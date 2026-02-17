#!/usr/bin/env python3
"""
2_gen_bias.py - Build true and observed colorectal labels with intersectional masking.

This script:
1. Loads baseline files from output/data/
2. Assigns each patient to an access plan via config/plan_rules.csv (age + income)
3. Builds true CRC outcomes and clinical feature columns
4. Applies probabilistic masking by plan stratum with age-band multipliers
5. Writes output/data/data.csv and output/info/2_bias_effect.md

Usage:
    uv run python scripts/2_gen_bias.py [--rules-path PATH] [--seed N]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"
CONFIG_DIR = PROJECT_DIR / "config"

CRC_STAGE_CODE_MAP = {
    "93761005": 1,
    "109838007": 2,
    "363406005": 3,
    "94260004": 4,
}

DIABETES_CODE = "44054006"
PREDIABETES_CODE = "714628002"
OBESITY_CODE = "162864005"
HYPERTENSION_CODE = "59621000"
HYPERLIPIDEMIA_CODE = "55822004"
CHF_CODE = "88805009"

BMI_CODE = "39156-5"
SMOKING_CODE = "72166-2"

AGE_BANDS = ["40-49", "50-59", "60-69", "70-79", "80+"]
AGE_BAND_MULTIPLIER = {
    "40-49": 1.30,
    "50-59": 1.15,
    "60-69": 1.00,
    "70-79": 0.90,
    "80+": 0.85,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate intersectional masking labels")
    parser.add_argument("--rules-path", type=Path, default=CONFIG_DIR / "plan_rules.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_inputs(rules_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    conditions = pd.read_csv(DATA_DIR / "conditions.csv")
    observations = pd.read_csv(DATA_DIR / "observations.csv")
    rules = pd.read_csv(rules_path)
    return patients, conditions, observations, rules


def infer_reference_date(conditions: pd.DataFrame) -> pd.Timestamp:
    if "START" in conditions.columns and len(conditions) > 0:
        starts = pd.to_datetime(conditions["START"], errors="coerce").dropna()
        if len(starts) > 0:
            return starts.max()
    return pd.Timestamp(datetime.utcnow().date())


def age_band_from_series(age: pd.Series) -> pd.Series:
    return pd.cut(
        age,
        bins=[0, 49, 59, 69, 79, 1000],
        labels=AGE_BANDS,
    ).astype(str)


def validate_rules(rules: pd.DataFrame) -> None:
    required = {
        "plan_id",
        "income_min",
        "income_max",
        "age_min",
        "age_max",
        "screening_start_age",
        "mask_rate_crc",
        "mask_rate_early",
    }
    missing = required - set(rules.columns)
    if missing:
        raise ValueError(f"Missing columns in rules file: {sorted(missing)}")

    for col in ["mask_rate_crc", "mask_rate_early"]:
        if ((rules[col] < 0) | (rules[col] > 1)).any():
            raise ValueError(f"{col} must be within [0, 1]")


def assign_plan(df: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """Assign exactly one plan row to each patient."""
    out = df.copy()
    out["_row_id"] = np.arange(len(out))
    age_for_plan = out["age"].clip(lower=40, upper=100)

    matches = []
    for _, rule in rules.iterrows():
        mask = (
            (out["income"] >= float(rule["income_min"]))
            & (out["income"] <= float(rule["income_max"]))
            & (age_for_plan >= float(rule["age_min"]))
            & (age_for_plan <= float(rule["age_max"]))
        )
        if mask.any():
            block = out.loc[mask, ["_row_id"]].copy()
            block["assigned_plan"] = rule["plan_id"]
            block["screening_start_age"] = float(rule["screening_start_age"])
            block["mask_rate_crc"] = float(rule["mask_rate_crc"])
            block["mask_rate_early"] = float(rule["mask_rate_early"])
            matches.append(block)

    if not matches:
        raise RuntimeError("No patients matched plan rules")

    matched = pd.concat(matches, ignore_index=True)
    counts = matched["_row_id"].value_counts()

    multi = counts[counts > 1]
    if len(multi) > 0:
        sample = multi.index[:5].tolist()
        raise RuntimeError(f"Plan rules overlap; multiple matches for row ids like {sample}")

    unmatched_ids = set(out["_row_id"]) - set(matched["_row_id"])
    if unmatched_ids:
        sample = sorted(list(unmatched_ids))[:5]
        raise RuntimeError(f"Plan rules incomplete; unmatched row ids like {sample}")

    out = out.merge(matched, on="_row_id", how="left")
    return out.drop(columns=["_row_id"])


def latest_observation_by_code(observations: pd.DataFrame, code: str) -> pd.Series:
    subset = observations[observations["CODE"].astype(str) == code].copy()
    if len(subset) == 0:
        return pd.Series(dtype=object)
    subset["DATE"] = pd.to_datetime(subset["DATE"], errors="coerce")
    subset = subset.sort_values("DATE").groupby("PATIENT").last()
    return subset["VALUE"]


def condition_flag(conditions: pd.DataFrame, code: str) -> set:
    return set(conditions[conditions["CODE"].astype(str) == code]["PATIENT"].unique())


def patient_numeric_column(patients: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in patients.columns:
        return pd.to_numeric(patients[name], errors="coerce").fillna(default)
    return pd.Series(default, index=patients.index, dtype=float)


def build_features(patients: pd.DataFrame, conditions: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    patients = patients.copy()
    conditions = conditions.copy()

    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    reference_date = infer_reference_date(conditions)

    df = pd.DataFrame()
    df["id"] = patients["Id"]
    df["age"] = ((reference_date - patients["BIRTHDATE"]).dt.days / 365.25).fillna(0).astype(int)
    df["male"] = (patients["GENDER"] == "M").astype(int)
    df["income"] = patient_numeric_column(patients, "INCOME", 0.0)

    # True CRC stage and targets
    cond = conditions.copy()
    cond["stage"] = cond["CODE"].astype(str).map(CRC_STAGE_CODE_MAP)
    cond = cond[cond["stage"].notna()].copy()
    cond["stage"] = cond["stage"].astype(int)
    stage_by_patient = cond.groupby("PATIENT")["stage"].max() if len(cond) > 0 else pd.Series(dtype=int)

    df["crc_stage_true"] = df["id"].map(stage_by_patient)
    df["has_crc_true"] = df["crc_stage_true"].notna().astype(int)
    df["has_early_crc_true"] = ((df["crc_stage_true"].fillna(99) <= 2) & (df["has_crc_true"] == 1)).astype(int)
    df["crc_stage_true"] = df["crc_stage_true"].fillna(0).astype(int)

    # Comorbidity flags
    diabetes_ids = condition_flag(conditions, DIABETES_CODE)
    prediabetes_ids = condition_flag(conditions, PREDIABETES_CODE)
    obesity_ids = condition_flag(conditions, OBESITY_CODE)
    hypertension_ids = condition_flag(conditions, HYPERTENSION_CODE)
    hyperlipidemia_ids = condition_flag(conditions, HYPERLIPIDEMIA_CODE)
    chf_ids = condition_flag(conditions, CHF_CODE)

    df["diabetes"] = df["id"].isin(diabetes_ids).astype(int)
    df["prediabetes"] = df["id"].isin(prediabetes_ids).astype(int)
    df["obesity"] = df["id"].isin(obesity_ids).astype(int)
    df["hypertension"] = df["id"].isin(hypertension_ids).astype(int)
    df["hyperlipidemia"] = df["id"].isin(hyperlipidemia_ids).astype(int)
    df["chf"] = df["id"].isin(chf_ids).astype(int)

    # BMI and smoking features
    bmi_latest = latest_observation_by_code(observations, BMI_CODE)
    df["bmi"] = pd.to_numeric(df["id"].map(bmi_latest), errors="coerce").fillna(25.0)

    smoker_latest = latest_observation_by_code(observations, SMOKING_CODE)
    smoker_map = smoker_latest.astype(str).str.lower().str.contains("current|daily|occasional", na=False)
    df["smoker"] = df["id"].map(smoker_map).fillna(False).astype(int)

    return df


def apply_masking(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    age_band = age_band_from_series(out["age"])

    out["eligible_for_screening"] = (out["age"] >= out["screening_start_age"]).astype(int)
    out["age_multiplier"] = age_band.map(AGE_BAND_MULTIPLIER).fillna(1.0)

    out["effective_mask_rate_crc"] = (out["mask_rate_crc"] * out["age_multiplier"]).clip(0, 1)
    out["effective_mask_rate_early"] = (out["mask_rate_early"] * out["age_multiplier"]).clip(0, 1)

    out["mask_crc"] = 0
    has_crc_idx = out.index[out["has_crc_true"] == 1]
    crc_draw = rng.random(len(has_crc_idx))
    crc_rates = out.loc[has_crc_idx, "effective_mask_rate_crc"].to_numpy()
    out.loc[has_crc_idx, "mask_crc"] = (crc_draw < crc_rates).astype(int)

    out["mask_early"] = 0
    has_early_idx = out.index[out["has_early_crc_true"] == 1]
    early_draw = rng.random(len(has_early_idx))
    early_rates = out.loc[has_early_idx, "effective_mask_rate_early"].to_numpy()
    out.loc[has_early_idx, "mask_early"] = (early_draw < early_rates).astype(int)

    out["observed_crc"] = ((out["has_crc_true"] == 1) & (out["mask_crc"] == 0)).astype(int)
    out["observed_early_crc"] = (
        (out["has_early_crc_true"] == 1)
        & (out["mask_early"] == 0)
        & (out["observed_crc"] == 1)
    ).astype(int)

    return out


def build_bias_report(df: pd.DataFrame, seed: int, rules_path: Path) -> str:
    total = len(df)

    true_crc = int(df["has_crc_true"].sum())
    obs_crc = int(df["observed_crc"].sum())
    true_early = int(df["has_early_crc_true"].sum())
    obs_early = int(df["observed_early_crc"].sum())

    by_plan_rows = []
    for plan_id, sub in df.groupby("assigned_plan"):
        n = len(sub)
        t = int(sub["has_crc_true"].sum())
        o = int(sub["observed_crc"].sum())
        te = int(sub["has_early_crc_true"].sum())
        oe = int(sub["observed_early_crc"].sum())
        mean_rate = sub["effective_mask_rate_crc"].mean()
        by_plan_rows.append(
            f"| {plan_id} | {n:,} | {t:,} | {o:,} | {te:,} | {oe:,} | {mean_rate:.3f} |"
        )

    by_age_rows = []
    age_band = age_band_from_series(df["age"])
    for band in AGE_BANDS:
        sub = df[age_band == band]
        if len(sub) == 0:
            continue
        t = int(sub["has_crc_true"].sum())
        o = int(sub["observed_crc"].sum())
        rate = sub["effective_mask_rate_crc"].mean()
        by_age_rows.append(
            f"| {band} | {len(sub):,} | {t:,} | {o:,} | {rate:.3f} | {(100 * t / len(sub)):.2f}% | {(100 * o / len(sub)):.2f}% |"
        )

    multiplier_rows = [
        f"| {band} | {AGE_BAND_MULTIPLIER[band]:.2f} |" for band in AGE_BANDS
    ]

    return f"""# Bias Effect Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Inputs

| Parameter | Value |
|-----------|-------|
| Rules file | `{rules_path}` |
| Seed | {seed} |

## Age Stratification Multipliers

| Age band | Mask multiplier |
|----------|-----------------|
{chr(10).join(multiplier_rows)}

## Overall Effect

| Metric | True | Observed | Relative drop |
|--------|------|----------|---------------|
| CRC cases | {true_crc:,} | {obs_crc:,} | {(100 * (true_crc - obs_crc) / true_crc) if true_crc else 0:.2f}% |
| Early CRC cases | {true_early:,} | {obs_early:,} | {(100 * (true_early - obs_early) / true_early) if true_early else 0:.2f}% |
| CRC prevalence | {(100 * true_crc / total) if total else 0:.2f}% | {(100 * obs_crc / total) if total else 0:.2f}% | - |

## By Assigned Plan

| Plan | Patients | True CRC | Observed CRC | True Early CRC | Observed Early CRC | Mean effective CRC mask rate |
|------|----------|----------|--------------|----------------|--------------------|-------------------------------|
{chr(10).join(by_plan_rows)}

## By Age Band

| Age band | Patients | True CRC | Observed CRC | Mean effective CRC mask rate | True prevalence | Observed prevalence |
|----------|----------|----------|--------------|-------------------------------|-----------------|---------------------|
{chr(10).join(by_age_rows)}
"""


def main() -> None:
    args = parse_args()
    patients, conditions, observations, rules = load_inputs(args.rules_path)

    validate_rules(rules)

    df = build_features(patients, conditions, observations)
    df = assign_plan(df, rules)
    df = apply_masking(df, seed=args.seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INFO_DIR.mkdir(parents=True, exist_ok=True)

    out_path = DATA_DIR / "data.csv"
    df.to_csv(out_path, index=False)

    report = build_bias_report(df, seed=args.seed, rules_path=args.rules_path)
    (INFO_DIR / "2_bias_effect.md").write_text(report)

    # Keep only the consolidated dataset to minimize retained raw exports.
    for name in ["patients.csv", "conditions.csv", "observations.csv", "procedures.csv", "encounters.csv"]:
        p = DATA_DIR / name
        if p.exists():
            os.remove(p)

    print(f"Wrote {out_path} ({len(df):,} rows)")
    print(f"Wrote {INFO_DIR / '2_bias_effect.md'}")


if __name__ == "__main__":
    main()
