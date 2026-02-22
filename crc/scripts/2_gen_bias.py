#!/usr/bin/env python3
"""Build CRC analysis dataset and apply intersectional masking bias."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
SYNTHEA_OUTPUT_CSV_DIR = REPO_ROOT / "synthea" / "output_crc" / "csv"
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"

CRC_SCREENING_CODE = "73761001"
OBESITY_CODE = "162864005"
TYPE2_DIABETES_CODE = "44054006"
HYPERTENSION_CODE = "59621000"
HYPERLIPIDEMIA_CODE = "55822004"
IBD_CODES = {"34000006", "64766004"}

SMOKING_CODE = "72166-2"
BMI_CODE = "39156-5"
AGE_MIN = 50
AGE_MAX = 80
AGE_BIN_EDGES = [49, 54, 59, 64, 69, 74, 80]
AGE_BIN_LABELS = ["50-54", "55-59", "60-64", "65-69", "70-74", "75-80"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CRC bias labels")
    parser.add_argument("-s", "--seed", type=int, default=160)
    return parser.parse_args()


def infer_reference_date(procedures: pd.DataFrame, conditions: pd.DataFrame, observations: pd.DataFrame) -> pd.Timestamp:
    candidates: list[pd.Timestamp] = []
    proc_time_col = "DATE" if "DATE" in procedures.columns else "START" if "START" in procedures.columns else None
    for frame, col in [(procedures, proc_time_col), (conditions, "START"), (observations, "DATE")]:
        if col and col in frame.columns:
            values = pd.to_datetime(frame[col], errors="coerce", utc=True).dropna()
            if len(values) > 0:
                candidates.append(values.dt.tz_convert(None).max())
    return max(candidates) if candidates else pd.Timestamp(datetime.utcnow().date())


def latest_observation(observations: pd.DataFrame, code: str) -> pd.Series:
    obs = observations[observations["CODE"].astype(str) == code].copy()
    if len(obs) == 0:
        return pd.Series(dtype=object)
    obs["DATE"] = pd.to_datetime(obs["DATE"], errors="coerce", utc=True).dt.tz_convert(None)
    obs = obs.sort_values("DATE").groupby("PATIENT").last()
    return obs["VALUE"]


def flagged_patients(conditions: pd.DataFrame, code_set: set[str]) -> set[str]:
    mask = conditions["CODE"].astype(str).isin(code_set)
    return set(conditions.loc[mask, "PATIENT"].astype(str).unique())


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(
        age,
        bins=AGE_BIN_EDGES,
        labels=AGE_BIN_LABELS,
        include_lowest=True,
    ).astype(str)


def income_band(income: pd.Series) -> pd.Series:
    return pd.cut(
        income,
        bins=[-1, 29999, 59999, 99999, np.inf],
        labels=["<30k", "30k-59k", "60k-99k", "100k+"],
    ).astype(str)


def build_dataset(seed: int) -> pd.DataFrame:
    if not SYNTHEA_OUTPUT_CSV_DIR.exists():
        raise FileNotFoundError(
            f"Missing Synthea CSV directory: {SYNTHEA_OUTPUT_CSV_DIR}. "
            "Run scripts/1_generate_data.py first."
        )

    patients = pd.read_csv(SYNTHEA_OUTPUT_CSV_DIR / "patients.csv")
    conditions = pd.read_csv(SYNTHEA_OUTPUT_CSV_DIR / "conditions.csv")
    procedures = pd.read_csv(SYNTHEA_OUTPUT_CSV_DIR / "procedures.csv")
    observations = pd.read_csv(SYNTHEA_OUTPUT_CSV_DIR / "observations.csv")
    encounters = pd.read_csv(SYNTHEA_OUTPUT_CSV_DIR / "encounters.csv")

    ref_date = infer_reference_date(procedures, conditions, observations)

    patients = patients.copy()
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    patients["age"] = ((ref_date - patients["BIRTHDATE"]).dt.days / 365.25).fillna(0).astype(int)
    patients = patients[(patients["age"] >= AGE_MIN) & (patients["age"] <= AGE_MAX)].copy()

    df = pd.DataFrame({
        "id": patients["Id"].astype(str),
        "age": patients["age"],
        "male": (patients["GENDER"] == "M").astype(int),
        "income_usd": pd.to_numeric(patients.get("INCOME", 0), errors="coerce").fillna(0),
    })

    smoker_latest = latest_observation(observations, SMOKING_CODE)
    bmi_latest = latest_observation(observations, BMI_CODE)

    current_smoker = smoker_latest.astype(str).str.lower().str.contains("current|daily|every day|some day", na=False)
    df["smoker"] = df["id"].map(current_smoker).fillna(False).astype(int)
    df["bmi"] = pd.to_numeric(df["id"].map(bmi_latest), errors="coerce").fillna(27.0)

    obesity_ids = flagged_patients(conditions, {OBESITY_CODE})
    diabetes_ids = flagged_patients(conditions, {TYPE2_DIABETES_CODE})
    htn_ids = flagged_patients(conditions, {HYPERTENSION_CODE})
    hld_ids = flagged_patients(conditions, {HYPERLIPIDEMIA_CODE})
    ibd_ids = flagged_patients(conditions, IBD_CODES)

    df["obesity"] = df["id"].isin(obesity_ids).astype(int)
    df["type2_diabetes"] = df["id"].isin(diabetes_ids).astype(int)
    df["hypertension"] = df["id"].isin(htn_ids).astype(int)
    df["hyperlipidemia"] = df["id"].isin(hld_ids).astype(int)
    df["ibd"] = df["id"].isin(ibd_ids).astype(int)

    enc = encounters.copy()
    enc["PATIENT"] = enc["PATIENT"].astype(str)
    enc["START"] = pd.to_datetime(enc["START"], errors="coerce", utc=True).dt.tz_convert(None)
    two_year_start = ref_date - pd.Timedelta(days=365 * 2)
    recent_enc = enc[enc["START"] >= two_year_start].copy()

    visit_count = recent_enc.groupby("PATIENT").size()
    preventive = recent_enc[recent_enc.get("ENCOUNTERCLASS", "").astype(str).str.lower().isin(["wellness", "ambulatory", "outpatient"])]
    preventive_flag = (preventive.groupby("PATIENT").size() > 0).astype(int)

    df["ambulatory_visits_last2y"] = df["id"].map(visit_count).fillna(0).astype(int)
    df["preventive_visit_last2y"] = df["id"].map(preventive_flag).fillna(0).astype(int)
    df["comorbidity_count"] = df[["obesity", "type2_diabetes", "hypertension", "hyperlipidemia", "ibd"]].sum(axis=1)

    proc = procedures.copy()
    proc["PATIENT"] = proc["PATIENT"].astype(str)
    proc_time_col = "DATE" if "DATE" in proc.columns else "START" if "START" in proc.columns else None
    if proc_time_col is None:
        raise KeyError("Missing procedure timestamp column (expected DATE or START)")
    proc[proc_time_col] = pd.to_datetime(proc[proc_time_col], errors="coerce", utc=True).dt.tz_convert(None)
    proc = proc[(proc["CODE"].astype(str) == CRC_SCREENING_CODE) & (proc[proc_time_col] >= ref_date - pd.Timedelta(days=365 * 5))]

    screened_ids = set(proc["PATIENT"].unique())
    df["true_screened_in_last_5y"] = df["id"].isin(screened_ids).astype(int)

    p_mask = np.clip(
        0.60 - 0.004 * (df["age"] - 50) - 0.000005 * (df["income_usd"] - 20000),
        0.05,
        0.70,
    )
    rng = np.random.default_rng(seed)
    draw = rng.random(len(df))

    df["mask_screening"] = 0
    mask_idx = df["true_screened_in_last_5y"] == 1
    df.loc[mask_idx, "mask_screening"] = (draw[mask_idx] < p_mask[mask_idx]).astype(int)
    df["observed_screened_in_last_5y"] = df["true_screened_in_last_5y"] * (1 - df["mask_screening"])

    df["age_band"] = age_band(df["age"])
    df["income_band"] = income_band(df["income_usd"])
    return df


def prevalence_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False).agg(
        n=("id", "count"),
        true_rate=("true_screened_in_last_5y", "mean"),
        observed_rate=("observed_screened_in_last_5y", "mean"),
        masked_rate=("mask_screening", "mean"),
    )
    return grouped.reset_index()


def write_report(df: pd.DataFrame) -> None:
    overall_true = df["true_screened_in_last_5y"].mean()
    overall_observed = df["observed_screened_in_last_5y"].mean()

    age_tbl = prevalence_table(df, ["age_band"]).to_markdown(index=False)
    income_tbl = prevalence_table(df, ["income_band"]).to_markdown(index=False)
    inter_tbl = prevalence_table(df, ["age_band", "income_band"]).to_markdown(index=False)

    text = f"""# CRC Bias Effect

- Cohort size: {len(df):,}
- True screening prevalence: {overall_true:.3f}
- Observed screening prevalence: {overall_observed:.3f}
- Relative loss from masking: {(overall_true - overall_observed) / max(overall_true, 1e-9):.1%}

## By Age Band (5-year bins)

{age_tbl}

## By Income Band

{income_tbl}

## By Age x Income

{inter_tbl}
"""
    (INFO_DIR / "2_bias_effect.md").write_text(text)


def main() -> None:
    args = parse_args()
    INFO_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset(args.seed)
    df.to_csv(DATA_DIR / "data.csv", index=False)
    write_report(df)
    print(f"Wrote {DATA_DIR / 'data.csv'} ({len(df):,} rows)")
    print(f"Wrote {INFO_DIR / '2_bias_effect.md'}")


if __name__ == "__main__":
    main()
