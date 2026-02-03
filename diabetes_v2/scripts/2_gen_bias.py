#!/usr/bin/env python3
"""
2_gen_bias.py - Build feature matrix and generate biased dataset.

This script:
1. Loads raw Synthea output (patients.csv, conditions.csv, observations.csv)
2. Builds feature matrix with age, gender, BMI, A1c, comorbidities, etc.
3. Adds hypertriglyceridemia flag and masks a portion to simulate documentation bias
4. Outputs single data.csv with all features needed for modeling
5. Deletes source CSV files to save space
6. Outputs 2_bias_effect.md with before/after prevalence statistics

Usage:
    uv run python scripts/2_gen_bias.py [--mask-rate 0.3] [--seed 42]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"

# Condition codes
DIABETES_CODE = "44054006"
PREDIABETES_CODE = "714628002"
HYPERTRIGLYCERIDEMIA_CODE = "302870006"
OBESITY_CODE = "162864005"
HYPERTENSION_CODE = "59621000"
HYPERLIPIDEMIA_CODE = "55822004"

# Observation codes
A1C_CODE = "4548-4"
BMI_CODE = "39156-5"
SMOKING_CODE = "72166-2"


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw Synthea output files."""
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    conditions = pd.read_csv(DATA_DIR / "conditions.csv")
    observations = pd.read_csv(DATA_DIR / "observations.csv")
    return patients, conditions, observations


def build_features(
    patients: pd.DataFrame,
    conditions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature matrix from raw Synthea data."""
    df = pd.DataFrame()
    df["id"] = patients["Id"]

    # Age from birthdate
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
    reference_date = patients["BIRTHDATE"].max() + pd.DateOffset(years=50)
    df["age"] = ((reference_date - patients["BIRTHDATE"]).dt.days / 365.25).astype(int)

    # Gender (1 = male)
    df["male"] = (patients["GENDER"] == "M").astype(int)

    # Income (normalized)
    df["income"] = patients["INCOME"] / 100000

    # Conditions - check if patient has each condition
    codes = conditions["CODE"].astype(str)

    # Target: Diabetes
    diabetes_patients = set(conditions[codes == DIABETES_CODE]["PATIENT"].unique())
    df["has_diabetes"] = patients["Id"].isin(diabetes_patients).astype(int)

    # Key metabolic condition (this will be masked)
    hypertriglyceridemia_patients = set(conditions[codes == HYPERTRIGLYCERIDEMIA_CODE]["PATIENT"].unique())
    df["has_hypertriglyceridemia"] = patients["Id"].isin(hypertriglyceridemia_patients).astype(int)

    # Other conditions (not masked)
    for code, name in [
        (PREDIABETES_CODE, "prediabetes"),
        (OBESITY_CODE, "obesity"),
        (HYPERTENSION_CODE, "hypertension"),
        (HYPERLIPIDEMIA_CODE, "hyperlipidemia"),
    ]:
        patient_ids = set(conditions[codes == code]["PATIENT"].unique())
        df[name] = patients["Id"].isin(patient_ids).astype(int)

    # A1c - get latest value per patient
    a1c_obs = observations[observations["CODE"].astype(str) == A1C_CODE].copy()
    if len(a1c_obs) > 0:
        a1c_obs["DATE"] = pd.to_datetime(a1c_obs["DATE"])
        a1c_latest = a1c_obs.sort_values("DATE").groupby("PATIENT").last()["VALUE"]
        df["a1c"] = patients["Id"].map(a1c_latest)
        df["a1c"] = pd.to_numeric(df["a1c"], errors="coerce")
        df["a1c"] = df["a1c"].fillna(5.4)
    else:
        df["a1c"] = 5.4

    # BMI - get latest value per patient
    bmi_obs = observations[observations["CODE"].astype(str) == BMI_CODE].copy()
    if len(bmi_obs) > 0:
        bmi_obs["DATE"] = pd.to_datetime(bmi_obs["DATE"])
        bmi_latest = bmi_obs.sort_values("DATE").groupby("PATIENT").last()["VALUE"]
        df["bmi"] = patients["Id"].map(bmi_latest).fillna(25.0)
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").fillna(25.0)
    else:
        df["bmi"] = 25.0

    # Smoking status - get latest value per patient
    smoking_obs = observations[observations["CODE"].astype(str) == SMOKING_CODE].copy()
    if len(smoking_obs) > 0:
        smoking_obs["DATE"] = pd.to_datetime(smoking_obs["DATE"])
        smoking_latest = smoking_obs.sort_values("DATE").groupby("PATIENT").last()["VALUE"]
        smoking_map = smoking_latest.str.lower().str.contains("current|daily|occasional", na=False)
        df["smoker"] = patients["Id"].map(smoking_map).fillna(False).astype(int)
    else:
        df["smoker"] = 0

    return df


def add_mask_flag(df: pd.DataFrame, mask_rate: float, seed: int) -> pd.DataFrame:
    """Add mask flag for hypertriglyceridemia to simulate documentation bias."""
    df = df.copy()
    rng = np.random.default_rng(seed)

    # Initialize mask column
    df["mask_hypertriglyceridemia"] = 0

    # Mask hypertriglyceridemia (random subset of patients with the condition)
    ht_indices = df[df["has_hypertriglyceridemia"] == 1].index
    n_to_mask = int(len(ht_indices) * mask_rate)
    if n_to_mask > 0:
        masked_indices = rng.choice(ht_indices, size=n_to_mask, replace=False)
        df.loc[masked_indices, "mask_hypertriglyceridemia"] = 1

    # Add observed (biased) feature
    df["observed_hypertriglyceridemia"] = ((df["has_hypertriglyceridemia"] == 1) & (df["mask_hypertriglyceridemia"] == 0)).astype(int)

    return df


def compute_prevalence_stats(df: pd.DataFrame, use_masked: bool = False) -> dict:
    """Compute prevalence statistics."""
    n_total = len(df)

    # Determine effective hypertriglyceridemia status
    if use_masked:
        effective_ht = df["observed_hypertriglyceridemia"]
    else:
        effective_ht = df["has_hypertriglyceridemia"]

    n_diabetes = df["has_diabetes"].sum()
    n_ht = effective_ht.sum()

    # Age decades
    decade = pd.cut(
        df["age"],
        bins=[0, 49, 59, 69, 79, 89, 150],
        labels=["40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    )

    stats = {
        "n_total": n_total,
        "n_diabetes": int(n_diabetes),
        "pct_diabetes": 100 * n_diabetes / n_total if n_total else 0,
        "n_hypertriglyceridemia": int(n_ht),
        "pct_hypertriglyceridemia": 100 * n_ht / n_total if n_total else 0,
    }

    # By gender
    for gender_name, gender_val in [("male", 1), ("female", 0)]:
        subset_mask = df["male"] == gender_val
        n = subset_mask.sum()
        n_ht_g = effective_ht[subset_mask].sum()
        n_diab = df.loc[subset_mask, "has_diabetes"].sum()
        stats[f"n_{gender_name}"] = int(n)
        stats[f"n_hypertriglyceridemia_{gender_name}"] = int(n_ht_g)
        stats[f"pct_hypertriglyceridemia_{gender_name}"] = 100 * n_ht_g / n if n else 0
        stats[f"n_diabetes_{gender_name}"] = int(n_diab)
        stats[f"pct_diabetes_{gender_name}"] = 100 * n_diab / n if n else 0

    # By decade
    decade_stats = []
    for dec in ["40-49", "50-59", "60-69", "70-79", "80-89", "90+"]:
        subset_mask = decade == dec
        n = subset_mask.sum()
        n_ht_d = effective_ht[subset_mask].sum()
        n_diab = df.loc[subset_mask, "has_diabetes"].sum()
        decade_stats.append({
            "decade": dec,
            "n": int(n),
            "n_hypertriglyceridemia": int(n_ht_d),
            "pct_hypertriglyceridemia": 100 * n_ht_d / n if n else 0,
            "n_diabetes": int(n_diab),
            "pct_diabetes": 100 * n_diab / n if n else 0,
        })
    stats["decade_stats"] = decade_stats

    # Co-occurrence with diabetes
    diabetes_mask = df["has_diabetes"] == 1
    n_diabetes_with_ht = (diabetes_mask & (effective_ht == 1)).sum()
    stats["n_diabetes_with_hypertriglyceridemia"] = int(n_diabetes_with_ht)
    stats["pct_diabetes_with_hypertriglyceridemia"] = 100 * n_diabetes_with_ht / n_diabetes if n_diabetes else 0

    return stats


def write_bias_effect_report(
    df: pd.DataFrame,
    before_stats: dict,
    after_stats: dict,
    mask_rate: float,
    seed: int
) -> None:
    """Write bias effect report to markdown file."""
    md_path = INFO_DIR / "2_bias_effect.md"

    # Masking summary
    n_ht = df["has_hypertriglyceridemia"].sum()
    n_masked = df["mask_hypertriglyceridemia"].sum()

    # Decade comparison table
    decade_rows = []
    for b_dec, a_dec in zip(before_stats["decade_stats"], after_stats["decade_stats"]):
        diff = a_dec["pct_hypertriglyceridemia"] - b_dec["pct_hypertriglyceridemia"]
        decade_rows.append(
            f"| {b_dec['decade']} | {b_dec['pct_hypertriglyceridemia']:.2f}% | {a_dec['pct_hypertriglyceridemia']:.2f}% | {diff:+.2f}% |"
        )
    decade_table = "\n".join(decade_rows)

    content = f"""# Bias Effect Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Masking Parameters

| Parameter | Value |
|-----------|-------|
| Mask rate | {mask_rate:.0%} |
| Random seed | {seed} |

## Masking Summary

| Metric | Value |
|--------|-------|
| Patients with hypertriglyceridemia | {n_ht:,} |
| Patients masked | {n_masked:,} |
| Actual mask rate | {100 * n_masked / n_ht:.1f}% |

## Overall Effect

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Hypertriglyceridemia cases | {before_stats['n_hypertriglyceridemia']:,} | {after_stats['n_hypertriglyceridemia']:,} | {after_stats['n_hypertriglyceridemia'] - before_stats['n_hypertriglyceridemia']:+,} |
| Prevalence | {before_stats['pct_hypertriglyceridemia']:.2f}% | {after_stats['pct_hypertriglyceridemia']:.2f}% | {after_stats['pct_hypertriglyceridemia'] - before_stats['pct_hypertriglyceridemia']:+.2f}% |

## Effect on Diabetes Association

Hypertriglyceridemia is a risk marker for diabetes. Masking it affects the apparent association.

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Diabetics with hypertriglyceridemia | {before_stats['n_diabetes_with_hypertriglyceridemia']:,} ({before_stats['pct_diabetes_with_hypertriglyceridemia']:.1f}%) | {after_stats['n_diabetes_with_hypertriglyceridemia']:,} ({after_stats['pct_diabetes_with_hypertriglyceridemia']:.1f}%) | {after_stats['pct_diabetes_with_hypertriglyceridemia'] - before_stats['pct_diabetes_with_hypertriglyceridemia']:+.1f}% |

## Prevalence by Gender

| Gender | Before | After | Change |
|--------|--------|-------|--------|
| Male | {before_stats['pct_hypertriglyceridemia_male']:.2f}% | {after_stats['pct_hypertriglyceridemia_male']:.2f}% | {after_stats['pct_hypertriglyceridemia_male'] - before_stats['pct_hypertriglyceridemia_male']:+.2f}% |
| Female | {before_stats['pct_hypertriglyceridemia_female']:.2f}% | {after_stats['pct_hypertriglyceridemia_female']:.2f}% | {after_stats['pct_hypertriglyceridemia_female'] - before_stats['pct_hypertriglyceridemia_female']:+.2f}% |

## Prevalence by Age Decade

| Decade | Before | After | Change |
|--------|--------|-------|--------|
{decade_table}

## Key Observations

- **Documentation bias is random**: This bias affects all patient groups equally (no demographic targeting)
- **Hypertriglyceridemia prevalence**: {before_stats['pct_hypertriglyceridemia']:.2f}% true -> {after_stats['pct_hypertriglyceridemia']:.2f}% observed
- **Impact on modeling**: Models trained on observed data will learn a weaker association between hypertriglyceridemia and diabetes
"""

    md_path.write_text(content)
    print(f"Wrote {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Build features and generate biased dataset")
    parser.add_argument("--mask-rate", "-m", type=float, default=0.3, help="Fraction of hypertriglyceridemia to mask (default: 0.3)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("Diabetes v2: Build Features & Generate Bias")
    print("=" * 60)

    # Load raw data
    print("\nLoading raw Synthea data...")
    patients, conditions, observations = load_raw_data()
    print(f"  Patients: {len(patients):,}")
    print(f"  Conditions: {len(conditions):,}")
    print(f"  Observations: {len(observations):,}")

    # Build features
    print("\nBuilding feature matrix...")
    df = build_features(patients, conditions, observations)
    print(f"  Features: age, male, income, a1c, bmi, smoker, prediabetes, obesity, hypertension, hyperlipidemia, hypertriglyceridemia")
    print(f"  Target (diabetes): {df['has_diabetes'].sum():,} ({100*df['has_diabetes'].mean():.2f}%)")
    print(f"  Hypertriglyceridemia: {df['has_hypertriglyceridemia'].sum():,} ({100*df['has_hypertriglyceridemia'].mean():.2f}%)")

    # Add mask flag
    print(f"\nApplying {args.mask_rate:.0%} documentation bias...")
    df = add_mask_flag(df, args.mask_rate, args.seed)
    n_masked = df["mask_hypertriglyceridemia"].sum()
    print(f"  Hypertriglyceridemia masked: {n_masked:,}")

    # Compute before/after stats
    print("\nComputing prevalence statistics...")
    before_stats = compute_prevalence_stats(df, use_masked=False)
    after_stats = compute_prevalence_stats(df, use_masked=True)

    # Save consolidated data.csv
    output_path = DATA_DIR / "data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nWrote {output_path} ({output_path.stat().st_size:,} bytes)")

    # Delete source files
    print("\nCleaning up source files...")
    for filename in ["patients.csv", "conditions.csv", "observations.csv"]:
        filepath = DATA_DIR / filename
        if filepath.exists():
            filepath.unlink()
            print(f"  Deleted {filename}")

    # Write report
    write_bias_effect_report(df, before_stats, after_stats, args.mask_rate, args.seed)

    # Console summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nDiabetes prevalence: {before_stats['pct_diabetes']:.2f}%")
    print(f"\nHypertriglyceridemia:")
    print(f"  True:     {before_stats['pct_hypertriglyceridemia']:.2f}%")
    print(f"  Observed: {after_stats['pct_hypertriglyceridemia']:.2f}%")
    print(f"\nDiabetics with hypertriglyceridemia:")
    print(f"  True:     {before_stats['pct_diabetes_with_hypertriglyceridemia']:.1f}%")
    print(f"  Observed: {after_stats['pct_diabetes_with_hypertriglyceridemia']:.1f}%")

    print("\n" + "=" * 60)
    print(f"Complete! Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
