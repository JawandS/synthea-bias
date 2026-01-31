#!/usr/bin/env python3
"""
2_gen_bias.py - Build feature matrix and generate biased dataset.

This script:
1. Loads raw Synthea output (patients.csv, conditions.csv, observations.csv)
2. Builds feature matrix with age, gender, BMI, comorbidities, etc.
3. Adds sleep apnea flags and masks rural patients to simulate underdiagnosis
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
SLEEP_APNEA_CODES = {"73430006", "78275009"}
HYPERTENSION_CODE = "59621000"
CHF_CODE = "88805009"
ALCOHOL_USE_CODE = "7200002"

# Observation codes
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
    reference_date = patients["BIRTHDATE"].max() + pd.DateOffset(years=70)
    df["age"] = ((reference_date - patients["BIRTHDATE"]).dt.days / 365.25).astype(int)

    # Gender (1 = male)
    df["male"] = (patients["GENDER"] == "M").astype(int)

    # Urban flag
    df["urban"] = patients["URBAN"].astype(int)

    # Income (normalized)
    df["income"] = patients["INCOME"] / 100000  # Scale to ~0-2 range

    # Conditions - check if patient has each condition
    codes = conditions["CODE"].astype(str)

    # Sleep apnea
    apnea_patients = set(conditions[codes.isin(SLEEP_APNEA_CODES)]["PATIENT"].unique())
    df["has_sleep_apnea"] = patients["Id"].isin(apnea_patients).astype(int)

    # Other conditions
    for code, name in [
        (HYPERTENSION_CODE, "hypertension"),
        (CHF_CODE, "chf"),
        (ALCOHOL_USE_CODE, "alcohol_use"),
    ]:
        patient_ids = set(conditions[codes == code]["PATIENT"].unique())
        df[name] = patients["Id"].isin(patient_ids).astype(int)

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
    """Add mask_sleep_apnea flag for underdiagnosed rural patients."""
    df = df.copy()
    rng = np.random.default_rng(seed)

    # Initialize all as 0
    df["mask_sleep_apnea"] = 0

    # Find rural patients with sleep apnea
    rural_apnea_mask = (df["urban"] == 0) & (df["has_sleep_apnea"] == 1)
    rural_apnea_indices = df[rural_apnea_mask].index

    # Randomly select patients to mask
    n_to_mask = int(len(rural_apnea_indices) * mask_rate)
    if n_to_mask > 0:
        masked_indices = rng.choice(rural_apnea_indices, size=n_to_mask, replace=False)
        df.loc[masked_indices, "mask_sleep_apnea"] = 1

    # Add observed (biased) label
    df["observed_sleep_apnea"] = ((df["has_sleep_apnea"] == 1) & (df["mask_sleep_apnea"] == 0)).astype(int)

    return df


def compute_prevalence_stats(df: pd.DataFrame, use_masked: bool = False) -> dict:
    """Compute prevalence statistics."""
    # Determine effective sleep apnea status
    if use_masked:
        effective_apnea = df["observed_sleep_apnea"]
    else:
        effective_apnea = df["has_sleep_apnea"]

    n_total = len(df)
    n_apnea = effective_apnea.sum()

    # Age decades
    decade = pd.cut(
        df["age"],
        bins=[0, 69, 79, 89, 150],
        labels=["60-69", "70-79", "80-89", "90+"]
    )

    stats = {
        "n_total": n_total,
        "n_apnea": int(n_apnea),
        "pct_apnea": 100 * n_apnea / n_total if n_total else 0,
    }

    # By location
    for loc_name, loc_val in [("urban", 1), ("rural", 0)]:
        subset_mask = df["urban"] == loc_val
        n = subset_mask.sum()
        n_apnea_loc = effective_apnea[subset_mask].sum()
        stats[f"n_{loc_name}"] = int(n)
        stats[f"n_apnea_{loc_name}"] = int(n_apnea_loc)
        stats[f"pct_apnea_{loc_name}"] = 100 * n_apnea_loc / n if n else 0

    # By gender
    for gender_name, gender_val in [("male", 1), ("female", 0)]:
        subset_mask = df["male"] == gender_val
        n = subset_mask.sum()
        n_apnea_g = effective_apnea[subset_mask].sum()
        stats[f"n_{gender_name}"] = int(n)
        stats[f"n_apnea_{gender_name}"] = int(n_apnea_g)
        stats[f"pct_apnea_{gender_name}"] = 100 * n_apnea_g / n if n else 0

    # By gender x location
    for gender_name, gender_val in [("male", 1), ("female", 0)]:
        for loc_name, loc_val in [("urban", 1), ("rural", 0)]:
            subset_mask = (df["male"] == gender_val) & (df["urban"] == loc_val)
            n = subset_mask.sum()
            n_apnea_gl = effective_apnea[subset_mask].sum()
            key = f"{gender_name}_{loc_name}"
            stats[f"n_{key}"] = int(n)
            stats[f"n_apnea_{key}"] = int(n_apnea_gl)
            stats[f"pct_apnea_{key}"] = 100 * n_apnea_gl / n if n else 0

    # By decade
    decade_stats = []
    for dec in ["60-69", "70-79", "80-89", "90+"]:
        subset_mask = decade == dec
        n = subset_mask.sum()
        n_apnea_d = effective_apnea[subset_mask].sum()
        decade_stats.append({
            "decade": dec,
            "n": int(n),
            "n_apnea": int(n_apnea_d),
            "pct": 100 * n_apnea_d / n if n else 0,
        })
    stats["decade_stats"] = decade_stats

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
    n_rural_apnea = ((df["urban"] == 0) & (df["has_sleep_apnea"] == 1)).sum()
    n_masked = df["mask_sleep_apnea"].sum()

    # Build comparison rows
    def comparison_row(label: str, key: str) -> str:
        b_n = before_stats.get(f"n_apnea_{key}", 0)
        b_pct = before_stats.get(f"pct_apnea_{key}", 0)
        a_n = after_stats.get(f"n_apnea_{key}", 0)
        a_pct = after_stats.get(f"pct_apnea_{key}", 0)
        diff = a_pct - b_pct
        return f"| {label} | {b_n:,} | {b_pct:.2f}% | {a_n:,} | {a_pct:.2f}% | {diff:+.2f}% |"

    # Decade comparison
    decade_rows = []
    for b_dec, a_dec in zip(before_stats["decade_stats"], after_stats["decade_stats"]):
        diff = a_dec["pct"] - b_dec["pct"]
        decade_rows.append(
            f"| {b_dec['decade']} | {b_dec['n_apnea']:,} | {b_dec['pct']:.2f}% | "
            f"{a_dec['n_apnea']:,} | {a_dec['pct']:.2f}% | {diff:+.2f}% |"
        )
    decade_table = "\n".join(decade_rows)

    content = f"""# Bias Effect Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Masking Parameters

| Parameter | Value |
|-----------|-------|
| Mask rate | {mask_rate:.0%} |
| Random seed | {seed} |
| Rural patients with sleep apnea | {n_rural_apnea:,} |
| Patients masked | {n_masked:,} |
| Actual mask rate | {100 * n_masked / n_rural_apnea:.1f}% |

## Overall Effect

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Sleep apnea cases | {before_stats['n_apnea']:,} | {after_stats['n_apnea']:,} | {after_stats['n_apnea'] - before_stats['n_apnea']:+,} |
| Prevalence | {before_stats['pct_apnea']:.2f}% | {after_stats['pct_apnea']:.2f}% | {after_stats['pct_apnea'] - before_stats['pct_apnea']:+.2f}% |

## Prevalence by Location

| Location | Before Cases | Before % | After Cases | After % | Change |
|----------|--------------|----------|-------------|---------|--------|
{comparison_row("Urban", "urban")}
{comparison_row("Rural", "rural")}

## Prevalence by Gender

| Gender | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
{comparison_row("Male", "male")}
{comparison_row("Female", "female")}

## Prevalence by Gender and Location

| Group | Before Cases | Before % | After Cases | After % | Change |
|-------|--------------|----------|-------------|---------|--------|
{comparison_row("Male Urban", "male_urban")}
{comparison_row("Male Rural", "male_rural")}
{comparison_row("Female Urban", "female_urban")}
{comparison_row("Female Rural", "female_rural")}

## Prevalence by Age Decade

| Decade | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
{decade_table}

## Key Observations

- **True prevalence**: {before_stats['pct_apnea']:.2f}% of all patients have sleep apnea
- **Observed prevalence**: {after_stats['pct_apnea']:.2f}% after rural underdiagnosis bias
- **Rural underdiagnosis**: {before_stats['pct_apnea_rural']:.2f}% -> {after_stats['pct_apnea_rural']:.2f}% ({after_stats['pct_apnea_rural'] - before_stats['pct_apnea_rural']:+.2f}%)
- **Urban (unaffected)**: {before_stats['pct_apnea_urban']:.2f}% -> {after_stats['pct_apnea_urban']:.2f}%
"""

    md_path.write_text(content)
    print(f"Wrote {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Build features and generate biased dataset")
    parser.add_argument("--mask-rate", "-m", type=float, default=0.3, help="Fraction of rural apnea to mask (default: 0.3)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("Sleep Apnea v2: Build Features & Generate Bias")
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
    print(f"  Features: age, male, urban, income, bmi, smoker, hypertension, chf, alcohol_use")
    print(f"  Patients with sleep apnea: {df['has_sleep_apnea'].sum():,} ({100*df['has_sleep_apnea'].mean():.2f}%)")

    # Add mask flag
    print(f"\nApplying {args.mask_rate:.0%} mask rate to rural patients...")
    df = add_mask_flag(df, args.mask_rate, args.seed)
    n_masked = df["mask_sleep_apnea"].sum()
    print(f"  Patients masked: {n_masked:,}")

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
    print(f"\nTrue prevalence:     {before_stats['pct_apnea']:.2f}%")
    print(f"Observed prevalence: {after_stats['pct_apnea']:.2f}%")
    print(f"\nRural effect:")
    print(f"  Before: {before_stats['pct_apnea_rural']:.2f}%")
    print(f"  After:  {after_stats['pct_apnea_rural']:.2f}%")
    print(f"\nUrban (unchanged):")
    print(f"  Before: {before_stats['pct_apnea_urban']:.2f}%")
    print(f"  After:  {after_stats['pct_apnea_urban']:.2f}%")

    print("\n" + "=" * 60)
    print(f"Complete! Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
