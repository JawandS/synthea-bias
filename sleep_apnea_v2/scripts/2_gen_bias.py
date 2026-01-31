#!/usr/bin/env python3
"""
2_gen_bias.py - Generate biased dataset by masking rural sleep apnea diagnoses.

This script:
1. Adds `has_sleep_apnea` boolean to patients based on conditions data
2. Adds `mask_sleep_apnea` boolean for rural patients randomly selected as "underdiagnosed"
3. Outputs bias_effect.md with before/after prevalence statistics

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

# Sleep apnea SNOMED codes
SLEEP_APNEA_CODES = {"73430006", "78275009"}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load patients and conditions data."""
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    conditions = pd.read_csv(DATA_DIR / "conditions.csv")
    return patients, conditions


def add_sleep_apnea_flag(patients: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """Add has_sleep_apnea boolean flag to patients."""
    patients = patients.copy()

    # Find patients with sleep apnea diagnosis
    apnea_patients = set(
        conditions[conditions["CODE"].astype(str).isin(SLEEP_APNEA_CODES)]["PATIENT"].unique()
    )

    patients["has_sleep_apnea"] = patients["Id"].isin(apnea_patients)
    return patients


def add_mask_flag(patients: pd.DataFrame, mask_rate: float, seed: int) -> pd.DataFrame:
    """Add mask_sleep_apnea boolean flag for underdiagnosed rural patients."""
    patients = patients.copy()
    rng = np.random.default_rng(seed)

    # Initialize all as False
    patients["mask_sleep_apnea"] = False

    # Find rural patients with sleep apnea
    rural_apnea_mask = (patients["URBAN"] == False) & (patients["has_sleep_apnea"] == True)
    rural_apnea_indices = patients[rural_apnea_mask].index

    # Randomly select patients to mask
    n_to_mask = int(len(rural_apnea_indices) * mask_rate)
    masked_indices = rng.choice(rural_apnea_indices, size=n_to_mask, replace=False)

    patients.loc[masked_indices, "mask_sleep_apnea"] = True
    return patients


def compute_prevalence_stats(patients: pd.DataFrame, use_masked: bool = False) -> dict:
    """Compute prevalence statistics.

    Args:
        patients: DataFrame with has_sleep_apnea and mask_sleep_apnea columns
        use_masked: If True, treat masked cases as not having sleep apnea (biased view)
    """
    patients = patients.copy()

    # Determine effective sleep apnea status
    if use_masked:
        patients["effective_apnea"] = patients["has_sleep_apnea"] & ~patients["mask_sleep_apnea"]
    else:
        patients["effective_apnea"] = patients["has_sleep_apnea"]

    n_total = len(patients)
    n_apnea = patients["effective_apnea"].sum()

    # Calculate age decades
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
    reference_date = patients["BIRTHDATE"].max() + pd.DateOffset(years=70)
    patients["age"] = ((reference_date - patients["BIRTHDATE"]).dt.days / 365.25).astype(int)
    patients["decade"] = pd.cut(
        patients["age"],
        bins=[0, 69, 79, 89, 150],
        labels=["60-69", "70-79", "80-89", "90+"]
    )

    stats = {
        "n_total": n_total,
        "n_apnea": int(n_apnea),
        "pct_apnea": 100 * n_apnea / n_total if n_total else 0,
    }

    # By location
    for loc_name, loc_val in [("urban", True), ("rural", False)]:
        subset = patients[patients["URBAN"] == loc_val]
        n = len(subset)
        n_apnea_loc = subset["effective_apnea"].sum()
        stats[f"n_{loc_name}"] = n
        stats[f"n_apnea_{loc_name}"] = int(n_apnea_loc)
        stats[f"pct_apnea_{loc_name}"] = 100 * n_apnea_loc / n if n else 0

    # By gender
    for gender_name, gender_val in [("male", "M"), ("female", "F")]:
        subset = patients[patients["GENDER"] == gender_val]
        n = len(subset)
        n_apnea_g = subset["effective_apnea"].sum()
        stats[f"n_{gender_name}"] = n
        stats[f"n_apnea_{gender_name}"] = int(n_apnea_g)
        stats[f"pct_apnea_{gender_name}"] = 100 * n_apnea_g / n if n else 0

    # By gender x location
    for gender_name, gender_val in [("male", "M"), ("female", "F")]:
        for loc_name, loc_val in [("urban", True), ("rural", False)]:
            subset = patients[(patients["GENDER"] == gender_val) & (patients["URBAN"] == loc_val)]
            n = len(subset)
            n_apnea_gl = subset["effective_apnea"].sum()
            key = f"{gender_name}_{loc_name}"
            stats[f"n_{key}"] = n
            stats[f"n_apnea_{key}"] = int(n_apnea_gl)
            stats[f"pct_apnea_{key}"] = 100 * n_apnea_gl / n if n else 0

    # By decade
    decade_stats = []
    for decade in ["60-69", "70-79", "80-89", "90+"]:
        subset = patients[patients["decade"] == decade]
        n = len(subset)
        n_apnea_d = subset["effective_apnea"].sum()
        decade_stats.append({
            "decade": decade,
            "n": n,
            "n_apnea": int(n_apnea_d),
            "pct": 100 * n_apnea_d / n if n else 0,
        })
    stats["decade_stats"] = decade_stats

    return stats


def write_bias_effect_report(
    patients: pd.DataFrame,
    before_stats: dict,
    after_stats: dict,
    mask_rate: float,
    seed: int
) -> None:
    """Write bias effect report to markdown file."""
    md_path = INFO_DIR / "bias_effect.md"

    # Masking summary
    n_rural_apnea = len(patients[(patients["URBAN"] == False) & (patients["has_sleep_apnea"] == True)])
    n_masked = patients["mask_sleep_apnea"].sum()

    # Build comparison rows for various breakdowns
    def comparison_row(label: str, before_key: str, after_key: str) -> str:
        b_n = before_stats.get(f"n_apnea_{before_key}", before_stats.get(f"n_{before_key}", 0))
        b_pct = before_stats.get(f"pct_apnea_{before_key}", before_stats.get(f"pct_{before_key}", 0))
        a_n = after_stats.get(f"n_apnea_{after_key}", after_stats.get(f"n_{after_key}", 0))
        a_pct = after_stats.get(f"pct_apnea_{after_key}", after_stats.get(f"pct_{after_key}", 0))

        # Handle the case where keys are for totals
        if before_key == "total":
            b_n = before_stats["n_apnea"]
            b_pct = before_stats["pct_apnea"]
            a_n = after_stats["n_apnea"]
            a_pct = after_stats["pct_apnea"]

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
{comparison_row("Urban", "urban", "urban")}
{comparison_row("Rural", "rural", "rural")}

## Prevalence by Gender

| Gender | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
{comparison_row("Male", "male", "male")}
{comparison_row("Female", "female", "female")}

## Prevalence by Gender and Location

| Group | Before Cases | Before % | After Cases | After % | Change |
|-------|--------------|----------|-------------|---------|--------|
{comparison_row("Male Urban", "male_urban", "male_urban")}
{comparison_row("Male Rural", "male_rural", "male_rural")}
{comparison_row("Female Urban", "female_urban", "female_urban")}
{comparison_row("Female Rural", "female_rural", "female_rural")}

## Prevalence by Age Decade

| Decade | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
{decade_table}

## Key Observations

- **True prevalence**: {before_stats['pct_apnea']:.2f}% of all patients have sleep apnea
- **Observed prevalence**: {after_stats['pct_apnea']:.2f}% after rural underdiagnosis bias
- **Rural underdiagnosis**: {before_stats['pct_apnea_rural']:.2f}% → {after_stats['pct_apnea_rural']:.2f}% ({after_stats['pct_apnea_rural'] - before_stats['pct_apnea_rural']:+.2f}%)
- **Urban (unaffected)**: {before_stats['pct_apnea_urban']:.2f}% → {after_stats['pct_apnea_urban']:.2f}%
"""

    md_path.write_text(content)
    print(f"Wrote {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate biased dataset by masking rural diagnoses")
    parser.add_argument("--mask-rate", "-m", type=float, default=0.3, help="Fraction of rural apnea to mask (default: 0.3)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("Sleep Apnea v2: Generate Bias")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    patients, conditions = load_data()
    print(f"  Patients: {len(patients):,}")
    print(f"  Conditions: {len(conditions):,}")

    # Add sleep apnea flag
    print("\nAdding sleep apnea flags...")
    patients = add_sleep_apnea_flag(patients, conditions)
    n_apnea = patients["has_sleep_apnea"].sum()
    print(f"  Patients with sleep apnea: {n_apnea:,} ({100*n_apnea/len(patients):.2f}%)")

    # Add mask flag
    print(f"\nApplying {args.mask_rate:.0%} mask rate to rural patients...")
    patients = add_mask_flag(patients, args.mask_rate, args.seed)
    n_masked = patients["mask_sleep_apnea"].sum()
    print(f"  Patients masked: {n_masked:,}")

    # Compute before/after stats
    print("\nComputing prevalence statistics...")
    before_stats = compute_prevalence_stats(patients, use_masked=False)
    after_stats = compute_prevalence_stats(patients, use_masked=True)

    # Save updated patients
    patients.to_csv(DATA_DIR / "patients.csv", index=False)
    print(f"\nUpdated {DATA_DIR / 'patients.csv'}")

    # Write report
    write_bias_effect_report(patients, before_stats, after_stats, args.mask_rate, args.seed)

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
    print(f"Complete! See {INFO_DIR / 'bias_effect.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
