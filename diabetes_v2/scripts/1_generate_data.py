#!/usr/bin/env python3
"""
1_generate_data.py - Generate synthetic patient data using Synthea.

This script:
1. Runs Synthea to generate a baseline population with diabetes cases
2. Copies relevant CSV files to output/data/
3. Filters observations to only include relevant codes (A1c, BMI, smoking)
4. Filters conditions to only include relevant codes (diabetes, metabolic conditions)
5. Generates summary statistics in output/info/1_summary_stats.md

Usage:
    uv run python scripts/1_generate_data.py [--population N] [--seed N] [--skip-synthea]

Options:
    --population, -p N   Number of patients to generate (default: 20000)
    --seed, -s N         Random seed for reproducibility (default: 160)
    --skip-synthea       Skip Synthea generation, only reprocess existing data

Optimizations:
    - Limits exported history to 5 years (not full patient lifetime)
    - Disables FHIR, C-CDA, and metadata exports
    - Filters conditions.csv to only relevant codes
    - Filters observations.csv to A1c + BMI + smoking codes only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Paths (relative to this script's location in scripts/)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
SYNTHEA_DIR = REPO_ROOT / "synthea"

# Output directories
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"

# Synthea output directory
SYNTHEA_OUTPUT_DIR = SYNTHEA_DIR / "output_diabetes_v2"

# Files to copy from Synthea output
RELEVANT_FILES = ["patients.csv", "conditions.csv", "observations.csv"]

# Observation codes to keep (A1c, BMI, and smoking status)
OBSERVATION_CODES = {
    "4548-4",   # Hemoglobin A1c
    "39156-5",  # BMI
    "72166-2",  # Smoking status
}

# Condition codes
DIABETES_CODE = "44054006"           # Diabetes mellitus type 2
PREDIABETES_CODE = "714628002"       # Prediabetes
HYPERGLYCEMIA_CODE = "80394007"      # Hyperglycemia
HYPERTRIGLYCERIDEMIA_CODE = "302870006"  # Hypertriglyceridemia
METABOLIC_SYNDROME_CODE = "237602007"    # Metabolic syndrome
OBESITY_CODE = "162864005"           # Obesity finding
HYPERTENSION_CODE = "59621000"       # Hypertension
HYPERLIPIDEMIA_CODE = "55822004"     # Hyperlipidemia

# Condition codes to keep for modeling
CONDITION_CODES = {
    DIABETES_CODE,
    PREDIABETES_CODE,
    HYPERGLYCEMIA_CODE,
    HYPERTRIGLYCERIDEMIA_CODE,
    METABOLIC_SYNDROME_CODE,
    OBESITY_CODE,
    HYPERTENSION_CODE,
    HYPERLIPIDEMIA_CODE,
}


def run_synthea(population: int, seed: int) -> None:
    """Run Synthea to generate synthetic patient data.

    Args:
        population: Number of patients to generate
        seed: Random seed for reproducibility
    """
    if SYNTHEA_OUTPUT_DIR.exists():
        print(f"Clearing old output: {SYNTHEA_OUTPUT_DIR}")
        shutil.rmtree(SYNTHEA_OUTPUT_DIR)

    print(f"Running Synthea: {population} patients, seed={seed}")

    cmd = [
        "./run_synthea",
        "-s", str(seed),
        "-cs", str(seed),
        "-p", str(population),
        "-a", "40-100",
        "--generate.only_alive_patients=false",
        # CSV export settings
        "--exporter.csv.export=true",
        "--exporter.csv.append_mode=false",
        # Disable all other exporters
        "--exporter.fhir.export=false",
        "--exporter.ccda.export=false",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        "--exporter.metadata.export=false",
        # Limit history to 5 years (0 = unlimited, which generates more data)
        "--exporter.years_of_history=5",
        f"--exporter.baseDirectory={SYNTHEA_OUTPUT_DIR.name}",
        # State (positional argument, must come last)
        "Montana",
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SYNTHEA_DIR)

    if result.returncode != 0:
        print(f"Synthea failed with return code {result.returncode}", file=sys.stderr)
        sys.exit(1)

    print("Synthea completed successfully")


def filter_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only observations needed for modeling (A1c + BMI + smoking status)."""
    return df[df["CODE"].astype(str).isin(OBSERVATION_CODES)].copy()


def filter_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only conditions needed for modeling."""
    return df[df["CODE"].astype(str).isin(CONDITION_CODES)].copy()


def copy_and_process_data() -> None:
    """Copy CSV files from Synthea output, filter observations and conditions."""
    csv_dir = SYNTHEA_OUTPUT_DIR / "csv"

    if not csv_dir.exists():
        print(f"Error: Synthea output not found at {csv_dir}", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INFO_DIR.mkdir(parents=True, exist_ok=True)

    for filename in RELEVANT_FILES:
        src = csv_dir / filename
        dst = DATA_DIR / filename

        if not src.exists():
            print(f"Warning: {filename} not found", file=sys.stderr)
            continue

        df = pd.read_csv(src)

        if filename == "observations.csv":
            original_len = len(df)
            df = filter_observations(df)
            print(f"  Filtered observations: {original_len:,} -> {len(df):,}")
        elif filename == "conditions.csv":
            original_len = len(df)
            df = filter_conditions(df)
            print(f"  Filtered conditions: {original_len:,} -> {len(df):,}")

        df.to_csv(dst, index=False)
        print(f"Wrote {filename} ({dst.stat().st_size:,} bytes)")


def compute_stats(patients: pd.DataFrame, conditions: pd.DataFrame, observations: pd.DataFrame) -> dict:
    """Compute summary statistics from the data."""
    n_patients = len(patients)
    codes = conditions["CODE"].astype(str)

    # Calculate age from birthdate
    patients = patients.copy()
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
    reference_date = patients["BIRTHDATE"].max() + pd.DateOffset(years=50)  # Approximate reference
    patients["age"] = ((reference_date - patients["BIRTHDATE"]).dt.days / 365.25).astype(int)

    # Age decades
    patients["decade"] = pd.cut(
        patients["age"],
        bins=[0, 49, 59, 69, 79, 89, 150],
        labels=["40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    )

    # Helper to get patient IDs with a condition
    def patient_ids_with_code(code_set: set) -> set:
        return set(conditions[codes.isin(code_set)]["PATIENT"].unique())

    # Gender counts
    male = patients[patients["GENDER"] == "M"]
    female = patients[patients["GENDER"] == "F"]

    # Condition counts
    diabetes_ids = patient_ids_with_code({DIABETES_CODE})
    hyperglycemia_ids = patient_ids_with_code({HYPERGLYCEMIA_CODE})
    hypertriglyceridemia_ids = patient_ids_with_code({HYPERTRIGLYCERIDEMIA_CODE})
    n_diabetes = len(diabetes_ids)
    n_hyperglycemia = len(hyperglycemia_ids)
    n_hypertriglyceridemia = len(hypertriglyceridemia_ids)
    n_prediabetes = len(patient_ids_with_code({PREDIABETES_CODE}))
    n_obesity = len(patient_ids_with_code({OBESITY_CODE}))
    n_hypertension = len(patient_ids_with_code({HYPERTENSION_CODE}))
    n_hyperlipidemia = len(patient_ids_with_code({HYPERLIPIDEMIA_CODE}))

    # Gender diabetes breakdown
    male_diabetes = len(diabetes_ids & set(male["Id"]))
    female_diabetes = len(diabetes_ids & set(female["Id"]))

    # Age decade breakdown for diabetes
    decade_stats = []
    for decade in ["40-49", "50-59", "60-69", "70-79", "80-89", "90+"]:
        decade_patients = patients[patients["decade"] == decade]
        n_decade = len(decade_patients)
        decade_diabetes = len(diabetes_ids & set(decade_patients["Id"]))
        decade_stats.append({
            "decade": decade,
            "n": n_decade,
            "diabetes": decade_diabetes,
            "pct": 100 * decade_diabetes / n_decade if n_decade else 0,
        })

    # Gender breakdown table
    gender_stats = []
    for gender in ["M", "F"]:
        for decade in ["40-49", "50-59", "60-69", "70-79", "80-89", "90+"]:
            subset = patients[
                (patients["GENDER"] == gender) &
                (patients["decade"] == decade)
            ]
            n_subset = len(subset)
            subset_diabetes = len(diabetes_ids & set(subset["Id"]))
            gender_stats.append({
                "gender": "Male" if gender == "M" else "Female",
                "decade": decade,
                "n": n_subset,
                "diabetes": subset_diabetes,
                "pct": 100 * subset_diabetes / n_subset if n_subset else 0,
            })

    # A1c stats (code 4548-4)
    a1c_obs = observations[observations["CODE"].astype(str) == "4548-4"]
    n_with_a1c = a1c_obs["PATIENT"].nunique()

    # BMI stats (code 39156-5)
    bmi_obs = observations[observations["CODE"].astype(str) == "39156-5"]
    n_with_bmi = bmi_obs["PATIENT"].nunique()

    # Smoking stats (code 72166-2)
    smoking_obs = observations[observations["CODE"].astype(str) == "72166-2"]
    n_with_smoking = smoking_obs["PATIENT"].nunique()

    return {
        "n_patients": n_patients,
        "n_male": len(male),
        "n_female": len(female),
        "pct_male": 100 * len(male) / n_patients if n_patients else 0,
        "pct_female": 100 * len(female) / n_patients if n_patients else 0,
        "n_diabetes": n_diabetes,
        "pct_diabetes": 100 * n_diabetes / n_patients if n_patients else 0,
        "n_prediabetes": n_prediabetes,
        "pct_prediabetes": 100 * n_prediabetes / n_patients if n_patients else 0,
        "n_hyperglycemia": n_hyperglycemia,
        "pct_hyperglycemia": 100 * n_hyperglycemia / n_patients if n_patients else 0,
        "n_hypertriglyceridemia": n_hypertriglyceridemia,
        "pct_hypertriglyceridemia": 100 * n_hypertriglyceridemia / n_patients if n_patients else 0,
        "n_obesity": n_obesity,
        "pct_obesity": 100 * n_obesity / n_patients if n_patients else 0,
        "n_hypertension": n_hypertension,
        "pct_hypertension": 100 * n_hypertension / n_patients if n_patients else 0,
        "n_hyperlipidemia": n_hyperlipidemia,
        "pct_hyperlipidemia": 100 * n_hyperlipidemia / n_patients if n_patients else 0,
        "male_diabetes": male_diabetes,
        "pct_male_diabetes": 100 * male_diabetes / len(male) if len(male) else 0,
        "female_diabetes": female_diabetes,
        "pct_female_diabetes": 100 * female_diabetes / len(female) if len(female) else 0,
        "decade_stats": decade_stats,
        "gender_stats": gender_stats,
        "n_with_a1c": n_with_a1c,
        "pct_with_a1c": 100 * n_with_a1c / n_patients if n_patients else 0,
        "n_with_bmi": n_with_bmi,
        "pct_with_bmi": 100 * n_with_bmi / n_patients if n_patients else 0,
        "n_with_smoking": n_with_smoking,
        "pct_with_smoking": 100 * n_with_smoking / n_patients if n_patients else 0,
    }


def write_summary_stats(stats: dict, population: int, seed: int) -> None:
    """Write summary statistics to markdown file."""
    md_path = INFO_DIR / "1_summary_stats.md"

    # Build decade table rows
    decade_rows = "\n".join(
        f"| {d['decade']} | {d['n']:,} | {d['diabetes']:,} | {d['pct']:.2f}% |"
        for d in stats["decade_stats"]
    )

    # Build gender x decade table rows
    gender_rows = "\n".join(
        f"| {d['gender']} | {d['decade']} | {d['n']:,} | {d['diabetes']:,} | {d['pct']:.2f}% |"
        for d in stats["gender_stats"]
    )

    content = f"""# Data Generation Summary

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Parameters

| Parameter | Value |
|-----------|-------|
| Population requested | {population:,} |
| Population generated | {stats['n_patients']:,} |
| Seed | {seed} |
| Age Range | 40-100 |
| State | Montana |

## Population Summary

| Metric | Count | Percentage |
|--------|------:|----------:|
| Total patients | {stats['n_patients']:,} | 100.00% |
| Male | {stats['n_male']:,} | {stats['pct_male']:.2f}% |
| Female | {stats['n_female']:,} | {stats['pct_female']:.2f}% |

## Condition Prevalence

| Condition | Count | Prevalence |
|-----------|------:|-----------:|
| Diabetes | {stats['n_diabetes']:,} | {stats['pct_diabetes']:.2f}% |
| Prediabetes | {stats['n_prediabetes']:,} | {stats['pct_prediabetes']:.2f}% |
| Hyperglycemia | {stats['n_hyperglycemia']:,} | {stats['pct_hyperglycemia']:.2f}% |
| Hypertriglyceridemia | {stats['n_hypertriglyceridemia']:,} | {stats['pct_hypertriglyceridemia']:.2f}% |
| Obesity | {stats['n_obesity']:,} | {stats['pct_obesity']:.2f}% |
| Hypertension | {stats['n_hypertension']:,} | {stats['pct_hypertension']:.2f}% |
| Hyperlipidemia | {stats['n_hyperlipidemia']:,} | {stats['pct_hyperlipidemia']:.2f}% |

## Diabetes by Gender

| Gender | Patients | Diabetes Cases | Prevalence |
|--------|----------|----------------|------------|
| Male | {stats['n_male']:,} | {stats['male_diabetes']:,} | {stats['pct_male_diabetes']:.2f}% |
| Female | {stats['n_female']:,} | {stats['female_diabetes']:,} | {stats['pct_female_diabetes']:.2f}% |

## Diabetes by Age Decade

| Age Group | Patients | Diabetes Cases | Prevalence |
|-----------|----------|----------------|------------|
{decade_rows}

## Diabetes by Gender and Age

| Gender | Age Group | Patients | Diabetes Cases | Prevalence |
|--------|-----------|----------|----------------|------------|
{gender_rows}

## Feature Availability

| Feature | Patients with Data | Coverage |
|---------|-------------------:|---------:|
| Hemoglobin A1c | {stats['n_with_a1c']:,} | {stats['pct_with_a1c']:.2f}% |
| BMI | {stats['n_with_bmi']:,} | {stats['pct_with_bmi']:.2f}% |
| Smoking status | {stats['n_with_smoking']:,} | {stats['pct_with_smoking']:.2f}% |
"""

    md_path.write_text(content)
    print(f"Wrote {md_path}")


def generate_summary(population: int, seed: int) -> None:
    """Generate and save summary statistics."""
    print("\n" + "=" * 60)
    print("Generating Summary Statistics")
    print("=" * 60)

    patients = pd.read_csv(DATA_DIR / "patients.csv")
    conditions = pd.read_csv(DATA_DIR / "conditions.csv")
    observations = pd.read_csv(DATA_DIR / "observations.csv")

    stats = compute_stats(patients, conditions, observations)
    write_summary_stats(stats, population, seed)

    # Print to console
    print(f"\nPatients: {stats['n_patients']:,}")
    print(f"  Male: {stats['n_male']:,} ({stats['pct_male']:.1f}%)")
    print(f"  Female: {stats['n_female']:,} ({stats['pct_female']:.1f}%)")
    print(f"\nDiabetes: {stats['n_diabetes']:,} ({stats['pct_diabetes']:.2f}%)")
    print(f"Hyperglycemia: {stats['n_hyperglycemia']:,} ({stats['pct_hyperglycemia']:.2f}%)")
    print(f"Hypertriglyceridemia: {stats['n_hypertriglyceridemia']:,} ({stats['pct_hypertriglyceridemia']:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patient data using Synthea")
    parser.add_argument("--population", "-p", type=int, default=20000, help="Number of patients (default: 20000)")
    parser.add_argument("--seed", "-s", type=int, default=160, help="Random seed (default: 160)")
    parser.add_argument("--skip-synthea", action="store_true", help="Skip Synthea, only copy existing data")
    args = parser.parse_args()

    print("=" * 60)
    print("Diabetes v2: Data Generation")
    print("=" * 60)

    if not args.skip_synthea:
        run_synthea(args.population, args.seed)
    else:
        print("Skipping Synthea generation (--skip-synthea)")

    copy_and_process_data()
    generate_summary(args.population, args.seed)

    print("\n" + "=" * 60)
    print(f"Complete! Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
