#!/usr/bin/env python3
"""
1_generate_data.py - Generate synthetic patient data using Synthea.

This script:
1. Runs Synthea to generate a baseline population with sleep apnea cases
2. Copies relevant CSV files to output/data/
3. Adds urban/rural flag to patients via SDoH county lookup
4. Filters observations to only include relevant codes (BMI, smoking)
5. Filters conditions to only include relevant codes (sleep apnea, hypertension, CHF, alcohol)
6. Generates summary statistics in output/info/1_summary_stats.md

Usage:
    uv run python scripts/1_generate_data.py [--population N] [--seed N] [--skip-synthea]

Options:
    --population, -p N   Number of patients to generate (default: 5000)
    --seed, -s N         Random seed for reproducibility (default: 42)
    --skip-synthea       Skip Synthea generation, only reprocess existing data

Optimizations:
    - Limits exported history to 5 years (not full patient lifetime)
    - Disables FHIR, C-CDA, and metadata exports
    - Filters conditions.csv to only relevant codes (reduces ~700MB to ~500KB)
    - Filters observations.csv to BMI + smoking codes only
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
SYNTHEA_OUTPUT_DIR = SYNTHEA_DIR / "output_sleep_apnea_v2"

# SDoH file with URBAN column
SDOH_PATH = SYNTHEA_DIR / "src" / "main" / "resources" / "geography" / "sdoh.csv"

# Files to copy from Synthea output
RELEVANT_FILES = ["patients.csv", "conditions.csv", "observations.csv"]

# Observation codes to keep (BMI and smoking status)
OBSERVATION_CODES = {"39156-5", "72166-2"}

# Condition codes
SLEEP_APNEA_CODES = {"73430006", "78275009"}
HYPERTENSION_CODE = "59621000"
CHF_CODE = "88805009"
OBESITY_CODE = "162864005"

# Condition codes to keep for modeling (sleep apnea, hypertension, CHF, alcohol use)
CONDITION_CODES = {
    "73430006",  # Obstructive sleep apnea
    "78275009",  # Obstructive sleep apnea syndrome
    "59621000",  # Hypertension
    "88805009",  # CHF
    "7200002",   # Alcohol use disorder
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
        "-a", "60-100",
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
        "Vermont",
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SYNTHEA_DIR)

    if result.returncode != 0:
        print(f"Synthea failed with return code {result.returncode}", file=sys.stderr)
        sys.exit(1)

    print("Synthea completed successfully")


def load_sdoh_urban_map() -> dict[tuple[str, str], bool]:
    """Return a mapping of (STATE, COUNTY) -> URBAN flag from SDoH data."""
    if not SDOH_PATH.exists():
        raise FileNotFoundError(f"SDoH file not found: {SDOH_PATH}")

    sdoh = pd.read_csv(SDOH_PATH, usecols=["STATE", "COUNTY", "URBAN"])
    sdoh["STATE"] = sdoh["STATE"].astype(str).str.strip().str.upper()
    sdoh["COUNTY"] = sdoh["COUNTY"].astype(str).str.strip().str.upper()
    sdoh = sdoh.dropna(subset=["STATE", "COUNTY", "URBAN"]).drop_duplicates(["STATE", "COUNTY"])

    return {
        (str(row["STATE"]), str(row["COUNTY"])): bool(row["URBAN"])
        for _, row in sdoh.iterrows()
    }


def append_urban_column(patients: pd.DataFrame, urban_map: dict[tuple[str, str], bool]) -> pd.DataFrame:
    """Append URBAN column to patients DataFrame via SDoH lookup."""
    patients = patients.copy()
    state_key = patients["STATE"].astype(str).str.strip().str.upper()
    county_key = patients["COUNTY"].astype(str).str.strip().str.upper()
    patients["URBAN"] = [
        urban_map.get((s, c)) for s, c in zip(state_key, county_key)
    ]
    return patients


def filter_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only observations needed for modeling (BMI + smoking status)."""
    return df[df["CODE"].astype(str).isin(OBSERVATION_CODES)].copy()


def filter_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only conditions needed for modeling."""
    return df[df["CODE"].astype(str).isin(CONDITION_CODES)].copy()


def copy_and_process_data() -> None:
    """Copy CSV files from Synthea output, add urban flag, filter observations."""
    csv_dir = SYNTHEA_OUTPUT_DIR / "csv"

    if not csv_dir.exists():
        print(f"Error: Synthea output not found at {csv_dir}", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INFO_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SDoH urban/rural mapping...")
    urban_map = load_sdoh_urban_map()
    print(f"  Loaded {len(urban_map):,} county mappings")

    for filename in RELEVANT_FILES:
        src = csv_dir / filename
        dst = DATA_DIR / filename

        if not src.exists():
            print(f"Warning: {filename} not found", file=sys.stderr)
            continue

        df = pd.read_csv(src)

        if filename == "patients.csv":
            df = append_urban_column(df, urban_map)
        elif filename == "observations.csv":
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
    reference_date = patients["BIRTHDATE"].max() + pd.DateOffset(years=70)  # Approximate reference
    patients["age"] = ((reference_date - patients["BIRTHDATE"]).dt.days / 365.25).astype(int)

    # Age decades
    patients["decade"] = pd.cut(
        patients["age"],
        bins=[0, 69, 79, 89, 150],
        labels=["60-69", "70-79", "80-89", "90+"]
    )

    # Helper to get patient IDs with a condition
    def patient_ids_with_code(code_set: set) -> set:
        return set(conditions[codes.isin(code_set)]["PATIENT"].unique())

    # Urban/rural counts
    urban = patients[patients["URBAN"] == True]
    rural = patients[patients["URBAN"] == False]
    n_urban = len(urban)
    n_rural = len(rural)

    # Gender counts
    male = patients[patients["GENDER"] == "M"]
    female = patients[patients["GENDER"] == "F"]

    # Condition counts
    apnea_ids = patient_ids_with_code(SLEEP_APNEA_CODES)
    n_apnea = len(apnea_ids)
    n_hypertension = len(patient_ids_with_code({HYPERTENSION_CODE}))
    n_chf = len(patient_ids_with_code({CHF_CODE}))

    # Urban/rural apnea breakdown
    urban_apnea = len(apnea_ids & set(urban["Id"]))
    rural_apnea = len(apnea_ids & set(rural["Id"]))

    # Gender apnea breakdown
    male_apnea = len(apnea_ids & set(male["Id"]))
    female_apnea = len(apnea_ids & set(female["Id"]))

    # Age decade breakdown
    decade_stats = []
    for decade in ["60-69", "70-79", "80-89", "90+"]:
        decade_patients = patients[patients["decade"] == decade]
        n_decade = len(decade_patients)
        decade_apnea = len(apnea_ids & set(decade_patients["Id"]))
        decade_stats.append({
            "decade": decade,
            "n": n_decade,
            "apnea": decade_apnea,
            "pct": 100 * decade_apnea / n_decade if n_decade else 0,
        })

    # Comprehensive cross-tabulation (gender x location x decade)
    cross_stats = []
    for gender in ["M", "F"]:
        for location in [True, False]:  # Urban, Rural
            for decade in ["60-69", "70-79", "80-89", "90+"]:
                subset = patients[
                    (patients["GENDER"] == gender) &
                    (patients["URBAN"] == location) &
                    (patients["decade"] == decade)
                ]
                n_subset = len(subset)
                subset_apnea = len(apnea_ids & set(subset["Id"]))
                cross_stats.append({
                    "gender": "Male" if gender == "M" else "Female",
                    "location": "Urban" if location else "Rural",
                    "decade": decade,
                    "n": n_subset,
                    "apnea": subset_apnea,
                    "pct": 100 * subset_apnea / n_subset if n_subset else 0,
                })

    # Gender x Location summary (without decade)
    gender_location_stats = []
    for gender in ["M", "F"]:
        for location in [True, False]:
            subset = patients[
                (patients["GENDER"] == gender) &
                (patients["URBAN"] == location)
            ]
            n_subset = len(subset)
            subset_apnea = len(apnea_ids & set(subset["Id"]))
            gender_location_stats.append({
                "gender": "Male" if gender == "M" else "Female",
                "location": "Urban" if location else "Rural",
                "n": n_subset,
                "apnea": subset_apnea,
                "pct": 100 * subset_apnea / n_subset if n_subset else 0,
            })

    # BMI stats (code 39156-5)
    bmi_obs = observations[observations["CODE"].astype(str) == "39156-5"]
    n_with_bmi = bmi_obs["PATIENT"].nunique()

    # Smoking stats (code 72166-2)
    smoking_obs = observations[observations["CODE"].astype(str) == "72166-2"]
    n_with_smoking = smoking_obs["PATIENT"].nunique()

    return {
        "n_patients": n_patients,
        "n_urban": n_urban,
        "n_rural": n_rural,
        "pct_urban": 100 * n_urban / n_patients if n_patients else 0,
        "pct_rural": 100 * n_rural / n_patients if n_patients else 0,
        "n_male": len(male),
        "n_female": len(female),
        "pct_male": 100 * len(male) / n_patients if n_patients else 0,
        "pct_female": 100 * len(female) / n_patients if n_patients else 0,
        "n_apnea": n_apnea,
        "pct_apnea": 100 * n_apnea / n_patients if n_patients else 0,
        "urban_apnea": urban_apnea,
        "pct_urban_apnea": 100 * urban_apnea / n_urban if n_urban else 0,
        "rural_apnea": rural_apnea,
        "pct_rural_apnea": 100 * rural_apnea / n_rural if n_rural else 0,
        "male_apnea": male_apnea,
        "pct_male_apnea": 100 * male_apnea / len(male) if len(male) else 0,
        "female_apnea": female_apnea,
        "pct_female_apnea": 100 * female_apnea / len(female) if len(female) else 0,
        "decade_stats": decade_stats,
        "cross_stats": cross_stats,
        "gender_location_stats": gender_location_stats,
        "n_hypertension": n_hypertension,
        "pct_hypertension": 100 * n_hypertension / n_patients if n_patients else 0,
        "n_chf": n_chf,
        "pct_chf": 100 * n_chf / n_patients if n_patients else 0,
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
        f"| {d['decade']} | {d['n']:,} | {d['apnea']:,} | {d['pct']:.2f}% |"
        for d in stats["decade_stats"]
    )

    # Build gender x location table rows
    gender_location_rows = "\n".join(
        f"| {d['gender']} | {d['location']} | {d['n']:,} | {d['apnea']:,} | {d['pct']:.2f}% |"
        for d in stats["gender_location_stats"]
    )

    # Build comprehensive cross-tab table rows
    cross_rows = "\n".join(
        f"| {d['gender']} | {d['location']} | {d['decade']} | {d['n']:,} | {d['apnea']:,} | {d['pct']:.2f}% |"
        for d in stats["cross_stats"]
    )

    content = f"""# Data Generation Summary

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Parameters

| Parameter | Value |
|-----------|-------|
| Population requested | {population:,} |
| Population generated | {stats['n_patients']:,} |
| Seed | {seed} |
| Age Range | 60-100 |
| State | Vermont |

## Population Summary

| Metric | Count | Percentage |
|--------|------:|----------:|
| Total patients | {stats['n_patients']:,} | 100.00% |
| Urban | {stats['n_urban']:,} | {stats['pct_urban']:.2f}% |
| Rural | {stats['n_rural']:,} | {stats['pct_rural']:.2f}% |
| Male | {stats['n_male']:,} | {stats['pct_male']:.2f}% |
| Female | {stats['n_female']:,} | {stats['pct_female']:.2f}% |

## Condition Prevalence

| Condition | Count | Prevalence |
|-----------|------:|-----------:|
| Sleep apnea | {stats['n_apnea']:,} | {stats['pct_apnea']:.2f}% |
| Hypertension | {stats['n_hypertension']:,} | {stats['pct_hypertension']:.2f}% |
| CHF | {stats['n_chf']:,} | {stats['pct_chf']:.2f}% |

## Sleep Apnea by Location

| Location | Patients | Apnea Cases | Prevalence |
|----------|----------|-------------|------------|
| Urban | {stats['n_urban']:,} | {stats['urban_apnea']:,} | {stats['pct_urban_apnea']:.2f}% |
| Rural | {stats['n_rural']:,} | {stats['rural_apnea']:,} | {stats['pct_rural_apnea']:.2f}% |

## Sleep Apnea by Gender

| Gender | Patients | Apnea Cases | Prevalence |
|--------|----------|-------------|------------|
| Male | {stats['n_male']:,} | {stats['male_apnea']:,} | {stats['pct_male_apnea']:.2f}% |
| Female | {stats['n_female']:,} | {stats['female_apnea']:,} | {stats['pct_female_apnea']:.2f}% |

## Sleep Apnea by Age Decade

| Age Group | Patients | Apnea Cases | Prevalence |
|-----------|----------|-------------|------------|
{decade_rows}

## Sleep Apnea by Gender and Location

| Gender | Location | Patients | Apnea Cases | Prevalence |
|--------|----------|----------|-------------|------------|
{gender_location_rows}

## Comprehensive Prevalence (Gender x Location x Age)

| Gender | Location | Age Group | Patients | Apnea Cases | Prevalence |
|--------|----------|-----------|----------|-------------|------------|
{cross_rows}

## Feature Availability

| Feature | Patients with Data | Coverage |
|---------|-------------------:|---------:|
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
    print(f"  Urban: {stats['n_urban']:,} ({stats['pct_urban']:.1f}%)")
    print(f"  Rural: {stats['n_rural']:,} ({stats['pct_rural']:.1f}%)")
    print(f"\nSleep apnea: {stats['n_apnea']:,} ({stats['pct_apnea']:.2f}%)")
    print(f"  Urban: {stats['urban_apnea']:,} ({stats['pct_urban_apnea']:.2f}%)")
    print(f"  Rural: {stats['rural_apnea']:,} ({stats['pct_rural_apnea']:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patient data using Synthea")
    parser.add_argument("--population", "-p", type=int, default=5000, help="Number of patients (default: 5000)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--skip-synthea", action="store_true", help="Skip Synthea, only copy existing data")
    args = parser.parse_args()

    print("=" * 60)
    print("Sleep Apnea v2: Data Generation")
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
