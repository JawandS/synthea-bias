#!/usr/bin/env python3
"""
1_generate_data.py - Generate synthetic patient data using Synthea.

This script:
1. Runs Synthea to generate a baseline population with sleep apnea cases
2. Copies relevant CSV files to the local data/ directory
3. Adds urban/rural flag to patients based on county FIPS codes

Usage:
    uv run python scripts/1_generate_data.py [--population N] [--seed N] [--skip-synthea]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Paths (relative to this script's location in scripts/)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
SYNTHEA_DIR = PROJECT_DIR.parent / "synthea"
DATA_DIR = PROJECT_DIR / "data"

# Synthea output directory (inside synthea folder)
SYNTHEA_OUTPUT_DIR = SYNTHEA_DIR / "output_sleep_apnea_v2"

# Files to copy from Synthea output
REQUIRED_FILES = ["patients.csv", "conditions.csv", "observations.csv"]

# Montana rural counties (FIPS codes for counties with population < 50k)
# Source: US Census Bureau urban/rural classification
RURAL_FIPS = {
    "30001", "30003", "30005", "30007", "30009", "30011", "30013", "30015",
    "30017", "30019", "30021", "30023", "30025", "30027", "30029", "30031",
    "30033", "30035", "30037", "30039", "30041", "30043", "30045", "30047",
    "30049", "30051", "30053", "30055", "30057", "30059", "30061", "30063",
    "30065", "30067", "30069", "30071", "30073", "30075", "30077", "30079",
    "30081", "30083", "30085", "30087", "30089", "30091", "30093", "30095",
    "30097", "30099", "30101", "30103", "30105", "30107", "30109", "30111",
}


def run_synthea(population: int, seed: int) -> None:
    """Run Synthea to generate synthetic patient data.

    Args:
        population: Number of patients to generate
        seed: Random seed for reproducibility
    """
    print(f"Running Synthea: {population} patients, seed={seed}")
    print(f"Output directory: {SYNTHEA_OUTPUT_DIR}")

    cmd = [
        "./run_synthea",
        "-s", str(seed),
        "-cs", str(seed),
        "-p", str(population),
        "-a", "60-100",  # Ages 60-100 ensures all patients evaluated for sleep apnea
        "--exporter.csv.export=true",
        "--exporter.csv.append_mode=false",
        "--exporter.fhir.export=false",
        "--exporter.ccda.export=false",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        "--exporter.years_of_history=0",  # Export full patient history
        f"--exporter.baseDirectory={SYNTHEA_OUTPUT_DIR.name}",
        "Montana",  # State with rural counties
    ]

    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=SYNTHEA_DIR,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"Synthea failed with return code {result.returncode}", file=sys.stderr)
        sys.exit(1)

    print("Synthea completed successfully")


def copy_and_process_data() -> None:
    """Copy CSV files from Synthea output and add urban/rural flag."""
    csv_dir = SYNTHEA_OUTPUT_DIR / "csv"

    if not csv_dir.exists():
        print(f"Error: Synthea output not found at {csv_dir}", file=sys.stderr)
        sys.exit(1)

    # Create data directory if needed
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Copy required files
    for filename in REQUIRED_FILES:
        src = csv_dir / filename
        dst = DATA_DIR / filename

        if not src.exists():
            print(f"Warning: {filename} not found in Synthea output", file=sys.stderr)
            continue

        shutil.copy2(src, dst)
        print(f"Copied {filename} ({src.stat().st_size:,} bytes)")

    # Add urban/rural flag to patients
    patients_path = DATA_DIR / "patients.csv"
    if patients_path.exists():
        add_urban_flag(patients_path)


def add_urban_flag(patients_path: Path) -> None:
    """Add 'urban' column to patients CSV based on county FIPS code.

    Args:
        patients_path: Path to patients.csv file
    """
    print("Adding urban/rural flag to patients...")

    df = pd.read_csv(patients_path, dtype={"FIPS": str})

    # Determine urban status based on FIPS code
    # Urban = True if FIPS not in rural set
    df["urban"] = ~df["FIPS"].isin(RURAL_FIPS)

    # Save updated file
    df.to_csv(patients_path, index=False)

    # Print summary
    n_urban = df["urban"].sum()
    n_rural = len(df) - n_urban
    print(f"  Urban patients: {n_urban:,} ({100*n_urban/len(df):.1f}%)")
    print(f"  Rural patients: {n_rural:,} ({100*n_rural/len(df):.1f}%)")


def print_summary() -> None:
    """Print summary statistics of the generated data."""
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)

    # Load data
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    conditions = pd.read_csv(DATA_DIR / "conditions.csv")

    print(f"\nPatients: {len(patients):,}")

    # Sleep apnea codes (SNOMED)
    apnea_codes = {"73430006", "78275009"}
    apnea_conditions = conditions[conditions["CODE"].astype(str).isin(apnea_codes)]
    apnea_patients = apnea_conditions["PATIENT"].nunique()

    print(f"Sleep apnea cases: {apnea_patients:,} ({100*apnea_patients/len(patients):.2f}%)")

    # Urban/rural breakdown
    if "urban" in patients.columns:
        urban_patients = patients[patients["urban"] == True]
        rural_patients = patients[patients["urban"] == False]

        urban_apnea = apnea_conditions[
            apnea_conditions["PATIENT"].isin(urban_patients["Id"])
        ]["PATIENT"].nunique()
        rural_apnea = apnea_conditions[
            apnea_conditions["PATIENT"].isin(rural_patients["Id"])
        ]["PATIENT"].nunique()

        print(f"\nUrban patients: {len(urban_patients):,}")
        print(f"  Sleep apnea: {urban_apnea:,} ({100*urban_apnea/len(urban_patients):.2f}%)")
        print(f"\nRural patients: {len(rural_patients):,}")
        print(f"  Sleep apnea: {rural_apnea:,} ({100*rural_apnea/len(rural_patients):.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient data using Synthea"
    )
    parser.add_argument(
        "--population", "-p",
        type=int,
        default=5000,
        help="Number of patients to generate (default: 5000)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-synthea",
        action="store_true",
        help="Skip Synthea generation, only copy existing data"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Sleep Apnea v2: Data Generation")
    print("=" * 60)

    if not args.skip_synthea:
        run_synthea(args.population, args.seed)
    else:
        print("Skipping Synthea generation (--skip-synthea)")

    copy_and_process_data()
    print_summary()

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print(f"Output directory: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
