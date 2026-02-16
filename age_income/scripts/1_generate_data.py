#!/usr/bin/env python3
"""
1_generate_data.py - Generate synthetic colorectal data using Synthea.

This script:
1. Runs Synthea for a baseline population (Montana, ages 40-100)
2. Copies relevant CSV files to output/data/
3. Filters conditions/procedures/observations to CRC + modeling-relevant records
4. Writes summary statistics to output/info/1_summary_stats.md

Usage:
    uv run python scripts/1_generate_data.py [--population N] [--seed N] [--skip-synthea]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
SYNTHEA_DIR = REPO_ROOT / "synthea"

OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"

DEFAULT_SYNTHEA_OUTPUT_DIR = SYNTHEA_DIR / "output_age_income"

RELEVANT_FILES = ["patients.csv", "conditions.csv", "observations.csv"]

# CRC stage codes used in synthea colorectal_cancer.json
CRC_STAGE_CODE_MAP = {
    "93761005": 1,
    "109838007": 2,
    "363406005": 3,
    "94260004": 4,
}

# Comorbidities used as model features
DIABETES_CODE = "44054006"
PREDIABETES_CODE = "714628002"
OBESITY_CODE = "162864005"
HYPERTENSION_CODE = "59621000"
HYPERLIPIDEMIA_CODE = "55822004"
CHF_CODE = "88805009"

CONDITION_CODES_TO_KEEP = set(CRC_STAGE_CODE_MAP.keys()) | {
    DIABETES_CODE,
    PREDIABETES_CODE,
    OBESITY_CODE,
    HYPERTENSION_CODE,
    HYPERLIPIDEMIA_CODE,
    CHF_CODE,
}

# Observation features used downstream
BMI_CODE = "39156-5"
SMOKING_CODE = "72166-2"
OBSERVATION_CODES_TO_KEEP = {BMI_CODE, SMOKING_CODE}


def run_synthea(population: int, seed: int, synthea_output_dir: Path) -> None:
    """Run Synthea to generate baseline data."""
    if synthea_output_dir.exists():
        print(f"Clearing old output: {synthea_output_dir}")
        shutil.rmtree(synthea_output_dir)

    cmd = [
        "./run_synthea",
        "-s", str(seed),
        "-cs", str(seed),
        "-p", str(population),
        "-a", "40-100",
        "--generate.only_alive_patients=false",
        "--exporter.csv.export=true",
        "--exporter.csv.append_mode=false",
        "--exporter.fhir.export=false",
        "--exporter.ccda.export=false",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        "--exporter.metadata.export=false",
        "--exporter.years_of_history=10",
        f"--exporter.baseDirectory={synthea_output_dir.name}",
        "Montana",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=SYNTHEA_DIR)
    if result.returncode != 0:
        print(f"Synthea failed with return code {result.returncode}", file=sys.stderr)
        sys.exit(1)


def infer_reference_date(conditions: pd.DataFrame, observations: pd.DataFrame) -> pd.Timestamp:
    """Infer a consistent reference date from event timestamps."""
    candidates: list[pd.Timestamp] = []

    for df, col in [(conditions, "START"), (observations, "DATE")]:
        if col in df.columns and len(df) > 0:
            series = pd.to_datetime(df[col], errors="coerce", utc=True).dropna()
            series = series.dt.tz_convert(None)
            if len(series) > 0:
                candidates.append(series.max())

    if not candidates:
        return pd.Timestamp(datetime.utcnow().date())
    return max(candidates)


def copy_and_filter_data(
    synthea_output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Copy and filter CSV files from Synthea output."""
    csv_dir = synthea_output_dir / "csv"
    if not csv_dir.exists():
        print(f"Synthea output not found: {csv_dir}", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INFO_DIR.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for filename in RELEVANT_FILES:
        src = csv_dir / filename
        if not src.exists():
            print(f"Missing source file: {filename}", file=sys.stderr)
            continue

        df = pd.read_csv(src)
        if filename == "conditions.csv":
            codes = df["CODE"].astype(str)
            df = df[codes.isin(CONDITION_CODES_TO_KEEP)].copy()
        elif filename == "observations.csv":
            codes = df["CODE"].astype(str)
            df = df[codes.isin(OBSERVATION_CODES_TO_KEEP)].copy()

        dst = DATA_DIR / filename
        df.to_csv(dst, index=False)
        print(f"Wrote {dst} ({len(df):,} rows)")
        dfs[filename] = df

    patients = dfs.get("patients.csv", pd.DataFrame())
    conditions = dfs.get("conditions.csv", pd.DataFrame())
    observations = dfs.get("observations.csv", pd.DataFrame())
    return patients, conditions, observations


def write_summary_stats(
    patients: pd.DataFrame,
    conditions: pd.DataFrame,
    observations: pd.DataFrame,
    population: int,
    seed: int,
) -> None:
    """Write markdown summary stats for baseline data."""
    if len(patients) == 0:
        print("No patients.csv found; skipping summary report", file=sys.stderr)
        return

    patients = patients.copy()
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")

    reference_date = infer_reference_date(conditions, observations)
    patients["age"] = ((reference_date - patients["BIRTHDATE"]).dt.days / 365.25).fillna(0).astype(int)

    stage_by_patient = {}
    codes = conditions["CODE"].astype(str) if len(conditions) > 0 else pd.Series(dtype=str)
    if len(conditions) > 0:
        cond = conditions.copy()
        cond["stage"] = cond["CODE"].astype(str).map(CRC_STAGE_CODE_MAP)
        cond = cond[cond["stage"].notna()].copy()
        stage_by_patient = cond.groupby("PATIENT")["stage"].max().to_dict()

    patients["has_crc_true"] = patients["Id"].isin(set(stage_by_patient.keys()))
    patients["crc_stage_true"] = patients["Id"].map(stage_by_patient)

    total = len(patients)
    n_crc = int(patients["has_crc_true"].sum())
    n_early = int((patients["crc_stage_true"].fillna(99) <= 2).sum())
    n_diabetes = int((codes == DIABETES_CODE).sum())
    n_prediabetes = int((codes == PREDIABETES_CODE).sum())
    n_obesity = int((codes == OBESITY_CODE).sum())
    n_hypertension = int((codes == HYPERTENSION_CODE).sum())
    n_hyperlipidemia = int((codes == HYPERLIPIDEMIA_CODE).sum())
    n_chf = int((codes == CHF_CODE).sum())

    age_band = pd.cut(
        patients["age"],
        bins=[0, 49, 59, 69, 79, 1000],
        labels=["40-49", "50-59", "60-69", "70-79", "80+"],
    )

    rows = []
    for band in ["40-49", "50-59", "60-69", "70-79", "80+"]:
        subset = patients[age_band == band]
        n = len(subset)
        n_band_crc = int(subset["has_crc_true"].sum())
        pct = (100 * n_band_crc / n) if n else 0.0
        rows.append(f"| {band} | {n:,} | {n_band_crc:,} | {pct:.2f}% |")

    content = f"""# Baseline Summary Stats

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Run Parameters

| Parameter | Value |
|-----------|-------|
| Requested population | {population:,} |
| Seed | {seed} |
| Reference date for age | {reference_date.date()} |

## Cohort Summary

| Metric | Value |
|--------|-------|
| Patients | {total:,} |
| True CRC cases | {n_crc:,} |
| True CRC prevalence | {(100 * n_crc / total) if total else 0:.2f}% |
| Early-stage CRC (I/II) | {n_early:,} |
| Early among CRC cases | {(100 * n_early / n_crc) if n_crc else 0:.2f}% |
| BMI/smoking observations kept | {len(observations):,} |

## Condition Rows Kept

| Condition | Rows |
|-----------|------|
| Diabetes | {n_diabetes:,} |
| Prediabetes | {n_prediabetes:,} |
| Obesity | {n_obesity:,} |
| Hypertension | {n_hypertension:,} |
| Hyperlipidemia | {n_hyperlipidemia:,} |
| CHF | {n_chf:,} |

## CRC by Age Band

| Age band | Patients | CRC cases | Prevalence |
|----------|----------|-----------|------------|
{chr(10).join(rows)}
"""

    (INFO_DIR / "1_summary_stats.md").write_text(content)
    print(f"Wrote {INFO_DIR / '1_summary_stats.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline colorectal dataset")
    parser.add_argument("-p", "--population", type=int, default=20000, help="Population size")
    parser.add_argument("-s", "--seed", type=int, default=160, help="Random seed")
    parser.add_argument("--skip-synthea", action="store_true", help="Skip generation and reuse existing output")
    parser.add_argument(
        "--synthea-output-dir",
        type=Path,
        default=DEFAULT_SYNTHEA_OUTPUT_DIR,
        help="Path to Synthea output directory (must contain csv/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthea_output_dir = args.synthea_output_dir.resolve()

    if not args.skip_synthea:
        run_synthea(population=args.population, seed=args.seed, synthea_output_dir=synthea_output_dir)

    patients, conditions, observations = copy_and_filter_data(synthea_output_dir=synthea_output_dir)
    write_summary_stats(
        patients=patients,
        conditions=conditions,
        observations=observations,
        population=args.population,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
