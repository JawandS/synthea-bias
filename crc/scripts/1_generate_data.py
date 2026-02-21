#!/usr/bin/env python3
"""Generate CRC case-study baseline data from Synthea outputs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
SYNTHEA_DIR = REPO_ROOT / "synthea"
SYNTHEA_OUTPUT_DIR = SYNTHEA_DIR / "output_crc"

OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"

RELEVANT_FILES = [
    "patients.csv",
    "conditions.csv",
    "procedures.csv",
    "observations.csv",
    "encounters.csv",
]

CRC_SCREENING_CODE = "73761001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CRC baseline data")
    parser.add_argument("-p", "--population", type=int, default=20000)
    parser.add_argument("-s", "--seed", type=int, default=160)
    parser.add_argument("--skip-synthea", action="store_true")
    return parser.parse_args()


def run_synthea(population: int, seed: int) -> None:
    if SYNTHEA_OUTPUT_DIR.exists():
        shutil.rmtree(SYNTHEA_OUTPUT_DIR)

    cmd = [
        "./run_synthea",
        "-s",
        str(seed),
        "-cs",
        str(seed),
        "-p",
        str(population),
        "-a",
        "50-100",
        "--generate.only_alive_patients=false",
        "--exporter.csv.export=true",
        "--exporter.csv.append_mode=false",
        "--exporter.fhir.export=false",
        "--exporter.ccda.export=false",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        "--exporter.metadata.export=false",
        "--exporter.years_of_history=10",
        f"--exporter.baseDirectory={SYNTHEA_OUTPUT_DIR.name}",
        "Montana",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=SYNTHEA_DIR)
    if result.returncode != 0:
        print(f"Synthea failed with code {result.returncode}", file=sys.stderr)
        sys.exit(1)


def infer_reference_date(conditions: pd.DataFrame, procedures: pd.DataFrame, observations: pd.DataFrame) -> pd.Timestamp:
    candidates: list[pd.Timestamp] = []
    for frame, col in [(conditions, "START"), (procedures, "DATE"), (observations, "DATE")]:
        if col not in frame.columns:
            continue
        s = pd.to_datetime(frame[col], errors="coerce", utc=True).dropna()
        if len(s) > 0:
            candidates.append(s.dt.tz_convert(None).max())
    return max(candidates) if candidates else pd.Timestamp.utcnow().normalize()


def copy_data() -> dict[str, pd.DataFrame]:
    csv_dir = SYNTHEA_OUTPUT_DIR / "csv"
    if not csv_dir.exists():
        print(f"Missing Synthea output at {csv_dir}", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INFO_DIR.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for file_name in RELEVANT_FILES:
        src = csv_dir / file_name
        if not src.exists():
            print(f"Warning: {file_name} missing", file=sys.stderr)
            continue
        df = pd.read_csv(src)
        df.to_csv(DATA_DIR / file_name, index=False)
        dfs[file_name] = df
        print(f"Wrote {file_name}: {len(df):,} rows")
    return dfs


def write_summary(dfs: dict[str, pd.DataFrame], population: int, seed: int) -> None:
    patients = dfs.get("patients.csv", pd.DataFrame()).copy()
    conditions = dfs.get("conditions.csv", pd.DataFrame())
    procedures = dfs.get("procedures.csv", pd.DataFrame())
    observations = dfs.get("observations.csv", pd.DataFrame())

    if len(patients) == 0:
        return

    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    ref_date = infer_reference_date(conditions, procedures, observations)
    patients["age"] = ((ref_date - patients["BIRTHDATE"]).dt.days / 365.25).fillna(0).astype(int)
    patients = patients[(patients["age"] >= 50) & (patients["age"] <= 100)].copy()

    procedures = procedures.copy()
    procedures["CODE"] = procedures.get("CODE", pd.Series(dtype=object)).astype(str)
    screened = procedures[procedures["CODE"] == CRC_SCREENING_CODE]

    summary = f"""# CRC Baseline Summary

- Requested population: {population:,}
- Seed: {seed}
- Cohort size (age 50-100): {len(patients):,}
- Male: {(patients['GENDER'] == 'M').mean():.1%}
- Mean age: {patients['age'].mean():.1f}
- Median age: {patients['age'].median():.0f}
- Mean income (USD): {pd.to_numeric(patients.get('INCOME', 0), errors='coerce').fillna(0).mean():,.0f}
- Raw colonoscopy procedure rows (SNOMED {CRC_SCREENING_CODE}): {len(screened):,}
"""
    (INFO_DIR / "1_summary_stats.md").write_text(summary)
    print(f"Wrote {INFO_DIR / '1_summary_stats.md'}")


def main() -> None:
    args = parse_args()
    if not args.skip_synthea:
        run_synthea(args.population, args.seed)

    dfs = copy_data()
    write_summary(dfs, args.population, args.seed)


if __name__ == "__main__":
    main()
