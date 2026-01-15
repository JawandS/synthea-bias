#!/usr/bin/env python3
"""Prepare sleep apnea CSV inputs from Synthea output directories.

- Copies relevant CSVs (patients/observations/conditions) into sleep_apnea/data.
- Repairs missing headers using reference output/csv headers.
- Appends a URBAN column to patients.csv via the SDoH county lookup.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

RELEVANT_FILES = ["patients.csv", "observations.csv", "conditions.csv"]
OBSERVATION_CODES = {"39156-5", "72166-2"}


def resolve_csv_dir(path: Path) -> Path:
    """Return the directory that contains CSV exports."""
    if (path / "csv").is_dir():
        return path / "csv"
    if (path / "patients.csv").exists():
        return path
    raise FileNotFoundError(f"No CSV directory found under {path}")


def load_reference_headers(reference_csv_dir: Path, filename: str) -> List[str]:
    """Load the header row from the reference CSV export."""
    ref_path = reference_csv_dir / filename
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference CSV missing: {ref_path}")
    with ref_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        headers = next(reader, [])
    if not headers:
        raise ValueError(f"Reference CSV has no headers: {ref_path}")
    return headers


def normalize_headers(headers: Sequence[str]) -> List[str]:
    return [header.strip().lower() for header in headers]


def source_has_header(source_path: Path, ref_headers: Sequence[str]) -> bool:
    """Return True if the source CSV already includes a header row."""
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, [])
    if not first_row:
        return False
    return normalize_headers(first_row) == normalize_headers(ref_headers)


def load_csv_with_headers(source_path: Path, ref_headers: Sequence[str]) -> pd.DataFrame:
    """Load a CSV, applying reference headers if they are missing."""
    if source_has_header(source_path, ref_headers):
        return pd.read_csv(source_path)
    return pd.read_csv(source_path, header=None, names=list(ref_headers))


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Return the first matching column name (case-insensitive)."""
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match:
            return match
    raise ValueError(f"Missing columns: {', '.join(candidates)}")


def filter_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only observations needed for modeling (BMI + smoking status)."""
    code_col = _find_column(df, ["code"])
    codes = df[code_col].astype(str).str.strip()
    return df.loc[codes.isin(OBSERVATION_CODES)].copy()


def load_sdoh_urban_map(sdoh_path: Path) -> Dict[Tuple[str, str], Optional[float]]:
    """Return a mapping of (STATE, COUNTY) -> URBAN flag."""
    if not sdoh_path.exists():
        raise FileNotFoundError(f"SDoH file not found: {sdoh_path}")
    sdoh = pd.read_csv(sdoh_path)
    required = {"STATE", "COUNTY", "URBAN"}
    missing = required.difference(sdoh.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"SDoH CSV missing columns: {missing_list}")

    sdoh = sdoh[list(required)].copy()
    sdoh["STATE"] = sdoh["STATE"].astype(str).str.strip().str.upper()
    sdoh["COUNTY"] = sdoh["COUNTY"].astype(str).str.strip().str.upper()
    sdoh["URBAN"] = pd.to_numeric(sdoh["URBAN"], errors="coerce")
    sdoh = sdoh.dropna(subset=["STATE", "COUNTY"]).drop_duplicates(["STATE", "COUNTY"])

    return {
        (row.STATE, row.COUNTY): row.URBAN
        for row in sdoh.itertuples(index=False)
    }


def append_urban_column(patients: pd.DataFrame, urban_map: Dict[Tuple[str, str], Optional[float]]) -> pd.DataFrame:
    """Append URBAN column to the patients DataFrame."""
    required = {"STATE", "COUNTY"}
    missing = required.difference(patients.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"patients.csv missing columns: {missing_list}")

    state_key = patients["STATE"].astype(str).str.strip().str.upper()
    county_key = patients["COUNTY"].astype(str).str.strip().str.upper()
    urban_values = [urban_map.get(key) for key in zip(state_key, county_key)]
    urban_series = pd.Series(urban_values, index=patients.index)

    if "URBAN" in patients.columns:
        patients = patients.copy()
        patients["URBAN"] = patients["URBAN"].where(patients["URBAN"].notna(), urban_series)
        return patients

    patients = patients.copy()
    patients["URBAN"] = urban_series
    return patients


def process_dataset(
    name: str,
    source_root: Path,
    dest_root: Path,
    reference_csv_dir: Path,
    sdoh_path: Path,
) -> None:
    """Copy and normalize relevant CSVs for a dataset."""
    source_csv_dir = resolve_csv_dir(source_root)
    dest_dir = dest_root / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    urban_map = load_sdoh_urban_map(sdoh_path)

    for filename in RELEVANT_FILES:
        source_path = source_csv_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source CSV: {source_path}")

        ref_headers = load_reference_headers(reference_csv_dir, filename)
        dest_path = dest_dir / filename
        if filename == "observations.csv" and dest_path.exists():
            df = load_csv_with_headers(dest_path, ref_headers)
        else:
            df = load_csv_with_headers(source_path, ref_headers)

        if filename == "patients.csv":
            df = append_urban_column(df, urban_map)
        elif filename == "observations.csv":
            df = filter_observations(df)

        df.to_csv(dest_path, index=False)
        print(f"Wrote {dest_path}")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Copy and normalize Synthea CSVs for sleep apnea modeling."
    )
    parser.add_argument(
        "--baseline",
        default=str(repo_root / "snythea" / "output_baseline"),
        help="Path to baseline Synthea output (root or csv directory).",
    )
    parser.add_argument(
        "--biased",
        default=str(repo_root / "snythea" / "output_rural_bias"),
        help="Path to biased Synthea output (root or csv directory).",
    )
    parser.add_argument(
        "--reference",
        default=str(repo_root / "snythea" / "output"),
        help="Path to reference output with headers (root or csv directory).",
    )
    parser.add_argument(
        "--sdoh",
        default=str(
            repo_root / "snythea" / "src" / "main" / "resources" / "geography" / "sdoh.csv"
        ),
        help="Path to SDoH CSV with URBAN column.",
    )
    parser.add_argument(
        "--out",
        default=str(repo_root / "sleep_apnea" / "data"),
        help="Output directory for baseline/biased CSVs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    reference_csv_dir = resolve_csv_dir(Path(args.reference))
    dest_root = Path(args.out)
    sdoh_path = Path(args.sdoh)

    process_dataset(
        "baseline",
        Path(args.baseline),
        dest_root,
        reference_csv_dir,
        sdoh_path,
    )
    process_dataset(
        "biased",
        Path(args.biased),
        dest_root,
        reference_csv_dir,
        sdoh_path,
    )


if __name__ == "__main__":
    main()
