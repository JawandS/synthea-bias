#!/usr/bin/env python3
"""Prepare diabetes CSV inputs from Synthea output directories.

- Copies relevant CSVs (patients/observations/conditions) into diabetes/data.
- Repairs missing headers using reference output/csv headers.
- Filters observations to relevant codes (A1c, BMI, smoking status).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import pandas as pd

RELEVANT_FILES = ["patients.csv", "observations.csv", "conditions.csv"]
# A1c, BMI, smoking status
OBSERVATION_CODES = {"4548-4", "39156-5", "72166-2"}


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
    """Keep only observations needed for modeling (A1c, BMI, smoking status)."""
    code_col = _find_column(df, ["code"])
    codes = df[code_col].astype(str).str.strip()
    return df.loc[codes.isin(OBSERVATION_CODES)].copy()


def process_dataset(
    name: str,
    source_root: Path,
    dest_root: Path,
    reference_csv_dir: Path,
) -> None:
    """Copy and normalize relevant CSVs for a dataset."""
    source_csv_dir = resolve_csv_dir(source_root)
    dest_dir = dest_root / name
    dest_dir.mkdir(parents=True, exist_ok=True)

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

        if filename == "observations.csv":
            df = filter_observations(df)

        df.to_csv(dest_path, index=False)
        print(f"Wrote {dest_path}")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Copy and normalize Synthea CSVs for diabetes modeling."
    )
    parser.add_argument(
        "--baseline",
        default=str(repo_root / "snythea" / "output_baseline_documentation"),
        help="Path to baseline Synthea output (root or csv directory).",
    )
    parser.add_argument(
        "--biased",
        default=str(repo_root / "snythea" / "output_documentation_bias"),
        help="Path to biased Synthea output (root or csv directory).",
    )
    parser.add_argument(
        "--reference",
        default=str(repo_root / "snythea" / "output"),
        help="Path to reference output with headers (root or csv directory).",
    )
    parser.add_argument(
        "--out",
        default=str(repo_root / "diabetes" / "data"),
        help="Output directory for baseline/biased CSVs.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only process baseline dataset (skip biased).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    reference_csv_dir = resolve_csv_dir(Path(args.reference))
    dest_root = Path(args.out)

    process_dataset(
        "baseline",
        Path(args.baseline),
        dest_root,
        reference_csv_dir,
    )

    if not args.baseline_only:
        biased_path = Path(args.biased)
        if biased_path.exists():
            process_dataset(
                "biased",
                biased_path,
                dest_root,
                reference_csv_dir,
            )
        else:
            print(f"Skipping biased dataset: {biased_path} does not exist")


if __name__ == "__main__":
    main()
