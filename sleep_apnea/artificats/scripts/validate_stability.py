#!/usr/bin/env python3
"""Validate Synthea generation stability/determinism.

This script generates multiple datasets with identical seeds and verifies
that the same patients with the same attributes are generated across runs.

Usage:
    cd synthea-bias/sleep_apnea
    uv run python artificats/scripts/validate_stability.py

    # Custom number of runs or population
    uv run python artificats/scripts/validate_stability.py --runs 3 --population 100
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from scipy import stats

# Default paths relative to repo structure
REPO_ROOT = Path(__file__).resolve().parents[3]  # synthea-bias/
SYNTHEA_DIR = REPO_ROOT / "synthea"


def run_synthea(
    synthea_dir: Path,
    output_dir: Path,
    population: int,
    seed: int,
    state: str = "Montana",
) -> bool:
    """Run Synthea with deterministic settings."""
    cmd = [
        "./run_synthea",
        "-s", str(seed),       # Random seed
        "-cs", str(seed),      # Clinician seed
        "-p", str(population),
        "-a", "30-100",        # Age range (sleep apnea module requirement)
        "-m", "4",             # Multi-threading
        "--exporter.csv.export=true",
        "--exporter.csv.append_mode=false",
        "--exporter.fhir.export=false",
        "--exporter.ccda.export=false",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        f"--exporter.baseDirectory={output_dir}",
        state,
    ]

    print(f"  Running Synthea (seed={seed}, pop={population})...")

    try:
        result = subprocess.run(
            cmd,
            cwd=synthea_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  ERROR: Synthea failed with code {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  ERROR: Synthea timed out after 10 minutes")
        return False
    except FileNotFoundError:
        print(f"  ERROR: run_synthea not found in {synthea_dir}")
        return False


def load_csv(output_dir: Path, filename: str) -> pd.DataFrame | None:
    """Load a CSV file from output directory."""
    path = output_dir / "csv" / filename
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=str)


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize datetime columns to date-only (strip time component)."""
    df = df.copy()
    date_cols = [c for c in df.columns if c in ("DATE", "START", "STOP", "BIRTHDATE", "DEATHDATE")]
    for col in date_cols:
        if col in df.columns:
            # Extract just the date part (YYYY-MM-DD) from datetime strings
            df[col] = df[col].str.split("T").str[0]
    return df


def normalize_df(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    """Normalize and sort DataFrame for comparison."""
    df = normalize_dates(df)
    cols = [c for c in sort_cols if c in df.columns]
    return df.sort_values(cols).reset_index(drop=True)


def test_statistical_difference(df1: pd.DataFrame, df2: pd.DataFrame, col: str) -> tuple[str, float]:
    """Test if column values differ significantly between two DataFrames.

    Returns (test_name, p_value).
    """
    s1, s2 = df1[col].dropna(), df2[col].dropna()

    if len(s1) == 0 or len(s2) == 0:
        return "N/A", 1.0

    # Try numeric comparison first
    try:
        n1, n2 = pd.to_numeric(s1), pd.to_numeric(s2)
        if len(n1) > 0 and len(n2) > 0:
            _, p = stats.mannwhitneyu(n1, n2, alternative='two-sided')
            return "Mann-Whitney U", p
    except (ValueError, TypeError):
        pass

    # Categorical: chi-square test of independence using contingency table
    counts1, counts2 = s1.value_counts(), s2.value_counts()
    all_vals = sorted(set(counts1.index) | set(counts2.index))

    if len(all_vals) < 2:
        return "N/A", 1.0

    # Build contingency table
    contingency = [
        [counts1.get(v, 0) for v in all_vals],
        [counts2.get(v, 0) for v in all_vals],
    ]

    try:
        _, p, _, _ = stats.chi2_contingency(contingency)
        return "Chi-square", p
    except ValueError:
        return "N/A", 1.0


def compare_dataframes(
    dfs: list[pd.DataFrame], sort_cols: list[str]
) -> tuple[bool, str, list[tuple[str, str, float]]]:
    """Compare DataFrames across runs.

    Returns (match, message, stat_tests) where stat_tests is list of (col, test, p_value).
    """
    if len(dfs) < 2:
        return False, "Need at least 2 runs", []

    base = normalize_df(dfs[0], sort_cols)
    stat_tests = []

    for run_num, df in enumerate(dfs[1:], start=2):
        other = normalize_df(df, sort_cols)

        if len(base) != len(other):
            return False, f"Row count differs: run 1 has {len(base)}, run {run_num} has {len(other)}", []

        if not base.equals(other):
            diff_cols = [c for c in base.columns if not base[c].equals(other[c])]
            # Run statistical tests on differing columns
            for col in diff_cols:
                test_name, p_val = test_statistical_difference(base, other, col)
                stat_tests.append((col, test_name, p_val))
            return False, f"Run 1 vs {run_num}: values differ in {diff_cols}", stat_tests

    return True, f"{len(base)} records match across {len(dfs)} runs", []


def main():
    parser = argparse.ArgumentParser(
        description="Validate Synthea generation stability across multiple runs."
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of generation runs to compare (default: 3)",
    )
    parser.add_argument(
        "--population", "-p", type=int, default=200,
        help="Population size per run (default: 200)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=160,
        help="Random seed for generation (default: 160)",
    )
    parser.add_argument(
        "--state", default="Montana",
        help="State for generation (default: Montana)",
    )
    parser.add_argument(
        "--synthea-dir", type=Path, default=SYNTHEA_DIR,
        help=f"Path to Synthea directory (default: {SYNTHEA_DIR})",
    )
    parser.add_argument(
        "--keep-outputs", action="store_true",
        help="Keep generated output directories",
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print(" Synthea Stability Validation ".center(60))
    print("=" * 60)
    print(f"  Runs: {args.runs}  |  Population: {args.population}  |  Seed: {args.seed}")
    print()

    # Verify Synthea directory
    run_script = args.synthea_dir / "run_synthea"
    if not run_script.exists():
        print(f"ERROR: run_synthea not found at {run_script}")
        return 1

    # Setup temp directory
    temp_dir = tempfile.mkdtemp(prefix="synthea_stability_")
    base_dir = Path(temp_dir)
    output_dirs = [base_dir / f"run_{i+1}" for i in range(args.runs)]

    try:
        # Generate datasets
        print("Generating datasets...")
        for out_dir in output_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
            success = run_synthea(
                args.synthea_dir, out_dir, args.population, args.seed, args.state
            )
            if not success:
                return 1
        print()

        # Files to compare: (filename, sort_cols, required)
        files_to_compare = [
            ("patients.csv", ["Id"], True),
            ("conditions.csv", ["PATIENT", "CODE", "START"], False),  # Optional
            ("observations.csv", ["PATIENT", "CODE", "DATE", "VALUE"], True),
        ]

        print("Comparing across runs...")
        print()
        all_pass = True

        for filename, sort_cols, required in files_to_compare:
            dfs = []
            missing_dirs = []
            for out_dir in output_dirs:
                df = load_csv(out_dir, filename)
                if df is None:
                    missing_dirs.append(out_dir.name)
                else:
                    dfs.append(df)

            # Handle missing files
            if len(missing_dirs) == len(output_dirs):
                # Missing in all runs - skip if optional, fail if required
                if required:
                    print(f"  {filename}: FAIL - missing in all runs")
                    all_pass = False
                else:
                    print(f"  {filename}: SKIP - not generated (optional)")
                continue
            elif missing_dirs:
                # Missing in some runs but not others - inconsistent
                print(f"  {filename}: FAIL - missing in {missing_dirs}")
                all_pass = False
                continue

            # Compare DataFrames
            match, message, stat_tests = compare_dataframes(dfs, sort_cols)
            status = "PASS" if match else "FAIL"
            print(f"  {filename}: {status} - {message}")
            if not match:
                all_pass = False
                # Show statistical significance of differences
                if stat_tests:
                    print("    Statistical tests for differing columns:")
                    for col, test, p in stat_tests:
                        sig = "significant" if p < 0.05 else "not significant"
                        print(f"      {col}: {test} p={p:.4f} ({sig})")

        print()
        print("=" * 60)
        if all_pass:
            print(f"  PASS: All files identical across {args.runs} runs")
            print(f"  Synthea generation is deterministic with seed={args.seed}")
        else:
            print("  FAIL: Differences detected between runs")
        print("=" * 60)
        print()

        return 0 if all_pass else 1

    finally:
        if not args.keep_outputs and base_dir.exists():
            shutil.rmtree(base_dir)


if __name__ == "__main__":
    sys.exit(main())
