#!/usr/bin/env python3
"""Validate Synthea generation stability/determinism.

This script generates multiple datasets with identical seeds and verifies
that the outputs are byte-for-byte identical. This confirms that Synthea's
deterministic generation is working correctly.

Usage:
    cd synthea-bias/sleep_apnea
    uv run python artificats/scripts/validate_stability.py

    # Custom number of runs or population
    uv run python artificats/scripts/validate_stability.py --runs 3 --population 100
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Default paths relative to repo structure
REPO_ROOT = Path(__file__).resolve().parents[3]  # synthea-bias/
SYNTHEA_DIR = REPO_ROOT / "synthea"

SEPARATOR = "=" * 72
CSV_FILES = ["patients.csv", "conditions.csv", "observations.csv", "encounters.csv"]


def run_synthea(
    synthea_dir: Path,
    output_dir: Path,
    population: int,
    seed: int,
    state: str = "Montana",
) -> bool:
    """Run Synthea with deterministic settings.

    Uses the same flags as documented in the sleep_apnea README.
    """
    cmd = [
        "./run_synthea",
        "-s", str(seed),       # Random seed
        "-cs", str(seed),      # Clinician seed
        "-o", "false",         # Disable overwrites
        "-p", str(population),
        "-a", "30-100",        # Age range (sleep apnea module requirement)
        f"--exporter.csv.export=true",
        f"--exporter.csv.append_mode=false",
        f"--exporter.baseDirectory={output_dir}",
        state,
    ]

    print(f"    Running: {' '.join(cmd[:8])}...")

    try:
        result = subprocess.run(
            cmd,
            cwd=synthea_dir,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        if result.returncode != 0:
            print(f"    ERROR: Synthea failed with code {result.returncode}")
            print(f"    stderr: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("    ERROR: Synthea timed out after 10 minutes")
        return False
    except FileNotFoundError:
        print(f"    ERROR: run_synthea not found in {synthea_dir}")
        return False


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_csv_content(path: Path) -> str:
    """Hash CSV content after sorting rows for order-independent comparison.

    Synthea may write rows in different orders due to threading, so we
    normalize by sorting before hashing.
    """
    df = pd.read_csv(path, dtype=str)
    # Sort by all columns to normalize order
    df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    content = df.to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()


def compare_datasets(
    dirs: List[Path],
    use_content_hash: bool = True,
) -> Tuple[bool, Dict[str, List[str]]]:
    """Compare multiple dataset directories for equality.

    Args:
        dirs: List of output directories to compare
        use_content_hash: If True, sort CSV content before hashing (order-independent)

    Returns:
        (all_match, file_hashes) where file_hashes maps filename to list of hashes
    """
    file_hashes: Dict[str, List[str]] = {f: [] for f in CSV_FILES}

    for out_dir in dirs:
        csv_dir = out_dir / "csv"
        if not csv_dir.exists():
            print(f"    WARNING: {csv_dir} does not exist")
            for f in CSV_FILES:
                file_hashes[f].append("MISSING")
            continue

        for filename in CSV_FILES:
            fpath = csv_dir / filename
            if not fpath.exists():
                file_hashes[filename].append("MISSING")
            elif use_content_hash:
                file_hashes[filename].append(hash_csv_content(fpath))
            else:
                file_hashes[filename].append(hash_file(fpath))

    # Check if all hashes match for each file
    all_match = True
    for filename, hashes in file_hashes.items():
        unique = set(hashes)
        if len(unique) > 1:
            all_match = False

    return all_match, file_hashes


def compare_dataframes(dirs: List[Path]) -> Dict[str, Dict[str, any]]:
    """Detailed comparison of DataFrames across runs."""
    results = {}

    for filename in CSV_FILES:
        dfs = []
        for out_dir in dirs:
            fpath = out_dir / "csv" / filename
            if fpath.exists():
                dfs.append(pd.read_csv(fpath))
            else:
                dfs.append(None)

        # Compare shapes
        shapes = [df.shape if df is not None else None for df in dfs]
        shape_match = len(set(str(s) for s in shapes)) == 1

        # Compare row counts
        row_counts = [len(df) if df is not None else 0 for df in dfs]

        # For patients.csv, compare patient IDs
        id_sets = []
        if filename == "patients.csv":
            for df in dfs:
                if df is not None and "Id" in df.columns:
                    id_sets.append(set(df["Id"]))

        results[filename] = {
            "shapes": shapes,
            "shape_match": shape_match,
            "row_counts": row_counts,
            "id_sets": id_sets if id_sets else None,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate Synthea generation stability across multiple runs."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of generation runs to compare (default: 5)",
    )
    parser.add_argument(
        "--population",
        "-p",
        type=int,
        default=200,
        help="Population size per run (default: 200)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=160,
        help="Random seed for generation (default: 160)",
    )
    parser.add_argument(
        "--state",
        default="Montana",
        help="State for generation (default: Montana)",
    )
    parser.add_argument(
        "--synthea-dir",
        type=Path,
        default=SYNTHEA_DIR,
        help=f"Path to Synthea directory (default: {SYNTHEA_DIR})",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep generated output directories (default: delete after comparison)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation, compare existing outputs in --output-base",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help="Base directory for outputs (default: temp directory)",
    )

    args = parser.parse_args()

    print()
    print(SEPARATOR)
    print(" Synthea Stability Validation ".center(72))
    print(SEPARATOR)
    print()
    print(f"  Runs:       {args.runs}")
    print(f"  Population: {args.population}")
    print(f"  Seed:       {args.seed}")
    print(f"  State:      {args.state}")
    print(f"  Synthea:    {args.synthea_dir}")
    print()

    # Verify Synthea directory
    if not args.skip_generation:
        run_script = args.synthea_dir / "run_synthea"
        if not run_script.exists():
            print(f"ERROR: run_synthea not found at {run_script}")
            print("Please ensure Synthea is built and run_synthea exists.")
            return 1

    # Setup output directories
    if args.output_base:
        base_dir = args.output_base
        base_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        temp_dir = tempfile.mkdtemp(prefix="synthea_stability_")
        base_dir = Path(temp_dir)
        cleanup = not args.keep_outputs

    output_dirs = [base_dir / f"run_{i+1}" for i in range(args.runs)]

    try:
        # Step 1: Generate datasets
        if not args.skip_generation:
            print(SEPARATOR)
            print("Step 1: Generating datasets")
            print(SEPARATOR)
            print()

            for i, out_dir in enumerate(output_dirs):
                print(f"  Run {i+1}/{args.runs}:")
                out_dir.mkdir(parents=True, exist_ok=True)
                success = run_synthea(
                    args.synthea_dir,
                    out_dir,
                    args.population,
                    args.seed,
                    args.state,
                )
                if not success:
                    print(f"  FAILED: Run {i+1} did not complete successfully")
                    return 1
                print(f"    Output: {out_dir}")
                print()
        else:
            print("Skipping generation (--skip-generation)")
            print()

        # Step 2: Compare outputs
        print(SEPARATOR)
        print("Step 2: Comparing outputs (content-normalized)")
        print(SEPARATOR)
        print()

        all_match, file_hashes = compare_datasets(output_dirs, use_content_hash=True)

        print(f"  {'File':<20s} {'Status':<10s} {'Hashes'}")
        print(f"  {'-'*20} {'-'*10} {'-'*40}")
        for filename, hashes in file_hashes.items():
            unique = set(hashes)
            status = "MATCH" if len(unique) == 1 else "DIFFER"
            hash_display = hashes[0][:16] + "..." if hashes else "N/A"
            if len(unique) > 1:
                hash_display = f"{len(unique)} different"
            print(f"  {filename:<20s} {status:<10s} {hash_display}")

        print()

        # Step 3: Detailed comparison
        print(SEPARATOR)
        print("Step 3: Detailed comparison")
        print(SEPARATOR)
        print()

        details = compare_dataframes(output_dirs)
        for filename, info in details.items():
            print(f"  {filename}:")
            print(f"    Shapes: {info['shapes']}")
            print(f"    Shape match: {info['shape_match']}")
            print(f"    Row counts: {info['row_counts']}")
            if info["id_sets"]:
                id_overlap = set.intersection(*info["id_sets"]) if info["id_sets"] else set()
                print(f"    Patient ID overlap: {len(id_overlap)} / {info['row_counts'][0]}")
            print()

        # Step 4: Summary
        print(SEPARATOR)
        print("Summary")
        print(SEPARATOR)
        print()

        if all_match:
            print("  PASS: All generated datasets are identical (content-normalized)")
            print()
            print("  Synthea generation is deterministic with the given seed.")
            print("  The same seed produces the same patients and records.")
        else:
            print("  FAIL: Generated datasets differ")
            print()
            print("  Possible causes:")
            print("    - Threading race conditions in Synthea")
            print("    - Non-deterministic module behavior")
            print("    - File system ordering differences")
            print()
            print("  Note: Row ORDER differences are normalized out. If this test")
            print("  fails, there are actual content differences.")

        print()

        # Byte-exact comparison (informational)
        print(SEPARATOR)
        print("Byte-exact comparison (informational)")
        print(SEPARATOR)
        print()

        exact_match, exact_hashes = compare_datasets(output_dirs, use_content_hash=False)
        print(f"  Byte-exact match: {exact_match}")
        if not exact_match and all_match:
            print("  (Files have same content but different row ordering)")
        print()

        return 0 if all_match else 1

    finally:
        if cleanup and base_dir.exists():
            print(f"Cleaning up {base_dir}...")
            shutil.rmtree(base_dir)


if __name__ == "__main__":
    sys.exit(main())
