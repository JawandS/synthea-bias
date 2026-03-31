"""
Patch script: add per-patient cost features from Synthea encounters.csv into data.csv.

Cost features added:
  - total_encounter_cost   : sum of TOTAL_CLAIM_COST across all encounters
  - total_payer_coverage   : sum of PAYER_COVERAGE across all encounters
  - out_of_pocket_cost     : total_encounter_cost - total_payer_coverage
  - num_encounters         : number of encounters
  - cost_per_encounter     : total_encounter_cost / num_encounters

Source: synthea/output_diabetes_v2/csv/encounters.csv  (724 MB, read in chunks)
Target: output/data/data.csv
"""

import argparse
from pathlib import Path

import pandas as pd

ENCOUNTERS_COLS = ["PATIENT", "TOTAL_CLAIM_COST", "PAYER_COVERAGE"]
CHUNK_SIZE = 100_000


def parse_args():
    parser = argparse.ArgumentParser(description="Add cost features to data.csv")
    parser.add_argument(
        "--encounters",
        type=Path,
        default=Path(__file__).parents[2] / "synthea/output_diabetes_v2/csv/encounters.csv",
        help="Path to Synthea encounters.csv",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent.parent / "output/data/data.csv",
        help="Path to data.csv to patch",
    )
    return parser.parse_args()


def aggregate_encounters(encounters_path: Path) -> pd.DataFrame:
    """Stream encounters.csv in chunks, aggregate per patient."""
    print(f"Reading {encounters_path} in chunks of {CHUNK_SIZE:,}...")
    agg_parts = []

    reader = pd.read_csv(
        encounters_path,
        usecols=ENCOUNTERS_COLS,
        chunksize=CHUNK_SIZE,
        dtype={"TOTAL_CLAIM_COST": float, "PAYER_COVERAGE": float},
    )

    for i, chunk in enumerate(reader):
        part = chunk.groupby("PATIENT", sort=False).agg(
            total_encounter_cost=("TOTAL_CLAIM_COST", "sum"),
            total_payer_coverage=("PAYER_COVERAGE", "sum"),
            num_encounters=("TOTAL_CLAIM_COST", "count"),
        )
        agg_parts.append(part)
        if (i + 1) % 10 == 0:
            print(f"  processed {(i + 1) * CHUNK_SIZE:,} rows...")

    print("Combining chunks...")
    combined = pd.concat(agg_parts).groupby(level=0).sum()
    combined["out_of_pocket_cost"] = (
        combined["total_encounter_cost"] - combined["total_payer_coverage"]
    ).clip(lower=0)
    combined["cost_per_encounter"] = (
        combined["total_encounter_cost"] / combined["num_encounters"]
    )
    combined.index.name = "id"
    return combined.reset_index()


def main():
    args = parse_args()

    if not args.encounters.exists():
        raise FileNotFoundError(f"encounters.csv not found: {args.encounters}")
    if not args.data.exists():
        raise FileNotFoundError(f"data.csv not found: {args.data}")

    cost_df = aggregate_encounters(args.encounters)
    print(f"Aggregated cost data for {len(cost_df):,} patients")

    print(f"Loading {args.data}...")
    data = pd.read_csv(args.data)
    n_before = len(data)
    print(f"  {n_before:,} rows in data.csv")

    # Drop any existing cost columns (idempotent re-run)
    cost_cols = ["total_encounter_cost", "total_payer_coverage", "out_of_pocket_cost",
                 "num_encounters", "cost_per_encounter"]
    data = data.drop(columns=[c for c in cost_cols if c in data.columns])

    merged = data.merge(cost_df, on="id", how="left")
    n_matched = merged["total_encounter_cost"].notna().sum()
    n_missing = merged["total_encounter_cost"].isna().sum()

    print(f"  matched: {n_matched:,} patients")
    if n_missing > 0:
        print(f"  WARNING: {n_missing:,} patients had no encounter records — cost columns will be NaN")

    merged.to_csv(args.data, index=False)
    print(f"Saved updated data.csv with cost columns -> {args.data}")

    print("\nCost feature summary:")
    print(merged[cost_cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
