#!/usr/bin/env python3
"""
Append an URBAN column to the SDoH county CSV using the Census
urban_delineation list (https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html).

URBAN = 1 when the county FIPS (zero-padded to 5 digits to restore leading
zeros in the state code) appears in scripts/data/urban_delineation.csv;
otherwise 0. Paths are resolved relative to this script so it can run from
any working directory.
"""

import csv
from pathlib import Path
from typing import List, Set

BASE_DIR = Path(__file__).resolve().parent
URBAN_DELINEATION_PATH = BASE_DIR / "data" / "urban_delineation.csv"
SDOH_PATH = BASE_DIR.parent / "src" / "main" / "resources" / "geography" / "sdoh.csv"
URBAN_COLUMN = "URBAN"


def load_urban_fips(path: Path) -> Set[str]:
    """Return a set of 5-digit FIPS codes present in the delineation file."""
    with path.open(newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        required = ["FIPS State Code", "FIPS County Code"]
        for column in required:
            if column not in (reader.fieldnames or []):
                raise ValueError(f"Missing column {column} in {path}")

        urban_fips: Set[str] = set()
        for row in reader:
            state = (row.get("FIPS State Code") or "").strip()
            county = (row.get("FIPS County Code") or "").strip()
            if not state or not county:
                continue
            fips = f"{state.zfill(2)}{county.zfill(3)}"
            urban_fips.add(fips)
    return urban_fips


def add_urban_flag() -> None:
    urban_fips = load_urban_fips(URBAN_DELINEATION_PATH)

    with SDOH_PATH.open(newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"No headers found in {SDOH_PATH}")

        fieldnames: List[str] = list(reader.fieldnames)
        if URBAN_COLUMN not in fieldnames:
            fieldnames.append(URBAN_COLUMN)

        rows = []
        matched = 0
        total = 0

        for row in reader:
            total += 1
            fips_raw = (row.get("FIPS_CODE") or "").strip()
            # Pad to 5 digits to restore leading zeros in the state code.
            padded_fips = fips_raw.zfill(5) if fips_raw else ""
            is_urban = padded_fips in urban_fips
            if is_urban:
                matched += 1
            row[URBAN_COLUMN] = "1" if is_urban else "0"
            rows.append(row)

    with SDOH_PATH.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {SDOH_PATH} with {URBAN_COLUMN} column. "
        f"Urban counties: {matched}/{total}."
    )


if __name__ == "__main__":
    add_urban_flag()
