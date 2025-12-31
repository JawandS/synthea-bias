"""Utility helpers for the sleep apnea regression workflow."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Collection, Dict, List, Optional, Sequence, Tuple


def find_csv_dir(path_str: str) -> Path:
    """Resolve a CSV output directory that contains patients.csv."""
    path = Path(path_str)
    candidates = [path]
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / path)
    for candidate in candidates:
        if candidate.is_dir():
            if (candidate / "patients.csv").exists():
                return candidate
            if (candidate / "csv" / "patients.csv").exists():
                return candidate / "csv"
    raise FileNotFoundError(
        f"Unable to locate patients.csv under {path_str}. "
        "Pass the output directory (e.g., output_baseline) or the csv subdir."
    )


def make_header_map(fieldnames: Optional[Sequence[str]]) -> Dict[str, str]:
    """Build a case-insensitive map of CSV fieldnames."""
    return {name.lower(): name for name in fieldnames or []}


def get_value(
    row: Dict[str, str],
    header_map: Dict[str, str],
    candidates: Sequence[str],
) -> Optional[str]:
    """Return the first matching CSV value for a list of candidate headers."""
    for candidate in candidates:
        key = header_map.get(candidate.lower())
        if key:
            return row.get(key)
    return None


def parse_date(value: Optional[str]) -> Optional[date]:
    """Parse an ISO-ish date string into a date object."""
    if not value:
        return None
    value = value.strip()
    if len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def find_dataset_end_date(csv_dir: Path) -> Optional[date]:
    """Return the latest date found across all CSVs in a dataset directory."""
    latest: Optional[date] = None
    for path in sorted(csv_dir.glob("*.csv")):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            date_fields = [
                name
                for name in fieldnames
                if any(token in name.lower() for token in ("date", "start", "stop"))
            ]
            if not date_fields:
                continue
            for row in reader:
                for field in date_fields:
                    parsed = parse_date(row.get(field))
                    if parsed and (latest is None or parsed > latest):
                        latest = parsed
    return latest


def parse_float(value: Optional[str]) -> Optional[float]:
    """Parse a numeric string into a float, returning None for blanks."""
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_costs_map(cost_paths: Sequence[Path]) -> Dict[str, float]:
    """Load per-code device/supply costs from one or more CSV files."""
    costs: Dict[str, float] = {}
    for path in cost_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header_map = make_header_map(reader.fieldnames)
            for row in reader:
                code = get_value(row, header_map, ["code"])
                cost = parse_float(get_value(row, header_map, ["mode"]))
                if code and cost is not None:
                    costs[code] = cost
    return costs


def load_sdoh_urban_map() -> Dict[Tuple[str, str], int]:
    """Load SDoH URBAN flags keyed by (STATE, COUNTY)."""
    repo_root = Path(__file__).resolve().parents[2]
    sdoh_path = repo_root / "src" / "main" / "resources" / "geography" / "sdoh.csv"
    urban_map: Dict[Tuple[str, str], int] = {}
    if not sdoh_path.exists():
        return urban_map
    with sdoh_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = make_header_map(reader.fieldnames)
        for row in reader:
            state = get_value(row, header_map, ["state"])
            county = get_value(row, header_map, ["county"])
            urban = parse_float(get_value(row, header_map, ["urban"]))
            if state and county and urban is not None:
                key = (state.strip().upper(), county.strip().upper())
                urban_map[key] = 1 if urban >= 0.5 else 0
    return urban_map


def load_patient_urban_flags(patients_path: Path) -> Dict[str, Optional[float]]:
    """Load urban/rural flag for each patient based on their location.
    
    Returns a dict mapping patient_id to urban flag:
    - 1.0 for urban
    - 0.0 for rural
    - None if location cannot be determined
    """
    urban_map = load_sdoh_urban_map()
    patient_urban: Dict[str, Optional[float]] = {}
    
    if not patients_path.exists():
        return patient_urban
    
    with patients_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = make_header_map(reader.fieldnames)
        for row in reader:
            patient_id = get_value(row, header_map, ["id"])
            if not patient_id:
                continue
            state = (get_value(row, header_map, ["state"]) or "").strip().upper()
            county = (get_value(row, header_map, ["county"]) or "").strip().upper()
            key = (state, county)
            if state and county and key in urban_map:
                patient_urban[patient_id] = float(urban_map[key])
            else:
                patient_urban[patient_id] = None
    return patient_urban


def load_condition_flags(
    conditions_path: Path,
    codes: Collection[str],
) -> Tuple[set, Optional[date]]:
    """Return a set of patients with any condition code and the latest date seen."""
    patients = set()
    max_date = None
    with conditions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = make_header_map(reader.fieldnames)
        for row in reader:
            code = get_value(row, header_map, ["code"])
            patient_id = get_value(row, header_map, ["patient"])
            start_date = parse_date(get_value(row, header_map, ["start", "date"]))
            if start_date and (max_date is None or start_date > max_date):
                max_date = start_date
            if patient_id and code in codes:
                patients.add(patient_id)
    return patients, max_date


def load_condition_patients(conditions_path: Path, codes: Collection[str]) -> set:
    """Return a set of patients with any condition code in the list."""
    patients = set()
    with conditions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = make_header_map(reader.fieldnames)
        for row in reader:
            code = get_value(row, header_map, ["code"])
            if code not in codes:
                continue
            patient_id = get_value(row, header_map, ["patient"])
            if patient_id:
                patients.add(patient_id)
    return patients


def load_latest_bmi(observations_path: Path, bmi_code: str) -> Dict[str, float]:
    """Load the latest BMI observation per patient."""
    bmi_by_patient: Dict[str, float] = {}
    date_by_patient: Dict[str, date] = {}

    if not observations_path.exists():
        return bmi_by_patient

    with observations_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = make_header_map(reader.fieldnames)
        for row in reader:
            code = get_value(row, header_map, ["code"])
            if code != bmi_code:
                continue
            patient_id = get_value(row, header_map, ["patient"])
            if not patient_id:
                continue
            value = parse_float(get_value(row, header_map, ["value"]))
            if value is None:
                continue
            obs_date = parse_date(get_value(row, header_map, ["date"]))
            if obs_date is None:
                continue
            if patient_id not in date_by_patient or obs_date > date_by_patient[patient_id]:
                date_by_patient[patient_id] = obs_date
                bmi_by_patient[patient_id] = value
    return bmi_by_patient


def compute_population_stats(
    csv_dir: Path,
    urban_map: Dict[Tuple[str, str], int],
    sleep_disorder_code: str,
    sleep_apnea_codes: Collection[str],
) -> Dict[str, float]:
    """Compute urban/rural totals and sleep condition counts for a dataset."""
    patients_path = csv_dir / "patients.csv"
    conditions_path = csv_dir / "conditions.csv"

    patient_locations: Dict[str, Tuple[str, str]] = {}
    with patients_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = make_header_map(reader.fieldnames)
        for row in reader:
            patient_id = get_value(row, header_map, ["id"])
            if not patient_id:
                continue
            state = (get_value(row, header_map, ["state"]) or "").strip().upper()
            county = (get_value(row, header_map, ["county"]) or "").strip().upper()
            patient_locations[patient_id] = (state, county)

    total = len(patient_locations)
    urban = 0
    rural = 0
    missing = 0
    # Classify by SDoH urban flag when available.
    for state, county in patient_locations.values():
        key = (state, county)
        if not state or not county or key not in urban_map:
            missing += 1
            continue
        if urban_map[key] == 1:
            urban += 1
        else:
            rural += 1

    patient_ids = set(patient_locations.keys())
    sleep_disorder = load_condition_patients(conditions_path, [sleep_disorder_code]) & patient_ids
    sleep_apnea = load_condition_patients(conditions_path, sleep_apnea_codes) & patient_ids
    dropout = sleep_disorder - sleep_apnea

    return {
        "total": total,
        "urban": urban,
        "rural": rural,
        "missing_urban": missing,
        "sleep_disorder": len(sleep_disorder),
        "sleep_apnea": len(sleep_apnea),
        "dropout": len(dropout),
        "apnea_prevalence": len(sleep_apnea) / total if total else 0.0,
        "dropout_rate_disorder": len(dropout) / len(sleep_disorder) if sleep_disorder else 0.0,
    }


def load_sleep_spend(
    csv_dir: Path,
    sleep_procedure_codes: Collection[str],
    sleep_reason_codes: Collection[str],
    sleep_encounter_codes: Collection[str],
    sleep_equipment_codes: Collection[str],
) -> Dict[str, float]:
    """Sum sleep-related spend per patient across procedures, encounters, meds, and supplies."""
    spend: Dict[str, float] = defaultdict(float)
    sleep_encounters = set()

    procedures_path = csv_dir / "procedures.csv"
    if procedures_path.exists():
        with procedures_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header_map = make_header_map(reader.fieldnames)
            for row in reader:
                code = get_value(row, header_map, ["code"])
                reason_code = get_value(row, header_map, ["reasoncode"])
                if code not in sleep_procedure_codes and reason_code not in sleep_reason_codes:
                    continue
                patient_id = get_value(row, header_map, ["patient"])
                cost = parse_float(get_value(row, header_map, ["base_cost"]))
                if patient_id and cost is not None:
                    spend[patient_id] += cost

    encounters_path = csv_dir / "encounters.csv"
    if encounters_path.exists():
        with encounters_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header_map = make_header_map(reader.fieldnames)
            for row in reader:
                code = get_value(row, header_map, ["code"])
                reason_code = get_value(row, header_map, ["reasoncode"])
                if reason_code not in sleep_reason_codes and code not in sleep_encounter_codes:
                    continue
                encounter_id = get_value(row, header_map, ["id"])
                if encounter_id:
                    sleep_encounters.add(encounter_id)
                patient_id = get_value(row, header_map, ["patient"])
                cost = parse_float(get_value(row, header_map, ["total_claim_cost"]))
                if cost is None:
                    cost = parse_float(get_value(row, header_map, ["base_encounter_cost"]))
                if patient_id and cost is not None:
                    spend[patient_id] += cost

    medications_path = csv_dir / "medications.csv"
    if medications_path.exists():
        with medications_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header_map = make_header_map(reader.fieldnames)
            for row in reader:
                reason_code = get_value(row, header_map, ["reasoncode"])
                if reason_code not in sleep_reason_codes:
                    continue
                patient_id = get_value(row, header_map, ["patient"])
                cost = parse_float(get_value(row, header_map, ["totalcost"]))
                if patient_id and cost is not None:
                    spend[patient_id] += cost

    repo_root = Path(__file__).resolve().parents[2]
    costs_dir = repo_root / "src" / "main" / "resources" / "costs"
    costs = _load_costs_map([costs_dir / "devices.csv", costs_dir / "supplies.csv"])

    devices_path = csv_dir / "devices.csv"
    if devices_path.exists():
        with devices_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header_map = make_header_map(reader.fieldnames)
            for row in reader:
                code = get_value(row, header_map, ["code"])
                if not code:
                    continue
                encounter_id = get_value(row, header_map, ["encounter"])
                # Allow device/supply costs if directly sleep-related or tied to a sleep encounter.
                if code not in sleep_equipment_codes and encounter_id not in sleep_encounters:
                    continue
                cost = costs.get(code)
                if cost is None:
                    continue
                patient_id = get_value(row, header_map, ["patient"])
                if patient_id:
                    spend[patient_id] += cost

    supplies_path = csv_dir / "supplies.csv"
    if supplies_path.exists():
        with supplies_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header_map = make_header_map(reader.fieldnames)
            for row in reader:
                code = get_value(row, header_map, ["code"])
                if not code:
                    continue
                encounter_id = get_value(row, header_map, ["encounter"])
                if code not in sleep_equipment_codes and encounter_id not in sleep_encounters:
                    continue
                cost = costs.get(code)
                if cost is None:
                    continue
                patient_id = get_value(row, header_map, ["patient"])
                if patient_id:
                    spend[patient_id] += cost

    return spend


def impute_missing(
    rows: List[List[Optional[float]]],
    reference: Optional[List[List[Optional[float]]]] = None,
) -> List[List[float]]:
    """Impute missing values with column means from a reference dataset."""
    if not rows:
        return []
    ref_rows = reference if reference is not None else rows
    means = []
    for col in range(len(ref_rows[0])):
        values = [row[col] for row in ref_rows if row[col] is not None]
        mean = sum(values) / len(values) if values else 0.0
        means.append(mean)
    imputed = []
    for row in rows:
        imputed.append([means[i] if value is None else float(value) for i, value in enumerate(row)])
    return imputed
