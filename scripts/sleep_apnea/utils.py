"""Utility helpers for the sleep apnea regression workflow."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Callable, Collection, Dict, Iterable, List, Optional, Sequence, Tuple

PATIENTS_HEADERS = [
    "Id",
    "BIRTHDATE",
    "DEATHDATE",
    "SSN",
    "DRIVERS",
    "PASSPORT",
    "PREFIX",
    "FIRST",
    "MIDDLE",
    "LAST",
    "SUFFIX",
    "MAIDEN",
    "MARITAL",
    "RACE",
    "ETHNICITY",
    "GENDER",
    "BIRTHPLACE",
    "ADDRESS",
    "CITY",
    "STATE",
    "COUNTY",
    "FIPS",
    "ZIP",
    "LAT",
    "LON",
    "HEALTHCARE_EXPENSES",
    "HEALTHCARE_COVERAGE",
    "INCOME",
]
CONDITIONS_HEADERS = [
    "START",
    "STOP",
    "PATIENT",
    "ENCOUNTER",
    "SYSTEM",
    "CODE",
    "DESCRIPTION",
]
OBSERVATIONS_HEADERS = [
    "DATE",
    "PATIENT",
    "ENCOUNTER",
    "CATEGORY",
    "CODE",
    "DESCRIPTION",
    "VALUE",
    "UNITS",
    "TYPE",
]
PROCEDURES_HEADERS = [
    "START",
    "STOP",
    "PATIENT",
    "ENCOUNTER",
    "SYSTEM",
    "CODE",
    "DESCRIPTION",
    "BASE_COST",
    "REASONCODE",
    "REASONDESCRIPTION",
]
ENCOUNTERS_HEADERS = [
    "Id",
    "START",
    "STOP",
    "PATIENT",
    "ORGANIZATION",
    "PROVIDER",
    "PAYER",
    "ENCOUNTERCLASS",
    "CODE",
    "DESCRIPTION",
    "BASE_ENCOUNTER_COST",
    "TOTAL_CLAIM_COST",
    "PAYER_COVERAGE",
    "REASONCODE",
    "REASONDESCRIPTION",
]
MEDICATIONS_HEADERS = [
    "START",
    "STOP",
    "PATIENT",
    "PAYER",
    "ENCOUNTER",
    "CODE",
    "DESCRIPTION",
    "BASE_COST",
    "PAYER_COVERAGE",
    "DISPENSES",
    "TOTALCOST",
    "REASONCODE",
    "REASONDESCRIPTION",
]
DEVICES_HEADERS = [
    "START",
    "STOP",
    "PATIENT",
    "ENCOUNTER",
    "CODE",
    "DESCRIPTION",
    "UDI",
]
SUPPLIES_HEADERS = [
    "DATE",
    "PATIENT",
    "ENCOUNTER",
    "CODE",
    "DESCRIPTION",
    "QUANTITY",
]


def _has_header(path: Path, expected_headers: Sequence[str]) -> bool:
    """Return True if the first row looks like a header row."""
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
    if not first_row:
        return False
    expected = {header.lower() for header in expected_headers}
    first_vals = {value.strip().lower() for value in first_row}
    return any(value in expected for value in first_vals)


def iter_csv_rows(
    path: Path,
    expected_headers: Sequence[str],
) -> Iterable[Tuple[Dict[str, str], Dict[str, str]]]:
    """Yield CSV rows and a header map, supporting headerless Synthea exports."""
    if not path.exists():
        return
    has_header = _has_header(path, expected_headers)
    with path.open(newline="", encoding="utf-8") as handle:
        if has_header:
            reader = csv.DictReader(handle)
        else:
            reader = csv.DictReader(handle, fieldnames=expected_headers)
        header_map = make_header_map(reader.fieldnames)
        for row in reader:
            yield row, header_map


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
    """Return the latest date found in the patients.csv file.
    
    Only checks birthdate and deathdate columns in patients.csv for speed,
    rather than scanning all CSV files which can be very slow.
    """
    patients_path = csv_dir / "patients.csv"
    if not patients_path.exists():
        return None
    
    latest: Optional[date] = None
    for row, header_map in iter_csv_rows(patients_path, PATIENTS_HEADERS):
        for field in ["deathdate", "birthdate"]:
            val = get_value(row, header_map, [field])
            parsed = parse_date(val)
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
    
    for row, header_map in iter_csv_rows(patients_path, PATIENTS_HEADERS):
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
    for row, header_map in iter_csv_rows(conditions_path, CONDITIONS_HEADERS):
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
    for row, header_map in iter_csv_rows(conditions_path, CONDITIONS_HEADERS):
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

    for row, header_map in iter_csv_rows(observations_path, OBSERVATIONS_HEADERS):
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


def load_latest_smoking_status(
    observations_path: Path,
    status_code: str,
    smoker_values: Collection[str],
    nonsmoker_values: Collection[str],
) -> Dict[str, float]:
    """Load the latest smoking status observation per patient."""
    status_by_patient: Dict[str, float] = {}
    date_by_patient: Dict[str, date] = {}

    if not observations_path.exists():
        return status_by_patient

    smoker_values_norm = {value.strip().lower() for value in smoker_values}
    nonsmoker_values_norm = {value.strip().lower() for value in nonsmoker_values}

    for row, header_map in iter_csv_rows(observations_path, OBSERVATIONS_HEADERS):
        code = get_value(row, header_map, ["code"])
        if code != status_code:
            continue
        patient_id = get_value(row, header_map, ["patient"])
        if not patient_id:
            continue
        obs_date = parse_date(get_value(row, header_map, ["date"]))
        if obs_date is None:
            continue
        value = (get_value(row, header_map, ["value"]) or "").strip().lower()
        if value in smoker_values_norm:
            status = 1.0
        elif value in nonsmoker_values_norm:
            status = 0.0
        else:
            continue
        if patient_id not in date_by_patient or obs_date > date_by_patient[patient_id]:
            date_by_patient[patient_id] = obs_date
            status_by_patient[patient_id] = status
    return status_by_patient


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
    for row, header_map in iter_csv_rows(patients_path, PATIENTS_HEADERS):
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
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, float]:
    """Sum sleep-specific spend per patient across procedures, encounters, meds, and supplies.
    
    Args:
        progress_callback: Optional callback(file_name, rows_processed) for progress reporting.
    """
    spend: Dict[str, float] = defaultdict(float)
    sleep_encounters = set()

    encounters_path = csv_dir / "encounters.csv"
    if encounters_path.exists():
        row_count = 0
        for row, header_map in iter_csv_rows(encounters_path, ENCOUNTERS_HEADERS):
            row_count += 1
            if progress_callback and row_count % 100000 == 0:
                progress_callback("encounters.csv", row_count)
            code = get_value(row, header_map, ["code"])
            reason_code = get_value(row, header_map, ["reasoncode"])
            if reason_code not in sleep_reason_codes:
                continue
            if code and code not in sleep_encounter_codes:
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
        if progress_callback:
            progress_callback("encounters.csv", row_count)

    procedures_path = csv_dir / "procedures.csv"
    if procedures_path.exists():
        row_count = 0
        for row, header_map in iter_csv_rows(procedures_path, PROCEDURES_HEADERS):
            row_count += 1
            if progress_callback and row_count % 100000 == 0:
                progress_callback("procedures.csv", row_count)
            code = get_value(row, header_map, ["code"])
            reason_code = get_value(row, header_map, ["reasoncode"])
            encounter_id = get_value(row, header_map, ["encounter"])
            if (
                code not in sleep_procedure_codes
                and reason_code not in sleep_reason_codes
                and encounter_id not in sleep_encounters
            ):
                continue
            patient_id = get_value(row, header_map, ["patient"])
            cost = parse_float(get_value(row, header_map, ["base_cost"]))
            if patient_id and cost is not None:
                spend[patient_id] += cost
        if progress_callback:
            progress_callback("procedures.csv", row_count)

    medications_path = csv_dir / "medications.csv"
    if medications_path.exists():
        row_count = 0
        for row, header_map in iter_csv_rows(medications_path, MEDICATIONS_HEADERS):
            row_count += 1
            if progress_callback and row_count % 100000 == 0:
                progress_callback("medications.csv", row_count)
            reason_code = get_value(row, header_map, ["reasoncode"])
            encounter_id = get_value(row, header_map, ["encounter"])
            if reason_code not in sleep_reason_codes and encounter_id not in sleep_encounters:
                continue
            patient_id = get_value(row, header_map, ["patient"])
            cost = parse_float(get_value(row, header_map, ["totalcost"]))
            if patient_id and cost is not None:
                spend[patient_id] += cost
        if progress_callback:
            progress_callback("medications.csv", row_count)

    repo_root = Path(__file__).resolve().parents[2]
    costs_dir = repo_root / "src" / "main" / "resources" / "costs"
    costs = _load_costs_map([costs_dir / "devices.csv", costs_dir / "supplies.csv"])

    devices_path = csv_dir / "devices.csv"
    if devices_path.exists():
        row_count = 0
        for row, header_map in iter_csv_rows(devices_path, DEVICES_HEADERS):
            row_count += 1
            if progress_callback and row_count % 50000 == 0:
                progress_callback("devices.csv", row_count)
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
        if progress_callback:
            progress_callback("devices.csv", row_count)

    supplies_path = csv_dir / "supplies.csv"
    if supplies_path.exists():
        row_count = 0
        for row, header_map in iter_csv_rows(supplies_path, SUPPLIES_HEADERS):
            row_count += 1
            if progress_callback and row_count % 50000 == 0:
                progress_callback("supplies.csv", row_count)
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
        if progress_callback:
            progress_callback("supplies.csv", row_count)

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
