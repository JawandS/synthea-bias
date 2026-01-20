#!/usr/bin/env python3
"""Summarize baseline vs biased diabetes datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

# Condition codes
DIABETES_CODE = "44054006"
PREDIABETES_CODE = "714628002"
HYPERGLYCEMIA_CODE = "80394007"
HYPERTRIGLYCERIDEMIA_CODE = "302870006"
METABOLIC_SYNDROME_CODE = "237602007"
OBESITY_CODE = "162864005"
HYPERTENSION_CODE = "59621000"

# Observation codes
A1C_CODE = "4548-4"
BMI_CODE = "39156-5"
SMOKING_STATUS_CODE = "72166-2"
SMOKER_VALUES = {"smokes tobacco daily (finding)"}
NON_SMOKER_VALUES = {"ex-smoker (finding)", "never smoked tobacco (finding)"}


def resolve_csv_dir(path: Path) -> Path:
    """Return the directory that contains patients.csv."""
    if (path / "csv").is_dir():
        return path / "csv"
    if (path / "patients.csv").exists():
        return path
    raise FileNotFoundError(f"No patients.csv found under {path}")


def _column_name(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Return the first matching column name (case-insensitive)."""
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match:
            return match
    raise ValueError(f"Missing columns: {', '.join(candidates)}")


def _optional_column_name(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match:
            return match
    return None


def _format_count(value: int) -> str:
    return f"{value:,}"


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _safe_div(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _normalize_patient_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def load_stats(csv_dir: Path) -> Dict[str, int | float]:
    """Compute summary stats for a dataset."""
    patients_path = csv_dir / "patients.csv"
    conditions_path = csv_dir / "conditions.csv"
    observations_path = csv_dir / "observations.csv"

    if not patients_path.exists():
        raise FileNotFoundError(f"Missing patients.csv at {patients_path}")
    if not conditions_path.exists():
        raise FileNotFoundError(f"Missing conditions.csv at {conditions_path}")
    if not observations_path.exists():
        raise FileNotFoundError(f"Missing observations.csv at {observations_path}")

    patients = pd.read_csv(patients_path)
    conditions = pd.read_csv(conditions_path)
    observations = pd.read_csv(observations_path)

    patient_id_col = _column_name(patients, ["Id", "ID", "id"])
    condition_patient_col = _column_name(conditions, ["PATIENT", "patient"])
    condition_code_col = _column_name(conditions, ["CODE", "code"])
    obs_patient_col = _column_name(observations, ["PATIENT", "patient"])
    obs_code_col = _column_name(observations, ["CODE", "code"])
    obs_value_col = _column_name(observations, ["VALUE", "value"])
    obs_date_col = _optional_column_name(observations, ["DATE", "date"])

    patients[patient_id_col] = _normalize_patient_ids(patients[patient_id_col])
    conditions[condition_patient_col] = _normalize_patient_ids(
        conditions[condition_patient_col]
    )
    conditions[condition_code_col] = conditions[condition_code_col].astype(str).str.strip()
    observations[obs_patient_col] = _normalize_patient_ids(observations[obs_patient_col])
    observations[obs_code_col] = observations[obs_code_col].astype(str).str.strip()

    patient_ids = set(patients[patient_id_col].dropna())
    conditions = conditions[conditions[condition_patient_col].isin(patient_ids)]
    observations = observations[observations[obs_patient_col].isin(patient_ids)]

    # Count conditions
    diabetes_ids = set(
        conditions.loc[
            conditions[condition_code_col] == DIABETES_CODE,
            condition_patient_col,
        ]
    )
    prediabetes_ids = set(
        conditions.loc[
            conditions[condition_code_col] == PREDIABETES_CODE,
            condition_patient_col,
        ]
    )
    hyperglycemia_ids = set(
        conditions.loc[
            conditions[condition_code_col] == HYPERGLYCEMIA_CODE,
            condition_patient_col,
        ]
    )
    hypertriglyceridemia_ids = set(
        conditions.loc[
            conditions[condition_code_col] == HYPERTRIGLYCERIDEMIA_CODE,
            condition_patient_col,
        ]
    )
    metabolic_syndrome_ids = set(
        conditions.loc[
            conditions[condition_code_col] == METABOLIC_SYNDROME_CODE,
            condition_patient_col,
        ]
    )

    total = len(patient_ids)

    # BMI observations
    if obs_date_col:
        observations[obs_date_col] = pd.to_datetime(
            observations[obs_date_col], errors="coerce"
        )

    bmi_obs = observations[observations[obs_code_col] == BMI_CODE].copy()
    bmi_obs[obs_value_col] = pd.to_numeric(bmi_obs[obs_value_col], errors="coerce")
    bmi_obs = bmi_obs.dropna(subset=[obs_value_col])
    if obs_date_col:
        bmi_obs = bmi_obs.sort_values(obs_date_col)
    bmi_latest = bmi_obs.drop_duplicates(subset=[obs_patient_col], keep="last")
    bmi_available = int(bmi_latest[obs_patient_col].nunique())
    obese_count = int((bmi_latest[obs_value_col] >= 30).sum())

    # A1c observations
    a1c_obs = observations[observations[obs_code_col] == A1C_CODE].copy()
    a1c_obs[obs_value_col] = pd.to_numeric(a1c_obs[obs_value_col], errors="coerce")
    a1c_obs = a1c_obs.dropna(subset=[obs_value_col])
    if obs_date_col:
        a1c_obs = a1c_obs.sort_values(obs_date_col)
    a1c_latest = a1c_obs.drop_duplicates(subset=[obs_patient_col], keep="last")
    a1c_available = int(a1c_latest[obs_patient_col].nunique())
    elevated_a1c = int((a1c_latest[obs_value_col] >= 5.7).sum())
    diabetic_a1c = int((a1c_latest[obs_value_col] >= 6.5).sum())

    # Smoking observations
    smoke_obs = observations[observations[obs_code_col] == SMOKING_STATUS_CODE].copy()
    smoke_obs[obs_value_col] = smoke_obs[obs_value_col].astype(str).str.strip().str.lower()
    if obs_date_col:
        smoke_obs = smoke_obs.sort_values(obs_date_col)
    smoke_latest = smoke_obs.drop_duplicates(subset=[obs_patient_col], keep="last")
    smoker_mask = smoke_latest[obs_value_col].isin(SMOKER_VALUES)
    nonsmoker_mask = smoke_latest[obs_value_col].isin(NON_SMOKER_VALUES)
    smoking_known = int((smoker_mask | nonsmoker_mask).sum())
    smoker_count = int(smoker_mask.sum())

    return {
        "total": total,
        "diabetes": len(diabetes_ids),
        "prediabetes": len(prediabetes_ids),
        "hyperglycemia": len(hyperglycemia_ids),
        "hypertriglyceridemia": len(hypertriglyceridemia_ids),
        "metabolic_syndrome": len(metabolic_syndrome_ids),
        "bmi_available": bmi_available,
        "obese": obese_count,
        "a1c_available": a1c_available,
        "elevated_a1c": elevated_a1c,
        "diabetic_a1c": diabetic_a1c,
        "smoking_known": smoking_known,
        "smoker": smoker_count,
        "diabetes_prevalence": len(diabetes_ids) / total if total else 0.0,
        "hyperglycemia_rate": len(hyperglycemia_ids) / total if total else 0.0,
        "hypertriglyceridemia_rate": len(hypertriglyceridemia_ids) / total if total else 0.0,
    }


def _print_table(title: str, rows: Iterable[Tuple[str, str, str]]) -> None:
    rows_list = list(rows)
    name_width = max(len("Metric"), *(len(row[0]) for row in rows_list))
    col_width = max(
        len("Baseline"),
        len("Biased"),
        *(len(row[1]) for row in rows_list),
        *(len(row[2]) for row in rows_list),
    )
    print(title)
    print(f"{'Metric':<{name_width}}  {'Baseline':>{col_width}}  {'Biased':>{col_width}}")
    print("-" * (name_width + col_width * 2 + 4))
    for label, baseline_val, biased_val in rows_list:
        print(f"{label:<{name_width}}  {baseline_val:>{col_width}}  {biased_val:>{col_width}}")
    print()


def _print_single_table(title: str, rows: Iterable[Tuple[str, str]]) -> None:
    rows_list = list(rows)
    name_width = max(len("Metric"), *(len(row[0]) for row in rows_list))
    col_width = max(len("Value"), *(len(row[1]) for row in rows_list))
    print(title)
    print(f"{'Metric':<{name_width}}  {'Value':>{col_width}}")
    print("-" * (name_width + col_width + 2))
    for label, val in rows_list:
        print(f"{label:<{name_width}}  {val:>{col_width}}")
    print()


def print_summary(baseline_stats: Dict[str, int | float], biased_stats: Optional[Dict[str, int | float]] = None) -> None:
    if biased_stats is None:
        # Single dataset mode
        count_rows = [
            ("Total patients", _format_count(int(baseline_stats["total"]))),
            ("Diabetes cases", _format_count(int(baseline_stats["diabetes"]))),
            ("Prediabetes cases", _format_count(int(baseline_stats["prediabetes"]))),
            ("Hyperglycemia cases", _format_count(int(baseline_stats["hyperglycemia"]))),
            ("Hypertriglyceridemia cases", _format_count(int(baseline_stats["hypertriglyceridemia"]))),
            ("Metabolic syndrome cases", _format_count(int(baseline_stats["metabolic_syndrome"]))),
        ]
        _print_single_table("Condition Counts", count_rows)

        rate_rows = [
            ("Diabetes prevalence", _format_pct(float(baseline_stats["diabetes_prevalence"]))),
            ("Hyperglycemia rate", _format_pct(float(baseline_stats["hyperglycemia_rate"]))),
            ("Hypertriglyceridemia rate", _format_pct(float(baseline_stats["hypertriglyceridemia_rate"]))),
            ("A1c available", _format_pct(_safe_div(int(baseline_stats["a1c_available"]), int(baseline_stats["total"])))),
            ("Elevated A1c (>=5.7)", _format_pct(_safe_div(int(baseline_stats["elevated_a1c"]), int(baseline_stats["a1c_available"])))),
            ("Diabetic A1c (>=6.5)", _format_pct(_safe_div(int(baseline_stats["diabetic_a1c"]), int(baseline_stats["a1c_available"])))),
            ("BMI available", _format_pct(_safe_div(int(baseline_stats["bmi_available"]), int(baseline_stats["total"])))),
            ("Obesity (BMI>=30)", _format_pct(_safe_div(int(baseline_stats["obese"]), int(baseline_stats["bmi_available"])))),
            ("Smoker rate (known)", _format_pct(_safe_div(int(baseline_stats["smoker"]), int(baseline_stats["smoking_known"])))),
        ]
        _print_single_table("Prevalence Rates", rate_rows)
        return

    # Two dataset comparison mode
    count_rows = [
        ("Total patients", _format_count(int(baseline_stats["total"])), _format_count(int(biased_stats["total"]))),
        ("Diabetes cases", _format_count(int(baseline_stats["diabetes"])), _format_count(int(biased_stats["diabetes"]))),
        ("Prediabetes cases", _format_count(int(baseline_stats["prediabetes"])), _format_count(int(biased_stats["prediabetes"]))),
        ("Hyperglycemia cases", _format_count(int(baseline_stats["hyperglycemia"])), _format_count(int(biased_stats["hyperglycemia"]))),
        ("Hypertriglyceridemia cases", _format_count(int(baseline_stats["hypertriglyceridemia"])), _format_count(int(biased_stats["hypertriglyceridemia"]))),
        ("Metabolic syndrome cases", _format_count(int(baseline_stats["metabolic_syndrome"])), _format_count(int(biased_stats["metabolic_syndrome"]))),
    ]
    _print_table("Condition Counts", count_rows)

    rate_rows = [
        (
            "Diabetes prevalence",
            _format_pct(float(baseline_stats["diabetes_prevalence"])),
            _format_pct(float(biased_stats["diabetes_prevalence"])),
        ),
        (
            "Hyperglycemia rate",
            _format_pct(float(baseline_stats["hyperglycemia_rate"])),
            _format_pct(float(biased_stats["hyperglycemia_rate"])),
        ),
        (
            "Hypertriglyceridemia rate",
            _format_pct(float(baseline_stats["hypertriglyceridemia_rate"])),
            _format_pct(float(biased_stats["hypertriglyceridemia_rate"])),
        ),
        (
            "A1c available",
            _format_pct(_safe_div(int(baseline_stats["a1c_available"]), int(baseline_stats["total"]))),
            _format_pct(_safe_div(int(biased_stats["a1c_available"]), int(biased_stats["total"]))),
        ),
        (
            "Elevated A1c (>=5.7)",
            _format_pct(_safe_div(int(baseline_stats["elevated_a1c"]), int(baseline_stats["a1c_available"]))),
            _format_pct(_safe_div(int(biased_stats["elevated_a1c"]), int(biased_stats["a1c_available"]))),
        ),
        (
            "Diabetic A1c (>=6.5)",
            _format_pct(_safe_div(int(baseline_stats["diabetic_a1c"]), int(baseline_stats["a1c_available"]))),
            _format_pct(_safe_div(int(biased_stats["diabetic_a1c"]), int(biased_stats["a1c_available"]))),
        ),
        (
            "BMI available",
            _format_pct(_safe_div(int(baseline_stats["bmi_available"]), int(baseline_stats["total"]))),
            _format_pct(_safe_div(int(biased_stats["bmi_available"]), int(biased_stats["total"]))),
        ),
        (
            "Obesity (BMI>=30)",
            _format_pct(_safe_div(int(baseline_stats["obese"]), int(baseline_stats["bmi_available"]))),
            _format_pct(_safe_div(int(biased_stats["obese"]), int(biased_stats["bmi_available"]))),
        ),
        (
            "Smoker rate (known)",
            _format_pct(_safe_div(int(baseline_stats["smoker"]), int(baseline_stats["smoking_known"]))),
            _format_pct(_safe_div(int(biased_stats["smoker"]), int(biased_stats["smoking_known"]))),
        ),
    ]
    _print_table("Prevalence Rates", rate_rows)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Summarize baseline vs biased datasets for sanity checks."
    )
    parser.add_argument(
        "--baseline",
        default=str(repo_root / "diabetes" / "data" / "baseline"),
        help="Baseline CSV directory (or parent containing csv/).",
    )
    parser.add_argument(
        "--biased",
        default=str(repo_root / "diabetes" / "data" / "biased"),
        help="Biased CSV directory (or parent containing csv/).",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only summarize baseline dataset.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    baseline_dir = resolve_csv_dir(Path(args.baseline))
    baseline_stats = load_stats(baseline_dir)

    if args.baseline_only:
        print_summary(baseline_stats)
    else:
        biased_path = Path(args.biased)
        if biased_path.exists():
            biased_dir = resolve_csv_dir(biased_path)
            biased_stats = load_stats(biased_dir)
            print_summary(baseline_stats, biased_stats)
        else:
            print(f"Biased dataset not found at {biased_path}, showing baseline only\n")
            print_summary(baseline_stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
