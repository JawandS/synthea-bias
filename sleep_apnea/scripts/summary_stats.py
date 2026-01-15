#!/usr/bin/env python3
"""Summarize baseline vs biased sleep apnea datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

SLEEP_APNEA_CODES = {"73430006", "78275009"}
SLEEP_DISORDER_CODE = "39898005"
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


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.2f}"


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
    urban_col = _optional_column_name(patients, ["URBAN", "urban"])
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

    sleep_apnea_ids = set(
        conditions.loc[
            conditions[condition_code_col].isin(SLEEP_APNEA_CODES),
            condition_patient_col,
        ]
    )
    sleep_disorder_ids = set(
        conditions.loc[
            conditions[condition_code_col] == SLEEP_DISORDER_CODE,
            condition_patient_col,
        ]
    )

    total = len(patient_ids)

    if urban_col:
        urban_values = pd.to_numeric(patients[urban_col], errors="coerce")
        urban_count = int((urban_values == 1).sum())
        rural_count = int((urban_values == 0).sum())
        missing_urban = int(urban_values.isna().sum())
    else:
        urban_count = 0
        rural_count = 0
        missing_urban = total

    sleep_disorder = len(sleep_disorder_ids)
    sleep_apnea = len(sleep_apnea_ids)
    dropout = len(sleep_disorder_ids - sleep_apnea_ids)

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
        "urban": urban_count,
        "rural": rural_count,
        "missing_urban": missing_urban,
        "sleep_disorder": sleep_disorder,
        "sleep_apnea": sleep_apnea,
        "dropout": dropout,
        "bmi_available": bmi_available,
        "obese": obese_count,
        "smoking_known": smoking_known,
        "smoker": smoker_count,
        "apnea_prevalence": sleep_apnea / total if total else 0.0,
        "dropout_rate_disorder": dropout / sleep_disorder if sleep_disorder else 0.0,
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


def print_summary(baseline_stats: Dict[str, int | float], biased_stats: Dict[str, int | float]) -> None:
    count_rows = [
        ("Total patients", _format_count(int(baseline_stats["total"])), _format_count(int(biased_stats["total"]))),
        ("Urban patients", _format_count(int(baseline_stats["urban"])), _format_count(int(biased_stats["urban"]))),
        ("Rural patients", _format_count(int(baseline_stats["rural"])), _format_count(int(biased_stats["rural"]))),
        (
            "Missing urban",
            _format_count(int(baseline_stats["missing_urban"])),
            _format_count(int(biased_stats["missing_urban"])),
        ),
        (
            "Sleep disorder cases",
            _format_count(int(baseline_stats["sleep_disorder"])),
            _format_count(int(biased_stats["sleep_disorder"])),
        ),
        (
            "Sleep apnea cases",
            _format_count(int(baseline_stats["sleep_apnea"])),
            _format_count(int(biased_stats["sleep_apnea"])),
        ),
        (
            "Dropout cases (disorder only)",
            _format_count(int(baseline_stats["dropout"])),
            _format_count(int(biased_stats["dropout"])),
        ),
    ]
    _print_table("Population Counts", count_rows)

    rate_rows = [
        (
            "Urban share",
            _format_pct(_safe_div(int(baseline_stats["urban"]), int(baseline_stats["total"]))),
            _format_pct(_safe_div(int(biased_stats["urban"]), int(biased_stats["total"]))),
        ),
        (
            "Rural share",
            _format_pct(_safe_div(int(baseline_stats["rural"]), int(baseline_stats["total"]))),
            _format_pct(_safe_div(int(biased_stats["rural"]), int(biased_stats["total"]))),
        ),
        (
            "Urban/Rural ratio",
            _format_ratio(int(baseline_stats["urban"]), int(baseline_stats["rural"])),
            _format_ratio(int(biased_stats["urban"]), int(biased_stats["rural"])),
        ),
        (
            "Missing urban share",
            _format_pct(_safe_div(int(baseline_stats["missing_urban"]), int(baseline_stats["total"]))),
            _format_pct(_safe_div(int(biased_stats["missing_urban"]), int(biased_stats["total"]))),
        ),
        (
            "Sleep disorder prevalence",
            _format_pct(_safe_div(int(baseline_stats["sleep_disorder"]), int(baseline_stats["total"]))),
            _format_pct(_safe_div(int(biased_stats["sleep_disorder"]), int(biased_stats["total"]))),
        ),
        (
            "Sleep apnea prevalence",
            _format_pct(float(baseline_stats["apnea_prevalence"])),
            _format_pct(float(biased_stats["apnea_prevalence"])),
        ),
        (
            "Apnea among disorder",
            _format_pct(
                _safe_div(int(baseline_stats["sleep_apnea"]), int(baseline_stats["sleep_disorder"]))
            ),
            _format_pct(
                _safe_div(int(biased_stats["sleep_apnea"]), int(biased_stats["sleep_disorder"]))
            ),
        ),
        (
            "Dropout rate (disorder -> apnea)",
            _format_pct(float(baseline_stats["dropout_rate_disorder"])),
            _format_pct(float(biased_stats["dropout_rate_disorder"])),
        ),
        (
            "BMI available share",
            _format_pct(_safe_div(int(baseline_stats["bmi_available"]), int(baseline_stats["total"]))),
            _format_pct(_safe_div(int(biased_stats["bmi_available"]), int(biased_stats["total"]))),
        ),
        (
            "Obesity prevalence (BMI>=30)",
            _format_pct(_safe_div(int(baseline_stats["obese"]), int(baseline_stats["bmi_available"]))),
            _format_pct(_safe_div(int(biased_stats["obese"]), int(biased_stats["bmi_available"]))),
        ),
        (
            "Smoking status available share",
            _format_pct(_safe_div(int(baseline_stats["smoking_known"]), int(baseline_stats["total"]))),
            _format_pct(_safe_div(int(biased_stats["smoking_known"]), int(biased_stats["total"]))),
        ),
        (
            "Current smoker rate (known)",
            _format_pct(_safe_div(int(baseline_stats["smoker"]), int(baseline_stats["smoking_known"]))),
            _format_pct(_safe_div(int(biased_stats["smoker"]), int(biased_stats["smoking_known"]))),
        ),
    ]
    _print_table("Population Rates", rate_rows)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Summarize baseline vs biased datasets for sanity checks."
    )
    parser.add_argument(
        "--baseline",
        default=str(repo_root / "sleep_apnea" / "data" / "baseline"),
        help="Baseline CSV directory (or parent containing csv/).",
    )
    parser.add_argument(
        "--biased",
        default=str(repo_root / "sleep_apnea" / "data" / "biased"),
        help="Biased CSV directory (or parent containing csv/).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    baseline_dir = resolve_csv_dir(Path(args.baseline))
    biased_dir = resolve_csv_dir(Path(args.biased))

    baseline_stats = load_stats(baseline_dir)
    biased_stats = load_stats(biased_dir)

    print_summary(baseline_stats, biased_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
