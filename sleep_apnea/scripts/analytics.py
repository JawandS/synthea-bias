#!/usr/bin/env python3
"""Analyze rural effects on sleep apnea under diagnosis."""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SLEEP_APNEA_CODES = {"73430006", "78275009"}
SLEEP_DISORDER_CODE = "39898005"
BMI_CODE = "39156-5"
SMOKING_STATUS_CODE = "72166-2"
SMOKER_VALUES = {"smokes tobacco daily (finding)"}
NON_SMOKER_VALUES = {"ex-smoker (finding)", "never smoked tobacco (finding)"}
HYPERTENSION_CODES = {"59621000"}
CHF_CODES = {"88805009"}
ALCOHOL_USE_CODES = {"7200002"}

FEATURE_NAMES = [
    "age_years",
    "male",
    "income",
    "bmi",
    "smoker",
    "alcohol_use",
    "hypertension",
    "chf",
    "rural",
]

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
    "URBAN",
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


@dataclass
class Dataset:
    name: str
    features: pd.DataFrame
    labels: pd.Series
    cohort_size: int
    sleep_apnea_count: int
    dropout_count: int
    rural_count: int
    urban_count: int
    missing_rural: int
    full_population: Optional[pd.DataFrame] = None


@dataclass
class PairwiseStats:
    rural_n: int
    urban_n: int
    rural_rate: Optional[float]
    urban_rate: Optional[float]
    risk_diff: Optional[float]
    risk_ratio: Optional[float]
    z_value: Optional[float]
    p_value: Optional[float]


@dataclass
class RegressionStats:
    rural_coef: Optional[float]
    rural_odds_ratio: Optional[float]
    rural_p_value: Optional[float]
    auc: Optional[float]
    n: int


@dataclass
class SummaryStats:
    """Summary statistics for a population subset."""
    n: int
    age_mean: Optional[float]
    age_std: Optional[float]
    male_pct: Optional[float]
    income_mean: Optional[float]
    income_std: Optional[float]
    bmi_mean: Optional[float]
    bmi_std: Optional[float]
    smoker_pct: Optional[float]
    alcohol_pct: Optional[float]
    hypertension_pct: Optional[float]
    chf_pct: Optional[float]


def _has_header(path: Path, expected_headers: Sequence[str]) -> bool:
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


def _read_csv(path: Path, headers: Sequence[str]) -> pd.DataFrame:
    if _has_header(path, headers):
        return pd.read_csv(path)
    return pd.read_csv(path, header=None, names=list(headers))


def _column_name(df: pd.DataFrame, candidates: Sequence[str]) -> str:
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


def _dataset_reference_date(patients: pd.DataFrame, birth_col: str, death_col: Optional[str]) -> date:
    birth_dates = pd.to_datetime(patients[birth_col], errors="coerce")
    max_birth = birth_dates.max()
    max_death = pd.NaT
    if death_col:
        death_dates = pd.to_datetime(patients[death_col], errors="coerce")
        max_death = death_dates.max()
    candidates = [value for value in [max_birth, max_death] if pd.notna(value)]
    max_date = max(candidates) if candidates else pd.NaT
    if pd.isna(max_date):
        return date.today()
    return max_date.date()


def _latest_observation(
    observations: pd.DataFrame,
    patient_col: str,
    code_col: str,
    value_col: str,
    date_col: Optional[str],
    code: str,
    value_parser: Callable[[object], Optional[float]],
) -> Dict[str, float]:
    subset = observations[observations[code_col] == code].copy()
    if subset.empty:
        return {}
    if date_col:
        subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce")
        subset = subset.sort_values(date_col)
    latest = subset.drop_duplicates(subset=[patient_col], keep="last")
    parsed_values = latest[value_col].apply(value_parser)
    parsed_values = pd.to_numeric(parsed_values, errors="coerce")
    latest = latest.assign(parsed_value=parsed_values)
    latest = latest.dropna(subset=["parsed_value"])
    return {
        str(patient_id): float(value)
        for patient_id, value in zip(latest[patient_col], latest["parsed_value"])
    }


def _parse_bmi(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _parse_smoking(value: object) -> Optional[float]:
    text = str(value).strip().lower()
    if text in SMOKER_VALUES:
        return 1.0
    if text in NON_SMOKER_VALUES:
        return 0.0
    return None


def _build_feature_frame(
    patients: pd.DataFrame,
    conditions: pd.DataFrame,
    observations: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, Dataset]:
    patient_id_col = _column_name(patients, ["id", "Id", "ID"])
    birth_col = _column_name(patients, ["birthdate", "BIRTHDATE"])
    death_col = _optional_column_name(patients, ["deathdate", "DEATHDATE"])
    gender_col = _column_name(patients, ["gender", "GENDER"])
    income_col = _column_name(patients, ["income", "INCOME"])
    urban_col = _optional_column_name(patients, ["urban", "URBAN"])

    condition_patient_col = _column_name(conditions, ["patient", "PATIENT"])
    condition_code_col = _column_name(conditions, ["code", "CODE"])

    obs_patient_col = _column_name(observations, ["patient", "PATIENT"])
    obs_code_col = _column_name(observations, ["code", "CODE"])
    obs_value_col = _column_name(observations, ["value", "VALUE"])
    obs_date_col = _optional_column_name(observations, ["date", "DATE"])

    patients = patients.copy()
    patients[patient_id_col] = patients[patient_id_col].astype(str).str.strip()

    conditions = conditions.copy()
    conditions[condition_patient_col] = (
        conditions[condition_patient_col].astype(str).str.strip()
    )
    conditions[condition_code_col] = conditions[condition_code_col].astype(str).str.strip()

    observations = observations.copy()
    observations[obs_patient_col] = (
        observations[obs_patient_col].astype(str).str.strip()
    )
    observations[obs_code_col] = observations[obs_code_col].astype(str).str.strip()

    patient_ids = patients[patient_id_col]
    patient_id_set = set(patient_ids)
    conditions = conditions[conditions[condition_patient_col].isin(patient_id_set)]
    observations = observations[observations[obs_patient_col].isin(patient_id_set)]

    reference_date = _dataset_reference_date(patients, birth_col, death_col)
    birth_dates = pd.to_datetime(patients[birth_col], errors="coerce")
    ref_ts = pd.Timestamp(reference_date)
    age_years = (ref_ts - birth_dates).dt.days / 365.25
    age_years = age_years.where(age_years >= 0)

    gender = patients[gender_col].astype(str).str.strip().str.lower()
    male = (gender == "m").astype(float)

    income = pd.to_numeric(patients[income_col], errors="coerce")

    bmi_by_patient = _latest_observation(
        observations,
        obs_patient_col,
        obs_code_col,
        obs_value_col,
        obs_date_col,
        BMI_CODE,
        _parse_bmi,
    )
    smoking_by_patient = _latest_observation(
        observations,
        obs_patient_col,
        obs_code_col,
        obs_value_col,
        obs_date_col,
        SMOKING_STATUS_CODE,
        _parse_smoking,
    )

    alcohol_patients = set(
        conditions.loc[conditions[condition_code_col].isin(ALCOHOL_USE_CODES), condition_patient_col]
    )
    hypertension_patients = set(
        conditions.loc[conditions[condition_code_col].isin(HYPERTENSION_CODES), condition_patient_col]
    )
    chf_patients = set(
        conditions.loc[conditions[condition_code_col].isin(CHF_CODES), condition_patient_col]
    )

    sleep_disorder_patients = set(
        conditions.loc[conditions[condition_code_col] == SLEEP_DISORDER_CODE, condition_patient_col]
    )
    sleep_apnea_patients = set(
        conditions.loc[conditions[condition_code_col].isin(SLEEP_APNEA_CODES), condition_patient_col]
    )
    dropout_patients = sleep_disorder_patients - sleep_apnea_patients

    features = pd.DataFrame(index=patient_ids)
    features["age_years"] = age_years.values
    features["male"] = male.values
    features["income"] = income.values
    features["bmi"] = features.index.map(bmi_by_patient)
    features["smoker"] = features.index.map(smoking_by_patient)
    features["alcohol_use"] = features.index.isin(alcohol_patients).astype(float)
    features["hypertension"] = features.index.isin(hypertension_patients).astype(float)
    features["chf"] = features.index.isin(chf_patients).astype(float)

    if urban_col:
        patients_indexed = patients.set_index(patient_id_col)
        urban_values = pd.to_numeric(patients_indexed[urban_col], errors="coerce")
        rural_map: Dict[str, float] = {}
        for patient_id, urban in urban_values.items():
            if pd.isna(urban):
                continue
            rural_map[str(patient_id)] = 1.0 if urban == 0 else 0.0
        features["rural"] = features.index.map(rural_map)
    else:
        features["rural"] = pd.Series([float("nan")] * len(features.index), index=features.index, dtype="float")

    cohort_ids = [pid for pid in features.index if pid in sleep_disorder_patients]
    cohort = features.loc[cohort_ids]
    labels = cohort.index.isin(dropout_patients).astype(int)

    rural_series = cohort["rural"]
    rural_count = int((rural_series == 1).sum())
    urban_count = int((rural_series == 0).sum())
    missing_rural = int(rural_series.isna().sum())

    dataset = Dataset(
        name="",
        features=cohort,
        labels=pd.Series(labels, index=cohort.index),
        cohort_size=len(cohort),
        sleep_apnea_count=len(sleep_apnea_patients & set(cohort.index)),
        dropout_count=len(dropout_patients & set(cohort.index)),
        rural_count=rural_count,
        urban_count=urban_count,
        missing_rural=missing_rural,
        full_population=features,
    )
    return cohort, pd.Series(labels, index=cohort.index), dataset


def load_dataset(name: str, data_dir: Path) -> Dataset:
    patients_path = data_dir / "patients.csv"
    conditions_path = data_dir / "conditions.csv"
    observations_path = data_dir / "observations.csv"

    if not patients_path.exists():
        raise FileNotFoundError(f"Missing patients.csv at {patients_path}")
    if not conditions_path.exists():
        raise FileNotFoundError(f"Missing conditions.csv at {conditions_path}")
    if not observations_path.exists():
        raise FileNotFoundError(f"Missing observations.csv at {observations_path}")

    patients = _read_csv(patients_path, PATIENTS_HEADERS)
    conditions = _read_csv(conditions_path, CONDITIONS_HEADERS)
    observations = _read_csv(observations_path, OBSERVATIONS_HEADERS)

    features, labels, dataset = _build_feature_frame(patients, conditions, observations)
    dataset.name = name
    dataset.features = features
    dataset.labels = labels
    return dataset


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def pairwise_comparison(dataset: Dataset) -> PairwiseStats:
    rural_mask = dataset.features["rural"] == 1
    urban_mask = dataset.features["rural"] == 0

    rural_n = int(rural_mask.sum())
    urban_n = int(urban_mask.sum())
    if rural_n == 0 or urban_n == 0:
        return PairwiseStats(
            rural_n=rural_n,
            urban_n=urban_n,
            rural_rate=None,
            urban_rate=None,
            risk_diff=None,
            risk_ratio=None,
            z_value=None,
            p_value=None,
        )

    rural_rate = float(dataset.labels[rural_mask].mean())
    urban_rate = float(dataset.labels[urban_mask].mean())
    risk_diff = rural_rate - urban_rate
    risk_ratio = rural_rate / urban_rate if urban_rate > 0 else None

    pooled = (dataset.labels[rural_mask].sum() + dataset.labels[urban_mask].sum()) / (
        rural_n + urban_n
    )
    denom = math.sqrt(pooled * (1 - pooled) * (1 / rural_n + 1 / urban_n))
    if denom == 0:
        z_value = None
        p_value = None
    else:
        z_value = risk_diff / denom
        p_value = 2 * (1 - _normal_cdf(abs(z_value)))

    return PairwiseStats(
        rural_n=rural_n,
        urban_n=urban_n,
        rural_rate=rural_rate,
        urban_rate=urban_rate,
        risk_diff=risk_diff,
        risk_ratio=risk_ratio,
        z_value=z_value,
        p_value=p_value,
    )


def regression_analysis(
    dataset: Dataset,
    seed: int,
    n_perm: int,
) -> RegressionStats:
    if dataset.labels.nunique() < 2:
        return RegressionStats(None, None, None, None, len(dataset.labels))

    X = dataset.features[FEATURE_NAMES].copy()
    y = dataset.labels

    rural_col = "rural"
    if rural_col not in X.columns or X[rural_col].isna().all():
        return RegressionStats(None, None, None, None, len(y))

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")),
        ]
    )

    fitted = clone(model).fit(X, y)
    coef = fitted.named_steps["model"].coef_[0][FEATURE_NAMES.index(rural_col)]
    odds_ratio = math.exp(coef)

    y_prob = fitted.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)

    rng = random.Random(seed)
    count_extreme = 0
    for _ in range(n_perm):
        permuted = X.copy()
        shuffled = list(permuted[rural_col].values)
        rng.shuffle(shuffled)
        permuted[rural_col] = shuffled
        perm_model = clone(model).fit(permuted, y)
        perm_coef = perm_model.named_steps["model"].coef_[0][FEATURE_NAMES.index(rural_col)]
        if abs(perm_coef) >= abs(coef):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_perm + 1)

    return RegressionStats(
        rural_coef=float(coef),
        rural_odds_ratio=float(odds_ratio),
        rural_p_value=float(p_value),
        auc=float(auc),
        n=len(y),
    )


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.1f}"
    return f"{mean:.1f} ± {std:.1f}"


def compute_summary_stats(features: pd.DataFrame, mask: Optional[pd.Series] = None) -> SummaryStats:
    """Compute summary statistics for a population subset."""
    if mask is not None:
        subset = features[mask]
    else:
        subset = features

    n = len(subset)
    if n == 0:
        return SummaryStats(
            n=0, age_mean=None, age_std=None, male_pct=None,
            income_mean=None, income_std=None, bmi_mean=None, bmi_std=None,
            smoker_pct=None, alcohol_pct=None, hypertension_pct=None, chf_pct=None,
        )

    age = subset["age_years"].dropna()
    male = subset["male"].dropna()
    income = subset["income"].dropna()
    bmi = subset["bmi"].dropna()
    smoker = subset["smoker"].dropna()
    alcohol = subset["alcohol_use"].dropna()
    hypertension = subset["hypertension"].dropna()
    chf = subset["chf"].dropna()

    return SummaryStats(
        n=n,
        age_mean=float(age.mean()) if len(age) > 0 else None,
        age_std=float(age.std()) if len(age) > 1 else None,
        male_pct=float(male.mean()) if len(male) > 0 else None,
        income_mean=float(income.mean()) if len(income) > 0 else None,
        income_std=float(income.std()) if len(income) > 1 else None,
        bmi_mean=float(bmi.mean()) if len(bmi) > 0 else None,
        bmi_std=float(bmi.std()) if len(bmi) > 1 else None,
        smoker_pct=float(smoker.mean()) if len(smoker) > 0 else None,
        alcohol_pct=float(alcohol.mean()) if len(alcohol) > 0 else None,
        hypertension_pct=float(hypertension.mean()) if len(hypertension) > 0 else None,
        chf_pct=float(chf.mean()) if len(chf) > 0 else None,
    )


def write_report(
    path: Path,
    datasets: List[Dataset],
    pairwise: Dict[str, PairwiseStats],
    regressions: Dict[str, RegressionStats],
    population_stats: Dict[str, Dict[str, SummaryStats]],
    n_perm: int,
) -> None:
    lines: List[str] = []
    lines.append("# Rural Under Diagnosis Analysis")
    lines.append("")
    lines.append(
        "Under diagnosis is defined as a sleep disorder diagnosis without a corresponding sleep apnea diagnosis "
        "(SLEEP_DISORDER_CODE 39898005 without SNOMED 73430006/78275009)."
    )
    lines.append("")

    # Population Summary Statistics table
    lines.append("## Population Summary Statistics (Full Population)")
    lines.append("")
    lines.append(
        "Summary statistics for the full patient population, stratified by rural/urban residence and dataset."
    )
    lines.append("")

    # Build column headers dynamically from datasets
    col_headers = []
    for dataset in datasets:
        col_headers.extend([f"{dataset.name} Urban", f"{dataset.name} Rural"])
    header_row = "| Characteristic | " + " | ".join(col_headers) + " |"
    separator_row = "| --- |" + " ---: |" * len(col_headers)
    lines.append(header_row)
    lines.append(separator_row)

    # Helper to get stats for a dataset/residence combo
    def get_stat(dataset_name: str, residence: str) -> SummaryStats:
        return population_stats.get(dataset_name, {}).get(residence, SummaryStats(
            n=0, age_mean=None, age_std=None, male_pct=None,
            income_mean=None, income_std=None, bmi_mean=None, bmi_std=None,
            smoker_pct=None, alcohol_pct=None, hypertension_pct=None, chf_pct=None,
        ))

    # N row
    n_cells = []
    for dataset in datasets:
        n_cells.append(f"{get_stat(dataset.name, 'urban').n:,}")
        n_cells.append(f"{get_stat(dataset.name, 'rural').n:,}")
    lines.append("| N | " + " | ".join(n_cells) + " |")

    # Age row
    age_cells = []
    for dataset in datasets:
        s = get_stat(dataset.name, "urban")
        age_cells.append(_format_mean_std(s.age_mean, s.age_std))
        s = get_stat(dataset.name, "rural")
        age_cells.append(_format_mean_std(s.age_mean, s.age_std))
    lines.append("| Age (years) | " + " | ".join(age_cells) + " |")

    # Male % row
    male_cells = []
    for dataset in datasets:
        male_cells.append(_format_pct(get_stat(dataset.name, "urban").male_pct))
        male_cells.append(_format_pct(get_stat(dataset.name, "rural").male_pct))
    lines.append("| Male (%) | " + " | ".join(male_cells) + " |")

    # Income row
    income_cells = []
    for dataset in datasets:
        s = get_stat(dataset.name, "urban")
        income_cells.append(_format_mean_std(s.income_mean, s.income_std))
        s = get_stat(dataset.name, "rural")
        income_cells.append(_format_mean_std(s.income_mean, s.income_std))
    lines.append("| Income ($) | " + " | ".join(income_cells) + " |")

    # BMI row
    bmi_cells = []
    for dataset in datasets:
        s = get_stat(dataset.name, "urban")
        bmi_cells.append(_format_mean_std(s.bmi_mean, s.bmi_std))
        s = get_stat(dataset.name, "rural")
        bmi_cells.append(_format_mean_std(s.bmi_mean, s.bmi_std))
    lines.append("| BMI | " + " | ".join(bmi_cells) + " |")

    # Smoker % row
    smoker_cells = []
    for dataset in datasets:
        smoker_cells.append(_format_pct(get_stat(dataset.name, "urban").smoker_pct))
        smoker_cells.append(_format_pct(get_stat(dataset.name, "rural").smoker_pct))
    lines.append("| Current Smoker (%) | " + " | ".join(smoker_cells) + " |")

    # Alcohol use % row
    alcohol_cells = []
    for dataset in datasets:
        alcohol_cells.append(_format_pct(get_stat(dataset.name, "urban").alcohol_pct))
        alcohol_cells.append(_format_pct(get_stat(dataset.name, "rural").alcohol_pct))
    lines.append("| Alcohol Use Disorder (%) | " + " | ".join(alcohol_cells) + " |")

    # Hypertension % row
    hypertension_cells = []
    for dataset in datasets:
        hypertension_cells.append(_format_pct(get_stat(dataset.name, "urban").hypertension_pct))
        hypertension_cells.append(_format_pct(get_stat(dataset.name, "rural").hypertension_pct))
    lines.append("| Hypertension (%) | " + " | ".join(hypertension_cells) + " |")

    # CHF % row
    chf_cells = []
    for dataset in datasets:
        chf_cells.append(_format_pct(get_stat(dataset.name, "urban").chf_pct))
        chf_cells.append(_format_pct(get_stat(dataset.name, "rural").chf_pct))
    lines.append("| CHF (%) | " + " | ".join(chf_cells) + " |")

    lines.append("")

    lines.append("## Cohort Summary (Sleep Disorder Patients)")
    lines.append("")
    has_missing_rural = any(d.missing_rural > 0 for d in datasets)
    if has_missing_rural:
        lines.append("| Dataset | Cohort N | Sleep Apnea | Under Diagnosed | Rural | Urban | Missing Rural |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    else:
        lines.append("| Dataset | Cohort N | Sleep Apnea | Under Diagnosed | Rural | Urban |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for dataset in datasets:
        if has_missing_rural:
            lines.append(
                "| {name} | {n:,} | {apnea:,} | {dropout:,} | {rural:,} | {urban:,} | {missing:,} |".format(
                    name=dataset.name,
                    n=dataset.cohort_size,
                    apnea=dataset.sleep_apnea_count,
                    dropout=dataset.dropout_count,
                    rural=dataset.rural_count,
                    urban=dataset.urban_count,
                    missing=dataset.missing_rural,
                )
            )
        else:
            lines.append(
                "| {name} | {n:,} | {apnea:,} | {dropout:,} | {rural:,} | {urban:,} |".format(
                    name=dataset.name,
                    n=dataset.cohort_size,
                    apnea=dataset.sleep_apnea_count,
                    dropout=dataset.dropout_count,
                    rural=dataset.rural_count,
                    urban=dataset.urban_count,
                )
            )
    lines.append("")

    lines.append("## Pairwise Comparison (Rural vs Urban)")
    lines.append("")
    lines.append(
        "Rates represent under diagnosis rates (proportion of sleep disorder patients without a sleep apnea diagnosis). "
        "P-values are from a two-proportion z-test."
    )
    lines.append("")
    lines.append("| Dataset | Rural N | Urban N | Rural Rate | Urban Rate | Risk Diff | Risk Ratio | z | p-value |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dataset in datasets:
        stats = pairwise[dataset.name]
        lines.append(
            "| {name} | {rural} | {urban} | {r_rate} | {u_rate} | {diff} | {ratio} | {z} | {p} |".format(
                name=dataset.name,
                rural=f"{stats.rural_n:,}",
                urban=f"{stats.urban_n:,}",
                r_rate=_format_pct(stats.rural_rate),
                u_rate=_format_pct(stats.urban_rate),
                diff=_format_pct(stats.risk_diff),
                ratio=_format_ratio(stats.risk_ratio),
                z=_format_metric(stats.z_value),
                p=_format_metric(stats.p_value),
            )
        )
    lines.append("")

    lines.append("## Regression (Adjusted Rural Effect)")
    lines.append("")
    lines.append(
        "Logistic regression includes age, gender, income, BMI, smoking status, alcohol use, hypertension, CHF, "
        "and a rural indicator. Rural coefficient significance is assessed via permutation testing "
        f"(n={n_perm})."
    )
    lines.append("")
    lines.append(
        "| Dataset | N | Rural Coef | Odds Ratio | Permutation p-value | In-sample AUC |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for dataset in datasets:
        stats = regressions[dataset.name]
        lines.append(
            "| {name} | {n:,} | {coef} | {odds} | {pval} | {auc} |".format(
                name=dataset.name,
                n=stats.n,
                coef=_format_metric(stats.rural_coef),
                odds=_format_metric(stats.rural_odds_ratio),
                pval=_format_metric(stats.rural_p_value),
                auc=_format_metric(stats.auc),
            )
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "If rural coefficients are positive with low permutation p-values, rural residence is associated with a "
        "higher chance of under diagnosis after adjusting for clinical risk factors. The pairwise comparison provides "
        "a direct rate difference, while the regression isolates the rural effect conditional on covariates."
    )
    lines.append("")
    lines.append(
        "In the biased dataset that simulates real-world barriers to care experienced by rural populations, "
        "we expect rural residence to be associated with a substantially higher odds of under diagnosis compared "
        "to the baseline dataset where urban and rural patients have equal access to care."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze rural effects on sleep apnea under diagnosis."
    )
    base_dir = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--baseline",
        default=str(base_dir / "data" / "baseline"),
        help="Baseline CSV directory.",
    )
    parser.add_argument(
        "--biased",
        default=str(base_dir / "data" / "biased"),
        help="Biased CSV directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=500,
        help="Permutation count for rural coefficient significance.",
    )
    parser.add_argument(
        "--out",
        default=str(base_dir / "output" / "sleep_apnea_analytics_report.md"),
        help="Output markdown path.",
    )
    return parser


def _compute_population_stats(dataset: Dataset) -> Dict[str, SummaryStats]:
    """Compute summary stats for urban and rural subsets of full population."""
    if dataset.full_population is None:
        return {}
    pop = dataset.full_population
    urban_mask = pop["rural"] == 0
    rural_mask = pop["rural"] == 1
    return {
        "urban": compute_summary_stats(pop, urban_mask),
        "rural": compute_summary_stats(pop, rural_mask),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    baseline = load_dataset("baseline", Path(args.baseline))
    biased = load_dataset("biased", Path(args.biased))

    pairwise = {
        baseline.name: pairwise_comparison(baseline),
        biased.name: pairwise_comparison(biased),
    }
    regressions = {
        baseline.name: regression_analysis(baseline, args.seed, args.n_perm),
        biased.name: regression_analysis(biased, args.seed, args.n_perm),
    }
    population_stats = {
        baseline.name: _compute_population_stats(baseline),
        biased.name: _compute_population_stats(biased),
    }

    write_report(
        Path(args.out),
        [baseline, biased],
        pairwise,
        regressions,
        population_stats,
        args.n_perm,
    )
    print(f"Wrote report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
