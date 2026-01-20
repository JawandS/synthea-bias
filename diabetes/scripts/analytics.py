#!/usr/bin/env python3
"""Analyze documentation bias effects on diabetes diagnosis."""

from __future__ import annotations

import argparse
import csv
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

DIABETES_CODE = "44054006"
BMI_CODE = "39156-5"
SMOKING_STATUS_CODE = "72166-2"
SMOKER_VALUES = {"smokes tobacco daily (finding)"}
NON_SMOKER_VALUES = {"ex-smoker (finding)", "never smoked tobacco (finding)"}
OBESITY_CODES = {"162864005", "408512008", "414915002"}
HYPERTENSION_CODES = {"59621000"}
HYPERLIPIDEMIA_CODES = {"55822004"}
HYPERGLYCEMIA_CODE = "80394007"
HYPERTRIGLYCERIDEMIA_CODE = "302870006"

FEATURE_NAMES = [
    "age_years",
    "male",
    "income",
    "bmi",
    "smoker",
    "obesity",
    "hypertension",
    "hyperlipidemia",
    "hyperglycemia",
    "hypertriglyceridemia",
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
    patient_count: int
    diabetes_count: int
    hyperglycemia_count: int
    hypertriglyceridemia_count: int
    diabetes_with_hyperglycemia: int
    diabetes_with_hypertriglyceridemia: int


@dataclass
class DocumentationStats:
    baseline_hyperglycemia_rate: float
    biased_hyperglycemia_rate: float
    hyperglycemia_reduction: float
    baseline_hypertriglyceridemia_rate: float
    biased_hypertriglyceridemia_rate: float
    hypertriglyceridemia_reduction: float
    baseline_diabetes_hyperglycemia_rate: float
    biased_diabetes_hyperglycemia_rate: float
    baseline_diabetes_hypertriglyceridemia_rate: float
    biased_diabetes_hypertriglyceridemia_rate: float


@dataclass
class RegressionStats:
    feature: str
    baseline_coef: float
    baseline_odds_ratio: float
    biased_coef: float
    biased_odds_ratio: float
    coef_change: float


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

    obesity_patients = set(
        conditions.loc[conditions[condition_code_col].isin(OBESITY_CODES), condition_patient_col]
    )
    hypertension_patients = set(
        conditions.loc[conditions[condition_code_col].isin(HYPERTENSION_CODES), condition_patient_col]
    )
    hyperlipidemia_patients = set(
        conditions.loc[conditions[condition_code_col].isin(HYPERLIPIDEMIA_CODES), condition_patient_col]
    )
    hyperglycemia_patients = set(
        conditions.loc[conditions[condition_code_col] == HYPERGLYCEMIA_CODE, condition_patient_col]
    )
    hypertriglyceridemia_patients = set(
        conditions.loc[conditions[condition_code_col] == HYPERTRIGLYCERIDEMIA_CODE, condition_patient_col]
    )
    diabetes_patients = set(
        conditions.loc[conditions[condition_code_col] == DIABETES_CODE, condition_patient_col]
    )

    features = pd.DataFrame(index=patient_ids)
    features["age_years"] = age_years.values
    features["male"] = male.values
    features["income"] = income.values
    features["bmi"] = features.index.map(bmi_by_patient)
    features["smoker"] = features.index.map(smoking_by_patient)
    features["obesity"] = features.index.isin(obesity_patients).astype(float)
    features["hypertension"] = features.index.isin(hypertension_patients).astype(float)
    features["hyperlipidemia"] = features.index.isin(hyperlipidemia_patients).astype(float)
    features["hyperglycemia"] = features.index.isin(hyperglycemia_patients).astype(float)
    features["hypertriglyceridemia"] = features.index.isin(hypertriglyceridemia_patients).astype(float)

    labels = features.index.isin(diabetes_patients).astype(int)

    diabetes_with_hyperglycemia = len(diabetes_patients & hyperglycemia_patients)
    diabetes_with_hypertriglyceridemia = len(diabetes_patients & hypertriglyceridemia_patients)

    dataset = Dataset(
        name="",
        features=features,
        labels=pd.Series(labels, index=features.index),
        patient_count=len(features),
        diabetes_count=len(diabetes_patients),
        hyperglycemia_count=len(hyperglycemia_patients),
        hypertriglyceridemia_count=len(hypertriglyceridemia_patients),
        diabetes_with_hyperglycemia=diabetes_with_hyperglycemia,
        diabetes_with_hypertriglyceridemia=diabetes_with_hypertriglyceridemia,
    )
    return features, pd.Series(labels, index=features.index), dataset


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


def compute_documentation_stats(baseline: Dataset, biased: Dataset) -> DocumentationStats:
    baseline_hyperglycemia_rate = baseline.hyperglycemia_count / baseline.patient_count
    biased_hyperglycemia_rate = biased.hyperglycemia_count / biased.patient_count
    hyperglycemia_reduction = (
        (baseline_hyperglycemia_rate - biased_hyperglycemia_rate) / baseline_hyperglycemia_rate
        if baseline_hyperglycemia_rate > 0
        else 0.0
    )

    baseline_hypertriglyceridemia_rate = baseline.hypertriglyceridemia_count / baseline.patient_count
    biased_hypertriglyceridemia_rate = biased.hypertriglyceridemia_count / biased.patient_count
    hypertriglyceridemia_reduction = (
        (baseline_hypertriglyceridemia_rate - biased_hypertriglyceridemia_rate)
        / baseline_hypertriglyceridemia_rate
        if baseline_hypertriglyceridemia_rate > 0
        else 0.0
    )

    baseline_diabetes_hyperglycemia_rate = (
        baseline.diabetes_with_hyperglycemia / baseline.diabetes_count
        if baseline.diabetes_count > 0
        else 0.0
    )
    biased_diabetes_hyperglycemia_rate = (
        biased.diabetes_with_hyperglycemia / biased.diabetes_count
        if biased.diabetes_count > 0
        else 0.0
    )

    baseline_diabetes_hypertriglyceridemia_rate = (
        baseline.diabetes_with_hypertriglyceridemia / baseline.diabetes_count
        if baseline.diabetes_count > 0
        else 0.0
    )
    biased_diabetes_hypertriglyceridemia_rate = (
        biased.diabetes_with_hypertriglyceridemia / biased.diabetes_count
        if biased.diabetes_count > 0
        else 0.0
    )

    return DocumentationStats(
        baseline_hyperglycemia_rate=baseline_hyperglycemia_rate,
        biased_hyperglycemia_rate=biased_hyperglycemia_rate,
        hyperglycemia_reduction=hyperglycemia_reduction,
        baseline_hypertriglyceridemia_rate=baseline_hypertriglyceridemia_rate,
        biased_hypertriglyceridemia_rate=biased_hypertriglyceridemia_rate,
        hypertriglyceridemia_reduction=hypertriglyceridemia_reduction,
        baseline_diabetes_hyperglycemia_rate=baseline_diabetes_hyperglycemia_rate,
        biased_diabetes_hyperglycemia_rate=biased_diabetes_hyperglycemia_rate,
        baseline_diabetes_hypertriglyceridemia_rate=baseline_diabetes_hypertriglyceridemia_rate,
        biased_diabetes_hypertriglyceridemia_rate=biased_diabetes_hypertriglyceridemia_rate,
    )


def regression_analysis(
    baseline: Dataset,
    biased: Dataset,
) -> Tuple[List[RegressionStats], float, float]:
    if baseline.labels.nunique() < 2 or biased.labels.nunique() < 2:
        return [], 0.0, 0.0

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")),
        ]
    )

    X_baseline = baseline.features[FEATURE_NAMES]
    y_baseline = baseline.labels
    fitted_baseline = clone(model).fit(X_baseline, y_baseline)
    baseline_coefs = fitted_baseline.named_steps["model"].coef_[0]
    baseline_probs = fitted_baseline.predict_proba(X_baseline)[:, 1]
    baseline_auc = roc_auc_score(y_baseline, baseline_probs)

    X_biased = biased.features[FEATURE_NAMES]
    y_biased = biased.labels
    fitted_biased = clone(model).fit(X_biased, y_biased)
    biased_coefs = fitted_biased.named_steps["model"].coef_[0]
    biased_probs = fitted_biased.predict_proba(X_biased)[:, 1]
    biased_auc = roc_auc_score(y_biased, biased_probs)

    results: List[RegressionStats] = []
    for idx, feature in enumerate(FEATURE_NAMES):
        baseline_coef = baseline_coefs[idx]
        biased_coef = biased_coefs[idx]
        results.append(
            RegressionStats(
                feature=feature,
                baseline_coef=float(baseline_coef),
                baseline_odds_ratio=float(2.718281828 ** baseline_coef),
                biased_coef=float(biased_coef),
                biased_odds_ratio=float(2.718281828 ** biased_coef),
                coef_change=float(biased_coef - baseline_coef),
            )
        )

    return results, float(baseline_auc), float(biased_auc)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def write_report(
    path: Path,
    datasets: List[Dataset],
    doc_stats: DocumentationStats,
    regression_results: List[RegressionStats],
    baseline_auc: float,
    biased_auc: float,
) -> None:
    lines: List[str] = []
    lines.append("# Documentation Bias Analysis")
    lines.append("")
    lines.append(
        "This report analyzes the effects of documentation bias on diabetes-related features. "
        "The biased dataset has 30% random under-documentation of hyperglycemia and hypertriglyceridemia diagnoses."
    )
    lines.append("")

    lines.append("## Dataset Summary")
    lines.append("")
    lines.append("| Dataset | Patients | Diabetes | Hyperglycemia | Hypertriglyceridemia |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for dataset in datasets:
        lines.append(
            "| {name} | {n:,} | {diabetes:,} | {hg:,} | {ht:,} |".format(
                name=dataset.name,
                n=dataset.patient_count,
                diabetes=dataset.diabetes_count,
                hg=dataset.hyperglycemia_count,
                ht=dataset.hypertriglyceridemia_count,
            )
        )
    lines.append("")

    lines.append("## Documentation Rate Comparison")
    lines.append("")
    lines.append(
        "Documentation rates for the metabolic conditions affected by the bias intervention."
    )
    lines.append("")
    lines.append("| Condition | Baseline Rate | Biased Rate | Reduction |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(
        "| Hyperglycemia | {b} | {bi} | {r} |".format(
            b=_format_pct(doc_stats.baseline_hyperglycemia_rate),
            bi=_format_pct(doc_stats.biased_hyperglycemia_rate),
            r=_format_pct(doc_stats.hyperglycemia_reduction),
        )
    )
    lines.append(
        "| Hypertriglyceridemia | {b} | {bi} | {r} |".format(
            b=_format_pct(doc_stats.baseline_hypertriglyceridemia_rate),
            bi=_format_pct(doc_stats.biased_hypertriglyceridemia_rate),
            r=_format_pct(doc_stats.hypertriglyceridemia_reduction),
        )
    )
    lines.append("")

    lines.append("## Documentation Among Diabetes Patients")
    lines.append("")
    lines.append(
        "Among patients with a diabetes diagnosis, the rate of documented metabolic conditions."
    )
    lines.append("")
    lines.append("| Condition | Baseline Rate | Biased Rate |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        "| Hyperglycemia | {b} | {bi} |".format(
            b=_format_pct(doc_stats.baseline_diabetes_hyperglycemia_rate),
            bi=_format_pct(doc_stats.biased_diabetes_hyperglycemia_rate),
        )
    )
    lines.append(
        "| Hypertriglyceridemia | {b} | {bi} |".format(
            b=_format_pct(doc_stats.baseline_diabetes_hypertriglyceridemia_rate),
            bi=_format_pct(doc_stats.biased_diabetes_hypertriglyceridemia_rate),
        )
    )
    lines.append("")

    lines.append("## Logistic Regression Coefficients")
    lines.append("")
    lines.append(
        "Logistic regression coefficients (standardized) for predicting diabetes. "
        "Higher positive coefficients indicate stronger association with diabetes diagnosis."
    )
    lines.append("")
    lines.append("| Feature | Baseline Coef | Baseline OR | Biased Coef | Biased OR | Δ Coef |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for result in regression_results:
        lines.append(
            "| {feature} | {b_coef} | {b_or} | {bi_coef} | {bi_or} | {delta} |".format(
                feature=result.feature,
                b_coef=_format_metric(result.baseline_coef),
                b_or=_format_metric(result.baseline_odds_ratio),
                bi_coef=_format_metric(result.biased_coef),
                bi_or=_format_metric(result.biased_odds_ratio),
                delta=f"{result.coef_change:+.3f}",
            )
        )
    lines.append("")

    lines.append("## In-Sample Model Performance")
    lines.append("")
    lines.append("| Dataset | AUC |")
    lines.append("| --- | ---: |")
    lines.append(f"| baseline | {_format_metric(baseline_auc)} |")
    lines.append(f"| biased | {_format_metric(biased_auc)} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Documentation bias affects the learned model in several ways:"
    )
    lines.append("")
    lines.append(
        "1. **Reduced feature prevalence**: The biased dataset shows lower rates of documented "
        "hyperglycemia and hypertriglyceridemia, matching the 30% under-documentation rate."
    )
    lines.append("")
    lines.append(
        "2. **Coefficient changes**: With fewer documented metabolic conditions, the model may "
        "compensate by increasing reliance on other features (demographics, comorbidities)."
    )
    lines.append("")
    lines.append(
        "3. **Impact on diabetes patients**: Among actual diabetes patients, the reduced documentation "
        "of highly predictive features creates a mismatch between true clinical state and recorded data."
    )
    lines.append("")
    lines.append(
        "4. **Generalization risk**: A model trained on biased data may perform worse when deployed "
        "on populations with complete documentation, as it has learned to under-weight the most "
        "predictive features."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze documentation bias effects on diabetes diagnosis."
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
        "--out",
        default=str(base_dir / "output" / "diabetes_analytics_report.md"),
        help="Output markdown path.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    baseline = load_dataset("baseline", Path(args.baseline))
    biased = load_dataset("biased", Path(args.biased))

    doc_stats = compute_documentation_stats(baseline, biased)
    regression_results, baseline_auc, biased_auc = regression_analysis(baseline, biased)

    write_report(
        Path(args.out),
        [baseline, biased],
        doc_stats,
        regression_results,
        baseline_auc,
        biased_auc,
    )
    print(f"Wrote report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
