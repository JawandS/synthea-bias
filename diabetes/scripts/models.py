#!/usr/bin/env python3
"""Train diabetes diagnosis models and write a markdown report."""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DIABETES_CODE = "44054006"
BMI_CODE = "39156-5"
SMOKING_STATUS_CODE = "72166-2"
SMOKER_VALUES = {"smokes tobacco daily (finding)"}
NON_SMOKER_VALUES = {"ex-smoker (finding)", "former smoker (finding)", "never smoked tobacco (finding)"}
OBESITY_CODES = {"162864005", "408512008", "414915002"}
HYPERTENSION_CODES = {"59621000"}
HYPERLIPIDEMIA_CODES = {"55822004"}
HYPERGLYCEMIA_CODE = "80394007"
HYPERTRIGLYCERIDEMIA_CODE = "302870006"

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


@dataclass
class Dataset:
    name: str
    features: pd.DataFrame
    labels: pd.Series
    prevalence: float


@dataclass
class ModelResult:
    dataset: str
    model: str
    estimator: Pipeline
    metrics: Dict[str, float]
    split_sizes: Tuple[int, int, int]
    params: Dict[str, object]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass
class CrossEvaluation:
    model: str
    baseline_metrics: Dict[str, float]
    biased_metrics: Dict[str, float]
    deltas: Dict[str, float]


@dataclass
class CohortStats:
    baseline_original: int
    biased_original: int
    common_ids: int
    baseline_dropped: int
    biased_dropped: int


@dataclass
class ProgressReporter:
    total: int
    current: int = 0
    bar_width: int = 28

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _render_bar(self) -> str:
        if self.total <= 0:
            return "[" + "-" * self.bar_width + "]"
        ratio = min(max(self.current / self.total, 0.0), 1.0)
        filled = int(round(self.bar_width * ratio))
        return "[" + "#" * filled + "-" * (self.bar_width - filled) + "]"

    def log(self, message: str) -> None:
        print(f"[{self._timestamp()}] [{self.current:>4}/{self.total:<4}] {message}")

    def advance(self, message: Optional[str] = None, step: int = 1) -> None:
        self.current = min(self.total, self.current + step)
        prefix = f"[{self._timestamp()}] [{self.current:>4}/{self.total:<4}] {self._render_bar()}"
        if message:
            print(f"{prefix} {message}")
        else:
            print(prefix)


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
) -> Tuple[pd.DataFrame, pd.Series, float]:
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
    prevalence = float(labels.mean()) if len(labels) else 0.0

    return features, pd.Series(labels, index=features.index), prevalence


def load_dataset(
    name: str,
    data_dir: Path,
    progress: Optional[ProgressReporter] = None,
) -> Dataset:
    patients_path = data_dir / "patients.csv"
    conditions_path = data_dir / "conditions.csv"
    observations_path = data_dir / "observations.csv"

    if progress:
        progress.log(f"Loading dataset '{name}' from {data_dir}")

    if not patients_path.exists():
        raise FileNotFoundError(f"Missing patients.csv at {patients_path}")
    if not conditions_path.exists():
        raise FileNotFoundError(f"Missing conditions.csv at {conditions_path}")
    if not observations_path.exists():
        raise FileNotFoundError(f"Missing observations.csv at {observations_path}")

    patients = _read_csv(patients_path, PATIENTS_HEADERS)
    conditions = _read_csv(conditions_path, CONDITIONS_HEADERS)
    observations = _read_csv(observations_path, OBSERVATIONS_HEADERS)

    features, labels, prevalence = _build_feature_frame(patients, conditions, observations)

    if progress:
        progress.advance(
            f"Loaded '{name}': {len(labels):,} patients, prevalence={_format_pct(prevalence)}"
        )

    return Dataset(name=name, features=features, labels=labels, prevalence=prevalence)


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be less than 1.0")

    stratify = y if y.nunique() > 1 else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - train_frac),
        random_state=seed,
        stratify=stratify,
    )

    val_ratio = val_frac / (val_frac + test_frac)
    stratify_temp = y_temp if y_temp.nunique() > 1 else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_ratio),
        random_state=seed,
        stratify=stratify_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_dataset(
    dataset: Dataset,
    train_frac: float,
    val_frac: float,
    seed: int,
    progress: Optional[ProgressReporter] = None,
) -> SplitData:
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        dataset.features,
        dataset.labels,
        train_frac,
        val_frac,
        seed,
    )
    if progress:
        progress.advance(f"Split dataset '{dataset.name}' into train/val/test")
    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def _evaluate(y_true: pd.Series, y_prob: Sequence[float]) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def _select_best_model(
    pipeline: Pipeline,
    param_grid: List[Dict[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    progress: Optional[ProgressReporter] = None,
) -> Tuple[Pipeline, Dict[str, object], Dict[str, float]]:
    best_score = -1.0
    best_params: Dict[str, object] = {}
    best_metrics: Dict[str, float] = {}

    if progress:
        progress.log(f"Searching hyperparameters ({len(param_grid)} candidates)")
    for idx, params in enumerate(param_grid, start=1):
        if progress:
            progress.advance(f"candidate {idx}/{len(param_grid)}")
        model = clone(pipeline)
        model.set_params(**params)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1].tolist()
        metrics = _evaluate(y_val, probs)
        score = metrics["auc"]
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    if progress:
        progress.log(f"Selected best params: {best_params}")
        progress.log("Retraining on train+val set")
    final_model = clone(pipeline)
    final_model.set_params(**best_params)
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    final_model.fit(X_train_val, y_train_val)
    if progress:
        progress.log("Finished retraining final model")
    return final_model, best_params, best_metrics


def _build_model_specs(seed: int) -> List[Tuple[str, Pipeline, List[Dict[str, object]]]]:
    logit = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, solver="lbfgs"),
            ),
        ]
    )
    logit_grid = [
        {"model__C": c, "model__class_weight": cw}
        for c in [0.1, 1.0, 10.0]
        for cw in [None, "balanced"]
    ]

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(random_state=seed, n_jobs=-1),
            ),
        ]
    )
    rf_grid = [
        {
            "model__n_estimators": n_estimators,
            "model__max_depth": max_depth,
            "model__min_samples_leaf": min_samples_leaf,
            "model__class_weight": class_weight,
        }
        for n_estimators in [200, 500]
        for max_depth in [None, 5, 10]
        for min_samples_leaf in [1, 5]
        for class_weight in [None, "balanced"]
    ]

    gbdt = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(random_state=seed),
            ),
        ]
    )
    gbdt_grid = [
        {
            "model__n_estimators": n_estimators,
            "model__learning_rate": learning_rate,
            "model__max_depth": max_depth,
            "model__min_samples_leaf": min_samples_leaf,
        }
        for n_estimators in [100, 200]
        for learning_rate in [0.05, 0.1]
        for max_depth in [2, 3]
        for min_samples_leaf in [5, 10]
    ]

    return [
        ("logistic", logit, logit_grid),
        ("rf", rf, rf_grid),
        ("gbdt", gbdt, gbdt_grid),
    ]


def train_models(
    dataset: Dataset,
    seed: int,
    train_frac: float,
    val_frac: float,
    split: Optional[SplitData] = None,
    progress: Optional[ProgressReporter] = None,
) -> List[ModelResult]:
    if split is None:
        split = split_dataset(dataset, train_frac, val_frac, seed, progress=progress)
    split_sizes = (len(split.y_train), len(split.y_val), len(split.y_test))

    results: List[ModelResult] = []
    if progress:
        progress.log(f"Training on dataset '{dataset.name}' (seed={seed})")
    for name, pipeline, grid in _build_model_specs(seed):
        if progress:
            progress.log(f"Starting model '{name}'")
        model, params, _ = _select_best_model(
            pipeline,
            grid,
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
            progress=progress,
        )
        probs = model.predict_proba(split.X_test)[:, 1].tolist()
        metrics = _evaluate(split.y_test, probs)
        if progress:
            progress.advance(
                f"Finished model '{name}': auc={metrics['auc']:.3f}, ap={metrics['ap']:.3f}"
            )
        results.append(
            ModelResult(
                dataset=dataset.name,
                model=name,
                estimator=model,
                metrics=metrics,
                split_sizes=split_sizes,
                params=params,
            )
        )
    return results


def filter_to_common_ids(
    baseline: Dataset,
    biased: Dataset,
    progress: Optional[ProgressReporter] = None,
) -> Tuple[Dataset, Dataset, CohortStats]:
    """Filter both datasets to only include patient IDs present in both."""
    baseline_ids = set(baseline.features.index)
    biased_ids = set(biased.features.index)
    common_ids = baseline_ids & biased_ids

    stats = CohortStats(
        baseline_original=len(baseline_ids),
        biased_original=len(biased_ids),
        common_ids=len(common_ids),
        baseline_dropped=len(baseline_ids) - len(common_ids),
        biased_dropped=len(biased_ids) - len(common_ids),
    )

    common_index = pd.Index(sorted(common_ids))

    baseline_features = baseline.features.loc[common_index]
    baseline_labels = baseline.labels.loc[common_index]
    baseline_prevalence = float(baseline_labels.mean()) if len(baseline_labels) else 0.0

    biased_features = biased.features.loc[common_index]
    biased_labels = biased.labels.loc[common_index]
    biased_prevalence = float(biased_labels.mean()) if len(biased_labels) else 0.0

    if progress:
        progress.advance(
            f"Filtered to {len(common_ids):,} common IDs "
            f"(dropped {stats.baseline_dropped:,} baseline, {stats.biased_dropped:,} biased)"
        )

    return (
        Dataset(name=baseline.name, features=baseline_features, labels=baseline_labels, prevalence=baseline_prevalence),
        Dataset(name=biased.name, features=biased_features, labels=biased_labels, prevalence=biased_prevalence),
        stats,
    )


def split_by_ids(
    baseline: Dataset,
    biased: Dataset,
    train_frac: float,
    val_frac: float,
    seed: int,
    progress: Optional[ProgressReporter] = None,
) -> Tuple[SplitData, SplitData]:
    """Split both datasets using the same patient IDs for train/val/test."""
    # Use baseline labels for stratification (same patients, same diabetes status)
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        baseline.features,
        baseline.labels,
        train_frac,
        val_frac,
        seed,
    )

    train_ids = X_train.index
    val_ids = X_val.index
    test_ids = X_test.index

    baseline_split = SplitData(
        X_train=baseline.features.loc[train_ids],
        X_val=baseline.features.loc[val_ids],
        X_test=baseline.features.loc[test_ids],
        y_train=baseline.labels.loc[train_ids],
        y_val=baseline.labels.loc[val_ids],
        y_test=baseline.labels.loc[test_ids],
    )

    biased_split = SplitData(
        X_train=biased.features.loc[train_ids],
        X_val=biased.features.loc[val_ids],
        X_test=biased.features.loc[test_ids],
        y_train=biased.labels.loc[train_ids],
        y_val=biased.labels.loc[val_ids],
        y_test=biased.labels.loc[test_ids],
    )

    if progress:
        progress.advance(
            f"Split datasets: train={len(train_ids):,}, val={len(val_ids):,}, test={len(test_ids):,}"
        )

    return baseline_split, biased_split


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def write_report(
    path: Path,
    datasets: List[Dataset],
    results: List[ModelResult],
    train_frac: float,
    val_frac: float,
    cross_results: List[CrossEvaluation],
    cross_test_size: int,
    cohort_stats: CohortStats,
    progress: Optional[ProgressReporter] = None,
) -> None:
    lines: List[str] = []
    lines.append("# Diabetes Diagnosis Model Report")
    lines.append("")
    lines.append(f"Train/val/test split: {train_frac:.2f}/{val_frac:.2f}/{1.0 - train_frac - val_frac:.2f}")
    lines.append("")
    lines.append("## Model Specification")
    lines.append("")
    lines.append("**Features:**")
    lines.append("")
    for feature in FEATURE_NAMES:
        lines.append(f"- {feature}")
    lines.append("")
    lines.append(
        "Models are trained with validation-based hyperparameter selection, then re-fit on the combined "
        "train+validation split before final evaluation on the held-out test set."
    )
    lines.append("")

    lines.append("## Cohort Selection")
    lines.append("")
    lines.append(
        "To ensure comparable results, both datasets are filtered to include only patient IDs that appear "
        "in both baseline and biased datasets. The same train/val/test split is then applied to both."
    )
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("| --- | ---: |")
    lines.append(f"| Baseline original patients | {cohort_stats.baseline_original:,} |")
    lines.append(f"| Biased original patients | {cohort_stats.biased_original:,} |")
    lines.append(f"| Common patient IDs (kept) | {cohort_stats.common_ids:,} |")
    lines.append(f"| Baseline patients dropped | {cohort_stats.baseline_dropped:,} |")
    lines.append(f"| Biased patients dropped | {cohort_stats.biased_dropped:,} |")
    lines.append("")

    lines.append("## Dataset Summary")
    lines.append("")
    lines.append("| Dataset | Patients | Diabetes Prevalence |")
    lines.append("| --- | ---: | ---: |")
    for dataset in datasets:
        lines.append(
            f"| {dataset.name} | {len(dataset.labels):,} | {_format_pct(dataset.prevalence)} |"
        )
    lines.append("")

    lines.append("## Test Performance")
    lines.append("")
    lines.append("| Dataset | Model | AUC | AP | Brier | Train/Val/Test |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for result in results:
        train_n, val_n, test_n = result.split_sizes
        lines.append(
            "| {dataset} | {model} | {auc} | {ap} | {brier} | {split} |".format(
                dataset=result.dataset,
                model=result.model,
                auc=_format_metric(result.metrics["auc"]),
                ap=_format_metric(result.metrics["ap"]),
                brier=_format_metric(result.metrics["brier"]),
                split=f"{train_n}/{val_n}/{test_n}",
            )
        )
    lines.append("")

    lines.append("## Cross-Dataset Evaluation")
    lines.append("")
    lines.append(
        "Both models are evaluated on the same held-out test set using baseline feature values. "
        "This directly compares how training on biased vs baseline data affects predictions for identical patients."
    )
    lines.append("")
    if cross_results:
        lines.append("| Model | Baseline AUC | Biased AUC | Δ AUC | Baseline AP | Biased AP | Δ AP | Baseline Brier | Biased Brier | Δ Brier | Test N |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result in cross_results:
            lines.append(
                "| {model} | {b_auc} | {bi_auc} | {d_auc} | {b_ap} | {bi_ap} | {d_ap} | {b_brier} | {bi_brier} | {d_brier} | {n} |".format(
                    model=result.model,
                    b_auc=_format_metric(result.baseline_metrics["auc"]),
                    bi_auc=_format_metric(result.biased_metrics["auc"]),
                    d_auc=_format_metric(result.deltas["auc"]),
                    b_ap=_format_metric(result.baseline_metrics["ap"]),
                    bi_ap=_format_metric(result.biased_metrics["ap"]),
                    d_ap=_format_metric(result.deltas["ap"]),
                    b_brier=_format_metric(result.baseline_metrics["brier"]),
                    bi_brier=_format_metric(result.biased_metrics["brier"]),
                    d_brier=_format_metric(result.deltas["brier"]),
                    n=f"{cross_test_size:,}",
                )
            )
        lines.append("")
    else:
        lines.append("_No overlapping baseline test patients available after filtering._")
        lines.append("")

    lines.append("## Performance Degradation Interpretation")
    lines.append("")
    if cross_results:
        lines.append(
            "Training on the biased dataset (with 30% under-documentation of hyperglycemia/hypertriglyceridemia) "
            "affects performance when evaluated on the baseline test cohort:"
        )
        lines.append("")
        for result in cross_results:
            delta_auc = result.deltas["auc"]
            delta_ap = result.deltas["ap"]
            delta_brier = result.deltas["brier"]
            lines.append(
                "- {model}: AUC {auc:+.3f}, AP {ap:+.3f}, Brier {brier:+.3f} "
                "(negative AUC/AP and positive Brier indicate worse performance).".format(
                    model=result.model,
                    auc=delta_auc,
                    ap=delta_ap,
                    brier=delta_brier,
                )
            )
        lines.append("")
        lines.append(
            "This gap suggests documentation bias in the training data affects model generalization, "
            "particularly because hyperglycemia and hypertriglyceridemia are highly predictive of diabetes."
        )
        lines.append("")
    else:
        lines.append(
            "Cross-dataset evaluation was skipped because no eligible overlapping baseline test patients remained "
            "after filtering, so degradation could not be quantified."
        )
        lines.append("")

    lines.append("## Selected Hyperparameters")
    lines.append("")
    for dataset in datasets:
        lines.append(f"### {dataset.name}")
        lines.append("")
        subset = [result for result in results if result.dataset == dataset.name]
        for result in subset:
            lines.append(f"- **{result.model}**: {result.params}")
        lines.append("")

    if progress:
        progress.log(f"Writing report to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    if progress:
        progress.advance(f"Report written to {path}")


def save_models(output_dir: Path, results: List[ModelResult], progress: Optional[ProgressReporter] = None) -> None:
    # Note: pickle is used here for sklearn model serialization, which is standard practice
    # for ML pipelines. These files are only used internally for model persistence.
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        filename = f"{result.dataset}_{result.model}.pkl"
        path = output_dir / filename
        with path.open("wb") as handle:
            pickle.dump(result.estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if progress:
            progress.advance(f"Saved model to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train diabetes diagnosis models and write a markdown report."
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
        "--train-frac",
        type=float,
        default=0.7,
        help="Training fraction.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation fraction.",
    )
    parser.add_argument(
        "--out",
        default=str(base_dir / "output" / "diabetes_model_report.md"),
        help="Output markdown path.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(base_dir / "output" / "models"),
        help="Directory to store trained model artifacts.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    model_specs = _build_model_specs(args.seed)
    total_candidates = sum(len(grid) for _, _, grid in model_specs)
    model_count = len(model_specs) * 2
    # Steps: 2 load + 1 filter + 1 split + 2*(candidates + models) + 1 cross-eval + 1 report + model_count save
    total_steps = 2 + 1 + 1 + (2 * (total_candidates + len(model_specs))) + 1 + 1 + model_count
    progress = ProgressReporter(total=total_steps)

    # Load both datasets
    baseline_raw = load_dataset("baseline", Path(args.baseline), progress=progress)
    biased_raw = load_dataset("biased", Path(args.biased), progress=progress)

    # Filter to common patient IDs
    baseline, biased, cohort_stats = filter_to_common_ids(baseline_raw, biased_raw, progress=progress)

    # Split using the same patient IDs for both datasets
    baseline_split, biased_split = split_by_ids(
        baseline, biased, args.train_frac, args.val_frac, args.seed, progress=progress
    )

    # Train models on each dataset
    results: List[ModelResult] = []
    results.extend(
        train_models(
            baseline,
            args.seed,
            args.train_frac,
            args.val_frac,
            split=baseline_split,
            progress=progress,
        )
    )
    results.extend(
        train_models(
            biased,
            args.seed,
            args.train_frac,
            args.val_frac,
            split=biased_split,
            progress=progress,
        )
    )

    # Cross-evaluation: evaluate both models on baseline test features
    baseline_models = {result.model: result.estimator for result in results if result.dataset == "baseline"}
    biased_models = {result.model: result.estimator for result in results if result.dataset == "biased"}

    cross_results: List[CrossEvaluation] = []
    cross_test_size = len(baseline_split.X_test)

    # Use baseline test features for both models (same patients, baseline feature values)
    X_test = baseline_split.X_test
    y_test = baseline_split.y_test

    for model_name, biased_model in biased_models.items():
        baseline_model = baseline_models.get(model_name)
        if baseline_model is None:
            continue
        baseline_probs = baseline_model.predict_proba(X_test)[:, 1].tolist()
        biased_probs = biased_model.predict_proba(X_test)[:, 1].tolist()
        baseline_metrics = _evaluate(y_test, baseline_probs)
        biased_metrics = _evaluate(y_test, biased_probs)
        deltas = {
            key: biased_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
        }
        cross_results.append(
            CrossEvaluation(
                model=model_name,
                baseline_metrics=baseline_metrics,
                biased_metrics=biased_metrics,
                deltas=deltas,
            )
        )

    if progress:
        progress.advance("Cross-dataset evaluation complete")

    write_report(
        Path(args.out),
        [baseline, biased],
        results,
        args.train_frac,
        args.val_frac,
        cross_results,
        cross_test_size,
        cohort_stats,
        progress=progress,
    )
    save_models(Path(args.model_dir), results, progress=progress)
    print(f"Wrote report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
