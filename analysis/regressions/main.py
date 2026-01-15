#!/usr/bin/env python3
"""Run linear regressions and diagnosis classifiers to assess disease modeling feasibility."""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent / "sleep_apnea"
sys.path.append(str(UTILS_DIR))

from utils import (  # type: ignore
    CONDITIONS_HEADERS,
    ENCOUNTERS_HEADERS,
    MEDICATIONS_HEADERS,
    OBSERVATIONS_HEADERS,
    PATIENTS_HEADERS,
    PROCEDURES_HEADERS,
    find_csv_dir,
    find_dataset_end_date,
    get_value,
    impute_missing,
    iter_csv_rows,
    load_latest_smoking_status,
    parse_date,
    parse_float,
)


SMOKING_STATUS_CODE = "72166-2"
SMOKER_VALUES = {"Smokes tobacco daily (finding)"}
NON_SMOKER_VALUES = {"Ex-smoker (finding)", "Never smoked tobacco (finding)"}


OBSERVATION_CODES = {
    "bmi": {"39156-5"},
    "a1c": {"4548-4"},
    "glucose": {"2339-0", "2345-7"},
    "triglycerides": {"2571-8"},
    "hdl": {"2085-9"},
    "ldl": {"18262-6"},
    "total_cholesterol": {"2093-3"},
    "systolic_bp": {"8480-6"},
    "diastolic_bp": {"8462-4"},
    "fev1_fvc": {"19926-5"},
}


CONDITION_GROUPS = {
    "hypertension": {"59621000"},
    "diabetes_dx": {"44054006"},
    "prediabetes": {"714628002"},
    "hyperlipidemia": {"55822004"},
    "copd_dx": {"87433001", "185086009"},
    "mi_dx": {"22298006", "401303003", "401314000", "399211009"},
    "asthma": {"233678006", "195967001"},
    "obesity": {"162864005", "408512008"},
    "hyperglycemia": {"80394007"},
    "hypertriglyceridemia": {"302870006"},
    "chf": {"88805009"},
}


UTILIZATION_FEATURES = [
    "encounter_count",
    "encounter_emergency",
    "encounter_inpatient",
    "encounter_ambulatory",
    "encounter_outpatient",
    "procedure_count",
    "medication_count",
    "condition_count",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    cohort: str  # "all" or "diagnosed"
    features: List[str]


@dataclass(frozen=True)
class DiseaseConfig:
    key: str
    label: str
    reason_codes: set
    cohort_codes: set
    specs: List[ModelSpec]
    target_label: str
    diagnosis_features: List[str]
    diagnosis_prob_feature: str
    diagnosis_flag_feature: str


@dataclass
class SpecResult:
    spec: ModelSpec
    n: int
    nonzero_rate: float
    metrics: Dict[str, float]
    missing_rates: Dict[str, float]
    dropped_features: List[str] = field(default_factory=list)
    note: Optional[str] = None


@dataclass
class DiagnosisResult:
    features: List[str]
    missing_rates: Dict[str, float]
    dropped_features: List[str]
    auc: float
    brier: float
    prevalence: float
    n_train: int
    n_test: int


@dataclass
class DiseaseResult:
    config: DiseaseConfig
    population_size: int
    diagnosed_size: int
    nonzero_rate: float
    mean_spend_nonzero: float
    spec_results: List[SpecResult]
    diagnosis_result: Optional[DiagnosisResult]
    judgement: str


def evaluate_predictions(y_true: Sequence[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Compute MAE, RMSE, and R2 for a set of predictions."""
    y_pred_list = list(y_pred)
    n = len(y_true)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    abs_err = [abs(p - t) for p, t in zip(y_pred_list, y_true)]
    sq_err = [(p - t) ** 2 for p, t in zip(y_pred_list, y_true)]
    mae = sum(abs_err) / n
    rmse = (sum(sq_err) / n) ** 0.5
    mean_true = sum(y_true) / n
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    ss_res = sum(sq_err)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def load_condition_groups(
    conditions_path: Path,
    groups: Dict[str, set],
) -> Tuple[Dict[str, set], Dict[str, int]]:
    """Load patient sets for multiple condition groups in one pass."""
    patients_by_group = {name: set() for name in groups}
    condition_counts: Dict[str, int] = defaultdict(int)
    code_to_groups: Dict[str, List[str]] = {}
    for group_name, codes in groups.items():
        for code in codes:
            code_to_groups.setdefault(code, []).append(group_name)

    for row, header_map in iter_csv_rows(conditions_path, CONDITIONS_HEADERS):
        code = get_value(row, header_map, ["code"])
        patient_id = get_value(row, header_map, ["patient"])
        if not patient_id:
            continue
        condition_counts[patient_id] += 1
        if not code:
            continue
        group_names = code_to_groups.get(code)
        if not group_names:
            continue
        for group_name in group_names:
            patients_by_group[group_name].add(patient_id)
    return patients_by_group, condition_counts


def load_latest_numeric_observations(
    observations_path: Path,
    code_groups: Dict[str, Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    """Load latest numeric observation for each code -> feature name mapping."""
    values_by_feature: Dict[str, Dict[str, float]] = {name: {} for name in code_groups}
    dates_by_feature: Dict[str, Dict[str, date]] = {name: {} for name in code_groups}
    code_map: Dict[str, str] = {}
    for feature_name, codes in code_groups.items():
        for code in codes:
            code_map[code] = feature_name

    for row, header_map in iter_csv_rows(observations_path, OBSERVATIONS_HEADERS):
        code = get_value(row, header_map, ["code"])
        if not code:
            continue
        feature = code_map.get(code)
        if not feature:
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
        prev_date = dates_by_feature[feature].get(patient_id)
        if prev_date is None or obs_date > prev_date:
            dates_by_feature[feature][patient_id] = obs_date
            values_by_feature[feature][patient_id] = value
    return values_by_feature


def load_encounter_counts(encounters_path: Path) -> Dict[str, Dict[str, int]]:
    """Load per-patient encounter counts by class."""
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tracked_classes = {
        "ambulatory",
        "emergency",
        "inpatient",
        "outpatient",
        "urgentcare",
        "wellness",
    }
    for row, header_map in iter_csv_rows(encounters_path, ENCOUNTERS_HEADERS):
        patient_id = get_value(row, header_map, ["patient"])
        if not patient_id:
            continue
        counts[patient_id]["encounter_count"] += 1
        encounter_class = (get_value(row, header_map, ["encounterclass"]) or "").strip().lower()
        if encounter_class in tracked_classes:
            counts[patient_id][f"encounter_{encounter_class}"] += 1
    return counts


def load_row_counts(path: Path, headers: Sequence[str]) -> Dict[str, int]:
    """Count rows per patient for a CSV file."""
    counts: Dict[str, int] = defaultdict(int)
    for row, header_map in iter_csv_rows(path, headers):
        patient_id = get_value(row, header_map, ["patient"])
        if patient_id:
            counts[patient_id] += 1
    return counts


def build_patient_features(
    csv_dir: Path,
    condition_groups: Dict[str, set],
    condition_counts: Dict[str, int],
    observation_values: Dict[str, Dict[str, float]],
    smoker_by_patient: Dict[str, float],
    encounter_counts: Dict[str, Dict[str, int]],
    procedure_counts: Dict[str, int],
    medication_counts: Dict[str, int],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Build feature dicts for each patient."""
    patients_path = csv_dir / "patients.csv"
    ref_date = find_dataset_end_date(csv_dir)
    if ref_date is None:
        ref_date = date.today()

    features_by_patient: Dict[str, Dict[str, Optional[float]]] = {}
    for row, header_map in iter_csv_rows(patients_path, PATIENTS_HEADERS):
        patient_id = get_value(row, header_map, ["id"])
        if not patient_id:
            continue

        birthdate = parse_date(get_value(row, header_map, ["birthdate"]))
        if birthdate is None:
            age_years = None
        else:
            age_years = (ref_date - birthdate).days / 365.25
            if age_years < 0:
                age_years = None

        gender = (get_value(row, header_map, ["gender"]) or "").strip().lower()
        male = 1.0 if gender == "m" else 0.0
        income = parse_float(get_value(row, header_map, ["income"]))

        features: Dict[str, Optional[float]] = {
            "age_years": age_years,
            "male": male,
            "income": income,
            "smoker": smoker_by_patient.get(patient_id),
            "condition_count": float(condition_counts.get(patient_id, 0)),
            "procedure_count": float(procedure_counts.get(patient_id, 0)),
            "medication_count": float(medication_counts.get(patient_id, 0)),
        }

        encounter_summary = encounter_counts.get(patient_id, {})
        features["encounter_count"] = float(encounter_summary.get("encounter_count", 0))
        features["encounter_emergency"] = float(encounter_summary.get("encounter_emergency", 0))
        features["encounter_inpatient"] = float(encounter_summary.get("encounter_inpatient", 0))
        features["encounter_ambulatory"] = float(encounter_summary.get("encounter_ambulatory", 0))
        features["encounter_outpatient"] = float(encounter_summary.get("encounter_outpatient", 0))

        for feature_name, values in observation_values.items():
            features[feature_name] = values.get(patient_id)

        for group_name, patient_set in condition_groups.items():
            features[group_name] = 1.0 if patient_id in patient_set else 0.0

        features["diabetes_any"] = 1.0 if (
            patient_id in condition_groups["diabetes_dx"]
            or patient_id in condition_groups["prediabetes"]
        ) else 0.0

        features_by_patient[patient_id] = features
    return features_by_patient


def load_disease_spend(
    csv_dir: Path,
    diseases: Sequence[DiseaseConfig],
) -> Dict[str, Dict[str, float]]:
    """Compute disease-specific spend per patient using reason codes."""
    spend_by_disease: Dict[str, Dict[str, float]] = {
        disease.key: defaultdict(float) for disease in diseases
    }
    disease_encounters: Dict[str, set] = {disease.key: set() for disease in diseases}

    encounters_path = csv_dir / "encounters.csv"
    for row, header_map in iter_csv_rows(encounters_path, ENCOUNTERS_HEADERS):
        reason_code = get_value(row, header_map, ["reasoncode"])
        if not reason_code:
            continue
        encounter_id = get_value(row, header_map, ["id"])
        patient_id = get_value(row, header_map, ["patient"])
        cost = parse_float(get_value(row, header_map, ["total_claim_cost"]))
        if cost is None:
            cost = parse_float(get_value(row, header_map, ["base_encounter_cost"]))
        for disease in diseases:
            if reason_code not in disease.reason_codes:
                continue
            if encounter_id:
                disease_encounters[disease.key].add(encounter_id)
            if patient_id and cost is not None:
                spend_by_disease[disease.key][patient_id] += cost

    procedures_path = csv_dir / "procedures.csv"
    for row, header_map in iter_csv_rows(procedures_path, PROCEDURES_HEADERS):
        reason_code = get_value(row, header_map, ["reasoncode"])
        encounter_id = get_value(row, header_map, ["encounter"])
        patient_id = get_value(row, header_map, ["patient"])
        cost = parse_float(get_value(row, header_map, ["base_cost"]))
        if not patient_id or cost is None:
            continue
        for disease in diseases:
            if (
                (reason_code and reason_code in disease.reason_codes)
                or (encounter_id and encounter_id in disease_encounters[disease.key])
            ):
                spend_by_disease[disease.key][patient_id] += cost

    medications_path = csv_dir / "medications.csv"
    for row, header_map in iter_csv_rows(medications_path, MEDICATIONS_HEADERS):
        reason_code = get_value(row, header_map, ["reasoncode"])
        encounter_id = get_value(row, header_map, ["encounter"])
        patient_id = get_value(row, header_map, ["patient"])
        cost = parse_float(get_value(row, header_map, ["totalcost"]))
        if not patient_id or cost is None:
            continue
        for disease in diseases:
            if (
                (reason_code and reason_code in disease.reason_codes)
                or (encounter_id and encounter_id in disease_encounters[disease.key])
            ):
                spend_by_disease[disease.key][patient_id] += cost

    return spend_by_disease


def build_dataset(
    patient_features: Dict[str, Dict[str, Optional[float]]],
    patient_ids: Sequence[str],
    target_by_patient: Dict[str, float],
    spec: ModelSpec,
    cohort_ids: Optional[set] = None,
) -> Tuple[List[List[Optional[float]]], List[float], List[str]]:
    """Build feature matrix and target vector."""
    X: List[List[Optional[float]]] = []
    y: List[float] = []
    ids: List[str] = []

    for patient_id in patient_ids:
        if cohort_ids is not None and patient_id not in cohort_ids:
            continue
        features = patient_features.get(patient_id)
        if not features:
            continue
        row = [features.get(name) for name in spec.features]
        X.append(row)
        y.append(float(target_by_patient.get(patient_id, 0.0)))
        ids.append(patient_id)
    return X, y, ids


def compute_missing_rates(
    X: Sequence[Sequence[Optional[float]]],
    feature_names: Sequence[str],
) -> Dict[str, float]:
    if not X:
        return {name: 1.0 for name in feature_names}
    missing_counts = [0] * len(feature_names)
    for row in X:
        for idx, value in enumerate(row):
            if value is None:
                missing_counts[idx] += 1
    total = len(X)
    return {name: missing_counts[idx] / total for idx, name in enumerate(feature_names)}


def filter_empty_features(
    X: Sequence[Sequence[Optional[float]]],
    feature_names: List[str],
) -> Tuple[List[List[Optional[float]]], List[str], List[str]]:
    missing_rates = compute_missing_rates(X, feature_names)
    keep_indices = [idx for idx, name in enumerate(feature_names) if missing_rates[name] < 1.0]
    dropped = [name for idx, name in enumerate(feature_names) if idx not in keep_indices]
    if not keep_indices:
        return [list(row) for row in X], feature_names, []
    filtered = [[row[idx] for idx in keep_indices] for row in X]
    filtered_names = [feature_names[idx] for idx in keep_indices]
    return filtered, filtered_names, dropped


def build_classification_dataset(
    patient_features: Dict[str, Dict[str, Optional[float]]],
    patient_ids: Sequence[str],
    feature_names: Sequence[str],
    labels_by_patient: Dict[str, int],
) -> Tuple[List[List[Optional[float]]], List[int], List[str]]:
    """Build feature matrix and labels for diagnosis modeling."""
    X: List[List[Optional[float]]] = []
    y: List[int] = []
    ids: List[str] = []

    for patient_id in patient_ids:
        features = patient_features.get(patient_id)
        if not features:
            continue
        row = [features.get(name) for name in feature_names]
        X.append(row)
        y.append(int(labels_by_patient.get(patient_id, 0)))
        ids.append(patient_id)
    return X, y, ids


def fit_diagnosis_model(
    patient_features: Dict[str, Dict[str, Optional[float]]],
    patient_ids: Sequence[str],
    feature_names: List[str],
    labels_by_patient: Dict[str, int],
    seed: int,
) -> Tuple[DiagnosisResult, Dict[str, float]]:
    """Fit logistic regression for diagnosis probability and return predictions."""
    X_raw, y, ids = build_classification_dataset(
        patient_features, patient_ids, feature_names, labels_by_patient
    )
    missing_rates = compute_missing_rates(X_raw, feature_names)
    filtered_X, filtered_features, dropped = filter_empty_features(X_raw, list(feature_names))

    if len(set(y)) < 2:
        return (
            DiagnosisResult(
                features=filtered_features,
                missing_rates=missing_rates,
                dropped_features=dropped,
                auc=float("nan"),
                brier=float("nan"),
                prevalence=sum(y) / len(y) if y else 0.0,
                n_train=0,
                n_test=0,
            ),
            {patient_id: 0.0 for patient_id in ids},
        )

    X = impute_missing(filtered_X)

    indices = list(range(len(y)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    model.fit(X_train_scaled, y_train)

    y_pred_test = model.predict_proba(X_test_scaled)[:, 1]
    if len(set(y_test)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_test, y_pred_test)
    brier = brier_score_loss(y_test, y_pred_test)

    # Refit on full data for downstream probability features.
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)
    model_full = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    model_full.fit(X_scaled, y)
    y_pred_full = model_full.predict_proba(X_scaled)[:, 1]
    prob_by_patient = {patient_id: prob for patient_id, prob in zip(ids, y_pred_full)}

    result = DiagnosisResult(
        features=filtered_features,
        missing_rates=missing_rates,
        dropped_features=dropped,
        auc=auc,
        brier=brier,
        prevalence=sum(y) / len(y) if y else 0.0,
        n_train=len(train_idx),
        n_test=len(test_idx),
    )
    return result, prob_by_patient


def train_linear_model(
    X: List[List[Optional[float]]],
    y: List[float],
    seed: int,
) -> Dict[str, float]:
    if len(y) < 5:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    indices = list(range(len(y)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train_raw = [X[i] for i in train_idx]
    X_test_raw = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    X_train = impute_missing(X_train_raw)
    X_test = impute_missing(X_test_raw, X_train_raw)

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    y_train_log = [math.log1p(value) for value in y_train]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train_log)

    y_pred_log = model.predict(X_test_scaled)
    y_pred = [max(math.expm1(value), 0.0) for value in y_pred_log]

    return evaluate_predictions(y_test, y_pred)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_metric(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def summarize_disease(
    config: DiseaseConfig,
    patient_ids: Sequence[str],
    patient_features: Dict[str, Dict[str, Optional[float]]],
    target_by_patient: Dict[str, float],
    cohort_ids: set,
    diagnosis_result: Optional[DiagnosisResult],
    seed: int,
) -> DiseaseResult:
    total_patients = len(patient_ids)
    diagnosed_size = len(cohort_ids)
    nonzero_values = [value for value in target_by_patient.values() if value > 0]
    nonzero_rate = len(nonzero_values) / total_patients if total_patients else 0.0
    mean_nonzero = sum(nonzero_values) / len(nonzero_values) if nonzero_values else 0.0

    spec_results: List[SpecResult] = []
    for spec in config.specs:
        cohort = cohort_ids if spec.cohort == "diagnosed" else None
        X_raw, y, _ = build_dataset(patient_features, patient_ids, target_by_patient, spec, cohort)
        missing_rates = compute_missing_rates(X_raw, spec.features)
        filtered_X, filtered_features, dropped = filter_empty_features(X_raw, list(spec.features))
        notes: List[str] = []
        if dropped:
            notes.append(f"Dropped empty features: {', '.join(dropped)}.")
        if any(name in UTILIZATION_FEATURES for name in spec.features):
            notes.append("Includes utilization counts (proxy for spend).")
        if config.diagnosis_flag_feature in spec.features:
            notes.append("Includes diagnosis indicator.")
        if config.diagnosis_prob_feature in spec.features:
            notes.append("Includes diagnosis probability.")
        note = " ".join(notes) if notes else None
        metrics = train_linear_model(filtered_X, y, seed)
        nonzero_rate_spec = sum(1 for value in y if value > 0) / len(y) if y else 0.0
        spec_results.append(
            SpecResult(
                spec=ModelSpec(spec.name, spec.cohort, filtered_features),
                n=len(y),
                nonzero_rate=nonzero_rate_spec,
                metrics=metrics,
                missing_rates=missing_rates,
                dropped_features=dropped,
                note=note,
            )
        )

    judgement = build_judgement(config, diagnosed_size, nonzero_rate, spec_results, diagnosis_result)

    return DiseaseResult(
        config=config,
        population_size=total_patients,
        diagnosed_size=diagnosed_size,
        nonzero_rate=nonzero_rate,
        mean_spend_nonzero=mean_nonzero,
        spec_results=spec_results,
        diagnosis_result=diagnosis_result,
        judgement=judgement,
    )


def build_judgement(
    config: DiseaseConfig,
    diagnosed_size: int,
    nonzero_rate: float,
    spec_results: Sequence[SpecResult],
    diagnosis_result: Optional[DiagnosisResult],
) -> str:
    if diagnosed_size < 200 or nonzero_rate < 0.005:
        return (
            "Insufficient volume of diagnosed cases or spending signal to support strong models."
        )
    best_r2 = max((result.metrics["r2"] for result in spec_results), default=float("nan"))
    if best_r2 >= 0.2:
        return "Good signal in linear models; data looks usable for modeling."
    if best_r2 >= 0.05:
        return "Weak-to-moderate signal; may be usable but expect limited performance."
    if diagnosis_result and diagnosis_result.auc >= 0.8:
        return (
            "Regression signal is weak, but diagnosis probability modeling is strong "
            "(AUC >= 0.8)."
        )
    if diagnosis_result and diagnosis_result.auc >= 0.7:
        return (
            "Regression signal is weak, but diagnosis probability modeling is moderate "
            "(AUC >= 0.7)."
        )
    return "Weak signal overall; likely not enough data for robust models."


def write_results(path: Path, results: Sequence[DiseaseResult], csv_dir: Path) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append("# Regression Feasibility Results")
    lines.append("")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Dataset: `{csv_dir}`")
    lines.append("Linear model: Ridge regression on log1p(spend) with standardized features.")
    lines.append("Diagnosis model: Logistic regression with standardized features.")
    lines.append("")

    for result in results:
        config = result.config
        lines.append(f"## {config.label}")
        lines.append("")
        lines.append(f"Target: {config.target_label}")
        lines.append(f"Reason codes: {', '.join(sorted(config.reason_codes))}")
        lines.append(f"Cohort codes: {', '.join(sorted(config.cohort_codes))}")
        lines.append("")
        lines.append(f"Population size: {result.population_size}")
        if result.population_size:
            pct = result.diagnosed_size / result.population_size
        else:
            pct = 0.0
        lines.append(
            f"Diagnosed patients: {result.diagnosed_size} ({format_percent(pct)})"
        )
        lines.append(f"Nonzero spend rate (population): {format_percent(result.nonzero_rate)}")
        lines.append(f"Mean spend (nonzero only): ${result.mean_spend_nonzero:,.2f}")
        lines.append("")

        if result.diagnosis_result:
            diagnosis = result.diagnosis_result
            lines.append("Diagnosis probability model (logistic regression):")
            lines.append(
                f"- AUC: {format_metric(diagnosis.auc)} | "
                f"Brier: {format_metric(diagnosis.brier)} | "
                f"Prevalence: {format_percent(diagnosis.prevalence)} | "
                f"Train/Test: {diagnosis.n_train}/{diagnosis.n_test}"
            )
            lines.append(f"- Features: {', '.join(diagnosis.features)}")
            if diagnosis.dropped_features:
                lines.append(
                    f"- Dropped empty features: {', '.join(diagnosis.dropped_features)}"
                )
            gaps = [
                f"{name} ({format_percent(rate)} missing)"
                for name, rate in diagnosis.missing_rates.items()
                if rate >= 0.3
            ]
            if gaps:
                lines.append("".join(["- Feature gaps (>30% missing): ", ", ".join(gaps)]))
            lines.append("")

        lines.append("| Spec | Cohort | Features | N | Nonzero | R2 | MAE | RMSE |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for spec_result in result.spec_results:
            metrics = spec_result.metrics
            lines.append(
                "| {name} | {cohort} | {features} | {n} | {nonzero} | {r2:.3f} | ${mae:,.0f} | ${rmse:,.0f} |".format(
                    name=spec_result.spec.name,
                    cohort=spec_result.spec.cohort,
                    features=len(spec_result.spec.features),
                    n=spec_result.n,
                    nonzero=format_percent(spec_result.nonzero_rate),
                    r2=metrics["r2"],
                    mae=metrics["mae"],
                    rmse=metrics["rmse"],
                )
            )
        lines.append("")

        lines.append("Specifications:")
        for spec_result in result.spec_results:
            features = ", ".join(spec_result.spec.features)
            lines.append(f"- {spec_result.spec.name} ({spec_result.spec.cohort}): {features}")
            if spec_result.note:
                lines.append(f"- {spec_result.spec.name} note: {spec_result.note}")
        lines.append("")

        for spec_result in result.spec_results:
            gaps = [
                f"{name} ({format_percent(rate)} missing)"
                for name, rate in spec_result.missing_rates.items()
                if rate >= 0.3
            ]
            if gaps:
                lines.append(
                    f"Feature gaps (>30% missing) for {spec_result.spec.name}: "
                    + ", ".join(gaps)
                )
        lines.append("")

        lines.append(f"Judgement: {result.judgement}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_configs() -> List[DiseaseConfig]:
    diabetes_base = [
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
    diabetes_labs = diabetes_base + [
        "a1c",
        "glucose",
        "triglycerides",
        "hdl",
        "ldl",
        "total_cholesterol",
        "systolic_bp",
        "diastolic_bp",
    ]
    diabetes_dx = "diabetes_dx"
    diabetes_prob = "diagnosis_prob_diabetes"
    diabetes_specs = [
        ModelSpec(name="Population risk", cohort="all", features=diabetes_base),
        ModelSpec(name="Population risk + labs/vitals", cohort="all", features=diabetes_labs),
        ModelSpec(
            name="Population risk + diagnosis flag",
            cohort="all",
            features=diabetes_base + [diabetes_dx],
        ),
        ModelSpec(
            name="Population risk + diagnosis probability",
            cohort="all",
            features=diabetes_base + [diabetes_prob],
        ),
        ModelSpec(
            name="Population risk + utilization",
            cohort="all",
            features=diabetes_base + UTILIZATION_FEATURES,
        ),
        ModelSpec(
            name="Diagnosed cohort risk + labs/vitals",
            cohort="diagnosed",
            features=diabetes_labs,
        ),
    ]

    copd_base = [
        "age_years",
        "male",
        "income",
        "bmi",
        "smoker",
        "asthma",
    ]
    copd_labs = copd_base + ["fev1_fvc"]
    copd_dx = "copd_dx"
    copd_prob = "diagnosis_prob_copd"
    copd_specs = [
        ModelSpec(name="Population risk", cohort="all", features=copd_base),
        ModelSpec(name="Population risk + FEV1/FVC", cohort="all", features=copd_labs),
        ModelSpec(
            name="Population risk + diagnosis flag",
            cohort="all",
            features=copd_base + [copd_dx],
        ),
        ModelSpec(
            name="Population risk + diagnosis probability",
            cohort="all",
            features=copd_base + [copd_prob],
        ),
        ModelSpec(
            name="Population risk + utilization",
            cohort="all",
            features=copd_base + UTILIZATION_FEATURES,
        ),
        ModelSpec(
            name="Diagnosed cohort risk + FEV1/FVC",
            cohort="diagnosed",
            features=copd_labs,
        ),
    ]

    mi_base = [
        "age_years",
        "male",
        "income",
        "bmi",
        "smoker",
        "obesity",
        "hypertension",
        "hyperlipidemia",
        "diabetes_any",
        "chf",
    ]
    mi_labs = mi_base + [
        "systolic_bp",
        "diastolic_bp",
        "total_cholesterol",
        "ldl",
        "hdl",
        "triglycerides",
        "glucose",
        "a1c",
    ]
    mi_dx = "mi_dx"
    mi_prob = "diagnosis_prob_mi"
    mi_specs = [
        ModelSpec(name="Population risk", cohort="all", features=mi_base),
        ModelSpec(name="Population risk + labs/vitals", cohort="all", features=mi_labs),
        ModelSpec(
            name="Population risk + diagnosis flag",
            cohort="all",
            features=mi_base + [mi_dx],
        ),
        ModelSpec(
            name="Population risk + diagnosis probability",
            cohort="all",
            features=mi_base + [mi_prob],
        ),
        ModelSpec(
            name="Population risk + utilization",
            cohort="all",
            features=mi_base + UTILIZATION_FEATURES,
        ),
        ModelSpec(
            name="Diagnosed cohort risk + labs/vitals",
            cohort="diagnosed",
            features=mi_labs,
        ),
    ]

    return [
        DiseaseConfig(
            key="diabetes",
            label="Diabetes care",
            reason_codes={"44054006", "714628002"},
            cohort_codes={"44054006"},
            specs=diabetes_specs,
            target_label=(
                "log1p(diabetes-related spend), from encounters/procedures/medications "
                "with diabetes/prediabetes reason codes"
            ),
            diagnosis_features=diabetes_labs,
            diagnosis_prob_feature=diabetes_prob,
            diagnosis_flag_feature=diabetes_dx,
        ),
        DiseaseConfig(
            key="copd",
            label="COPD",
            reason_codes={"87433001", "185086009"},
            cohort_codes={"87433001", "185086009"},
            specs=copd_specs,
            target_label=(
                "log1p(COPD-related spend), from encounters/procedures/medications "
                "with COPD reason codes"
            ),
            diagnosis_features=copd_labs,
            diagnosis_prob_feature=copd_prob,
            diagnosis_flag_feature=copd_dx,
        ),
        DiseaseConfig(
            key="mi",
            label="Myocardial infarction",
            reason_codes={"22298006", "401303003", "401314000", "399211009"},
            cohort_codes={"22298006", "401303003", "401314000", "399211009"},
            specs=mi_specs,
            target_label=(
                "log1p(MI-related spend), from encounters/procedures/medications "
                "with MI reason codes"
            ),
            diagnosis_features=mi_labs,
            diagnosis_prob_feature=mi_prob,
            diagnosis_flag_feature=mi_dx,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run linear regressions for diabetes, COPD, and MI on Synthea outputs."
    )
    parser.add_argument(
        "--data",
        default="output_baseline",
        help="Path to Synthea output directory (root or csv subdir).",
    )
    parser.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "RESULTS.md"),
        help="Output markdown path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    csv_dir = find_csv_dir(args.data)

    conditions_path = csv_dir / "conditions.csv"
    observations_path = csv_dir / "observations.csv"

    condition_groups, condition_counts = load_condition_groups(conditions_path, CONDITION_GROUPS)

    smoker_by_patient = load_latest_smoking_status(
        observations_path,
        SMOKING_STATUS_CODE,
        SMOKER_VALUES,
        NON_SMOKER_VALUES,
    )

    observation_values = load_latest_numeric_observations(
        observations_path,
        OBSERVATION_CODES,
    )

    encounter_counts = load_encounter_counts(csv_dir / "encounters.csv")
    procedure_counts = load_row_counts(csv_dir / "procedures.csv", PROCEDURES_HEADERS)
    medication_counts = load_row_counts(csv_dir / "medications.csv", MEDICATIONS_HEADERS)

    patient_features = build_patient_features(
        csv_dir,
        condition_groups,
        condition_counts,
        observation_values,
        smoker_by_patient,
        encounter_counts,
        procedure_counts,
        medication_counts,
    )

    patient_ids = list(patient_features.keys())

    configs = build_configs()
    spend_by_disease = load_disease_spend(csv_dir, configs)

    cohort_map = {
        "diabetes": condition_groups.get("diabetes_dx", set()),
        "copd": condition_groups.get("copd_dx", set()),
        "mi": condition_groups.get("mi_dx", set()),
    }

    diagnosis_results: Dict[str, DiagnosisResult] = {}
    for config in configs:
        cohort_ids = cohort_map.get(config.key, set())
        labels_by_patient = {
            patient_id: 1 if patient_id in cohort_ids else 0 for patient_id in patient_ids
        }
        diagnosis_result, prob_by_patient = fit_diagnosis_model(
            patient_features,
            patient_ids,
            config.diagnosis_features,
            labels_by_patient,
            args.seed,
        )
        diagnosis_results[config.key] = diagnosis_result
        for patient_id, prob in prob_by_patient.items():
            patient_features[patient_id][config.diagnosis_prob_feature] = prob

    results: List[DiseaseResult] = []
    for config in configs:
        cohort_ids = cohort_map.get(config.key, set())
        results.append(
            summarize_disease(
                config=config,
                patient_ids=patient_ids,
                patient_features=patient_features,
                target_by_patient=spend_by_disease.get(config.key, {}),
                cohort_ids=cohort_ids,
                diagnosis_result=diagnosis_results.get(config.key),
                seed=args.seed,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_results(out_path, results, csv_dir)
    print(f"Wrote results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
