#!/usr/bin/env python3
"""Analyze diabetes diagnosis probability with upstream features."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent / "sleep_apnea"
sys.path.append(str(UTILS_DIR))

from utils import (  # type: ignore
    CONDITIONS_HEADERS,
    OBSERVATIONS_HEADERS,
    PATIENTS_HEADERS,
    find_csv_dir,
    find_dataset_end_date,
    get_value,
    impute_missing,
    iter_csv_rows,
    load_latest_smoking_status,
    parse_date,
    parse_float,
)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


SMOKING_STATUS_CODE = "72166-2"
SMOKER_VALUES = {"Smokes tobacco daily (finding)"}
NON_SMOKER_VALUES = {"Ex-smoker (finding)", "Never smoked tobacco (finding)"}

DIABETES_CODES = {"44054006"}

OBSERVATION_CODES = {
    "bmi": {"39156-5"},
    "systolic_bp": {"8480-6"},
    "diastolic_bp": {"8462-4"},
}

CONDITION_CODES = {
    "obesity": {"162864005", "408512008"},
    "hypertension": {"59621000"},
    "hyperlipidemia": {"55822004"},
    "hyperglycemia": {"80394007"},
    "hypertriglyceridemia": {"302870006"},
    "prediabetes": {"714628002"},
}


MODULE_SUMMARY = [
    "Diabetes onset modeled in metabolic_syndrome_disease with age-based prevalence and progression.",
    "Metabolic syndrome care module screens via A1c and metabolic criteria, then diagnoses prediabetes/diabetes.",
    "Care includes a diabetes self-management plan, medications, glucose monitoring, and DME/supplies.",
    "Complications modeled: kidney disease, neuropathy, anemia, and amputations (via submodules).",
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    features: List[str]


@dataclass(frozen=True)
class ModelResult:
    spec_name: str
    model_name: str
    auc: float
    average_precision: float
    brier: float


@dataclass(frozen=True)
class ModelScore:
    model_name: str
    auc: float
    average_precision: float
    brier: float


@dataclass(frozen=True)
class SpecSummary:
    spec: FeatureSpec
    missing_rates: Dict[str, float]
    dropped_features: List[str]
    n_train: int
    n_test: int


def load_condition_groups(
    conditions_path: Path,
    code_groups: Dict[str, Sequence[str]],
) -> Dict[str, set]:
    patients_by_group = {name: set() for name in code_groups}
    code_to_groups: Dict[str, List[str]] = {}
    for group_name, codes in code_groups.items():
        for code in codes:
            code_to_groups.setdefault(code, []).append(group_name)

    for row, header_map in iter_csv_rows(conditions_path, CONDITIONS_HEADERS):
        code = get_value(row, header_map, ["code"])
        patient_id = get_value(row, header_map, ["patient"])
        if not code or not patient_id:
            continue
        group_names = code_to_groups.get(code)
        if not group_names:
            continue
        for group_name in group_names:
            patients_by_group[group_name].add(patient_id)
    return patients_by_group


def load_latest_numeric_observations(
    observations_path: Path,
    code_groups: Dict[str, Sequence[str]],
) -> Dict[str, Dict[str, float]]:
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


def build_patient_features(
    csv_dir: Path,
    condition_groups: Dict[str, set],
    observation_values: Dict[str, Dict[str, float]],
    smoker_by_patient: Dict[str, float],
) -> Dict[str, Dict[str, Optional[float]]]:
    patients_path = csv_dir / "patients.csv"
    ref_date = find_dataset_end_date(csv_dir) or date.today()

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
        }

        for feature_name, values in observation_values.items():
            features[feature_name] = values.get(patient_id)

        for group_name, patient_set in condition_groups.items():
            features[group_name] = 1.0 if patient_id in patient_set else 0.0

        features_by_patient[patient_id] = features
    return features_by_patient


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


def build_dataset(
    patient_features: Dict[str, Dict[str, Optional[float]]],
    patient_ids: Sequence[str],
    labels: Dict[str, int],
    feature_names: Sequence[str],
) -> Tuple[List[List[Optional[float]]], List[int], List[str]]:
    X: List[List[Optional[float]]] = []
    y: List[int] = []
    ids: List[str] = []
    for patient_id in patient_ids:
        features = patient_features.get(patient_id)
        if not features:
            continue
        X.append([features.get(name) for name in feature_names])
        y.append(int(labels.get(patient_id, 0)))
        ids.append(patient_id)
    return X, y, ids


def evaluate_predictions(y_true: Sequence[int], y_prob: Sequence[float]) -> Tuple[float, float, float]:
    if len(set(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return auc, ap, brier


def train_models(
    X: List[List[Optional[float]]],
    y: List[int],
    seed: int,
) -> Tuple[List[ModelScore], Tuple[int, int]]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y))
    X_train_raw = [X[i] for i in train_idx]
    X_test_raw = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    X_train = impute_missing(X_train_raw)
    X_test = impute_missing(X_test_raw, X_train_raw)

    results: List[ModelScore] = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    log_reg.fit(X_train_scaled, y_train)
    prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]
    auc, ap, brier = evaluate_predictions(y_test, prob_log)
    results.append(ModelScore("logistic", auc, ap, brier))

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    prob_rf = rf.predict_proba(X_test)[:, 1]
    auc, ap, brier = evaluate_predictions(y_test, prob_rf)
    results.append(ModelScore("rf", auc, ap, brier))

    gb = GradientBoostingClassifier(random_state=seed)
    gb.fit(X_train, y_train)
    prob_gb = gb.predict_proba(X_test)[:, 1]
    auc, ap, brier = evaluate_predictions(y_test, prob_gb)
    results.append(ModelScore("gbdt", auc, ap, brier))

    return results, (len(train_idx), len(test_idx))


def load_module_summary(paths: Sequence[Path]) -> Dict[str, List[str]]:
    summary = {
        "condition_codes": [],
        "procedure_codes": [],
        "observation_codes": [],
        "medication_codes": [],
        "device_codes": [],
        "attributes": [],
        "submodules": [],
    }
    attributes = set()
    condition_codes = set()
    procedure_codes = set()
    observation_codes = set()
    medication_codes = set()
    device_codes = set()
    submodules = set()

    for module_path in paths:
        if not module_path.exists():
            continue
        data = json.loads(module_path.read_text(encoding="utf-8"))
        for state in data.get("states", {}).values():
            state_type = state.get("type")
            if state_type in {"ConditionOnset", "ConditionEnd"}:
                for code in state.get("codes", []):
                    if code.get("code"):
                        condition_codes.add(code["code"])
            if state_type in {"Procedure", "Encounter"}:
                for code in state.get("codes", []):
                    if code.get("code"):
                        procedure_codes.add(code["code"])
            if state_type in {"Observation", "MultiObservation"}:
                for code in state.get("codes", []):
                    if code.get("code"):
                        observation_codes.add(code["code"])
            if state_type == "DiagnosticReport":
                for report_code in state.get("codes", []):
                    if report_code.get("code"):
                        observation_codes.add(report_code["code"])
                for obs in state.get("observations", []):
                    for code in obs.get("codes", []):
                        if code.get("code"):
                            observation_codes.add(code["code"])
            if state_type == "MedicationOrder":
                for code in state.get("codes", []):
                    if code.get("code"):
                        medication_codes.add(code["code"])
            if state_type in {"Device", "DeviceEnd"}:
                device = state.get("code") or {}
                if device.get("code"):
                    device_codes.add(device["code"])
                for code in state.get("codes", []):
                    if code.get("code"):
                        device_codes.add(code["code"])
            if state_type == "CallSubmodule":
                submodule = state.get("submodule")
                if submodule:
                    submodules.add(submodule)
            for key in ["attribute", "assign_to_attribute", "referenced_by_attribute"]:
                value = state.get(key)
                if value:
                    attributes.add(value)
            for transition in state.get("distributed_transition", []):
                distribution = transition.get("distribution")
                if isinstance(distribution, dict):
                    attr = distribution.get("attribute")
                    if attr:
                        attributes.add(attr)
            for transition in state.get("conditional_transition", []):
                condition = transition.get("condition", {})
                if condition.get("condition_type") == "Attribute":
                    attributes.add(condition.get("attribute"))

    summary["condition_codes"] = sorted(str(code) for code in condition_codes if code)
    summary["procedure_codes"] = sorted(str(code) for code in procedure_codes if code)
    summary["observation_codes"] = sorted(str(code) for code in observation_codes if code)
    summary["medication_codes"] = sorted(str(code) for code in medication_codes if code)
    summary["device_codes"] = sorted(str(code) for code in device_codes if code)
    summary["attributes"] = sorted(attr for attr in attributes if attr)
    summary["submodules"] = sorted(submodules)
    return summary


def write_report(
    path: Path,
    csv_dir: Path,
    prevalence: float,
    spec_summaries: List[SpecSummary],
    results: List[ModelResult],
    module_summary: Dict[str, List[str]],
) -> None:
    lines: List[str] = []
    lines.append("# Diabetes Diagnosis Probability Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Dataset: `{csv_dir}`")
    lines.append("")
    lines.append("## Module summary (metabolic syndrome modules)")
    lines.extend([f"- {item}" for item in MODULE_SUMMARY])
    if module_summary["attributes"]:
        lines.append(f"- Module attributes: {', '.join(module_summary['attributes'])}")
    if module_summary["submodules"]:
        lines.append(f"- Submodules: {', '.join(module_summary['submodules'])}")
    if module_summary["condition_codes"]:
        lines.append(f"- Condition codes: {', '.join(module_summary['condition_codes'])}")
    if module_summary["procedure_codes"]:
        lines.append(f"- Procedure/encounter codes: {', '.join(module_summary['procedure_codes'])}")
    if module_summary["observation_codes"]:
        lines.append(f"- Observation codes: {', '.join(module_summary['observation_codes'])}")
    if module_summary["medication_codes"]:
        lines.append(f"- Medication codes: {', '.join(module_summary['medication_codes'])}")
    if module_summary["device_codes"]:
        lines.append(f"- Device codes: {', '.join(module_summary['device_codes'])}")
    lines.append("")
    lines.append("## Diagnosis prevalence")
    lines.append(f"- Diabetes prevalence: {format_percent(prevalence)}")
    lines.append("")
    lines.append("## Feature specifications")
    for summary in spec_summaries:
        lines.append(f"- {summary.spec.name}: {', '.join(summary.spec.features)}")
        if summary.dropped_features:
            lines.append(f"  Dropped empty: {', '.join(summary.dropped_features)}")
        gaps = [
            f"{name} ({format_percent(rate)} missing)"
            for name, rate in summary.missing_rates.items()
            if rate >= 0.3
        ]
        if gaps:
            lines.append(f"  Missing >30%: {', '.join(gaps)}")
    lines.append("")
    lines.append("## Model results")
    lines.append("| Spec | Model | AUC | AP | Brier | Train/Test |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for summary in spec_summaries:
        spec_results = [r for r in results if r.spec_name == summary.spec.name]
        for result in spec_results:
            lines.append(
                "| {spec} | {model} | {auc:.3f} | {ap:.3f} | {brier:.3f} | {train}/{test} |".format(
                    spec=summary.spec.name,
                    model=result.model_name,
                    auc=result.auc,
                    ap=result.average_precision,
                    brier=result.brier,
                    train=summary.n_train,
                    test=summary.n_test,
                )
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diabetes diagnosis probability analysis.")
    parser.add_argument(
        "--data",
        default="output_baseline",
        help="Path to Synthea output directory (root or csv subdir).",
    )
    parser.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "DIABETES_ANALYSIS.md"),
        help="Output markdown path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    csv_dir = find_csv_dir(args.data)
    conditions_path = csv_dir / "conditions.csv"
    observations_path = csv_dir / "observations.csv"

    condition_groups = load_condition_groups(conditions_path, CONDITION_CODES)
    diabetes_patients = load_condition_groups(conditions_path, {"diabetes": DIABETES_CODES})[
        "diabetes"
    ]

    smoker_by_patient = load_latest_smoking_status(
        observations_path,
        SMOKING_STATUS_CODE,
        SMOKER_VALUES,
        NON_SMOKER_VALUES,
    )
    observation_values = load_latest_numeric_observations(observations_path, OBSERVATION_CODES)

    patient_features = build_patient_features(
        csv_dir, condition_groups, observation_values, smoker_by_patient
    )
    patient_ids = list(patient_features.keys())

    labels = {
        patient_id: 1 if patient_id in diabetes_patients else 0 for patient_id in patient_ids
    }
    prevalence = sum(labels.values()) / len(labels) if labels else 0.0

    specs = [
        FeatureSpec("Demographics", ["age_years", "male", "income"]),
        FeatureSpec("Risk basic", ["age_years", "male", "income", "bmi", "smoker"]),
        FeatureSpec(
            "Risk + comorbidities",
            ["age_years", "male", "income", "bmi", "smoker", "obesity", "hypertension", "hyperlipidemia"],
        ),
        FeatureSpec(
            "Risk + metabolic",
            [
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
            ],
        ),
        FeatureSpec(
            "Risk + vitals",
            [
                "age_years",
                "male",
                "income",
                "bmi",
                "smoker",
                "obesity",
                "hypertension",
                "hyperlipidemia",
                "systolic_bp",
                "diastolic_bp",
            ],
        ),
    ]

    spec_summaries: List[SpecSummary] = []
    all_results: List[ModelResult] = []

    for spec in specs:
        X_raw, y, _ = build_dataset(patient_features, patient_ids, labels, spec.features)
        missing_rates = compute_missing_rates(X_raw, spec.features)
        filtered_X, filtered_features, dropped = filter_empty_features(X_raw, list(spec.features))
        model_results, split_sizes = train_models(filtered_X, y, args.seed)
        spec_summaries.append(
            SpecSummary(
                spec=FeatureSpec(spec.name, filtered_features),
                missing_rates=missing_rates,
                dropped_features=dropped,
                n_train=split_sizes[0],
                n_test=split_sizes[1],
            )
        )
        for result in model_results:
            all_results.append(
                ModelResult(
                    spec_name=spec.name,
                    model_name=result.model_name,
                    auc=result.auc,
                    average_precision=result.average_precision,
                    brier=result.brier,
                )
            )

    module_summary = load_module_summary(
        [
            Path(__file__).resolve().parents[2]
            / "src"
            / "main"
            / "resources"
            / "modules"
            / "metabolic_syndrome_disease.json",
            Path(__file__).resolve().parents[2]
            / "src"
            / "main"
            / "resources"
            / "modules"
            / "metabolic_syndrome_care.json",
        ]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(out_path, csv_dir, prevalence, spec_summaries, all_results, module_summary)
    print(f"Wrote report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
