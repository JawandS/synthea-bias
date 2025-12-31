#!/usr/bin/env python3
"""Train and compare sleep apnea spend models across baseline and biased datasets."""

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

from analyze_sleep_apnea import analyze_models, evaluate_predictions
from sleep_apnea_report import ReportInputs, write_report

from utils import (
    find_csv_dir,
    get_value,
    impute_missing,
    load_condition_flags,
    load_latest_bmi,
    load_sleep_spend,
    make_header_map,
    parse_date,
    parse_float,
)

from sklearn.ensemble import GradientBoostingRegressor

SLEEP_APNEA_CODES = {"73430006", "78275009"}
SLEEP_DISORDER_CODE = "39898005"
SLEEP_REASON_CODES = {SLEEP_DISORDER_CODE} | SLEEP_APNEA_CODES
SLEEP_PROCEDURE_CODES = {
    "103750000",
    "82808001",
    "60554003",
    "10563004",
    "446573003",
    "698560000",
}
SLEEP_ENCOUNTER_CODES = {"185345009", "185347001", "185389009"}
SLEEP_DEVICE_CODES = {
    "272265001",
    "701077002",
    "701100002",
    "702172008",
    "706180003",
    "720253003",
}
SLEEP_SUPPLY_CODES = {"463659001", "467645007", "704718009", "706226000", "972002"}
SLEEP_EQUIPMENT_CODES = SLEEP_DEVICE_CODES | SLEEP_SUPPLY_CODES

HYPERTENSION_CODES = {"59621000"}
BMI_CODE = "39156-5"
BASELINE_GENERATION_CMD = (
    "./run_synthea -p 25000 --exporter.csv.export=true --exporter.baseDirectory=./output_baseline"
)
BIASED_GENERATION_CMD = (
    "./run_synthea -p 25000 --exporter.csv.export=true --exporter.baseDirectory=./output_rural_bias "
    "--module_override=/home/js/contracts/synthea-bias/config/overrides_rural_sleep_apnea.properties"
)


@dataclass
class Dataset:
    """Feature matrix, labels, and metadata for one dataset."""
    label: str
    csv_dir: Path
    X: List[List[Optional[float]]]
    y: List[float]
    feature_names: List[str]
    patient_ids: List[str]


@dataclass
class Split:
    """Train/validation/test split with patient identifiers."""
    X_train: List[List[float]]
    y_train: List[float]
    X_val: List[List[float]]
    y_val: List[float]
    X_test: List[List[float]]
    y_test: List[float]
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]


class GBDTParams(TypedDict):
    """Parameter grid entries for GradientBoostingRegressor."""

    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_leaf: int


def build_dataset(label: str, base_dir: str) -> Dataset:
    """Build the regression dataset from Synthea CSV exports."""
    csv_dir = find_csv_dir(base_dir)
    patients_path = csv_dir / "patients.csv"
    conditions_path = csv_dir / "conditions.csv"
    observations_path = csv_dir / "observations.csv"

    hypertension_patients, ref_date = load_condition_flags(conditions_path, HYPERTENSION_CODES)
    if ref_date is None:
        ref_date = date.today()
    # Anchor age calculations to a dataset-relevant reference date.

    # Use the most recent BMI per patient for a consistent point-in-time feature.
    bmi_by_patient = load_latest_bmi(observations_path, BMI_CODE)
    sleep_spend = load_sleep_spend(
        csv_dir,
        SLEEP_PROCEDURE_CODES,
        SLEEP_REASON_CODES,
        SLEEP_ENCOUNTER_CODES,
        SLEEP_EQUIPMENT_CODES,
    )

    feature_names = [
        "age_years",
        "male",
        "income",
        "healthcare_expenses",
        "healthcare_coverage",
        "bmi",
        "hypertension",
    ]

    X: List[List[Optional[float]]] = []
    y: List[float] = []
    patient_ids: List[str] = []

    with patients_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = make_header_map(reader.fieldnames)
        for row in reader:
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
            expenses = parse_float(get_value(row, header_map, ["healthcare_expenses"]))
            coverage = parse_float(get_value(row, header_map, ["healthcare_coverage"]))
            bmi = bmi_by_patient.get(patient_id)

            hypertension = 1.0 if patient_id in hypertension_patients else 0.0

            features = [
                age_years,
                male,
                income,
                expenses,
                coverage,
                bmi,
                hypertension,
            ]

            X.append(features)
            y.append(float(sleep_spend.get(patient_id, 0.0)))
            patient_ids.append(patient_id)

    return Dataset(
        label=label,
        csv_dir=csv_dir,
        X=X,
        y=y,
        feature_names=feature_names,
        patient_ids=patient_ids,
    )


def split_dataset(
    dataset: Dataset,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Split:
    """Shuffle and split the dataset, then impute missing values."""
    rng = random.Random(seed)
    indices = list(range(len(dataset.y)))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = max(1, int(n_total * train_frac))
    n_val = max(1, int(n_total * val_frac))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]

    def _gather(idx_list: List[int]) -> Tuple[List[List[Optional[float]]], List[float], List[str]]:
        """Collect features, labels, and ids for a list of indices."""
        X_split = [dataset.X[i] for i in idx_list]
        y_split = [dataset.y[i] for i in idx_list]
        ids_split = [dataset.patient_ids[i] for i in idx_list]
        return X_split, y_split, ids_split

    X_train, y_train, train_ids = _gather(train_idx)
    X_val, y_val, val_ids = _gather(val_idx)
    X_test, y_test, test_ids = _gather(test_idx)

    return Split(
        X_train=impute_missing(X_train),
        y_train=y_train,
        X_val=impute_missing(X_val, X_train),
        y_val=y_val,
        X_test=impute_missing(X_test, X_train),
        y_test=y_test,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )


def train_with_validation(
    split: Split,
    param_grid: Sequence[GBDTParams],
    seed: int,
) -> Tuple[GradientBoostingRegressor, GBDTParams, Dict[str, float]]:
    """Select hyperparameters by validation MAE and refit on train+val."""
    best_model = None
    best_params = None
    best_metrics = None

    for params in param_grid:
        model = GradientBoostingRegressor(random_state=seed, **params)
        model.fit(split.X_train, split.y_train)
        preds = model.predict(split.X_val)
        metrics = evaluate_predictions(split.y_val, preds)
        if best_metrics is None or metrics["mae"] < best_metrics["mae"]:
            best_model = model
            best_params = params
            best_metrics = metrics

    if best_model is None or best_params is None or best_metrics is None:
        raise RuntimeError("Failed to train model on validation grid.")

    # Refit on combined train+val to maximize training data with the chosen params.
    X_train_val = split.X_train + split.X_val
    y_train_val = split.y_train + split.y_val
    final_model = GradientBoostingRegressor(random_state=seed, **best_params)
    final_model.fit(X_train_val, y_train_val)

    return final_model, best_params, best_metrics


def main() -> int:
    """Entry point for training and reporting."""
    parser = argparse.ArgumentParser(
        description="Train GBDT models to predict sleep-related spend and compare baseline vs bias."
    )
    parser.add_argument("--baseline", default="output_baseline", help="Baseline output directory.")
    parser.add_argument("--biased", default="output_rural_bias", help="Biased output directory.")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Training fraction.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "sleep_apnea_demand_report.md"),
        help="Markdown output path.",
    )
    args = parser.parse_args()

    baseline = build_dataset("baseline", args.baseline)
    biased = build_dataset("biased", args.biased)

    split_baseline = split_dataset(baseline, args.train_frac, args.val_frac, args.seed)
    split_biased = split_dataset(biased, args.train_frac, args.val_frac, args.seed + 1)

    param_grid: List[GBDTParams] = [
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "min_samples_leaf": 20},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20},
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 2, "min_samples_leaf": 10},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 10},
    ]

    baseline_model, baseline_params, baseline_val_metrics = train_with_validation(
        split_baseline, param_grid, args.seed
    )
    biased_model, biased_params, biased_val_metrics = train_with_validation(
        split_biased, param_grid, args.seed + 1
    )

    analysis = analyze_models(
        baseline=baseline,
        biased=biased,
        split_baseline=split_baseline,
        split_biased=split_biased,
        baseline_model=baseline_model,
        biased_model=biased_model,
        sleep_disorder_code=SLEEP_DISORDER_CODE,
        sleep_apnea_codes=SLEEP_APNEA_CODES,
    )

    report_inputs = ReportInputs(
        baseline_csv_dir=baseline.csv_dir,
        biased_csv_dir=biased.csv_dir,
        feature_names=baseline.feature_names,
        baseline_params=baseline_params,
        biased_params=biased_params,
        baseline_val_metrics=baseline_val_metrics,
        biased_val_metrics=biased_val_metrics,
        analysis=analysis,
        baseline_generation_cmd=BASELINE_GENERATION_CMD,
        biased_generation_cmd=BIASED_GENERATION_CMD,
        sleep_reason_codes=SLEEP_REASON_CODES,
        sleep_procedure_codes=SLEEP_PROCEDURE_CODES,
        sleep_encounter_codes=SLEEP_ENCOUNTER_CODES,
        sleep_device_codes=SLEEP_DEVICE_CODES,
        sleep_supply_codes=SLEEP_SUPPLY_CODES,
    )

    write_report(Path(args.out), report_inputs)

    print(f"Wrote report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
