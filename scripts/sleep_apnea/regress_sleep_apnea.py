#!/usr/bin/env python3
import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is not available. Run with scripts/.venv/bin/python "
        "or ensure sklearn is installed."
    ) from exc


SLEEP_REASON_CODES = {"39898005", "73430006", "78275009"}
SLEEP_PROCEDURE_CODES = {
    "103750000",
    "82808001",
    "60554003",
    "10563004",
    "446573003",
    "698560000",
}
SLEEP_ENCOUNTER_CODES = {"185345009", "185347001", "185389009"}

HYPERTENSION_CODES = {"59621000"}
BMI_CODE = "39156-5"


@dataclass
class Dataset:
    label: str
    csv_dir: Path
    X: List[List[Optional[float]]]
    y: List[float]
    feature_names: List[str]
    patient_ids: List[str]


@dataclass
class Split:
    X_train: List[List[float]]
    y_train: List[float]
    X_val: List[List[float]]
    y_val: List[float]
    X_test: List[List[float]]
    y_test: List[float]
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]


def _find_csv_dir(path_str: str) -> Path:
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


def _make_header_map(fieldnames: Sequence[str]) -> Dict[str, str]:
    return {name.lower(): name for name in fieldnames or []}


def _get_value(row: Dict[str, str], header_map: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        key = header_map.get(candidate.lower())
        if key:
            return row.get(key)
    return None


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    value = value.strip()
    if len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_condition_flags(
    conditions_path: Path,
    codes: Sequence[str],
) -> Tuple[set, Optional[date]]:
    patients = set()
    max_date = None
    with conditions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = _make_header_map(reader.fieldnames)
        for row in reader:
            code = _get_value(row, header_map, ["code"])
            patient_id = _get_value(row, header_map, ["patient"])
            start_date = _parse_date(_get_value(row, header_map, ["start", "date"]))
            if start_date and (max_date is None or start_date > max_date):
                max_date = start_date
            if patient_id and code in codes:
                patients.add(patient_id)
    return patients, max_date


def load_latest_bmi(observations_path: Path) -> Dict[str, float]:
    bmi_by_patient: Dict[str, float] = {}
    date_by_patient: Dict[str, date] = {}

    if not observations_path.exists():
        return bmi_by_patient

    with observations_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = _make_header_map(reader.fieldnames)
        for row in reader:
            code = _get_value(row, header_map, ["code"])
            if code != BMI_CODE:
                continue
            patient_id = _get_value(row, header_map, ["patient"])
            if not patient_id:
                continue
            value = _parse_float(_get_value(row, header_map, ["value"]))
            if value is None:
                continue
            obs_date = _parse_date(_get_value(row, header_map, ["date"]))
            if obs_date is None:
                continue
            if patient_id not in date_by_patient or obs_date > date_by_patient[patient_id]:
                date_by_patient[patient_id] = obs_date
                bmi_by_patient[patient_id] = value
    return bmi_by_patient


def load_sleep_spend(csv_dir: Path) -> Dict[str, float]:
    spend = defaultdict(float)

    procedures_path = csv_dir / "procedures.csv"
    if procedures_path.exists():
        with procedures_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header_map = _make_header_map(reader.fieldnames)
            for row in reader:
                code = _get_value(row, header_map, ["code"])
                reason_code = _get_value(row, header_map, ["reasoncode"])
                if code not in SLEEP_PROCEDURE_CODES and reason_code not in SLEEP_REASON_CODES:
                    continue
                patient_id = _get_value(row, header_map, ["patient"])
                cost = _parse_float(_get_value(row, header_map, ["base_cost"]))
                if patient_id and cost is not None:
                    spend[patient_id] += cost

    encounters_path = csv_dir / "encounters.csv"
    if encounters_path.exists():
        with encounters_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header_map = _make_header_map(reader.fieldnames)
            for row in reader:
                code = _get_value(row, header_map, ["code"])
                reason_code = _get_value(row, header_map, ["reasoncode"])
                if reason_code not in SLEEP_REASON_CODES and code not in SLEEP_ENCOUNTER_CODES:
                    continue
                patient_id = _get_value(row, header_map, ["patient"])
                cost = _parse_float(_get_value(row, header_map, ["total_claim_cost"]))
                if cost is None:
                    cost = _parse_float(_get_value(row, header_map, ["base_encounter_cost"]))
                if patient_id and cost is not None:
                    spend[patient_id] += cost

    medications_path = csv_dir / "medications.csv"
    if medications_path.exists():
        with medications_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header_map = _make_header_map(reader.fieldnames)
            for row in reader:
                reason_code = _get_value(row, header_map, ["reasoncode"])
                if reason_code not in SLEEP_REASON_CODES:
                    continue
                patient_id = _get_value(row, header_map, ["patient"])
                cost = _parse_float(_get_value(row, header_map, ["totalcost"]))
                if patient_id and cost is not None:
                    spend[patient_id] += cost

    return spend


def build_dataset(label: str, base_dir: str, include_bmi: bool) -> Dataset:
    csv_dir = _find_csv_dir(base_dir)
    patients_path = csv_dir / "patients.csv"
    conditions_path = csv_dir / "conditions.csv"
    observations_path = csv_dir / "observations.csv"

    hypertension_patients, ref_date = load_condition_flags(conditions_path, HYPERTENSION_CODES)
    if ref_date is None:
        ref_date = date.today()

    bmi_by_patient = load_latest_bmi(observations_path) if include_bmi else {}
    sleep_spend = load_sleep_spend(csv_dir)

    feature_names = [
        "age_years",
        "male",
        "race_black",
        "race_white",
        "ethnicity_hispanic",
        "income",
        "healthcare_expenses",
        "healthcare_coverage",
        "hypertension",
    ]
    if include_bmi:
        feature_names.insert(8, "bmi")

    X: List[List[Optional[float]]] = []
    y: List[float] = []
    patient_ids: List[str] = []

    with patients_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = _make_header_map(reader.fieldnames)
        for row in reader:
            patient_id = _get_value(row, header_map, ["id"])
            if not patient_id:
                continue

            birthdate = _parse_date(_get_value(row, header_map, ["birthdate"]))
            if birthdate is None:
                age_years = None
            else:
                age_years = (ref_date - birthdate).days / 365.25
                if age_years < 0:
                    age_years = None

            gender = (_get_value(row, header_map, ["gender"]) or "").strip().lower()
            male = 1.0 if gender == "m" else 0.0

            race = (_get_value(row, header_map, ["race"]) or "").strip().lower()
            race_black = 1.0 if race == "black" else 0.0
            race_white = 1.0 if race == "white" else 0.0

            ethnicity = (_get_value(row, header_map, ["ethnicity"]) or "").strip().lower()
            ethnicity_hispanic = 1.0 if ethnicity == "hispanic" else 0.0

            income = _parse_float(_get_value(row, header_map, ["income"]))
            expenses = _parse_float(_get_value(row, header_map, ["healthcare_expenses"]))
            coverage = _parse_float(_get_value(row, header_map, ["healthcare_coverage"]))
            bmi = bmi_by_patient.get(patient_id) if include_bmi else None

            hypertension = 1.0 if patient_id in hypertension_patients else 0.0

            features = [
                age_years,
                male,
                race_black,
                race_white,
                ethnicity_hispanic,
                income,
                expenses,
                coverage,
            ]
            if include_bmi:
                features.append(bmi)
            features.append(hypertension)

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


def impute_missing(rows: List[List[Optional[float]]], reference: Optional[List[List[Optional[float]]]] = None) -> List[List[float]]:
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


def evaluate_predictions(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    abs_err = [abs(p - t) for p, t in zip(y_pred, y_true)]
    sq_err = [(p - t) ** 2 for p, t in zip(y_pred, y_true)]
    mae = sum(abs_err) / n
    rmse = math.sqrt(sum(sq_err) / n)
    mean_true = sum(y_true) / n
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    ss_res = sum(sq_err)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_with_validation(
    split: Split,
    param_grid: List[Dict[str, float]],
    seed: int,
) -> Tuple[GradientBoostingRegressor, Dict[str, float], Dict[str, float]]:
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

    X_train_val = split.X_train + split.X_val
    y_train_val = split.y_train + split.y_val
    final_model = GradientBoostingRegressor(random_state=seed, **best_params)
    final_model.fit(X_train_val, y_train_val)

    return final_model, best_params, best_metrics


def summarize_dataset(dataset: Dataset) -> Dict[str, float]:
    n = len(dataset.y)
    total_spend = sum(dataset.y)
    mean_spend = total_spend / n if n else 0.0
    nonzero = sum(1 for value in dataset.y if value > 0)
    return {
        "n": n,
        "total_spend": total_spend,
        "mean_spend": mean_spend,
        "nonzero_rate": nonzero / n if n else 0.0,
    }


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def generate_report(
    output_path: Path,
    baseline: Dataset,
    biased: Dataset,
    split_baseline: Split,
    split_biased: Split,
    baseline_model: GradientBoostingRegressor,
    biased_model: GradientBoostingRegressor,
    baseline_params: Dict[str, float],
    biased_params: Dict[str, float],
    baseline_val_metrics: Dict[str, float],
    biased_val_metrics: Dict[str, float],
    include_bmi: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_summary = summarize_dataset(baseline)
    biased_summary = summarize_dataset(biased)

    baseline_test_preds = baseline_model.predict(split_baseline.X_test)
    biased_test_preds = biased_model.predict(split_biased.X_test)

    baseline_test_metrics = evaluate_predictions(split_baseline.y_test, baseline_test_preds)
    biased_test_metrics = evaluate_predictions(split_biased.y_test, biased_test_preds)

    biased_on_baseline_preds = biased_model.predict(split_baseline.X_test)
    baseline_on_biased_preds = baseline_model.predict(split_biased.X_test)

    biased_on_baseline_metrics = evaluate_predictions(split_baseline.y_test, biased_on_baseline_preds)
    baseline_on_biased_metrics = evaluate_predictions(split_biased.y_test, baseline_on_biased_preds)

    def _bias_summary(y_true: List[float], y_pred: Sequence[float]) -> Dict[str, float]:
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"mean_true": 0.0, "mean_pred": 0.0, "diff": 0.0, "rel": float("nan")}
        mean_true = sum(y_true) / len(y_true)
        mean_pred = sum(float(value) for value in y_pred) / len(y_pred)
        diff = mean_pred - mean_true
        rel = diff / mean_true if mean_true != 0 else float("nan")
        return {"mean_true": mean_true, "mean_pred": mean_pred, "diff": diff, "rel": rel}

    baseline_bias = _bias_summary(split_baseline.y_test, baseline_test_preds)
    biased_bias = _bias_summary(split_baseline.y_test, biased_on_baseline_preds)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Sleep Apnea Demand Modeling Report\n\n")
        handle.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"- Baseline dataset: `{baseline.csv_dir}`\n")
        handle.write(f"- Biased dataset: `{biased.csv_dir}`\n")
        handle.write(f"- BMI feature included: {include_bmi}\n\n")

        handle.write("## Target Definition\n")
        handle.write(
            "Total sleep-related spend per patient, computed as the sum of:\n"
            "- `procedures.csv` BASE_COST where CODE in sleep-related procedure codes or REASONCODE "
            "in sleep-related condition codes.\n"
            "- `encounters.csv` TOTAL_CLAIM_COST (fallback BASE_ENCOUNTER_COST) where REASONCODE "
            "in sleep-related condition codes or encounter CODE in sleep-specific codes.\n"
            "- `medications.csv` TOTALCOST where REASONCODE in sleep-related condition codes.\n"
        )
        handle.write("\n")
        handle.write(f"Sleep-related condition codes: {sorted(SLEEP_REASON_CODES)}\n")
        handle.write(f"Sleep-related procedure codes: {sorted(SLEEP_PROCEDURE_CODES)}\n")
        handle.write(f"Sleep-related encounter codes: {sorted(SLEEP_ENCOUNTER_CODES)}\n\n")

        handle.write("## Features (No Urban/Rural)\n")
        handle.write(", ".join(baseline.feature_names) + "\n\n")

        handle.write("## Dataset Summary\n")
        for label, summary in [("Baseline", baseline_summary), ("Biased", biased_summary)]:
            handle.write(f"- {label}: n={summary['n']}, mean_spend={format_currency(summary['mean_spend'])}, ")
            handle.write(f"nonzero_rate={summary['nonzero_rate'] * 100:.2f}%\n")
        handle.write("\n")

        handle.write("## Split Configuration\n")
        handle.write(
            f"- Train/Val/Test sizes (baseline): "
            f"{len(split_baseline.y_train)}/{len(split_baseline.y_val)}/{len(split_baseline.y_test)}\n"
        )
        handle.write(
            f"- Train/Val/Test sizes (biased): "
            f"{len(split_biased.y_train)}/{len(split_biased.y_val)}/{len(split_biased.y_test)}\n\n"
        )

        handle.write("## Model Selection (Validation MAE)\n")
        handle.write(f"- Baseline best params: {baseline_params}\n")
        handle.write(
            f"  - Val MAE={baseline_val_metrics['mae']:.2f}, "
            f"RMSE={baseline_val_metrics['rmse']:.2f}, R2={baseline_val_metrics['r2']:.3f}\n"
        )
        handle.write(f"- Biased best params: {biased_params}\n")
        handle.write(
            f"  - Val MAE={biased_val_metrics['mae']:.2f}, "
            f"RMSE={biased_val_metrics['rmse']:.2f}, R2={biased_val_metrics['r2']:.3f}\n\n"
        )

        handle.write("## Test Results (In-Dataset)\n")
        handle.write(
            f"- Baseline model on baseline test: "
            f"MAE={baseline_test_metrics['mae']:.2f}, "
            f"RMSE={baseline_test_metrics['rmse']:.2f}, R2={baseline_test_metrics['r2']:.3f}\n"
        )
        handle.write(
            f"- Biased model on biased test: "
            f"MAE={biased_test_metrics['mae']:.2f}, "
            f"RMSE={biased_test_metrics['rmse']:.2f}, R2={biased_test_metrics['r2']:.3f}\n\n"
        )

        handle.write("## Cross-Dataset Evaluation\n")
        handle.write(
            f"- Biased model on baseline test: "
            f"MAE={biased_on_baseline_metrics['mae']:.2f}, "
            f"RMSE={biased_on_baseline_metrics['rmse']:.2f}, R2={biased_on_baseline_metrics['r2']:.3f}\n"
        )
        handle.write(
            f"- Baseline model on biased test: "
            f"MAE={baseline_on_biased_metrics['mae']:.2f}, "
            f"RMSE={baseline_on_biased_metrics['rmse']:.2f}, R2={baseline_on_biased_metrics['r2']:.3f}\n\n"
        )

        handle.write("## Demand Bias (Baseline Test Set)\n")
        handle.write(
            f"- Baseline model mean prediction: {format_currency(baseline_bias['mean_pred'])} "
            f"(actual {format_currency(baseline_bias['mean_true'])}, "
            f"diff {format_currency(baseline_bias['diff'])})\n"
        )
        handle.write(
            f"- Biased model mean prediction: {format_currency(biased_bias['mean_pred'])} "
            f"(actual {format_currency(biased_bias['mean_true'])}, "
            f"diff {format_currency(biased_bias['diff'])}, "
            f"rel {biased_bias['rel'] * 100:.2f}% )\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train GBDT models to predict sleep-related spend and compare baseline vs bias."
    )
    parser.add_argument("--baseline", default="output_baseline", help="Baseline output directory.")
    parser.add_argument("--biased", default="output_rural_bias", help="Biased output directory.")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Training fraction.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--include-bmi",
        action="store_true",
        help="Include BMI feature (scans observations.csv, which can be large).",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "sleep_apnea_demand_report.md"),
        help="Markdown output path.",
    )
    args = parser.parse_args()

    baseline = build_dataset("baseline", args.baseline, args.include_bmi)
    biased = build_dataset("biased", args.biased, args.include_bmi)

    split_baseline = split_dataset(baseline, args.train_frac, args.val_frac, args.seed)
    split_biased = split_dataset(biased, args.train_frac, args.val_frac, args.seed + 1)

    param_grid = [
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

    generate_report(
        Path(args.out),
        baseline,
        biased,
        split_baseline,
        split_biased,
        baseline_model,
        biased_model,
        baseline_params,
        biased_params,
        baseline_val_metrics,
        biased_val_metrics,
        args.include_bmi,
    )

    print(f"Wrote report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
