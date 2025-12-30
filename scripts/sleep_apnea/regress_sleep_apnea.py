#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SLEEP_APNEA_CODES = {"73430006", "78275009"}
HYPERTENSION_CODES = {"59621000"}
BMI_CODE = "39156-5"
EPS = 1e-12


@dataclass
class Dataset:
    label: str
    csv_dir: Path
    X: List[List[Optional[float]]]
    y: List[int]
    feature_names: List[str]
    numeric_indices: List[int]
    prevalence: float
    ref_date: date


def _find_csv_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        if (path / "patients.csv").exists():
            return path
        if (path / "csv" / "patients.csv").exists():
            return path / "csv"
    raise FileNotFoundError(
        f"Unable to locate patients.csv under {path_str}. "
        "Pass the output directory (e.g., output_baseline) or the csv subdir."
    )


def _make_header_map(fieldnames: Sequence[str]) -> Dict[str, str]:
    return {name.lower(): name for name in fieldnames}


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
    sleep_codes: Sequence[str],
    hypertension_codes: Sequence[str],
) -> Tuple[set, set, Optional[date]]:
    sleep_patients = set()
    hypertension_patients = set()
    max_date = None

    with conditions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = _make_header_map(reader.fieldnames or [])
        for row in reader:
            code = _get_value(row, header_map, ["code"])
            patient_id = _get_value(row, header_map, ["patient"])
            start_date = _parse_date(_get_value(row, header_map, ["start", "date"]))
            if start_date and (max_date is None or start_date > max_date):
                max_date = start_date
            if patient_id and code in sleep_codes:
                sleep_patients.add(patient_id)
            if patient_id and code in hypertension_codes:
                hypertension_patients.add(patient_id)

    return sleep_patients, hypertension_patients, max_date


def load_latest_bmi(observations_path: Path) -> Dict[str, float]:
    bmi_by_patient: Dict[str, float] = {}
    date_by_patient: Dict[str, date] = {}

    if not observations_path.exists():
        return bmi_by_patient

    with observations_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = _make_header_map(reader.fieldnames or [])
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


def compute_numeric_stats(
    rows: List[List[Optional[float]]],
    numeric_indices: Sequence[int],
) -> Dict[int, Tuple[float, float]]:
    stats: Dict[int, Tuple[float, float]] = {}
    for idx in numeric_indices:
        values = [row[idx] for row in rows if row[idx] is not None]
        if not values:
            stats[idx] = (0.0, 1.0)
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std == 0.0:
            std = 1.0
        stats[idx] = (mean, std)
    return stats


def apply_standardization(
    rows: List[List[Optional[float]]],
    numeric_indices: Sequence[int],
    stats: Dict[int, Tuple[float, float]],
) -> None:
    for row in rows:
        for idx in numeric_indices:
            mean, std = stats[idx]
            value = row[idx]
            if value is None:
                value = mean
            row[idx] = (value - mean) / std


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def logistic_fit(
    X: List[List[float]],
    y: List[int],
    max_iter: int = 200,
    lr: float = 0.1,
    tol: float = 1e-6,
    l2: float = 0.0,
) -> Tuple[List[float], float, int, bool]:
    n = len(y)
    if n == 0:
        raise ValueError("No rows available for model fitting.")
    p = len(X[0])
    beta = [0.0] * p
    prev_avg_ll = None

    for iteration in range(1, max_iter + 1):
        grad = [0.0] * p
        ll = 0.0

        for x_row, y_val in zip(X, y):
            z = 0.0
            for j in range(p):
                z += beta[j] * x_row[j]
            p_hat = sigmoid(z)
            ll += y_val * math.log(p_hat + EPS) + (1 - y_val) * math.log(1 - p_hat + EPS)
            diff = y_val - p_hat
            for j in range(p):
                grad[j] += diff * x_row[j]

        if l2 > 0.0:
            for j in range(p):
                grad[j] -= l2 * beta[j]

        step = lr / n
        for j in range(p):
            beta[j] += step * grad[j]

        avg_ll = ll / n
        if prev_avg_ll is not None and abs(avg_ll - prev_avg_ll) < tol:
            return beta, avg_ll, iteration, True
        prev_avg_ll = avg_ll

    return beta, prev_avg_ll if prev_avg_ll is not None else float("nan"), max_iter, False


def predict_proba(X: List[List[float]], beta: Sequence[float]) -> List[float]:
    probs = []
    for x_row in X:
        z = 0.0
        for j in range(len(beta)):
            z += beta[j] * x_row[j]
        probs.append(sigmoid(z))
    return probs


def build_dataset(label: str, base_dir: str, include_bmi: bool) -> Dataset:
    csv_dir = _find_csv_dir(base_dir)
    patients_path = csv_dir / "patients.csv"
    conditions_path = csv_dir / "conditions.csv"
    observations_path = csv_dir / "observations.csv"

    sleep_patients, hypertension_patients, ref_date = load_condition_flags(
        conditions_path, SLEEP_APNEA_CODES, HYPERTENSION_CODES
    )
    if ref_date is None:
        ref_date = date.today()

    bmi_by_patient = load_latest_bmi(observations_path) if include_bmi else {}

    feature_names = [
        "intercept",
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
        feature_names.insert(-1, "bmi")
    numeric_features = {"age_years", "income", "healthcare_expenses", "healthcare_coverage"}
    if include_bmi:
        numeric_features.add("bmi")
    numeric_indices = [i for i, name in enumerate(feature_names) if name in numeric_features]

    X: List[List[Optional[float]]] = []
    y: List[int] = []

    with patients_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header_map = _make_header_map(reader.fieldnames or [])
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
            sleep_apnea = 1 if patient_id in sleep_patients else 0

            features = [
                1.0,
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
            y.append(sleep_apnea)

    prevalence = sum(y) / len(y) if y else 0.0

    return Dataset(
        label=label,
        csv_dir=csv_dir,
        X=X,
        y=y,
        feature_names=feature_names,
        numeric_indices=numeric_indices,
        prevalence=prevalence,
        ref_date=ref_date,
    )


def report_model(dataset: Dataset, beta: Sequence[float], avg_ll: float, iterations: int, converged: bool) -> None:
    probs = predict_proba(dataset.X, beta)
    mean_prob = sum(probs) / len(probs) if probs else 0.0

    print(f"\nDataset: {dataset.label}")
    print(f"- csv_dir: {dataset.csv_dir}")
    print(f"- patients: {len(dataset.y)}")
    print(f"- sleep apnea prevalence: {dataset.prevalence * 100:.2f}%")
    print(f"- avg predicted probability: {mean_prob * 100:.2f}%")
    print(f"- avg log-likelihood: {avg_ll:.6f}")
    print(f"- iterations: {iterations} (converged={converged})")
    print("Coefficients (odds ratio):")
    for name, coef in zip(dataset.feature_names, beta):
        odds_ratio = math.exp(coef)
        print(f"- {name}: {coef:.4f} (OR={odds_ratio:.3f})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit MLE logistic models for sleep apnea on baseline and biased datasets."
    )
    parser.add_argument(
        "--baseline",
        default="output_baseline",
        help="Baseline output directory (or csv subdir).",
    )
    parser.add_argument(
        "--biased",
        default="output_rural_bias",
        help="Biased output directory (or csv subdir).",
    )
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations for MLE.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for MLE.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for convergence.")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 penalty (0 for pure MLE).")
    parser.add_argument(
        "--include-bmi",
        action="store_true",
        help="Include BMI as a feature (scans observations.csv, which can be large).",
    )
    args = parser.parse_args()

    baseline = build_dataset("baseline", args.baseline, args.include_bmi)
    biased = build_dataset("biased", args.biased, args.include_bmi)

    stats = compute_numeric_stats(baseline.X, baseline.numeric_indices)
    apply_standardization(baseline.X, baseline.numeric_indices, stats)
    apply_standardization(biased.X, biased.numeric_indices, stats)

    baseline_beta, baseline_ll, baseline_iters, baseline_conv = logistic_fit(
        baseline.X, baseline.y, max_iter=args.max_iter, lr=args.lr, tol=args.tol, l2=args.l2
    )
    biased_beta, biased_ll, biased_iters, biased_conv = logistic_fit(
        biased.X, biased.y, max_iter=args.max_iter, lr=args.lr, tol=args.tol, l2=args.l2
    )

    print("Logistic regression for sleep apnea (MLE, numeric features standardized by baseline stats).")
    report_model(baseline, baseline_beta, baseline_ll, baseline_iters, baseline_conv)
    report_model(biased, biased_beta, biased_ll, biased_iters, biased_conv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
