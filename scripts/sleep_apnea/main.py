#!/usr/bin/env python3
"""Train and compare sleep apnea spend models across baseline and biased datasets.

This study builds spend prediction models and then quantifies geographic access
bias using a separate linear regression that includes the urban/rural flag.

Model Variants:
- base GBDT: Core features only (age, gender, income, BMI, smoking, alcohol use, hypertension, CHF)
- linear bias model: Core features + urban/rural indicator (used for bias measurement)

The study uses bootstrap resampling for confidence intervals and permutation
testing for cross-population hypothesis tests.
"""

import argparse
import csv
import random
from dataclasses import dataclass, field
from datetime import date
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()

from analyze_sleep_apnea import analyze_models, evaluate_predictions
from sleep_apnea_report import ReportInputs, write_report

from utils import (
    find_csv_dir,
    find_dataset_end_date,
    get_value,
    impute_missing,
    load_condition_patients,
    load_latest_bmi,
    load_latest_smoking_status,
    load_patient_urban_flags,
    load_sleep_spend,
    make_header_map,
    parse_date,
    parse_float,
)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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
CHF_CODES = {"88805009"}
ALCOHOL_USE_CODES = {"7200002"}
BMI_CODE = "39156-5"
SMOKING_STATUS_CODE = "72166-2"
SMOKER_VALUES = {"Smokes tobacco daily (finding)"}
NON_SMOKER_VALUES = {"Ex-smoker (finding)", "Never smoked tobacco (finding)"}
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
    urban_flags: Dict[str, Optional[float]] = field(default_factory=dict)


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
    # Urban flags for stratified analysis
    train_urban: List[Optional[float]] = field(default_factory=list)
    val_urban: List[Optional[float]] = field(default_factory=list)
    test_urban: List[Optional[float]] = field(default_factory=list)


class GBDTParams(TypedDict):
    """Parameter grid entries for GradientBoostingRegressor."""

    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_leaf: int


def build_dataset(
    label: str,
    base_dir: str,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dataset:
    """Build the regression dataset from Synthea CSV exports.
    
    Features extracted (base model):
    - age_years: Patient age at dataset end date
    - male: Gender indicator (1.0 for male, 0.0 otherwise)
    - income: Annual income
    - bmi: Latest recorded BMI
    - smoker: Current smoker indicator
    - alcohol_use: Alcohol use disorder indicator
    - hypertension: Hypertension diagnosis indicator
    - chf: Congestive heart failure indicator
    
    The urban flag is stored separately for the linear bias model.
    
    Args:
        progress_callback: Optional callback(file_name, rows_processed) for progress.
    """
    csv_dir = find_csv_dir(base_dir)
    patients_path = csv_dir / "patients.csv"
    conditions_path = csv_dir / "conditions.csv"
    observations_path = csv_dir / "observations.csv"

    hypertension_patients = load_condition_patients(conditions_path, HYPERTENSION_CODES)
    chf_patients = load_condition_patients(conditions_path, CHF_CODES)
    alcohol_patients = load_condition_patients(conditions_path, ALCOHOL_USE_CODES)
    bmi_by_patient = load_latest_bmi(observations_path, BMI_CODE)
    smoking_by_patient = load_latest_smoking_status(
        observations_path,
        SMOKING_STATUS_CODE,
        SMOKER_VALUES,
        NON_SMOKER_VALUES,
    )
    ref_date = find_dataset_end_date(csv_dir)
    if ref_date is None:
        ref_date = date.today()
    # Anchor age calculations to the dataset end date.

    sleep_spend = load_sleep_spend(
        csv_dir,
        SLEEP_PROCEDURE_CODES,
        SLEEP_REASON_CODES,
        SLEEP_ENCOUNTER_CODES,
        SLEEP_EQUIPMENT_CODES,
        progress_callback=progress_callback,
    )
    
    # Load urban/rural flags for each patient
    urban_flags = load_patient_urban_flags(patients_path)

    feature_names = [
        "age_years",
        "male",
        "income",
        "bmi",
        "smoker",
        "alcohol_use",
        "hypertension",
        "chf",
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

            bmi = bmi_by_patient.get(patient_id)
            smoker = smoking_by_patient.get(patient_id)
            alcohol_use = 1.0 if patient_id in alcohol_patients else 0.0
            hypertension = 1.0 if patient_id in hypertension_patients else 0.0
            chf = 1.0 if patient_id in chf_patients else 0.0

            features = [
                age_years,
                male,
                income,
                bmi,
                smoker,
                alcohol_use,
                hypertension,
                chf,
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
        urban_flags=urban_flags,
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

    def _gather(idx_list: List[int]) -> Tuple[List[List[Optional[float]]], List[float], List[str], List[Optional[float]]]:
        """Collect features, labels, ids, and urban flags for a list of indices."""
        X_split = [dataset.X[i] for i in idx_list]
        y_split = [dataset.y[i] for i in idx_list]
        ids_split = [dataset.patient_ids[i] for i in idx_list]
        urban_split = [dataset.urban_flags.get(dataset.patient_ids[i]) for i in idx_list]
        return X_split, y_split, ids_split, urban_split

    X_train, y_train, train_ids, train_urban = _gather(train_idx)
    X_val, y_val, val_ids, val_urban = _gather(val_idx)
    X_test, y_test, test_ids, test_urban = _gather(test_idx)

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
        train_urban=train_urban,
        val_urban=val_urban,
        test_urban=test_urban,
    )


def train_with_validation(
    split: Split,
    param_grid: Sequence[GBDTParams],
    seed: int,
    feature_names: List[str],
    progress_callback: Optional[Callable[[], None]] = None,
) -> Tuple[GradientBoostingRegressor, GBDTParams, Dict[str, float], List[str]]:
    """Select hyperparameters by validation MAE and refit on train+val.
    
    Returns:
        Tuple of (final_model, best_params, validation_metrics, feature_names)
    """
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
        if progress_callback:
            progress_callback()

    if best_model is None or best_params is None or best_metrics is None:
        raise RuntimeError("Failed to train model on validation grid.")

    # Refit on combined train+val to maximize training data with the chosen params.
    X_train_val = split.X_train + split.X_val
    y_train_val = split.y_train + split.y_val
    final_model = GradientBoostingRegressor(random_state=seed, **best_params)
    final_model.fit(X_train_val, y_train_val)

    return final_model, best_params, best_metrics, feature_names


def add_urban_feature(
    split: Split,
    feature_names: List[str],
) -> Tuple[Split, List[str]]:
    """Add urban flag as a feature for an optional urban model variant.
    
    Creates a new Split with the urban feature appended to each feature vector.
    Missing urban values are imputed as 0.5 (unknown).
    """
    def _append_urban(X: List[List[float]], urban_flags: List[Optional[float]]) -> List[List[float]]:
        """Append urban flag to each feature vector."""
        return [
            row + [1.0 if u == 1 else 0.0 if u == 0 else 0.5]
            for row, u in zip(X, urban_flags)
        ]
    
    new_split = Split(
        X_train=_append_urban(split.X_train, split.train_urban),
        y_train=split.y_train,
        X_val=_append_urban(split.X_val, split.val_urban),
        y_val=split.y_val,
        X_test=_append_urban(split.X_test, split.test_urban),
        y_test=split.y_test,
        train_ids=split.train_ids,
        val_ids=split.val_ids,
        test_ids=split.test_ids,
        train_urban=split.train_urban,
        val_urban=split.val_urban,
        test_urban=split.test_urban,
    )
    new_feature_names = feature_names + ["urban"]
    return new_split, new_feature_names


def bootstrap_metrics(
    y_true: List[float],
    y_pred: List[float],
    n_bootstrap: int,
    seed: int,
    confidence: float = 0.95,
    progress_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for evaluation metrics.
    
    Returns dict with 'mae', 'rmse', 'r2' keys, each containing:
        - 'point': point estimate
        - 'ci_low': lower bound of CI
        - 'ci_high': upper bound of CI
        - 'std': bootstrap standard error
    """
    rng = random.Random(seed)
    n = len(y_true)
    
    point_metrics = evaluate_predictions(y_true, y_pred)
    
    bootstrap_results: Dict[str, List[float]] = {"mae": [], "rmse": [], "r2": []}
    
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]
        metrics = evaluate_predictions(y_true_boot, y_pred_boot)
        for key in bootstrap_results:
            bootstrap_results[key].append(metrics[key])
        if progress_callback:
            progress_callback()
    
    alpha = 1 - confidence
    result = {}
    for key in ["mae", "rmse", "r2"]:
        sorted_vals = sorted(bootstrap_results[key])
        low_idx = int(alpha / 2 * n_bootstrap)
        high_idx = int((1 - alpha / 2) * n_bootstrap) - 1
        mean_val = sum(bootstrap_results[key]) / n_bootstrap
        std_val = (sum((v - mean_val) ** 2 for v in bootstrap_results[key]) / n_bootstrap) ** 0.5
        result[key] = {
            "point": point_metrics[key],
            "ci_low": sorted_vals[low_idx],
            "ci_high": sorted_vals[high_idx],
            "std": std_val,
        }
    return result


@dataclass
class BiasAnalysis:
    """Results from linear regression bias quantification."""
    # Coefficients (unstandardized)
    coefficients: Dict[str, float]
    # Standardized coefficients for comparison
    standardized_coefficients: Dict[str, float]
    # Bootstrap confidence intervals for each coefficient
    coefficient_cis: Dict[str, Dict[str, float]]
    # Model fit statistics
    r2: float
    # The key metric: urban coefficient represents access disparity
    urban_coefficient: float
    urban_ci_low: float
    urban_ci_high: float
    urban_pvalue: float  # Permutation-based p-value


def fit_linear_bias_model(
    split: Split,
    feature_names: List[str],
    n_bootstrap: int,
    seed: int,
    progress_callback: Optional[Callable[[], None]] = None,
) -> BiasAnalysis:
    """Fit linear regression with urban feature to quantify geographic bias.
    
    The urban coefficient represents the effect of being urban (vs rural)
    on healthcare spending, controlling for the base demographic and clinical features.
    
    A significant negative coefficient in the biased dataset indicates rural
    patients have lower spending due to access barriers, not lower need.
    
    Returns:
        BiasAnalysis with coefficients, CIs, and significance tests.
    """
    rng = random.Random(seed)
    
    # Build feature matrix with urban flag
    def _add_urban(X: List[List[float]], urban_flags: List[Optional[float]]) -> List[List[float]]:
        return [
            row + [1.0 if u == 1.0 else 0.0 if u == 0.0 else 0.5]
            for row, u in zip(X, urban_flags)
        ]
    
    X_train = _add_urban(split.X_train, split.train_urban)
    X_test = _add_urban(split.X_test, split.test_urban)
    y_train = split.y_train
    y_test = split.y_test
    
    feature_names_with_urban = feature_names + ["urban"]
    
    # Fit on full training data
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get unstandardized coefficients
    coefficients = {name: coef for name, coef in zip(feature_names_with_urban, model.coef_)}
    coefficients["intercept"] = model.intercept_
    
    # Standardized coefficients (for relative importance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model_scaled = LinearRegression()
    model_scaled.fit(X_train_scaled, y_train)
    standardized_coefficients = {
        name: coef for name, coef in zip(feature_names_with_urban, model_scaled.coef_)
    }
    
    # R² on test set
    y_pred = model.predict(X_test)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_test, y_pred))
    ss_tot = sum((yt - sum(y_test) / len(y_test)) ** 2 for yt in y_test)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Bootstrap confidence intervals for coefficients
    n_train = len(X_train)
    bootstrap_coefs: Dict[str, List[float]] = {name: [] for name in feature_names_with_urban}
    bootstrap_coefs["intercept"] = []
    
    for _ in range(n_bootstrap):
        # Resample training data
        indices = [rng.randint(0, n_train - 1) for _ in range(n_train)]
        X_boot = [X_train[i] for i in indices]
        y_boot = [y_train[i] for i in indices]
        
        # Fit bootstrap model
        boot_model = LinearRegression()
        boot_model.fit(X_boot, y_boot)
        
        for name, coef in zip(feature_names_with_urban, boot_model.coef_):
            bootstrap_coefs[name].append(coef)
        bootstrap_coefs["intercept"].append(boot_model.intercept_)
        
        if progress_callback:
            progress_callback()
    
    # Compute CIs
    alpha = 0.05
    coefficient_cis = {}
    for name in list(feature_names_with_urban) + ["intercept"]:
        sorted_vals = sorted(bootstrap_coefs[name])
        low_idx = int(alpha / 2 * n_bootstrap)
        high_idx = int((1 - alpha / 2) * n_bootstrap) - 1
        mean_val = sum(bootstrap_coefs[name]) / n_bootstrap
        std_val = (sum((v - mean_val) ** 2 for v in bootstrap_coefs[name]) / n_bootstrap) ** 0.5
        coefficient_cis[name] = {
            "point": coefficients.get(name, coefficients.get("intercept", 0.0)),
            "ci_low": sorted_vals[low_idx],
            "ci_high": sorted_vals[high_idx],
            "std": std_val,
        }
    
    # Permutation test for urban coefficient significance
    # Test H0: urban coefficient = 0 by permuting urban labels
    observed_urban_coef = coefficients["urban"]
    urban_flags_train = [row[-1] for row in X_train]  # Extract urban from last column
    X_train_no_urban = [row[:-1] for row in X_train]
    
    count_extreme = 0
    n_perm = min(n_bootstrap, 1000)  # Use fewer permutations for speed
    for _ in range(n_perm):
        # Shuffle urban flags
        shuffled_urban = urban_flags_train.copy()
        rng.shuffle(shuffled_urban)
        X_perm = [row + [u] for row, u in zip(X_train_no_urban, shuffled_urban)]
        
        perm_model = LinearRegression()
        perm_model.fit(X_perm, y_train)
        perm_urban_coef = perm_model.coef_[-1]  # Urban is last feature
        
        if abs(perm_urban_coef) >= abs(observed_urban_coef):
            count_extreme += 1
    
    urban_pvalue = (count_extreme + 1) / (n_perm + 1)
    
    return BiasAnalysis(
        coefficients=coefficients,
        standardized_coefficients=standardized_coefficients,
        coefficient_cis=coefficient_cis,
        r2=r2,
        urban_coefficient=coefficients["urban"],
        urban_ci_low=coefficient_cis["urban"]["ci_low"],
        urban_ci_high=coefficient_cis["urban"]["ci_high"],
        urban_pvalue=urban_pvalue,
    )


def permutation_test(
    y_true_a: List[float],
    y_pred_a: List[float],
    y_true_b: List[float],
    y_pred_b: List[float],
    n_permutations: int,
    seed: int,
    metric: str = "mae",
    progress_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, float]:
    """Two-sample permutation test for metric difference between populations.
    
    Tests whether the difference in metric between populations A and B
    is statistically significant.
    
    Returns:
        - 'observed_diff': observed difference (metric_a - metric_b)
        - 'p_value': two-sided p-value
        - 'metric_a': metric for population A
        - 'metric_b': metric for population B
    """
    rng = random.Random(seed)
    
    metrics_a = evaluate_predictions(y_true_a, y_pred_a)
    metrics_b = evaluate_predictions(y_true_b, y_pred_b)
    observed_diff = metrics_a[metric] - metrics_b[metric]
    
    # Pool (y_true, y_pred) pairs and permute group labels.
    pooled = list(zip(y_true_a, y_pred_a)) + list(zip(y_true_b, y_pred_b))
    n_a = len(y_true_a)
    
    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_a = pooled[:n_a]
        perm_b = pooled[n_a:]

        perm_true_a = [t for t, _ in perm_a]
        perm_pred_a = [p for _, p in perm_a]
        perm_true_b = [t for t, _ in perm_b]
        perm_pred_b = [p for _, p in perm_b]

        perm_metrics_a = evaluate_predictions(perm_true_a, perm_pred_a)
        perm_metrics_b = evaluate_predictions(perm_true_b, perm_pred_b)
        perm_diff = perm_metrics_a[metric] - perm_metrics_b[metric]
        
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
        if progress_callback:
            progress_callback()
    
    p_value = (count_extreme + 1) / (n_permutations + 1)
    
    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "metric_a": metrics_a[metric],
        "metric_b": metrics_b[metric],
    }


@dataclass
class ModelResult:
    """Results from training a single model variant."""
    name: str
    model: GradientBoostingRegressor
    params: GBDTParams
    val_metrics: Dict[str, float]
    feature_names: List[str]
    split: Split


@dataclass
class CrossEvaluation:
    """Cross-population evaluation results with bootstrap CIs."""
    model_name: str
    source_dataset: str
    target_dataset: str
    metrics: Dict[str, Dict[str, float]]  # With CIs
    predictions: List[float]


def build_param_grid() -> List[GBDTParams]:
    """Build expanded hyperparameter grid for model selection.
    
    Grid covers:
    - n_estimators: [100, 200, 300, 500] - tree count
    - learning_rate: [0.01, 0.05, 0.1] - shrinkage
    - max_depth: [2, 3, 4] - tree complexity
    - min_samples_leaf: [5, 10, 20] - regularization
    """
    n_estimators_values = [100, 200, 300, 500]
    learning_rate_values = [0.01, 0.05, 0.1]
    max_depth_values = [2, 3, 4]
    min_samples_leaf_values = [5, 10, 20]
    
    grid: List[GBDTParams] = []
    for n_est, lr, depth, min_leaf in product(
        n_estimators_values,
        learning_rate_values,
        max_depth_values,
        min_samples_leaf_values,
    ):
        grid.append({
            "n_estimators": n_est,
            "learning_rate": lr,
            "max_depth": depth,
            "min_samples_leaf": min_leaf,
        })
    return grid


def main() -> int:
    """Entry point for training and reporting.
    
    Trains multiple model variants:
    1. base_baseline: Base features trained on baseline data (GBDT)
    2. base_biased: Base features trained on biased data (GBDT)
    3. bias_baseline: Linear regression with urban feature on baseline data
    4. bias_biased: Linear regression with urban feature on biased data
    
    Performs cross-population evaluation with bootstrap confidence intervals
    and permutation tests for statistical significance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train GBDT models to predict sleep-related spend and quantify bias with a linear model."
        )
    )
    parser.add_argument("--baseline", default="output_baseline", help="Baseline output directory.")
    parser.add_argument("--biased", default="output_rural_bias", help="Biased output directory.")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Training fraction.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations.")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Permutation test iterations.")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "sleep_apnea_demand_report.md"),
        help="Markdown output path.",
    )
    args = parser.parse_args()

    # Print header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Sleep Apnea Spend Model Training[/bold cyan]\n"
        "[dim]Comparing baseline vs biased datasets with linear bias analysis[/dim]",
        border_style="cyan"
    ))
    console.print()

    # Create progress display
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )

    with progress:
        # Loading datasets - estimate ~8M rows total per dataset (procedures + encounters + meds + devices + supplies)
        estimated_rows_per_dataset = 8_000_000
        load_task = progress.add_task(
            "[cyan]Loading baseline...",
            total=estimated_rows_per_dataset
        )

        def make_load_callback(task_id: TaskID, dataset_name: str) -> Callable[[str, int], None]:
            """Create a progress callback for dataset loading."""
            file_progress: Dict[str, int] = {}
            def callback(file_name: str, rows: int) -> None:
                prev = file_progress.get(file_name, 0)
                file_progress[file_name] = rows
                progress.advance(task_id, rows - prev)
                progress.update(task_id, description=f"[cyan]{dataset_name}: {file_name} ({rows:,} rows)")
            return callback

        baseline = build_dataset("baseline", args.baseline, make_load_callback(load_task, "Baseline"))
        progress.update(load_task, completed=estimated_rows_per_dataset)

        load_task2 = progress.add_task(
            "[cyan]Loading biased...",
            total=estimated_rows_per_dataset
        )
        biased = build_dataset("biased", args.biased, make_load_callback(load_task2, "Biased"))
        progress.update(load_task2, completed=estimated_rows_per_dataset)

        # Show dataset summary
        progress.stop()
        table = Table(title="Dataset Summary", show_header=True, header_style="bold magenta")
        table.add_column("Dataset", style="cyan")
        table.add_column("Patients", justify="right")
        table.add_column("Features", justify="right")
        table.add_row("Baseline", f"{len(baseline.patient_ids):,}", str(len(baseline.feature_names)))
        table.add_row("Biased", f"{len(biased.patient_ids):,}", str(len(biased.feature_names)))
        console.print(table)
        console.print()
        progress.start()

        # Splitting datasets
        split_task = progress.add_task("[cyan]Splitting datasets...", total=2)
        split_baseline = split_dataset(baseline, args.train_frac, args.val_frac, args.seed)
        progress.advance(split_task)
        split_biased = split_dataset(biased, args.train_frac, args.val_frac, args.seed)
        progress.advance(split_task)

        param_grid = build_param_grid()
        total_grid = len(param_grid)

        # Training base GBDT models (without urban feature) - 2 models
        train_task = progress.add_task(
            "[green]Training GBDT models (grid search)...",
            total=total_grid * 2
        )

        def advance_train() -> None:
            progress.advance(train_task)

        base_baseline_model, base_baseline_params, base_baseline_val, _ = train_with_validation(
            split_baseline, param_grid, args.seed, baseline.feature_names, advance_train
        )
        base_biased_model, base_biased_params, base_biased_val, _ = train_with_validation(
            split_biased, param_grid, args.seed, biased.feature_names, advance_train
        )

        # Linear regression bias analysis (replaces urban GBDT model)
        # This quantifies the urban coefficient to measure access disparity
        bias_task = progress.add_task(
            "[blue]Linear regression bias analysis...",
            total=args.n_bootstrap * 2
        )

        def advance_bias() -> None:
            progress.advance(bias_task)

        bias_baseline = fit_linear_bias_model(
            split_baseline, baseline.feature_names, args.n_bootstrap, args.seed, advance_bias
        )
        bias_biased = fit_linear_bias_model(
            split_biased, biased.feature_names, args.n_bootstrap, args.seed, advance_bias
        )

        # In-dataset test predictions (base models only)
        base_baseline_preds = list(base_baseline_model.predict(split_baseline.X_test))
        base_biased_preds = list(base_biased_model.predict(split_biased.X_test))

        # Cross-dataset predictions (base models only)
        base_biased_on_baseline_preds = list(base_biased_model.predict(split_baseline.X_test))
        base_baseline_on_biased_preds = list(base_baseline_model.predict(split_biased.X_test))

        # Bootstrap CIs - 4 evaluations × n_bootstrap iterations (reduced from 8)
        bootstrap_task = progress.add_task(
            "[yellow]Bootstrap confidence intervals...",
            total=args.n_bootstrap * 4
        )

        def advance_bootstrap() -> None:
            progress.advance(bootstrap_task)

        bootstrap_results = {
            "base_baseline_in": bootstrap_metrics(
                split_baseline.y_test, base_baseline_preds, args.n_bootstrap, args.seed,
                progress_callback=advance_bootstrap
            ),
            "base_biased_in": bootstrap_metrics(
                split_biased.y_test, base_biased_preds, args.n_bootstrap, args.seed,
                progress_callback=advance_bootstrap
            ),
            "base_biased_on_baseline": bootstrap_metrics(
                split_baseline.y_test, base_biased_on_baseline_preds, args.n_bootstrap, args.seed,
                progress_callback=advance_bootstrap
            ),
            "base_baseline_on_biased": bootstrap_metrics(
                split_biased.y_test, base_baseline_on_biased_preds, args.n_bootstrap, args.seed,
                progress_callback=advance_bootstrap
            ),
        }

        # Permutation test for cross-population performance
        perm_task = progress.add_task(
            "[magenta]Permutation tests...",
            total=args.n_permutations
        )

        def advance_perm() -> None:
            progress.advance(perm_task)

        permutation_results = {
            "base_cross_pop": permutation_test(
                split_baseline.y_test, base_biased_on_baseline_preds,
                split_biased.y_test, base_biased_preds,
                args.n_permutations, args.seed, "mae", advance_perm
            ),
        }

    # Print bias analysis summary
    console.print()
    console.print(Panel.fit(
        "[bold green]Bias Quantification Results[/bold green]",
        border_style="green"
    ))
    
    bias_table = Table(title="Urban Coefficient (Rural Access Disparity)", show_header=True, header_style="bold")
    bias_table.add_column("Dataset", style="cyan")
    bias_table.add_column("Urban Coef ($)", justify="right")
    bias_table.add_column("95% CI", justify="right")
    bias_table.add_column("p-value", justify="right")
    bias_table.add_column("Significant?", justify="center")
    
    def format_coef(ba: BiasAnalysis) -> Tuple[str, str, str, str]:
        coef = f"${ba.urban_coefficient:,.0f}"
        ci = f"[${ba.urban_ci_low:,.0f}, ${ba.urban_ci_high:,.0f}]"
        pval = f"{ba.urban_pvalue:.4f}"
        sig = "✓ Yes" if ba.urban_pvalue < 0.05 else "✗ No"
        return coef, ci, pval, sig
    
    coef_b, ci_b, pval_b, sig_b = format_coef(bias_baseline)
    coef_bi, ci_bi, pval_bi, sig_bi = format_coef(bias_biased)
    
    bias_table.add_row("Baseline (unbiased)", coef_b, ci_b, pval_b, sig_b)
    bias_table.add_row("Biased (rural dropout)", coef_bi, ci_bi, pval_bi, sig_bi)
    console.print(bias_table)
    
    # Interpretation
    diff = bias_biased.urban_coefficient - bias_baseline.urban_coefficient
    console.print()
    console.print(f"[bold]Interpretation:[/bold]")
    console.print(f"  • Baseline urban coefficient: ${bias_baseline.urban_coefficient:,.0f}")
    console.print(f"  • Biased urban coefficient: ${bias_biased.urban_coefficient:,.0f}")
    console.print(f"  • Difference (bias effect): ${diff:,.0f}")
    if diff < 0:
        console.print(f"  • [yellow]Rural patients in biased data show ${abs(diff):,.0f} less spending[/yellow]")
        console.print(f"    [dim]This represents the access disparity captured by the model[/dim]")
    console.print()

    # Legacy analysis for backward compatibility with report
    analysis = analyze_models(
        baseline=baseline,
        biased=biased,
        split_baseline=split_baseline,
        split_biased=split_biased,
        baseline_model=base_baseline_model,
        biased_model=base_biased_model,
        sleep_disorder_code=SLEEP_DISORDER_CODE,
        sleep_apnea_codes=SLEEP_APNEA_CODES,
    )

    # Feature names with urban for report
    feature_names_urban = baseline.feature_names + ["urban"]

    report_inputs = ReportInputs(
        baseline_csv_dir=baseline.csv_dir,
        biased_csv_dir=biased.csv_dir,
        feature_names=baseline.feature_names,
        feature_names_urban=feature_names_urban,
        baseline_params=base_baseline_params,
        biased_params=base_biased_params,
        urban_baseline_params=None,  # No longer using urban GBDT
        urban_biased_params=None,
        baseline_val_metrics=base_baseline_val,
        biased_val_metrics=base_biased_val,
        urban_baseline_val_metrics=None,
        urban_biased_val_metrics=None,
        analysis=analysis,
        baseline_generation_cmd=BASELINE_GENERATION_CMD,
        biased_generation_cmd=BIASED_GENERATION_CMD,
        sleep_reason_codes=SLEEP_REASON_CODES,
        sleep_procedure_codes=SLEEP_PROCEDURE_CODES,
        sleep_encounter_codes=SLEEP_ENCOUNTER_CODES,
        sleep_device_codes=SLEEP_DEVICE_CODES,
        sleep_supply_codes=SLEEP_SUPPLY_CODES,
        bootstrap_results=bootstrap_results,
        permutation_results=permutation_results,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        base_baseline_model=base_baseline_model,
        urban_baseline_model=None,  # No longer using urban GBDT
        base_biased_model=base_biased_model,
        urban_biased_model=None,
        # New bias analysis results
        bias_baseline=bias_baseline,
        bias_biased=bias_biased,
    )

    write_report(Path(args.out), report_inputs)

    print(f"Wrote report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
