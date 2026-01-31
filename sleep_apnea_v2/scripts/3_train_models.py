#!/usr/bin/env python3
"""
3_train_models.py - Train and evaluate GBDT models on baseline vs biased data.

This script:
1. Builds feature matrix from patients, conditions, and observations
2. Trains baseline model on true labels (has_sleep_apnea)
3. Trains biased model on observed labels (has_sleep_apnea & ~mask_sleep_apnea)
4. Evaluates both models on true labels to measure bias impact
5. Outputs 3_model.md with specifications and performance comparison

Usage:
    uv run python scripts/3_train_models.py [--test-size 0.15] [--val-size 0.15]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, method: str = "f1") -> float:
    """Find optimal classification threshold.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        method: 'f1' to maximize F1 score, 'youden' for Youden's J statistic

    Returns:
        Optimal threshold value
    """
    if method == "f1":
        # Find threshold that maximizes F1 score
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # Avoid division by zero - compute denominator first
        denom = precision + recall
        f1_scores = np.zeros_like(precision)
        nonzero = denom > 0
        f1_scores[nonzero] = 2 * (precision[nonzero] * recall[nonzero]) / denom[nonzero]
        # precision_recall_curve returns n+1 values, thresholds has n values
        best_idx = np.argmax(f1_scores[:-1])
        return thresholds[best_idx]
    elif method == "youden":
        # Youden's J = sensitivity + specificity - 1 = TPR - FPR
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]
    elif method == "prevalence":
        # Set threshold to match expected prevalence
        prevalence = y_true.mean()
        return np.percentile(y_proba, 100 * (1 - prevalence))
    else:
        return 0.5

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"

# Condition codes
HYPERTENSION_CODE = "59621000"
CHF_CODE = "88805009"
ALCOHOL_USE_CODE = "7200002"

# Observation codes
BMI_CODE = "39156-5"
SMOKING_CODE = "72166-2"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load patients, conditions, and observations data."""
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    conditions = pd.read_csv(DATA_DIR / "conditions.csv")
    observations = pd.read_csv(DATA_DIR / "observations.csv")
    return patients, conditions, observations


def build_features(
    patients: pd.DataFrame,
    conditions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature matrix for modeling."""
    df = patients[["Id"]].copy()

    # Age from birthdate
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
    reference_date = patients["BIRTHDATE"].max() + pd.DateOffset(years=70)
    df["age"] = ((reference_date - patients["BIRTHDATE"]).dt.days / 365.25).astype(int)

    # Gender (1 = male)
    df["male"] = (patients["GENDER"] == "M").astype(int)

    # Urban flag
    df["urban"] = patients["URBAN"].astype(int)

    # Income (normalized)
    df["income"] = patients["INCOME"] / 100000  # Scale to ~0-2 range

    # Conditions - check if patient has each condition
    codes = conditions["CODE"].astype(str)
    for code, name in [
        (HYPERTENSION_CODE, "hypertension"),
        (CHF_CODE, "chf"),
        (ALCOHOL_USE_CODE, "alcohol_use"),
    ]:
        patient_ids = set(conditions[codes == code]["PATIENT"].unique())
        df[name] = patients["Id"].isin(patient_ids).astype(int)

    # BMI - get latest value per patient
    bmi_obs = observations[observations["CODE"].astype(str) == BMI_CODE].copy()
    if len(bmi_obs) > 0:
        bmi_obs["DATE"] = pd.to_datetime(bmi_obs["DATE"])
        bmi_latest = bmi_obs.sort_values("DATE").groupby("PATIENT").last()["VALUE"]
        df["bmi"] = patients["Id"].map(bmi_latest).fillna(df["age"] * 0 + 25)  # Default BMI
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").fillna(25)
    else:
        df["bmi"] = 25.0

    # Smoking status - get latest value per patient
    smoking_obs = observations[observations["CODE"].astype(str) == SMOKING_CODE].copy()
    if len(smoking_obs) > 0:
        # Map smoking status to binary (current smoker = 1)
        smoking_obs["DATE"] = pd.to_datetime(smoking_obs["DATE"])
        smoking_latest = smoking_obs.sort_values("DATE").groupby("PATIENT").last()["VALUE"]
        smoking_map = smoking_latest.str.lower().str.contains("current|daily|occasional", na=False)
        df["smoker"] = patients["Id"].map(smoking_map).fillna(False).astype(int)
    else:
        df["smoker"] = 0

    # Target variables
    df["has_sleep_apnea"] = patients["has_sleep_apnea"].astype(int)
    df["mask_sleep_apnea"] = patients["mask_sleep_apnea"].astype(int)
    df["observed_sleep_apnea"] = (
        patients["has_sleep_apnea"] & ~patients["mask_sleep_apnea"]
    ).astype(int)

    return df


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test_true: np.ndarray,
    model_params: dict,
    threshold_method: str = "f1",
) -> tuple[GradientBoostingClassifier, dict, float]:
    """Train GBDT and evaluate on test set with adaptive threshold.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (used for threshold tuning)
        X_test, y_test_true: Test data with true labels
        model_params: GBDT hyperparameters
        threshold_method: Method for finding optimal threshold ('f1', 'youden', 'prevalence')

    Returns:
        Trained model, metrics dict, optimal threshold
    """
    model = GradientBoostingClassifier(**model_params)
    model.fit(X_train, y_train)

    # Validation performance and threshold tuning
    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0

    # Find optimal threshold on validation set
    if len(np.unique(y_val)) > 1:
        threshold = find_optimal_threshold(y_val, val_proba, method=threshold_method)
    else:
        threshold = 0.5

    # Test performance (evaluated on TRUE labels) with adaptive threshold
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "val_auc": val_auc,
        "threshold": threshold,
        "test_auc": roc_auc_score(y_test_true, test_proba) if len(np.unique(y_test_true)) > 1 else 0,
        "test_ap": average_precision_score(y_test_true, test_proba) if len(np.unique(y_test_true)) > 1 else 0,
        "test_accuracy": accuracy_score(y_test_true, test_pred),
        "test_precision": precision_score(y_test_true, test_pred, zero_division=0),
        "test_recall": recall_score(y_test_true, test_pred, zero_division=0),
        "test_f1": f1_score(y_test_true, test_pred, zero_division=0),
    }

    return model, metrics, threshold


def compute_subgroup_metrics(
    model: GradientBoostingClassifier,
    X_test: pd.DataFrame,
    y_test_true: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.5,
) -> dict:
    """Compute metrics for urban/rural subgroups with specified threshold."""
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    # Get urban column index
    urban_idx = feature_names.index("urban")
    urban_mask = X_test[:, urban_idx] == 1
    rural_mask = ~urban_mask

    subgroup_metrics = {}
    for name, mask in [("urban", urban_mask), ("rural", rural_mask)]:
        if mask.sum() > 0 and len(np.unique(y_test_true[mask])) > 1:
            subgroup_metrics[f"{name}_auc"] = roc_auc_score(y_test_true[mask], test_proba[mask])
            subgroup_metrics[f"{name}_recall"] = recall_score(y_test_true[mask], test_pred[mask], zero_division=0)
            subgroup_metrics[f"{name}_precision"] = precision_score(y_test_true[mask], test_pred[mask], zero_division=0)
            subgroup_metrics[f"{name}_f1"] = f1_score(y_test_true[mask], test_pred[mask], zero_division=0)
            subgroup_metrics[f"{name}_n"] = mask.sum()
            subgroup_metrics[f"{name}_pos"] = y_test_true[mask].sum()
        else:
            subgroup_metrics[f"{name}_auc"] = 0
            subgroup_metrics[f"{name}_recall"] = 0
            subgroup_metrics[f"{name}_precision"] = 0
            subgroup_metrics[f"{name}_f1"] = 0
            subgroup_metrics[f"{name}_n"] = mask.sum()
            subgroup_metrics[f"{name}_pos"] = 0

    return subgroup_metrics


def write_model_report(
    baseline_metrics: dict,
    biased_metrics: dict,
    baseline_subgroup: dict,
    biased_subgroup: dict,
    model_params: dict,
    feature_names: list[str],
    data_stats: dict,
    threshold_method: str,
) -> None:
    """Write model comparison report to markdown file."""
    md_path = INFO_DIR / "3_model.md"

    # Calculate deltas
    def delta(key: str) -> str:
        diff = biased_metrics[key] - baseline_metrics[key]
        return f"{diff:+.4f}"

    def delta_sub(key: str) -> str:
        diff = biased_subgroup[key] - baseline_subgroup[key]
        return f"{diff:+.4f}"

    content = f"""# Model Training Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Specification

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosted Decision Tree |
| n_estimators | {model_params['n_estimators']} |
| max_depth | {model_params['max_depth']} |
| learning_rate | {model_params['learning_rate']} |
| min_samples_split | {model_params['min_samples_split']} |
| min_samples_leaf | {model_params['min_samples_leaf']} |
| subsample | {model_params['subsample']} |

## Threshold Selection

| Parameter | Value |
|-----------|-------|
| Method | {threshold_method} (maximize F1 on validation set) |
| Baseline threshold | {baseline_metrics['threshold']:.4f} |
| Biased threshold | {biased_metrics['threshold']:.4f} |

> **Note**: With ~9% class prevalence, the default 0.5 threshold would rarely predict positives.
> Adaptive thresholding finds the optimal operating point that balances precision and recall.

## Features

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| male | Gender (1 = male) |
| urban | Location (1 = urban) |
| income | Household income (scaled) |
| bmi | Body mass index |
| smoker | Current smoker (1 = yes) |
| hypertension | Has hypertension diagnosis |
| chf | Has CHF diagnosis |
| alcohol_use | Has alcohol use disorder |

## Data Split

| Split | Patients | Sleep Apnea Cases | Prevalence |
|-------|----------|-------------------|------------|
| Train | {data_stats['n_train']:,} | {data_stats['train_pos']:,} | {100*data_stats['train_pos']/data_stats['n_train']:.2f}% |
| Validation | {data_stats['n_val']:,} | {data_stats['val_pos']:,} | {100*data_stats['val_pos']/data_stats['n_val']:.2f}% |
| Test | {data_stats['n_test']:,} | {data_stats['test_pos']:,} | {100*data_stats['test_pos']/data_stats['n_test']:.2f}% |

## Training Labels

| Model | Training Labels | Evaluation Labels |
|-------|-----------------|-------------------|
| Baseline | `has_sleep_apnea` (true) | `has_sleep_apnea` (true) |
| Biased | `observed_sleep_apnea` (masked) | `has_sleep_apnea` (true) |

## Overall Performance

| Metric | Baseline | Biased | Delta |
|--------|----------|--------|-------|
| AUC-ROC | {baseline_metrics['test_auc']:.4f} | {biased_metrics['test_auc']:.4f} | {delta('test_auc')} |
| Avg Precision | {baseline_metrics['test_ap']:.4f} | {biased_metrics['test_ap']:.4f} | {delta('test_ap')} |
| F1 Score | {baseline_metrics['test_f1']:.4f} | {biased_metrics['test_f1']:.4f} | {delta('test_f1')} |
| Precision | {baseline_metrics['test_precision']:.4f} | {biased_metrics['test_precision']:.4f} | {delta('test_precision')} |
| Recall | {baseline_metrics['test_recall']:.4f} | {biased_metrics['test_recall']:.4f} | {delta('test_recall')} |
| Accuracy | {baseline_metrics['test_accuracy']:.4f} | {biased_metrics['test_accuracy']:.4f} | {delta('test_accuracy')} |

## Subgroup Performance (Urban vs Rural)

### Test Set Composition

| Subgroup | Patients | Apnea Cases | Prevalence |
|----------|----------|-------------|------------|
| Urban | {baseline_subgroup['urban_n']:,} | {baseline_subgroup['urban_pos']:,} | {100*baseline_subgroup['urban_pos']/baseline_subgroup['urban_n']:.2f}% |
| Rural | {baseline_subgroup['rural_n']:,} | {baseline_subgroup['rural_pos']:,} | {100*baseline_subgroup['rural_pos']/baseline_subgroup['rural_n']:.2f}% |

### AUC-ROC by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | {baseline_subgroup['urban_auc']:.4f} | {biased_subgroup['urban_auc']:.4f} | {delta_sub('urban_auc')} |
| Rural | {baseline_subgroup['rural_auc']:.4f} | {biased_subgroup['rural_auc']:.4f} | {delta_sub('rural_auc')} |

### Recall by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | {baseline_subgroup['urban_recall']:.4f} | {biased_subgroup['urban_recall']:.4f} | {delta_sub('urban_recall')} |
| Rural | {baseline_subgroup['rural_recall']:.4f} | {biased_subgroup['rural_recall']:.4f} | {delta_sub('rural_recall')} |

### Precision by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | {baseline_subgroup['urban_precision']:.4f} | {biased_subgroup['urban_precision']:.4f} | {delta_sub('urban_precision')} |
| Rural | {baseline_subgroup['rural_precision']:.4f} | {biased_subgroup['rural_precision']:.4f} | {delta_sub('rural_precision')} |

### F1 Score by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | {baseline_subgroup['urban_f1']:.4f} | {biased_subgroup['urban_f1']:.4f} | {delta_sub('urban_f1')} |
| Rural | {baseline_subgroup['rural_f1']:.4f} | {biased_subgroup['rural_f1']:.4f} | {delta_sub('rural_f1')} |

## Key Findings

- **Rural AUC degradation**: {delta_sub('rural_auc')} (from {baseline_subgroup['rural_auc']:.4f} to {biased_subgroup['rural_auc']:.4f})
  - The biased model has reduced ability to discriminate sleep apnea in rural patients
- **Urban AUC change**: {delta_sub('urban_auc')} (relatively stable)
- **Disparity gap**: Rural AUC changes by {delta_sub('rural_auc')} vs Urban by {delta_sub('urban_auc')}
  - Bias introduces/widens performance gap between subgroups
- **Threshold shift**: Biased model uses lower threshold ({biased_metrics['threshold']:.4f} vs {baseline_metrics['threshold']:.4f})
  - Compensates for reduced signal from missing rural positives in training
"""

    md_path.write_text(content)
    print(f"Wrote {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GBDT models")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test set fraction (default: 0.15)")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation set fraction (default: 0.15)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("Sleep Apnea v2: Model Training")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    patients, conditions, observations = load_data()

    # Check required columns
    if "has_sleep_apnea" not in patients.columns:
        print("Error: Run 2_gen_bias.py first to add sleep apnea flags")
        return

    # Build features
    print("Building features...")
    df = build_features(patients, conditions, observations)

    feature_cols = ["age", "male", "urban", "income", "bmi", "smoker", "hypertension", "chf", "alcohol_use"]
    X = df[feature_cols].to_numpy()
    y_true = df["has_sleep_apnea"].to_numpy()
    y_observed = df["observed_sleep_apnea"].to_numpy()

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X):,}")
    print(f"  True prevalence: {np.mean(y_true):.2%}")
    print(f"  Observed prevalence: {np.mean(y_observed):.2%}")

    # Train/val/test split
    print("\nSplitting data...")
    X_temp, X_test, y_true_temp, y_true_test, y_obs_temp, y_obs_test = train_test_split(
        X, y_true, y_observed, test_size=args.test_size, random_state=args.seed, stratify=y_true
    )

    val_ratio = args.val_size / (1 - args.test_size)
    X_train, X_val, y_true_train, y_true_val, y_obs_train, y_obs_val = train_test_split(
        X_temp, y_true_temp, y_obs_temp, test_size=val_ratio, random_state=args.seed, stratify=y_true_temp
    )

    data_stats = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "train_pos": y_true_train.sum(),
        "val_pos": y_true_val.sum(),
        "test_pos": y_true_test.sum(),
    }

    print(f"  Train: {len(X_train):,} ({np.mean(y_true_train):.2%} positive)")
    print(f"  Val: {len(X_val):,} ({np.mean(y_true_val):.2%} positive)")
    print(f"  Test: {len(X_test):,} ({np.mean(y_true_test):.2%} positive)")

    # Model parameters - improved for imbalanced classification
    model_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "subsample": 0.8,  # Stochastic gradient boosting for regularization
        "random_state": args.seed,
    }

    # Threshold selection method
    threshold_method = "f1"  # Maximize F1 score on validation set

    # Train baseline model (on true labels)
    print("\nTraining baseline model (true labels)...")
    baseline_model, baseline_metrics, baseline_threshold = train_and_evaluate(
        X_train, y_true_train, X_val, y_true_val, X_test, y_true_test, model_params, threshold_method
    )
    print(f"  Val AUC: {baseline_metrics['val_auc']:.4f}")
    print(f"  Test AUC: {baseline_metrics['test_auc']:.4f}")
    print(f"  Optimal threshold: {baseline_threshold:.4f}")
    print(f"  Test F1: {baseline_metrics['test_f1']:.4f}")

    # Train biased model (on observed/masked labels)
    print("\nTraining biased model (observed labels)...")
    biased_model, biased_metrics, biased_threshold = train_and_evaluate(
        X_train, y_obs_train, X_val, y_obs_val, X_test, y_true_test, model_params, threshold_method
    )
    print(f"  Val AUC: {biased_metrics['val_auc']:.4f}")
    print(f"  Test AUC: {biased_metrics['test_auc']:.4f}")
    print(f"  Optimal threshold: {biased_threshold:.4f}")
    print(f"  Test F1: {biased_metrics['test_f1']:.4f}")

    # Compute subgroup metrics with respective thresholds
    print("\nComputing subgroup metrics...")
    baseline_subgroup = compute_subgroup_metrics(
        baseline_model, X_test, y_true_test, feature_cols, baseline_threshold
    )
    biased_subgroup = compute_subgroup_metrics(
        biased_model, X_test, y_true_test, feature_cols, biased_threshold
    )

    # Write report
    write_model_report(
        baseline_metrics,
        biased_metrics,
        baseline_subgroup,
        biased_subgroup,
        model_params,
        feature_cols,
        data_stats,
        threshold_method,
    )

    # Console summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print(f"\nThresholds (F1-optimized):")
    print(f"  Baseline: {baseline_threshold:.4f}")
    print(f"  Biased:   {biased_threshold:.4f}")

    print(f"\nOverall F1:")
    print(f"  Baseline: {baseline_metrics['test_f1']:.4f}")
    print(f"  Biased:   {biased_metrics['test_f1']:.4f}")
    print(f"  Delta:    {biased_metrics['test_f1'] - baseline_metrics['test_f1']:+.4f}")

    print(f"\nRural Recall (sensitivity to true positives):")
    print(f"  Baseline: {baseline_subgroup['rural_recall']:.4f}")
    print(f"  Biased:   {biased_subgroup['rural_recall']:.4f}")
    print(f"  Delta:    {biased_subgroup['rural_recall'] - baseline_subgroup['rural_recall']:+.4f}")

    print(f"\nRural F1:")
    print(f"  Baseline: {baseline_subgroup['rural_f1']:.4f}")
    print(f"  Biased:   {biased_subgroup['rural_f1']:.4f}")
    print(f"  Delta:    {biased_subgroup['rural_f1'] - baseline_subgroup['rural_f1']:+.4f}")

    print("\n" + "=" * 60)
    print(f"Complete! See {INFO_DIR / '3_model.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
