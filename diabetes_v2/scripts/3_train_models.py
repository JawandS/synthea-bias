#!/usr/bin/env python3
"""
3_train_models.py - Train and evaluate GBDT models on baseline vs biased features.

This script:
1. Loads feature matrix from data.csv (created by 2_gen_bias.py)
2. Trains baseline model using true features (has_hyperglycemia, has_hypertriglyceridemia)
3. Trains biased model using observed features (observed_hyperglycemia, observed_hypertriglyceridemia)
4. Both models predict the same target: has_diabetes
5. Compares feature importance and performance differences
6. Outputs 3_model.md with specifications and performance comparison

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
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        denom = precision + recall
        f1_scores = np.zeros_like(precision)
        nonzero = denom > 0
        f1_scores[nonzero] = 2 * (precision[nonzero] * recall[nonzero]) / denom[nonzero]
        best_idx = np.argmax(f1_scores[:-1])
        return thresholds[best_idx]
    elif method == "youden":
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]
    elif method == "prevalence":
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

# Feature columns (shared between baseline and biased models)
SHARED_FEATURE_COLS = ["age", "male", "income", "a1c", "bmi", "smoker", "prediabetes", "obesity", "hypertension", "hyperlipidemia"]

# Metabolic features (different between baseline and biased)
BASELINE_METABOLIC_COLS = ["has_hyperglycemia", "has_hypertriglyceridemia"]
BIASED_METABOLIC_COLS = ["observed_hyperglycemia", "observed_hypertriglyceridemia"]


def load_data() -> pd.DataFrame:
    """Load consolidated data.csv."""
    return pd.read_csv(DATA_DIR / "data.csv")


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_params: dict,
    threshold_method: str = "f1",
) -> tuple[GradientBoostingClassifier, dict, float]:
    """Train GBDT and evaluate on test set with adaptive threshold."""
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

    # Test performance with adaptive threshold
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "val_auc": val_auc,
        "threshold": threshold,
        "test_auc": roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0,
        "test_ap": average_precision_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0,
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred, zero_division=0),
        "test_f1": f1_score(y_test, test_pred, zero_division=0),
    }

    return model, metrics, threshold


def get_feature_importance(model: GradientBoostingClassifier, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importance from trained model."""
    importances = model.feature_importances_
    return {name: float(imp) for name, imp in zip(feature_names, importances)}


def compute_subgroup_metrics(
    model: GradientBoostingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_test: pd.DataFrame,
    threshold: float,
) -> dict:
    """Compute metrics for diabetic subgroups based on true metabolic conditions."""
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    subgroup_metrics = {}

    # Subgroups based on TRUE metabolic conditions
    has_hg = df_test["has_hyperglycemia"] == 1
    has_ht = df_test["has_hypertriglyceridemia"] == 1
    has_both = has_hg & has_ht
    has_neither = ~has_hg & ~has_ht

    for name, mask in [
        ("with_hyperglycemia", has_hg.values),
        ("with_hypertriglyceridemia", has_ht.values),
        ("with_both", has_both.values),
        ("with_neither", has_neither.values),
    ]:
        if mask.sum() > 0 and len(np.unique(y_test[mask])) > 1:
            subgroup_metrics[f"{name}_auc"] = roc_auc_score(y_test[mask], test_proba[mask])
            subgroup_metrics[f"{name}_recall"] = recall_score(y_test[mask], test_pred[mask], zero_division=0)
            subgroup_metrics[f"{name}_precision"] = precision_score(y_test[mask], test_pred[mask], zero_division=0)
            subgroup_metrics[f"{name}_f1"] = f1_score(y_test[mask], test_pred[mask], zero_division=0)
        else:
            subgroup_metrics[f"{name}_auc"] = 0
            subgroup_metrics[f"{name}_recall"] = 0
            subgroup_metrics[f"{name}_precision"] = 0
            subgroup_metrics[f"{name}_f1"] = 0
        subgroup_metrics[f"{name}_n"] = int(mask.sum())
        subgroup_metrics[f"{name}_pos"] = int(y_test[mask].sum()) if mask.sum() > 0 else 0

    return subgroup_metrics


def write_model_report(
    baseline_metrics: dict,
    biased_metrics: dict,
    baseline_importance: dict,
    biased_importance: dict,
    baseline_subgroup: dict,
    biased_subgroup: dict,
    model_params: dict,
    data_stats: dict,
    threshold_method: str,
) -> None:
    """Write model comparison report to markdown file."""
    md_path = INFO_DIR / "3_model.md"

    def delta(key: str) -> str:
        diff = biased_metrics[key] - baseline_metrics[key]
        return f"{diff:+.4f}"

    def delta_sub(key: str) -> str:
        diff = biased_subgroup[key] - baseline_subgroup[key]
        return f"{diff:+.4f}"

    # Build feature importance table
    all_features = SHARED_FEATURE_COLS + ["hyperglycemia", "hypertriglyceridemia"]
    importance_rows = []
    for feat in all_features:
        if feat == "hyperglycemia":
            base_key = "has_hyperglycemia"
            bias_key = "observed_hyperglycemia"
        elif feat == "hypertriglyceridemia":
            base_key = "has_hypertriglyceridemia"
            bias_key = "observed_hypertriglyceridemia"
        else:
            base_key = feat
            bias_key = feat
        base_imp = baseline_importance.get(base_key, 0)
        bias_imp = biased_importance.get(bias_key, 0)
        diff = bias_imp - base_imp
        importance_rows.append(f"| {feat} | {base_imp:.4f} | {bias_imp:.4f} | {diff:+.4f} |")
    importance_table = "\n".join(importance_rows)

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

## Features

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| male | Gender (1 = male) |
| income | Household income (scaled) |
| a1c | Hemoglobin A1c level |
| bmi | Body mass index |
| smoker | Current smoker (1 = yes) |
| prediabetes | Has prediabetes diagnosis |
| obesity | Has obesity diagnosis |
| hypertension | Has hypertension diagnosis |
| hyperlipidemia | Has hyperlipidemia diagnosis |
| hyperglycemia | Has hyperglycemia (true or observed) |
| hypertriglyceridemia | Has hypertriglyceridemia (true or observed) |

## Model Comparison

| Model | Metabolic Features Used | Target |
|-------|-------------------------|--------|
| Baseline | `has_hyperglycemia`, `has_hypertriglyceridemia` (true) | `has_diabetes` |
| Biased | `observed_hyperglycemia`, `observed_hypertriglyceridemia` (30% masked) | `has_diabetes` |

## Data Split

| Split | Patients | Diabetes Cases | Prevalence |
|-------|----------|----------------|------------|
| Train | {data_stats['n_train']:,} | {data_stats['train_pos']:,} | {100*data_stats['train_pos']/data_stats['n_train']:.2f}% |
| Validation | {data_stats['n_val']:,} | {data_stats['val_pos']:,} | {100*data_stats['val_pos']/data_stats['n_val']:.2f}% |
| Test | {data_stats['n_test']:,} | {data_stats['test_pos']:,} | {100*data_stats['test_pos']/data_stats['n_test']:.2f}% |

## Overall Performance

| Metric | Baseline | Biased | Delta |
|--------|----------|--------|-------|
| AUC-ROC | {baseline_metrics['test_auc']:.4f} | {biased_metrics['test_auc']:.4f} | {delta('test_auc')} |
| Avg Precision | {baseline_metrics['test_ap']:.4f} | {biased_metrics['test_ap']:.4f} | {delta('test_ap')} |
| F1 Score | {baseline_metrics['test_f1']:.4f} | {biased_metrics['test_f1']:.4f} | {delta('test_f1')} |
| Precision | {baseline_metrics['test_precision']:.4f} | {biased_metrics['test_precision']:.4f} | {delta('test_precision')} |
| Recall | {baseline_metrics['test_recall']:.4f} | {biased_metrics['test_recall']:.4f} | {delta('test_recall')} |
| Accuracy | {baseline_metrics['test_accuracy']:.4f} | {biased_metrics['test_accuracy']:.4f} | {delta('test_accuracy')} |

## Feature Importance

| Feature | Baseline | Biased | Delta |
|---------|----------|--------|-------|
{importance_table}

## Subgroup Performance

Performance on patients grouped by TRUE metabolic condition status (regardless of which model).

### Test Set Composition by Metabolic Status

| Subgroup | Patients | Diabetes Cases | Prevalence |
|----------|----------|----------------|------------|
| With hyperglycemia | {baseline_subgroup['with_hyperglycemia_n']:,} | {baseline_subgroup['with_hyperglycemia_pos']:,} | {100*baseline_subgroup['with_hyperglycemia_pos']/max(baseline_subgroup['with_hyperglycemia_n'],1):.2f}% |
| With hypertriglyceridemia | {baseline_subgroup['with_hypertriglyceridemia_n']:,} | {baseline_subgroup['with_hypertriglyceridemia_pos']:,} | {100*baseline_subgroup['with_hypertriglyceridemia_pos']/max(baseline_subgroup['with_hypertriglyceridemia_n'],1):.2f}% |
| With both conditions | {baseline_subgroup['with_both_n']:,} | {baseline_subgroup['with_both_pos']:,} | {100*baseline_subgroup['with_both_pos']/max(baseline_subgroup['with_both_n'],1):.2f}% |
| With neither | {baseline_subgroup['with_neither_n']:,} | {baseline_subgroup['with_neither_pos']:,} | {100*baseline_subgroup['with_neither_pos']/max(baseline_subgroup['with_neither_n'],1):.2f}% |

### AUC-ROC by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| With hyperglycemia | {baseline_subgroup['with_hyperglycemia_auc']:.4f} | {biased_subgroup['with_hyperglycemia_auc']:.4f} | {delta_sub('with_hyperglycemia_auc')} |
| With hypertriglyceridemia | {baseline_subgroup['with_hypertriglyceridemia_auc']:.4f} | {biased_subgroup['with_hypertriglyceridemia_auc']:.4f} | {delta_sub('with_hypertriglyceridemia_auc')} |
| With both | {baseline_subgroup['with_both_auc']:.4f} | {biased_subgroup['with_both_auc']:.4f} | {delta_sub('with_both_auc')} |
| With neither | {baseline_subgroup['with_neither_auc']:.4f} | {biased_subgroup['with_neither_auc']:.4f} | {delta_sub('with_neither_auc')} |

### Recall by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| With hyperglycemia | {baseline_subgroup['with_hyperglycemia_recall']:.4f} | {biased_subgroup['with_hyperglycemia_recall']:.4f} | {delta_sub('with_hyperglycemia_recall')} |
| With hypertriglyceridemia | {baseline_subgroup['with_hypertriglyceridemia_recall']:.4f} | {biased_subgroup['with_hypertriglyceridemia_recall']:.4f} | {delta_sub('with_hypertriglyceridemia_recall')} |
| With both | {baseline_subgroup['with_both_recall']:.4f} | {biased_subgroup['with_both_recall']:.4f} | {delta_sub('with_both_recall')} |
| With neither | {baseline_subgroup['with_neither_recall']:.4f} | {biased_subgroup['with_neither_recall']:.4f} | {delta_sub('with_neither_recall')} |

## Key Findings

- **Overall AUC change**: {delta('test_auc')} (from {baseline_metrics['test_auc']:.4f} to {biased_metrics['test_auc']:.4f})
- **Overall F1 change**: {delta('test_f1')} (from {baseline_metrics['test_f1']:.4f} to {biased_metrics['test_f1']:.4f})
- **Hyperglycemia importance**: {baseline_importance.get('has_hyperglycemia', 0):.4f} -> {biased_importance.get('observed_hyperglycemia', 0):.4f}
- **Hypertriglyceridemia importance**: {baseline_importance.get('has_hypertriglyceridemia', 0):.4f} -> {biased_importance.get('observed_hypertriglyceridemia', 0):.4f}

> **Documentation bias effect**: When metabolic conditions are under-documented, the model learns
> weaker associations between these conditions and diabetes. This may cause the model to rely more
> heavily on other features (like A1c or demographics), potentially changing its behavior on
> populations where documentation practices differ.
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
    print("Diabetes v2: Model Training")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_data()

    # Check required columns
    required_cols = SHARED_FEATURE_COLS + BASELINE_METABOLIC_COLS + BIASED_METABOLIC_COLS + ["has_diabetes"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print("Run 2_gen_bias.py first to create data.csv")
        return

    # Build feature matrices
    baseline_feature_cols = SHARED_FEATURE_COLS + BASELINE_METABOLIC_COLS
    biased_feature_cols = SHARED_FEATURE_COLS + BIASED_METABOLIC_COLS

    X_baseline = df[baseline_feature_cols].to_numpy()
    X_biased = df[biased_feature_cols].to_numpy()
    y = df["has_diabetes"].to_numpy()

    print(f"  Samples: {len(df):,}")
    print(f"  Features: {len(baseline_feature_cols)}")
    print(f"  Diabetes prevalence: {np.mean(y):.2%}")

    # Train/val/test split (same indices for both models)
    print("\nSplitting data...")
    indices = np.arange(len(df))
    idx_temp, idx_test = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    val_ratio = args.val_size / (1 - args.test_size)
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=val_ratio, random_state=args.seed, stratify=y[idx_temp]
    )

    # Split data
    X_baseline_train, X_baseline_val, X_baseline_test = X_baseline[idx_train], X_baseline[idx_val], X_baseline[idx_test]
    X_biased_train, X_biased_val, X_biased_test = X_biased[idx_train], X_biased[idx_val], X_biased[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    df_test = df.iloc[idx_test]

    data_stats = {
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
        "train_pos": int(y_train.sum()),
        "val_pos": int(y_val.sum()),
        "test_pos": int(y_test.sum()),
    }

    print(f"  Train: {len(idx_train):,} ({np.mean(y_train):.2%} positive)")
    print(f"  Val: {len(idx_val):,} ({np.mean(y_val):.2%} positive)")
    print(f"  Test: {len(idx_test):,} ({np.mean(y_test):.2%} positive)")

    # Model parameters
    model_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "subsample": 0.8,
        "random_state": args.seed,
    }

    threshold_method = "f1"

    # Train baseline model (true metabolic features)
    print("\nTraining baseline model (true features)...")
    baseline_model, baseline_metrics, baseline_threshold = train_and_evaluate(
        X_baseline_train, y_train, X_baseline_val, y_val, X_baseline_test, y_test, model_params, threshold_method
    )
    baseline_importance = get_feature_importance(baseline_model, baseline_feature_cols)
    print(f"  Val AUC: {baseline_metrics['val_auc']:.4f}")
    print(f"  Test AUC: {baseline_metrics['test_auc']:.4f}")
    print(f"  Optimal threshold: {baseline_threshold:.4f}")
    print(f"  Test F1: {baseline_metrics['test_f1']:.4f}")

    # Train biased model (observed metabolic features)
    print("\nTraining biased model (observed features)...")
    biased_model, biased_metrics, biased_threshold = train_and_evaluate(
        X_biased_train, y_train, X_biased_val, y_val, X_biased_test, y_test, model_params, threshold_method
    )
    biased_importance = get_feature_importance(biased_model, biased_feature_cols)
    print(f"  Val AUC: {biased_metrics['val_auc']:.4f}")
    print(f"  Test AUC: {biased_metrics['test_auc']:.4f}")
    print(f"  Optimal threshold: {biased_threshold:.4f}")
    print(f"  Test F1: {biased_metrics['test_f1']:.4f}")

    # Compute subgroup metrics
    print("\nComputing subgroup metrics...")
    baseline_subgroup = compute_subgroup_metrics(baseline_model, X_baseline_test, y_test, df_test, baseline_threshold)
    biased_subgroup = compute_subgroup_metrics(biased_model, X_biased_test, y_test, df_test, biased_threshold)

    # Write report
    write_model_report(
        baseline_metrics,
        biased_metrics,
        baseline_importance,
        biased_importance,
        baseline_subgroup,
        biased_subgroup,
        model_params,
        data_stats,
        threshold_method,
    )

    # Console summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print(f"\nOverall Performance:")
    print(f"  Baseline AUC: {baseline_metrics['test_auc']:.4f}")
    print(f"  Biased AUC:   {biased_metrics['test_auc']:.4f}")
    print(f"  Delta:        {biased_metrics['test_auc'] - baseline_metrics['test_auc']:+.4f}")

    print(f"\nFeature Importance (metabolic conditions):")
    print(f"  Hyperglycemia:       {baseline_importance.get('has_hyperglycemia', 0):.4f} -> {biased_importance.get('observed_hyperglycemia', 0):.4f}")
    print(f"  Hypertriglyceridemia: {baseline_importance.get('has_hypertriglyceridemia', 0):.4f} -> {biased_importance.get('observed_hypertriglyceridemia', 0):.4f}")

    print(f"\nRecall on patients with hyperglycemia:")
    print(f"  Baseline: {baseline_subgroup['with_hyperglycemia_recall']:.4f}")
    print(f"  Biased:   {biased_subgroup['with_hyperglycemia_recall']:.4f}")

    print("\n" + "=" * 60)
    print(f"Complete! See {INFO_DIR / '3_model.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
