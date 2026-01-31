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
)
from sklearn.model_selection import train_test_split

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
) -> tuple[GradientBoostingClassifier, dict]:
    """Train GBDT and evaluate on test set."""
    model = GradientBoostingClassifier(**model_params)
    model.fit(X_train, y_train)

    # Validation performance (for tuning)
    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0

    # Test performance (evaluated on TRUE labels)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = model.predict(X_test)

    metrics = {
        "val_auc": val_auc,
        "test_auc": roc_auc_score(y_test_true, test_proba) if len(np.unique(y_test_true)) > 1 else 0,
        "test_ap": average_precision_score(y_test_true, test_proba) if len(np.unique(y_test_true)) > 1 else 0,
        "test_accuracy": accuracy_score(y_test_true, test_pred),
        "test_precision": precision_score(y_test_true, test_pred, zero_division=0),
        "test_recall": recall_score(y_test_true, test_pred, zero_division=0),
        "test_f1": f1_score(y_test_true, test_pred, zero_division=0),
    }

    return model, metrics


def compute_subgroup_metrics(
    model: GradientBoostingClassifier,
    X_test: pd.DataFrame,
    y_test_true: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Compute metrics for urban/rural subgroups."""
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = model.predict(X_test)

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
        else:
            subgroup_metrics[f"{name}_auc"] = 0
            subgroup_metrics[f"{name}_recall"] = 0
            subgroup_metrics[f"{name}_precision"] = 0

    return subgroup_metrics


def write_model_report(
    baseline_metrics: dict,
    biased_metrics: dict,
    baseline_subgroup: dict,
    biased_subgroup: dict,
    model_params: dict,
    feature_names: list[str],
    data_stats: dict,
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
| Accuracy | {baseline_metrics['test_accuracy']:.4f} | {biased_metrics['test_accuracy']:.4f} | {delta('test_accuracy')} |
| Precision | {baseline_metrics['test_precision']:.4f} | {biased_metrics['test_precision']:.4f} | {delta('test_precision')} |
| Recall | {baseline_metrics['test_recall']:.4f} | {biased_metrics['test_recall']:.4f} | {delta('test_recall')} |
| F1 Score | {baseline_metrics['test_f1']:.4f} | {biased_metrics['test_f1']:.4f} | {delta('test_f1')} |

## Subgroup Performance (Urban vs Rural)

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

## Key Findings

- **Overall AUC degradation**: {delta('test_auc')} (from {baseline_metrics['test_auc']:.4f} to {biased_metrics['test_auc']:.4f})
- **Rural recall degradation**: {delta_sub('rural_recall')} (most affected by underdiagnosis bias)
- **Urban performance**: Relatively stable as bias only affects rural population
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

    # Model parameters
    model_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": args.seed,
    }

    # Train baseline model (on true labels)
    print("\nTraining baseline model (true labels)...")
    baseline_model, baseline_metrics = train_and_evaluate(
        X_train, y_true_train, X_val, y_true_val, X_test, y_true_test, model_params
    )
    print(f"  Val AUC: {baseline_metrics['val_auc']:.4f}")
    print(f"  Test AUC: {baseline_metrics['test_auc']:.4f}")

    # Train biased model (on observed/masked labels)
    print("\nTraining biased model (observed labels)...")
    biased_model, biased_metrics = train_and_evaluate(
        X_train, y_obs_train, X_val, y_obs_val, X_test, y_true_test, model_params
    )
    print(f"  Val AUC: {biased_metrics['val_auc']:.4f}")
    print(f"  Test AUC: {biased_metrics['test_auc']:.4f}")

    # Compute subgroup metrics
    print("\nComputing subgroup metrics...")
    baseline_subgroup = compute_subgroup_metrics(baseline_model, X_test, y_true_test, feature_cols)
    biased_subgroup = compute_subgroup_metrics(biased_model, X_test, y_true_test, feature_cols)

    # Write report
    write_model_report(
        baseline_metrics,
        biased_metrics,
        baseline_subgroup,
        biased_subgroup,
        model_params,
        feature_cols,
        data_stats,
    )

    # Console summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\nOverall AUC:")
    print(f"  Baseline: {baseline_metrics['test_auc']:.4f}")
    print(f"  Biased:   {biased_metrics['test_auc']:.4f}")
    print(f"  Delta:    {biased_metrics['test_auc'] - baseline_metrics['test_auc']:+.4f}")

    print(f"\nRural Recall:")
    print(f"  Baseline: {baseline_subgroup['rural_recall']:.4f}")
    print(f"  Biased:   {biased_subgroup['rural_recall']:.4f}")
    print(f"  Delta:    {biased_subgroup['rural_recall'] - baseline_subgroup['rural_recall']:+.4f}")

    print("\n" + "=" * 60)
    print(f"Complete! See {INFO_DIR / '3_model.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
