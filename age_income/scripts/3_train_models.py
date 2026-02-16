#!/usr/bin/env python3
"""
3_train_models.py - Train baseline vs biased models for CRC and early detection.

This script trains two tasks:
1. CRC diagnosis model (target: has_crc_true vs observed_crc)
2. Early CRC detection model (target: has_early_crc_true vs observed_early_crc)

For each task, it trains:
- Baseline model: trained on true labels
- Biased model: trained on observed labels

Both are evaluated on the same held-out test set against true labels.
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
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"
MODELS_DIR = OUTPUT_DIR / "models"

MODEL_FEATURES = [
    "age",
    "male",
    "bmi",
    "smoker",
    "diabetes",
    "prediabetes",
    "obesity",
    "hypertension",
    "hyperlipidemia",
    "chf",
]

AGE_BANDS = ["40-49", "50-59", "60-69", "70-79", "80+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRC models on true vs observed labels")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "data.csv")


def make_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    feature_df = df[MODEL_FEATURES].copy()
    return feature_df.to_numpy(), list(feature_df.columns)


def safe_metric(fn, y_true: np.ndarray, y_pred: np.ndarray | None = None, y_score: np.ndarray | None = None) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    if y_score is not None:
        return float(fn(y_true, y_score))
    if y_pred is not None:
        return float(fn(y_true, y_pred, zero_division=0))
    return 0.0


def compute_fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def select_dynamic_threshold(y_val: np.ndarray, val_proba: np.ndarray) -> float:
    """Choose a threshold robustly under class imbalance.

    Strategy:
    1. Search a dense threshold grid and optimize an F2-oriented utility.
    2. Constrain to thresholds that emit at least a minimum number of positives.
    3. If no threshold satisfies the constraint, fall back to prevalence matching.
    """
    if len(np.unique(y_val)) < 2:
        return 0.5

    n = len(y_val)
    prevalence = float(y_val.mean())
    # Keep a usable operating point: not too sparse, not "everyone positive".
    min_rate = max(0.02, 0.8 * prevalence)
    max_rate = min(0.25, 6.0 * prevalence)
    min_pos_pred = max(1, int(min_rate * n))
    max_pos_pred = max(min_pos_pred, int(max_rate * n))
    recall_floor = max(0.15, min(0.50, 4 * prevalence))

    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 501),
                np.quantile(val_proba, np.linspace(0.0, 1.0, 201)),
            ]
        )
    )

    best_threshold = 0.5
    best_score = -1.0
    found_recall_constrained = False
    found_rate_constrained = False

    for t in thresholds:
        pred = (val_proba >= t).astype(int)
        n_pos_pred = int(pred.sum())
        precision = precision_score(y_val, pred, zero_division=0)
        recall = recall_score(y_val, pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, zero_division=0)
        # Seek usable balance: favor F1/precision under realistic alert volume.
        utility = (0.50 * f1) + (0.30 * precision) + (0.20 * bal_acc)

        within_rate = min_pos_pred <= n_pos_pred <= max_pos_pred

        if within_rate and recall >= recall_floor:
            found_recall_constrained = True
            if utility > best_score:
                best_score = utility
                best_threshold = float(t)
        elif within_rate and not found_recall_constrained:
            found_rate_constrained = True
            if utility > best_score:
                best_score = utility
                best_threshold = float(t)
        elif not found_recall_constrained and not found_rate_constrained:
            # As a final fallback during search, keep best available candidate.
            if utility > best_score:
                best_score = utility
                best_threshold = float(t)

    if found_recall_constrained or found_rate_constrained:
        return best_threshold

    # Fallback: predicted positive rate ~ observed prevalence.
    target_rate = min(0.25, max(prevalence * 3.0, 1 / max(n, 1)))
    fallback = float(np.quantile(val_proba, max(0.0, 1 - target_rate)))
    return fallback


def train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test_true: np.ndarray,
    seed: int,
) -> tuple[dict, np.ndarray]:
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=seed,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = select_dynamic_threshold(y_val, val_proba)
    val_pred = (val_proba >= threshold).astype(int)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "val_pred_pos": int(val_pred.sum()),
        "test_pred_pos": int(pred.sum()),
        "val_prevalence": float(np.mean(y_val)),
        "test_prevalence": float(np.mean(y_test_true)),
        "auc": safe_metric(roc_auc_score, y_test_true, y_score=proba),
        "ap": safe_metric(average_precision_score, y_test_true, y_score=proba),
        "accuracy": float(accuracy_score(y_test_true, pred)),
        "precision": safe_metric(precision_score, y_test_true, y_pred=pred),
        "recall": safe_metric(recall_score, y_test_true, y_pred=pred),
        "f1": safe_metric(f1_score, y_test_true, y_pred=pred),
    }
    return metrics, pred


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    group_name: str,
    task: str,
    model_variant: str,
) -> pd.DataFrame:
    rows = []
    temp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups.astype(str)})

    for group_val, sub in temp.groupby("group"):
        positives = int(sub["y_true"].sum())
        if positives > 0:
            tp = int(((sub["y_true"] == 1) & (sub["y_pred"] == 1)).sum())
            fn = int(((sub["y_true"] == 1) & (sub["y_pred"] == 0)).sum())
            recall = tp / positives
            fnr = fn / positives
        else:
            recall = 0.0
            fnr = 0.0

        rows.append(
            {
                "task": task,
                "model": model_variant,
                "group_type": group_name,
                "group": group_val,
                "n": len(sub),
                "positives": positives,
                "recall": recall,
                "fnr": fnr,
            }
        )

    return pd.DataFrame(rows)


def subgroup_table_markdown(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "No subgroup rows available."
    lines = [
        "| Group | N | Positives | Recall | FNR |",
        "|-------|---:|----------:|-------:|----:|",
    ]
    for _, r in df.sort_values("group").iterrows():
        lines.append(
            f"| {r['group']} | {int(r['n']):,} | {int(r['positives']):,} | {r['recall']:.3f} | {r['fnr']:.3f} |"
        )
    return "\n".join(lines)


def build_task_section(task_name: str, baseline: dict, biased: dict) -> str:
    rows = []
    for metric in ["auc", "ap", "accuracy", "precision", "recall", "f1"]:
        b = baseline[metric]
        o = biased[metric]
        rows.append(f"| {metric.upper()} | {b:.4f} | {o:.4f} | {(o - b):+.4f} |")

    return f"""## {task_name}

Thresholds: baseline `{baseline['threshold']:.3f}`, biased `{biased['threshold']:.3f}`
Operating points:
- Baseline: val positives `{baseline['val_pred_pos']}`, test positives `{baseline['test_pred_pos']}`, test prevalence `{baseline['test_prevalence']:.3%}`
- Biased: val positives `{biased['val_pred_pos']}`, test positives `{biased['test_pred_pos']}`, test prevalence `{biased['test_prevalence']:.3%}`

| Metric | Baseline (train=true) | Biased (train=observed) | Delta |
|--------|------------------------|--------------------------|-------|
{chr(10).join(rows)}
"""


def main() -> None:
    args = parse_args()
    if args.test_size <= 0 or args.test_size >= 1:
        raise ValueError("--test-size must be in (0, 1)")
    if args.val_size <= 0 or (args.test_size + args.val_size) >= 1:
        raise ValueError("--val-size must be > 0 and test_size + val_size must be < 1")

    df = load_data()
    X, feature_names = make_feature_matrix(df)

    idx = np.arange(len(df))
    idx_train_val, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    val_fraction = args.val_size / (1 - args.test_size)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=val_fraction, random_state=args.seed)

    y_crc_true = df["has_crc_true"].to_numpy()
    y_crc_obs = df["observed_crc"].to_numpy()

    y_early_true = df["has_early_crc_true"].to_numpy()
    y_early_obs = df["observed_early_crc"].to_numpy()

    age_groups = df.loc[idx_test, "age_band"].astype(str)
    income_groups = pd.qcut(df.loc[idx_test, "income"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop").astype(str)
    age_income_groups = age_groups + "|" + income_groups

    results = {}
    subgroup_frames = []

    # Task 1: CRC diagnosis
    crc_base_metrics, crc_base_pred = train_and_eval(
        X[idx_train], y_crc_true[idx_train], X[idx_val], y_crc_true[idx_val], X[idx_test], y_crc_true[idx_test], args.seed
    )
    crc_bias_metrics, crc_bias_pred = train_and_eval(
        X[idx_train], y_crc_obs[idx_train], X[idx_val], y_crc_obs[idx_val], X[idx_test], y_crc_true[idx_test], args.seed
    )
    results["CRC Diagnosis"] = {"baseline": crc_base_metrics, "biased": crc_bias_metrics}

    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_base_pred, age_groups, "age_band", "CRC Diagnosis", "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_base_pred, income_groups, "income_quintile", "CRC Diagnosis", "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_base_pred, age_income_groups, "age_x_income", "CRC Diagnosis", "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_bias_pred, age_groups, "age_band", "CRC Diagnosis", "biased"))
    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_bias_pred, income_groups, "income_quintile", "CRC Diagnosis", "biased"))
    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_bias_pred, age_income_groups, "age_x_income", "CRC Diagnosis", "biased"))

    # Task 2: Early CRC detection
    early_base_metrics, early_base_pred = train_and_eval(
        X[idx_train], y_early_true[idx_train], X[idx_val], y_early_true[idx_val], X[idx_test], y_early_true[idx_test], args.seed
    )
    early_bias_metrics, early_bias_pred = train_and_eval(
        X[idx_train], y_early_obs[idx_train], X[idx_val], y_early_obs[idx_val], X[idx_test], y_early_true[idx_test], args.seed
    )
    results["Early CRC Detection"] = {"baseline": early_base_metrics, "biased": early_bias_metrics}

    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_base_pred, age_groups, "age_band", "Early CRC Detection", "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_base_pred, income_groups, "income_quintile", "Early CRC Detection", "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_base_pred, age_income_groups, "age_x_income", "Early CRC Detection", "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_bias_pred, age_groups, "age_band", "Early CRC Detection", "biased"))
    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_bias_pred, income_groups, "income_quintile", "Early CRC Detection", "biased"))
    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_bias_pred, age_income_groups, "age_x_income", "Early CRC Detection", "biased"))

    subgroup_df = pd.concat(subgroup_frames, ignore_index=True)

    INFO_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    summary = [
        "# Model Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- Samples: {len(df):,}",
        f"- Features: {', '.join(feature_names)}",
        "- Excluded from model features: income, poverty_ratio, assigned_plan, eligible_for_screening",
        f"- Train size: {len(idx_train):,}",
        f"- Validation size: {len(idx_val):,}",
        f"- Test size: {len(idx_test):,}",
        f"- Seed: {args.seed}",
        "",
    ]

    for task, payload in results.items():
        summary.append(build_task_section(task, payload["baseline"], payload["biased"]))
        for group_type, title in [
            ("age_band", "Age Band"),
            ("income_quintile", "Income Quintile"),
            ("age_x_income", "Age x Income"),
        ]:
            base_sub = subgroup_df[
                (subgroup_df["task"] == task)
                & (subgroup_df["model"] == "baseline")
                & (subgroup_df["group_type"] == group_type)
            ]
            bias_sub = subgroup_df[
                (subgroup_df["task"] == task)
                & (subgroup_df["model"] == "biased")
                & (subgroup_df["group_type"] == group_type)
            ]

            summary.append(f"### {task} - {title} Subgroup Metrics (Baseline)")
            summary.append("")
            summary.append(subgroup_table_markdown(base_sub))
            summary.append("")
            summary.append(f"### {task} - {title} Subgroup Metrics (Biased)")
            summary.append("")
            summary.append(subgroup_table_markdown(bias_sub))
            summary.append("")

    output_path = INFO_DIR / "3_model.md"
    output_path.write_text("\n".join(summary))

    meta = pd.DataFrame(
        [
            {
                "task": task,
                "metric": metric,
                "baseline": payload["baseline"][metric],
                "biased": payload["biased"][metric],
            }
            for task, payload in results.items()
            for metric in ["auc", "ap", "accuracy", "precision", "recall", "f1"]
        ]
    )
    meta.to_csv(MODELS_DIR / "metrics.csv", index=False)
    subgroup_df.to_csv(MODELS_DIR / "subgroup_metrics.csv", index=False)

    print(f"Wrote {output_path}")
    print(f"Wrote {MODELS_DIR / 'metrics.csv'}")
    print(f"Wrote {MODELS_DIR / 'subgroup_metrics.csv'}")


if __name__ == "__main__":
    main()
