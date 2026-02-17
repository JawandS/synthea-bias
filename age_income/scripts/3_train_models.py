#!/usr/bin/env python3
"""
3_train_models.py - Train baseline vs biased screening recommendation models.

This script trains two tasks:
1. CRC risk screening recommendation (target: has_crc_true vs observed_crc)
2. Early-stage capture recommendation (target: has_early_crc_true vs observed_early_crc)

For each task, it trains:
- Equity-oriented baseline model: trained on true labels
- Historically-biased model: trained on observed labels

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
CRC_TASK_NAME = "CRC Screening Recommendation (Risk of CRC)"
INCOME_CUTOFF_LOW = 40000
INCOME_CUTOFF_HIGH = 90000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train screening recommendation models on true vs observed labels")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "data.csv")


def age_band_from_series(age: pd.Series) -> pd.Series:
    return pd.cut(
        age,
        bins=[0, 49, 59, 69, 79, 1000],
        labels=AGE_BANDS,
    ).astype(str)


def income_band_from_series(income: pd.Series) -> pd.Series:
    return pd.cut(
        income,
        bins=[-np.inf, INCOME_CUTOFF_LOW, INCOME_CUTOFF_HIGH, np.inf],
        labels=["low_income", "middle_income", "high_income"],
        right=False,
    ).astype(str)


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


def aggregate_delta_table(results: dict[str, dict[str, dict]]) -> str:
    lines = [
        "| Task | Metric | Baseline | Biased | Delta (Biased - Baseline) |",
        "|------|--------|---------:|-------:|---------------------------:|",
    ]
    for task, payload in results.items():
        for metric in ["auc", "ap", "accuracy", "precision", "recall", "f1"]:
            b = payload["baseline"][metric]
            o = payload["biased"][metric]
            lines.append(f"| {task} | {metric.upper()} | {b:.4f} | {o:.4f} | {(o - b):+.4f} |")
    return "\n".join(lines)


def subgroup_delta_table(subgroup_df: pd.DataFrame, group_type: str, task_name: str | None = None) -> str:
    rows = []
    base = subgroup_df[subgroup_df["model"] == "baseline"]
    bias = subgroup_df[subgroup_df["model"] == "biased"]
    merged = base.merge(
        bias,
        on=["task", "group_type", "group", "n", "positives"],
        suffixes=("_baseline", "_biased"),
    )
    merged = merged[merged["group_type"] == group_type].copy()
    if task_name is not None:
        merged = merged[merged["task"] == task_name].copy()
    merged["delta_recall"] = merged["recall_biased"] - merged["recall_baseline"]
    merged["delta_fnr"] = merged["fnr_biased"] - merged["fnr_baseline"]
    merged = merged.sort_values(["task", "group"])

    if len(merged) == 0:
        return "No subgroup rows available."

    lines = [
        "| Task | Group | N | Positives | Baseline Recall | Biased Recall | Delta Recall | Baseline FNR | Biased FNR | Delta FNR |",
        "|------|-------|--:|----------:|----------------:|--------------:|-------------:|-------------:|-----------:|----------:|",
    ]
    for _, r in merged.iterrows():
        lines.append(
            f"| {r['task']} | {r['group']} | {int(r['n']):,} | {int(r['positives']):,} | "
            f"{r['recall_baseline']:.3f} | {r['recall_biased']:.3f} | {r['delta_recall']:+.3f} | "
            f"{r['fnr_baseline']:.3f} | {r['fnr_biased']:.3f} | {r['delta_fnr']:+.3f} |"
        )
    return "\n".join(lines)


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

    age_groups = age_band_from_series(df.loc[idx_test, "age"])
    income_groups = income_band_from_series(df.loc[idx_test, "income"])
    age_income_groups = age_groups + "|" + income_groups

    results = {}
    subgroup_frames = []

    # Task 1: Screening recommendation for CRC risk
    crc_base_metrics, crc_base_pred = train_and_eval(
        X[idx_train], y_crc_true[idx_train], X[idx_val], y_crc_true[idx_val], X[idx_test], y_crc_true[idx_test], args.seed
    )
    crc_bias_metrics, crc_bias_pred = train_and_eval(
        X[idx_train], y_crc_obs[idx_train], X[idx_val], y_crc_obs[idx_val], X[idx_test], y_crc_true[idx_test], args.seed
    )
    results[CRC_TASK_NAME] = {"baseline": crc_base_metrics, "biased": crc_bias_metrics}

    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_base_pred, age_groups, "age_band", CRC_TASK_NAME, "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_base_pred, income_groups, "income_band", CRC_TASK_NAME, "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_base_pred, age_income_groups, "age_x_income", CRC_TASK_NAME, "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_bias_pred, age_groups, "age_band", CRC_TASK_NAME, "biased"))
    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_bias_pred, income_groups, "income_band", CRC_TASK_NAME, "biased"))
    subgroup_frames.append(compute_subgroup_metrics(y_crc_true[idx_test], crc_bias_pred, age_income_groups, "age_x_income", CRC_TASK_NAME, "biased"))

    # Task 2: Screening recommendation to catch early-stage CRC
    early_base_metrics, early_base_pred = train_and_eval(
        X[idx_train], y_early_true[idx_train], X[idx_val], y_early_true[idx_val], X[idx_test], y_early_true[idx_test], args.seed
    )
    early_bias_metrics, early_bias_pred = train_and_eval(
        X[idx_train], y_early_obs[idx_train], X[idx_val], y_early_obs[idx_val], X[idx_test], y_early_true[idx_test], args.seed
    )
    results["Early-Stage Screening Recommendation (Catch Early CRC)"] = {"baseline": early_base_metrics, "biased": early_bias_metrics}

    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_base_pred, age_groups, "age_band", "Early-Stage Screening Recommendation (Catch Early CRC)", "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_base_pred, income_groups, "income_quintile", "Early-Stage Screening Recommendation (Catch Early CRC)", "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_base_pred, age_income_groups, "age_x_income", "Early-Stage Screening Recommendation (Catch Early CRC)", "baseline"))
    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_bias_pred, age_groups, "age_band", "Early-Stage Screening Recommendation (Catch Early CRC)", "biased"))
    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_bias_pred, income_groups, "income_quintile", "Early-Stage Screening Recommendation (Catch Early CRC)", "biased"))
    subgroup_frames.append(compute_subgroup_metrics(y_early_true[idx_test], early_bias_pred, age_income_groups, "age_x_income", "Early-Stage Screening Recommendation (Catch Early CRC)", "biased"))

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
        "- Policy objective: recommend individuals for CRC screening based on predicted risk.",
        "- Secondary objective: maximize capture of early-stage CRC cases.",
        f"- Samples: {len(df):,}",
        f"- Features: {', '.join(feature_names)}",
        "- Excluded from model features for equity: income, assigned_plan, eligible_for_screening",
        "- Historical-bias simulation: observed labels reflect access barriers that disproportionately affect lower-income groups.",
        f"- Train size: {len(idx_train):,}",
        f"- Validation size: {len(idx_val):,}",
        f"- Test size: {len(idx_test):,}",
        f"- Seed: {args.seed}",
        "",
    ]

    summary.extend(
        [
            "## Aggregate Performance Difference",
            "",
            aggregate_delta_table(results),
            "",
            "## Income Subgroup Performance Difference (CRC Screening Recommendation)",
            "",
            subgroup_delta_table(subgroup_df, "income_band", task_name=CRC_TASK_NAME),
            "",
            "## Age Subgroup Performance Difference (CRC Screening Recommendation)",
            "",
            subgroup_delta_table(subgroup_df, "age_band", task_name=CRC_TASK_NAME),
            "",
        ]
    )

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
