#!/usr/bin/env python3
"""Train baseline and biased CRC screening recommendation models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"

FEATURES = [
    "age",
    "male",
    "smoker",
    "bmi",
    "obesity",
    "type2_diabetes",
    "hypertension",
    "hyperlipidemia",
    "ibd",
    "ambulatory_visits_last2y",
    "preventive_visit_last2y",
    "comorbidity_count",
]


def select_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    candidates = np.linspace(0.1, 0.9, 81)
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        pred = (prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def evaluate(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "auc": safe_auc(y_true, prob),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def train_variant(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test_true: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model = GradientBoostingClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=3,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_prob)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    metrics = evaluate(y_test_true, test_prob, threshold)
    metrics["threshold"] = threshold
    return metrics, test_prob, test_pred


def subgroup_metrics(df: pd.DataFrame, pred_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_values, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        y = sub["true_screened_in_last_5y"].to_numpy()
        p = sub[pred_col].to_numpy()
        rows.append(
            {
                **{group_cols[i]: group_values[i] for i in range(len(group_cols))},
                "n": len(sub),
                "true_rate": float(y.mean()) if len(y) else float("nan"),
                "pred_rate": float(p.mean()) if len(p) else float("nan"),
                "recall": float(recall_score(y, p, zero_division=0)),
                "f1": float(f1_score(y, p, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    seed = 160
    INFO_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "data.csv")
    X = df[FEATURES].copy()

    y_true = df["true_screened_in_last_5y"].astype(int).to_numpy()
    y_obs = df["observed_screened_in_last_5y"].astype(int).to_numpy()

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y_true)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=seed, stratify=y_true[train_idx])

    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]

    baseline_metrics, _, baseline_pred = train_variant(
        X_train,
        y_true[train_idx],
        X_val,
        y_true[val_idx],
        X_test,
        y_true[test_idx],
        seed,
    )

    biased_metrics, _, biased_pred = train_variant(
        X_train,
        y_obs[train_idx],
        X_val,
        y_obs[val_idx],
        X_test,
        y_true[test_idx],
        seed,
    )

    test_df = df.iloc[test_idx].copy()
    test_df["baseline_pred"] = baseline_pred
    test_df["biased_pred"] = biased_pred

    age_metrics = subgroup_metrics(test_df, "biased_pred", ["age_band"])
    inc_metrics = subgroup_metrics(test_df, "biased_pred", ["income_band"])
    inter_metrics = subgroup_metrics(test_df, "biased_pred", ["age_band", "income_band"])

    overall = pd.DataFrame([
        {"model": "baseline", **baseline_metrics},
        {"model": "biased", **biased_metrics},
    ])

    report = f"""# CRC Model Results

## Overall (evaluated against true_screened_in_last_5y)

{overall.to_markdown(index=False)}

## Biased model subgroup performance by age

{age_metrics.sort_values(['age_band']).to_markdown(index=False)}

## Biased model subgroup performance by income

{inc_metrics.sort_values(['income_band']).to_markdown(index=False)}

## Biased model subgroup performance by age x income

{inter_metrics.sort_values(['age_band', 'income_band']).to_markdown(index=False)}
"""

    (INFO_DIR / "3_model.md").write_text(report)
    print(f"Wrote {INFO_DIR / '3_model.md'}")


if __name__ == "__main__":
    main()
