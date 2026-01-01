#!/usr/bin/env python3
"""Analysis helpers for sleep apnea modeling outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Collection, Dict, Iterable, List, Sequence, Tuple, TYPE_CHECKING

from utils import compute_population_stats, load_sdoh_urban_map

if TYPE_CHECKING:
    from scripts.sleep_apnea.main import Dataset, Split


@dataclass
class AnalysisResults:
    """Computed metrics and summaries used in report generation."""

    baseline_summary: Dict[str, float]
    biased_summary: Dict[str, float]
    baseline_test_metrics: Dict[str, float]
    biased_test_metrics: Dict[str, float]
    biased_on_baseline_metrics: Dict[str, float]
    baseline_on_biased_metrics: Dict[str, float]
    baseline_bias: Dict[str, float]
    biased_bias: Dict[str, float]
    importance_table: List[Tuple[str, float, float]]
    baseline_population: Dict[str, float]
    biased_population: Dict[str, float]
    baseline_split_sizes: Tuple[int, int, int]
    biased_split_sizes: Tuple[int, int, int]


def evaluate_predictions(y_true: List[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Compute MAE, RMSE, and R2 for a set of predictions."""
    y_pred_list = list(y_pred)
    n = len(y_true)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    abs_err = [abs(p - t) for p, t in zip(y_pred_list, y_true)]
    sq_err = [(p - t) ** 2 for p, t in zip(y_pred_list, y_true)]
    mae = sum(abs_err) / n
    rmse = (sum(sq_err) / n) ** 0.5
    mean_true = sum(y_true) / n
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    ss_res = sum(sq_err)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def summarize_dataset(dataset: Dataset) -> Dict[str, float]:
    """Summarize dataset size, spend totals, and nonzero rate."""
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


def build_importance_table(
    baseline_model: Any,
    biased_model: Any,
    feature_names: Sequence[str],
) -> List[Tuple[str, float, float]]:
    """Build an ordered list of feature importances for both models."""
    baseline_importances = dict(zip(feature_names, baseline_model.feature_importances_))
    biased_importances = dict(zip(feature_names, biased_model.feature_importances_))
    ordered = sorted(feature_names, key=lambda name: baseline_importances.get(name, 0.0), reverse=True)
    return [
        (name, baseline_importances.get(name, 0.0), biased_importances.get(name, 0.0))
        for name in ordered
    ]


def analyze_models(
    baseline: Dataset,
    biased: Dataset,
    split_baseline: Split,
    split_biased: Split,
    baseline_model: Any,
    biased_model: Any,
    sleep_disorder_code: str,
    sleep_apnea_codes: Collection[str],
    log_spend: bool = False,
) -> AnalysisResults:
    """Compute evaluation metrics, feature importances, and population stats."""
    def _to_raw(preds: Iterable[float]) -> List[float]:
        pred_list = list(preds)
        if not log_spend:
            return pred_list
        return [max(math.expm1(value), 0.0) for value in pred_list]

    baseline_summary = summarize_dataset(baseline)
    biased_summary = summarize_dataset(biased)

    baseline_test_preds = _to_raw(baseline_model.predict(split_baseline.X_test))
    biased_test_preds = _to_raw(biased_model.predict(split_biased.X_test))

    baseline_test_metrics = evaluate_predictions(split_baseline.y_test, baseline_test_preds)
    biased_test_metrics = evaluate_predictions(split_biased.y_test, biased_test_preds)

    # Cross-dataset generalization checks.
    biased_on_baseline_preds = _to_raw(biased_model.predict(split_baseline.X_test))
    baseline_on_biased_preds = _to_raw(baseline_model.predict(split_biased.X_test))

    biased_on_baseline_metrics = evaluate_predictions(split_baseline.y_test, biased_on_baseline_preds)
    baseline_on_biased_metrics = evaluate_predictions(split_biased.y_test, baseline_on_biased_preds)

    def _bias_summary(y_true: List[float], y_pred: Iterable[float]) -> Dict[str, float]:
        """Summarize mean prediction bias vs. the true mean."""
        y_pred_list = list(y_pred)
        if len(y_true) == 0 or len(y_pred_list) == 0:
            return {"mean_true": 0.0, "mean_pred": 0.0, "diff": 0.0, "rel": float("nan")}
        mean_true = sum(y_true) / len(y_true)
        mean_pred = sum(float(value) for value in y_pred_list) / len(y_pred_list)
        diff = mean_pred - mean_true
        rel = diff / mean_true if mean_true != 0 else float("nan")
        return {"mean_true": mean_true, "mean_pred": mean_pred, "diff": diff, "rel": rel}

    baseline_bias = _bias_summary(split_baseline.y_test, baseline_test_preds)
    biased_bias = _bias_summary(split_baseline.y_test, biased_on_baseline_preds)

    urban_map = load_sdoh_urban_map()
    baseline_population = compute_population_stats(
        baseline.csv_dir, urban_map, sleep_disorder_code, sleep_apnea_codes
    )
    biased_population = compute_population_stats(
        biased.csv_dir, urban_map, sleep_disorder_code, sleep_apnea_codes
    )

    importance_table = build_importance_table(
        baseline_model, biased_model, baseline.feature_names
    )

    return AnalysisResults(
        baseline_summary=baseline_summary,
        biased_summary=biased_summary,
        baseline_test_metrics=baseline_test_metrics,
        biased_test_metrics=biased_test_metrics,
        biased_on_baseline_metrics=biased_on_baseline_metrics,
        baseline_on_biased_metrics=baseline_on_biased_metrics,
        baseline_bias=baseline_bias,
        biased_bias=biased_bias,
        importance_table=importance_table,
        baseline_population=baseline_population,
        biased_population=biased_population,
        baseline_split_sizes=(
            len(split_baseline.y_train),
            len(split_baseline.y_val),
            len(split_baseline.y_test),
        ),
        biased_split_sizes=(
            len(split_biased.y_train),
            len(split_biased.y_val),
            len(split_biased.y_test),
        ),
    )
