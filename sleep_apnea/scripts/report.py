#!/usr/bin/env python3
"""Generate a comprehensive sleep apnea case study report combining model and analytics results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, confusion_matrix

from analytics import (
    Dataset as AnalyticsDataset,
    PairwiseStats,
    RegressionStats,
    SummaryStats,
    compute_summary_stats,
    load_dataset as load_analytics_dataset,
    pairwise_comparison,
    regression_analysis,
)
from models import (
    CrossEvaluation,
    Dataset as ModelDataset,
    ModelResult,
    SplitData,
    load_dataset as load_model_dataset,
    save_models,
    split_dataset,
    train_models,
    _evaluate,
)


@dataclass
class SubgroupMetrics:
    """Metrics for a single subgroup."""
    n: int
    n_positive: int
    auc: Optional[float]
    auc_ci: Optional[Tuple[float, float]]  # 95% bootstrap CI
    ap: Optional[float]
    fpr: Optional[float]  # False positive rate
    fnr: Optional[float]  # False negative rate (miss rate)
    fnr_ci: Optional[Tuple[float, float]]  # 95% Wilson CI
    tpr: Optional[float]  # True positive rate (recall/sensitivity)
    precision: Optional[float]
    mean_prediction: Optional[float]
    mean_prediction_ci: Optional[Tuple[float, float]]  # 95% CI
    calibration_error: Optional[float]  # Expected calibration error
    threshold: Optional[float]  # Classification threshold used


@dataclass
class FairnessMetrics:
    """Fairness metrics comparing subgroups."""
    model_name: str
    dataset_name: str
    rural: SubgroupMetrics
    urban: SubgroupMetrics
    auc_gap: Optional[float]  # rural AUC - urban AUC
    fnr_gap: Optional[float]  # rural FNR - urban FNR (positive = rural worse)
    tpr_gap: Optional[float]  # rural TPR - urban TPR (negative = rural worse)
    prediction_gap: Optional[float]  # rural mean pred - urban mean pred


def _compute_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find classification threshold maximizing Youden's J (sensitivity + specificity - 1)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr_vals - fpr_vals
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def _bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Optional[Tuple[float, float]]:
    """Compute 95% bootstrap CI for AUC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true[idx], y_prob[idx])))
    if len(aucs) < 100:
        return None
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def _wilson_ci(
    successes: int,
    n: int,
    z: float = 1.96,
) -> Optional[Tuple[float, float]]:
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return None
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _mean_ci(
    values: np.ndarray,
    z: float = 1.96,
) -> Optional[Tuple[float, float]]:
    """Normal-approximation 95% CI for a mean."""
    n = len(values)
    if n < 2:
        return None
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    return (mean - z * se, mean + z * se)


def _compute_subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> SubgroupMetrics:
    """Compute metrics for a single subgroup."""
    n = len(y_true)
    n_positive = int(y_true.sum())

    if n == 0:
        return SubgroupMetrics(
            n=0, n_positive=0, auc=None, auc_ci=None, ap=None,
            fpr=None, fnr=None, fnr_ci=None, tpr=None, precision=None,
            mean_prediction=None, mean_prediction_ci=None,
            calibration_error=None, threshold=threshold,
        )

    mean_prediction = float(y_prob.mean())
    mean_prediction_ci = _mean_ci(y_prob)

    # AUC and AP require both classes
    if n_positive == 0 or n_positive == n:
        auc = None
        auc_ci = None
        ap = None
    else:
        auc = float(roc_auc_score(y_true, y_prob))
        auc_ci = _bootstrap_auc_ci(y_true, y_prob)
        ap = float(average_precision_score(y_true, y_prob))

    # Confusion matrix metrics at the provided threshold
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else None
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else None
    fnr_ci = _wilson_ci(int(fn), int(fn + tp)) if (fn + tp) > 0 else None
    tpr = float(tp / (fn + tp)) if (fn + tp) > 0 else None
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None

    # Calibration error (simplified ECE)
    try:
        if n_positive > 0 and n_positive < n:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy='uniform')
            calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))
        else:
            calibration_error = None
    except Exception:
        calibration_error = None

    return SubgroupMetrics(
        n=n,
        n_positive=n_positive,
        auc=auc,
        auc_ci=auc_ci,
        ap=ap,
        fpr=fpr,
        fnr=fnr,
        fnr_ci=fnr_ci,
        tpr=tpr,
        precision=precision,
        mean_prediction=mean_prediction,
        mean_prediction_ci=mean_prediction_ci,
        calibration_error=calibration_error,
        threshold=threshold,
    )


def compute_fairness_metrics(
    model_name: str,
    dataset_name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    rural_indicator: pd.Series,
    threshold: Optional[float] = None,
) -> FairnessMetrics:
    """Compute fairness metrics comparing rural vs urban subgroups.

    If *threshold* is None the optimal threshold (Youden's J) is computed
    on the full test set and then applied to each subgroup.
    """
    y_prob = estimator.predict_proba(X)[:, 1]

    if threshold is None:
        threshold = _compute_optimal_threshold(y.values, y_prob)

    # Split by rural/urban
    rural_mask = rural_indicator == 1
    urban_mask = rural_indicator == 0

    rural_metrics = _compute_subgroup_metrics(
        y[rural_mask].values,
        y_prob[rural_mask],
        threshold,
    )
    urban_metrics = _compute_subgroup_metrics(
        y[urban_mask].values,
        y_prob[urban_mask],
        threshold,
    )

    # Compute gaps (positive = rural disadvantaged for FNR, negative for TPR/AUC)
    auc_gap = None
    if rural_metrics.auc is not None and urban_metrics.auc is not None:
        auc_gap = rural_metrics.auc - urban_metrics.auc

    fnr_gap = None
    if rural_metrics.fnr is not None and urban_metrics.fnr is not None:
        fnr_gap = rural_metrics.fnr - urban_metrics.fnr

    tpr_gap = None
    if rural_metrics.tpr is not None and urban_metrics.tpr is not None:
        tpr_gap = rural_metrics.tpr - urban_metrics.tpr

    prediction_gap = None
    if rural_metrics.mean_prediction is not None and urban_metrics.mean_prediction is not None:
        prediction_gap = rural_metrics.mean_prediction - urban_metrics.mean_prediction

    return FairnessMetrics(
        model_name=model_name,
        dataset_name=dataset_name,
        rural=rural_metrics,
        urban=urban_metrics,
        auc_gap=auc_gap,
        fnr_gap=fnr_gap,
        tpr_gap=tpr_gap,
        prediction_gap=prediction_gap,
    )


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.1f}"
    return f"{mean:.1f} ± {std:.1f}"


def _compute_population_stats(dataset: AnalyticsDataset) -> Dict[str, SummaryStats]:
    """Compute summary stats for urban and rural subsets of full population."""
    if dataset.full_population is None:
        return {}
    pop = dataset.full_population
    urban_mask = pop["rural"] == 0
    rural_mask = pop["rural"] == 1
    return {
        "urban": compute_summary_stats(pop, urban_mask),
        "rural": compute_summary_stats(pop, rural_mask),
    }


def write_comprehensive_report(
    path: Path,
    # Model data
    model_datasets: List[ModelDataset],
    model_results: List[ModelResult],
    train_frac: float,
    val_frac: float,
    cross_results: List[CrossEvaluation],
    cross_note: str,
    cross_test_size: int,
    # Analytics data
    analytics_datasets: List[AnalyticsDataset],
    pairwise: Dict[str, PairwiseStats],
    regressions: Dict[str, RegressionStats],
    population_stats: Dict[str, Dict[str, SummaryStats]],
    n_perm: int,
    # Fairness data
    fairness_metrics: Optional[List[FairnessMetrics]] = None,
) -> None:
    """Write a comprehensive markdown report combining all analyses."""
    lines: List[str] = []

    # ==========================================================================
    # TITLE AND EXECUTIVE SUMMARY
    # ==========================================================================
    lines.append("# Sleep Apnea Case Study: Rural Access Bias")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This case study demonstrates how healthcare access disparities can introduce systematic bias "
        "into clinical datasets and downstream machine learning models. Using Synthea, we generated "
        "two synthetic patient populations: a **baseline** dataset where all patients have equal access "
        "to care, and a **biased** dataset where rural patients face barriers that cause them to drop "
        "out of the sleep apnea care pathway before receiving a diagnosis."
    )
    lines.append("")

    # Key findings summary
    biased_pairwise = pairwise.get("biased")
    biased_reg = regressions.get("biased")
    if biased_pairwise and biased_reg:
        lines.append("**Key Findings:**")
        lines.append("")
        if biased_pairwise.rural_rate is not None and biased_pairwise.urban_rate is not None:
            lines.append(
                f"- Rural underdiagnosis rate in biased dataset: **{_format_pct(biased_pairwise.rural_rate)}** "
                f"vs urban rate of **{_format_pct(biased_pairwise.urban_rate)}**"
            )
        if biased_reg.rural_odds_ratio is not None:
            lines.append(
                f"- Adjusted odds ratio for rural underdiagnosis: **{_format_metric(biased_reg.rural_odds_ratio)}** "
                f"(p={_format_metric(biased_reg.rural_p_value)})"
            )
        if cross_results:
            avg_delta_auc = sum(r.deltas["auc"] for r in cross_results) / len(cross_results)
            if avg_delta_auc < -0.01:
                lines.append(
                    "- Models trained on biased data show degraded performance when evaluated on baseline population"
                )
            else:
                lines.append(
                    "- Cross-dataset evaluation reveals how biased training data interacts with model generalization"
                )
        lines.append("")

    # ==========================================================================
    # BACKGROUND
    # ==========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 1. Background")
    lines.append("")

    lines.append("### 1.1 Sleep Apnea Overview")
    lines.append("")
    lines.append(
        "Sleep apnea is a common sleep disorder characterized by repeated interruption of breathing "
        "during sleep. The Synthea `sleep_apnea.json` module models the clinical pathway:"
    )
    lines.append("")
    lines.append("1. **Risk Assessment**: Patients aged 30-60 are evaluated based on BMI, smoking, alcohol use, and CHF")
    lines.append("2. **Initial Presentation**: Symptoms include loud snoring and excessive daytime sleepiness")
    lines.append("3. **Referral**: Primary care refers to sleep specialist")
    lines.append("4. **Diagnostic Testing**: Home sleep study or in-lab polysomnography")
    lines.append("5. **Treatment**: CPAP therapy or oral appliances with ongoing follow-up")
    lines.append("")

    lines.append("### 1.2 The Access Bias Problem")
    lines.append("")
    lines.append(
        "In real-world healthcare, rural patients often face barriers to specialty care including:"
    )
    lines.append("")
    lines.append("- Longer travel distances to sleep centers")
    lines.append("- Fewer available sleep specialists")
    lines.append("- Difficulty scheduling follow-up appointments")
    lines.append("- Higher costs of repeated travel")
    lines.append("")
    lines.append(
        "These barriers cause rural patients to \"drop out\" of the care pathway before receiving "
        "diagnosis and treatment, leading to **underdiagnosis** and **undertreatment** of sleep apnea "
        "in rural populations."
    )
    lines.append("")

    lines.append("### 1.3 Bias Simulation")
    lines.append("")
    lines.append(
        "We simulate this access disparity using Synthea's module override mechanism. The biased dataset "
        "modifies two transition points in the sleep apnea care pathway:"
    )
    lines.append("")
    lines.append("| Transition Point | Baseline (Rural) | Biased (Rural) |")
    lines.append("| --- | :---: | :---: |")
    lines.append("| Wait Until Overnight Study | 100% continue | 20% continue, **80% drop out** |")
    lines.append("| Appointment Delay | 100% continue | 20% continue, **80% drop out** |")
    lines.append("")
    lines.append(
        "Urban patients continue to receive full care in both datasets. This creates a scenario where "
        "the biased dataset reflects lower sleep apnea diagnosis rates in rural populations despite "
        "equivalent underlying disease prevalence."
    )
    lines.append("")

    # ==========================================================================
    # DATA SUMMARY
    # ==========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 2. Data Summary")
    lines.append("")

    lines.append("### 2.1 Dataset Overview")
    lines.append("")
    lines.append("| Dataset | Total Patients | Sleep Apnea Prevalence | Description |")
    lines.append("| --- | ---: | ---: | --- |")
    for dataset in model_datasets:
        desc = "Equal access for all patients" if dataset.name == "baseline" else "80% rural dropout at diagnostic stages"
        lines.append(
            f"| {dataset.name} | {len(dataset.labels):,} | {_format_pct(dataset.prevalence)} | {desc} |"
        )
    lines.append("")

    # Note the prevalence difference
    baseline_ds = next((d for d in model_datasets if d.name == "baseline"), None)
    biased_ds = next((d for d in model_datasets if d.name == "biased"), None)
    if baseline_ds and biased_ds:
        prev_diff = baseline_ds.prevalence - biased_ds.prevalence
        lines.append(
            f"The biased dataset shows a **{_format_pct(prev_diff)} lower** sleep apnea prevalence "
            f"({_format_pct(biased_ds.prevalence)} vs {_format_pct(baseline_ds.prevalence)}), "
            "reflecting the rural underdiagnosis effect."
        )
        lines.append("")

    lines.append("### 2.2 Population Characteristics by Residence")
    lines.append("")
    lines.append(
        "Summary statistics for the full patient population, stratified by urban/rural residence. "
        "Note that demographic characteristics are identical across datasets—only diagnosis rates differ."
    )
    lines.append("")

    # Build population stats table
    col_headers = []
    for dataset in analytics_datasets:
        col_headers.extend([f"{dataset.name} Urban", f"{dataset.name} Rural"])
    header_row = "| Characteristic | " + " | ".join(col_headers) + " |"
    separator_row = "| --- |" + " ---: |" * len(col_headers)
    lines.append(header_row)
    lines.append(separator_row)

    def get_stat(dataset_name: str, residence: str) -> SummaryStats:
        return population_stats.get(dataset_name, {}).get(residence, SummaryStats(
            n=0, age_mean=None, age_std=None, male_pct=None,
            income_mean=None, income_std=None, bmi_mean=None, bmi_std=None,
            smoker_pct=None, alcohol_pct=None, hypertension_pct=None, chf_pct=None,
        ))

    # N row
    n_cells = []
    for dataset in analytics_datasets:
        n_cells.append(f"{get_stat(dataset.name, 'urban').n:,}")
        n_cells.append(f"{get_stat(dataset.name, 'rural').n:,}")
    lines.append("| N | " + " | ".join(n_cells) + " |")

    # Age row
    age_cells = []
    for dataset in analytics_datasets:
        s = get_stat(dataset.name, "urban")
        age_cells.append(_format_mean_std(s.age_mean, s.age_std))
        s = get_stat(dataset.name, "rural")
        age_cells.append(_format_mean_std(s.age_mean, s.age_std))
    lines.append("| Age (years) | " + " | ".join(age_cells) + " |")

    # Male % row
    male_cells = []
    for dataset in analytics_datasets:
        male_cells.append(_format_pct(get_stat(dataset.name, "urban").male_pct))
        male_cells.append(_format_pct(get_stat(dataset.name, "rural").male_pct))
    lines.append("| Male (%) | " + " | ".join(male_cells) + " |")

    # Income row
    income_cells = []
    for dataset in analytics_datasets:
        s = get_stat(dataset.name, "urban")
        income_cells.append(_format_mean_std(s.income_mean, s.income_std))
        s = get_stat(dataset.name, "rural")
        income_cells.append(_format_mean_std(s.income_mean, s.income_std))
    lines.append("| Income ($) | " + " | ".join(income_cells) + " |")

    # BMI row
    bmi_cells = []
    for dataset in analytics_datasets:
        s = get_stat(dataset.name, "urban")
        bmi_cells.append(_format_mean_std(s.bmi_mean, s.bmi_std))
        s = get_stat(dataset.name, "rural")
        bmi_cells.append(_format_mean_std(s.bmi_mean, s.bmi_std))
    lines.append("| BMI | " + " | ".join(bmi_cells) + " |")

    # Smoker % row
    smoker_cells = []
    for dataset in analytics_datasets:
        smoker_cells.append(_format_pct(get_stat(dataset.name, "urban").smoker_pct))
        smoker_cells.append(_format_pct(get_stat(dataset.name, "rural").smoker_pct))
    lines.append("| Current Smoker (%) | " + " | ".join(smoker_cells) + " |")

    # Alcohol use % row
    alcohol_cells = []
    for dataset in analytics_datasets:
        alcohol_cells.append(_format_pct(get_stat(dataset.name, "urban").alcohol_pct))
        alcohol_cells.append(_format_pct(get_stat(dataset.name, "rural").alcohol_pct))
    lines.append("| Alcohol Use Disorder (%) | " + " | ".join(alcohol_cells) + " |")

    # Hypertension % row
    hypertension_cells = []
    for dataset in analytics_datasets:
        hypertension_cells.append(_format_pct(get_stat(dataset.name, "urban").hypertension_pct))
        hypertension_cells.append(_format_pct(get_stat(dataset.name, "rural").hypertension_pct))
    lines.append("| Hypertension (%) | " + " | ".join(hypertension_cells) + " |")

    # CHF % row
    chf_cells = []
    for dataset in analytics_datasets:
        chf_cells.append(_format_pct(get_stat(dataset.name, "urban").chf_pct))
        chf_cells.append(_format_pct(get_stat(dataset.name, "rural").chf_pct))
    lines.append("| CHF (%) | " + " | ".join(chf_cells) + " |")
    lines.append("")

    # ==========================================================================
    # UNDERDIAGNOSIS ANALYSIS
    # ==========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 3. Rural Underdiagnosis Analysis")
    lines.append("")
    lines.append(
        "We define **underdiagnosis** as a patient who enters the sleep disorder care pathway "
        "(receives a sleep disorder diagnosis, SNOMED 39898005) but does not receive a sleep apnea "
        "diagnosis (SNOMED 73430006 or 78275009). This captures patients who \"drop out\" before "
        "completing diagnostic testing."
    )
    lines.append("")

    lines.append("### 3.1 Sleep Disorder Cohort Summary")
    lines.append("")
    lines.append("| Dataset | Cohort N | Sleep Apnea Dx | Underdiagnosed | Rural | Urban |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for dataset in analytics_datasets:
        lines.append(
            "| {name} | {n:,} | {apnea:,} | {dropout:,} | {rural:,} | {urban:,} |".format(
                name=dataset.name,
                n=dataset.cohort_size,
                apnea=dataset.sleep_apnea_count,
                dropout=dataset.dropout_count,
                rural=dataset.rural_count,
                urban=dataset.urban_count,
            )
        )
    lines.append("")

    lines.append("### 3.2 Pairwise Comparison (Rural vs Urban)")
    lines.append("")
    lines.append(
        "Underdiagnosis rates represent the proportion of sleep disorder patients who did not "
        "receive a sleep apnea diagnosis. P-values are from a two-proportion z-test."
    )
    lines.append("")
    lines.append("| Dataset | Rural N | Urban N | Rural Rate | Urban Rate | Risk Diff | Risk Ratio | z | p-value |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dataset in analytics_datasets:
        stats = pairwise[dataset.name]
        lines.append(
            "| {name} | {rural} | {urban} | {r_rate} | {u_rate} | {diff} | {ratio} | {z} | {p} |".format(
                name=dataset.name,
                rural=f"{stats.rural_n:,}",
                urban=f"{stats.urban_n:,}",
                r_rate=_format_pct(stats.rural_rate),
                u_rate=_format_pct(stats.urban_rate),
                diff=_format_pct(stats.risk_diff),
                ratio=_format_ratio(stats.risk_ratio),
                z=_format_metric(stats.z_value),
                p=_format_metric(stats.p_value),
            )
        )
    lines.append("")

    # Interpret pairwise results
    baseline_pair = pairwise.get("baseline")
    biased_pair = pairwise.get("biased")
    if baseline_pair and biased_pair:
        lines.append("**Interpretation:**")
        lines.append("")
        if baseline_pair.p_value is not None and baseline_pair.p_value > 0.05:
            lines.append(
                f"- **Baseline**: No significant difference in underdiagnosis rates between rural and urban "
                f"patients (p={_format_metric(baseline_pair.p_value)}), confirming equal access to care."
            )
        if biased_pair.p_value is not None and biased_pair.p_value < 0.05:
            lines.append(
                f"- **Biased**: Highly significant rural-urban disparity (p={_format_metric(biased_pair.p_value)}). "
                f"Rural patients are {_format_ratio(biased_pair.risk_ratio)}x more likely to be underdiagnosed."
            )
        lines.append("")

    lines.append("### 3.3 Adjusted Regression Analysis")
    lines.append("")
    lines.append(
        "Logistic regression models the probability of underdiagnosis as a function of clinical "
        "risk factors plus a rural indicator. This isolates the rural effect after controlling for "
        "potential confounders."
    )
    lines.append("")
    lines.append("**Covariates:** age, gender, income, BMI, smoking status, alcohol use, hypertension, CHF, rural indicator")
    lines.append("")
    lines.append(f"**Significance testing:** Permutation test with n={n_perm:,} permutations")
    lines.append("")
    lines.append("| Dataset | N | Rural Coef | Odds Ratio | Permutation p | In-sample AUC |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for dataset in analytics_datasets:
        stats = regressions[dataset.name]
        lines.append(
            "| {name} | {n:,} | {coef} | {odds} | {pval} | {auc} |".format(
                name=dataset.name,
                n=stats.n,
                coef=_format_metric(stats.rural_coef),
                odds=_format_metric(stats.rural_odds_ratio),
                pval=_format_metric(stats.rural_p_value),
                auc=_format_metric(stats.auc),
            )
        )
    lines.append("")

    # Interpret regression results
    baseline_reg = regressions.get("baseline")
    biased_reg = regressions.get("biased")
    if baseline_reg and biased_reg:
        lines.append("**Interpretation:**")
        lines.append("")
        if baseline_reg.rural_p_value is not None and baseline_reg.rural_p_value > 0.05:
            lines.append(
                f"- **Baseline**: Rural coefficient is not significant (p={_format_metric(baseline_reg.rural_p_value)}), "
                "indicating no access-driven underdiagnosis."
            )
        if biased_reg.rural_p_value is not None and biased_reg.rural_p_value < 0.05:
            lines.append(
                f"- **Biased**: Rural coefficient is highly significant (p={_format_metric(biased_reg.rural_p_value)}). "
                f"After adjusting for clinical factors, rural patients have **{_format_metric(biased_reg.rural_odds_ratio)}x "
                "the odds** of being underdiagnosed compared to urban patients."
            )
        lines.append("")

    # ==========================================================================
    # MODEL SPECIFICATION
    # ==========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 4. Predictive Model Specification")
    lines.append("")
    lines.append(
        "We train machine learning models to predict sleep apnea diagnosis from clinical features. "
        "Importantly, **urban/rural residence is excluded** from the feature set to examine how "
        "biased training data affects model performance independent of explicit location features."
    )
    lines.append("")

    lines.append("### 4.1 Features")
    lines.append("")
    lines.append("| Feature | Source | Description |")
    lines.append("| --- | --- | --- |")
    lines.append("| `age_years` | patients.csv | Patient age at reference date |")
    lines.append("| `male` | patients.csv | Gender indicator (1.0 for male) |")
    lines.append("| `income` | patients.csv | Annual household income |")
    lines.append("| `bmi` | observations.csv | Latest recorded BMI (LOINC 39156-5) |")
    lines.append("| `smoker` | observations.csv | Current smoking status (LOINC 72166-2) |")
    lines.append("| `alcohol_use` | conditions.csv | Alcohol use disorder (SNOMED 7200002) |")
    lines.append("| `hypertension` | conditions.csv | Hypertension diagnosis (SNOMED 59621000) |")
    lines.append("| `chf` | conditions.csv | Congestive heart failure (SNOMED 88805009) |")
    lines.append("")

    lines.append("### 4.2 Target Variable")
    lines.append("")
    lines.append(
        "Binary classification: patient has a sleep apnea diagnosis (SNOMED 73430006 or 78275009) in conditions.csv."
    )
    lines.append("")

    lines.append("### 4.3 Model Families")
    lines.append("")
    lines.append("| Model | Description |")
    lines.append("| --- | --- |")
    lines.append("| **Logistic Regression** | L2-regularized logistic regression with standardized inputs |")
    lines.append("| **Random Forest** | Ensemble of decision trees with bootstrap aggregation |")
    lines.append("| **Gradient Boosted DT** | Sequential boosting of shallow decision trees |")
    lines.append("")

    lines.append("### 4.4 Training Protocol")
    lines.append("")
    lines.append(f"- **Train/Validation/Test Split**: {train_frac:.0%}/{val_frac:.0%}/{1.0-train_frac-val_frac:.0%}")
    lines.append("- **Hyperparameter Selection**: Grid search optimizing validation AUC")
    lines.append("- **Final Training**: Re-fit on combined train+validation set")
    lines.append("- **Evaluation**: Held-out test set metrics")
    lines.append("")

    # ==========================================================================
    # MODEL RESULTS
    # ==========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 5. Model Performance Results")
    lines.append("")

    lines.append("### 5.1 Test Set Performance")
    lines.append("")
    lines.append("| Dataset | Model | AUC | Avg Precision | Brier Score | Train/Val/Test |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for result in model_results:
        train_n, val_n, test_n = result.split_sizes
        lines.append(
            "| {dataset} | {model} | {auc} | {ap} | {brier} | {split} |".format(
                dataset=result.dataset,
                model=result.model,
                auc=_format_metric(result.metrics["auc"]),
                ap=_format_metric(result.metrics["ap"]),
                brier=_format_metric(result.metrics["brier"]),
                split=f"{train_n}/{val_n}/{test_n}",
            )
        )
    lines.append("")

    lines.append("### 5.2 Selected Hyperparameters")
    lines.append("")
    for dataset in model_datasets:
        lines.append(f"**{dataset.name.title()}:**")
        lines.append("")
        subset = [r for r in model_results if r.dataset == dataset.name]
        for result in subset:
            params_str = ", ".join(f"{k}={v}" for k, v in result.params.items())
            lines.append(f"- *{result.model}*: {params_str}")
        lines.append("")

    # ==========================================================================
    # CROSS-DATASET EVALUATION
    # ==========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 6. Cross-Dataset Evaluation: Bias Impact on Model Generalization")
    lines.append("")
    lines.append(
        "To quantify how training on biased data affects real-world performance, we evaluate "
        "biased-trained models on the baseline test set. This simulates deploying a model trained "
        "on data with access disparities to a population with equitable care access."
    )
    lines.append("")
    lines.append(f"**Methodology:** {cross_note}")
    lines.append("")

    if cross_results:
        lines.append("### 6.1 Performance Comparison")
        lines.append("")
        lines.append("| Model | Baseline AUC | Biased AUC | Δ AUC | Baseline AP | Biased AP | Δ AP | Test N |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result in cross_results:
            lines.append(
                "| {model} | {b_auc} | {bi_auc} | {d_auc} | {b_ap} | {bi_ap} | {d_ap} | {n} |".format(
                    model=result.model,
                    b_auc=_format_metric(result.baseline_metrics["auc"]),
                    bi_auc=_format_metric(result.biased_metrics["auc"]),
                    d_auc="{:+.3f}".format(result.deltas["auc"]),
                    b_ap=_format_metric(result.baseline_metrics["ap"]),
                    bi_ap=_format_metric(result.biased_metrics["ap"]),
                    d_ap="{:+.3f}".format(result.deltas["ap"]),
                    n=f"{cross_test_size:,}",
                )
            )
        lines.append("")

        lines.append("### 6.2 Cross-Dataset Analysis")
        lines.append("")
        for result in cross_results:
            delta_auc = result.deltas["auc"]
            delta_ap = result.deltas["ap"]
            if delta_auc > 0.01:
                direction = "improved"
            elif delta_auc < -0.01:
                direction = "degraded"
            else:
                direction = "remained similar"
            lines.append(
                f"- **{result.model}**: AUC {direction} (Δ={delta_auc:+.3f}), AP Δ={delta_ap:+.3f}"
            )
        lines.append("")
        lines.append(
            "Note: The biased models were trained on labels where rural patients are underdiagnosed, "
            "but urban/rural residence is not an input feature.  Because the feature distributions "
            "are similar across rural and urban populations, the models cannot easily learn the "
            "rural-specific label bias, limiting the expected degradation effect."
        )
        lines.append("")
    else:
        lines.append(
            "*Cross-dataset evaluation was skipped because no eligible overlapping baseline test patients "
            "remained after filtering.*"
        )
        lines.append("")

    # ==========================================================================
    # FAIRNESS ANALYSIS
    # ==========================================================================
    if fairness_metrics:
        lines.append("---")
        lines.append("")
        lines.append("## 7. Fairness Analysis: Subgroup Performance Disparities")
        lines.append("")
        lines.append(
            "Aggregate metrics like AUC can mask disparities between subgroups. This section examines "
            "model performance separately for rural and urban patients to reveal how bias in training "
            "data translates to differential model behavior."
        )
        lines.append("")

        # Note threshold methodology
        lines.append(
            "**Classification threshold:** Instead of the default 0.5 (which produces 100% false-negative "
            "rates at low prevalence), the threshold is set per-model by maximizing Youden's J "
            "(sensitivity + specificity - 1) on the full test set, then applied to each subgroup."
        )
        lines.append("")

        # Group by dataset
        biased_fairness = [f for f in fairness_metrics if f.dataset_name == "biased"]

        # --- 7.1 AUC by Subgroup (with 95% bootstrap CI) ---
        lines.append("### 7.1 AUC by Subgroup")
        lines.append("")
        lines.append(
            "AUC measures discriminative ability — how well the model ranks positive cases above "
            "negative cases.  95% bootstrap confidence intervals (2 000 resamples) are shown in brackets."
        )
        lines.append("")
        lines.append("| Dataset | Model | Rural AUC [95% CI] | Urban AUC [95% CI] | Gap (R-U) | Rural N (pos) | Urban N (pos) |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for fm in fairness_metrics:
            gap_str = f"{fm.auc_gap:+.3f}" if fm.auc_gap is not None else "n/a"

            def _auc_cell(sm: SubgroupMetrics) -> str:
                if sm.auc is None:
                    return "n/a"
                s = f"{sm.auc:.3f}"
                if sm.auc_ci is not None:
                    s += f" [{sm.auc_ci[0]:.2f}, {sm.auc_ci[1]:.2f}]"
                return s

            lines.append(
                f"| {fm.dataset_name} | {fm.model_name} | "
                f"{_auc_cell(fm.rural)} | {_auc_cell(fm.urban)} | {gap_str} | "
                f"{fm.rural.n:,} ({fm.rural.n_positive}) | {fm.urban.n:,} ({fm.urban.n_positive}) |"
            )
        lines.append("")

        # --- 7.2 FNR by Subgroup (with Wilson CI) ---
        lines.append("### 7.2 False Negative Rate by Subgroup")
        lines.append("")
        lines.append(
            "The **false negative rate (FNR)** is the proportion of true positive cases that the model "
            "misses (predicts as negative).  A higher FNR for rural patients means more rural patients "
            "with sleep apnea are incorrectly told they don't have it.  95% Wilson score intervals "
            "are shown in brackets; wide intervals indicate small subgroup sample sizes."
        )
        lines.append("")
        lines.append("| Dataset | Model | Threshold | Rural FNR [95% CI] | Urban FNR [95% CI] | Gap (R-U) |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for fm in fairness_metrics:
            gap_str = f"{fm.fnr_gap:+.3f}" if fm.fnr_gap is not None else "n/a"
            thresh_str = f"{fm.rural.threshold:.3f}" if fm.rural.threshold is not None else "n/a"

            def _fnr_cell(sm: SubgroupMetrics) -> str:
                if sm.fnr is None:
                    return "n/a"
                s = _format_pct(sm.fnr)
                if sm.fnr_ci is not None:
                    s += f" [{sm.fnr_ci[0]*100:.1f}%, {sm.fnr_ci[1]*100:.1f}%]"
                return s

            lines.append(
                f"| {fm.dataset_name} | {fm.model_name} | {thresh_str} | "
                f"{_fnr_cell(fm.rural)} | {_fnr_cell(fm.urban)} | {gap_str} |"
            )
        lines.append("")

        # --- 7.3 Mean Predicted Probability (with CI) ---
        lines.append("### 7.3 Mean Predicted Probability by Subgroup")
        lines.append("")
        lines.append(
            "The mean predicted probability reveals systematic differences in how the model scores "
            "patients from different subgroups.  A lower mean prediction for rural patients indicates "
            "the model has learned to associate rural-correlated features with lower disease probability.  "
            "95% normal-approximation CIs are shown in brackets."
        )
        lines.append("")
        lines.append("| Dataset | Model | Rural Mean Pred [95% CI] | Urban Mean Pred [95% CI] | Gap (R-U) |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for fm in fairness_metrics:
            gap_str = f"{fm.prediction_gap:+.4f}" if fm.prediction_gap is not None else "n/a"

            def _pred_cell(sm: SubgroupMetrics) -> str:
                if sm.mean_prediction is None:
                    return "n/a"
                s = f"{sm.mean_prediction:.3f}"
                if sm.mean_prediction_ci is not None:
                    s += f" [{sm.mean_prediction_ci[0]:.4f}, {sm.mean_prediction_ci[1]:.4f}]"
                return s

            lines.append(
                f"| {fm.dataset_name} | {fm.model_name} | "
                f"{_pred_cell(fm.rural)} | {_pred_cell(fm.urban)} | {gap_str} |"
            )
        lines.append("")

        # --- 7.4 Interpretation ---
        lines.append("### 7.4 Fairness Interpretation")
        lines.append("")

        # Check if biased models show worse rural metrics
        if biased_fairness:
            has_fnr_disparity = any(
                f.fnr_gap is not None and f.fnr_gap > 0.05 for f in biased_fairness
            )
            has_pred_gap = any(
                f.prediction_gap is not None and f.prediction_gap < -0.01 for f in biased_fairness
            )

            if has_fnr_disparity or has_pred_gap:
                lines.append("**Key findings from fairness analysis:**")
                lines.append("")
                if has_pred_gap:
                    lines.append(
                        "- **Systematic underscoring of rural patients**: Models trained on biased data assign "
                        "lower predicted probabilities to rural patients, even though urban/rural is not an input feature. "
                        "This occurs because the model learns associations between rural-correlated features "
                        "(e.g., lower income) and the biased labels where rural patients are less likely to be diagnosed."
                    )
                    lines.append("")
                if has_fnr_disparity:
                    lines.append(
                        "- **Higher miss rate for rural patients**: The false negative rate gap shows that "
                        "rural patients with sleep apnea are more likely to be incorrectly classified as negative. "
                        "This perpetuates the underdiagnosis pattern present in the training data."
                    )
                    lines.append("")
                lines.append(
                    "These findings demonstrate that **aggregate metrics like AUC can hide fairness harms**. "
                    "A model can achieve good overall discrimination while systematically disadvantaging specific subgroups."
                )
                lines.append("")
            else:
                # Honest finding: bias didn't transmit strongly through models
                lines.append(
                    "The fairness metrics do not show strong evidence of systematic rural disadvantage "
                    "in model predictions.  Because urban/rural residence is excluded from the feature set "
                    "and rural vs urban patients have similar clinical feature distributions, the models have "
                    "limited ability to distinguish the two groups.  The bias in training labels (rural "
                    "underdiagnosis) therefore does not translate into large differential prediction errors.  "
                    "This is an important finding: **when bias is in labels but features are group-invariant, "
                    "standard ML models may not propagate the bias into predictions**.  However, the biased "
                    "training data still reduces overall model utility by providing fewer true positive "
                    "examples for the model to learn from."
                )
                lines.append("")

                # Note sample-size limitations
                biased_rural_pos = [f.rural.n_positive for f in biased_fairness]
                if any(p < 30 for p in biased_rural_pos):
                    lines.append(
                        "**Caution:** The biased test set contains very few rural positive cases "
                        f"({', '.join(str(p) for p in biased_rural_pos)} per model), so subgroup metrics "
                        "have wide confidence intervals.  Larger datasets would be needed for definitive "
                        "fairness conclusions."
                    )
                    lines.append("")

    # ==========================================================================
    # CONCLUSIONS
    # ==========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 8. Conclusions and Implications" if fairness_metrics else "## 7. Conclusions and Implications")
    lines.append("")

    section_num = "8" if fairness_metrics else "7"
    lines.append(f"### {section_num}.1 Key Findings")
    lines.append("")
    lines.append(
        "1. **Access barriers create measurable underdiagnosis**: The 80% dropout rate for rural "
        "patients at diagnostic stages produces a dramatic underdiagnosis disparity "
        f"({_format_pct(biased_pair.rural_rate if biased_pair else None)} rural vs "
        f"{_format_pct(biased_pair.urban_rate if biased_pair else None)} urban)."
    )
    lines.append("")
    lines.append(
        "2. **Bias persists after covariate adjustment**: The significant rural coefficient in the "
        "adjusted regression confirms that underdiagnosis is driven by access barriers, not "
        "differences in clinical risk factors."
    )
    lines.append("")
    lines.append(
        "3. **Label bias does not always transmit through models**: When the protected attribute "
        "(rural/urban) is excluded from features and the remaining features have similar distributions "
        "across groups, models may not learn group-specific biases from the corrupted labels. "
        "However, the biased data still reduces overall model utility by providing fewer true positive "
        "training examples."
    )
    lines.append("")

    lines.append(f"### {section_num}.2 Real-World Implications")
    lines.append("")
    lines.append(
        "- **Clinical Decision Support**: ML models for sleep apnea screening trained on data from "
        "healthcare systems with rural access barriers may underperform in settings with better "
        "rural care access."
    )
    lines.append("")
    lines.append(
        "- **Health Equity Monitoring**: Organizations should audit training data for geographic "
        "and socioeconomic disparities that could introduce systematic bias."
    )
    lines.append("")
    lines.append(
        "- **Policy Interventions**: Addressing rural healthcare access (telemedicine, mobile sleep "
        "clinics, transportation assistance) could reduce underdiagnosis and improve data quality "
        "for downstream applications."
    )
    lines.append("")

    lines.append(f"### {section_num}.3 Limitations")
    lines.append("")
    lines.append(
        "- Synthea generates synthetic data that may not capture all real-world complexities"
    )
    lines.append(
        "- The 80% dropout rate is illustrative; actual rural dropout rates vary by region and condition"
    )
    lines.append(
        "- Very few rural positive cases in the biased test set limit the power of subgroup fairness analysis"
    )
    lines.append("")

    # ==========================================================================
    # APPENDIX
    # ==========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## Appendix: Technical Details")
    lines.append("")

    lines.append("### A.1 SNOMED/LOINC Codes")
    lines.append("")
    lines.append("| Concept | Code System | Code |")
    lines.append("| --- | --- | --- |")
    lines.append("| Sleep Apnea | SNOMED | 73430006, 78275009 |")
    lines.append("| Sleep Disorder | SNOMED | 39898005 |")
    lines.append("| Hypertension | SNOMED | 59621000 |")
    lines.append("| CHF | SNOMED | 88805009 |")
    lines.append("| Alcohol Use Disorder | SNOMED | 7200002 |")
    lines.append("| BMI | LOINC | 39156-5 |")
    lines.append("| Smoking Status | LOINC | 72166-2 |")
    lines.append("")

    lines.append("### A.2 Data Generation Commands")
    lines.append("")
    lines.append("**Baseline:**")
    lines.append("```bash")
    lines.append("./run_synthea -s 160 -cs 160 -o false -p 20000 -a 30-100 \\")
    lines.append("  --exporter.csv.export=true \\")
    lines.append("  --exporter.csv.append_mode=false \\")
    lines.append("  --exporter.baseDirectory=./output_baseline \\")
    lines.append("  Montana")
    lines.append("```")
    lines.append("")
    lines.append("**Biased:**")
    lines.append("```bash")
    lines.append("./run_synthea -s 160 -cs 160 -o false -p 20000 -a 30-100 \\")
    lines.append("  --exporter.csv.export=true \\")
    lines.append("  --exporter.csv.append_mode=false \\")
    lines.append("  --exporter.baseDirectory=./output_rural_bias \\")
    lines.append("  --module_override=./config/overrides_rural_sleep_apnea.properties \\")
    lines.append("  Montana")
    lines.append("```")
    lines.append("")
    lines.append("**Note:** The `-a 30-100` flag restricts the population to ages 30-100, matching")
    lines.append("the sleep apnea module's risk assessment age range.")
    lines.append("")

    # Write the report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate comprehensive sleep apnea case study report."
    )
    base_dir = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--baseline",
        default=str(base_dir / "data" / "baseline"),
        help="Baseline CSV directory.",
    )
    parser.add_argument(
        "--biased",
        default=str(base_dir / "data" / "biased"),
        help="Biased CSV directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Training fraction.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation fraction.",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=500,
        help="Permutation count for rural coefficient significance.",
    )
    parser.add_argument(
        "--out",
        default=str(base_dir / "output" / "sleep_apnea_report.md"),
        help="Output markdown path.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(base_dir / "output" / "models"),
        help="Directory to store trained model artifacts.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model training (use for faster report generation if models already exist).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("Sleep Apnea Case Study: Comprehensive Report Generator")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Load datasets for both analytics and models
    # -------------------------------------------------------------------------
    print("[1/5] Loading datasets...")

    # Analytics datasets (with rural/urban and full population)
    analytics_baseline = load_analytics_dataset("baseline", Path(args.baseline))
    analytics_biased = load_analytics_dataset("biased", Path(args.biased))
    analytics_datasets = [analytics_baseline, analytics_biased]

    # Model datasets (for prediction)
    model_baseline = load_model_dataset("baseline", Path(args.baseline))
    model_biased = load_model_dataset("biased", Path(args.biased))
    model_datasets = [model_baseline, model_biased]

    print(f"  - Baseline: {len(model_baseline.labels):,} patients, prevalence={model_baseline.prevalence:.2%}")
    print(f"  - Biased: {len(model_biased.labels):,} patients, prevalence={model_biased.prevalence:.2%}")
    print()

    # -------------------------------------------------------------------------
    # Analytics: Pairwise and regression analysis
    # -------------------------------------------------------------------------
    print("[2/5] Running underdiagnosis analysis...")

    pairwise_stats = {
        analytics_baseline.name: pairwise_comparison(analytics_baseline),
        analytics_biased.name: pairwise_comparison(analytics_biased),
    }

    regression_stats = {
        analytics_baseline.name: regression_analysis(analytics_baseline, args.seed, args.n_perm),
        analytics_biased.name: regression_analysis(analytics_biased, args.seed, args.n_perm),
    }

    population_stats = {
        analytics_baseline.name: _compute_population_stats(analytics_baseline),
        analytics_biased.name: _compute_population_stats(analytics_biased),
    }

    biased_pair = pairwise_stats["biased"]
    print(f"  - Biased rural underdiagnosis rate: {biased_pair.rural_rate:.1%}")
    print(f"  - Biased urban underdiagnosis rate: {biased_pair.urban_rate:.1%}")
    print()

    # -------------------------------------------------------------------------
    # Model training
    # -------------------------------------------------------------------------
    model_results: List[ModelResult] = []
    cross_results: List[CrossEvaluation] = []
    cross_note = ""
    cross_test_size = 0

    baseline_split = None
    biased_split = None

    if args.skip_models:
        print("[3/5] Skipping model training (--skip-models)")
        print()
    else:
        print("[3/5] Training prediction models...")

        # Split baseline dataset first
        baseline_split = split_dataset(model_baseline, args.train_frac, args.val_frac, args.seed)

        # Use the SAME patient-ID split for biased dataset so cross-evaluation
        # can use the full test set.  Both datasets share the same patients
        # (same Synthea seed), so we align by index.
        biased_split = SplitData(
            X_train=model_biased.features.loc[baseline_split.X_train.index],
            X_val=model_biased.features.loc[baseline_split.X_val.index],
            X_test=model_biased.features.loc[baseline_split.X_test.index],
            y_train=model_biased.labels.loc[baseline_split.X_train.index],
            y_val=model_biased.labels.loc[baseline_split.X_val.index],
            y_test=model_biased.labels.loc[baseline_split.X_test.index],
        )

        # Train models
        model_results.extend(
            train_models(model_baseline, args.seed, args.train_frac, args.val_frac, split=baseline_split)
        )
        model_results.extend(
            train_models(model_biased, args.seed, args.train_frac, args.val_frac, split=biased_split)
        )

        print(f"  - Trained {len(model_results)} models")

        # Cross-dataset evaluation: score the baseline test set (ground-truth
        # labels) with both baseline-trained and biased-trained models.
        print("[4/5] Running cross-dataset evaluation...")

        baseline_models = {r.model: r.estimator for r in model_results if r.dataset == "baseline"}
        biased_models = {r.model: r.estimator for r in model_results if r.dataset == "biased"}

        X_cross = baseline_split.X_test
        y_cross = baseline_split.y_test
        cross_test_size = len(y_cross)
        cross_note = (
            f"Both datasets share the same patients (same Synthea seed).  A single "
            f"patient-ID split is used so that the biased models are never evaluated "
            f"on patients they trained on.  Test N={cross_test_size:,}; "
            f"positives={int(y_cross.sum()):,}."
        )

        for model_name, biased_model in biased_models.items():
            baseline_model = baseline_models.get(model_name)
            if baseline_model is None:
                continue
            baseline_probs = baseline_model.predict_proba(X_cross)[:, 1].tolist()
            biased_probs = biased_model.predict_proba(X_cross)[:, 1].tolist()
            baseline_metrics = _evaluate(y_cross, baseline_probs)
            biased_metrics = _evaluate(y_cross, biased_probs)
            deltas = {key: biased_metrics[key] - baseline_metrics[key] for key in baseline_metrics}
            cross_results.append(
                CrossEvaluation(
                    model=model_name,
                    baseline_metrics=baseline_metrics,
                    biased_metrics=biased_metrics,
                    deltas=deltas,
                )
            )

        print(f"  - Cross-evaluation test size: {cross_test_size}")

        # Save models
        save_models(Path(args.model_dir), model_results)
        print(f"  - Saved models to {args.model_dir}")
        print()

    # -------------------------------------------------------------------------
    # Fairness analysis
    # -------------------------------------------------------------------------
    fairness_results: List[FairnessMetrics] = []

    if model_results and baseline_split is not None and biased_split is not None:
        print("[4.5/5] Computing fairness metrics...")

        # Get rural indicator from analytics datasets
        baseline_rural = analytics_baseline.full_population["rural"] if analytics_baseline.full_population is not None else None
        biased_rural = analytics_biased.full_population["rural"] if analytics_biased.full_population is not None else None

        # Compute fairness metrics for each model on its own test set.
        # Threshold is computed per-model via Youden's J on the full test set.
        for result in model_results:
            if result.dataset == "baseline" and baseline_rural is not None:
                test_ids = baseline_split.X_test.index
                rural_indicator = baseline_rural.reindex(test_ids)
                valid_mask = rural_indicator.notna()
                if valid_mask.sum() > 0:
                    X_test = baseline_split.X_test[valid_mask]
                    y_test = baseline_split.y_test[valid_mask]
                    rural_test = rural_indicator[valid_mask]
                    fm = compute_fairness_metrics(
                        result.model, result.dataset, result.estimator,
                        X_test, y_test, rural_test,
                    )
                    fairness_results.append(fm)

            elif result.dataset == "biased" and biased_rural is not None:
                test_ids = biased_split.X_test.index
                rural_indicator = biased_rural.reindex(test_ids)
                valid_mask = rural_indicator.notna()
                if valid_mask.sum() > 0:
                    X_test = biased_split.X_test[valid_mask]
                    y_test = biased_split.y_test[valid_mask]
                    rural_test = rural_indicator[valid_mask]
                    fm = compute_fairness_metrics(
                        result.model, result.dataset, result.estimator,
                        X_test, y_test, rural_test,
                    )
                    fairness_results.append(fm)

        print(f"  - Computed fairness metrics for {len(fairness_results)} model-dataset combinations")
        print()

    # -------------------------------------------------------------------------
    # Write comprehensive report
    # -------------------------------------------------------------------------
    print("[5/5] Writing comprehensive report...")

    write_comprehensive_report(
        Path(args.out),
        model_datasets=model_datasets,
        model_results=model_results,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        cross_results=cross_results,
        cross_note=cross_note,
        cross_test_size=cross_test_size,
        analytics_datasets=analytics_datasets,
        pairwise=pairwise_stats,
        regressions=regression_stats,
        population_stats=population_stats,
        n_perm=args.n_perm,
        fairness_metrics=fairness_results if fairness_results else None,
    )

    print(f"  - Report written to {args.out}")
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
