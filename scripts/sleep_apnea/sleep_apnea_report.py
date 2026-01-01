#!/usr/bin/env python3
"""Render the sleep apnea modeling report to Markdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

from analyze_sleep_apnea import AnalysisResults


@dataclass
class ReportInputs:
    """Inputs required to render the markdown report."""

    baseline_csv_dir: Path
    biased_csv_dir: Path
    feature_names: Sequence[str]
    baseline_params: Mapping[str, object]
    biased_params: Mapping[str, object]
    baseline_val_metrics: Dict[str, float]
    biased_val_metrics: Dict[str, float]
    analysis: AnalysisResults
    baseline_generation_cmd: str
    biased_generation_cmd: str
    sleep_reason_codes: Collection[str]
    sleep_procedure_codes: Collection[str]
    sleep_encounter_codes: Collection[str]
    sleep_device_codes: Collection[str]
    sleep_supply_codes: Collection[str]
    # New fields for extended analysis
    feature_names_urban: Sequence[str] = field(default_factory=list)
    urban_baseline_params: Optional[Mapping[str, object]] = None
    urban_biased_params: Optional[Mapping[str, object]] = None
    urban_baseline_val_metrics: Optional[Dict[str, float]] = None
    urban_biased_val_metrics: Optional[Dict[str, float]] = None
    bootstrap_results: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    permutation_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_bootstrap: int = 1000
    n_permutations: int = 1000
    base_baseline_model: Optional[Any] = None
    urban_baseline_model: Optional[Any] = None
    base_biased_model: Optional[Any] = None
    urban_biased_model: Optional[Any] = None
    # Linear regression bias analysis
    bias_baseline: Optional[Any] = None
    bias_biased: Optional[Any] = None


def format_currency(value: float) -> str:
    """Format a numeric value as USD currency."""
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.2f}%"


def write_report(output_path: Path, inputs: ReportInputs) -> None:
    """Write a markdown report comparing baseline and biased model performance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analysis = inputs.analysis
    baseline_population = analysis.baseline_population
    biased_population = analysis.biased_population

    baseline_total = baseline_population["total"]
    biased_total = biased_population["total"]
    baseline_urban_rate = baseline_population["urban"] / baseline_total if baseline_total else 0.0
    biased_urban_rate = biased_population["urban"] / biased_total if biased_total else 0.0
    baseline_rural_rate = baseline_population["rural"] / baseline_total if baseline_total else 0.0
    biased_rural_rate = biased_population["rural"] / biased_total if biased_total else 0.0

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Sleep Apnea Demand Modeling: Findings Report\n\n")
        handle.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        handle.write("> See [README.md](README.md) for study methodology, feature definitions, and clinical codes.\n\n")
        handle.write("## Run Configuration\n\n")
        handle.write(f"| Parameter | Value |\n")
        handle.write(f"| --- | --- |\n")
        handle.write(f"| Baseline dataset | `{inputs.baseline_csv_dir}` |\n")
        handle.write(f"| Biased dataset | `{inputs.biased_csv_dir}` |\n")
        handle.write(f"| Bootstrap iterations | {inputs.n_bootstrap} |\n")
        handle.write(f"| Permutation iterations | {inputs.n_permutations} |\n")
        handle.write("\n")

        handle.write("## Population Summary\n\n")
        handle.write("| Metric | Baseline | Biased |\n")
        handle.write("| --- | ---: | ---: |\n")
        handle.write(f"| Total patients | {baseline_total:,} | {biased_total:,} |\n")
        handle.write(f"| Urban | {format_pct(baseline_urban_rate)} | {format_pct(biased_urban_rate)} |\n")
        handle.write(f"| Rural | {format_pct(baseline_rural_rate)} | {format_pct(biased_rural_rate)} |\n")
        handle.write(f"| Sleep disorder (module entry) | {baseline_population['sleep_disorder']} | {biased_population['sleep_disorder']} |\n")
        handle.write(f"| Sleep apnea diagnosed | {baseline_population['sleep_apnea']} | {biased_population['sleep_apnea']} |\n")
        handle.write(f"| Sleep apnea prevalence | {format_pct(baseline_population['apnea_prevalence'])} | {format_pct(biased_population['apnea_prevalence'])} |\n")
        handle.write(f"| Dropouts | {baseline_population['dropout']} | {biased_population['dropout']} |\n")
        handle.write(f"| Dropout rate (of sleep disorder) | {format_pct(baseline_population['dropout_rate_disorder'])} | {format_pct(biased_population['dropout_rate_disorder'])} |\n")
        handle.write("\n")
        if baseline_population["rural"] == 0 and biased_population["rural"] == 0:
            handle.write("> ⚠️ **Warning**: Both datasets are 100% urban. The rural dropout branch was not exercised.\n\n")
        if baseline_population["missing_urban"] or biased_population["missing_urban"]:
            handle.write(f"> ℹ️ Urban/rural lookup missing: baseline {baseline_population['missing_urban']}, biased {biased_population['missing_urban']}\n\n")

        handle.write("## Spend Distribution\n\n")

        handle.write("## Dataset Summary\n")
        handle.write("| Metric | Baseline | Biased |\n")
        handle.write("| --- | ---: | ---: |\n")
        for label, summary in [("Baseline", analysis.baseline_summary), ("Biased", analysis.biased_summary)]:
            pass  # Will build rows below
        handle.write(f"| Sample size | {analysis.baseline_summary['n']:,} | {analysis.biased_summary['n']:,} |\n")
        handle.write(f"| Mean spend | {format_currency(analysis.baseline_summary['mean_spend'])} | {format_currency(analysis.biased_summary['mean_spend'])} |\n")
        handle.write(f"| Nonzero rate | {analysis.baseline_summary['nonzero_rate'] * 100:.1f}% | {analysis.biased_summary['nonzero_rate'] * 100:.1f}% |\n")
        handle.write(f"| Train/Val/Test split | {analysis.baseline_split_sizes[0]}/{analysis.baseline_split_sizes[1]}/{analysis.baseline_split_sizes[2]} | {analysis.biased_split_sizes[0]}/{analysis.biased_split_sizes[1]}/{analysis.biased_split_sizes[2]} |\n")
        handle.write("\n")

        handle.write("## GBDT Model Selection\n\n")
        handle.write("| Dataset | Best Hyperparameters | Val MAE | Val R² |\n")
        handle.write("| --- | --- | ---: | ---: |\n")
        handle.write(f"| Baseline | n={inputs.baseline_params['n_estimators']}, lr={inputs.baseline_params['learning_rate']}, depth={inputs.baseline_params['max_depth']}, leaf={inputs.baseline_params['min_samples_leaf']} | ${inputs.baseline_val_metrics['mae']:,.0f} | {inputs.baseline_val_metrics['r2']:.3f} |\n")
        handle.write(f"| Biased | n={inputs.biased_params['n_estimators']}, lr={inputs.biased_params['learning_rate']}, depth={inputs.biased_params['max_depth']}, leaf={inputs.biased_params['min_samples_leaf']} | ${inputs.biased_val_metrics['mae']:,.0f} | {inputs.biased_val_metrics['r2']:.3f} |\n")
        handle.write("\n")

        handle.write("## Feature Importances\n\n")
        handle.write("| Feature | Baseline | Biased |\n")
        handle.write("| --- | --- | --- |\n")
        for feature, baseline_value, biased_value in analysis.importance_table:
            handle.write(f"| {feature} | {baseline_value:.3f} | {biased_value:.3f} |\n")
        handle.write("\n")

        handle.write("## Model Performance\n\n")
        handle.write("### In-Dataset Test Results\n\n")
        handle.write("| Model | Test Set | MAE | RMSE | R² |\n")
        handle.write("| --- | --- | ---: | ---: | ---: |\n")
        handle.write(
            f"| Baseline | Baseline | ${analysis.baseline_test_metrics['mae']:,.0f} | "
            f"${analysis.baseline_test_metrics['rmse']:,.0f} | "
            f"{analysis.baseline_test_metrics['r2']:.3f} |\n"
        )
        handle.write(
            f"| Biased | Biased | ${analysis.biased_test_metrics['mae']:,.0f} | "
            f"${analysis.biased_test_metrics['rmse']:,.0f} | "
            f"{analysis.biased_test_metrics['r2']:.3f} |\n"
        )
        handle.write("\n")

        handle.write("### Cross-Dataset Test Results\n\n")
        handle.write("| Model | Test Set | MAE | RMSE | R² |\n")
        handle.write("| --- | --- | ---: | ---: | ---: |\n")
        handle.write(
            f"| Biased | Baseline | ${analysis.biased_on_baseline_metrics['mae']:,.0f} | "
            f"${analysis.biased_on_baseline_metrics['rmse']:,.0f} | "
            f"{analysis.biased_on_baseline_metrics['r2']:.3f} |\n"
        )
        handle.write(
            f"| Baseline | Biased | ${analysis.baseline_on_biased_metrics['mae']:,.0f} | "
            f"${analysis.baseline_on_biased_metrics['rmse']:,.0f} | "
            f"{analysis.baseline_on_biased_metrics['r2']:.3f} |\n"
        )
        handle.write("\n")

        handle.write("### Prediction Bias (Baseline Test Set)\n\n")
        handle.write("| Model | Mean Prediction | Actual Mean | Difference | Rel. Error |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        handle.write(
            f"| Baseline | {format_currency(analysis.baseline_bias['mean_pred'])} | "
            f"{format_currency(analysis.baseline_bias['mean_true'])} | "
            f"{format_currency(analysis.baseline_bias['diff'])} | - |\n"
        )
        rel_pct = analysis.biased_bias['rel'] * 100
        handle.write(
            f"| Biased | {format_currency(analysis.biased_bias['mean_pred'])} | "
            f"{format_currency(analysis.biased_bias['mean_true'])} | "
            f"{format_currency(analysis.biased_bias['diff'])} | {rel_pct:+.2f}% |\n"
        )
        handle.write("\n")

        # New sections for extended analysis
        _write_extended_analysis(handle, inputs)


def _format_ci(metrics: Dict[str, float]) -> str:
    """Format a metric with its confidence interval."""
    return f"{metrics['point']:.2f} [{metrics['ci_low']:.2f}, {metrics['ci_high']:.2f}]"


def _write_extended_analysis(handle, inputs: ReportInputs) -> None:
    """Write extended analysis sections including bias quantification and statistical tests."""
    
    if not inputs.bootstrap_results and not inputs.bias_baseline:
        return
    
    handle.write("---\n\n")
    handle.write("# Bias Quantification\n\n")
    
    # New bias quantification section using linear regression
    if inputs.bias_baseline is not None and inputs.bias_biased is not None:
        handle.write("## Geographic Disparity (Linear Regression)\n\n")
        handle.write("The urban coefficient measures spending difference between urban and rural patients\\n")
        handle.write("after controlling for clinical factors. See README.md for methodology.\\n\\n")
        
        handle.write("### Urban Effect Summary\\n\\n")
        handle.write("| Dataset | Urban Effect (%) | 95% CI | p-value | Significant? |\\n")
        handle.write("| --- | ---: | --- | ---: | :---: |\\n")
        
        bb = inputs.bias_baseline
        bi = inputs.bias_biased
        
        sig_b = "Yes" if bb.urban_pvalue < 0.05 else "No"
        sig_bi = "Yes" if bi.urban_pvalue < 0.05 else "No"
        
        bb_effect = math.expm1(bb.urban_coefficient) * 100
        bi_effect = math.expm1(bi.urban_coefficient) * 100
        bb_ci = (math.expm1(bb.urban_ci_low) * 100, math.expm1(bb.urban_ci_high) * 100)
        bi_ci = (math.expm1(bi.urban_ci_low) * 100, math.expm1(bi.urban_ci_high) * 100)
        handle.write(
            f"| Baseline | {bb_effect:+.2f}% | "
            f"[{bb_ci[0]:+.2f}%, {bb_ci[1]:+.2f}%] | {bb.urban_pvalue:.4f} | {sig_b} |\\n"
        )
        handle.write(
            f"| Biased | {bi_effect:+.2f}% | "
            f"[{bi_ci[0]:+.2f}%, {bi_ci[1]:+.2f}%] | {bi.urban_pvalue:.4f} | {sig_bi} |\\n"
        )
        handle.write("\\n")
        
        diff = bi_effect - bb_effect
        handle.write("### Key Finding\\n\\n")
        handle.write(f"| Metric | Value |\\n")
        handle.write(f"| --- | ---: |\\n")
        handle.write(f"| Baseline urban effect | {bb_effect:+.2f}% |\\n")
        handle.write(f"| Biased urban effect | {bi_effect:+.2f}% |\\n")
        handle.write(f"| **Bias-induced disparity** | **{diff:+.2f}%** |\\n")
        handle.write(f"| Baseline R² | {bb.r2:.4f} |\\n")
        handle.write(f"| Biased R² | {bi.r2:.4f} |\\n")
        handle.write("\\n")
        
        if abs(diff) > 1:
            if diff < 0:
                handle.write(
                    f"> The biased dataset shows rural patients spending **{abs(diff):.1f}% less** "
                    f"than in baseline, indicating access disparity from rural dropout.\\n\\n"
                )
            else:
                handle.write(
                    f"> The biased dataset shows rural patients spending **{diff:.1f}% more** "
                    f"than in baseline, which is unexpected.\\n\\n"
                )
        
        handle.write("### Standardized Coefficients\\n\\n")
        handle.write("| Feature | Baseline | Biased |\\n")
        handle.write("| --- | ---: | ---: |\\n")
        
        for feature in inputs.feature_names_urban:
            b_val = bb.standardized_coefficients.get(feature, 0.0)
            bi_val = bi.standardized_coefficients.get(feature, 0.0)
            handle.write(f"| {feature} | {b_val:,.2f} | {bi_val:,.2f} |\\n")
        handle.write("\\n")
    
    # Bootstrap results for GBDT base model
    if inputs.bootstrap_results:
        handle.write("## Statistical Confidence (Bootstrap)\\n\\n")
        
        bootstrap = inputs.bootstrap_results
        handle.write("| Evaluation | MAE [95% CI] | R² [95% CI] |\\n")
        handle.write("| --- | --- | --- |\\n")
        
        if "base_baseline_in" in bootstrap:
            m = bootstrap["base_baseline_in"]
            handle.write(f"| Baseline → Baseline | {_format_ci(m['mae'])} | {_format_ci(m['r2'])} |\\n")
        if "base_biased_in" in bootstrap:
            m = bootstrap["base_biased_in"]
            handle.write(f"| Biased → Biased | {_format_ci(m['mae'])} | {_format_ci(m['r2'])} |\\n")
        if "base_biased_on_baseline" in bootstrap:
            m = bootstrap["base_biased_on_baseline"]
            handle.write(f"| Biased → Baseline | {_format_ci(m['mae'])} | {_format_ci(m['r2'])} |\\n")
        if "base_baseline_on_biased" in bootstrap:
            m = bootstrap["base_baseline_on_biased"]
            handle.write(f"| Baseline → Biased | {_format_ci(m['mae'])} | {_format_ci(m['r2'])} |\\n")
        handle.write("\\n")
        handle.write(f"*{inputs.n_bootstrap} bootstrap iterations, 95% CIs*\\n\\n")
    
    # Permutation tests
    if inputs.permutation_results:
        perm = inputs.permutation_results
        if "base_cross_pop" in perm:
            p = perm["base_cross_pop"]
            sig = "**significant**" if p["p_value"] < 0.05 else "not significant"
            handle.write("## Hypothesis Test (Permutation)\\n\\n")
            handle.write(f"Cross-population MAE difference: {p['observed_diff']:.2f} (p = {p['p_value']:.4f}, {sig})\\n\\n")
            handle.write(f"*{inputs.n_permutations} permutations, two-sided test*\\n")
