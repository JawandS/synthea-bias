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
        handle.write("# Sleep Apnea Demand Modeling Report\n\n")
        handle.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"- Baseline dataset: `{inputs.baseline_csv_dir}`\n")
        handle.write(f"- Biased dataset: `{inputs.biased_csv_dir}`\n")
        handle.write("- BMI feature included: True (always on)\n")
        handle.write("- Smoking, alcohol use, and CHF features included: True\n\n")

        handle.write("## Study Setup\n")
        handle.write("Commands used (from `scripts/NOTES.md`):\n")
        handle.write("```bash\n")
        handle.write(f"{inputs.baseline_generation_cmd}\n")
        handle.write(f"{inputs.biased_generation_cmd}\n")
        handle.write("```\n\n")

        handle.write("## Module Override (Rural Access Bias)\n")
        handle.write(
            "The override file `config/overrides_rural_sleep_apnea.properties` adjusts two complex "
            "transitions in `sleep_apnea.json` for patients with `urban == false`:\n"
            "- `Wait Until Overnight Study`: changes the rural branch from 0% `Terminal` / 100% "
            "`Overnight Test` to 70% / 30%.\n"
            "- `Appointment Delay`: changes the rural branch from 0% `Terminal` / 100% `Follow Up` "
            "to 70% / 30%.\n"
            "This simulates rural patients dropping out before testing or follow-up.\n\n"
        )

        handle.write("## Population Stats\n")
        handle.write(
            f"- Baseline patients: {baseline_total} "
            f"(urban {format_pct(baseline_urban_rate)}, rural {format_pct(baseline_rural_rate)})\n"
        )
        handle.write(
            f"- Biased patients: {biased_total} "
            f"(urban {format_pct(biased_urban_rate)}, rural {format_pct(biased_rural_rate)})\n"
        )
        handle.write(
            f"- Sleep apnea prevalence (unique patients with condition): baseline "
            f"{format_pct(baseline_population['apnea_prevalence'])} "
            f"({baseline_population['sleep_apnea']}/{baseline_total}), biased "
            f"{format_pct(biased_population['apnea_prevalence'])} "
            f"({biased_population['sleep_apnea']}/{biased_total})\n"
        )
        handle.write(
            f"- Sleep disorder patients (module entry): baseline {baseline_population['sleep_disorder']}, "
            f"biased {biased_population['sleep_disorder']}\n"
        )
        handle.write(
            f"- Dropouts (sleep disorder without sleep apnea): baseline {baseline_population['dropout']} "
            f"({format_pct(baseline_population['dropout_rate_disorder'])} of sleep disorder), biased "
            f"{biased_population['dropout']} "
            f"({format_pct(biased_population['dropout_rate_disorder'])} of sleep disorder)\n"
        )
        handle.write(
            "- Urban/rural classification uses `geography/sdoh.csv` (URBAN field) via SDoH attributes.\n"
        )
        if baseline_population["rural"] == 0 and biased_population["rural"] == 0:
            handle.write(
                "- These runs are 100% urban, so the rural dropout branch is not exercised.\n"
            )
        if baseline_population["missing_urban"] or biased_population["missing_urban"]:
            handle.write(
                f"- Urban/rural lookup missing: baseline {baseline_population['missing_urban']}, "
                f"biased {biased_population['missing_urban']}\n"
            )
        handle.write("\n")

        handle.write("## Target Definition\n")
        handle.write(
            "Total sleep-related spend per patient, computed as the sum of:\n"
            "- `encounters.csv` TOTAL_CLAIM_COST (fallback BASE_ENCOUNTER_COST) where REASONCODE "
            "is a sleep-related condition code (encounter CODE is used as a secondary filter).\n"
            "- `procedures.csv` BASE_COST where CODE is a sleep procedure code, REASONCODE is "
            "sleep-related, or the procedure is tied to a sleep encounter.\n"
            "- `medications.csv` TOTALCOST where REASONCODE is sleep-related or the medication is "
            "tied to a sleep encounter.\n"
            "- `devices.csv` MODE cost from costs/devices.csv or costs/supplies.csv for sleep-related "
            "device/supply codes or sleep-related encounters.\n"
            "- `supplies.csv` MODE cost from costs/supplies.csv or costs/devices.csv for sleep-related "
            "device/supply codes or sleep-related encounters.\n"
        )
        handle.write("\n")
        handle.write(f"Sleep-related condition codes: {sorted(inputs.sleep_reason_codes)}\n")
        handle.write(f"Sleep-related procedure codes: {sorted(inputs.sleep_procedure_codes)}\n")
        handle.write(f"Sleep-related encounter codes: {sorted(inputs.sleep_encounter_codes)}\n\n")
        handle.write(f"Sleep-related device codes: {sorted(inputs.sleep_device_codes)}\n")
        handle.write(f"Sleep-related supply codes: {sorted(inputs.sleep_supply_codes)}\n\n")
        handle.write(
            "Modeling uses log1p(spend) for training; evaluation metrics are reported in dollars "
            "after back-transforming predictions.\n\n"
        )

        handle.write("## Features (No Urban/Rural or Race/Ethnicity)\n")
        handle.write(", ".join(inputs.feature_names) + "\n\n")

        handle.write("## Dataset Summary\n")
        for label, summary in [("Baseline", analysis.baseline_summary), ("Biased", analysis.biased_summary)]:
            handle.write(
                f"- {label}: n={summary['n']}, mean_spend={format_currency(summary['mean_spend'])}, "
            )
            handle.write(f"nonzero_rate={summary['nonzero_rate'] * 100:.2f}%\n")
        handle.write("\n")

        handle.write("## Split Configuration\n")
        baseline_train, baseline_val, baseline_test = analysis.baseline_split_sizes
        handle.write(
            f"- Train/Val/Test sizes (baseline): "
            f"{baseline_train}/{baseline_val}/{baseline_test}\n"
        )
        biased_train, biased_val, biased_test = analysis.biased_split_sizes
        handle.write(
            f"- Train/Val/Test sizes (biased): "
            f"{biased_train}/{biased_val}/{biased_test}\n\n"
        )

        handle.write("## Model Selection (Validation MAE)\n")
        handle.write(f"- Baseline best params: {inputs.baseline_params}\n")
        handle.write(
            f"  - Val MAE={inputs.baseline_val_metrics['mae']:.2f}, "
            f"RMSE={inputs.baseline_val_metrics['rmse']:.2f}, R2={inputs.baseline_val_metrics['r2']:.3f}\n"
        )
        handle.write(f"- Biased best params: {inputs.biased_params}\n")
        handle.write(
            f"  - Val MAE={inputs.biased_val_metrics['mae']:.2f}, "
            f"RMSE={inputs.biased_val_metrics['rmse']:.2f}, R2={inputs.biased_val_metrics['r2']:.3f}\n\n"
        )

        handle.write("## Gradient Boosted Decision Tree (Feature Importances)\n")
        handle.write("Feature importances are normalized and sum to 1.0 per model.\n\n")
        handle.write("| Feature | Baseline | Biased |\n")
        handle.write("| --- | --- | --- |\n")
        for feature, baseline_value, biased_value in analysis.importance_table:
            handle.write(f"| {feature} | {baseline_value:.3f} | {biased_value:.3f} |\n")
        handle.write("\n")

        handle.write("## Test Results (In-Dataset)\n")
        handle.write(
            f"- Baseline model on baseline test: "
            f"MAE={analysis.baseline_test_metrics['mae']:.2f}, "
            f"RMSE={analysis.baseline_test_metrics['rmse']:.2f}, "
            f"R2={analysis.baseline_test_metrics['r2']:.3f}\n"
        )
        handle.write(
            f"- Biased model on biased test: "
            f"MAE={analysis.biased_test_metrics['mae']:.2f}, "
            f"RMSE={analysis.biased_test_metrics['rmse']:.2f}, "
            f"R2={analysis.biased_test_metrics['r2']:.3f}\n\n"
        )

        handle.write("## Cross-Dataset Evaluation\n")
        handle.write(
            f"- Biased model on baseline test: "
            f"MAE={analysis.biased_on_baseline_metrics['mae']:.2f}, "
            f"RMSE={analysis.biased_on_baseline_metrics['rmse']:.2f}, "
            f"R2={analysis.biased_on_baseline_metrics['r2']:.3f}\n"
        )
        handle.write(
            f"- Baseline model on biased test: "
            f"MAE={analysis.baseline_on_biased_metrics['mae']:.2f}, "
            f"RMSE={analysis.baseline_on_biased_metrics['rmse']:.2f}, "
            f"R2={analysis.baseline_on_biased_metrics['r2']:.3f}\n\n"
        )

        handle.write("## Demand Bias (Baseline Test Set)\n")
        handle.write(
            f"- Baseline model mean prediction: {format_currency(analysis.baseline_bias['mean_pred'])} "
            f"(actual {format_currency(analysis.baseline_bias['mean_true'])}, "
            f"diff {format_currency(analysis.baseline_bias['diff'])})\n"
        )
        handle.write(
            f"- Biased model mean prediction: {format_currency(analysis.biased_bias['mean_pred'])} "
            f"(actual {format_currency(analysis.biased_bias['mean_true'])}, "
            f"diff {format_currency(analysis.biased_bias['diff'])}, "
            f"rel {analysis.biased_bias['rel'] * 100:.2f}% )\n"
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
    handle.write("# Extended Analysis: Bias Quantification\n\n")
    
    # New bias quantification section using linear regression
    if inputs.bias_baseline is not None and inputs.bias_biased is not None:
        handle.write("## Geographic Bias Quantification (Linear Regression)\n\n")
        handle.write("This analysis uses linear regression to quantify the effect of urban/rural status\n")
        handle.write(
            "on log1p healthcare spending, controlling for age, gender, income, BMI, smoking status,\n"
        )
        handle.write("alcohol use disorder, hypertension, and CHF.\n\n")
        handle.write("**Key insight**: The urban coefficient represents the spending difference between\n")
        handle.write("urban and rural patients *after controlling for other factors*. A significant\n")
        handle.write("difference between baseline and biased datasets indicates measurable access disparity.\n\n")
        
        handle.write("### Urban Effect (Access Disparity Measure)\n\n")
        handle.write("| Dataset | Urban Effect (%) | 95% CI | p-value | Significant? |\n")
        handle.write("| --- | ---: | --- | ---: | :---: |\n")
        
        bb = inputs.bias_baseline
        bi = inputs.bias_biased
        
        sig_b = "Yes" if bb.urban_pvalue < 0.05 else "No"
        sig_bi = "Yes" if bi.urban_pvalue < 0.05 else "No"
        
        bb_effect = math.expm1(bb.urban_coefficient) * 100
        bi_effect = math.expm1(bi.urban_coefficient) * 100
        bb_ci = (math.expm1(bb.urban_ci_low) * 100, math.expm1(bb.urban_ci_high) * 100)
        bi_ci = (math.expm1(bi.urban_ci_low) * 100, math.expm1(bi.urban_ci_high) * 100)
        handle.write(
            f"| Baseline (unbiased) | {bb_effect:+.2f}% | "
            f"[{bb_ci[0]:+.2f}%, {bb_ci[1]:+.2f}%] | {bb.urban_pvalue:.4f} | {sig_b} |\n"
        )
        handle.write(
            f"| Biased (rural dropout) | {bi_effect:+.2f}% | "
            f"[{bi_ci[0]:+.2f}%, {bi_ci[1]:+.2f}%] | {bi.urban_pvalue:.4f} | {sig_bi} |\n"
        )
        handle.write("\n")
        
        diff = bi_effect - bb_effect
        handle.write("### Interpretation\n\n")
        handle.write(f"- **Baseline urban effect**: {bb_effect:+.2f}%\n")
        handle.write(f"- **Biased urban effect**: {bi_effect:+.2f}%\n")
        handle.write(f"- **Difference (bias effect)**: {diff:+.2f}%\n\n")
        
        if diff < 0:
            handle.write(
                f"The biased dataset shows rural patients spending **{abs(diff):.2f}% less** than\n"
            )
            handle.write(
                "in the baseline dataset, controlling for other factors. This difference represents\n"
            )
            handle.write("the access disparity introduced by rural dropout.\n\n")
        elif diff > 0:
            handle.write(
                f"The biased dataset shows rural patients spending **{diff:.2f}% more** than\n"
            )
            handle.write(
                "in the baseline dataset. This is unexpected and may indicate other confounding factors.\n\n"
            )
        else:
            handle.write("No measurable difference in urban coefficient between datasets.\n\n")
        
        handle.write("### All Coefficients (Standardized)\n\n")
        handle.write("Standardized coefficients allow comparison of relative feature importance.\n\n")
        handle.write("| Feature | Baseline (std) | Biased (std) |\n")
        handle.write("| --- | ---: | ---: |\n")
        
        for feature in inputs.feature_names_urban:
            b_val = bb.standardized_coefficients.get(feature, 0.0)
            bi_val = bi.standardized_coefficients.get(feature, 0.0)
            handle.write(f"| {feature} | {b_val:,.2f} | {bi_val:,.2f} |\n")
        handle.write("\n")
        
        handle.write("### Coefficient Confidence Intervals (log1p scale)\n\n")
        handle.write("| Feature | Baseline Coef [95% CI] | Biased Coef [95% CI] |\n")
        handle.write("| --- | --- | --- |\n")
        
        for feature in list(inputs.feature_names_urban) + ["intercept"]:
            b_ci = bb.coefficient_cis.get(feature, {})
            bi_ci = bi.coefficient_cis.get(feature, {})
            if b_ci and bi_ci:
                b_str = f"{b_ci['point']:.4f} [{b_ci['ci_low']:.4f}, {b_ci['ci_high']:.4f}]"
                bi_str = f"{bi_ci['point']:.4f} [{bi_ci['ci_low']:.4f}, {bi_ci['ci_high']:.4f}]"
                handle.write(f"| {feature} | {b_str} | {bi_str} |\n")
        handle.write("\n")
        
        handle.write(f"### Linear Model R² (Test Set)\n\n")
        handle.write(f"- Baseline (log1p spend): R² = {bb.r2:.4f}\n")
        handle.write(f"- Biased (log1p spend): R² = {bi.r2:.4f}\n\n")
    
    # Bootstrap results for GBDT base model
    if inputs.bootstrap_results:
        handle.write("## GBDT Model Performance (Bootstrap CIs)\n\n")
        handle.write(f"Bootstrap iterations: {inputs.n_bootstrap}\n")
        handle.write("95% confidence intervals shown as [lower, upper]\n\n")
        
        handle.write("### In-Dataset Performance\n\n")
        handle.write("| Model | Dataset | MAE [95% CI] | RMSE [95% CI] | R² [95% CI] |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        
        bootstrap = inputs.bootstrap_results
        if "base_baseline_in" in bootstrap:
            m = bootstrap["base_baseline_in"]
            handle.write(f"| Base | Baseline | {_format_ci(m['mae'])} | {_format_ci(m['rmse'])} | {_format_ci(m['r2'])} |\n")
        if "base_biased_in" in bootstrap:
            m = bootstrap["base_biased_in"]
            handle.write(f"| Base | Biased | {_format_ci(m['mae'])} | {_format_ci(m['rmse'])} | {_format_ci(m['r2'])} |\n")
        handle.write("\n")
        
        handle.write("### Cross-Population Performance\n\n")
        handle.write("| Model | Source → Target | MAE [95% CI] | RMSE [95% CI] | R² [95% CI] |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        
        if "base_biased_on_baseline" in bootstrap:
            m = bootstrap["base_biased_on_baseline"]
            handle.write(f"| Base | Biased → Baseline | {_format_ci(m['mae'])} | {_format_ci(m['rmse'])} | {_format_ci(m['r2'])} |\n")
        if "base_baseline_on_biased" in bootstrap:
            m = bootstrap["base_baseline_on_biased"]
            handle.write(f"| Base | Baseline → Biased | {_format_ci(m['mae'])} | {_format_ci(m['rmse'])} | {_format_ci(m['r2'])} |\n")
        handle.write("\n")
    
    # Permutation tests
    if inputs.permutation_results:
        handle.write("## Hypothesis Tests (Permutation)\n\n")
        handle.write(f"Permutation iterations: {inputs.n_permutations}\n")
        handle.write("Two-sided permutation tests for MAE differences.\n")
        handle.write("p < 0.05 indicates statistically significant difference.\n\n")
        
        handle.write("| Test | Description | Observed Diff | p-value | Significant |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        
        perm = inputs.permutation_results
        if "base_cross_pop" in perm:
            p = perm["base_cross_pop"]
            sig = "Yes" if p["p_value"] < 0.05 else "No"
            handle.write(f"| Base Cross-Pop | Biased model: baseline vs biased test | {p['observed_diff']:.2f} | {p['p_value']:.4f} | {sig} |\n")
        handle.write("\n")
    
    handle.write("## Methodology Notes\n\n")
    handle.write("### Bias Quantification Approach\n\n")
    handle.write("Instead of using urban/rural as a feature in a predictive model (which would encode\n")
    handle.write("the disparity), we use linear regression to *measure* the urban coefficient:\n\n")
    handle.write("```\n")
    handle.write(
        "log1p(spend) ~ age + gender + income + bmi + smoker + alcohol_use + hypertension + chf + urban\n"
    )
    handle.write("```\n\n")
    handle.write("The urban coefficient represents the average spending difference between urban and\n")
    handle.write("rural patients on the log1p scale, controlling for other factors. By comparing this\n")
    handle.write("coefficient across datasets, we can quantify the access disparity introduced by the bias.\n\n")
    handle.write("### Key Findings\n\n")
    handle.write("1. **The urban coefficient measures disparity, not need**: A positive coefficient means\n")
    handle.write("   urban patients spend more (controlling for demographics), indicating rural underutilization.\n\n")
    handle.write("2. **Comparing coefficients reveals bias**: If the biased dataset has a larger urban\n")
    handle.write("   coefficient than baseline, the rural dropout is creating measurable disparity.\n\n")
    handle.write("3. **Statistical significance**: Bootstrap CIs and permutation p-values indicate whether\n")
    handle.write("   the measured disparity is statistically robust.\n")
