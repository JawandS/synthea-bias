#!/usr/bin/env python3
"""Render the sleep apnea modeling report to Markdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    urban_baseline_params: Mapping[str, object] = field(default_factory=dict)
    urban_biased_params: Mapping[str, object] = field(default_factory=dict)
    urban_baseline_val_metrics: Dict[str, float] = field(default_factory=dict)
    urban_biased_val_metrics: Dict[str, float] = field(default_factory=dict)
    bootstrap_results: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    permutation_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_bootstrap: int = 1000
    n_permutations: int = 1000
    base_baseline_model: Optional[Any] = None
    urban_baseline_model: Optional[Any] = None
    base_biased_model: Optional[Any] = None
    urban_biased_model: Optional[Any] = None


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
        handle.write("- BMI feature included: True (always on)\n\n")

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
            "`Overnight Test` to 50% / 50%.\n"
            "- `Appointment Delay`: changes the rural branch from 0% `Terminal` / 100% `Follow Up` "
            "to 50% / 50%.\n"
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
            "- `procedures.csv` BASE_COST where CODE in sleep-related procedure codes or REASONCODE "
            "in sleep-related condition codes.\n"
            "- `encounters.csv` TOTAL_CLAIM_COST (fallback BASE_ENCOUNTER_COST) where REASONCODE "
            "in sleep-related condition codes or encounter CODE in sleep-specific codes.\n"
            "- `medications.csv` TOTALCOST where REASONCODE in sleep-related condition codes.\n"
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
    """Write extended analysis sections including urban model and statistical tests."""
    
    if not inputs.bootstrap_results:
        return
    
    handle.write("---\n\n")
    handle.write("# Extended Analysis: Model Variants and Statistical Testing\n\n")
    
    handle.write("## Model Variants\n\n")
    handle.write("This analysis compares two model variants to assess whether including\n")
    handle.write("geographic (urban/rural) information can control for access bias:\n\n")
    handle.write("1. **Base Model**: Core features only (age, gender, income, BMI, hypertension)\n")
    handle.write("2. **Urban Model**: Core features + urban/rural indicator\n\n")
    
    handle.write("### Feature Sets\n")
    handle.write(f"- Base features: {', '.join(inputs.feature_names)}\n")
    if inputs.feature_names_urban:
        handle.write(f"- Urban features: {', '.join(inputs.feature_names_urban)}\n")
    handle.write("\n")
    
    handle.write("## Urban Model Hyperparameters\n\n")
    if inputs.urban_baseline_params:
        handle.write(f"- Urban baseline best params: {dict(inputs.urban_baseline_params)}\n")
        if inputs.urban_baseline_val_metrics:
            handle.write(
                f"  - Val MAE={inputs.urban_baseline_val_metrics['mae']:.2f}, "
                f"RMSE={inputs.urban_baseline_val_metrics['rmse']:.2f}, "
                f"R2={inputs.urban_baseline_val_metrics['r2']:.3f}\n"
            )
    if inputs.urban_biased_params:
        handle.write(f"- Urban biased best params: {dict(inputs.urban_biased_params)}\n")
        if inputs.urban_biased_val_metrics:
            handle.write(
                f"  - Val MAE={inputs.urban_biased_val_metrics['mae']:.2f}, "
                f"RMSE={inputs.urban_biased_val_metrics['rmse']:.2f}, "
                f"R2={inputs.urban_biased_val_metrics['r2']:.3f}\n"
            )
    handle.write("\n")
    
    # Feature importances for urban models
    if inputs.urban_baseline_model is not None and inputs.feature_names_urban:
        handle.write("## Urban Model Feature Importances\n\n")
        handle.write("| Feature | Urban Baseline | Urban Biased |\n")
        handle.write("| --- | --- | --- |\n")
        
        baseline_imp = dict(zip(inputs.feature_names_urban, 
                                inputs.urban_baseline_model.feature_importances_))
        biased_imp = {}
        if inputs.urban_biased_model is not None:
            biased_imp = dict(zip(inputs.feature_names_urban,
                                  inputs.urban_biased_model.feature_importances_))
        
        for feature in inputs.feature_names_urban:
            b_val = baseline_imp.get(feature, 0.0)
            bi_val = biased_imp.get(feature, 0.0)
            handle.write(f"| {feature} | {b_val:.3f} | {bi_val:.3f} |\n")
        handle.write("\n")
    
    handle.write("## Bootstrap Confidence Intervals\n\n")
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
    if "urban_baseline_in" in bootstrap:
        m = bootstrap["urban_baseline_in"]
        handle.write(f"| Urban | Baseline | {_format_ci(m['mae'])} | {_format_ci(m['rmse'])} | {_format_ci(m['r2'])} |\n")
    if "urban_biased_in" in bootstrap:
        m = bootstrap["urban_biased_in"]
        handle.write(f"| Urban | Biased | {_format_ci(m['mae'])} | {_format_ci(m['rmse'])} | {_format_ci(m['r2'])} |\n")
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
    if "urban_biased_on_baseline" in bootstrap:
        m = bootstrap["urban_biased_on_baseline"]
        handle.write(f"| Urban | Biased → Baseline | {_format_ci(m['mae'])} | {_format_ci(m['rmse'])} | {_format_ci(m['r2'])} |\n")
    if "urban_baseline_on_biased" in bootstrap:
        m = bootstrap["urban_baseline_on_biased"]
        handle.write(f"| Urban | Baseline → Biased | {_format_ci(m['mae'])} | {_format_ci(m['rmse'])} | {_format_ci(m['r2'])} |\n")
    handle.write("\n")
    
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
    if "urban_cross_pop" in perm:
        p = perm["urban_cross_pop"]
        sig = "Yes" if p["p_value"] < 0.05 else "No"
        handle.write(f"| Urban Cross-Pop | Urban model: baseline vs biased test | {p['observed_diff']:.2f} | {p['p_value']:.4f} | {sig} |\n")
    if "base_vs_urban_baseline" in perm:
        p = perm["base_vs_urban_baseline"]
        sig = "Yes" if p["p_value"] < 0.05 else "No"
        handle.write(f"| Base vs Urban (Baseline) | Base vs Urban on baseline data | {p['observed_diff']:.2f} | {p['p_value']:.4f} | {sig} |\n")
    if "base_vs_urban_biased" in perm:
        p = perm["base_vs_urban_biased"]
        sig = "Yes" if p["p_value"] < 0.05 else "No"
        handle.write(f"| Base vs Urban (Biased) | Base vs Urban on biased data | {p['observed_diff']:.2f} | {p['p_value']:.4f} | {sig} |\n")
    handle.write("\n")
    
    handle.write("## Interpretation\n\n")
    handle.write("### Key Questions Addressed\n\n")
    handle.write("1. **Does the urban feature improve model performance?**\n")
    handle.write("   Compare Base vs Urban model MAE confidence intervals.\n\n")
    handle.write("2. **Does including urban/rural control for geographic access bias?**\n")
    handle.write("   If the Urban model shows smaller cross-population MAE differences\n")
    handle.write("   than the Base model, the urban feature partially controls for bias.\n\n")
    handle.write("3. **Is the bias statistically significant?**\n")
    handle.write("   The permutation test p-values indicate whether observed differences\n")
    handle.write("   are unlikely to occur by chance.\n\n")
    handle.write("### Methodology Notes\n\n")
    handle.write("- **Data leakage prevention**: Healthcare expenses and coverage removed from features\n")
    handle.write("  as they include the target (sleep-related spending).\n")
    handle.write("- **Consistent splits**: Same random seed used for both datasets to ensure\n")
    handle.write("  comparable train/val/test partitions.\n")
    handle.write("- **Bootstrap CIs**: Non-parametric confidence intervals via resampling.\n")
    handle.write("- **Permutation tests**: Distribution-free hypothesis testing by shuffling sample assignments.\n")
