#!/usr/bin/env python3
"""
4_create_report.py - Generate comprehensive case study report.

This script compiles all analysis outputs into a single report.md file that
documents the complete sleep apnea underdiagnosis bias case study.

Requires:
    - output/data/data.csv (from 2_gen_bias.py)
    - output/info/1_summary_stats.md (optional, from 1_generate_data.py)
    - output/info/2_bias_effect.md (optional, from 2_gen_bias.py)
    - output/info/3_model.md (optional, from 3_train_models.py)

Usage:
    uv run python scripts/4_create_report.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
INFO_DIR = OUTPUT_DIR / "info"

# Input files
SUMMARY_STATS_PATH = INFO_DIR / "1_summary_stats.md"
BIAS_EFFECT_PATH = INFO_DIR / "2_bias_effect.md"
MODEL_PATH = INFO_DIR / "3_model.md"

# Output
REPORT_PATH = OUTPUT_DIR / "report.md"


def extract_table_from_md(md_content: str, table_header: str) -> str | None:
    """Extract a markdown table that follows a given header."""
    lines = md_content.split("\n")
    in_table = False
    table_lines = []

    for i, line in enumerate(lines):
        if table_header in line:
            # Find the table (skip blank lines)
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("|"):
                    in_table = True
                    table_lines.append(lines[j])
                elif in_table and not lines[j].strip().startswith("|"):
                    break
            break

    return "\n".join(table_lines) if table_lines else None


def extract_key_findings(md_content: str) -> str | None:
    """Extract key findings section from model report."""
    lines = md_content.split("\n")
    findings_lines = []
    in_findings = False

    for line in lines:
        if "## Key Findings" in line:
            in_findings = True
            continue
        if in_findings:
            # Stop at next section or end
            if line.startswith("## ") or line.startswith("---"):
                break
            findings_lines.append(line)

    content = "\n".join(findings_lines).strip()
    return content if content else None


def extract_metric_from_table(md_content: str, table_header: str, row_label: str, col_index: int) -> float | None:
    """Extract a specific metric value from a markdown table.

    Args:
        md_content: Full markdown content
        table_header: Header that precedes the table (e.g., "### Recall by Subgroup")
        row_label: Label in first column to find (e.g., "Rural")
        col_index: 0-based index of column to extract (after the label column)

    Returns:
        Float value or None if not found
    """
    lines = md_content.split("\n")
    in_table = False
    found_header = False

    for line in lines:
        if table_header in line:
            found_header = True
            continue
        if found_header and line.strip().startswith("|"):
            # Skip header row and separator
            if "---" in line or row_label not in line:
                in_table = True
                continue
            if row_label in line:
                # Parse the row
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) > col_index + 1:
                    try:
                        return float(parts[col_index + 1])
                    except ValueError:
                        return None
        elif found_header and in_table and not line.strip().startswith("|"):
            break
    return None


def read_stats_from_data() -> dict[str, int | float]:
    """Read statistics directly from data.csv."""
    df = pd.read_csv(DATA_DIR / "data.csv")

    stats: dict[str, int | float] = {
        "n_patients": len(df),
        "n_urban": int((df["urban"] == 1).sum()),
        "n_rural": int((df["urban"] == 0).sum()),
        "n_male": int((df["male"] == 1).sum()),
        "n_female": int((df["male"] == 0).sum()),
    }

    stats["pct_urban"] = 100 * stats["n_urban"] / stats["n_patients"]
    stats["pct_rural"] = 100 * stats["n_rural"] / stats["n_patients"]

    if "has_sleep_apnea" in df.columns:
        stats["n_apnea"] = int(df["has_sleep_apnea"].sum())
        stats["pct_apnea"] = 100 * stats["n_apnea"] / stats["n_patients"]

        # By location
        urban_mask = df["urban"] == 1
        rural_mask = df["urban"] == 0
        stats["urban_apnea"] = int(df.loc[urban_mask, "has_sleep_apnea"].sum())
        stats["rural_apnea"] = int(df.loc[rural_mask, "has_sleep_apnea"].sum())
        stats["pct_urban_apnea"] = 100 * stats["urban_apnea"] / stats["n_urban"] if stats["n_urban"] else 0
        stats["pct_rural_apnea"] = 100 * stats["rural_apnea"] / stats["n_rural"] if stats["n_rural"] else 0

        if "mask_sleep_apnea" in df.columns:
            stats["n_masked"] = int(df["mask_sleep_apnea"].sum())
            n_rural_apnea = int(((df["urban"] == 0) & (df["has_sleep_apnea"] == 1)).sum())
            stats["n_rural_apnea"] = n_rural_apnea
            stats["mask_rate"] = 100 * stats["n_masked"] / n_rural_apnea if n_rural_apnea else 0

    return stats


def read_md_file(path: Path) -> str | None:
    """Read markdown file if it exists."""
    if path.exists():
        return path.read_text()
    return None


def generate_report() -> str:
    """Generate the complete case study report."""
    stats = read_stats_from_data()
    summary_md = read_md_file(SUMMARY_STATS_PATH)
    bias_md = read_md_file(BIAS_EFFECT_PATH)
    model_md = read_md_file(MODEL_PATH)

    report = f"""# Sleep Apnea Underdiagnosis Bias: A Case Study

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Overview

### Study Goal

This case study demonstrates how **barriers to care resulting in underdiagnosis** can
introduce systematic bias into healthcare machine learning models. Specifically, we
examine how rural populations face reduced access to diagnostic services for sleep
apnea, leading to lower observed diagnosis rates despite similar true prevalence.

When ML models are trained on this biased "observed" data, they learn to underpredict
sleep apnea in rural populations, perpetuating and potentially amplifying existing
healthcare disparities.

### The Problem

Sleep apnea affects approximately 9-38% of the general adult population, with higher
rates among the elderly. Diagnosis requires specialized testing:

- **Polysomnography**: Overnight sleep study at a specialized clinic
- **Home sleep testing**: Ambulatory monitoring devices

Rural populations face significant barriers to diagnosis:
- Fewer sleep specialists and clinics
- Longer travel distances to diagnostic facilities
- Reduced access to follow-up care

This creates a **label masking bias**: rural patients with sleep apnea are less likely
to receive a formal diagnosis, making them appear "healthy" in electronic health records.

### Study Design

1. **Generate synthetic population** with realistic sleep apnea prevalence
2. **Apply rural underdiagnosis bias** by masking a portion of rural diagnoses
3. **Train ML models** on both true and biased labels
4. **Compare performance** to quantify the bias impact

---

## 2. Data Generation

### Synthea Patient Generator

Data is generated using [Synthea](https://github.com/synthetichealth/synthea), an
open-source synthetic patient population simulator. Synthea creates realistic (but
not real) patient data including:

- Demographics (age, gender, location)
- Medical conditions with onset/resolution dates
- Observations (BMI, smoking status)
- Encounters and procedures

### Sleep Apnea Module

The sleep apnea disease progression is modeled using a Synthea Generic Module
(`complex_sleep_apnea.json`). Key characteristics:

**Risk Score Calculation**:
The module calculates a cumulative risk score based on:
- BMI >= 35 (severe obesity): +2 points
- BMI >= 30 (obese): +1 point
- Current smoker: +1 point
- Alcohol use disorder: +1 point
- Congestive heart failure: +2 points

**Prevalence by Risk and Gender**:

| Risk Score | Male Prevalence | Female Prevalence |
|------------|-----------------|-------------------|
| >= 4 | 60% | 40% |
| >= 2 | 44% | 24% |
| >= 1 | 32% | 16% |
| 0 | 26% | 12% |

This reflects epidemiological data showing higher sleep apnea rates in males and
individuals with obesity, smoking history, and cardiovascular disease.

**Diagnostic Pathway**:
1. Initial encounter for symptoms (snoring, daytime sleepiness)
2. Referral to sleep specialist
3. Sleep study (polysomnography or home testing)
4. Diagnosis and treatment (CPAP or oral appliance)

### Generated Population

"""

    # Add population stats
    report += f"""| Metric | Value |
|--------|-------|
| Total patients | {stats['n_patients']:,} |
| Urban | {stats['n_urban']:,} ({stats['pct_urban']:.1f}%) |
| Rural | {stats['n_rural']:,} ({stats['pct_rural']:.1f}%) |
| Male | {stats['n_male']:,} |
| Female | {stats['n_female']:,} |
"""

    if "n_apnea" in stats:
        report += f"""| Sleep apnea cases | {stats['n_apnea']:,} ({stats['pct_apnea']:.1f}%) |
"""

    report += """
---

## 3. Bias Application

### Rural Underdiagnosis Simulation

To simulate real-world underdiagnosis bias, we apply **label masking** to rural
patients with sleep apnea. This models the scenario where patients have the condition
but never receive a formal diagnosis due to barriers to care.

**Masking Process**:
1. Identify all rural patients with true sleep apnea diagnosis
2. Randomly select a portion (mask rate) to have their diagnosis "hidden"
3. Create two label sets:
   - `has_sleep_apnea`: True underlying condition
   - `observed_sleep_apnea`: What appears in medical records (true & ~masked)

"""

    if "n_masked" in stats:
        report += f"""### Masking Statistics

| Metric | Value |
|--------|-------|
| Rural patients with sleep apnea | {stats['n_rural_apnea']:,} |
| Patients masked (underdiagnosed) | {stats['n_masked']:,} |
| Effective mask rate | {stats['mask_rate']:.1f}% |

"""

    # Include bias effect details if available
    if bias_md:
        # Extract key tables from bias report
        report += """### Prevalence Impact

The masking creates a gap between true and observed prevalence, particularly
affecting rural populations:

"""
        # Try to extract the prevalence by location table
        location_table = extract_table_from_md(bias_md, "## Prevalence by Location")
        if location_table:
            report += location_table + "\n\n"

        gender_loc_table = extract_table_from_md(bias_md, "## Prevalence by Gender and Location")
        if gender_loc_table:
            report += "**By Gender and Location**:\n\n" + gender_loc_table + "\n\n"

    report += """
---

## 4. Model Training and Evaluation

### Approach

We train two Gradient Boosted Decision Tree (GBDT) models:

1. **Baseline Model**: Trained on true labels (`has_sleep_apnea`)
   - Represents the ideal scenario with complete diagnosis information

2. **Biased Model**: Trained on observed labels (`observed_sleep_apnea`)
   - Represents real-world scenario with underdiagnosis bias

Both models are evaluated against **true labels** to measure actual predictive
performance and fairness across subgroups.

### Features

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| male | Gender indicator (1 = male) |
| urban | Location indicator (1 = urban) |
| income | Household income (scaled) |
| bmi | Body mass index |
| smoker | Current smoker indicator |
| hypertension | Hypertension diagnosis |
| chf | Congestive heart failure diagnosis |
| alcohol_use | Alcohol use disorder diagnosis |

"""

    # Include model results if available
    if model_md:
        # Extract model specification from 3_model.md
        spec_table = extract_table_from_md(model_md, "## Model Specification")
        if spec_table:
            report += "### Model Specification\n\n" + spec_table + "\n\n"

        # Extract threshold selection info
        threshold_table = extract_table_from_md(model_md, "## Threshold Selection")
        if threshold_table:
            report += """### Threshold Selection

With ~9% class prevalence, the default 0.5 classification threshold would rarely
predict positives. We use **adaptive thresholding** to find the optimal operating
point that maximizes F1 score on the validation set.

""" + threshold_table + "\n\n"

        # Extract performance tables
        overall_table = extract_table_from_md(model_md, "## Overall Performance")
        if overall_table:
            report += "### Overall Performance\n\n" + overall_table + "\n\n"

        # Extract test set composition
        composition_table = extract_table_from_md(model_md, "### Test Set Composition")
        if composition_table:
            report += "### Test Set Composition\n\n" + composition_table + "\n\n"

        auc_table = extract_table_from_md(model_md, "### AUC-ROC by Subgroup")
        if auc_table:
            report += "### Subgroup AUC-ROC\n\n" + auc_table + "\n\n"

        recall_table = extract_table_from_md(model_md, "### Recall by Subgroup")
        if recall_table:
            report += "### Subgroup Recall\n\n" + recall_table + "\n\n"

        f1_table = extract_table_from_md(model_md, "### F1 Score by Subgroup")
        if f1_table:
            report += "### Subgroup F1 Score\n\n" + f1_table + "\n\n"

        # Extract key findings from model report
        key_findings = extract_key_findings(model_md)
        if key_findings:
            report += "### Key Model Findings\n\n" + key_findings + "\n\n"
    else:
        # Fallback if no model report
        report += """### Model Specification

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosted Decision Tree |
| n_estimators | 200 |
| max_depth | 5 |
| learning_rate | 0.05 |
| min_samples_split | 20 |
| min_samples_leaf | 10 |
| subsample | 0.8 |

*Note: Run 3_train_models.py to generate detailed model results.*

"""

    # Extract recall metrics for dynamic findings if model report exists
    findings_content = ""
    if model_md:
        urban_recall_base = extract_metric_from_table(model_md, "### Recall by Subgroup", "Urban", 0)
        urban_recall_bias = extract_metric_from_table(model_md, "### Recall by Subgroup", "Urban", 1)
        rural_recall_base = extract_metric_from_table(model_md, "### Recall by Subgroup", "Rural", 0)
        rural_recall_bias = extract_metric_from_table(model_md, "### Recall by Subgroup", "Rural", 1)

        if all(v is not None for v in [urban_recall_base, urban_recall_bias, rural_recall_base, rural_recall_bias]):
            urban_recall_delta = urban_recall_bias - urban_recall_base
            rural_recall_delta = rural_recall_bias - rural_recall_base
            rural_recall_pct_drop = 100 * rural_recall_delta / rural_recall_base if rural_recall_base else 0

            findings_content = f"""
1. **Rural Recall Collapse**: The biased model's recall for rural patients drops
   dramatically ({rural_recall_base:.1%} → {rural_recall_bias:.1%}, {rural_recall_delta:+.1%}), meaning it
   misses {-rural_recall_pct_drop:.0f}% more true sleep apnea cases in rural populations.

2. **Urban Recall Improvement**: Meanwhile, urban recall actually increases
   ({urban_recall_base:.1%} → {urban_recall_bias:.1%}, {urban_recall_delta:+.1%}), as the model shifts its
   predictions toward the majority group with complete labels.

3. **Disparity Amplification**: The recall gap between urban and rural widens from
   {(urban_recall_base - rural_recall_base):+.1%} to {(urban_recall_bias - rural_recall_bias):+.1%}, showing how
   training on biased data amplifies existing healthcare inequities.
"""
        else:
            findings_content = """
1. **Rural Recall Drop**: The biased model catches fewer true sleep apnea cases
   in rural populations, as it learns from data where rural cases are underrepresented.

2. **Urban Performance Stable/Improved**: Urban predictions remain accurate or improve,
   as the training data fully represents urban sleep apnea cases.

3. **Fairness Gap Widens**: The performance disparity between subgroups increases,
   demonstrating how label bias compounds into prediction bias.
"""
    else:
        findings_content = """
1. **Expected Rural Recall Drop**: Models trained on biased data will underpredict
   sleep apnea in rural populations where diagnoses were masked.

2. **Fairness Gap**: Performance disparity between urban and rural subgroups
   is expected to widen under the biased model.

3. **Run model training** (script 3) to see quantified results.
"""

    report += f"""
---

## 5. Key Findings

### Impact of Underdiagnosis Bias
{findings_content}
### Why Recall Matters More Than AUC

- **AUC** measures ranking ability across all thresholds - it may remain stable
  even when the model systematically underpredicts for a subgroup.
- **Recall** measures how many true positives are caught at the operating threshold -
  this directly translates to missed diagnoses in clinical deployment.
- A model with good AUC but poor rural recall will still fail rural patients.

### Implications

- **Model Deployment Risk**: Deploying models trained on biased data perpetuates
  underdiagnosis in already underserved populations.

- **Data Quality Matters**: "Ground truth" labels from EHR data may reflect access
  patterns rather than true disease prevalence.

- **Fairness Monitoring**: Subgroup recall and F1 metrics are essential for detecting
  bias - overall AUC alone is insufficient.

### Mitigation Strategies

1. **Active case finding** in underserved populations to improve label quality
2. **Subgroup-stratified evaluation** with recall/F1 metrics, not just AUC
3. **Fairness-aware training** with constraints on subgroup performance parity
4. **Regular audits** comparing model predictions to external prevalence estimates

---

## 6. Conclusion

This case study demonstrates how structural barriers to healthcare access create
biased training data that, when used for ML model development, can perpetuate and
amplify existing health disparities. Rural underdiagnosis of sleep apnea is just
one example; similar patterns exist across many conditions and populations.

Responsible ML development in healthcare requires:
- Understanding the data generation process and its biases
- Evaluating models on true outcomes when possible
- Monitoring fairness across relevant subgroups
- Implementing mitigation strategies for known biases

---

## Appendix: Pipeline Execution

```bash
# 1. Generate synthetic population (Vermont, ages 60-100)
uv run python scripts/1_generate_data.py -p <N> -s 42

# 2. Apply rural underdiagnosis bias (30% mask rate)
uv run python scripts/2_gen_bias.py --mask-rate 0.3

# 3. Train and evaluate models
uv run python scripts/3_train_models.py

# 4. Generate this report
uv run python scripts/4_create_report.py
```

**This run**: """ + f"{stats['n_patients']:,}" + """ patients generated

---

*This report was generated as part of the Synthea Bias Case Study project.*
"""

    return report


def main():
    print("=" * 60)
    print("Sleep Apnea v2: Generate Report")
    print("=" * 60)

    # Check for required data
    if not (DATA_DIR / "data.csv").exists():
        print("Error: data.csv not found. Run 1_generate_data.py and 2_gen_bias.py first.")
        return

    print("\nReading data and generating report...")
    report = generate_report()

    # Write report
    REPORT_PATH.write_text(report)
    print(f"\nWrote {REPORT_PATH}")

    # Report status of input files
    print("\nInput files status:")
    for name, path in [
        ("Summary stats", SUMMARY_STATS_PATH),
        ("Bias effect", BIAS_EFFECT_PATH),
        ("Model results", MODEL_PATH),
    ]:
        status = "found" if path.exists() else "not found (section will be minimal)"
        print(f"  {name}: {status}")

    print("\n" + "=" * 60)
    print(f"Complete! Report saved to {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
