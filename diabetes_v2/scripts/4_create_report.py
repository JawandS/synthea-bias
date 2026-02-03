#!/usr/bin/env python3
"""
4_create_report.py - Generate comprehensive case study report.

This script compiles all analysis outputs into a single report.md file that
documents the complete diabetes documentation bias case study.

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
        "n_male": int((df["male"] == 1).sum()),
        "n_female": int((df["male"] == 0).sum()),
    }

    stats["pct_male"] = 100 * stats["n_male"] / stats["n_patients"]
    stats["pct_female"] = 100 * stats["n_female"] / stats["n_patients"]

    if "has_diabetes" in df.columns:
        stats["n_diabetes"] = int(df["has_diabetes"].sum())
        stats["pct_diabetes"] = 100 * stats["n_diabetes"] / stats["n_patients"]

    if "has_hyperglycemia" in df.columns:
        stats["n_hyperglycemia"] = int(df["has_hyperglycemia"].sum())
        stats["pct_hyperglycemia"] = 100 * stats["n_hyperglycemia"] / stats["n_patients"]

    if "has_hypertriglyceridemia" in df.columns:
        stats["n_hypertriglyceridemia"] = int(df["has_hypertriglyceridemia"].sum())
        stats["pct_hypertriglyceridemia"] = 100 * stats["n_hypertriglyceridemia"] / stats["n_patients"]

    if "mask_hyperglycemia" in df.columns:
        stats["n_masked_hg"] = int(df["mask_hyperglycemia"].sum())
        n_hg = int(df["has_hyperglycemia"].sum())
        stats["mask_rate_hg"] = 100 * stats["n_masked_hg"] / n_hg if n_hg else 0

    if "mask_hypertriglyceridemia" in df.columns:
        stats["n_masked_ht"] = int(df["mask_hypertriglyceridemia"].sum())
        n_ht = int(df["has_hypertriglyceridemia"].sum())
        stats["mask_rate_ht"] = 100 * stats["n_masked_ht"] / n_ht if n_ht else 0

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

    report = f"""# Diabetes Documentation Bias: A Case Study

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Overview

### Study Goal

This case study demonstrates how **documentation bias** in electronic health records can
affect machine learning models that predict diabetes. Specifically, we examine how
incomplete documentation of metabolic conditions (hyperglycemia and hypertriglyceridemia)
reduces the apparent association between these conditions and diabetes.

When ML models are trained on data with incomplete documentation, they learn weaker
associations for the under-documented features, potentially relying more heavily on
other features like demographics or lab values.

### The Problem

Diabetes affects approximately 10-12% of the adult population. Several metabolic
conditions are highly predictive of diabetes:

- **Hyperglycemia**: Elevated blood sugar levels (SNOMED 80394007)
- **Hypertriglyceridemia**: Elevated triglyceride levels (SNOMED 302870006)

These conditions may be under-documented in EHR data due to:
- Time pressure during clinical encounters
- Variability in documentation practices across providers
- Incomplete lab panels or follow-up testing
- EHR usability issues and alert fatigue
- Focus on primary diagnosis rather than secondary findings

This creates **documentation bias**: patients may have these conditions but they
are not recorded in the medical record, weakening the apparent predictive signal.

### Study Design

1. **Generate synthetic population** with realistic diabetes and metabolic condition prevalence
2. **Apply documentation bias** by randomly masking a portion of metabolic condition diagnoses
3. **Train ML models** using both true and observed (biased) features
4. **Compare performance** to quantify the bias impact on feature importance and predictions

---

## 2. Data Generation

### Synthea Patient Generator

Data is generated using [Synthea](https://github.com/synthetichealth/synthea), an
open-source synthetic patient population simulator. Synthea creates realistic (but
not real) patient data including:

- Demographics (age, gender, income)
- Medical conditions with onset/resolution dates
- Observations (A1c, BMI, smoking status)
- Encounters and procedures

### Metabolic Syndrome Module

The metabolic syndrome disease progression is modeled using a Synthea Generic Module
(`metabolic_syndrome_care.json`). Key characteristics:

**Risk Factors**:
- BMI >= 30 (obesity)
- Family history of diabetes
- Age > 40 years
- Sedentary lifestyle

**Progression Pathway**:
1. Metabolic risk factor accumulation
2. Development of prediabetes
3. Potential progression to type 2 diabetes
4. Associated conditions: hyperglycemia, hypertriglyceridemia, hypertension

### Generated Population

"""

    # Add population stats
    report += f"""| Metric | Value |
|--------|-------|
| Total patients | {stats['n_patients']:,} |
| Male | {stats['n_male']:,} ({stats['pct_male']:.1f}%) |
| Female | {stats['n_female']:,} ({stats['pct_female']:.1f}%) |
"""

    if "n_diabetes" in stats:
        report += f"""| Diabetes cases | {stats['n_diabetes']:,} ({stats['pct_diabetes']:.1f}%) |
"""
    if "n_hyperglycemia" in stats:
        report += f"""| Hyperglycemia cases | {stats['n_hyperglycemia']:,} ({stats['pct_hyperglycemia']:.1f}%) |
"""
    if "n_hypertriglyceridemia" in stats:
        report += f"""| Hypertriglyceridemia cases | {stats['n_hypertriglyceridemia']:,} ({stats['pct_hypertriglyceridemia']:.1f}%) |
"""

    report += """
---

## 3. Bias Application

### Documentation Bias Simulation

To simulate real-world documentation bias, we apply **random masking** to
hyperglycemia and hypertriglyceridemia diagnoses. This models the scenario where
patients have these conditions but they are not recorded due to documentation gaps.

**Masking Process**:
1. Identify all patients with true hyperglycemia/hypertriglyceridemia diagnosis
2. Randomly select a portion (mask rate) to have their diagnosis "hidden"
3. Create two feature sets:
   - `has_*`: True underlying condition status
   - `observed_*`: What appears in medical records (true & ~masked)

**Key Difference from Label Bias**:
Unlike underdiagnosis bias which affects the target variable, documentation bias
affects *features* used to predict the target. The diabetes diagnosis itself
remains accurate - only the metabolic condition features are masked.

"""

    if "n_masked_hg" in stats or "n_masked_ht" in stats:
        report += """### Masking Statistics

| Condition | Total Cases | Masked | Effective Mask Rate |
|-----------|-------------|--------|---------------------|
"""
        if "n_masked_hg" in stats:
            report += f"""| Hyperglycemia | {stats['n_hyperglycemia']:,} | {stats['n_masked_hg']:,} | {stats['mask_rate_hg']:.1f}% |
"""
        if "n_masked_ht" in stats:
            report += f"""| Hypertriglyceridemia | {stats['n_hypertriglyceridemia']:,} | {stats['n_masked_ht']:,} | {stats['mask_rate_ht']:.1f}% |
"""
        report += "\n"

    # Include bias effect details if available
    if bias_md:
        report += """### Impact on Observed Prevalence

Documentation bias reduces the apparent prevalence of metabolic conditions:

"""
        overall_table = extract_table_from_md(bias_md, "## Overall Effect")
        if overall_table:
            report += overall_table + "\n\n"

        assoc_table = extract_table_from_md(bias_md, "## Effect on Diabetes Association")
        if assoc_table:
            report += "**Effect on Diabetes Association**:\n\n" + assoc_table + "\n\n"

    report += """
---

## 4. Model Training and Evaluation

### Approach

We train two Gradient Boosted Decision Tree (GBDT) models:

1. **Baseline Model**: Uses true features (`has_hyperglycemia`, `has_hypertriglyceridemia`)
   - Represents the ideal scenario with complete documentation

2. **Biased Model**: Uses observed features (`observed_hyperglycemia`, `observed_hypertriglyceridemia`)
   - Represents real-world scenario with documentation bias

Both models predict the **same target** (`has_diabetes`) and are evaluated on
identical test data to measure the impact of feature bias.

### Features

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| male | Gender indicator (1 = male) |
| income | Household income (scaled) |
| a1c | Hemoglobin A1c level |
| bmi | Body mass index |
| smoker | Current smoker indicator |
| prediabetes | Prediabetes diagnosis |
| obesity | Obesity diagnosis |
| hypertension | Hypertension diagnosis |
| hyperlipidemia | Hyperlipidemia diagnosis |
| hyperglycemia | Hyperglycemia (true or observed) |
| hypertriglyceridemia | Hypertriglyceridemia (true or observed) |

"""

    # Include model results if available
    if model_md:
        # Extract model specification
        spec_table = extract_table_from_md(model_md, "## Model Specification")
        if spec_table:
            report += "### Model Specification\n\n" + spec_table + "\n\n"

        # Extract threshold info
        threshold_table = extract_table_from_md(model_md, "## Threshold Selection")
        if threshold_table:
            report += "### Threshold Selection\n\n" + threshold_table + "\n\n"

        # Extract performance tables
        overall_table = extract_table_from_md(model_md, "## Overall Performance")
        if overall_table:
            report += "### Overall Performance\n\n" + overall_table + "\n\n"

        # Extract feature importance
        importance_table = extract_table_from_md(model_md, "## Feature Importance")
        if importance_table:
            report += "### Feature Importance\n\n" + importance_table + "\n\n"

        # Extract subgroup tables
        composition_table = extract_table_from_md(model_md, "### Test Set Composition by Metabolic Status")
        if composition_table:
            report += "### Subgroup Composition (by True Metabolic Status)\n\n" + composition_table + "\n\n"

        recall_table = extract_table_from_md(model_md, "### Recall by Subgroup")
        if recall_table:
            report += "### Subgroup Recall\n\n" + recall_table + "\n\n"

        # Extract key findings
        key_findings = extract_key_findings(model_md)
        if key_findings:
            report += "### Key Model Findings\n\n" + key_findings + "\n\n"
    else:
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

    # Generate findings section
    if model_md:
        findings_content = """
1. **Reduced Feature Importance**: The biased model assigns lower importance to
   hyperglycemia and hypertriglyceridemia features, as the masked values dilute
   the apparent signal.

2. **Compensatory Reliance**: Other features (like A1c, BMI, or demographics) may
   see increased importance as the model compensates for the weakened metabolic signals.

3. **Potential Subgroup Effects**: Patients who truly have metabolic conditions
   may see different prediction patterns, particularly if documentation rates
   vary by demographic factors.
"""
    else:
        findings_content = """
1. **Expected Feature Importance Reduction**: Models trained with masked features
   will learn weaker associations for hyperglycemia and hypertriglyceridemia.

2. **Run model training** (script 3) to see quantified results.
"""

    report += f"""
---

## 5. Key Findings

### Impact of Documentation Bias
{findings_content}
### Why Documentation Bias Matters

- **Different from Label Bias**: Unlike underdiagnosis bias (where the target is
  wrong), documentation bias affects features while the target remains correct.

- **Feature Importance Shifts**: The model may learn to rely more heavily on
  other features, changing its behavior in populations with different documentation
  practices.

- **Generalization Risk**: A model trained on data with documentation bias may
  perform differently when deployed in settings with better or worse documentation.

### Implications

- **Model Robustness**: Models should be evaluated on data with varying
  documentation quality to understand sensitivity.

- **Feature Quality Monitoring**: Track documentation completeness for key
  predictive features, not just the target variable.

- **Institutional Variation**: Documentation practices vary across providers,
  departments, and EHR systems - models may not generalize well.

### Mitigation Strategies

1. **Documentation improvement initiatives** to increase capture of metabolic conditions
2. **Feature imputation** based on lab values when diagnoses are missing
3. **Multi-task learning** to jointly predict both the target and missing features
4. **Sensitivity analysis** to understand model behavior under varying documentation rates

---

## 6. Conclusion

This case study demonstrates how incomplete documentation of clinical findings
creates biased training data that affects ML model behavior. Unlike label bias
which directly corrupts the target, documentation bias works through features,
reducing the apparent predictive power of under-documented conditions.

Key takeaways:

- Documentation gaps are a form of measurement error in feature values
- Feature importance can shift away from under-documented conditions
- Models may not generalize to settings with different documentation practices
- Monitoring documentation completeness is essential for reliable ML deployment

---

## Appendix: Pipeline Execution

```bash
# 1. Generate synthetic population (Montana, ages 40-100)
uv run python scripts/1_generate_data.py -p 20000 -s 160

# 2. Apply documentation bias (30% mask rate)
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
    print("Diabetes v2: Generate Report")
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
