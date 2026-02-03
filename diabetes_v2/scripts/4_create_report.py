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
            if line.startswith("## ") or line.startswith("---"):
                break
            findings_lines.append(line)

    content = "\n".join(findings_lines).strip()
    return content if content else None


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

    if "has_hypertriglyceridemia" in df.columns:
        stats["n_hypertriglyceridemia"] = int(df["has_hypertriglyceridemia"].sum())
        stats["pct_hypertriglyceridemia"] = 100 * stats["n_hypertriglyceridemia"] / stats["n_patients"]

    if "mask_hypertriglyceridemia" in df.columns:
        stats["n_masked"] = int(df["mask_hypertriglyceridemia"].sum())
        n_ht = int(df["has_hypertriglyceridemia"].sum())
        stats["mask_rate"] = 100 * stats["n_masked"] / n_ht if n_ht else 0

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
incomplete documentation of hypertriglyceridemia (elevated triglycerides) reduces the
apparent association between this metabolic condition and diabetes.

When ML models are trained on data with incomplete documentation, they learn weaker
associations for the under-documented features, potentially relying more heavily on
other features like A1c, BMI, or demographics.

### The Problem

Diabetes affects approximately 10-12% of the adult population. Hypertriglyceridemia
(elevated triglycerides, SNOMED 302870006) is a key metabolic risk marker:

- Part of **metabolic syndrome** cluster
- Signals underlying insulin resistance
- Often co-occurs with or precedes diabetes
- Distinct from diabetes itself (unlike hyperglycemia)

Hypertriglyceridemia may be under-documented in EHR data due to:
- Labs show elevated triglycerides but diagnosis code not entered
- Time pressure during clinical encounters
- Focus on primary diagnosis rather than secondary findings
- Variability in documentation practices across providers

This creates **documentation bias**: the condition exists but isn't recorded,
weakening its apparent predictive signal for diabetes.

### Study Design

1. **Generate synthetic population** with realistic diabetes and hypertriglyceridemia prevalence
2. **Apply documentation bias** by randomly masking a portion of hypertriglyceridemia diagnoses
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

The metabolic syndrome disease progression is modeled using a Synthea Generic Module.
Key characteristics:

**Risk Factors**:
- BMI >= 30 (obesity)
- Age > 40 years
- Sedentary lifestyle
- Family history

**Metabolic Conditions**:
- Hypertriglyceridemia (elevated triglycerides)
- Prediabetes
- Type 2 diabetes

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
    if "n_hypertriglyceridemia" in stats:
        report += f"""| Hypertriglyceridemia cases | {stats['n_hypertriglyceridemia']:,} ({stats['pct_hypertriglyceridemia']:.1f}%) |
"""

    report += """
---

## 3. Bias Application

### Documentation Bias Simulation

To simulate real-world documentation bias, we apply **random masking** to
hypertriglyceridemia diagnoses. This models the scenario where patients have
the condition but it's not recorded due to documentation gaps.

**Masking Process**:
1. Identify all patients with true hypertriglyceridemia diagnosis
2. Randomly select a portion (mask rate) to have their diagnosis "hidden"
3. Create two feature versions:
   - `has_hypertriglyceridemia`: True underlying condition status
   - `observed_hypertriglyceridemia`: What appears in medical records

**Key Characteristic**:
Unlike demographic-based bias (e.g., rural underdiagnosis), documentation bias
affects patients randomly across all groups. This isolates the effect of
incomplete feature information from confounding demographic factors.

"""

    if "n_masked" in stats:
        report += f"""### Masking Statistics

| Metric | Value |
|--------|-------|
| Patients with hypertriglyceridemia | {stats['n_hypertriglyceridemia']:,} |
| Patients masked (under-documented) | {stats['n_masked']:,} |
| Effective mask rate | {stats['mask_rate']:.1f}% |

"""

    # Include bias effect details if available
    if bias_md:
        report += """### Impact on Observed Prevalence

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

1. **Baseline Model**: Uses true feature (`has_hypertriglyceridemia`)
   - Represents the ideal scenario with complete documentation

2. **Biased Model**: Uses observed feature (`observed_hypertriglyceridemia`)
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
| hypertriglyceridemia | Hypertriglyceridemia (true or observed) |

"""

    # Include model results if available
    if model_md:
        spec_table = extract_table_from_md(model_md, "## Model Specification")
        if spec_table:
            report += "### Model Specification\n\n" + spec_table + "\n\n"

        overall_table = extract_table_from_md(model_md, "## Overall Performance")
        if overall_table:
            report += "### Overall Performance\n\n" + overall_table + "\n\n"

        importance_table = extract_table_from_md(model_md, "## Feature Importance")
        if importance_table:
            report += "### Feature Importance\n\n" + importance_table + "\n\n"

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

*Note: Run 3_train_models.py to generate detailed model results.*

"""

    report += """
---

## 5. Key Findings

### Impact of Documentation Bias

1. **Reduced Feature Importance**: The biased model assigns lower importance to
   the hypertriglyceridemia feature, as masked values dilute the signal.

2. **Compensatory Reliance**: Other features (A1c, BMI, prediabetes) may see
   increased importance as the model compensates for the weakened signal.

3. **Clinical Validity**: Unlike hyperglycemia (which is definitionally diabetes),
   hypertriglyceridemia is a genuine risk marker - making this a clinically
   realistic documentation bias scenario.

### Why This Matters

- **Different from Label Bias**: Documentation bias affects *features* while the
  target remains correct. The model learns from incomplete information rather
  than incorrect labels.

- **Feature Importance Shifts**: Models may over-rely on well-documented features
  and under-utilize poorly documented but clinically important ones.

- **Generalization Risk**: A model trained on data with documentation bias may
  perform differently in settings with better or worse documentation practices.

### Implications for ML in Healthcare

- **Data Quality**: Feature completeness matters as much as label accuracy
- **Institutional Variation**: Documentation practices vary - models may not generalize
- **Monitoring**: Track documentation rates for key predictive features
- **Robustness**: Evaluate models under varying documentation completeness

### Mitigation Strategies

1. **Documentation improvement** initiatives at the clinical level
2. **Feature imputation** based on related lab values when diagnoses are missing
3. **Sensitivity analysis** to understand model behavior under documentation gaps
4. **Multi-source validation** using data with different documentation practices

---

## 6. Conclusion

This case study demonstrates how incomplete documentation of clinical findings
creates biased training data that affects ML model behavior. By focusing on
hypertriglyceridemia (a genuine risk marker rather than a defining characteristic),
we isolate the effect of documentation bias in a clinically realistic scenario.

Key takeaways:

- Documentation gaps are measurement error in feature values
- Feature importance shifts away from under-documented conditions
- Models may not generalize across documentation practices
- Monitoring feature completeness is essential for reliable ML deployment

---

## Appendix: Pipeline Execution

```bash
# 1. Generate synthetic population (Montana, ages 40-100)
uv run python scripts/1_generate_data.py -p 20000 -s 160

# 2. Apply documentation bias (30% mask rate)
uv run python scripts/2_gen_bias.py

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
