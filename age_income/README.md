# Intersectional Case Study v1: Age-Income Access Bias

This case study demonstrates how age and income can jointly shape observed colorectal cancer outcomes through differential screening access.

- **True outcomes** are derived from colorectal stage diagnosis codes.
- **Observed outcomes** are created by masking true outcomes using a plan policy table that depends on age and income.

## Quick Start

```bash
# 1) Generate baseline Synthea data (Montana, age 40-100)
uv run python scripts/1_generate_data.py -p 20000 -s 160

# 2) Apply intersectional masking policy
uv run python scripts/2_gen_bias.py --seed 42

# 3) Train baseline vs biased models
uv run python scripts/3_train_models.py --seed 42

# 4) Build full report
uv run python scripts/4_create_report.py
```

## Output Structure

```
output/
├── data/
│   └── data.csv
├── info/
│   ├── 1_summary_stats.md
│   ├── 2_bias_effect.md
│   └── 3_model.md
├── models/
│   ├── metrics.csv
│   └── subgroup_metrics.csv
└── report.md
```

## Canonical Columns (`output/data/data.csv`)

| Column | Meaning |
|---|---|
| `has_crc_true` | True CRC diagnosis indicator |
| `crc_stage_true` | True maximum stage (1-4, 0 if none) |
| `has_early_crc_true` | True early-stage CRC indicator (stage I/II) |
| `assigned_plan` | Plan assigned by age-income rule table |
| `eligible_for_screening` | Whether age meets plan screening threshold |
| `mask_crc` | Whether CRC diagnosis was masked |
| `mask_early` | Whether early-stage flag was masked |
| `observed_crc` | Biased observed CRC diagnosis |
| `observed_early_crc` | Biased observed early-stage outcome |

## Policy Configuration

Edit `config/plan_rules.csv` to change:
- plan assignment ranges (`income_min/max`, `age_min/max`)
- screening eligibility age (`screening_start_age`)
- masking severity (`mask_rate_crc`, `mask_rate_early`)

The rules must cover all generated patients with no overlap.

## Modeling Features

Training uses: `age, male, bmi, smoker, diabetes, prediabetes, obesity, hypertension, hyperlipidemia, chf`.

Training excludes direct policy/bias variables: `income, poverty_ratio, assigned_plan, eligible_for_screening`.

Intermediate filtered files (`patients.csv`, `conditions.csv`, `observations.csv`) are removed after `2_gen_bias.py` to keep only the consolidated dataset.
