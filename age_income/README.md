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

## Column Overview (`output/data/data.csv`)

### Identity + Demographics
- `id`: patient identifier
- `age`: age at reference date
- `male`: sex flag (`1` male, `0` otherwise)
- `income`: annual income

### True Labels
- `crc_stage_true`: true max CRC stage (`1-4`, `0` if none)
- `has_crc_true`: true CRC diagnosis flag
- `has_early_crc_true`: true early-stage CRC flag (stage I/II)

### Clinical Features
- `diabetes`, `prediabetes`, `obesity`, `hypertension`, `hyperlipidemia`, `chf`: comorbidity flags
- `bmi`: latest BMI value
- `smoker`: smoking flag

### Plan Policy Inputs
- `assigned_plan`: plan assigned by age-income rules
- `screening_start_age`: plan-specific screening threshold
- `mask_rate_crc`: base CRC mask rate from policy table
- `mask_rate_early`: base early-CRC mask rate from policy table
- `eligible_for_screening`: whether age meets plan threshold

### Derived Masking Variables
- `age_multiplier`: age-band multiplier applied to mask rates
- `effective_mask_rate_crc`: `mask_rate_crc * age_multiplier` (clipped to `[0,1]`)
- `effective_mask_rate_early`: `mask_rate_early * age_multiplier` (clipped to `[0,1]`)
- `mask_crc`: sampled mask outcome for CRC label
- `mask_early`: sampled mask outcome for early-stage label

### Observed (Biased) Labels
- `observed_crc`: observed CRC after masking
- `observed_early_crc`: observed early-stage CRC after masking

## Policy Configuration

Edit `config/plan_rules.csv` to change:
- plan assignment ranges (`income_min/max`, `age_min/max`)
- screening eligibility age (`screening_start_age`)
- masking severity (`mask_rate_crc`, `mask_rate_early`)

The rules must cover all generated patients with no overlap.

## Modeling Features

Training uses: `age, male, bmi, smoker, diabetes, prediabetes, obesity, hypertension, hyperlipidemia, chf`.

Training excludes direct policy/bias variables: `income, assigned_plan, eligible_for_screening`.

Intermediate filtered files (`patients.csv`, `conditions.csv`, `observations.csv`) are removed after `2_gen_bias.py` to keep only the consolidated dataset.
