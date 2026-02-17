# Intersectional Case Study v1: Age-Income Access Bias

This case study demonstrates how age and income can jointly shape observed colorectal cancer outcomes through differential screening access.
The modeling objective is to recommend individuals at high risk of CRC for screening, with a secondary goal of catching cases early.

- **True outcomes** are derived from colorectal stage diagnosis codes.
- **Observed outcomes** are created by masking true outcomes using a plan policy table that depends on age and income.
- **Equity framing**: model features exclude wealth-related policy variables, while the biased training labels still encode historical access disadvantage for lower-income patients.

## Quick Start

```bash
# 1) Generate baseline Synthea data (Montana, age 40-100)
uv run python scripts/1_generate_data.py -p 20000 -s 160

# 2) Apply intersectional masking policy
uv run python scripts/2_gen_bias.py --seed 42

# 3) Train screening recommendation models (equity baseline vs historically biased labels)
uv run python scripts/3_train_models.py --seed 42

# 4) Build full report
uv run python scripts/4_create_report.py
```

## Pipeline Walkthrough

### `scripts/1_generate_data.py` - Generate + filter baseline inputs
- Runs Synthea (unless `--skip-synthea`) for a Montana cohort age 40-100.
- Copies `patients.csv`, `conditions.csv`, and `observations.csv` into `output/data/`.
- Filters:
  - conditions to CRC-stage + model-comorbidity codes
  - observations to BMI + smoking codes
- Writes baseline summary stats to `output/info/1_summary_stats.md`.

### `scripts/2_gen_bias.py` - Build labels + apply access bias policy
- Loads filtered baseline CSVs from `output/data/`.
- Builds consolidated patient-level features:
  - demographics (`age`, `male`, `income`)
  - clinical features (comorbidity flags, latest BMI, smoker flag)
  - true outcomes (`crc_stage_true`, `has_crc_true`, `has_early_crc_true`)
- Assigns one policy plan per patient using `config/plan_rules.csv` (age + income ranges).
- Computes masking mechanics:
  - eligibility (`eligible_for_screening`)
  - age multiplier by band
  - effective mask rates (`effective_mask_rate_crc`, `effective_mask_rate_early`)
  - sampled mask outcomes (`mask_crc`, `mask_early`)
- Produces observed labels (`observed_crc`, `observed_early_crc`).
- Writes final dataset to `output/data/data.csv` and bias summary to `output/info/2_bias_effect.md`.
- Removes intermediate raw filtered CSVs (`patients.csv`, `conditions.csv`, `observations.csv`) from `output/data/`.

### `scripts/3_train_models.py` - Train baseline vs biased models
- Loads `output/data/data.csv`.
- Trains two tasks:
  - CRC screening recommendation (risk of CRC)
  - Early-stage screening recommendation (catch early CRC)
- For each task, trains:
  - equity-oriented baseline model on true labels
  - historically biased model on observed labels
- Evaluates both against true labels on the same held-out test split.
- Uses clinical + demographic features only (no income/policy variables), then reports subgroup performance by income to quantify historical-practice bias carryover.
- Writes:
  - narrative results to `output/info/3_model.md`
  - aggregate metrics to `output/models/metrics.csv`
  - subgroup metrics to `output/models/subgroup_metrics.csv`.

### `scripts/4_create_report.py` - Assemble final case-study report
- Loads generated markdown artifacts from scripts 1-3.
- Reads `config/plan_rules.csv` and formats policy table.
- Produces a single end-to-end report at `output/report.md`.

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

`config/plan_rules.csv` stores base policy rates by age-income strata, not final per-patient rates.
Each patient is first assigned a plan using income + age ranges, then receives row-level effective
mask rates after applying age-band multipliers.

### Derived Masking Variables
- `age_multiplier`: age-band multiplier applied to mask rates
- `effective_mask_rate_crc`: `mask_rate_crc * age_multiplier` (clipped to `[0,1]`)
- `effective_mask_rate_early`: `mask_rate_early * age_multiplier` (clipped to `[0,1]`)
- `mask_crc`: sampled mask outcome for CRC label
- `mask_early`: sampled mask outcome for early-stage label

These values are computed per individual because masking is applied at the patient level
(after assigning a plan from age + income), not as a single constant per rule table row.

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

Training excludes direct policy/bias variables for equity: `income, assigned_plan, eligible_for_screening`.

The biased model is still expected to underperform for poorer populations because it is trained on
historically masked observed labels created by access barriers in `plan_rules.csv`.

Intermediate filtered files (`patients.csv`, `conditions.csv`, `observations.csv`) are removed after `2_gen_bias.py` to keep only the consolidated dataset.
