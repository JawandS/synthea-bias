# Diabetes Case Study v2: Label Masking Bias

Demonstrates how documentation bias in metabolic conditions affects ML model performance. Generates a baseline population via Synthea (Montana, ages 40-100), then masks a portion of hyperglycemia/hypertriglyceridemia diagnoses to simulate under-documentation.

## Quick Start

```bash
# 1. Generate baseline data (~20k patients)
uv run python scripts/1_generate_data.py -p 20000 -s 160

# 2. Apply bias (30% of hyperglycemia/hypertriglyceridemia masked)
uv run python scripts/2_gen_bias.py --mask-rate 0.3

# 3. Train and evaluate models
uv run python scripts/3_train_models.py

# 4. Generate comprehensive report
uv run python scripts/4_create_report.py
```

## Output Structure

```
output/
├── data/
│   ├── patients.csv      # Demographics + canonical condition flags
│   ├── conditions.csv    # All conditions (source for diagnoses)
│   └── observations.csv  # A1c, BMI, smoking status
├── info/
│   ├── 1_summary_stats.md  # Population statistics
│   ├── 2_bias_effect.md    # Before/after bias analysis
│   └── 3_model.md          # Model training results
└── report.md               # Complete case study report
```

## Canonical Condition Flags

After running `2_gen_bias.py`, `patients.csv` contains:

| Column | Description |
|--------|-------------|
| `has_diabetes` | **Target** - patient has diabetes diagnosis (SNOMED 44054006) |
| `has_hyperglycemia` | **True label** - patient has hyperglycemia (SNOMED 80394007) |
| `has_hypertriglyceridemia` | **True label** - patient has hypertriglyceridemia (SNOMED 302870006) |
| `mask_hyperglycemia` | Whether hyperglycemia is masked (under-documented) |
| `mask_hypertriglyceridemia` | Whether hypertriglyceridemia is masked (under-documented) |

**For modeling**: Use `has_hyperglycemia & ~mask_hyperglycemia` as the observed/biased feature.

## Scripts

| Script | Purpose |
|--------|---------|
| `1_generate_data.py` | Run Synthea (Montana), extract relevant CSVs, output summary stats |
| `2_gen_bias.py` | Add condition flags and mask columns for documentation bias |
| `3_train_models.py` | Train/evaluate GBDT models on true vs biased features |
| `4_create_report.py` | Generate comprehensive `report.md` case study |

## Key Codes

| Condition | Code |
|-----------|------|
| Diabetes mellitus type 2 | SNOMED 44054006 |
| Hyperglycemia | SNOMED 80394007 |
| Hypertriglyceridemia | SNOMED 302870006 |
| Hemoglobin A1c | LOINC 4548-4 |
| BMI | LOINC 39156-5 |
