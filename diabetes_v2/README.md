# Diabetes Case Study v2: Documentation Bias

Demonstrates how documentation bias affects ML model performance. Generates a baseline population via Synthea (Montana, ages 40-100), then masks a portion of hypertriglyceridemia diagnoses to simulate under-documentation of this metabolic risk factor.

## Quick Start

```bash
# 1. Generate baseline data (~20k patients)
uv run python scripts/1_generate_data.py -p 20000 -s 160

# 2. Apply bias (30% of hypertriglyceridemia masked)
uv run python scripts/2_gen_bias.py

# 3. Train and evaluate models
uv run python scripts/3_train_models.py

# 4. Generate comprehensive report
uv run python scripts/4_create_report.py
```

## Quick Start (with cost data)

```bash
# 1. Generate baseline data (~20k patients)
uv run python scripts/1_generate_data.py -p 20000 -s 160

# 1b. (Optional) Patch in per-patient cost features from Synthea encounters
uv run python scripts/1b_add_cost_data.py

# 2. Apply bias (30% of hypertriglyceridemia masked)
uv run python scripts/2_gen_bias.py

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

After running `2_gen_bias.py`, `data.csv` contains:

| Column | Description |
|--------|-------------|
| `has_diabetes` | **Target** - patient has diabetes diagnosis (SNOMED 44054006) |
| `has_hypertriglyceridemia` | **True feature** - patient has hypertriglyceridemia (SNOMED 302870006) |
| `mask_hypertriglyceridemia` | Whether hypertriglyceridemia is masked (under-documented) |
| `observed_hypertriglyceridemia` | Biased feature for modeling (`has & ~mask`) |

## Scripts

| Script | Purpose |
|--------|---------|
| `1_generate_data.py` | Run Synthea (Montana), extract relevant CSVs, output summary stats |
| `1b_add_cost_data.py` | Patch per-patient cost features into `data.csv` from Synthea `encounters.csv` |
| `2_gen_bias.py` | Add condition flags and mask column for documentation bias |
| `3_train_models.py` | Train/evaluate GBDT models on true vs biased features |
| `4_create_report.py` | Generate comprehensive `report.md` case study |

### `1b_add_cost_data.py`

Reads `synthea/output_diabetes_v2/csv/encounters.csv` (724 MB) in streaming chunks and aggregates five cost features per patient, then left-joins them into `output/data/data.csv`:

| Column | Description |
|--------|-------------|
| `total_encounter_cost` | Sum of `TOTAL_CLAIM_COST` across all encounters |
| `total_payer_coverage` | Sum of `PAYER_COVERAGE` across all encounters |
| `out_of_pocket_cost` | `total_encounter_cost - total_payer_coverage` (floored at 0) |
| `num_encounters` | Number of recorded encounters |
| `cost_per_encounter` | `total_encounter_cost / num_encounters` |

These features support a cost-disparity angle on the bias narrative: patients whose hypertriglyceridemia is under-documented may show different utilization and financial burden patterns, even before the model is aware of the diagnosis gap. The script is idempotent — re-running it replaces existing cost columns.

```bash
# Uses default paths (run from diabetes_v2/)
uv run python scripts/1b_add_cost_data.py

# Or specify paths explicitly
uv run python scripts/1b_add_cost_data.py \
  --encounters ../../synthea/output_diabetes_v2/csv/encounters.csv \
  --data output/data/data.csv
```

## Key Codes

| Condition | Code |
|-----------|------|
| Diabetes mellitus type 2 | SNOMED 44054006 |
| Hypertriglyceridemia | SNOMED 302870006 |
| Hemoglobin A1c | LOINC 4548-4 |
| BMI | LOINC 39156-5 |

## Clinical Context

Hypertriglyceridemia (elevated triglycerides) is part of metabolic syndrome and a genuine risk marker for diabetes. Unlike hyperglycemia (which is definitionally diabetes), hypertriglyceridemia is a distinct condition that signals metabolic dysfunction and often co-occurs with or precedes diabetes.

Documentation bias scenario: Labs may show elevated triglycerides, but the formal diagnosis code isn't entered due to time pressure, documentation practices, or focus on the primary diagnosis.
