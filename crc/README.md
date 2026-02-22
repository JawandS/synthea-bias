# Colorectal Screening Case Study (CRC)

Bias case study for CRC screening recommendations with age-income intersectional label masking.

## Current Setup

- Target population for generation: `15,000` (default in script 1)
- Cohort used for analysis/modeling: ages `50-80`
- Age subgroup bins: `50-54`, `55-59`, `60-64`, `65-69`, `70-74`, `75-80`
- Label window: colonoscopy in the last 5 years (`SNOMED 73761001`)
- Models:
  - Baseline trained on `true_screened_in_last_5y`
  - Biased trained on `observed_screened_in_last_5y`
  - Both evaluated against `true_screened_in_last_5y`

## Overall Story
This case study is trying to model which individuals are screened for colorectal cancer (CRC) in the last 5 years. The bias mechanism is that younger and lower-income patients are more likely to have their screening status masked, which simulates a scenario where certain populations are underrepresented in the observed data. Since some insurance companies may not have access to income of individuals this analysis shows how not controlling for key variables can perpetuate bias in model predictions and lead to worse outcomes for certain subgroups.

## Data Flow

- Source data comes from `synthea/output_crc/csv`.
- `scripts/1_generate_data.py` runs Synthea and writes `output/info/1_summary_stats.md`.
- `scripts/2_gen_bias.py` reads directly from `synthea/output_crc/csv` and writes:
  - `output/data/data.csv`
  - `output/info/2_bias_effect.md`
- `scripts/3_train_models.py` writes `output/info/3_model.md`.
- `scripts/4_create_report.py` writes `output/report.md`.

## Bias Mechanism

Masking is applied only when `true_screened_in_last_5y = 1`:

`p_mask = clip(0.60 - 0.004 * (age - 50) - 0.000005 * (income_usd - 20000), 0.05, 0.70)`

`observed_screened_in_last_5y = true_screened_in_last_5y * (1 - mask_screening)`

Lower income and younger patients are more likely to be masked.

## Run

```bash
cd crc
uv run python scripts/1_generate_data.py -p 12000 -s 160
uv run python scripts/2_gen_bias.py -s 160
uv run python scripts/3_train_models.py
uv run python scripts/4_create_report.py
```
