# Colorectal Screening Case Study: Intersectional Bias (Age x Income)

Goal: build a GBDT that recommends CRC screening from demographics + clinical features, while simulating bias from missing income.

## Scope

- Use Synthea CRC behavior where possible.
- Restrict cohort to ages `50-100` (all eligible for screening).
- Define labels:
  - `true_screened_in_last_5y`
  - `observed_screened_in_last_5y` (after masking)
- Train:
  - Baseline model on `true_screened_in_last_5y`
  - Biased model on `observed_screened_in_last_5y`
- Evaluate both against `true_screened_in_last_5y`.

## Exact Bias Mechanism

Mask only patients with `true_screened_in_last_5y = 1`.

`p_mask = clip(0.60 - 0.004 * (age - 50) - 0.000005 * (income_usd - 20000), 0.05, 0.70)`

Then:
- sample `u ~ Uniform(0, 1)`
- `mask_screening = 1 if u < p_mask else 0`
- `observed_screened_in_last_5y = true_screened_in_last_5y * (1 - mask_screening)`

Effect: older and higher-income patients are less likely to be masked, creating an age-income intersectional bias.

## Required Columns

Training features (income excluded):
- `age`
- `male`
- `smoker`
- `bmi`
- `obesity` (`SNOMED 162864005`)
- `type2_diabetes` (`SNOMED 44054006`)
- `hypertension` (`SNOMED 59621000`)
- `hyperlipidemia` (`SNOMED 55822004`)
- `ibd` (`Crohn 34000006` or `Ulcerative colitis 64766004`; use `0` if absent)
- `ambulatory_visits_last2y`
- `preventive_visit_last2y`
- `comorbidity_count`

Analysis-only columns:
- `income_usd`
- `income_band`
- `age_band`
- `true_screened_in_last_5y`
- `observed_screened_in_last_5y`
- `mask_screening`

## Label and Leakage Rules

- Primary target window: screening in last 5 years.
- Start with colonoscopy code `SNOMED 73761001`.
- Do not train on:
  - any direct screening indicator in the target window
  - target-derived columns
  - post-index outcomes/treatments
- Do not include `income_usd` in model features.

## 4-Script Plan

1. `scripts/1_generate_data.py`
- Generate/extract `patients.csv`, `conditions.csv`, `procedures.csv`, `observations.csv`.
- Filter to age `50-100`.
- Write `output/info/1_summary_stats.md`.

2. `scripts/2_gen_bias.py`
- Build `output/data/data.csv`.
- Create true/observed screening labels and `mask_screening` using formula above.
- Write `output/info/2_bias_effect.md` with age, income, and age x income prevalence tables.

3. `scripts/3_train_models.py`
- Train baseline vs biased GBDT with identical splits.
- Tune threshold on validation (`f1`).
- Report AUC, recall, F1 overall and by:
  - age bands: `50-59`, `60-69`, `70-79`, `80+`
  - income bands
  - age x income intersections
- Write `output/info/3_model.md`.

4. `scripts/4_create_report.py`
- Build `output/report.md` summarizing data, bias mechanism, model deltas, and subgroup disparities.

## Expected Result

Because income is omitted from training but drives masking, the biased model should over-recommend screening for groups with better observed capture (especially older/higher-income) and under-recommend for younger/lower-income groups.
