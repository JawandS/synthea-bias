# Sleep Apnea Demand Modeling Report

- Generated: 2025-12-31T11:03:40
- Baseline dataset: `output_baseline/csv`
- Biased dataset: `output_rural_bias/csv`
- BMI feature included: False

## Study Setup
Commands used (from `scripts/NOTES.md`):
```bash
./run_synthea -p 25000 --exporter.csv.export=true --exporter.baseDirectory=./output_baseline
./run_synthea -p 25000 --exporter.csv.export=true --exporter.baseDirectory=./output_rural_bias --module_override=/home/js/contracts/synthea-bias/config/overrides_rural_sleep_apnea.properties
```

## Module Override (Rural Access Bias)
The override file `config/overrides_rural_sleep_apnea.properties` adjusts two complex transitions in `sleep_apnea.json` for patients with `urban == false`:
- `Wait Until Overnight Study`: changes the rural branch from 0% `Terminal` / 100% `Overnight Test` to 50% / 50%.
- `Appointment Delay`: changes the rural branch from 0% `Terminal` / 100% `Follow Up` to 50% / 50%.
This simulates rural patients dropping out before testing or follow-up.

## Population Stats
- Baseline patients: 27199 (urban 100.00%, rural 0.00%)
- Biased patients: 22112 (urban 100.00%, rural 0.00%)
- Sleep apnea prevalence (unique patients with condition): baseline 4.54% (1235/27199), biased 4.30% (951/22112)
- Sleep disorder patients (module entry): baseline 1237, biased 955
- Dropouts (sleep disorder without sleep apnea): baseline 2 (0.16% of sleep disorder), biased 4 (0.42% of sleep disorder)
- Urban/rural classification uses `geography/sdoh.csv` (URBAN field) via SDoH attributes.
- These runs are 100% urban, so the rural dropout branch is not exercised.

## Target Definition
Total sleep-related spend per patient, computed as the sum of:
- `procedures.csv` BASE_COST where CODE in sleep-related procedure codes or REASONCODE in sleep-related condition codes.
- `encounters.csv` TOTAL_CLAIM_COST (fallback BASE_ENCOUNTER_COST) where REASONCODE in sleep-related condition codes or encounter CODE in sleep-specific codes.
- `medications.csv` TOTALCOST where REASONCODE in sleep-related condition codes.
- `devices.csv` MODE cost from costs/devices.csv or costs/supplies.csv for sleep-related device/supply codes or sleep-related encounters.
- `supplies.csv` MODE cost from costs/supplies.csv or costs/devices.csv for sleep-related device/supply codes or sleep-related encounters.

Sleep-related condition codes: ['39898005', '73430006', '78275009']
Sleep-related procedure codes: ['103750000', '10563004', '446573003', '60554003', '698560000', '82808001']
Sleep-related encounter codes: ['185345009', '185347001', '185389009']

Sleep-related device codes: ['272265001', '701077002', '701100002', '702172008', '706180003', '720253003']
Sleep-related supply codes: ['463659001', '467645007', '704718009', '706226000', '972002']

## Features (No Urban/Rural)
age_years, male, race_black, race_white, ethnicity_hispanic, income, healthcare_expenses, healthcare_coverage, hypertension

## Dataset Summary
- Baseline: n=27199, mean_spend=$37,798.26, nonzero_rate=97.21%
- Biased: n=22112, mean_spend=$38,509.99, nonzero_rate=97.07%

## Split Configuration
- Train/Val/Test sizes (baseline): 19039/4079/4081
- Train/Val/Test sizes (biased): 15478/3316/3318

## Model Selection (Validation MAE)
- Baseline best params: {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 20}
  - Val MAE=28267.00, RMSE=94245.38, R2=0.794
- Biased best params: {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 20}
  - Val MAE=29982.18, RMSE=96080.84, R2=0.840

## Gradient Boosted Decision Tree (Feature Importances)
Feature importances are normalized and sum to 1.0 per model.

| Feature | Baseline | Biased |
| --- | --- | --- |
| healthcare_coverage | 0.820 | 0.767 |
| male | 0.105 | 0.123 |
| healthcare_expenses | 0.028 | 0.029 |
| age_years | 0.024 | 0.061 |
| income | 0.015 | 0.005 |
| hypertension | 0.007 | 0.014 |
| race_white | 0.002 | 0.000 |
| race_black | 0.000 | 0.000 |
| ethnicity_hispanic | 0.000 | 0.000 |

## Test Results (In-Dataset)
- Baseline model on baseline test: MAE=31315.35, RMSE=117950.31, R2=0.692
- Biased model on biased test: MAE=30574.12, RMSE=107876.87, R2=0.664

## Cross-Dataset Evaluation
- Biased model on baseline test: MAE=30186.40, RMSE=103344.97, R2=0.764
- Baseline model on biased test: MAE=30261.69, RMSE=109091.59, R2=0.656

## Demand Bias (Baseline Test Set)
- Baseline model mean prediction: $36,679.94 (actual $36,973.35, diff $-293.40)
- Biased model mean prediction: $35,139.86 (actual $36,973.35, diff $-1,833.48, rel -4.96% )
