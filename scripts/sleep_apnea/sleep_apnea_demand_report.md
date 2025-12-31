# Sleep Apnea Demand Modeling Report

- Generated: 2025-12-30T19:16:49
- Baseline dataset: `output_baseline/csv`
- Biased dataset: `output_rural_bias/csv`
- BMI feature included: False

## Target Definition
Total sleep-related spend per patient, computed as the sum of:
- `procedures.csv` BASE_COST where CODE in sleep-related procedure codes or REASONCODE in sleep-related condition codes.
- `encounters.csv` TOTAL_CLAIM_COST (fallback BASE_ENCOUNTER_COST) where REASONCODE in sleep-related condition codes or encounter CODE in sleep-specific codes.
- `medications.csv` TOTALCOST where REASONCODE in sleep-related condition codes.

Sleep-related condition codes: ['39898005', '73430006', '78275009']
Sleep-related procedure codes: ['103750000', '10563004', '446573003', '60554003', '698560000', '82808001']
Sleep-related encounter codes: ['185345009', '185347001', '185389009']

## Features (No Urban/Rural)
age_years, male, race_black, race_white, ethnicity_hispanic, income, healthcare_expenses, healthcare_coverage, hypertension

## Dataset Summary
- Baseline: n=27199, mean_spend=$37,762.34, nonzero_rate=97.21%
- Biased: n=22112, mean_spend=$38,475.84, nonzero_rate=97.07%

## Split Configuration
- Train/Val/Test sizes (baseline): 19039/4079/4081
- Train/Val/Test sizes (biased): 15478/3316/3318

## Model Selection (Validation MAE)
- Baseline best params: {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 20}
  - Val MAE=28221.36, RMSE=92940.80, R2=0.799
- Biased best params: {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 20}
  - Val MAE=29957.98, RMSE=96848.20, R2=0.838

## Test Results (In-Dataset)
- Baseline model on baseline test: MAE=31439.39, RMSE=118778.53, R2=0.688
- Biased model on biased test: MAE=30563.76, RMSE=106978.13, R2=0.669

## Cross-Dataset Evaluation
- Biased model on baseline test: MAE=30320.55, RMSE=103741.79, R2=0.762
- Baseline model on biased test: MAE=30138.00, RMSE=108164.92, R2=0.662

## Demand Bias (Baseline Test Set)
- Baseline model mean prediction: $36,436.37 (actual $36,934.62, diff $-498.25)
- Biased model mean prediction: $35,088.97 (actual $36,934.62, diff $-1,845.65, rel -5.00% )
