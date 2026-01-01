# Sleep Apnea Demand Modeling Report

- Generated: 2025-12-31T17:27:31
- Baseline dataset: `output_baseline/csv`
- Biased dataset: `output_rural_bias/csv`
- BMI feature included: True (always on)

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
- Baseline patients: 23014 (urban 68.31%, rural 31.68%)
- Biased patients: 23003 (urban 69.14%, rural 30.85%)
- Sleep apnea prevalence (unique patients with condition): baseline 5.02% (1156/23014), biased 3.98% (916/23003)
- Sleep disorder patients (module entry): baseline 1158, biased 1106
- Dropouts (sleep disorder without sleep apnea): baseline 2 (0.17% of sleep disorder), biased 190 (17.18% of sleep disorder)
- Urban/rural classification uses `geography/sdoh.csv` (URBAN field) via SDoH attributes.
- Urban/rural lookup missing: baseline 1, biased 1

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

## Features (No Urban/Rural or Race/Ethnicity)
age_years, male, income, hypertension

## Dataset Summary
- Baseline: n=23014, mean_spend=$39,712.20, nonzero_rate=97.36%
- Biased: n=23003, mean_spend=$38,099.49, nonzero_rate=97.51%

## Split Configuration
- Train/Val/Test sizes (baseline): 16109/3452/3453
- Train/Val/Test sizes (biased): 16102/3450/3451

## Model Selection (Validation MAE)
- Baseline best params: {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 3, 'min_samples_leaf': 5}
  - Val MAE=52518.68, RMSE=188232.61, R2=0.062
- Biased best params: {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 5}
  - Val MAE=50963.19, RMSE=196894.75, R2=0.087

## Gradient Boosted Decision Tree (Feature Importances)
Feature importances are normalized and sum to 1.0 per model.

| Feature | Baseline | Biased |
| --- | --- | --- |
| age_years | 0.743 | 0.756 |
| male | 0.133 | 0.164 |
| income | 0.096 | 0.034 |
| hypertension | 0.028 | 0.046 |

## Test Results (In-Dataset)
- Baseline model on baseline test: MAE=56864.08, RMSE=217318.12, R2=0.084
- Biased model on biased test: MAE=55195.68, RMSE=213018.90, R2=0.068

## Cross-Dataset Evaluation
- Biased model on baseline test: MAE=55780.45, RMSE=216137.25, R2=0.094
- Baseline model on biased test: MAE=55956.07, RMSE=213115.06, R2=0.067

## Demand Bias (Baseline Test Set)
- Baseline model mean prediction: $37,989.74 (actual $43,151.48, diff $-5,161.73)
- Biased model mean prediction: $37,297.02 (actual $43,151.48, diff $-5,854.46, rel -13.57% )

---

# Extended Analysis: Model Variants and Statistical Testing

## Model Variants

This analysis compares two model variants to assess whether including
geographic (urban/rural) information can control for access bias:

1. **Base Model**: Core features only (age, gender, income, BMI, hypertension)
2. **Urban Model**: Core features + urban/rural indicator

### Feature Sets
- Base features: age_years, male, income, hypertension
- Urban features: age_years, male, income, hypertension, urban

## Urban Model Hyperparameters

- Urban baseline best params: {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 3, 'min_samples_leaf': 5}
  - Val MAE=52462.32, RMSE=188218.29, R2=0.063
- Urban biased best params: {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 5}
  - Val MAE=50972.99, RMSE=196898.77, R2=0.087

## Urban Model Feature Importances

| Feature | Urban Baseline | Urban Biased |
| --- | --- | --- |
| age_years | 0.741 | 0.756 |
| male | 0.136 | 0.164 |
| income | 0.082 | 0.034 |
| hypertension | 0.027 | 0.046 |
| urban | 0.014 | 0.000 |

## Bootstrap Confidence Intervals

Bootstrap iterations: 1000
95% confidence intervals shown as [lower, upper]

### In-Dataset Performance

| Model | Dataset | MAE [95% CI] | RMSE [95% CI] | R² [95% CI] |
| --- | --- | --- | --- | --- |
| Base | Baseline | 56864.08 [49717.56, 64078.75] | 217318.12 [180173.50, 252553.59] | 0.08 [0.06, 0.11] |
| Base | Biased | 55195.68 [48632.18, 62556.97] | 213018.90 [177526.95, 246502.16] | 0.07 [0.03, 0.10] |
| Urban | Baseline | 56870.76 [49727.97, 64094.30] | 217304.35 [180016.96, 252830.04] | 0.08 [0.06, 0.11] |
| Urban | Biased | 55198.71 [48637.39, 62562.23] | 213020.92 [177529.83, 246502.83] | 0.07 [0.03, 0.10] |

### Cross-Population Performance

| Model | Source → Target | MAE [95% CI] | RMSE [95% CI] | R² [95% CI] |
| --- | --- | --- | --- | --- |
| Base | Biased → Baseline | 55780.45 [48715.50, 63159.48] | 216137.25 [178883.87, 251524.31] | 0.09 [0.06, 0.12] |
| Base | Baseline → Biased | 55956.07 [49296.58, 63096.11] | 213115.06 [177251.33, 246542.88] | 0.07 [0.04, 0.09] |
| Urban | Biased → Baseline | 55784.38 [48717.63, 63165.47] | 216137.20 [178881.51, 251526.77] | 0.09 [0.06, 0.12] |
| Urban | Baseline → Biased | 56028.61 [49419.59, 63136.20] | 212889.64 [177220.55, 246191.09] | 0.07 [0.04, 0.10] |

## Hypothesis Tests (Permutation)

Permutation iterations: 1000
Two-sided permutation tests for MAE differences.
p < 0.05 indicates statistically significant difference.

| Test | Description | Observed Diff | p-value | Significant |
| --- | --- | --- | --- | --- |
| Base Cross-Pop | Biased model: baseline vs biased test | 584.78 | 0.8881 | No |
| Urban Cross-Pop | Urban model: baseline vs biased test | 585.67 | 0.8911 | No |
| Base vs Urban (Baseline) | Base vs Urban on baseline data | -6.69 | 1.0000 | No |
| Base vs Urban (Biased) | Base vs Urban on biased data | -3.03 | 1.0000 | No |

## Interpretation

### Key Questions Addressed

1. **Does the urban feature improve model performance?**
   Compare Base vs Urban model MAE confidence intervals.

2. **Does including urban/rural control for geographic access bias?**
   If the Urban model shows smaller cross-population MAE differences
   than the Base model, the urban feature partially controls for bias.

3. **Is the bias statistically significant?**
   The permutation test p-values indicate whether observed differences
   are unlikely to occur by chance.

### Methodology Notes

- **Data leakage prevention**: Healthcare expenses and coverage removed from features
  as they include the target (sleep-related spending).
- **Consistent splits**: Same random seed used for both datasets to ensure
  comparable train/val/test partitions.
- **Bootstrap CIs**: Non-parametric confidence intervals via resampling.
- **Permutation tests**: Distribution-free hypothesis testing by shuffling sample assignments.
