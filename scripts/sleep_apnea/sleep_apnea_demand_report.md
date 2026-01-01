# Sleep Apnea Demand Modeling Report

- Generated: 2026-01-01T08:44:51
- Baseline dataset: `output_baseline/csv`
- Biased dataset: `output_rural_bias/csv`
- BMI feature included: True (always on)
- Smoking, alcohol use, and CHF features included: True

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
- Baseline patients: 1163 (urban 69.56%, rural 30.44%)
- Biased patients: 1168 (urban 69.26%, rural 30.74%)
- Sleep apnea prevalence (unique patients with condition): baseline 4.99% (58/1163), biased 4.11% (48/1168)
- Sleep disorder patients (module entry): baseline 58, biased 60
- Dropouts (sleep disorder without sleep apnea): baseline 0 (0.00% of sleep disorder), biased 12 (20.00% of sleep disorder)
- Urban/rural classification uses `geography/sdoh.csv` (URBAN field) via SDoH attributes.

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
age_years, male, income, bmi, smoker, alcohol_use, hypertension, chf

## Dataset Summary
- Baseline: n=1163, mean_spend=$22,384.23, nonzero_rate=97.68%
- Biased: n=1168, mean_spend=$22,791.75, nonzero_rate=97.69%

## Split Configuration
- Train/Val/Test sizes (baseline): 814/174/175
- Train/Val/Test sizes (biased): 817/175/176

## Model Selection (Validation MAE)
- Baseline best params: {'n_estimators': 100, 'learning_rate': 0.01, 'max_depth': 2, 'min_samples_leaf': 10}
  - Val MAE=33201.30, RMSE=152340.88, R2=-0.008
- Biased best params: {'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 4, 'min_samples_leaf': 20}
  - Val MAE=20872.61, RMSE=46098.24, R2=0.033

## Gradient Boosted Decision Tree (Feature Importances)
Feature importances are normalized and sum to 1.0 per model.

| Feature | Baseline | Biased |
| --- | --- | --- |
| age_years | 0.493 | 0.705 |
| income | 0.354 | 0.101 |
| bmi | 0.086 | 0.122 |
| hypertension | 0.068 | 0.071 |
| male | 0.000 | 0.001 |
| smoker | 0.000 | 0.000 |
| alcohol_use | 0.000 | 0.000 |
| chf | 0.000 | 0.000 |

## Test Results (In-Dataset)
- Baseline model on baseline test: MAE=28030.05, RMSE=56360.41, R2=0.059
- Biased model on biased test: MAE=44669.21, RMSE=190212.17, R2=0.014

## Cross-Dataset Evaluation
- Biased model on baseline test: MAE=24864.87, RMSE=52714.94, R2=0.177
- Baseline model on biased test: MAE=44396.25, RMSE=184915.89, R2=0.069

## Demand Bias (Baseline Test Set)
- Baseline model mean prediction: $24,086.93 (actual $22,617.25, diff $1,469.68)
- Biased model mean prediction: $22,523.47 (actual $22,617.25, diff $-93.78, rel -0.41% )

---

# Extended Analysis: Bias Quantification

## Geographic Bias Quantification (Linear Regression)

This analysis uses linear regression to quantify the effect of urban/rural status
on healthcare spending, controlling for age, gender, income, BMI, smoking status,
alcohol use disorder, hypertension, and CHF.

**Key insight**: The urban coefficient represents the spending difference between
urban and rural patients *after controlling for other factors*. A significant
difference between baseline and biased datasets indicates measurable access disparity.

### Urban Coefficient (Access Disparity Measure)

| Dataset | Urban Coef ($) | 95% CI | p-value | Significant? |
| --- | ---: | --- | ---: | :---: |
| Baseline (unbiased) | $-1,434 | [$-17,087, $14,044] | 0.8581 | No |
| Biased (rural dropout) | $-1,585 | [$-16,378, $12,738] | 0.8551 | No |

### Interpretation

- **Baseline urban coefficient**: $-1,434
- **Biased urban coefficient**: $-1,585
- **Difference (bias effect)**: $-151

The biased dataset shows rural patients spending **$151 less** than
in the baseline dataset, controlling for other factors. This difference represents
the access disparity introduced by rural dropout.

### All Coefficients (Standardized)

Standardized coefficients allow comparison of relative feature importance.

| Feature | Baseline (std) | Biased (std) |
| --- | ---: | ---: |
| age_years | 22,809.24 | 20,592.82 |
| male | 8,138.28 | 6,598.92 |
| income | -1,236.05 | -166.14 |
| bmi | -9,398.27 | -10,253.11 |
| smoker | -505.58 | 2,467.88 |
| alcohol_use | -3,412.80 | -3,389.83 |
| hypertension | 9,091.95 | 7,188.54 |
| chf | -1,739.00 | -442.13 |
| urban | -662.35 | -738.37 |

### Coefficient Confidence Intervals

| Feature | Baseline Coef [95% CI] | Biased Coef [95% CI] |
| --- | --- | --- |
| age_years | $878 [$311, $1,637] | $778 [$351, $1,339] |
| male | $16,277 [$4,854, $30,760] | $13,214 [$4,055, $24,732] |
| income | $-0 [$-0, $0] | $-0 [$-0, $0] |
| bmi | $-1,989 [$-4,118, $-420] | $-2,229 [$-4,088, $-689] |
| smoker | $-6,471 [$-22,721, $10,208] | $31,643 [$-11,619, $112,090] |
| alcohol_use | $-36,962 [$-77,663, $-1,246] | $-32,477 [$-56,887, $-15,277] |
| hypertension | $22,086 [$-1,703, $47,896] | $17,522 [$-4,093, $42,014] |
| chf | $-10,969 [$-49,751, $16,886] | $-2,861 [$-33,269, $24,478] |
| urban | $-1,434 [$-17,087, $14,044] | $-1,585 [$-16,378, $12,738] |
| intercept | $23,025 [$-1,016, $51,912] | $34,192 [$9,351, $64,071] |

### Linear Model R² (Test Set)

- Baseline: R² = 0.1416
- Biased: R² = 0.0316

## GBDT Model Performance (Bootstrap CIs)

Bootstrap iterations: 1000
95% confidence intervals shown as [lower, upper]

### In-Dataset Performance

| Model | Dataset | MAE [95% CI] | RMSE [95% CI] | R² [95% CI] |
| --- | --- | --- | --- | --- |
| Base | Baseline | 28030.05 [21206.91, 35503.48] | 56360.41 [36974.39, 74114.61] | 0.06 [-0.17, 0.25] |
| Base | Biased | 44669.21 [22099.46, 74474.85] | 190212.17 [50273.03, 298163.41] | 0.01 [-0.38, 0.07] |

### Cross-Population Performance

| Model | Source → Target | MAE [95% CI] | RMSE [95% CI] | R² [95% CI] |
| --- | --- | --- | --- | --- |
| Base | Biased → Baseline | 24864.87 [18111.26, 31949.69] | 52714.94 [34132.85, 69816.59] | 0.18 [-0.00, 0.36] |
| Base | Baseline → Biased | 44396.25 [22539.46, 73239.18] | 184915.89 [46304.06, 292277.51] | 0.07 [-0.19, 0.18] |

## Hypothesis Tests (Permutation)

Permutation iterations: 1000
Two-sided permutation tests for MAE differences.
p < 0.05 indicates statistically significant difference.

| Test | Description | Observed Diff | p-value | Significant |
| --- | --- | --- | --- | --- |
| Base Cross-Pop | Biased model: baseline vs biased test | -19804.34 | 0.2078 | No |

## Methodology Notes

### Bias Quantification Approach

Instead of using urban/rural as a feature in a predictive model (which would encode
the disparity), we use linear regression to *measure* the urban coefficient:

```
spend ~ age + gender + income + bmi + smoker + alcohol_use + hypertension + chf + urban
```

The urban coefficient represents the average spending difference between urban and
rural patients, controlling for other factors. By comparing this coefficient across
datasets, we can quantify the access disparity introduced by the bias.

### Key Findings

1. **The urban coefficient measures disparity, not need**: A positive coefficient means
   urban patients spend more (controlling for demographics), indicating rural underutilization.

2. **Comparing coefficients reveals bias**: If the biased dataset has a larger urban
   coefficient than baseline, the rural dropout is creating measurable disparity.

3. **Statistical significance**: Bootstrap CIs and permutation p-values indicate whether
   the measured disparity is statistically robust.
