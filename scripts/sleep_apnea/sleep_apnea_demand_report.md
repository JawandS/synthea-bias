# Sleep Apnea Demand Modeling: Findings Report

Generated: 2026-01-01T10:02:51

> See [README.md](README.md) for study methodology, feature definitions, and clinical codes.

## Run Configuration

| Parameter | Value |
| --- | --- |
| Baseline dataset | `output_baseline/csv` |
| Biased dataset | `output_rural_bias/csv` |
| Bootstrap iterations | 1000 |
| Permutation iterations | 1000 |
| Target transform | log1p(spend) |
| Metrics scale | dollars (back-transformed) |

## Population Summary

| Metric | Baseline | Biased |
| --- | ---: | ---: |
| Total patients | 5,000 | 5,000 |
| Urban | 69.14% | 69.14% |
| Rural | 30.86% | 30.86% |
| Sleep disorder (module entry) | 224 | 224 |
| Sleep apnea diagnosed | 224 | 179 |
| Sleep apnea prevalence | 4.48% | 3.58% |
| Dropouts | 0 | 45 |
| Dropout rate (of sleep disorder) | 0.00% | 20.09% |

## Spend Distribution

| Metric | Baseline | Biased |
| --- | ---: | ---: |
| Median spend | $0.00 | $0.00 |
| 90th percentile spend | $0.00 | $0.00 |
| Mean spend (nonzero) | $9,521.77 | $7,387.97 |

## Dataset Summary
| Metric | Baseline | Biased |
| --- | ---: | ---: |
| Sample size | 5,000 | 5,000 |
| Mean spend | $426.58 | $330.98 |
| Nonzero rate | 4.5% | 4.5% |
| Train/Val/Test split | 3500/750/750 | 3500/750/750 |

## GBDT Model Selection

| Dataset | Best Hyperparameters | Val MAE | Val R² |
| --- | --- | ---: | ---: |
| Baseline | n=500, lr=0.1, depth=4, leaf=5 | $323 | -0.022 |
| Biased | n=100, lr=0.01, depth=2, leaf=20 | $235 | -0.018 |

## Feature Importances

| Feature | Baseline | Biased |
| --- | --- | --- |
| age_years | 0.357 | 0.071 |
| income | 0.302 | 0.020 |
| hypertension | 0.180 | 0.778 |
| bmi | 0.107 | 0.000 |
| male | 0.039 | 0.132 |
| alcohol_use | 0.011 | 0.000 |
| chf | 0.004 | 0.000 |
| smoker | 0.000 | 0.000 |

## Model Performance

### In-Dataset Test Results

| Model | Test Set | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| Baseline | Baseline | $487 | $2,887 | -0.026 |
| Biased | Biased | $360 | $2,468 | -0.021 |

### Cross-Dataset Test Results

| Model | Test Set | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| Biased | Baseline | $483 | $2,891 | -0.028 |
| Baseline | Biased | $351 | $2,399 | 0.034 |

### Prediction Bias (Baseline Test Set)

| Model | Mean Prediction | Actual Mean | Difference | Rel. Error |
| --- | ---: | ---: | ---: | ---: |
| Baseline | $5.90 | $482.87 | $-476.97 | -98.78% |
| Biased | $0.54 | $482.87 | $-482.33 | -99.89% |

---

# Bias Quantification

## Geographic Disparity (Linear Regression)

The urban coefficient measures the percent difference in log1p spend between urban and
rural patients after controlling for clinical factors. See README.md for methodology.

### Urban Effect Summary

| Dataset | Urban Effect (%) | 95% CI | p-value | Significant? |
| --- | ---: | --- | ---: | :---: |
| Baseline | -0.22% | [-12.52%, +13.17%] | 0.9680 | No |
| Biased | +11.75% | [-0.44%, +26.14%] | 0.0819 | No |

### Key Finding

| Metric | Value |
| --- | ---: |
| Baseline urban effect | -0.22% |
| Biased urban effect | +11.75% |
| **Bias-induced disparity** | **+11.98%** |
| Baseline R² (log1p) | 0.0965 |
| Biased R² (log1p) | 0.0835 |

> The biased dataset shows rural patients spending **12.0% more** than in baseline, which is unexpected.

### Standardized Coefficients

| Feature | Baseline | Biased |
| --- | ---: | ---: |
| age_years | 0.13 | 0.18 |
| male | 0.15 | 0.12 |
| income | -0.00 | 0.00 |
| bmi | 0.04 | 0.01 |
| smoker | -0.00 | -0.00 |
| alcohol_use | 0.08 | 0.03 |
| hypertension | 0.44 | 0.43 |
| chf | -0.01 | -0.05 |
| urban | -0.00 | 0.05 |

## Statistical Confidence (Bootstrap)

| Evaluation | MAE [95% CI] | R² [95% CI] |
| --- | --- | --- |
| Baseline → Baseline | 487.23 [298.86, 682.74] | -0.03 [-0.04, -0.02] |
| Biased → Biased | 359.78 [205.17, 538.36] | -0.02 [-0.03, -0.01] |
| Biased → Baseline | 483.28 [294.29, 678.12] | -0.03 [-0.04, -0.02] |
| Baseline → Biased | 351.23 [200.95, 524.28] | 0.03 [-0.02, 0.12] |

*1000 bootstrap iterations, 95% CIs*

## Hypothesis Test (Permutation)

Cross-population MAE difference: 123.50 (p = 0.3726, not significant)

*1000 permutations, two-sided test*
