# Sleep Apnea Demand Modeling: Findings Report

Generated: 2026-01-01T11:31:37

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
| Total patients | 20,106 | 20,061 |
| Urban | 72.56% | 72.56% |
| Rural | 27.44% | 27.44% |
| Sleep disorder (module entry) | 1712 | 1707 |
| Sleep apnea diagnosed | 1710 | 1322 |
| Sleep apnea prevalence | 8.50% | 6.59% |
| Dropouts | 2 | 385 |
| Dropout rate (of sleep disorder) | 0.12% | 22.55% |

## Spend Distribution

| Metric | Baseline | Biased |
| --- | ---: | ---: |
| Median spend | $0.00 | $0.00 |
| 90th percentile spend | $0.00 | $0.00 |
| Mean spend (nonzero) | $9,576.02 | $7,797.39 |

## Nonzero Spend Rate

| Cohort | Baseline | Biased |
| --- | ---: | ---: |
| Overall | 8.51% (1712/20106) | 8.51% (1707/20061) |
| Sleep disorder cohort | 100.00% (1712/1712) | 100.00% (1707/1707) |
| Sleep apnea cohort | 100.00% (1710/1710) | 100.00% (1322/1322) |

## Dataset Summary
| Metric | Baseline | Biased |
| --- | ---: | ---: |
| Sample size | 20,106 | 20,061 |
| Mean spend | $815.39 | $663.48 |
| Nonzero rate | 8.5% | 8.5% |
| Train/Val/Test split | 14074/3015/3017 | 14042/3009/3010 |

## GBDT Model Selection

| Dataset | Best Hyperparameters | Val MAE | Val R² |
| --- | --- | ---: | ---: |
| Baseline | n=100, lr=0.01, depth=2, leaf=5 | $829 | -0.044 |
| Biased | n=100, lr=0.01, depth=3, leaf=20 | $687 | -0.044 |

## Feature Importances

| Feature | Baseline | Biased |
| --- | --- | --- |
| hypertension | 0.800 | 0.735 |
| age_years | 0.133 | 0.157 |
| male | 0.067 | 0.102 |
| income | 0.000 | 0.000 |
| bmi | 0.000 | 0.006 |
| smoker | 0.000 | 0.000 |
| alcohol_use | 0.000 | 0.000 |
| chf | 0.000 | 0.000 |

## Model Performance

### In-Dataset Test Results

| Model | Test Set | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| Baseline | Baseline | $801 | $3,259 | -0.063 |
| Biased | Biased | $717 | $3,191 | -0.052 |

### Cross-Dataset Test Results

| Model | Test Set | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| Biased | Baseline | $801 | $3,259 | -0.063 |
| Baseline | Biased | $717 | $3,191 | -0.052 |

### Prediction Bias (Baseline Test Set)

| Model | Mean Prediction | Actual Mean | Difference | Rel. Error |
| --- | ---: | ---: | ---: | ---: |
| Baseline | $1.70 | $800.37 | $-798.67 | -99.79% |
| Biased | $1.61 | $800.37 | $-798.77 | -99.80% |

---

# Bias Quantification

## Geographic Disparity (Linear Regression)

The urban coefficient measures the percent difference in log1p spend between urban and
rural patients after controlling for clinical factors. See README.md for methodology.

### Urban Effect Summary

| Dataset | Urban Effect (%) | 95% CI | p-value | Significant? |
| --- | ---: | --- | ---: | :---: |
| Baseline | -1.23% | [-9.18%, +7.38%] | 0.7662 | No |
| Biased | +8.83% | [+0.61%, +17.98%] | 0.0310 | Yes |

### Key Finding

| Metric | Value |
| --- | ---: |
| Baseline urban effect | -1.23% |
| Biased urban effect | +8.83% |
| **Bias-induced disparity** | **+10.06%** |
| Baseline R² (log1p) | 0.1679 |
| Biased R² (log1p) | 0.1628 |

> The biased dataset shows rural patients spending **10.1% more** than in baseline, which is unexpected.

### Standardized Coefficients

| Feature | Baseline | Biased |
| --- | ---: | ---: |
| age_years | 0.35 | 0.31 |
| male | 0.25 | 0.23 |
| income | -0.02 | 0.02 |
| bmi | 0.04 | 0.04 |
| smoker | -0.01 | -0.01 |
| alcohol_use | 0.01 | 0.04 |
| hypertension | 0.71 | 0.66 |
| chf | 0.02 | 0.04 |
| urban | -0.01 | 0.04 |

## Statistical Confidence (Bootstrap)

| Evaluation | MAE [95% CI] | R² [95% CI] |
| --- | --- | --- |
| Baseline → Baseline | 801.22 [686.09, 918.24] | -0.06 [-0.08, -0.05] |
| Biased → Biased | 716.51 [605.70, 828.60] | -0.05 [-0.06, -0.04] |
| Biased → Baseline | 801.11 [686.00, 918.14] | -0.06 [-0.08, -0.05] |
| Baseline → Biased | 716.60 [605.80, 828.71] | -0.05 [-0.06, -0.04] |

*1000 bootstrap iterations, 95% CIs*

## Hypothesis Test (Permutation)

Cross-population MAE difference: 84.60 (p = 0.3057, not significant)

*1000 permutations, two-sided test*
