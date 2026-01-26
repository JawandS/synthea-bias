# Rural Under Diagnosis Analysis

Under diagnosis is defined as a sleep disorder diagnosis without a corresponding sleep apnea diagnosis (SLEEP_DISORDER_CODE 39898005 without SNOMED 73430006/78275009).

## Population Summary Statistics (Full Population)

Summary statistics for the full patient population, stratified by rural/urban residence and dataset.

| Characteristic | baseline Urban | baseline Rural | biased Urban | biased Rural |
| --- | ---: | ---: | ---: | ---: |
| N | 14,507 | 5,493 | 14,507 | 5,493 |
| Age (years) | 39.6 ± 23.7 | 40.1 ± 24.6 | 39.6 ± 23.7 | 40.1 ± 24.6 |
| Male (%) | 50.45% | 50.94% | 50.45% | 50.94% |
| Income ($) | 71991.4 ± 101351.5 | 60887.1 ± 87209.0 | 71991.4 ± 101351.5 | 60887.1 ± 87209.0 |
| BMI | 26.1 ± 5.0 | 26.0 ± 5.2 | 26.1 ± 5.0 | 26.0 ± 5.2 |
| Current Smoker (%) | 0.53% | 0.24% | 0.53% | 0.24% |
| Alcohol Use Disorder (%) | 0.29% | 0.40% | 0.31% | 0.44% |
| Hypertension (%) | 19.15% | 18.93% | 19.08% | 18.93% |
| CHF (%) | 1.93% | 1.98% | 1.96% | 2.00% |

## Cohort Summary (Sleep Disorder Patients)

| Dataset | Cohort N | Sleep Apnea | Under Diagnosed | Rural | Urban |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 740 | 739 | 1 | 198 | 542 |
| biased | 749 | 577 | 172 | 205 | 544 |

## Pairwise Comparison (Rural vs Urban)

Rates represent under diagnosis rates (proportion of sleep disorder patients without a sleep apnea diagnosis). P-values are from a two-proportion z-test.

| Dataset | Rural N | Urban N | Rural Rate | Urban Rate | Risk Diff | Risk Ratio | z | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 198 | 542 | 0.00% | 0.18% | -0.18% | 0.00 | -0.605 | 0.545 |
| biased | 205 | 544 | 82.93% | 0.37% | 82.56% | 225.56 | 23.951 | 0.000 |

## Regression (Adjusted Rural Effect)

Logistic regression includes age, gender, income, BMI, smoking status, alcohol use, hypertension, CHF, and a rural indicator. Rural coefficient significance is assessed via permutation testing (n=500).

| Dataset | N | Rural Coef | Odds Ratio | Permutation p-value | In-sample AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 740 | -1.545 | 0.213 | 0.711 | 0.997 |
| biased | 749 | 3.065 | 21.432 | 0.002 | 0.970 |

## Interpretation

If rural coefficients are positive with low permutation p-values, rural residence is associated with a higher chance of under diagnosis after adjusting for clinical risk factors. The pairwise comparison provides a direct rate difference, while the regression isolates the rural effect conditional on covariates.

In the biased dataset that simulates real-world barriers to care experienced by rural populations, we expect rural residence to be associated with a substantially higher odds of under diagnosis compared to the baseline dataset where urban and rural patients have equal access to care.
