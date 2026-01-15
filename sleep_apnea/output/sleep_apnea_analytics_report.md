# Rural Misdiagnosis Analysis

Misdiagnosis is defined as a sleep disorder diagnosis without a corresponding sleep apnea diagnosis (SLEEP_DISORDER_CODE 39898005 without SNOMED 73430006/78275009).

## Cohort Summary (Sleep Disorder Patients)

| Dataset | Cohort N | Sleep Apnea | Misdiagnosed | Rural | Urban | Missing Rural |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1,712 | 1,710 | 2 | 477 | 1,235 | 0 |
| biased | 1,707 | 1,322 | 385 | 476 | 1,231 | 0 |

## Pairwise Comparison (Rural vs Urban)

Rates are computed within the sleep disorder cohort. P-values are from a two-proportion z-test.

| Dataset | Rural N | Urban N | Rural Rate | Urban Rate | Risk Diff | Risk Ratio | z | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 477 | 1,235 | 0.00% | 0.16% | -0.16% | 0.00 | -0.879 | 0.379 |
| biased | 476 | 1,231 | 80.46% | 0.16% | 80.30% | 495.24 | 35.597 | 0.000 |

## Regression (Adjusted Rural Effect)

Logistic regression includes age, gender, income, BMI, smoking status, alcohol use, hypertension, CHF, and a rural indicator. Rural coefficient significance is assessed via permutation testing (n=500).

| Dataset | N | Rural Coef | Odds Ratio | Permutation p-value | In-sample AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1,712 | -2.795 | 0.061 | 0.509 | 0.973 |
| biased | 1,707 | 3.385 | 29.524 | 0.002 | 0.967 |

## Interpretation

If rural coefficients are positive with low permutation p-values, rural residence is associated with a higher chance of misdiagnosis after adjusting for clinical risk factors. The pairwise comparison provides a direct rate difference, while the regression isolates the rural effect conditional on covariates.
