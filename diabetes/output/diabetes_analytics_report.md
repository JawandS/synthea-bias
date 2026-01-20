# Documentation Bias Analysis

This report analyzes the effects of documentation bias on diabetes-related features. The biased dataset has 30% random under-documentation of hyperglycemia and hypertriglyceridemia diagnoses.

## Dataset Summary

| Dataset | Patients | Diabetes | Hyperglycemia | Hypertriglyceridemia |
| --- | ---: | ---: | ---: | ---: |
| baseline | 20,000 | 1,253 | 595 | 1,223 |
| biased | 20,000 | 1,257 | 476 | 1,112 |

## Documentation Rate Comparison

Documentation rates for the metabolic conditions affected by the bias intervention.

| Condition | Baseline Rate | Biased Rate | Reduction |
| --- | ---: | ---: | ---: |
| Hyperglycemia | 2.97% | 2.38% | 20.00% |
| Hypertriglyceridemia | 6.12% | 5.56% | 9.08% |

## Documentation Among Diabetes Patients

Among patients with a diabetes diagnosis, the rate of documented metabolic conditions.

| Condition | Baseline Rate | Biased Rate |
| --- | ---: | ---: |
| Hyperglycemia | 47.49% | 37.87% |
| Hypertriglyceridemia | 95.85% | 87.11% |

## Logistic Regression Coefficients

Logistic regression coefficients (standardized) for predicting diabetes. Higher positive coefficients indicate stronger association with diabetes diagnosis.

| Feature | Baseline Coef | Baseline OR | Biased Coef | Biased OR | Δ Coef |
| --- | ---: | ---: | ---: | ---: | ---: |
| age_years | 0.216 | 1.241 | 0.404 | 1.498 | +0.188 |
| male | 0.001 | 1.001 | -0.003 | 0.997 | -0.004 |
| income | 0.135 | 1.145 | 0.071 | 1.074 | -0.064 |
| bmi | 0.531 | 1.700 | 0.459 | 1.583 | -0.071 |
| smoker | -0.106 | 0.899 | -0.214 | 0.807 | -0.108 |
| obesity | 0.319 | 1.376 | 0.449 | 1.567 | +0.130 |
| hypertension | 0.027 | 1.027 | 0.166 | 1.180 | +0.139 |
| hyperlipidemia | 0.058 | 1.060 | 0.042 | 1.043 | -0.017 |
| hyperglycemia | 0.906 | 2.475 | 1.438 | 4.212 | +0.532 |
| hypertriglyceridemia | 2.178 | 8.826 | 2.001 | 7.395 | -0.177 |

## In-Sample Model Performance

| Dataset | AUC |
| --- | ---: |
| baseline | 0.988 |
| biased | 0.982 |

## Interpretation

Documentation bias affects the learned model in several ways:

1. **Reduced feature prevalence**: The biased dataset shows lower rates of documented hyperglycemia and hypertriglyceridemia, matching the 30% under-documentation rate.

2. **Coefficient changes**: With fewer documented metabolic conditions, the model may compensate by increasing reliance on other features (demographics, comorbidities).

3. **Impact on diabetes patients**: Among actual diabetes patients, the reduced documentation of highly predictive features creates a mismatch between true clinical state and recorded data.

4. **Generalization risk**: A model trained on biased data may perform worse when deployed on populations with complete documentation, as it has learned to under-weight the most predictive features.
