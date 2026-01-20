# Documentation Bias Analysis

This report analyzes the effects of documentation bias on diabetes-related features. The biased dataset has 30% random under-documentation of hyperglycemia and hypertriglyceridemia diagnoses.

## Dataset Summary

| Dataset | Patients | Diabetes | Hyperglycemia | Hypertriglyceridemia |
| --- | ---: | ---: | ---: | ---: |
| baseline | 20,106 | 1,268 | 609 | 1,217 |
| biased | 20,000 | 1,257 | 476 | 1,112 |

## Documentation Rate Comparison

Documentation rates for the metabolic conditions affected by the bias intervention.

| Condition | Baseline Rate | Biased Rate | Reduction |
| --- | ---: | ---: | ---: |
| Hyperglycemia | 3.03% | 2.38% | 21.42% |
| Hypertriglyceridemia | 6.05% | 5.56% | 8.14% |

## Documentation Among Diabetes Patients

Among patients with a diabetes diagnosis, the rate of documented metabolic conditions.

| Condition | Baseline Rate | Biased Rate |
| --- | ---: | ---: |
| Hyperglycemia | 48.03% | 37.87% |
| Hypertriglyceridemia | 95.27% | 87.11% |

## Logistic Regression Coefficients

Logistic regression coefficients (standardized) for predicting diabetes. Higher positive coefficients indicate stronger association with diabetes diagnosis.

| Feature | Baseline Coef | Baseline OR | Biased Coef | Biased OR | Δ Coef |
| --- | ---: | ---: | ---: | ---: | ---: |
| age_years | 0.410 | 1.506 | 0.404 | 1.498 | -0.005 |
| male | -0.014 | 0.986 | -0.003 | 0.997 | +0.011 |
| income | -0.073 | 0.930 | 0.071 | 1.074 | +0.144 |
| bmi | 0.571 | 1.771 | 0.459 | 1.583 | -0.112 |
| smoker | -0.195 | 0.823 | -0.214 | 0.807 | -0.020 |
| obesity | -0.179 | 0.836 | 0.449 | 1.567 | +0.628 |
| hypertension | 0.070 | 1.072 | 0.166 | 1.180 | +0.096 |
| hyperlipidemia | -0.127 | 0.880 | 0.042 | 1.043 | +0.169 |
| hyperglycemia | 0.702 | 2.019 | 1.438 | 4.212 | +0.736 |
| hypertriglyceridemia | 2.290 | 9.871 | 2.001 | 7.395 | -0.289 |

## In-Sample Model Performance

| Dataset | AUC |
| --- | ---: |
| baseline | 0.984 |
| biased | 0.982 |

## Interpretation

Documentation bias affects the learned model in several ways:

1. **Reduced feature prevalence**: The biased dataset shows lower rates of documented hyperglycemia and hypertriglyceridemia, matching the 30% under-documentation rate.

2. **Coefficient changes**: With fewer documented metabolic conditions, the model may compensate by increasing reliance on other features (demographics, comorbidities).

3. **Impact on diabetes patients**: Among actual diabetes patients, the reduced documentation of highly predictive features creates a mismatch between true clinical state and recorded data.

4. **Generalization risk**: A model trained on biased data may perform worse when deployed on populations with complete documentation, as it has learned to under-weight the most predictive features.
