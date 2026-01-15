# Sleep Apnea Diagnosis Probability Analysis

Generated: 2026-01-11T19:50:43
Dataset: `output_baseline/csv`

## Module summary (sleep_apnea.json)
- Prevalence gated to ages 30-60 with higher rates for men.
- Sleep disorder leads to referral, home or in-lab sleep study, and apnea diagnosis.
- Diagnosis codes include sleep disorder and obstructive sleep apnea.
- Treatment includes CPAP or oral appliance, with ongoing device/supply renewals.
- Urban attribute gates drop-off in the pathway (used for access bias overrides).
- Module attributes: hypertension, sleep_apnea, sleep_apnea_treatment, years_until_sleep_apnea_renewal
- Condition codes: 39898005, 73430006, 78275009
- Procedure/encounter codes: 103750000, 10563004, 185345009, 185347001, 185389009, 446573003, 60554003, 698560000, 82808001
- Device codes: 272265001, 701077002, 701100002, 702172008, 706180003, 720253003
- Supply codes: 463659001, 467645007, 704718009, 706226000, 972002

## Diagnosis prevalence
- Sleep apnea prevalence: 8.50%

## Feature specifications
- Demographics: age_years, male, income
- Risk basic: age_years, male, income, bmi, smoker
- Risk + comorbidities: age_years, male, income, bmi, smoker, alcohol_use, hypertension, chf
- Risk + comorbidities + vitals: age_years, male, income, bmi, smoker, alcohol_use, hypertension, chf, systolic_bp, diastolic_bp

## Model results
| Spec | Model | AUC | AP | Brier | Train/Test |
| --- | --- | ---: | ---: | ---: | ---: |
| Demographics | logistic | 0.809 | 0.224 | 0.192 | 16084/4022 |
| Demographics | rf | 0.819 | 0.246 | 0.181 | 16084/4022 |
| Demographics | gbdt | 0.821 | 0.237 | 0.069 | 16084/4022 |
| Risk basic | logistic | 0.816 | 0.237 | 0.192 | 16084/4022 |
| Risk basic | rf | 0.824 | 0.252 | 0.183 | 16084/4022 |
| Risk basic | gbdt | 0.827 | 0.251 | 0.069 | 16084/4022 |
| Risk + comorbidities | logistic | 0.858 | 0.306 | 0.167 | 16084/4022 |
| Risk + comorbidities | rf | 0.862 | 0.313 | 0.163 | 16084/4022 |
| Risk + comorbidities | gbdt | 0.866 | 0.334 | 0.065 | 16084/4022 |
| Risk + comorbidities + vitals | logistic | 0.859 | 0.306 | 0.167 | 16084/4022 |
| Risk + comorbidities + vitals | rf | 0.864 | 0.306 | 0.159 | 16084/4022 |
| Risk + comorbidities + vitals | gbdt | 0.868 | 0.323 | 0.065 | 16084/4022 |
