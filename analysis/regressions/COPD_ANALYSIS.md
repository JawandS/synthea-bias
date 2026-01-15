# COPD Diagnosis Probability Analysis

COPD: Chronic Obstructive Pulmonary Disease (lung disease)
Generated: 2026-01-11T19:38:04
Dataset: `output_baseline/csv`

## Module summary (copd.json)
- COPD incidence begins at age 18 and branches by smoking status and socioeconomic status.
- Two COPD variants: emphysema and chronic bronchitis, tracked via `copd_variant`.
- Diagnosis encounter includes spirometry and FEV1/FVC observation to stage severity.
- Care plans include smoking cessation, pulmonary rehab, oxygen therapy, and inhaled meds.
- Severe disease may trigger lung volume reduction surgery or transplant; hospice possible.
- Submodules: anemia workup and wheelchair durable medical equipment.
- Module attributes: anemia, copd_variant, hospice, hospice_reason, smoker
- Submodules: anemia/anemia_sub, dme/wheelchair
- Condition codes: 185086009, 87433001
- Procedure/encounter codes: 127783003, 133899007, 15081005, 225415001, 305411003, 429609002, 88039007
- Observation codes: 19926-5
- Medication codes: 245314, 896209
- Device codes: 170615005, 261323006, 336621006, 464173000

## Diagnosis prevalence
- COPD prevalence: 2.69%

## Feature specifications
- Demographics: age_years, male, income
- Risk basic: age_years, male, income, bmi, smoker
- Risk + asthma: age_years, male, income, bmi, smoker, asthma
- Risk + asthma + vitals: age_years, male, income, bmi, smoker, asthma, systolic_bp, diastolic_bp

## Model results
| Spec | Model | AUC | AP | Brier | Train/Test |
| --- | --- | ---: | ---: | ---: | ---: |
| Demographics | logistic | 0.793 | 0.081 | 0.206 | 16084/4022 |
| Demographics | rf | 0.782 | 0.078 | 0.178 | 16084/4022 |
| Demographics | gbdt | 0.782 | 0.070 | 0.028 | 16084/4022 |
| Risk basic | logistic | 0.791 | 0.081 | 0.208 | 16084/4022 |
| Risk basic | rf | 0.791 | 0.086 | 0.184 | 16084/4022 |
| Risk basic | gbdt | 0.783 | 0.069 | 0.028 | 16084/4022 |
| Risk + asthma | logistic | 0.791 | 0.081 | 0.208 | 16084/4022 |
| Risk + asthma | rf | 0.792 | 0.090 | 0.191 | 16084/4022 |
| Risk + asthma | gbdt | 0.776 | 0.065 | 0.027 | 16084/4022 |
| Risk + asthma + vitals | logistic | 0.800 | 0.090 | 0.208 | 16084/4022 |
| Risk + asthma + vitals | rf | 0.773 | 0.083 | 0.184 | 16084/4022 |
| Risk + asthma + vitals | gbdt | 0.769 | 0.067 | 0.027 | 16084/4022 |
