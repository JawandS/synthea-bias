# MI Diagnosis Probability Analysis

Generated: 2026-01-11T19:24:40
Dataset: `output_baseline/csv`

## Module summary (myocardial_infarction.json)
- Monthly MI onset governed by patient attribute `mi_risk`.
- MI onset records acute MI and can result in pre-hospital death.
- Emergency encounter triggers cardiac assessment, ECG, labs, imaging, and diagnostic assessment.
- Splits into STEMI vs NSTEACS pathways (submodules) and can lead to PCI or CABG.
- Records history of MI after recovery; includes discharge care plan and medications.
- Uses attributes such as `chance_of_mi_death`, `cardiac_surgery_reason`, and `ACS_CABG_referral`.
- Module attributes: ACS_CABG_referral, cardiac_surgery_reason, chance_of_mi_death, mi_risk
- Condition codes: 22298006, 399211009
- Procedure/encounter codes: 15220000, 165197003, 29303009, 399208008, 433236007, 50849002, 710839006

## Diagnosis prevalence
- MI prevalence: 2.99%

## Feature specifications
- Demographics: age_years, male, income
- Risk basic: age_years, male, income, bmi, smoker
- Risk + comorbidities: age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, diabetes_any
- Risk + comorbidities + vitals: age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, diabetes_any, systolic_bp, diastolic_bp

## Model results
| Spec | Model | AUC | AP | Brier | Train/Test |
| --- | --- | ---: | ---: | ---: | ---: |
| Demographics | logistic | 0.906 | 0.185 | 0.143 | 16084/4022 |
| Demographics | rf | 0.900 | 0.170 | 0.129 | 16084/4022 |
| Demographics | gbdt | 0.896 | 0.151 | 0.028 | 16084/4022 |
| Risk basic | logistic | 0.905 | 0.183 | 0.144 | 16084/4022 |
| Risk basic | rf | 0.904 | 0.183 | 0.131 | 16084/4022 |
| Risk basic | gbdt | 0.898 | 0.162 | 0.028 | 16084/4022 |
| Risk + comorbidities | logistic | 0.909 | 0.197 | 0.142 | 16084/4022 |
| Risk + comorbidities | rf | 0.904 | 0.216 | 0.127 | 16084/4022 |
| Risk + comorbidities | gbdt | 0.901 | 0.168 | 0.027 | 16084/4022 |
| Risk + comorbidities + vitals | logistic | 0.911 | 0.202 | 0.141 | 16084/4022 |
| Risk + comorbidities + vitals | rf | 0.907 | 0.232 | 0.123 | 16084/4022 |
| Risk + comorbidities + vitals | gbdt | 0.905 | 0.190 | 0.027 | 16084/4022 |
