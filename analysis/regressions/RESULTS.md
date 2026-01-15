# Regression Feasibility Results

Generated: 2026-01-11T19:10:59
Dataset: `output_baseline/csv`
Linear model: Ridge regression on log1p(spend) with standardized features.
Diagnosis model: Logistic regression with standardized features.

## Diabetes care

Target: log1p(diabetes-related spend), from encounters/procedures/medications with diabetes/prediabetes reason codes
Reason codes: 44054006, 714628002
Cohort codes: 44054006

Population size: 20106
Diagnosed patients: 1268 (6.31%)
Nonzero spend rate (population): 9.24%
Mean spend (nonzero only): $23,298.63

Diagnosis probability model (logistic regression):
- AUC: 1.000 | Brier: 0.002 | Prevalence: 6.31% | Train/Test: 16084/4022
- Features: age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, hyperglycemia, hypertriglyceridemia, a1c, glucose, triglycerides, hdl, ldl, total_cholesterol, systolic_bp, diastolic_bp
- Feature gaps (>30% missing): a1c (61.96% missing), glucose (56.36% missing), triglycerides (41.68% missing), hdl (41.68% missing), ldl (41.68% missing), total_cholesterol (41.68% missing)

| Spec | Cohort | Features | N | Nonzero | R2 | MAE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Population risk | all | 10 | 20106 | 9.24% | 0.124 | $1,807 | $7,426 |
| Population risk + labs/vitals | all | 18 | 20106 | 9.24% | -481.041 | $14,049 | $174,226 |
| Population risk + diagnosis flag | all | 11 | 20106 | 9.24% | 0.126 | $1,800 | $7,420 |
| Population risk + diagnosis probability | all | 11 | 20106 | 9.24% | 0.124 | $1,807 | $7,427 |
| Population risk + utilization | all | 18 | 20106 | 9.24% | -9972.648 | $22,400 | $792,497 |
| Diagnosed cohort risk + labs/vitals | diagnosed | 18 | 1268 | 92.11% | 0.143 | $13,399 | $20,615 |

Specifications:
- Population risk (all): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, hyperglycemia, hypertriglyceridemia
- Population risk + labs/vitals (all): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, hyperglycemia, hypertriglyceridemia, a1c, glucose, triglycerides, hdl, ldl, total_cholesterol, systolic_bp, diastolic_bp
- Population risk + diagnosis flag (all): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, hyperglycemia, hypertriglyceridemia, diabetes_dx
- Population risk + diagnosis flag note: Includes diagnosis indicator.
- Population risk + diagnosis probability (all): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, hyperglycemia, hypertriglyceridemia, diagnosis_prob_diabetes
- Population risk + diagnosis probability note: Includes diagnosis probability.
- Population risk + utilization (all): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, hyperglycemia, hypertriglyceridemia, encounter_count, encounter_emergency, encounter_inpatient, encounter_ambulatory, encounter_outpatient, procedure_count, medication_count, condition_count
- Population risk + utilization note: Includes utilization counts (proxy for spend).
- Diagnosed cohort risk + labs/vitals (diagnosed): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, hyperglycemia, hypertriglyceridemia, a1c, glucose, triglycerides, hdl, ldl, total_cholesterol, systolic_bp, diastolic_bp

Feature gaps (>30% missing) for Population risk + labs/vitals: a1c (61.96% missing), glucose (56.36% missing), triglycerides (41.68% missing), hdl (41.68% missing), ldl (41.68% missing), total_cholesterol (41.68% missing)

Judgement: Weak-to-moderate signal; may be usable but expect limited performance.

## COPD

Target: log1p(COPD-related spend), from encounters/procedures/medications with COPD reason codes
Reason codes: 185086009, 87433001
Cohort codes: 185086009, 87433001

Population size: 20106
Diagnosed patients: 540 (2.69%)
Nonzero spend rate (population): 2.69%
Mean spend (nonzero only): $158,137.29

Diagnosis probability model (logistic regression):
- AUC: 0.754 | Brier: 0.211 | Prevalence: 2.69% | Train/Test: 16084/4022
- Features: age_years, male, income, bmi, smoker, asthma, fev1_fvc
- Feature gaps (>30% missing): fev1_fvc (97.31% missing)

| Spec | Cohort | Features | N | Nonzero | R2 | MAE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Population risk | all | 6 | 20106 | 2.69% | -0.012 | $4,459 | $40,515 |
| Population risk + FEV1/FVC | all | 7 | 20106 | 2.69% | -0.012 | $4,459 | $40,515 |
| Population risk + diagnosis flag | all | 7 | 20106 | 2.69% | 0.389 | $2,583 | $31,469 |
| Population risk + diagnosis probability | all | 7 | 20106 | 2.69% | -0.012 | $4,459 | $40,515 |
| Population risk + utilization | all | 14 | 20106 | 2.69% | -0.012 | $4,459 | $40,515 |
| Diagnosed cohort risk + FEV1/FVC | diagnosed | 7 | 540 | 100.00% | 0.275 | $68,413 | $115,041 |

Specifications:
- Population risk (all): age_years, male, income, bmi, smoker, asthma
- Population risk + FEV1/FVC (all): age_years, male, income, bmi, smoker, asthma, fev1_fvc
- Population risk + diagnosis flag (all): age_years, male, income, bmi, smoker, asthma, copd_dx
- Population risk + diagnosis flag note: Includes diagnosis indicator.
- Population risk + diagnosis probability (all): age_years, male, income, bmi, smoker, asthma, diagnosis_prob_copd
- Population risk + diagnosis probability note: Includes diagnosis probability.
- Population risk + utilization (all): age_years, male, income, bmi, smoker, asthma, encounter_count, encounter_emergency, encounter_inpatient, encounter_ambulatory, encounter_outpatient, procedure_count, medication_count, condition_count
- Population risk + utilization note: Includes utilization counts (proxy for spend).
- Diagnosed cohort risk + FEV1/FVC (diagnosed): age_years, male, income, bmi, smoker, asthma, fev1_fvc

Feature gaps (>30% missing) for Population risk + FEV1/FVC: fev1_fvc (97.31% missing)

Judgement: Good signal in linear models; data looks usable for modeling.

## Myocardial infarction

Target: log1p(MI-related spend), from encounters/procedures/medications with MI reason codes
Reason codes: 22298006, 399211009, 401303003, 401314000
Cohort codes: 22298006, 399211009, 401303003, 401314000

Population size: 20106
Diagnosed patients: 601 (2.99%)
Nonzero spend rate (population): 3.82%
Mean spend (nonzero only): $52,932.54

Diagnosis probability model (logistic regression):
- AUC: 0.910 | Brier: 0.136 | Prevalence: 2.99% | Train/Test: 16084/4022
- Features: age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, diabetes_any, chf, systolic_bp, diastolic_bp, total_cholesterol, ldl, hdl, triglycerides, glucose, a1c
- Feature gaps (>30% missing): total_cholesterol (41.68% missing), ldl (41.68% missing), hdl (41.68% missing), triglycerides (41.68% missing), glucose (56.36% missing), a1c (61.96% missing)

| Spec | Cohort | Features | N | Nonzero | R2 | MAE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Population risk | all | 10 | 20106 | 3.82% | -0.018 | $1,562 | $11,570 |
| Population risk + labs/vitals | all | 18 | 20106 | 3.82% | -0.018 | $1,562 | $11,570 |
| Population risk + diagnosis flag | all | 11 | 20106 | 3.82% | 0.768 | $630 | $5,519 |
| Population risk + diagnosis probability | all | 11 | 20106 | 3.82% | -0.018 | $1,562 | $11,570 |
| Population risk + utilization | all | 18 | 20106 | 3.82% | -0.018 | $1,562 | $11,570 |
| Diagnosed cohort risk + labs/vitals | diagnosed | 18 | 601 | 100.00% | -0.117 | $26,681 | $35,096 |

Specifications:
- Population risk (all): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, diabetes_any, chf
- Population risk + labs/vitals (all): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, diabetes_any, chf, systolic_bp, diastolic_bp, total_cholesterol, ldl, hdl, triglycerides, glucose, a1c
- Population risk + diagnosis flag (all): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, diabetes_any, chf, mi_dx
- Population risk + diagnosis flag note: Includes diagnosis indicator.
- Population risk + diagnosis probability (all): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, diabetes_any, chf, diagnosis_prob_mi
- Population risk + diagnosis probability note: Includes diagnosis probability.
- Population risk + utilization (all): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, diabetes_any, chf, encounter_count, encounter_emergency, encounter_inpatient, encounter_ambulatory, encounter_outpatient, procedure_count, medication_count, condition_count
- Population risk + utilization note: Includes utilization counts (proxy for spend).
- Diagnosed cohort risk + labs/vitals (diagnosed): age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, diabetes_any, chf, systolic_bp, diastolic_bp, total_cholesterol, ldl, hdl, triglycerides, glucose, a1c

Feature gaps (>30% missing) for Population risk + labs/vitals: total_cholesterol (41.68% missing), ldl (41.68% missing), hdl (41.68% missing), triglycerides (41.68% missing), glucose (56.36% missing), a1c (61.96% missing)

Judgement: Good signal in linear models; data looks usable for modeling.
