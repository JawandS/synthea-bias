# Diabetes Diagnosis Probability Analysis

Generated: 2026-01-11T19:49:02
Dataset: `output_baseline/csv`

## Module summary (metabolic syndrome modules)
- Diabetes onset modeled in metabolic_syndrome_disease with age-based prevalence and progression.
- Metabolic syndrome care module screens via A1c and metabolic criteria, then diagnoses prediabetes/diabetes.
- Care includes a diabetes self-management plan, medications, glucose monitoring, and DME/supplies.
- Complications modeled: kidney disease, neuropathy, anemia, and amputations (via submodules).
- Module attributes: anemia, diabetes, diabetes_amputation_necessary, diabetes_severity, diabetes_stage, diabetic_nerve_damage, diabetic_neuropathy, diabetic_retinopathy_stage, hyperglycemia, hypertension, hypertriglyceridemia, macular_edema, metabolic_syndrome, metabolic_syndrome_review, neuropathy, prediabetes, time_until_diabetes_onset, veteran
- Submodules: anemia/anemia_sub, metabolic_syndrome/amputations, metabolic_syndrome/diabetic_retinopathy_progression, metabolic_syndrome/dme_supplies, metabolic_syndrome/kidney_conditions, metabolic_syndrome/medications
- Condition codes: 237602007, 302870006, 368581000119106, 44054006, 714628002, 80394007
- Observation codes: 4548-4
- Device codes: 337414009

## Diagnosis prevalence
- Diabetes prevalence: 6.31%

## Feature specifications
- Demographics: age_years, male, income
- Risk basic: age_years, male, income, bmi, smoker
- Risk + comorbidities: age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia
- Risk + metabolic: age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, hyperglycemia, hypertriglyceridemia
- Risk + vitals: age_years, male, income, bmi, smoker, obesity, hypertension, hyperlipidemia, systolic_bp, diastolic_bp

## Model results
| Spec | Model | AUC | AP | Brier | Train/Test |
| --- | --- | ---: | ---: | ---: | ---: |
| Demographics | logistic | 0.810 | 0.168 | 0.197 | 16084/4022 |
| Demographics | rf | 0.806 | 0.161 | 0.185 | 16084/4022 |
| Demographics | gbdt | 0.799 | 0.147 | 0.057 | 16084/4022 |
| Risk basic | logistic | 0.809 | 0.171 | 0.198 | 16084/4022 |
| Risk basic | rf | 0.807 | 0.160 | 0.187 | 16084/4022 |
| Risk basic | gbdt | 0.803 | 0.153 | 0.056 | 16084/4022 |
| Risk + comorbidities | logistic | 0.831 | 0.185 | 0.186 | 16084/4022 |
| Risk + comorbidities | rf | 0.830 | 0.196 | 0.175 | 16084/4022 |
| Risk + comorbidities | gbdt | 0.828 | 0.176 | 0.055 | 16084/4022 |
| Risk + metabolic | logistic | 0.986 | 0.965 | 0.006 | 16084/4022 |
| Risk + metabolic | rf | 0.990 | 0.968 | 0.008 | 16084/4022 |
| Risk + metabolic | gbdt | 0.982 | 0.911 | 0.004 | 16084/4022 |
| Risk + vitals | logistic | 0.832 | 0.186 | 0.186 | 16084/4022 |
| Risk + vitals | rf | 0.832 | 0.214 | 0.170 | 16084/4022 |
| Risk + vitals | gbdt | 0.833 | 0.197 | 0.054 | 16084/4022 |
