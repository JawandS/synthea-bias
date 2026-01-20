# Diabetes Case Study: Documentation Bias

This case study demonstrates how documentation bias can affect machine learning models in healthcare.
We compare a baseline Synthea simulation against a biased version where metabolic conditions
(hyperglycemia, hypertriglyceridemia) are randomly under-documented in 30% of cases.

## Technical Overview

- `scripts/load_data.py`: copies relevant CSV files from Synthea and filters observations to include A1c, BMI, and smoking status.
- `scripts/summary_stats.py`: provides summary statistics on the datasets.
- `scripts/models.py`: trains logistic regression, random forest, and gradient boosted decision tree models to predict diabetes diagnosis using defined feature sets. Writes a markdown report to `diabetes/output`.
- `scripts/analytics.py`: analyzes documentation bias effects and model performance differences, writing a report to `diabetes/output`.

## Data Generation

Use Montana for consistency with the sleep apnea case study.

### Module Setup

Before generating the biased dataset, install the modified module that supports override-based bias:
```bash
# From snythea directory
cp ../diabetes/artifacts/metabolic_syndrome_care_biased.json \
   src/main/resources/modules/metabolic_syndrome_care.json
```

The modified module uses `complex_transition` with 100%/0% distributions (identical behavior to original).
The override file changes these to 70%/30% to introduce documentation bias.

### Baseline Dataset
```bash
./run_synthea -s 160 -cs 160 -o false -p 20000 \
  --exporter.csv.export=true \
  --exporter.csv.append_mode=true \
  --exporter.baseDirectory=./output_baseline \
  Montana
```

### Biased Dataset (Documentation Bias)
```bash
./run_synthea -s 160 -cs 160 -o false -p 20000 \
  --exporter.csv.export=true \
  --exporter.csv.append_mode=true \
  --exporter.baseDirectory=./output_documentation_bias \
  --module_override=/home/js/contracts/synthea-bias/diabetes/artifacts/overrides_documentation_bias.properties \
  Montana
```

## Background

### Documentation Bias in Healthcare

Documentation bias occurs when certain clinical findings are systematically under-recorded in medical records.
This can happen due to:
- Time pressure during clinical encounters
- Variability in documentation practices across providers
- Incomplete lab panels or follow-up testing
- EHR usability issues

### The Diabetes Context

The Synthea metabolic syndrome modules record hyperglycemia and hypertriglyceridemia as distinct
conditions when patients meet clinical thresholds. These conditions are highly predictive of diabetes
(AUC 0.986-0.990 when included as features). When documentation is incomplete, models trained on
such data may:
- Rely more heavily on demographic features
- Miss important clinical signals
- Potentially perform worse on populations with complete documentation

### Module Modifications

The `artifacts/` directory contains:
- `base_metabolic_syndrome_care.json` - original Synthea module (reference)
- `metabolic_syndrome_care_biased.json` - modified module with `complex_transition` (100%/0% distributions)
- `overrides_documentation_bias.properties` - changes distributions to 70%/30%

The biased dataset introduces 30% random under-documentation of:
- **Hyperglycemia** (SNOMED 80394007)
- **Hypertriglyceridemia** (SNOMED 302870006)

In the baseline, these conditions are always recorded when clinical criteria are met.
The override modifies the `Blood_Sugar_Check` and `Triglyceride_Check` state transitions
to skip recording 30% of the time, regardless of patient characteristics.

## Model Features and Target

### Features

| Feature | Source | Description |
|---------|--------|-------------|
| `age_years` | `patients.csv` | Patient age at dataset reference date |
| `male` | `patients.csv` | Gender indicator (1.0 for male) |
| `income` | `patients.csv` | Annual household income |
| `bmi` | `observations.csv` | Latest recorded BMI (LOINC 39156-5) |
| `smoker` | `observations.csv` | Latest smoking status (LOINC 72166-2) |
| `obesity` | `conditions.csv` | Obesity diagnosis |
| `hypertension` | `conditions.csv` | Hypertension diagnosis |
| `hyperlipidemia` | `conditions.csv` | Hyperlipidemia diagnosis |
| `hyperglycemia` | `conditions.csv` | Hyperglycemia diagnosis (SNOMED 80394007) |
| `hypertriglyceridemia` | `conditions.csv` | Hypertriglyceridemia diagnosis (SNOMED 302870006) |

### Feature Sets

- **Demographics**: age_years, male, income
- **Risk basic**: age_years, male, income, bmi, smoker
- **Risk + comorbidities**: + obesity, hypertension, hyperlipidemia
- **Risk + metabolic**: + hyperglycemia, hypertriglyceridemia

### Target

Binary label: patient has a diabetes diagnosis (SNOMED 44054006) in `conditions.csv`.

## Key Codes

- **Diabetes mellitus type 2**: SNOMED 44054006
- **Prediabetes**: SNOMED 714628002
- **Hyperglycemia**: SNOMED 80394007
- **Hypertriglyceridemia**: SNOMED 302870006
- **Metabolic syndrome**: SNOMED 237602007
- **Hemoglobin A1c**: LOINC 4548-4
- **BMI**: LOINC 39156-5
