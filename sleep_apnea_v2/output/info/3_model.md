# Model Training Report

Generated: 2026-01-31 10:39:21

## Model Specification

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosted Decision Tree |
| n_estimators | 100 |
| max_depth | 4 |
| learning_rate | 0.1 |
| min_samples_split | 20 |
| min_samples_leaf | 10 |

## Features

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| male | Gender (1 = male) |
| urban | Location (1 = urban) |
| income | Household income (scaled) |
| bmi | Body mass index |
| smoker | Current smoker (1 = yes) |
| hypertension | Has hypertension diagnosis |
| chf | Has CHF diagnosis |
| alcohol_use | Has alcohol use disorder |

## Data Split

| Split | Patients | Sleep Apnea Cases | Prevalence |
|-------|----------|-------------------|------------|
| Train | 1,233 | 101 | 8.19% |
| Validation | 265 | 22 | 8.30% |
| Test | 265 | 22 | 8.30% |

## Training Labels

| Model | Training Labels | Evaluation Labels |
|-------|-----------------|-------------------|
| Baseline | `has_sleep_apnea` (true) | `has_sleep_apnea` (true) |
| Biased | `observed_sleep_apnea` (masked) | `has_sleep_apnea` (true) |

## Overall Performance

| Metric | Baseline | Biased | Delta |
|--------|----------|--------|-------|
| AUC-ROC | 0.6629 | 0.6650 | +0.0021 |
| Avg Precision | 0.1467 | 0.1570 | +0.0103 |
| Accuracy | 0.9057 | 0.9057 | +0.0000 |
| Precision | 0.0000 | 0.0000 | +0.0000 |
| Recall | 0.0000 | 0.0000 | +0.0000 |
| F1 Score | 0.0000 | 0.0000 | +0.0000 |

## Subgroup Performance (Urban vs Rural)

### AUC-ROC by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.6570 | 0.6586 | +0.0016 |
| Rural | 0.0000 | 0.0000 | +0.0000 |

### Recall by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.0000 | 0.0000 | +0.0000 |
| Rural | 0.0000 | 0.0000 | +0.0000 |

### Precision by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.0000 | 0.0000 | +0.0000 |
| Rural | 0.0000 | 0.0000 | +0.0000 |

## Key Findings

- **Overall AUC degradation**: +0.0021 (from 0.6629 to 0.6650)
- **Rural recall degradation**: +0.0000 (most affected by underdiagnosis bias)
- **Urban performance**: Relatively stable as bias only affects rural population
