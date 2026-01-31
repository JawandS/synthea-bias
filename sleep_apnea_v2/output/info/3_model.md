# Model Training Report

Generated: 2026-01-31 15:34:20

## Model Specification

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosted Decision Tree |
| n_estimators | 200 |
| max_depth | 5 |
| learning_rate | 0.05 |
| min_samples_split | 20 |
| min_samples_leaf | 10 |
| subsample | 0.8 |

## Threshold Selection

| Parameter | Value |
|-----------|-------|
| Method | f1 (maximize F1 on validation set) |
| Baseline threshold | 0.0767 |
| Biased threshold | 0.0543 |

> **Note**: With ~9% class prevalence, the default 0.5 threshold would rarely predict positives.
> Adaptive thresholding finds the optimal operating point that balances precision and recall.

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
| Train | 641 | 59 | 9.20% |
| Validation | 138 | 13 | 9.42% |
| Test | 138 | 13 | 9.42% |

## Training Labels

| Model | Training Labels | Evaluation Labels |
|-------|-----------------|-------------------|
| Baseline | `has_sleep_apnea` (true) | `has_sleep_apnea` (true) |
| Biased | `observed_sleep_apnea` (masked) | `has_sleep_apnea` (true) |

## Overall Performance

| Metric | Baseline | Biased | Delta |
|--------|----------|--------|-------|
| AUC-ROC | 0.5415 | 0.5200 | -0.0215 |
| Avg Precision | 0.1141 | 0.1166 | +0.0026 |
| F1 Score | 0.1905 | 0.1961 | +0.0056 |
| Precision | 0.1379 | 0.1316 | -0.0064 |
| Recall | 0.3077 | 0.3846 | +0.0769 |
| Accuracy | 0.7536 | 0.7029 | -0.0507 |

## Subgroup Performance (Urban vs Rural)

### Test Set Composition

| Subgroup | Patients | Apnea Cases | Prevalence |
|----------|----------|-------------|------------|
| Urban | 113 | 10 | 8.85% |
| Rural | 25 | 3 | 12.00% |

### AUC-ROC by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.5553 | 0.5311 | -0.0243 |
| Rural | 0.4848 | 0.4545 | -0.0303 |

### Recall by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.3000 | 0.4000 | +0.1000 |
| Rural | 0.3333 | 0.3333 | +0.0000 |

### Precision by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.1154 | 0.1143 | -0.0011 |
| Rural | 0.3333 | 0.3333 | +0.0000 |

### F1 Score by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.1667 | 0.1778 | +0.0111 |
| Rural | 0.3333 | 0.3333 | +0.0000 |

## Key Findings

- **Rural AUC degradation**: -0.0303 (from 0.4848 to 0.4545)
  - The biased model has reduced ability to discriminate sleep apnea in rural patients
- **Urban AUC change**: -0.0243 (relatively stable)
- **Disparity gap**: Rural AUC changes by -0.0303 vs Urban by -0.0243
  - Bias introduces/widens performance gap between subgroups
- **Threshold shift**: Biased model uses lower threshold (0.0543 vs 0.0767)
  - Compensates for reduced signal from missing rural positives in training
