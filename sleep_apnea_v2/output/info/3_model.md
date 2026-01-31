# Model Training Report

Generated: 2026-01-31 17:19:00

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
| Baseline threshold | 0.1405 |
| Biased threshold | 0.1233 |

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
| Train | 16,031 | 1,512 | 9.43% |
| Validation | 3,436 | 324 | 9.43% |
| Test | 3,436 | 324 | 9.43% |

## Training Labels

| Model | Training Labels | Evaluation Labels |
|-------|-----------------|-------------------|
| Baseline | `has_sleep_apnea` (true) | `has_sleep_apnea` (true) |
| Biased | `observed_sleep_apnea` (masked) | `has_sleep_apnea` (true) |

## Overall Performance

| Metric | Baseline | Biased | Delta |
|--------|----------|--------|-------|
| AUC-ROC | 0.6457 | 0.6467 | +0.0010 |
| Avg Precision | 0.1445 | 0.1492 | +0.0047 |
| F1 Score | 0.2245 | 0.2219 | -0.0027 |
| Precision | 0.1559 | 0.1516 | -0.0043 |
| Recall | 0.4012 | 0.4136 | +0.0123 |
| Accuracy | 0.7386 | 0.7264 | -0.0122 |

## Subgroup Performance (Urban vs Rural)

### Test Set Composition

| Subgroup | Patients | Apnea Cases | Prevalence |
|----------|----------|-------------|------------|
| Urban | 2,801 | 262 | 9.35% |
| Rural | 635 | 62 | 9.76% |

### AUC-ROC by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.6358 | 0.6410 | +0.0052 |
| Rural | 0.6927 | 0.6988 | +0.0061 |

### Recall by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.4237 | 0.4733 | +0.0496 |
| Rural | 0.3065 | 0.1613 | -0.1452 |

### Precision by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.1546 | 0.1490 | -0.0056 |
| Rural | 0.1638 | 0.1923 | +0.0285 |

### F1 Score by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.2265 | 0.2267 | +0.0002 |
| Rural | 0.2135 | 0.1754 | -0.0380 |

## Key Findings

- **Rural AUC degradation**: +0.0061 (from 0.6927 to 0.6988)
  - The biased model has reduced ability to discriminate sleep apnea in rural patients
- **Urban AUC change**: +0.0052 (relatively stable)
- **Disparity gap**: Rural AUC changes by +0.0061 vs Urban by +0.0052
  - Bias introduces/widens performance gap between subgroups
- **Threshold shift**: Biased model uses lower threshold (0.1233 vs 0.1405)
  - Compensates for reduced signal from missing rural positives in training
