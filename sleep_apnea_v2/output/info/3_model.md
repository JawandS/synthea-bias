# Model Training Report

Generated: 2026-01-31 16:51:55

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
| Baseline threshold | 0.1469 |
| Biased threshold | 0.1542 |

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
| Train | 2,594 | 244 | 9.41% |
| Validation | 557 | 53 | 9.52% |
| Test | 557 | 53 | 9.52% |

## Training Labels

| Model | Training Labels | Evaluation Labels |
|-------|-----------------|-------------------|
| Baseline | `has_sleep_apnea` (true) | `has_sleep_apnea` (true) |
| Biased | `observed_sleep_apnea` (masked) | `has_sleep_apnea` (true) |

## Overall Performance

| Metric | Baseline | Biased | Delta |
|--------|----------|--------|-------|
| AUC-ROC | 0.6498 | 0.6348 | -0.0150 |
| Avg Precision | 0.1531 | 0.1510 | -0.0021 |
| F1 Score | 0.2236 | 0.2400 | +0.0164 |
| Precision | 0.1667 | 0.1856 | +0.0189 |
| Recall | 0.3396 | 0.3396 | +0.0000 |
| Accuracy | 0.7756 | 0.7953 | +0.0197 |

## Subgroup Performance (Urban vs Rural)

### Test Set Composition

| Subgroup | Patients | Apnea Cases | Prevalence |
|----------|----------|-------------|------------|
| Urban | 462 | 44 | 9.52% |
| Rural | 95 | 9 | 9.47% |

### AUC-ROC by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.6334 | 0.6328 | -0.0005 |
| Rural | 0.7558 | 0.6744 | -0.0814 |

### Recall by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.3864 | 0.3864 | +0.0000 |
| Rural | 0.1111 | 0.1111 | +0.0000 |

### Precision by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.1683 | 0.1828 | +0.0145 |
| Rural | 0.1429 | 0.2500 | +0.1071 |

### F1 Score by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.2345 | 0.2482 | +0.0137 |
| Rural | 0.1250 | 0.1538 | +0.0288 |

## Key Findings

- **Rural AUC degradation**: -0.0814 (from 0.7558 to 0.6744)
  - The biased model has reduced ability to discriminate sleep apnea in rural patients
- **Urban AUC change**: -0.0005 (relatively stable)
- **Disparity gap**: Rural AUC changes by -0.0814 vs Urban by -0.0005
  - Bias introduces/widens performance gap between subgroups
- **Threshold shift**: Biased model uses lower threshold (0.1542 vs 0.1469)
  - Compensates for reduced signal from missing rural positives in training
