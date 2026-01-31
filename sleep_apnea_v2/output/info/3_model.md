# Model Training Report

Generated: 2026-01-31 17:21:28

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

- **Rural Recall Drop**: -0.1452 (from 0.3065 to 0.1613)
  - The biased model misses more true sleep apnea cases in rural patients
- **Urban Recall Change**: +0.0496 (from 0.4237 to 0.4733)
  - Urban recall may improve as model shifts toward majority group
- **Recall Disparity**: Rural recall changes by -0.1452 vs Urban by +0.0496
  - Bias widens the gap in who gets correctly identified
- **Rural F1 Drop**: -0.0380 (from 0.2135 to 0.1754)
  - Overall rural prediction quality degrades
- **Threshold shift**: Biased model threshold 0.1233 vs baseline 0.1405

> **Note**: AUC measures ranking across all thresholds and may remain stable even when
> recall drops significantly. Recall at the operating threshold directly measures
> missed diagnoses and is the key fairness metric for this use case.
