# Model Training Report

Generated: 2026-02-03 17:27:48

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
| Baseline threshold | 0.8651 |
| Biased threshold | 0.6254 |

## Features

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| male | Gender indicator (1 = male) |
| income | Household income (scaled) |
| a1c | Hemoglobin A1c level |
| bmi | Body mass index |
| smoker | Current smoker indicator |
| prediabetes | Prediabetes diagnosis |
| obesity | Obesity diagnosis |
| hypertension | Hypertension diagnosis |
| hyperlipidemia | Hyperlipidemia diagnosis |
| hypertriglyceridemia | Hypertriglyceridemia (true or observed) |

## Model Comparison

| Model | Hypertriglyceridemia Feature | Target |
|-------|------------------------------|--------|
| Baseline | `has_hypertriglyceridemia` (true) | `has_diabetes` |
| Biased | `observed_hypertriglyceridemia` (30% masked) | `has_diabetes` |

## Data Split

| Split | Patients | Diabetes Cases | Prevalence |
|-------|----------|----------------|------------|
| Train | 23,240 | 3,783 | 16.28% |
| Validation | 4,980 | 811 | 16.29% |
| Test | 4,980 | 811 | 16.29% |

## Overall Performance

| Metric | Baseline | Biased | Delta |
|--------|----------|--------|-------|
| AUC-ROC | 0.9997 | 0.9976 | -0.0021 |
| Avg Precision | 0.9939 | 0.9900 | -0.0038 |
| F1 Score | 0.9963 | 0.9598 | -0.0365 |
| Precision | 0.9939 | 0.9934 | -0.0005 |
| Recall | 0.9988 | 0.9285 | -0.0703 |
| Accuracy | 0.9988 | 0.9873 | -0.0114 |

## Feature Importance

| Feature | Baseline | Biased | Delta |
|---------|----------|--------|-------|
| age | 0.0004 | 0.0018 | +0.0014 |
| male | 0.0000 | 0.0003 | +0.0003 |
| income | 0.0011 | 0.0037 | +0.0025 |
| a1c | 0.0268 | 0.2252 | +0.1984 |
| bmi | 0.0079 | 0.0128 | +0.0049 |
| smoker | 0.0000 | 0.0000 | +0.0000 |
| prediabetes | 0.0001 | 0.0289 | +0.0288 |
| obesity | 0.0000 | 0.0006 | +0.0006 |
| hypertension | 0.0001 | 0.0237 | +0.0236 |
| hyperlipidemia | 0.0000 | 0.0004 | +0.0004 |
| hypertriglyceridemia | 0.9636 | 0.7026 | -0.2610 |

## Subgroup Performance

Performance grouped by TRUE hypertriglyceridemia status.

### Test Set Composition

| Subgroup | Patients | Diabetes Cases | Prevalence |
|----------|----------|----------------|------------|
| With hypertriglyceridemia | 805 | 794 | 98.63% |
| Without hypertriglyceridemia | 4,175 | 17 | 0.41% |

### AUC-ROC by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| With hypertriglyceridemia | 0.9746 | 0.9336 | -0.0410 |
| Without hypertriglyceridemia | 0.9998 | 0.9998 | +0.0000 |

### Recall by Subgroup

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| With hypertriglyceridemia | 0.9987 | 0.9270 | -0.0718 |
| Without hypertriglyceridemia | 1.0000 | 1.0000 | +0.0000 |

## Key Findings

- **Overall AUC change**: -0.0021 (from 0.9997 to 0.9976)
- **Overall F1 change**: -0.0365 (from 0.9963 to 0.9598)
- **Hypertriglyceridemia importance**: 0.9636 -> 0.7026 (-0.2610)

> **Documentation bias effect**: When hypertriglyceridemia is under-documented, the model learns
> a weaker association between this condition and diabetes. The model may compensate by relying
> more heavily on other features like A1c, BMI, or other comorbidities.
