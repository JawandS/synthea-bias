# Intersectional Case Study: Age + Income Access Bias in Colorectal Cancer

Generated: 2026-02-16 15:16:20

## Scenario

This case study models an intersectional access pattern where age and income jointly determine
screening access generosity. Generous plans screen earlier; restrictive plans screen later.
The underlying disease burden (true CRC) remains unchanged, while observed diagnosis and
observed early-stage detection are masked by plan-specific access barriers.

## Policy Rules

| Plan | Income Min | Income Max | Age Range | Screen Start Age | CRC Mask Rate | Early Mask Rate |
|------|------------:|-----------:|----------:|-----------------:|--------------:|----------------:|
| high_access | $90,000 | $1,000,000,000 | 40-49 | 45 | 8% | 18% |
| high_access | $90,000 | $1,000,000,000 | 50-64 | 45 | 5% | 12% |
| high_access | $90,000 | $1,000,000,000 | 65-74 | 45 | 4% | 10% |
| high_access | $90,000 | $1,000,000,000 | 75-100 | 50 | 8% | 14% |
| standard_access | $40,000 | $90,000 | 40-49 | 50 | 18% | 32% |
| standard_access | $40,000 | $90,000 | 50-64 | 50 | 12% | 24% |
| standard_access | $40,000 | $90,000 | 65-74 | 50 | 10% | 20% |
| standard_access | $40,000 | $90,000 | 75-100 | 55 | 14% | 24% |
| restricted_access | $0 | $40,000 | 40-49 | 55 | 36% | 55% |
| restricted_access | $0 | $40,000 | 50-64 | 55 | 28% | 45% |
| restricted_access | $0 | $40,000 | 65-74 | 55 | 20% | 34% |
| restricted_access | $0 | $40,000 | 75-100 | 60 | 26% | 40% |

## Baseline Data Summary

# Baseline Summary Stats

Generated: 2026-02-16 15:16:01

## Run Parameters

| Parameter | Value |
|-----------|-------|
| Requested population | 20,000 |
| Seed | 160 |
| Reference date for age | 2026-02-03 |

## Cohort Summary

| Metric | Value |
|--------|-------|
| Patients | 33,200 |
| True CRC cases | 1,057 |
| True CRC prevalence | 3.18% |
| Early-stage CRC (I/II) | 536 |
| Early among CRC cases | 50.71% |
| BMI/smoking observations kept | 766,043 |

## Condition Rows Kept

| Condition | Rows |
|-----------|------|
| Diabetes | 5,405 |
| Prediabetes | 20,149 |
| Obesity | 25,959 |
| Hypertension | 14,558 |
| Hyperlipidemia | 8,570 |
| CHF | 2,788 |

## CRC by Age Band

| Age band | Patients | CRC cases | Prevalence |
|----------|----------|-----------|------------|
| 40-49 | 3,479 | 7 | 0.20% |
| 50-59 | 3,669 | 60 | 1.64% |
| 60-69 | 4,241 | 147 | 3.47% |
| 70-79 | 4,699 | 207 | 4.41% |
| 80+ | 17,112 | 636 | 3.72% |

## Bias Effect Analysis

# Bias Effect Report

Generated: 2026-02-16 15:16:05

## Inputs

| Parameter | Value |
|-----------|-------|
| Rules file | `/home/js/synthea-bias/age_income/config/plan_rules.csv` |
| Seed | 42 |

## Age Stratification Multipliers

| Age band | Mask multiplier |
|----------|-----------------|
| 40-49 | 1.30 |
| 50-59 | 1.15 |
| 60-69 | 1.00 |
| 70-79 | 0.90 |
| 80+ | 0.85 |

## Overall Effect

| Metric | True | Observed | Relative drop |
|--------|------|----------|---------------|
| CRC cases | 1,057 | 890 | 15.80% |
| Early CRC cases | 536 | 352 | 34.33% |
| CRC prevalence | 3.18% | 2.68% | - |

## By Assigned Plan

| Plan | Patients | True CRC | Observed CRC | True Early CRC | Observed Early CRC | Mean effective CRC mask rate |
|------|----------|----------|--------------|----------------|--------------------|-------------------------------|
| high_access | 6,568 | 219 | 201 | 110 | 92 | 0.066 |
| restricted_access | 14,253 | 467 | 353 | 228 | 119 | 0.257 |
| standard_access | 12,379 | 371 | 336 | 198 | 141 | 0.130 |

## By Age Band

| Age band | Patients | True CRC | Observed CRC | Mean effective CRC mask rate | True prevalence | Observed prevalence |
|----------|----------|----------|--------------|-------------------------------|-----------------|---------------------|
| 40-49 | 3,479 | 7 | 3 | 0.302 | 0.20% | 0.09% |
| 50-59 | 3,669 | 60 | 45 | 0.196 | 1.64% | 1.23% |
| 60-69 | 4,241 | 147 | 122 | 0.150 | 3.47% | 2.88% |
| 70-79 | 4,699 | 207 | 172 | 0.141 | 4.41% | 3.66% |
| 80+ | 17,112 | 636 | 548 | 0.154 | 3.72% | 3.20% |

## Age x Income Quintile

| Age band | Income quintile | Patients | True CRC | Observed CRC |
|----------|------------------|----------|----------|--------------|
| 40-49 | Q1 | 661 | 1 | 0 |
| 40-49 | Q2 | 677 | 2 | 0 |
| 40-49 | Q3 | 682 | 1 | 1 |
| 40-49 | Q4 | 696 | 1 | 1 |
| 40-49 | Q5 | 763 | 2 | 1 |
| 50-59 | Q1 | 718 | 7 | 4 |
| 50-59 | Q2 | 681 | 17 | 11 |
| 50-59 | Q3 | 780 | 14 | 11 |
| 50-59 | Q4 | 713 | 12 | 9 |
| 50-59 | Q5 | 777 | 10 | 10 |
| 60-69 | Q1 | 800 | 22 | 15 |
| 60-69 | Q2 | 844 | 27 | 20 |
| 60-69 | Q3 | 880 | 31 | 25 |
| 60-69 | Q4 | 851 | 33 | 29 |
| 60-69 | Q5 | 866 | 34 | 33 |
| 70-79 | Q1 | 948 | 48 | 38 |
| 70-79 | Q2 | 909 | 43 | 35 |
| 70-79 | Q3 | 968 | 38 | 30 |
| 70-79 | Q4 | 948 | 37 | 34 |
| 70-79 | Q5 | 926 | 41 | 35 |
| 80+ | Q1 | 3,515 | 134 | 101 |
| 80+ | Q2 | 3,527 | 136 | 111 |
| 80+ | Q3 | 3,331 | 117 | 108 |
| 80+ | Q4 | 3,434 | 113 | 102 |
| 80+ | Q5 | 3,305 | 136 | 126 |

## Modeling Results

# Model Results

Generated: 2026-02-16 15:16:19

## Configuration

- Samples: 33,200
- Features: age, male, bmi, smoker, diabetes, prediabetes, obesity, hypertension, hyperlipidemia, chf
- Excluded from model features: income, poverty_ratio, assigned_plan, eligible_for_screening
- Train size: 23,240
- Validation size: 4,980
- Test size: 4,980
- Seed: 42

## CRC Diagnosis

Thresholds: baseline `0.100`, biased `0.100`

| Metric | Baseline (train=true) | Biased (train=observed) | Delta |
|--------|------------------------|--------------------------|-------|
| AUC | 0.6072 | 0.6145 | +0.0073 |
| AP | 0.0463 | 0.0485 | +0.0022 |
| ACCURACY | 0.9637 | 0.9635 | -0.0002 |
| PRECISION | 0.0000 | 0.0000 | +0.0000 |
| RECALL | 0.0000 | 0.0000 | +0.0000 |
| F1 | 0.0000 | 0.0000 | +0.0000 |

### CRC Diagnosis - Age Band Subgroup Metrics (Baseline)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49 | 531 | 0 | 0.000 | 0.000 |
| 50-59 | 504 | 9 | 0.000 | 1.000 |
| 60-69 | 634 | 26 | 0.000 | 1.000 |
| 70-79 | 682 | 29 | 0.000 | 1.000 |
| 80+ | 2,629 | 112 | 0.000 | 1.000 |

### CRC Diagnosis - Age Band Subgroup Metrics (Biased)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49 | 531 | 0 | 0.000 | 0.000 |
| 50-59 | 504 | 9 | 0.000 | 1.000 |
| 60-69 | 634 | 26 | 0.000 | 1.000 |
| 70-79 | 682 | 29 | 0.000 | 1.000 |
| 80+ | 2,629 | 112 | 0.000 | 1.000 |

### CRC Diagnosis - Income Quintile Subgroup Metrics (Baseline)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| Q1 | 996 | 30 | 0.000 | 1.000 |
| Q2 | 996 | 30 | 0.000 | 1.000 |
| Q3 | 996 | 36 | 0.000 | 1.000 |
| Q4 | 996 | 40 | 0.000 | 1.000 |
| Q5 | 996 | 40 | 0.000 | 1.000 |

### CRC Diagnosis - Income Quintile Subgroup Metrics (Biased)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| Q1 | 996 | 30 | 0.000 | 1.000 |
| Q2 | 996 | 30 | 0.000 | 1.000 |
| Q3 | 996 | 36 | 0.000 | 1.000 |
| Q4 | 996 | 40 | 0.000 | 1.000 |
| Q5 | 996 | 40 | 0.000 | 1.000 |

### CRC Diagnosis - Age x Income Subgroup Metrics (Baseline)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49|Q1 | 98 | 0 | 0.000 | 0.000 |
| 40-49|Q2 | 104 | 0 | 0.000 | 0.000 |
| 40-49|Q3 | 98 | 0 | 0.000 | 0.000 |
| 40-49|Q4 | 106 | 0 | 0.000 | 0.000 |
| 40-49|Q5 | 125 | 0 | 0.000 | 0.000 |
| 50-59|Q1 | 103 | 2 | 0.000 | 1.000 |
| 50-59|Q2 | 86 | 0 | 0.000 | 0.000 |
| 50-59|Q3 | 117 | 4 | 0.000 | 1.000 |
| 50-59|Q4 | 96 | 2 | 0.000 | 1.000 |
| 50-59|Q5 | 102 | 1 | 0.000 | 1.000 |
| 60-69|Q1 | 112 | 1 | 0.000 | 1.000 |
| 60-69|Q2 | 131 | 5 | 0.000 | 1.000 |
| 60-69|Q3 | 127 | 4 | 0.000 | 1.000 |
| 60-69|Q4 | 130 | 6 | 0.000 | 1.000 |
| 60-69|Q5 | 134 | 10 | 0.000 | 1.000 |
| 70-79|Q1 | 113 | 7 | 0.000 | 1.000 |
| 70-79|Q2 | 128 | 6 | 0.000 | 1.000 |
| 70-79|Q3 | 144 | 2 | 0.000 | 1.000 |
| 70-79|Q4 | 145 | 7 | 0.000 | 1.000 |
| 70-79|Q5 | 152 | 7 | 0.000 | 1.000 |
| 80+|Q1 | 570 | 20 | 0.000 | 1.000 |
| 80+|Q2 | 547 | 19 | 0.000 | 1.000 |
| 80+|Q3 | 510 | 26 | 0.000 | 1.000 |
| 80+|Q4 | 519 | 25 | 0.000 | 1.000 |
| 80+|Q5 | 483 | 22 | 0.000 | 1.000 |

### CRC Diagnosis - Age x Income Subgroup Metrics (Biased)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49|Q1 | 98 | 0 | 0.000 | 0.000 |
| 40-49|Q2 | 104 | 0 | 0.000 | 0.000 |
| 40-49|Q3 | 98 | 0 | 0.000 | 0.000 |
| 40-49|Q4 | 106 | 0 | 0.000 | 0.000 |
| 40-49|Q5 | 125 | 0 | 0.000 | 0.000 |
| 50-59|Q1 | 103 | 2 | 0.000 | 1.000 |
| 50-59|Q2 | 86 | 0 | 0.000 | 0.000 |
| 50-59|Q3 | 117 | 4 | 0.000 | 1.000 |
| 50-59|Q4 | 96 | 2 | 0.000 | 1.000 |
| 50-59|Q5 | 102 | 1 | 0.000 | 1.000 |
| 60-69|Q1 | 112 | 1 | 0.000 | 1.000 |
| 60-69|Q2 | 131 | 5 | 0.000 | 1.000 |
| 60-69|Q3 | 127 | 4 | 0.000 | 1.000 |
| 60-69|Q4 | 130 | 6 | 0.000 | 1.000 |
| 60-69|Q5 | 134 | 10 | 0.000 | 1.000 |
| 70-79|Q1 | 113 | 7 | 0.000 | 1.000 |
| 70-79|Q2 | 128 | 6 | 0.000 | 1.000 |
| 70-79|Q3 | 144 | 2 | 0.000 | 1.000 |
| 70-79|Q4 | 145 | 7 | 0.000 | 1.000 |
| 70-79|Q5 | 152 | 7 | 0.000 | 1.000 |
| 80+|Q1 | 570 | 20 | 0.000 | 1.000 |
| 80+|Q2 | 547 | 19 | 0.000 | 1.000 |
| 80+|Q3 | 510 | 26 | 0.000 | 1.000 |
| 80+|Q4 | 519 | 25 | 0.000 | 1.000 |
| 80+|Q5 | 483 | 22 | 0.000 | 1.000 |

## Early CRC Detection

Thresholds: baseline `0.100`, biased `0.100`

| Metric | Baseline (train=true) | Biased (train=observed) | Delta |
|--------|------------------------|--------------------------|-------|
| AUC | 0.6071 | 0.5954 | -0.0116 |
| AP | 0.0230 | 0.0213 | -0.0017 |
| ACCURACY | 0.9821 | 0.9817 | -0.0004 |
| PRECISION | 0.0000 | 0.0000 | +0.0000 |
| RECALL | 0.0000 | 0.0000 | +0.0000 |
| F1 | 0.0000 | 0.0000 | +0.0000 |

### Early CRC Detection - Age Band Subgroup Metrics (Baseline)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49 | 531 | 0 | 0.000 | 0.000 |
| 50-59 | 504 | 5 | 0.000 | 1.000 |
| 60-69 | 634 | 12 | 0.000 | 1.000 |
| 70-79 | 682 | 15 | 0.000 | 1.000 |
| 80+ | 2,629 | 53 | 0.000 | 1.000 |

### Early CRC Detection - Age Band Subgroup Metrics (Biased)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49 | 531 | 0 | 0.000 | 0.000 |
| 50-59 | 504 | 5 | 0.000 | 1.000 |
| 60-69 | 634 | 12 | 0.000 | 1.000 |
| 70-79 | 682 | 15 | 0.000 | 1.000 |
| 80+ | 2,629 | 53 | 0.000 | 1.000 |

### Early CRC Detection - Income Quintile Subgroup Metrics (Baseline)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| Q1 | 996 | 16 | 0.000 | 1.000 |
| Q2 | 996 | 12 | 0.000 | 1.000 |
| Q3 | 996 | 14 | 0.000 | 1.000 |
| Q4 | 996 | 26 | 0.000 | 1.000 |
| Q5 | 996 | 17 | 0.000 | 1.000 |

### Early CRC Detection - Income Quintile Subgroup Metrics (Biased)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| Q1 | 996 | 16 | 0.000 | 1.000 |
| Q2 | 996 | 12 | 0.000 | 1.000 |
| Q3 | 996 | 14 | 0.000 | 1.000 |
| Q4 | 996 | 26 | 0.000 | 1.000 |
| Q5 | 996 | 17 | 0.000 | 1.000 |

### Early CRC Detection - Age x Income Subgroup Metrics (Baseline)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49|Q1 | 98 | 0 | 0.000 | 0.000 |
| 40-49|Q2 | 104 | 0 | 0.000 | 0.000 |
| 40-49|Q3 | 98 | 0 | 0.000 | 0.000 |
| 40-49|Q4 | 106 | 0 | 0.000 | 0.000 |
| 40-49|Q5 | 125 | 0 | 0.000 | 0.000 |
| 50-59|Q1 | 103 | 1 | 0.000 | 1.000 |
| 50-59|Q2 | 86 | 0 | 0.000 | 0.000 |
| 50-59|Q3 | 117 | 3 | 0.000 | 1.000 |
| 50-59|Q4 | 96 | 0 | 0.000 | 0.000 |
| 50-59|Q5 | 102 | 1 | 0.000 | 1.000 |
| 60-69|Q1 | 112 | 0 | 0.000 | 0.000 |
| 60-69|Q2 | 131 | 2 | 0.000 | 1.000 |
| 60-69|Q3 | 127 | 1 | 0.000 | 1.000 |
| 60-69|Q4 | 130 | 3 | 0.000 | 1.000 |
| 60-69|Q5 | 134 | 6 | 0.000 | 1.000 |
| 70-79|Q1 | 113 | 3 | 0.000 | 1.000 |
| 70-79|Q2 | 128 | 3 | 0.000 | 1.000 |
| 70-79|Q3 | 144 | 1 | 0.000 | 1.000 |
| 70-79|Q4 | 145 | 6 | 0.000 | 1.000 |
| 70-79|Q5 | 152 | 2 | 0.000 | 1.000 |
| 80+|Q1 | 570 | 12 | 0.000 | 1.000 |
| 80+|Q2 | 547 | 7 | 0.000 | 1.000 |
| 80+|Q3 | 510 | 9 | 0.000 | 1.000 |
| 80+|Q4 | 519 | 17 | 0.000 | 1.000 |
| 80+|Q5 | 483 | 8 | 0.000 | 1.000 |

### Early CRC Detection - Age x Income Subgroup Metrics (Biased)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49|Q1 | 98 | 0 | 0.000 | 0.000 |
| 40-49|Q2 | 104 | 0 | 0.000 | 0.000 |
| 40-49|Q3 | 98 | 0 | 0.000 | 0.000 |
| 40-49|Q4 | 106 | 0 | 0.000 | 0.000 |
| 40-49|Q5 | 125 | 0 | 0.000 | 0.000 |
| 50-59|Q1 | 103 | 1 | 0.000 | 1.000 |
| 50-59|Q2 | 86 | 0 | 0.000 | 0.000 |
| 50-59|Q3 | 117 | 3 | 0.000 | 1.000 |
| 50-59|Q4 | 96 | 0 | 0.000 | 0.000 |
| 50-59|Q5 | 102 | 1 | 0.000 | 1.000 |
| 60-69|Q1 | 112 | 0 | 0.000 | 0.000 |
| 60-69|Q2 | 131 | 2 | 0.000 | 1.000 |
| 60-69|Q3 | 127 | 1 | 0.000 | 1.000 |
| 60-69|Q4 | 130 | 3 | 0.000 | 1.000 |
| 60-69|Q5 | 134 | 6 | 0.000 | 1.000 |
| 70-79|Q1 | 113 | 3 | 0.000 | 1.000 |
| 70-79|Q2 | 128 | 3 | 0.000 | 1.000 |
| 70-79|Q3 | 144 | 1 | 0.000 | 1.000 |
| 70-79|Q4 | 145 | 6 | 0.000 | 1.000 |
| 70-79|Q5 | 152 | 2 | 0.000 | 1.000 |
| 80+|Q1 | 570 | 12 | 0.000 | 1.000 |
| 80+|Q2 | 547 | 7 | 0.000 | 1.000 |
| 80+|Q3 | 510 | 9 | 0.000 | 1.000 |
| 80+|Q4 | 519 | 17 | 0.000 | 1.000 |
| 80+|Q5 | 483 | 8 | 0.000 | 1.000 |

## Model Inputs Policy

- Model training includes only clinical + demographic features:
  `age, male, bmi, smoker, diabetes, prediabetes, obesity, hypertension, hyperlipidemia, chf`
- Model training excludes direct bias-policy features:
  `income, poverty_ratio, assigned_plan, eligible_for_screening`
- Fairness evaluation is reported by:
  age band, income quintile, and age x income intersections.

## Interpretation Notes

- `has_crc_true` and `has_early_crc_true` are the canonical outcomes derived from stage codes.
- `observed_crc` and `observed_early_crc` represent what appears in biased data after masking.
- Any performance drop from baseline to biased model quantifies information loss induced by
  the age-income access policy.
