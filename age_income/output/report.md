# Intersectional Case Study: Age + Income Access Bias in Colorectal Cancer Screening

Generated: 2026-02-17 18:04:17

## Scenario

This case study models an intersectional access pattern where age and income jointly determine
screening access generosity. Generous plans screen earlier; restrictive plans screen later.
The underlying disease burden (true CRC) remains unchanged, while observed diagnosis and
observed early-stage detection are masked by plan-specific access barriers.

Policy goal: recommend individuals at high risk of CRC for screening, while preserving early-stage
case capture. The equity-oriented model excludes wealth variables, but the historically biased model
learns from observed labels that encode unequal access.

## Policy Rules

| Plan | Income Min | Income Max | Age Range | Screen Start Age | CRC Mask Rate | Early Mask Rate |
|------|------------:|-----------:|----------:|-----------------:|--------------:|----------------:|
| high_access | $90,000 | $1,000,000,000 | 40-49 | 45 | 8% | 18% |
| high_access | $90,000 | $1,000,000,000 | 50-64 | 45 | 5% | 12% |
| high_access | $90,000 | $1,000,000,000 | 65-74 | 45 | 4% | 10% |
| high_access | $90,000 | $1,000,000,000 | 75-100 | 45 | 8% | 14% |
| standard_access | $40,000 | $90,000 | 40-49 | 50 | 18% | 32% |
| standard_access | $40,000 | $90,000 | 50-64 | 50 | 12% | 24% |
| standard_access | $40,000 | $90,000 | 65-74 | 50 | 10% | 20% |
| standard_access | $40,000 | $90,000 | 75-100 | 50 | 14% | 24% |
| restricted_access | $0 | $40,000 | 40-49 | 55 | 36% | 55% |
| restricted_access | $0 | $40,000 | 50-64 | 55 | 28% | 45% |
| restricted_access | $0 | $40,000 | 65-74 | 55 | 20% | 34% |
| restricted_access | $0 | $40,000 | 75-100 | 55 | 26% | 40% |

## Baseline Data Summary

# Baseline Summary Stats

Generated: 2026-02-17 17:58:52

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

Generated: 2026-02-17 17:58:55

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

## Modeling Results

# Model Results

Generated: 2026-02-17 18:04:17

## Configuration

- Policy objective: recommend individuals for CRC screening based on predicted risk.
- Secondary objective: maximize capture of early-stage CRC cases.
- Samples: 33,200
- Features: age, male, bmi, smoker, diabetes, prediabetes, obesity, hypertension, hyperlipidemia, chf
- Excluded from model features for equity: income, assigned_plan, eligible_for_screening
- Historical-bias simulation: observed labels reflect access barriers that disproportionately affect lower-income groups.
- Train size: 23,240
- Validation size: 4,980
- Test size: 4,980
- Seed: 42

## Aggregate Performance Difference

| Task | Metric | Baseline | Biased | Delta (Biased - Baseline) |
|------|--------|---------:|-------:|---------------------------:|
| CRC Screening Recommendation (Risk of CRC) | AUC | 0.6125 | 0.6132 | +0.0008 |
| CRC Screening Recommendation (Risk of CRC) | AP | 0.0465 | 0.0451 | -0.0015 |
| CRC Screening Recommendation (Risk of CRC) | ACCURACY | 0.7064 | 0.7745 | +0.0681 |
| CRC Screening Recommendation (Risk of CRC) | PRECISION | 0.0433 | 0.0451 | +0.0018 |
| CRC Screening Recommendation (Risk of CRC) | RECALL | 0.3466 | 0.2670 | -0.0795 |
| CRC Screening Recommendation (Risk of CRC) | F1 | 0.0770 | 0.0772 | +0.0002 |
| Early-Stage Screening Recommendation (Catch Early CRC) | AUC | 0.5980 | 0.5935 | -0.0044 |
| Early-Stage Screening Recommendation (Catch Early CRC) | AP | 0.0228 | 0.0210 | -0.0019 |
| Early-Stage Screening Recommendation (Catch Early CRC) | ACCURACY | 0.9090 | 0.6765 | -0.2325 |
| Early-Stage Screening Recommendation (Catch Early CRC) | PRECISION | 0.0233 | 0.0219 | -0.0014 |
| Early-Stage Screening Recommendation (Catch Early CRC) | RECALL | 0.1059 | 0.4118 | +0.3059 |
| Early-Stage Screening Recommendation (Catch Early CRC) | F1 | 0.0382 | 0.0416 | +0.0034 |

## Income Subgroup Performance Difference (CRC Screening Recommendation)

| Task | Group | N | Positives | Baseline Recall | Biased Recall | Delta Recall | Baseline FNR | Biased FNR | Delta FNR |
|------|-------|--:|----------:|----------------:|--------------:|-------------:|-------------:|-----------:|----------:|
| CRC Screening Recommendation (Risk of CRC) | high_income | 1,011 | 40 | 0.325 | 0.275 | -0.050 | 0.675 | 0.725 | +0.050 |
| CRC Screening Recommendation (Risk of CRC) | low_income | 2,152 | 63 | 0.333 | 0.286 | -0.048 | 0.667 | 0.714 | +0.048 |
| CRC Screening Recommendation (Risk of CRC) | middle_income | 1,817 | 73 | 0.370 | 0.247 | -0.123 | 0.630 | 0.753 | +0.123 |

## Age Subgroup Performance Difference (CRC Screening Recommendation)

| Task | Group | N | Positives | Baseline Recall | Biased Recall | Delta Recall | Baseline FNR | Biased FNR | Delta FNR |
|------|-------|--:|----------:|----------------:|--------------:|-------------:|-------------:|-----------:|----------:|
| CRC Screening Recommendation (Risk of CRC) | 40-49 | 531 | 0 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| CRC Screening Recommendation (Risk of CRC) | 50-59 | 504 | 9 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| CRC Screening Recommendation (Risk of CRC) | 60-69 | 634 | 26 | 0.231 | 0.115 | -0.115 | 0.769 | 0.885 | +0.115 |
| CRC Screening Recommendation (Risk of CRC) | 70-79 | 682 | 29 | 0.379 | 0.345 | -0.034 | 0.621 | 0.655 | +0.034 |
| CRC Screening Recommendation (Risk of CRC) | 80+ | 2,629 | 112 | 0.393 | 0.304 | -0.089 | 0.607 | 0.696 | +0.089 |

## Model Inputs Policy

- Model training includes only clinical + demographic features:
  `age, male, bmi, smoker, diabetes, prediabetes, obesity, hypertension, hyperlipidemia, chf`
- Model training excludes direct bias-policy features for equity:
  `income, assigned_plan, eligible_for_screening`
- Fairness evaluation is reported by:
  age band and policy-aligned income band.

## Interpretation Notes

- `has_crc_true` and `has_early_crc_true` are the canonical outcomes derived from stage codes.
- `observed_crc` and `observed_early_crc` represent what appears in biased data after masking.
- Any performance drop from baseline to biased model quantifies information loss induced by
  the age-income access policy.
- Income is intentionally excluded from features, so income-related disparities in biased-model
  performance arise from historical label distortion rather than explicit wealth inputs.
