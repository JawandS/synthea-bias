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
