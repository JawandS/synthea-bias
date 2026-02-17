# Model Results

Generated: 2026-02-17 17:45:20

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

## CRC Screening Recommendation (Risk of CRC)

Thresholds: baseline `0.558`, biased `0.589`
Operating points:
- Baseline: val positives `1395`, test positives `1408`, test prevalence `3.534%`
- Biased: val positives `1021`, test positives `1041`, test prevalence `3.534%`

| Metric | Baseline (train=true labels) | Biased (train=observed labels) | Delta |
|--------|------------------------|--------------------------|-------|
| AUC | 0.6125 | 0.6132 | +0.0008 |
| AP | 0.0465 | 0.0451 | -0.0015 |
| ACCURACY | 0.7064 | 0.7745 | +0.0681 |
| PRECISION | 0.0433 | 0.0451 | +0.0018 |
| RECALL | 0.3466 | 0.2670 | -0.0795 |
| F1 | 0.0770 | 0.0772 | +0.0002 |

### CRC Screening Recommendation (Risk of CRC) - Age Band Subgroup Metrics (Baseline)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49 | 531 | 0 | 0.000 | 0.000 |
| 50-59 | 504 | 9 | 0.000 | 1.000 |
| 60-69 | 634 | 26 | 0.231 | 0.769 |
| 70-79 | 682 | 29 | 0.379 | 0.621 |
| 80+ | 2,629 | 112 | 0.393 | 0.607 |

### CRC Screening Recommendation (Risk of CRC) - Age Band Subgroup Metrics (Biased)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49 | 531 | 0 | 0.000 | 0.000 |
| 50-59 | 504 | 9 | 0.000 | 1.000 |
| 60-69 | 634 | 26 | 0.115 | 0.885 |
| 70-79 | 682 | 29 | 0.345 | 0.655 |
| 80+ | 2,629 | 112 | 0.304 | 0.696 |

### CRC Screening Recommendation (Risk of CRC) - Income Quintile Subgroup Metrics (Baseline)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| Q1 | 996 | 30 | 0.433 | 0.567 |
| Q2 | 996 | 30 | 0.233 | 0.767 |
| Q3 | 996 | 36 | 0.361 | 0.639 |
| Q4 | 996 | 40 | 0.375 | 0.625 |
| Q5 | 996 | 40 | 0.325 | 0.675 |

### CRC Screening Recommendation (Risk of CRC) - Income Quintile Subgroup Metrics (Biased)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| Q1 | 996 | 30 | 0.333 | 0.667 |
| Q2 | 996 | 30 | 0.267 | 0.733 |
| Q3 | 996 | 36 | 0.222 | 0.778 |
| Q4 | 996 | 40 | 0.250 | 0.750 |
| Q5 | 996 | 40 | 0.275 | 0.725 |

### CRC Screening Recommendation (Risk of CRC) - Age x Income Subgroup Metrics (Baseline)

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
| 60-69|Q3 | 127 | 4 | 0.250 | 0.750 |
| 60-69|Q4 | 130 | 6 | 0.167 | 0.833 |
| 60-69|Q5 | 134 | 10 | 0.400 | 0.600 |
| 70-79|Q1 | 113 | 7 | 0.286 | 0.714 |
| 70-79|Q2 | 128 | 6 | 0.500 | 0.500 |
| 70-79|Q3 | 144 | 2 | 0.500 | 0.500 |
| 70-79|Q4 | 145 | 7 | 0.429 | 0.571 |
| 70-79|Q5 | 152 | 7 | 0.286 | 0.714 |
| 80+|Q1 | 570 | 20 | 0.550 | 0.450 |
| 80+|Q2 | 547 | 19 | 0.211 | 0.789 |
| 80+|Q3 | 510 | 26 | 0.423 | 0.577 |
| 80+|Q4 | 519 | 25 | 0.440 | 0.560 |
| 80+|Q5 | 483 | 22 | 0.318 | 0.682 |

### CRC Screening Recommendation (Risk of CRC) - Age x Income Subgroup Metrics (Biased)

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
| 60-69|Q3 | 127 | 4 | 0.250 | 0.750 |
| 60-69|Q4 | 130 | 6 | 0.000 | 1.000 |
| 60-69|Q5 | 134 | 10 | 0.200 | 0.800 |
| 70-79|Q1 | 113 | 7 | 0.286 | 0.714 |
| 70-79|Q2 | 128 | 6 | 0.500 | 0.500 |
| 70-79|Q3 | 144 | 2 | 0.500 | 0.500 |
| 70-79|Q4 | 145 | 7 | 0.143 | 0.857 |
| 70-79|Q5 | 152 | 7 | 0.429 | 0.571 |
| 80+|Q1 | 570 | 20 | 0.400 | 0.600 |
| 80+|Q2 | 547 | 19 | 0.263 | 0.737 |
| 80+|Q3 | 510 | 26 | 0.231 | 0.769 |
| 80+|Q4 | 519 | 25 | 0.360 | 0.640 |
| 80+|Q5 | 483 | 22 | 0.273 | 0.727 |

## Early-Stage Screening Recommendation (Catch Early CRC)

Thresholds: baseline `0.656`, biased `0.524`
Operating points:
- Baseline: val positives `367`, test positives `386`, test prevalence `1.707%`
- Biased: val positives `1538`, test positives `1596`, test prevalence `1.707%`

| Metric | Baseline (train=true labels) | Biased (train=observed labels) | Delta |
|--------|------------------------|--------------------------|-------|
| AUC | 0.5980 | 0.5935 | -0.0044 |
| AP | 0.0228 | 0.0210 | -0.0019 |
| ACCURACY | 0.9090 | 0.6765 | -0.2325 |
| PRECISION | 0.0233 | 0.0219 | -0.0014 |
| RECALL | 0.1059 | 0.4118 | +0.3059 |
| F1 | 0.0382 | 0.0416 | +0.0034 |

### Early-Stage Screening Recommendation (Catch Early CRC) - Age Band Subgroup Metrics (Baseline)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49 | 531 | 0 | 0.000 | 0.000 |
| 50-59 | 504 | 5 | 0.000 | 1.000 |
| 60-69 | 634 | 12 | 0.000 | 1.000 |
| 70-79 | 682 | 15 | 0.133 | 0.867 |
| 80+ | 2,629 | 53 | 0.132 | 0.868 |

### Early-Stage Screening Recommendation (Catch Early CRC) - Age Band Subgroup Metrics (Biased)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| 40-49 | 531 | 0 | 0.000 | 0.000 |
| 50-59 | 504 | 5 | 0.000 | 1.000 |
| 60-69 | 634 | 12 | 0.417 | 0.583 |
| 70-79 | 682 | 15 | 0.600 | 0.400 |
| 80+ | 2,629 | 53 | 0.396 | 0.604 |

### Early-Stage Screening Recommendation (Catch Early CRC) - Income Quintile Subgroup Metrics (Baseline)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| Q1 | 996 | 16 | 0.312 | 0.688 |
| Q2 | 996 | 12 | 0.000 | 1.000 |
| Q3 | 996 | 14 | 0.000 | 1.000 |
| Q4 | 996 | 26 | 0.077 | 0.923 |
| Q5 | 996 | 17 | 0.118 | 0.882 |

### Early-Stage Screening Recommendation (Catch Early CRC) - Income Quintile Subgroup Metrics (Biased)

| Group | N | Positives | Recall | FNR |
|-------|---:|----------:|-------:|----:|
| Q1 | 996 | 16 | 0.562 | 0.438 |
| Q2 | 996 | 12 | 0.417 | 0.583 |
| Q3 | 996 | 14 | 0.286 | 0.714 |
| Q4 | 996 | 26 | 0.269 | 0.731 |
| Q5 | 996 | 17 | 0.588 | 0.412 |

### Early-Stage Screening Recommendation (Catch Early CRC) - Age x Income Subgroup Metrics (Baseline)

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
| 70-79|Q1 | 113 | 3 | 0.333 | 0.667 |
| 70-79|Q2 | 128 | 3 | 0.000 | 1.000 |
| 70-79|Q3 | 144 | 1 | 0.000 | 1.000 |
| 70-79|Q4 | 145 | 6 | 0.167 | 0.833 |
| 70-79|Q5 | 152 | 2 | 0.000 | 1.000 |
| 80+|Q1 | 570 | 12 | 0.333 | 0.667 |
| 80+|Q2 | 547 | 7 | 0.000 | 1.000 |
| 80+|Q3 | 510 | 9 | 0.000 | 1.000 |
| 80+|Q4 | 519 | 17 | 0.059 | 0.941 |
| 80+|Q5 | 483 | 8 | 0.250 | 0.750 |

### Early-Stage Screening Recommendation (Catch Early CRC) - Age x Income Subgroup Metrics (Biased)

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
| 60-69|Q3 | 127 | 1 | 1.000 | 0.000 |
| 60-69|Q4 | 130 | 3 | 0.333 | 0.667 |
| 60-69|Q5 | 134 | 6 | 0.500 | 0.500 |
| 70-79|Q1 | 113 | 3 | 0.333 | 0.667 |
| 70-79|Q2 | 128 | 3 | 1.000 | 0.000 |
| 70-79|Q3 | 144 | 1 | 1.000 | 0.000 |
| 70-79|Q4 | 145 | 6 | 0.333 | 0.667 |
| 70-79|Q5 | 152 | 2 | 1.000 | 0.000 |
| 80+|Q1 | 570 | 12 | 0.667 | 0.333 |
| 80+|Q2 | 547 | 7 | 0.286 | 0.714 |
| 80+|Q3 | 510 | 9 | 0.222 | 0.778 |
| 80+|Q4 | 519 | 17 | 0.235 | 0.765 |
| 80+|Q5 | 483 | 8 | 0.625 | 0.375 |
