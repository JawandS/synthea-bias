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
