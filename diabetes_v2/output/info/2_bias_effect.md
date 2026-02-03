# Bias Effect Report

Generated: 2026-02-03 17:27:09

## Masking Parameters

| Parameter | Value |
|-----------|-------|
| Mask rate | 30% |
| Random seed | 42 |

## Masking Summary

| Metric | Value |
|--------|-------|
| Patients with hypertriglyceridemia | 5,400 |
| Patients masked | 1,620 |
| Actual mask rate | 30.0% |

## Overall Effect

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Hypertriglyceridemia cases | 5,400 | 3,780 | -1,620 |
| Prevalence | 16.27% | 11.39% | -4.88% |

## Effect on Diabetes Association

Hypertriglyceridemia is a risk marker for diabetes. Masking it affects the apparent association.

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Diabetics with hypertriglyceridemia | 5,316 (98.4%) | 3,718 (68.8%) | -29.6% |

## Prevalence by Gender

| Gender | Before | After | Change |
|--------|--------|-------|--------|
| Male | 15.73% | 10.85% | -4.88% |
| Female | 16.90% | 12.03% | -4.88% |

## Prevalence by Age Decade

| Decade | Before | After | Change |
|--------|--------|-------|--------|
| 40-49 | 0.00% | 0.00% | +0.00% |
| 50-59 | 5.01% | 3.54% | -1.47% |
| 60-69 | 10.71% | 7.33% | -3.38% |
| 70-79 | 15.41% | 10.78% | -4.62% |
| 80-89 | 18.79% | 13.15% | -5.64% |
| 90+ | 19.26% | 13.51% | -5.75% |

## Key Observations

- **Documentation bias is random**: This bias affects all patient groups equally (no demographic targeting)
- **Hypertriglyceridemia prevalence**: 16.27% true -> 11.39% observed
- **Impact on modeling**: Models trained on observed data will learn a weaker association between hypertriglyceridemia and diabetes
