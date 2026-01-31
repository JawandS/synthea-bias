# Bias Effect Report

Generated: 2026-01-31 17:18:50

## Masking Parameters

| Parameter | Value |
|-----------|-------|
| Mask rate | 30% |
| Random seed | 42 |
| Rural patients with sleep apnea | 390 |
| Patients masked | 117 |
| Actual mask rate | 30.0% |

## Overall Effect

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Sleep apnea cases | 2,160 | 2,043 | -117 |
| Prevalence | 9.43% | 8.92% | -0.51% |

## Prevalence by Location

| Location | Before Cases | Before % | After Cases | After % | Change |
|----------|--------------|----------|-------------|---------|--------|
| Urban | 1,770 | 9.50% | 1,770 | 9.50% | +0.00% |
| Rural | 390 | 9.13% | 273 | 6.39% | -2.74% |

## Prevalence by Gender

| Gender | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
| Male | 1,554 | 12.51% | 1,480 | 11.91% | -0.60% |
| Female | 606 | 5.78% | 563 | 5.37% | -0.41% |

## Prevalence by Gender and Location

| Group | Before Cases | Before % | After Cases | After % | Change |
|-------|--------------|----------|-------------|---------|--------|
| Male Urban | 1,281 | 12.65% | 1,281 | 12.65% | +0.00% |
| Male Rural | 273 | 11.86% | 199 | 8.64% | -3.21% |
| Female Urban | 489 | 5.75% | 489 | 5.75% | +0.00% |
| Female Rural | 117 | 5.94% | 74 | 3.76% | -2.18% |

## Prevalence by Age Decade

| Decade | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
| 60-69 | 0 | 0.00% | 0 | 0.00% | +0.00% |
| 70-79 | 340 | 8.96% | 324 | 8.54% | -0.42% |
| 80-89 | 391 | 9.03% | 369 | 8.52% | -0.51% |
| 90+ | 1,429 | 9.67% | 1,350 | 9.14% | -0.53% |

## Key Observations

- **True prevalence**: 9.43% of all patients have sleep apnea
- **Observed prevalence**: 8.92% after rural underdiagnosis bias
- **Rural underdiagnosis**: 9.13% -> 6.39% (-2.74%)
- **Urban (unaffected)**: 9.50% -> 9.50%
