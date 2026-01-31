# Bias Effect Report

Generated: 2026-01-31 10:31:17

## Masking Parameters

| Parameter | Value |
|-----------|-------|
| Mask rate | 30% |
| Random seed | 42 |
| Rural patients with sleep apnea | 29 |
| Patients masked | 8 |
| Actual mask rate | 27.6% |

## Overall Effect

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Sleep apnea cases | 96 | 88 | -8 |
| Prevalence | 10.49% | 9.62% | -0.87% |

## Prevalence by Location

| Location | Before Cases | Before % | After Cases | After % | Change |
|----------|--------------|----------|-------------|---------|--------|
| Urban | 67 | 10.36% | 67 | 10.36% | +0.00% |
| Rural | 29 | 10.82% | 21 | 7.84% | -2.99% |

## Prevalence by Gender

| Gender | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
| Male | 72 | 14.60% | 68 | 13.79% | -0.81% |
| Female | 24 | 5.69% | 20 | 4.74% | -0.95% |

## Prevalence by Gender and Location

| Group | Before Cases | Before % | After Cases | After % | Change |
|-------|--------------|----------|-------------|---------|--------|
| Male Urban | 51 | 14.25% | 51 | 14.25% | +0.00% |
| Male Rural | 21 | 15.56% | 17 | 12.59% | -2.96% |
| Female Urban | 16 | 5.54% | 16 | 5.54% | +0.00% |
| Female Rural | 8 | 6.02% | 4 | 3.01% | -3.01% |

## Prevalence by Age Decade

| Decade | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
| 60-69 | 0 | 0.00% | 0 | 0.00% | +0.00% |
| 70-79 | 24 | 14.72% | 24 | 14.72% | +0.00% |
| 80-89 | 17 | 9.39% | 14 | 7.73% | -1.66% |
| 90+ | 55 | 9.65% | 50 | 8.77% | -0.88% |

## Key Observations

- **True prevalence**: 10.49% of all patients have sleep apnea
- **Observed prevalence**: 9.62% after rural underdiagnosis bias
- **Rural underdiagnosis**: 10.82% → 7.84% (-2.99%)
- **Urban (unaffected)**: 10.36% → 10.36%
