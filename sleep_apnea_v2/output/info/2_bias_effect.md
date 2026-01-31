# Bias Effect Report

Generated: 2026-01-31 10:39:13

## Masking Parameters

| Parameter | Value |
|-----------|-------|
| Mask rate | 30% |
| Random seed | 42 |
| Rural patients with sleep apnea | 15 |
| Patients masked | 4 |
| Actual mask rate | 26.7% |

## Overall Effect

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Sleep apnea cases | 145 | 141 | -4 |
| Prevalence | 8.22% | 8.00% | -0.23% |

## Prevalence by Location

| Location | Before Cases | Before % | After Cases | After % | Change |
|----------|--------------|----------|-------------|---------|--------|
| Urban | 130 | 8.39% | 130 | 8.39% | +0.00% |
| Rural | 15 | 7.04% | 11 | 5.16% | -1.88% |

## Prevalence by Gender

| Gender | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
| Male | 107 | 11.33% | 104 | 11.02% | -0.32% |
| Female | 38 | 4.64% | 37 | 4.52% | -0.12% |

## Prevalence by Gender and Location

| Group | Before Cases | Before % | After Cases | After % | Change |
|-------|--------------|----------|-------------|---------|--------|
| Male Urban | 99 | 11.73% | 99 | 11.73% | +0.00% |
| Male Rural | 8 | 8.00% | 5 | 5.00% | -3.00% |
| Female Urban | 31 | 4.39% | 31 | 4.39% | +0.00% |
| Female Rural | 7 | 6.19% | 6 | 5.31% | -0.88% |

## Prevalence by Age Decade

| Decade | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
| 60-69 | 0 | 0.00% | 0 | 0.00% | +0.00% |
| 70-79 | 29 | 10.36% | 28 | 10.00% | -0.36% |
| 80-89 | 30 | 8.11% | 29 | 7.84% | -0.27% |
| 90+ | 86 | 7.73% | 84 | 7.55% | -0.18% |

## Key Observations

- **True prevalence**: 8.22% of all patients have sleep apnea
- **Observed prevalence**: 8.00% after rural underdiagnosis bias
- **Rural underdiagnosis**: 7.04% → 5.16% (-1.88%)
- **Urban (unaffected)**: 8.39% → 8.39%
