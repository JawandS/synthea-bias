# Bias Effect Report

Generated: 2026-01-31 15:34:14

## Masking Parameters

| Parameter | Value |
|-----------|-------|
| Mask rate | 30% |
| Random seed | 42 |
| Rural patients with sleep apnea | 10 |
| Patients masked | 3 |
| Actual mask rate | 30.0% |

## Overall Effect

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Sleep apnea cases | 85 | 82 | -3 |
| Prevalence | 9.27% | 8.94% | -0.33% |

## Prevalence by Location

| Location | Before Cases | Before % | After Cases | After % | Change |
|----------|--------------|----------|-------------|---------|--------|
| Urban | 75 | 9.77% | 75 | 9.77% | +0.00% |
| Rural | 10 | 6.71% | 7 | 4.70% | -2.01% |

## Prevalence by Gender

| Gender | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
| Male | 69 | 13.35% | 66 | 12.77% | -0.58% |
| Female | 16 | 4.00% | 16 | 4.00% | +0.00% |

## Prevalence by Gender and Location

| Group | Before Cases | Before % | After Cases | After % | Change |
|-------|--------------|----------|-------------|---------|--------|
| Male Urban | 60 | 14.32% | 60 | 14.32% | +0.00% |
| Male Rural | 9 | 9.18% | 6 | 6.12% | -3.06% |
| Female Urban | 15 | 4.30% | 15 | 4.30% | +0.00% |
| Female Rural | 1 | 1.96% | 1 | 1.96% | +0.00% |

## Prevalence by Age Decade

| Decade | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
| 60-69 | 0 | 0.00% | 0 | 0.00% | +0.00% |
| 70-79 | 11 | 7.43% | 11 | 7.43% | +0.00% |
| 80-89 | 15 | 8.43% | 14 | 7.87% | -0.56% |
| 90+ | 59 | 10.00% | 57 | 9.66% | -0.34% |

## Key Observations

- **True prevalence**: 9.27% of all patients have sleep apnea
- **Observed prevalence**: 8.94% after rural underdiagnosis bias
- **Rural underdiagnosis**: 6.71% → 4.70% (-2.01%)
- **Urban (unaffected)**: 9.77% → 9.77%
