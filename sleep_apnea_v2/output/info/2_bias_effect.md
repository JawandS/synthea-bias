# Bias Effect Report

Generated: 2026-01-31 16:51:46

## Masking Parameters

| Parameter | Value |
|-----------|-------|
| Mask rate | 30% |
| Random seed | 42 |
| Rural patients with sleep apnea | 62 |
| Patients masked | 18 |
| Actual mask rate | 29.0% |

## Overall Effect

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Sleep apnea cases | 350 | 332 | -18 |
| Prevalence | 9.44% | 8.95% | -0.49% |

## Prevalence by Location

| Location | Before Cases | Before % | After Cases | After % | Change |
|----------|--------------|----------|-------------|---------|--------|
| Urban | 288 | 9.59% | 288 | 9.59% | +0.00% |
| Rural | 62 | 8.79% | 44 | 6.24% | -2.55% |

## Prevalence by Gender

| Gender | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
| Male | 255 | 13.12% | 242 | 12.45% | -0.67% |
| Female | 95 | 5.39% | 90 | 5.10% | -0.28% |

## Prevalence by Gender and Location

| Group | Before Cases | Before % | After Cases | After % | Change |
|-------|--------------|----------|-------------|---------|--------|
| Male Urban | 209 | 13.25% | 209 | 13.25% | +0.00% |
| Male Rural | 46 | 12.53% | 33 | 8.99% | -3.54% |
| Female Urban | 79 | 5.54% | 79 | 5.54% | +0.00% |
| Female Rural | 16 | 4.73% | 11 | 3.25% | -1.48% |

## Prevalence by Age Decade

| Decade | Before Cases | Before % | After Cases | After % | Change |
|--------|--------------|----------|-------------|---------|--------|
| 60-69 | 0 | 0.00% | 0 | 0.00% | +0.00% |
| 70-79 | 61 | 8.97% | 60 | 8.82% | -0.15% |
| 80-89 | 64 | 8.74% | 61 | 8.33% | -0.41% |
| 90+ | 225 | 9.80% | 211 | 9.19% | -0.61% |

## Key Observations

- **True prevalence**: 9.44% of all patients have sleep apnea
- **Observed prevalence**: 8.95% after rural underdiagnosis bias
- **Rural underdiagnosis**: 8.79% -> 6.24% (-2.55%)
- **Urban (unaffected)**: 9.59% -> 9.59%
