# Diabetes Case Study: Implementation Plan

## Phase 1: Data Generation

- [x] Generate baseline dataset (20k patients, Montana, seed 160)
- [x] Verify baseline diabetes prevalence (~6.31%)

## Phase 2: Module Override

- [x] Analyze `metabolic_syndrome_care.json` states for Hyperglycemia and Hypertriglyceridemia
- [x] Save original module to `artifacts/base_metabolic_syndrome_care.json`
- [x] Create modified module `artifacts/metabolic_syndrome_care_biased.json` with complex_transition (100%/0%)
- [x] Create `artifacts/overrides_documentation_bias.properties` with 30% random skip rate
- [ ] Generate biased dataset with override

## Phase 3: Scripts

- [x] `scripts/load_data.py` - copy/filter CSVs, handle headers
- [x] `scripts/summary_stats.py` - dataset statistics
- [ ] `scripts/models.py` - train models with feature sets (demographics, risk basic, risk+comorbidities, risk+metabolic)
- [ ] `scripts/analytics.py` - compare baseline vs biased performance

## Phase 4: Reports

- [ ] `output/diabetes_model_report.md`
- [ ] `output/diabetes_analytics_report.md`
- [ ] Update README.md with results

## Directory Structure

```
diabetes/
├── PLAN.md
├── README.md
├── pyproject.toml
├── artifacts/
│   ├── base_metabolic_syndrome_care.json
│   ├── metabolic_syndrome_care_biased.json
│   └── overrides_documentation_bias.properties
├── scripts/
│   ├── load_data.py
│   ├── summary_stats.py
│   ├── models.py
│   └── analytics.py
├── data/
│   ├── baseline/
│   └── biased/
└── output/
```
