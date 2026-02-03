# Diabetes Documentation Bias: A Case Study

Generated: 2026-02-03 17:27:48

---

## 1. Overview

### Study Goal

This case study demonstrates how **documentation bias** in electronic health records can
affect machine learning models that predict diabetes. Specifically, we examine how
incomplete documentation of hypertriglyceridemia (elevated triglycerides) reduces the
apparent association between this metabolic condition and diabetes.

When ML models are trained on data with incomplete documentation, they learn weaker
associations for the under-documented features, potentially relying more heavily on
other features like A1c, BMI, or demographics.

### The Problem

Diabetes affects approximately 10-12% of the adult population. Hypertriglyceridemia
(elevated triglycerides, SNOMED 302870006) is a key metabolic risk marker:

- Part of **metabolic syndrome** cluster
- Signals underlying insulin resistance
- Often co-occurs with or precedes diabetes
- Distinct from diabetes itself (unlike hyperglycemia)

Hypertriglyceridemia may be under-documented in EHR data due to:
- Labs show elevated triglycerides but diagnosis code not entered
- Time pressure during clinical encounters
- Focus on primary diagnosis rather than secondary findings
- Variability in documentation practices across providers

This creates **documentation bias**: the condition exists but isn't recorded,
weakening its apparent predictive signal for diabetes.

### Study Design

1. **Generate synthetic population** with realistic diabetes and hypertriglyceridemia prevalence
2. **Apply documentation bias** by randomly masking a portion of hypertriglyceridemia diagnoses
3. **Train ML models** using both true and observed (biased) features
4. **Compare performance** to quantify the bias impact on feature importance and predictions

---

## 2. Data Generation

### Synthea Patient Generator

Data is generated using [Synthea](https://github.com/synthetichealth/synthea), an
open-source synthetic patient population simulator. Synthea creates realistic (but
not real) patient data including:

- Demographics (age, gender, income)
- Medical conditions with onset/resolution dates
- Observations (A1c, BMI, smoking status)
- Encounters and procedures

### Metabolic Syndrome Module

The metabolic syndrome disease progression is modeled using a Synthea Generic Module.
Key characteristics:

**Risk Factors**:
- BMI >= 30 (obesity)
- Age > 40 years
- Sedentary lifestyle
- Family history

**Metabolic Conditions**:
- Hypertriglyceridemia (elevated triglycerides)
- Prediabetes
- Type 2 diabetes

### Generated Population

| Metric | Value |
|--------|-------|
| Total patients | 33,200 |
| Male | 18,085 (54.5%) |
| Female | 15,115 (45.5%) |
| Diabetes cases | 5,405 (16.3%) |
| Hypertriglyceridemia cases | 5,400 (16.3%) |

---

## 3. Bias Application

### Documentation Bias Simulation

To simulate real-world documentation bias, we apply **random masking** to
hypertriglyceridemia diagnoses. This models the scenario where patients have
the condition but it's not recorded due to documentation gaps.

**Masking Process**:
1. Identify all patients with true hypertriglyceridemia diagnosis
2. Randomly select a portion (mask rate) to have their diagnosis "hidden"
3. Create two feature versions:
   - `has_hypertriglyceridemia`: True underlying condition status
   - `observed_hypertriglyceridemia`: What appears in medical records

**Key Characteristic**:
Unlike demographic-based bias (e.g., rural underdiagnosis), documentation bias
affects patients randomly across all groups. This isolates the effect of
incomplete feature information from confounding demographic factors.

### Masking Statistics

| Metric | Value |
|--------|-------|
| Patients with hypertriglyceridemia | 5,400 |
| Patients masked (under-documented) | 1,620 |
| Effective mask rate | 30.0% |

### Impact on Observed Prevalence

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Hypertriglyceridemia cases | 5,400 | 3,780 | -1,620 |
| Prevalence | 16.27% | 11.39% | -4.88% |

**Effect on Diabetes Association**:

| Metric | Before (True) | After (Observed) | Change |
|--------|---------------|------------------|--------|
| Diabetics with hypertriglyceridemia | 5,316 (98.4%) | 3,718 (68.8%) | -29.6% |


---

## 4. Model Training and Evaluation

### Approach

We train two Gradient Boosted Decision Tree (GBDT) models:

1. **Baseline Model**: Uses true feature (`has_hypertriglyceridemia`)
   - Represents the ideal scenario with complete documentation

2. **Biased Model**: Uses observed feature (`observed_hypertriglyceridemia`)
   - Represents real-world scenario with documentation bias

Both models predict the **same target** (`has_diabetes`) and are evaluated on
identical test data to measure the impact of feature bias.

### Features

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| male | Gender indicator (1 = male) |
| income | Household income (scaled) |
| a1c | Hemoglobin A1c level |
| bmi | Body mass index |
| smoker | Current smoker indicator |
| prediabetes | Prediabetes diagnosis |
| obesity | Obesity diagnosis |
| hypertension | Hypertension diagnosis |
| hyperlipidemia | Hyperlipidemia diagnosis |
| hypertriglyceridemia | Hypertriglyceridemia (true or observed) |

### Model Specification

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosted Decision Tree |
| n_estimators | 200 |
| max_depth | 5 |
| learning_rate | 0.05 |
| min_samples_split | 20 |
| min_samples_leaf | 10 |
| subsample | 0.8 |

### Overall Performance

| Metric | Baseline | Biased | Delta |
|--------|----------|--------|-------|
| AUC-ROC | 0.9997 | 0.9976 | -0.0021 |
| Avg Precision | 0.9939 | 0.9900 | -0.0038 |
| F1 Score | 0.9963 | 0.9598 | -0.0365 |
| Precision | 0.9939 | 0.9934 | -0.0005 |
| Recall | 0.9988 | 0.9285 | -0.0703 |
| Accuracy | 0.9988 | 0.9873 | -0.0114 |

### Feature Importance

| Feature | Baseline | Biased | Delta |
|---------|----------|--------|-------|
| age | 0.0004 | 0.0018 | +0.0014 |
| male | 0.0000 | 0.0003 | +0.0003 |
| income | 0.0011 | 0.0037 | +0.0025 |
| a1c | 0.0268 | 0.2252 | +0.1984 |
| bmi | 0.0079 | 0.0128 | +0.0049 |
| smoker | 0.0000 | 0.0000 | +0.0000 |
| prediabetes | 0.0001 | 0.0289 | +0.0288 |
| obesity | 0.0000 | 0.0006 | +0.0006 |
| hypertension | 0.0001 | 0.0237 | +0.0236 |
| hyperlipidemia | 0.0000 | 0.0004 | +0.0004 |
| hypertriglyceridemia | 0.9636 | 0.7026 | -0.2610 |

### Key Model Findings

- **Overall AUC change**: -0.0021 (from 0.9997 to 0.9976)
- **Overall F1 change**: -0.0365 (from 0.9963 to 0.9598)
- **Hypertriglyceridemia importance**: 0.9636 -> 0.7026 (-0.2610)

> **Documentation bias effect**: When hypertriglyceridemia is under-documented, the model learns
> a weaker association between this condition and diabetes. The model may compensate by relying
> more heavily on other features like A1c, BMI, or other comorbidities.


---

## 5. Key Findings

### Impact of Documentation Bias

1. **Reduced Feature Importance**: The biased model assigns lower importance to
   the hypertriglyceridemia feature, as masked values dilute the signal.

2. **Compensatory Reliance**: Other features (A1c, BMI, prediabetes) may see
   increased importance as the model compensates for the weakened signal.

3. **Clinical Validity**: Unlike hyperglycemia (which is definitionally diabetes),
   hypertriglyceridemia is a genuine risk marker - making this a clinically
   realistic documentation bias scenario.

### Why This Matters

- **Different from Label Bias**: Documentation bias affects *features* while the
  target remains correct. The model learns from incomplete information rather
  than incorrect labels.

- **Feature Importance Shifts**: Models may over-rely on well-documented features
  and under-utilize poorly documented but clinically important ones.

- **Generalization Risk**: A model trained on data with documentation bias may
  perform differently in settings with better or worse documentation practices.

### Implications for ML in Healthcare

- **Data Quality**: Feature completeness matters as much as label accuracy
- **Institutional Variation**: Documentation practices vary - models may not generalize
- **Monitoring**: Track documentation rates for key predictive features
- **Robustness**: Evaluate models under varying documentation completeness

### Mitigation Strategies

1. **Documentation improvement** initiatives at the clinical level
2. **Feature imputation** based on related lab values when diagnoses are missing
3. **Sensitivity analysis** to understand model behavior under documentation gaps
4. **Multi-source validation** using data with different documentation practices

---

## 6. Conclusion

This case study demonstrates how incomplete documentation of clinical findings
creates biased training data that affects ML model behavior. By focusing on
hypertriglyceridemia (a genuine risk marker rather than a defining characteristic),
we isolate the effect of documentation bias in a clinically realistic scenario.

Key takeaways:

- Documentation gaps are measurement error in feature values
- Feature importance shifts away from under-documented conditions
- Models may not generalize across documentation practices
- Monitoring feature completeness is essential for reliable ML deployment

---

## Appendix: Pipeline Execution

```bash
# 1. Generate synthetic population (Montana, ages 40-100)
uv run python scripts/1_generate_data.py -p 20000 -s 160

# 2. Apply documentation bias (30% mask rate)
uv run python scripts/2_gen_bias.py

# 3. Train and evaluate models
uv run python scripts/3_train_models.py

# 4. Generate this report
uv run python scripts/4_create_report.py
```

**This run**: 33,200 patients generated

---

*This report was generated as part of the Synthea Bias Case Study project.*
