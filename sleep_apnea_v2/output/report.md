# Sleep Apnea Underdiagnosis Bias: A Case Study

Generated: 2026-01-31 17:21:34

---

## 1. Overview

### Study Goal

This case study demonstrates how **barriers to care resulting in underdiagnosis** can
introduce systematic bias into healthcare machine learning models. Specifically, we
examine how rural populations face reduced access to diagnostic services for sleep
apnea, leading to lower observed diagnosis rates despite similar true prevalence.

When ML models are trained on this biased "observed" data, they learn to underpredict
sleep apnea in rural populations, perpetuating and potentially amplifying existing
healthcare disparities.

### The Problem

Sleep apnea affects approximately 9-38% of the general adult population, with higher
rates among the elderly. Diagnosis requires specialized testing:

- **Polysomnography**: Overnight sleep study at a specialized clinic
- **Home sleep testing**: Ambulatory monitoring devices

Rural populations face significant barriers to diagnosis:
- Fewer sleep specialists and clinics
- Longer travel distances to diagnostic facilities
- Reduced access to follow-up care

This creates a **label masking bias**: rural patients with sleep apnea are less likely
to receive a formal diagnosis, making them appear "healthy" in electronic health records.

### Study Design

1. **Generate synthetic population** with realistic sleep apnea prevalence
2. **Apply rural underdiagnosis bias** by masking a portion of rural diagnoses
3. **Train ML models** on both true and biased labels
4. **Compare performance** to quantify the bias impact

---

## 2. Data Generation

### Synthea Patient Generator

Data is generated using [Synthea](https://github.com/synthetichealth/synthea), an
open-source synthetic patient population simulator. Synthea creates realistic (but
not real) patient data including:

- Demographics (age, gender, location)
- Medical conditions with onset/resolution dates
- Observations (BMI, smoking status)
- Encounters and procedures

### Sleep Apnea Module

The sleep apnea disease progression is modeled using a Synthea Generic Module
(`complex_sleep_apnea.json`). Key characteristics:

**Risk Score Calculation**:
The module calculates a cumulative risk score based on:
- BMI >= 35 (severe obesity): +2 points
- BMI >= 30 (obese): +1 point
- Current smoker: +1 point
- Alcohol use disorder: +1 point
- Congestive heart failure: +2 points

**Prevalence by Risk and Gender**:

| Risk Score | Male Prevalence | Female Prevalence |
|------------|-----------------|-------------------|
| >= 4 | 60% | 40% |
| >= 2 | 44% | 24% |
| >= 1 | 32% | 16% |
| 0 | 26% | 12% |

This reflects epidemiological data showing higher sleep apnea rates in males and
individuals with obesity, smoking history, and cardiovascular disease.

**Diagnostic Pathway**:
1. Initial encounter for symptoms (snoring, daytime sleepiness)
2. Referral to sleep specialist
3. Sleep study (polysomnography or home testing)
4. Diagnosis and treatment (CPAP or oral appliance)

### Generated Population

| Metric | Value |
|--------|-------|
| Total patients | 22,903 |
| Urban | 18,632 (81.4%) |
| Rural | 4,271 (18.6%) |
| Male | 12,427 |
| Female | 10,476 |
| Sleep apnea cases | 2,160 (9.4%) |

---

## 3. Bias Application

### Rural Underdiagnosis Simulation

To simulate real-world underdiagnosis bias, we apply **label masking** to rural
patients with sleep apnea. This models the scenario where patients have the condition
but never receive a formal diagnosis due to barriers to care.

**Masking Process**:
1. Identify all rural patients with true sleep apnea diagnosis
2. Randomly select a portion (mask rate) to have their diagnosis "hidden"
3. Create two label sets:
   - `has_sleep_apnea`: True underlying condition
   - `observed_sleep_apnea`: What appears in medical records (true & ~masked)

### Masking Statistics

| Metric | Value |
|--------|-------|
| Rural patients with sleep apnea | 390 |
| Patients masked (underdiagnosed) | 117 |
| Effective mask rate | 30.0% |

### Prevalence Impact

The masking creates a gap between true and observed prevalence, particularly
affecting rural populations:

| Location | Before Cases | Before % | After Cases | After % | Change |
|----------|--------------|----------|-------------|---------|--------|
| Urban | 1,770 | 9.50% | 1,770 | 9.50% | +0.00% |
| Rural | 390 | 9.13% | 273 | 6.39% | -2.74% |

**By Gender and Location**:

| Group | Before Cases | Before % | After Cases | After % | Change |
|-------|--------------|----------|-------------|---------|--------|
| Male Urban | 1,281 | 12.65% | 1,281 | 12.65% | +0.00% |
| Male Rural | 273 | 11.86% | 199 | 8.64% | -3.21% |
| Female Urban | 489 | 5.75% | 489 | 5.75% | +0.00% |
| Female Rural | 117 | 5.94% | 74 | 3.76% | -2.18% |


---

## 4. Model Training and Evaluation

### Approach

We train two Gradient Boosted Decision Tree (GBDT) models:

1. **Baseline Model**: Trained on true labels (`has_sleep_apnea`)
   - Represents the ideal scenario with complete diagnosis information

2. **Biased Model**: Trained on observed labels (`observed_sleep_apnea`)
   - Represents real-world scenario with underdiagnosis bias

Both models are evaluated against **true labels** to measure actual predictive
performance and fairness across subgroups.

### Features

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| male | Gender indicator (1 = male) |
| urban | Location indicator (1 = urban) |
| income | Household income (scaled) |
| bmi | Body mass index |
| smoker | Current smoker indicator |
| hypertension | Hypertension diagnosis |
| chf | Congestive heart failure diagnosis |
| alcohol_use | Alcohol use disorder diagnosis |

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

### Threshold Selection

With ~9% class prevalence, the default 0.5 classification threshold would rarely
predict positives. We use **adaptive thresholding** to find the optimal operating
point that maximizes F1 score on the validation set.

| Parameter | Value |
|-----------|-------|
| Method | f1 (maximize F1 on validation set) |
| Baseline threshold | 0.1405 |
| Biased threshold | 0.1233 |

### Overall Performance

| Metric | Baseline | Biased | Delta |
|--------|----------|--------|-------|
| AUC-ROC | 0.6457 | 0.6467 | +0.0010 |
| Avg Precision | 0.1445 | 0.1492 | +0.0047 |
| F1 Score | 0.2245 | 0.2219 | -0.0027 |
| Precision | 0.1559 | 0.1516 | -0.0043 |
| Recall | 0.4012 | 0.4136 | +0.0123 |
| Accuracy | 0.7386 | 0.7264 | -0.0122 |

### Test Set Composition

| Subgroup | Patients | Apnea Cases | Prevalence |
|----------|----------|-------------|------------|
| Urban | 2,801 | 262 | 9.35% |
| Rural | 635 | 62 | 9.76% |

### Subgroup AUC-ROC

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.6358 | 0.6410 | +0.0052 |
| Rural | 0.6927 | 0.6988 | +0.0061 |

### Subgroup Recall

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.4237 | 0.4733 | +0.0496 |
| Rural | 0.3065 | 0.1613 | -0.1452 |

### Subgroup F1 Score

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.2265 | 0.2267 | +0.0002 |
| Rural | 0.2135 | 0.1754 | -0.0380 |

### Key Model Findings

- **Rural Recall Drop**: -0.1452 (from 0.3065 to 0.1613)
  - The biased model misses more true sleep apnea cases in rural patients
- **Urban Recall Change**: +0.0496 (from 0.4237 to 0.4733)
  - Urban recall may improve as model shifts toward majority group
- **Recall Disparity**: Rural recall changes by -0.1452 vs Urban by +0.0496
  - Bias widens the gap in who gets correctly identified
- **Rural F1 Drop**: -0.0380 (from 0.2135 to 0.1754)
  - Overall rural prediction quality degrades
- **Threshold shift**: Biased model threshold 0.1233 vs baseline 0.1405

> **Note**: AUC measures ranking across all thresholds and may remain stable even when
> recall drops significantly. Recall at the operating threshold directly measures
> missed diagnoses and is the key fairness metric for this use case.


---

## 5. Key Findings

### Impact of Underdiagnosis Bias

1. **Rural Recall Collapse**: The biased model's recall for rural patients drops
   dramatically (30.6% → 16.1%, -14.5%), meaning it
   misses 47% more true sleep apnea cases in rural populations.

2. **Urban Recall Improvement**: Meanwhile, urban recall actually increases
   (42.4% → 47.3%, +5.0%), as the model shifts its
   predictions toward the majority group with complete labels.

3. **Disparity Amplification**: The recall gap between urban and rural widens from
   +11.7% to +31.2%, showing how
   training on biased data amplifies existing healthcare inequities.

### Why Recall Matters More Than AUC

- **AUC** measures ranking ability across all thresholds - it may remain stable
  even when the model systematically underpredicts for a subgroup.
- **Recall** measures how many true positives are caught at the operating threshold -
  this directly translates to missed diagnoses in clinical deployment.
- A model with good AUC but poor rural recall will still fail rural patients.

### Implications

- **Model Deployment Risk**: Deploying models trained on biased data perpetuates
  underdiagnosis in already underserved populations.

- **Data Quality Matters**: "Ground truth" labels from EHR data may reflect access
  patterns rather than true disease prevalence.

- **Fairness Monitoring**: Subgroup recall and F1 metrics are essential for detecting
  bias - overall AUC alone is insufficient.

### Mitigation Strategies

1. **Active case finding** in underserved populations to improve label quality
2. **Subgroup-stratified evaluation** with recall/F1 metrics, not just AUC
3. **Fairness-aware training** with constraints on subgroup performance parity
4. **Regular audits** comparing model predictions to external prevalence estimates

---

## 6. Conclusion

This case study demonstrates how structural barriers to healthcare access create
biased training data that, when used for ML model development, can perpetuate and
amplify existing health disparities. Rural underdiagnosis of sleep apnea is just
one example; similar patterns exist across many conditions and populations.

Responsible ML development in healthcare requires:
- Understanding the data generation process and its biases
- Evaluating models on true outcomes when possible
- Monitoring fairness across relevant subgroups
- Implementing mitigation strategies for known biases

---

## Appendix: Pipeline Execution

```bash
# 1. Generate synthetic population (Vermont, ages 60-100)
uv run python scripts/1_generate_data.py -p <N> -s 42

# 2. Apply rural underdiagnosis bias (30% mask rate)
uv run python scripts/2_gen_bias.py --mask-rate 0.3

# 3. Train and evaluate models
uv run python scripts/3_train_models.py

# 4. Generate this report
uv run python scripts/4_create_report.py
```

**This run**: 22,903 patients generated

---

*This report was generated as part of the Synthea Bias Case Study project.*
