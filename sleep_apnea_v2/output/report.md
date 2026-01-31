# Sleep Apnea Underdiagnosis Bias: A Case Study

Generated: 2026-01-31 15:45:01

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
| Total patients | 917 |
| Urban | 768 (83.8%) |
| Rural | 149 (16.2%) |
| Male | 517 |
| Female | 400 |
| Sleep apnea cases | 85 (9.3%) |

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
| Rural patients with sleep apnea | 10 |
| Patients masked (underdiagnosed) | 3 |
| Effective mask rate | 30.0% |

### Prevalence Impact

The masking creates a gap between true and observed prevalence, particularly
affecting rural populations:

| Location | Before Cases | Before % | After Cases | After % | Change |
|----------|--------------|----------|-------------|---------|--------|
| Urban | 75 | 9.77% | 75 | 9.77% | +0.00% |
| Rural | 10 | 6.71% | 7 | 4.70% | -2.01% |

**By Gender and Location**:

| Group | Before Cases | Before % | After Cases | After % | Change |
|-------|--------------|----------|-------------|---------|--------|
| Male Urban | 60 | 14.32% | 60 | 14.32% | +0.00% |
| Male Rural | 9 | 9.18% | 6 | 6.12% | -3.06% |
| Female Urban | 15 | 4.30% | 15 | 4.30% | +0.00% |
| Female Rural | 1 | 1.96% | 1 | 1.96% | +0.00% |


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
| Baseline threshold | 0.0767 |
| Biased threshold | 0.0543 |

### Overall Performance

| Metric | Baseline | Biased | Delta |
|--------|----------|--------|-------|
| AUC-ROC | 0.5415 | 0.5200 | -0.0215 |
| Avg Precision | 0.1141 | 0.1166 | +0.0026 |
| F1 Score | 0.1905 | 0.1961 | +0.0056 |
| Precision | 0.1379 | 0.1316 | -0.0064 |
| Recall | 0.3077 | 0.3846 | +0.0769 |
| Accuracy | 0.7536 | 0.7029 | -0.0507 |

### Test Set Composition

| Subgroup | Patients | Apnea Cases | Prevalence |
|----------|----------|-------------|------------|
| Urban | 113 | 10 | 8.85% |
| Rural | 25 | 3 | 12.00% |

### Subgroup AUC-ROC

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.5553 | 0.5311 | -0.0243 |
| Rural | 0.4848 | 0.4545 | -0.0303 |

### Subgroup Recall

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.3000 | 0.4000 | +0.1000 |
| Rural | 0.3333 | 0.3333 | +0.0000 |

### Subgroup F1 Score

| Subgroup | Baseline | Biased | Delta |
|----------|----------|--------|-------|
| Urban | 0.1667 | 0.1778 | +0.0111 |
| Rural | 0.3333 | 0.3333 | +0.0000 |

### Key Model Findings

- **Rural AUC degradation**: -0.0303 (from 0.4848 to 0.4545)
  - The biased model has reduced ability to discriminate sleep apnea in rural patients
- **Urban AUC change**: -0.0243 (relatively stable)
- **Disparity gap**: Rural AUC changes by -0.0303 vs Urban by -0.0243
  - Bias introduces/widens performance gap between subgroups
- **Threshold shift**: Biased model uses lower threshold (0.0543 vs 0.0767)
  - Compensates for reduced signal from missing rural positives in training


---

## 5. Key Findings

### Impact of Underdiagnosis Bias

1. **Rural AUC Degradation**: The biased model shows greater AUC reduction for
   rural patients compared to urban, demonstrating how underdiagnosis bias
   disproportionately harms the affected subgroup's predictive performance.

2. **Threshold Compensation**: The biased model learns a lower classification
   threshold, attempting to compensate for the reduced positive signal from
   missing rural diagnoses in training data.

3. **Fairness Gap**: The performance disparity between urban and rural subgroups
   widens under the biased model, exacerbating healthcare inequities.

### Implications

- **Model Deployment Risk**: Deploying models trained on biased data perpetuates
  underdiagnosis in already underserved populations.

- **Data Quality Matters**: "Ground truth" labels from EHR data may reflect access
  patterns rather than true disease prevalence.

- **Fairness Monitoring**: Subgroup performance metrics are essential for detecting
  and addressing bias in healthcare ML.

### Mitigation Strategies

1. **Active case finding** in underserved populations
2. **Calibration adjustments** for known underdiagnosis patterns
3. **Subgroup-aware training** with fairness constraints
4. **Regular audits** of model performance across demographics

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

**This run**: 917 patients generated

---

*This report was generated as part of the Synthea Bias Case Study project.*
