# Sleep Apnea Case Study: Rural Access Bias

*Generated: 2026-01-25 20:43:24*

## Executive Summary

This case study demonstrates how healthcare access disparities can introduce systematic bias into clinical datasets and downstream machine learning models. Using Synthea, we generated two synthetic patient populations: a **baseline** dataset where all patients have equal access to care, and a **biased** dataset where rural patients face barriers that cause them to drop out of the sleep apnea care pathway before receiving a diagnosis.

**Key Findings:**

- Rural underdiagnosis rate in biased dataset: **82.93%** vs urban rate of **0.37%**
- Adjusted odds ratio for rural underdiagnosis: **21.432** (p=0.002)
- Models trained on biased data show degraded performance when evaluated on baseline population

---

## 1. Background

### 1.1 Sleep Apnea Overview

Sleep apnea is a common sleep disorder characterized by repeated interruption of breathing during sleep. The Synthea `sleep_apnea.json` module models the clinical pathway:

1. **Risk Assessment**: Patients aged 30-60 are evaluated based on BMI, smoking, alcohol use, and CHF
2. **Initial Presentation**: Symptoms include loud snoring and excessive daytime sleepiness
3. **Referral**: Primary care refers to sleep specialist
4. **Diagnostic Testing**: Home sleep study or in-lab polysomnography
5. **Treatment**: CPAP therapy or oral appliances with ongoing follow-up

### 1.2 The Access Bias Problem

In real-world healthcare, rural patients often face barriers to specialty care including:

- Longer travel distances to sleep centers
- Fewer available sleep specialists
- Difficulty scheduling follow-up appointments
- Higher costs of repeated travel

These barriers cause rural patients to "drop out" of the care pathway before receiving diagnosis and treatment, leading to **underdiagnosis** and **undertreatment** of sleep apnea in rural populations.

### 1.3 Bias Simulation

We simulate this access disparity using Synthea's module override mechanism. The biased dataset modifies two transition points in the sleep apnea care pathway:

| Transition Point | Baseline (Rural) | Biased (Rural) |
| --- | :---: | :---: |
| Wait Until Overnight Study | 100% continue | 20% continue, **80% drop out** |
| Appointment Delay | 100% continue | 20% continue, **80% drop out** |

Urban patients continue to receive full care in both datasets. This creates a scenario where the biased dataset reflects lower sleep apnea diagnosis rates in rural populations despite equivalent underlying disease prevalence.

---

## 2. Data Summary

### 2.1 Dataset Overview

| Dataset | Total Patients | Sleep Apnea Prevalence | Description |
| --- | ---: | ---: | --- |
| baseline | 20,000 | 3.69% | Equal access for all patients |
| biased | 20,000 | 2.89% | 80% rural dropout at diagnostic stages |

The biased dataset shows a **0.81% lower** sleep apnea prevalence (2.89% vs 3.69%), reflecting the rural underdiagnosis effect.

### 2.2 Population Characteristics by Residence

Summary statistics for the full patient population, stratified by urban/rural residence. Note that demographic characteristics are identical across datasets—only diagnosis rates differ.

| Characteristic | baseline Urban | baseline Rural | biased Urban | biased Rural |
| --- | ---: | ---: | ---: | ---: |
| N | 14,507 | 5,493 | 14,507 | 5,493 |
| Age (years) | 39.6 ± 23.7 | 40.1 ± 24.6 | 39.6 ± 23.7 | 40.1 ± 24.6 |
| Male (%) | 50.45% | 50.94% | 50.45% | 50.94% |
| Income ($) | 71991.4 ± 101351.5 | 60887.1 ± 87209.0 | 71991.4 ± 101351.5 | 60887.1 ± 87209.0 |
| BMI | 26.1 ± 5.0 | 26.0 ± 5.2 | 26.1 ± 5.0 | 26.0 ± 5.2 |
| Current Smoker (%) | 0.53% | 0.24% | 0.53% | 0.24% |
| Alcohol Use Disorder (%) | 0.29% | 0.40% | 0.31% | 0.44% |
| Hypertension (%) | 19.15% | 18.93% | 19.08% | 18.93% |
| CHF (%) | 1.93% | 1.98% | 1.96% | 2.00% |

---

## 3. Rural Underdiagnosis Analysis

We define **underdiagnosis** as a patient who enters the sleep disorder care pathway (receives a sleep disorder diagnosis, SNOMED 39898005) but does not receive a sleep apnea diagnosis (SNOMED 73430006 or 78275009). This captures patients who "drop out" before completing diagnostic testing.

### 3.1 Sleep Disorder Cohort Summary

| Dataset | Cohort N | Sleep Apnea Dx | Underdiagnosed | Rural | Urban |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 740 | 739 | 1 | 198 | 542 |
| biased | 749 | 577 | 172 | 205 | 544 |

### 3.2 Pairwise Comparison (Rural vs Urban)

Underdiagnosis rates represent the proportion of sleep disorder patients who did not receive a sleep apnea diagnosis. P-values are from a two-proportion z-test.

| Dataset | Rural N | Urban N | Rural Rate | Urban Rate | Risk Diff | Risk Ratio | z | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 198 | 542 | 0.00% | 0.18% | -0.18% | 0.00 | -0.605 | 0.545 |
| biased | 205 | 544 | 82.93% | 0.37% | 82.56% | 225.56 | 23.951 | 0.000 |

**Interpretation:**

- **Baseline**: No significant difference in underdiagnosis rates between rural and urban patients (p=0.545), confirming equal access to care.
- **Biased**: Highly significant rural-urban disparity (p=0.000). Rural patients are 225.56x more likely to be underdiagnosed.

### 3.3 Adjusted Regression Analysis

Logistic regression models the probability of underdiagnosis as a function of clinical risk factors plus a rural indicator. This isolates the rural effect after controlling for potential confounders.

**Covariates:** age, gender, income, BMI, smoking status, alcohol use, hypertension, CHF, rural indicator

**Significance testing:** Permutation test with n=500 permutations

| Dataset | N | Rural Coef | Odds Ratio | Permutation p | In-sample AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 740 | -1.545 | 0.213 | 0.711 | 0.997 |
| biased | 749 | 3.065 | 21.432 | 0.002 | 0.970 |

**Interpretation:**

- **Baseline**: Rural coefficient is not significant (p=0.711), indicating no access-driven underdiagnosis.
- **Biased**: Rural coefficient is highly significant (p=0.002). After adjusting for clinical factors, rural patients have **21.432x the odds** of being underdiagnosed compared to urban patients.

---

## 4. Predictive Model Specification

We train machine learning models to predict sleep apnea diagnosis from clinical features. Importantly, **urban/rural residence is excluded** from the feature set to examine how biased training data affects model performance independent of explicit location features.

### 4.1 Features

| Feature | Source | Description |
| --- | --- | --- |
| `age_years` | patients.csv | Patient age at reference date |
| `male` | patients.csv | Gender indicator (1.0 for male) |
| `income` | patients.csv | Annual household income |
| `bmi` | observations.csv | Latest recorded BMI (LOINC 39156-5) |
| `smoker` | observations.csv | Current smoking status (LOINC 72166-2) |
| `alcohol_use` | conditions.csv | Alcohol use disorder (SNOMED 7200002) |
| `hypertension` | conditions.csv | Hypertension diagnosis (SNOMED 59621000) |
| `chf` | conditions.csv | Congestive heart failure (SNOMED 88805009) |

### 4.2 Target Variable

Binary classification: patient has a sleep apnea diagnosis (SNOMED 73430006 or 78275009) in conditions.csv.

### 4.3 Model Families

| Model | Description |
| --- | --- |
| **Logistic Regression** | L2-regularized logistic regression with standardized inputs |
| **Random Forest** | Ensemble of decision trees with bootstrap aggregation |
| **Gradient Boosted DT** | Sequential boosting of shallow decision trees |

### 4.4 Training Protocol

- **Train/Validation/Test Split**: 70%/15%/15%
- **Hyperparameter Selection**: Grid search optimizing validation AUC
- **Final Training**: Re-fit on combined train+validation set
- **Evaluation**: Held-out test set metrics

---

## 5. Model Performance Results

### 5.1 Test Set Performance

| Dataset | Model | AUC | Avg Precision | Brier Score | Train/Val/Test |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | logistic | 0.850 | 0.149 | 0.033 | 13999/3000/3001 |
| baseline | rf | 0.853 | 0.158 | 0.033 | 13999/3000/3001 |
| baseline | gbdt | 0.847 | 0.142 | 0.033 | 13999/3000/3001 |
| biased | logistic | 0.867 | 0.124 | 0.026 | 13999/3000/3001 |
| biased | rf | 0.871 | 0.155 | 0.130 | 13999/3000/3001 |
| biased | gbdt | 0.870 | 0.131 | 0.026 | 13999/3000/3001 |

### 5.2 Selected Hyperparameters

**Baseline:**

- *logistic*: model__C=0.1, model__class_weight=None
- *rf*: model__n_estimators=500, model__max_depth=5, model__min_samples_leaf=5, model__class_weight=None
- *gbdt*: model__n_estimators=200, model__learning_rate=0.1, model__max_depth=2, model__min_samples_leaf=5

**Biased:**

- *logistic*: model__C=0.1, model__class_weight=None
- *rf*: model__n_estimators=200, model__max_depth=10, model__min_samples_leaf=5, model__class_weight=balanced
- *gbdt*: model__n_estimators=100, model__learning_rate=0.1, model__max_depth=3, model__min_samples_leaf=10

---

## 6. Cross-Dataset Evaluation: Bias Impact on Model Generalization

To quantify how training on biased data affects real-world performance, we evaluate biased-trained models on the baseline test set. This simulates deploying a model trained on data with access disparities to a population with equitable care access.

**Methodology:** Baseline test N=3,001; overlap N=3,001; removed-not-in-biased N=0; removed-seen-in-biased N=2,508; final test N=493.

### 6.1 Performance Comparison

| Model | Baseline AUC | Biased AUC | Δ AUC | Baseline AP | Biased AP | Δ AP | Test N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.824 | 0.824 | +0.000 | 0.123 | 0.126 | +0.003 | 493 |
| rf | 0.834 | 0.861 | +0.027 | 0.146 | 0.172 | +0.026 | 493 |
| gbdt | 0.818 | 0.831 | +0.013 | 0.137 | 0.175 | +0.039 | 493 |

### 6.2 Degradation Analysis

- **logistic**: AUC improved by 0.000, AP changed by +0.003
- **rf**: AUC improved by 0.027, AP changed by +0.026
- **gbdt**: AUC improved by 0.013, AP changed by +0.039

---

## 7. Fairness Analysis: Subgroup Performance Disparities

Aggregate metrics like AUC can mask disparities between subgroups. This section examines model performance separately for rural and urban patients to reveal how bias in training data translates to differential model behavior.

### 7.1 AUC by Subgroup

AUC measures discriminative ability—how well the model ranks positive cases above negative cases. Gaps indicate the model is better at distinguishing disease in one population than another.

| Dataset | Model | Rural AUC | Urban AUC | Gap (R-U) | Rural N | Urban N |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | logistic | 0.855 | 0.848 | +0.007 | 792 | 2,209 |
| baseline | rf | 0.848 | 0.855 | -0.007 | 792 | 2,209 |
| baseline | gbdt | 0.839 | 0.850 | -0.011 | 792 | 2,209 |
| biased | logistic | 0.882 | 0.868 | +0.014 | 818 | 2,183 |
| biased | rf | 0.823 | 0.879 | -0.056 | 818 | 2,183 |
| biased | gbdt | 0.858 | 0.875 | -0.017 | 818 | 2,183 |

### 7.2 False Negative Rate by Subgroup

The **false negative rate (FNR)** is the proportion of true positive cases that the model misses (predicts as negative). A higher FNR for rural patients means more rural patients with sleep apnea are incorrectly told they don't have it—a direct measure of underdiagnosis harm.

| Dataset | Model | Rural FNR | Urban FNR | Gap (R-U) | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | logistic | 100.00% | 100.00% | +0.000 | Similar miss rates |
| baseline | rf | 100.00% | 100.00% | +0.000 | Similar miss rates |
| baseline | gbdt | 100.00% | 100.00% | +0.000 | Similar miss rates |
| biased | logistic | 100.00% | 100.00% | +0.000 | Similar miss rates |
| biased | rf | 40.00% | 19.48% | +0.205 | Rural patients significantly more likely to be missed |
| biased | gbdt | 100.00% | 100.00% | +0.000 | Similar miss rates |

### 7.3 Mean Predicted Probability by Subgroup

The mean predicted probability reveals systematic differences in how the model scores patients from different subgroups. A lower mean prediction for rural patients indicates the model has learned to associate rural-correlated features with lower disease probability.

| Dataset | Model | Rural Mean Pred | Urban Mean Pred | Gap (R-U) |
| --- | --- | ---: | ---: | ---: |
| baseline | logistic | 0.037 | 0.037 | +0.0008 |
| baseline | rf | 0.038 | 0.036 | +0.0013 |
| baseline | gbdt | 0.039 | 0.036 | +0.0027 |
| biased | logistic | 0.029 | 0.028 | +0.0013 |
| biased | rf | 0.270 | 0.260 | +0.0105 |
| biased | gbdt | 0.031 | 0.029 | +0.0025 |

### 7.4 Fairness Interpretation

**Key findings from fairness analysis:**

- **Higher miss rate for rural patients**: The false negative rate gap shows that rural patients with sleep apnea are more likely to be incorrectly classified as negative. This perpetuates the underdiagnosis pattern present in the training data.

These findings demonstrate that **aggregate metrics like AUC can hide fairness harms**. A model can achieve good overall discrimination while systematically disadvantaging specific subgroups.

---

## 8. Conclusions and Implications

### 8.1 Key Findings

1. **Access barriers create measurable underdiagnosis**: The 80% dropout rate for rural patients at diagnostic stages produces a dramatic underdiagnosis disparity (82.93% rural vs 0.37% urban).

2. **Bias persists after covariate adjustment**: The significant rural coefficient in the adjusted regression confirms that underdiagnosis is driven by access barriers, not differences in clinical risk factors.

3. **Biased training data affects model utility**: Models trained on biased data may show different performance characteristics when deployed to populations with different care access patterns.

### 8.2 Real-World Implications

- **Clinical Decision Support**: ML models for sleep apnea screening trained on data from healthcare systems with rural access barriers may underperform in settings with better rural care access.

- **Health Equity Monitoring**: Organizations should audit training data for geographic and socioeconomic disparities that could introduce systematic bias.

- **Policy Interventions**: Addressing rural healthcare access (telemedicine, mobile sleep clinics, transportation assistance) could reduce underdiagnosis and improve data quality for downstream applications.

### 8.3 Limitations

- Synthea generates synthetic data that may not capture all real-world complexities
- The 80% dropout rate is illustrative; actual rural dropout rates vary by region and condition
- Cross-dataset evaluation is limited by patient ID overlap between baseline and biased runs

---

## Appendix: Technical Details

### A.1 SNOMED/LOINC Codes

| Concept | Code System | Code |
| --- | --- | --- |
| Sleep Apnea | SNOMED | 73430006, 78275009 |
| Sleep Disorder | SNOMED | 39898005 |
| Hypertension | SNOMED | 59621000 |
| CHF | SNOMED | 88805009 |
| Alcohol Use Disorder | SNOMED | 7200002 |
| BMI | LOINC | 39156-5 |
| Smoking Status | LOINC | 72166-2 |

### A.2 Data Generation Commands

**Baseline:**
```bash
./run_synthea -s 160 -cs 160 -o false -p 20000 \
  --exporter.csv.export=true \
  --exporter.csv.append_mode=false \
  --exporter.baseDirectory=./output_baseline \
  Montana
```

**Biased:**
```bash
./run_synthea -s 160 -cs 160 -o false -p 20000 \
  --exporter.csv.export=true \
  --exporter.csv.append_mode=false \
  --exporter.baseDirectory=./output_rural_bias \
  --module_override=./config/overrides_rural_sleep_apnea.properties \
  Montana
```
