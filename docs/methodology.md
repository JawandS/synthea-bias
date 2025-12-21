# Evaluating Representativeness in Healthcare Claims Data

## Methodology Documentation

---

## 1. Overview and Objectives

This methodology provides a structured, implementation-ready framework for evaluating **representativeness and bias in healthcare claims data** used in actuarial and AI/ML applications. It is designed to support compliance with actuarial standards, improve model reliability, and reduce the risk of unfair or misleading outcomes arising from non-representative data.

The methodology emphasizes **representative results rather than representative sampling**, acknowledging that administrative claims data are generated through utilization processes rather than population-based sampling designs.

**Primary objectives:**

* Define and operationalize representativeness in claims data
* Quantify representation gaps across key dimensions
* Mitigate bias during model development
* Monitor bias during deployment
* Support transparent documentation and governance

---

## 2. Conceptual Framework

### 2.1 Defining Representativeness in Claims Data

Representativeness is defined as the extent to which analytical results and model outputs are expected to generalize to a **clearly specified target population**.

Key distinctions:

* **Observed claims population:** Individuals with coverage who generated claims
* **Target population:** Population to which actuarial inference or AI deployment is intended

Claims data reflect successful navigation of multiple access points (coverage, provider access, coding, adjudication), creating systematic representation gaps.

### 2.2 Representative Sampling vs. Representative Results

* **Representative sampling:** Dataset mirrors population composition (rarely feasible)
* **Representative results:** Model relationships remain stable across subgroups

This methodology focuses on achieving and validating representative results through statistical assessment and adjustment.

---

## 3. Three-Stage Bias Assessment Framework

The framework evaluates bias across the AI/ML lifecycle:

1. **Input Bias** – data representativeness
2. **Model Bias** – development and estimation choices
3. **Application Bias** – deployment and use

---

## 4. Stage 1: Input Bias Assessment

### 4.1 Purpose

Assess whether claims data are appropriate and sufficient for the intended actuarial application by identifying representation gaps.

### 4.2 Dimensions Evaluated

* Demographics (age, sex)
* Geography (e.g., RUCA classifications)
* Clinical conditions and coding
* Utilization patterns
* Insurance type and plan characteristics

### 4.3 Quantitative Metrics

**Distributional metrics:**

* Standardized Mean Differences (SMD)

  * |SMD| > 0.10 indicates meaningful imbalance
* Variance ratios
* Kullback–Leibler (KL) divergence

**Statistical tests (used cautiously):**

* Chi-square tests (categorical variables)
* Confidence intervals for group differences

Statistical significance must be interpreted alongside **clinical and actuarial relevance**.

### 4.4 Interpretation Guidelines

* Emphasize magnitude over p-values in large samples
* Identify gaps that are material to financial or operational decisions
* Document known sources of systematic underrepresentation

---

## 5. Stage 2: Model Bias Mitigation

### 5.1 Purpose

Reduce bias arising from input imbalance and modeling decisions during development.

### 5.2 Propensity Score Approaches

* Matching
* Stratification
* Covariate adjustment
* Inverse Probability Weighting (IPW)

**Key considerations:**

* Diagnostics for common support
* Weight stability and trimming
* Selection of estimand (ATE vs. ATT)

### 5.3 Entropy Balancing

Entropy balancing enforces covariate balance through optimization rather than estimated treatment probabilities.

Advantages:

* Direct balance constraints
* Improved efficiency in some settings
* Reduced reliance on model specification

### 5.4 Calibration and Post-Stratification

* Post-stratification using external benchmarks
* Raking (iterative proportional fitting)
* Calibration weighting with linear constraints

Useful when reliable population margins are available.

### 5.5 Missing Data Treatment

* Missingness is often **not MCAR**
* Listwise deletion risks excluding underrepresented groups
* Methods:

  * Single imputation (mean, regression, kNN)
  * Multiple imputation
  * Pattern-mixture models (MNAR sensitivity)

Higher imputation accuracy does **not guarantee fairness**.

### 5.6 Sensitivity Analysis

Recommended analyses include:

* Alternative covariate definitions
* Alternative model forms (e.g., GLM vs. tree-based)
* Alternative weighting methods
* Outcome definition robustness

---

## 6. Stage 3: Application Bias Monitoring

### 6.1 Purpose

Ensure deployed models perform consistently and fairly across subgroups.

### 6.2 Performance Monitoring

**Continuous outcomes:**

* RMSE, MAE by subgroup

**Binary outcomes:**

* Brier score
* Classification error

**Discrimination and calibration:**

* ROC/AUC by subgroup
* Calibration plots

### 6.3 Fairness Metrics

Select metrics based on error types most likely to cause harm:

* Equal opportunity (TPR parity)
* Equalized odds (TPR + FPR parity)
* Predictive parity
* False negative rate parity
* Demographic parity (context-dependent)

---

## 7. Actuarial Standards Alignment

The methodology aligns with:

* **ASOP 23 (Data Quality):** appropriateness, sufficiency, disclosure
* **ASOP 12 (Risk Classification):** avoidance of unfair discrimination
* **ASOP 41 (Actuarial Communications):** transparency and documentation
* **ASOP 56 (Modeling):** validation, sensitivity, limitation disclosure

---

## 8. Synthetic Data Application

### 8.1 Purpose

Demonstrate the framework using controlled, reproducible scenarios.

### 8.2 Scenarios

1. **Urban vs. Rural Underrepresentation**
2. **Clinical Documentation Bias**
3. **Age Distribution Bias**

Each scenario includes:

* Reference population
* Biased claims dataset
* Validation of representation gaps

### 8.3 Validation Criteria

* Distributional tests (chi-square, KL divergence)
* Standardized mean differences
* Cost concentration metrics (5/50 analysis)

---

## 9. Model Documentation Using Model Cards

### 9.1 Purpose

Model cards function as **"nutrition labels"** for actuarial AI models.

### 9.2 Core Elements

* Model details and provenance
* Intended and out-of-scope use
* Training and evaluation data
* Subgroup performance metrics
* Ethical considerations and limitations

### 9.3 Actuarial Extensions

* ASOP mapping
* Claims-specific data limitations
* Financial materiality context
* Regulatory environment

---

## 10. Implementation Guidance

* Integrate documentation into validation workflows
* Maintain model cards through version control
* Update documentation following material changes
* Use scenario-based disclosures to guide deployment decisions

---

## 11. Summary

This methodology provides actuaries with a rigorous, standards-aligned approach to assessing and governing representativeness in healthcare claims data used for AI/ML applications. It treats representativeness as a **measurable source of model risk** and embeds fairness considerations within existing actuarial practice rather than as a separate ethical overlay.
