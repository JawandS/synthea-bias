# Methodology Summary: Evaluating Representativeness in Healthcare Claims Data

> **Purpose of this Document**
>
> This markdown document provides a comprehensive, implementation-oriented summary of the *Methodology Documentation* for **Evaluating Representativeness in Healthcare Claims Data**. It is intended to serve as a **practical guide for synthetic data generation, model development, validation, and documentation**, not merely as an academic summary.
>
> The goal is to enable reproducible implementation of representativeness assessment, bias mitigation, and AI governance in actuarial applications using healthcare claims data.

---

## 1. Conceptual Foundation

### 1.1 Core Problem

Healthcare claims data are **administrative utilization records**, not population health data. Individuals appear in claims data only if they:

1. Have insurance coverage
2. Successfully access healthcare services
3. Receive services that are documented and coded
4. Generate reimbursable claims

Each step introduces **systematic selection effects** that disproportionately exclude or distort representation of:

* Rural populations
* Racial and ethnic minorities
* Uninsured or underinsured individuals
* Pediatric populations (commercial data)
* Oldest-old populations (Medicare data)
* Patients with underdiagnosed or undercoded conditions

AI/ML models trained on claims data therefore risk **structural bias**, not random noise.

---

### 1.2 Representativeness Redefined

The methodology explicitly rejects naive notions of representativeness based on sample mirroring.

**Key distinction:**

* *Representative sampling* → Rarely achievable with claims data
* *Representative results* → Central objective

A dataset is considered *representative* if **model conclusions are expected to generalize to a clearly defined target population**, even when sample composition differs.

Three operational definitions:

1. **Representative composition** – Sample proportions match target population
2. **Representative estimates** – Aggregate statistics generalize
3. **Representative heterogeneous effects** – Model performance is stable across subgroups

The methodology primarily targets (2) and (3).

---

## 2. Target Population Specification (Mandatory Step)

Every representativeness assessment begins with **explicit target population definition**.

Required components:

* Eligibility criteria (age, coverage, enrollment duration)
* Insurance product type (Commercial, Medicare, Medicaid)
* Geographic scope (state, county, RUCA category)
* Time period
* Intended actuarial application (pricing, risk adjustment, UM, fraud)

> **Implementation rule:** No representativeness analysis is valid without an explicit target population statement.

---

## 3. Three-Stage Bias Assessment Framework

The methodology structures bias evaluation across the full AI/ML lifecycle.

---

### Stage 1: Input Bias Assessment (Data Representativeness)

**Objective:** Quantify how and where claims data differ from the target population.

#### 3.1 Dimensions of Assessment

* Demographics: age, sex
* Geography: urban/rural (RUCA), region
* Clinical: chronic condition prevalence, comorbidity burden
* Utilization: visits, admissions, pharmacy claims
* Costs: spending distribution and concentration

#### 3.2 Quantitative Metrics

| Metric                             | Purpose               | Interpretation Thresholds                   |
| ---------------------------------- | --------------------- | ------------------------------------------- |
| Standardized Mean Difference (SMD) | Scale-free imbalance  | |SMD| > 0.10 indicates meaningful imbalance |
| KL Divergence                      | Information loss      | < 0.10 typically acceptable                 |
| Chi-square + Effect Size           | Categorical imbalance | Use Cramér’s V, not p-values alone          |
| Variance Ratios                    | Distribution spread   | Deviations from 1 indicate heterogeneity    |
| 5/50 Cost Metric                   | Cost concentration    | Validates realism and representation        |

#### 3.3 Statistical Significance vs Materiality

* Large datasets make p-values misleading
* Emphasis is placed on **magnitude and actuarial relevance**, not hypothesis rejection
* Statistical tests are diagnostic, not dispositive

---

### Stage 2: Model Bias Mitigation (Development Phase)

**Objective:** Reduce sensitivity of model outputs to representation gaps while preserving interpretability and stability.

#### 3.4 Covariate Balancing Methods

1. **Propensity Score Methods**

   * Matching
   * Stratification
   * Regression adjustment
   * Inverse Probability Weighting (IPW)

2. **Entropy Balancing (Preferred)**

   * Directly enforces covariate balance
   * Avoids extreme weights
   * Efficient when treated group is large

3. **Calibration & Post-Stratification**

   * Aligns marginal distributions to external benchmarks
   * Supports census- or survey-based adjustments

4. **Raking / Iterative Proportional Fitting**

   * Balances across multiple dimensions simultaneously

> **Key principle:** No single method is sufficient; methods are iterative and context-dependent.

---

#### 3.5 Missing Data Handling

Key findings from literature:

* Missingness is rarely MCAR
* Dropping incomplete cases disproportionately excludes marginalized populations
* Imputation accuracy ≠ fairness

Recommended practices:

* Diagnose missingness mechanism (MCAR / MAR / MNAR)
* Use multiple imputation where feasible
* Include subgroup indicators in imputation models
* Conduct sensitivity analyses using pattern-mixture models
* Explicitly document residual uncertainty

---

#### 3.6 Sensitivity Analysis

Required robustness checks:

* Alternative model forms (GLM vs tree-based)
* Alternative covariate definitions
* Alternative balancing specifications
* Alternative missing data assumptions
* Threshold sensitivity for classification models

Sensitivity analysis is treated as **model risk disclosure**, not optimization.

---

### Stage 3: Application Bias Monitoring (Deployment Phase)

**Objective:** Detect bias that emerges during real-world use.

#### 3.7 Performance Monitoring

Metrics reported **overall and by subgroup**:

* Regression: RMSE, MAE, R²
* Classification: AUC, sensitivity, specificity
* Calibration: observed vs predicted risk curves

#### 3.8 Fairness Metrics (Context-Dependent)

| Metric                | Focus                     |
| --------------------- | ------------------------- |
| Equal Opportunity     | True positive rate parity |
| Equalized Odds        | TPR + FPR parity          |
| Predictive Parity     | Precision parity          |
| False Negative Parity | Missed cases              |
| Demographic Parity    | Output independence       |

Metric choice depends on **harm asymmetry** of errors.

#### 3.9 Operational Bias Risks

* Automation bias (over-trust)
* Dismissal bias (alert fatigue)
* Feedback loops (biased decisions → biased data)

Monitoring is continuous, not one-time.

---

## 4. Synthetic Data Generation Framework

### 4.1 Design Principles

Synthetic data are used to:

* Isolate bias mechanisms
* Test representativeness diagnostics
* Validate mitigation strategies

Each scenario includes:

* Reference population dataset
* Biased claims dataset
* Sample size ≥ 10,000
* Realistic utilization and cost distributions

---

### 4.2 Scenario 1: Urban / Rural Bias

**Mechanism:** Differential access to care

Key features:

* RUCA-based geographic classification
* Rural underrepresentation
* Lower utilization rates
* Higher ED reliance
* Undercoded chronic conditions

Assessment metrics:

* Geographic distribution tests
* Utilization SMDs
* Cost concentration metrics

---

### 4.3 Scenario 2: Clinical Documentation Bias

**Mechanism:** Provider coding practices

Key features:

* Undercoding of diabetes complications, mental health
* Specialty-based coding intensity
* Pharmacy–diagnosis discordance

Assessment metrics:

* Diagnosis prevalence tests
* Prescription–diagnosis concordance
* Provider-level KL divergence

---

### 4.4 Scenario 3: Age Distribution Bias

**Mechanism:** Coverage and survivorship effects

Key features:

* Pediatric underrepresentation (commercial)
* Oldest-old underrepresentation (Medicare)
* Healthier-than-average observed cohorts

Assessment metrics:

* Age-band chi-square tests
* KS tests for age distributions
* Age-specific utilization SMDs

---

### 4.5 Validation Criteria

Synthetic data must satisfy:

* Realistic spending concentration (5/50 rule)
* Adequate subgroup sample sizes
* Plausible utilization–condition relationships
* Scenario-specific imbalance thresholds

---

## 5. Model Documentation via Model Cards

### 5.1 Purpose

Model cards operationalize:

* ASOP 41 (communication)
* ASOP 23 (data quality)
* ASOP 56 (modeling)

They function as **actuarial governance artifacts**, not marketing summaries.

---

### 5.2 Core Sections

1. Model Details
2. Intended Use / Out-of-Scope Use
3. Factors and Metrics
4. Training Data
5. Evaluation Data
6. Ethical Considerations and Limitations
7. Caveats and Recommendations

---

### 5.3 Actuarial Extensions

Additional required documentation:

* ASOP mapping
* Claims data lineage
* Coding intensity variation
* Claims lag and maturity
* Plan design effects
* Network configuration
* Financial materiality context

---

## 6. Mitigation and Disclosure Strategy

Key principle:

> Not all bias can be removed. All material bias must be documented.

Mitigation options:

* Data supplementation (census, surveys)
* Statistical adjustment
* Recalibration
* Use restrictions
* Enhanced monitoring

Residual risks are disclosed, not hidden.

---

## 7. Implementation Guidance

### 7.1 Workflow Integration

1. Define target population
2. Perform Stage 1 diagnostics
3. Apply Stage 2 adjustments
4. Validate with synthetic scenarios
5. Document via model card
6. Deploy with Stage 3 monitoring

### 7.2 Maintenance

* Annual documentation refresh
* Triggered updates for material changes
* Continuous subgroup monitoring

---

## 8. Bottom Line

This methodology establishes **representativeness as a core component of actuarial model risk management**.

It provides:

* Quantitative diagnostics
* Mitigation tools
* Documentation standards
* Governance processes

The objective is not perfect fairness, but **defensible, transparent, and professionally compliant AI use in healthcare actuarial practice**.
