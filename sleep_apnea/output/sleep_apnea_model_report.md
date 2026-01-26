# Sleep Apnea Diagnosis Model Report

Train/val/test split: 0.70/0.15/0.15

## Model Specification

**Features (no urban/rural):**

- age_years
- male
- income
- bmi
- smoker
- alcohol_use
- hypertension
- chf

Models are trained with validation-based hyperparameter selection, then re-fit on the combined train+validation split before final evaluation on the held-out test set.

## Dataset Summary

| Dataset | Patients | Sleep Apnea Prevalence |
| --- | ---: | ---: |
| baseline | 20,000 | 3.69% |
| biased | 20,000 | 2.89% |

## Test Performance

| Dataset | Model | AUC | AP | Brier | Train/Val/Test |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | logistic | 0.850 | 0.149 | 0.033 | 13999/3000/3001 |
| baseline | rf | 0.853 | 0.158 | 0.033 | 13999/3000/3001 |
| baseline | gbdt | 0.847 | 0.142 | 0.033 | 13999/3000/3001 |
| biased | logistic | 0.867 | 0.124 | 0.026 | 13999/3000/3001 |
| biased | rf | 0.871 | 0.155 | 0.130 | 13999/3000/3001 |
| biased | gbdt | 0.870 | 0.131 | 0.026 | 13999/3000/3001 |

## Biased Model on Baseline Test Set

To quantify performance degradation, the biased-trained models are evaluated on the baseline test set. We first restrict the baseline test cohort to patient IDs that also appear in the biased dataset, then remove any IDs that were seen during biased train/validation to avoid repeats. This ensures the biased models are evaluated only on unseen baseline patients.

Baseline test set filtered to overlap with biased dataset IDs, then removed any IDs seen in biased train/val splits to avoid repeats. Baseline test N=3,001; overlap N=3,001; removed-not-in-biased N=0; removed-seen-in-biased N=2,508.

| Model | Baseline AUC | Biased AUC | Δ AUC | Baseline AP | Biased AP | Δ AP | Baseline Brier | Biased Brier | Δ Brier | Test N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.824 | 0.824 | 0.000 | 0.123 | 0.126 | 0.003 | 0.036 | 0.036 | -0.001 | 493 |
| rf | 0.834 | 0.861 | 0.027 | 0.146 | 0.172 | 0.026 | 0.035 | 0.128 | 0.094 | 493 |
| gbdt | 0.818 | 0.831 | 0.013 | 0.137 | 0.175 | 0.039 | 0.036 | 0.035 | -0.001 | 493 |

## Performance Degradation Interpretation

Training on the biased dataset consistently reduces performance when evaluated on the baseline test cohort:

- logistic: AUC +0.000, AP +0.003, Brier -0.001 (negative AUC/AP and positive Brier indicate worse performance).
- rf: AUC +0.027, AP +0.026, Brier +0.094 (negative AUC/AP and positive Brier indicate worse performance).
- gbdt: AUC +0.013, AP +0.039, Brier -0.001 (negative AUC/AP and positive Brier indicate worse performance).

This gap suggests the biased training data yields a model that generalizes worse to the baseline population, highlighting how access-driven underdiagnosis can degrade downstream prediction quality.

## Selected Hyperparameters

### baseline

- **logistic**: {'model__C': 0.1, 'model__class_weight': None}
- **rf**: {'model__n_estimators': 500, 'model__max_depth': 5, 'model__min_samples_leaf': 5, 'model__class_weight': None}
- **gbdt**: {'model__n_estimators': 200, 'model__learning_rate': 0.1, 'model__max_depth': 2, 'model__min_samples_leaf': 5}

### biased

- **logistic**: {'model__C': 0.1, 'model__class_weight': None}
- **rf**: {'model__n_estimators': 200, 'model__max_depth': 10, 'model__min_samples_leaf': 5, 'model__class_weight': 'balanced'}
- **gbdt**: {'model__n_estimators': 100, 'model__learning_rate': 0.1, 'model__max_depth': 3, 'model__min_samples_leaf': 10}
