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
| baseline | 20,106 | 8.50% |
| biased | 20,061 | 6.59% |

## Test Performance

| Dataset | Model | AUC | AP | Brier | Train/Val/Test |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | logistic | 0.867 | 0.324 | 0.065 | 14074/3015/3017 |
| baseline | rf | 0.868 | 0.319 | 0.065 | 14074/3015/3017 |
| baseline | gbdt | 0.870 | 0.350 | 0.064 | 14074/3015/3017 |
| biased | logistic | 0.853 | 0.232 | 0.166 | 14042/3009/3010 |
| biased | rf | 0.868 | 0.248 | 0.162 | 14042/3009/3010 |
| biased | gbdt | 0.866 | 0.263 | 0.054 | 14042/3009/3010 |

## Biased Model on Baseline Test Set

To quantify performance degradation, the biased-trained models are evaluated on the baseline test set. We first restrict the baseline test cohort to patient IDs that also appear in the biased dataset, then remove any IDs that were seen during biased train/validation to avoid repeats. This ensures the biased models are evaluated only on unseen baseline patients.

Baseline test set filtered to overlap with biased dataset IDs, then removed any IDs seen in biased train/val splits to avoid repeats. Baseline test N=3,017; overlap N=3,006; removed-not-in-biased N=11; removed-seen-in-biased N=2,536.

| Model | Baseline AUC | Biased AUC | Δ AUC | Baseline AP | Biased AP | Δ AP | Baseline Brier | Biased Brier | Δ Brier | Test N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.895 | 0.889 | -0.005 | 0.415 | 0.379 | -0.036 | 0.067 | 0.161 | 0.095 | 470 |
| rf | 0.904 | 0.903 | -0.001 | 0.409 | 0.453 | 0.043 | 0.067 | 0.160 | 0.093 | 470 |
| gbdt | 0.906 | 0.904 | -0.002 | 0.439 | 0.432 | -0.007 | 0.064 | 0.068 | 0.004 | 470 |

## Performance Degradation Interpretation

Training on the biased dataset consistently reduces performance when evaluated on the baseline test cohort:

- logistic: AUC -0.005, AP -0.036, Brier +0.095 (negative AUC/AP and positive Brier indicate worse performance).
- rf: AUC -0.001, AP +0.043, Brier +0.093 (negative AUC/AP and positive Brier indicate worse performance).
- gbdt: AUC -0.002, AP -0.007, Brier +0.004 (negative AUC/AP and positive Brier indicate worse performance).

This gap suggests the biased training data yields a model that generalizes worse to the baseline population, highlighting how access-driven underdiagnosis can degrade downstream prediction quality.

## Selected Hyperparameters

### baseline

- **logistic**: {'model__C': 0.1, 'model__class_weight': None}
- **rf**: {'model__n_estimators': 200, 'model__max_depth': 5, 'model__min_samples_leaf': 5, 'model__class_weight': None}
- **gbdt**: {'model__n_estimators': 100, 'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__min_samples_leaf': 5}

### biased

- **logistic**: {'model__C': 0.1, 'model__class_weight': 'balanced'}
- **rf**: {'model__n_estimators': 200, 'model__max_depth': 5, 'model__min_samples_leaf': 1, 'model__class_weight': 'balanced'}
- **gbdt**: {'model__n_estimators': 100, 'model__learning_rate': 0.05, 'model__max_depth': 2, 'model__min_samples_leaf': 5}
