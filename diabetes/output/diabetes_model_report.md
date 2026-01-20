# Diabetes Diagnosis Model Report

Train/val/test split: 0.70/0.15/0.15

## Model Specification

**Features:**

- age_years
- male
- income
- bmi
- smoker
- obesity
- hypertension
- hyperlipidemia
- hyperglycemia
- hypertriglyceridemia

Models are trained with validation-based hyperparameter selection, then re-fit on the combined train+validation split before final evaluation on the held-out test set.

## Cohort Selection

To ensure comparable results, both datasets are filtered to include only patient IDs that appear in both baseline and biased datasets. The same train/val/test split is then applied to both.

| Metric | Count |
| --- | ---: |
| Baseline original patients | 20,000 |
| Biased original patients | 20,000 |
| Common patient IDs (kept) | 20,000 |
| Baseline patients dropped | 0 |
| Biased patients dropped | 0 |

## Dataset Summary

| Dataset | Patients | Diabetes Prevalence |
| --- | ---: | ---: |
| baseline | 20,000 | 6.26% |
| biased | 20,000 | 6.29% |

## Test Performance

| Dataset | Model | AUC | AP | Brier | Train/Val/Test |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | logistic | 0.988 | 0.963 | 0.006 | 13999/3000/3001 |
| baseline | rf | 0.987 | 0.968 | 0.005 | 13999/3000/3001 |
| baseline | gbdt | 0.990 | 0.968 | 0.003 | 13999/3000/3001 |
| biased | logistic | 0.982 | 0.947 | 0.005 | 13999/3000/3001 |
| biased | rf | 0.982 | 0.949 | 0.021 | 13999/3000/3001 |
| biased | gbdt | 0.984 | 0.950 | 0.005 | 13999/3000/3001 |

## Cross-Dataset Evaluation

Both models are evaluated on the same held-out test set using baseline feature values. This directly compares how training on biased vs baseline data affects predictions for identical patients.

| Model | Baseline AUC | Biased AUC | Δ AUC | Baseline AP | Biased AP | Δ AP | Baseline Brier | Biased Brier | Δ Brier | Test N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.988 | 0.990 | 0.002 | 0.963 | 0.968 | 0.005 | 0.006 | 0.004 | -0.002 | 3,001 |
| rf | 0.987 | 0.990 | 0.003 | 0.968 | 0.972 | 0.004 | 0.005 | 0.020 | 0.015 | 3,001 |
| gbdt | 0.990 | 0.991 | 0.000 | 0.968 | 0.970 | 0.002 | 0.003 | 0.004 | 0.000 | 3,001 |

## Performance Degradation Interpretation

Training on the biased dataset (with 30% under-documentation of hyperglycemia/hypertriglyceridemia) affects performance when evaluated on the baseline test cohort:

- logistic: AUC +0.002, AP +0.005, Brier -0.002 (negative AUC/AP and positive Brier indicate worse performance).
- rf: AUC +0.003, AP +0.004, Brier +0.015 (negative AUC/AP and positive Brier indicate worse performance).
- gbdt: AUC +0.000, AP +0.002, Brier +0.000 (negative AUC/AP and positive Brier indicate worse performance).

This gap suggests documentation bias in the training data affects model generalization, particularly because hyperglycemia and hypertriglyceridemia are highly predictive of diabetes.

## Selected Hyperparameters

### baseline

- **logistic**: {'model__C': 10.0, 'model__class_weight': 'balanced'}
- **rf**: {'model__n_estimators': 200, 'model__max_depth': 10, 'model__min_samples_leaf': 5, 'model__class_weight': 'balanced'}
- **gbdt**: {'model__n_estimators': 100, 'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__min_samples_leaf': 10}

### biased

- **logistic**: {'model__C': 10.0, 'model__class_weight': None}
- **rf**: {'model__n_estimators': 200, 'model__max_depth': 5, 'model__min_samples_leaf': 5, 'model__class_weight': 'balanced'}
- **gbdt**: {'model__n_estimators': 100, 'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__min_samples_leaf': 5}
