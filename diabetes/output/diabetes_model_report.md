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

## Dataset Summary

| Dataset | Patients | Diabetes Prevalence |
| --- | ---: | ---: |
| baseline | 20,106 | 6.31% |
| biased | 20,000 | 6.29% |

## Test Performance

| Dataset | Model | AUC | AP | Brier | Train/Val/Test |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | logistic | 0.983 | 0.954 | 0.004 | 14074/3015/3017 |
| baseline | rf | 0.984 | 0.958 | 0.005 | 14074/3015/3017 |
| baseline | gbdt | 0.987 | 0.960 | 0.004 | 14074/3015/3017 |
| biased | logistic | 0.980 | 0.939 | 0.012 | 13999/3000/3001 |
| biased | rf | 0.980 | 0.939 | 0.021 | 13999/3000/3001 |
| biased | gbdt | 0.984 | 0.941 | 0.006 | 13999/3000/3001 |

## Biased Model on Baseline Test Set

To quantify performance degradation, the biased-trained models are evaluated on the baseline test set. We first restrict the baseline test cohort to patient IDs that also appear in the biased dataset, then remove any IDs that were seen during biased train/validation to avoid repeats. This ensures the biased models are evaluated only on unseen baseline patients.

Baseline test set filtered to overlap with biased dataset IDs, then removed any IDs seen in biased train/val splits to avoid repeats. Baseline test N=3,017; overlap N=3,008; removed-not-in-biased N=9; removed-seen-in-biased N=2,554.

| Model | Baseline AUC | Biased AUC | Δ AUC | Baseline AP | Biased AP | Δ AP | Baseline Brier | Biased Brier | Δ Brier | Test N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.981 | 0.981 | 0.000 | 0.968 | 0.970 | 0.002 | 0.004 | 0.012 | 0.007 | 454 |
| rf | 0.976 | 0.977 | 0.001 | 0.952 | 0.950 | -0.002 | 0.005 | 0.021 | 0.016 | 454 |
| gbdt | 0.985 | 0.980 | -0.005 | 0.970 | 0.961 | -0.009 | 0.004 | 0.004 | 0.000 | 454 |

## Performance Degradation Interpretation

Training on the biased dataset (with 30% under-documentation of hyperglycemia/hypertriglyceridemia) affects performance when evaluated on the baseline test cohort:

- logistic: AUC +0.000, AP +0.002, Brier +0.007 (negative AUC/AP and positive Brier indicate worse performance).
- rf: AUC +0.001, AP -0.002, Brier +0.016 (negative AUC/AP and positive Brier indicate worse performance).
- gbdt: AUC -0.005, AP -0.009, Brier +0.000 (negative AUC/AP and positive Brier indicate worse performance).

This gap suggests documentation bias in the training data affects model generalization, particularly because hyperglycemia and hypertriglyceridemia are highly predictive of diabetes.

## Selected Hyperparameters

### baseline

- **logistic**: {'model__C': 0.1, 'model__class_weight': None}
- **rf**: {'model__n_estimators': 500, 'model__max_depth': None, 'model__min_samples_leaf': 5, 'model__class_weight': 'balanced'}
- **gbdt**: {'model__n_estimators': 100, 'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__min_samples_leaf': 5}

### biased

- **logistic**: {'model__C': 1.0, 'model__class_weight': 'balanced'}
- **rf**: {'model__n_estimators': 500, 'model__max_depth': 5, 'model__min_samples_leaf': 5, 'model__class_weight': 'balanced'}
- **gbdt**: {'model__n_estimators': 100, 'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__min_samples_leaf': 10}
