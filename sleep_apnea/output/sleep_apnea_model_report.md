# Sleep Apnea Diagnosis Model Report

Train/val/test split: 0.70/0.15/0.15

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

## Selected Hyperparameters

### baseline

- **logistic**: {'model__C': 0.1, 'model__class_weight': None}
- **rf**: {'model__n_estimators': 200, 'model__max_depth': 5, 'model__min_samples_leaf': 5, 'model__class_weight': None}
- **gbdt**: {'model__n_estimators': 100, 'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__min_samples_leaf': 5}

### biased

- **logistic**: {'model__C': 0.1, 'model__class_weight': 'balanced'}
- **rf**: {'model__n_estimators': 200, 'model__max_depth': 5, 'model__min_samples_leaf': 1, 'model__class_weight': 'balanced'}
- **gbdt**: {'model__n_estimators': 100, 'model__learning_rate': 0.05, 'model__max_depth': 2, 'model__min_samples_leaf': 5}
