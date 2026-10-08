# Evaluation Metrics & Fit Diagnostics

Source: [scikit-learn Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) and [Validation curves / learning curves](https://scikit-learn.org/stable/modules/learning_curve.html).

## 1. Decision framework

1. **If an external scoring function is already specified** (e.g. a competition metric, a business KPI), use that one — don't substitute a "better" metric.
2. **Otherwise, pick based on what you're actually trying to predict:**
   - Predicting the **mean** → MSE / Log Loss / Brier Score (all "strictly consistent" for the mean).
   - Predicting the **median** → MAE.
   - Predicting a **specific quantile** → Pinball loss at that quantile (`alpha`).
   - Making a **decision** (not just predicting a number) → confusion-matrix-derived metrics (precision, recall, F1, F-beta).

## 2. Classification metrics

| Condition | Recommended metric | Why |
|---|---|---|
| Balanced binary/multiclass | Accuracy | Simple, each class equally important |
| Imbalanced classes | Balanced Accuracy | Macro-average of per-class recall — avoids inflated scores from majority-class dominance |
| Minority class matters more | F1 score | Harmonic mean of precision & recall |
| Specific precision/recall tradeoff (known cost ratio) | F-beta score | `beta` weights recall vs. precision |
| Need well-calibrated probabilities | Log Loss | Strictly consistent, rewards confident-and-correct predictions, penalizes confident-and-wrong heavily |
| Need probability quality on binary target | Brier Score | Squared error on predicted probabilities |
| Threshold-invariant ranking quality | ROC AUC | Good general-purpose ranking metric, less informative under severe imbalance than PR-AUC |
| Multilabel, weight all sample-class pairs equally | Micro-averaged F1 | — |
| Multilabel, weight all classes equally regardless of frequency | Macro-averaged F1 | — |
| Multilabel, imbalanced classes | Weighted F1 | Accounts for class support |

## 3. Regression metrics

| Condition | Recommended metric | Why |
|---|---|---|
| Standard regression, no outliers | MSE / RMSE | Strictly consistent for the mean; penalizes large errors more |
| Standard regression, want interpretable 0–1 scale | R² | Same ranking as MSE, easier to communicate |
| Data has outliers | MAE | Robust; strictly consistent for the median instead of the mean |
| Need a specific quantile prediction | Pinball loss (`alpha` = target quantile) | Strictly consistent for quantiles |
| Log-scale / multiplicative-error targets | Mean Squared Log Error | Handles targets spanning orders of magnitude |
| Count data (Poisson-distributed) | Mean Poisson Deviance | Matches the GLM family of the target |
| Strictly positive, skewed data (Gamma-distributed) | Mean Gamma Deviance | — |
| Compound/insurance-style distributions | Mean Tweedie Deviance (`power` parameter) | Flexible across distribution families |

## 4. Time series / forecasting metrics

*(Sources: [Time Series Forecasting Part 2: Selecting Algorithms](https://medium.com/data-science-at-microsoft/time-series-forecasting-part-2-of-3-selecting-algorithms-11b6635f61bb), [Training a Time-Series Forecasting Model Using Automated Machine Learning](https://techcommunity.microsoft.com/blog/educatordeveloperblog/training-a-time-series-forecasting-model-using-automated-machine-learning/4212384). See `model_selection_timeseries.md` for the modeling side.)*

| Condition | Recommended metric | Why |
|---|---|---|
| Want all errors weighted equally, including large ones from anomalies | MAE | Treats every error the same regardless of magnitude; robust, less sensitive to outliers |
| Anomaly/outlier detection matters, want to penalize large misses more | MSE / RMSE | Squaring amplifies large errors — a low MSE/RMSE indicates accurate predictions, but both are sensitive to outliers |
| Want a percentage-based error, scale-independent | MAPE | Mean absolute percentage difference between forecast and actual — intuitive, but penalizes over- and under-forecasts asymmetrically and blows up when actuals are near zero |
| Comparing forecast accuracy across series of different scales, want to avoid MAPE's asymmetry | SMAPE | Symmetric percentage error, bounded 0–200% — generally preferred over plain MAPE for this reason |
| Want a baseline-relative measure, or actuals are near zero (breaks percentage-based metrics) | MASE | Scaled by a naive seasonal baseline's own error rather than by the actual values — see formula below |

**Pipeline rule:** on any time-series task this pipeline computes **MAE, RMSE, SMAPE and MASE**
every Training call, on train and validation, regardless of which is primary — see `../training.md`
step 7. The table above governs which one is *primary* and which further metrics to add, not
whether these four get computed. SMAPE's zero-actual instability is handled by an explicitly
recorded convention, not by dropping the metric.

**MASE (Mean Absolute Scaled Error)** *(added from the user's production training-skill convention — closes a prior gap)*: preferred over MAPE/SMAPE specifically because it compares against a naive seasonal baseline rather than a percentage of the actual value, so it doesn't blow up when actuals are near zero.

- **Formula:** `MASE = MAE(model) / MAE(seasonal naive baseline)`, where the seasonal naive baseline predicts `ŷ_t = y_{t-P}` for seasonal period `P` (e.g. lag-7 for weekly-seasonal daily data).
- **Procedure:** compute the naive baseline's MAE on the **training set only**, then use that as a fixed denominator to normalize both train and validation MAE. Never recompute the denominator on validation data — doing so would let future values leak into what counts as a "good" score on past data (the same leakage principle as fitting a scaler on validation data).
- **Interpretation:** MASE < 1.0 means the model beats the seasonal naive baseline; MASE ≥ 1.0 means it doesn't — a useful sanity floor before trusting any more complex model.

## 5. Clustering metrics

See `model_selection_unsupervised.md` §1.3 for the full table (Rand Index/AMI/V-measure when ground truth is available; Silhouette/Davies-Bouldin/Calinski-Harabasz when it isn't).

## 6. Validation curves and learning curves — diagnosing fit quality

Run these **before** finalizing a model — they're cheap and catch problems that a single CV score can hide.

### Validation curve (score vs. one hyperparameter value)

| Training score | Validation score | Diagnosis |
|---|---|---|
| Low | Low | Underfitting — model too simple / too much regularization for this hyperparameter value |
| High | Low | Overfitting — model too complex / too little regularization for this hyperparameter value |
| High | High | Good fit at this hyperparameter value |

### Learning curve (score vs. training set size)

| Pattern | Diagnosis | Action |
|---|---|---|
| Both training and validation scores low and converged | High bias (underfitting) | More data won't help much — need a more expressive model or better features instead |
| Large, persistent gap between training and validation scores | High variance (overfitting) | More training data (or regularization, or a simpler model) should help |
| Both scores converge at a high level | Good fit | Model and data size are well matched |

**Practical use:** validation curves tell you *which hyperparameter values* cause over/underfitting (feeding directly into the ranges in `hyperparameter_tuning.md` §4); learning curves tell you whether *collecting more data* is worth it before spending more effort on model/hyperparameter changes.
