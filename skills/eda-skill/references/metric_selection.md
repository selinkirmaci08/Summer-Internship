# Evaluation Metric Selection

Situation-to-metric tables for every task type EDA step 4 can encounter. The tables are the lookup; the reasoning about *this* dataset's goal, segments and row count stays in the skill.

Used by: `eda.md` step 4.

---

**Binary classification**

| Situation | Metric | Why |
|---|---|---|
| Classes are balanced and every error type costs the same | Accuracy | Simple and interpretable, but only trustworthy here — misleading under imbalance (a 99%-negative dataset scores 99% by always predicting negative) |
| False positives are the costly error (e.g. flagging a legitimate email as spam) | Precision | Minimizes wrongly-flagged positives |
| False negatives are the costly/dangerous error (e.g. missing a sick patient) | Recall | Minimizes missed true positives |
| Both precision and recall matter, especially under imbalance | F1 | Harmonic mean — penalizes a large gap between precision and recall rather than averaging it away |
| Need threshold-independent ranking ability across the whole range | ROC-AUC | Measures separability regardless of the eventual decision threshold |
| Severe class imbalance (minority/positive class is rare) | Average precision / PR-AUC | ROC-AUC can look misleadingly good under severe imbalance because it's dominated by the (easy) majority class; PR-AUC focuses specifically on positive-class performance |
| Want one robust number even under strong imbalance | MCC (Matthews correlation coefficient) | Uses all four confusion-matrix quadrants; doesn't inflate with imbalance the way accuracy does |
| The predicted probabilities themselves matter, not just the final decision | Log loss | Penalizes confident-but-wrong predictions and rewards well-calibrated probabilities |

**Multiclass classification** — same considerations as binary, extended with an averaging
strategy, plus:

| Situation | Metric | Why |
|---|---|---|
| Imbalanced classes, want to avoid a majority class dominating the score | Balanced accuracy | Macro-average of per-class recall — for balanced data it equals plain accuracy, but doesn't inflate under imbalance |

**Regression**

| Situation | Metric | Why |
|---|---|---|
| Want all errors weighted equally, robust to outliers | MAE | Doesn't square errors, so a few large misses don't dominate the score |
| Large errors are much worse than small ones | MSE / RMSE | Squaring amplifies large errors; RMSE keeps the same units as the target, which is more interpretable than raw MSE |
| Need to express error in relative/percentage terms | MAPE | Useful when the target's scale varies a lot, or stakeholders think in relative rather than absolute terms |
| Want an overall "how well does the model explain the variance" summary | R² | Standard goodness-of-fit measure, but can mislead with nonlinear relationships or when predicting outside the training range |
| Target is a non-negative count (e.g. demand, event counts) | Poisson deviance | Matches the count-data assumption better than squared error, especially for skewed count distributions |
| Target is strictly positive and right-skewed (e.g. revenue, duration) | Gamma deviance | Better suited to strictly-positive, multiplicative-feeling targets than MSE |

**Time series / forecasting** — the regression metrics above still apply, plus two that are
specific to forecasting:

| Situation | Metric | Why |
|---|---|---|
| Need an error metric comparable across series of different scale | SMAPE (symmetric MAPE) | Normalizes by the average of actual and predicted values, avoiding plain MAPE's blow-up when actuals are near zero |
| Want to judge the model against a naive baseline (e.g. "last value repeated") | MASE (mean absolute scaled error) | Expresses error relative to a naive forecast's error — a value under 1 means the model beats the naive baseline, which plain MAE/RMSE don't tell you on their own |

**Ordinal regression**

| Situation | Metric | Why |
|---|---|---|
| Categories have a meaningful order (e.g. severity 1–5) and being off by one category is a smaller error than being off by three | Weighted Cohen's kappa (linear or quadratic weights) | Accounts for the *distance* between predicted and actual category, unlike accuracy or unweighted kappa, which treat every misclassification as equally wrong |
| Want a simpler baseline to start with | MAE treating categories as their ordinal integer codes | Cruder than weighted kappa but easy to compute and interpret as "average categories off by" |

**Multi-label classification**

| Situation | Metric | Why |
|---|---|---|
| Every label matters equally regardless of how common it is | Macro-averaged F1 across labels | Each label's F1 contributes equally to the average, so a rare label isn't drowned out by common ones |
| Overall, frequency-weighted correctness across labels is what the goal cares about | Micro-averaged F1 across labels | Aggregates true/false positives across all labels before computing F1, so common labels naturally dominate the score |
| Need to check whether the *entire* label set was predicted exactly right, not just individual labels | Subset accuracy (exact match ratio) | Stricter than per-label metrics — useful when partial credit for getting some labels right isn't meaningful for the goal |
