# Model Selection — Supervised Learning

Source: [scikit-learn Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html) and linked pages. Scope: classical ML (linear models, SVM, nearest neighbors, trees, ensembles, naive Bayes, SGD, discriminant analysis). Neural networks (MLP) are out of scope here — flag them only if the earlier pipeline stages explicitly call for deep learning.

Preprocessing requirements a model implies (scaling, encoding) are noted inline as flags but documented in full in `../../shared-references/preprocessing_and_feature_engineering.md` — that file is owned by/reusable from the EDA/preprocessing/feature-engineering skill, not duplicated here.

## 1. Quick decision table — condition → recommended family

**General principle (bias-variance framing):** for small training sets, prefer high-bias/low-variance models (Naive Bayes, linear/logistic regression, LDA) over low-bias/high-variance ones (deep trees, KNN with small k, high-degree/RBF SVM) — the latter need more data to avoid overfitting. This is a coarse heuristic, not a substitute for the row-by-row table below. *(Source: towardsdatascience.com algorithm-selection guide.)*

Read top to bottom; multiple rows can apply, in which case shortlist 2–3 families and let hyperparameter tuning + CV decide between them.

| Condition | Recommended family | Notes |
|---|---|---|
| Need a fast, interpretable baseline | Linear/Logistic Regression | Always fit one first, even if you expect to beat it |
| Need probability estimates that are well-calibrated | Logistic Regression | Native `predict_proba`; SVM/tree probs need extra calibration |
| Multicollinear features | Ridge or Elastic-Net | OLS/Lasso become unstable under correlated features |
| Need automatic feature selection (sparse coefficients) | Lasso or Elastic-Net | Elastic-Net is more stable when features are correlated |
| High-dimensional, n_features ≫ n_samples | SVM (linear kernel), Lasso, LinearSVC | SVMs and L1 models handle this natively |
| Non-linear decision boundary, small-to-medium data | SVM with RBF kernel | Needs feature scaling; tune `C` and `gamma` together |
| Non-linear boundary, tabular data, no scaling desired | Decision Tree / Ensemble (RF, Gradient Boosting) | Trees are scale-invariant and need minimal preprocessing |
| Large dataset (>10,000 rows), need best accuracy | HistGradientBoosting | Orders of magnitude faster than plain GradientBoosting at this scale, handles missing values & categoricals natively |
| Small-to-medium dataset, need best accuracy | GradientBoostingClassifier/Regressor | More accurate than RF on small data, more tuning required |
| Need a fast, robust baseline ensemble with little tuning | Random Forest | Good default; parallelizable; gives feature importances |
| Need feature importances but small/limited data | Random Forest or single Decision Tree | Tree gives max interpretability, RF gives more stable importances |
| Very large n_samples (streaming/online, memory constrained) | SGDClassifier/Regressor | Linear models fit incrementally with `partial_fit` |
| Extremely large n_samples & n_features, need speed | LinearSVC | Near-linear scaling vs. libsvm's SVC |
| Text / high-dimensional sparse count data | Multinomial or Complement Naive Bayes | Fast, strong baseline for text classification |
| Text classification, <100K samples, want non-linear separation | Kernel Approximation (Nystroem/RBFSampler) + linear model | scikit-learn's own algorithm cheat-sheet routes text data here specifically — see §11 |
| Imbalanced text/count classification | Complement Naive Bayes | Designed to correct MultinomialNB's bias on imbalanced data |
| Very little training data available | Naive Bayes, LDA | Both need few samples to estimate parameters |
| Need uncertainty/confidence intervals on predictions | Bayesian Ridge, ARD, Gaussian Process | GP is best for small data where uncertainty quantification matters most (not deep-dived here — see scikit-learn `gaussian_process.html` if needed) |
| Data has significant outliers | RANSAC, Theil-Sen, or HuberRegressor (regression); tree ensembles (classification) | RANSAC for large y-outliers, Theil-Sen for x-direction outliers, Huber as a fast default |
| Irregular/local decision boundary, small-moderate data | k-Nearest Neighbors | No training phase; prediction cost scales with data size |
| Multi-class with small samples relative to features | LDA | Closed-form, no hyperparameters, handles small-sample regimes better than QDA |
| Multi-class, classes have different covariance structure, ample data | QDA | More flexible than LDA but needs more data per class |
| Count / non-negative / skewed targets (regression) | Poisson / Gamma / Tweedie Regressor | Matches GLM family to the target's distribution instead of forcing OLS |
| Need to combine several already-good models | Voting (quick) or Stacking (max accuracy) | Voting = simple average/majority; Stacking = learns optimal combination via meta-learner |

## 2. Linear models

| Model | Problem type | Key hyperparameters | Prefer when | Avoid / limitations |
|---|---|---|---|---|
| OLS (`LinearRegression`) | Regression | `positive` | Small/medium data, low multicollinearity, need raw interpretable coefficients | Unstable under multicollinearity |
| Ridge | Regression/Classification | `alpha`, `solver` | Multicollinear features; want stability, keep all features | Doesn't zero out features (no selection) |
| Lasso | Regression | `alpha`, `selection` | Need sparse solution / feature selection, high-dimensional data | Unstable pick among correlated features |
| Elastic-Net | Regression | `alpha`, `l1_ratio` | Correlated feature groups + need sparsity | Two hyperparameters to tune instead of one |
| Logistic Regression | Classification | `C`, `penalty`, `solver`, `l1_ratio` | Default classification baseline; need probabilities | Solver choice matters at scale (see below) |
| Multi-Task Lasso/ElasticNet | Multi-output regression | `alpha`, `l1_ratio` | Multiple related regression targets sharing feature relevance | Only for 2D `Y` |
| LARS / LARS-Lasso | Regression | — | n_features ≫ n_samples, need full solution path | Sensitive to noise in high dimensions |
| Bayesian Ridge / ARD | Regression | (auto-tuned) | Need uncertainty estimates + automatic regularization | Slower than plain Ridge |
| Poisson / Gamma / Tweedie Regressor | Regression | `power`, `link`, `alpha` | Count data, positive skewed data, insurance-style compound distributions | Wrong choice if target is roughly Gaussian (use OLS/Ridge instead) |
| RANSAC / Theil-Sen / Huber | Regression | varies | Outliers present | Slower (Theil-Sen), or less robust to extreme cases (Huber) |

**Logistic Regression solver selection:**

| Scenario | Recommended solver |
|---|---|
| Default / robust choice | `lbfgs` |
| n_samples ≫ n_features, need high precision | `newton-cholesky` |
| Large datasets | `sag` or `saga` |
| Need sparse (L1) solution | `saga` or `liblinear` |

## 3. Support Vector Machines

| Aspect | Guidance |
|---|---|
| Classification | `SVC`/`NuSVC` (kernelized), `LinearSVC` (fast, linear-only) |
| Regression | `SVR`/`NuSVR` (kernelized), `LinearSVR` (fast, linear-only) |
| Key hyperparameters | `C` (regularization, inverse), `kernel`, `gamma` (RBF/poly/sigmoid), `epsilon` (SVR margin) |
| Good fit when | High-dimensional data (n_features ≫ n_samples), need max-margin classifier, small-medium datasets |
| Avoid / caution | `predict_proba` is expensive (5-fold internal CV via Platt scaling); libsvm-backed classes scale poorly (O(n²)–O(n³)) beyond tens of thousands of rows — use `LinearSVC` or `SGDClassifier` instead at that scale. Needs scaled input features — see `../../shared-references/preprocessing_and_feature_engineering.md` §1 |
| Imbalanced classes | `class_weight='balanced'` (SVC only, not NuSVC) |
| Too much data for a full kernel, but still need non-linear separation | See §11, Kernel Approximation |

## 4. Nearest Neighbors

| Aspect | Guidance |
|---|---|
| Key hyperparameters | `n_neighbors`, `weights` (`uniform`/`distance`), `algorithm` (`auto`/`kd_tree`/`ball_tree`/`brute`), `metric` |
| Good fit when | Small-moderate datasets, irregular/local decision boundaries, quick baseline |
| Avoid / caution | Curse of dimensionality (D > ~15–20 degrades performance sharply); high prediction-time cost since it's a lazy learner (no real "training") |
| Mitigations | Use tree-based `algorithm` for larger N. Needs scaled input features and benefits from dimensionality reduction beforehand in high dimensions — see `../../shared-references/preprocessing_and_feature_engineering.md` §§1, 3 |

## 5. Decision Trees

| Aspect | Guidance |
|---|---|
| Key hyperparameters | `max_depth` (start ~3, increase gradually), `min_samples_split`, `min_samples_leaf` (try 5), `criterion` (`gini`/`entropy` for classification, `squared_error`/`absolute_error`/`poisson` for regression) |
| Strengths | Interpretable, minimal preprocessing, handles mixed/non-linear data natively, fast prediction O(log n) |
| Weaknesses | Prone to overfitting without depth/leaf limits; unstable (small data changes → different tree); biased toward dominant classes in imbalanced data |
| Single tree vs. ensemble | Use a single tree only when interpretability is the priority or the dataset is small; use an ensemble whenever accuracy/stability matters more than reading the tree directly |

## 6. Ensemble methods

| Method | Dataset size | Speed | Accuracy | Interpretability | Best for |
|---|---|---|---|---|---|
| Random Forest | Medium–large | Medium | High | Moderate (feature importances) | Strong default baseline, balanced needs |
| Extra Trees | Medium–large | Fast | Medium-high | Moderate | Slightly faster than RF, similar accuracy |
| Gradient Boosting | Small–medium | Slow | Highest (small data) | Lower | Max accuracy when data is small/medium |
| HistGradientBoosting | Large (>10k rows) | Fastest | Very high | Lower | Large tabular data, native missing-value & categorical support |
| AdaBoost | Small–medium | Medium | Medium | Moderate | Simple boosting with weak learners, binary classification |
| Bagging | Any | Medium | Medium | Moderate | Reducing variance of an overfitting base estimator (e.g. KNN, deep tree) |
| Voting | Any | Fast | Medium-high | Low | Quickly combining several already-trained diverse models |
| Stacking | Medium | Slow | Highest (with diverse base models) | Low | Squeezing out max accuracy by learning how to combine base models |

**Key ensemble hyperparameters:** `n_estimators`, `max_features`, `max_depth`, `learning_rate` (boosting only, typically 0.01–0.1 — lower needs more estimators), `subsample` (stochastic boosting, typically 0.5–0.8).

**Rule of thumb:** start with Random Forest as a fast baseline → move to HistGradientBoosting (large data) or GradientBoosting (small/medium data) for higher accuracy → consider Stacking/Voting only once you already have several good, diverse models to combine.

## 7. Naive Bayes

| Variant | Data type | Notes |
|---|---|---|
| GaussianNB | Continuous features | No hyperparameters; assumes per-class Gaussian likelihood |
| MultinomialNB | Counts/frequencies (text) | `alpha` smoothing; standard text-classification baseline |
| BernoulliNB | Binary features | `binarize` threshold; can beat MultinomialNB on short documents |
| ComplementNB | Counts, imbalanced text | Corrects MultinomialNB's bias on imbalanced classes |
| CategoricalNB | Nominal/categorical features | `alpha`; requires integer-encoded categories — see `../../shared-references/preprocessing_and_feature_engineering.md` §2 |

**Good baseline when:** text/high-dimensional sparse data, very little training data, need fast predictions. **Avoid when:** features are strongly correlated (violates independence assumption), or calibrated probabilities are the actual goal (`predict_proba` is known to be poorly calibrated). All variants support `partial_fit` for out-of-core/streaming learning.

## 8. SGD-based linear models

| Aspect | Guidance |
|---|---|
| Use when | Very large datasets (>10k rows), online/streaming data, memory-constrained environments |
| Roughly equivalent to | `SGDClassifier(loss='log_loss')` ≈ `LogisticRegression`; `SGDRegressor(loss='squared_error', penalty='l2')` ≈ `Ridge` — just a different (scalable) solver |
| Key hyperparameters | `loss` (`hinge`=linear SVM, `log_loss`=logistic, `modified_huber`, `squared_error`, `huber`, `epsilon_insensitive`), `penalty` (`l1`/`l2`/`elasticnet`), `alpha`, `learning_rate` schedule |
| Mandatory practice | Needs scaled input features (highly sensitive to feature magnitude) — see `../../shared-references/preprocessing_and_feature_engineering.md` §1 |
| Other tips | Shuffle data each epoch (default); use `early_stopping=True` with `validation_fraction`; expect convergence around ~10⁶ samples observed total |

## 9. Discriminant Analysis (LDA / QDA)

| Aspect | LDA | QDA |
|---|---|---|
| Decision surface | Linear | Quadratic |
| Covariance assumption | Shared across classes | Per-class |
| Data needed | Works well with small samples | Needs more samples per class |
| Extra capability | Can also do supervised dimensionality reduction (project onto ≤ K−1 dims) | Classification only |
| Hyperparameters | None required; optional `shrinkage='auto'` for small-sample covariance regularization | None |

Relation to Naive Bayes: QDA with a diagonal covariance matrix reduces to Gaussian Naive Bayes.

## 10. Gaussian Processes (brief mention, not deep-dived)

Good choice specifically when the dataset is small-to-medium **and** calibrated uncertainty estimates on predictions matter more than raw scale. Not covered in depth in this pass — pull scikit-learn's `gaussian_process.html` if a task actually calls for it.

## 11. Kernel Approximation

*(Source: [scikit-learn Choosing the right estimator](https://scikit-learn.org/stable/machine_learning_map.html).)*

When a problem needs a non-linear decision boundary but the dataset is too large for a full-kernel SVM to be practical (see §3's O(n²)–O(n³) scaling caution), scikit-learn's own flowchart routes to **kernel approximation** instead: methods like `Nystroem` or `RBFSampler` approximate a non-linear kernel's feature mapping explicitly, so a fast linear model (`SGDClassifier`, `LinearSVC`) can be used in place of a full kernelized SVM. This isn't a model family so much as a preprocessing trick that lets a linear model behave non-linearly at scale — worth reaching for specifically in the gap between "SVM with RBF kernel would be ideal" and "the dataset is too big for that to be practical."

**Resolved note — SGD size threshold:** two scikit-learn sources gave different cutoffs for when SGD-based linear models become preferred (`sgd.html` said >10,000 samples; the [algorithm cheat-sheet](https://scikit-learn.org/stable/machine_learning_map.html) used >100,000 samples for its regression branch). Standardized on **>10,000 samples** throughout this file (see §8) — decided 2026-08-21.
