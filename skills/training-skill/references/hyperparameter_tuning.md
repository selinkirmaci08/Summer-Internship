# Hyperparameter Tuning Methods

Source: [scikit-learn Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html).

Pair this with `cross_validation.md` (how folds are built) and `evaluation_metrics.md` (what score to optimize) — a search method alone doesn't determine either of those.

## 1. Method comparison

| Method | Exhaustiveness | Speed | Best for | Budget control | API stability |
|---|---|---|---|---|---|
| GridSearchCV | Complete (every combination) | Slow | Small, discrete parameter spaces | None (grows combinatorially) | Stable |
| RandomizedSearchCV | Samples n_iter candidates | Moderate | Large / continuous parameter spaces | `n_iter` | Stable |
| HalvingGridSearchCV | Complete grid, tournament elimination | Fast | Budget-constrained, discrete spaces | `factor`, `min_resources` | Experimental |
| HalvingRandomSearchCV | Random sampling, tournament elimination | Fast | Budget-constrained, continuous spaces | `factor`, `min_resources` | Experimental |
| Model-specific CV (`*CV` estimators) | Full regularization path | Very fast | Tuning a single regularization parameter only | None | Stable |
| Out-of-Bag (OOB) estimates | Single fit | Very fast | Bagging-based ensembles only (RF, Extra Trees, GB) | None | Stable |
| Bayesian Optimization *(external library — not in scikit-learn core)* | Sequential, model-based sampling | Fast convergence per candidate evaluated | Expensive-to-fit models (large ensembles, big data) where each trial's cost matters more than exhaustiveness | Configurable (n_trials) | Depends on library (Optuna, scikit-optimize, Hyperopt) |

## 2. When to prefer each

- **GridSearchCV** — parameter space is small and discrete, and you want to be exhaustive (e.g. final confirmation pass after narrowing the range with a randomized search).
- **RandomizedSearchCV** — default choice for most real tuning jobs. Handles continuous distributions (`scipy.stats.expon`, `loguniform`), and increasing `n_iter` always refines the search without exploding runtime the way grid search does.
- **Successive halving (`Halving*SearchCV`)** — when compute budget is the binding constraint and poor candidates can be identified early on a small resource allocation (fewer samples/trees/iterations), promoting only the survivors to larger allocations. Still experimental in scikit-learn — treat results as directional if reproducibility across versions matters.
- **Model-specific CV estimators** (`RidgeCV`, `LassoCV`, `ElasticNetCV`, `LogisticRegressionCV`, `LassoLarsCV`, etc.) — always prefer these over a generic search when only a single model's own regularization parameter needs tuning; they compute the whole path far more cheaply than repeated `fit()` calls.
- **`LassoLarsIC`** — even cheaper than CV when an AIC/BIC-based single path computation is acceptable in place of cross-validation.
- **OOB estimates** — free generalization-error estimate for `RandomForest*`, `ExtraTrees*`, `GradientBoosting*` without a separate validation split; useful for quick sanity checks but not a substitute for CV during actual hyperparameter search across model families.
- **Bayesian Optimization** *(source: general model-selection guidance, e.g. thedatatwins medium article — not scikit-learn itself)* — builds a probabilistic surrogate model of the objective and picks the next candidate to evaluate based on expected improvement, rather than searching blindly (grid/random) or by elimination (halving). Worth reaching for when each model fit is genuinely expensive (large ensembles, large data, deep hyperparameter interactions) and a smarter-than-random search pays for itself. Requires an external library (`optuna`, `scikit-optimize`, `hyperopt`) since scikit-learn doesn't implement this natively — flag this dependency if the training skill's environment is scikit-learn-only.

## 3. Practical tips checklist

| Tip | Why |
|---|---|
| Set `scoring` explicitly for imbalanced data | Defaults (`accuracy`, `r2`) can be misleading — see `evaluation_metrics.md` |
| Use `<estimator>__<parameter>` syntax inside Pipelines | Needed to tune parameters nested inside `Pipeline`/`ColumnTransformer`/`VotingClassifier` |
| Keep a held-out evaluation set separate from the development set used for the search | Prevents the reported final score from being optimistic |
| Use `n_jobs=-1` | Parallelizes candidate evaluation |
| Set `error_score` deliberately (`"raise"` vs a numeric fallback) | Controls whether one failing candidate crashes the whole search |
| Convert `cv_results_` to a DataFrame and inspect it | Successive halving adds `iter`/`n_resources` columns worth checking for convergence |
| Read the estimator's docstring before building a large grid | Usually only a small subset of hyperparameters actually move the needle — tuning all of them wastes budget |
| Use `cross_validate` with a dict of scorers for multi-metric evaluation, and set `refit='<metric_name>'` | Needed whenever more than one metric matters for the final choice |

## 4. Typical hyperparameter ranges worth searching (starting points)

| Model | Parameters to search | Typical range |
|---|---|---|
| Ridge / Lasso / ElasticNet | `alpha` | log-spaced, e.g. `10.0 ** np.arange(-4, 4)` |
| ElasticNet | `l1_ratio` | `[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]` |
| Logistic Regression | `C` | log-spaced, e.g. `10.0 ** np.arange(-4, 4)` |
| SVM (RBF) | `C`, `gamma` | both log-spaced, e.g. `np.logspace(-3, 3, 7)` |
| KNN | `n_neighbors` | odd values, e.g. `3–31` |
| Decision Tree | `max_depth`, `min_samples_leaf` | `max_depth`: `3–20` or `None`; `min_samples_leaf`: `1–20` |
| Random Forest / Extra Trees | `n_estimators`, `max_features`, `max_depth` | `n_estimators`: `100–1000`; `max_features`: `"sqrt"`, `"log2"`, or a fraction |
| Gradient Boosting / HistGradientBoosting | `learning_rate`, `n_estimators`/`max_iter`, `max_depth`, `subsample` | `learning_rate`: `0.01–0.3`; shallow `max_depth` (2–8) |
| SGD | `alpha`, `learning_rate`, `eta0` | `alpha`: log-spaced `10.0**-np.arange(1,7)` |

## 5. Time series-specific tuning

*(Source: [Fine-Tuning for Time Series Analysis](https://www.meegle.com/en_us/topics/fine-tuning/fine-tuning-for-time-series-analysis). See `model_selection_timeseries.md` for the models themselves.)*

| Aspect | Guidance |
|---|---|
| Validation strategy while tuning | Use **walk-forward validation** for tuning, not just for the final evaluation — this is the forecasting-literature name for the same `TimeSeriesSplit`/forward-chaining approach in `cross_validation.md`. Tuning against an ordinary random split (or worse, in-sample) will overstate performance the same way it would for final evaluation. |
| What to search for classical models | `(p,d,q)` for ARIMA, `(p,d,q)(P,D,Q)m` for SARIMA/SARIMAX — grid search or Bayesian optimization over these orders, same method comparison as §1 above applies |
| What to search for the ML-as-regression path | Same hyperparameter ranges as the underlying model in §4 above (e.g. gradient boosting ranges), plus the lag/rolling window sizes themselves as an additional tunable choice — window size is a feature-engineering decision, not just a model hyperparameter, so it needs to go through the same walk-forward-validated search rather than being fixed by guess |
| AutoML options | H2O.ai and Google AutoML are named as platforms that automate this tuning end-to-end — alongside Auto ARIMA/Prophet already noted in `model_selection_timeseries.md` §1 |
| Overfitting risk | Flagged as a real risk specifically because a time series model can look good on the training window and fail once the underlying process shifts — regularization (L1/L2, dropout) is the suggested fix, though dropout is only relevant if a deep-learning model is in play (out of scope per `model_selection_timeseries.md`) |
