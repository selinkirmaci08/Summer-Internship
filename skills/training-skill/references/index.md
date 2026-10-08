# Training Skill — Reference Index

Source: scikit-learn documentation (`model_selection.html`, `supervised_learning.html`, `unsupervised_learning.html`, and their linked sub-pages). Scope: **classical ML only** (no deep learning).

These files assume the previous pipeline stage (EDA / preprocessing / feature engineering) has already handled scaling, encoding, dimensionality reduction, and feature selection. Where a model choice *implies* one of those requirements (e.g. "SVM needs scaled features"), it's flagged inline with a pointer rather than explained in full — the full reusable reference lives outside this skill at `../../shared-references/preprocessing_and_feature_engineering.md`. Check the previous stage's output against that file before training if there's any doubt it was actually done.

Use this index to decide which reference file to open. Don't load every file for every job — jump to the one that matches the decision you're making right now.

| I need to... | Open |
|---|---|
| Pick a model family for a supervised (labeled) problem | `model_selection_supervised.md` |
| Pick a clustering algorithm or a dimensionality-reduction method | `model_selection_unsupervised.md` |
| Pick a time series forecasting approach | `model_selection_timeseries.md` |
| Decide how to split data for validation (k-fold, grouped, time series, etc.) | `cross_validation.md` |
| Decide how to search hyperparameters (grid vs random vs successive halving) | `hyperparameter_tuning.md` |
| Decide which metric to optimize/report, or diagnose over/underfitting | `evaluation_metrics.md` |
| Decide which diagnostic plots to generate, and when | `plotting_guide.md` |
| Report feature importance when features are correlated (grouping, masking, why split-count is not importance) | `../../shared-references/feature_attribution.md` |

## Sanity gate before modeling

Per scikit-learn's own [algorithm cheat-sheet](https://scikit-learn.org/stable/machine_learning_map.html): if the dataset has **fewer than ~50 samples**, its guidance is to get more data before attempting to model at all — every branch of the flowchart assumes at least that much. Treat this as a hard stop worth surfacing back to the pipeline rather than silently training on it.

## Time series

Time series forecasting is covered separately — start at `model_selection_timeseries.md`, with supporting content in `cross_validation.md` (TimeSeriesSplit), `evaluation_metrics.md` §4 (forecasting metrics, including MASE), and `../../shared-references/preprocessing_and_feature_engineering.md` §4 (lag/rolling/calendar features, stationarity, decomposition). The AR/ARIMA seasonality framing question has been resolved in favor of the machinelearningmastery cheat sheet (see `model_selection_timeseries.md`).

## Recommended order of operations for the training skill

1. **Frame the problem.** From the previous pipeline stage's output, determine: supervised or unsupervised, classification/regression/clustering/dim-reduction, dataset size (rows/cols), presence of groups/time structure, class balance, and any hard constraints (interpretability, latency, memory).
2. **Shortlist model candidates** — `model_selection_supervised.md` or `model_selection_unsupervised.md`.
3. **Choose a validation scheme** for that data — `cross_validation.md`. Do this before tuning; the wrong CV scheme (e.g. plain K-Fold on grouped or time-ordered data) invalidates every score that follows.
4. **Choose the scoring metric(s)** to optimize and report — `evaluation_metrics.md`.
5. **Tune hyperparameters** for the shortlisted candidates using the chosen CV scheme and metric — `hyperparameter_tuning.md`.
6. **Diagnose fit quality** with validation/learning curves (also in `evaluation_metrics.md`) before finalizing — cheap to run, catches over/underfitting early.
7. **Train the final model** on the winning configuration and hand off the fitted estimator + metrics + chosen CV/metric rationale to the next stage.
8. **Generate diagnostic plots** per `plotting_guide.md` — a fixed floor of always-generate plots plus condition-triggered ones (classification-specific, iterative-model training curves, time-series-specific, etc.), each with a written insight, not just an image.