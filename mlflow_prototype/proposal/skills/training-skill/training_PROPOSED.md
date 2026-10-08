---
name: training-agent-skill
description: Trains a model on the feature-engineered dataset, evaluates it on the validation set, and produces the artifacts (trained model, metrics, feature importance, residuals) that Diagnosis needs to decide whether to loop back or accept the result. Never touches the test set — that is end_model's job.
---

# Training Skill

## Purpose

Given the current feature set and a model family recommendation from context, fit a model on the
training split, evaluate it on the validation split, and surface the diagnostics Diagnosis needs
to reason about: metrics, feature importance, and residual patterns. This is the step that
produces a concrete number for the Planner and a concrete residual signal for Diagnosis — not the
final answer (that is end_model) but the best current answer given the feature set at hand.

## Reference material

This skill has a `references/` directory (general ML model-selection/tuning/evaluation knowledge,
built from scikit-learn docs and vetted blog sources) and shares a `shared-references/` directory
with Preprocess/FE (model-specific data-readiness requirements). Consult these instead of
re-deriving general ML knowledge inline — the procedure below points to the relevant section at
each step:

- `references/index.md` — routing index for everything below
- `references/model_selection_supervised.md` — full classical-ML decision framework (linear, SVM,
  KNN, trees, ensembles, Naive Bayes, SGD, LDA/QDA, kernel approximation) — broader than the
  narrowed GBT/RF/Linear table this skill uses in step 2b
- `references/model_selection_timeseries.md` — classical statistical forecasting (ARIMA/SARIMA/
  exponential smoothing family) vs. the ML-as-regression-on-lag-features path this skill defaults to
- `references/cross_validation.md` — CV strategy selection, including why time-series data needs
  `TimeSeriesSplit`/walk-forward validation instead of random k-fold
- `references/hyperparameter_tuning.md` — search method comparison (grid/random/successive
  halving/Bayesian optimization), typical hyperparameter ranges, and (§5) time-series-specific
  tuning notes
- `references/evaluation_metrics.md` — metric-selection framework (including MASE) and
  validation/learning-curve interpretation for diagnosing over/underfitting
- `references/plotting_guide.md` — condition → plot table covering when to generate each
  diagnostic beyond the fixed floor in step 14, with the insight/interpretation each one needs
- `references/mlflow_logging.md` — how to mirror this iteration's params/metrics/artifacts into a
  shared MLflow tracking store for cross-run comparison (step 14a); optional and never blocking —
  see Guardrails
- `../shared-references/preprocessing_and_feature_engineering.md` — model-specific scaling,
  encoding, and missing-value requirements, owned jointly with Preprocess; consult before
  assuming a feature is training-ready

## Inputs

- `dataset_path` — the feature-engineered dataset. Must contain a `_split` column with values
  `train`, `validation`, and `test`. **Training only ever reads `train` and `validation` rows.**
- `target_column` — the column to predict.
- `task_type` — one of `regression`, `binary_classification`, or `multiclass_classification`.
  Should be established by EDA and carried forward; Training does not re-derive it.
- `feature_subset` — the specific feature columns to use this call. Must not include
  `target_column`, `date`/timestamp columns, or `_split`. When the FE prediction-time
  availability audit flagged concurrent features, `feature_subset` should reflect whichever
  deployment scenario applies (see step 4).
- `deployment_scenario` — one of:
  - `retrospective` — the model explains past demand using all available features including
    same-period concurrent measurements. Concurrent features may be included in `feature_subset`.
  - `live_forecast` — the model will be used to predict a future period before it occurs.
    Concurrent features must be excluded from `feature_subset`; only `pre_period` features apply.
  - `unspecified` — the deployment scenario has not been confirmed. Training proceeds with the
    full feature set but records a `clarification_needed` entry.
- `context` — prior reasoning from any upstream node: EDA `recommended_strategy`, Diagnosis
  hypothesis, Training observation from a prior iteration, or a direct Planner instruction.
  Always record which node and iteration each piece of context came from.
- `mode` — one of:
  - `initial` — first Training call this run.
  - `incremental` — re-train with a changed feature set, different hyperparameters, or a
    Diagnosis-driven modification. Loads the prior model path from `state/current.json` for
    comparison.
  - `rebuild` — discard the prior model entirely and start fresh (e.g. model family change).
- `mode_reasoning` — why this mode was chosen.
- `prior_model_path` — path to the previously saved model (used in `incremental` mode for
  comparison; null in `initial` or `rebuild`).
- `iteration` — the global iteration number, used to build every output path.

## Tools available

- Code execution sandbox with `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`,
  `optuna` or `sklearn.model_selection` (for hyperparameter search), `joblib` (for model
  persistence)
- `mlflow`, plus whichever model-flavor logging module matches `model_class` (`mlflow.lightgbm`,
  `mlflow.xgboost`, or `mlflow.sklearn`) — for step 14a, see `references/mlflow_logging.md`
- Plotting library for artifact generation — same "numbers, not images" rule: every plot gets a
  paired CSV of the values plotted
- File write access to `artifacts/training/`, `data/`, `history/`, `code/`, `reports/`, and
  `mlflow_tracking/` (project root, shared across runs — see `references/mlflow_logging.md`)
- Read-only access to `state/current.json` and any file under `history/`

## Procedure

**Every step below is a floor, not a ceiling.** If an additional check, technique, or diagnostic
beyond what's specified here would genuinely help for this specific dataset and model family,
pursue it. Record anything beyond the prescribed procedure in `agent_initiated_findings` with the
same `reasoning` and `confidence` discipline used throughout.

1. **Resolve dataset and validate split integrity.**
   Load `dataset_path`, verify `_split` column exists with the expected three values. Count rows
   per split and confirm none is empty. Verify `target_column` is present and has no NaN values
   in `train` or `validation` rows. Verify no column in `feature_subset` is the target or leaks
   it (sanity-check: no feature in `feature_subset` should have Spearman correlation > 0.98 with
   the target on the training set — flag any that do as suspected leakage and record in `issues`).
   In `incremental` mode, confirm `feature_subset` matches what's actually in the file.

2. **Read context and select the specific model class to train.**

   **2a. Check for an override first.** If `context` contains an explicit model class from a
   Planner instruction or a Diagnosis hypothesis (e.g. "switch to XGBoost", "try Ridge as a
   baseline"), use it directly and skip the decision table below. Document the override source.

   **2b. Apply the model selection decision table.** When no override exists, derive the model
   class from observable dataset properties already measured by EDA and Preprocess. Work through
   the table in order — earlier rows take precedence when multiple conditions apply. Record every
   property that fired and why, not just the final choice.

   This table is a narrowed decision table specific to this pipeline's three supported families
   (gradient boosted trees, random forest, linear). For the full classical-ML decision space —
   SVM, KNN, Naive Bayes, discriminant analysis, kernel approximation, or whenever none of the
   rows below clearly fire — consult `references/model_selection_supervised.md`. For whether this
   GBT/RF/Linear-on-engineered-features approach is the right approach at all versus a classical
   statistical forecasting model (ARIMA/SARIMA/exponential smoothing), consult
   `references/model_selection_timeseries.md` §1 — this pipeline defaults to the "ML-as-regression
   on engineered features" path documented there, and that file's XGBoost/gradient-boosting note
   confirms the choice.

   | Dataset property (from EDA / Preprocess / FE context) | Implication |
   |---|---|
   | n_train < 500 | Avoid GBT — overfits easily on small data. Prefer Ridge (regression) or shallow Random Forest (max_depth ≤ 5) |
   | n_train 500–5000 | GBT is appropriate with moderate regularisation (lower learning rate, min_child_samples > 10) |
   | n_train > 5000 | Full GBT or neural network if feature complexity warrants it |
   | NaN values remain in any feature after preprocessing | Choose LightGBM — it handles NaN natively without imputation. Classic scikit-learn trees require NaN-free input; see `../shared-references/preprocessing_and_feature_engineering.md` §6 for the full model-by-model breakdown, including an open question about XGBoost's NaN handling worth resolving before treating this row as settled |
   | Mixed feature types (OHE categoricals + numeric + sparse) | Tree-based over linear — linear models need careful scaling and encoding to handle this mix; trees split on raw values. See `../shared-references/preprocessing_and_feature_engineering.md` §§1–2 for the full scaling/encoding requirements by model family |
   | Confirmed strong linear relationship with target (EDA R² > 0.7 from a linear fit, or target is near-Gaussian) | Consider Ridge/Lasso as a competitor alongside tree-based; a linear model that performs similarly is strictly preferable for interpretability |
   | Target has confirmed trend + seasonality (time-series task) | Tree-based preferred — can learn non-linear seasonal interactions. Linear models need manual interaction terms to achieve comparable accuracy |
   | n_features > 200 after encoding | LightGBM over XGBoost — LightGBM's histogram-based algorithm scales better with high feature count |
   | Target distribution is heavily right-skewed (skew > 3) | Consider Tweedie or Poisson loss for count/rate targets (GBT supports this via `objective` parameter); or log-transform of the target (see step 6 sub-step) |
   | Binary classification with class imbalance (minority class < 20%) | GBT with `scale_pos_weight` (XGBoost) or `class_weight="balanced"` (LightGBM); record the imbalance ratio |
   | Hard interpretability requirement stated in goal | Logistic Regression, Ridge, or single Decision Tree (depth ≤ 4); do not use GBT if interpretability is non-negotiable |

   **Specific class mapping from family to library:**
   - `gradient_boosted_trees` + NaN features present → `LGBMRegressor` / `LGBMClassifier`
   - `gradient_boosted_trees` + no NaN → `LGBMRegressor` / `LGBMClassifier` (preferred for speed) or `XGBRegressor` / `XGBClassifier`
   - `random_forest` → `sklearn.ensemble.RandomForestRegressor` / `RandomForestClassifier`
   - `linear` → `sklearn.linear_model.Ridge` (regression) / `LogisticRegression` (classification)

   Record `model_class` (the exact library class name), `model_family` (the broad family), and
   `model_selection_reasoning` (which table rows fired and why) in the output.

   **2c. Extract metric configuration from `context`.**
   - **Primary metric** — from EDA's `recommended_metric` or a Diagnosis override. If
     unspecified, default to MAE for regression, F1 for binary classification, macro-F1 for
     multiclass. Always document why this metric was chosen or defaulted. These are this
     pipeline's fixed defaults — see `references/evaluation_metrics.md` for the full
     decision framework (e.g. when Balanced Accuracy, F-beta, ROC AUC, or Pinball loss would be
     preferable) if Diagnosis or the Planner questions a default.
   - **Secondary metrics** — one or more supplementary metrics. For time-series regression,
     always include MASE (see step 7). For classification, always include AUC-ROC.
   - **Feature-selection posture** — whether to include all features in `feature_subset` or
     apply any last-minute pruning before fitting.

3. **Resolve feature subset per `deployment_scenario`.**
   If `deployment_scenario == "live_forecast"`: confirm no `concurrent` feature from the FE
   prediction-time audit is in `feature_subset`. Remove any that are and log the removal.
   If `deployment_scenario == "retrospective"`: all features in `feature_subset` are permitted.
   If `deployment_scenario == "unspecified"`: proceed with the full `feature_subset` and raise a
   non-blocking `clarification_needed` entry asking the Planner to confirm the scenario. Note in
   the report that metrics are computed on the concurrent-included feature set.

4. **Pre-training integrity checks.** Before fitting anything:
   - Confirm the training split is strictly earlier than the validation split (for time-series
     tasks). If rows overlap in time, record a blocking `issues` entry and stop.
   - Confirm no NaN values remain in any column in `feature_subset` for training or validation
     rows. If NaN exists, record in `issues` — do not silently drop rows or impute here (that is
     Preprocess's job; either request a Preprocess fix or fail the call).
   - Check rows-to-features ratio on the training set (n_train_rows / n_features). If below 10×,
     record a warning in `issues` — proceed but flag the risk.

5. **Configure model and hyperparameter strategy.** Choose one of:
   - **Use defaults** (`mode == "initial"` with no prior search results): set sensible defaults
     for the chosen model family (e.g. for gradient boosted trees: n_estimators=500,
     max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8). Document each
     default and the reasoning.
   - **Lightweight search** (Diagnosis suggested a hyperparameter hypothesis, or `mode ==
     "incremental"` with a search instruction): use `TimeSeriesSplit` (not random k-fold — see
     Guardrails) with 3–5 folds over the training set only. Evaluate on the fold's held-out
     portion using the primary metric. Never let any fold's validation rows overlap with the
     global validation or test splits. See `references/hyperparameter_tuning.md` §§1–2 for the
     full grid/random/halving/Bayesian-optimization comparison if a different search strategy is
     warranted, and §5 specifically for time-series tuning — walk-forward validation during
     search (not just final evaluation), and treating the lag/rolling window size itself as a
     tunable choice rather than a fixed guess.
   - **Accept explicit hyperparameters** (Planner or Diagnosis passed specific values in
     `context`): use them directly and document the source.

   For all strategies: document every hyperparameter used and the reasoning for its value or the
   search range. A hyperparameter left at its library default without documentation is an
   unrecorded assumption.

6. **Train the model on `_split == "train"` rows only.** Fit using only the resolved
   `feature_subset` columns and `target_column`. Record training time. After fitting, save the
   model to `artifacts/training/iteration_<NNN>_training_model.pkl` using `joblib.dump`.

   **6a. Baseline model sub-step (`mode == "initial"` only).** When this is the first Training
   call (`mode == "initial"`), also fit a simple baseline model on the same training rows
   immediately after the primary model. The baseline has two roles: it gives MASE a second point
   of comparison (alongside the seasonal naive denominator) and it tells Diagnosis whether the
   primary model's complexity is earning its keep. Repeat the baseline in later iterations only
   when Diagnosis explicitly requests a re-comparison.

   Choose the baseline type from the table below:

   | Task type | Baseline |
   |---|---|
   | Time-series regression | `Ridge` from `sklearn.linear_model`, fit on `[time_index, <strongest_lag_feature>]` only (two features maximum). If time_index does not exist, use the single highest-importance lag feature. Compute predictions on both train and val. |
   | Plain regression (non-time-series) | Mean predictor: always predicts the train-set mean of `target_column`. |
   | Binary classification | Majority-class predictor: always predicts the most-frequent class in training. |
   | Multiclass classification | Majority-class predictor. |

   Record the baseline's train and validation primary metric values in `baseline_model`. The
   baseline is never saved as a `.pkl` artifact — it exists only for metric comparison. Do not
   compute feature importance or residuals for the baseline. If the primary model's validation
   metric is worse than the baseline's, raise this as a high-confidence `agent_initiated_finding`
   — it almost always means a structural problem with the feature set or target leakage.

   **6b. Target log-transform check.** If EDA reported target skewness > 3 (or if the decision
   table in step 2b flagged it), apply `np.log1p` to `target_column` for the training fit and
   invert with `np.expm1` before computing metrics. Document the transform in the output under
   `training_summary.target_transform`. Do not transform the target without recording it — an
   unrecorded transform makes Diagnosis's residual interpretation wrong. (Log transformation as a
   target-skew fix is also documented in `../shared-references/preprocessing_and_feature_engineering.md`
   §5.2 alongside differencing, for consistency with how Preprocess/FE handle skew and
   non-stationarity elsewhere in the pipeline.)

7. **Evaluate on `_split == "validation"` rows. Never use `_split == "test"` rows.**
   Compute the primary metric and all secondary metrics on both train and validation sets (train
   metrics help assess overfitting in step 8). For each metric record both the value and the
   computation method.

   **Metric selection.** This pipeline's fixed defaults per task type:

   | Task type | Primary (default) | Secondary (always include) |
   |---|---|---|
   | Regression | MAE | RMSE, R², MAPE |
   | Time-series regression | MAE | MASE, RMSE |
   | Binary classification | F1 | AUC-ROC, precision, recall, confusion matrix |
   | Multiclass classification | Macro-F1 | Per-class F1, weighted-F1, confusion matrix |

   See `references/evaluation_metrics.md` §§2–4 for the full reasoning behind each of these (why
   MAE vs. RMSE vs. MAPE, why AUC-ROC pairs with F1, why MASE for time series) and for metrics to
   reach for outside these defaults (e.g. Balanced Accuracy under severe imbalance, Pinball loss
   for quantile targets). Report threshold used for classification (default 0.5); note if class
   imbalance is present. Flag MAPE if the target has near-zero values (division instability) —
   this is exactly why MASE is mandatory for time-series regression instead.

   **MASE sub-step (time-series regression only):**
   See `references/evaluation_metrics.md` §4 for the MASE formula and seasonal-naive-baseline
   procedure. This pipeline's specific requirement: compute the naive baseline MAE on the
   **training set only** — fit the denominator on train, apply it to normalise both train and val
   MAE (see Guardrails — this is the same leakage principle as fitting a scaler on train only).
   Record the seasonal period P used, the naive baseline MAE (train), and the resulting MASE for
   both train and validation.

8. **Overfitting assessment.** Compute the ratio of validation primary metric to train primary
   metric. For error metrics (MAE, RMSE, log-loss): ratio = val / train (> 1 means worse on val).
   For score metrics (R², AUC, F1): ratio = train / val (> 1 means worse on val). Classify:

   | Ratio | Assessment |
   |---|---|
   | < 1.05 | `good_fit` — negligible gap |
   | 1.05 – 1.25 | `mild_overfit` — monitor; may resolve with more data or lighter regularisation |
   | 1.25 – 2.0 | `moderate_overfit` — worth addressing; Diagnosis should investigate feature count or regularisation |
   | > 2.0 | `severe_overfit` — likely cause: too many features relative to training rows, or data leakage |

   Also flag `underfitting` when validation metric is poor AND train metric is similarly poor
   (both far from a reasonable baseline), which suggests the model family or feature set is
   insufficient regardless of overfitting. When this ratio-based read is ambiguous or Diagnosis
   wants more than a single-number verdict, pair it with a validation/learning curve — see
   `references/evaluation_metrics.md` §6 for how to read those.

9. **Extract and rank feature importance.** For tree-based models use the model's built-in
   `feature_importances_` attribute (split-gain or impurity-based, whichever the library
   provides — document which). For linear models use `abs(coef_)`. For other families use
   permutation importance on the validation set (not train — permutation importance on train
   inflates the importance of memorised features). Record the top-N features (at minimum the top
   10 or all features if fewer than 10) with their importance values and ranks. Flag any feature
   with near-zero importance (< 0.5% of total importance for tree models) — Diagnosis may choose
   to drop it.

10. **Compute residual diagnostics for Diagnosis.** On the validation set:
    - Compute `residuals = y_actual - y_predicted` (for regression) or predicted-probability
      minus a threshold-based indicator (for classification).
    - Save the full residuals series (with date/index column if available) to
      `artifacts/training/iteration_<NNN>_training_val_residuals.csv`.
    - Compute and record: mean residual, std of residuals, and the 5th/95th percentile.
    - For time-series: compute Spearman autocorrelation of residuals at lag-1 and at the
      seasonal lag (lag-7 for weekly, lag-12 for monthly, etc.). A high residual autocorrelation
      (|r| > 0.2) indicates the model is not capturing temporal structure — record this as a
      finding for Diagnosis.
    - Compute mean residual by time period (e.g. by year or quarter) to detect systematic
      temporal bias — if residuals are strongly positive in early periods and negative in later
      ones (or vice versa), the trend or regime-change feature may be insufficient. (This kind of
      drift is also flagged as an operational concern in `references/model_selection_timeseries.md`
      — worth surfacing to whatever stage owns model monitoring/retraining triggers downstream of
      this skill, not just Diagnosis.)
    - For classification: compute calibration — mean predicted probability vs observed positive
      rate across probability bins.

11. **Record `reasoning` and `confidence` for every major decision** — model family choice,
    hyperparameter configuration, metric selection, feature subset resolution — per
    `conventions/storage_conventions.md`. A decision left without reasoning is an unrecorded
    assumption that Diagnosis cannot act on.

12. **Raise `clarification_needed` for genuine ambiguity.** Same rules as all other nodes: a
    `blocking` flag, a `confidence`, and never a self-decision to pause the workflow. Typical
    triggers: deployment_scenario is unspecified, metric preference is ambiguous, seasonal
    period for MASE is unclear, or a suspected leakage feature was found in step 1.

13. **Record any execution failures in `issues`.** A model that failed to converge, a metric
    that returned NaN, a NaN found in the feature set — record rather than silently omitting.

14. **Write outputs and generate artifacts.** Save:
    - Model: `artifacts/training/iteration_<NNN>_training_model.pkl`
    - Residuals: `artifacts/training/iteration_<NNN>_training_val_residuals.csv`
    - History JSON: `history/iteration_<NNN>_training.json`
    - Code: `code/iteration_<NNN>_training.py`
    - Report: `reports/iteration_<NNN>_training.md`
    - Update `state/current.json`: `latest_model_path`, `latest_metrics`, `current_hypothesis`,
      `mlflow_parent_run_id` (set on the first Training call in this run; carried forward
      unchanged by every later one — see step 14a)

    **Minimum required artifacts** (each plot paired with a CSV of the plotted values; see
    `references/plotting_guide.md` for the full condition → plot table this is the floor of,
    including classification-specific, iterative-model-training-curve, hyperparameter-search,
    interpretability, and iteration-tracking plots that apply beyond this fixed list):
    - `feature_importance` — ranked bar chart of all features by importance
    - `actual_vs_predicted_val` — scatter or time-series overlay of actual vs predicted on
      validation set
    - `residuals_over_time` — residuals plotted in chronological order on validation set (reveals
      temporal bias)
    - `residuals_distribution` — histogram of validation residuals (reveals skew, outliers)

    For time-series, also generate:
    - `residual_acf` — autocorrelation of residuals at lags 1 through at least 14 (reveals
      whether temporal structure remains uncaptured)

    **14a. Mirror this iteration to MLflow (optional, never blocking).** After the artifacts above
    are written and `state/current.json` is otherwise ready to update, log this iteration's
    params/metrics/tags/artifacts/model into the project's shared MLflow tracking store. Full
    mechanics — tracking URI, experiment naming, parent/child run nesting, and the exact field
    mapping — are in `references/mlflow_logging.md`; don't re-derive them here. This step must
    never block or fail the call: wrap it in `try/except` and see the MLflow guardrail below if
    anything goes wrong. Record the resulting `mlflow` output fields (all null if this step was
    skipped or failed).

15. **Generate a human-readable report.** Write `reports/iteration_<NNN>_training.md` covering:
    mode and what triggered this call; feature subset used and deployment scenario; model family
    and hyperparameters with reasoning; train and validation metrics (primary + all secondary);
    MASE and its interpretation relative to the seasonal naive baseline (for time-series);
    overfitting assessment; top-10 feature importance ranking; residual diagnostic findings and
    what they imply for Diagnosis; clarification_needed entries; issues. Write explanations, not
    just numbers — a metric without context is not actionable.

## Guardrails

- **Never use `_split == "test"` rows for any purpose in this node** — not for evaluation, not
  for hyperparameter search, not for residual computation. Test rows must remain untouched until
  end_model. Any contact with test rows here constitutes optimistic bias in the final reported
  metric.
- **Never use random cross-validation (random k-fold, stratified k-fold) for time-series data.**
  Random folds allow future rows to appear in training folds and past rows in validation folds,
  leaking temporal structure. Always use `TimeSeriesSplit` or the fixed chronological split. See
  `references/cross_validation.md` for the full CV-strategy decision table and the reasoning
  behind it.
- **Never fit any statistic on validation or test rows** — including scaling, imputation, or any
  preprocessing applied inside the training loop. All such statistics must come from Preprocess
  (already fit on train) or from training rows only within this node. (Same rule `references/
  cross_validation.md` §3 states generally: fit on training folds only, apply to validation.)
- **Never impute, scale, or encode features here** — that is Preprocess's job. If the feature
  set has NaN values or unencoded categoricals, record an `issues` entry and request a Preprocess
  fix rather than silently patching it in the training loop. Model-specific scaling/encoding/
  NaN-handling requirements that inform what to ask Preprocess for are documented in
  `../shared-references/preprocessing_and_feature_engineering.md` (e.g. §6 explains why LightGBM's
  native NaN handling matters for the step 2b decision table).
- Never treat model training as a substitute for feature selection — high feature importance does
  not retroactively justify a feature that failed the FE variance-in-train or relevance checks.
  Low feature importance on a kept feature is a signal for Diagnosis to act on, not an error.
- **Never report final performance against the test set** — that number belongs exclusively to
  end_model. If validation metrics are strong enough to warrant reporting, note they are
  validation metrics, not test metrics.
- For MASE: always compute the denominator on training data only. Using the full dataset to
  compute the naive baseline denominator would let future demand levels influence what counts as
  a "good" score on past data.
- Every artifact plot must have a paired CSV. Never load that CSV into your own reasoning
  context.
- Every decision — model family, hyperparameter, metric choice — needs `reasoning` and
  `confidence`. A decision recorded only as "default" is an unrecorded assumption.
- The `reports/` Markdown file must only restate what is already in the JSON output.
- Never decide on your own to pause the workflow — raise `clarification_needed` and let the
  Planner decide.
- **MLflow logging (step 14a) is best-effort bookkeeping, not a required deliverable.** A failure
  there must never block writing the required outputs (history JSON, report, state update) —
  record it as a non-blocking `issues` entry and continue. Never log free-text reasoning as a
  metric value; see `references/mlflow_logging.md` for what's a param/metric/tag/artifact.

## Input / output contract

Full input/output example: `contracts/training.json`.

Top-level output sections: `schema_version`, `trained_model_path`, `model_family`,
`model_class`, `model_selection_reasoning` (which decision-table rows fired and why — or override
source if step 2a applied), `feature_subset_used`, `feature_subset_type` (`concurrent_included` or
`pre_period_only`), `deployment_scenario`, `hyperparameters` (each with `value` and `reasoning`),
`hyperparameter_search` (method, n_trials, search_space — null if defaults used),
`training_summary` (n_train_rows, n_val_rows, n_features, training_time_seconds,
`target_transform` — the transform applied to the target column, or null if none),
`metrics` (train and validation objects, each with all computed metrics; for time-series include
`mase` with `seasonal_period`, `naive_baseline_mae_train`, and `mase_value`),
`baseline_model` (present when `mode == "initial"`; contains `baseline_type`,
`train_<primary_metric>`, `val_<primary_metric>`, and `vs_primary_model` assessment: one of
`primary_better`, `primary_worse` — flag as finding if `primary_worse`; null in `incremental`/
`rebuild` mode unless Diagnosis requested a re-comparison),
`overfitting_check` (ratio, assessment string),
`feature_importance` (ranked list, each entry: feature, importance, rank, near_zero_flag),
`residuals_summary` (val_residuals_path, mean, std, p5, p95; for time-series: acf_lag1,
acf_seasonal_lag, temporal_bias_by_period),
`mlflow` (`tracking_uri`, `experiment_name`, `parent_run_id`, `run_id` — all null if step 14a was
skipped or failed; see `references/mlflow_logging.md`),
`clarification_needed`, `agent_initiated_findings`, `issues`,
`artifact_paths` (plots list, each with path and data_path).
