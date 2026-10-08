---
name: training-agent-skill
description: Trains a model on the feature-engineered dataset, evaluates it on the validation set, and produces the artifacts (trained model, metrics, feature importance, residuals) that Diagnosis needs to decide whether to loop back or accept the result. Every number it reports is a validation number; the unbiased held-out score is produced outside this pipeline after the run ends.
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

- `../baseline-skill/baseline.md` — **required at step 6a** (`mode == "baseline"`). The
  closed six-rule menu that sets `baseline_floor`, and the five rules that keep the floor
  honest. Cheap to load and cheap to run; a wrong floor invalidates every later number in
  the run
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
- `../shared-references/preprocessing_and_feature_engineering.md` — model-specific scaling,
  encoding, and missing-value requirements, owned jointly with Preprocess; consult before
  assuming a feature is training-ready

## Inputs

- `dataset_path` — the feature-engineered dataset. Must contain a `_split` column with values
  `train` and `validation` only. **If a `test` value appears, stop and raise it as an `issues`
  entry** — the genuinely held-out data was removed before this run began, so a partition named
  `test` here means something upstream invented one, and any metric quoted against it would be
  mistaken for a final number.
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
  - `baseline` — **the first Training call of every run.** Fits no learned model: it scores the
    trivial baseline on the real train/validation split and establishes the floor every later
    iteration is measured against. See step 6a.
  - `initial` — first *learned* model this run.
  - `incremental` — re-train with a changed feature set, different hyperparameters, or a
    Diagnosis-driven modification. Loads the prior model path from `state/current.json` for
    comparison.
  - `rebuild` — discard the prior model entirely and start fresh (e.g. model family change).
- `mode_reasoning` — why this mode was chosen.
- `prior_model_path` — path to the previously saved model (used in `incremental` mode for
  comparison; null in `initial` or `rebuild`). On a multi-horizon task this is a dict keyed by
  horizon; only the horizons being retrained this call need an entry.
- `target_horizons` — **optional.** A subset of `context.forecast_horizons` specifying which
  horizons to train this call. When absent or null, all horizons in `context.forecast_horizons`
  are trained. When present, only the listed horizons are fitted — all others are left as-is in
  `state/current.json`. This allows the Planner to loop on a single underperforming horizon
  without discarding the other horizons' already-validated models. The horizons listed must be a
  subset of `context.forecast_horizons`; an unlisted value is a blocking `issues` entry.
- `iteration` — the global iteration number, used to build every output path.

## Tools available

- Code execution sandbox with `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`,
  `optuna` or `sklearn.model_selection` (for hyperparameter search), `joblib` (for model
  persistence)
- Plotting library for artifact generation — same "numbers, not images" rule: every plot gets a
  paired CSV of the values plotted
- File write access to `artifacts/training/`, `data/`, `history/`, `code/`, and `reports/`
- Read-only access to `state/current.json` and any file under `history/`

## Procedure

**Every step below is a floor, not a ceiling.** If an additional check, technique, or diagnostic
beyond what's specified here would genuinely help for this specific dataset and model family,
pursue it. Record anything beyond the prescribed procedure in `agent_initiated_findings` with the
same `reasoning` and `confidence` discipline used throughout.

0. **Resolve the horizon set before anything else.**

   Read `context.target_horizons` if present; otherwise read `context.forecast_horizons`.
   This is the set of horizons **this call** trains — call it `active_horizons`. Record it in
   the output as `horizons_trained_this_call`. When `target_horizons` was used, also record
   `horizons_skipped_this_call` — the horizons in `forecast_horizons` that are not being
   retrained — so the Planner and Diagnosis can see exactly which models were touched.

   When `active_horizons` holds more than one value, **this call trains one model per active
   horizon** — not one model scored at several lead times. Each horizon gets its own fit, on its
   own target column (`<target>_h<h>`, built by Feature Engineering), using its own feature
   subset (`horizon_feature_subsets[h]`, which filters on **all** bounds — it excludes every
   feature whose `min_horizon` exceeds `h`, every feature whose `max_horizon` is below `h` (the
   naive predictors, each of which is correct only at its own horizon), and every feature whose
   `known_future_horizon` is below `h` — and marks each remaining feature
   `evaluated_at: origin | target_date`), its own
   hyperparameters, and its own artifact: `iteration_<NNN>_training_model_h<h>.pkl`.

   The reason is not tidiness. A single model pooled across horizons learns mostly from the short
   ones — they dominate the sample and they are easier — and the long horizons inherit a fit
   built on features they will not have at their own origin. The per-horizon models can also
   legitimately differ: a lag-heavy model at `h = 1` and a level-and-calendar model at `h = 30`
   is the correct outcome, not an inconsistency to be smoothed away. Record each horizon's model
   family and hyperparameters separately, and never report a metric pooled across horizons without
   the per-horizon breakdown beside it.

   Every step below then runs per active horizon. Where a decision is genuinely shared — the
   primary metric, the deployment scenario — record it once and say it applies to all; where it
   differs, record it per horizon. `mode == "baseline"` follows the same rule: the floor is scored
   per horizon, because the naive error grows with lead time and one pooled floor would be wrong at
   both ends. On a baseline call, `target_horizons` must not be used — the floor must be
   established for every horizon at once so it is never missing when a later call needs it.

   A single-horizon task — `forecast_horizons` absent, or one value — runs exactly as before.

1. **Resolve dataset and validate split integrity.**
   Load `dataset_path`, verify `_split` column exists with the expected three values. Count rows
   per split and confirm none is empty. Verify `target_column` is present and has no NaN values
   in `train` or `validation` rows. Verify no column in `feature_subset` is the target or leaks
   it (sanity-check: no feature in `feature_subset` should have Spearman correlation > 0.98 with
   the target on the training set — flag any that do as suspected leakage and record in `issues`).

   **The naive predictors are the one expected exception to that 0.98 check.** A short-horizon
   naive predictor is a near-copy of the target by design — that is what makes it a strong feature
   — and it is strictly causal, computed at origin `target − h`, which is the property the check
   is actually screening for. When the column is in `context.protected_features` and its
   `max_horizon` equals the horizon being trained, record the correlation as expected rather than
   as suspected leakage, and do not remove it. When it is **not** protected, or its `max_horizon`
   does not match, the flag stands and is a real finding: a naive predictor showing up in the
   wrong horizon's subset is exactly the leak this check catches.

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

   **2b-ts. Time-series tasks: name this fit's approach, and check the other one has a run.**
   `references/model_selection_timeseries.md` §1 names three approaches. Two of them are
   **non-automatic** — **classical statistical forecasting** (ARIMA/SARIMA/SARIMAX/exponential
   smoothing, fitted to the time-ordered series itself) and **ML-as-regression on engineered
   features** (the GBT/RF/linear path of the table above, fed lag/rolling/calendar features). The
   third — Auto ARIMA, Prophet, an AutoML forecasting framework — is automatic and counts as
   neither.

   Record `ts_approach` in the output on every time-series call: `classical_statistical`,
   `ml_as_regression`, or `automatic`.

   **Both non-automatic approaches must have been fitted at least once before the run may stop.**
   Defaulting to `ml_as_regression` on every iteration and never fitting a classical model is not
   an acceptable run: the two put the structure in different places — one in the series' own
   autoregressive and seasonal terms, the other in engineered features — and which one this
   series rewards is an empirical question that only a fit answers. It is not settled by the
   decision table above, which already assumes the ML-as-regression framing.

   Read `context.ts_approaches_tried`, which the Planner carries forward. If this call fits an
   approach that is already covered while the other has never been tried, fit what `context` asked
   for, then set `ts_approach_gap` in the output to the missing approach and note it in
   `agent_initiated_findings` — that is the field the Planner's coverage check and Diagnosis read.

   **The classical slot is filled by the ARIMA family, and plain ARIMA is fitted at least once.**
   Exponential smoothing (SES, Holt-Winters) is a legitimate classical model and may be run in
   addition, but on its own it does **not** close the classical slot. The slot is closed by a fit
   from `references/model_selection_timeseries.md` §2's ARIMA family — ARIMA, SARIMA, or SARIMAX
   — chosen by that section's decision shortcut against what the series actually has (trend only
   → ARIMA; trend + seasonality → SARIMA; external regressors on top → SARIMAX).

   When the shortcut sends you past plain ARIMA to SARIMA or SARIMAX, **fit plain ARIMA anyway,
   once, alongside it.** It is a cheap ablation and it answers a question the seasonal fit cannot:
   what the seasonal (or exogenous) terms are actually worth on this series. Report both numbers.
   Record which member of the family closed the slot in `model_class`, and record the plain-ARIMA
   comparison in `agent_initiated_findings` when ARIMA was not itself the chosen member.

   Order selection `(p,d,q)` — and `(P,D,Q)m` where seasonal — follows
   `references/hyperparameter_tuning.md` §§1–2 and its classical-model row: search the orders,
   do not guess them, and do not leave `d` at a default. Establish `d` from a stationarity check
   (ADF plus the rolling mean/std plot in `references/plotting_guide.md`) and record the value with
   its evidence. An Auto ARIMA search may be used *to propose* orders, but the fit that closes the
   slot is the explicit one whose orders are recorded — the `automatic` approach still counts as
   `automatic`.

   **On a multi-horizon task, both approaches are fitted per horizon.** A classical ARIMA-family
   fit is naturally per-horizon anyway (its forecast function is evaluated at each lead time from
   the same origin); the ML-as-regression path needs one model per horizon per step 0. The
   comparison between the two approaches is then made **at each horizon**, and it is allowed to
   come out differently at different lead times — classical winning at `h = 30` while
   ML-as-regression wins at `h = 1` is a real and common result, and reporting only the pooled
   winner would hide it.

   **The comparison only means something if the split and the metric match.** A classical fit is
   evaluated on the *same* validation rows with the *same* primary metric as the ML-as-regression
   fits. That a classical model cannot consume the engineered feature set is expected — it is the
   substance of the comparison, not a defect — but a different split or a different metric makes
   the two numbers unreadable next to each other.

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

   **Never prune a protected feature out of `feature_subset`.** Every column in
   `context.protected_features` whose `max_horizon` matches the horizon being trained stays in,
   whatever the feature-selection posture says and whatever a last-minute pruning step would
   otherwise do. If this call has evidence that one should go, fit with it anyway and raise a
   `blocking` `clarification_needed` naming the column and the evidence — the user decides, per
   the Planner's Clarification Handling section. Leaving a horizon's own naive predictor out is
   not a pruning decision this node gets to make.

   A protected feature is still subject to horizon scoping: `naive_h7_*` is absent from the
   `h = 30` subset because it is only correct at `h = 7`, not because anything was pruned.

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
     global validation split. See `references/hyperparameter_tuning.md` §§1–2 for the
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

   **6a. Baseline sub-step.** Two different things are called a baseline here and the difference
   matters:

   **The floor (`mode == "baseline"`, the run's first Training call).** Follow
   `../baseline-skill/baseline.md` — it is the whole procedure for this sub-step and is
   deliberately small: six trivial rules, one pass, no tuning. Do not hand-roll a single rule
   here. It scores the *trivial* predictors on the real split and reports the **strongest** as the
   floor, which is the part a hand-rolled version gets wrong: a lone seasonal naive
   (`ŷ_t = y_{t-P}`) is usually not the strongest trivial rule, and adopting it unchallenged
   sets the floor too low for the rest of the run. For plain regression the floor is the training
   mean, for classification the majority class, and the skill's rules on origin discipline and
   row alignment still apply. Compute the full metric set on it — for a time series that means MAE, RMSE and
   SMAPE (MASE against itself is 1.0 by construction; record it anyway so the tables line up) —
   **and compute it per segment the goal is stated over**, per horizon for a multi-horizon
   forecast. Write it to `baseline_floor`, then run the skill's **Part B** and write
   `naive_predictor_features` — the per-horizon naive predictor specs, selected on train rows
   only, that the next `feature_engineering` call materialises as protected columns. Both blocks
   go in the handoff in full, with formulas and parameters, because FE reimplements each one and
   cannot do that from a name. Then stop: this call trains nothing, saves no `.pkl`, and takes an
   iteration number like any other node.

   **Part B is not optional and not deferrable.** It is the only point in the run where the naive
   predictors get selected, and Feature Engineering runs next. A baseline call that returns a
   floor without them leaves the run's feature sets permanently missing the one predictor class
   selected specifically for each horizon, and nothing downstream reports the absence.

   This is the number the rest of the run exists to beat, and it is established first precisely so
   that no later iteration can quietly redefine what "good" means. EDA's `trivial_baseline` is the
   provisional version of it; this one is authoritative because it uses the same rows, the same
   metric and the same segments as every model that follows.

   **The reference model (`mode == "initial"`).** When the first *learned* model is fitted, also
   fit a simple reference model on the same training rows immediately after it. This is a
   different question from the floor: it asks whether the primary model's complexity is earning
   its keep against something cheap but non-trivial. Repeat it in later iterations only when
   Diagnosis explicitly requests a re-comparison.

   Choose the reference type from the table below:

   | Task type | Baseline |
   |---|---|
   | Time-series regression | `Ridge` from `sklearn.linear_model`, fit on `[time_index, <strongest_lag_feature>]` only (two features maximum). If time_index does not exist, use the single highest-importance lag feature. Compute predictions on both train and val. |
   | Plain regression (non-time-series) | Mean predictor: always predicts the train-set mean of `target_column`. |
   | Binary classification | Majority-class predictor: always predicts the most-frequent class in training. |
   | Multiclass classification | Majority-class predictor. |

   Record the reference model's train and validation primary metric values in `baseline_model`.
   Neither the floor nor the reference model is saved as a `.pkl` artifact — both exist only for
   metric comparison. Do not compute feature importance or residuals for either. If the primary
   model's validation metric is worse than the reference model's, raise this as a high-confidence
   `agent_initiated_finding` — it almost always means a structural problem with the feature set or
   target leakage.

   **Every Training call reports its standing against the floor.** Read `context.baseline_floor`
   and record `vs_baseline_floor`: the margin per segment and per horizon, not just pooled, and
   whether the model beats the floor **everywhere** or only on average. A model that wins pooled
   while losing on a third of the segments is the specific failure the per-segment floor exists to
   catch — the aggregate will never say so on its own.

   **6b. Target log-transform check.** If EDA reported target skewness > 3 (or if the decision
   table in step 2b flagged it), apply `np.log1p` to `target_column` for the training fit and
   invert with `np.expm1` before computing metrics. Document the transform in the output under
   `training_summary.target_transform`. Do not transform the target without recording it — an
   unrecorded transform makes Diagnosis's residual interpretation wrong. (Log transformation as a
   target-skew fix is also documented in `../shared-references/preprocessing_and_feature_engineering.md`
   §5.2 alongside differencing, for consistency with how Preprocess/FE handle skew and
   non-stationarity elsewhere in the pipeline.)

7. **Evaluate on `_split == "validation"` rows.**
   Compute the primary metric and all secondary metrics on both train and validation sets (train
   metrics help assess overfitting in step 8). For each metric record both the value and the
   computation method.

   **Metric selection.** This pipeline's fixed defaults per task type:

   | Task type | Primary (default) | Secondary (always include) |
   |---|---|---|
   | Regression | MAE | RMSE, R², MAPE |
   | Time-series regression | Chosen by the agent from EDA's `recommended_metric` — MAE unless the goal argues otherwise | **MAE, RMSE, SMAPE and MASE, always, whichever of them is not the primary** — plus anything else the agent judges relevant |
   | Binary classification | F1 | AUC-ROC, precision, recall, confusion matrix |
   | Multiclass classification | Macro-F1 | Per-class F1, weighted-F1, confusion matrix |

   See `references/evaluation_metrics.md` §§2–4 for the full reasoning behind each of these (why
   MAE vs. RMSE vs. MAPE, why AUC-ROC pairs with F1, why MASE for time series) and for metrics to
   reach for outside these defaults (e.g. Balanced Accuracy under severe imbalance, Pinball loss
   for quantile targets). Report threshold used for classification (default 0.5); note if class
   imbalance is present. Flag MAPE if the target has near-zero values (division instability) —
   this is exactly why MASE is mandatory for time-series regression instead.

   **The time-series floor is four metrics, and it is not negotiable.** On any time-series task,
   compute **MAE, RMSE, SMAPE and MASE** on both train and validation, every call, regardless of
   which one is primary. They are there because each answers a question the others cannot: MAE
   gives the typical error in the target's own units; RMSE weights the rare large miss, so the
   RMSE/MAE ratio says whether the error is spread evenly or concentrated in a few bad days; SMAPE
   is scale-free, so it is the only one of the four that compares honestly across segments, series
   or horizons of different magnitude; MASE prices the model against the seasonal naive.

   **Choosing the primary metric is still the agent's call** — take EDA's `recommended_metric`, or
   depart from it with recorded reasoning, and add any further metric the task warrants. What this
   floor removes is the option of *not computing* one of the four, which is how a run ends up
   unable to answer "was that a uniformly small error or three catastrophic days?" after the fact.

   **When the four disagree, that disagreement is a finding, not noise.** If RMSE ranks two
   iterations differently from MAE, the difference is outlier-driven and belongs in the report and
   in the handoff to Diagnosis — do not report only the primary metric's verdict and leave the
   contradiction unmentioned.

   **SMAPE and zeros:** SMAPE is undefined where actual and predicted are both zero, and unstable
   where the actual is zero (closed days, stock-outs, off-season). State the convention used —
   excluding those rows, or defining `0/0 = 0` — record how many rows it affected, and never let
   the choice go unrecorded, because the two conventions can differ by several points on a series
   with many zero days.

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

   **8b. Achievable-range check — required for every regression model, and a hard gate for
   tree-based ones.** Overfitting is not the only way a model can be wrong in a way the primary
   metric hides. A tree ensemble predicts an average of training-set leaf values, so **it can
   never output a value outside the range of the target it was trained on.** If the series moves
   out of that range — a declining trend, a level shift, a regime change — the model cannot
   follow it, and no amount of tuning will fix it because the failure is in the target
   representation, not the hyperparameters.

   Compute and record, on the validation rows:

   ```
   pred_min, pred_max                     the model's achievable output range
   y_min, y_max                           the actual range it is being asked to reach
   out_of_range_pct    = share of actual values outside [pred_min, pred_max]
   one_sided_bias_pct  = share of residuals sharing a sign
   ```

   | Reading | Assessment | What it means |
   |---|---|---|
   | `out_of_range_pct` < 1% and `one_sided_bias_pct` < 65% | `range_ok` | the model can reach where the data goes |
   | `out_of_range_pct` 1–5%, or `one_sided_bias_pct` 65–80% | `range_warning` | record it, hand it to Diagnosis by name |
   | `out_of_range_pct` > 5%, or `one_sided_bias_pct` > 80% | `range_failure` | **the model is mis-specified.** Do not present it as a candidate without saying so |

   On `range_failure` with a tree-based model, the fix is a **target representation** change, not
   a hyperparameter change — see `featureengineering.md` step 2b. Say so explicitly in `issues`
   and in the handoff, and name the anchor the model already has available. A recalibration or
   bias-correction proposal is a *downstream* remedy for this and must not be offered as the
   primary one: it treats the symptom in the serving layer while the model stays unable to
   represent the level.

   **Watch for this being invisible on validation.** These figures are computed on validation
   because that is the only partition Training may touch — but the defect they detect only
   appears once the target moves outside the training range. If Preprocess reported a larger
   drifting series and warned that validation sits closer to train than the unseen holdout will
   (`split_info.holdout_representativeness`, see `preprocess.md`
   step 3), then validation *understates* this failure and a `range_ok` here is not evidence of
   safety. Record that caveat alongside the numbers rather than letting a clean validation read
   stand unqualified.

   Record all four numbers plus the assessment under `overfitting_check` as
   `achievable_range`. Plot the predicted-vs-actual range comparison when the assessment is
   anything other than `range_ok` — a histogram of actuals with the achievable range shaded is
   enough, and it makes the failure obvious to a reader who would skim past the numbers.

9. **Extract and rank feature importance — by group, per
   `../shared-references/feature_attribution.md`.** That file is the specification; this step is
   the summary of what it requires.

   Feature importance is not identifiable when features are correlated, so the reported value is
   **grouped permutation importance on the validation set** (never on train — permutation
   importance on train inflates the importance of memorised features):

   a. Cluster features by Spearman rank correlation, merging at `|rho| >= 0.6`. Record the clusters
      — "these features are the same variable" is itself a finding.
   b. Permute every column in a cluster together with the same row permutation, re-score the
      **final** prediction, and record the MAE increase over >= 5 repeats with its standard deviation.
   c. Record each group's `masked` gap: joint importance minus the sum of its members' solo
      importances. A positive gap is credit no single feature can claim.

   For linear models `abs(coef_)` on standardised inputs remains acceptable, since collinearity
   there is visible in the coefficients themselves — but the clustering in (a) is still run and
   reported.

   Built-in `feature_importances_` (split-gain or impurity-based) may be recorded alongside as a
   diagnostic of how the tree was built, clearly labelled as such. It is substantially a measure of
   feature cardinality — in `run_20260902T120000Z` it ranked `covid_flag` last of 25 while
   permutation ranked it fifth — so it is never the reported importance and never the basis for a
   flag.

   `near_zero_flag` is a **group-level** verdict: set it when a group's joint importance and its
   masked gap are both indistinguishable from zero. Never flag a column on its solo score — a
   redundant feature scores zero precisely because its twin covers for it.

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
    `conventions/naming_conventions.md`. A decision left without reasoning is an unrecorded
    assumption that Diagnosis cannot act on.

12. **Raise `clarification_needed` for genuine ambiguity.** Same rules as all other nodes: a
    `blocking` flag, a `confidence`, a `recommendation` with its `recommendation_reasoning` and
    `impact_if_wrong` (field table in the Planner's **Clarification Handling** section), and never
    a self-decision to pause the workflow. Typical triggers: deployment_scenario is unspecified,
    metric preference is ambiguous, seasonal period for MASE is unclear, or a suspected leakage
    feature was found in step 1. Each of those has an obvious default this node can name — the
    conservative scenario, the metric EDA recommended, the period the ACF actually shows — so name
    it rather than presenting the question bare.

13. **Record any execution failures in `issues`.** A model that failed to converge, a metric
    that returned NaN, a NaN found in the feature set — record rather than silently omitting.

14. **Write outputs and generate artifacts.** Save:
    - Model: `artifacts/training/iteration_<NNN>_training_model.pkl`
    - Residuals: `artifacts/training/iteration_<NNN>_training_val_residuals.csv`
    - History JSON: `history/iteration_<NNN>_training.json`
    - Code: `code/iteration_<NNN>_training.py`
    - Report: `reports/iteration_<NNN>_training.md`
    - Update `state/current.json`: `latest_model_path`, `latest_metrics`, `current_hypothesis`

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

15. **Generate a human-readable report.** Write `reports/iteration_<NNN>_training.md` covering:
    mode and what triggered this call; feature subset used and deployment scenario; model family
    and hyperparameters with reasoning; train and validation metrics (primary + all secondary);
    MASE and its interpretation relative to the seasonal naive baseline (for time-series);
    overfitting assessment; top-10 feature importance ranking; residual diagnostic findings and
    what they imply for Diagnosis; clarification_needed entries; issues. Write explanations, not
    just numbers — a metric without context is not actionable.


16. **Log to MLflow (required for every training call).**

   After all artifacts and the history JSON have been written, append the following block to the
   **end** of the generated `code/iteration_<NNN>_training_*.py` script. This is the last thing
   the script does; nothing after it can affect training results.

   ```python
   # ── MLflow logging ──────────────────────────────────────────────────────
   import glob as _glob
   import mlflow as _mlflow
   import mlflow.lightgbm as _mlflow_lgb
   import lightgbm as _lgb

   _HERE     = os.path.dirname(os.path.abspath(__file__))   # code/
   _RUN_DIR  = os.path.dirname(_HERE)                       # run root
   _PROJ_DIR = os.path.dirname(os.path.dirname(_RUN_DIR))   # General Plan/
   _MLDB     = os.path.join(_PROJ_DIR, "mlflow.db")

   _mlflow.set_tracking_uri("sqlite:///" + _MLDB.replace("\\", "/"))
   _mlflow.set_experiment("bread_demand_forecast")

   _RUN_ID = os.path.basename(_RUN_DIR)
   _ITER   = os.path.splitext(os.path.basename(__file__))[0]  # e.g. iteration_005_training_ml_regression

   with _mlflow.start_run(run_name=f"{_RUN_ID}__{_ITER}"):
       # ── tags ──
       _mlflow.set_tags({
           "pipeline_run_id": _RUN_ID,
           "iteration_id":    _ITER,
           "mode":            MODE,            # set earlier in this script
           "model_type":      MODEL_TYPE,      # set earlier in this script (e.g. "LightGBM")
       })

       # ── params ──
       _mlflow.log_params({k: v for k, v in LGBM_PARAMS.items()
                           if isinstance(v, (int, float, str, bool))})
       _mlflow.log_param("n_features",           len(SAFE_FEATURES))
       _mlflow.log_param("early_stopping_rounds", EARLY_STOPPING_ROUNDS)

       # ── per-horizon metrics ──
       _pooled_maes = []
       for _h, _mae in VAL_MAE_BY_HORIZON.items():   # dict populated during training loop
           _mlflow.log_metric(f"val_mae_h{_h}", _mae)
           _pooled_maes.append(_mae)
       if _pooled_maes:
           _mlflow.log_metric("val_mae_pooled", sum(_pooled_maes) / len(_pooled_maes))

       # ── artifacts: plots ──
       _art_dir = os.path.join(_RUN_DIR, "artifacts", "training")
       for _png in _glob.glob(os.path.join(_art_dir, f"{_ITER}_*.png")):
           _mlflow.log_artifact(_png, artifact_path="plots")

       # ── models: LightGBM text dumps → MLflow model registry ──
       _models_dir = os.path.join(_art_dir, "models")
       for _h in HORIZONS:
           _dump = os.path.join(_models_dir, f"lgbm_h{_h}.txt")
           if os.path.exists(_dump):
               _booster = _lgb.Booster(model_file=_dump)
               _mlflow_lgb.log_model(_booster, name=f"lgbm_h{_h}")
   # ── end MLflow logging ───────────────────────────────────────────────────
   ```

   **Adapt the snippet to the model family being trained:**
   - For classical statistical (ARIMA/SARIMA): replace the LightGBM params block with the
     fitted orders (`arima_p`, `arima_d`, `arima_q`, `sarima_P`, `sarima_D`, `sarima_Q`, `sarima_m`,
     `arima_aic`, `sarima_aic`). Log per-horizon MAE under `arima_val_mae_h{h}` and
     `sarima_val_mae_h{h}`. Omit the model registration block (statsmodels ARIMA results files are
     logged as plain artifacts instead via `mlflow.log_artifact`).
   - For baseline (no learned model): log only per-horizon MAE from `baseline_results` under
     `val_mae_h{h}`. No params or model registration.
   - For any model family with a sklearn-compatible `.predict()`: use `mlflow.sklearn.log_model()`
     in place of the LightGBM call.

   **Required dict `VAL_MAE_BY_HORIZON`:** the training loop must populate a `dict[int, float]`
   named exactly `VAL_MAE_BY_HORIZON` (keys = horizon integers) so the MLflow block can read it
   without knowledge of the loop's internals. Add `VAL_MAE_BY_HORIZON: dict[int, float] = {}`
   before the training loop and `VAL_MAE_BY_HORIZON[h] = float(val_mae)` inside it, immediately
   after computing `val_mae` for horizon `h`.

   **Guardrails for this step:**
   - The MLflow block must be the last thing the script does — never place it before a model save
     or a history-JSON write, so a logging failure cannot corrupt outputs.
   - All paths are constructed from `__file__` — never hard-code an absolute path.
   - The experiment name `"bread_demand_forecast"` must be constant across all calls in a project
     so all pipeline runs land in the same MLflow experiment and can be compared side by side.
   - `mlflow` and `lightgbm` must already be installed in the sandbox (they are listed under
     "Tools available"). Do not add `pip install` calls to generated training code.


## Handoff block (required)

Before writing the output, add a top-level `handoff` block to this node's
`history/iteration_<NNN>_<node>.json`. **The field rules, the required fields, and the
externalisation pass that must be run before finishing are all in
`conventions/handoff_contract.md`** — that file is the specification, and it is what
`tools/handoff_check.py` validates against.

Two things worth restating here because they are the failures this node can actually cause:

- A finding recorded in `agent_initiated_findings` that changes what a later node should do must
  be duplicated into the handoff. The two serve different readers: one is the honest record of
  what was explored, the other is the instruction set for whoever runs next.
- A node that genuinely changes nothing downstream still writes the block, with empty lists and
  `context_clearable: true`. An absent block is indistinguishable from a forgotten one.

## Guardrails

- **Generated code is a cell-marked `.py`, and every plot names itself**
  (`conventions/code_and_plot_conventions.md`). Scripts carry `# %%` markers so the same file
  stays diffable and converts to a notebook on demand — never write a `.ipynb` as the artifact.
  Every figure carries its declared `name` as the title, its declared `type` stated in the
  figure, and its `iteration - node` provenance, using the same strings recorded in the JSON. A
  plot that is unidentifiable once separated from its report is not finished.
- **Never end a call without a `handoff` block** (`conventions/handoff_contract.md`). Context may
  be cleared immediately after this node returns; anything load-bearing that exists only in prose,
  only in a code file, or only in `agent_initiated_findings` is lost at that point. Never store a
  count where a later node needs the content.
- **There are no held-out rows in this dataset to protect.** The genuinely held-out set was
  removed before the run began and is scored outside this pipeline. What must be protected here
  is the validation partition: it is the only ranking signal the run has, and it is reused on
  every iteration, so treat it the way a test set would be treated within a single call — fit
  nothing on it, tune nothing directly against a single reading of it, and prefer the
  cross-validated estimate where `split_info.cv_design` provides one.
- **Never use random cross-validation (random k-fold, stratified k-fold) for time-series data.**
  Random folds allow future rows to appear in training folds and past rows in validation folds,
  leaking temporal structure. Always use `TimeSeriesSplit` or the fixed chronological split. See
  `references/cross_validation.md` for the full CV-strategy decision table and the reasoning
  behind it.
- **Never fit any statistic on validation rows** — including scaling, imputation, or any
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
- **Never report a per-feature importance ranking as the top-level answer, and never set
  `near_zero_flag` from a column's solo score.** Both are the identifiability failure that
  `../shared-references/feature_attribution.md` exists to prevent: under correlation, a per-feature
  number describes which twin the fit reached first, not what the model depends on.
- **Never present any number this node produces as final performance.** Every metric here is a
  validation metric and must be labelled as one. The unbiased held-out score is produced after
  the pipeline finishes, against data this run never had.
- **Respect the selection-noise band.** Where `split_info.selection_noise_band` is set, a
  difference smaller than it is a tie, not an improvement — say so rather than ranking on it.
  This run is expected to compare many candidates, and without this guard the winner is
  partly whichever model happened to suit one validation draw.
- For MASE: always compute the denominator on training data only. Using the full dataset to
  compute the naive baseline denominator would let future demand levels influence what counts as
  a "good" score on past data.
- Every artifact plot must have a paired CSV. Never load that CSV into your own reasoning
  context.
- Every decision — model family, hyperparameter, metric choice — needs `reasoning` and
  `confidence`. A decision recorded only as "default" is an unrecorded assumption.
- The `reports/` Markdown file must only restate what is already in the JSON output.
- **Never train one model for several horizons.** Each active horizon gets its own fit, its own
  feature subset, its own target column and its own artifact. A metric pooled across horizons is
  never reported without the per-horizon table beside it.
- **Never use `target_horizons` on a baseline call.** The floor must be established for every
  horizon in `forecast_horizons` at once — a horizon missing a baseline cannot be stopped against.
- **Never use a feature whose `min_horizon` exceeds the horizon being trained.** A lag-1 feature
  in a 30-day-ahead model is not a feature, it is a value nobody will have at prediction time —
  the same failure as a concurrent feature in a live forecast, one lead time further out.
- **Never use a feature whose `known_future_horizon` is below the horizon being trained, and
  never substitute its origin value instead.** This is the reciprocal bound and the one more
  easily missed, because nothing about the stale value looks wrong: origin weather at `h = 90`
  is a well-formed number that carries no information about the target, and a model handed it
  will spend capacity on noise. Drop the feature for that horizon. Read
  `evaluated_at` and honour it: a feature marked `target_date` is joined on the target date,
  not the origin, and reading it at the wrong anchor silently changes what the column means.
- **Never drop, prune, or exclude a column listed in `context.protected_features`** from the
  horizon it belongs to. Not for redundancy, not for near-zero importance, not for a >0.98
  correlation with the target — that correlation is the feature working. Evidence against one is
  raised as a `blocking` clarification and answered by the user, never acted on here.
- **Never feed a naive predictor to a horizon other than its own.** `naive_h7_*` is computed at
  origin `target − 7`; in the `h = 30` model it reads the series 23 periods after that model's
  origin. The resulting model scores better on validation and worse on the holdout, which is the
  hardest failure in this pipeline to detect after the fact.
- **Take `horizon_feature_subsets` as given; never re-derive it here.** If a horizon's subset
  looks wrong, that is a finding routed back to Feature Engineering, not something to patch
  in the training loop — two nodes deciding the same thing is how they drift apart.
- **Default hyperparameters are a starting point, never the run's final answer.** Fitting on
  library defaults for a first pass is correct and cheap. Closing the run without a single search
  is not: it leaves the largest cheap axis unmeasured, and `tools/stop_check.py` blocks the stop
  for exactly this reason. "Re-tuning would invalidate the baseline comparison" is not a reason —
  a baseline is a fixed reference and does not move when a model is tuned. Where a search is
  genuinely inadvisable (too few rows, a validation window too small to select on), say so in
  `hyperparameter_search.reasoning` with the row count that makes it so, rather than leaving the
  field null and the axis silently closed.
- **Never report a time-series iteration without all four of MAE, RMSE, SMAPE and MASE**, on
  train and validation both, whichever one is primary. Choosing the primary is the agent's call;
  computing the other three is not optional. Where they disagree on the ranking, say so.
- **Never let a time-series run consist only of ML-as-regression fits.** Both non-automatic
  approaches in `references/model_selection_timeseries.md` §1 must be fitted at least once, on the
  same validation rows and the same primary metric. This node does not choose when — the Planner
  routes — but it does report the gap (`ts_approach_gap`) every call until it is closed.
- **Never close the classical slot without an ARIMA-family fit.** ARIMA, SARIMA, or SARIMAX —
  exponential smoothing alone does not close it, and neither does an Auto ARIMA run whose orders
  were never recorded. Where the series' structure sends the choice to SARIMA/SARIMAX, plain
  ARIMA is still fitted once as the ablation that prices the seasonal terms.
- Never decide on your own to pause the workflow — raise `clarification_needed` and let the
  Planner decide.

## Input / output contract

Full input/output example: `contracts/training.json`.

On a multi-horizon task every model-specific section below is recorded **per horizon**, keyed by
horizon — `trained_model_path`, `model_family`, `model_class`, `hyperparameters`,
`feature_subset_used`, `metrics`, `feature_importance`, `residuals_summary`, `vs_baseline_floor`
— alongside `horizons_trained_this_call` (the active set for this call) and
`horizons_skipped_this_call` (horizons in `forecast_horizons` not touched this call — empty list
when `target_horizons` was absent). Sections that are genuinely shared (`deployment_scenario`,
the primary metric choice) stay single and say so.

Top-level output sections: `schema_version`, `trained_model_path`, `model_family`,
`model_class`, `model_selection_reasoning` (which decision-table rows fired and why — or override
source if step 2a applied), `ts_approach` (time-series calls only: `classical_statistical`,
`ml_as_regression`, or `automatic` — step 2b-ts), `ts_approach_gap` (the non-automatic approach
the run has still never fitted, or null), `feature_subset_used`, `feature_subset_type` (`concurrent_included` or
`pre_period_only`), `deployment_scenario`, `hyperparameters` (each with `value` and `reasoning`),
`hyperparameter_search` (method, n_trials, search_space — null if defaults used),
`training_summary` (n_train_rows, n_val_rows, n_features, training_time_seconds,
`target_transform` — the transform applied to the target column, or null if none),
`metrics` (train and validation objects, each with all computed metrics; for time-series always
include `mae`, `rmse` and `smape` — the last with its zero-handling convention and the affected
row count — alongside `mase` with `seasonal_period`, `naive_baseline_mae_train`, and `mase_value`,
regardless of which is primary),
`naive_predictor_features` (present when `mode == "baseline"`: the per-horizon predictor specs
from `baseline-skill/baseline.md` Part B, with the search grid, the deduplication result and
`selection_basis: "train_rows_only"`),
`baseline_floor` (present when `mode == "baseline"`: `baseline_type`, the rule or seasonal period,
the full metric set pooled and per segment/horizon), `vs_baseline_floor` (every other call: margin
per segment and per horizon, and whether the model beats the floor everywhere or only pooled),
`baseline_model` (present when `mode == "initial"`; contains `baseline_type`,
`train_<primary_metric>`, `val_<primary_metric>`, and `vs_primary_model` assessment: one of
`primary_better`, `primary_worse` — flag as finding if `primary_worse`; null in `incremental`/
`rebuild` mode unless Diagnosis requested a re-comparison),
`overfitting_check` (ratio, assessment string),
`feature_importance` (`groups`: ranked list, each entry: group_name, members, importance, importance_sd, sum_solo, masked, rank, near_zero_flag; `per_feature`: solo permutation value per column, reported only inside its group; `builtin_importance`: optional, labelled as a build diagnostic — see `../shared-references/feature_attribution.md`),
`residuals_summary` (val_residuals_path, mean, std, p5, p95; for time-series: acf_lag1,
acf_seasonal_lag, temporal_bias_by_period),
`clarification_needed`, `agent_initiated_findings`, `issues`,
`artifact_paths` (plots list, each with path and data_path).