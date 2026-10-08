---
name: preprocess-agent-skill
description: Cleans, encodes, scales, and imputes existing raw fields for a specific, Planner-scoped subset of features. Never creates new features (Feature Engineering's job) and never decides feature relevance (also Feature Engineering's job) — Preprocess only acts on columns it's told to process, following EDA's (or a later node's) recommendations, and can override them with its own stated reasoning.
---

# Preprocess Skill

## Purpose

Given a specific set of raw or partially-processed columns and the relevant findings about them
(from EDA or a later diagnostic step), clean, encode, scale, and impute those columns — never the
whole dataset at once, and never columns outside the current scope. This is what turns your
supervisor's "preprocess one feature first" into a node the Planner can call repeatedly with a
growing scope, rather than a single batch step.

## Reference material

- `references/transformation_catalog.md` — the five per-feature sub-decisions and their technique tables (step 5)
- `references/report_example.md` — a worked example of the report this node writes
- `../shared-references/preprocessing_and_feature_engineering.md` — model-specific scaling, encoding and NaN requirements, owned jointly with Feature Engineering and Training
- `conventions/handoff_contract.md` — the handoff block's fields and the externalisation pass
- `conventions/naming_conventions.md` — where every file this node writes goes, and what it is named
- `conventions/code_and_plot_conventions.md` — cell-marked scripts; every plot names itself

## Inputs

- `dataset_path` — the dataset to start from *this call*. What this actually points to depends
  on `mode` (see below) — the Planner resolves this, Preprocess doesn't need to know why.
- `target_column` — provided for reference only. **Preprocess never transforms the target column**
  — see Guardrails.
- `features_to_process` — the specific column(s) to work on this call. Never "all columns."
- `mode` — one of:
  - `initial` — first Preprocess call this run; `dataset_path` is the raw dataset.
  - `incremental` — build on this node's own prior output; `dataset_path` is the most recently
    processed file, and `already_processed` lists what not to redo.
  - `rebuild_from_raw` — start over from the raw dataset even though prior processed output
    exists, because whatever triggered this call concluded the existing approach needs to change,
    not just extend. `dataset_path` is the raw dataset in this case too.
  This is the Planner's judgment call each time, based on whatever triggered the call (a
  Diagnosis hypothesis, a user clarification, or anything else) — it is not tied to any specific
  upstream node.
- `mode_reasoning` — why this mode was chosen this call.
- `context` — prior reasoning relevant to this call, per `conventions/naming_conventions.md`
  (a `previous_step` entry plus zero or more `planner_selected` entries). This is where EDA's
  missingness/flags findings and any triggering hypothesis arrive.
- `already_processed` — which columns are already handled (only meaningful in `incremental` mode).
- `rows_to_filter` — zero or more Planner-authorized row-filtering instructions for this call
  (e.g. `{"condition": "num_orders < 10", "reason": "...", "source": "eda iteration 1
  data_quality_issues.out_of_range_values, confirmed non-representative per user clarification"}`).
  This is dataset-wide, not scoped to `features_to_process` — a row that's excluded is excluded
  regardless of which feature happens to be in scope this call. Always empty unless the Planner
  has explicitly authorized a filter, typically after a blocking clarification was resolved.
- `iteration` — the global iteration number, used to build every output path.

## Tools available

- Code execution sandbox with `pandas`, `numpy`, `scikit-learn` (preprocessing transformers)
- Plotting library for before/after artifact generation only — same "numbers, not images" rule as
  EDA: every plot gets a paired CSV of the actual values plotted, per
  `conventions/naming_conventions.md`
- File write access to `artifacts/preprocess/`, `data/`, `history/`, `code/`, and `reports/`
- Read-only access to `state/current.json` and any file under `history/`, per
  `conventions/naming_conventions.md`, for anything not already covered by `context`

## Procedure

**Every step below is a floor, not a ceiling — and so is the procedure as a whole**, the same
rule EDA follows. If a technique or an entire additional step beyond what's specified here would
genuinely help for this specific dataset, pursue it — this doesn't excuse skipping what's
actually required, it's permission to go further when warranted. Record anything beyond what's
specified in `agent_initiated_findings`, never blended into the standard output fields as if it
were part of the prescribed procedure — same shape as EDA's version: `related_step`, `title`,
`method`, `findings`, `reasoning`, `confidence`, with `"new"` for something that isn't an
extension of any listed step.

0. **Combine the sources first, when EDA's `join_plan` says to.** In `initial` and
   `rebuild_from_raw` mode, if `context` carries a `join_plan` with more than one `used` source,
   building the combined table is this call's first act — before row filtering, before any
   per-feature work. Execute the plan's joins on its stated keys and granularity, then record a
   `join_result`: rows in and out per source, the realised match rate against EDA's estimate, how
   many rows gained nulls from a left join, and any key that multiplied rows. A match rate well
   below EDA's estimate is an `issues` entry and usually a `clarification_needed` — not something
   to absorb quietly by dropping the unmatched rows. The combined table is what the rest of this
   call, and every downstream node, treats as the dataset. In `incremental` mode the combination
   has already happened; do not redo it.

1. **Resolve the working dataset per `mode`.** Load `dataset_path` as given. In `incremental`
   mode, confirm `already_processed` matches what's actually already reflected in that file
   before proceeding — if it doesn't, treat this as an `issues` entry rather than silently
   proceeding on a mismatched assumption.

2. **Apply row filtering, if `rows_to_filter` is non-empty.** This is dataset-wide — remove the
   matching rows before anything else happens, so the train/validation split (next step) is created
   only over rows that will actually be used, and per-feature statistics later never get skewed
   by rows that are about to be excluded anyway. **Never filter rows on your own initiative** —
   only ever act on an explicit `rows_to_filter` instruction from the Planner. If EDA's output
   flagged a row-level anomaly but no filtering instruction was given this call, leave those rows
   in place; that flag not yet being acted on is the Planner's call, not something to infer.
   Record what was removed (condition, row count, reason) — never silently drop rows without
   this being visible in the output.

3. **Resolve the train/validation split.** Check whether a `_split` column (values
   `"train"`/`"validation"`) already exists for this dataset. If not, this is the first
   time any node has needed one for this run — create it now.

   **The dataset you were given is already a training partition.** The true held-out set was
   carved off before this run started and is not present in `dataset_path`, not reachable from
   it, and not described anywhere this node can read. Everything below therefore splits *what
   you have* into two parts, `train` and `validation`, and nothing else. **Never create a
   `"test"` partition, never name a partition `test`, and never hold rows back "for a final
   check" — that check happens outside this pipeline, on data this run will never see.** A third
   partition carved out here would not be a held-out set; it would just be a smaller validation
   set with a misleading name, and every number quoted against it would read as more final than
   it is.

   **First check EDA's `grouping_structure`.** If rows are not independent (e.g. the same
   patient appears across multiple rows), the split must be group-aware — every row belonging to
   the same group goes entirely into one partition (all of a given patient's rows in train, or
   all in validation, never split across both). Split by the group
   identifier first, then assign whole groups to partitions using the method below — never split
   by individual row when a `grouping_structure` was found, regardless of task type. If no
   grouping structure was found (or EDA didn't report one), split by row as usual.

   The table below is the **starting point, not the answer.** It says what the task type
   requires; it does not know anything about this dataset. Read it, then reason about whether it
   is right *here* — and record that reasoning either way, including when you follow the table
   unchanged. A split is the single most consequential choice in the run: every metric anyone
   later quotes is conditional on it, and unlike a feature or a hyperparameter it is fixed for
   the whole run and cannot be revisited without invalidating everything measured against it.

   | Task type | Split method |
   |---|---|
   | Classification (binary or multiclass) | Stratified two-way split, stratified on the target so both partitions preserve the same class distribution |
   | Regression, not time series | Random two-way split |
   | Time series / sequential | Chronological two-way split — earlier portion as train, most recent as validation, never random. **Prefer rolling-origin (walk-forward) cross-validation over a single static cut** whenever the series is short or noisy, and treat it as **required** where the run has identified a regime change: a single static cut cannot both keep validation recent enough to be representative and keep the new regime inside training, and rolling origin is what resolves that conflict — every fold trains on everything before it, so the last fold trains through the regime. See the note on selection noise and the regime-coverage check below |

   Typical proportions: roughly 75–85% train, 15–25% validation — adjust for dataset size (a
   small dataset may need a larger validation *percentage* just to get a statistically
   meaningful row count; a large dataset can use a smaller percentage since the absolute count
   stays large). These are larger train shares than a three-way split would give, because the
   held-out partition is no longer being carved out of this data.

   **Multi-horizon forecasts need a gap, sized by the longest horizon.** A row carries two dates:
   the **origin** it is predicted from and the **target** it predicts, `h` periods later. A
   chronological cut on origins alone leaks: a train row at origin `T` forecasting `h` ahead has
   its answer at `T + h`, and if that date falls inside the validation window the model was fitted
   on a value validation is about to score.

   **The rule: no train row's target date may fall on or after the first validation target.**
   Equivalently, stop train origins `max(forecast_horizons)` periods before validation's first
   origin, and drop the train rows in between. Record it as `split_info.horizon_gap` with the
   value, the horizon that set it, and the row count dropped.

   Be precise about which two endpoints that gap sits between, because getting it wrong is
   expensive in one direction and unsafe in the other. It is `max(h)` between the last train
   **origin** and the first validation **origin** — which leaves the last train *target* landing
   just before the first validation *target*, a gap of one period in target space. Sizing it as
   `max(h)` in *target* space instead discards a further `max(h)` periods of perfectly usable
   training data at the most recent end of the series, which is also the most relevant end.

   State the cost either way: at `h = 90` on a daily series the gap alone consumes a quarter of a
   year of origins. One split serves every horizon; do not build a different partition per horizon,
   or the horizons stop being comparable to each other.

   **Selection noise — read this before choosing a single static window.** Every candidate this
   run produces will be ranked on whatever partition you create here, and the run is expected to
   try many. With one fixed window and enough candidates, the winner is partly whichever model
   happened to suit that window — the same "teaching to the test" failure the old third
   partition existed to absorb, now relocated onto validation. Two defences, and you should
   supply at least one:

   - **Rolling-origin or k-fold CV instead of a single cut**, so each candidate is scored on
     several windows rather than one. Record the fold definition in `split_info.cv_design`.
   - **A noise band**: the spread of the primary metric across folds (or across neighbouring
     configurations, if only one window is available). Record it as
     `split_info.selection_noise_band`. Downstream, Diagnosis and Training read this and must
     not call a difference smaller than the band an improvement — it is a tie. **Record the band
     per horizon** on a multi-horizon task: long horizons are noisier and have fewer scored rows,
     so a single pooled band calls real gains at h=1 noise and noise at h=30 a gain.

   If neither is possible, say so explicitly in `split_info.reasoning` and state the
   consequence: candidate rankings from this run are provisional and differences between the top
   few should be treated as unresolved.

   **Reason about the split before you fix it.** Work through these and record an answer to each
   under `split_info.reasoning` — a sentence apiece is enough, but a blank is not:

   - **What deployment situation is this split simulating?** Name it. "Forecasting forward into a
     period the model has not seen" is a different simulation from "scoring new customers drawn
     from the same population", and they justify different splits.
   - **Is the validation partition a fair stand-in for the unseen holdout?** This is the question
     that decides whether model selection can work at all, and it is harder here than it looks,
     because the holdout cannot be inspected — it sits after the end of this dataset and this
     node has no access to it. So reason about it indirectly. Compute
     `split_info.level_gap_pct` — for regression, validation's target mean as a percentage
     difference from train's; for classification, validation's class balance against train's.
     Then ask whether the *trend* that gap sits on is likely to continue past the end of the
     data. **If the series is drifting, state plainly that validation sits closer to train than
     the holdout will, and that every accuracy number this run produces is therefore optimistic
     by an unknown margin.** Downstream nodes need that sentence — Training step 8b and Feature
     Engineering step 2b both read it. Record the answer as
     `split_info.holdout_representativeness`, and never assert the holdout resembles validation
     as if it had been checked.
   - **Is every regime the run identified represented in the TRAINING rows?** Not "does the
     boundary land on a regime change" — the harder case is a regime change that falls entirely
     *after* the boundary, which leaves training with none of it while the unseen holdout is
     made of nothing else. Count the rows per regime per partition and record it as
     `split_info.regime_coverage`. **If the training partition contains 0% of a regime that the
     validation partition or the expected holdout contains, the split is rejected** — move to
     rolling origin so the last fold trains through the change, or move the boundary later and
     accept a shorter validation. Say which you did and what it cost.

     A model cannot learn a regime it has no examples of, and the failure is quiet: the metrics
     on a validation window that straddles the change still look reasonable, because the
     pre-change rows carry them.

     > `run_20260904T140437Z` identified the 2020-06-01 regime change itself, recorded it in
     > `confirmed_decisions`, and then split so that training was **0% post-regime**, validation
     > 32.6%, and the holdout 100%. Its per-horizon models learned a weekday relationship that
     > held only under the old six-day week and was wrong on every holdout row. Refitting the
     > same models on data that merely *included* seven months of the new regime recovered
     > 8.5 MAE, with nothing else changed.

   - **Does either boundary fall somewhere meaningful?** A split that happens to cut at a regime
     change, a policy change, or a data-collection change is not a neutral 70/15/15 — it is a
     decision about which regime the model learns. Report `regime_exposure` per partition when
     EDA flagged a break.
   - **What would a reasonable person split differently, and why did you not?** If the table's
     method is wrong for this dataset, deviate from it and justify the deviation here. The table
     is a floor.

   Everything above goes in the Markdown report as well as the JSON — a named
   **"Split design and its consequences"** section, not a line in a table. Whoever reads the
   run's final numbers needs to be able to see what those numbers are conditional on without
   opening the JSON.

   **Why two, not three:** re-evaluating against the same held-out data on every iteration —
   which is exactly what this loop's Diagnosis step does to decide what to try next — causes the
   model to implicitly overfit to that data over repeated iterations, the same failure mode as
   "teaching to the test." Validation exists precisely to absorb that repeated, iterative use.
   The genuinely unbiased number comes from the external holdout, scored once, after this
   pipeline has finished and by someone other than this pipeline. That is the whole reason the
   holdout was removed from the data before the run began: a set this pipeline cannot read is a
   set it cannot quietly tune against, which is a stronger guarantee than a rule telling it not
   to.

   The consequence to keep in mind: **no number this run produces is a final number.** Every
   metric in every report from this run is a validation metric, and must be labelled as one.

   Persist the split as a column in the processed dataset. Once created, this split is fixed for
   the rest of the run — every later Preprocess call and Training itself reuse the exact same
   assignment, never re-splitting. This is what makes it safe for both Preprocess (fitting
   statistics) and Training (resampling within CV folds) to avoid leaking validation-row
   information into anything fit on "train" data — they're all working from one shared,
   consistent partition.

4. **Read `context` for what's relevant to `features_to_process`.** Pull each feature's
   missingness finding, any data-quality flags, and the reasoning behind whatever triggered this
   call. You may follow EDA's `suggested_action` as given, or override it — but an override needs
   its own `reasoning` and `confidence`, same as any other judgment call here.

5. **For each feature in `features_to_process`, apply these in order. Every statistic fit here —
   an imputation value, a scaler's mean/std, an encoding map — must be fit using only
   `_split == "train"` rows, then applied (transformed) to every row regardless of its split.
   This
   is the leakage-prevention rule from `conventions/naming_conventions.md` applied concretely:
   the validation rows must never influence what "normal" looks like. Each sub-decision below (b–e) is
   a separate judgment call and needs its own `reasoning` and `confidence` — do not bundle them
   into one confidence for the whole feature. You might be very sure about the imputation method
   and much less sure about the scaling choice; collapsing that into a single number loses
   exactly the information a downstream reader would need.**

   The five sub-decisions (a. normalise categories, b. missingness, c. outliers,
   d. scaling, e. encoding) and the technique table for each are in
   `references/transformation_catalog.md`. Each is a separate judgment call and needs
   its own `reasoning` and `confidence`.

6. **Drop flagged columns in scope.** If a column in `features_to_process` is flagged as leakage
   or PII in `context`, drop it and record it in `dropped_columns` with the reason. Preprocess
   acts on flags as given — if a flag was ambiguous enough to need human confirmation, that
   confirmation already happened via the Planner before this call was made; Preprocess doesn't
   re-litigate it.

7. **Raise `clarification_needed` for genuine ambiguity encountered during this call** — not
   things already resolved via `context`, but something new this node hits mid-execution that
   it isn't confident enough to decide alone (e.g. two plausible encoding strategies with no
   clear winner given what's known, or a value pattern that looks like it needs domain input).
   Same shape and rules as EDA's: each entry needs a `blocking` flag (true only if proceeding
   without an answer could materially change the outcome), a `confidence`, and a `recommendation`
   with its `recommendation_reasoning` and `impact_if_wrong` — the field table is in the Planner's
   **Clarification Handling** section. Two plausible encoding strategies with no clear winner is
   still a question with a leaning; name it. Like EDA, Preprocess never decides on its own to
   pause the workflow — raising a clarification is not the same as stopping, and a blocking entry
   halts the run at the Planner, not here.

8. **Compute `post_transform_stats`** for each processed feature (missing_pct, skew, outlier
   count, or whatever's relevant to what changed) so the transform can be verified against its
   `before` state, not just trusted from the description. **Recompute the outlier count using
   the same method EDA declared** (`feature_stats_summary.numeric.outlier_detection_method` in
   its output) — comparing a before-count from IQR against an after-count from a different
   method would make the before/after comparison meaningless, not just imprecise.

9. **Generate a before/after plot per transformed feature**, following the same rule as EDA: the
   image is for the report, and the actual before/after values are saved as a paired CSV — never
   only the image.

10. **Record any execution failures in `issues`** — a transform that errored, a column that didn't
   parse as expected — rather than silently producing incomplete output.

11. **Write the output.** Save the full JSON to `history/iteration_<NNN>_preprocess.json`, the
    generated code to `code/iteration_<NNN>_preprocess.py`, and the processed dataset to
    `data/iteration_<NNN>_preprocess.csv`.

12. **Generate a human-readable report.** Write a short Markdown report to
    `reports/iteration_<NNN>_preprocess.md`, separate from the JSON — for a person to read
    directly, not for another node to consume. This must cover every section of the output, not
    just a summary of the highlights:

    - **Call context:** `mode` and `mode_reasoning`; which features were in `features_to_process`
      this call.
    - **Rows filtered:** each entry in `rows_filtered` — the condition, how many rows were
      removed, how many remain, and the reason. State plainly if none were filtered this call.
    - **Split design and its consequences:** a named section of its own, not a line in a table.
      Whether the split was created this call or reused, the method (stratified, time-based,
      etc.), the train/validation proportions, and the CV design and selection-noise band if one
      was set — and then the reasoning from step 3: what
      deployment situation this split simulates, the `level_gap_pct`, whether
      validation is a fair stand-in for the unseen holdout and what follows if it is not, whether either boundary
      lands somewhere meaningful, and what a reasonable person might have split differently.
      Every number the run later reports is conditional on this, so a reader must be able to see
      what they are conditional on without opening the JSON.
    - **Per-feature transformations:** for every entry in `transformations_applied` — the
      missingness handling (method, value used, whether a missing indicator was added), outlier
      handling, scaling/transform method, encoding method where applicable, the `reasoning`, and
      the `confidence`. State explicitly that statistics were fit on train-split rows only.
    - **Before/after comparison:** a table or list contrasting each feature's `post_transform_stats`
      against what it looked like beforehand (missing_pct, skew, outlier count, or whatever
      changed) — not just a link to a plot, the actual before/after numbers themselves.
    - **Dropped columns:** every entry in `dropped_columns` with its reason.
    - **Generated artifacts:** every plot in `artifact_paths.plots`, each with its `reason` and a
      link to both the image and its paired values CSV.
    - **Issues:** every entry in `issues`, or state plainly that none occurred.
    - **Open questions:** every entry in `clarification_needed`, with its `blocking` status, or
      state plainly that none arose this call.

    Keep this readable on its own — someone should be able to reconstruct exactly what this run
    did without opening the raw JSON, not just get a general impression of it. This report is a
    byproduct of the run, not a substitute for the structured output — if something is in the
    JSON, it belongs in the report too.

    **Write explanations, not just facts.** A number or a method name on its own doesn't tell a
    reader why it mattered. Don't just state "median imputation, value 142.0" — explain why
    median was chosen over mean here, what the missingness pattern suggested, and what would have
    gone wrong with a different choice. Don't just report "skew dropped from 1.4 to 0.3" — say
    what that means practically (the distribution is now close to symmetric, which is what the
    downstream scaling step assumed). Every method, bound, or number in the report should be
    followed by a sentence of *why*, the same depth of reasoning that's in the JSON's `reasoning`
    fields, not a stripped-down restatement of just the JSON's field values.


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
- Anything explored beyond what a step specifies must go in `agent_initiated_findings`, never
  blended into the standard fields — and needs the same `reasoning` and `confidence` discipline
  as everything else in this skill.
- Never process a feature outside `features_to_process`. Scope discipline is what makes this node
  callable incrementally — processing more than asked defeats that.
- Never create a new feature (a derived/combined column). That is Feature Engineering's job —
  Preprocess only transforms columns that already exist.
- Never transform `target_column`. If a target-side transform is genuinely needed (e.g.
  log-transforming a skewed regression target), that decision and its inverse-transform handling
  belongs to Training/Diagnosis, which own metric interpretation — keeping it out of Preprocess
  avoids splitting that responsibility across two nodes.
- Never fit an imputation value, scaler parameter, or encoding map using `_split == "validation"`
  rows. Fit on `_split == "train"` rows only, then apply the result to
  every row. This is the single most common preprocessing mistake per the sources reviewed for
  this skill — treat it as a hard rule, not a preference.
- **Never create a `test` partition.** The genuinely held-out data was removed before this run
  started and is not in `dataset_path`. A partition named `test` created here would be a
  validation set wearing a more authoritative name, and every downstream node would read it as
  the final honest number it is not. Two partitions, `train` and `validation`, and no others.
- No node in this pipeline produces a final performance number. Per-iteration decisions
  (Diagnosis, and whatever computes the metrics it reasons over) use `_split == "validation"`;
  the unbiased number is produced outside this pipeline against the external holdout, after the
  run has finished. Label every metric this run emits as a validation metric.
- Never perform class-imbalance resampling (oversampling, undersampling, SMOTE, or similar) in
  this node. Resampling must happen inside Training's cross-validation loop, applied per-fold to
  training rows only — doing it here, once, globally, risks resampling rows that later end up in
  a validation fold, which corrupts evaluation. Preprocess's job stops at recommending
  that resampling may be warranted (echoing EDA's `target` findings) — Training executes it.
- Never filter or drop rows on your own initiative, even if `context` clearly shows EDA flagged a
  row-level anomaly. Only act on an explicit `rows_to_filter` instruction from the Planner — the
  decision to act on an ambiguous, potentially `blocking` finding belongs to the Planner (and
  often a human), not to Preprocess inferring it from what it can see in EDA's output.
- Never decide encoding, scaling, or imputation arbitrarily — follow the decision tables above,
  or state a clear `reasoning` for departing from them. Every sub-decision (missingness
  handling, outlier handling, scaling, encoding) needs its own `reasoning` and `confidence`, per
  `conventions/naming_conventions.md` — never one bundled confidence for the whole feature.
- Never leave a structured field `null` without explanation. A field like `bounds` on a clip
  operation or `value_used` on an imputation must always be populated when the method implies a
  value exists — a `null` is only acceptable when it's structurally inapplicable given the
  method (e.g. `value_used: null` when `method: "none_needed"`), the same exception EDA uses.
- **Never silently drop rows that failed to join.** An unmatched key is a finding — record the
  count and the realised match rate against EDA's estimate, and raise it rather than letting an
  inner join quietly shrink the dataset.
- Never decide on your own to pause the workflow — raise genuine ambiguity via
  `clarification_needed` and let the Planner decide whether to act on it, same rule as EDA.
- Never silently drop a flagged column without recording it in `dropped_columns`.
- **Never drop a column listed in `context.protected_features`**, even for a data-quality reason
  this node would normally act on without asking. These are the per-horizon naive predictors
  (`../baseline-skill/baseline.md` Part B) and anything else the Planner has protected. Preprocess
  usually runs upstream of where they are created and will not see them; when it does see them —
  a later Preprocess call on an engineered dataset — record the finding and raise a `blocking`
  clarification instead of acting. A genuine data-quality problem in one of these columns means it
  was built wrong, which is a Feature Engineering fix, not a drop.
- Never generate a before/after plot image without its paired values CSV, per
  `conventions/naming_conventions.md` — and never load that CSV into your own reasoning context;
  it's for other nodes to load with code.
- If EDA reported a `grouping_structure`, the split must be group-aware — never split by
  individual row in that case. A group's rows ending up in more than one partition is a real
  leakage path (e.g. a model partly memorizing a specific patient from training, then being
  "tested" on that same patient) even when row-level splitting was otherwise done correctly.
- If `mode` is `incremental` and the loaded dataset doesn't actually match `already_processed`,
  treat it as an `issues` entry — don't proceed on a silently mismatched assumption.
- The `reports/` Markdown file must only restate what's already in the JSON output — never state
  a finding, number, or decision in the report that isn't also present in the structured output.

## Input / output contract

Full input/output examples: `contracts/preprocess.json` (`mode: initial`),
`contracts/preprocess_incremental_example.json` (`mode: incremental`, triggered by a Diagnosis
hypothesis — though the mode decision itself never depends on which node triggered the call), and
`contracts/preprocess_row_filter_example.json` (dataset-wide row filtering, authorized via
`rows_to_filter` after a blocking clarification was resolved).

Top-level output sections: `schema_version`, `processed_dataset_path`, `rows_filtered` (empty
unless `rows_to_filter` was given — each entry includes the condition, row count removed, and
reason), `split_info` (whether the train/validation split was created this call or reused, which
method — stratified, random, or time-based — matching the task type, plus `cv_design`,
`selection_noise_band`, `regime_coverage` and `holdout_representativeness` where step 3 set them),
`transformations_applied` (each sub-decision — `missing_handling`, `outlier_handling`, `scaling`,
`encoding` — includes its own `reasoning` and `confidence`, plus `fit_on` showing statistics were
fit on train-split rows only; `missing_handling` includes `source_mechanism` referencing EDA's
MCAR/MAR/MNAR classification and `missing_indicator_added` where relevant), `encoding_applied`,
`dropped_columns`, `post_transform_stats` (outlier counts using the same
`outlier_detection_method` EDA declared, for a valid before/after comparison),
`clarification_needed` (each entry carries `blocking`, `confidence`, `recommendation`, `recommendation_reasoning` and `impact_if_wrong` — see the Planner's Clarification Handling section), `agent_initiated_findings`
(anything explored beyond what a step specifies), `issues`, and `artifact_paths`
(each plot entry includes a `path` and a `data_path` to its paired before/after values CSV). A
human-readable report is also written to `reports/iteration_<NNN>_preprocess.md` per step 12,
alongside the JSON.