---
name: feature-engineering-agent-skill
description: Creates new derived features from existing columns, and selects or drops features based on relevance to the target — never for data-quality reasons, which is Preprocess's job. This is the only node that introduces genuinely new information into the feature set, or judges a feature's usefulness rather than its correctness.
---

# Feature Engineering Skill

## Purpose

Given the current feature set and whatever triggered this call — an EDA finding, a Diagnosis
hypothesis, a Training observation, a prior FE iteration result, or a direct instruction —
create new features that capture information not already present as a raw column, and/or select
which features actually earn a place in the model. This is the boundary line from Preprocess:
Preprocess changes how existing information is *represented*; Feature Engineering changes *what
information exists*, or judges *whether it's worth keeping*.

## Reference material

- `../baseline-skill/baseline.md` — **required at step 2a.** Part B defines the naive predictor
  specs this node materialises, and B4 defines what protecting them means
- `references/feature_techniques.md` — the trigger-to-technique table (step 3)
- `references/report_example.md` — a worked example of the report this node writes
- `../shared-references/preprocessing_and_feature_engineering.md` — model-specific scaling, encoding and NaN requirements, owned jointly with Preprocess and Training
- `conventions/handoff_contract.md` — the handoff block's fields and the externalisation pass
- `conventions/naming_conventions.md` — where every file this node writes goes, and what it is named
- `conventions/code_and_plot_conventions.md` — cell-marked scripts; every plot names itself

## Inputs

- `dataset_path` — the most recently processed dataset to build features on top of (normally
  Preprocess's latest output).
- `target_column` — provided for reference and for computing quick relevance sanity-checks.
  **Feature Engineering never transforms the target column** — see Guardrails.
- `context` — prior reasoning relevant to this call, per `conventions/naming_conventions.md`.
  This can come from **any upstream node** — EDA findings (`feature_interactions`,
  `grouping_structure`, `time_series_analysis`, `recommended_strategy`), a Diagnosis hypothesis
  naming a specific feature to add or remove, a Training observation about model behaviour, a
  prior FE iteration's open questions, or a direct Planner instruction. The source does not
  change what FE does — only the specific trigger and technique change. Always record which node
  and iteration the triggering context came from.
- `current_features` — what's currently in the model.
- `naive_predictor_features` — the per-horizon naive predictor specifications produced by
  `../baseline-skill/baseline.md` Part B, carried forward by the Planner. Present on every call
  once the baseline has run, which is always before this node's first call. Step 2a builds these.
- `protected_features` — the column names this node may not drop on its own judgment, and the
  reason each is protected. Every naive predictor is on this list; anything else the Planner has
  put there is on it for a reason recorded alongside. See step 6's protected-feature rule.
- `mode` — one of:
  - `initial` — first Feature Engineering call this run; builds on Preprocess's latest output.
  - `incremental` — builds on this node's own prior output; add or remove features from what
    already exists without discarding it.
  - `rebuild` — discard this node's own prior derived features and start over from Preprocess's
    current output, because whatever triggered this call concluded the existing feature set's
    approach needs to change, not just extend. **`rebuild` does not discard protected features** —
    step 2a rebuilds every naive predictor from its spec on a rebuild call, identically. A rebuild
    that silently loses them is the most likely way for the protection to fail, because nobody
    decided to drop anything.
  Same principle as Preprocess's `mode`: this is the Planner's judgment call each time, based on
  whatever triggered the call — never tied to a specific upstream node.
- `mode_reasoning` — why this mode was chosen this call.
- `iteration` — the global iteration number, used to build every output path.

## Tools available

- Code execution sandbox with `pandas`, `numpy`, `scikit-learn` (`PolynomialFeatures`, `PCA`,
  `KBinsDiscretizer`, and similar)
- Plotting library for artifact generation only — same "numbers, not images" rule as EDA and
  Preprocess: every plot gets a paired CSV of the actual values plotted
- File write access to `artifacts/feature_engineering/`, `data/`, `history/`, `code/`, and
  `reports/`
- Read-only access to `state/current.json` and any file under `history/`, per
  `conventions/naming_conventions.md`, for anything not already covered by `context`

## Procedure

**Every step below is a floor, not a ceiling — and so is the procedure as a whole**, the same
rule EDA and Preprocess follow. If a technique, trigger, or entire additional step beyond what's
specified here would genuinely help for this specific dataset, pursue it — this doesn't excuse
skipping what's actually required, it's permission to go further when warranted. Record anything
beyond what's specified in `agent_initiated_findings`, never blended into the standard output
fields as if it were part of the prescribed procedure — same shape as EDA's version:
`related_step`, `title`, `method`, `findings`, `reasoning`, `confidence`, with `"new"` for
something that isn't an extension of any listed step.


1. **Resolve the working dataset per `mode`.** Load `dataset_path` as given. In `incremental`
   mode, confirm `current_features` matches what's actually already reflected in that file before
   proceeding — if it doesn't, record an `issues` entry rather than silently proceeding on a
   mismatched assumption.

2. **Read `context` for what triggered this call and what's relevant.** Identify the specific
   observation about the data or model behaviour that is driving this call, and which node
   produced it — this determines which technique in step 3 applies and how confident you should
   be in the trigger. Also extract the recommended model family and feature-selection posture
   (filter/wrapper/embedded) from `context`; these govern choices later in this procedure. If
   `context` does not specify a posture (e.g. FE is triggered by Diagnosis without a fresh EDA
   recommendation), carry forward the posture from the most recent prior EDA output in
   `state/current.json`, or default to `embedded` if none has ever been set.

2a. **Build the naive predictor features from their specs.** Do this before any other feature
   creation, on the **first** FE call of the run and on every `rebuild` call. On `incremental`
   calls, verify they are still present in the working dataset and rebuild any that are missing —
   a missing protected column is an `issues` entry as well as a rebuild, because something removed
   it without asking.

   For each entry in `naive_predictor_features`, create the column named in `column_name` by
   applying its `formula` at origin `target_date − horizon`, for every row of the dataset. These
   are `fit_on: "row_local_causal"` features: the value at row date `d` depends only on the series
   at or before `d − h`, never on anything after it.

   **Three things to get right, because each has a wrong version that looks identical.**

   - **Walk back to the last OBSERVED period, never index a fixed date.** The same rule as
     `baseline.md`'s second rule. A closure day holds a zero or nothing at all, and indexing it
     turns a shut shop into a demand figure.
   - **Compute the origin by DATE, not by row position.** `shift(h)` moves `h` rows, and rows are
     not days on a frame with non-trading days filtered out. This is the same failure step 7's
     horizon-target construction warns about, and it produces a predictor for the wrong weekday —
     which on a weekly-seasonal series is most of the signal.
   - **Each predictor belongs to its own horizon and no other.** `naive_h7_*` is computed at
     origin `d − 7`; handed to the `h = 30` model it reads the series 23 periods after that
     model's origin. Step 7 records this as `min_horizon = h` **and** `max_horizon = h`, and
     `horizon_feature_subsets` gives each horizon its own predictors only.

   **Verify each column against the number the baseline already computed.** The baseline scored
   each kept rule on the train rows; scoring the column you just built on those same rows must
   reproduce that number. If it does not, the column is not the rule it is named after: record an
   `issues` entry, fix the construction, and re-verify. If it still cannot be made to reproduce
   the baseline's number, keep the column — it is protected — but mark it `verified: false` and
   raise a `blocking` clarification, because a protected column nobody can reconstruct is worse
   than either keeping or dropping it knowingly. This check costs one line and is the only thing
   standing between a mis-built predictor and a run that trusts it completely.

   Record each built column in `features_created` like any other feature, with
   `technique: "naive_predictor"`, its source spec, the verification result, and
   `protected: true`. Steps 4, 5 and 6 treat them differently — see each.

   **The NaN rule is different for these.** Rows early in the series have no origin to read, so
   every naive predictor is NaN at the start. Do not drop those rows — that silently shortens the
   training window for every feature, not just this one. Fill or declare them per the usual lag
   NaN strategy and record it.

2b. **Decide how the target itself is represented.** Before creating any feature beyond step 2a's
   naive predictors, settle what the model is being asked to predict. This is not a feature decision and it is not
   Training's to make — the anchor a relative target needs is a column, and columns are built
   here.

   **The failure this prevents.** Tree ensembles predict averages of training-set leaf values, so
   their output is bounded by the range of the target they saw. A series that trends or shifts
   level moves outside that range, and the model then cannot follow it *at all* — every
   prediction is clipped at the edge of what it learned, producing a large one-sided bias that no
   hyperparameter search can touch. Linear models extrapolate and do not have this problem; tree
   models are the recommended family for most tabular work, so this comes up often.

   **Check first, using what EDA already measured.** Raw-level modelling is disqualified when any
   of these hold:

   | Signal | Source |
   |---|---|
   | a significant trend in the target | EDA `target.time_series_analysis.trend` |
   | a level step or structural break | EDA structural-break output |
   | the most recent window's mean sits outside the training window's interquartile range | Preprocess `split_info.level_gap_pct` |

   **When disqualified, model the target relative to a past-only anchor:**

   | Representation | Use when | Inverse |
   |---|---|---|
   | ratio, `y / anchor` | the target is strictly positive and varies proportionally with its level — most demand, volume and count series | `pred * anchor` |
   | difference, `y - anchor` | the target can be zero or negative, or varies in absolute terms independent of level | `pred + anchor` |
   | raw `y` | no trend, no level shift, and a model family that extrapolates | none |

   **Choosing the anchor.** It must be computed strictly from data at or before the origin — it
   is a feature, subject to step 7's prediction-time availability audit like any other. Prefer a
   **trailing level over a long window** (e.g. a 60–120 period trailing mean) over a single recent
   lag. The anchor's job is to carry the *level*; a long window estimates level with less noise,
   while a single lag carries level plus that period's noise, which the model then has to undo. A
   long anchor is also more robust to a large level shift, which is precisely the case that
   motivated the change.

   **The naive predictors built in step 2a are anchor candidates, and usually the best ones.** A
   predictor selected for horizon `h` is already a level estimate computed at exactly that
   horizon's origin, already causal, already verified against the baseline's own number, and
   already protected from being dropped later. Where a wide-window seasonal mean was kept for a
   horizon, it is a trailing level over a long window — which is what this step asks for — with
   the season's shape included. Using it makes the target a residual against a decent forecast,
   which is the framing the whole of step 2a exists to enable. Where you choose a different
   anchor, say why the naive predictor was not it.

   **Do not expect validation to settle the anchor length.** Where validation contains less level
   drift than the deployment period will, a shorter anchor scores better on validation and worse
   in deployment — the ranking inverts. Choose the anchor on the reasoning above and record it;
   if you use a validation comparison, say what the validation window's level gap was so the next
   node knows how much weight that comparison carries.

   **The representation is per model family, not per run.** A family that cannot consume a
   transformed target does not impose raw levels on the families that can. Classical ARIMA-family
   fits need levels; a gradient-boosted tree does not, and the tree is the one the disqualification
   above was written for. Emit a representation per family where they differ, record both inverses,
   and let Training apply the right one — never collapse them to the most restrictive.

   **An override expires with the reason for it.** If raw levels are kept despite a disqualifying
   trigger, record the override with the family that required it in `target_representation.override`.
   The moment that family stops being a live candidate, the override is void and the decision
   returns to the table above. Diagnosis reads this: a champion from a family that never needed the
   override, while the override is still in force, is a finding and a routing trigger back here.

   > `run_20260904T140437Z` recorded a significant downward trend (slope -0.039/day, -35.9%,
   > p < 0.001) — a disqualifying trigger — then kept level targets "for ARIMA compatibility",
   > planned a trailing-level feature to compensate, and dropped that feature as concurrent two
   > steps later. ARIMA and SARIMA both lost to LightGBM. So the run paid for a constraint imposed
   > by a family it did not ship, lost the compensation, and never revisited either decision. The
   > shipped model's predictions bottom out at 222.9 while 16% of holdout days fall below that:
   > with level targets a tree cannot leave its training range, and no feature can lift the floor.

   **`log1p` is not a substitute.** A log transform stabilises variance and handles
   right-skew; it does not let a tree predict below its training range, because the bound simply
   moves into log space. If the trigger is level drift, a log transform does not address it.

   Record under `target_representation`: the chosen form, the anchor column, the inverse
   transform, the trigger that justified it, and `reasoning`/`confidence`. Carry all of it in the
   handoff — Training must apply the inverse to report metrics in the target's real units, and a
   model artifact whose output is a ratio nobody documented is a trap.

3. **Create derived features, using the table below to match technique to trigger.** Do not
   create features speculatively with no basis — every created feature should trace back to a
   specific signal in `context`, not "this seemed like it might help."

   The "Trigger" column describes an **observation about the data or model** — not a specific
   upstream node. The same observation can be surfaced by EDA, confirmed by Diagnosis, noted in
   a Training residual analysis, or stated directly in a Planner instruction. The source does not
   change which technique applies; record both the observation and its source in `reasoning`.

   The trigger-to-technique table is in `references/feature_techniques.md`. Match on the
   observation, not on which node reported it, and record both the observation and its
   source in `reasoning`.


   **Two different leakage rules apply here, not one.** Techniques that compute a single
   statistic by looking across many rows (group aggregation, PCA, bin edges) must fit that
   statistic using `_split == "train"` rows only, then apply the result to every row — the same
   rule Preprocess uses for its own transformations. Techniques that are inherently row-local
   (lag features, rolling windows, calendar features) don't have that cross-row fitting step at
   all — their leakage rule is simpler and different: a row's feature value must never depend on
   anything after that row's own timestamp. Record which rule applied via `fit_on` —
   `"train_split_only"` for the first kind, `"row_local_causal"` for the second, so it's clear
   which discipline was actually being enforced.

4. **Run a variance-in-train check on every newly created feature.** A feature with zero or
   near-zero variance in the training set is useless regardless of its theoretical motivation —
   the model cannot learn any mapping from a constant or near-constant input. This is a separate
   check from the relevance sanity-check (step 5): a feature can have non-zero correlation with
   the target on the full dataset yet still have zero variance in the training split alone (e.g.
   a regime-change distance feature created from an event that falls entirely outside the
   training window).

   For each created feature, compute variance (or unique-value count for categoricals) on
   `_split == "train"` rows:
   - **Zero variance in train** (constant value, or Spearman/correlation returns NaN/undefined):
     drop the feature. A constant cannot be split on and contributes no signal to any model
     family. Record the drop in `features_dropped` with reasoning. **Exception — a protected
     feature is never dropped here.** A naive predictor that is constant in train is a
     construction failure, not a relevance finding: the rule scored a real error in the baseline,
     so a constant column means step 2a built the wrong thing. Record it in `issues`, raise the
     blocking clarification of step 6's protected-feature rule, and keep the column.
   - **Near-zero variance** (> 95% of train rows share one value): flag with `confidence: low`
     and require an explicit domain justification to keep. "The model can handle it" is not a
     justification — state specifically what signal the non-majority values are expected to carry.
   - If a feature is zero-variance in train but non-zero in validation (e.g. because the
     trigger condition first occurs after the training window), also note this: the model will
     encounter values it has no learned mapping for at inference time — and the same will be
     true, more severely, on the held-out data this run never sees.

5. **Run a quick relevance sanity-check on each created feature** that survived step 4 — its
   correlation with the target, using the metric appropriate to the feature type: Spearman for
   numeric, mutual information or Cramér's V for categorical. Compute this check using
   `_split == "train"` rows only.

   **Interpreting the result:**
   - If the check returns undefined (NaN) or the feature was found constant in step 4: treat
     as zero-relevance and drop unless overridden by strong domain reasoning.
   - **Guideline for dropping:** |Spearman| < 0.02 (or equivalent MI ≈ 0) on a train set with
     more than 200 rows is a near-certain signal of zero value — drop the feature unless there
     is an explicit domain argument (e.g. a known binary event with very low base rate, a safety
     feature whose value lies in rare-but-critical rows). Document the argument if you keep it.
   - For 0.02 ≤ |Spearman| < 0.05: flag with `confidence: low`. The feature may add marginal
     value or may be noise at this sample size — note this explicitly rather than silently keeping
     or dropping.
   - For features in the gray zone, note the effective sample size: a Spearman of 0.07 on 50
     train rows is much weaker evidence than 0.07 on 2000 rows.
   - This check is not a full evaluation of whether the feature helps the model — that requires
     a trained model and belongs to Training/Diagnosis. This step only catches features that are
     obviously useless before they are carried forward.

   **Protected features are measured here but never dropped here.** Compute and record the check
   for them like any other feature — the number is worth having — but a low result routes to step
   6's protected-feature rule, not to a drop. A naive predictor with weak train correlation is
   telling you the series is hard to forecast at that horizon, which is a finding about the
   problem rather than a verdict on the column.

   **Never compute this check on the full dataset** — letting validation rows influence
   which features get kept is a real leakage path, even though this is only a screening decision.

6. **Select or drop features for relevance**, informed by the feature-selection posture extracted
   in step 2 (from EDA, Diagnosis, Planner instruction, or the `embedded` default):

   | Posture | What this step does |
   |---|---|
   | Incremental / forward | Only add the specific feature(s) this call was triggered to create — don't reconsider the whole feature set each call |
   | Broad-start / backward elimination | Start with a wider candidate set (existing + newly created) and drop the weakest by relevance (low correlation, high multicollinearity with a stronger feature) |
   | Embedded (relying on the recommended model family's built-in selection) | Keep a broader candidate set and let Training's model do the weighted pruning — **but this does not mean skipping this step**. See clarification below |

   **Clarification on the embedded posture:** "Let the model prune" applies to the gray zone —
   features with modest, uncertain relevance that a regularised model or tree pruner can
   down-weight on its own. It does not apply to features that have already failed steps 4 or 5:
   - Always drop from any posture: zero-variance-in-train features, features with undefined
     relevance check, features perfectly redundant with an already-kept stronger feature, and
     features with |Spearman| < 0.02 with no domain justification.
   - The embedded model requires non-trivial signal to distinguish useful from useless. A
     constant, a zero-correlation feature, or a perfect duplicate gives it nothing to learn from
     and wastes a coefficient slot or a split budget.

   Regardless of posture, always drop a feature that is redundant with a stronger one based on
   any `multicollinearity` findings in `context` (from EDA, Diagnosis, or prior FE iterations),
   and always act on an explicit instruction naming a specific feature to add or remove.

   **The protected-feature rule — this node may not drop one, and must ask instead.** Every column
   in `protected_features` (every naive predictor, plus anything else the Planner listed) is
   exempt from every drop rule in steps 4, 5 and 6, including the "always drop from any posture"
   list and the multicollinearity rule above. If the evidence says one should go:

   1. **Keep the column.** It stays in the output dataset and in `features_selected`.
   2. **Raise a `clarification_needed` entry with `blocking: true`**, naming the column, the
      evidence, the measurement it came from, a `recommendation` (drop or keep), its
      `recommendation_reasoning`, and `impact_if_wrong`.
   3. **The user decides.** The Planner halts the run on a blocking entry and surfaces the
      question; it never answers this one on the user's behalf and never adopts the
      recommendation as an assumption. Only an explicit user answer removes a protected column,
      and when it does, record the answer in `features_dropped` as its reasoning.

   **Why this one column type gets a veto.** A naive predictor is correlated with the target by
   construction and correlated with every nearby lag feature by construction. That makes it
   trip every redundancy heuristic in this skill *and* score near zero on solo importance in
   Training, for the reason `../shared-references/feature_attribution.md` §4 sets out — the
   correlated twin covers for it. Each of those signals is normally right; on this column each of
   them is wrong for a knowable reason. The pipeline has no way to tell the two cases apart on its
   own, so it asks.

   **Horizon scoping is not a drop.** Giving `naive_h7_*` to the `h = 7` model only, and leaving
   it out of every other horizon's subset, is the construction working as specified (step 7).
   Do not raise a clarification for it, and do not record it in `features_dropped`.

7. **Audit prediction-time availability for every feature in the dataset.** This step is
   specifically about deployment correctness — it is distinct from temporal row-ordering leakage
   (step 3's leakage rules) and from data-quality leakage (EDA/Preprocess). The question here is:
   *would this feature be computable at the moment a real prediction is needed?*

   Classify each feature into one of three availability tiers:
   - `pre_period` — the value is known before the prediction period begins. Calendar features,
     lag features, rolling windows anchored at least one period behind, external reference data
     (e.g. a scheduled event calendar): all pre-period. These are always safe to use.
   - `concurrent` — the value is only known after the period being predicted has ended. A
     same-day sales total, a same-day order count, a same-day operational metric: all concurrent.
     Using concurrent features in training is valid for a retrospective analysis or explanation
     model. For a live forward-looking forecast they are not available at decision time.
   - `post_period` — the value depends on an outcome that occurs after the prediction period.
     This is strict target leakage regardless of use case. Drop immediately.

   **On a multi-horizon task, availability is not one verdict — it is one per horizon.** A
   `pre_period` classification answers "is this known before the prediction period begins?", which
   silently assumes a single lead time. At horizon `h` the origin sits `h` periods before the
   target, so a lag-1 feature is available for `h = 1` and unavailable for `h = 7`; a 7-day
   rolling mean anchored one period back is available at `h = 1` and not at `h = 14`. Record
   `min_horizon` on **every** feature: the smallest horizon at which it is computable at the
   origin, `1` for a calendar feature that is known arbitrarily far ahead.

   **A naive predictor is bounded at both ends, and it is the only feature type that is.** It is
   defined by its origin — `naive_h{h}_*` is computed at `target − h` — so it is not merely
   *available* at horizon `h`, it is *only correct* at horizon `h`. Record `max_horizon = h`
   alongside `min_horizon = h` for every one of them, and let no horizon's subset contain another
   horizon's predictors. Reused at a longer horizon the column reads the series after that
   model's origin, which is a leak that will not show up as one: the model scores better, the
   validation number agrees, and only the holdout disagrees.

   **The other bound: how far ahead is this feature *knowable*?** `min_horizon` asks when a
   feature becomes computable at the origin. It does not ask the reciprocal question, and the
   reciprocal question is where per-horizon feature sets actually come from: **at horizon `h`, is
   this feature's value at the TARGET date knowable when the forecast is made?** Many exogenous
   series are — that is what a forecast is. Record `known_future_horizon` on every non-target
   feature:

   | feature | `known_future_horizon` | why |
   |---|---|---|
   | calendar, holidays, school terms | unbounded | the date is known forever ahead |
   | a published schedule (competitor closures, promotions) | to the end of the published window | confirm the window; do not assume |
   | weather | ~7–10 days | a forecast exists that far out and no further |
   | the target's own realisations | 0 | never knowable past the origin — these stay lag features |

   **A feature with `known_future_horizon >= h` should be evaluated at the TARGET date for that
   horizon, not at the origin.** Origin weather at `h = 90` is a three-month-old observation of a
   variable with no memory; target-date weather at `h = 1` is tomorrow's forecast, which is real
   information. Same column, opposite value, and only the second is what the feature was added
   for. Where `known_future_horizon < h`, the feature is dropped for that horizon rather than
   silently replaced by its stale origin value.

   **Two things this obliges you to do.**

   First, it is a claim about the *deployment environment*, not about the training data — it
   asserts that a forecast feed will exist at serving time. Raise it as `clarification_needed`
   with the conservative reading as the recommendation, exactly as the concurrent-feature audit
   does. If no feed will exist, the feature is not known-future at any horizon and the answer is
   to drop it, not to fall back on the origin value.

   Second, if you train on *observed* target-date values and serve on *forecast* ones, the model
   is fitted on a cleaner signal than it will ever see. Say so in the audit and treat every score
   from it as an upper bound; where archived forecasts exist, train on those instead. This is the
   trap in an otherwise correct design, and it is worth one sentence in the handoff.

   > `run_20260904T140437Z` gave all six horizon models the same 37 columns, recording only that
   > "all features are available at min_horizon=1, subsets are identical for all h" — which
   > answers a leakage question and skips the modelling one. Weather was read at the origin for
   > every horizon and measured **0.34 / 0.06 / 0.12 / −0.25 / −0.06 / −0.23** MAE across
   > h = 1…90: no contribution anywhere. The version that might have worked — tomorrow's forecast
   > at h = 1 and h = 7, nothing at h = 30+ — was never built, so twelve weather columns were
   > carried at every horizon and earned nothing at any of them.

   Then emit `horizon_feature_subsets` — for each horizon in `context.forecast_horizons`, the
   list of features with `min_horizon <= h` and (where set) `max_horizon >= h`, each marked
   `evaluated_at: origin | target_date`, and with every feature whose `known_future_horizon < h`
   excluded. Each horizon's subset must contain that horizon's own naive predictors and no other
   horizon's; assert this before writing, because it is a one-line check for a failure that is
   otherwise invisible. Training uses these directly; deriving them per
   horizon is not something to leave to the node that fits the model. Where a horizon's subset
   loses the features that carry most of the signal, say so as a finding: it is the reason long
   horizons score worse, and it is far more useful stated up front than rediscovered in residuals.

   **Build one target column per horizon, by DATE, never by row position.** For each `h`, create
   `<target>_h<h>` holding the value at `row_date + h` **in the series' calendar unit** — join on
   the date, do not use a positional shift. Record the row count each one loses at the end of the
   series. This is target representation, which is this node's job — leaving Training to shift the
   target itself puts the same construction in several places and eventually they disagree.

   **Why the distinction is not pedantic.** `shift(-h)` moves `h` *rows*. Rows equal days only on
   a gapless daily frame, and this pipeline filters non-trading days, so they do not. The gap
   between the two is not a rounding error — it changes which **weekday** the model is predicting,
   and on a weekly-seasonal series the weekday is most of the signal. Worse, the offset is not even
   constant: it moves whenever the trading calendar changes, so a model trained before such a
   change learns a rule that is wrong after it.

   **Verify it, and record the check.** After building the targets, compute the realised calendar
   gap between each row and its target for every `h` — median and range — and write it to
   `horizon_construction`. If the median is not `h`, the column is not what its name says. Where
   the series has a regime change, compute the gap on each side separately: a construction that
   holds on one side and not the other is the failure this check exists to catch.

   > `run_20260904T140437Z` built targets with `shift(-h)` on a frame with pre-2020 Mondays
   > removed, so a week was six rows. Its `h=30` target sat **35** calendar days ahead before the
   > regime change and 30 after; `h=90` sat **105** days ahead, then 90. Because 30 rows was
   > exactly five six-day weeks, the h=30 model learned "my target is the same weekday as my
   > origin" from 89% of its training rows — and the holdout, on a 7-day week, is +2 weekdays at
   > every row. Five of its six horizon models learned a weekday offset that was wrong on every
   > holdout row; h=1, the only one that survived, is the only horizon where the run beats an
   > AutoML arm that does no feature engineering at all.

   For every `concurrent` feature, explicitly note:
   - Whether a lagged version (`feature_lag_1` or equivalent) already exists in the dataset as
     a deployable proxy.
   - If not, whether one should be created this call (apply step 3's lag technique).
   - Whether the use case for this specific run is retrospective analysis (concurrent features
     permitted) or live forecasting (concurrent features must be excluded at inference time).

   This audit is **non-blocking by default** — concurrent features are kept in the dataset so
   that training can use them for the retrospective case. But the audit result must be recorded
   as a `clarification_needed` entry so the Planner and Training steps can choose the correct
   feature subset for their deployment scenario.

8. **Record `reasoning` and `confidence` for every decision** — each created feature, each
   selection/drop decision — per `conventions/naming_conventions.md`. A feature that was created
   but is genuinely uncertain to help should say so via `confidence: low`, not be silently
   dropped or silently kept without comment.

9. **Raise `clarification_needed` for genuine ambiguity** encountered during this call, same
   rules as EDA and Preprocess: a `blocking` flag, a `confidence`, a `recommendation` with its
   `recommendation_reasoning` and `impact_if_wrong` (field table in the Planner's **Clarification
   Handling** section), and never a self-decision to pause the workflow — only the Planner decides
   that. Prediction-time availability ambiguity (step 7) always produces a `clarification_needed`
   entry when concurrent features are present, and its recommendation is the conservative reading
   — treat the feature as unavailable — unless the evidence in `context` says otherwise.

   A proposed drop of a protected feature (step 6) always produces a `blocking: true` entry. It is
   the one clarification this node raises that the Planner may not resolve by adopting the
   recommendation: the whole point is that a person decides.

10. **Record any execution failures in `issues`** — a feature that failed to compute, a
    dimensionality reduction that didn't converge — rather than silently omitting it.

11. **Write the output.** Save the full JSON to `history/iteration_<NNN>_feature_engineering.json`,
    the generated code to `code/iteration_<NNN>_feature_engineering.py`, and the resulting dataset
    to `data/iteration_<NNN>_feature_engineering.csv`.

12. **Generate a human-readable report.** Write a short Markdown report to
    `reports/iteration_<NNN>_feature_engineering.md`, covering every section of the output, the
    same exhaustive standard as Preprocess's report: what triggered this call and in what `mode`;
    every feature created, with the technique used, its `reasoning`, `confidence`, variance-in-
    train result, lag NaN fill strategy (if applicable), and relevance sanity-check result; every
    feature selected or dropped, with reasoning; the prediction-time availability audit summary;
    `clarification_needed` entries; and `issues`. Write explanations, not just facts — don't
    just state a technique name, explain why it fit this specific trigger.


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
- Never transform, impute, scale, or encode an existing column's representation — that is
  Preprocess's job. This node only creates genuinely new columns or makes relevance/selection
  judgments; it does not fix or re-represent what already exists.
- **Never drop, exclude, or silently fail to rebuild a feature listed in `protected_features`.**
  Every drop rule in this skill has an exception for them, and the exception is always the same:
  keep the column, raise a `blocking` clarification, let the user decide. A `rebuild`-mode call
  rebuilds them from their specs rather than discarding them with the rest of the prior feature
  set — that path loses them without anyone deciding anything, which is the failure mode this
  protection is actually guarding against.
- **Never build a naive predictor at an origin other than `target − h`, and never let one appear
  in another horizon's feature subset.** The column is defined by its origin; used at a longer
  horizon it reads the future and the resulting leak flatters every validation number in the run.
- **Never carry a naive predictor whose train score does not reproduce the baseline's number for
  the same rule.** A mismatch means the column is not the rule it is named after, and every
  downstream reading of it — importance, residuals, the anchor decision — is then about a
  different quantity than the one the report names.
- Never drop a feature for data-quality reasons (leakage, PII, missingness, inconsistent
  categories) — that is EDA's flag and Preprocess's action. This node only drops for *relevance*
  (low target relationship, redundancy with a stronger feature, zero variance in train) or an
  explicit instruction.
- Never transform `target_column`. If a target-side transform is genuinely needed, that decision
  belongs to Training/Diagnosis, which own metric interpretation.
- Never create a feature with no traceable basis in `context` — every feature needs a specific
  trigger, not general speculation that it "might help."
- Never create a large batch of features in one call without checking the cumulative effect on
  the rows-to-features ratio, even when each individual feature has a valid trigger. Check the
  ratio *before* creating features, not after. This is the same ~10× rows-to-features floor EDA
  already uses for feasibility. If a call would push the ratio uncomfortably low, note this
  explicitly and consider spreading additions across more than one iteration.
- Never carry forward a feature with zero variance in the training split without explicit
  justification. Zero variance in train means the model has no signal to learn from during
  fitting; the feature's non-zero values in validation are encountered without any
  learned mapping, producing arbitrary predictions.
- For any cyclically-ordered ordinal feature (month 1–12, hour 0–23, day-of-week treated as
  ordinal, compass bearing, etc.), always use Fourier sin/cos encoding rather than the raw
  integer. The raw integer imposes an artificial discontinuity at the wrap-around point. Never
  retain the raw integer alongside its Fourier pair — it reintroduces the discontinuity being
  fixed.
- Never leave NaNs introduced by lag or rolling-window feature creation unresolved. These NaNs
  are produced by Feature Engineering and are its responsibility to handle. Apply and document a
  fill strategy (drop rows, train-computed statistic, domain constant, or explicit native-NaN
  declaration) before writing the output dataset.
- Never classify a feature as "the embedded model will prune it" without first verifying it
  passes the variance-in-train (step 4) and relevance (step 5) checks. Embedded pruning relies
  on the model seeing non-trivial signal differences between features. A constant or a
  zero-correlation feature gives no such signal and must be dropped here, not deferred.
- Never compute a statistic used in a feature (an aggregate, a rolling window, a PCA fit, a bin
  edge, a lag NaN fill value) using anything other than `_split == "train"` rows, then apply the
  result to every row. This is the same fit-on-train-only rule as Preprocess.
- Never independently perform rigorous model-dependent feature evaluation (permutation
  importance, partial dependence, SHAP interaction values) — this node's own checks are sanity
  screens only; the real verdict comes from Training/Diagnosis once a model exists.
- Every created feature and every selection/drop decision needs its own `reasoning` and
  `confidence` — never bundled into one blanket judgment for the whole call.
- Never leave a structured field `null` without explanation, with the same structural-null
  exception used elsewhere.
- Never generate a plot without its paired values CSV, and never load that CSV into your own
  reasoning context.
- Never decide on your own to pause the workflow — raise ambiguity via `clarification_needed`
  and let the Planner decide.
- The `reports/` Markdown file must only restate what's already in the JSON output.

## Input / output contract

Full input/output example: `contracts/feature_engineering.json`.

Top-level output sections: `schema_version`, `engineered_dataset_path`,
`rows_to_features_check` (computed *before* creation and *after*, so the delta is visible),
`features_created` (each entry includes `technique`, `reasoning`, `confidence`, `fit_on`,
`variance_in_train` result, `lag_nan_fill` strategy if applicable, `protected` where it applies,
and its relevance sanity-check result computed on `_split == "train"` only), `features_selected`,
`features_dropped` (each with `reasoning`; a protected feature may appear here only with an
explicit user answer as that reasoning), `naive_predictors_built` (one entry per spec in
`naive_predictor_features`: the column built, the source spec, the train-score verification
against the baseline's own number, and its NaN fill), `protected_features_status` (every
protected column, whether it is present in the output dataset, and any blocking clarification
raised against it), `prediction_time_availability_audit` (one entry
per feature classified as `concurrent` or `post_period`, plus `min_horizon`, `max_horizon` where bounded, and `known_future_horizon` on every feature),
`horizon_construction` (per horizon: the realised calendar gap median and range, and the per-regime gap where a regime change exists), `horizon_feature_subsets` (one entry per horizon in `context.forecast_horizons`: the feature list
usable at that lead time, and what it loses relative to the shortest horizon), `horizon_targets`
(the `<target>_h<h>` column built for each horizon and the rows it loses), `clarification_needed` (each entry carries `blocking`, `confidence`, `recommendation`, `recommendation_reasoning` and `impact_if_wrong` — see the Planner's Clarification Handling section), `agent_initiated_findings` (anything explored beyond what a step
specifies), `issues`, and `artifact_paths` (each plot entry includes a `path` and a `data_path`
to its paired values CSV).
