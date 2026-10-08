---
name: diagnosis-skill
description: After Training produces a model, decides whether the run has met its goal and, if not, root-causes why and proposes a ranked set of two to three concrete, falsifiable next hypotheses — each naming which node would test it and what to change there. Diagnosis never decides what actually runs next; that is the Planner's call, weighing these hypotheses against everything else in state and history.
---

# Diagnosis Skill

## Purpose

Diagnosis is the pipeline's judgment call on "is this good enough, and if not, why not." It runs
after every Training call, and the Planner may also call it again after any other node (e.g. after
a Preprocess or Feature Engineering change made without a new Training call) when it wants a fresh
read on where things stand. Its evidence is always anchored on the most recent Training output —
metrics, overfitting assessment, feature importance, residual diagnostics — read alongside the
run's broader history so it doesn't repeat a hypothesis already tried.

Diagnosis produces exactly two things: a stop-or-continue read, and — if continuing — one root
cause and a **ranked set of two to three** concrete hypotheses for what to try next. It does
**not** pick the next skill to run. That decision belongs to the Planner, which weighs these
against the rest of state and history and may follow the ranking, pick a lower-ranked one, or
choose something else entirely.

**Why a set and not a single hypothesis.** An earlier version of this skill required exactly one.
That reliably produced a run shaped as one forward pass followed by a `training ↔ diagnosis`
ping-pong: with only one option on the table, the Planner's stated authority to "adapt or override"
had nothing to choose between, and in practice it followed the suggestion every time. Ranking two
or three costs almost nothing — the evidence for each is already in hand by step 3 — and it is what
makes the Planner's routing decision real rather than nominal. The ranking is still an opinion:
say which you would run first and why, and the Planner may still disagree.

## Reference material

- `../shared-references/interrogating_results.md` — **required, as step 0 below.** Five cheap
  checks that a number measures what its name says. A wrong reading of a correct number gets
  caught later; a correct reading of a number that measures something else never does
- `../shared-references/feature_attribution.md` — **required before proposing to drop, add or
  consolidate any feature.** Feature importance is not identifiable under correlated features; this
  file gives the grouping rules and the masking test that decide whether a near-zero number is
  evidence or an artifact
- `../training-skill/references/evaluation_metrics.md` — for interpreting overfitting ratio, MASE,
  and validation/learning-curve reads when the ratio-based verdict alone is ambiguous
- `../training-skill/references/plotting_guide.md` — if an additional diagnostic plot beyond what
  Training already generated would clarify the root cause
- `conventions/naming_conventions.md` — the `reasoning`/`confidence` discipline and the
  context/history access rules below are identical to every other node's

## Inputs

- `training_output` — the full JSON output of the most recent Training call: `metrics`,
  `overfitting_check`, `feature_importance`, `residuals_summary`, `baseline_model`,
  `vs_prior_iteration` (if `mode == "incremental"`)
- `goal` — the plain-language modeling objective, from `state/current.json`. If the goal statement
  (or prior user input) includes an explicit numeric target, Diagnosis checks against it directly;
  otherwise "good enough" is judged from trend, not a fixed number (see step 2).
- `context` — curated prior-step context per the standard rules (`previous_step` entry always
  present; `planner_selected` entries for anything specific from further back the Planner judged
  relevant)
- `features_used_so_far` — the current feature subset in play
- `protected_features` — the columns this node may not propose dropping as an ordinary
  hypothesis: the per-horizon naive predictors selected by `../baseline-skill/baseline.md` Part B,
  plus anything else the Planner has protected, each with its reason. Evidence against one becomes
  a blocking clarification, not a hypothesis — see step 4.
- `prior_hypotheses_tried` — every hypothesis a past Diagnosis call already proposed in this run,
  with its outcome if known (confirmed / rejected / untested). Never propose one of these again
  without new evidence explaining why it's worth revisiting.
- `run_history_summary` — the `history` array from this run's `state/current.json`: one
  `{iteration, node, summary}` entry per call so far. Always available, always cheap.
- `cross_run_history_index` (optional) — for other `runs/run_<id>/` folders tackling the same or a
  closely related goal, their own `state/current.json["history"]` summaries. Skim these the same
  way as `run_history_summary` — a quick scan, not a mandatory deep read. Only open a specific past
  run's `history/iteration_<NNN>_<node_slug>.json` in full if a specific piece of information from
  it looks like it would actually change this call's conclusion (e.g. "a similar lag feature was
  tried in an earlier run — what happened?"). This is available in case it's needed, not an
  instruction to cross-reference every past run on every call.

## Tools available

- Read-only access to `training_output`, `context`, `state/current.json`, every file under this
  run's `history/`, and (per above) other runs' `state/current.json` files
- File write access to `history/`, `code/`, `reports/`, and `artifacts/diagnosis/`
- Plotting library for the one artifact this skill can usefully add that Training itself can't
  (see step 6) — same "numbers, not images" rule: any plot gets a paired CSV

## Procedure

**Every step below is a floor, not a ceiling.** If a check beyond what's specified here would
genuinely clarify the stop/continue read or the root cause, pursue it — record it in
`agent_initiated_findings` with the same `reasoning`/`confidence` discipline used throughout.

0. **Check the measurement before interpreting it.** Run the five checks in
   `../shared-references/interrogating_results.md`: units (rows vs time), regime (does the
   relationship survive the break the run already found), rows (are both sides of every comparison
   on the same `n`), floor (is the baseline actually trying), justification (are the features that
   were argued for actually used). Minutes, not iterations.

   A failed check is **not** a hypothesis and does not compete with one for ranking. It invalidates
   evidence the other hypotheses are resting on, so it is reported first, routed to the node that
   owns the definition, and the stop/continue read is made without the number it broke. Recommending
   a modelling change on top of a number that measures something else is the most expensive mistake
   available at this node, because everything downstream inherits it and nothing downstream can
   detect it.

1. **Read the evidence.** Pull `metrics.validation` (primary + secondary), `overfitting_check`,
   `feature_importance` — read `groups` and each group's `masked` gap **first**, and treat
   `per_feature` values as meaningful only inside their group (`../shared-references/feature_attribution.md`);
   a group with a large masked gap is over-parameterised, and a column with a near-zero solo value
   inside a high-importance group is redundant, not useless — and `residuals_summary`
   (autocorrelation flags, temporal bias by period) from `training_output`. Pull
   `vs_prior_iteration` if present — this is often the single most informative field for step 2.

   **On a multi-horizon task, read every horizon separately.** There is one model per horizon, so
   there is one diagnosis per horizon: a residual autocorrelation at `h = 1` and a level bias at
   `h = 30` are different problems with different fixes, and averaging them produces a hypothesis
   that helps neither. Say which horizons a hypothesis is for. A horizon losing to the baseline
   floor while the others beat it is a `continue` even when the pooled figure looks finished —
   and it usually points at that horizon's feature subset, which lost the short lags the others
   still have.

   **On a time-series task, read all four of MAE, RMSE, SMAPE and MASE, not just the primary.**
   They are always present (Training computes them every call). The RMSE/MAE ratio separates error
   spread evenly across days from error concentrated in a few large misses — two different root
   causes needing two different hypotheses. SMAPE is the one that survives comparison across
   segments or horizons of different magnitude, so it is what a per-segment read should lean on.
   Where the four rank iterations differently, resolve the contradiction explicitly rather than
   quoting whichever one supports the reading you already have.

2. **Stop/continue assessment — per horizon first, then run-level.** This merges what an earlier
   design had as a separate Evaluate step — there is no separate node for it; Diagnosis owns this
   read directly, every time it runs.

   **On a multi-horizon task, run the assessment once per horizon before synthesizing the
   run-level verdict.** Each horizon gets its own `stop`/`continue` read, its own binding
   constraint (where `stop`) or its own primary issue (where `continue`), and its own ranked
   hypotheses. These go into `per_horizon_verdicts[h]`. The run-level `stop_recommendation` is
   then synthesized: `stop` only if **every** horizon individually reads `stop`; `continue` if any
   one horizon reads `continue`, with the continuing horizons named. A horizon that is `stop` while
   others are `continue` should not be retrained just to keep it busy — note it as settled and let
   the Planner focus iterations on the continuing ones. The run-level `hypotheses` list is the
   union of all per-horizon hypothesis sets, de-duplicated and re-ranked by cost and expected impact
   across horizons; hypotheses that apply to one specific horizon carry `applies_to_horizons: [h]`.

   Work through all of the following checks for each horizon; any one of them can override an
   otherwise-positive read:

   | Check | What it looks for |
   |---|---|
   | Explicit target | If the goal or a prior user answer states a numeric target (e.g. "MAE under 20"), compare directly. Met target + no other red flag below → recommend `stop`. |
   | No explicit target | Judge "good enough" from trend and context instead of a number — see the next three rows. Never invent a target that wasn't actually given. |
   | Diminishing returns | Compare this iteration's primary metric to the last 1–2 Training iterations (`vs_prior_iteration`, and one further back via `run_history_summary` if needed). A small relative improvement (rule of thumb: under ~3–5%) two iterations running, with no specific untried hypothesis in sight, is a diminishing-returns signal. Read it against `split_info.selection_noise_band` where one is set: a "gain" smaller than the band is not a small gain, it is no gain, and calling it one is how a run talks itself into a winner that is really a lucky validation draw. **This row alone never justifies a stop** — it must be paired with the coverage row below, because decelerating gains along one axis say nothing about axes never tried. |
   | Coverage (breadth) | Count what has actually been explored, from `run_history_summary`. A `stop` is **not** available while any of these is true: (a) a node other than `end_model` has been visited only once; (b) only one model family has ever been fitted; (c) **any hypothesis any Diagnosis call has ranked remains untested** — every one, not only the top-ranked, and not only the ones that still look promising. A ranked hypothesis is a falsifiable claim the run committed to; the run closes when they are answered, not when they stop feeling worth answering. Only three things retire one: it was **tested** and the outcome recorded; it was **answered by a measurement already in the run**, citing that measurement; or it is **infeasible**, naming the specific input that does not exist. An estimate of its expected gain is not one of the three — "low expected gain relative to the noise band" is a prediction about the result, and predicting a result is what running the test replaces. The same applies to every `alternative_if_wrong`; (d) a cheap external reference (see the row below) has never been run; (e) **no hyperparameter search has ever been run** — the `hyperparameter_search` field is null on every Training call in the run. An axis closed by argument rather than by measurement is untried, and "re-tuning would invalidate the baseline comparison" is not a reason: the baseline is fixed and does not move when a model is tuned. One search, recorded, closes this; (f) **feature_engineering has run only once** — the run has never revised a feature in light of a Training result, so no feature rationale has ever been tested against evidence; (g) the task is a time series and only one of the two non-automatic approaches — classical statistical forecasting, ML-as-regression on engineered features — has ever been fitted (read `ts_approach` / `ts_approach_gap` in `training_output`; an automatic tool such as Auto ARIMA or Prophet fills neither slot, and the classical slot is closed only by an ARIMA-family fit — ARIMA, SARIMA or SARIMAX, with plain ARIMA run at least once — not by exponential smoothing alone). If one of these blocks the stop, say which, and make the untried item one of this call's ranked hypotheses. |
   | Baseline floor | `current.baseline_floor` is established by the run's first Training call, before any learned model exists. Read `vs_baseline_floor` from `training_output` every call: the margin **per segment and per horizon**, not pooled. A model losing to the floor on any segment the goal is stated over is a `continue`, and the hypothesis is about those segments. Where the model loses **overall** and no untried axis clears the noise band, say so explicitly — that is the one case where a `stop` is correct despite the model being worse than doing nothing, and the Planner needs the recommendation to say it plainly rather than looping. **Beating the floor is the minimum bar, not a stop signal on its own.** A model that has just cleared it should keep improving: if the coverage check (above) still has untried axes, the correct recommendation is `continue` regardless of how comfortably the naive is beaten. A stop is only available when the model beats the floor *and* the coverage check finds nothing remaining worth trying. |
| External reference | At least one cheap reference model must have been fitted on this run's own train/validation split by mid-run, not only at the end — a linear/ridge fit, and where practical a generic AutoML search over the agent's own engineered features. That last one is the informative comparison: handing a generic search the same features measures exactly what the agent's model choice and tuning are worth. A near-zero or negative gap is a strong signal that further Training iterations are not where the value is, and a strong prompt toward an axis the run has not touched. These are validation-scored, like everything else here; the full baseline comparison against held-out data happens after the pipeline finishes and cannot inform a decision inside it. |
   | Residual/segment red flags | An unresolved autocorrelation flag (`\|r\| > 0.2`), a temporal bias pattern that hasn't been addressed, or (when segment-level performance is available) a segment trailing well behind the rest — any of these means "not done," even if the aggregate metric looks fine. |
   | Baseline comparison | If `baseline_model.vs_primary_model == "primary_worse"` or the margin over baseline is thin, that's a structural red flag, not a stopping condition. |
   | Unresolved clarification | Any `clarification_needed` from Training (or from this Diagnosis call) that could plausibly change the feature set or deployment scenario stays open — note it in this call's own `clarification_needed` even if Diagnosis didn't raise it originally. It is surfaced to the user regardless of how "close to done" the metrics look (see Guardrails). |

   **Before recommending `stop`, name the binding constraint.** Not "the metric has plateaued" —
   the mechanism holding the model back: an unexplained structural break, a segment with too few
   rows to learn from, a signal absent from every source, irreducible noise at this granularity.
   If you cannot name one, the honest recommendation is `continue`, because "I do not know what is
   limiting this" is a description of an unfinished diagnosis, not of a finished model. The
   Planner's Step 1c-bis will ask the same question and record the answer as
   `stop_justification`; a `stop` recommendation that arrives without a mechanism just moves the
   work one node downstream.

   **The same test blocks low-value continuation.** A hypothesis earns an iteration by naming a
   mechanism and an expected effect larger than `split_info.selection_noise_band`. A fourth sweep
   of an axis whose last two passes moved less than the band is not a hypothesis, it is momentum —
   say so and let the `stop` stand on the other rows.

   Record `stop_recommendation`: `stop` or `continue`, with `reasoning` and `confidence`, the
   binding constraint where the call is `stop`, and which of the rows above drove the call. A `continue` recommendation is the default whenever there's a
   concrete, untried, evidence-backed hypothesis left to test — Diagnosis should be reluctant to
   recommend `stop` just because things "look fine," if a specific improvement is still plausible
   and cheap to try.

   Also record `coverage_check` alongside it: which nodes have been visited how many times, which
   model families have been fitted, which prior hypotheses and alternatives are still untried, and
   whether the external reference has been run. This is the evidence for the coverage row and it is
   cheap to produce from `run_history_summary`. Its real purpose is to make absent exploration
   *visible*: the escape clause in the diminishing-returns row is "no specific untried hypothesis
   in sight," and an agent cannot see what it has not thought of. A list of node visit counts can.

3. **If `continue`: identify the primary issue and its root cause.** Don't stop at restating a
   metric — trace it to a mechanism, the same way `contracts/diagnosis.json`'s existing example
   does (e.g. "high false-negative rate" → "only 2 of 87 features used" → "insufficient feature
   coverage"). Ground every claim in a specific field from `training_output` or `context`, not a
   general impression. Explicitly check `prior_hypotheses_tried` — if the evidence points toward
   something already tried and rejected, say so and look for a different mechanism instead of
   re-proposing it.

   **Some root causes are not modelling problems, and naming them as such is the whole job.** A
   level shift, a regime break, a variable behaving unlike its description, a segment failing for
   no reason visible in the features — the root cause of these is often that nobody has
   established what the data *means* at that point, and no amount of feature engineering or
   retraining can supply it. When that is the honest read, say so plainly and rank a hypothesis
   whose `suggested_action.node` is `eda` or `domain_research`, with the specific question that
   node is being asked to answer. Routing upstream is not an admission of failure; proposing
   another model against an unexplained break is the failure.

4. **Formulate two to three ranked hypotheses.** Write them into `hypotheses`, ordered best-first.
   Each one falsifiable, specific, and testable by one concrete next action:
   - `rank` — 1, 2, 3. Rank 1 is what you would run next if the choice were yours.
   - `statement` — what you believe and why (tie back to the root cause in step 3)
   - `applies_to_horizons` — on a multi-horizon task, the list of horizons this hypothesis is for,
     or the string `"all"` when it applies equally to every horizon. A hypothesis for a single
     underperforming horizon should name it — the Planner can then route a targeted training call
     using `target_horizons` rather than retraining everything.
   - `suggested_action` — `{node, action, reasoning}`. `node` is whichever node would actually test
     it: `preprocess`, `feature_engineering`, `training` (different model family, hyperparameters,
     or a retrain with an adjusted feature set — for a horizon-specific hypothesis, say so here so
     the Planner knows `target_horizons` is appropriate), `eda` (a specific data-meaning question
     worth re-examining), or `domain_research` (a specific real-world explanation worth chasing
     down). This is a recommendation, not a directive — the Planner may run any of them, or none.
   - `falsification` — the pre-registered result that would reject it, fixed before the test runs
   - `cost` — `cheap` (one node call, no upstream rework), `moderate`, or `expensive`. The Planner
     needs this to trade a high-confidence expensive test against a speculative cheap one.
   - `confidence` — per the usual `low`/`medium`/`high` scale
   - `why_this_rank` — one line on why it sits above or below its neighbours

   **The set must not be three variations of one idea.** Rank 1 may address the most likely root
   cause; ranks 2 and 3 should reach for a *different* mechanism or a *different* node wherever the
   evidence allows one. If every plausible hypothesis genuinely points at the same node, say so
   explicitly in `ranking_note` rather than padding the list — but treat that as a signal worth
   examining, because it is also what a run looks like shortly before it stops exploring.

   Draw candidates from more than the immediate residuals: an axis the coverage check flagged as
   untried, a lower-ranked hypothesis from an earlier Diagnosis call, an `alternative_if_wrong`
   that was never reached, or a gap the external reference exposed all qualify.

   **A drop of a protected feature is never a hypothesis.** If the evidence points at removing a
   column in `protected_features` — a naive predictor measuring near zero, or looking redundant
   with a lag feature beside it — do not write it as a ranked hypothesis for the Planner to route.
   Write it as a `blocking` clarification in step 5, naming the column, the measurement, your
   recommendation and what the run loses either way. The Planner halts and the user answers.

   The reason is specific to this column type, not a general caution. A naive predictor is
   correlated with the target and with its neighbouring lags **by construction**, so the
   solo-importance reading that `../shared-references/feature_attribution.md` §4 already warns
   about is guaranteed to understate it, and the redundancy screen is guaranteed to fire. Both
   signals are normally trustworthy; on this column both are wrong for a knowable reason, and this
   node cannot tell that case from a genuine dead feature. Report the measurement — it is a real
   finding either way — and let a person decide what to do with it.

   Also record `alternative_if_all_wrong` — what the next Diagnosis call should look at if every
   hypothesis here is rejected, so it isn't starting from zero.

5. **Record every clarification, not just blocking ones.** Same field shape as every other node
   (`topic`, `description`, `blocking`, `confidence`, `recommendation`, `recommendation_reasoning`,
   `impact_if_wrong` — field table in the Planner's **Clarification Handling** section), but the
   Planner surfaces *all* of these to the user before running anything else. `blocking` no longer
   determines whether the Planner asks; what it now determines is whether the run **halts** until
   an answer arrives, so set it from `impact_if_wrong`, not from how strongly you feel about the
   question. Diagnosis is the node best placed to say what a wrong answer would cost, because it
   is the one reading the consequences of every earlier decision.

6. **Optional artifact: metric trend across this run's iterations.** Diagnosis is the one node with
   visibility across every Training call in the run so far, which Training itself doesn't have.
   When there have been 2+ Training iterations, generate `metric_trend_across_iterations` — the
   primary metric (train and validation) plotted against iteration number, paired with its CSV —
   since this is exactly the comparison a person would otherwise have to reconstruct by hand from
   several `history/iteration_<NNN>_training.json` files.

7. **Write outputs.** Save:
   - History JSON: `history/iteration_<NNN>_diagnosis.json`
   - Code: `code/iteration_<NNN>_diagnosis.py` (if any computation beyond reading/reasoning was
     needed — e.g. the trend plot in step 6)
   - Report: `reports/iteration_<NNN>_diagnosis.md`
   - Update `state/current.json`: append EVERY hypothesis from this call to `prior_hypotheses_tried`
     (each `untested` until a later Diagnosis call resolves it, each carrying its `rank` and `cost`),
     and record `stop_recommendation` and `coverage_check` under `current` so the Planner can read
     them without re-opening this call's full history file. Appending only the top-ranked one is the
     bug this change exists to prevent: the lower-ranked hypotheses are exactly what a later call
     needs to know were already on the table and never run.

8. **Generate a human-readable report.** Write `reports/iteration_<NNN>_diagnosis.md` covering:
   the stop/continue read and why, including the coverage check; the primary issue and root cause
   with the evidence trail; ALL ranked hypotheses with their suggested actions, costs and
   falsification criteria — not just rank 1, since the ranking is the part the Planner acts on; any
   clarifications raised. Explanations, not just numbers — a `stop_recommendation` without the
   reasoning behind it isn't actionable for the Planner.


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

### `do_not_repeat` entries this node writes carry two extra fields

Diagnosis is the node that closes axes, and a closed axis stays closed for the rest of the run.
That is the point — but "rejected" is not one claim, it is several, and collapsing them loses the
distinction that matters most. So every `do_not_repeat` entry written here adds:

- **`evidence_basis`** — what the rejection actually rests on: `validation_only`,
  `train_cv_only`, `cross_run`, `domain_reasoning`, or `external_holdout`. A rejection measured
  on one split is a claim about that split, not about the world. Nothing this run does can
  produce `external_holdout` evidence — that tier exists only to label findings inherited from a
  previous effort that was scored against a held-out set, and it comes with a restriction (see
  below).
- **`reopen_if`** — the specific condition under which this axis becomes worth re-testing, or the
  string `"no condition — closed by mechanism"` when the rejection genuinely generalises. Ruling
  something out by *mechanism* (the reasoning applies to every variant) is much stronger than
  ruling it out by *measurement* (one variant scored badly once), and the two should not read
  identically to whoever inherits them.

These are additive: they do not replace `tried`, `outcome`, `evidence` or `why_not_again`, and
existing entries without them stay valid.

**The failure this prevents is real and has happened.** In `run_20260901T095109Z`, iteration 012
tested a trailing-residual correction, measured it making validation *worse*, and closed the axis.
The same correction was later measured worth **+9 MAE on held-out data** — because validation sat
+2.3% against the training level and the held-out period sat −18.1%, so validation structurally
could not exhibit the effect. With `evidence_basis: validation_only` and `reopen_if: "a period
containing a level shift becomes available"`, that axis would have been reopened rather than
inherited as settled.

### Reading findings from other runs

Prior runs are a legitimate and encouraged source of hypotheses — what was tried, what moved the
metric, and what did not, is exactly the kind of evidence that stops this run repeating work. Two
rules govern it:

- **Carry the tier across.** A finding imported from another run keeps the `evidence_basis` it
  was recorded with, and gains `cross_run`. It never gets promoted by being reused.
- **Never let an `external_holdout` finding drive a decision in this run.** Report it, cite it as
  motivation, and then justify the choice on this run's own validation evidence — or mark the
  hypothesis as untested and hand it to the Planner. This is not pedantry: in
  `run_20260901T095109Z` the 56-day recalibration window was imported from
  `run_20260827T154407Z`, which had measured it on *its* held-out period. That run's own report
  admits the resulting number is optimistic. When runs share a held-out set and findings flow
  forward unmarked, the holdout is spent across the whole project rather than one run, and
  nobody can tell when it happened.

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
- **Diagnosis never runs the next skill and never tells the Planner what it "must" do.**
  `suggested_action` is this call's best recommendation; framing it as anything stronger
  misrepresents the Planner's actual authority over routing.
- **Never propose dropping a feature on its solo importance.** Under correlation a redundant
  feature scores near zero because its twin covers for it; dropping both loses signal that the
  group-level number would have shown. A drop hypothesis cites a group whose joint importance *and*
  masked gap are both ~zero (`../shared-references/feature_attribution.md` §4). In
  `run_20260902T120000Z` this distinction separated six genuinely unused weather features from a
  `month` pair that scored −0.02 solo while carrying ~1.4 MAE.
- **Never propose dropping a feature in `protected_features` as a hypothesis.** It goes to the
  user as a blocking clarification instead. The heuristics that would justify the drop —
  near-zero solo importance, redundancy with an adjacent lag — misfire on this column type by
  construction, and routing it as a hypothesis lets the pipeline remove it without anyone
  deciding to.
- **Never read a naive predictor's dominance as a finding on its own.** A model whose importance
  sits almost entirely on `naive_h{h}_*` is doing what the feature was added for: predicting the
  residual of a decent forecast. That is only a problem when the model also fails to beat
  `baseline_floor` at that horizon — then the model has learned to copy the naive rule and add
  nothing, which is worth saying plainly, in those words, because the metric alone will look
  respectable.
- **Never propose a hypothesis already in `prior_hypotheses_tried` without new evidence explaining
  why it's worth revisiting.** Re-proposing a rejected hypothesis silently wastes an iteration. The
  exception is an entry whose `reopen_if` condition has now been met — that is not a repeat, it is
  the mechanism working, and the entry that authorised it should be cited.
- **Never emit a single hypothesis when a second plausible one exists.** The set is two to three,
  ranked, reaching for different mechanisms where the evidence allows. One hypothesis leaves the
  Planner with nothing to decide, which is how a run stops exploring without anyone choosing to.
- **Never recommend `stop` on diminishing returns alone.** It must be paired with the coverage
  check. Decelerating gains along the axis being worked say nothing about axes never tried, and
  this is the single most likely way for a run to end early while still holding cheap wins.
- **Never recommend `stop` solely because the model beats the naive baseline.** Beating the floor
  is the entry condition for stopping, not the reason to stop. A stop is only available when the
  model beats the floor *and* the coverage check finds no remaining axis whose expected effect
  exceeds the noise band. If the floor is beaten but untried axes remain, the recommendation is
  `continue` with those axes as the hypotheses.
- **Never invent a numeric target that wasn't actually given.** Absent an explicit target, the
  stop/continue read comes from trend and residual/segment evidence, not a made-up threshold.
- **Every clarification gets recorded, whether or not it's blocking** — the `blocking` field is
  informational (severity/urgency), not a filter on what reaches the user; that filtering decision
  belongs to the Planner, not to this skill deciding what's worth asking about.
- **Cross-run history reads are read-only and lightweight-first**, exactly like in-run history
  access: skim summaries, open a specific full file only when a specific need is identified — never
  an open-ended scan across every past run "just to check."
- Every decision — stop/continue, root cause, hypothesis, suggested action — needs `reasoning` and
  `confidence`. A verdict recorded without either is an unrecorded assumption the Planner can't
  weigh properly.
- The `reports/` Markdown file must only restate what is already in the JSON output.

## Input / output contract

Full input/output example: `contracts/diagnosis.json`.

Top-level output sections: `schema_version`,

`per_horizon_verdicts` — **on multi-horizon tasks only**: a dict keyed by horizon value, each
entry containing `stop_recommendation` (`stop`/`continue`), `reasoning`, `confidence`,
`binding_constraint` (when `stop`), `primary_issue` and `root_cause` (when `continue`), and
`hypotheses` for that horizon specifically. This is the authoritative per-model reasoning. On a
single-horizon task this section is absent and the run-level fields below are the only output.

`stop_recommendation` (`recommendation`: `stop` or `continue`; `reasoning`; `confidence`; which
check(s) from step 2's table drove it; on multi-horizon tasks, `horizons_still_continuing` — the
list of horizons whose per-horizon verdict is `continue` — and `horizons_settled` — those whose
verdict is `stop`),

`coverage_check` (node visit counts, model families fitted, untried prior hypotheses, whether the
external reference has been run),

`diagnosis` (`primary_issue`, `evidence` list, `root_cause` — run-level summary; on multi-horizon
tasks this synthesizes across all continuing horizons),

`hypotheses` (a ranked list of 2-3 run-level hypotheses, each with `rank`, `statement`,
`applies_to_horizons`, `suggested_action` — `node`/`action`/`reasoning` — `falsification`,
`cost`, `confidence`, `why_this_rank`),

`ranking_note` (only when all hypotheses point at the same node), `alternative_if_all_wrong`,
`clarification_needed`, `agent_initiated_findings`, `issues`, `artifact_paths` (present only when
step 6's trend plot was generated).

Note the rename: `hypothesis` (object) became `hypotheses` (ranked list). A Planner reading an
older run's history will find the singular form and should treat it as a one-element list.
