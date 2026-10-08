---
name: planner-skill
description: Orchestrates the full ML pipeline for a single dataset and goal by reasoning over state and history to decide, at every point, which skill runs next — not a fixed sequence. Creates the run directory, curates context for whichever skill it chooses, runs report_generator after every skill execution unless the run's report_generator mode is closed (it defaults to open), and surfaces every clarification a skill raises to the user before proceeding. While reports are open, no skill output is considered complete until its HTML report is produced.
---

# Planner Skill

## Purpose

The Planner is the entry point for every pipeline run, and the pipeline's only decision-maker.
Given a dataset path and a goal, it:

1. Creates the run directory and all required sub-directories.
2. Initialises `state/current.json` and a bare-bones `reports/index.html`.
3. At every point — including the very first action — reasons over this run's state and history
   (and, lightly, other runs' history on the same goal) to decide which skill to run next. Nothing
   is fixed-order past what the evidence actually supports; the Planner can jump to any skill,
   including one it has already run before.
4. Curates focused context for whichever skill it chooses.
5. Invokes the `report_generator` skill immediately after every skill execution, and a skill is
   not considered complete until its HTML report exists at the correct path. This is
   unconditional while the run's `report_generator` mode is `"open"` — the default whenever the
   input is absent. Under `"closed"` (see Optional overrides) report generation is suspended for
   the run and nothing else changes.
6. Surfaces every `clarification_needed` entry any skill raises — blocking or not — to the user
   with the raising node's recommended answer, and halts the run outright on a blocking one,
   and waits for an answer before deciding what runs next.
7. Keeps `state/current.json` and `reports/index.html` up to date after every step.
8. Repeats steps 3–7 until a stop is decided (see Step 1c below), then runs `end_model` and a final
   verification pass.

The Planner does **not** perform analysis itself. It reads, reasons, routes, curates, and enforces.
A skill's own recommendation (most notably Diagnosis's `suggested_action`) is one input to that
reasoning, never a command the Planner is obliged to follow.

---

## Inputs

- `dataset_path` — absolute path to the dataset for this run. **This is a training partition, not
  the full dataset.** The genuinely held-out data was carved off before the run started and is
  neither present at this path nor reachable from it. Everything the run does — including the
  train/validation split Preprocess creates — happens inside what this path contains, and every
  number the run produces is a validation number. The unbiased score is produced after the run
  ends, outside this pipeline, against data the run never saw. Record this in
  `current.run_config.holdout` as `"external — scored after the run"` on the first pass, so a
  reader of the run directory is never left wondering where the test metrics went.

  **It may also be more than one file.** `dataset_path` can name a single table, a directory, or a
  set of paths. When it resolves to more than one file, every one of them is in scope for the run
  — there is no "the main file" that the others are optional decoration around. Resolve the full
  list at Step 0, record it as `current.run_config.data_sources`, and pass it whole to EDA. A
  source the run chooses not to use is a decision recorded with a reason; a source nobody opened
  is a hole in the analysis.
- `goal` — plain-language statement of the modelling objective
  (e.g. "forecast daily bread demand in units for branch 01")
- `target_column` — the column name to predict

### Optional overrides

- `run_id` — force a specific run ID (ISO compact timestamp, e.g. `20260821T090000Z`).
  If omitted, generate from the current timestamp: `run_YYYYMMDDTHHMMSSZ`.
- `skip_skills` — list of skill slugs the Planner should never choose this run.
  **Domain Research may never be skipped if a live signal recommends it** (see the decision table
  in Step 1b) — attempting to skip it while such a signal is active is an error.
- `stop_after` — force a stop once this skill slug next completes (inclusive of its report),
  overriding whatever Diagnosis or the Planner's own reasoning would otherwise decide.
- `report_generator` — `"open"` or `"closed"`. **Defaults to `"open"` when absent or
  unrecognised**, which is the behaviour described throughout this skill: `report_generator` runs
  after every skill execution and no skill is complete until its HTML report exists.

  `"closed"` suspends report generation for the run. Nothing else changes — `history/`, `code/`
  and `artifacts/` are still written in full, exactly as under `"open"`, so every report can be
  generated retroactively from what is on disk. `tools/handoff_check.py` validates history, code
  and artifacts and does not look for reports, so a closed run still passes the full check.

  Record the mode and its origin in `current.run_config.report_generator` on the first pass —
  `"closed (user instruction)"` or `"open (default, not specified)"`. A reader of the run
  directory should never have to guess why the reports are missing. Where the rest of this skill
  says report generation is unconditional, read it as unconditional **within `"open"`**; under
  `"closed"` every step that produces a report is skipped and nothing else is.

---

## Available skills

```
eda | domain_research | preprocess | feature_engineering | training | diagnosis | end_model
```

There is no canonical order. In practice, the first several decisions in a run will usually look
like `eda → (domain_research) → preprocess → feature_engineering → training`, simply because
there's nothing to reason from before data is understood, processed, and featurized — but that's an
emergent consequence of Step 1b's decision table, not a hardcoded rule. The Planner could in
principle choose differently even at iteration 1 if the evidence supported it (e.g. re-running EDA
on a re-supplied dataset, or resuming a paused run partway through).

`report_generator` is not in this list because it is never a Planner *decision* — while reports
are `"open"` it is invoked automatically after every other skill (Step 1e), and while they are
`"closed"` it is never invoked at all. Either way it is not reasoned about per iteration; the mode
is set once for the run and then simply obeyed.

---

## Run Directory Layout

Every run lives under:

```
runs/run_{RUN_ID}/
  code/          ← one .py per iteration (e.g. code/iteration_001_eda.py)
  history/       ← one .json per iteration (e.g. history/iteration_001_eda.json)
  artifacts/
    eda/
    domain_research/
    preprocess/
    feature_engineering/
    training/
    diagnosis/
  data/          ← processed CSVs (written by Preprocess, FE)
  artifacts/training/  ← trained model files (written by Training; corrected from an earlier
                          draft of this file, which said `models/` — `artifacts/training/` is
                          what every actual run and `training.md` itself already use)
  reports/       ← HTML reports + index.html
  state/         ← state/current.json (always the latest snapshot)
```

Create all directories at run initialisation. A missing directory is a blocking error.

---

## Procedure

### Step 0 — Initialise run

1. Determine `run_id`. Create `runs/run_{run_id}/` and all sub-directories listed above.

2. Write `state/current.json` (initial):

```json
{
  "run_id": "run_{run_id}",
  "current": {
    "goal": "{goal}",
    "dataset_path": "{dataset_path}",
    "target_column": "{target_column}",
    "task_type": null,
    "stop_condition_met": false,
    "pending_clarifications": [],
    "next_iteration_contract": {
      "node": null,
      "iteration": null,
      "status": "RUN INITIALISED - first routing decision not yet made"
    }
  },
  "history": [],
  "archive": {}
}
```

The contract starts explicitly null rather than absent, so "not decided yet" is never
indistinguishable from "forgotten." Step 1g fills it the moment the first skill completes. Do not
clear context between Step 0 and the first Step 1g — there is no resume point yet, and
`handoff_check.py` will say so.

3. Write `reports/index.html` — a skeleton using the element61 design system (dark navy `#301D56`
   hero, orange logo, tab bar with "By Skill Type" and "Timeline" tabs). Mark every skill card as
   `PENDING`. This file is regenerated by `report_generator` after each step — the skeleton is
   only so the file exists and is not empty during the run.

   The index skeleton must include the run ID, goal summary, and dataset name in the hero. Use the
   same design as `runs/run_20260820T110000Z/reports/index.html` as the canonical reference.

4. **Resolve the data sources.** Expand `dataset_path` into the complete list of **resources** the
   run may read. A sub-directory whose files share one schema (per-day exports, monthly extracts)
   is one resource, recorded once with its file pattern and file count — not one entry per file; a
   standalone file is one resource. Write the list to `current.run_config.data_sources`, each entry
   with its path (or directory plus pattern), format, file count and size. This list is what
   EDA is measured against: its `source_inventory` must account for every entry. Do not pre-judge
   which files matter — a name that looks like metadata (`Winkelfile.csv`, an opening-hours table,
   a holiday list) is exactly the kind of file that explains a structural break nothing else can.

5. **Stamp the conventions version.** Write `current.run_config.conventions_version = "2026-09-04"`
   (or later, if these skills have moved on since). `tools/handoff_check.py` applies the checks
   introduced with that revision — the baseline floor before any learned model, the written stop
   justification, the blocking-clarification halt, one model per horizon, both non-automatic
   time-series approaches — **only** to runs that declare it. A run started under the older
   conventions is not retroactively broken, and one started now cannot quietly skip them.

6. Log: "Planner initialised. Run: {run_id}. Goal: {goal}. Dataset: {dataset_path} ({N} source files)."

---

### Step 1 — Decide, run, and record (repeats until a stop is decided)

This is the entire loop. It runs once to pick the very first skill, and again after every skill
completes, until Step 1c's stop condition is met.

**1a. Gather decision inputs.**

- **If resuming after a context clear** (Step 1h), read `current.next_iteration_contract` first.
  It names the node, says why it was chosen, and carries the curated context that node needs. When
  it is complete, that is the whole read — do not pull the full snapshot for a decision it already
  answers. Fall through to the full read below only if the contract is missing, has a null `node`
  on a live run, or lacks a field you actually need.
- Otherwise read `state/current.json` in full — `current` and the complete `history` array. Never
  rely on in-memory state from an earlier iteration of this loop; the state file is the single
  source of truth.
- If any entries exist in `current.pending_clarifications`, resolve those first — see Step 1f
  before doing anything else this pass.
- Lightweight cross-run check: list `runs/` and, for any other `run_<id>/state/current.json` whose
  `current.goal` matches or closely resembles this run's goal, read just that file's `history`
  array (a few KB — iteration/node/summary tuples, exactly like this run's own). This is a skim,
  not a mandatory deep read: use it to notice "this was tried before" or "that run's approach might
  be relevant," and only then open a specific `history/iteration_<NNN>_<node_slug>.json` from that
  other run if a specific piece of information from it looks like it would actually change this
  decision. Never open-endedly read every file of every past run.

  **Reason about *why* past results differed, not just what they were.** The useful question is
  which change moved the metric and in which direction — a past run's feature set, split design,
  model family or training window, set against its outcome. A prior run that scored worse is as
  informative as one that scored better, provided the difference can be attributed to something
  specific. Record what you took from another run in `why_this_node`, naming the run id, so a
  reader can tell an inherited idea from a fresh one.

  **One restriction, and it matters.** A finding another run measured against *held-out* data
  (`evidence_basis: external_holdout`, or any figure that run described as a final or test
  score) may motivate a hypothesis here but may never justify a decision here. Re-establish it
  on this run's own validation evidence, or hand it forward as untested. When runs share a
  held-out set and its findings flow forward unmarked, the holdout is quietly spent across every
  run that inherits them — which has already happened once in this project, between
  `run_20260827T154407Z` and `run_20260901T095109Z`.

**1b. Decide which skill runs next.** Reason over everything gathered in 1a. There is no scoring
formula — this is a judgment call, the same way Diagnosis's own root-causing is — but the table
below is the floor of what to consider, not the ceiling:

| Situation | Consider |
|---|---|
| **A Training call has completed and has not been diagnosed.** Checked FIRST, before every row below, and it is not conditional on the metrics looking unfinished. Diagnosis is the only node that proposes what to try next, so a Training call that skips it leaves the Planner with nothing to route to and the run walks forward instead of looping | `diagnosis` |
| No EDA has run yet this run, or something learned since makes an earlier reading of the data look wrong, incomplete, or worth re-examining on a specific question | `eda` |
| EDA flagged `domain_research_recommended`, or a specific data-meaning question has come up since (from Diagnosis, or from the Planner's own read of the evidence) | `domain_research` |
| No processed dataset exists yet, or a specific issue has been raised with the current one (leakage, missing/wrong split, unhandled missingness) | `preprocess` |
| No engineered dataset exists yet, or Diagnosis (or the Planner) has a specific feature to add, remove, or reconsider | `feature_engineering` — **but not before the baseline call below has run.** See the ordering rule under the table |
| **A protected feature has a blocking clarification against it** (a node proposed dropping a naive predictor). The run is already halted by the blocking tier; this row exists so the resumption is routed rather than improvised — re-run the node that raised it, carrying the user's answer | the node that raised it |
| **A Training call's grouped feature importance contradicts the rationale Feature Engineering recorded for those features** — a group added as the core signal measuring at or below zero, or a group carrying the model that nobody argued for. The feature set is a hypothesis and this is the measurement that tests it; leaving it untested is how a run ships 37 features of which 6 do anything | `feature_engineering` |
| A processed dataset with a `_split` column exists and no baseline floor has been established | `training` in `baseline` mode — this is always the first Training call of a run, **and it must come before the first `feature_engineering` call** |
| The task is a multi-horizon forecast and a model exists for some horizons but not others | `training`, naming the horizons still missing — the run is not comparable until every horizon in `current.forecast_horizons` has one |
| The task is a multi-horizon forecast and Diagnosis's `per_horizon_verdicts` shows that one or more specific horizons are still `continue` while others are `stop` | `training` with `target_horizons` set to the continuing horizons only — settled horizons keep their current model and are not retrained |
| No trained model exists yet for the current feature set, or there's a specific reason to retrain (different model family, hyperparameter search, a changed feature set, Diagnosis's suggested action) | `training` |
| **Any hypothesis a Diagnosis call ranked is still untested.** Route to the node that tests it, highest rank first, and keep routing until every one is answered. The Planner chooses the *order*, never whether to test them at all — a ranked hypothesis is retired by a measurement, by an existing measurement cited, or by naming the input that makes it impossible, and by nothing else | the node named in `which_node` |
| A node has been visited only once, or only one model family has ever been fitted | revisit that node — see **1b-bis** |
| **No hyperparameter search has ever been run** (`hyperparameter_search` is null on every Training call), or **`feature_engineering` has run only once**. Both are untried axes, and both block a stop in Diagnosis's coverage check. An axis closed by argument rather than measurement is untried — "re-tuning would invalidate the baseline comparison" is not a reason, since a fixed baseline does not move when a model is tuned | `training` with a search, or `feature_engineering` — see **1b-bis** |
| The task is a time series and only one of the two non-automatic approaches (classical statistical vs. ML-as-regression) has ever been fitted | `training`, with context naming the untried approach — see **1b-bis** |
| The external reference (`automl-baseline2`) has never been run and a trained model exists | run it before the next Training call — see **1b-bis** |
| Diagnosis's `stop_recommendation` is `stop`, **every ranked hypothesis in the run has been tested or formally retired**, `python tools/stop_check.py runs/run_{run_id}` exits 0, and the Planner's own reasoning independently concurs | `end_model` |

**One ordering rule is hard: the baseline call precedes the first `feature_engineering` call.**
Everything else in this table is a priority list the Planner may reason its way around; this one
is not. `training` in `baseline` mode produces two things — the floor, and
`naive_predictor_features`, the per-horizon naive predictor specs that Feature Engineering
materialises as protected columns (`baseline-skill/baseline.md` Part B). Those specs are an input
to the first FE call. An FE call made before them builds a feature set with no naive predictor in
it, and nothing later in the run forces a revisit: the columns are simply absent from every model
the run fits, and the absence looks exactly like a design choice. So when both rows match —
"no engineered dataset exists" and "no baseline floor has been established" — the baseline row
wins, every time, with no reasoning required.

Carry `naive_predictor_features` and `protected_features` in the context of **every** subsequent
`feature_engineering`, `training` and `diagnosis` call, not just the first. A node that cannot see
which columns are protected will treat them as ordinary features, and the protection is then a
label in the state file with nothing enforcing it.

**The diagnosis row is checked first, always.** The rows below it are a priority list, and several
Training rows can match at once — "a model exists for some horizons but not others" and "no trained
model for the current feature set" will both keep matching while models are still being fitted. If
Diagnosis sits below them it is never reached, and the run becomes a single forward pass with the
hypothesis-generating node called once at the end to rubber-stamp a stop. That is exactly what
`run_20260904T140437Z` did: three consecutive Training calls, one Diagnosis, no loop, and the
feature set never revised after iteration 003.

**The table lists common situations. It is not the set of permitted moves.** Any node may be
chosen at any iteration whenever reasoning over the evidence supports it — including a node
already run several times, and including one that sits "earlier" in the usual order. There is no
forward-only constraint anywhere in this pipeline. A finding that arrives at iteration 12 can send
the run back to `eda` or `domain_research`, and doing so is an ordinary routing decision, not an
exception needing special justification. What the decision always needs is a recorded reason in
`why_this_node`: what was learned, why it makes an earlier node the right next call, and what
specific question that node is being asked to answer.

The situation to watch for is a finding the modelling loop cannot absorb — a level shift, a
regime break, a residual pattern, a variable behaving unlike its description — where the honest
answer is that nobody has established what the data *means* at that point. Another
`feature_engineering` or `training` pass cannot supply that; `eda` or `domain_research` can. A run
that keeps cycling `training → diagnosis → training` against an unexplained structural break is
not converging, it is stuck, and the way out is usually upstream rather than another model.

**This has never once happened in this project.** Across every run that got past three
iterations, not one has returned to `eda` or `domain_research` after reaching `training` — the
loop has always stayed inside `training`, `diagnosis`, `feature_engineering` and occasionally
`preprocess`. Meanwhile several of those runs concluded with an unexplained level shift as their
headline limitation, while the shop's own `Winkelfile.csv` records both a system migration and the
date the shop went 7/7, and no run has opened it. Treat a long unbroken stretch of modelling-only
iterations as a signal to ask whether the blocker is actually a data-understanding question.

A skill's own suggestion — most importantly Diagnosis's `hypotheses` — is strong evidence, not a
directive. Diagnosis now returns a **ranked set of two to three**, each with a `cost` and a
`falsification` criterion, precisely so this decision has something to weigh. The Planner may run
rank 1, run a lower-ranked one, adapt any of them (e.g. Diagnosis suggested `feature_engineering`,
but the Planner's broader read says the real blocker is upstream in `preprocess`), or override the
whole set. Record which happened and why (see Step 1d).

**Taking rank 1 every time is a signal worth noticing.** The ranking is Diagnosis's opinion formed
from the residuals in front of it; the Planner sees the whole run, including which nodes have gone
unvisited and which axes an earlier call left untried. If several consecutive passes accept rank 1
without examining the alternatives, the routing has become nominal — that is exactly the failure
the ranked set was introduced to fix, and it reappears silently if the lower ranks are never read.

**1b-bis. Weigh breadth, not just the strongest lead.** Before settling on a node, look at what the
run has actually covered: how many times each node has been visited, how many model families have
been fitted, and which hypotheses from earlier Diagnosis calls were never run. A cheap test on an
untouched axis is often worth more than a better-argued test on the axis already being worked,
because the second only refines what is known while the first can reveal that the run has been
optimising the wrong thing. Weigh Diagnosis's `cost` field here: a `cheap` rank-2 hypothesis on an
untried node usually beats an `expensive` rank-1 on the node just worked.

The external reference is the sharpest form of this. Once a trained model exists, fit a generic
search over the agent's own engineered features on this run's own train/validation split: the gap
between it and the agent's model measures exactly what the Planner's model-side work has been
worth. A near-zero gap says further Training iterations are not where the value is, which is
information the loop cannot generate about itself. Do this mid-run, not only at the end. The full
tiered baseline comparison against held-out data happens after the pipeline finishes and cannot
feed back into it.

**On a time-series task, breadth has one named floor.** The run must fit both non-automatic
approaches from `training-skill/references/model_selection_timeseries.md` §1 — classical
statistical forecasting and ML-as-regression on engineered features — at least once each, on the
same validation rows and the same primary metric, before a stop is available. Training reports
which is still missing in `ts_approach_gap`; carry the approaches already fitted forward as
`context.ts_approaches_tried` on every Training call so the node can see its own coverage. An
automatic tool (Auto ARIMA, Prophet, an AutoML forecaster) is a useful reference point but fills
neither slot. The classical slot specifically requires an ARIMA-family fit — ARIMA, SARIMA or
SARIMAX — with plain ARIMA run at least once even when the series' seasonality sends the main fit
to SARIMA; exponential smoothing on its own does not close it. Every run in this project so far has taken the ML-as-regression path by default and
never once fitted a classical model against it — the pipeline has never tested the assumption it
opens with.

**Breadth is the default, not a luxury.** This pipeline is expected to try as many distinct,
reasonable approaches as the evidence supports — several model families, several feature
framings, several treatments of whatever the data's central difficulty turns out to be — rather
than converging early on the first thing that works and refining it. A run that fitted one model
family well is a worse run than one that fitted four and can say why three lost. The limit is
reasonableness, not appetite: an approach needs a stated reason to expect it might help, and its
result is recorded either way.

The guard that makes breadth safe is `split_info.selection_noise_band`. Comparing many candidates
against one validation partition will eventually surface a winner that is just a lucky draw, so
any difference smaller than the band is a tie and must be reported as one. Prefer the
cross-validated estimate whenever `split_info.cv_design` provides one.

**1c. Check for a stop.** A stop is decided when either: Diagnosis's most recent
`stop_recommendation.recommendation == "stop"` and the Planner's own reasoning doesn't override it,
or the run's `stop_after` override has just been satisfied. On a stop, the next skill is `end_model`
(Step 2), not another iteration of this loop.

**The coverage check gates the stop as well.** Read `current.coverage_check`, which Diagnosis
records alongside its recommendation. A stop is not available while a node other than `end_model`
has been visited only once, while only one model family has ever been fitted, while an earlier
Diagnosis call's lower-ranked hypotheses remain untried, while the external reference has never
been run, or — on a time-series task — while either of the two non-automatic approaches in
`training-skill/references/model_selection_timeseries.md` §1 (classical statistical forecasting,
ML-as-regression on engineered features) has never been fitted — where the classical one is
closed only by an ARIMA-family fit with recorded orders, plain ARIMA among them. Diagnosis is required to apply this itself, but the Planner is the one that can see a
deviation the user authorised — a `skip_skills` entry, say — that legitimately makes one of these
unreachable. Where that is the case, record it explicitly rather than treating the gate as passed.

Diminishing returns alone is never sufficient. Decelerating gains describe the axis being worked;
they say nothing about an axis never touched, and a run that stops on them while holding untried
cheap hypotheses has ended early without anyone deciding to.

**The naive floor gates the stop — but beating it does not open one.** The floor is established by the run's first Training call
(`mode == "baseline"`) before any learned model exists, and lives in `current.baseline_floor` —
the seasonal naive for a time series, the majority class for classification, the training mean for
plain regression, scored **on the same rows, and separately for every segment the goal is stated
over.** For a multi-horizon forecast that means per horizon, not pooled; for a per-segment goal it
means per segment. Establishing it first is deliberate: a floor computed at stop time is a floor
computed by a run that already knows what it wants the answer to be.

Beating the naive is a necessary condition for stopping, not a sufficient one. A model that just
cleared the floor should keep improving: the naive is the minimum bar, not the target. A stop is
only available once the model beats the naive **and** Step 1c-bis finds no remaining axis whose
expected effect exceeds the noise band.

| Result | What the Planner does |
|---|---|
| The model beats the naive on every segment AND Step 1c-bis names untried axes worth more than the noise band | **do not stop** — beating the naive is the floor, not the ceiling. Route to the highest-value untried axis |
| The model beats the naive on every segment AND Step 1c-bis names no remaining axis worth trying | a stop is available; weigh it as normal |
| The model beats the naive overall but loses on some segments | **do not stop.** Route to the node that can address those segments and say which ones and by how much |
| The model does not beat the naive overall, and Step 1c-bis names an untried axis worth more than the noise band | **do not stop**, regardless of what Diagnosis recommends. Route to that axis |
| The model does not beat the naive overall, and Step 1c-bis names **no** remaining axis worth trying | **stop, and say so plainly.** See below |

**The honest negative result.** A run that has worked every axis with a stated mechanism, has
nothing left whose expected effect clears the noise band, and still cannot beat the trivial
baseline has produced a real finding: on this data, at this granularity, with these sources, a
learned model does not beat doing nothing. That is a legitimate ending and a useful one — it is
also the only thing standing between this pipeline and an infinite loop on a genuinely hard
problem, since the row above would otherwise forbid stopping forever.

Ending this way has requirements. `stop_justification.binding_constraint` must name why — absent
signal, too few rows, noise that dominates at this granularity — and question 3 must show the
untried axes were weighed and found wanting, not merely unenumerated. `end_model` is told the
champion is the naive baseline, not the least-bad learned model, and the run's conclusion is that
no learned model earned deployment. Never dress this up: a model that loses to the seasonal naive
does not become acceptable because a lot of work went into it.

A pooled average can sit comfortably above the naive while the model loses on a third of the
segments, because the segments it wins are averaged together with the ones it loses. Stopping
there ships a model that is worse than doing nothing over part of its intended range, and the
aggregate metric will never say so. When the Planner overrides this gate — there are legitimate
reasons, such as a segment with too few rows to be measured — record which segments still lose,
by how much, and why stopping anyway is defensible.

Record the per-segment naive comparison in `state/current.json` under
`current.naive_floor_check` each time it is run, so `end_model` and the final report inherit it
rather than recomputing it.

**1c-bis. Answer the exhaustiveness question, in writing, before any stop.** The coverage check
counts things: node visits, model families, untried hypotheses. Counting is not the same as
having asked whether this run did everything worth doing. Before Step 2 may be entered, write
`current.stop_justification` into state, answering all five of these — in prose, not as a
checklist echo:

1. **What is the binding constraint on this model's accuracy?** Name the mechanism: a level shift
   nobody explained, a segment with too few rows, a signal that is genuinely absent from the data,
   irreducible day-to-day noise. "Gains got smaller" is not a mechanism, it is the observation
   that made you ask the question. A stop whose justification cannot name what is holding the
   model back is not justified.
2. **Which axes were worked, and what did each one buy?** Go through them by name — data
   understanding, source coverage, feature framing, target representation, model family,
   hyperparameters, training window, segmentation — with the metric movement attributable to each.
   An axis worked with no measurable effect is as informative as one that paid.
3. **Which axes were never worked, and for each: would it plausibly move the metric by more than
   `split_info.selection_noise_band`?** If the answer is yes for any of them, **a stop is not
   available** — route there instead and say so. If no, say concretely why: what about this
   dataset makes that axis unpromising.
4. **What is on the list that should *not* be tried?** This is the other half of the question and
   it is not rhetorical. Anything whose expected effect is smaller than the noise band, or that
   only refines an axis already shown to be saturated, is declined — explicitly, in
   `considered_and_declined`, with the reason. A run that keeps going because it can always think
   of one more variant is failing this test from the opposite side: it is spending iterations that
   buy nothing and calling that thoroughness.
5. **What would reopen this?** Name the evidence that would make the stop wrong — a held-out score
   far from validation, a segment the goal turns out to care about, a source that arrives later.
   This is what a resuming agent reads before deciding the run is genuinely finished.

The value test cuts both ways, and that is the point. An untried axis is *owed* a run when there
is a stated mechanism by which it could help and an expected effect larger than the noise band;
it is owed nothing when there is not. Neither "we have not tried everything" nor "returns are
diminishing" decides this on its own — the named mechanism does.

`stop_justification` is written before `end_model` is invoked, not after. Written afterwards it
is a rationalisation of a decision already taken, which is exactly the failure it exists to
prevent.

**1d. Build curated context for the chosen skill.** Same discipline regardless of which skill was
picked — see Context Curation Rules below. Record in this call's own reasoning (for the eventual
report) which candidates from 1b's table were considered and why this one won, especially whenever
the choice departed from what Diagnosis suggested.

**1e. Invoke the skill, then invoke `report_generator` immediately** — the second half applies
while the run's `report_generator` mode is `"open"`. Under `"closed"`, run the skill and its output
verification exactly as below, then skip straight to Step 1f: no report is generated and no report
path is verified.

```
history_json_path = history/iteration_<NNN>_<node_slug>.json
run_dir            = runs/run_{run_id}/
```

After the skill completes:
- Verify its required outputs exist (history JSON at minimum; code, data, or artifact files per
  that skill's own contract).
- If anything required is missing, log the violation and **do not proceed** to `report_generator`.

After `report_generator` completes (`"open"` mode only):
- Verify `reports/iteration_<NNN>_<node_slug>.html` exists.
- Verify `reports/index.html` was updated.
- If either is missing, log "PLANNER ERROR: report_generator did not produce expected output for
  {node_slug}. Pipeline halted." and stop.

Under `"closed"`, the skill's own output verification above is still enforced in full. That is
what keeps a closed run recoverable: `history/`, `code/` and `artifacts/` are complete, so every
report can be generated retroactively by running `report_generator` over the finished run.

**1f. Surface every clarification before continuing.** Collect every `clarification_needed` entry
the just-completed skill raised — regardless of its `blocking` value — plus anything still sitting
unresolved in `current.pending_clarifications` from earlier calls. Present all of them to the user
together (not just the newest one), each with the raising node's `recommendation` and what it
costs if that recommendation is wrong, and wait for a reply before moving on to 1g. Do not attempt
to answer on the user's behalf, and do not deprioritize a clarification because it was marked
non-blocking — that field communicates severity, not whether the Planner asks. Fold the answers
into context for whatever skill runs next, and clear the resolved entries from
`current.pending_clarifications`.

**If any outstanding entry is `blocking: true`, the run halts here.** Write
`current.run_status = "halted_awaiting_answer"` naming the entries and the node that raised them,
finish Step 1g's state write so nothing is lost, and stop. Do not invoke the next node, do not
clear context, and do not proceed on the recommendation — stating it is the node's job, acting on
it without an answer is exactly what `blocking` forbids. The run resumes only once the user has
answered every blocking entry; set `run_status` back to `"running"` at that point and continue
from 1g. See **Clarification Handling** for the full tier rules and the entry shape.

**1f-bis. Verify the handoff block.** Before updating state, confirm the just-completed
skill wrote a conformant `handoff` block to its history JSON per
`conventions/handoff_contract.md`:

```
python tools/handoff_check.py runs/run_{run_id} --iteration-only
```

`--iteration-only` is required here. At this point in the pass, `next_iteration_contract` still
holds the *previous* pass's value — Step 1g has not run yet — so the full check would report a
stale resume point that is not actually a defect. The full check is Step 1h's job.

If this reports a blocking problem, the iteration is not complete: the next agent would be missing
something load-bearing, and the failure is silent — it produces a plausible wrong answer rather
than an error. Fix it before proceeding.

Promote into `current` any handoff `parameters` marked `must_not_change`, any `do_not_repeat`
entry, and the CONTENT (never the count) of any list a later node needs. A `do_not_repeat`
entry must travel with its `evidence_basis` and `reopen_if` fields where the writing node
supplied them (Diagnosis always does) — stripping them on the way into state turns a
conditional rejection into a permanent one, which is the failure those fields exist to
prevent.

**1g. Update `state/current.json`.** Append this iteration to `history`
(`{iteration, node, timestamp, summary, full_output_path, report_html, why_this_node,
candidates_considered}`). Refresh whichever `current` fields that skill owns (e.g. `task_type`
after EDA, `preprocess_output_path` after Preprocess, `latest_model_path`/`latest_metrics` after
Training, `stop_condition_met` and the latest `stop_recommendation` after Diagnosis). Add any
newly-raised clarifications from 1f to `pending_clarifications` if they weren't resolved
immediately.

`why_this_node` and `candidates_considered` are Step 1b's reasoning, written down. Recording them
only in this call's own reasoning is not enough: context may be cleared at 1h, and a resuming
Planner that can see *what* ran but not *why* will re-consider options that were already weighed
and rejected. This is the Planner's handoff to itself, and it is as load-bearing as any skill's.

**Then write `current.next_iteration_contract`** — per `conventions/handoff_contract.md`. This is
the field a resuming agent reads first, and the Planner is the only writer of it:

- `node` and `iteration` — what runs next. Both `null` only when the run is complete.
- `why_this_node` — the same routing reason recorded in the history entry.
- `read_this_first` — the one thing that would most change how that node behaves.
- `required_state_fields` — what that node cannot run without. Keep aligned with
  `NODE_REQUIREMENTS` in `tools/handoff_check.py`, which now also requires `data_sources` for
  `eda`, `join_plan` for `preprocess`, and `stop_justification` + `baseline_floor` for
  `end_model`. Note that `forecast_horizons` and `horizon_grid` are different things and both
  exist: the first is the lead times that each get their own model, the second the (usually
  denser) lead times predictions are scored at.
- `context` — the curated bundle built in Step 1d, stored in full. Writing it now, while the
  reasoning that produced it is still to hand, is what lets the next agent read one contract
  instead of the whole snapshot.

Log: "Step complete — {node_slug}, iteration {NNN}. Report written to
reports/iteration_{NNN}_{node_slug}.html. Next: {next_node}."

**1h. Clear context (optional, and safe only here).** A node boundary is the only place context
may be cleared. Clearing is what keeps a long run affordable; clearing at the wrong moment loses
reasoning that was never written down.

Preconditions — all three, no exceptions:

1. `tools/handoff_check.py runs/run_{run_id}` — the **full** check, no flags, run now that Step 1g
   has written the contract — exited **0**. Step 1f-bis's `--iteration-only` pass does not satisfy
   this: it deliberately skips the resume contract, which is the half that matters here. A
   non-zero exit names exactly what a fresh agent would be missing; fix that first. Never clear on
   a judgment call that it "looks complete."
2. Step 1g has written `next_iteration_contract` with a non-null `node`, or `current.run_complete`
   is true.
3. No unresolved entries in `current.pending_clarifications` (Step 1f).

Never clear **inside** a skill call. Mid-skill there is no handoff block on disk yet, so nothing
gates the clear and the loss is silent — the run continues and produces a plausible wrong answer
rather than an error.

After clearing, a fresh agent re-enters at Step 1a. It reads `next_iteration_contract` first; it
reads the rest of `current` only if the contract is missing something it needs. Then return to
Step 1a.

---

### Step 1c-ter — the coverage gate is a script, not a paragraph

Before invoking `end_model`, run:

```
python tools/stop_check.py runs/run_{run_id}
```

It evaluates the coverage conditions in `diagnosis.md` mechanically from `state/` and `history/`:
single-visit nodes, model families, untested hypotheses, the external reference, hyperparameter
search, whether `feature_engineering` was ever revised, and both time-series approaches. Exit 0
means the gate is clear; non-zero names the axes nobody measured.

**A non-zero exit blocks the stop.** Fix it by running the named node, never by arguing the metric
looks finished — that is precisely how `run_20260904T140437Z` closed at iteration 8 with four
single-visit nodes and no hyperparameter search ever run, and its conclusion did not survive the
holdout. The tool checks coverage only; a clear exit does not mean the run *should* stop, it means
the Planner's own reasoning is now the only thing standing between the run and `end_model`.

### Step 2 — end_model

Once Step 1c decides to stop: confirm `current.stop_justification` is written (Step 1c-bis) and
that its answer to question 3 named no untried axis worth more than the noise band. If it is
missing, Step 1c-bis has not been done and the run is not ready to stop — do it now, and be
prepared for it to send the run back into Step 1 instead. Then invoke `end_model` (its skill file
carries the procedure and its own contract governs inputs/outputs — this file doesn't restate
them), and `report_generator` for it exactly as in Step 1e.

Pass `stop_justification` and `baseline_floor` to `end_model` in its context: the champion's
write-up should be able to say what stopped the run and what the model beat, not just which model
won. Where no learned model beat the floor, say so in the invocation — `end_model` names the naive
baseline as the outcome and records that no learned model earned deployment, rather than crowning
whichever learned candidate lost by the smallest margin.

**What `end_model` is, under this pipeline's split design.** It does *not* produce a final,
unbiased performance number, because this run holds no data capable of producing one — the
held-out set was removed before the run began and is scored afterwards, outside the pipeline.
`end_model`'s job is therefore: choose the champion from everything the run trained, **on
validation evidence and nothing else**, refit it on all the data the run was given, freeze that
artifact, and emit its predictions in the form the later comparison expects. State this when
invoking it, and state the rules that follow from it:

- **Selection is on validation (or the CV design Preprocess recorded), with the reason written
  down.** Never rank candidates on anything else. Where `split_info.selection_noise_band` is set
  and the top candidates fall inside it, say they are tied and break the tie on a stated
  criterion — generalisation gap, artifact simplicity, feature count — rather than presenting a
  noise-sized metric difference as a decision.
- **The champion is refit on train + validation before it is frozen, and the refit is what
  ships.** Selection spends the validation partition; freezing the selected fit spends it twice
  and leaves the shipped model fitted on the smaller, older part of the data — missing exactly
  the most recent rows, which are the ones closest to what it will face. Refit means refit and
  nothing else: same features, same hyperparameters, same target transform, no re-tuning. Where
  a hyperparameter came from early stopping, pin it to the value already chosen; it cannot be
  re-derived once validation is inside the fit. Keep both artifacts — the selected fit is the
  evidence behind the choice, the refit is the deliverable — and point `final_model_path`,
  `predictions_path` and `prediction_manifest` at the refit.
- **The refit is never scored.** It has seen the validation rows, so a validation metric taken
  from it is contaminated, lower than the honest one, and sits in the field a reader trusts
  most. `output.refit.metrics` is `null`; the reported metrics stay attached to the selected fit.
- **Every number it reports is a validation number and must be labelled as one.** No field in
  its output may be named or described as a test or final metric.
- **It must emit `predictions_path` and `prediction_manifest`** (see `contracts/end_model.json`).
  The manifest records the model path, feature table, feature order, target transform and its
  inverse, and the prediction key. This is the run's only handoff to the external holdout
  evaluation — a frozen model with no reproducible feature order cannot be scored later, and the
  run then never produces a real number at all.

`end_model` reads `skills/end_model-skill/end_model.md` for its procedure and
`contracts/end_model.json` for its input/output shape, with the contract authoritative where the
two disagree. It was for a long time the one node with no skill file, and its handoff requirements
were written into the contract because of that — they are now in both. State them when invoking it
anyway; they are the requirements this node has historically been most likely to miss:

- It **must** write a top-level `handoff` block to `history/iteration_<NNN>_end_model.json` per
  `conventions/handoff_contract.md`, including the externalisation pass. Being the last node does
  not exempt it: `handoff_check.py` requires a conformant block in every history file, and a
  resumed or extended run reads this one first.
- Its `do_not_repeat` and `methodological_cautions` are the most valuable in the run — they are
  what stops a later agent re-opening an axis this run already closed by measurement.

**Then close the run out** (the Step 1f—1h sequence still applies, in its terminal form):

1. Run `tools/handoff_check.py runs/run_{run_id} --iteration-only` as in 1f-bis.
2. Update `state/current.json`: append the history entry with `why_this_node`, set
   `run_complete: true`, and set `final_model_path`. **Both are required** — `run_complete`
   without `final_model_path` is a blocking error in the validator, because it claims a finished
   run with nothing to point at.
3. Write the terminal `next_iteration_contract`: `node` and `iteration` `null`, a `status` saying
   the run is complete, and `if_resumed` describing what a later agent must know before adding to
   the run (which axes are closed and on what evidence, and that no held-out score exists inside
   this run).
4. Run the **full** `tools/handoff_check.py runs/run_{run_id}`. It should report the run as
   COMPLETE and exit 0. Only then is the run safe to walk away from.

---

### Step 3 — Final verification

After `end_model` completes, perform a post-run audit:

**File audit:** for every entry in `history`, confirm its `full_output_path` and `report_html` both
exist on disk. Confirm `reports/index.html` and `state/current.json` exist. Because the number and
order of skills is no longer fixed, this audit is driven by what's actually in `history`, not a
fixed checklist of five expected files.

**Report audit:** open `reports/index.html`. Confirm every completed iteration has a `status-done`
badge, a valid `href` pointing to a `.html` file that exists on disk, and no broken links.

**Final log:** "Pipeline complete. Run: {run_id}. {N} iterations across {M} skill calls, all
reports written. Final model: {final_model_path}. Open reports/index.html to review."

---

## Context Curation Rules

When building the `context` for whichever skill Step 1d chose:

1. **Always read from `state/current.json`** — never rely on in-memory state from an earlier pass
   of the loop. The state file is the single source of truth.
2. **Never pass raw history JSON files as context.** Extract the relevant fields and curate.
   Downstream skills receive summaries, not full dumps — this applies identically whether the
   context comes from this run's history or another run's.
3. **Domain research findings are always passed downstream** (to Preprocess and FE) once Domain
   Research has run, regardless of how long ago — even if the downstream skill says it already has
   everything it needs.
4. **Context must reference exact column names** when passing leakage or feature lists — use the
   names from the history JSON verbatim.
5. **Pass the split details (row counts + date ranges) to every skill after Preprocess** — every
   downstream skill needs to know the exact train/val/test boundaries.
6. **When context comes from another run**, curate it exactly as tightly as in-run context — a
   one-line summary plus the specific fields that made it relevant, never that other run's raw
   history file.
7. **`current.forecast_horizons` goes to every node after EDA**, as the explicit list, never as a
   count. Preprocess needs it to size the split gap, Feature Engineering to compute each feature's
   `min_horizon` and build the per-horizon targets, Training to know how many models it owes,
   Diagnosis to read per-horizon metrics. A node that receives "4 horizons" instead of
   `[1, 7, 14, 30]` cannot do any of that.
   When routing to `training` for a horizon-specific call, also pass `target_horizons` — the
   subset of `forecast_horizons` to train this iteration. Include the Diagnosis
   `per_horizon_verdicts` entry for each listed horizon in the curated context so Training knows
   what the continuing issue is. Never pass `target_horizons` on a baseline call.
8. **`current.naive_predictor_features` and `current.protected_features` pass in full to every
   `feature_engineering`, `training` and `diagnosis` call** once the baseline has run — the specs
   with their formulas and parameters, not a count and not a list of column names alone. Feature
   Engineering reimplements each formula (its step 2a) and cannot do that from a name; Training
   needs to know which columns belong to which horizon's subset; Diagnosis needs to know which
   columns it may not propose dropping without asking. A node that receives "3 protected features"
   will treat them as ordinary columns and drop one on the first redundancy screen.
9. **EDA's `source_inventory` and `join_plan` pass downstream in full** — to Preprocess and
   Feature Engineering — once EDA has run: which sources were used, which were not and why, and
   the exact join keys and granularity for the ones that were. Preprocess is the node that
   physically combines them, and it cannot do that from a count or a summary sentence. This is the
   one place where "extract the relevant fields" means the whole structure.

---

## Report Generator Enforcement Rules

These rules are non-negotiable:

1. **report_generator is always the next call after any skill.** No exceptions. No skill output
   is considered complete until `reports/iteration_NNN_<node_slug>.html` exists on disk.
2. **If report_generator fails** (file not found, execution error, missing HTML output), the
   pipeline halts. Log the error prominently. Do not proceed to the next skill.
3. **Never silently skip report_generator** — not even in simulation mode, not even when
   revisiting a skill that already produced a report earlier in the run. A re-run of a skill always
   produces a fresh report at that iteration's own path.
4. **index.html must be updated by report_generator** after every skill. If report_generator did
   not update it, update it manually before proceeding (but log this as a report_generator
   violation).
5. **The Planner verifies the output** — after calling report_generator, read the target HTML
   path and confirm it is non-empty and contains the element61 nav pattern
   (`class="nav-logo"`) and the hero section. A file with only `<html></html>` is not acceptable.

---

## Clarification Handling

### The entry shape — every node, every clarification

| Field | What it holds |
|---|---|
| `topic` / `question` | what is being asked, as a complete question |
| `description` | the ambiguity itself, and which downstream decision hangs on it |
| `blocking` | `true` only when proceeding without an answer could materially change the outcome |
| `confidence` | the raising node's confidence in its own reading of the ambiguity |
| `recommendation` | **the answer the raising node would give if it had to choose** — one concrete option, named |
| `recommendation_reasoning` | why that is the best answer available on the evidence the node has |
| `impact_if_wrong` | what changes downstream if the recommendation turns out to be wrong |

**A question without a recommendation is half an output.** A node that can see the ambiguity can
almost always also say which way it leans and why — refusing to means handing the user the node's
own reasoning to do. `impact_if_wrong` is what makes `blocking` legible: a question whose wrong
answer costs one re-run of one node is not blocking; one whose wrong answer invalidates every
metric produced after it is. The Planner still surfaces an entry that arrives without these
fields, and records the omission against the node that raised it.

### Two tiers, one moment of surfacing

Both tiers are surfaced identically — all outstanding entries together at Step 1f, never just the
newest, never deprioritised for being non-blocking. What differs is what may happen next.

| Tier | What the Planner may do |
|---|---|
| `blocking: true` | **The run halts.** No node is invoked, no context is cleared, and the recommendation is stated but **never acted on**. The halt is written to state so it survives a context clear — see below |
| `blocking: false` | Waits one loop pass, as before. If the user defers, or answers some entries and not others, the Planner may adopt the stated `recommendation` and continue — recording it in `current.assumptions` with the node, iteration and entry it came from |

**One kind of blocking entry may never be resolved by adopting its recommendation, even in the
cases where the Planner otherwise could: a proposed drop of a feature in
`current.protected_features`.** These are the naive predictor columns the baseline selected per
horizon (`baseline-skill/baseline.md` Part B), plus anything else the Planner has protected. Every
heuristic in the pipeline that would drop one — multicollinearity, redundancy with a stronger
feature, near-zero solo importance — misfires on this column type for a knowable reason: the
predictor is correlated with the target and with its neighbouring lags *by construction*. The
pipeline cannot distinguish "this heuristic is right" from "this heuristic is wrong here" on its
own, so the user answers. Surface the column, the evidence, the raising node's recommendation, and
what the run loses either way; adopt nothing; wait. If the user says drop it, remove it from
`current.protected_features` in the same update so the question is not asked again, and record the
answer as the drop's reasoning. If the user says keep it, record that too — a heuristic that fired
once will fire again, and the recorded answer is what stops the second FE call re-raising it.

**Halting is a state change, not a pause in conversation.** On a blocking entry, write
`current.run_status = "halted_awaiting_answer"` along with the halting entries' topics and the
node that raised them. A resuming agent reads `run_status` before anything else and re-surfaces
the questions rather than resuming the loop — which is the whole point of writing it down. Clear
`run_status` back to `"running"` only when every blocking entry has an actual answer from the
user. The Planner never answers a blocking question on the user's behalf, and never reclassifies
one as non-blocking to get moving again.

**An adopted recommendation is an assumption, and assumptions get reported.** Every entry in
`current.assumptions` carries what was assumed, why, whose recommendation it was, and what it
would invalidate if wrong. `end_model` and the final report list them — a result resting on three
unanswered questions is a different result from one resting on none, and the reader is entitled to
know which they have.

- Once answered, fold the answer into context for the next skill and clear it from
  `current.pending_clarifications`. An unanswered non-blocking clarification stays in that list
  and is re-surfaced every loop pass until resolved or adopted.

---

## State Schema

`state/current.json` is a living document. Its schema grows as the pipeline progresses. The
Planner is the **only** agent allowed to write the `current` and `history` root keys — skills
may not overwrite these. Skills write their own dedicated `*_summary_ref` fields.

Representative schema mid-run (note `history` has no fixed length or fixed order of node types —
this example shows a loop that has already revisited `feature_engineering` and `training` several
times after an initial pass, which is the normal shape of a real run, not an edge case):

```json
{
  "run_id": "run_20260821T130000Z",
  "current": {
    "goal": "forecast daily bread demand (bread_quantity units sold per day) for Aernoudt Bakery branch 01",
    "dataset_path": "...",
    "target_column": "bread_quantity",
    "task_type": "time_series_forecasting",
    "run_config": {"conventions_version": "2026-09-04", "holdout": "external — scored after the run",
                   "data_sources": [{"path": "data/aernoudt_bread_daily.csv", "format": "csv", "size_kb": 412}]},
    "forecast_horizons": {"unit": "days", "values": [1, 7, 14, 30], "source": "user goal, confirmed at iteration 1"},
    "models_by_horizon": {"1": "artifacts/training/iteration_014_training_model_h1.pkl",
                          "7": "artifacts/training/iteration_014_training_model_h7.pkl"},
    "ts_approaches_tried": ["ml_as_regression"],
    "naive_predictor_features": {"established_at_iteration": 3, "selection_basis": "train_rows_only",
                                 "1": [{"column_name": "naive_h1_last_value", "rule": "last observed value", "selected_by": "part_a_winner", "train_mae": 24.8},
                                       {"column_name": "naive_h1_trailing_mean_w7", "rule": "trailing mean, w=7 observed periods", "selected_by": "train_search", "train_mae": 26.1}],
                                 "7": [{"column_name": "naive_h7_seasonal_median_w13", "rule": "seasonal median, P=7, window=13 seasons", "selected_by": "part_a_winner", "train_mae": 33.4}],
                                 "ref": "history/iteration_003_training.json"},
    "protected_features": [
      {"column": "naive_h1_last_value", "reason": "naive predictor selected for h=1", "protected_since_iteration": 3},
      {"column": "naive_h7_seasonal_median_w13", "reason": "naive predictor selected for h=7", "protected_since_iteration": 3}
    ],
    "stop_condition_met": false,
    "run_status": "running",
    "pending_clarifications": [
      {"topic": "weather_forecast_feed", "raised_by": "training", "raised_at_iteration": 14, "blocking": false,
       "recommendation": "assume observed weather only, and exclude every weather feature from the live_forecast feature set",
       "recommendation_reasoning": "no forecast feed is documented in any source file; excluding is the reversible choice",
       "impact_if_wrong": "up to 6 usable features are being left on the table; costs one FE + training pass to reverse"}
    ],
    "assumptions": [
      {"assumed": "weather features excluded from live_forecast", "from_clarification": "weather_forecast_feed",
       "raised_by": "training", "adopted_at_iteration": 15, "impact_if_wrong": "..."}
    ],
    "eda_summary_ref": "history/iteration_001_eda.json",
    "domain_research_ref": "history/iteration_001b_domain_research.json",
    "preprocess_output_path": "data/iteration_007_preprocess.csv",
    "fe_output_path": "data/iteration_013_feature_engineering.csv",
    "latest_model_path": "artifacts/training/iteration_014_training_model.pkl",
    "latest_metrics": {"val_mae": 21.09, "val_mase": 0.672},
    "latest_stop_recommendation": {"recommendation": "continue", "confidence": "medium"},
    "baseline_floor": {"established_at_iteration": 3, "baseline_type": "seasonal_naive", "seasonal_period": 7,
                       "pooled": {"mae": 38.05, "rmse": 51.2, "smape": 14.7},
                       "per_segment": {"weekday": {"mae": 33.1}, "saturday": {"mae": 61.4}},
                       "per_horizon": {"1": {"mae": 31.0}, "7": {"mae": 38.1}, "14": {"mae": 41.9}, "30": {"mae": 45.2}}},
    "stop_justification": {
      "binding_constraint": "an unexplained level shift in Sep 2023 that no feature accounts for; residuals before and after it have different means",
      "axes_worked": [{"axis": "feature framing", "iterations": [8, 11, 13], "bought": "-2.4 MAE"},
                      {"axis": "model family", "iterations": [14, 16], "bought": "within noise band"}],
      "axes_untried": [{"axis": "domain research on the migration date", "expected_effect": "larger than noise band", "verdict": "blocks the stop — routing there"}],
      "considered_and_declined": [{"candidate": "a fourth GBT hyperparameter sweep", "reason": "the axis is saturated; last two sweeps moved less than the noise band"}],
      "would_reopen_if": "the held-out score lands far from validation, or the shop confirms a second migration"
    },
    "coverage_check": {"node_visits": {"eda": 1, "preprocess": 2, "feature_engineering": 3, "training": 4, "diagnosis": 3},
                       "model_families_fitted": ["lightgbm"], "untried_hypotheses": [{"from_iteration": 7, "rank": 2, "statement": "..."}],
                       "external_reference_run": false},
    "prior_hypotheses_tried": [
      {"iteration": 8, "statement": "lag_14 captures bi-weekly patterns", "outcome": "confirmed"},
      {"iteration": 11, "statement": "is_covid_lockdown captures 2020 disruption", "outcome": "rejected"}
    ],
    "next_iteration_contract": {
      "node": "training",
      "iteration": 15,
      "why_this_node": "iteration 14 is the best model so far but has not been diagnosed against a frozen-hyperparameter rerun; this is a training change, not an FE one",
      "read_this_first": "hyperparameters are frozen from iteration 9 -- re-tuning invalidates the comparison this call exists to make",
      "required_state_fields": ["fe_output_path", "feature_list", "recommended_metric"],
      "context": {"...": "the curated bundle for this node, written by Step 1d"}
    }
  },
  "history": [
    {"iteration": 1,  "node": "eda",                  "summary": "...", "why_this_node": "...", "full_output_path": "history/iteration_001_eda.json", "report_html": "reports/iteration_001_eda.html"},
    {"iteration": "1b","node": "domain_research",      "summary": "...", "why_this_node": "...", "full_output_path": "history/iteration_001b_domain_research.json", "report_html": "reports/iteration_001b_domain_research.html"},
    {"iteration": 2,  "node": "preprocess",            "summary": "...", "why_this_node": "...", "full_output_path": "history/iteration_002_preprocess.json", "report_html": "reports/iteration_002_preprocess.html"},
    {"iteration": 3,  "node": "feature_engineering",   "summary": "...", "why_this_node": "...", "full_output_path": "history/iteration_003_feature_engineering.json", "report_html": "reports/iteration_003_feature_engineering.html"},
    {"iteration": 6,  "node": "training",              "summary": "...", "why_this_node": "...", "full_output_path": "history/iteration_006_training.json", "report_html": "reports/iteration_006_training.html"},
    {"iteration": 7,  "node": "diagnosis",             "summary": "...", "why_this_node": "...", "full_output_path": "history/iteration_007_diagnosis.json", "report_html": "reports/iteration_007_diagnosis.html"},
    {"iteration": 8,  "node": "feature_engineering",   "summary": "...", "why_this_node": "...", "full_output_path": "history/iteration_008_feature_engineering.json", "report_html": "reports/iteration_008_feature_engineering.html"},
    {"iteration": 9,  "node": "training",              "summary": "...", "why_this_node": "...", "full_output_path": "history/iteration_009_training.json", "report_html": "reports/iteration_009_training.html"}
  ]
}
```

### Retention: `current` vs `archive`

`current` is a **live** snapshot, not a ledger. Every key in it must describe the state of the run
*now*. Keys that describe a superseded moment move to a sibling `archive` root key; they are not
deleted, because the reports and the audit trail still reference them.

Two rules keep it live:

1. **Supersede on write.** When a node produces the newer version of something `current` already
   holds, the older value moves to `archive` in the same update. `iteration_009_metrics` does not
   sit next to `iteration_011_metrics`; the first moves and the second stays as `latest_metrics`.
2. **One key per fact.** If you are about to add a key that is a variant of one already present,
   you are recording a *new value*, not a new fact — update the existing key. A `current` holding
   `three_model_comparison`, `four_model_comparison`, and `model_family_comparison` has one fact
   and three keys, and a resuming agent cannot tell which is authoritative.

Anything named for the iteration that produced it (`iteration_009_metrics`,
`features_added_iteration_008`, `metric_comparison_006_vs_009`) belongs in `archive` the moment a
later iteration supersedes it. The per-iteration detail already lives in `history/` and in that
iteration's `handoff` block, which is where a later agent should read it from.

`tools/state_prune.py runs/run_<id>` reports which keys look superseded and, with `--apply`, moves
them. Run it periodically; a `current` that has grown past roughly 20 KB is usually carrying a
run's worth of dead keys on the hot path of every resume.

Skills read `current`. Nothing reads `archive` except an agent that has followed a specific
reference into it.

---

---

## Guardrails

- **Generated code is a cell-marked `.py`, and every plot names itself**
  (`conventions/code_and_plot_conventions.md`). Scripts carry `# %%` markers so the same file
  stays diffable and converts to a notebook on demand — never write a `.ipynb` as the artifact.
  Every figure carries its declared `name` as the title, its declared `type` stated in the
  figure, and its `iteration - node` provenance, using the same strings recorded in the JSON. A
  plot that is unidentifiable once separated from its report is not finished.
- **Never let an iteration close without a validated handoff block.** Context may be cleared
  at any node boundary; `conventions/handoff_contract.md` defines what must be on disk for that to
  be safe, and `tools/handoff_check.py` is the check. Counts are not content: promoting
  `n_recommendations: 12` into state instead of the twelve recommendations is the exact failure
  the contract exists to prevent.
- **Never clear context anywhere but Step 1h, and never with a precondition unmet.** A clear is
  safe only when `handoff_check.py` exited 0, `next_iteration_contract` is written, and no
  clarification is outstanding. Clearing mid-skill is never safe: no handoff exists on disk yet,
  so nothing gates it and the loss is silent.
- **Never close a pass without writing `next_iteration_contract`.** It is the only thing telling a
  fresh agent what runs next. A live run whose contract has a null `node` has lost its resume
  point, and the next agent must re-derive routing from prose.
- **Never leave the routing reason only in reasoning.** `why_this_node` goes into the history
  entry and the contract. The Planner's own decision trail is lost at a clear exactly like a
  skill's would be, and nothing else on disk records why one node was picked over another.
- **Never run analysis yourself.** The Planner reads, reasons, routes, curates, and enforces — it
  does not compute statistics, build features, or train models.
- **Never invent context.** When building curated context for a downstream skill, use only what is
  in the history JSON files (this run's or another run's). If a field is absent, note it as
  absent — do not fill in a plausible value.
- **Never treat a skill's own suggestion as binding.** Diagnosis's `hypotheses` are one input to
  Step 1b's reasoning, not a command — the Planner weighs them against the rest of state and
  history and records when and why it departs from the ranking. Read the lower-ranked entries
  before accepting rank 1; accepting rank 1 unexamined across several passes is how routing
  authority becomes nominal.
- **Never stop on diminishing returns without the coverage check.** A node visited once, a single
  model family, an untried lower-ranked hypothesis, or an unrun external reference each block a
  stop. If a run is ending with any of those true, say so and say why it is acceptable.
- **Never route to `feature_engineering` before the baseline call has run.** The baseline
  produces the naive predictor specs FE materialises; an FE call made first builds a feature set
  those columns are simply missing from, and no later routing rule notices, because an absent
  feature looks identical to a feature nobody wanted. This is the one hard ordering constraint in
  the table.
- **Never resolve a protected-feature drop by adopting the recommendation.** A blocking entry
  naming a column in `current.protected_features` is answered by the user or not at all. Adopting
  the recommendation here would let the pipeline remove, on a heuristic known to misfire on that
  exact column type, the one feature the run was told to keep — and it would be recorded as an
  assumption rather than a decision.
- **Never treat a multi-horizon forecast as one model.** Every horizon in
  `current.forecast_horizons` gets its own model, its own feature subset (a lag shorter than the
  horizon is not available at that origin) and its own metrics. Pooling horizons into one model
  and reporting per-horizon numbers from it measures something nobody asked for: the short
  horizons carry the fit and the long ones inherit a level they cannot actually see.
- **Never let a stop be decided on pooled horizons.** The coverage check, the naive floor and
  Step 1c-bis all read per horizon. A champion that wins at h=1 and loses at h=30 is two different
  results, and the average is neither of them.
- **A settled horizon does not need to be retrained.** When Diagnosis marks some horizons `stop`
  and others `continue`, route a training call with `target_horizons` set to the continuing ones
  only. Do not retrain a settled horizon unless a later Feature Engineering or Preprocess change
  would invalidate its model.
- **Never fit a learned model before the floor exists.** The run's first Training call is
  `mode == "baseline"`, and every later iteration reports `vs_baseline_floor` per segment. A run
  that discovers its own floor at the end has spent every intermediate decision without knowing
  what "better" meant.
- **Never let the floor make the run unstoppable.** Failing to beat the naive baseline blocks a
  stop only while an axis worth trying remains. Once Step 1c-bis names none, the run stops and
  reports the negative result honestly — that is an outcome, not a failure to be worked around.
- **Never enter Step 2 without `stop_justification`.** Naming the binding constraint, what each
  worked axis bought, which untried axes could still beat the noise band, and which candidates
  were declined as low-value is the pre-stop question in full. A stop that cannot answer question
  1 with a mechanism is a stop taken because the run got tired.
- **Never spend an iteration on an axis you cannot state a mechanism for.** Breadth is the
  default, but an approach earns its run by naming how it could plausibly help and by an expected
  effect larger than `split_info.selection_noise_band`. Variants that only refine a saturated axis
  are declined in `considered_and_declined`, not run to look thorough.
- **Never inherit a `do_not_repeat` entry as settled without reading its `evidence_basis`.** An
  axis closed on `validation_only` evidence is a claim about that split, not about the world; if
  the entry carries a `reopen_if` condition and that condition is now met, re-opening it is the
  mechanism working, not a repeat.
- **Never skip report_generator.** If the user asks to skip it, explain that the planner design
  requires it and decline. Offer to run it immediately after the skill if that was the confusion.
- **Never advance past Step 1f if a clarification is still unresolved.** An unanswered
  clarification — blocking or not — blocks the next loop pass, not just blocking ones.
- **Never run a node while `run_status == "halted_awaiting_answer"`.** A blocking clarification
  halts the run until the user answers it. Not "until it looks answerable," not "until the
  recommendation seems safe enough" — the recommendation exists to inform the user's answer, not
  to substitute for it. Reclassifying a blocking entry as non-blocking to resume is the same
  violation wearing a different label.
- **Never adopt a recommendation silently.** A non-blocking question the user declines to answer
  may be resolved by the raising node's own recommendation, but it lands in `current.assumptions`
  with its provenance and its `impact_if_wrong`, and the final report lists it. An assumption that
  is not written down is indistinguishable from a fact, which is how a run's caveats disappear.
- **Never advance to the next skill if the current skill's report is missing.** Every iteration's
  report is a prerequisite for the next one, regardless of which skill produced it.
- **Iteration numbers are globally chronological, not per-node.** One counter, incremented every
  time any skill runs — including a skill visited for the second or fifth time. Do not restart
  numbering for a revisited node, and do not reserve a fixed suffix for any particular node: an
  earlier version of this design hardcoded Domain Research to always be iteration `001b`, which
  assumed it could only ever run once, early. Domain Research can now be chosen at any point,
  including more than once — it takes the next global iteration number like any other skill.
- **Cross-run history reads are read-only and lightweight-first.** Skim other runs' `history`
  summaries; open a specific full file only when a specific need is identified. Never write to
  another run's folder. Findings another run measured on held-out data motivate but never
  justify — see Step 1a.
- **The `current` key in state always reflects the most recent completed step** — it is updated
  immediately after each skill+report pair (Step 1g), not after the full pipeline.
