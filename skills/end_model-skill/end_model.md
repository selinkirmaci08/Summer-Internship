---
name: end_model-skill
description: Closes a run once the Planner has decided to stop. Chooses the champion from every model the run trained, on validation evidence and nothing else; refits that champion on train + validation so the shipped artifact uses all the data the run was given; freezes it; and emits the predictions and manifest the external holdout evaluation needs. Produces no held-out number and never claims one — the unbiased figure is produced after the pipeline finishes, by whoever holds the holdout.
---

# End Model Skill

## Purpose

End model is the last node of a run. It does **not** produce a final, unbiased performance number,
because the run holds no data capable of producing one: the held-out set was removed before the run
began and is scored afterwards, outside the pipeline. Four jobs, in this order:

1. **Choose** the champion from every model the run trained — on validation evidence and nothing
   else, with the reason written down.
2. **Refit** that champion on train + validation, so the artifact that ships is fitted on all the
   data the run was given rather than the fraction left over after selection.
3. **Freeze** the refit and emit its predictions plus a `prediction_manifest`, which is the run's
   only handoff to the external holdout evaluation.
4. **Write the run up** — what stopped it, what the model beat, what is still unknown.

Steps 1 and 2 pull in opposite directions and both are mandatory. Selection must never see anything
but validation; the shipped model must never be fitted on less than everything. The way both hold at
once is that selection finishes first and is never revisited: the refit changes no choice, it only
changes how much data the chosen specification was fitted on.

## When this runs

Invoked by the Planner once Step 1c has decided to stop and `current.stop_justification` is written.
It consumes an iteration number like any other node and is followed by `report_generator`, then
optionally the user-invoked `final_report` skill once the holdout has been scored.

A stop is not available while a node other than End model is next, so by the time this runs the
routing decision is closed. If `stop_justification` is missing, say so and hand back — the run is
not ready to close, and writing the justification afterwards makes it a rationalisation of a
decision already taken.

## Reference material

- `contracts/end_model.json` — the full input/output example. **Authoritative** where this file and
  it disagree.
- `conventions/handoff_contract.md` — the handoff block specification, validated by
  `tools/handoff_check.py`
- `conventions/naming_conventions.md` — paths for every artifact this node writes, including
  `models/<stem>.pkl` for the refit
- `conventions/code_and_plot_conventions.md` — cell-marked `.py`, self-naming figures
- `../shared-references/feature_attribution.md` — required if the write-up says anything about which
  features the model uses
- `../training-skill/references/evaluation_metrics.md` — for MASE, skill scores, and the
  overfit-ratio read used as a tiebreak

## Inputs

From the Planner's call and `state/current.json`:

- `goal`, `run_id`, `iteration`, `final_stop_reason`
- `stop_justification` — what closed the run, written before this node was invoked
- `baseline_floor` — the naive floor the run has been measured against throughout
- `candidates` — **every** model the run trained, built per `candidates_built_from`:
  `state.current.all_saved_models` joined to each model's validation metrics from its training
  history JSON. Never `state.current.latest_model_path` alone — taking only the latest silently
  reduces this node to a one-model run with nothing to hold the ranking against.
- `selection_criteria_source` — the tiebreak criterion, normally from the last Diagnosis handoff
- `split_info` — partition dates, `cv_design`, `selection_noise_band`, `holdout_representativeness`
- `fe_output_path`, `feature_list`, `recommended_metric`, `horizon_grid`
- `durable_rules_in_force` — the accumulated rules from every prior handoff in the run

## Tools available

Read/write of the run directory, and whatever training library produced the candidates. The refit in
step 4 needs the same library and version that fitted the candidate — a champion saved as a
LightGBM booster is refit with LightGBM, not with a lookalike wrapper.

## Procedure

### 1. Assemble the full candidate set

Build the table from `candidates_built_from`. Every model the run trained appears, including the
ones that lost badly and the baseline floor. A candidate whose validation metrics cannot be found is
listed with the reason rather than dropped.

Where the run is horizon-conditioned, a "candidate" may be a set of per-horizon models. Keep it as
one candidate — the choice is between specifications, not between individual boosters.

### 2. Rank on validation, and respect the noise band

Rank on `recommended_metric` over the validation partition, or over the CV design recorded in
`split_info.cv_design` where one exists — that estimate is the better one and should be preferred
when both are available.

Then check the spread against `split_info.selection_noise_band`. Where the top candidates fall
inside it they are **tied**, and saying so is mandatory. Reporting a winner on a noise-sized
difference is reporting a coin flip in the voice of a measurement. Record the tie in
`tie_declared` with the models, the spread, and the band.

### 3. Choose, and write down what decided it

Break a tie on a stated criterion that is not the metric — generalisation gap, artifact simplicity,
feature count — and name where the criterion came from (`selection_criteria_source`). Record
`choice.selected_on = "validation"`, the reasoning, and what was rejected and why.

Where no learned candidate beat `baseline_floor`, the naive baseline is the outcome. Say that
plainly and record that no learned model earned deployment. Crowning whichever learned candidate
lost by the smallest margin is the failure this rule exists to prevent.

### 4. Refit the champion on train + validation

**This is the step that produces the artifact that ships.** Refit the chosen specification on the
training partition *and* the validation partition together.

Refit means refit and nothing else:

- identical feature list, identical hyperparameters, identical target transform
- no re-tuning, no re-selection, no new features, no second look at the candidate table
- where a hyperparameter was set by **early stopping** against validation — `n_estimators`,
  `best_iteration`, an epoch count — pin it to the value already chosen. It cannot be re-derived
  once validation is inside the fit, and there is no honest window left to derive it from. Record
  the pinned value and note that pinning is mildly conservative: the optimal count is usually a
  little higher on more rows, so this understates the refit rather than flattering it.

Why this is not optional: selection *spends* the validation partition. Freezing the selected fit
spends it twice — once to choose, once by never using it — and leaves the deliverable fitted on the
smaller and older part of the data. The withheld rows are the most recent ones, which makes them the
rows closest to whatever the model will actually face. On a drifting series that is the worst
possible slice to leave out.

Keep both artifacts and record both paths. The selected fit is the evidence behind the choice and
stays where Training wrote it; the refit is the deliverable and goes to
`models/iteration_<NNN>_end_model.pkl`. `final_model_path` points at the refit.

Where the champion has no fit step — a naive baseline, a rule, a constant — record
`refit.performed = false` with the reason, and `final_model_path` stays whatever the champion is.

### 5. Verify the refit, then freeze

Before anything downstream uses it, assert against the selected fit:

- feature order identical, name for name and position for position
- target transform identical, and its inverse still present
- the artifact loads by the same call the manifest will document

A refit that silently reordered its columns produces predictions that look plausible and are wrong,
and nothing downstream can detect it. Check it here or it is not checked at all.

**Do not score the refit.** See the guardrail below — this is the one mistake in this node that
produces a number rather than an error.

### 6. Emit predictions and the manifest

Write the refit's predictions to a CSV keyed the way the downstream comparison expects — for a
horizon-conditioned forecast, `target_date, h, y_pred` — and a `prediction_manifest` recording:

- the model path and the exact call that loads it
- the feature table, and the feature order as a list
- the target transform and its inverse, or `identity` where there is none
- the prediction key and what an origin means

Without this the model cannot be scored against the holdout later, and the run never produces a real
number at all. A frozen model with no reproducible feature order is not a deliverable.

### 7. Write the run up

`headline`, `final_validation_metrics`, `validation_error_by_horizon`, `model_comparison`,
`hypotheses_summary`, `remaining_risks`, `deployment_recommendation`. Every metric carries its
validation label. The headline says what the model beat, what stopped the run, and what is not yet
known — in that order, because the third is the part a reader skips.

`deployment_recommendation.ship` is normally "do not ship on validation evidence alone — score the
frozen model against the external holdout first." A run that has never seen held-out data is not in
a position to recommend deployment, and saying so is the honest end of the run.

At least one plot: the candidates on validation with the noise band drawn as a shaded band. Drawing
the band is what makes the tie visible; a bare ranking chart argues the opposite of the text.

## Handoff block (required)

Before writing the output, add a top-level `handoff` block to
`history/iteration_<NNN>_end_model.json` per `conventions/handoff_contract.md`, including the
externalisation pass.

Being the last node is not an exemption. `tools/handoff_check.py` requires a conformant block in
every history file, and a resumed or extended run reads this one first. This node's `do_not_repeat`
and `methodological_cautions` are the most valuable in the run — they are what stops a later agent
re-opening an axis this run already closed by measurement.

`parameters.final_model_path` is the refit, marked `must_not_change: true`, with a `why` that says
explicitly it is the refit and not the selected fit — otherwise a later agent reads the training
artifact path out of the candidate table and ships the wrong one.

## Guardrails

- **Selection is on validation, or the recorded CV design, and nothing else.** There is no held-out
  partition in this run to choose on, and inventing one by carving a third slice out of the training
  data does not create one — it produces a smaller validation set with a more authoritative name.
  This is the single most important rule in the node.
- **The refit is never scored, and `refit.metrics` is `null`.** It has been fitted on the validation
  rows, so any validation metric taken from it is contaminated — and contaminated *downward*, which
  makes it the most convincing wrong number the run could produce: better than the honest figure,
  sitting in the field a reader trusts most, with nothing in the output to mark it. Every metric
  this node reports stays attached to the **selected fit** and keeps its validation label. The
  refit's real number comes from the external holdout, later, from whoever holds it.
- **No field may be named or described as a test, final, or held-out score.** Every number here is a
  validation number. `final_validation_metrics` carries an explicit label saying so.
- **Never present a difference smaller than `split_info.selection_noise_band` as an improvement.**
  Below the band the ranking is a lucky draw.
- **Compare every candidate, not the latest one.** See `candidates_built_from` in step 1.
- **Never end a call without a `handoff` block.** Context may be cleared immediately after this node
  returns; anything load-bearing that exists only in prose is lost at that point.
- **Generated code is a cell-marked `.py` and every plot names itself**
  (`conventions/code_and_plot_conventions.md`).

## Input / output contract

Full input/output example: `contracts/end_model.json` — authoritative where this file disagrees.

Top-level output sections: `schema_version`, `node`, `iteration`, `run_id`, `contract`,
`final_model_path` (the refit), `refit` (`performed`, `selected_model_path`, `shipped_model_path`,
`fitted_on`, `fitted_through`, `rows_selected_fit`, `rows_refit`, `pct_of_available_rows`,
`unchanged_from_selected_fit`, `pinned_from_selection`, `metrics` — always `null`, `why_no_metrics`,
`verification`), `choice` (`chosen`, `selected_on`, `selection_noise_band`, `criteria_source`,
`reasoning`, `rejected`), `tie_declared`, `seasonal_naive_reference`, `final_validation_metrics`,
`all_candidates_on_validation`, `validation_error_by_horizon`, `model_comparison`,
`hypotheses_summary`, `headline`, `deployment_recommendation`, `remaining_risks`, `artifact_paths`,
`predictions_path`, `prediction_manifest`, `issues`, `final_summary`, `report_html`, `handoff`.
