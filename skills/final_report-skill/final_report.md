---
name: final-report-skill
description: Produces the final self-contained HTML report for a finished modelling effort — comparing the agentic run, the AutoML baseline arms, and a naive baseline on one shared held-out test set, with per-horizon tables and charts, an error decomposition, and grouped explainability. User-invoked with the paths to each system's predictions (or models); never routed by the Planner and never part of an iteration loop.
---

# Final Report Skill

## Purpose

One document that answers three questions about a finished modelling effort, in this order:

1. **Did it beat the alternatives?** — agentic vs AutoML baseline vs naive, on one shared test set.
2. **Where and why does it win or lose?** — per horizon, per period, and split into level error vs
   shape error, so "worse than naive" becomes a mechanism rather than a verdict.
3. **What is the model actually using?** — grouped explainability, so the answer survives correlated
   features.

The output is a **single self-contained HTML file** written directly by this skill (no
`report_generator` pass, no external assets). It is a document a supervisor reads end to end, not a
slideshow — see `references/report_layout.md` for the section list and chart specs.

## When this runs

**User-invoked only.** The user triggers it after all systems have been fitted and scored on the
same held-out test set. It is not in the Planner's node vocabulary, does not consume an iteration
number from the run counter, and never loops. It may be re-run any number of times; a re-run
overwrites its own output and nothing else.

Because it is invoked directly, its inputs come from the invocation, not from `state/current.json`.
Read state only to *enrich* the report (run ids, split definition, feature list, hyperparameters) —
never to decide which systems to compare.

## Inputs

A `systems` mapping, one entry per system to compare. Each entry gives **either** predictions or a
model, plus a label and a role:

```yaml
test_window:   [2020-12-13, 2023-02-08]     # optional; inferred and verified if absent
key:           [target_date, h]              # join key; [date] for single-horizon problems
target_column: bread_quantity
systems:
  agentic:
    label: "Agentic run (iter_006 residual)"
    role: champion
    predictions: runs/run_.../data/iteration_006_predictions.csv
    pred_column: pred_residual
  automl_b1:
    label: "AutoML · agent features"
    role: baseline
    predictions: automl-baseline2/outputs/run_.../predictions.csv
    pred_column: pred_B1_agent_features
  naive:
    label: "Seasonal naive"
    role: naive                              # exactly one system must carry this role
    predictions: ...
    pred_column: pred_naive
explainability:
  model:    runs/run_.../artifacts/training/iteration_006_lgbm_residual.pkl
  features: runs/run_.../data/iteration_003_feature_engineering.parquet
  feature_order: [...]                       # required — pickles rarely carry names
  scoring_split: validation                  # never test; see step 6
```

**If a system supplies `model` instead of `predictions`**, this skill loads it and scores it, and
must be told `feature_order`, and — for a residual or ratio model — the `base_column` and any
clipping, exactly as the training code applied them. Getting this wrong silently produces a wrong
comparison, so a system whose reconstruction cannot be stated is reported as **not scored** rather
than guessed at.

Optional: `run_dir` (to enrich provenance), `output_path` (defaults to
`<run_dir>/reports/final_report.html`).

## Tools available

- Read access to every path named in the invocation, plus `state/current.json` and `history/` of any
  run referenced
- Model loading and `.predict()` — only for systems supplying `model`, and for the explainability
  section
- Plotting; write access to the output HTML and to `artifacts/final_report/` for the paired CSVs

## Reference material

- `references/report_layout.md` — section list, what each section must contain, chart specs,
  and the self-contained HTML rules
- `references/comparison_protocol.md` — alignment rules, the metric set, and the error
  decomposition
- `../shared-references/feature_attribution.md` — **binding** for section 7. Grouped permutation
  importance, the masking test, and why split-count importance is never reported
- `../../conventions/code_and_plot_conventions.md` — cell-marked `.py`, every plot self-identifying,
  every plot paired with a CSV

## Procedure

**Every step is a floor, not a ceiling.** A check beyond these that would change a conclusion is
worth running; record it in `agent_initiated_findings`.

### 1. Align the systems before computing anything

This is the step that makes the report true, and it is the one most likely to be skipped.

a. Load every system's predictions. Inner-join all of them on `key`. **The comparison set is the
   intersection**, never one system's own rows.
b. Report `n` for each system before the join and after. If any system loses more than 5% of its
   rows to the join, that is a finding, stated in the report — not a silent truncation.
c. **Verify the test windows match.** If two systems were scored on different date ranges, they are
   not comparable, and no amount of shared-window filtering fixes the fact that they were *fitted*
   against different splits. Restrict to the overlap, and state plainly in the report that the
   comparison is on the overlap and why.
d. **Check the horizon grids match.** Where a system's grid lacks a lead time another has, pair to
   the nearest only when within 3 days, and label the pair (`91~90`). Beyond 3 days, **drop the
   horizon** and say so — a 150-day forecast compared against a 120-day forecast is two different
   problems, and the difference flatters whichever system got the shorter lead.

**Failure this prevents, from this project.** `run_20260902T120000Z` recorded
`beats_automl_baseline2: true` against a baseline MAE of 42.87 — a number taken from a benchmark
scored on a *different* test window (2021-07-02 onward vs 2020-12-13 onward), and which was in fact
the previous agentic run's own score, not an AutoML score at all. The conclusion happened to
survive re-measurement; the evidence for it did not exist. Every headline claim in this report
names the exact file and window it came from.

### 2. Compute the metric set

Per system, pooled over the aligned set: `MAE`, `RMSE`, `SMAPE`, `bias` (mean signed error, actual
minus predicted), `MASE` against the naive system, and `n`. Per system **per horizon**: `MAE`,
`SMAPE`, `MASE`, `n`.

`SMAPE` carries the same zero-handling convention the run used, stated once in the report. Applying
a different convention here than Training used makes the report's numbers incomparable with the
run's own.

`MASE` is computed against the naive system's error **at the same horizon**, not against a global
denominator — a single denominator hides that naive itself degrades with lead time.

The naive system is mandatory. If the invocation supplies none, derive a seasonal naive (last
observed same-weekday value at or before the origin) and label it as derived. A report comparing
only learned systems cannot tell the reader whether any of them were worth building.

### 3. Decompose the error into level and shape

For each system, alongside `MAE` and `bias`, compute **MAE after removing the constant bias**
(`mean|e - mean(e)|`). Report the three together. This one extra column is what converts "the model
is worse" into a mechanism:

| system | MAE | bias | MAE debiased |
|---|---|---|---|
| agentic | 41.44 | +32.1 | **28.73** |
| naive | 38.05 | +9.1 | 37.39 |

Read as: the learned model's day-to-day shape is far better; it draws that shape at the wrong level.
A large `bias` with a much smaller debiased MAE is a **level** problem (distribution shift,
extrapolation) and is not fixed by more feature engineering. A bias near zero with a large debiased
MAE is a **shape** problem, and is.

When several independently-fitted systems share the same bias sign and magnitude, say so
explicitly — that pattern is evidence of distribution shift rather than of any one model's mistake,
and it is the single most useful sentence such a report can contain.

### 4. Break the comparison down by horizon and by period

- **By horizon** — the chart and table in `references/report_layout.md` §4. Count and report how
  many horizons each system wins. A pooled win built on one dominant horizon band is a different
  claim from a uniform win.
- **By period** — the same metrics per calendar year (or per regime, if the run identified one).
  A bias that grows year over year is direct evidence of drift, and belongs in the report as a table.

### 5. State the verdict at the top, with its caveats attached

The report opens with the answer, not with methodology. Three plain statements: does the champion
beat the AutoML baseline, does it beat naive, and what explains the gap. Each carries its number
and the horizon count behind it. A verdict that holds only pooled, or only on some horizons, says
so in the same sentence — never in a footnote.

### 6. Explainability — grouped, per `../shared-references/feature_attribution.md`

That file is binding; this step only says what the report needs from it.

- Compute grouped permutation importance on the **scoring split named in the invocation, which must
  not be the test set**. Test-set attribution answers "what did it use on the data we just judged
  it on," which contaminates the held-out set's one job. Validation is the default.
- Report three things: (a) group importance with its standard deviation, (b) each group's masking
  gap, (c) the disagreement between split-count and permutation ranking, as a table of the features
  that move most. All three have a two-to-three sentence explanation, no more.
- If a group's importance and masking gap are both indistinguishable from zero, state that those
  features are unused and can be dropped. That is a genuine result and belongs in the report.
- **Never present split-count importance as importance.** Include it only in the rank-disagreement
  table, labelled as a build diagnostic.

### 7. Write the explanations

Every table and every chart carries a short prose block: **two to four sentences**, stating what the
number means, why it matters, and what follows from it. Not a caption, not an essay. If a finding
needs more than four sentences, it is more than one finding — split it.

Write for a reader who knows ML but not this project. Expand a metric the first time it appears
(`MASE — model error divided by the naive method's error at the same horizon; below 1.00 beats
naive`). Never restate a number the table already shows without adding an interpretation to it.

### 8. Write the outputs

- The self-contained HTML at `output_path`
- A paired CSV for every chart, under `artifacts/final_report/`, per
  `conventions/code_and_plot_conventions.md` — the report is a rendering, and every number in it
  must be recoverable without it
- `code/final_report.py` — cell-marked, re-runnable, so the report can be regenerated
- A `final_report.json` beside the HTML with the full metric set, the alignment record from step 1,
  and every path consumed. This is what a later run or a thesis appendix cites

## Guardrails

- **Never compare systems that were not scored on the same test rows.** Alignment (step 1) runs
  before any metric. Restricting to an overlap is acceptable and must be stated; comparing across
  different windows is not.
- **Never pair horizons more than 3 days apart**, and never silently. A dropped horizon is listed.
- **Never report a headline number without the file it came from.** Each verdict in section 1 names
  its source path and window. The 42.87 failure in step 1 is why.
- **Never compute explainability on the test set.** It is spent on the comparison; attribution goes
  on validation.
- **Never report per-feature importance as the top-level answer**, and never set a near-zero verdict
  from a column's solo score (`../shared-references/feature_attribution.md`).
- **Never omit the naive baseline**, even when it wins — especially when it wins. A report that
  compares only learned systems cannot answer whether the work was worth doing.
- **Never soften a losing result.** If the champion loses to naive, the verdict line says so, and
  the decomposition explains why. The error decomposition exists to make an honest loss
  informative, not to argue the loss away.
- **Never invent a number.** Every figure traces to a prediction file or to a computation in
  `code/final_report.py`. If a system could not be scored, it appears in the report as
  **not scored**, with the reason.
- **The HTML is self-contained** — no CDN, no external images, no network fetch at view time.
  Charts are inline SVG; fonts degrade to a system stack. It must render from a USB stick in five
  years.
