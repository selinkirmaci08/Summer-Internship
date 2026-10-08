# automl-baseline2

The reference point for a *specific* agentic run, scored on that run's own held-out
rows. Point it at a run directory and it answers one question:

> Given the same data, the same split and the same rows, how much of what the
> agentic pipeline achieved could an off-the-shelf AutoML tool have achieved on
> its own — and **which part** of the pipeline is responsible for the difference?

```bash
python -m automl_baseline2 --run ../runs/run_20260827T154407Z --budget 900
```

## Why not just use `../automl-baseline`?

Because v1 owns its protocol, and that is the right design for what v1 does. It
picks its own cutoff, its own held-out window, its own horizon grid, and it scores
on every calendar day including the ones the shop was shut. That produces an honest
number about *the dataset*.

It is not a number you can put next to `MAE 43.509` from
`run_20260827T154407Z`, because that run scored 21 horizons on 512 trading dates
after dropping 242 non-trading days, split 70/15/15 on the target date. Two systems
measured on different rows do not have a comparable error, and averaging over a
different set of days changes the answer more than the modelling does.

So this package does not replace v1 — **it imports it**:

| reused from v1, unchanged | what for |
| --- | --- |
| `automl_baseline.features.HorizonDataset` | the generic lag / rolling / calendar / Fourier recipe for tiers B2 and B3 |
| `automl_baseline.features.prepare_series` | regular-grid reindexing, so a lag of 7 means seven days |
| `automl_baseline.metrics` | every metric, so a number here means what a number there means |
| `automl_baseline.search._prepare_matrix` | the FLAML dtype contract |

What is added is the part v1 has no way to express: reading an agentic run's
protocol out of the run directory and pinning every system to it.

The sibling package is located automatically. `AUTOML_BASELINE_V1` overrides the
path if it moves.

## The four reference points

Each tier is handed a different amount of the agent's work, and everything else is
held fixed. The gaps between adjacent tiers are the interesting part — a single
"AutoML got X" number cannot tell you *where* the pipeline earned its keep.

| | given to it | isolates |
| --- | --- | --- |
| **seasonal naive** | last observed same-weekday value at the origin | the floor |
| **B3 assembled** | the raw assembled frame; generic features | AutoML alone, end to end |
| **B2 preprocessed** | the agent's cleaned frame; generic features | the agent's **preprocessing** (B3→B2) |
| **B1 agent features** | the agent's engineered matrix, unchanged | the agent's **feature engineering** (B2→B1) |
| **agentic** | the run's chosen model | the agent's **model choice and tuning** (B1→agentic) |

B1 is the sharpest of them. It gets the exact 56 columns the chosen model used, on
the exact same rows, so the *only* thing that differs from the agentic model is who
picked the estimator and its hyperparameters. If AutoML wins B1, the agent's
feature work was sound and its training work left something on the table. If the
agentic model wins B1, the tuning was doing real work that a generic search misses.

## What is held fixed

Read from the run rather than configured, so pointing this at a different run picks
up that run's protocol instead of silently applying this one's:

- **Rows.** The (target date, horizon) pairs in the run's test partition. Checked
  pair-by-pair for every tier before anything is scored — `assert_grid_matches`
  raises rather than scoring a near-match, because a comparison whose two sides sit
  on slightly different rows is worse than no comparison: it still looks like a
  result.
- **Split.** The run's own chronological partitions, by target date.
- **Tuning.** Every AutoML tier selects hyperparameters on the run's *validation*
  partition — never on test, and never on a split of its own choosing. A baseline
  that tunes on a friendlier window is measuring the window, not the model.
- **Trading days.** The days the run kept. Days it filtered as non-trading are
  excluded from every system; unexported days stay absent on the calendar so lags
  still span real time.
- **Information at the origin.** Exogenous columns are lagged unless the run
  declared them knowable in advance. `must_drop_columns` are dropped from B3 too —
  lagging same-day revenue and transaction counts would hand the AutoML tier
  information the agentic model was denied, and the gap would stop meaning what it
  appears to mean.
- **MASE denominators.** Both of the run's own: the train-window denominator, which
  is comparable with every MASE the run quotes, and the test-window one, which is
  the honest same-window verdict. Reporting only one of them flatters or damns
  every system at once.

## No recursion

Each row predicts `t+h` in one jump from data at origin `t = (t+h) − h`. Nothing is
fed back into itself, so `t+90` does not inherit 89 steps of accumulated error and
you do not need `t+89` to get `t+90`. This matches the run's horizon-conditioned
direct design, and the layout is verified on load rather than assumed.

## Output

```
outputs/<run-id>/
  metrics.json     every number, plus the protocol and what each tier was given
  predictions.csv  one row per (target_date, horizon); y_true and one column per system
  per_horizon.csv  the per-horizon table on its own
  report.md        the same tables in Markdown
  report.html      self-contained styled report with embedded plots
  plots/           MAE by system, MAE and skill by horizon, error by year,
                   the h=7 forecast across the window, residual spread
  models/          each fitted AutoML model (with --save-models) + FLAML search logs
```

`predictions.csv` is the file to keep. Every claim in the report is recomputable
from it, and it is the join key for anything else you want to ask later.

## The comparison page

`report.html` is written by every run and is the working record. For something to
hand to someone else, `tools/build_comparison_page.py` renders the same numbers as
a designed page with computed SVG charts:

```bash
python tools/build_comparison_page.py --run outputs/run_20260827T154407Z
```

It reads that run's own `metrics.json` and `per_horizon.csv` and writes two files
beside them, so the page cannot drift from the numbers the run produced:

| file | what it is |
| --- | --- |
| `comparison.html` | standalone - open it in a browser |
| `comparison.body.html` | the same page with no `<!doctype>`/`<head>`/`<body>`, which is the form the Artifact publisher expects |

Three tables (overall metrics, horizon bands, all 21 horizons) and five charts
(MAE by system, the B3 -> B1 -> agentic decomposition ladder, MAE by horizon,
skill against the naive by horizon, error by year). The two line charts carry
hover tooltips.

Chart coordinates are computed rather than hand-placed, and the y-domains are
derived from the data each time. That is not tidiness: the first version pinned
the MAE axis to the models' range and silently drew the seasonal naive - which
climbs to 56 at h=364 - outside the plot.

The horizon axis is **evenly spaced, not to scale**. The grid is 21 irregular
values from 1 to 364; log spacing crushes h1-7, which is exactly where the
naive-beats-everything result lives. The axis label says so.

## Tests

Seven checks, none of which test the modelling. They test the things that would
silently invalidate a comparison:

```bash
python tests/test_alignment.py
```

The first one is the load-bearing one: it rescores the run's own chosen model
through this harness and asserts the result reproduces the MAE and RMSE the run
reported. If that drifts, the harness is reading the run wrong and every other
number it produces is wrong in the same way. The same check runs inside the CLI on
every invocation and prints `match` or a warning.

The leakage check is constructive rather than name-based: it blanks the series
after an origin and asserts that none of that origin's past-only features moved.

## Options

| flag | default | what it does |
| --- | --- | --- |
| `--run` | required | agentic run directory |
| `--tiers` | all three | comma-separated subset of `B1_agent_features,B2_preprocessed,B3_assembled` |
| `--budget` | `600` | AutoML search seconds **per tier** |
| `--metric` | `mae` | what the search optimises |
| `--estimators` | FLAML's list | e.g. `lgbm,xgboost,rf` |
| `--seed` | `42` | |
| `--target` | from the run | override the target column |
| `--naive-col` | `lag_sw_1` | the run's seasonal-naive column |
| `--save-models` | off | persist each fitted AutoML model |
| `--output-dir` | `outputs` | |

## Known limits

- **One cutoff, not nested CV.** The comparison inherits the run's single split. If
  two systems land within noise of each other, that gap is not resolvable here —
  re-run the agentic pipeline on a second cutoff before reading much into it.
- **Weather is lagged for the AutoML tiers**, where the agentic features used
  observed weather as a forecast proxy at h ≤ 7. That is a small advantage to the
  agentic side, on a feature the run measured at Spearman −0.076 against the target.
- **B2 and B3 share one generic recipe.** They differ in their input frame, not in
  their feature engineering, so the B3→B2 gap is specifically about the agent's
  cleaning decisions and not about anything else.
- **Search budget is wall time**, so results move a little between machines. The
  budget actually used is recorded in `metrics.json`.
