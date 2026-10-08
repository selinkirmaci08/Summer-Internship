# AutoML baseline

The reference point the agentic ML pipeline gets measured against. You give it a
dataset path and a goal in plain language; it returns the accuracy an
off-the-shelf AutoML tool reaches at **t+1, t+7, t+14, t+30 and t+90**, plus the
prediction-level data needed to diff the agentic pipeline against it row by row.

```bash
python -m automl_baseline --data "../data" --goal "forecast daily bread quantity sold" --budget 120
```

```
[target] inferred 'bread_quantity' from the goal
[data]   4468 rows x 30 cols | 2010-11-16 -> 2023-02-08 | freq=D | seasonal period=7
[split]  cutoff 2022-08-12 | train 4288 periods | held-out 180 periods
[model]  ONE model across horizons 1-90 (44 of them) | 155209 pooled train rows, 155 features
[model]  picked lgbm | validation mae 42.273
[h=1]    MASE 1.443 | MAE 82.07 | WAPE 44.7% | vs seasonal-naive +16.2% (open days +22.5%)
...
```

Then hand the trained model a time:

```bash
python -m automl_baseline.predict --run outputs/<run> --at 2023-03-15
```

## Install

```bash
pip install -r requirements.txt
```

FLAML is the AutoML engine: CPU-only, installs cleanly on Windows, and its
cost-frugal search is a defensible "what a good off-the-shelf tool finds in N
seconds" reference. Nothing outside `automl_baseline/search.py` depends on it, so
swapping in AutoGluon or H2O later means rewriting one function.

## What it does

1. **Ingests a path, not a file.** Every tabular file under `--data` is read
   (`.csv .tsv .parquet .xlsx .json .jsonl`), clustered by schema similarity,
   stacked within a cluster, then joined across clusters on the time key with
   overlapping columns coalesced most-complete-first. Pointing it at the project's
   `data/` folder collapses the four `aernoudt_bread_daily*.csv` files into one
   3 655-row × 31-column table and drops `adult.csv`, which has no time key.
2. **Infers the target from the goal.** `"forecast daily bread quantity sold"` →
   `bread_quantity`, by token overlap with column names. `--target` overrides it,
   and the ranked candidates are recorded in the output so the choice is auditable.
3. **Reframes the series as supervised learning, once per horizon.** Lags,
   rolling statistics, calendar and Fourier terms, plus lagged exogenous columns.
4. **Runs an AutoML search per horizon** against a forward-chaining time-series CV.
5. **Scores on a held-out window** against seasonal-naive, naive-last and
   rolling-mean references, and writes JSON, CSV, Markdown, HTML and plots.

## One model, not five

The default is **one** model. Every `(origin, horizon)` pair is stacked into a
single training matrix with `horizon` as an ordinary input column, so one fitted
model answers "what will the value be at time X" for any horizon in its training
grid — that is the `predict` command above.

On the bread data the single pooled model beat five per-horizon models outright,
in less wall time:

| | pooled (1 model) | direct (5 models) |
|---|---|---|
| mean MASE | **1.484** | 1.602 |
| mean skill vs seasonal-naive | **+11.0%** | +3.8% |
| horizons beating seasonal-naive | **4 / 5** | 3 / 5 |
| mean skill, open days only | **+28.0%** | +18.3% |
| wall time | **296 s** | 459 s |

Pooling is what does it: 155 000 training rows instead of ~3 500 per horizon. Each
horizon still gets scored separately, so the comparison table is unchanged.

`--strategy direct` fits a separate model per horizon instead — worth running when
you want each horizon's hyperparameters tuned in isolation, or as a second
reference point.

## Evaluation protocol

This is the part that has to match whatever scores the agentic pipeline.

- **No recursion.** Both strategies predict `t+h` in one jump from data at origin
  `t`. Nothing is fed back into itself, so t+90 does not inherit 89 steps of
  accumulated error — you do not need to predict t+89 to get t+90.
- **Split on the target timestamp, not the origin.** Everything with a target at
  or before the cutoff trains; everything after is held out. A horizon-90 model
  therefore cannot have been fitted on a value it is later asked to predict.
- **Rolling-origin evaluation.** Every valid origin inside the held-out window is
  forecast, so t+90 is scored on ~90 forecasts rather than one lucky shot. The
  window defaults to 2× the longest horizon (180 periods), capped at 35% of the
  series.
- **Fitted once, not refitted as the window advances.** Each origin's features use
  actuals up to that origin — you know the past when you forecast — but the model
  itself is frozen at the cutoff.
- **Validation inside the search is time-ordered** (`split_type="time"`), so
  hyperparameter selection never sees data from after the fold it is scored on.

Two test files enforce the parts that would silently invalidate a comparison —
feature leakage, target misalignment, a wrong naive reference, a comparison scored
on rows only one system saw:

```bash
python tests/test_protocol.py
python tests/test_compare.py
```

The leakage test is constructive: it blanks the series after an origin and asserts
that none of the origin's features change.

## Metrics

Per horizon: **MASE, MAE, RMSE, SMAPE, WAPE, R², bias**, plus the same set for
three naive references.

- **MASE** (Hyndman) is the scale-free number: MAE divided by the in-sample
  one-step seasonal-naive MAE of the *training* window. Comparable across horizons
  and models, and it does not blow up near zero the way MAPE does. Because its
  denominator comes from the training period, MASE < 1 is not by itself proof of
  beating naive on the test window.
- **skill** = `1 - MAE_model / MAE_seasonal_naive` on the *identical held-out rows*
  is the honest same-window verdict, and is what decides whether a horizon was won.

Reporting both is deliberate: a model can score MASE 0.38 and still only tie
seasonal-naive on the recent window if the series changed scale over its history.

**Zero and sparse days get their own view.** Closed days sit at exactly 0 and send
SMAPE/MAPE to ~200% for any model that predicts a normal day, so every horizon is
also scored on open days only (`actual > 0`), and the run flags a held-out window
that is sparse or zero-heavy. On the bread data this matters a lot: the Aug 2022 –
Feb 2023 window has an observation on only 52% of days and 24% of those are zeros.
Read MAE and WAPE on the full window, SMAPE on the open-days table.

## Output

```
outputs/<run>/
  metrics.json      every number, config, protocol and library version
  predictions.csv   horizon, origin_time, target_time, y_true, pred_automl, pred_<baseline>...
  report.md         the same table in Markdown
  report.html       self-contained styled report with embedded plots
  plots/            series overview, MASE by horizon, error spread, per-horizon fits
  models/           the fitted model per horizon (joblib) + FLAML search logs
```

## Using the model

```bash
# a specific date
python -m automl_baseline.predict --run outputs/<run> --at 2023-03-15

# a horizon instead
python -m automl_baseline.predict --run outputs/<run> --horizon 30

# the whole path t+1 .. t+90, one row per step
python -m automl_baseline.predict --run outputs/<run> --through 90 --csv forecast.csv
```

The origin defaults to the last timestamp with an observed value — "where we are
now" — and `--origin` moves it. Horizons that fall outside the trained range are
flagged as extrapolation; horizons between grid points are interpolated over the
numeric `horizon` feature, and `--max-train-rows` at training time controls how
dense that grid is.

## The Aernoudt raw drop

The raw POS export cannot be read by the generic ingest: ~12 000 XML files, one per
store-day, with the element names changing mid-history, plus a product master that
must be joined before a product group means anything. `sources/aernoudt.py` does
that translation and `run_aernoudt.py` chains it to the baseline:

```bash
python -m automl_baseline.run_aernoudt     --data "C:/Users/selin/OneDrive/Desktop/alldata/aernoudt"     --goal "predicting the bread demand" --budget 240
```

The ETL makes **no judgements about the data**. Every product group in the master
gets its own quantity/revenue/line columns, every weather column is carried
through, nothing is dropped for being small, and no ratios or flags are invented --
selecting what matters is the model's job. Dutch group names are translated for
column readability only.

The four things that cannot be derived from the data are parameters, and their
confirmed values live in `DECISIONS` at the top of `sources/aernoudt.py`:

| decision | confirmed value |
| --- | --- |
| what counts as bread | product group 1, `Brood`, alone |
| how demand is measured | quantity (units sold) |
| series scope | chain total, every POS code summed per day |
| store-days exported twice | keep the newest export |

The competitor files are **not** joined: nothing in the data links their shop codes
(`GENTDEPINT`) to POS codes, and inventing that mapping would be a judgement about
the data. `--competitor-map <csv with winkel,store>` includes them.

Mechanical rules the ETL applies, forced by the file formats rather than chosen:
two XML schemas (`<order>/<orderdetail>` and `<sale>/<saledetail>`), store identity
from the POS code in the filename rather than the folder (folder `01` holds both
POS 001 and POS 301/B2B), decimal commas, ISO-8859-1 XML and latin-1 CSVs, and
weather de-duplication by `drop_duplicates()` because 23 of every 24 rows are
byte-identical.

Validated against the previously derived `aernoudt_bread_daily_all_years.csv`: for
POS 001 on 2010-11-16 it reproduces num_orders=473, num_items_sold=1175,
total_revenue=1933.80 and group-1 quantity=395 exactly.

## Data shapes it handles

Tested on daily, weekly (`W-SUN`, parquet + xlsx across nested directories) and
monthly (`MS`) data, with the frequency, seasonal period and target inferred each
time.

**Several series in one file** (store_id, sku, region) is detected rather than
silently collapsed — modelling a panel as one series would keep one entity and
throw the rest away. The run stops and asks:

```bash
python -m automl_baseline --data ./sales --goal "forecast daily sales" --series-id store_id --series C
python -m automl_baseline --data ./sales --goal "forecast daily sales" --series-id store_id --aggregate sum
```

Detection keys on *recurrence* — a real key like `store_id` takes the same 3
values at every timestamp, while a measurement takes fresh values — so repeated
dates coming from four overlapping source files are not mistaken for a panel.
Forecasting every series with one shared model is not implemented yet; pick one
or aggregate.

**Short series** are handled by skipping the horizons they cannot support rather
than by producing a number: 60 daily points evaluates t+1 and t+7 and skips
t+14/30/90, saying why for each.

## Comparing the agentic pipeline against it

Have the agentic pipeline forecast the same cutoff window and the same horizons,
then:

```bash
python -m automl_baseline.compare --baseline-run outputs/<run> --candidate agentic_preds.csv --name "agentic v3"
```

The candidate CSV needs `horizon`, `target_time` and a prediction column
(`y_pred` / `prediction` / `pred` / `forecast` / `yhat`); include `y_true` and it
is cross-checked against the baseline's ground truth. Rows are joined on
`(horizon, target_time)` and every metric is recomputed on the intersection, so
neither system is scored on rows the other never saw. Output is a per-horizon
win/loss table plus `comparison.json`.

## Scoring a model trained elsewhere

To put an agentic-pipeline model (or any pickled estimator) on this protocol:

```bash
python -m automl_baseline.score_external     --model runs/<run>/artifacts/training/iteration_014_training_model.pkl     --features runs/<run>/data/iteration_013_feature_engineering.csv     --baseline-run outputs/<baseline-run>     --exclude-split train --name "agentic it_014"
```

Contamination is checked rather than assumed. Rows the external model trained on
are excluded via `--exclude-split`, rows at or before the baseline's cutoff are
excluded because the baseline trained on those, and targets are reconciled between
the two tables. If nothing survives, the tool says so instead of returning a
number.

One condition cannot be fixed by dropping rows: if the external model's training
set contains dates **after** the scored window, it was fitted with access to the
future. That is reported as `future_training_rows`, and when it is non-zero the
output is labelled an optimistic bound on the external model, not a like-for-like
score. An interleaved or blocked CV split will always trip this — such splits are
valid for estimating generalisation, but not for a forecasting comparison.

## Options

| flag | default | what it does |
| --- | --- | --- |
| `--data` | required | file or directory, searched recursively |
| `--goal` | required | plain-language goal; the target is inferred from it |
| `--target` | inferred | target column, overriding inference |
| `--time-col` | auto | time index column |
| `--horizons` | `1,7,14,30,90` | steps ahead to evaluate |
| `--strategy` | `pooled` | `pooled` = one model with horizon as a feature; `direct` = one model per horizon |
| `--budget` | `60` | search seconds (total for `pooled`, per horizon for `direct`) |
| `--sparse-grid` | off | pooled: train only on the scored horizons, not every step 1..max |
| `--max-train-rows` | `200000` | pooled: row cap before the horizon grid is thinned |
| `--series-id` | none | column splitting the data into several series |
| `--series` | none | with `--series-id`: model this one series |
| `--aggregate` | none | with `--series-id`: `sum`/`mean` every series into one total |
| `--test-days` | `2 × max horizon` | held-out window length in periods |
| `--train-window` | all history | cap training to the last N periods (lag features still use the full history) |
| `--freq` | inferred | pandas frequency, e.g. `D`, `W`, `MS` |
| `--known-future` | none | columns whose future values are known at forecast time (holidays, planned promos, weather forecasts) — used at `t+h` instead of lagged |
| `--metric` | `mae` | what the search optimises |
| `--estimators` | FLAML's list | e.g. `lgbm,xgboost,rf` |
| `--seed` | `42` | |
| `--mlflow` | off | log params, metrics and artifacts to MLflow |
| `--tabular` | off | force plain tabular AutoML (no horizons) |

## Choices worth knowing about

- **Exogenous columns are lagged, never read at `t+h`.** At forecast time you do
  not know next month's revenue, so the baseline does not get to use it. Anything
  genuinely known in advance goes in `--known-future`; calendar features are the
  built-in case of that.
- **Missing periods become NaN rows on a regular grid**, so "lag 7" means seven
  days back, not seven recorded observations back. The bread series has 813 such
  gaps across 2010–2023.
- **All history is used by default**, including the pre-2020 regime with a
  different schema. That is the honest off-the-shelf behaviour; `--train-window`
  exists to test whether the old regime hurts.
- **No refitting inside the held-out window.** A production system would refit as
  data arrives; both systems being compared are frozen at the cutoff, which keeps
  the comparison about modelling rather than retraining cadence.
- **A single cutoff, not nested cross-validation.** Cheaper, and adequate given
  the rolling-origin window; if the agentic pipeline and the baseline end up close,
  repeat with several `--test-days` / cutoffs before reading much into the gap.
- **Non-temporal datasets fall back to plain tabular AutoML** (`--tabular`, or
  automatically when no time column exists). Horizons do not apply there and the
  output says so.

## Layout

```
automl_baseline/
  ingest.py     discover, cluster, stack and join multi-source files; panel detection
  config.py     RunConfig + target inference from the goal text
  features.py   supervised reframing: lags, rolling stats, calendar, exogenous, pooled matrix
  baselines.py  seasonal-naive, naive-last, rolling-mean references
  search.py     the FLAML boundary -- swap engines here
  metrics.py    MAE, RMSE, MAPE, SMAPE, WAPE, MASE, R², bias
  runner.py     the forecasting protocol end to end
  tabular.py    non-temporal fallback
  report.py     plots, Markdown and HTML
  predict.py    give the trained model a time; it predicts
  run_aernoudt.py       raw Aernoudt path + goal -> model + horizon scores
  sources/aernoudt.py   the XML/product-master ETL for that raw drop
  compare.py    head-to-head against a candidate model
  score_external.py     put a model trained elsewhere on this protocol
  cli.py        argument parsing
```
