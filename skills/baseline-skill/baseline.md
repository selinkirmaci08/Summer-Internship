---
name: baseline-skill
description: Picks the floor every later model is measured against, and picks the naive predictors that later become features. Part A scores a fixed menu of six trivial forecasting rules — no fitting, no tuning, no search — on the run's own validation rows and reports the STRONGEST as baseline_floor. Part B runs one bounded, train-rows-only search for the best naive rules PER HORIZON and emits them as protected feature specifications for Feature Engineering to materialise. Loaded by Training at mode == "baseline", which must complete before the run's first feature_engineering call.
---

# Baseline Skill

## Purpose

The floor decides what "good" means for the rest of the run. A floor that is too easy makes every
later number look like a win, and nothing downstream can detect it — the metric is real arithmetic
against a bad reference.

This skill exists to stop that, cheaply. It is a **closed list of six rules and one pass**. Do not
expand the list, do not tune anything, do not iterate. If the six are all beaten by a wide margin,
the answer is "the floor is low," not "try more baselines."

**The failure this prevents, from this project.** `run_20260904T140437Z` set its floor with a rule
that looked up a fixed calendar date and scored whatever it found there, including public-holiday
closures where demand is 0. The floor came out at MAE 85.31 pooled. A correct rule on the identical
window scores 44.14. The run reported its champion as +35.5% against the floor; against a working
floor the same champion is **−24.5%**. The model was losing the whole time and every artifact in the
run said it was winning.

## When this runs

Training's first call of every run (`mode == "baseline"`). It fits nothing, saves no `.pkl`, and
returns `baseline_floor` and `naive_predictor_features`. Re-run it only if the split or the
trading-day rule changes.

**It must complete before the run's first `feature_engineering` call.** Part B's output is an
input to that call — Feature Engineering builds the naive predictor columns from the specs this
skill emits, and a feature set built before those specs exist is missing them for the rest of the
run unless someone thinks to go back. The Planner enforces the ordering; this skill states the
dependency so the reason survives.

## The two things this skill produces

| | Part A — the floor | Part B — naive predictor features |
|---|---|---|
| question | what does "good" mean for this run? | which naive rules carry enough signal to be worth handing the model? |
| candidates | the closed six, never expanded | the six plus a small bounded grid of parameterised variants |
| scored on | validation rows | **train rows only** |
| selection | the single strongest rule, pooled and per horizon | the best few rules **per horizon**, deduplicated |
| output | `baseline_floor` | `naive_predictor_features`, marked protected |
| may be tuned | **never** | yes — and that is exactly why it cannot set the floor |

Keeping these apart is the whole design. A rule whose window was chosen by search is a
one-parameter model, and a floor set by one flatters every model measured against it. The same
arithmetic produces both, so Part B costs almost nothing on top of Part A — but its results never
touch `baseline_floor`.

## Budget

Part A: one pass over the validation rows, six rules, vectorised — under 30 seconds. Part B: one
further pass over the train rows, at most 24 candidates per horizon, vectorised — under 60
seconds. One script covering both, target under two minutes end to end. No model fitting, no
cross-validation, no hyperparameter search on anything that is not a naive rule's own single
parameter, no plots beyond two.

**"Search" in Part B means one sweep of a written-down grid, not a study.** If it is taking
longer than the rest of this node, it has stopped being a baseline step. Cut the grid, record
what was cut, and move on — the marginal rule that needed the extra minute was never going to be
the one that carried the model.

## Part A — the floor

### The six candidates

Fixed list. `P` is the seasonal period from EDA (7 for daily retail); `S` is one seasonal year
(52 for weekly seasonality on daily data).

| # | rule | prediction at origin `t` for target `t+h` |
|---|---|---|
| 1 | last value | the most recent observed value at or before `t` |
| 2 | seasonal naive | the last observed value at or before `t` **on the target's own season slot** |
| 3 | seasonal median | median of the last `S` observations on the target's season slot |
| 4 | seasonal mean | mean of the same `S` observations |
| 5 | drift | last value + `h ×` (average per-period change over the training rows) |
| 6 | train mean | the mean of the target over training rows only |

Rules 3 and 4 are usually the strongest and are the reason this list is not just "seasonal naive".
A single seasonal observation carries the season's signal plus one period's noise; averaging a
season-year of them keeps the signal and drops most of the noise.

### Procedure

1. **Build the (target, origin) grid.** Origin is `target_date − h` in the series' own time unit,
   for every horizon in `horizon_grid`. Use the exact rows the model will later be scored on.

2. **Score all six in one pass.** Per horizon and pooled: MAE plus the run's primary metric.

3. **Report the strongest as `baseline_floor`.** Not the most familiar, not the contract's named
   one, not the most convenient — the lowest error. Where a scoring contract names a specific
   reference, report **both**: the contract's rule under its own name, and the strongest rule as
   the floor, and say which is which.

4. **Write one plot:** MAE by horizon, all six rules, per `code_and_plot_conventions.md`.

5. **Then run Part B.** The (target, origin) grid built in step 1 is the same construction Part B
   needs, so it is built once and reused. Part B scores on train rows, not the validation rows
   scored here.

### The five rules that make the number real

Each one is a failure this project actually shipped.

- **Only data at or before the origin.** For `h=90` that means nothing from the last 90 periods.
  A rule that indexes `target − P` regardless of `h` silently reads the future for `h > P`. Check
  one long-horizon row by hand before trusting the table.

- **Walk back to the last OBSERVED period; never index a fixed date.** If the shop was shut, that
  date holds a 0 or nothing at all. Indexing it scores a closure as a sales figure — a ~300-unit
  error that has nothing to do with forecasting. Step back to the most recent period the series
  actually records.

- **Declare what you do with closures and zeros, and apply it to every system.** Whether a zero-demand
  day may serve as a reference value is a real choice with a real cost (~2 MAE here). Record it in
  `baseline_floor.zero_handling`. Do not silently pick the version that flatters the floor.

- **Score the floor on the same rows as the model.** Different `n` per system means the comparison
  is not a comparison. Record `n` per horizon and assert it matches what Training scores.

- **Never tune a window on validation.** `S` comes from the calendar — one seasonal year — not from
  a grid search. If a window is chosen by search, it must be searched on training rows only and
  labelled as tuned, at which point it is a one-parameter model and not a baseline.

**All five apply to Part B as well**, with one substitution: Part B's rows are the train rows,
so "score the floor on the same rows as the model" becomes "select on train rows and record the
train row count per horizon."

---

## Part B — naive predictor features

A naive rule is not only a yardstick. At horizon `h` the rule's own prediction is a legitimate,
strictly causal feature: it compresses the target's recent history into the one number that
history most obviously implies, and it hands the model a starting point to correct rather than a
level to rediscover from lags. A gradient-boosted tree given `naive_h7_seasonal_median` is
learning the *residual* of a decent forecast; the same tree given only raw lags has to
reconstruct that forecast internally, out of splits, before it can improve on it.

**Different horizons want different rules.** At `h = 1` the last observed value is hard to beat.
At `h = 30` it is nearly worthless and a seasonal average over a full season-year is far stronger.
Selecting one rule pooled across horizons therefore hands most horizons a predictor chosen for a
different problem, which is why this selection is per horizon and not run-wide.

### B1. The candidate grid

The six fixed rules, plus a bounded set of parameterised variants. This grid is the whole search
space — write it down in the script, do not grow it while looking at results.

| family | parameters swept | count |
|---|---|---|
| the six fixed rules (Part A) | none | 6 |
| seasonal mean over the target's season slot | window ∈ {4, 8, 13, 26, 52} seasons | 5 |
| seasonal median over the target's season slot | window ∈ {4, 8, 13, 26, 52} seasons | 5 |
| trailing mean over the last `w` observed periods | `w` ∈ {7, 14, 28, 56} | 4 |
| trailing median over the last `w` observed periods | `w` ∈ {7, 28} | 2 |
| damped drift: last value + `φ·h·`(mean per-period change) | `φ` ∈ {0.25, 0.5} | 2 |

At most 24 candidates per horizon. Where the series has a second seasonality EDA actually
measured (e.g. a yearly cycle on daily data alongside the weekly one), one seasonal mean and one
seasonal median at that period may be added, and the two weakest windows above dropped to stay
inside the budget. Nothing else is added. If a candidate cannot be computed for a horizon — a
window longer than the training rows allow — record it as skipped with the reason rather than
silently shrinking the grid.

### B2. Select per horizon, on train rows only

For every horizon `h` in `horizon_grid`, score every candidate on the **training rows** using the
run's primary metric plus MAE, on the same (target, origin) construction Part A uses. Then:

1. Rank by train error.
2. **Always keep the horizon's Part A winner**, whether or not it ranks first here. It is the one
   predictor in the set whose parameter was not chosen by looking at data, which makes it the one
   that cannot have been selected by noise.
3. Add candidates down the ranking until three are kept, skipping any whose predictions correlate
   above 0.98 (Pearson, on train rows) with one already kept. Two rules that agree on 98% of the
   variance are one feature and one wasted column.
4. Stop at three per horizon. Fewer is fine — if only two survive deduplication, keep two.

**Selection happens on train rows and nowhere else.** These become model inputs, so a window
chosen by looking at validation error puts validation information inside the feature set, and
every validation number the run produces afterwards is measuring a model that has already seen
its own test. This is the same rule Part A's fifth guardrail states, applied to the case where
the tuned rule is actually used instead of discarded.

### B3. Emit the specs — do not build the columns here

This skill produces **specifications**, not columns. Feature Engineering materialises them,
because that is the node that owns creating columns and the node whose availability and causality
audits every feature must pass. Building them here would put the same construction in two places,
which is how the two eventually disagree.

Each entry in `naive_predictor_features` carries:

| field | what it holds |
|---|---|
| `horizon` | the horizon this predictor is for — it is valid for that horizon's model and no other |
| `column_name` | `naive_h{h}_{rule}`, e.g. `naive_h7_seasonal_median_w13` |
| `rule` | the rule's name and its parameters, exactly enough to reimplement it |
| `formula` | the prediction for target date `d` at origin `d − h`, written as an expression over the series |
| `train_mae`, `train_primary` | its score on train rows, and `rank` within the horizon |
| `corr_with_kept` | its highest correlation with the other kept predictors at that horizon |
| `selected_by` | `"part_a_winner"` (untuned) or `"train_search"` (tuned) |
| `protected` | `true` — see below |
| `zero_handling` | inherited from `baseline_floor.zero_handling`, restated per predictor |

### B4. Why these are protected, and what protection means

These columns are the run's one guaranteed floor-level signal, and they are also the columns most
likely to be dropped by a rule of thumb that is right about everything else. A naive predictor is
by construction highly correlated with the target *and* with every lag feature near it, so a
multicollinearity screen flags it, a redundancy check flags it, and a solo-importance reading
under correlated features scores it near zero for exactly the reason
`../shared-references/feature_attribution.md` §4 exists. Each of those is a reasonable heuristic
producing a wrong answer on this specific column.

So: **no node may drop, exclude, or fail to rebuild a protected feature on its own judgment.** A
node that concludes one should go raises a `blocking` `clarification_needed` naming the column,
the evidence, and its recommendation, and leaves the column in place. The user decides. The
Planner's Clarification Handling section already halts the run on a blocking entry; this is that
mechanism, not a new one.

Two things that are **not** drops and need no clarification:

- **Horizon scoping.** `naive_h7_*` belongs to the `h = 7` model's feature subset and to no
  other. Feeding it to the `h = 30` model would read the series 23 periods after that model's
  origin — a leak, not a feature. Restricting each predictor to its own horizon is the
  construction working as specified.
- **A predictor that scores badly.** A kept predictor whose measured contribution is zero is a
  finding, reported and carried. It is not grounds for removal without asking, because the
  measurement that says "zero" is the same masked-importance measurement this section exists to
  distrust.

### B5. What Part B does not do

- It does not change `baseline_floor`. A tuned rule beating the floor's winner on train rows
  means the tuned rule is a small model, not that the floor was wrong.
- It does not fit anything, and no entry in the grid has more than one free parameter.
- It does not iterate. One grid, one pass, one selection. If the selection looks disappointing,
  that is the answer.
- It does not decide whether the model should predict the residual against the naive prediction.
  That is Feature Engineering's `target_representation` decision (its step 2b), which now has an
  anchor available that it did not have before — say so in the handoff and let that node choose.

**Write one plot:** train MAE by horizon for the kept predictors against the Part A winner, per
`code_and_plot_conventions.md`, with its paired values CSV.

## Output

Into `baseline_floor`:

- `baseline_type` — the winning rule's name, and `contract_reference` where a contract names one
- `seasonal_period`, `window` — `P` and `S`, with where each came from
- `zero_handling` — the declared closure policy
- `per_horizon` — MAE and `n` for every rule, not only the winner; the losers are what show the
  floor was chosen rather than assumed
- `pooled` — the same, pooled
- `note` — one sentence: which rule won, by how much over the contract reference, and the
  consequence if a later model fails to beat it

Into `naive_predictor_features` (Part B), a list with one entry per kept predictor per horizon,
fields per the B3 table, plus alongside it:

- `search_grid` — every candidate scored, per horizon, with its train score and rank, and every
  candidate skipped with the reason. The rejected ones are what show the selection was made
  rather than assumed
- `dedup_dropped` — candidates removed for correlating above 0.98 with a kept one, naming which
- `selection_basis` — `"train_rows_only"`, with the train row count per horizon
- `note` — one sentence per horizon: which predictors were kept, and how much better the best is
  than the last-value rule at that horizon

Both blocks go in the handoff. `naive_predictor_features` is load-bearing for the next
`feature_engineering` call and must be carried in full — a count is not enough for a node that
has to reimplement the formula.

## Guardrails

- **Never report a floor without having scored at least rules 1, 2 and 3.** A single-rule floor is
  an assumption wearing a measurement's clothes.
- **Never expand the list to find a floor the model beats.** If the champion loses to a trivial
  rule, that is the result. Say so in `vs_baseline_floor` and let Diagnosis route on it.
- **Never carry a floor across a change to the split or the trading-day rule.** Re-score it.
- **Never let a Part B result change `baseline_floor`.** The two selections have different rules
  and different scoring rows on purpose. A tuned rule is a one-parameter model; the moment one
  sets the floor, every model in the run is measured against something that was itself fitted.
- **Never select a Part B predictor on validation rows.** These are model inputs. A window chosen
  on validation is validation data inside the feature set, and it silently invalidates every
  validation number the run reports afterwards — including the one the stop decision rests on.
- **Never emit a Part B predictor for a horizon other than its own.** The column is defined by its
  origin, and the origin is `target − h`. Reused at a longer horizon it reads the future.
- **Never return a run's naive predictors without `protected: true` and the formula.** A protected
  column that a later node cannot reconstruct will be silently regenerated wrong or dropped on the
  first `rebuild`, and the protection is then a label with nothing behind it.
