---
name: eda-agent-skill
description: Performs exploratory data analysis on a tabular dataset. Judges training feasibility, profiles features, surfaces relationships and data quality issues, recommends an evaluation metric, and selectively generates plots — converting all visual findings into numerical summaries before returning results. Triggered as the first node the Planner routes to for any new dataset.
---

# EDA Skill

## Purpose

Given a raw dataset and a stated goal, determine whether the dataset is trainable, understand its
structure, and produce a compact, structured summary that lets the Planner and downstream agents
(Preprocess, Feature Engineering, Train, Evaluate, Diagnosis) reason about the data without ever
needing to see it directly.

## Reference material

This skill has a `references/` directory holding the lookup tables the procedure below points at.
Consult them at the step that names them rather than loading them all up front:

- `../shared-references/interrogating_results.md` — **check 1 (units) and check 2 (regime) belong
  to this node.** A lag or seasonality figure measured in rows on a filtered frame does not mean
  what its name says, and every feature built on it inherits the error

- `references/feasibility_criteria.md` — the seven hard-blocker checks and their surrounding rules (step 2)
- `references/metric_selection.md` — situation-to-metric tables for every task type (step 4)
- `references/missingness_framework.md` — MCAR/MAR/MNAR classification and treatments (step 7)
- `references/plot_selection.md` — the condition-to-plot table (step 10)
- `references/report_example.md` — a worked example of the report this node writes (step 14)
- `conventions/handoff_contract.md` — the handoff block's fields and the externalisation pass
- `conventions/naming_conventions.md` — where every file this node writes goes, and what it is named
- `conventions/code_and_plot_conventions.md` — cell-marked scripts; every plot names itself

## Inputs

- `dataset_path` — path to the raw dataset. **May resolve to several files** (a directory, or a
  set of paths). The full list arrives as `context.data_sources`; every entry in it is in scope
  for this call — see step 1a.
- `goal` — the stated modeling objective, in plain language
- `target_column` — the column to predict
- `iteration` — the global iteration number assigned by the Planner, used to build every output
  path per `conventions/naming_conventions.md`
- `context` — prior reasoning relevant to this call, per `conventions/naming_conventions.md`.
  Usually empty, since EDA is normally the first node to run, but not guaranteed to be — e.g. if
  EDA is re-invoked after a user answers a clarification or more data is pulled in

## Tools available

- Code execution sandbox with `pandas`, `numpy`, `scipy.stats`
- Plotting library (`matplotlib`/`plotly`) for artifact generation only — plot images are saved to
  disk and never passed into the agent's own context
- File write access to `artifacts/eda/`, `history/`, and `reports/`
- Read-only access to `state/current.json` and any file under `history/`, per
  `conventions/naming_conventions.md` — useful if something relevant wasn't included in the
  input the Planner curated (rare for EDA specifically, since it's usually the first node to run,
  but the capability is general to every node)

## Procedure

Work through these steps in order. Each step's findings feed the next — do not skip ahead without
computing the numbers a later step depends on.

**Every step below is a floor, not a ceiling — and so is the procedure as a whole.** If, for this
specific dataset, a technique, check, or method beyond what a step's table or instructions
specify would genuinely be useful, pursue it. If the dataset calls for an entirely new analysis
this 14-step procedure doesn't include at all, run it. This is not license to skip or replace
what's actually required — every step's prescribed checks still have to run regardless — it's
permission to go further when going further is warranted.

Anything beyond what's specified gets recorded in `agent_initiated_findings`, a top-level output
list, **never blended into a step's standard fields as if it were part of the prescribed
procedure.** This keeps the output honest about what was actually asked for versus what you
decided, on your own judgment, was worth doing. Each entry:

```json
{
  "related_step": "7",
  "title": "short description of what was explored",
  "method": "...",
  "findings": "...",
  "reasoning": "why this was worth doing beyond what the step specifies",
  "confidence": "low | medium | high"
}
```

Use `"related_step": "new"` for something that isn't an extension of any existing step — a whole
additional analysis the procedure never anticipated. Every entry needs the same discipline as
everything else in this skill — a stated `method`, `reasoning`, and `confidence` — being
agent-initiated doesn't relax any other rule in this document.

1a. **Inventory every resource before profiling any of them.** Read `context.data_sources` (the
   Planner resolves it at Step 0) and open **every resource** in it — not the biggest one, not the
   one whose name matches the target, and not only the ones an earlier run happened to use.

   **The unit here is the resource, not the file.** A directory of files sharing one schema — 3000
   per-day XML exports, a year of monthly CSVs — is **one** resource, profiled as a set; a
   standalone file is one resource. Inventorying 3000 files individually buries exactly the
   findings that matter about them.

   For each resource record in `source_inventory`: path (or directory and file pattern), format,
   row and column counts, its key columns, its **granularity** (one row per what?), its **time
   coverage** (first and last timestamp, and any gap), and one sentence on what it holds that no
   other resource does. For a multi-file resource, add what only shows up across the set: how many
   files, how many expected periods have **no** file, whether the schema changes partway through,
   and whether any file is empty or unparseable. A missing day and a present-but-empty day are
   different facts and downstream nodes treat them differently.

   Then reason explicitly about **using them together**, and record it as `join_plan`:

   | Question | What to record |
   |---|---|
   | What key joins these? | The exact column names on each side (`date` ↔ `Datum`, `branch_id` ↔ `winkel`), and whether that key is unique on each side |
   | Do the granularities match? | One row per day vs. per transaction vs. per store-day. Say which direction the aggregation has to go, and on which column |
   | Do the coverage windows overlap? | The intersection is what a joined dataset can actually cover. A source spanning only 2022 cannot enrich a 2020–2024 series without leaving most rows null — say so, with the row percentages |
   | What does each source add? | The specific signal, named. "Weather" is not an answer; "daily max temperature and precipitation, which plausibly drive walk-in traffic" is |
   | What breaks if they are joined? | Row multiplication from a non-unique key, timezone or date-format mismatch, rows silently dropped by an inner join. Estimate the match rate before recommending the join, and report it |

   Every resource in `context.data_sources` gets a verdict: `used`, or `not_used` **with a
   reason**. "Not obviously relevant" is not a reason — open it first, then say what it contains
   and why that does not help. A resource left unmentioned is a defect in this node's output.

   **This is the specific failure being corrected.** Runs in this project have repeatedly modelled
   one resource and left whole sibling directories — competitor data, delivery data, product data
   — unopened for the entire run, without ever recording a decision not to use them. Nobody chose
   that; it happened because nothing forced the question. An unopened resource is now a defect
   whether or not it would have helped, because "it would not have helped" is a claim that
   requires having looked.

   **This is where structural mysteries get solved, or don't.** Level shifts, regime breaks and
   unexplained segments are far more often explained by a file nobody opened — a migration date, an
   opening-hours change, a store that went 7/7 — than by a model nobody tuned hard enough. If a
   source plainly bears on the goal but cannot be joined, that is a `clarification_needed`, not a
   silent omission.

   When `context.data_sources` holds exactly one file, record that single entry and move on — the
   step costs nothing and the record still shows the question was asked.

1. **Load and profile the dataset.** Shape, dtypes, memory footprint, duplicate rows (and whether
   duplicates cluster around a specific segment), missingness at the dataset level.

   **Also check whether rows are independent or secretly grouped** — does any column look like a
   repeated-entity identifier, where each unique value appears across multiple rows (e.g.
   `patient_id` repeated once per visit)? This is different from the leakage check on a
   near-unique ID column — here, a value appearing many times is normal, expected structure, not
   a red flag. If found, record it as `grouping_structure` (the group column, average and max
   rows per group). This matters because rows sharing a group are not statistically independent —
   if the same entity's rows end up split across train and validation, that's a real leakage path
   even if the split was otherwise done correctly (e.g. a model could partly memorize a specific
   patient's pattern from training and then be scored on that same patient). Preprocess needs
   this to split by *group*, not by row, when creating `_split`.

2. **Check feasibility before anything else, using the criteria below.** Feasibility fails only
   on true hard blockers — apply these checks in order; if none trigger, the dataset is feasible
   even if imperfect (imbalance, missingness, and multicollinearity are handled downstream, not
   here).

   The seven checks themselves, the row-count pre-scan rule, the borderline-verdict rule,
   and the narrow escape hatch for a blocker none of them capture are in
   `references/feasibility_criteria.md`.

   **Feasibility failing does not mean stopping the analysis.** It means the Planner should not
   proceed to Preprocess/Training as normal — it does not mean skipping steps 3 and 5–9 or leaving
   their output fields empty. Continue computing and populate every field for which you have
   actual data (target stats, feature stats, correlations, missingness, quality issues) regardless
   of the feasibility verdict — a `null` field should only ever mean "genuinely could not be
   computed" (e.g. an empty dataset), never "feasibility was false so this was skipped." The
   Planner and any human reviewing the run need to see the real numbers behind an infeasibility
   call, not just a prose explanation.

3. **Analyze the target.** Task type — not limited to binary/multiclass classification and
   regression; also consider **time series / forecasting** (the target is a sequence over time,
   not independent rows), **ordinal regression** (categories with a meaningful order, e.g.
   severity levels 1–5, where treating them as unordered classes or unweighted numbers both lose
   information), and **multi-label classification** (multiple non-exclusive labels per row,
   distinct from multiclass where labels are mutually exclusive) — plus a definition sanity check
   (consistent encoding, no ambiguous labels), a breakdown by key segments (e.g. region,
   day-of-week, time period) to catch subgroup effects, and a check for temporal drift if a
   datetime field exists.

   **Survival analysis / time-to-event is a plausible task type this skill doesn't yet have full
   treatment for** — worth specifically checking whether the goal is really "when does X happen"
   rather than "does X happen" (e.g. "predict readmission risk" could be framed as time-until-
   readmission, not just yes/no), and whether some rows are censored (the event hadn't happened
   yet by the end of the observation window, so the true time-to-event is unknown, not zero or
   missing). If this framing genuinely fits, record it via `agent_initiated_findings`
   (`related_step: "3"`) rather than forcing it into one of the task types above, which don't
   represent censoring at all.

   **If the task is time series / forecasting, target analysis needs content none of the other
   task types require:**
   - **Trend** — is there a sustained long-run increase or decrease in the target over the full
     time range, not just noise (e.g. the 28% multi-year decline found in one of the bread-demand
     runs this skill was tested against)?
   - **Seasonality** — does the target show a repeating pattern tied to a calendar cycle (weekly,
     monthly, annual)?
   - **Stationarity** — does the target's statistical properties (mean, variance) stay stable
     over time, or drift? A formal check (e.g. an Augmented Dickey-Fuller test) is worth running
     if this isn't visually obvious; a non-stationary target usually needs differencing or a
     trend feature before many model families can use it well.
   - **Autocorrelation** — does the target correlate with its own past values (e.g. via ACF/PACF)?
     A strongly autocorrelated target means lagged versions of itself are likely to be genuinely
     useful features later, worth noting in `recommended_strategy` for Feature Engineering.

   - **Forecast horizons** — how many periods ahead is this forecast actually for? Read it from
     the goal. A goal that says "day-ahead" is single-horizon; one that says "plan next month's
     production" implies a set. Record `forecast_horizons` as an explicit list (e.g. `[1, 7, 14,
     30]`) with the unit, and the reasoning that produced it. **If the goal does not state it,
     raise a `clarification_needed`** — with a recommendation, per the Planner's field table —
     rather than assuming one; the horizon set determines the split, the usable feature set and
     the number of models the run has to build, so a wrong guess here is expensive to unwind.
     Report the cost per horizon too: at horizon `h` the last `h` periods of the series have no
     target, so the effective training rows shrink as `h` grows — state the row count per horizon,
     because a 30-day horizon on a short series can be infeasible while a 1-day horizon is fine.
   - **The trivial baseline's error** — before any model exists, score the naive predictor on the
     series itself: seasonal naive (`ŷ_t = y_{t-P}`, with `P` from the seasonality finding above)
     for a time series, the training mean for plain regression, the majority class for
     classification. Record it as `trivial_baseline` with the metric, the value, the period or
     rule used, and the window it was computed over. **This is provisional** — there is no
     train/validation split yet, so it is a difficulty reading, not the floor a model gets
     measured against; Training establishes that on the real split. What it buys here is
     perspective from the very first iteration: a run that knows the naive error is 38 units knows
     what a 36-unit model is actually worth, and feasibility (step 2) can weigh whether the gap
     between naive error and the goal's implied accuracy is plausibly closable at all.

   Record these under `target.time_series_analysis` when applicable; leave the field out entirely
   for non-time-series tasks — a structural omission the task type already explains. Record `trivial_baseline` for **every** task type, time series or not — and compute it **per
horizon** when `forecast_horizons` holds more than one, since the naive error grows with lead time
and a single pooled figure would flatter the long horizons and punish the short ones.

   **The shape of the distribution field must match the task, not default to a single number:**
   for binary classification, report `class_distribution` as a two-entry breakdown (e.g.
   `{"0": 0.68, "1": 0.32}`), not a single "balance" float — a lone number only conveys the
   positive rate and doesn't generalize; for multiclass classification, `class_distribution`
   covers every class present, not just two; for multi-label classification, report the
   positive rate independently per label, since a row can carry more than one; for ordinal
   regression, report the distribution across the ordered categories, in order; for regression
   and time series, there's no class distribution at all — use `range` and `distribution_notes`
   instead, as already established.

   The segment breakdown must use a metric that fits the actual task, not a fixed field: for
   binary classification, report the positive rate per segment; for multiclass, report the full
   per-class distribution per segment (the segment's `value` becomes a distribution object, not a
   single float, when `metric` is `class_distribution`); for regression or time series, report a
   central-tendency stat (mean or median) per segment. Always include the sample count (`n`) for
   each segment alongside the stat, since a stat computed on very few rows (e.g. a handful of
   samples for one weekday) is far less reliable than one computed on many — this `n` is what
   lets a downstream node judge whether a segment finding is trustworthy enough to act on, and
   what should drive whether a segment gets an `underrepresented_categories` flag in step 9. Be
   especially cautious with a segment showing an unusually extreme value on a small `n` — this is
   exactly the signature of regression to the mean (an extreme measurement is disproportionately
   likely by chance alone in a small sample), not necessarily a real effect.

4. **Recommend an evaluation metric, using the table below.** Reason from the target's
   distribution and the stated goal about which error type is costlier (false negative vs. false
   positive), and recommend a primary + secondary metric with explicit reasoning. This is not a
   default (e.g. "always AUC") — it must follow from the specific goal and target distribution.

   **On a time-series task, recommend the primary and stop worrying about the floor.** Training
   computes MAE, RMSE, SMAPE and MASE on every time-series call whatever EDA says, so this
   recommendation's whole job is to name which of them (or which other metric) should be *primary*
   and why, given the target's distribution, its zero days, and the stated goal. Listing the other
   three as "secondary" adds nothing; arguing for a primary that is not one of them — pinball loss
   for a quantile goal, say — is a real recommendation and is welcome.

   **Also reason from what else EDA has already found, not just target distribution and goal:**
   - If `segment_breakdown` (step 3) found a segment with meaningfully different target behavior
     (e.g. a notably higher positive rate in one region), note in `recommended_metric.reasoning`
     that a single aggregate metric could mask this disparity, and recommend per-segment metric
     reporting as a secondary check — a model that looks good in aggregate can still perform
     badly for a specific, meaningful subgroup.
   - If `feasibility` found the row count borderline (per step 2's criteria), prefer a metric
     that's less sensitive to small-sample variance where a reasonable choice exists (e.g. a
     metric that doesn't require a large positive-class count to be stable), and note this
     tradeoff explicitly rather than picking a metric that happens to need more data than is
     actually available.

   **For multiclass classification, also specify an averaging strategy** (`macro`, `micro`, or
   `weighted`) and why — `macro` if every class matters equally regardless of frequency (e.g. a
   rare-but-critical class shouldn't be drowned out), `weighted` if overall frequency-weighted
   performance is what the goal actually cares about. Leave `averaging` out entirely for binary
   classification and regression, where it doesn't apply — a structural omission the task type
   already explains, not something requiring its own note.

   The situation-to-metric tables for binary, multiclass, regression, time-series,
   ordinal and multi-label tasks are in `references/metric_selection.md`. Pick from them
   using the reasoning above — the table narrows the options, it does not make the choice.

5. **Profile features.** Compute full per-feature stats for every column — including mean,
   median, std, skewness, kurtosis, and outlier count (e.g. IQR-based) for numeric features — but
   only surface the top-N (by target correlation / mutual information) individually in the
   output. Aggregate the rest into summary statistics (counts, average missingness, skew counts,
   cardinality bands) by dtype group — never list all columns individually in a wide dataset.
   Also check whether any feature is dominated by a single value even though it isn't technically
   constant or missing (e.g. a numeric column that is 90%+ zero) — this makes a feature nearly
   as useless as a constant one, and won't be caught by missingness or constant-value checks
   alone. For each feature in `top_features`, add a short `reasoning` explaining *why* it's
   relevant — not just that it correlates, but what that suggests (e.g. "strong positive
   correlation with the target and low missingness — likely to be a primary driver of
   predictions"), plus a `confidence` (`low`/`medium`/`high`) reflecting how solid that
   interpretation is given the sample size and any competing explanations (e.g. confounding with
   another top feature).

6. **Analyze relationships.** Feature-target correlation and mutual information (for the top
   features), plus feature-feature multicollinearity (high pairs only, not the full matrix in the
   output — save the full matrix as a CSV artifact instead). Use the method that matches the pair
   of types involved, not one generic "correlation":

   | Pair of types | Method |
   |---|---|
   | Numeric – numeric | Spearman's rank correlation (captures monotonic nonlinear relationships that Pearson misses) |
   | Categorical – categorical | Cramér's V |
   | Numeric – categorical | Cramér's V, with the numeric feature discretized first |

   In `top_features`, a categorical entry should report `target_cramers_v` instead of
   `target_corr` (Cramér's V doesn't have the same numeric-only correlation meaning) — set
   `target_corr` to `null` on those entries, but this is one of the allowed "structural null"
   cases per the guardrails, since the field genuinely doesn't apply to that dtype; it doesn't
   need a `note` the way a skipped-but-applicable computation would.

   **`target_mi` is not optional and not a structural-null case for a numeric feature.** Every
   entry in `top_features` must either report a real computed mutual information value, or, if it
   was deliberately skipped (e.g. judged unreliable at this sample size, or redundant given a
   strong correlation already found), say so explicitly via a `mi_note` — never leave it `null`
   with no explanation. A silent `null` here is exactly the failure mode the no-unexplained-null
   guardrail exists to prevent, and it is not covered by the categorical `target_corr` exception
   above, which only applies when the field is genuinely inapplicable to the dtype — MI applies
   to every feature regardless of type.

   **Screen for feature interactions, not just pairwise correlation.** A feature's relationship
   with the target can change depending on another feature's value (e.g. `blood_sugar` might
   correlate strongly with the outcome for older patients but weakly for younger ones) — this is
   invisible to a correlation matrix, which only ever measures each feature's relationship with
   the target in isolation. A lightweight, model-free screen: for each top feature, check whether
   its target correlation shifts meaningfully across the segments already computed in
   `segment_breakdown`. **Only report a shift when every segment being compared has at least the
   same minimum sample size used for `underrepresented_categories` (e.g. `n` ≥ 30)** — the same
   regression-to-the-mean risk flagged in step 3 applies here: an apparent correlation shift in a
   small segment is more likely to be sampling noise than a real interaction. Record any found in
   a `feature_interactions` list, with the `n` for each segment included alongside the shift, and
   a `confidence` — this screen is a coarse signal, not a rigorous interaction test, even when the
   sample-size gate is met.

   **Rigorous interaction detection (partial dependence plots, Friedman's H-statistic, SHAP
   interaction values) is explicitly out of scope for this node.** These all require a trained
   model to compute, the same reason `sklearn.inspection`-style tools (partial dependence,
   permutation importance) were already assigned to Training/Diagnosis rather than EDA — nothing
   here changes that boundary. If the lightweight screen above finds a plausible interaction,
   note it in `recommended_strategy` as something worth a deeper look once a model exists, rather
   than attempting to resolve it here. Separately, automatically *creating* interaction features
   (e.g. products or ratios of feature pairs) is Feature Engineering's job, not EDA's or
   Training's — EDA only ever flags that an interaction might exist, never constructs a feature
   to capture it.

7. **Analyze missingness using the MCAR / MAR / MNAR framework, not just "structural vs.
   sporadic."** Different missingness mechanisms require genuinely different treatments — using
   the wrong one introduces bias, not just noise. This node classifies the mechanism and
   recommends a treatment; it never imputes or removes anything itself, same boundary as always.

   The classification checks and the treatment recommended for each mechanism are in
   `references/missingness_framework.md`. Classify first, then recommend — this node never
   imputes or removes anything itself.

8. **Check data quality.** Inconsistent categorical labels (e.g. "M"/"Male"/"male"), out-of-range
   values, mixed units, and outliers. These are usually data entry issues or genuine extreme
   values, not necessarily errors — note them and recommend a next step (e.g. clip, investigate,
   or leave as-is pending domain input) but do not remove, cap, or transform anything here;
   detection and recommendation only, same boundary as missing data. While checking a numeric
   column, also note if its values cluster unnaturally at round numbers (multiples of 5, 10) —
   this usually signals human rounding at data entry rather than genuine structure, and is worth
   a one-line note in that column's `out_of_range_values` entry (or a new entry if the column
   otherwise looks clean) rather than a separate check of its own. Also check for label noise —
   rows with near-identical feature values but conflicting target labels — and record the rate
   found, even when it's too low to affect the feasibility verdict.

   **Outlier detection needs more than a per-feature IQR/Z-score pass.** Use that as the
   baseline. Per-feature checks miss a row that isn't extreme on any single column but is
   anomalous in combination — if you have reason to suspect this (e.g. a feasibility concern
   about data quality, or a domain hint that fraud/error patterns tend to show up as unusual
   combinations rather than single extreme values), a multivariate check (e.g. Isolation Forest)
   is worth running; otherwise the per-feature pass is sufficient and this is not a required step.
   **Before recommending an extreme value be clipped or dropped, consider whether it could be
   genuine signal rather than an error** — automatically discarding real extreme values as
   "impossible" has caused real, well-documented failures (e.g. atmospheric monitoring software
   that discarded the actual ozone hole readings because they fell below an assumed floor). Lower
   the `confidence` on an outlier recommendation when there's a plausible reason the value could
   be real, rather than defaulting to high confidence that it's an error.

   **Check whether the dataset itself might be a biased sample of what it claims to represent**
   — this is different from any column-level check above; it's about whether the row population
   is systematically skewed by how it was collected. Consider whether a plausible mechanism
   exists for the goal at hand: survivorship bias (does the dataset only include cases that
   "survived" some process — e.g. patients who returned for follow-up, excluding those who
   didn't), self-selection bias (did subjects opt in, in a way correlated with the outcome), or
   recall bias (does the data depend on retrospective self-report). This cannot be confirmed
   statistically from the dataset alone — if a plausible mechanism exists given the goal and
   domain, raise it via `clarification_needed` asking about the data collection methodology,
   rather than silently assuming the sample is representative.

   **Check for duplicate columns** — two or more columns with identical (or near-identical, e.g.
   a renamed copy) values across all rows. This is distinct from `multicollinearity`, which
   flags strong correlation, not exact duplication. Record any found in
   `data_quality_issues.duplicate_columns`. Do not decide on your own whether they're safe to
   drop — a duplicate could be intentional (e.g. a legacy column kept during a schema migration)
   and dropping data is consequential. Instead raise it via `clarification_needed`, same pattern
   as an ambiguous leakage finding: ask whether the duplicate is intentional and, if not, which
   copy should be kept. Mark it `blocking: true` only if the duplicate involves the target or an
   already-leakage-flagged column; otherwise it's a normal (non-blocking) clarification, since
   the run can reasonably proceed with either copy in the meantime.

9. **Compile flags.** Group findings into: high cardinality, possible leakage, possible temporal
   leakage (a field not known at prediction time), high missingness, near-constant columns,
   dominant-value columns, underrepresented categories, and PII/sensitive fields. Underrepresented
   categories are distinct from `segment_balance` — a category can have too few samples for the
   model to learn it reliably regardless of whether the target rate within it looks balanced,
   which is a data-sufficiency and fairness concern in its own right (particularly for sensitive
   attributes). These are the columns/values downstream nodes should treat specially or drop.

   **Every entry in every flag list must be a plain column name (a string) and nothing else,
   with exactly one column per entry.** Never embed an explanation, caveat, or sentence into what
   should be an identifier, and never join multiple real column names into a single entry (e.g.
   `"name/street/telephone"` is three columns, not one — each must be its own separate entry). If
   several similar columns share the same flag reason (e.g. a batch of new PII columns added by a
   schema change), list each column as its own entry and put the shared reasoning in each one's
   `note`, or in `limitations` if it applies dataset-wide — never collapse them into one string.
   If a flag needs a note, the entry must be a structured object — `{"name": "col_name", "note":
   "..."}`, not a string with the note appended inline. This applies uniformly across every flag
   category, not just PII, and applies to any dataset regardless of how many columns happen to be
   affected.

   **This same rule applies to every `name` field anywhere in this output, not just `flags`.**
   `data_quality_issues.out_of_range_values`, `inconsistent_categories`, `duplicate_columns`, and
   any similar entry all follow the identical discipline: one column per entry, never several
   names joined into one string (e.g. `"weather_avg_temp, total_revenue, avg_order_value"` is
   three separate findings, not one, even if they share the same conclusion — "no action needed"
   for three columns is three entries with the same `suggested_action`, not one entry naming all
   three).

10. **Decide which plots to generate, using the rules below.** Do not generate a fixed set —
    apply these conditions against what steps 3–9 surfaced. "Always" rows are unconditional and
    cheap; every other row is gated so plot generation stays proportional to relevance, not
    dataset width. Record the triggering reason for every plot chosen.

    **This is the plot-specific case of the general "floor, not ceiling" rule above** — if
    something about this dataset stands out that no row here anticipated, generate a plot for it
    with the same discipline as every other plot (paired values CSV, `reason`, `insight`,
    `confidence`), and add `"agent_initiated": true` to that plot's entry in `artifact_paths.plots`
    so it's clearly marked as beyond the table — plots stay in their normal structure rather than
    moving to `agent_initiated_findings`, which has no `path`/`data_path` fields suited to them.

    **Every plot must save two files, not one: the rendered image, and the exact values that
    went into it, as a paired CSV file with the same stem** (e.g.
    `iteration_001_eda_hist_blood_sugar.png` and `iteration_001_eda_hist_blood_sugar.csv`). CSV,
    not JSON — this data is inherently tabular, and CSV loads directly into pandas with no
    parsing logic, matching how `full_correlation_matrix` is already stored. The image is for a
    human looking at the report; the CSV is for other nodes to load and recompute from — no
    downstream node should ever need to open an image to get information a plot represents, and
    no node should be stuck with only EDA's chosen bin count or aggregation if it needs something
    different (e.g. Preprocess re-binning the same feature differently later).

    **Every plot entry needs both a `reason` and an `insight`, and both must be substantive.**

    `reason` must answer two things, not just name the triggering condition:
    - **What specific question does this plot answer** for this dataset and goal? Not "autocorrelation
      needs a visual" — but "ACF quantifies at which lag horizons the target's self-correlation is
      meaningful, which directly determines which lag features Feature Engineering should construct
      and how far back to look; without it, lag selection would be arbitrary."
    - **Why this plot type specifically**, rather than an alternative? A scatter answers "is the
      relationship linear or curved?"; a histogram answers "what is the marginal distribution shape?";
      an ACF answers "at which periodic lags does self-correlation persist beyond noise?" — state
      what this type reveals that another type would miss for this specific question.

    `insight` must cover four layers, in this order, in a single coherent paragraph:
    1. **Observation** — what the plot actually shows, with the key numbers (spike height, skew
       direction, cluster location, correlation at specific lags). Never just "confirms what was
       already found" — state the visual fact and its magnitude.
    2. **Interpretation** — what this pattern means for the data or the domain. Why does this
       shape or spike make sense given what this dataset represents? What does it reveal about the
       underlying process generating the data?
    3. **Possible explanation** — what might cause this pattern? If there are multiple plausible
       causes, name them. If one is more likely given what else EDA has already found, say so and why.
    4. **Modeling implication** — what should a downstream node (Feature Engineering, Preprocess,
       Training) do differently because of this finding? Be specific: which features to create,
       which transformation to apply, which model assumption might be violated, which data issue
       to handle first.

    A one-sentence insight ("clear spike at lag 7, confirms seasonality") is not acceptable output
    regardless of how obvious the finding seems. The insight is the primary artifact a data scientist
    reads — it must give them enough to reason and act without opening the raw JSON.

    Include a `confidence` (`low`/`medium`/`high`) reflecting how clear-cut that reading is.

    **Store the actual values that went into the plot, not just a summary derived from them** —
    e.g. a histogram's CSV is the real (non-missing) values of that column, not just the bin
    edges and counts; a box plot's CSV is the real values, not just the quartiles. This also
    captures whatever filtering produced the plot (e.g. missing rows excluded) as an exact,
    reusable snapshot, rather than something a downstream node has to re-derive and hope matches.
    If a plot's numbers are already captured elsewhere in the output as a genuine duplicate (e.g.
    the correlation heatmap is already `full_correlation_matrix`), don't create a second copy —
    reference the existing file instead.

    **This CSV is a file for code to load, never something to read directly into your own
    reasoning context.** The same rule that applies to the raw dataset applies here: a node opens
    it with `pandas` and computes whatever it needs, it never pastes the file's contents into a
    prompt. Saving real values here does not relax the "never pass raw data into context" rule
    from the guardrails below — it only means the values exist on disk for code to use.

   The condition-to-plot table is in `references/plot_selection.md`. Apply it against what
   steps 3-9 actually surfaced, and record the triggering reason for every plot chosen.

11. **Synthesize a strategy note.** One paragraph combining the findings above into an actionable
    recommendation: what model family class is likely to work well, what needs to be dropped or
    normalized before preprocessing, and which feature to preprocess first.

    **Also recommend an overall feature-selection posture, not just a starting feature.** There
    are three general approaches — filter (rank features by a statistical measure, independent of
    any model — this node already does this via `top_features`), wrapper (iteratively add
    features one at a time and test, or start with everything and remove the weakest — an
    incremental loop like this one is a manual form of wrapper selection), and embedded (let
    feature selection happen automatically inside model fitting, e.g. Lasso or tree-based
    importance). Reason from the feature count and dataset size about which posture fits: a
    small number of candidate features (e.g. under ~15) often doesn't need one-at-a-time
    incremental addition — starting broader and letting embedded importance or a backward
    elimination pass prune the weak ones can reach a good result in far fewer iterations; a wide
    dataset (many candidate features) is where incremental, evidence-driven addition earns its
    cost. State which posture is recommended and why — this is not a fixed default, and should
    not silently default to "add one feature at a time" regardless of dataset shape.

12. **Surface key findings, limitations, and clarification requests.** State the goal being
    analyzed and list the 3–5 most important findings in plain language (`key_findings`) — what a
    human reviewing the run should see first. Separately note any real limitations or caveats in
    the analysis itself (`limitations` — e.g. a collection gap, a segment with too little data to
    judge reliably). Separately, list anything genuinely ambiguous that a domain expert could
    resolve better than you can (`clarification_needed`), each with a `blocking` flag — reserve
    `blocking: true` for cases where proceeding without an answer could materially change the
    outcome (e.g. genuine leakage uncertainty), not minor or low-stakes uncertainty. You do not
    decide whether to pause the workflow for these; that is the Planner's decision, not yours.

    **Every entry carries a `recommendation`** — the answer you would give if you had to choose
    one — plus `recommendation_reasoning` and `impact_if_wrong`. The full field table is in the
    Planner's **Clarification Handling** section. Raising a question you can see clearly enough to
    describe, without saying which way you lean, hands the user your reasoning to do. A blocking
    entry halts the run until the user answers, and the recommendation is what they weigh; it is
    never acted on in their absence.

13. **Write the output.** Build every path (the history file, generated code file, and each
    artifact) per `conventions/naming_conventions.md`, using the `iteration` number provided in the input.
    Save the full structured JSON to `history/iteration_<NNN>_eda.json`, and write a one-to-two
    sentence summary + the file path into the Shared State `history` log.

14. **Generate a human-readable report.** Write a short Markdown report to
    `reports/iteration_<NNN>_eda.md`, separate from the JSON — this is for a person to read
    directly (you, a supervisor, or documentation), not for another node to consume. Include:
    the goal and dataset this run analyzed; the feasibility verdict and why, in plain language;
    `key_findings` as a bulleted list; a short section per top feature with its `reasoning` and
    `confidence`; `limitations`; `clarification_needed` questions if any, including any duplicate
    columns found; the `recommended_strategy` paragraph with its `confidence`; and a list of the
    generated plots, each with its `reason`, its `insight`, and its `confidence`, referencing
    their `artifacts/eda/` paths so a reader can open them. Keep this readable on its own —
    someone should be able to understand what happened in this run without opening the raw JSON.
    This report is a byproduct of the run, not a substitute for the structured output — the JSON
    in `history/` remains the source of truth for every other node.


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
- Never pass a raw dataframe, raw row samples, or an image into your own reasoning context —
  everything must be reduced to numbers, aggregates, or short text before you reason over it.
- Never list every column individually when the dataset is wide — always aggregate by group and
  surface only the top-N individually relevant features.
- Never skip the feasibility check, even if the dataset looks obviously fine — it must be an
  explicit, reasoned judgment in the output, not assumed.
- Every recommendation (metric, missingness handling, outlier handling, plots chosen) must
  include a `reasoning` field — a bare conclusion without reasoning is not acceptable output.
- Every field that carries a `reasoning`, `note` explaining a judgment call, or `suggested_action`
  must also carry a sibling `confidence` (`low`/`medium`/`high`) — this includes
  `recommended_metric`, each missingness entry's `suggested_action`, each `data_quality_issues`
  finding, `label_noise`, `recommended_strategy`, each `top_features` entry's `reasoning`, and
  each plot's `insight`. Use qualitative levels, not a percentage — a number like "73% confident"
  implies a precision these judgments don't actually have. Raw measured stats (correlations,
  counts, percentages) don't need a confidence — they're computed, not inferred; confidence is
  only for the interpretation layered on top of them.
- **Never profile only the primary file.** Every entry in `context.data_sources` is opened and
  accounted for in `source_inventory`, with a `used` / `not_used` verdict and a reason drawn from
  its actual contents. Judging a source irrelevant from its filename is the failure step 1a exists
  to prevent.
- Never remove rows, impute values, cap outliers, or transform data — this node detects and
  recommends only; acting on that recommendation is Preprocess's job.
- Never fail feasibility on leakage alone — leakage is fixable (drop the column) and belongs in
  `flags`, not in the feasibility verdict. Feasibility fails only on the hard blockers listed in
  step 2's criteria table, or — rarely — a genuinely novel blocker explicitly justified at
  `confidence: high` per step 2's exception. Never on a vague or unstated judgment call.
- Every computed value in the output must make its method explicit — either via a dedicated
  `method` field, or because the field name itself unambiguously implies the method (e.g.
  `target_cramers_v` vs. `target_corr` already tells you which was used, no redundant field
  needed). A number with no stated method (e.g. an outlier count with no mention of IQR vs.
  Z-score, or a "likely reason" for missingness with no stated detection approach) can't be
  trusted or reproduced by whoever reads it — this applies to every computation in this skill,
  not just correlations.
- Never leave a structured field `null` without explanation — whether due to feasibility, an
  unreliable computation (e.g. mutual information estimated on too few rows to trust), or any
  other reason. `null` alone tells a downstream node nothing about whether the value is missing,
  inapplicable, or simply not yet computed. If a value can't or shouldn't be computed, either
  compute it anyway and flag it as low-confidence, or state why it was skipped in a nearby `note`
  or in `limitations` — never leave a bare `null` for anyone to guess at. The one exception is a
  field that is structurally inapplicable for the row's dtype (e.g. `target_corr` on a
  categorical feature, which reports `target_cramers_v` instead) — that kind of `null` doesn't
  need its own note, since the schema itself explains it.
- Never put a sentence, caveat, or more than one column name inside any `name` field anywhere in
  this output — not just `flags`, but `out_of_range_values`, `inconsistent_categories`,
  `duplicate_columns`, and anywhere else a column is being identified. Use `{"name": ..., "note":
  ...}` if context is needed, and one entry per column even when several share the same finding.
- If something looks wrong but you're not confident (e.g. ambiguous leakage), flag it rather than
  silently deciding either way.
- Never decide on your own to pause the workflow or ask the user something — only raise it via
  `clarification_needed`. Routing to the user is the Planner's decision, not this node's.
- The `reports/` Markdown file must only restate what's already in the JSON output — never state
  a finding, number, or recommendation in the report that isn't also present in the structured
  output. The report is a readable rendering of the same facts, not a second, independent place
  for conclusions to live.
- Never generate a plot image without also saving its paired values as a CSV (same stem), unless
  the plot's numbers are a genuine duplicate of something already in the output — in that case,
  reference the existing field or file instead. An image with no accessible data twin means a
  downstream node would have to open the image to use it, which defeats the entire
  "numbers, not images" design of this system.
- Saving real values to a plot's CSV does not permit loading that file into your own reasoning
  context. It exists for other nodes to load with code (e.g. `pandas`) and compute from — the
  same "never pass raw data/rows into context" rule applies to it as to the raw dataset itself.
- If any step fails (a computation errors, a file can't be parsed, a plot can't be generated),
  record it in `issues` rather than silently omitting the affected output or letting the whole
  run crash. `issues` is not for findings about the data itself — that's what `flags` and
  `data_quality_issues` are for — it's specifically for problems with the *execution* of this
  node.

## Input / output contract

Full input/output examples: `contracts/eda.json` (feasible case) and
`contracts/eda_infeasible_example.json` (infeasible case — note that every field is still
populated with whatever was actually computed; only `feasibility.trainable` and the downstream
recommendation change).

Top-level output sections: `schema_version`, `dataset_overview` (including `grouping_structure`
when rows are not independent), `feasibility`, `target`
(`task_type` — binary/multiclass/regression, or time series, ordinal regression, and multi-label
classification where applicable; `class_distribution` shaped to the task; `time_series_analysis`
when the task is time series (trend, seasonality, stationarity, autocorrelation); plus
`segment_breakdown` with a task-appropriate `metric`, and `recommended_metric` with `confidence`,
an `averaging` strategy for multiclass, and reasoning informed by segment disparities and
row-count reliability where relevant),
`source_inventory` (one entry per file in `context.data_sources`: path, format, shape, key
columns, granularity, time coverage, what it uniquely holds, and a `used` / `not_used` verdict
with its reason), `join_plan` (join keys per source pair, granularity alignment, coverage overlap
with row percentages, estimated match rate, and the risks named in step 1a — empty only when
there is genuinely one source),
`feature_stats_summary`, `top_features` (each entry includes `reasoning` and `confidence`),
`top_correlated_pairs`, `multicollinearity`, `feature_interactions` (segment-conditional
relationship shifts, gated by a minimum `n` per segment, with an explicit `method` — a coarse
model-free screen, not a substitute for rigorous interaction detection, which happens later in
Training/Diagnosis), `missingness` (each column's `mechanism` classified as MCAR/MAR/MNAR with
supporting `mechanism_evidence`), `data_quality_issues`
(including `label_noise` and `duplicate_columns`, each with `confidence`), `flags` (including
`dominant_value` and `underrepresented_categories`), `recommended_strategy` (with `confidence`),
`key_findings`, `limitations`, `clarification_needed` (each entry carries `blocking`, `confidence`, `recommendation`, `recommendation_reasoning` and `impact_if_wrong` — see the Planner's Clarification Handling section; blocking halts the run until the user answers —
including any duplicate-column questions), `issues` (execution problems, distinct from
data-quality findings), `agent_initiated_findings` (anything explored beyond what a step
specifies, or an entirely new analysis the procedure didn't anticipate — each entry tagged with
its `related_step`, `method`, `reasoning`, and `confidence`, never blended into the standard
fields above), and `artifact_paths` (each plot entry includes a `reason`, an `insight`,
a `confidence`, a `path` to the image, and a `data_path` to its paired values CSV — or, only when
the plot is a genuine duplicate of a large artifact already saved elsewhere, such as the
correlation heatmap and `full_correlation_matrix`, a `data_source` reference instead; a plot
generated beyond the table in step 10 also carries `"agent_initiated": true`).