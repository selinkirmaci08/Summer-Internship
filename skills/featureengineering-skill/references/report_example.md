# Feature Engineering Report — Iteration 5

## Call Context

**Mode:** `initial` — first Feature Engineering call this run, building on Preprocess's latest
output.
**Features already in the model:** `day_of_week`

## Features Created

### bread_quantity_lag_7 *(confidence: high)*

**Technique:** lag feature — `bread_quantity` shifted 7 rows (one week) back, using each row's
own history only.

**Why:** EDA's time-series analysis found strong autocorrelation specifically at lag 7 — today's
demand correlates more with demand exactly one week ago than with adjacent days. Rather than
leaving the model to rediscover this weekly rhythm on its own, this feature hands it the exact
comparison EDA already confirmed matters. Leakage rule: row-local causal — this depends only on
each row's own past, so it never looks at another row's data regardless of train/validation
split.

**Relevance check:** Spearman correlation with the target is 0.61 — strong, consistent with the
autocorrelation EDA found.

### is_weekend *(confidence: medium)*

**Technique:** calendar feature — 1 if Saturday or Sunday, else 0.

**Why:** EDA found a weekend effect (Saturday and Sunday notably higher demand), but the
underlying signal is really a two-way weekend/weekday contrast, not the full 7-level distinction
`day_of_week` already carries. Collapsing it to this binary form may be an easier signal for some
model families to pick up on than sifting it out of seven separate categories.

**Relevance check:** Spearman correlation with the target is 0.48 — moderate, and expected to
overlap somewhat with `day_of_week`'s own signal rather than being fully independent of it.

### time_index *(confidence: high)*

**Technique:** trend feature — sequential integer position of each row in chronological order.

**Why:** EDA confirmed a real, non-noise decline in demand (~28% from 2011 to 2021) and flagged
the target as non-stationary. Without an explicit way to represent "how far into the decline is
this row," a model trained mostly on early years would systematically over-predict demand in
later years.

**Relevance check:** Spearman correlation with the target is -0.52 — moderate-to-strong and
negative, consistent with the confirmed downward trend.

## Feature Selection

**Selected:** `day_of_week`, `bread_quantity_lag_7`, `is_weekend`, `time_index`
**Dropped:** none this call.

EDA's `recommended_strategy` favored a broad-start posture over incremental addition, since only
~10 candidate features remain after leakage columns were already dropped in Preprocess — with a
feature set this small, embedded importance from a tree-based model in Training is expected to do
the pruning, rather than this step aggressively narrowing the set itself.

## Open Questions

None this call.

## Issues

None.

---
*Full structured output: `history/iteration_005_feature_engineering.json`*