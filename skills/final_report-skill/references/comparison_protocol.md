# Comparison Protocol

The alignment rules, metric definitions, and decomposition used by `final_report.md` steps 1–4.
Open at step 1.

## 1. Alignment

A comparison is only as valid as the rows it is computed on.

| Check | Rule | If violated |
|---|---|---|
| Join key | Inner-join all systems on `(target_date, h)` — or `(date)` when single-horizon. The comparison set is the intersection. | — |
| Row loss | Report `n` per system before and after the join. | Loss > 5% for any system is a stated finding, not a silent truncation |
| Test window | All systems must have been *fitted against the same split*, not merely scored on overlapping dates. | Restrict to the overlap **and** state that fitting splits differed — the overlap does not repair it |
| Horizon grid | Pair a missing lead time to the nearest only within 3 days; label the pair (`91~90`). | Beyond 3 days, drop the horizon and list it as dropped |
| Target identity | Confirm every system's `y_true` agrees on the joined rows (max absolute difference ≈ 0). | A mismatch means different targets or different filtering — stop, do not report |

That last check is cheap and catches the worst class of error: two files that look joinable and are
measuring different things. Run it.

## 2. Metrics

Pooled and per-horizon, per system:

| Metric | Definition | Why it is here |
|---|---|---|
| `MAE` | `mean(abs(y - ŷ))` | Primary. In the target's own units, so it is readable by a domain reader |
| `RMSE` | `sqrt(mean((y - ŷ)²))` | Alongside MAE only — a much larger RMSE/MAE ratio than the other systems means one system's errors are concentrated in rare large misses |
| `SMAPE` | `mean(2·abs(y - ŷ) / (abs(y) + abs(ŷ)))` × 100 | Scale-free, so it is the only metric here that compares honestly across horizons and segments of different magnitude. State the zero-actual convention, matching the one Training used |
| `bias` | `mean(y - ŷ)` | The single most diagnostic number in a drifting series. Never omit |
| `MASE` | `MAE_system / MAE_naive` **at the same horizon** | Makes horizons comparable to each other. A global denominator hides that naive itself degrades with lead time |
| `n` | rows in the aligned set | Small `n` at long horizons is why long-horizon differences are usually noise |

Report `MASE` to three decimals, everything else to two. A difference smaller than the spread across
neighbouring horizons is noise — say "tie", not a winner.

## 3. Error decomposition

For each system compute:

```
bias          = mean(e)                    where e = y - ŷ
MAE           = mean(|e|)
MAE_debiased  = mean(|e - mean(e)|)
```

`MAE_debiased` is the error that would remain if the system's level were corrected by a single
constant. The pair `(bias, MAE_debiased)` separates two failures that look identical in `MAE`:

| pattern | reading | what fixes it |
|---|---|---|
| large `bias`, much smaller `MAE_debiased` | **level** error — the shape is right, drawn at the wrong height. Typical of tree models on a trending series: a GBDT cannot extrapolate, so a test period outside the training range collapses to the terminal leaf | Predict a ratio to a recent level, or model differences; a level correction. **Not** more features |
| `bias` near zero, large `MAE_debiased` | **shape** error — the level is right, the day-to-day pattern is wrong | Features, model family, interactions |

**The cross-system check.** When several independently-fitted systems share the same bias sign and
magnitude, that is evidence of distribution shift in the data, not of any one model's mistake. State
it explicitly — it reframes a whole set of losing results as one property of the problem.

Worked example from this project: agentic `bias +30.0`, AutoML arms `+31.7 / +32.6 / +31.0`, naive
`+5.2`. Four independent fits, one shared error. The naive method wins not by forecasting better but
by carrying no level memory.

## 4. Win counting

Per horizon, record which system has the lowest MAE. Report:

- horizons won per system, out of the total compared
- whether wins are contiguous (a band) or scattered (noise)
- the margin at each win — a win under the noise floor is reported as a tie

A pooled win assembled from one dominant band is a materially different claim from a uniform win,
and the report must let a reader tell them apart without recomputing anything.

## 5. Per-period breakdown

Same metrics grouped by calendar year (or by regime where one was identified). The column that
matters is `bias` per period: a bias that grows monotonically across periods is drift, and is
direct evidence for the level reading in §3. A stable bias across periods with a large magnitude is
a fixed offset instead — a different problem with a cheaper fix.
