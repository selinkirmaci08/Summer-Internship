# Feasibility Criteria

The seven hard-blocker checks EDA step 2 applies, in order, plus the rules governing the row-count pre-scan, borderline verdicts, leakage, and the narrow escape hatch for a blocker none of the seven capture.

Used by: `eda.md` step 2.

---

| Check | Hard blocker if... | Otherwise |
|---|---|---|
| Row count (general, i.i.d. tabular) | fewer than ~10× the number of candidate features, or under an absolute floor (e.g. <100 rows) | feasible; note the ratio in `limitations` if it's on the low side |
| Row count (time series / sequential) | fewer full task-relevant cycles than needed to learn the pattern (e.g. daily data aimed at weekly patterns needs multiple weeks; a task needing seasonal/holiday effects needs multiple years, not one window) | if only one cycle is present, feasible but must be noted in `limitations` — the model can't be validated against a second, unseen cycle |
| Target variance (classification) | the minority class has fewer than ~20–30 samples, or only one class is present at all | imbalanced-but-workable (each class has enough samples) is feasible — handled via `recommended_metric`, not a blocker |
| Target variance (regression) | near-zero variance/std relative to scale (e.g. >99% of values identical) | otherwise feasible |
| Missingness — target | the target column itself is missing beyond a hard threshold (e.g. >20% of rows) | below that, feasible — row-level target missingness is just fewer usable rows, handled in `missingness` |
| Missingness — features | after excluding leakage/PII-flagged columns, zero or near-zero features remain with usable (non-catastrophic) missingness | otherwise feasible — missingness is handled per-column in step 7 |
| Label noise | a substantial share of rows share near-identical feature values but conflicting target values (check by grouping on the top features and looking for target disagreement within groups) — this sets a ceiling on achievable model performance no feature engineering can fix | isolated conflicts are normal noise, not a blocker — note them in `data_quality_issues` rather than failing feasibility |

**The row-count check uses the raw column count, not a post-exclusion count** — leakage
detection happens later, in steps 5–6, so the true usable feature count isn't known yet at
this point in the procedure. Reordering the procedure to determine leakage first isn't the
fix, since that would mean running the expensive analysis before the cheap gate meant to
filter out hopeless datasets early. Instead, run a **cheap heuristic pre-scan** for columns
that are near-certain to end up excluded — a column that's unique or near-unique per row, or
whose name matches a common ID pattern (`*_id`, `*_number`, `ssn`, and similar) — this is not
the real leakage analysis, just a rough estimate. If this rough estimate would flip the
row-count verdict (e.g. the dataset passes using the raw count but would likely fail using the
estimated usable count, or vice versa), state this discrepancy explicitly in
`feasibility.reasoning`. The official verdict still uses the raw count, but flagging the
discrepancy means whoever reads the output knows the real answer, once flags are actually
compiled in step 9, could turn out different from what this early check assumed.

**A row-count check that lands near its threshold (not a clear pass or a clear fail) should
not be forced into a definitive true/false call.** Mark feasibility `trainable: true` but note
the borderline volume in `limitations`, and recommend in `recommended_strategy` that Training
run a learning-curve check (train a baseline model on increasing fractions of the data and
watch whether validation performance is still rising at 100% or has plateaued) as an early
sanity check — this is empirical evidence EDA cannot produce itself, since it requires actually
training a model, which is Training's job, not EDA's.

**Leakage is never one of these checks and never fails feasibility on its own** — it is always
fixable (drop the column) and always belongs in `flags.possible_leakage`, whether or not
feasibility passes.

**These 7 checks are the default, not an absolute ceiling.** In rare cases, something about a
specific dataset may present a genuine hard blocker none of these checks capture. Only invoke
this if you can state explicitly why none of the 7 standard checks apply to what you found,
and mark the finding `confidence: high` — this is a deliberately narrow escape hatch, not a
license for vague or unstated infeasibility calls. The guardrail against failing feasibility on
an unlisted, unjustified judgment call still applies; this exception requires clearing a high
bar, not lowering it.
