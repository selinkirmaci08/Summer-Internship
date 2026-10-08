# Missingness: MCAR / MAR / MNAR

The classification checks EDA step 7 runs to decide a column's missingness mechanism, and the treatment recommended for each mechanism. Classifying wrongly introduces bias rather than noise, which is why the mechanism is decided before any treatment is named.

Used by: `eda.md` step 7.

---

**Classify each column with meaningful missingness using these checks, in order:**

| Check | Result if triggered | Mechanism |
|---|---|---|
| The missingness indicator for this column (is it missing: yes/no) correlates with the *target* | Informative missingness — the fact that it's missing is itself predictive | **MNAR** |
| The missingness indicator correlates with *other observed features*, but not the target | Missingness is explainable by other known variables | **MAR** |
| Neither of the above; no detectable pattern | Missingness looks unrelated to anything measurable | **MCAR** |
| Domain reasoning suggests the value is missing *because of its own magnitude* (e.g. high earners underreporting income) even without a clean statistical signal | Can't be confirmed statistically — depends on unobserved data by definition | **MNAR** (flag via `clarification_needed`, not a confident classification) |

**Recommend treatment based on the classified mechanism:**

| Mechanism | Recommended treatment | Why |
|---|---|---|
| MCAR | Simple imputation (median/mode) is valid; complete-case deletion is also valid if the missing rate is low (e.g. <5% of rows) | Missingness has no relationship to anything, so neither approach introduces systematic bias |
| MAR | Multivariate/model-based imputation (leveraging the correlated feature(s) found above), not simple imputation | Simple imputation ignores the systematic relationship and will distort it; model-based imputation can use the correlation to make a genuinely better estimate. Note: this can conflict with the one-feature-at-a-time scope of Preprocess — flag which other feature(s) the imputation would depend on. **Before recommending standard model-based MAR imputation, check whether the correlated feature actually has overlapping support** — rows where both the correlated feature and this one are observed together. If the correlation is deterministic or near-deterministic with zero (or near-zero) overlap (e.g. a column that's missing for every row before a known collection-start date, with no rows straddling that boundary where both are present), standard MAR imputation has nothing to learn from — there's no case where the model could see both values together. In that specific case, recommend a missing-indicator with no point-estimate imputation instead, even though the column technically meets the MAR correlation test — the mechanism label is correct, but the standard treatment for it isn't usable |
| MNAR | Treat as its own category (if structurally absent) or recommend a missing-indicator alongside whatever imputation is used, and flag via `clarification_needed` if the handling isn't obvious | Imputing a "normal" value here erases real signal — simple imputation is actively wrong, not just suboptimal, here |

**If the dataset is a time series** (per the feasibility check), do not default to the
mechanism-based recommendations above without adjustment — plain median/mode imputation
flattens exactly the cyclical/seasonal structure a time-series task depends on. Recommend
forward-fill or interpolation for small gaps, or a time-aware model-based approach (e.g.
regression on a time index, or a model incorporating lagged values) for larger or nonlinear
gaps — and say which, with reasoning, rather than defaulting to a generic method.

**Column and row deletion thresholds**, as a starting point for `suggested_action`, not a
rigid rule: a column missing more than roughly 50% of its values, with no strong target
relationship, is usually reasonable to recommend dropping entirely; row-level deletion
(complete-case analysis) is only reasonable when missingness is both low-rate (e.g. <5%) and
classified MCAR — otherwise recommend imputation instead, since deletion under MAR or MNAR
introduces bias rather than avoiding it.

Check for correlated missingness between column pairs, since jointly-missing fields usually
share a cause and shouldn't be imputed independently — this is a different check from the MAR
test above (which asks whether missingness relates to *observed values* elsewhere, not other
columns' *missingness*), and both are worth recording.
