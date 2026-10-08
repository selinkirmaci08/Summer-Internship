# Preprocess Report — Iteration 2

## Call Context

**Mode:** `initial` — this is the first Preprocess call this run, so there's nothing to build on
yet. Everything here starts from the raw dataset rather than a previously processed file.

**Features in scope this call:** `blood_sugar`. Per the incremental design of this pipeline, only
one feature is processed at this stage — `blood_sugar` was chosen because EDA identified it as
having the strongest correlation with the target (0.41), making it the most valuable single
feature to get right before expanding to others in later iterations.

## Rows Filtered

None this call. No row-level anomalies were flagged for this dataset, so the full row set is
carried forward unchanged.

## Train/Validation Split

- **Created this call:** yes — since no `_split` column existed yet for this dataset, one was
  created now and will be reused for the rest of the run.
- **Method:** stratified on the target. A plain random split risked producing a validation set
  with a meaningfully different positive rate than the training set, given the target's known
  imbalance (32% positive, per EDA) — stratifying ensures both partitions keep roughly the same
  32% rate, so validation performance isn't an artifact of an unlucky split.
- **Ratio:** 80/20 — this dataset's held-out set was removed before the run began, so the whole
  80/20 is carved out of training data. The 20% validation portion is large enough to give a
  stable ranking signal across the many candidates this run will compare, while keeping most of
  the data available for the model to actually learn from.
- **Selection noise band:** 0.011 AUC, the spread of the primary metric across the five
  stratified folds. Differences between candidates smaller than this are ties, not improvements.
- **Is validation a fair stand-in for the holdout?** Partly. The rows are drawn from the same
  population and the split is random rather than temporal, so there is no drift argument here —
  but validation is used repeatedly across iterations and the holdout is scored once, so every
  number below is expected to be mildly optimistic. No number in this report is a final number.
- **Why this matters going forward:** every later Preprocess call and Training reuse this exact
  row assignment. This consistency is what lets any statistic fit later (a scaler's mean, an
  imputation value, an encoding map) be trusted as genuinely unseen-data-safe — if the split were
  recreated each time, there'd be no way to guarantee a given row hadn't already influenced a
  fitted parameter in an earlier iteration.

## Per-Feature Transformations

### blood_sugar *(confidence: medium)*

**Missing values — median imputation (value used: 142.0).** 2% of rows were missing this value,
and the missingness looked sporadic — no relationship to any other column, no clustering by time
or segment — which is exactly the situation median imputation is meant for: a single global
estimate stands in reasonably well for missing values that aren't systematically different from
the observed ones. Median was chosen over mean specifically because `blood_sugar` is right-skewed
(skew 1.4 before this step) — a mean would be pulled upward by the long tail of high values and
would systematically overestimate a "typical" missing value, whereas the median is robust to that
skew.

**Outliers — clipped to [40, 310] using IQR bounds.** The IQR method flags any value more than
1.5× the interquartile range beyond the 25th/75th percentiles as an outlier; 12 such values were
found. Clipping (rather than dropping the rows entirely) was chosen because these looked like
genuine extreme readings, not data entry errors — the values were physiologically plausible, just
rare. Clipping preserves the row's other feature values for training while preventing these
extreme blood_sugar readings from disproportionately influencing a distance-based or gradient-
based downstream model.

**Scaling — standard scaling (mean 145.2, std 38.7).** This centers the feature at zero with
unit variance, which matters if a later iteration trains a linear or distance-based model —
without scaling, blood_sugar's raw range (roughly 40-310) would dominate a feature like `age`
(roughly 0-100) purely because of its larger numeric scale, not because it's actually more
important. All three statistics above (the imputation median, the clip bounds, and the scaler's
mean/std) were computed using only the 80% of rows assigned to `_split == "train"`, then applied
identically to every row including the validation 20% — this is what prevents the validation
set's own distribution from quietly influencing what "normal" looks like, which would make every
per-iteration comparison an optimistic overestimate of real performance.

## Before/After Comparison

| Feature | Metric | Before | After | What changed |
|---|---|---|---|---|
| blood_sugar | missing_pct | 2.0% | 0.0% | Every missing value was filled with the train-fit median; no missing values remain |
| blood_sugar | skew | 1.4 | 0.3 | The right tail was pulled in by outlier clipping, bringing the distribution much closer to symmetric — closer to what standard scaling assumes |
| blood_sugar | outliers | 12 | 0 | All 12 previously-flagged extreme values now sit exactly at the clip bounds, no longer counted as outliers under the same IQR rule |

## Dropped Columns

| Column | Reason |
|---|---|
| visit_id | Flagged as possible leakage in EDA (iteration 1) — this column correlated almost perfectly with the target, which is a sign it may encode information that wouldn't actually be available at prediction time. Dropping it removes the risk of the model learning a shortcut that won't generalize. |
| patient_name | Personally identifiable information, flagged by EDA. Not usable as a feature regardless of any statistical relationship it might have with the target — retaining it would be both a privacy risk and a near-certain source of spurious, non-generalizable correlation (e.g. via name-based demographic proxies). |

## Generated Artifacts

- [`blood_sugar_before_after.png`](../artifacts/preprocess/iteration_002_preprocess_blood_sugar_before_after.png) —
  a visual side-by-side of the distribution before and after this call's transformations, meant
  as a quick sanity check that the imputation and scaling behaved as described above rather than,
  say, collapsing the distribution or introducing an artificial spike at the imputed value.
  Underlying values (the actual before/after numbers, not just the rendered image):
  [`blood_sugar_before_after.csv`](../artifacts/preprocess/iteration_002_preprocess_blood_sugar_before_after.csv).

## Issues

None — every step in this call completed as intended, with no execution failures or unexpected
data conditions to report.

---
*Full structured output: `history/iteration_002_preprocess.json`*