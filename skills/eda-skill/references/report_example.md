# EDA Report — Iteration 1

**Goal:** predict readmission risk
**Dataset:** `path/to/data.csv` (250,000 rows, 87 columns)

## Feasibility

**Trainable: Yes.** 250k rows, the target has a usable class balance, and no blocking issues were
found.

## Key Findings

- Target is imbalanced (32% positive) — recall recommended as the primary metric (**confidence:
  high**), since missing a true readmission is likely costlier than a false alarm.
- Region=east has a notably higher target rate (52%) than other regions — possible subgroup bias
  worth investigating.
- 87 columns are available, but only a handful show strong target relevance — most predictive
  signal is concentrated in a small feature subset.

## Top Features

- **blood_sugar** *(confidence: high)* — strongest target correlation of any feature (0.41) and
  low missingness. Likely a primary driver of predictions, not just a passenger correlate.
- **bmi** *(confidence: medium)* — moderate target correlation (0.29), but also correlates with
  blood_sugar (0.35) — some of its apparent signal may be redundant rather than independent.
- **age** *(confidence: low)* — weak direct correlation (0.15), and flagged as multicollinear
  with `years_since_first_visit`. Its true relevance is unclear until that redundancy is
  resolved.

## Limitations

- There's a collection gap from 2020-03 to 2020-05 — that period is underrepresented, and
  findings may not generalize well to it.
- The region=east segment has fewer rows than other regions, so its 52% balance estimate is less
  reliable than the overall figure.

## Open Questions

- **[Blocking]** Is `visit_id`'s near-perfect correlation with the target a genuine leakage risk,
  or expected in this domain? *(low confidence — needs domain input before proceeding)*
- **[Blocking]** Is `follow_up_scheduled`, recorded after the outcome, safe to exclude as
  temporal leakage? *(medium confidence)*
- **[Not blocking]** `provider_id` and `attending_provider_id` look like exact duplicates — is
  that intentional (e.g. a renamed column kept for compatibility), and if not, which should be
  kept? *(medium confidence — doesn't hold up the next step, since retaining both is redundant
  but not harmful)*

## Recommended Strategy *(confidence: medium)*

Wide dataset with several high-cardinality categorical fields — tree-based models and
target/frequency encoding likely outperform one-hot encoding with linear models. Normalize
gender categories and drop PII/leakage columns before preprocessing. Start with `blood_sugar`
given it has the strongest target correlation.

## Generated Plots

- [`hist_blood_sugar.png`](../artifacts/eda/iteration_001_eda_hist_blood_sugar.png)
  *(confidence: high)* — generated because `blood_sugar` has the highest target correlation.
  **Insight:** distribution is right-skewed with a long tail past 250, consistent with the
  outlier count already flagged — most values cluster between 120-200.
- [`class_balance.png`](../artifacts/eda/iteration_001_eda_class_balance.png)
  *(confidence: high)* — generated because target imbalance was detected (32%). **Insight:** the
  positive class is a clear minority but not extreme — imbalanced-but-workable, matching the
  feasibility assessment that resampling isn't required.

---
*Full structured output: `history/iteration_001_eda.json`*