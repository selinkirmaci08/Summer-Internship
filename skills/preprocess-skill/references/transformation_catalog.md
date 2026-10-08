# Per-Feature Transformation Catalog

The five sub-decisions Preprocess step 5 applies to each feature in scope - category normalisation, missingness, outliers, scaling, encoding - with the technique tables for each. Every statistic named here is fit on `_split == "train"` rows only.

Used by: `preprocess.md` step 5.

---

a. **Normalize inconsistent categories**, if this feature was flagged in
   `data_quality_issues.inconsistent_categories` — map variants to one canonical form before
   anything else touches this column.

b. **Handle missingness, keyed to the `mechanism` EDA already classified** (MCAR/MAR/MNAR,
   with `mechanism_evidence`) — don't re-derive "structural vs. sporadic" from scratch; consume
   what EDA already determined:

   | `mechanism` (from EDA) | Action |
   |---|---|
   | MNAR | Treat as its own category / explicit sentinel value if EDA's evidence points to structural absence (the value genuinely doesn't apply); otherwise add a missing-indicator alongside whatever imputation is used. Never impute a "normal" value here — that would erase the informative signal MNAR represents |
   | MCAR | Median imputation for numeric (robust to skew), mode for categorical — a simple univariate fill is valid here because the missingness has no relationship to anything else |
   | MAR | Multivariate/model-based imputation (e.g. `IterativeImputer`, `KNNImputer`), leveraging the specific feature(s) EDA's `mechanism_evidence` identified as correlated with the missingness — a simple imputation would ignore that relationship and distort it. **This creates real tension with the one-feature-at-a-time scope of `features_to_process`** — it needs the correlated feature(s) to already be processed and available. If those columns aren't yet available, note this in `reasoning` and fall back to simple imputation with a lower `confidence`, flagging that a better MAR-aware imputation should happen once those columns are in scope. **Override this default when EDA's `mechanism_evidence` describes a deterministic, zero-overlap separator** (e.g. missingness explained entirely by a collection-start date, with no rows where both the correlated feature and this one are observed together) — standard model-based imputation has nothing to learn from in that case, since there's no overlapping support. Use a missing-indicator with no point-estimate imputation instead, and state this override explicitly in `reasoning` rather than silently following the generic MAR row |
   | No `mechanism` provided (e.g. `context` didn't include this feature's missingness finding) | Fall back to checking `correlated_missingness_pairs`: if this feature co-occurs with another's missingness, add a missing-indicator rather than imputing independently; otherwise default to simple imputation with `confidence: low` and note that EDA's mechanism classification wasn't available for this call |

   Regardless of mechanism, if the feature was flagged in `correlated_missingness_pairs`, add
   a binary "was missing" indicator column alongside imputing (a `MissingIndicator`, in
   scikit-learn's terms) — this preserves the joint-missingness signal even after the value
   itself is filled in.

c. **Handle outliers**, per `data_quality_issues.out_of_range_values` if present:

   | Situation | Action |
   |---|---|
   | Flagged as a likely data entry error (e.g. impossible value like negative age) | Clip to a valid range, or drop the specific row only if the value is truly unusable — dropping rows should be rare and always justified in `reasoning` |
   | Flagged as a genuine extreme value (not an error) | Clip to a reasoned bound (e.g. IQR-based) if the downstream model family is sensitive to scale (linear/distance-based); leave as-is if a tree-based model family was recommended, which is largely outlier-robust |

   **Check step (d) below before clipping a heavily skewed feature.** A log or power
   transform, applied there for the feature's distribution shape, often resolves apparent
   outliers on its own — a value only looks "extreme" because the untransformed scale is
   skewed. Applying both a raw-value clip here and a shape transform in (d) risks
   double-treating the same values; prefer the transform alone when the feature is heavily
   skewed, and reserve clipping here for outliers that remain genuinely extreme even after
   accounting for the distribution's natural shape.

d. **Scale numeric features**, informed primarily by the feature's *distribution shape* (per
   EDA's stats), with EDA's `recommended_strategy` gating whether scaling is needed at all.
   Note: "normalize" is used two different ways in this skill — step 5a normalizes *category
   labels* (canonical spelling), while normalization here means scaling numeric *values* to a
   standard range; don't conflate the two just because they share a name.

   | Situation | Action |
   |---|---|
   | Tree-based model family recommended | Scaling is usually unnecessary — record `"method": "none_needed"` with reasoning, don't scale by default |
   | Feature is roughly uniformly distributed across its range, few or no extreme outliers, and bounds are stable | Linear (Min-Max) scaling — appropriate specifically for a flat-shaped distribution, not a default choice |
   | Feature is roughly normally (Gaussian) distributed | Standard scaling (Z-score) — appropriate for a bell-shaped distribution |
   | Feature is heavily skewed or power-law/heavy-tailed (e.g. \|skew\| > 1, per EDA) | Log scaling (`log1p`) can serve as the scaling method itself for this shape, not just a pre-step — or a power transform (Yeo-Johnson if negative values are present, Box-Cox if strictly positive) followed by standard scaling if more flexibility is needed than a plain log |
   | Roughly Gaussian bulk but contaminated by heavy outliers | Robust scaling (median/IQR-based), or standard (Z-score) scaling combined with clipping the resulting score (e.g. clip Z-scores beyond ±3) — either avoids letting a handful of extreme values dominate the scaled range |
   | Downstream model needs a bounded input range (e.g. a neural network with a sigmoid/bounded activation), or the feature genuinely has a fixed, meaningful min/max (e.g. a percentage, a bounded score) | Min-Max normalization to [0, 1] (or another fixed range) — this bounded-range requirement overrides the shape-based default above, since the model's own constraint matters more than the feature's natural distribution |

e. **Encode categorical features**, informed by cardinality and EDA's flags:

   | Situation | Action |
   |---|---|
   | Low cardinality (roughly <10 categories), no ordinal relationship | One-hot encoding |
   | Moderate cardinality, not flagged `high_cardinality` but with a long tail of rare categories | Group infrequent categories (e.g. below a frequency threshold) into a single "infrequent" bucket, then one-hot the rest — a middle ground before resorting to frequency/target encoding |
   | Flagged `high_cardinality` by EDA, and you want a compact encoding with no leakage risk and no collision between categories that happen to share a frequency | Binary encoding — represents each category as a binary number split across `log2(n_categories)` columns, far fewer than one-hot, and (unlike frequency encoding) two categories with the same frequency don't become indistinguishable |
   | Flagged `high_cardinality` by EDA, and new/unseen categories may appear later (e.g. at inference time, or in a future iteration's incoming data) | Hashing encoding (feature hashing) — maps categories to a fixed number of buckets via a hash function, so it never needs the full category set known in advance, unlike every other option in this table |
   | Flagged `high_cardinality` by EDA, and the goal specifically benefits from encoding the category's relationship to the target | Frequency encoding or target encoding — never one-hot, which would explode dimensionality. **Target encoding must use cross-fitting (fit on k-1 folds of the train-split rows, transform the held-out fold, rotate) — never a single global fit on all train rows, which leaks target information into the same rows it's derived from. This applies within the train split only; validation rows are still encoded using the mapping learned from the full train split, per the fit-on-train-only rule above.** |
   | Natural order exists (e.g. severity levels) | Ordinal encoding preserving that order, not one-hot |
   | EDA's recommended model family is a neural network | An embedding layer (learned during model training) is often preferred over any of the above for very high cardinality — note this as a recommendation for Training rather than something Preprocess does itself, since it requires the model architecture to exist |
