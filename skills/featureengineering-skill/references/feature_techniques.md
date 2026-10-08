# Feature Creation Techniques

The trigger-to-technique table Feature Engineering step 3 applies. The trigger column describes an *observation about the data or model*, not an upstream node - the same observation can arrive from EDA, Diagnosis, Training or the Planner, and the technique is the same either way.

Used by: `featureengineering.md` step 3.

---

| Trigger (observation about the data or model) | Technique | Notes |
|---|---|---|
| A feature's relationship with the target varies across levels of another feature — a segment-conditional shift (e.g. the correlation between X and Y is strong in group A but weak or reversed in group B) | Feature crossing — a sum (e.g. combining sibling and parent counts into `family_size`), a product (e.g. `height × weight`), a ratio (e.g. `price_per_sqft`), or a polynomial term (e.g. `age²`) capturing the interaction | Choose the crossing type based on the *nature* of the interaction, not by default — a sum fits a combined-count relationship, a ratio fits a rate-like relationship, a product fits a combined-magnitude effect, a polynomial fits curvature in a single feature's own relationship with the target |
| The dataset has repeated-entity structure — multiple rows share a common entity identifier (e.g. multiple visits per patient, multiple transactions per customer, multiple readings per device) | Aggregation features per group (e.g. mean/count/max of a feature within each entity's history) | **Fit on train-split rows only** — an aggregate computed using validation rows leaks information about those rows. Compute the aggregate from each entity's `_split == "train"` rows only, then apply the same value to every row in that entity regardless of split. Record the number of distinct entities — this is the effective sample size for the relevance check, not the number of rows |
| Meaningful autocorrelation exists in the target or a key predictor — the current value correlates with its own recent past values | Lag features (the target or a feature's own value N periods ago) | Only include lags that don't leak future information relative to what would actually be known at prediction time. Use date-based joins (not row-shifts) when the series has calendar gaps — a row-shift silently links the wrong time points. **After creating the feature: record the NaN rate on train and apply a fill strategy** — see the lag NaN sub-step immediately below this table |
| A seasonal or periodic pattern exists in the target — the mean or distribution varies predictably across calendar positions (day of week, month, quarter, hour of day, etc.) | Calendar features (day-of-week, month, is-holiday) | Cheap, interpretable, and directly operationalises the observed pattern rather than leaving the model to rediscover it. **For any cyclically-ordered ordinal (month 1–12, hour 0–23, day-of-week as ordinal), use Fourier sin/cos encoding** (see the cyclic-encoding row below) rather than the raw integer — integers impose an artificial discontinuity at the wrap-around point (December=12 treated as far from January=1 despite adjacency) |
| A feature is a cyclically-ordered ordinal whose last value wraps back to its first (e.g. month 1–12, hour 0–23, day-of-week 0–6, compass bearing 0–359) | Fourier cyclic encoding — `sin(2π × value / period)` and `cos(2π × value / period)` — creates two features that place all values evenly on a unit circle | Always create both sin and cos: a single sinusoid cannot represent an arbitrary-phase pattern. The period must be documented (12 for months, 24 for hours, 7 for days). The raw integer must not be retained alongside the Fourier pair — it reintroduces the discontinuity being fixed |
| A trend exists in the target — the mean shifts monotonically or structurally over time, or a model's residuals show systematic bias that increases or decreases with time | A time-index or rolling-window statistic (e.g. rolling 7-day mean, days-since-event) | This is a different leakage concern from the group-aggregation row above — a rolling/lag feature is inherently row-local (each row's value depends only on that row's own prior history), so `_split` membership doesn't constrain it the way a cross-row aggregate does. The actual rule: never let a row's feature value depend on anything after that row's own timestamp (no negative shift, no centered windows). **After creating: apply the lag NaN sub-step** |
| A raw column bundles multiple distinct pieces of information into one field (e.g. a full address, a combined datetime string, a composite code) | Feature splitting — parse into separate columns (street/city/zip; year/month/day) | This is feature creation, not encoding, because it extracts genuinely separate pieces of information the raw column was compressing together |
| A feature has a demonstrably nonlinear relationship with the target and the model family cannot capture it directly (e.g. a linear model, a shallow tree with restricted depth) | Binning/discretization — equal-width or equal-frequency | Equal-width preserves absolute scale meaning; equal-frequency guarantees balanced bin sizes but loses that. Tree-based models rarely need this — they can already find thresholds directly |
| Very high dimensionality remains after individual feature decisions and is degrading model training (e.g. hundreds of one-hot-encoded columns, many correlated numeric features) | Dimensionality reduction (e.g. PCA) | This trades interpretability for compactness — components are combinations of original features, not something a human can read directly. Only worth it when dimensionality itself is the problem, not as a default step. **Fit PCA on train-split rows only** |
| A Diagnosis or Training step identifies that the model's residuals or errors show a pattern correlated with a specific feature or time window | Targeted new feature derived from that finding (crossing, lag, interaction, or split as appropriate) | This row exists specifically for Diagnosis- and Training-triggered FE calls. The feature must address the identified residual pattern directly — general "might help" additions do not qualify |

**Lag NaN sub-step (applies after every lag or rolling-window feature is created):**
Lag and rolling features introduce NaNs at the start of the series (rows where the lookback
window predates available data). These NaNs are produced by Feature Engineering, not by the
raw data — they are this step's responsibility to resolve before the output dataset is saved.
After creating each lag or rolling feature:
- Record the NaN rate on `_split == "train"` rows.
- Choose a fill strategy from the options below and document it explicitly:
  - **Drop rows** — valid only when the NaN rate is very low and the affected rows carry no
    unique information; document how many rows are dropped and from which period.
  - **Fill with a train-computed statistic** (median preferred for skewed distributions,
    mean otherwise) — fit the statistic on non-NaN train rows only, apply to all splits.
  - **Fill with a domain-appropriate constant** (e.g. 0 for a 'count of prior events'
    feature where no prior events is the correct interpretation).
  - **Propagate NaN** — only if the downstream model is confirmed to handle NaN natively
    (e.g. LightGBM, XGBoost with `missing` parameter); must be documented as a deliberate
    choice, not an oversight.
- Never leave lag/rolling NaNs unresolved — an unresolved NaN will silently drop rows or
  cause errors at model-fit time without any warning in the output.
