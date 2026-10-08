# Model-Specific Data-Readiness Requirements

Owned jointly by **Preprocess**, **Feature Engineering** and **Training**. Preprocess and FE
decide what to do to a column; Training decides which model family to fit. Those two decisions
constrain each other, and this file is where the constraint is written down once instead of
being re-derived in three places.

> **Provenance note.** This file was cited by `training.md` (§§1–2 and §6) and by four of its
> reference files before it existed. It was written to close that gap, not extracted from an
> existing document — so unlike the other reference files, its content has not been through a
> run. Treat the tables as standard practice rather than as this project's measured experience,
> and correct them when a run shows otherwise.

The other direction of this relationship already exists and is not repeated here: the
distribution-keyed scaling and cardinality-keyed encoding tables live in
`../preprocess-skill/references/transformation_catalog.md` (step 5, sub-decisions d and e).
Those answer "what does this *column* need?" This file answers "what does this *model* require?"
Both have to be satisfied.

---

## 1. Scaling requirements by model family

| Model family | Scaling required? | Why |
|---|---|---|
| Decision tree, random forest, extra trees | **No** | Splits are on thresholds within a single feature; a monotonic rescaling changes nothing about which split is chosen |
| Gradient boosted trees (LightGBM, XGBoost, HistGB, CatBoost) | **No** | Same reason. Scaling a tree model's inputs is wasted effort, not a safety measure |
| Linear/logistic regression, unregularised | Not for correctness | Coefficients absorb the scale. Helps numerical conditioning on badly scaled data |
| Ridge, Lasso, ElasticNet | **Yes** | The penalty is applied to coefficient magnitude, so an unscaled feature is penalised in proportion to its units. Skipping this silently changes which features get shrunk |
| SVM (RBF, polynomial) | **Yes** | The kernel is a distance, and distance is dominated by whichever feature has the widest range |
| KNN, k-means, DBSCAN | **Yes** | Same distance argument |
| Neural networks | **Yes** | Unscaled inputs produce badly conditioned gradients and slow or unstable convergence |
| Gaussian Naive Bayes | No | Each feature's likelihood is estimated in its own units |
| PCA and any variance-based decomposition | **Yes** | Components follow whichever feature has the largest variance, which is a units artefact if unscaled |

The rule that catches most mistakes: **if the model computes a distance, an inner product, or a
penalty on coefficient magnitude, it needs scaling. If it computes a threshold, it does not.**

## 2. Encoding requirements by model family

| Model family | Acceptable encodings | Avoid |
|---|---|---|
| Tree ensembles | Ordinal/label codes are fine — a tree can isolate any category with enough splits. One-hot is also fine but costs depth on high-cardinality columns | One-hot on a high-cardinality column: each dummy is a weak split and the tree spends depth recovering what one column held |
| LightGBM | Native categorical support (`categorical_feature`) — usually better than either alternative | Pre-one-hotting a column LightGBM could have handled natively |
| CatBoost | Native ordered target statistics, built to avoid the leakage plain target encoding causes | Hand-rolled target encoding on top of it |
| Linear, logistic, SVM, KNN | One-hot, or a learned embedding | **Ordinal codes** — they assert an order and a spacing that usually do not exist, and the model will read `3` as three times `1` |
| Neural networks | Embedding layers for high cardinality; one-hot for low | Ordinal codes, same reason |

**Ordinal encoding is safe for trees and unsafe for everything else.** That single asymmetry is
the most common encoding error, because the same column encoded the same way is correct for one
model and wrong for the next.

## 3. Target transformations

A skewed *target* is a different problem from a skewed *feature*, and a different problem again
from a target whose **level drifts**. All three get reached for with the same tool, which is why
they get confused:

| Problem | Symptom | Treatment |
|---|---|---|
| Right-skewed positive target | A few large values dominate the loss; residuals fan out with the fitted value | `log1p`, with the inverse applied before any metric is reported |
| Heteroscedastic residuals | Error variance grows with the level | `log1p` or a variance-stabilising transform |
| Target level drifts outside the training range | Large one-sided bias; a tree model's predictions clipped at the edge of what it saw | **Not a transform.** Model the target relative to a past-only anchor — see `../featureengineering-skill/featureengineering.md` step 2b |

The third row is the one worth reading twice. `log1p` moves the bound into log space; it does not
remove it. A tree fitted on a log target still cannot predict below its training minimum once the
prediction is transformed back.

Whatever is applied, the inverse must travel with the model artifact. A pickled estimator whose
output is in log space or in ratio space, with the inverse recorded nowhere, returns plausible
small numbers and raises no error — the failure is silent by construction.

## 4. Missing-value handling by model family

| Model family | Handles NaN natively? | What to do |
|---|---|---|
| LightGBM | **Yes** | Learns a default direction per split. Imputing first *removes* information — the fact a value was missing is often predictive |
| XGBoost | **Yes** (default direction per split) | `training.md` step 2b flags an open question about the exact behaviour across versions and objectives. Treat native handling as working, but verify before relying on it for a load-bearing feature |
| CatBoost | **Yes** | Native |
| sklearn `HistGradientBoosting*` | **Yes** | Native, same mechanism as LightGBM |
| sklearn `DecisionTree*`, `RandomForest*`, `ExtraTrees*` | **No** | Raises on NaN. Requires imputation upstream |
| Linear, logistic, Ridge, Lasso, SVM, KNN | **No** | Requires imputation |
| Neural networks | **No** | Requires imputation |

**A model that handles NaN natively is a reason not to impute, not merely permission to skip
it.** Where missingness is informative — MNAR, in EDA's classification — imputing a "normal"
value destroys the signal that the value was absent. Preprocess's missingness treatment should
be chosen knowing which model family will consume the column, which is the whole reason this
file is shared rather than owned by one node.

## 5. Where each node's responsibility ends

| Question | Whose call |
|---|---|
| Does this column need scaling/encoding/imputation, and which method? | Preprocess |
| Does this column earn a place in the model at all? | Feature Engineering |
| What is the target's representation, and what anchor does it need? | Feature Engineering (step 2b) — the anchor is a column, and columns are built there |
| Which model family, given what the columns now look like? | Training |
| The column is not ready for the chosen family | Training records an `issues` entry and requests a Preprocess fix — **it never patches this inside the training loop**, because a statistic fit there is fit on the wrong rows |
