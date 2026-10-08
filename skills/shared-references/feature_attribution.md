# Feature Attribution Under Correlated Features

Shared reference for `training-skill` (which produces feature importance) and `diagnosis-skill`
(which acts on it). Both must use the same rules, or Diagnosis will drop features on evidence
Training never actually established.

## 1. The problem this file exists to prevent

Feature importance is **not identifiable when features are correlated**. If two features carry the
same information, a tree splits arbitrarily between them, and any per-feature attribution splits
the credit arbitrarily too. Neither number is a fact about the world; both are facts about which
twin the fitting procedure happened to reach first.

This is not a subtlety. Two concrete failures it causes:

| Failure | What it looks like | Cost |
|---|---|---|
| **Masking** | Two redundant features each score near zero, because permuting one leaves the other to cover for it. Both get flagged `near_zero_flag` and Diagnosis drops both — losing signal that was genuinely there. | Silent accuracy loss, attributed to nothing |
| **False attribution** | The finer-grained twin absorbs the shared signal and ranks first. The report concludes "X drives demand" when X and Y are the same variable in two costumes. | A wrong causal story in the write-up |

**Worked example from `run_20260902T120000Z`.** `month_sin` and `doy_sin` correlate at
Spearman **ρ = +0.97** — the same annual position, at two resolutions. Measured on validation:

| permuted | ΔMAE |
|---|---|
| `month_sin` alone | **−0.02** (looks useless) |
| day-of-year pair alone | 5.09 |
| month pair + day-of-year pair together | **6.47** |

Month was carrying ~1.4 MAE the whole time. Alone it scores *negative* and would have been dropped
as noise. The signature is **joint > sum of individuals** — that inequality is what masking *is*.

## 2. Split-count importance is not importance — never report it as such

For tree ensembles, `feature_importances_` defaults to split count or impurity gain. Both are
substantially a measure of **cardinality**: a continuous feature offers thousands of usable
thresholds, a binary flag offers one, so the continuous feature accumulates splits regardless of
how much the answer depends on it.

From the same run, split-count rank against permutation rank:

| feature | splits | rank | perm ΔMAE | rank |
|---|---|---|---|---|
| `doy_sin` | 11,918 | 1 | 1.45 | 8 |
| `year_trend` | 3,810 | 3 | 0.09 | 16 |
| `dow_cos` | 834 | 18 | **5.55** | 2 |
| `covid_flag` | **26** | 25 | **2.00** | 5 |

`covid_flag` is last by split count and fifth by actual effect. A `near_zero_flag` assigned on
split count would have dropped the fifth most important feature in the model.

**Rule:** split-count importance may be recorded as a diagnostic of *how the tree was built*. It is
never the value in `feature_importance`, and never the basis for `near_zero_flag`.

## 3. The required procedure

### Step A — cluster features by rank correlation

On the **validation** split (never train — see §5):

```python
rho = spearmanr(X_val).statistic
rho = np.nan_to_num(rho, nan=0.0)          # constant-on-val features
d = 1 - np.abs(rho); np.fill_diagonal(d, 0); d = (d + d.T) / 2; d[d < 0] = 0
Z = linkage(squareform(d, checks=False), "average")
labels = fcluster(Z, 0.4, "distance")      # merge when |rho| >= 0.6
```

Merge at `|rho| >= 0.6` by default. Record the resulting clusters — they are an output, not an
intermediate: "these features are the same variable" is itself a finding for Diagnosis and for the
report.

Semantic groups may override the clustering where a group is obviously one concept measured several
ways (all lags and rolling means are "recent level" even if some pairs correlate below 0.6). State
which grouping was used and why.

### Step B — permute groups, not columns

Permute **every column in a cluster together, with the same row permutation**, and re-score on
validation. Report the MAE increase. Use >= 5 repeats and report the standard deviation; a group
whose effect is within ~2 sd of zero is not distinguishable from zero.

Score the **final prediction**, not an internal target. For a residual model the score is
`|y - clip(base + model.predict(X_perm), 0, None)|` — permuting a feature must not also change the
base the residual is added to.

### Step C — report the masking gap

For every group, record both the sum of its members' solo importances and the group's joint
importance. The difference is the credit no single feature can claim:

| group | Σ solo | grouped | masked |
|---|---|---|---|
| Recent level (10 lag/rolling) | 18.80 | 23.37 | **+4.57** |
| Annual seasonality | 4.78 | 6.39 | **+1.62** |
| Day of week | 8.83 | 9.24 | +0.41 |
| Weather (6) | −0.01 | −0.01 | 0.00 |
| **Total** | **34.70** | **40.96** | **+6.27** |

Per-feature accounting explained only **85%** of what the groups explain. That 15% is the number
this whole procedure exists to surface.

## 4. Reading the result

- **Group ΔMAE ≈ 0 *and* masked gap ≈ 0** → genuinely unused. Safe to propose dropping the whole
  group. In the run above, all six weather features scored −0.01 jointly, and killing calendar and
  weather together cost no more than killing calendar alone — so weather was not hiding behind
  season, it was simply unused. That is a legitimate `near_zero_flag`, at group level.
- **Solo ≈ 0 but group ΔMAE large** → masking. **Never drop these.** The individual number is an
  artifact of the twin covering for it.
- **A protected feature reading near zero** → report it, propose nothing. The per-horizon naive
  predictors (`../baseline-skill/baseline.md` Part B) are the worst case this whole file describes:
  each one is correlated with the target by construction *and* with every lag feature beside it by
  construction, so a solo reading is guaranteed to understate it and a group reading pools it with
  the very features masking it. There is no attribution measurement that settles the question here.
  Record the number, say which measurement produced it, and route the decision to the user as a
  blocking clarification — never to a drop.
- **Large masked gap within a group** → the group is over-parameterised. A reasonable hypothesis for
  Diagnosis is consolidating it (one lag feature instead of five), which usually costs little
  accuracy and buys interpretability.

## 5. Guardrails

- **Permutation importance is computed on validation, never on train.** On train it measures
  memorisation. This matches the existing rule in `training-skill` step 9 and is restated here
  because grouped permutation makes it easy to forget which split is in play.
- **`near_zero_flag` is a group-level verdict, never a column-level one.** A column may score zero
  because it is redundant; a group scoring zero is the only evidence that actually supports
  dropping. Diagnosis must not propose dropping a feature on its solo score alone.
- **SHAP does not solve this.** It has the same identifiability problem under correlation — it
  distributes credit smoothly between twins rather than arbitrarily, which reads better but is
  equally unidentifiable. If SHAP is used (see `training-skill/references/plotting_guide.md` §6),
  it is still grouped before it is reported. Grouping is the fix, whichever attribution method sits
  underneath.
- **A negative solo importance is noise, not evidence against a feature.** Report it as
  "indistinguishable from zero," not as a harmful feature.
- **Report groups as the headline; per-feature values only inside a group.** A per-feature ranking
  presented as the top-level answer is the artifact this file rejects.
