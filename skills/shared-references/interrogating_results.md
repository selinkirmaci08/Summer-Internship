# Interrogating results

Five checks that ask whether a number measures what its name says, before anyone reasons about
what it means. They are cheap — minutes, not iterations — and each one is a failure this project
actually shipped and reported as a success.

A wrong interpretation of a correct number gets caught later, by the next iteration or the holdout.
A correct interpretation of a number that measures something else does not get caught at all: it is
real arithmetic, it is internally consistent, and every artifact downstream repeats it. That is why
these run first.

**Run them when a number first appears, not when it looks wrong.** All five failures below looked
entirely reasonable.

---

## 1. Units — is this index rows, or time?

`series.shift(k)` moves `k` **rows**. It moves `k` days only if every calendar day is a row. The
moment a frame filters non-trading days, drops closures, or has missing exports, rows and time come
apart, and every lag, rolling window and horizon built on `shift` silently changes meaning.

**Check:** for one lag and one horizon, print the actual calendar gap between a row and the row `k`
positions away — its median and its range. If the median is not `k`, the feature is not what its
name says.

> **What happened here.** EDA measured autocorrelation on a frame with pre-2020 Mondays deleted, so
> a week was six rows. It read `lag-30` at `r = 0.601` and reported a **monthly** signal. On the
> calendar, lag-30-days is `r = 0.022`; 30 rows was five weeks, i.e. the weekly signal wearing a
> monthly label. Four of the seven lag features it then built (`bq_lag14`, `bq_lag28`, `bq_lag91`,
> `bq_lag364`) point at weekday-misaligned offsets and measure noise.

## 2. Regime — does the relationship survive the break the run already found?

If the run identified a structural break, check every learned relationship across it, not just the
level. A relationship defined by row position is especially exposed: when the calendar spacing of
rows changes at the break, what the model learned changes with it.

**Check:** for the relationship in question, compute it separately on each side of the break. If the
two sides disagree, and the training split sits entirely on one side, the model has learned
something that is false where it will be deployed.

> **What happened here.** Targets were `shift(-h)` rows. The h=30 model learned "my target is the
> **same weekday** as my origin" from 89.4% of its training rows (30 rows = 5 six-day weeks). The
> training split ends 2018-10-04, so **100%** of its examples were pre-regime. After the shop went
> 7/7, 30 days later is **+2 weekdays**. The single relationship it learned is wrong on every
> holdout row, and h=30 is its worst horizon (MAE 73.32 against 41.61 at h=1, the one horizon whose
> weekday offset survived).

## 3. Rows — are the two sides of the comparison on the same rows?

A margin between two systems scored on different row sets is not a margin. Report `n` per system
per segment, and assert equality before computing any difference.

> **What happened here.** The run's champion was scored on n = 656/650/643/627/597/567 across its
> horizons; its baseline floor on n = 582/650/643/553/523/494. Only two of six horizons match. The
> reported "beats baseline by 23–35 MAE" was arithmetic across two different row sets.

## 4. Floor — is the thing being beaten actually trying?

A floor sets what "good" means for the whole run. Score more than one trivial rule and take the
strongest — see `../baseline-skill/baseline.md`, which exists because of this failure.

> **What happened here.** The floor was one rule that indexed a fixed calendar date and scored
> public-holiday closures as sales figures: MAE 85.31 pooled. A correct rule on the identical
> window scores 44.14. The champion was reported at **+35.5%** against the floor; against a working
> floor the same champion is **−24.5%**. It was losing for the whole run.

## 5. Justification — is the feature you argued for actually being used?

Every feature was added for a stated reason. Grouped permutation importance says whether that reason
held (`feature_attribution.md` for the grouping rules — never read solo values alone). A feature
group whose importance is indistinguishable from zero did not do what it was added to do, and that
is a result worth reporting, not an embarrassment to leave out.

**Check:** after any Training call, compare the feature groups' measured importance against the
rationale recorded when they were created. Name every group where the two disagree.

> **What happened here.** Grouped permutation importance on validation: Fourier calendar terms
> `+2.46` MAE — the only group carrying the model. The seven lag features, the ones EDA justified as
> the core signal: `−1.44`, i.e. permuting them **improves** accuracy. Weather (12 features) and
> competitor (7) are indistinguishable from zero. `year_centered`, the only trend term, is exactly
> `0.00` — never split on once. That last one explains the run's whole holdout failure: a model with
> no usable trend term cannot follow a drifting series, and its bias grew +32 → +54 → +60 across
> 2021–2023 while a naive's stayed flat.

---

## What to do with a failed check

Record it as a finding with the number that failed and the number it should have been, and route it
to the node that owns the definition — EDA for a units or seasonality question, Preprocess for a
split or filter, Feature Engineering for a feature's rationale, Training for a floor or a scoring
row set. A failed check is not a modelling hypothesis and does not compete with one for ranking:
it invalidates evidence that other hypotheses were resting on, so it is fixed first.
