# AutoML baseline vs agentic pipeline

Run `run_20260903T152603Z` | target `bread_quantity` | scored on **2034 rows** (345 target dates x 6 horizons), 2020-01-01 to 2020-12-31.

Every system below was fitted on the same training partition, tuned on the same validation partition, and scored on the same held-out rows. No system saw a test row before it was scored.


## Headline

**B1 - AutoML on agent features wins.** MAE 37.628 against the agentic pipeline's 40.743 -- a gap of 3.115 (8.3%).


## What each system was given

| system | what it is | features | estimator |
| --- | --- | --- | --- |
| Agentic pipeline | the run's chosen model | 18 | LGBMRegressor |
| B1 - AutoML on agent features | identical features and rows; AutoML chose the estimator | 18 | extra_tree |
| Seasonal naive | last observed same-weekday value at the origin | 1 | seasonal naive |


## Test-set metrics

| system | MAE | RMSE | R2 | SMAPE_pct | WAPE_pct | bias | MASE_train_denom | MASE_test_denom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agentic pipeline | 40.7426 | 54.4367 | 0.2449 | 14.9594 | 13.7329 | -5.1377 | 0.6268 | 0.631 |
| B1 - AutoML on agent features | 37.6278 | 52.5242 | 0.297 | 13.783 | 12.683 | 0.1068 | 0.5789 | 0.5828 |
| Seasonal naive | 64.5654 | 85.0213 | -0.842 | 22.8932 | 21.7626 | 1.3029 | 0.9933 | 1.0 |


MASE against the **train** denominator (65.003) is comparable with every figure the agentic run quotes. MASE against the **test** denominator (64.565) is the same-window verdict: below 1.0 means better than the seasonal naive on these rows.


## Skill against the seasonal naive

| system | MAE | skill vs naive (%) | horizons won (of 6) |
| --- | --- | --- | --- |
| Agentic pipeline | 40.7426 | 36.9 | 0 |
| B1 - AutoML on agent features | 37.6278 | 41.72 | 6 |
| Seasonal naive | 64.5654 | 0.0 | 0 |


## By horizon band

| band | n | Agentic pipeline | B1 - AutoML on agent features | Seasonal naive |
| --- | --- | --- | --- | --- |
| h1-7 (within the week) | 686 | 40.683 | 37.705 | 57.636 |
| h8-30 (within the month) | 682 | 40.367 | 37.29 | 64.513 |
| h31-90 (within the quarter) | 666 | 41.189 | 37.894 | 71.757 |


## By horizon

| h | n | actual mean | Agentic pipeline | B1 - AutoML on agent features | Seasonal naive |
| --- | --- | --- | --- | --- | --- |
| 1 | 342 | 296.79 | 40.695 | 38.006 | 68.544 |
| 7 | 344 | 297.17 | 40.671 | 37.406 | 46.791 |
| 14 | 343 | 297.56 | 40.542 | 37.438 | 49.563 |
| 30 | 339 | 297.48 | 40.19 | 37.14 | 79.64 |
| 60 | 335 | 298.14 | 41.13 | 38.041 | 81.313 |
| 90 | 331 | 292.84 | 41.249 | 37.746 | 62.085 |


## Across the test window

| year | n | actual mean | Agentic pipeline | B1 - AutoML on agent features | Seasonal naive |
| --- | --- | --- | --- | --- | --- |
| 2020 | 2034 | 296.68 | 40.743 | 37.628 | 64.565 |


## Protocol

- Rows: the 2034 (target date, horizon) pairs in the agentic run's test partition, checked pair-by-pair against every tier before scoring.
- Split: the agentic run's own chronological 70/15/15 by target date. Train ends 2018-12-31, validation ends 2019-12-31, test runs 2020-01-01 to 2020-12-31.
- Tuning: every AutoML tier selects hyperparameters on the run's validation partition, never on test, and never on a split of its own choosing.
- No recursion: each row predicts t+h in one jump from data at the origin t = t+h minus h days, so long horizons do not inherit accumulated one-step error.
- Trading days: the 2920 dates the agentic run kept. The 242 sub-10-unit non-trading days it filtered are excluded from every system, and unexported dates stay absent on the calendar so a lag of 7 still means seven days.
- Exogenous columns are lagged unless the run declared them knowable in advance (calendar terms, opening pattern, competitor closures). Weather is lagged for the AutoML tiers, where the agentic features used observed weather as a forecast proxy at h&nbsp;&le;&nbsp;7 - a small advantage to the agentic side on a feature the run measured at Spearman -0.076.
- B3 drops the run's must_drop_columns (same-day revenue, transaction and line counts). Lagging them instead would give the AutoML tier information the agentic model was denied, and the gap would stop meaning what it appears to mean.
- MASE denominators: train 65.003, test 64.565, both from the run's own seasonal-naive column.
- AutoML engine: FLAML, 180s per tier, optimising mae.
