# AutoML baseline vs agentic pipeline

Run `run_20260904T140437Z` | target `bread_quantity` | scored on **3473 rows** (655 target dates x 6 horizons), 2019-01-03 to 2020-12-31.

Every system below was fitted on the same training partition, tuned on the same validation partition, and scored on the same held-out rows. No system saw a test row before it was scored.


## Headline

**B1 - AutoML on agent features wins.** MAE 59.659 against the agentic pipeline's 72.739 -- a gap of 13.080 (21.9%).


## What each system was given

| system | what it is | features | estimator |
| --- | --- | --- | --- |
| Agentic pipeline | the run's chosen model | 37 | LGBMRegressor |
| B1 - AutoML on agent features | identical features and rows; AutoML chose the estimator | 37 | lgbm |
| Seasonal naive | last observed same-weekday value at the origin | 1 | seasonal naive |


## Test-set metrics

| system | MAE | RMSE | R2 | SMAPE_pct | WAPE_pct | bias | MASE_train_denom | MASE_test_denom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agentic pipeline | 72.7388 | 106.289 | -0.3185 | 30.1611 | 26.2243 | 37.8188 | 2.0122 | 1.1452 |
| B1 - AutoML on agent features | 59.659 | 92.4577 | 0.0023 | 26.8136 | 21.5087 | 12.5187 | 1.6504 | 0.9393 |
| Seasonal naive | 63.5152 | 106.5906 | -0.326 | 35.0046 | 22.899 | 0.6326 | 1.757 | 1.0 |


MASE against the **train** denominator (36.149) is comparable with every figure the agentic run quotes. MASE against the **test** denominator (63.515) is the same-window verdict: below 1.0 means better than the seasonal naive on these rows.


## Skill against the seasonal naive

| system | MAE | skill vs naive (%) | horizons won (of 6) |
| --- | --- | --- | --- |
| Agentic pipeline | 72.7388 | -14.52 | 1 |
| B1 - AutoML on agent features | 59.659 | 6.07 | 4 |
| Seasonal naive | 63.5152 | 0.0 | 1 |


## By horizon band

| band | n | Agentic pipeline | B1 - AutoML on agent features | Seasonal naive |
| --- | --- | --- | --- | --- |
| h1-7 (within the week) | 1233 | 57.549 | 58.526 | 62.308 |
| h8-30 (within the month) | 1201 | 71.234 | 60.831 | 62.898 |
| h31-90 (within the quarter) | 1039 | 92.504 | 59.649 | 65.662 |


## By horizon

| h | n | actual mean | Agentic pipeline | B1 - AutoML on agent features | Seasonal naive |
| --- | --- | --- | --- | --- | --- |
| 1 | 583 | 279.08 | 53.16 | 61.421 | 61.429 |
| 7 | 650 | 278.52 | 61.485 | 55.929 | 63.096 |
| 14 | 643 | 279.02 | 61.152 | 55.238 | 62.778 |
| 30 | 558 | 280.72 | 82.852 | 67.276 | 63.036 |
| 60 | 532 | 282.66 | 97.005 | 63.658 | 64.769 |
| 90 | 507 | 262.61 | 87.781 | 55.443 | 66.599 |


## Across the test window

| year | n | actual mean | Agentic pipeline | B1 - AutoML on agent features | Seasonal naive |
| --- | --- | --- | --- | --- | --- |
| 2019 | 1517 | 282.69 | 61.977 | 50.423 | 47.909 |
| 2020 | 1956 | 273.24 | 81.086 | 66.822 | 75.619 |


## Protocol

- Rows: the 3473 (target date, horizon) pairs in the agentic run's test partition, checked pair-by-pair against every tier before scoring.
- Split: the agentic run's own chronological 70/15/15 by target date. Train ends 2017-07-29, validation ends 2019-01-02, test runs 2019-01-03 to 2020-12-31.
- Tuning: every AutoML tier selects hyperparameters on the run's validation partition, never on test, and never on a split of its own choosing.
- No recursion: each row predicts t+h in one jump from data at the origin t = t+h minus h days, so long horizons do not inherit accumulated one-step error.
- Trading days: the 2821 dates the agentic run kept. The 242 sub-10-unit non-trading days it filtered are excluded from every system, and unexported dates stay absent on the calendar so a lag of 7 still means seven days.
- Exogenous columns are lagged unless the run declared them knowable in advance (calendar terms, opening pattern, competitor closures). Weather is lagged for the AutoML tiers, where the agentic features used observed weather as a forecast proxy at h&nbsp;&le;&nbsp;7 - a small advantage to the agentic side on a feature the run measured at Spearman -0.076.
- B3 drops the run's must_drop_columns (same-day revenue, transaction and line counts). Lagging them instead would give the AutoML tier information the agentic model was denied, and the gap would stop meaning what it appears to mean.
- MASE denominators: train 36.149, test 63.515, both from the run's own seasonal-naive column.
- AutoML engine: FLAML, 120s per tier, optimising mae.
