# AutoML baseline vs agentic pipeline

Run `run_20260827T154407Z` | target `bread_quantity` | scored on **10752 rows** (512 target dates x 21 horizons), 2020-12-13 to 2023-02-08.

Every system below was fitted on the same training partition, tuned on the same validation partition, and scored on the same held-out rows. No system saw a test row before it was scored.


## Headline

**The agentic pipeline wins.** MAE 43.509 against 43.705 for the closest AutoML tier (B1 - AutoML on agent features).


## What each system was given

| system | what it is | features | estimator |
| --- | --- | --- | --- |
| Agentic pipeline | the run's chosen model | 56 | LGBMRegressor |
| B1 - AutoML on agent features | identical features and rows; AutoML chose the estimator | 56 | lgbm |
| B2 - AutoML on cleaned data | the agent's cleaned frame, generic lags and calendar terms | 156 | lgbm |
| B3 - AutoML on assembled data | the assembled frame, generic lags and calendar terms | 114 | lgbm |
| Seasonal naive | last observed same-weekday value at the origin | 1 | seasonal naive |


## Test-set metrics

| system | MAE | RMSE | R2 | SMAPE_pct | WAPE_pct | bias | MASE_train_denom | MASE_test_denom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agentic pipeline | 43.5093 | 52.9273 | 0.1372 | 16.4669 | 16.6249 | 34.0214 | 1.1478 | 0.9632 |
| B1 - AutoML on agent features | 43.7045 | 53.1726 | 0.1292 | 16.5925 | 16.6995 | 34.6978 | 1.153 | 0.9675 |
| B2 - AutoML on cleaned data | 47.565 | 57.7897 | -0.0286 | 17.7721 | 18.1746 | 34.8892 | 1.2548 | 1.053 |
| B3 - AutoML on assembled data | 46.2443 | 56.4257 | 0.0194 | 17.3022 | 17.67 | 33.3842 | 1.22 | 1.0237 |
| Seasonal naive | 45.1729 | 58.6646 | -0.06 | 17.0124 | 17.2606 | 20.0136 | 1.1917 | 1.0 |


MASE against the **train** denominator (37.906) is comparable with every figure the agentic run quotes. MASE against the **test** denominator (45.173) is the same-window verdict: below 1.0 means better than the seasonal naive on these rows.


## Skill against the seasonal naive

| system | MAE | skill vs naive (%) | horizons won (of 21) |
| --- | --- | --- | --- |
| Agentic pipeline | 43.5093 | 3.68 | 6 |
| B1 - AutoML on agent features | 43.7045 | 3.25 | 6 |
| B2 - AutoML on cleaned data | 47.565 | -5.3 | 0 |
| B3 - AutoML on assembled data | 46.2443 | -2.37 | 2 |
| Seasonal naive | 45.1729 | 0.0 | 7 |


## By horizon band

| band | n | Agentic pipeline | B1 - AutoML on agent features | B2 - AutoML on cleaned data | B3 - AutoML on assembled data | Seasonal naive |
| --- | --- | --- | --- | --- | --- | --- |
| h1-7 (within the week) | 3584 | 41.655 | 41.996 | 47.19 | 46.026 | 39.797 |
| h8-30 (within the month) | 2048 | 43.016 | 42.962 | 47.389 | 46.091 | 44.79 |
| h31-90 (within the quarter) | 3072 | 43.874 | 43.756 | 47.61 | 46.073 | 46.577 |
| h91+ (beyond the quarter) | 2048 | 46.699 | 47.36 | 48.328 | 47.037 | 52.857 |


## By horizon

| h | n | actual mean | Agentic pipeline | B1 - AutoML on agent features | B2 - AutoML on cleaned data | B3 - AutoML on assembled data | Seasonal naive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 261.71 | 41.675 | 41.922 | 47.195 | 45.916 | 39.797 |
| 2 | 512 | 261.71 | 41.643 | 41.997 | 47.215 | 46.004 | 39.797 |
| 3 | 512 | 261.71 | 41.681 | 42.005 | 47.19 | 45.998 | 39.797 |
| 4 | 512 | 261.71 | 41.683 | 41.992 | 47.173 | 46.035 | 39.797 |
| 5 | 512 | 261.71 | 41.688 | 42.02 | 47.245 | 46.091 | 39.797 |
| 6 | 512 | 261.71 | 41.604 | 42.047 | 47.146 | 46.057 | 39.797 |
| 7 | 512 | 261.71 | 41.613 | 41.992 | 47.169 | 46.081 | 39.797 |
| 10 | 512 | 261.71 | 42.422 | 42.654 | 47.245 | 46.122 | 43.766 |
| 14 | 512 | 261.71 | 42.483 | 42.6 | 47.329 | 46.054 | 43.766 |
| 21 | 512 | 261.71 | 43.472 | 43.215 | 47.513 | 46.143 | 45.469 |
| 28 | 512 | 261.71 | 43.687 | 43.381 | 47.47 | 46.047 | 46.158 |
| 35 | 512 | 261.71 | 43.835 | 43.482 | 47.617 | 46.084 | 46.604 |
| 42 | 512 | 261.71 | 43.835 | 43.741 | 47.44 | 45.893 | 47.064 |
| 56 | 512 | 261.71 | 43.147 | 43.4 | 47.406 | 45.97 | 46.686 |
| 70 | 512 | 261.71 | 43.684 | 43.753 | 47.599 | 46.022 | 46.719 |
| 84 | 512 | 261.71 | 44.345 | 44.012 | 47.825 | 46.149 | 46.896 |
| 90 | 512 | 261.71 | 44.4 | 44.145 | 47.775 | 46.321 | 45.496 |
| 120 | 512 | 261.71 | 45.485 | 45.618 | 48.119 | 46.649 | 50.015 |
| 180 | 512 | 261.71 | 46.841 | 47.153 | 48.499 | 46.808 | 51.987 |
| 270 | 512 | 261.71 | 47.121 | 48.052 | 48.379 | 47.073 | 53.466 |
| 364 | 512 | 261.71 | 47.35 | 48.616 | 48.315 | 47.616 | 55.961 |


## Across the test window

| year | n | actual mean | Agentic pipeline | B1 - AutoML on agent features | B2 - AutoML on cleaned data | B3 - AutoML on assembled data | Seasonal naive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 399 | 290.47 | 34.898 | 34.008 | 45.349 | 40.078 | 44.107 |
| 2021 | 7602 | 268.81 | 39.68 | 40.122 | 41.802 | 40.766 | 39.042 |
| 2022 | 1932 | 238.0 | 52.356 | 52.718 | 60.416 | 59.608 | 63.654 |
| 2023 | 819 | 237.72 | 62.382 | 60.425 | 71.825 | 68.57 | 59.004 |


## Protocol

- Rows: the 10752 (target date, horizon) pairs in the agentic run's test partition, checked pair-by-pair against every tier before scoring.
- Split: the agentic run's own chronological 70/15/15 by target date. Train ends 2019-05-30, validation ends 2020-12-12, test runs 2020-12-13 to 2023-02-08.
- Tuning: every AutoML tier selects hyperparameters on the run's validation partition, never on test, and never on a split of its own choosing.
- No recursion: each row predicts t+h in one jump from data at the origin t = t+h minus h days, so long horizons do not inherit accumulated one-step error.
- Trading days: the 3413 dates the agentic run kept. The 242 sub-10-unit non-trading days it filtered are excluded from every system, and unexported dates stay absent on the calendar so a lag of 7 still means seven days.
- Exogenous columns are lagged unless the run declared them knowable in advance (calendar terms, opening pattern, competitor closures). Weather is lagged for the AutoML tiers, where the agentic features used observed weather as a forecast proxy at h&nbsp;&le;&nbsp;7 - a small advantage to the agentic side on a feature the run measured at Spearman -0.076.
- B3 drops the run's must_drop_columns (same-day revenue, transaction and line counts). Lagging them instead would give the AutoML tier information the agentic model was denied, and the gap would stop meaning what it appears to mean.
- MASE denominators: train 37.906, test 45.173, both from the run's own seasonal-naive column.
- AutoML engine: FLAML, 900s per tier, optimising mae.
