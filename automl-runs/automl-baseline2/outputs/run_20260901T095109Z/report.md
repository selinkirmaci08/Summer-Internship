# AutoML baseline vs agentic pipeline

Run `bench_run_20260901T095109Z` | target `bread_quantity` | scored on **5530 rows** (313 target dates x 23 horizons), 2021-07-02 to 2023-02-08.

Every system below was fitted on the same training partition, tuned on the same validation partition, and scored on the same held-out rows. No system saw a test row before it was scored.


## Headline

**Seasonal naive wins.** MAE 38.077 against the agentic pipeline's 42.873 -- a gap of 4.796 (12.6%).


## What each system was given

| system | what it is | features | estimator |
| --- | --- | --- | --- |
| Agentic pipeline | the run's chosen model | 32 | LGBMRegressor |
| B1 - AutoML on agent features | identical features and rows; AutoML chose the estimator | 32 | xgboost |
| B2 - AutoML on cleaned data | the agent's cleaned frame, generic lags and calendar terms | 122 | lgbm |
| B3 - AutoML on assembled data | the assembled frame, generic lags and calendar terms | 118 | xgboost |
| Seasonal naive | last observed same-weekday value at the origin | 1 | seasonal naive |


## Test-set metrics

| system | MAE | RMSE | R2 | SMAPE_pct | WAPE_pct | bias | MASE_train_denom | MASE_test_denom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agentic pipeline | 42.8729 | 50.1961 | -0.0595 | 16.9478 | 17.5304 | 38.8261 | 1.3489 | 1.1259 |
| B1 - AutoML on agent features | 42.7694 | 50.0337 | -0.0526 | 17.0576 | 17.488 | 38.8733 | 1.3456 | 1.1232 |
| B2 - AutoML on cleaned data | 52.0776 | 59.8118 | -0.5043 | 20.1981 | 21.2941 | 49.0145 | 1.6385 | 1.3677 |
| B3 - AutoML on assembled data | 50.222 | 57.3801 | -0.3844 | 19.4423 | 20.5354 | 47.7812 | 1.5801 | 1.319 |
| Seasonal naive | 38.0773 | 50.5152 | -0.073 | 15.1968 | 15.5695 | 12.5257 | 1.198 | 1.0 |


MASE against the **train** denominator (31.784) is comparable with every figure the agentic run quotes. MASE against the **test** denominator (38.077) is the same-window verdict: below 1.0 means better than the seasonal naive on these rows.


## Skill against the seasonal naive

| system | MAE | skill vs naive (%) | horizons won (of 23) |
| --- | --- | --- | --- |
| Agentic pipeline | 42.8729 | -12.59 | 1 |
| B1 - AutoML on agent features | 42.7694 | -12.32 | 0 |
| B2 - AutoML on cleaned data | 52.0776 | -36.77 | 2 |
| B3 - AutoML on assembled data | 50.222 | -31.89 | 0 |
| Seasonal naive | 38.0773 | 0.0 | 20 |


## By horizon band

| band | n | Agentic pipeline | B1 - AutoML on agent features | B2 - AutoML on cleaned data | B3 - AutoML on assembled data | Seasonal naive |
| --- | --- | --- | --- | --- | --- | --- |
| h1-7 (within the week) | 1687 | 39.221 | 39.268 | 53.378 | 49.911 | 28.183 |
| h8-30 (within the month) | 942 | 39.101 | 38.849 | 52.453 | 49.007 | 33.899 |
| h31-90 (within the quarter) | 1115 | 40.939 | 40.279 | 50.699 | 48.451 | 35.644 |
| h91+ (beyond the quarter) | 1786 | 49.519 | 49.699 | 51.512 | 52.263 | 51.146 |


## By horizon

| h | n | actual mean | Agentic pipeline | B1 - AutoML on agent features | B2 - AutoML on cleaned data | B3 - AutoML on assembled data | Seasonal naive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 241 | 243.72 | 39.221 | 39.265 | 53.441 | 49.911 | 28.183 |
| 2 | 241 | 243.72 | 39.221 | 39.264 | 53.413 | 49.911 | 28.183 |
| 3 | 241 | 243.72 | 39.221 | 39.264 | 53.408 | 49.911 | 28.183 |
| 4 | 241 | 243.72 | 39.221 | 39.273 | 53.378 | 49.911 | 28.183 |
| 5 | 241 | 243.72 | 39.221 | 39.271 | 53.361 | 49.911 | 28.183 |
| 6 | 241 | 243.72 | 39.221 | 39.271 | 53.329 | 49.911 | 28.183 |
| 7 | 241 | 243.72 | 39.221 | 39.271 | 53.317 | 49.911 | 28.183 |
| 10 | 240 | 245.94 | 38.681 | 38.522 | 52.22 | 49.013 | 32.967 |
| 14 | 240 | 245.94 | 38.681 | 38.564 | 52.178 | 49.013 | 32.967 |
| 21 | 237 | 244.87 | 39.388 | 39.31 | 52.957 | 48.927 | 34.97 |
| 28 | 225 | 245.48 | 39.697 | 39.015 | 52.463 | 49.078 | 34.76 |
| 35 | 222 | 246.09 | 39.763 | 38.484 | 51.092 | 47.973 | 34.27 |
| 42 | 219 | 245.84 | 40.06 | 39.369 | 50.669 | 48.203 | 35.918 |
| 56 | 223 | 244.93 | 41.305 | 40.528 | 51.045 | 48.62 | 34.359 |
| 70 | 224 | 246.92 | 41.591 | 41.298 | 50.277 | 48.832 | 35.241 |
| 84 | 227 | 246.4 | 41.932 | 41.662 | 50.421 | 48.613 | 38.383 |
| 91 | 234 | 245.47 | 41.714 | 42.112 | 50.128 | 49.294 | 37.744 |
| 120 | 237 | 244.58 | 45.281 | 44.647 | 50.524 | 50.431 | 42.987 |
| 150 | 247 | 245.34 | 46.011 | 45.403 | 49.578 | 50.601 | 44.077 |
| 182 | 248 | 243.94 | 49.401 | 49.574 | 50.622 | 51.835 | 48.266 |
| 240 | 258 | 242.58 | 54.297 | 53.835 | 51.817 | 54.382 | 59.609 |
| 300 | 276 | 242.76 | 54.171 | 55.415 | 52.911 | 54.38 | 61.243 |
| 364 | 286 | 243.16 | 53.752 | 54.663 | 54.278 | 54.062 | 60.099 |


## Across the test window

| year | n | actual mean | Agentic pipeline | B1 - AutoML on agent features | B2 - AutoML on cleaned data | B3 - AutoML on assembled data | Seasonal naive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | 4152 | 246.05 | 45.023 | 42.958 | 50.243 | 49.247 | 39.271 |
| 2022 | 923 | 240.13 | 35.713 | 42.28 | 54.693 | 51.324 | 35.889 |
| 2023 | 455 | 239.97 | 37.776 | 42.04 | 63.509 | 56.883 | 31.622 |


## Protocol

- Rows: the 7199 (target date, horizon) pairs in the agentic run's test partition, checked pair-by-pair against every tier before scoring.
- Split: the agentic run's own chronological 70/15/15 by target date. Train ends 2020-08-20, validation ends 2021-07-01, test runs 2021-07-02 to 2023-02-08.
- Tuning: every AutoML tier selects hyperparameters on the run's validation partition, never on test, and never on a split of its own choosing.
- No recursion: each row predicts t+h in one jump from data at the origin t = t+h minus h days, so long horizons do not inherit accumulated one-step error.
- Trading days: the 1960 dates the agentic run kept. The 242 sub-10-unit non-trading days it filtered are excluded from every system, and unexported dates stay absent on the calendar so a lag of 7 still means seven days.
- Exogenous columns are lagged unless the run declared them knowable in advance (calendar terms, opening pattern, competitor closures). Weather is lagged for the AutoML tiers, where the agentic features used observed weather as a forecast proxy at h&nbsp;&le;&nbsp;7 - a small advantage to the agentic side on a feature the run measured at Spearman -0.076.
- B3 drops the run's must_drop_columns (same-day revenue, transaction and line counts). Lagging them instead would give the AutoML tier information the agentic model was denied, and the gap would stop meaning what it appears to mean.
- MASE denominators: train 31.784, test 38.077, both from the run's own seasonal-naive column.
- AutoML engine: FLAML, 900s per tier, optimising mae.
