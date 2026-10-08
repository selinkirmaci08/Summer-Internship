# Every run against one AutoML baseline, on one set of rows

Horizon 1, **270 dates**, 2021-07-02 to 2023-02-08. Every system below predicts the same 270 days and is scored against the same truth.


## What could be compared

| run | feature table | models | in the comparison |
| --- | --- | --- | --- |
| run_20260811T141909Z | -- | 0 | no model produced |
| run_20260811T144500Z | -- | 0 | no model produced |
| run_20260811T151500Z | -- | 0 | no model produced |
| run_20260811T160000Z | -- | 0 | no model produced |
| run_20260814T082956Z | -- | 0 | no model produced |
| run_20260814T131044Z | -- | 0 | no model produced |
| run_20260814T140822Z | -- | 0 | no model produced |
| run_20260818T000000Z | -- | 0 | no model produced |
| run_20260818T082437Z | -- | 0 | no model produced |
| run_20260818T121822Z | -- | 0 | no model produced |
| run_20260819T090000Z | iteration_005_feature_engineering.csv | 1 | yes |
| run_20260820T090000Z | -- | 0 | no model produced |
| run_20260820T110000Z | iteration_003_feature_engineering.csv | 1 | yes |
| run_20260821T090000Z | -- | 0 | no model produced |
| run_20260821T110000Z | -- | 0 | no model produced |
| run_20260821T130000Z | iteration_013_feature_engineering.csv | 5 | excluded -- see below |
| run_20260821T140000Z | iteration_004_feature_engineering.csv | 2 | yes |
| run_20260827T154407Z | iteration_008_feature_engineering.csv | 5 | yes |
| run_20260901T095109Z | iteration_010_feature_engineering.csv | 4 | yes |

6 of 19 runs reached a trained model. 5 of those are scored here.


## Results

| system | MAE | RMSE | MASE | WAPE % | bias | vs AutoML | vs naive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seasonal_naive_h1 | 32.963 | 46.7141 | 1.0 | 13.3606 | -2.0222 | +18.92% | +0.00% |
| run_20260901T095109Z | 36.6909 | 44.166 | 1.1131 | 14.8716 | 31.627 | +9.75% | -11.31% |
| automl_common_h1 | 40.6529 | 51.9936 | 1.2333 | 16.4774 | 35.2733 | +0.00% | -23.33% |
| run_20260820T110000Z | 45.7805 | 54.398 | 1.3888 | 18.5558 | 42.4063 | -12.61% | -38.88% |
| run_20260827T154407Z | 48.667 | 56.5188 | 1.4764 | 19.7257 | 46.594 | -19.71% | -47.64% |
| run_20260821T140000Z | 52.2641 | 66.2588 | 1.5855 | 21.1837 | 46.4106 | -28.56% | -58.55% |
| run_20260819T090000Z | 77.2443 | 84.3839 | 2.3434 | 31.3087 | 77.1707 | -90.01% | -134.34% |

Positive skill means better than that reference. The AutoML row is the baseline: FLAML on target lags, rolling windows and calendar only, trained on every day through 2021-07-01 and never on a scored day.


## Why no model was scored on a row it had seen

| run | trained through | tuned through | days before the first scored date |
| --- | --- | --- | --- |
| run_20260819T090000Z | 2019-06-23 | 2020-12-09 | 205 |
| run_20260820T110000Z | 2019-06-22 | 2020-12-09 | 205 |
| run_20260821T140000Z | 2019-04-03 | 2020-11-14 | 230 |
| run_20260827T154407Z | 2019-05-30 | 2020-12-12 | 202 |
| run_20260901T095109Z | 2020-08-20 | 2021-07-01 | 1 |

The frozen index starts the day after the latest validation window of any run in it, and every scored date was asserted to be inside every run's own test partition before anything was predicted.


Dates removed from the window:

| run | reason | dates |
| --- | --- | --- |
| run_20260820T110000Z | duplicate rows for the same date | 2 |
| all | runs disagree on the true value | 1 |

## Runs that could not be ranked


**run_20260821T130000Z** -- split is not chronological (train 2014-06-21..2023-02-08, validation 2015-04-28..2021-12-23, test 2015-05-29..2022-03-01) -- training rows sit inside and after the test window, so any score it produces is contaminated

| scored dates | of which it trained on | MAE (all) | MAE where it trained | MAE where it did not |
| --- | --- | --- | --- | --- |
| 270 | 221 | 21.4877 | 20.5281 | 25.8158 |

The gap between the last two columns is the size of the contamination, not a modelling result.

