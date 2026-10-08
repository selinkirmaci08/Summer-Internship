# Iteration 6 — Training: LightGBM (initial, ml_as_regression)
Run: run_20260908T075709Z | Date: 2026-09-08

## Mode and trigger
Mode: `initial` — first learned model this run. Follows baseline (iter 4) and feature engineering (iter 5).
ts_approach: `ml_as_regression`. **Gap: `classical_statistical` (SARIMA) must be fitted next.**

## Feature subset and deployment scenario
Deployment scenario: `live_forecast`. One feature subset per horizon from `horizon_feature_subsets`.
Weather (is_hot_day, is_heavy_rain) included for h≤7 only (7-day forecast horizon).
Lag features: causal DOW-aligned lag (bread_lag_{ceil(h/7)*7}d).

## Model: LightGBMRegressor
Hyperparameters: defaults (n_estimators=500, max_depth=5, lr=0.05, min_child_samples=20, L1/L2=0.1).
Objective: regression_l1 (MAE-aligned with primary metric). Early stopping: 50 rounds.
**Note:** Hyperparameter search deferred. Training.md guardrail requires at least one search before run close.

## Results: Validation MAE by horizon

| h | Val MAE | Floor | Margin | Beat floor | Overfit | Range |
|---|---------|-------|--------|------------|---------|-------|
| 1 | 49.197 | 31.263 | -17.934 | NO | severe_overfit | range_warning |
| 7 | 72.056 | 31.263 | -40.793 | NO | severe_overfit | range_failure |
| 14 | 74.038 | 32.846 | -41.192 | NO | severe_overfit | range_failure |
| 30 | 78.972 | 35.913 | -43.059 | NO | severe_overfit | range_failure |
| 60 | 62.374 | 40.042 | -22.332 | NO | severe_overfit | range_failure |
| 90 | 114.440 | 42.143 | -72.297 | NO | severe_overfit | range_failure |

**Pooled val MAE: 75.180  Floor: 35.578  Margin: -39.601**
**All horizons beat floor: False**

## Ridge reference model (initial mode)

| h | Ridge val MAE | LightGBM val MAE | LightGBM better |
|---|---------------|-----------------|-----------------|
| 1 | 59.385 | 49.197 | YES |
| 7 | 44.762 | 72.056 | **NO — INVESTIGATE** |
| 14 | 47.108 | 74.038 | **NO — INVESTIGATE** |
| 30 | 69.498 | 78.972 | **NO — INVESTIGATE** |
| 60 | 63.696 | 62.374 | YES |
| 90 | 100.952 | 114.440 | **NO — INVESTIGATE** |

## Secondary metrics (RMSE, SMAPE, MASE)

| h | Train MAE | Val MAE | Train RMSE | Val RMSE | Val SMAPE | Val MASE |
|---|-----------|---------|------------|----------|-----------|----------|
| 1 | 23.262 | 49.197 | 36.308 | 63.670 | 19.1% | 0.5068 |
| 7 | 27.550 | 72.056 | 42.669 | 81.571 | 29.1% | 0.9837 |
| 14 | 23.241 | 74.038 | 35.399 | 84.939 | 31.1% | 1.0114 |
| 30 | 27.950 | 78.972 | 43.312 | 87.937 | 33.4% | 0.7831 |
| 60 | 28.134 | 62.374 | 44.984 | 80.577 | 29.2% | 0.7402 |
| 90 | 37.087 | 114.440 | 53.255 | 134.251 | 66.7% | 1.708 |

## Residual diagnostics

| h | Resid mean | ACF lag1 | ACF lag7 | Range check |
|---|-----------|----------|----------|------------|
| 1 | +23.7 | -0.236 | 0.330 | range_warning |
| 7 | +60.7 | -0.105 | 0.040 | range_failure |
| 14 | +62.4 | -0.178 | 0.061 | range_failure |
| 30 | +65.3 | -0.025 | 0.004 | range_failure |
| 60 | +28.1 | -0.114 | -0.036 | range_failure |
| 90 | +20.3 | -0.204 | 0.000 | range_failure |

**ACF guidance**: |r| > 0.2 indicates uncaptured temporal structure (see plots).

## Gaps and next steps
1. **REQUIRED**: Fit SARIMA per horizon (iteration 007) to close classical_statistical slot.
2. **REQUIRED**: Hyperparameter search (Optuna, n_trials≥50) before run close.
3. Diagnosis: examine residual ACF patterns and achievable-range caveat for deployment.

## Clarifications needed
- CQ-TR-001 (non-blocking): SARIMA slot unfilled — schedule classical TS training next.
- CQ-TR-002 (non-blocking): Hyperparameter search deferred — schedule before run close.