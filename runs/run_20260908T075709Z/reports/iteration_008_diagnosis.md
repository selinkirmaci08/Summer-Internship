# Iteration 008 — Diagnosis
**Run**: run_20260908T075709Z | **Date**: 2026-09-08

---

## Stop / Continue: CONTINUE

**Verdict**: No horizon beats the baseline floor. Three coverage blocks remain:
1. No hyperparameter search run (guardrail)
2. Feature engineering visited only once — no feature rationale tested against evidence
3. No model beats floor on any horizon

---

## Measurement Checks (Step 0)

| Check | Result | Notes |
|-------|--------|-------|
| 1. Units (rows vs time) | PASS | FE uses calendar-day offsets throughout; lag features DOW-aligned |
| 2. Regime | WARN | Only 30 post-regime training rows (1.1%) vs 100% post-regime val |
| 3. Rows | MINOR | Baseline h=1 n=95, LGBM h=1 n=94 (1-row Dec31 difference); non-material |
| 4. Floor | PASS | 22-candidate baseline grid; smedian_w4 is a strong floor at 31.263 |
| 5. Justification | WARN | 12-tree LGBM cannot use lag features (Spearman r=0.76); only DOW + naive group used |

---

## Primary Issue and Root Cause

**Issue**: LightGBM achieves val MAE=39.36 (h=1) vs floor 31.263. ARIMA achieves 70.86. Neither beats the baseline.

**Root cause** (two compounding mechanisms):

1. **COVID contamination**: 41 lockdown training rows (March-May 2020, mean demand=264) create extreme gradients. LightGBM's early stopping (regression_l1) fires at 12 trees because these rows produce noisy, unstable gradients on the 95-row validation set. With only 12 trees, the model cannot utilize lag features (bread_lag_7d, Spearman r=0.76) that require deeper splits to express.

2. **Regime underfitting**: Only 30 post-regime training rows exist (1.1%) vs all 95 validation rows being post-regime. Monday demand pattern learned from ~4 examples; validation has 14 Mondays.

**Why the naive predictor wins**: smedian_w4 uses only the 4 most recent same-DOW observations. For Oct-Dec 2020 Mondays, these 4 observations are all post-regime (Jun-Sep 2020), correctly capturing the new Monday demand level. LightGBM and ARIMA both use the full training distribution.

---

## Metric Trend

| Model | h=1 | h=7 | h=14 | h=30 | h=60 | h=90 | Pooled |
|-------|-----|-----|------|------|------|------|--------|
| Baseline floor | 31.26 | 31.26 | 32.85 | 35.91 | 40.04 | 42.14 | 35.58 |
| LightGBM (iter006) | 39.36 | 39.39 | 40.19 | 43.23 | 50.08 | 79.22 | 48.58 |
| ARIMA (iter007) | 70.86 | 71.23 | 72.21 | 70.40 | 68.58 | 86.84 | 73.35 |

LightGBM > ARIMA > (floor wins). No model beats floor.

---

## Ranked Hypotheses

### Rank 1 — Ratio target + COVID exclusion (FE + Training, moderate cost, medium confidence)

**Statement**: Replace the raw target with a multiplicative ratio: ratio_h{h} = bread_quantity_h{h} / anchor_h{h} (anchor = smedian_w4/w26 per horizon). Train LightGBM on non-COVID rows only. Final prediction = pred_ratio x anchor.

**Evidence**: The additive residual approach (iter006) failed because the train/val residual mean asymmetry persisted (train=+13.45, val=-13.15 for h=1). The ratio target on non-COVID rows has much more similar means: train=1.0616, val=0.9724.

**Oracle check**: predicting naive x mean_train_ratio as a constant gives val MAE=58.166 for h=1 — still above the floor (31.263) by 26.903 units. Features are needed to beat the floor; the ratio alone is not sufficient.

**Falsification**: If LightGBM ratio-target model (non-COVID rows) achieves val_mae >= 31.263 for h=1, reject.

**Suggested action (node=feature_engineering)**: Add ratio_h{h} columns. Training filters COVID rows.

---

### Rank 2 — COVID exclusion only (Training only, cheap, medium confidence)

**Statement**: Exclude COVID lockdown rows (is_covid_lockdown_period==1) from LightGBM training. No FE change needed. This removes the extreme gradient contamination and allows early stopping to find a more useful tree count.

**Evidence**: 41 COVID rows (1.5% of training) with mean demand=264 vs normal 319. These create large MAE-gradient spikes.

**Falsification**: If LGBM without COVID rows still achieves val_mae >= 31.263 for h=1, COVID contamination alone is not the primary failure cause.

**Suggested action (node=training)**: Filter train_mask to exclude is_covid_lockdown_period==1. Use fold-1 and fold-2 val rows for inner early stopping (not last 15% of training, which falls in COVID period).

---

### Rank 3 — Rolling window training (Training only, cheap, low confidence)

**Statement**: Use only 2019-01-01 to 2020-02-28 as training (~320 rows). Removes COVID AND level trend; recent data closer to validation distribution.

**Falsification**: If rolling-window LGBM (n=320) achieves val_mae >= 31.263, the dataset is too small for tree-based learning.

**Suggested action (node=training)**: Filter train_mask to date >= 2019-01-01 AND is_covid_lockdown_period==0.

---

## Alternative if All Hypotheses Fail

Accept smedian_w4/w26 as the per-horizon champion naive predictor. Proceed to end_model. Document that ML/ARIMA cannot beat the naive predictor on this training distribution.

---

## Clarifications

**CQ-D-001** (non-blocking): Ratio target requires anchor (smedian_w4) at serving time. Confirm anchor availability in production system.

**CQ-D-002** (non-blocking): Optuna search (n_trials>=50) required before run close per training.md guardrail. Defer until ratio-target model established.

---

*Artifacts*: `artifacts/diagnosis/plot1_metric_trend.png`, `plot2_naive_residuals_val.png`, `plot3_covid_contamination.png`, `metric_trend.csv`, `naive_residuals_val.csv`
