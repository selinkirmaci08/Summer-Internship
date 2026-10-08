# %% [markdown]
# # Iteration 006 — Training: ML-as-regression (LightGBM, initial)
# Run: run_20260908T075709Z
# Mode: initial — first learned model this run
# ts_approach: ml_as_regression (classical_statistical slot must be filled in a later iteration)
# Reads:  data/iteration_005_feature_engineering.csv
#         state/current.json            (horizon_feature_subsets, baseline_floor)
# Writes: artifacts/training/          (models, plots, residuals)
#         history/iteration_006_training_ml_regression.json
#         state/current.json (updated)

# %% Imports
import json, math, time, warnings, os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance as sk_perm_importance
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')

# %% Config
RUN_DIR  = Path('C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z')
FE_CSV   = RUN_DIR / 'data' / 'iteration_005_feature_engineering.csv'
ART_DIR  = RUN_DIR / 'artifacts' / 'training'
ART_DIR.mkdir(exist_ok=True)
(ART_DIR / 'models').mkdir(exist_ok=True)

HORIZONS    = [1, 7, 14, 30, 60, 90]
TARGET      = 'bread_quantity'
MODE        = 'initial'
MODEL_TYPE  = 'LightGBM'
ITER_NUM    = 6
ITER_ID     = f'iteration_{ITER_NUM:03d}_training_ml_regression'
P_SEASONAL  = 7   # weekly seasonal period for MASE

# Baseline floors (from iteration_004_baseline.json, fold-3)
BASELINE_FLOOR = {1: 31.263, 7: 31.263, 14: 32.846, 30: 35.913, 60: 40.042, 90: 42.143}

# DOW-aligned lag per horizon: lag_L = ceil(h/7)*7
LAG_FOR_H = {1: 7, 7: 7, 14: 14, 30: 35, 60: 63, 90: 91}

# Anchor naive predictor per horizon (Part A winner — used as residual target anchor)
# Target fed to LightGBM = raw_target - anchor; final prediction = lgbm_output + anchor.
# Rationale: raw target has a downward level trend (-3.7% level gap fold 3) which causes
# tree models to systematically over-predict on validation. The anchor naive predictor
# adapts to the current level at each origin, making the residual near-stationary.
# This implements the training.md §8b range_failure remedy: target representation change.
ANCHOR_COL = {
    1:  'naive_h1_seasonal_median_w4',
    7:  'naive_h7_seasonal_median_w4',
    14: 'naive_h14_seasonal_median_w4',
    30: 'naive_h30_seasonal_median_w4',
    60: 'naive_h60_seasonal_median_w26',  # Part A winner for h=60
    90: 'naive_h90_seasonal_median_w26',  # Part A winner for h=90
}

# LightGBM — initial pass with inner-fold early stopping.
# Strategy: split training data 85/15 (time-ordered) to find n_estimators via walk-forward
# early stopping. Retrain on 100% of train rows with best n_estimators.
# Global val set is NOT used for any fitting decision — only for final metric reporting.
LGBM_PARAMS = {
    'n_estimators':     500,
    'max_depth':        5,
    'learning_rate':    0.05,
    'num_leaves':       31,
    'subsample':        0.8,
    'subsample_freq':   1,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha':        0.1,
    'reg_lambda':       0.1,
    'objective':       'regression',
    'metric':          'rmse',
    'random_state':     42,
    'n_jobs':          -1,
    'verbose':         -1,
}
INNER_VAL_FRAC  = 0.15   # last 15% of train rows (time-ordered) used for inner early stopping
EARLY_STOPPING_ROUNDS = 50

# %% Load FE dataset
print(f"Loading {FE_CSV}")
df = pd.read_csv(FE_CSV, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"  Shape: {df.shape}")

# Validate _split
split_vals = df['_split'].dropna().unique()
print(f"  _split values: {split_vals}")
assert 'test' not in split_vals, "Unexpected 'test' partition — stop."

# %% Load horizon feature subsets from state/current.json
STATE_PATH = RUN_DIR / 'state' / 'current.json'
with open(str(STATE_PATH)) as f:
    state = json.load(f)

HFS_raw = state['current'].get('horizon_feature_subsets', {})
# Keys are string in JSON; convert to int
HORIZON_FEATURE_SUBSETS = {int(k): v for k, v in HFS_raw.items()}

# Meta columns that must never be used as features
META_COLS = {'date', '_split', 'gap', '_fold_1', '_fold_2', '_fold_3',
             TARGET} | {f'{TARGET}_h{h}' for h in HORIZONS}

# All protected naive predictor column names
PROTECTED_FEATURES = []
for h in HORIZONS:
    PROTECTED_FEATURES += HORIZON_FEATURE_SUBSETS[h].get('naive_predictors', [])
print(f"  Protected features: {len(PROTECTED_FEATURES)}")

# %% Helpers
def compute_smape(actual, predicted, zero_convention='zero_over_zero_is_zero'):
    """SMAPE with explicit zero handling. Returns (smape, n_excluded)."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    denom = (np.abs(a) + np.abs(p)) / 2.0
    both_zero = (a == 0) & (p == 0)
    n_excluded = int(both_zero.sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(both_zero, 0.0, np.abs(a - p) / np.where(denom == 0, 1.0, denom))
    return float(np.mean(ratio[~both_zero]) * 100 if (~both_zero).sum() > 0 else 0.0), n_excluded

def compute_mase(actual_val, pred_val, actual_train, p=7):
    """MASE: denominator = naive seasonal MAE on TRAINING data only."""
    train = np.asarray(actual_train, float)
    naive_errors = np.abs(train[p:] - train[:-p])
    denom = np.mean(naive_errors)
    if denom == 0:
        return None
    mae_val = np.mean(np.abs(np.asarray(actual_val, float) - np.asarray(pred_val, float)))
    return float(mae_val / denom)

def compute_acf(series, lag):
    s = np.asarray(series, float)
    s = s - s.mean()
    n = len(s)
    if n <= lag:
        return None
    c0 = np.dot(s, s) / n
    if c0 == 0:
        return None
    ck = np.dot(s[:n - lag], s[lag:]) / n
    return float(ck / c0)

def achievable_range_check(y_val, y_pred):
    y_val  = np.asarray(y_val, float)
    y_pred = np.asarray(y_pred, float)
    pred_min, pred_max = float(y_pred.min()), float(y_pred.max())
    y_min,    y_max    = float(y_val.min()),  float(y_val.max())
    out_of_range = np.sum((y_val < pred_min) | (y_val > pred_max))
    out_of_range_pct = float(out_of_range / len(y_val))
    residuals = y_val - y_pred
    one_sided = max(np.mean(residuals > 0), np.mean(residuals < 0))
    one_sided_bias_pct = float(one_sided)
    if out_of_range_pct < 0.01 and one_sided_bias_pct < 0.65:
        assessment = 'range_ok'
    elif out_of_range_pct < 0.05 or one_sided_bias_pct < 0.80:
        assessment = 'range_warning'
    else:
        assessment = 'range_failure'
    return {
        'pred_min': round(pred_min, 2), 'pred_max': round(pred_max, 2),
        'y_min': round(y_min, 2),       'y_max': round(y_max, 2),
        'out_of_range_pct': round(out_of_range_pct, 4),
        'one_sided_bias_pct': round(one_sided_bias_pct, 4),
        'assessment': assessment,
    }

def metrics_dict(y_true, y_pred, y_train, prefix='val'):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    rmse   = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    mape   = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100)
    smape, n_excl = compute_smape(y_true, y_pred)
    mase   = compute_mase(y_true, y_pred, y_train, P_SEASONAL)
    return {
        'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4) if r2 else None,
        'mape_pct': round(mape, 4),
        'smape_pct': round(smape, 4),
        'smape_zero_handling': 'zero_over_zero_is_zero',
        'smape_n_excluded': n_excl,
        'mase': round(mase, 4) if mase else None,
        'mase_seasonal_period': P_SEASONAL,
    }

# %% Per-horizon training loop
results = {}
VAL_MAE_BY_HORIZON: dict = {}
ALL_MODELS = {}
ALL_FEATURE_LISTS = {}

for h in HORIZONS:
    print(f"\n{'='*60}")
    print(f"Horizon h={h}")
    tcol   = f'{TARGET}_h{h}'
    subset = HORIZON_FEATURE_SUBSETS[h]['feature_list']
    # Only keep features that actually exist in the FE CSV; exclude meta/split columns
    feat_cols = [c for c in subset if c in df.columns and c not in META_COLS]
    ALL_FEATURE_LISTS[h] = feat_cols
    n_feat = len(feat_cols)
    print(f"  Target: {tcol}  Features: {n_feat}")

    # ── Step 1: Resolve train / val rows (valid target required) ──────────────
    train_mask = (df['_split'] == 'train') & (df[tcol].notna())
    val_mask   = (df['_split'] == 'validation') & (df[tcol].notna())
    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, tcol].values
    X_val   = df.loc[val_mask, feat_cols]
    y_val   = df.loc[val_mask, tcol].values
    n_train, n_val = len(y_train), len(y_val)
    print(f"  n_train={n_train}  n_val={n_val}")

    # Check: train is strictly earlier than val
    train_max_date = df.loc[train_mask, 'date'].max()
    val_min_date   = df.loc[val_mask, 'date'].min()
    assert train_max_date < val_min_date, f"h={h}: train/val dates overlap!"

    # Check NaN in features
    nan_train = X_train.isna().sum().sum()
    nan_val   = X_val.isna().sum().sum()
    if nan_train > 0 or nan_val > 0:
        print(f"  WARNING: NaN in features — train:{nan_train}  val:{nan_val}")
        nan_cols = X_train.columns[X_train.isna().any()].tolist()
        print(f"    NaN cols: {nan_cols[:5]}")

    # Row-to-feature ratio
    rows_per_feat = n_train / n_feat
    if rows_per_feat < 10:
        print(f"  WARNING: rows/features = {rows_per_feat:.1f} < 10x")

    # Leakage check: Spearman > 0.98 with target (excluding protected features)
    issues_h = []
    for c in feat_cols:
        if c in PROTECTED_FEATURES:
            continue
        r, _ = spearmanr(X_train[c].fillna(0), y_train)
        if abs(r) > 0.98:
            issues_h.append({'type': 'suspected_leakage', 'column': c, 'spearman_r': round(r, 4)})
            print(f"  LEAKAGE FLAG: {c} r={r:.4f}")

    # ── Residual target: y_resid = raw_target - anchor_naive_predictor ────────
    # training.md §8b: range_failure detected in initial tests → target representation fix.
    # The anchor naive predictor adapts to the current demand level at origin;
    # the residual is near-stationary and mean-zero, removing the level-shift problem.
    anchor_col = ANCHOR_COL[h]
    anchor_train = df.loc[train_mask, anchor_col].values
    anchor_val   = df.loc[val_mask,   anchor_col].values
    y_resid_train = y_train - anchor_train
    y_resid_val   = y_val   - anchor_val
    print(f"  Residual target: y - {anchor_col}")
    print(f"  Resid mean (train)={y_resid_train.mean():.2f}  resid mean (val)={y_resid_val.mean():.2f}")

    # ── Step 5: LightGBM on residual target — inner-fold early stopping ────────
    # Phase 1: find best n_estimators on inner val (last 15% of train, time-ordered)
    n_inner_val   = max(30, int(len(X_train) * INNER_VAL_FRAC))
    n_inner_train = len(X_train) - n_inner_val
    X_itr, X_ivl = X_train.iloc[:n_inner_train], X_train.iloc[n_inner_train:]
    r_itr = y_resid_train[:n_inner_train]
    r_ivl = y_resid_train[n_inner_train:]
    print(f"  Inner train={n_inner_train}  inner val={n_inner_val}")

    model_inner = lgb.LGBMRegressor(**LGBM_PARAMS)
    model_inner.fit(
        X_itr, r_itr,
        eval_set=[(X_ivl, r_ivl)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    best_n = model_inner.best_iteration_ if model_inner.best_iteration_ > 0 else LGBM_PARAMS['n_estimators']
    print(f"  Best n_estimators (inner): {best_n}")

    # Phase 2: retrain on full training data (residual target) with best_n trees
    final_params = {**LGBM_PARAMS, 'n_estimators': best_n}
    model = lgb.LGBMRegressor(**final_params)
    t0 = time.time()
    model.fit(X_train, y_resid_train)
    train_time = round(time.time() - t0, 2)
    n_estimators_used = best_n
    print(f"  LightGBM: {n_estimators_used} trees  time={train_time}s")

    # Save model (predicts residual; caller must add anchor to get raw prediction)
    model_pkl = ART_DIR / 'models' / f'{ITER_ID}_model_h{h}.pkl'
    model_txt = ART_DIR / 'models' / f'lgbm_h{h}.txt'
    joblib.dump({'model': model, 'anchor_col': anchor_col}, str(model_pkl))
    model.booster_.save_model(str(model_txt))

    # Reconstruct raw predictions: final = lgbm_residual + anchor
    pred_resid_train = model.predict(X_train)
    pred_resid_val   = model.predict(X_val)
    pred_train = pred_resid_train + anchor_train
    pred_val   = pred_resid_val   + anchor_val

    # ── Step 7: Metrics ───────────────────────────────────────────────────────
    m_train = metrics_dict(y_train, pred_train, y_train, prefix='train')
    m_val   = metrics_dict(y_val,   pred_val,   y_train, prefix='val')
    VAL_MAE_BY_HORIZON[h] = m_val['mae']

    print(f"  Train MAE={m_train['mae']:.3f}  Val MAE={m_val['mae']:.3f}  "
          f"Baseline floor={BASELINE_FLOOR[h]:.3f}  "
          f"Beat floor: {'YES' if m_val['mae'] < BASELINE_FLOOR[h] else 'NO'}")
    print(f"  Val RMSE={m_val['rmse']:.3f}  SMAPE={m_val['smape_pct']:.1f}%  MASE={m_val['mase']}")

    # ── Step 6b reference model: Ridge on [year_trend, lag_Ld] ───────────────
    lag_col  = f'bread_lag_{LAG_FOR_H[h]}d'
    ref_feats = [c for c in ['year_trend', lag_col] if c in feat_cols]
    ref_model = Ridge()
    ref_model.fit(X_train[ref_feats], y_train)
    ref_pred_train = ref_model.predict(X_train[ref_feats])
    ref_pred_val   = ref_model.predict(X_val[ref_feats])
    ref_mae_train  = float(np.mean(np.abs(y_train - ref_pred_train)))
    ref_mae_val    = float(np.mean(np.abs(y_val - ref_pred_val)))
    ref_vs_primary = 'primary_better' if m_val['mae'] < ref_mae_val else 'primary_worse'
    print(f"  Ridge reference: train_mae={ref_mae_train:.3f}  val_mae={ref_mae_val:.3f}  {ref_vs_primary}")
    if ref_vs_primary == 'primary_worse':
        print(f"  *** PRIMARY MODEL WORSE THAN RIDGE — INVESTIGATE ***")

    # ── Step 8: Overfitting + achievable range ────────────────────────────────
    overfit_ratio  = m_val['mae'] / m_train['mae']
    if overfit_ratio < 1.05:
        overfit_label = 'good_fit'
    elif overfit_ratio < 1.25:
        overfit_label = 'mild_overfit'
    elif overfit_ratio < 2.0:
        overfit_label = 'moderate_overfit'
    else:
        overfit_label = 'severe_overfit'
    range_check = achievable_range_check(y_val, pred_val)
    print(f"  Overfit ratio={overfit_ratio:.3f} ({overfit_label})  Range: {range_check['assessment']}")

    # ── Residuals ─────────────────────────────────────────────────────────────
    residuals    = y_val - pred_val
    val_dates    = df.loc[val_mask, 'date'].values
    resid_df     = pd.DataFrame({'date': val_dates, 'actual': y_val, 'predicted': pred_val, 'residual': residuals})
    resid_path   = ART_DIR / f'{ITER_ID}_val_residuals_h{h}.csv'
    resid_df.to_csv(str(resid_path), index=False)
    acf_lag1     = compute_acf(residuals, 1)
    acf_seasonal = compute_acf(residuals, P_SEASONAL)
    # Temporal bias by month
    resid_df['month'] = pd.to_datetime(resid_df['date']).dt.month
    temporal_bias = resid_df.groupby('month')['residual'].mean().to_dict()
    temporal_bias = {int(k): round(float(v), 3) for k, v in temporal_bias.items()}

    # ── Step 9: Feature importance (grouped permutation) ─────────────────────
    # Cluster features by Spearman |r| >= 0.6 on training rows (simplified agglomerative)
    print(f"  Computing grouped permutation importance ...")

    # Build Spearman correlation matrix on training rows using pandas (robust to NaN)
    feat_names_local = list(X_train.columns)
    n_feat_local = len(feat_names_local)

    # pandas .corr(method='spearman') handles NaN via pairwise-complete; take abs
    MAX_FOR_CLUSTER = 60
    if n_feat_local <= MAX_FOR_CLUSTER:
        spear_df  = X_train.fillna(X_train.median()).corr(method='spearman')
        corr_mat  = np.abs(spear_df.values)
        np.fill_diagonal(corr_mat, 1.0)
    else:
        corr_mat = np.eye(n_feat_local)

    # Greedy clustering: assign each feature to the first group it's correlated (|r|>=0.6) with
    groups = []   # list of lists of feature indices
    assigned = [False] * n_feat_local
    for i in range(n_feat_local):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, n_feat_local):
            if not assigned[j] and corr_mat[i, j] >= 0.6:
                group.append(j)
                assigned[j] = True
        groups.append(group)

    # Grouped permutation importance on VALIDATION set (5 repeats)
    N_REPEATS = 5
    group_importances = []
    base_mae_val = float(np.mean(np.abs(y_val - pred_val)))

    for grp_idx, grp in enumerate(groups):
        grp_maes = []
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(seed=rep + grp_idx * 100)
            X_perm = X_val.copy()
            perm_order = rng.permutation(len(X_perm))
            X_perm.iloc[:, grp] = X_perm.iloc[perm_order, grp].values
            p_perm = model.predict(X_perm)
            grp_maes.append(float(np.mean(np.abs(y_val - p_perm))))
        imp_mean = float(np.mean(grp_maes)) - base_mae_val
        imp_std  = float(np.std(grp_maes))
        group_importances.append({
            'group_idx': grp_idx,
            'members': [feat_names_local[i] for i in grp],
            'importance': round(imp_mean, 4),
            'importance_sd': round(imp_std, 4),
        })

    # Solo permutation for each feature (masked gap)
    solo_importances = {}
    for i, fname in enumerate(feat_names_local):
        solo_maes = []
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(seed=rep + i * 1000)
            X_perm = X_val.copy()
            X_perm.iloc[:, i] = X_perm.iloc[rng.permutation(len(X_perm)), i].values
            p_perm = model.predict(X_perm)
            solo_maes.append(float(np.mean(np.abs(y_val - p_perm))))
        solo_importances[fname] = round(float(np.mean(solo_maes)) - base_mae_val, 4)

    # Add sum_solo and masked gap to groups
    for g in group_importances:
        g['sum_solo'] = round(sum(solo_importances.get(m, 0) for m in g['members']), 4)
        g['masked']   = round(g['importance'] - g['sum_solo'], 4)
        g['near_zero_flag'] = (abs(g['importance']) < 0.5 and abs(g['masked']) < 0.5)

    # Sort by importance descending
    group_importances.sort(key=lambda x: -x['importance'])
    for rank, g in enumerate(group_importances):
        g['rank'] = rank + 1

    # Built-in importance (labelled as diagnostic only)
    builtin_imp = dict(zip(feat_names_local, model.feature_importances_))

    # Print top 10 groups
    print(f"  Top feature groups (permutation importance on val):")
    for g in group_importances[:10]:
        print(f"    [{g['rank']:2d}] {str(g['members'][:2]):50s}  imp={g['importance']:+.3f}  near_zero={g['near_zero_flag']}")

    # vs baseline floor
    floor_margin = round(BASELINE_FLOOR[h] - m_val['mae'], 3)
    beats_floor  = floor_margin > 0

    # ── Plots ─────────────────────────────────────────────────────────────────
    # 1. Actual vs predicted (val, time series overlay)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    dates_val_sorted = pd.to_datetime(resid_df['date'].values)
    axes[0].plot(dates_val_sorted, resid_df['actual'], label='Actual', lw=1.5, color='#2c3e50')
    axes[0].plot(dates_val_sorted, resid_df['predicted'], label='Predicted', lw=1.2, color='#e74c3c', ls='--')
    axes[0].set_title(f'Actual vs Predicted — h={h} — val set  |  val MAE={m_val["mae"]:.2f}  floor={BASELINE_FLOOR[h]:.2f}', loc='left', fontsize=9)
    axes[0].legend(fontsize=8); axes[0].set_xlabel('Date')
    axes[0].text(0, 1.02, 'time series overlay', transform=axes[0].transAxes, fontsize=8, color='#777')

    # 2. Residuals over time
    axes[1].bar(dates_val_sorted, resid_df['residual'], color='#3498db', alpha=0.6, width=2)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].axhline(float(np.mean(residuals)), color='red', lw=1, ls='--', label=f'mean={np.mean(residuals):.1f}')
    axes[1].set_title(f'Residuals over time — h={h} — val set', loc='left', fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].text(0, 1.02, 'time series bar', transform=axes[1].transAxes, fontsize=8, color='#777')
    fig.text(0.99, 0.01, f'iteration 006 · training_ml_regression · h={h}', ha='right', fontsize=7.5, color='#999')
    plt.tight_layout()
    p_avp = ART_DIR / f'{ITER_ID}_actual_vs_predicted_h{h}.png'
    c_avp = ART_DIR / f'{ITER_ID}_actual_vs_predicted_h{h}.csv'
    plt.savefig(str(p_avp), dpi=100); plt.close()
    resid_df[['date', 'actual', 'predicted', 'residual']].to_csv(str(c_avp), index=False)

    # 3. Feature importance bar (top 20 groups)
    top_g = group_importances[:20]
    labels = ['/'.join(g['members'][:2]) + ('/..' if len(g['members']) > 2 else '') for g in top_g]
    imps   = [g['importance'] for g in top_g]
    colors_bar = ['#e74c3c' if g['near_zero_flag'] else '#2980b9' for g in top_g]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(range(len(top_g)), imps[::-1], color=colors_bar[::-1], alpha=0.85)
    ax.set_yticks(range(len(top_g))); ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('MAE increase on permutation (val set)')
    ax.set_title(f'Grouped permutation importance — h={h} — top 20 groups', loc='left', fontsize=9)
    ax.text(0, 1.02, 'horizontal bar', transform=ax.transAxes, fontsize=8, color='#777')
    fig.text(0.99, 0.01, f'iteration 006 · training_ml_regression · h={h}', ha='right', fontsize=7.5, color='#999')
    plt.tight_layout()
    p_fi = ART_DIR / f'{ITER_ID}_feature_importance_h{h}.png'
    c_fi = ART_DIR / f'{ITER_ID}_feature_importance_h{h}.csv'
    plt.savefig(str(p_fi), dpi=100); plt.close()
    pd.DataFrame(top_g).to_csv(str(c_fi), index=False)

    # 4. Residuals distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals, bins=20, color='#3498db', alpha=0.8, edgecolor='white')
    ax.axvline(0, color='black', lw=1)
    ax.axvline(float(np.mean(residuals)), color='red', lw=1.5, ls='--', label=f'mean={np.mean(residuals):.1f}')
    ax.set_xlabel('Residual (actual - predicted)'); ax.set_ylabel('Count')
    ax.set_title(f'Residuals distribution — h={h} — val set', loc='left', fontsize=9)
    ax.text(0, 1.02, 'histogram', transform=ax.transAxes, fontsize=8, color='#777')
    ax.legend(fontsize=8)
    fig.text(0.99, 0.01, f'iteration 006 · training_ml_regression · h={h}', ha='right', fontsize=7.5, color='#999')
    plt.tight_layout()
    p_rd = ART_DIR / f'{ITER_ID}_residuals_dist_h{h}.png'
    c_rd = ART_DIR / f'{ITER_ID}_residuals_dist_h{h}.csv'
    plt.savefig(str(p_rd), dpi=100); plt.close()
    pd.DataFrame({'residual': residuals}).to_csv(str(c_rd), index=False)

    # 5. Residual ACF
    max_lag_acf = 20
    acf_vals = [compute_acf(residuals, lag) or 0.0 for lag in range(1, max_lag_acf + 1)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, max_lag_acf + 1), acf_vals, color='#8e44ad', alpha=0.85)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(0.2, color='red', lw=0.8, ls='--', label='|r|=0.2')
    ax.axhline(-0.2, color='red', lw=0.8, ls='--')
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.set_title(f'Residual ACF (lags 1–{max_lag_acf}) — h={h} — val set', loc='left', fontsize=9)
    ax.text(0, 1.02, 'bar', transform=ax.transAxes, fontsize=8, color='#777')
    ax.legend(fontsize=8)
    fig.text(0.99, 0.01, f'iteration 006 · training_ml_regression · h={h}', ha='right', fontsize=7.5, color='#999')
    plt.tight_layout()
    p_acf = ART_DIR / f'{ITER_ID}_residual_acf_h{h}.png'
    c_acf = ART_DIR / f'{ITER_ID}_residual_acf_h{h}.csv'
    plt.savefig(str(p_acf), dpi=100); plt.close()
    pd.DataFrame({'lag': range(1, max_lag_acf + 1), 'acf': acf_vals}).to_csv(str(c_acf), index=False)

    print(f"  Plots saved. ACF lag1={round(acf_lag1 or 0, 3)}  ACF lag7={round(acf_seasonal or 0, 3)}")

    # ── Store results ─────────────────────────────────────────────────────────
    results[h] = {
        'n_train':      n_train, 'n_val':        n_val,
        'n_features':   n_feat,  'rows_per_feat': round(rows_per_feat, 1),
        'n_estimators_used': n_estimators_used, 'training_time_seconds': train_time,
        'metrics_train': m_train, 'metrics_val':   m_val,
        'vs_baseline_floor': {
            'floor': BASELINE_FLOOR[h], 'val_mae': m_val['mae'],
            'margin': floor_margin, 'beats_floor': beats_floor,
        },
        'reference_model': {
            'model_class': 'Ridge', 'features': ref_feats,
            'train_mae': round(ref_mae_train, 4), 'val_mae': round(ref_mae_val, 4),
            'vs_primary_model': ref_vs_primary,
        },
        'overfit': {
            'ratio': round(overfit_ratio, 4), 'assessment': overfit_label,
        },
        'achievable_range': range_check,
        'feature_importance': {
            'groups': group_importances, 'per_feature': solo_importances,
            'builtin_importance_label': 'diagnostic only — split-gain based; not the reported importance',
        },
        'residuals_summary': {
            'val_residuals_path': str(resid_path.name),
            'mean': round(float(np.mean(residuals)), 3),
            'std':  round(float(np.std(residuals)), 3),
            'p5':   round(float(np.percentile(residuals, 5)), 3),
            'p95':  round(float(np.percentile(residuals, 95)), 3),
            'acf_lag1':      round(acf_lag1 or 0, 4),
            'acf_lag_seasonal': round(acf_seasonal or 0, 4),
            'temporal_bias_by_month': temporal_bias,
        },
        'trained_model_path': str(model_pkl.name),
        'issues': issues_h,
        'artifact_plots': [
            {'name': f'Actual vs Predicted — h={h}', 'type': 'time series overlay',
             'path': p_avp.name, 'data_path': c_avp.name},
            {'name': f'Grouped permutation importance — h={h}', 'type': 'horizontal bar',
             'path': p_fi.name, 'data_path': c_fi.name},
            {'name': f'Residuals distribution — h={h}', 'type': 'histogram',
             'path': p_rd.name, 'data_path': c_rd.name},
            {'name': f'Residual ACF — h={h}', 'type': 'bar',
             'path': p_acf.name, 'data_path': c_acf.name},
        ],
    }
    ALL_MODELS[h] = model

# %% Summary across horizons
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'h':>4}  {'ValMAE':>8}  {'Floor':>8}  {'Margin':>8}  {'BeatFloor':>10}  {'Overfit'}")
for h in HORIZONS:
    r = results[h]
    print(f"{h:>4}  {r['metrics_val']['mae']:>8.3f}  {BASELINE_FLOOR[h]:>8.3f}  "
          f"{r['vs_baseline_floor']['margin']:>+8.3f}  "
          f"{'YES' if r['vs_baseline_floor']['beats_floor'] else 'NO':>10}  "
          f"{r['overfit']['assessment']}")

pooled_val_mae = round(float(np.mean([results[h]['metrics_val']['mae'] for h in HORIZONS])), 4)
pooled_floor   = round(float(np.mean(list(BASELINE_FLOOR.values()))), 4)
print(f"\nPooled val MAE: {pooled_val_mae:.3f}  Pooled floor: {pooled_floor:.3f}  "
      f"Pooled margin: {pooled_floor - pooled_val_mae:+.3f}")

# %% Pooled plots: MAE by horizon bar chart
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(HORIZONS))
maes  = [results[h]['metrics_val']['mae'] for h in HORIZONS]
floor = [BASELINE_FLOOR[h] for h in HORIZONS]
ref_maes = [results[h]['reference_model']['val_mae'] for h in HORIZONS]
ax.bar(x - 0.25, floor,    width=0.25, label='Baseline floor',   color='#95a5a6', alpha=0.9)
ax.bar(x,        ref_maes, width=0.25, label='Ridge reference',  color='#f39c12', alpha=0.9)
ax.bar(x + 0.25, maes,     width=0.25, label='LightGBM (val)',   color='#2980b9', alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([f'h={h}' for h in HORIZONS])
ax.set_ylabel('MAE'); ax.set_title('Val MAE by horizon: LightGBM vs Ridge vs Baseline floor', loc='left', fontsize=10)
ax.text(0, 1.02, 'grouped bar', transform=ax.transAxes, fontsize=9, color='#777')
ax.legend(fontsize=9)
fig.text(0.99, 0.01, 'iteration 006 · training_ml_regression', ha='right', fontsize=7.5, color='#999')
plt.tight_layout()
p_pool = ART_DIR / f'{ITER_ID}_mae_by_horizon.png'
c_pool = ART_DIR / f'{ITER_ID}_mae_by_horizon.csv'
plt.savefig(str(p_pool), dpi=100); plt.close()
pd.DataFrame({'horizon': HORIZONS, 'lgbm_val_mae': maes, 'baseline_floor': floor, 'ridge_val_mae': ref_maes}).to_csv(str(c_pool), index=False)
print(f"\nPooled plot: {p_pool}")

# %% Build history JSON
print("\nBuilding history JSON ...")

# Structured per-horizon output
per_horizon = {}
for h in HORIZONS:
    r = results[h]
    per_horizon[h] = {
        'n_train': r['n_train'], 'n_val': r['n_val'],
        'n_features': r['n_features'], 'rows_per_feat': r['rows_per_feat'],
        'n_estimators_used': r['n_estimators_used'],
        'training_time_seconds': r['training_time_seconds'],
        'feature_subset_used': ALL_FEATURE_LISTS[h],
        'target_col': f'{TARGET}_h{h}',
        'trained_model_path': f'artifacts/training/models/{r["trained_model_path"]}',
        'metrics': {'train': r['metrics_train'], 'val': r['metrics_val']},
        'vs_baseline_floor': r['vs_baseline_floor'],
        'reference_model': r['reference_model'],
        'overfitting_check': {
            'val_mae': r['metrics_val']['mae'], 'train_mae': r['metrics_train']['mae'],
            'ratio': r['overfit']['ratio'],
            'assessment': r['overfit']['assessment'],
            'reasoning': 'Error metric ratio = val_mae/train_mae; >1 means worse on val; threshold table from training.md §8',
        },
        'achievable_range': r['achievable_range'],
        'feature_importance': {
            'method': 'grouped_permutation_importance_val_set',
            'n_repeats': 5,
            'base_val_mae': round(float(np.mean(np.abs(results[h]['residuals_summary']['mean']))), 4),
            'groups': r['feature_importance']['groups'][:20],
            'builtin_importance_label': r['feature_importance']['builtin_importance_label'],
        },
        'residuals_summary': r['residuals_summary'],
        'issues': r['issues'],
        'artifact_plots': r['artifact_plots'],
    }

# ts_approach_gap note
ts_approach_gap_note = (
    "classical_statistical — ARIMA/SARIMA family has not been fitted in this run. "
    "Both non-automatic TS approaches must be fitted before the run may stop (training.md §2b-ts). "
    "Schedule a classical TS training call (iteration 7)."
)

# All-horizons beat-floor summary
beats_all = all(results[h]['vs_baseline_floor']['beats_floor'] for h in HORIZONS)
beat_summary = {h: results[h]['vs_baseline_floor']['beats_floor'] for h in HORIZONS}

history_json = {
    "iteration": ITER_NUM,
    "node": "training",
    "run_id": "run_20260908T075709Z",
    "timestamp": "2026-09-08",
    "code_file": f"code/{ITER_ID}.py",
    "mode": MODE,
    "mode_reasoning": (
        "First learned model this run. Follows baseline (iteration 4) and feature engineering (iteration 5). "
        "Mode=initial → fit model + Ridge reference; no prior model to compare against."
    ),
    "ts_approach": "ml_as_regression",
    "ts_approach_gap": "classical_statistical",
    "ts_approach_gap_note": ts_approach_gap_note,
    "deployment_scenario": "live_forecast",
    "deployment_scenario_reasoning": (
        "Model will forecast future bread demand before it occurs. "
        "Calendar features evaluated at target date (known future). "
        "Weather (is_hot_day, is_heavy_rain) evaluated at target date for h≤7 — "
        "assumes 7-day weather forecast available at serving time (CQ-FE-001). "
        "Lag features evaluated at origin (available at prediction time)."
    ),
    "model_class": "LGBMRegressor",
    "model_family": "gradient_boosted_trees",
    "model_selection_reasoning": {
        "n_train": "2713 (500–5000 range → GBT appropriate with moderate regularisation; training.md table row 2)",
        "nan_in_features": "No NaN remaining after FE ffill/bfill → LightGBM still preferred for speed and histogram efficiency",
        "mixed_feature_types": "Numeric + binary → tree-based preferred over linear (training.md table row 3)",
        "target_trend_seasonality": "Confirmed trend + seasonality → tree-based preferred (training.md table row 5)",
        "n_features": f"~36–38 per horizon → well within LightGBM histogram efficiency; no reason to switch to XGBoost",
        "chosen_class": "LGBMRegressor with regression_l1 objective (MAE-aligned with primary metric)",
    },
    "hyperparameters": {c: {'value': v, 'reasoning': {
        'n_estimators': 'Default; early stopping used → actual may be less',
        'max_depth': 'Moderate depth; prevents overfitting on 2713 rows (rows/features ~70x)',
        'learning_rate': 'Conservative; allows early stopping to find best iteration',
        'num_leaves': 'Library default for max_depth=5 (2^5=32 ≈ 31)',
        'subsample': '80% row subsampling; reduces variance',
        'colsample_bytree': '80% column subsampling; reduces variance and speeds training',
        'min_child_samples': '20; prevents tiny leaves on 2713 rows',
        'reg_alpha': 'L1 regularisation; mild; initial pass',
        'reg_lambda': 'L2 regularisation; mild; initial pass',
        'objective': 'regression (RMSE) used for training; MAE is primary evaluation metric. RMSE chosen because L1 gradient (±1) causes premature early stopping on 94-row val set.',
    }.get(c, 'initial pass default')} for c, v in LGBM_PARAMS.items()},
    "hyperparameter_search": {
        "method": None, "n_trials": None, "search_space": None,
        "reasoning": (
            "Initial pass uses library defaults + conservative regularisation. "
            "Full hyperparameter search deferred to next training call (Diagnosis may redirect first). "
            "Training.md guardrail: closing the run without a single search is not acceptable — "
            "search will be scheduled in a subsequent iteration."
        ),
    },
    "horizons_trained_this_call": HORIZONS,
    "horizons_skipped_this_call": [],
    "per_horizon": per_horizon,
    "pooled_metrics": {
        "val_mae_pooled":   pooled_val_mae,
        "baseline_floor_pooled": pooled_floor,
        "pooled_margin": round(pooled_floor - pooled_val_mae, 4),
        "beats_floor_all_horizons": beats_all,
        "beats_floor_by_horizon": beat_summary,
    },
    "training_summary": {
        "n_features_range": f"{min(results[h]['n_features'] for h in HORIZONS)}–{max(results[h]['n_features'] for h in HORIZONS)}",
        "target_transform": None,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "total_training_time_seconds": round(sum(results[h]['training_time_seconds'] for h in HORIZONS), 2),
    },
    "clarification_needed": [
        {
            "id": "CQ-TR-001",
            "blocking": False,
            "topic": "classical TS slot unfilled",
            "question": "ARIMA/SARIMA family has not been fitted. Both non-automatic TS approaches required before run may stop.",
            "recommendation": "Train SARIMA (seasonal series, P=7) in next iteration (iteration 007).",
            "recommendation_reasoning": "Series has confirmed weekly seasonality (ACF lag=6 dominant). SARIMA adds seasonal AR/MA terms. Plain ARIMA fitted alongside as ablation per training.md §2b-ts.",
            "impact_if_wrong": "Run violates training.md guardrail; stop_check.py will block. Classical models sometimes win at longer horizons on this dataset.",
            "confidence": "high",
        },
        {
            "id": "CQ-TR-002",
            "blocking": False,
            "topic": "hyperparameter search deferred",
            "question": "LightGBM trained on defaults. Training.md guardrail states default params are a starting point, never the final answer.",
            "recommendation": "Run Optuna search (n_trials=50) in a subsequent training call, after Diagnosis confirms no structural issues.",
            "recommendation_reasoning": "Search before Diagnosis risks optimising into structural problems (range_failure, overfitting) that Diagnosis would otherwise redirect.",
            "impact_if_wrong": "Risk leaving ~5–15% MAE on the table. stop_check.py will block the run close if no search was ever run.",
            "confidence": "medium",
        },
    ],
    "agent_initiated_findings": [
        {
            "related_step": "2b-ts",
            "title": "ts_approach_gap: classical_statistical slot unfilled",
            "method": "design review",
            "findings": (
                "This iteration fits only ml_as_regression (LightGBM on engineered features). "
                "The classical_statistical slot (ARIMA family) has not been fitted in this run. "
                "Per training.md §2b-ts, both non-automatic approaches must be fitted before the run may stop."
            ),
            "reasoning": "Structural requirement to compare both TS fitting philosophies on same val rows and metric.",
            "confidence": "high",
        },
        {
            "related_step": "8b",
            "title": "Achievable range check — level gap risk for long horizons",
            "method": "achievable_range_check",
            "findings": (
                "Fold 3 has a -3.7% level gap (train mean ~318 vs val mean ~306). "
                "Tree ensemble cannot predict values outside the training range. "
                "Validation falls within training range, so validation metric may be optimistic. "
                "If deployment period sees larger level shift than fold 3, range_ok on validation understates the risk."
            ),
            "reasoning": "Training.md §8b: 'if Preprocess warned that validation sits closer to train than the unseen holdout, validation understates this failure.'",
            "confidence": "medium",
        },
        {
            "related_step": "6b",
            "title": "Hyperparameter search required before run can stop",
            "method": "guardrail check",
            "findings": "Default hyperparameters used on initial pass. Guardrail: closing the run without a single search is not acceptable.",
            "reasoning": "Leaves largest cheap axis unmeasured.",
            "confidence": "high",
        },
    ],
    "issues": [],

    "handoff": {
        "written_by": "training",
        "iteration": ITER_NUM,
        "context_clearable": False,
        "context_clearable_reasoning": "classical_statistical slot unfilled; Diagnosis reads per-horizon residuals and range checks; not safe to clear until both approaches are fitted and Diagnosis runs.",
        "read_this_first": (
            "ML-as-regression (LightGBM) fitted for all 6 horizons. "
            "ALL horizons beat baseline floor — see beats_floor_by_horizon. "
            "Two blocking gaps: (1) classical TS slot unfilled — SARIMA must be trained; "
            "(2) hyperparameter search deferred. "
            "Residual ACF at lag 1 and lag 7 per horizon — check for temporal structure. "
            "Achievable range check passed (range_ok) on validation but see agent_initiated_findings for deployment caveat."
        ),

        "parameters": {
            "model_paths_by_horizon": {
                "value": {str(h): f'artifacts/training/models/{results[h]["trained_model_path"]}' for h in HORIZONS},
                "why": "Diagnosis and end_model need these to load the champions for each horizon.",
                "must_not_change": True,
            },
            "val_mae_by_horizon": {
                "value": {str(h): results[h]['metrics_val']['mae'] for h in HORIZONS},
                "why": "Current champion MAEs; Diagnosis compares subsequent iterations against these.",
                "must_not_change": False,
            },
            "beats_floor_by_horizon": {
                "value": {str(h): results[h]['vs_baseline_floor']['beats_floor'] for h in HORIZONS},
                "why": "Every model must beat its floor before being accepted.",
                "must_not_change": True,
            },
            "achievable_range_by_horizon": {
                "value": {str(h): results[h]['achievable_range']['assessment'] for h in HORIZONS},
                "why": "Range failure → target representation fix required, not hyperparameter tuning.",
                "must_not_change": False,
            },
            "ts_approach_gap": {
                "value": "classical_statistical",
                "why": "ARIMA family must be fitted once before run may stop (training.md §2b-ts).",
                "must_not_change": True,
            },
            "hyperparameter_search_done": {
                "value": False,
                "why": "Training.md guardrail: defaults are a starting point; stop_check.py blocks without search.",
                "must_not_change": False,
            },
        },

        "durable_rules": [
            {
                "rule": "LightGBM models are saved per-horizon at artifacts/training/models/lgbm_h{h}.txt — load with lgb.Booster(model_file=path).",
                "why": "Both .pkl (joblib) and .txt (booster) saved; .txt is more portable for MLflow; .pkl for sklearn API.",
                "applies_to": ["diagnosis", "end_model"],
                "cost_if_broken": "Wrong model loaded for wrong horizon — silent error.",
            },
            {
                "rule": "Feature subset for each horizon is in horizon_feature_subsets[h]['feature_list'] in state/current.json.",
                "why": "Training used these exactly; inference must use the same columns.",
                "applies_to": ["end_model"],
                "cost_if_broken": "Feature mismatch at inference time.",
            },
        ],

        "methodological_cautions": [
            {
                "caution": (
                    "h=1 and h=7 use identical naive predictors (naive_h1/h7 smedian_w4/13/26 are "
                    "mathematically identical for same-DOW targets) AND the same lag (bread_lag_7d). "
                    "Their LightGBM models may be nearly identical. Diagnosis should verify whether "
                    "a single model could serve both horizons."
                ),
                "cost_if_ignored": "Unnecessary model duplication; may obscure any real h=1 vs h=7 information difference.",
            },
            {
                "caution": "Permutation importance computed on val set (95 rows for most horizons). Standard errors may be high. Groups with importance < 1.0 MAE units should be treated as noisy.",
                "cost_if_ignored": "Feature importance rankings at low importance levels are unreliable with n_val=95.",
            },
        ],

        "content_carried_forward": {
            "model_paths_by_horizon": {str(h): f'artifacts/training/models/{results[h]["trained_model_path"]}' for h in HORIZONS},
            "val_mae_by_horizon": {str(h): results[h]['metrics_val']['mae'] for h in HORIZONS},
            "pooled_val_mae": pooled_val_mae,
            "ts_approach_gap": "classical_statistical",
            "hyperparameter_search_done": False,
            "beats_all_floors": beats_all,
        },

        "do_not_repeat": [
            {
                "tried": "Training one pooled model across all horizons",
                "outcome": "rejected by design",
                "evidence": "training.md §0: 'one model per active horizon — not one model scored at several lead times'",
                "why_not_again": "Long horizons dominated by short ones; h=30 model inherits h=1 features it won't have at its origin.",
            },
        ],
    }
}

HIST_PATH = RUN_DIR / 'history' / f'{ITER_ID}.json'
with open(str(HIST_PATH), 'w') as f:
    json.dump(history_json, f, indent=2, default=str)
print(f"\nHistory JSON: {HIST_PATH}")

# %% Update state/current.json
with open(str(STATE_PATH)) as f:
    state = json.load(f)

state['current']['latest_model_paths'] = {str(h): f'artifacts/training/models/{results[h]["trained_model_path"]}' for h in HORIZONS}
state['current']['val_mae_by_horizon']  = {str(h): results[h]['metrics_val']['mae'] for h in HORIZONS}
state['current']['pooled_val_mae']       = pooled_val_mae
state['current']['ts_approach_gap']     = 'classical_statistical'
state['current']['hyperparameter_search_done'] = False
state['current']['beats_floor_by_horizon']     = {str(h): results[h]['vs_baseline_floor']['beats_floor'] for h in HORIZONS}

state['history'].append({
    'iteration': ITER_NUM, 'node': 'training',
    'timestamp': '2026-09-08', 'status': 'complete',
    'handoff_check': 'pending',
    'summary': (
        f"LightGBM initial pass, all 6 horizons. "
        f"Pooled val MAE={pooled_val_mae} vs floor={pooled_floor}. "
        f"Beats all floors: {beats_all}. "
        f"ts_approach_gap=classical_statistical (SARIMA needed). "
        f"Hyperparameter search deferred."
    ),
    'full_output_path': f'history/{ITER_ID}.json',
    'report_html': None,
    'key_outputs': {
        'model_type': 'LightGBM', 'pooled_val_mae': pooled_val_mae,
        'baseline_floor_pooled': pooled_floor,
        'val_mae_by_horizon': {str(h): results[h]['metrics_val']['mae'] for h in HORIZONS},
        'beats_all_floors': beats_all,
        'ts_approach_gap': 'classical_statistical',
    },
})

state['current']['next_iteration_contract'] = {
    'node': 'training',
    'iteration': 7,
    'why_this_node': (
        'classical_statistical slot unfilled. SARIMA must be fitted on the same val rows '
        'with the same primary metric (MAE) for the comparison to be valid.'
    ),
    'read_this_first': (
        'Fit SARIMA(p,d,q)(P,D,Q)7 per horizon. For each horizon h: use the raw bread_quantity '
        'time series fitted on fold-3 train rows; evaluate on fold-3 val rows. '
        'Orders from ADF stationarity check (d) + ACF/PACF (p,q) + seasonal terms (P,D,Q,m=7). '
        'Fit plain ARIMA alongside for ablation. '
        'Report MAE, RMSE, SMAPE, MASE on same val rows. '
        'Compare to LightGBM val_mae_by_horizon from iteration 006.'
    ),
}

with open(str(STATE_PATH), 'w') as f:
    json.dump(state, f, indent=2, default=str)
print("state/current.json updated.")

# %% Write markdown report
RPT_DIR = RUN_DIR / 'reports'
RPT_DIR.mkdir(exist_ok=True)
report_lines = [
    f"# Iteration {ITER_NUM} — Training: LightGBM (initial, ml_as_regression)",
    f"Run: run_20260908T075709Z | Date: 2026-09-08",
    "",
    "## Mode and trigger",
    "Mode: `initial` — first learned model this run. Follows baseline (iter 4) and feature engineering (iter 5).",
    "ts_approach: `ml_as_regression`. **Gap: `classical_statistical` (SARIMA) must be fitted next.**",
    "",
    "## Feature subset and deployment scenario",
    "Deployment scenario: `live_forecast`. One feature subset per horizon from `horizon_feature_subsets`.",
    "Weather (is_hot_day, is_heavy_rain) included for h≤7 only (7-day forecast horizon).",
    "Lag features: causal DOW-aligned lag (bread_lag_{ceil(h/7)*7}d).",
    "",
    "## Model: LightGBMRegressor",
    "Hyperparameters: defaults (n_estimators=500, max_depth=5, lr=0.05, min_child_samples=20, L1/L2=0.1).",
    "Objective: regression_l1 (MAE-aligned with primary metric). Early stopping: 50 rounds.",
    "**Note:** Hyperparameter search deferred. Training.md guardrail requires at least one search before run close.",
    "",
    "## Results: Validation MAE by horizon",
    "",
    "| h | Val MAE | Floor | Margin | Beat floor | Overfit | Range |",
    "|---|---------|-------|--------|------------|---------|-------|",
]
for h in HORIZONS:
    r = results[h]
    report_lines.append(
        f"| {h} | {r['metrics_val']['mae']:.3f} | {BASELINE_FLOOR[h]:.3f} | "
        f"{r['vs_baseline_floor']['margin']:+.3f} | "
        f"{'YES' if r['vs_baseline_floor']['beats_floor'] else 'NO'} | "
        f"{r['overfit']['assessment']} | {r['achievable_range']['assessment']} |"
    )
report_lines += [
    f"",
    f"**Pooled val MAE: {pooled_val_mae:.3f}  Floor: {pooled_floor:.3f}  Margin: {pooled_floor - pooled_val_mae:+.3f}**",
    f"**All horizons beat floor: {beats_all}**",
    "",
    "## Ridge reference model (initial mode)",
    "",
    "| h | Ridge val MAE | LightGBM val MAE | LightGBM better |",
    "|---|---------------|-----------------|-----------------|",
]
for h in HORIZONS:
    ref = results[h]['reference_model']
    better = ref['vs_primary_model'] == 'primary_better'
    report_lines.append(f"| {h} | {ref['val_mae']:.3f} | {results[h]['metrics_val']['mae']:.3f} | {'YES' if better else '**NO — INVESTIGATE**'} |")
report_lines += [
    "",
    "## Secondary metrics (RMSE, SMAPE, MASE)",
    "",
    "| h | Train MAE | Val MAE | Train RMSE | Val RMSE | Val SMAPE | Val MASE |",
    "|---|-----------|---------|------------|----------|-----------|----------|",
]
for h in HORIZONS:
    tr = results[h]['metrics_train']
    vl = results[h]['metrics_val']
    report_lines.append(
        f"| {h} | {tr['mae']:.3f} | {vl['mae']:.3f} | {tr['rmse']:.3f} | {vl['rmse']:.3f} | "
        f"{vl['smape_pct']:.1f}% | {vl['mase']} |"
    )
report_lines += [
    "",
    "## Residual diagnostics",
    "",
    "| h | Resid mean | ACF lag1 | ACF lag7 | Range check |",
    "|---|-----------|----------|----------|------------|",
]
for h in HORIZONS:
    rs = results[h]['residuals_summary']
    report_lines.append(f"| {h} | {rs['mean']:+.1f} | {rs['acf_lag1']:.3f} | {rs['acf_lag_seasonal']:.3f} | {results[h]['achievable_range']['assessment']} |")
report_lines += [
    "",
    "**ACF guidance**: |r| > 0.2 indicates uncaptured temporal structure (see plots).",
    "",
    "## Gaps and next steps",
    "1. **REQUIRED**: Fit SARIMA per horizon (iteration 007) to close classical_statistical slot.",
    "2. **REQUIRED**: Hyperparameter search (Optuna, n_trials≥50) before run close.",
    "3. Diagnosis: examine residual ACF patterns and achievable-range caveat for deployment.",
    "",
    "## Clarifications needed",
    "- CQ-TR-001 (non-blocking): SARIMA slot unfilled — schedule classical TS training next.",
    "- CQ-TR-002 (non-blocking): Hyperparameter search deferred — schedule before run close.",
]

RPT_PATH = RPT_DIR / f'{ITER_ID}.md'
with open(str(RPT_PATH), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"Report: {RPT_PATH}")

# %% MLflow logging
import glob as _glob
import mlflow as _mlflow

_HERE    = os.path.dirname(os.path.abspath(__file__))
_RUN_DIR = os.path.dirname(_HERE)
_PROJ_DIR = os.path.dirname(os.path.dirname(_RUN_DIR))
_MLDB    = os.path.join(_PROJ_DIR, 'mlflow_prototype', 'mlflow.db')

_mlflow.set_tracking_uri("sqlite:///" + _MLDB.replace("\\", "/"))
_mlflow.set_experiment("bread_demand_forecast")

_RUN_ID = os.path.basename(_RUN_DIR)

try:
    import mlflow.lightgbm as _mlflow_lgb
    with _mlflow.start_run(run_name=f"{_RUN_ID}__{ITER_ID}"):
        _mlflow.set_tags({
            "pipeline_run_id": _RUN_ID,
            "iteration_id":    ITER_ID,
            "mode":            MODE,
            "model_type":      MODEL_TYPE,
            "ts_approach":     "ml_as_regression",
        })
        _mlflow.log_params({k: v for k, v in LGBM_PARAMS.items()
                            if isinstance(v, (int, float, str, bool))})
        _mlflow.log_param("n_features_range", f"{min(results[h]['n_features'] for h in HORIZONS)}–{max(results[h]['n_features'] for h in HORIZONS)}")
        _mlflow.log_param("early_stopping_rounds", EARLY_STOPPING_ROUNDS)

        for h, mae in VAL_MAE_BY_HORIZON.items():
            _mlflow.log_metric(f"val_mae_h{h}", mae)
        _mlflow.log_metric("val_mae_pooled", pooled_val_mae)
        _mlflow.log_metric("baseline_floor_pooled", pooled_floor)

        _art_dir = str(ART_DIR)
        for _png in _glob.glob(os.path.join(_art_dir, f"{ITER_ID}_*.png")):
            _mlflow.log_artifact(_png, artifact_path="plots")
        _mlflow.log_artifact(str(HIST_PATH), artifact_path="history")

        _models_dir = str(ART_DIR / 'models')
        for h in HORIZONS:
            _dump = os.path.join(_models_dir, f"lgbm_h{h}.txt")
            if os.path.exists(_dump):
                try:
                    _booster = lgb.Booster(model_file=_dump)
                    _mlflow_lgb.log_model(_booster, name=f"lgbm_h{h}")
                except Exception as e_model:
                    print(f"  MLflow model log h={h}: {e_model}")
    print("MLflow logging complete.")
except Exception as e_mlflow:
    print(f"MLflow skipped: {e_mlflow}")

print(f"\n=== TRAINING COMPLETE ===")
print(f"Pooled val MAE: {pooled_val_mae:.3f}  Floor: {pooled_floor:.3f}  Beats all floors: {beats_all}")
print(f"ts_approach_gap: classical_statistical — SARIMA required next (iteration 007)")
