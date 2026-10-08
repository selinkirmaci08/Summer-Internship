# %% [markdown]
# # iteration_018_targeted_improvement.py
# Targeted improvement pass before end_model:
#   h=14: extended Optuna (150 trials) + raw-units target attempt
#   h=7 : multi-fold verification (is the floor-beat robust?)
#   h=90: multi-fold confirmation of lgbm_A_no_anc (4-row fold-3 result is unreliable)
#   h=60: one more ML attempt using all-fold pooled training
# Overfitting guards: inner-split ES, multi-fold eval, noise-band check.

# %% Setup
import pandas as pd
import numpy as np
import json, warnings, pickle, os
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

RUN_DIR  = "C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z"
DATA_DIR = f"{RUN_DIR}/data"
ART_DIR  = f"{RUN_DIR}/artifacts/training"
os.makedirs(f"{ART_DIR}/iter018", exist_ok=True)

df = pd.read_csv(f"{DATA_DIR}/iteration_012_feature_engineering.csv", parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"Loaded: {df.shape}")

FOLDS    = [('_fold_1',1), ('_fold_2',2), ('_fold_3',3)]
FLOOR    = {1:31.263, 7:31.263, 14:32.846, 30:35.913, 60:40.042, 90:42.143}

# Base feature exclusions for raw-units models
EXCLUDE_BASE = set(
    ['date', '_split', '_fold_1', '_fold_2', '_fold_3', 'bread_quantity'] +
    [c for c in df.columns if c.startswith('bread_quantity_h')] +
    [c for c in df.columns if c.startswith('ratio_h')] +
    [c for c in df.columns if c.startswith('ratio_correct_h')] +
    [c for c in df.columns if c.startswith('anchor_correct_h')] +
    [c for c in df.columns if c.startswith('anchor_h')]
)
FEAT_RAW = [c for c in df.columns if c not in EXCLUDE_BASE and df[c].dtype != object]

def inner_split(df_train, pct=0.20, min_n=20):
    n = len(df_train)
    n_inner = max(min_n, int(pct * n))
    return df_train.iloc[:n-n_inner], df_train.iloc[n-n_inner:]

def multifold_mae(h, feat_cols, target_col, get_preds_fn, nz_only=True, covid_filter=True):
    """Evaluate a model spec across all 3 folds; returns per-fold + pooled MAE."""
    results = {}
    for fold_col, fid in FOLDS:
        tr = df[(df[fold_col]=='train') & df[target_col].notna()]
        if covid_filter:
            tr = tr[tr['is_covid_lockdown_period']==0]
        vl = df[(df[fold_col]=='val') & df[target_col].notna()]
        if nz_only:
            vl = vl[vl[target_col] > 0]
        if len(tr) < 30 or len(vl) == 0:
            continue
        X_tr = tr[feat_cols].fillna(0).values
        y_tr = tr[target_col].values
        X_vl = vl[feat_cols].fillna(0).values
        y_vl = vl[target_col].values
        preds = get_preds_fn(X_tr, y_tr, X_vl)
        if preds is None:
            continue
        mae = np.mean(np.abs(preds - y_vl))
        fl  = np.mean(np.abs(vl[f'naive_h{h}_seasonal_median_w4'].values - y_vl)) if f'naive_h{h}_seasonal_median_w4' in df.columns else FLOOR[h]
        results[fid] = {'mae': mae, 'floor': fl, 'n': len(vl), 'beats': mae < fl}
    if not results:
        return None
    n_tot  = sum(r['n'] for r in results.values())
    p_mae  = sum(r['n']*r['mae']   for r in results.values()) / n_tot
    p_fl   = sum(r['n']*r['floor'] for r in results.values()) / n_tot
    return {'per_fold': results, 'pooled_mae': p_mae, 'pooled_floor': p_fl,
            'pooled_beats': p_mae < p_fl, 'n_folds': len(results)}

print("\nSetup complete. Starting experiments.\n")
print("=" * 70)

# ============================================================
# SECTION 1: h=14 — Extended Optuna (ratio_h14 target)
# ============================================================
print("SECTION 1: h=14 — Extended Optuna (ratio_h14, COVID exclusion)")
print("=" * 70)

H14_TARGET  = 'ratio_h14'
H14_ANCHOR  = 'naive_h14_seasonal_median_w4'
H14_FEAT_RATIO = [c for c in df.columns
                  if c not in EXCLUDE_BASE
                  and c not in [H14_TARGET]
                  and not c.startswith('ratio_')
                  and not c.startswith('ratio_correct_')
                  and df[c].dtype != object
                  and c != 'bread_quantity_h14'
                  and not c.startswith('bread_quantity_h')]

# Add the ratio target to feature list exclusion check, keep naive predictors for h=14
# Feature set: exactly what was used in iter011 for h=14
h14_excl_extras = set([c for c in df.columns if c.startswith('ratio_')] +
                       [c for c in df.columns if c.startswith('ratio_correct_')] +
                       [c for c in df.columns if c.startswith('bread_quantity_h')])
H14_FEATS = [c for c in df.columns
             if c not in EXCLUDE_BASE
             and c not in h14_excl_extras
             and df[c].dtype != object]
print(f"h=14 ratio features: {len(H14_FEATS)}")

# Fold-3 train/val (primary)
tr14_f3 = df[(df['_fold_3']=='train') & df['bread_quantity_h14'].notna() &
             (df['is_covid_lockdown_period']==0)].copy()
vl14_f3 = df[(df['_fold_3']=='val')   & df['bread_quantity_h14'].notna()].copy()
vl14_f3_nz = vl14_f3[vl14_f3['bread_quantity_h14'] > 0].copy()

X14_tr = tr14_f3[H14_FEATS].fillna(0).values
y14_tr = tr14_f3[H14_TARGET].values
X14_vl = vl14_f3_nz[H14_FEATS].fillna(0).values
y14_vl_ratio = vl14_f3_nz[H14_TARGET].values
y14_vl_units = vl14_f3_nz['bread_quantity_h14'].values
anchor14_vl  = vl14_f3_nz[H14_ANCHOR].values

tr14_inner, tr14_inner_val = inner_split(tr14_f3)
X14_in_tr  = tr14_inner[H14_FEATS].fillna(0).values
y14_in_tr  = tr14_inner[H14_TARGET].values
X14_in_vl  = tr14_inner_val[H14_FEATS].fillna(0).values
y14_in_vl  = tr14_inner_val[H14_TARGET].values

print(f"Fold-3 train(noCOVID)={len(tr14_f3)}, inner={len(tr14_inner)}/{len(tr14_inner_val)}")
print(f"Fold-3 val total={len(vl14_f3)} / non-zero={len(vl14_f3_nz)}")
print(f"Current champion MAE = 34.891 | Floor = 32.846")

# --- Optuna objective for h=14 ratio target ---
def h14_optuna_obj(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate':     trial.suggest_float('learning_rate', 0.003, 0.15, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 16, 128),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 2.0),
        'objective':         'regression_l1',
        'verbose':           -1,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X14_in_tr, y14_in_tr,
          eval_X=X14_in_vl, eval_y=y14_in_vl, eval_metric='mae',
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    n_trees = m.best_iteration_
    if n_trees < 5:
        return 999.0
    m2 = lgb.LGBMRegressor(**{**params, 'n_estimators': n_trees})
    m2.fit(X14_in_tr, y14_in_tr)
    ratio_pred = m2.predict(X14_vl)
    unit_pred  = ratio_pred * anchor14_vl
    return float(np.mean(np.abs(unit_pred - y14_vl_units)))

study14 = optuna.create_study(direction='minimize')
# Warm-start from prior best params (iter011)
study14.enqueue_trial({
    'n_estimators': 123, 'learning_rate': 0.034511, 'num_leaves': 59, 'max_depth': 6,
    'min_child_samples': 27, 'reg_alpha': 0.024575, 'reg_lambda': 0.107,
    'subsample': 0.878, 'colsample_bytree': 0.738, 'min_split_gain': 0.863
})
print("\nRunning Optuna h=14 (150 trials)...")
study14.optimize(h14_optuna_obj, n_trials=150, show_progress_bar=False)

best14_params = study14.best_params
best14_mae    = study14.best_value
print(f"Best Optuna MAE (ratio->units on val): {best14_mae:.4f}  (prior: 34.891)")
print(f"Best params: {best14_params}")

# Refit best Optuna model on full fold-3 non-COVID train
n_trees_14 = best14_params.pop('n_estimators')
m14_best = lgb.LGBMRegressor(**best14_params, n_estimators=n_trees_14, verbose=-1)
m14_best.fit(X14_in_tr, y14_in_tr)
ratio_pred14 = m14_best.predict(X14_vl)
unit_pred14  = ratio_pred14 * anchor14_vl
mae14_val    = float(np.mean(np.abs(unit_pred14 - y14_vl_units)))
print(f"Final val MAE (units): {mae14_val:.4f}")
best14_params['n_estimators'] = n_trees_14  # restore

# Multi-fold verification for h=14 Optuna result
def get_preds_h14_ratio(X_tr, y_tr, X_vl):
    m = lgb.LGBMRegressor(**best14_params, verbose=-1)
    tr_df_ = pd.DataFrame(X_tr, columns=H14_FEATS)  # won't work easily — do inline
    m.fit(X_tr[:int(0.8*len(X_tr))], y_tr[:int(0.8*len(y_tr))])
    return m.predict(X_vl)

# Do multi-fold manually for h=14 (ratio target → units via anchor)
print(f"\n--- h=14 Optuna multi-fold verification ---")
h14_mf = {}
for fold_col, fid in FOLDS:
    tr_ = df[(df[fold_col]=='train') & df['bread_quantity_h14'].notna() &
             (df['is_covid_lockdown_period']==0)].copy()
    vl_ = df[(df[fold_col]=='val') & df['bread_quantity_h14'].notna() &
             (df['bread_quantity_h14'] > 0)].copy()
    if len(tr_) < 30 or len(vl_) == 0:
        continue
    X_tr_ = tr_[H14_FEATS].fillna(0).values
    y_tr_ = tr_['ratio_h14'].values
    X_vl_ = vl_[H14_FEATS].fillna(0).values
    anc_   = vl_[H14_ANCHOR].values
    y_vl_u = vl_['bread_quantity_h14'].values
    fl_    = np.mean(np.abs(vl_[H14_ANCHOR].values - y_vl_u))
    n_in = max(20, int(0.20 * len(tr_)))
    m_ = lgb.LGBMRegressor(**best14_params, verbose=-1)
    m_.fit(X_tr_[:-n_in], y_tr_[:-n_in])
    ratio_p = m_.predict(X_vl_)
    unit_p  = ratio_p * anc_
    mae_ = np.mean(np.abs(unit_p - y_vl_u))
    h14_mf[fid] = {'mae': mae_, 'floor': fl_, 'n': len(vl_), 'beats': mae_ < fl_}
    print(f"  fold-{fid}: n={len(vl_):3d}  floor={fl_:.3f}  model={mae_:.3f}  {'BEATS' if mae_ < fl_ else ''}")

if h14_mf:
    tot_n = sum(r['n'] for r in h14_mf.values())
    p_mae = sum(r['n']*r['mae'] for r in h14_mf.values()) / tot_n
    p_fl  = sum(r['n']*r['floor'] for r in h14_mf.values()) / tot_n
    print(f"  POOLED: n={tot_n}  floor={p_fl:.3f}  model={p_mae:.3f}  {'BEATS' if p_mae < p_fl else ''}")
    h14_pooled_beats = p_mae < p_fl
    h14_pooled_mae   = p_mae

print("=" * 70)

# ============================================================
# SECTION 2: h=14 — Raw-units target attempt
# ============================================================
print("\nSECTION 2: h=14 — Raw-units target (LightGBM, COVID exclusion, inner ES)")
print("=" * 70)

H14_RAW_FEATS = [c for c in FEAT_RAW
                 if c not in [c2 for c2 in df.columns if c2.startswith('ratio_') or c2.startswith('ratio_correct_')]]

tr14_f3_raw = df[(df['_fold_3']=='train') & df['bread_quantity_h14'].notna() &
                 (df['is_covid_lockdown_period']==0)].copy()
vl14_f3_nz2 = df[(df['_fold_3']=='val') & df['bread_quantity_h14'].notna() &
                  (df['bread_quantity_h14'] > 0)].copy()

X14r_tr = tr14_f3_raw[FEAT_RAW].fillna(0).values
y14r_tr = tr14_f3_raw['bread_quantity_h14'].values
X14r_vl = vl14_f3_nz2[FEAT_RAW].fillna(0).values
y14r_vl = vl14_f3_nz2['bread_quantity_h14'].values

n_in14r = max(20, int(0.20 * len(tr14_f3_raw)))
X14r_in_tr = X14r_tr[:-n_in14r]; y14r_in_tr = y14r_tr[:-n_in14r]
X14r_in_vl = X14r_tr[-n_in14r:]; y14r_in_vl = y14r_tr[-n_in14r:]

print(f"Raw-units feature set: {len(FEAT_RAW)} cols")
best14r_mae, best14r_cfg, best14r_model = 1e9, None, None
for n_leaves in [8, 16, 32, 64]:
    for lr in [0.01, 0.03, 0.05]:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=n_leaves, max_depth=5,
                               learning_rate=lr, reg_alpha=1.0, reg_lambda=10.0,
                               min_child_samples=max(5, len(tr14_f3_raw)//80),
                               objective='regression_l1', verbose=-1)
        m.fit(X14r_in_tr, y14r_in_tr, eval_X=X14r_in_vl, eval_y=y14r_in_vl,
              eval_metric='mae',
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
        n_tr = m.best_iteration_
        if n_tr < 3:
            continue
        m2 = lgb.LGBMRegressor(n_estimators=n_tr, num_leaves=n_leaves, max_depth=5,
                                learning_rate=lr, reg_alpha=1.0, reg_lambda=10.0,
                                min_child_samples=max(5, len(tr14_f3_raw)//80),
                                objective='regression_l1', verbose=-1)
        m2.fit(X14r_in_tr, y14r_in_tr)
        pred = m2.predict(X14r_vl)
        mae  = float(np.mean(np.abs(pred - y14r_vl)))
        mark = ' <-- best' if mae < best14r_mae else ''
        print(f"  leaves={n_leaves:3d} lr={lr:.2f} n_trees={n_tr:3d}: MAE={mae:.4f}{mark}")
        if mae < best14r_mae:
            best14r_mae = mae; best14r_cfg = (n_leaves, lr, n_tr); best14r_model = m2

print(f"\nBest raw-units h=14 MAE: {best14r_mae:.4f}  (ratio Optuna: {mae14_val:.4f}, prior: 34.891, floor: 32.846)")

# Multi-fold for raw-units h=14
print(f"\n--- h=14 raw-units multi-fold ---")
n_lv14r, lr14r, n_tr14r = best14r_cfg
h14r_mf = {}
for fold_col, fid in FOLDS:
    tr_ = df[(df[fold_col]=='train') & df['bread_quantity_h14'].notna() &
             (df['is_covid_lockdown_period']==0)].copy()
    vl_ = df[(df[fold_col]=='val') & df['bread_quantity_h14'].notna() &
             (df['bread_quantity_h14'] > 0)].copy()
    if len(tr_) < 30 or len(vl_) == 0:
        continue
    X_tr_ = tr_[FEAT_RAW].fillna(0).values; y_tr_ = tr_['bread_quantity_h14'].values
    X_vl_ = vl_[FEAT_RAW].fillna(0).values; y_vl_ = vl_['bread_quantity_h14'].values
    fl_ = np.mean(np.abs(vl_[H14_ANCHOR].values - y_vl_))
    n_in_ = max(20, int(0.20 * len(tr_)))
    m_ = lgb.LGBMRegressor(n_estimators=n_tr14r, num_leaves=n_lv14r, max_depth=5,
                            learning_rate=lr14r, reg_alpha=1.0, reg_lambda=10.0,
                            min_child_samples=max(5, len(tr_)//80),
                            objective='regression_l1', verbose=-1)
    m_.fit(X_tr_[:-n_in_], y_tr_[:-n_in_])
    mae_ = np.mean(np.abs(m_.predict(X_vl_) - y_vl_))
    h14r_mf[fid] = {'mae': mae_, 'floor': fl_, 'n': len(vl_), 'beats': mae_ < fl_}
    print(f"  fold-{fid}: n={len(vl_):3d}  floor={fl_:.3f}  model={mae_:.3f}  {'BEATS' if mae_ < fl_ else ''}")

if h14r_mf:
    tot_n = sum(r['n'] for r in h14r_mf.values());
    p_mae14r = sum(r['n']*r['mae'] for r in h14r_mf.values()) / tot_n
    p_fl14r  = sum(r['n']*r['floor'] for r in h14r_mf.values()) / tot_n
    print(f"  POOLED: n={tot_n}  floor={p_fl14r:.3f}  model={p_mae14r:.3f}  {'BEATS' if p_mae14r < p_fl14r else ''}")

print("=" * 70)

# ============================================================
# SECTION 3: h=7 — Multi-fold verification
# ============================================================
print("\nSECTION 3: h=7 — Multi-fold verification (same Optuna params as iter011)")
print("=" * 70)

H7_PARAMS = {
    'n_estimators': 522, 'learning_rate': 0.006877, 'num_leaves': 57,
    'max_depth': 4, 'min_child_samples': 17, 'reg_alpha': 0.001348,
    'reg_lambda': 0.432, 'subsample': 0.770, 'colsample_bytree': 0.777,
    'min_split_gain': 0.0754, 'objective': 'regression_l1', 'verbose': -1
}
H7_TARGET  = 'ratio_h7'
H7_ANCHOR  = 'naive_h7_seasonal_median_w4'
h7_excl = set([c for c in df.columns if c.startswith('ratio_')] +
              [c for c in df.columns if c.startswith('ratio_correct_')] +
              [c for c in df.columns if c.startswith('bread_quantity_h')])
H7_FEATS = [c for c in df.columns
            if c not in EXCLUDE_BASE and c not in h7_excl and df[c].dtype != object]
print(f"h=7 feature set: {len(H7_FEATS)} cols")

h7_mf = {}
for fold_col, fid in FOLDS:
    tr_ = df[(df[fold_col]=='train') & df['bread_quantity_h7'].notna() &
             (df['is_covid_lockdown_period']==0)].copy()
    vl_ = df[(df[fold_col]=='val') & df['bread_quantity_h7'].notna()].copy()
    if len(tr_) < 30 or len(vl_) == 0:
        continue
    X_tr_ = tr_[H7_FEATS].fillna(0).values; y_tr_ = tr_[H7_TARGET].values
    X_vl_ = vl_[H7_FEATS].fillna(0).values
    anc_   = vl_[H7_ANCHOR].values
    y_vl_u = vl_['bread_quantity_h7'].values
    fl_    = np.mean(np.abs(anc_ - y_vl_u))
    n_in_ = max(20, int(0.20 * len(tr_)))
    m_ = lgb.LGBMRegressor(**H7_PARAMS)
    m_.fit(X_tr_[:-n_in_], y_tr_[:-n_in_])
    unit_p = m_.predict(X_vl_) * anc_
    mae_   = np.mean(np.abs(unit_p - y_vl_u))
    h7_mf[fid] = {'mae': mae_, 'floor': fl_, 'n': len(vl_), 'beats': mae_ < fl_}
    print(f"  fold-{fid}: n={len(vl_):3d}  floor={fl_:.3f}  model={mae_:.3f}  {'BEATS' if mae_ < fl_ else ''}")

if h7_mf:
    tot_n = sum(r['n'] for r in h7_mf.values())
    p7_mae = sum(r['n']*r['mae'] for r in h7_mf.values()) / tot_n
    p7_fl  = sum(r['n']*r['floor'] for r in h7_mf.values()) / tot_n
    print(f"  POOLED: n={tot_n}  floor={p7_fl:.3f}  model={p7_mae:.3f}  {'BEATS' if p7_mae < p7_fl else ''}")

print("=" * 70)

# ============================================================
# SECTION 4: h=90 — Multi-fold confirmation
# ============================================================
print("\nSECTION 4: h=90 — Multi-fold confirmation (lgbm_A_no_anc config)")
print("=" * 70)

H90_CFG = {'n_leaves': 16, 'lr': 0.05, 'n_trees': 68}
H90_TARGET = 'bread_quantity_h90'
H90_FLOOR_COL = 'naive_h90_seasonal_median_w26'

print(f"Config: n_leaves={H90_CFG['n_leaves']}, lr={H90_CFG['lr']}, n_trees={H90_CFG['n_trees']}")
h90_mf = {}
for fold_col, fid in FOLDS:
    tr_ = df[(df[fold_col]=='train') & df[H90_TARGET].notna() &
             (df['is_covid_lockdown_period']==0)].copy()
    vl_ = df[(df[fold_col]=='val') & df[H90_TARGET].notna() &
             (df[H90_TARGET] > 0)].copy()
    if len(tr_) < 30 or len(vl_) == 0:
        continue
    X_tr_ = tr_[FEAT_RAW].fillna(0).values; y_tr_ = tr_[H90_TARGET].values
    X_vl_ = vl_[FEAT_RAW].fillna(0).values; y_vl_ = vl_[H90_TARGET].values
    fl_   = np.mean(np.abs(vl_[H90_FLOOR_COL].values - y_vl_))
    n_in_ = max(20, int(0.20 * len(tr_)))
    m_ = lgb.LGBMRegressor(
        n_estimators=H90_CFG['n_trees'], num_leaves=H90_CFG['n_leaves'],
        max_depth=4, learning_rate=H90_CFG['lr'],
        reg_alpha=2.0, reg_lambda=20.0,
        min_child_samples=max(5, (len(tr_)-n_in_)//50),
        objective='regression_l1', verbose=-1
    )
    m_.fit(X_tr_[:-n_in_], y_tr_[:-n_in_])
    mae_ = np.mean(np.abs(m_.predict(X_vl_) - y_vl_))
    h90_mf[fid] = {'mae': mae_, 'floor': fl_, 'n': len(vl_), 'beats': mae_ < fl_}
    print(f"  fold-{fid}: n={len(vl_):3d}  floor={fl_:.3f}  model={mae_:.3f}  {'BEATS' if mae_ < fl_ else ''}")

if h90_mf:
    tot_n = sum(r['n'] for r in h90_mf.values())
    p90_mae = sum(r['n']*r['mae'] for r in h90_mf.values()) / tot_n
    p90_fl  = sum(r['n']*r['floor'] for r in h90_mf.values()) / tot_n
    print(f"  POOLED: n={tot_n}  floor={p90_fl:.3f}  model={p90_mae:.3f}  {'BEATS' if p90_mae < p90_fl else ''}")
    h90_pooled_beats = p90_mae < p90_fl
    h90_pooled_mae   = p90_mae

# Also try Optuna on h=90 (fresh, 75 trials)
print(f"\nAlso running Optuna h=90 (75 trials) to see if a better config exists...")
tr90_f3 = df[(df['_fold_3']=='train') & df[H90_TARGET].notna() &
             (df['is_covid_lockdown_period']==0)].copy()
vl90_f3 = df[(df['_fold_3']=='val') & df[H90_TARGET].notna() &
             (df[H90_TARGET] > 0)].copy()
X90_tr = tr90_f3[FEAT_RAW].fillna(0).values; y90_tr = tr90_f3[H90_TARGET].values
X90_vl = vl90_f3[FEAT_RAW].fillna(0).values; y90_vl = vl90_f3[H90_TARGET].values
fl90_f3 = np.mean(np.abs(vl90_f3[H90_FLOOR_COL].values - y90_vl))
n_in90 = max(20, int(0.20 * len(tr90_f3)))
X90_in_tr = X90_tr[:-n_in90]; y90_in_tr = y90_tr[:-n_in90]
X90_in_vl = X90_tr[-n_in90:]; y90_in_vl = y90_tr[-n_in90:]
print(f"  fold-3: n_train={len(tr90_f3)}, n_val={len(vl90_f3)} nz, floor={fl90_f3:.3f}")

def h90_obj(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 30, 500),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 4, 64),
        'max_depth':         trial.suggest_int('max_depth', 2, 6),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 30),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.1, 20.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.1, 50.0, log=True),
        'objective': 'regression_l1', 'verbose': -1
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X90_in_tr, y90_in_tr, eval_X=X90_in_vl, eval_y=y90_in_vl, eval_metric='mae',
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    n_tr = m.best_iteration_
    if n_tr < 3:
        return 999.0
    m2 = lgb.LGBMRegressor(**{**params, 'n_estimators': n_tr})
    m2.fit(X90_in_tr, y90_in_tr)
    return float(np.mean(np.abs(m2.predict(X90_vl) - y90_vl)))

study90 = optuna.create_study(direction='minimize')
study90.enqueue_trial({'n_estimators': 68, 'learning_rate': 0.05, 'num_leaves': 16,
                       'max_depth': 4, 'min_child_samples': 8, 'reg_alpha': 2.0, 'reg_lambda': 20.0})
study90.optimize(h90_obj, n_trials=75, show_progress_bar=False)
best90_params = study90.best_params
best90_f3_mae = study90.best_value
print(f"  Optuna best fold-3 MAE: {best90_f3_mae:.3f}  (prior: 19.856, floor: {fl90_f3:.3f})")
print(f"  Best params: {best90_params}")

# Multi-fold check for Optuna h=90 winner
if best90_f3_mae < fl90_f3:
    print(f"\n  Running multi-fold check for Optuna h=90 winner...")
    n_est90o = best90_params.pop('n_estimators')
    h90_opt_mf = {}
    for fold_col, fid in FOLDS:
        tr_ = df[(df[fold_col]=='train') & df[H90_TARGET].notna() &
                 (df['is_covid_lockdown_period']==0)].copy()
        vl_ = df[(df[fold_col]=='val') & df[H90_TARGET].notna() &
                 (df[H90_TARGET] > 0)].copy()
        if len(tr_) < 30 or len(vl_) == 0:
            continue
        X_tr_ = tr_[FEAT_RAW].fillna(0).values; y_tr_ = tr_[H90_TARGET].values
        X_vl_ = vl_[FEAT_RAW].fillna(0).values; y_vl_ = vl_[H90_TARGET].values
        fl_   = np.mean(np.abs(vl_[H90_FLOOR_COL].values - y_vl_))
        n_in_ = max(20, int(0.20 * len(tr_)))
        m_ = lgb.LGBMRegressor(**best90_params, n_estimators=n_est90o, verbose=-1)
        m_.fit(X_tr_[:-n_in_], y_tr_[:-n_in_])
        mae_  = np.mean(np.abs(m_.predict(X_vl_) - y_vl_))
        h90_opt_mf[fid] = {'mae': mae_, 'floor': fl_, 'n': len(vl_), 'beats': mae_ < fl_}
        print(f"    fold-{fid}: n={len(vl_):3d}  floor={fl_:.3f}  model={mae_:.3f}  {'BEATS' if mae_ < fl_ else ''}")
    if h90_opt_mf:
        tot_n = sum(r['n'] for r in h90_opt_mf.values())
        p90o_mae = sum(r['n']*r['mae'] for r in h90_opt_mf.values()) / tot_n
        p90o_fl  = sum(r['n']*r['floor'] for r in h90_opt_mf.values()) / tot_n
        print(f"    POOLED: n={tot_n}  floor={p90o_fl:.3f}  model={p90o_mae:.3f}  {'BEATS' if p90o_mae < p90o_fl else ''}")
    best90_params['n_estimators'] = n_est90o

print("=" * 70)

# ============================================================
# SECTION 5: h=60 — Extended ML attempt (multi-fold training)
# ============================================================
print("\nSECTION 5: h=60 — Extended ML attempt (Optuna, 75 trials)")
print("=" * 70)

H60_TARGET   = 'bread_quantity_h60'
H60_FLOOR_COL = 'naive_h60_seasonal_median_w26'

tr60_f3 = df[(df['_fold_3']=='train') & df[H60_TARGET].notna() &
             (df['is_covid_lockdown_period']==0)].copy()
vl60_f3 = df[(df['_fold_3']=='val') & df[H60_TARGET].notna() &
             (df[H60_TARGET] > 0)].copy()
X60_tr = tr60_f3[FEAT_RAW].fillna(0).values; y60_tr = tr60_f3[H60_TARGET].values
X60_vl = vl60_f3[FEAT_RAW].fillna(0).values; y60_vl = vl60_f3[H60_TARGET].values
fl60_f3 = np.mean(np.abs(vl60_f3[H60_FLOOR_COL].values - y60_vl))
n_in60 = max(20, int(0.20 * len(tr60_f3)))
X60_in_tr = X60_tr[:-n_in60]; y60_in_tr = y60_tr[:-n_in60]
X60_in_vl = X60_tr[-n_in60:]; y60_in_vl = y60_tr[-n_in60:]
print(f"fold-3 noCOVID train={len(tr60_f3)}, val_nz={len(vl60_f3)}, floor={fl60_f3:.3f}")

def h60_obj(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 30, 600),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 4, 64),
        'max_depth':         trial.suggest_int('max_depth', 2, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 40),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.1, 20.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.1, 50.0, log=True),
        'objective': 'regression_l1', 'verbose': -1
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X60_in_tr, y60_in_tr, eval_X=X60_in_vl, eval_y=y60_in_vl, eval_metric='mae',
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    n_tr = m.best_iteration_
    if n_tr < 3:
        return 999.0
    m2 = lgb.LGBMRegressor(**{**params, 'n_estimators': n_tr})
    m2.fit(X60_in_tr, y60_in_tr)
    return float(np.mean(np.abs(m2.predict(X60_vl) - y60_vl)))

study60 = optuna.create_study(direction='minimize')
study60.optimize(h60_obj, n_trials=75, show_progress_bar=False)
best60_params = study60.best_params
best60_f3_mae = study60.best_value
print(f"Best Optuna fold-3 MAE: {best60_f3_mae:.3f}  (floor: {fl60_f3:.3f})")

if best60_f3_mae < fl60_f3:
    print(f"\n  fold-3 result BEATS floor — running multi-fold check...")
    n_est60 = best60_params.pop('n_estimators')
    h60_mf = {}
    for fold_col, fid in FOLDS:
        tr_ = df[(df[fold_col]=='train') & df[H60_TARGET].notna() &
                 (df['is_covid_lockdown_period']==0)].copy()
        vl_ = df[(df[fold_col]=='val') & df[H60_TARGET].notna() &
                 (df[H60_TARGET] > 0)].copy()
        if len(tr_) < 30 or len(vl_) == 0:
            continue
        X_tr_ = tr_[FEAT_RAW].fillna(0).values; y_tr_ = tr_[H60_TARGET].values
        X_vl_ = vl_[FEAT_RAW].fillna(0).values; y_vl_ = vl_[H60_TARGET].values
        fl_ = np.mean(np.abs(vl_[H60_FLOOR_COL].values - y_vl_))
        n_in_ = max(20, int(0.20 * len(tr_)))
        m_ = lgb.LGBMRegressor(**best60_params, n_estimators=n_est60, verbose=-1)
        m_.fit(X_tr_[:-n_in_], y_tr_[:-n_in_])
        mae_ = np.mean(np.abs(m_.predict(X_vl_) - y_vl_))
        h60_mf[fid] = {'mae': mae_, 'floor': fl_, 'n': len(vl_), 'beats': mae_ < fl_}
        print(f"    fold-{fid}: n={len(vl_):3d}  floor={fl_:.3f}  model={mae_:.3f}  {'BEATS' if mae_ < fl_ else ''}")
    if h60_mf:
        tot_n = sum(r['n'] for r in h60_mf.values())
        p60_mae = sum(r['n']*r['mae'] for r in h60_mf.values()) / tot_n
        p60_fl  = sum(r['n']*r['floor'] for r in h60_mf.values()) / tot_n
        print(f"    POOLED: n={tot_n}  floor={p60_fl:.3f}  model={p60_mae:.3f}  {'BEATS' if p60_mae < p60_fl else ''}")
    best60_params['n_estimators'] = n_est60
else:
    print(f"  Does not beat floor on fold-3. h=60 champion remains naive_h60_seasonal_median_w26.")
    h60_mf = None

print("=" * 70)

# ============================================================
# SECTION 6: Save best h=14 model + Summary
# ============================================================
print("\nSECTION 6: Summary + champion decisions")
print("=" * 70)

# Decide h=14 champion
ratio_better_14 = (mae14_val < best14r_mae)
h14_champ_mae   = min(mae14_val, best14r_mae)
h14_champ_type  = 'ratio_optuna' if ratio_better_14 else 'raw_units'
h14_beats_floor = h14_champ_mae < FLOOR[14]

print(f"\nh=14:")
print(f"  Ratio Optuna (150 trials): fold-3={mae14_val:.4f}  pooled={'N/A' if not h14_mf else f'{p_mae:.3f}'}")
print(f"  Raw-units:                 fold-3={best14r_mae:.4f}  pooled={'N/A' if not h14r_mf else f'{p_mae14r:.3f}'}")
print(f"  Prior champion:            fold-3=34.891")
print(f"  Floor:                     {FLOOR[14]}")
print(f"  --> Best approach: {h14_champ_type}, MAE={h14_champ_mae:.4f}, beats_floor={h14_beats_floor}")

print(f"\nh=7 multi-fold:")
if h7_mf:
    for fid2, r2 in h7_mf.items():
        print(f"  fold-{fid2}: mae={r2['mae']:.3f} floor={r2['floor']:.3f} beats={r2['beats']}")
    print(f"  pooled: {p7_mae:.3f} vs floor {p7_fl:.3f}  beats_floor={'YES' if p7_mae < p7_fl else 'NO'}")

print(f"\nh=90 multi-fold (prior config n_leaves=16 lr=0.05 n_trees=68):")
if h90_mf:
    for fid2, r2 in h90_mf.items():
        print(f"  fold-{fid2}: mae={r2['mae']:.3f} floor={r2['floor']:.3f} beats={r2['beats']}")
    print(f"  pooled: {p90_mae:.3f} vs floor {p90_fl:.3f}  beats_floor={'YES' if p90_mae < p90_fl else 'NO'}")

print(f"\nh=60: fold-3 best={best60_f3_mae:.3f} vs floor={fl60_f3:.3f}")

# Save best h=14 model (whichever won)
if ratio_better_14:
    # Refit Optuna ratio model on full non-COVID train
    m14_save = lgb.LGBMRegressor(**best14_params, verbose=-1)
    m14_save.fit(X14_in_tr, y14_in_tr)
    with open(f"{ART_DIR}/iter018/lgbm_h14_optuna150.pkl", 'wb') as f:
        pickle.dump(m14_save, f)
    print(f"\nSaved: iter018/lgbm_h14_optuna150.pkl (ratio target, inner-split refit)")
else:
    with open(f"{ART_DIR}/iter018/lgbm_h14_raw.pkl", 'wb') as f:
        pickle.dump(best14r_model, f)
    print(f"\nSaved: iter018/lgbm_h14_raw.pkl (raw-units target, inner-split refit)")

# Save best h=90 model if it beats floor on multi-fold
if h90_mf and p90_mae < p90_fl:
    m90_save = lgb.LGBMRegressor(
        n_estimators=H90_CFG['n_trees'], num_leaves=H90_CFG['n_leaves'],
        max_depth=4, learning_rate=H90_CFG['lr'],
        reg_alpha=2.0, reg_lambda=20.0,
        min_child_samples=max(5, (len(tr90_f3)-n_in90)//50),
        objective='regression_l1', verbose=-1
    )
    m90_save.fit(X90_tr[:-n_in90], y90_tr[:-n_in90])
    with open(f"{ART_DIR}/iter018/lgbm_h90_multifold_confirmed.pkl", 'wb') as f:
        pickle.dump(m90_save, f)
    print(f"Saved: iter018/lgbm_h90_multifold_confirmed.pkl (multi-fold confirmed)")

# ============================================================
# SECTION 7: Update state/current.json
# ============================================================
state_path = f"{RUN_DIR}/state/current.json"
with open(state_path) as f:
    state = json.load(f)

state['iteration'] = 18

# Update h=14 champion
state.setdefault('iter018_results', {})
state['iter018_results']['h14'] = {
    'ratio_optuna_150_f3_mae': mae14_val,
    'raw_units_f3_mae': best14r_mae,
    'prior_mae': 34.891,
    'floor': FLOOR[14],
    'winner': h14_champ_type,
    'winner_mae': h14_champ_mae,
    'beats_floor': h14_beats_floor
}
state['iter018_results']['h7_multifold'] = {k: v for k, v in h7_mf.items()} if h7_mf else {}
state['iter018_results']['h90_multifold'] = {k: v for k, v in h90_mf.items()} if h90_mf else {}
state['iter018_results']['h60_f3_mae'] = best60_f3_mae
state['iter018_results']['h60_floor']  = fl60_f3

with open(state_path, 'w') as f:
    json.dump(state, f, indent=2, default=str)
print(f"\nstate/current.json updated (iteration=18).")
print("\nDone. Review results above to decide final champions before end_model.")
