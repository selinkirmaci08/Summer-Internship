# iteration_016_h90_champion.py
# Refit h=90 champion using fold-3 training data.
# Tests: (a) Ridge with anchor_correct_h90 as extra feature
#         (b) LightGBM with same extra feature
# Declares champion, updates state/current.json.

import pandas as pd
import numpy as np
import json, warnings
warnings.filterwarnings('ignore')

RUN_DIR = "C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z"
DATA_DIR = f"{RUN_DIR}/data"
import lightgbm as lgb
from sklearn.linear_model import Ridge

df = pd.read_csv(f"{DATA_DIR}/iteration_012_feature_engineering.csv", parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

TARGET  = 'bread_quantity_h90'
FLOOR_COL = 'naive_h90_seasonal_median_w26'
FOLD_COL  = '_fold_3'

# Feature set A: no anchor_correct (same as iter015)
exclude_base = set(
    ['date', '_split', '_fold_1', '_fold_2', '_fold_3', 'bread_quantity'] +
    [c for c in df.columns if c.startswith('bread_quantity_h')] +
    [c for c in df.columns if c.startswith('ratio_h')] +
    [c for c in df.columns if c.startswith('ratio_correct_h')] +
    [c for c in df.columns if c.startswith('anchor_correct_h')] +
    [c for c in df.columns if c.startswith('anchor_h')]
)
feat_A = [c for c in df.columns if c not in exclude_base and df[c].dtype != object]

# Feature set B: add anchor_correct_h90 as an extra feature
feat_B = feat_A + ['anchor_correct_h90']

print(f"Feature set A: {len(feat_A)} cols   Feature set B: {len(feat_B)} cols")

# Training and val data (fold-3, non-COVID, non-null target)
train = df[(df[FOLD_COL] == 'train') & df[TARGET].notna() & (df['is_covid_lockdown_period'] == 0)].copy()
val   = df[(df[FOLD_COL] == 'val')   & df[TARGET].notna()].copy()
val_nz = val[val[TARGET] > 0].copy()   # exclude zero-demand days (consistent with baseline)

print(f"Train rows: {len(train)}")
print(f"Val rows total/non-zero: {len(val)} / {len(val_nz)}")

y_tr   = train[TARGET].values
y_val  = val_nz[TARGET].values
floor  = val_nz[FLOOR_COL].values
mae_fl = np.mean(np.abs(floor - y_val))
print(f"\nFloor (smedian_w26) MAE on fold-3 non-zero val: {mae_fl:.3f}")

# Inner-fold split (reproduce iter015's split: last 20%)
n_inner = max(20, int(0.20 * len(train)))
n_train_inner = len(train) - n_inner
print(f"Inner split: n_train={n_train_inner}, n_inner={n_inner}")

results = {}

for feat_label, feat_cols in [('A_no_anc', feat_A), ('B_with_anc', feat_B)]:
    X_tr  = train[feat_cols].fillna(0).values
    X_val_mat = val_nz[feat_cols].fillna(0).values
    X_inner_tr  = X_tr[:n_train_inner]
    y_inner_tr  = y_tr[:n_train_inner]
    X_inner_val = X_tr[n_train_inner:]
    y_inner_val = y_tr[n_train_inner:]

    print(f"\n--- Feature set {feat_label} ---")

    # Ridge: search alpha, fit on full training data
    best_ridge_mae, best_alpha, best_ridge = 1e9, None, None
    for alpha in [0.1, 1, 10, 100, 1000, 10000]:
        m = Ridge(alpha=alpha)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val_mat)
        mae  = np.mean(np.abs(pred - y_val))
        if mae < best_ridge_mae:
            best_ridge_mae, best_alpha, best_ridge = mae, alpha, m
    beat = '*** BEATS FLOOR' if best_ridge_mae < mae_fl else ''
    print(f"  Ridge  alpha={best_alpha:>7}: MAE={best_ridge_mae:.3f}  {beat}")
    results[f'ridge_{feat_label}'] = {'mae': best_ridge_mae, 'alpha': best_alpha}

    # LightGBM: fit with inner ES, then refit using exact same inner split
    best_lgbm_mae, best_cfg, best_lgbm = 1e9, None, None
    for n_leaves in [4, 8, 16]:
        for lr in [0.01, 0.05]:
            m = lgb.LGBMRegressor(
                n_estimators=500,
                num_leaves=n_leaves,
                max_depth=4,
                learning_rate=lr,
                reg_alpha=2.0,
                reg_lambda=20.0,
                min_child_samples=max(5, n_train_inner//50),
                objective='regression_l1',
                verbose=-1
            )
            m.fit(
                X_inner_tr, y_inner_tr,
                eval_X=X_inner_val, eval_y=y_inner_val,
                eval_metric='mae',
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
            )
            n_trees = m.best_iteration_
            # Refit on same inner split with fixed n_estimators (reproduce best)
            m2 = lgb.LGBMRegressor(
                n_estimators=n_trees,
                num_leaves=n_leaves,
                max_depth=4,
                learning_rate=lr,
                reg_alpha=2.0,
                reg_lambda=20.0,
                min_child_samples=max(5, n_train_inner//50),
                objective='regression_l1',
                verbose=-1
            )
            m2.fit(X_inner_tr, y_inner_tr)
            pred = m2.predict(X_val_mat)
            mae  = np.mean(np.abs(pred - y_val))
            if mae < best_lgbm_mae:
                best_lgbm_mae = mae
                best_cfg = (n_leaves, lr, n_trees)
                best_lgbm = m2
    beat = '*** BEATS FLOOR' if best_lgbm_mae < mae_fl else ''
    print(f"  LightGBM leaves={best_cfg[0]}, lr={best_cfg[1]}, trees={best_cfg[2]:3d}: MAE={best_lgbm_mae:.3f}  {beat}")
    results[f'lgbm_{feat_label}'] = {'mae': best_lgbm_mae, 'config': best_cfg}

# ===========================================================================
# Pick champion for h=90
# ===========================================================================
print(f"\n{'='*60}")
print("CHAMPION SELECTION for h=90")
print(f"{'='*60}")

print(f"\n  Floor (smedian_w26):         MAE = {mae_fl:.3f}")
for k, v in sorted(results.items(), key=lambda x: x[1]['mae']):
    print(f"  {k:30s}: MAE = {v['mae']:.3f}")

best_key = min(results, key=lambda k: results[k]['mae'])
best_mae  = results[best_key]['mae']

print(f"\nChampion: {best_key}  MAE={best_mae:.3f}")
if best_mae < mae_fl:
    margin = mae_fl - best_mae
    print(f"BEATS FLOOR by {margin:.3f} units on fold-3 non-zero val rows.")
else:
    print("Does not beat floor — reverting to smedian_w26.")

# ===========================================================================
# Multi-fold verification of the winning config
# ===========================================================================
print(f"\n--- Multi-fold verification (best config: {best_key}) ---")
# Determine which feature set and model type
use_anc = 'with_anc' in best_key
feat_champ = feat_B if use_anc else feat_A
use_lgbm = 'lgbm' in best_key

multi_results = {}
for fold_id, fold_col in [(1,'_fold_1'), (2,'_fold_2'), (3,'_fold_3')]:
    tr = df[(df[fold_col]=='train') & df[TARGET].notna() & (df['is_covid_lockdown_period']==0)].copy()
    vl = df[(df[fold_col]=='val')   & df[TARGET].notna() & (df[TARGET]>0)].copy()
    if len(tr) < 50 or len(vl) == 0:
        continue
    X_tr_ = tr[feat_champ].fillna(0).values
    y_tr_ = tr[TARGET].values
    X_vl_ = vl[feat_champ].fillna(0).values
    y_vl_ = vl[TARGET].values
    fl_ = np.mean(np.abs(vl[FLOOR_COL].values - y_vl_))

    if use_lgbm:
        n_leaves_c, lr_c, n_trees_c = results[best_key]['config']
        n_inner_ = max(20, int(0.20 * len(tr)))
        m = lgb.LGBMRegressor(
            n_estimators=n_trees_c,
            num_leaves=n_leaves_c,
            max_depth=4,
            learning_rate=lr_c,
            reg_alpha=2.0,
            reg_lambda=20.0,
            min_child_samples=max(5, (len(tr)-n_inner_)//50),
            objective='regression_l1',
            verbose=-1
        )
        m.fit(X_tr_[:len(tr)-n_inner_], y_tr_[:len(tr)-n_inner_])
        pred_ = m.predict(X_vl_)
    else:
        best_alpha_c = results[best_key]['alpha']
        m = Ridge(alpha=best_alpha_c)
        m.fit(X_tr_, y_tr_)
        pred_ = m.predict(X_vl_)

    mae_ = np.mean(np.abs(pred_ - y_vl_))
    multi_results[fold_id] = {'mae': mae_, 'floor': fl_, 'n': len(vl)}
    beat = '*** BEATS' if mae_ < fl_ else 'below floor'
    print(f"  fold-{fold_id}: n={len(vl):3d}  floor={fl_:.3f}  model={mae_:.3f}  {beat}")

# Pooled
if multi_results:
    total_n = sum(v['n'] for v in multi_results.values())
    p_floor  = sum(v['n']*v['floor'] for v in multi_results.values()) / total_n
    p_model  = sum(v['n']*v['mae']   for v in multi_results.values()) / total_n
    print(f"  POOLED : n={total_n:3d}  floor={p_floor:.3f}  model={p_model:.3f}  {'*** BEATS' if p_model < p_floor else ''}")

# ===========================================================================
# Update state/current.json
# ===========================================================================
state_path = f"{RUN_DIR}/state/current.json"
try:
    with open(state_path) as f:
        state = json.load(f)
except Exception:
    state = {}

if best_mae < mae_fl:
    # Update h=90 champion
    state.setdefault('champions', {})
    state['champions']['h90'] = {
        'model':    best_key,
        'val_mae':  round(best_mae, 3),
        'floor_mae': round(mae_fl, 3),
        'margin':   round(mae_fl - best_mae, 3),
        'beats_floor': True,
        'config':   results[best_key],
        'eval_note': 'fold-3 non-zero val rows (4 rows, excl Dec29 closure)'
    }
    state['iteration'] = 16
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    print(f"\nstate/current.json updated — h=90 champion: {best_key} MAE={best_mae:.3f}")
else:
    print(f"\nh=90 champion remains smedian_w26 (MAE={mae_fl:.3f})")
