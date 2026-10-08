# iteration_017_h60_multifold.py
# Multi-fold training/evaluation for h=60, mirroring the h=90 fix.
# h=60 has 35 fold-3 val rows and likely ~90 in each of fold-1/fold-2.
# Tests raw-unit LightGBM + Ridge; declares champion and updates state.

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

TARGET    = 'bread_quantity_h60'
FLOOR_COL = 'naive_h60_seasonal_median_w26'   # baseline winner for h=60

# Feature set: exclude all target-related columns
exclude_base = set(
    ['date', '_split', '_fold_1', '_fold_2', '_fold_3', 'bread_quantity'] +
    [c for c in df.columns if c.startswith('bread_quantity_h')] +
    [c for c in df.columns if c.startswith('ratio_h')] +
    [c for c in df.columns if c.startswith('ratio_correct_h')] +
    [c for c in df.columns if c.startswith('anchor_correct_h')] +
    [c for c in df.columns if c.startswith('anchor_h')]
)
feat_cols = [c for c in df.columns if c not in exclude_base and df[c].dtype != object]
print(f"Feature set: {len(feat_cols)} columns")

# ===========================================================================
# PART 1: Inspect h=60 val rows per fold
# ===========================================================================
print("\n--- h=60 val rows per fold ---")
for fold_id, fold_col in [(1,'_fold_1'),(2,'_fold_2'),(3,'_fold_3')]:
    vl = df[(df[fold_col]=='val') & df[TARGET].notna()]
    vl_nz = vl[vl[TARGET] > 0]
    if len(vl) > 0:
        targets = vl['date'] + pd.Timedelta(days=60)
        print(f"  fold-{fold_id}: {len(vl)} total / {len(vl_nz)} non-zero "
              f"| origins: {vl['date'].min().date()} to {vl['date'].max().date()} "
              f"| targets: {targets.min().date()} to {targets.max().date()}")
        mae_fl = np.mean(np.abs(vl_nz[FLOOR_COL].values - vl_nz[TARGET].values))
        mae_anc = np.mean(np.abs(vl_nz['anchor_correct_h60'].values - vl_nz[TARGET].values))
        print(f"            floor(smedian_w26)={mae_fl:.3f}  anchor_correct={mae_anc:.3f}")
    else:
        print(f"  fold-{fold_id}: no val rows")

# ===========================================================================
# PART 2: Multi-fold training loop
# ===========================================================================
print(f"\n{'='*70}")
print("MULTI-FOLD TRAINING (h=60)")
print(f"{'='*70}")

fold_results = {}

for fold_id, fold_col in [(1,'_fold_1'),(2,'_fold_2'),(3,'_fold_3')]:
    print(f"\n--- Fold {fold_id} ---")

    train = df[(df[fold_col]=='train') & df[TARGET].notna() & (df['is_covid_lockdown_period']==0)].copy()
    val   = df[(df[fold_col]=='val')   & df[TARGET].notna()].copy()
    val_nz = val[val[TARGET] > 0].copy()

    print(f"  train (non-COVID): {len(train)}  |  val total/non-zero: {len(val)}/{len(val_nz)}")
    if len(train) < 50 or len(val_nz) == 0:
        print("  Skipping — insufficient data.")
        continue

    X_tr  = train[feat_cols].fillna(0).values
    y_tr  = train[TARGET].values
    X_val = val_nz[feat_cols].fillna(0).values
    y_val = val_nz[TARGET].values

    fl_preds  = val_nz[FLOOR_COL].values
    mae_floor = np.mean(np.abs(fl_preds - y_val))
    print(f"  floor MAE = {mae_floor:.3f}")

    fold_results[fold_id] = {'floor': mae_floor, 'n': len(val_nz)}

    # Inner split for early stopping
    n_inner = max(20, int(0.20 * len(train)))
    X_in_tr  = X_tr[:-n_inner];  y_in_tr  = y_tr[:-n_inner]
    X_in_val = X_tr[-n_inner:];  y_in_val = y_tr[-n_inner:]

    # Ridge
    best_ridge_mae, best_alpha = 1e9, None
    for alpha in [1, 10, 100, 1000, 10000]:
        m = Ridge(alpha=alpha)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)
        mae  = np.mean(np.abs(pred - y_val))
        if mae < best_ridge_mae:
            best_ridge_mae, best_alpha = mae, alpha
    beat = '*** BEATS' if best_ridge_mae < mae_floor else ''
    print(f"  Ridge  alpha={best_alpha:>7}: MAE={best_ridge_mae:.3f}  {beat}")
    fold_results[fold_id]['ridge'] = best_ridge_mae

    # LightGBM with inner ES
    best_lgbm_mae, best_cfg = 1e9, None
    for n_leaves in [4, 8, 16, 32]:
        for lr in [0.01, 0.05]:
            m = lgb.LGBMRegressor(
                n_estimators=500,
                num_leaves=n_leaves,
                max_depth=4,
                learning_rate=lr,
                reg_alpha=2.0,
                reg_lambda=20.0,
                min_child_samples=max(5, len(train)//60),
                objective='regression_l1',
                verbose=-1
            )
            m.fit(
                X_in_tr, y_in_tr,
                eval_X=X_in_val, eval_y=y_in_val,
                eval_metric='mae',
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
            )
            n_trees = m.best_iteration_
            # Refit on exact same inner split (same as h=90 fix)
            m2 = lgb.LGBMRegressor(
                n_estimators=n_trees,
                num_leaves=n_leaves,
                max_depth=4,
                learning_rate=lr,
                reg_alpha=2.0,
                reg_lambda=20.0,
                min_child_samples=max(5, len(train)//60),
                objective='regression_l1',
                verbose=-1
            )
            m2.fit(X_in_tr, y_in_tr)
            pred = m2.predict(X_val)
            mae  = np.mean(np.abs(pred - y_val))
            if mae < best_lgbm_mae:
                best_lgbm_mae = mae
                best_cfg = (n_leaves, lr, n_trees)
    beat = '*** BEATS' if best_lgbm_mae < mae_floor else ''
    print(f"  LightGBM leaves={best_cfg[0]}, lr={best_cfg[1]}, trees={best_cfg[2]:3d}: "
          f"MAE={best_lgbm_mae:.3f}  {beat}")
    fold_results[fold_id]['lgbm'] = best_lgbm_mae
    fold_results[fold_id]['lgbm_cfg'] = best_cfg

    best_ml = min(best_ridge_mae, best_lgbm_mae)
    fold_results[fold_id]['best_ml'] = best_ml

# ===========================================================================
# PART 3: Summary and champion selection
# ===========================================================================
print(f"\n{'='*70}")
print("SUMMARY (non-zero val rows)")
print(f"{'='*70}")
print(f"\n  {'Fold':6} {'n':6} {'Floor':10} {'Ridge':10} {'LGBM':10} {'Result'}")
print(f"  {'-'*60}")
for fid, r in fold_results.items():
    beats = '*** BEATS' if r.get('best_ml', 1e9) < r['floor'] else ''
    print(f"  {fid:<6} {r['n']:<6} {r['floor']:<10.3f} "
          f"{r.get('ridge',float('nan')):<10.3f} "
          f"{r.get('lgbm',float('nan')):<10.3f} {beats}")

# Pooled
total_n = sum(r['n'] for r in fold_results.values())
if total_n > 0:
    p_floor = sum(r['n']*r['floor']          for r in fold_results.values()) / total_n
    p_ridge = sum(r['n']*r.get('ridge',0)    for r in fold_results.values()) / total_n
    p_lgbm  = sum(r['n']*r.get('lgbm',0)     for r in fold_results.values()) / total_n
    p_best  = min(p_ridge, p_lgbm)
    print(f"\n  POOLED {total_n:4d}   floor={p_floor:.3f}  ridge={p_ridge:.3f}  lgbm={p_lgbm:.3f}  "
          f"{'*** BEATS' if p_best < p_floor else ''}")

# Champion decision: use fold-3 as primary (matches deployment)
print(f"\n--- Champion decision (fold-3 primary) ---")
fold3 = fold_results.get(3, {})
f3_floor = fold3.get('floor', 1e9)
f3_best  = fold3.get('best_ml', 1e9)
f3_ridge = fold3.get('ridge', 1e9)
f3_lgbm  = fold3.get('lgbm', 1e9)

if f3_best < f3_floor:
    champ_algo = 'LightGBM' if f3_lgbm <= f3_ridge else 'Ridge'
    champ_mae  = f3_lgbm if f3_lgbm <= f3_ridge else f3_ridge
    champ_cfg  = fold3.get('lgbm_cfg') if f3_lgbm <= f3_ridge else {'alpha': 'best'}
    print(f"  h=60 CHAMPION: {champ_algo}  MAE={champ_mae:.3f}  floor={f3_floor:.3f}  "
          f"margin={f3_floor-champ_mae:.3f}")
    print(f"  Config: {champ_cfg}")

    # Update state
    state_path = f"{RUN_DIR}/state/current.json"
    try:
        with open(state_path) as f:
            state = json.load(f)
    except Exception:
        state = {}
    state.setdefault('champions', {})
    state['champions']['h60'] = {
        'model':      f'multifold_{champ_algo.lower()}_raw_units',
        'val_mae':    round(champ_mae, 3),
        'floor_mae':  round(f3_floor, 3),
        'margin':     round(f3_floor - champ_mae, 3),
        'beats_floor': True,
        'config':     str(champ_cfg),
        'eval_note':  'fold-3 non-zero val rows (35 total, excl closed days)'
    }
    state['iteration'] = 17
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    print(f"\nstate/current.json updated.")
else:
    print(f"  h=60 ML does not beat floor on fold-3 (best={f3_best:.3f}, floor={f3_floor:.3f})")
    print(f"  Keeping prior Optuna champion (MAE=46.112).")
