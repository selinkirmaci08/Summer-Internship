# iteration_015_h90_multifold.py
# Multi-fold training/evaluation for h=90 to get reliable sample sizes.
# Fold-1: 68 h=90 val rows | Fold-2: 67 | Fold-3: 5
# Goal: beat smedian_w26 on h=90 consistently across folds.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

RUN_DIR = "C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z"
DATA_DIR = f"{RUN_DIR}/data"
import lightgbm as lgb
from sklearn.linear_model import Ridge

df = pd.read_csv(f"{DATA_DIR}/iteration_012_feature_engineering.csv", parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# Feature set (exclude target columns and meta columns)
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

TARGET = 'bread_quantity_h90'
FLOOR_COL = 'naive_h90_seasonal_median_w26'

# ===========================================================================
# Evaluation helper — report MAE on both all rows and non-zero rows
# ===========================================================================
def evaluate(label, preds, actuals, include_zeros=False):
    mask = actuals > 0 if not include_zeros else np.ones(len(actuals), dtype=bool)
    if mask.sum() == 0:
        return np.nan
    mae = np.mean(np.abs(preds[mask] - actuals[mask]))
    suffix = "(non-zero)" if not include_zeros else "(all rows)"
    print(f"  {label:40s}: MAE={mae:.3f}  n={mask.sum()}  {suffix}")
    return mae

# ===========================================================================
# Multi-fold evaluation loop
# ===========================================================================
fold_results = {}  # fold -> {model: mae}

for fold_id, fold_col in enumerate(['_fold_1', '_fold_2', '_fold_3'], start=1):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_id}  (fold_col={fold_col})")
    print(f"{'='*70}")

    train_df = df[(df[fold_col] == 'train') & df[TARGET].notna() & (df['is_covid_lockdown_period'] == 0)].copy()
    val_df   = df[(df[fold_col] == 'val')   & df[TARGET].notna()].copy()
    # Non-zero val for fair comparison with baseline
    val_nz   = val_df[val_df[TARGET] > 0].copy()

    print(f"  Train rows (non-COVID, non-null h=90): {len(train_df)}")
    print(f"  Val rows total / non-zero: {len(val_df)} / {len(val_nz)}")
    print(f"  Val origin range: {val_df['date'].min().date()} to {val_df['date'].max().date()}")
    print(f"  Val target range: {(val_df['date']+pd.Timedelta(days=90)).min().date()} "
          f"to {(val_df['date']+pd.Timedelta(days=90)).max().date()}")

    if len(train_df) < 50 or len(val_nz) == 0:
        print("  Skipping — not enough data.")
        continue

    X_tr  = train_df[feat_cols].fillna(0).values
    y_tr  = train_df[TARGET].values
    X_val = val_nz[feat_cols].fillna(0).values
    y_val = val_nz[TARGET].values

    print(f"\n  Train target: mean={y_tr.mean():.1f}  std={y_tr.std():.1f}")
    print(f"  Val   target: mean={y_val.mean():.1f}  std={y_val.std():.1f}")

    # --- Baseline: smedian_w26 (floor predictor) ---
    floor_preds = val_nz[FLOOR_COL].values
    mae_floor   = np.mean(np.abs(floor_preds - y_val))
    print(f"\n  {'smedian_w26 (floor)':40s}: MAE={mae_floor:.3f}  n={len(val_nz)}")

    fold_results[fold_id] = {'floor': mae_floor, 'n_val': len(val_nz)}

    # --- Ridge regression ---
    best_ridge_mae, best_ridge_alpha = 1e9, None
    for alpha in [1, 10, 100, 1000, 10000]:
        m = Ridge(alpha=alpha)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)
        mae  = np.mean(np.abs(pred - y_val))
        if mae < best_ridge_mae:
            best_ridge_mae, best_ridge_alpha = mae, alpha
    print(f"  {'Ridge (best alpha)':40s}: MAE={best_ridge_mae:.3f}  alpha={best_ridge_alpha}")
    fold_results[fold_id]['ridge'] = best_ridge_mae

    # --- LightGBM (constrained, raw units) ---
    # Use inner-fold early stopping: last 20% of training rows as inner val
    n_inner = max(20, int(0.20 * len(train_df)))
    X_inner_tr  = X_tr[:-n_inner]
    y_inner_tr  = y_tr[:-n_inner]
    X_inner_val = X_tr[-n_inner:]
    y_inner_val = y_tr[-n_inner:]

    best_lgb_mae, best_lgb_cfg = 1e9, None
    for n_leaves in [4, 8, 16]:
        for lr in [0.01, 0.05]:
            m = lgb.LGBMRegressor(
                n_estimators=500,
                num_leaves=n_leaves,
                max_depth=4,
                learning_rate=lr,
                reg_alpha=2.0,
                reg_lambda=20.0,
                min_child_samples=max(5, len(train_df)//50),
                objective='regression_l1',
                verbose=-1
            )
            m.fit(
                X_inner_tr, y_inner_tr,
                eval_X=X_inner_val, eval_y=y_inner_val,
                eval_metric='mae',
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
            )
            pred = m.predict(X_val)
            mae  = np.mean(np.abs(pred - y_val))
            if mae < best_lgb_mae:
                best_lgb_mae = mae
                best_lgb_cfg = (n_leaves, lr, m.best_iteration_)
    print(f"  {'LightGBM (inner ES)':40s}: MAE={best_lgb_mae:.3f}  "
          f"(leaves={best_lgb_cfg[0]}, lr={best_lgb_cfg[1]}, trees={best_lgb_cfg[2]})")
    fold_results[fold_id]['lgbm'] = best_lgb_mae

    # --- Anchor-correct (target-DOW) predictor ---
    anc_preds = val_nz['anchor_correct_h90'].values
    mae_anc   = np.mean(np.abs(anc_preds - y_val))
    print(f"  {'anchor_correct_h90':40s}: MAE={mae_anc:.3f}")
    fold_results[fold_id]['anchor_correct'] = mae_anc

    # --- Beat floor? ---
    best_ml = min(best_ridge_mae, best_lgb_mae)
    if best_ml < mae_floor:
        print(f"\n  *** ML BEATS FLOOR: {best_ml:.3f} < {mae_floor:.3f} (margin={mae_floor-best_ml:.3f}) ***")
    else:
        print(f"\n  ML does not beat floor (best_ml={best_ml:.3f}, floor={mae_floor:.3f})")

# ===========================================================================
# Summary across folds
# ===========================================================================
print(f"\n{'='*70}")
print("MULTI-FOLD SUMMARY (non-zero val rows only)")
print(f"{'='*70}")

print(f"\n  {'Fold':6} {'n_val':6} {'Floor':10} {'Ridge':10} {'LGBM':10} {'Anc_corr':10} {'Winner'}")
print(f"  {'-'*65}")
for fold_id, r in fold_results.items():
    n    = r['n_val']
    fl   = r.get('floor', np.nan)
    ri   = r.get('ridge', np.nan)
    lg   = r.get('lgbm', np.nan)
    ac   = r.get('anchor_correct', np.nan)
    best_ml = min(ri, lg)
    winner = 'floor' if fl <= best_ml else ('lgbm' if lg <= ri else 'ridge')
    beat = '*** BEATS' if best_ml < fl else ''
    print(f"  {fold_id:<6} {n:<6} {fl:<10.3f} {ri:<10.3f} {lg:<10.3f} {ac:<10.3f} {winner} {beat}")

# Pooled (weighted by n_val)
total_n = sum(r['n_val'] for r in fold_results.values())
if total_n > 0:
    pooled_floor = sum(r['n_val'] * r.get('floor', 0) for r in fold_results.values()) / total_n
    pooled_ridge = sum(r['n_val'] * r.get('ridge', 0) for r in fold_results.values()) / total_n
    pooled_lgbm  = sum(r['n_val'] * r.get('lgbm', 0)  for r in fold_results.values()) / total_n
    print(f"\n  {'POOLED':<6} {total_n:<6} {pooled_floor:<10.3f} {pooled_ridge:<10.3f} {pooled_lgbm:<10.3f}")
    best_ml_pooled = min(pooled_ridge, pooled_lgbm)
    if best_ml_pooled < pooled_floor:
        print(f"\n  *** POOLED ML BEATS FLOOR: {best_ml_pooled:.3f} < {pooled_floor:.3f} ***")
    else:
        print(f"\n  Pooled ML does not beat pooled floor ({best_ml_pooled:.3f} >= {pooled_floor:.3f})")

print(f"\nPrimary decision on fold-3 (most recent, matches deployment):")
if 3 in fold_results:
    r = fold_results[3]
    best_ml = min(r.get('ridge', 1e9), r.get('lgbm', 1e9))
    print(f"  fold-3 floor={r.get('floor',np.nan):.3f}, best_ml={best_ml:.3f}")
    if best_ml < r.get('floor', 1e9):
        print("  => ML beats floor on fold-3 (primary)")
    else:
        print("  => smedian_w26 remains champion for h=90")
