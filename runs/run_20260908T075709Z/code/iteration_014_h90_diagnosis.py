# iteration_014_h90_diagnosis.py
# Diagnose why anchor_correct_h90 gives MAE=102.1 (worse than floor 42.143)
# Then attempt to beat the floor.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

RUN_DIR = "C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z"
DATA_DIR = f"{RUN_DIR}/data"

df = pd.read_csv(f"{DATA_DIR}/iteration_012_feature_engineering.csv", parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# ===========================================================================
# PART 1: Inspect the 5 h=90 val rows
# ===========================================================================
print("=" * 70)
print("PART 1: h=90 val row details")
print("=" * 70)

val90 = df[(df['_fold_3'] == 'val') & df['bread_quantity_h90'].notna()].copy()
print(f"\nh=90 val rows: {len(val90)}")

for _, r in val90.iterrows():
    origin = r['date']
    target = origin + pd.Timedelta(days=90)
    o_dow = origin.dayofweek
    t_dow = target.dayofweek
    actual   = r['bread_quantity_h90']
    anc_corr = r['anchor_correct_h90']
    naive_fl = r['naive_h90_seasonal_median_w26']
    ratio_c  = r.get('ratio_correct_h90', np.nan)

    print(f"\n  Origin: {origin.date()} ({DOW[o_dow]})  ->  Target: {target.date()} ({DOW[t_dow]})")
    print(f"    actual          = {actual:.1f}")
    print(f"    anchor_correct  = {anc_corr:.1f}   (ratio_correct={ratio_c:.3f})")
    print(f"    naive_floor     = {naive_fl:.1f}")
    print(f"    err(anc_corr)   = {abs(actual - anc_corr):.1f}")
    print(f"    err(naive_fl)   = {abs(actual - naive_fl):.1f}")

mae_ac = np.mean(np.abs(val90['bread_quantity_h90'] - val90['anchor_correct_h90']))
mae_nf = np.mean(np.abs(val90['bread_quantity_h90'] - val90['naive_h90_seasonal_median_w26']))
print(f"\nMAE(anchor_correct_h90) = {mae_ac:.3f}   [current champion: 102.100]")
print(f"MAE(smedian_w26)        = {mae_nf:.3f}   [floor: 42.143]")

# ===========================================================================
# PART 2: Why does anchor_correct hurt? — check target DOW demand history
# ===========================================================================
print("\n" + "=" * 70)
print("PART 2: Target-DOW demand history")
print("=" * 70)

for _, r in val90.iterrows():
    origin = r['date']
    target = origin + pd.Timedelta(days=90)
    t_dow  = target.dayofweek

    hist = df[(df['date'] < origin) & (df['date'].dt.dayofweek == t_dow)].copy()
    n     = len(hist)
    mean_all  = hist['bread_quantity'].mean() if n > 0 else np.nan
    zeros     = (hist['bread_quantity'] == 0).sum()
    last26    = hist.tail(26)['bread_quantity'].mean() if n >= 1 else np.nan
    last26_nz = hist.tail(26).loc[hist.tail(26)['bread_quantity'] > 0, 'bread_quantity'].mean()

    print(f"\n  Target {target.date()} ({DOW[t_dow]}) from origin {origin.date()}:")
    print(f"    historical {DOW[t_dow]} rows: {n}  |  zeros: {zeros}  |  mean_all: {mean_all:.1f}")
    print(f"    last-26 same-DOW mean: {last26:.1f}  |  last-26 nonzero mean: {last26_nz:.1f}")
    print(f"    anchor_correct = {r['anchor_correct_h90']:.1f}")

# ===========================================================================
# PART 3: Anchor candidates — compare origin-DOW vs target-DOW vs mixed
# ===========================================================================
print("\n" + "=" * 70)
print("PART 3: Alternative anchor strategies for h=90")
print("=" * 70)

# Also check w26 origin-DOW (= floor) and w52 origin-DOW
naive_cols_90 = [c for c in df.columns if 'h90' in c and c.startswith('naive_')]
print(f"Available h=90 naive predictors: {naive_cols_90}")

for col in naive_cols_90:
    mae = np.mean(np.abs(val90['bread_quantity_h90'] - val90[col]))
    print(f"  {col}: MAE = {mae:.3f}")

print(f"  anchor_correct_h90:           MAE = {mae_ac:.3f}")

# Shrinkage: blend anchor_correct with naive_floor
print("\nShrinkage toward smedian_w26 (lam = weight on anchor_correct):")
best_lam, best_mae_s = None, 1e9
for lam in np.linspace(0.0, 1.0, 11):
    pred = lam * val90['anchor_correct_h90'] + (1 - lam) * val90['naive_h90_seasonal_median_w26']
    mae  = np.mean(np.abs(val90['bread_quantity_h90'] - pred))
    marker = " <-- best" if mae < best_mae_s else ""
    print(f"  lam={lam:.1f}: MAE = {mae:.3f}{marker}")
    if mae < best_mae_s:
        best_mae_s, best_lam = mae, lam

# ===========================================================================
# PART 4: ML attempt using raw-unit target (not ratio) — avoid broken anchor
# ===========================================================================
print("\n" + "=" * 70)
print("PART 4: ML on raw-unit target for h=90")
print("=" * 70)

from sklearn.linear_model import Ridge, RidgeCV

train90 = df[
    (df['_fold_3'] == 'train') &
    df['bread_quantity_h90'].notna() &
    (df['is_covid_lockdown_period'] == 0)
].copy()
print(f"Non-COVID train rows for h=90: {len(train90)}")

# Feature set: exclude future-leaking / target columns
exclude = set(
    ['date', 'split', 'fold', 'bread_quantity'] +
    [c for c in df.columns if c.startswith('bread_quantity_h')] +
    [c for c in df.columns if c.startswith('ratio_h')] +
    [c for c in df.columns if c.startswith('ratio_correct_h')] +
    [c for c in df.columns if c.startswith('anchor_correct_h')] +
    [c for c in df.columns if c.startswith('anchor_h')]
)
feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype != object]
print(f"Features: {len(feat_cols)}")

X_tr  = train90[feat_cols].fillna(0).values
y_tr  = train90['bread_quantity_h90'].values
X_val = val90[feat_cols].fillna(0).values
y_val = val90['bread_quantity_h90'].values

print("\nRidge (raw units target):")
best_ridge_mae = 1e9
for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    m = Ridge(alpha=alpha)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_val)
    mae  = np.mean(np.abs(pred - y_val))
    marker = " <-- best" if mae < best_ridge_mae else ""
    print(f"  alpha={alpha}: MAE = {mae:.3f}{marker}")
    if mae < best_ridge_mae:
        best_ridge_mae = mae

# LightGBM with heavy regularization on raw units
import lightgbm as lgb
print("\nLightGBM (raw units, constrained):")
best_lgb_mae = 1e9
for n_trees, n_leaves in [(5,4),(10,4),(20,4),(10,8),(20,8)]:
    m = lgb.LGBMRegressor(
        n_estimators=n_trees,
        num_leaves=n_leaves,
        max_depth=3,
        learning_rate=0.05,
        reg_alpha=5.0,
        reg_lambda=50.0,
        min_child_samples=10,
        objective='regression_l1',
        verbose=-1
    )
    m.fit(X_tr, y_tr)
    pred = m.predict(X_val)
    mae  = np.mean(np.abs(pred - y_val))
    marker = " <-- best" if mae < best_lgb_mae else ""
    print(f"  n_trees={n_trees}, n_leaves={n_leaves}: MAE = {mae:.3f}{marker}")
    if mae < best_lgb_mae:
        best_lgb_mae = mae

# ===========================================================================
# PART 5: Multi-fold evaluation — use fold-1 and fold-2 val rows for h=90
# ===========================================================================
print("\n" + "=" * 70)
print("PART 5: Multi-fold h=90 val (increase sample size)")
print("=" * 70)

for fold_col in ['_fold_1', '_fold_2', '_fold_3']:
    fold_val = df[(df[fold_col] == 'val') & df['bread_quantity_h90'].notna()]
    print(f"  {fold_col}: {len(fold_val)} h=90 val rows", end="")
    if len(fold_val) > 0:
        mae_nf_fold = np.mean(np.abs(fold_val['bread_quantity_h90'] - fold_val['naive_h90_seasonal_median_w26']))
        print(f"  |  MAE(smedian_w26)={mae_nf_fold:.3f}  |  "
              f"origin range: {fold_val['date'].min().date()} to {fold_val['date'].max().date()}")
    else:
        print()

# ===========================================================================
# PART 6: Summary
# ===========================================================================
print("\n" + "=" * 70)
print("PART 6: Summary")
print("=" * 70)

print(f"\nCurrent champion (anchor_correct_h90):  MAE = {mae_ac:.3f}  [WORSE than floor]")
print(f"Floor (smedian_w26):                    MAE = {mae_nf:.3f}")
print(f"Best shrinkage:                         MAE = {best_mae_s:.3f}  (lam={best_lam:.1f})")
print(f"Best Ridge (raw units):                 MAE = {best_ridge_mae:.3f}")
print(f"Best LightGBM (raw units):              MAE = {best_lgb_mae:.3f}")

if best_ridge_mae < mae_nf or best_lgb_mae < mae_nf or best_mae_s < mae_nf:
    print("\nAt least one method BEATS the floor! See candidates above.")
else:
    print(f"\nNo ML method beats the floor for h=90. Recommend adopting smedian_w26 as champion.")
    print(f"Immediate fix: switch champion from anchor_correct_h90 to naive_h90_seasonal_median_w26.")
