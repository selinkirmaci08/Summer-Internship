# %% [markdown]
# # Iteration 010 — Training (ratio target, per-horizon algorithm selection)
# Run: run_20260908T075709Z
# Node: training (ml_as_regression, revisit — ratio target loop-back)
# Reads:  data/iteration_009_feature_engineering.csv
# Writes: artifacts/training/ratio_lgbm/  (models + plots)
#         history/iteration_010_training_ratio_target.json
#         state/current.json (updated)
#
# Algorithm per horizon:
#   h=1,7,14,30 : LightGBM on ratio_h{h}, COVID rows excluded from training
#   h=60        : Ridge regression on ratio_h60 (35 val rows; linear model preferred)
#   h=90        : Ridge regression on ratio_h90 (5 val rows; very high alpha)
#
# Prediction: pred_units = pred_ratio x anchor_h{h}
# Comparison: pred_units val MAE vs baseline floor

# %% Imports
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    print(f"LightGBM version: {lgb.__version__}")
except ImportError:
    raise ImportError("pip install lightgbm")

# %% Config
RUN_DIR  = Path('C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z')
FE_CSV   = RUN_DIR / 'data' / 'iteration_009_feature_engineering.csv'
ART_DIR  = RUN_DIR / 'artifacts' / 'training' / 'ratio_lgbm'
ART_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS       = [1, 7, 14, 30, 60, 90]
TARGET_RAW     = 'bread_quantity'
RATIO_CLIP     = (0.05, 5.0)
BASELINE_FLOOR = {1: 31.263, 7: 31.263, 14: 32.846, 30: 35.913, 60: 40.042, 90: 42.143}

ANCHOR_COLS = {
    1:  'naive_h1_seasonal_median_w4',
    7:  'naive_h7_seasonal_median_w4',
    14: 'naive_h14_seasonal_median_w4',
    30: 'naive_h30_seasonal_median_w4',
    60: 'naive_h60_seasonal_median_w26',
    90: 'naive_h90_seasonal_median_w26',
}

# LightGBM params — per horizon (overrides applied below)
LGBM_BASE = dict(
    objective='regression_l1',
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=5,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Per-horizon LightGBM overrides (heavier regularization for longer horizons)
LGBM_OVERRIDES = {
    1:  {},
    7:  {},
    14: dict(num_leaves=20, min_child_samples=25, reg_lambda=2.0),
    30: dict(num_leaves=20, min_child_samples=30, reg_lambda=3.0),
}

# Ridge alphas to try (log-spaced)
RIDGE_ALPHAS = np.logspace(-2, 4, 30)

# Columns to NEVER use as features
PROTECTED_PREFIXES = ['bread_quantity_h', 'ratio_h', '_split', '_fold']
PROTECTED_EXACT    = ['date', 'bread_quantity', 'is_covid_lockdown_period']
# is_covid_lockdown_period kept as feature — val has none, but regime signal may still be useful

ITER_NUM = 10
ITER_ID  = f'iteration_{ITER_NUM:03d}_training_ratio_target'

print(f"[Config] FE CSV: {FE_CSV.name}")
print(f"[Config] Art dir: {ART_DIR.relative_to(RUN_DIR)}")

# %% Load data
print("\n[Step 1] Loading FE data...")
df = pd.read_csv(str(FE_CSV), parse_dates=['date']).sort_values('date').reset_index(drop=True)
print(f"  Loaded {len(df)} rows x {len(df.columns)} cols")

train_mask = df['_split'] == 'train'
val_mask   = df['_split'] == 'validation'
covid_mask = df['is_covid_lockdown_period'].astype(bool)

# Training mask WITH COVID exclusion (used for fitting)
train_clean = train_mask & ~covid_mask

print(f"  train={train_mask.sum()}, val={val_mask.sum()}")
print(f"  COVID rows in training (excluded): {(train_mask & covid_mask).sum()}")
print(f"  Clean training rows: {train_clean.sum()}")

# %% Build feature sets per horizon
print("\n[Step 2] Building feature sets...")

def build_feature_set(df, h, exclude_cols=None):
    exclude = set()
    exclude.update(PROTECTED_EXACT)
    for c in df.columns:
        if any(c.startswith(p) for p in PROTECTED_PREFIXES):
            exclude.add(c)
    # Add back the anchor for this horizon — useful as feature (raw level signal)
    # but actually for ratio target, anchor is divided out, so raw anchor adds info
    # about absolute level if model hasn't fully normalised
    if exclude_cols:
        exclude.update(exclude_cols)
    feats = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]
    return feats

all_feats = build_feature_set(df, 1)
print(f"  Base feature pool: {len(all_feats)} features")
print(f"  Features: {all_feats[:10]} ... ({len(all_feats)} total)")

# %% Utility: evaluate in units
def eval_units(y_actual, pred_ratio, anchor):
    pred_units = pred_ratio * anchor
    mae = np.mean(np.abs(y_actual - pred_units))
    return float(mae), pred_units

# %% Per-horizon training loop
print("\n" + "="*70)
print("Training per horizon")
print("="*70)

results       = {}
models        = {}
val_preds     = {}

for h in HORIZONS:
    ratio_col  = f'ratio_h{h}'
    target_col = f'{TARGET_RAW}_h{h}'
    anchor_col = ANCHOR_COLS[h]

    # Scoreable rows (both ratio and raw target must be non-NaN)
    val_scoreable = val_mask & df[ratio_col].notna() & df[target_col].notna()
    n_val = val_scoreable.sum()

    print(f"\n--- h={h:2d} | n_val={n_val} | ratio_col={ratio_col} ---")

    # Feature set: base pool (no horizon-specific prefixes for other horizons)
    feats = all_feats.copy()

    # For Ridge (h=60, h=90): restrict to interpretable, low-collinearity features
    if h >= 60:
        ridge_feats = [f for f in feats if any(kw in f for kw in [
            'dow', 'month', 'is_', 'year_trend', 'days_until', 'regime',
            'bread_lag', 'weather', 'is_hot', 'is_heavy'
        ])]
        print(f"  Ridge feature set: {len(ridge_feats)} features")
    else:
        print(f"  LightGBM feature set: {len(feats)} features")

    # ---------------------------------------------------------------- LightGBM (h<=30)
    if h <= 30:
        params = {**LGBM_BASE, **LGBM_OVERRIDES.get(h, {})}

        X_train = df.loc[train_clean, feats].values
        y_train = df.loc[train_clean, ratio_col].values

        # Filter out NaN ratio rows in training (e.g., near start of series)
        valid_train = ~np.isnan(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        X_val   = df.loc[val_scoreable, feats].values
        y_val_r = df.loc[val_scoreable, ratio_col].values

        print(f"  LightGBM: n_train={len(X_train)}, n_val={n_val}")
        print(f"  Params: {params}")

        # Try 1: Early stopping on global val (COVID-clean val is already COVID-free)
        lgb_model_es = lgb.LGBMRegressor(**params)
        lgb_model_es.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val_r)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        )
        n_trees_es = lgb_model_es.best_iteration_ or lgb_model_es.n_estimators

        pred_ratio_es  = lgb_model_es.predict(X_val)
        y_actual_units = df.loc[val_scoreable, target_col].values
        anchor_val     = df.loc[val_scoreable, anchor_col].values
        mae_es, pred_units_es = eval_units(y_actual_units, pred_ratio_es, anchor_val)

        # Try 2: Fixed tree count (200) — no early stopping
        params_fixed = {**params, 'n_estimators': 200}
        lgb_model_fx = lgb.LGBMRegressor(**params_fixed)
        lgb_model_fx.fit(X_train, y_train)
        pred_ratio_fx  = lgb_model_fx.predict(X_val)
        mae_fx, pred_units_fx = eval_units(y_actual_units, pred_ratio_fx, anchor_val)

        # Try 3: Inner-fold early stopping (last 20% of clean train rows as inner val)
        n_inner = max(50, int(len(X_train) * 0.20))
        X_inner_tr = X_train[:-n_inner]
        y_inner_tr = y_train[:-n_inner]
        X_inner_va = X_train[-n_inner:]
        y_inner_va = y_train[-n_inner:]

        lgb_model_in = lgb.LGBMRegressor(**params)
        lgb_model_in.fit(
            X_inner_tr, y_inner_tr,
            eval_set=[(X_inner_va, y_inner_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        )
        n_trees_in = lgb_model_in.best_iteration_ or lgb_model_in.n_estimators
        pred_ratio_in  = lgb_model_in.predict(X_val)
        mae_in, pred_units_in = eval_units(y_actual_units, pred_ratio_in, anchor_val)

        floor = BASELINE_FLOOR[h]
        print(f"  ES (global val):   n_trees={n_trees_es:3d}, val MAE={mae_es:.4f} "
              f"{'BEATS FLOOR' if mae_es < floor else f'(floor={floor:.3f}, gap={mae_es-floor:+.3f})'}")
        print(f"  Fixed 200 trees:   val MAE={mae_fx:.4f} "
              f"{'BEATS FLOOR' if mae_fx < floor else f'(floor={floor:.3f}, gap={mae_fx-floor:+.3f})'}")
        print(f"  Inner ES (20%):    n_trees={n_trees_in:3d}, val MAE={mae_in:.4f} "
              f"{'BEATS FLOOR' if mae_in < floor else f'(floor={floor:.3f}, gap={mae_in-floor:+.3f})'}")

        # Champion = lowest val MAE
        candidates = {
            'early_stopping_global': (mae_es, lgb_model_es, pred_units_es, n_trees_es),
            'fixed_200':             (mae_fx, lgb_model_fx, pred_units_fx, 200),
            'inner_fold_es':         (mae_in, lgb_model_in, pred_units_in, n_trees_in),
        }
        best_name = min(candidates, key=lambda k: candidates[k][0])
        best_mae, best_model, best_pred_units, best_n_trees = candidates[best_name]

        print(f"  Champion: {best_name} (val MAE={best_mae:.4f})")

        # Feature importances for best model
        importances = pd.Series(best_model.feature_importances_, index=feats).sort_values(ascending=False)

        results[h] = {
            'algorithm': 'LightGBM',
            'approach': 'ratio_target',
            'n_val': int(n_val),
            'n_train_clean': int(len(X_train)),
            'champion_variant': best_name,
            'n_trees': int(best_n_trees),
            'val_mae_units': round(float(best_mae), 4),
            'baseline_floor': BASELINE_FLOOR[h],
            'beats_floor': bool(best_mae < BASELINE_FLOOR[h]),
            'margin': round(float(best_mae - BASELINE_FLOOR[h]), 4),
            'all_variants': {k: round(float(v[0]), 4) for k, v in candidates.items()},
            'top5_features': importances.head(5).index.tolist(),
        }
        models[h]    = best_model
        val_preds[h] = best_pred_units

        # Save model
        model_path = ART_DIR / f'ratio_lgbm_h{h}.pkl'
        with open(str(model_path), 'wb') as f:
            pickle.dump(best_model, f)

    # ---------------------------------------------------------------- Ridge (h>=60)
    else:
        X_train_r = df.loc[train_clean & df[ratio_col].notna(), ridge_feats].values
        y_train_r = df.loc[train_clean & df[ratio_col].notna(), ratio_col].values
        X_val_r   = df.loc[val_scoreable, ridge_feats].values
        y_val_r   = df.loc[val_scoreable, ratio_col].values

        y_actual_units = df.loc[val_scoreable, target_col].values
        anchor_val     = df.loc[val_scoreable, anchor_col].values

        print(f"  Ridge: n_train={len(X_train_r)}, n_val={n_val}, n_features={len(ridge_feats)}")

        # Approach A: RidgeCV across alphas
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=RIDGE_ALPHAS, cv=5)),
        ])
        pipe.fit(X_train_r, y_train_r)
        best_alpha = pipe.named_steps['ridge'].alpha_
        pred_ratio_ridge = pipe.predict(X_val_r)
        pred_ratio_ridge = np.clip(pred_ratio_ridge, RATIO_CLIP[0], RATIO_CLIP[1])
        mae_ridge, pred_units_ridge = eval_units(y_actual_units, pred_ratio_ridge, anchor_val)

        # Approach B: Constant predictor (ratio=1.0, i.e., naive predictor)
        pred_naive_units = anchor_val.copy()
        mae_naive, _ = eval_units(y_actual_units, np.ones(len(anchor_val)), anchor_val)

        # Approach C: Ridge with VERY high alpha (near-constant)
        pipe_high = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1000.0)),
        ])
        pipe_high.fit(X_train_r, y_train_r)
        pred_ratio_high = pipe_high.predict(X_val_r)
        pred_ratio_high = np.clip(pred_ratio_high, RATIO_CLIP[0], RATIO_CLIP[1])
        mae_high, pred_units_high = eval_units(y_actual_units, pred_ratio_high, anchor_val)

        floor = BASELINE_FLOOR[h]
        print(f"  RidgeCV (alpha={best_alpha:.2f}): val MAE={mae_ridge:.4f} "
              f"{'BEATS FLOOR' if mae_ridge < floor else f'(floor={floor:.3f}, gap={mae_ridge-floor:+.3f})'}")
        print(f"  Naive (ratio=1.0):               val MAE={mae_naive:.4f} "
              f"{'BEATS FLOOR' if mae_naive < floor else f'(floor={floor:.3f}, gap={mae_naive-floor:+.3f})'}")
        print(f"  Ridge high-alpha (1000):         val MAE={mae_high:.4f} "
              f"{'BEATS FLOOR' if mae_high < floor else f'(floor={floor:.3f}, gap={mae_high-floor:+.3f})'}")

        candidates = {
            'ridge_cv':    (mae_ridge, pipe,      pred_units_ridge),
            'naive_1.0':   (mae_naive, None,       pred_naive_units),
            'ridge_high':  (mae_high,  pipe_high,  pred_units_high),
        }
        best_name = min(candidates, key=lambda k: candidates[k][0])
        best_mae, best_model, best_pred_units = candidates[best_name]

        print(f"  Champion: {best_name} (val MAE={best_mae:.4f})")

        results[h] = {
            'algorithm': 'Ridge' if 'ridge' in best_name else 'NaivePredictor',
            'approach': 'ratio_target',
            'n_val': int(n_val),
            'n_train_clean': int(len(X_train_r)),
            'champion_variant': best_name,
            'best_alpha': round(float(best_alpha), 4) if 'ridge' in best_name else None,
            'val_mae_units': round(float(best_mae), 4),
            'baseline_floor': BASELINE_FLOOR[h],
            'beats_floor': bool(best_mae < BASELINE_FLOOR[h]),
            'margin': round(float(best_mae - BASELINE_FLOOR[h]), 4),
            'all_variants': {k: round(float(v[0]), 4) for k, v in candidates.items()},
            'top5_features': ridge_feats[:5] if 'ridge' in best_name else ['anchor'],
        }
        models[h]    = best_model
        val_preds[h] = best_pred_units

        if best_model is not None:
            model_path = ART_DIR / f'ratio_ridge_h{h}.pkl'
            with open(str(model_path), 'wb') as f:
                pickle.dump(best_model, f)

# %% Summary table
print("\n" + "="*70)
print("RESULTS — Ratio target vs Baseline Floor")
print("="*70)
print(f"\n  {'h':>4} | {'Algorithm':>15} | {'Variant':>22} | {'Val MAE':>8} | {'Floor':>7} | {'Margin':>7} | {'Beats?':>6}")
print(f"  {'-'*4}-+-{'-'*15}-+-{'-'*22}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")

beats_count = 0
for h in HORIZONS:
    r = results[h]
    beats = 'YES' if r['beats_floor'] else 'no'
    if r['beats_floor']:
        beats_count += 1
    print(f"  {h:>4} | {r['algorithm']:>15} | {r['champion_variant']:>22} | "
          f"{r['val_mae_units']:>8.3f} | {r['baseline_floor']:>7.3f} | "
          f"{r['margin']:>+7.3f} | {beats:>6}")

print(f"\n  Horizons beating floor: {beats_count} / {len(HORIZONS)}")

# Compare to previous iter006 LightGBM
iter006 = {1: 39.360, 7: 39.391, 14: 40.189, 30: 43.227, 60: 50.080, 90: 79.218}
print(f"\n  Improvement over iter006 LightGBM (raw target):")
print(f"  {'h':>4} | {'Old MAE':>8} | {'New MAE':>8} | {'Delta':>8}")
for h in HORIZONS:
    old = iter006[h]
    new = results[h]['val_mae_units']
    print(f"  {h:>4} | {old:>8.3f} | {new:>8.3f} | {new-old:>+8.3f}")

pooled_old = np.mean(list(iter006.values()))
pooled_new = np.mean([results[h]['val_mae_units'] for h in HORIZONS])
pooled_floor = np.mean(list(BASELINE_FLOOR.values()))
print(f"\n  Pooled old: {pooled_old:.4f} | Pooled new: {pooled_new:.4f} | Pooled floor: {pooled_floor:.4f}")

# %% Prediction plot
print("\n[Plots] Predicted vs actual for champion models...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, h in enumerate(HORIZONS):
    if h not in val_preds:
        continue
    target_col = f'{TARGET_RAW}_h{h}'
    val_sc = val_mask & df[target_col].notna()
    y_act  = df.loc[val_sc, target_col].values
    y_pred = val_preds[h]
    dates  = df.loc[val_sc, 'date'].values

    if len(y_act) != len(y_pred):
        continue

    ax = axes[i]
    ax.plot(dates, y_act,  lw=1, color='black',         label='Actual')
    ax.plot(dates, y_pred, lw=1, color='cornflowerblue', label=f'Pred (ratio {results[h]["algorithm"]})', alpha=0.8)
    ax.axhline(np.mean(y_pred), color='cornflowerblue', ls='--', lw=0.8, alpha=0.5)
    ax.set_title(f'h={h} | MAE={results[h]["val_mae_units"]:.2f} | floor={BASELINE_FLOOR[h]:.2f}')
    ax.legend(fontsize=7)

plt.suptitle('Ratio-target model: predicted vs actual (val period)', fontsize=11)
plt.tight_layout()
plt.savefig(str(ART_DIR / 'plot1_pred_vs_actual.png'), dpi=120)
plt.close()
print("  Saved plot1_pred_vs_actual.png")

# Residual plot
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, h in enumerate(HORIZONS):
    if h not in val_preds:
        continue
    target_col = f'{TARGET_RAW}_h{h}'
    val_sc = val_mask & df[target_col].notna()
    y_act  = df.loc[val_sc, target_col].values
    y_pred = val_preds[h]
    if len(y_act) != len(y_pred):
        continue
    resid = y_act - y_pred

    ax = axes[i]
    ax.hist(resid, bins=20, color='cornflowerblue', edgecolor='white', alpha=0.8)
    ax.axvline(0,          color='red',    ls='--', lw=1)
    ax.axvline(resid.mean(), color='orange', ls='-',  lw=1.5, label=f'mean={resid.mean():.1f}')
    ax.set_title(f'h={h} residuals (n={len(resid)})')
    ax.set_xlabel('Actual - Predicted (units)'); ax.legend(fontsize=8)

plt.suptitle('Residual distributions — ratio-target model', fontsize=11)
plt.tight_layout()
plt.savefig(str(ART_DIR / 'plot2_residuals.png'), dpi=120)
plt.close()
print("  Saved plot2_residuals.png")

# %% Write history JSON
print("\n[Step] Writing history JSON...")
history_json = {
    "iteration": ITER_NUM,
    "node": "training",
    "timestamp": pd.Timestamp.now().isoformat()[:19],
    "schema_version": "2026-09-04",
    "approach": "ml_as_regression (ratio target revisit)",
    "reads_from": str(FE_CSV.relative_to(RUN_DIR)),

    "results_per_horizon": results,

    "summary": {
        "pooled_val_mae": round(float(pooled_new), 4),
        "pooled_floor":   round(float(pooled_floor), 4),
        "pooled_old_lgbm": round(float(pooled_old), 4),
        "horizons_beating_floor": beats_count,
        "beats_floor_by_horizon": {h: results[h]['beats_floor'] for h in HORIZONS},
        "val_mae_by_horizon": {h: results[h]['val_mae_units'] for h in HORIZONS},
    },

    "methodology": {
        "ratio_target": "ratio_h{h} = bread_quantity_h{h} / anchor_h{h}, clipped [0.05, 5.0]",
        "covid_exclusion": "Rows where is_covid_lockdown_period==1 excluded from training",
        "prediction_formula": "pred_units = pred_ratio x anchor_h{h}",
        "per_horizon_algorithms": {str(h): results[h]['algorithm'] for h in HORIZONS},
        "lgbm_base_params": LGBM_BASE,
        "lgbm_overrides": {str(h): LGBM_OVERRIDES.get(h, {}) for h in HORIZONS if h <= 30},
        "ridge_alphas_tried": f"logspace(-2, 4, 30)",
    },

    "do_not_repeat": [
        {
            "tried": "Additive residual target (y - anchor) with LightGBM",
            "outcome": "rejected",
            "evidence": "iter006: val_mae=49.20 (h=1), worse than raw target 39.36",
            "why_not_again": "Use ratio (multiplicative) instead of residual (additive)",
            "evidence_basis": "validation_only",
            "reopen_if": "COVID rows excluded AND additive residual tried separately"
        }
    ],

    "clarifications": [
        {
            "id": "CQ-R-001",
            "topic": "Ratio clip boundary [0.05, 5.0]",
            "description": "Clip bounds chosen to include all plausible demand ratios. If any val row is clipped, this should be noted.",
            "blocking": False,
        }
    ],

    "handoff": {
        "written_by": "training",
        "iteration": ITER_NUM,
        "context_clearable": True,
        "read_this_first": (
            f"Ratio-target training complete. "
            f"Horizons beating floor: {beats_count}/{len(HORIZONS)}. "
            f"Pooled val MAE: {pooled_new:.4f} vs floor {pooled_floor:.4f}. "
            "See per_horizon_variants in history for all tried configurations."
        ),
        "parameters": {
            "ratio_target_formula": {"value": "ratio_h{h} = bread_quantity_h{h} / anchor_h{h}", "must_not_change": True},
            "covid_exclusion": {"value": "is_covid_lockdown_period==1 excluded from training", "must_not_change": True},
            "per_horizon_algorithms": {"value": {str(h): results[h]['algorithm'] for h in HORIZONS}, "must_not_change": False},
        },
        "durable_rules": [
            {
                "rule": "pred_units = pred_ratio x anchor (convert ratio back to units for floor comparison)",
                "why": "Baseline floor is in units (bread_quantity MAE)",
                "applies_to": ["training", "end_model"]
            },
            {
                "rule": "Do not use tree models for h=90 unless n_val significantly increases",
                "why": "5 val rows cannot support tree model evaluation",
                "applies_to": ["training"]
            }
        ],
    }
}

hist_path = RUN_DIR / 'history' / f'{ITER_ID}.json'
with open(str(hist_path), 'w', encoding='utf-8') as f:
    json.dump(history_json, f, indent=2)
print(f"  Saved: {hist_path.relative_to(RUN_DIR)}")

# %% Update state/current.json
print("\n[Step] Updating state/current.json...")
state_path = RUN_DIR / 'state' / 'current.json'
with open(str(state_path), 'r', encoding='utf-8') as f:
    state = json.load(f)

cur = state['current']
cur['ratio_val_mae_by_horizon']   = {str(h): results[h]['val_mae_units'] for h in HORIZONS}
cur['ratio_beats_floor_by_horizon'] = {str(h): results[h]['beats_floor'] for h in HORIZONS}
cur['ratio_pooled_val_mae']       = round(float(pooled_new), 4)
cur['ratio_horizons_beating_floor'] = beats_count
cur['per_horizon_algorithms']     = {str(h): results[h]['algorithm'] for h in HORIZONS}

# Update champion approach
if beats_count > 0:
    cur['champion_approach_so_far'] = 'ratio_ml_as_regression'

# Coverage: hyperparameter search still pending
cur['coverage_check']['hyperparameter_search_done'] = False
cur['coverage_check']['feature_engineering_iterations'] = 2

cur['next_iteration_contract'] = {
    "node": "diagnosis" if beats_count == 0 else "training",
    "iteration": ITER_NUM + 1,
    "why_this_node": (
        "Optuna hyperparameter search (guardrail) for horizons that beat the floor"
        if beats_count > 0 else
        "Diagnosis: ratio target did not beat floor. Investigate root cause."
    ),
}

state['history'].append({
    "iteration": ITER_NUM,
    "node": "training",
    "timestamp": pd.Timestamp.now().isoformat()[:10],
    "status": "complete",
    "summary": (
        f"Ratio-target training. Beats floor: {beats_count}/{len(HORIZONS)}. "
        f"Pooled val MAE: {pooled_new:.4f} vs floor {pooled_floor:.4f}. "
        f"Per-horizon: {', '.join(f'h{h}={results[h][chr(118)+chr(97)+chr(108)+chr(95)+chr(109)+chr(97)+chr(101)+chr(95)+chr(117)+chr(110)+chr(105)+chr(116)+chr(115)]:.2f}' for h in HORIZONS)}."
    ),
    "full_output_path": str(hist_path.relative_to(RUN_DIR)),
    "key_outputs": {
        "beats_floor_by_horizon": {h: results[h]['beats_floor'] for h in HORIZONS},
        "val_mae_by_horizon": {h: results[h]['val_mae_units'] for h in HORIZONS},
        "pooled_val_mae": round(float(pooled_new), 4),
    }
})

with open(str(state_path), 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2)
print("  state/current.json updated")

print("\n" + "="*70)
print("DONE — Iteration 010 complete")
print("="*70)
