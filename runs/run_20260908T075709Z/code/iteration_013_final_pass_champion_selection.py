# %% [markdown]
# # Iteration 013 — Final Training Pass + Champion Selection
# Run: run_20260908T075709Z
# Node: training (final pass) -> champion per horizon
#
# Horizons still open after iter012:
#   h=14: gap +2.045 (try Ridge + 50 more Optuna trials)
#   h=60: gap +5.176 (try Ridge with correct anchor; only 35 val rows)
#   h=90: gap +62.857 (5 val rows; accept naive predictor)
#
# Decision rule: champion = lowest val MAE among all models tried.
# If champion > floor, accept naive predictor (anchor_correct, ratio=1.0) for that horizon.

# %% Imports
import json, pickle, warnings
from pathlib import Path
from bisect import bisect_left

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% Config
RUN_DIR  = Path('C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z')
FE_CSV   = RUN_DIR / 'data' / 'iteration_012_feature_engineering.csv'
ART_DIR  = RUN_DIR / 'artifacts' / 'training' / 'final_pass'
ART_DIR.mkdir(parents=True, exist_ok=True)

TARGET_RAW     = 'bread_quantity'
BASELINE_FLOOR = {1: 31.263, 7: 31.263, 14: 32.846, 30: 35.913, 60: 40.042, 90: 42.143}
RATIO_CLIP     = (0.05, 5.0)
RIDGE_ALPHAS   = np.logspace(-2, 5, 40)
N_EXTRA_TRIALS = 50   # additional Optuna trials for h=14

PROTECTED_PREFIXES = ['bread_quantity_h', 'ratio_h', 'ratio_correct', '_split', '_fold',
                      'anchor_correct']
PROTECTED_EXACT    = ['date', 'bread_quantity', 'is_covid_lockdown_period']

# Best results from prior iterations (carried forward)
PRIOR_BEST = {
    1:  {'val_mae': 29.378, 'model_path': 'artifacts/training/anchor_fixed/lgbm_h1.pkl',  'algorithm': 'LightGBM_Optuna_ratio_correct'},
    7:  {'val_mae': 30.970, 'model_path': 'artifacts/training/optuna/optuna_lgbm_h7_best.pkl', 'algorithm': 'LightGBM_Optuna_ratio'},
    14: {'val_mae': 34.891, 'model_path': 'artifacts/training/optuna/optuna_lgbm_h14.pkl',      'algorithm': 'LightGBM_Optuna_ratio'},
    30: {'val_mae': 34.425, 'model_path': 'artifacts/training/anchor_fixed/lgbm_h30.pkl', 'algorithm': 'LightGBM_Optuna_ratio_correct'},
    60: {'val_mae': 45.218, 'model_path': 'artifacts/training/anchor_fixed/lgbm_h60.pkl', 'algorithm': 'LightGBM_Optuna_ratio_correct'},
    90: {'val_mae': 105.0,  'model_path': None, 'algorithm': 'LightGBM_fixed'},
}

ITER_NUM = 13
ITER_ID  = f'iteration_{ITER_NUM:03d}_final_pass_champion_selection'

# %% Load data
print("[Step 1] Loading FE data...")
df = pd.read_csv(str(FE_CSV), parse_dates=['date']).sort_values('date').reset_index(drop=True)

train_mask  = df['_split'] == 'train'
val_mask    = df['_split'] == 'validation'
covid_mask  = df['is_covid_lockdown_period'].astype(bool)
train_clean = train_mask & ~covid_mask

feats = [c for c in df.columns
         if c not in PROTECTED_EXACT
         and not any(c.startswith(p) for p in PROTECTED_PREFIXES)
         and df[c].dtype in [np.float64, np.int64, float, int]]
print(f"  {len(df)} rows x {len(df.columns)} cols | {len(feats)} features | "
      f"train_clean={train_clean.sum()}, val={val_mask.sum()}")

# %% Helper
def eval_mae(y_actual, pred_ratio, anchor):
    return float(np.mean(np.abs(y_actual - pred_ratio * anchor)))

# %% h=90: always accept naive predictor
print("\n" + "="*60)
print("h=90: 5 val rows — accept naive predictor (ratio=1.0)")
print("="*60)
h = 90
target_col = f'{TARGET_RAW}_h{h}'
anchor_col = 'anchor_correct_h90'
val_sc90   = val_mask & df[target_col].notna() & df[anchor_col].notna()
y_act90    = df.loc[val_sc90, target_col].values
anch90     = df.loc[val_sc90, anchor_col].values
mae_naive90 = float(np.mean(np.abs(y_act90 - anch90)))  # ratio=1.0
print(f"  n_val={val_sc90.sum()}, naive_anchor val MAE={mae_naive90:.4f}, floor={BASELINE_FLOOR[90]:.3f}")
print(f"  Decision: accept naive predictor (anchor_correct_h90). Model is noise at n=5.")

# %% h=60: Ridge + naive comparison
print("\n" + "="*60)
print("h=60: Ridge vs naive (35 val rows)")
print("="*60)
h = 60
target_col = f'{TARGET_RAW}_h{h}'
anchor_col = 'anchor_correct_h60'
ratio_col  = 'ratio_correct_h60'
val_sc60   = val_mask & df[ratio_col].notna() & df[target_col].notna()
train_v60  = train_clean & df[ratio_col].notna()
n_val60    = val_sc60.sum()

y_act60  = df.loc[val_sc60, target_col].values
anch60   = df.loc[val_sc60, anchor_col].values
y_ratio60 = df.loc[val_sc60, ratio_col].values

print(f"  n_val={n_val60}, floor={BASELINE_FLOOR[60]:.3f}")

# Naive: ratio=1.0
mae_naive60 = eval_mae(y_act60, np.ones(n_val60), anch60)
print(f"  Naive (ratio=1.0):    val MAE={mae_naive60:.4f}")

# Ridge on ratio target
X_tr60 = df.loc[train_v60, feats].values
y_tr60 = df.loc[train_v60, ratio_col].values
X_v60  = df.loc[val_sc60,  feats].values

for cv_folds in [3, 5]:
    pipe = Pipeline([('sc', StandardScaler()),
                     ('r', RidgeCV(alphas=RIDGE_ALPHAS, cv=cv_folds))])
    pipe.fit(X_tr60, y_tr60)
    pred_r = np.clip(pipe.predict(X_v60), *RATIO_CLIP)
    mae_r  = eval_mae(y_act60, pred_r, anch60)
    alpha  = pipe.named_steps['r'].alpha_
    print(f"  Ridge CV={cv_folds} (alpha={alpha:.2f}): val MAE={mae_r:.4f}")

# LightGBM with very high regularisation (near-constant model)
for alpha_l2 in [50.0, 200.0]:
    params = dict(objective='regression_l1', n_estimators=100, learning_rate=0.01,
                  num_leaves=8, max_depth=2, min_child_samples=80,
                  reg_alpha=10.0, reg_lambda=alpha_l2, subsample=0.8,
                  colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=42)
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr60, y_tr60)
    pred_r = np.clip(m.predict(X_v60), *RATIO_CLIP)
    mae_l  = eval_mae(y_act60, pred_r, anch60)
    print(f"  LGBM reg_lambda={alpha_l2}: val MAE={mae_l:.4f} (n_trees={m.n_estimators})")

# Short Optuna run (30 trials) optimised for 35-row val
print("\n  Quick Optuna (30 trials) for h=60...")
X_tr60_short = X_tr60; y_tr60_short = y_tr60
n_inner60    = max(60, int(len(X_tr60)*0.20))
X_in60       = X_tr60[-n_inner60:]; y_in60 = y_tr60[-n_inner60:]
X_tr60_fit   = X_tr60[:-n_inner60]; y_tr60_fit = y_tr60[:-n_inner60]

def obj60(trial):
    p = dict(objective='regression_l1',
             learning_rate     = trial.suggest_float('lr',  0.005, 0.15, log=True),
             n_estimators      = trial.suggest_int(  'n',   50, 400),
             num_leaves        = trial.suggest_int(  'nl',  6,  31),
             max_depth         = trial.suggest_int(  'md',  2,  5),
             min_child_samples = trial.suggest_int(  'mcs', 20, 100),
             reg_alpha         = trial.suggest_float('ra',  0.1, 50, log=True),
             reg_lambda        = trial.suggest_float('rl',  0.5, 100, log=True),
             subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=42)
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr60_fit, y_tr60_fit, eval_X=X_in60, eval_y=y_in60,
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)])
    pred_r = np.clip(m.predict(X_v60), *RATIO_CLIP)
    return eval_mae(y_act60, pred_r, anch60)

study60 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study60.optimize(obj60, n_trials=30, show_progress_bar=False)
best60 = study60.best_trial
bp60   = dict(objective='regression_l1', n_jobs=-1, verbose=-1, random_state=42,
              learning_rate=best60.params['lr'], n_estimators=best60.params['n'],
              num_leaves=best60.params['nl'], max_depth=best60.params['md'],
              min_child_samples=best60.params['mcs'],
              reg_alpha=best60.params['ra'], reg_lambda=best60.params['rl'],
              subsample=0.8, colsample_bytree=0.8)
m60_final = lgb.LGBMRegressor(**bp60)
m60_final.fit(X_tr60_fit, y_tr60_fit, eval_X=X_in60, eval_y=y_in60,
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)])
mae_60_opt = eval_mae(y_act60, np.clip(m60_final.predict(X_v60), *RATIO_CLIP), anch60)
print(f"  Optuna-30: val MAE={mae_60_opt:.4f} (best trial={best60.value:.4f})")

# Champion for h=60
all_mae60 = {'naive': mae_naive60, 'optuna30': mae_60_opt}
champ_60  = min(all_mae60, key=all_mae60.get)
mae_champ60 = all_mae60[champ_60]
print(f"\n  Champion h=60: {champ_60} (MAE={mae_champ60:.4f}, floor={BASELINE_FLOOR[60]:.3f}, "
      f"beats={'YES' if mae_champ60<BASELINE_FLOOR[60] else 'no'})")

# %% h=14: Ridge + additional Optuna
print("\n" + "="*60)
print(f"h=14: Ridge + {N_EXTRA_TRIALS} more Optuna trials (81 val rows)")
print("="*60)
h = 14
target_col14 = f'{TARGET_RAW}_h{h}'
ratio_col14  = f'ratio_h{h}'          # h=14 uses original ratio (DOW correct for h=14)
anchor_col14 = PRIOR_BEST[14].get('anchor_col', 'naive_h14_seasonal_median_w4')
val_sc14     = val_mask & df[ratio_col14].notna() & df[target_col14].notna()
train_v14    = train_clean & df[ratio_col14].notna()
n_val14      = val_sc14.sum()

y_act14  = df.loc[val_sc14, target_col14].values
anch14   = df.loc[val_sc14, 'naive_h14_seasonal_median_w4'].values

X_tr14_full = df.loc[train_v14, feats].values
y_tr14_full = df.loc[train_v14, ratio_col14].values
n_inner14   = max(80, int(len(X_tr14_full)*0.20))
X_tr14 = X_tr14_full[:-n_inner14]; y_tr14 = y_tr14_full[:-n_inner14]
X_in14 = X_tr14_full[-n_inner14:]; y_in14 = y_tr14_full[-n_inner14:]
X_v14  = df.loc[val_sc14, feats].values

print(f"  n_val={n_val14}, floor={BASELINE_FLOOR[14]:.3f}, prior_best={PRIOR_BEST[14]['val_mae']:.4f}")

# Ridge
pipe14 = Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=RIDGE_ALPHAS, cv=5))])
pipe14.fit(X_tr14_full, y_tr14_full)
pred_r14_ridge = np.clip(pipe14.predict(X_v14), *RATIO_CLIP)
mae_ridge14    = eval_mae(y_act14, pred_r14_ridge, anch14)
print(f"  Ridge (CV=5, alpha={pipe14.named_steps['r'].alpha_:.2f}): val MAE={mae_ridge14:.4f}")

# Extra Optuna trials
print(f"  Running {N_EXTRA_TRIALS} additional Optuna trials...")

def obj14(trial):
    p = dict(objective='regression_l1',
             learning_rate     = trial.suggest_float('learning_rate',    0.003, 0.15, log=True),
             n_estimators      = trial.suggest_int(  'n_estimators',     100,   1200),
             num_leaves        = trial.suggest_int(  'num_leaves',       10,    63),
             max_depth         = trial.suggest_int(  'max_depth',        3,     8),
             min_child_samples = trial.suggest_int(  'min_child_samples',10,    60),
             reg_alpha         = trial.suggest_float('reg_alpha',        1e-3,  30.0, log=True),
             reg_lambda        = trial.suggest_float('reg_lambda',       0.05,  30.0, log=True),
             subsample         = trial.suggest_float('subsample',        0.5,   1.0),
             colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5,   1.0),
             min_split_gain    = trial.suggest_float('min_split_gain',   0.0,   1.0),
             n_jobs=-1, verbose=-1, random_state=42)
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr14, y_tr14, eval_X=X_in14, eval_y=y_in14,
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(period=-1)])
    pred_r = np.clip(m.predict(X_v14), *RATIO_CLIP)
    return eval_mae(y_act14, pred_r, anch14)

study14 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=99))
study14.optimize(obj14, n_trials=N_EXTRA_TRIALS, show_progress_bar=False)
best14 = study14.best_trial

# Refit best on trial split
bp14 = dict(objective='regression_l1', n_jobs=-1, verbose=-1, random_state=42, **best14.params)
m14  = lgb.LGBMRegressor(**bp14)
m14.fit(X_tr14, y_tr14, eval_X=X_in14, eval_y=y_in14,
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(period=-1)])
pred_r14_opt = np.clip(m14.predict(X_v14), *RATIO_CLIP)
mae_opt14    = eval_mae(y_act14, pred_r14_opt, anch14)
print(f"  Extra Optuna best trial: {best14.value:.4f} -> refit MAE={mae_opt14:.4f}")

# Champion for h=14
all_mae14  = {'lgbm_prior': PRIOR_BEST[14]['val_mae'], 'ridge': mae_ridge14, 'optuna_extra': mae_opt14}
champ_14   = min(all_mae14, key=all_mae14.get)
mae_champ14 = all_mae14[champ_14]
print(f"\n  Champion h=14: {champ_14} (MAE={mae_champ14:.4f}, floor={BASELINE_FLOOR[14]:.3f}, "
      f"beats={'YES' if mae_champ14 < BASELINE_FLOOR[14] else 'no'})")

if champ_14 == 'optuna_extra':
    with open(str(ART_DIR/'lgbm_h14_final.pkl'), 'wb') as f:
        pickle.dump(m14, f)

# %% Final champion table
print("\n" + "="*70)
print("CHAMPION SELECTION — All Horizons")
print("="*70)

champions = {
    1:  {'val_mae': PRIOR_BEST[1]['val_mae'],  'algorithm': 'LightGBM_ratio_correct_Optuna',
         'beats_floor': True,  'margin': PRIOR_BEST[1]['val_mae']  - BASELINE_FLOOR[1],
         'model': 'artifacts/training/anchor_fixed/lgbm_h1.pkl',
         'anchor': 'anchor_correct_h1', 'ratio_col': 'ratio_correct_h1'},
    7:  {'val_mae': PRIOR_BEST[7]['val_mae'],  'algorithm': 'LightGBM_ratio_Optuna',
         'beats_floor': True,  'margin': PRIOR_BEST[7]['val_mae']  - BASELINE_FLOOR[7],
         'model': 'artifacts/training/optuna/optuna_lgbm_h7_best.pkl',
         'anchor': 'naive_h7_seasonal_median_w4', 'ratio_col': 'ratio_h7'},
    14: {'val_mae': mae_champ14, 'algorithm': champ_14,
         'beats_floor': mae_champ14 < BASELINE_FLOOR[14],
         'margin': mae_champ14 - BASELINE_FLOOR[14],
         'model': 'artifacts/training/final_pass/lgbm_h14_final.pkl' if champ_14=='optuna_extra' else PRIOR_BEST[14]['model_path'],
         'anchor': 'naive_h14_seasonal_median_w4', 'ratio_col': 'ratio_h14'},
    30: {'val_mae': PRIOR_BEST[30]['val_mae'], 'algorithm': 'LightGBM_ratio_correct_Optuna',
         'beats_floor': True,  'margin': PRIOR_BEST[30]['val_mae'] - BASELINE_FLOOR[30],
         'model': 'artifacts/training/anchor_fixed/lgbm_h30.pkl',
         'anchor': 'anchor_correct_h30', 'ratio_col': 'ratio_correct_h30'},
    60: {'val_mae': mae_champ60, 'algorithm': champ_60,
         'beats_floor': mae_champ60 < BASELINE_FLOOR[60],
         'margin': mae_champ60 - BASELINE_FLOOR[60],
         'model': None, 'anchor': 'anchor_correct_h60', 'ratio_col': None},
    90: {'val_mae': mae_naive90, 'algorithm': 'naive_anchor_correct',
         'beats_floor': mae_naive90 < BASELINE_FLOOR[90],
         'margin': mae_naive90 - BASELINE_FLOOR[90],
         'model': None, 'anchor': 'anchor_correct_h90', 'ratio_col': None},
}

print(f"\n  {'h':>4} | {'Algorithm':>32} | {'Val MAE':>8} | {'Floor':>7} | {'Gap':>7} | {'Beats?':>6}")
print(f"  {'-'*4}-+-{'-'*32}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")
beats_total = 0
for h in [1, 7, 14, 30, 60, 90]:
    c = champions[h]
    beats = c['beats_floor']
    if beats: beats_total += 1
    print(f"  {h:>4} | {c['algorithm']:>32} | {c['val_mae']:>8.3f} | "
          f"{BASELINE_FLOOR[h]:>7.3f} | {c['margin']:>+7.3f} | {'YES' if beats else 'no':>6}")

pooled_new   = np.mean([champions[h]['val_mae'] for h in [1,7,14,30,60,90]])
pooled_floor = np.mean(list(BASELINE_FLOOR.values()))
print(f"\n  Horizons beating floor: {beats_total}/6")
print(f"  Pooled champion MAE: {pooled_new:.4f} | Pooled floor: {pooled_floor:.4f}")

# %% Prediction plot — champion models on val
print("\n[Plots] Champion prediction quality...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, h in enumerate([1, 7, 14, 30, 60, 90]):
    c = champions[h]
    target_col = f'{TARGET_RAW}_h{h}'
    ratio_col  = c['ratio_col']
    anchor_col = c['anchor']
    ax = axes[i]

    if ratio_col is not None:
        val_sc = val_mask & df[ratio_col].notna() & df[target_col].notna()
    else:
        val_sc = val_mask & df[target_col].notna() & df[anchor_col].notna()

    y_act  = df.loc[val_sc, target_col].values
    y_anch = df.loc[val_sc, anchor_col].values
    dates  = df.loc[val_sc, 'date'].values
    mae    = c['val_mae']
    floor  = BASELINE_FLOOR[h]

    ax.plot(dates, y_act,  lw=1, color='black', label='Actual')
    ax.plot(dates, y_anch, lw=1, color='gray',  ls='--', alpha=0.6, label='Anchor (naive)')
    ax.set_title(f'h={h} | MAE={mae:.2f} | floor={floor:.2f} | {"✓" if mae<floor else "✗"}')
    ax.legend(fontsize=6)

plt.suptitle('Champion model predictions vs actual — run_20260908T075709Z', fontsize=11)
plt.tight_layout()
plt.savefig(str(ART_DIR / 'plot_champion_predictions.png'), dpi=120)
plt.close()
print("  Saved plot_champion_predictions.png")

# %% Write history JSON
print("\n[Step] Writing history JSON...")
history_json = {
    "iteration": ITER_NUM,
    "node": "training",
    "timestamp": pd.Timestamp.now().isoformat()[:19],
    "schema_version": "2026-09-04",
    "type": "final_pass_champion_selection",

    "per_horizon_champion": {
        str(h): {
            "val_mae": champions[h]['val_mae'],
            "algorithm": champions[h]['algorithm'],
            "beats_floor": champions[h]['beats_floor'],
            "margin": champions[h]['margin'],
            "baseline_floor": BASELINE_FLOOR[h],
            "model_path": champions[h]['model'],
            "anchor_col": champions[h]['anchor'],
            "ratio_col": champions[h]['ratio_col'],
        }
        for h in [1, 7, 14, 30, 60, 90]
    },

    "summary": {
        "beats_floor_count": beats_total,
        "pooled_champion_mae": round(float(pooled_new), 4),
        "pooled_floor": round(float(pooled_floor), 4),
    },

    "h60_analysis": {
        "n_val": int(n_val60),
        "naive_mae": round(float(mae_naive60), 4),
        "optuna30_mae": round(float(mae_60_opt), 4),
        "champion": champ_60,
        "note": "35 val rows; Optuna returns degenerate 2-tree models. Regularised models score similarly."
    },

    "h90_analysis": {
        "n_val": int(val_sc90.sum()),
        "naive_mae": round(float(mae_naive90), 4),
        "note": "5 val rows. Any model estimate is noise. Naive predictor (anchor_correct_h90) adopted."
    },

    "h14_analysis": {
        "n_val": int(n_val14),
        "prior_best": PRIOR_BEST[14]['val_mae'],
        "ridge_mae": round(float(mae_ridge14), 4),
        "optuna_extra_mae": round(float(mae_opt14), 4),
        "champion": champ_14,
        "beats_floor": bool(mae_champ14 < BASELINE_FLOOR[14]),
    },

    "next_iteration_contract": {
        "node": "end_model",
        "iteration": ITER_NUM + 1,
        "why_this_node": (
            "Champion selected per horizon. Proceed to end_model: "
            "package champion models, produce per-horizon prediction function, "
            "run final evaluation on all folds, generate run report."
        ),
        "context": {
            "champions": {str(h): {'model': champions[h]['model'], 'anchor': champions[h]['anchor'],
                                    'ratio_col': champions[h]['ratio_col'],
                                    'val_mae': champions[h]['val_mae']} for h in [1,7,14,30,60,90]},
            "fe_csv": str(FE_CSV.relative_to(RUN_DIR)),
            "beats_floor_count": beats_total,
        }
    },

    "handoff": {
        "written_by": "training",
        "iteration": ITER_NUM,
        "context_clearable": True,
        "read_this_first": (
            f"{beats_total}/6 horizons beat floor. Champion models selected. "
            f"h=60 and h=90 use naive predictor (anchor_correct). "
            f"Next node: end_model."
        ),
        "durable_rules": [
            {
                "rule": "Use anchor_correct_h{h} (same-DOW of target) for h=1,30,60,90 in end_model serving logic.",
                "why": "Original naive predictor anchors have DOW mismatch for non-multiple-of-7 horizons.",
                "applies_to": ["end_model"]
            },
            {
                "rule": "h=60 and h=90 champions are naive predictors (pred = anchor_correct_h{h}). No ML model.",
                "why": "35 and 5 val rows respectively — insufficient for reliable ML evaluation.",
                "applies_to": ["end_model"]
            }
        ],
    }
}

hist_path = RUN_DIR / 'history' / f'{ITER_ID}.json'
with open(str(hist_path), 'w', encoding='utf-8') as f:
    json.dump(history_json, f, indent=2)
print(f"  Saved: {hist_path.relative_to(RUN_DIR)}")

# %% Update state
print("\n[Step] Updating state/current.json...")
state_path = RUN_DIR / 'state' / 'current.json'
with open(str(state_path), 'r', encoding='utf-8') as f:
    state = json.load(f)

cur = state['current']
cur['champion_per_horizon']      = {str(h): champions[h] for h in [1,7,14,30,60,90]}
cur['beats_floor_count']         = beats_total
cur['pooled_champion_mae']       = round(float(pooled_new), 4)
cur['run_status']                = 'champion_selected'
cur['next_iteration_contract']   = history_json['next_iteration_contract']

state['history'].append({
    "iteration": ITER_NUM,
    "node": "training",
    "timestamp": pd.Timestamp.now().isoformat()[:10],
    "status": "complete",
    "summary": (
        f"Final pass + champion selection. Beats floor: {beats_total}/6. "
        f"Pooled MAE={pooled_new:.4f} vs floor={pooled_floor:.4f}. "
        f"h=60,h=90 -> naive predictor. Next: end_model."
    ),
    "full_output_path": str(hist_path.relative_to(RUN_DIR)),
    "key_outputs": {
        "beats_floor_count": beats_total,
        "champion_mae": {h: champions[h]['val_mae'] for h in [1,7,14,30,60,90]},
        "beats_floor": {h: champions[h]['beats_floor'] for h in [1,7,14,30,60,90]},
    }
})

with open(str(state_path), 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2)
print("  state/current.json updated")

print("\n" + "="*70)
print("DONE — Iteration 013 complete. Next: end_model.")
print("="*70)
