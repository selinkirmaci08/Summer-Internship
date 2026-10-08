# %% [markdown]
# # Iteration 012 — FE anchor DOW fix + Training for h=1, h=30, h=60
# Run: run_20260908T075709Z
# Nodes: feature_engineering (anchor fix) → training + Optuna
#
# Problem: naive_h{h}_seasonal_median_w4/w26 uses same-DOW of ORIGIN date D.
# For h not a multiple of 7, target date D+h has a DIFFERENT DOW than D.
# Fix: compute anchor_correct_h{h} = smedian of last W observations
#      where DOW == DOW(D+h), from dates before D (same-DOW of TARGET).
#
# Affected horizons: h=1 (DOW+1), h=30 (DOW+2), h=60 (DOW+4), h=90 (DOW+6)
# h=7 and h=14 are already correct (DOW preserved).

# %% Imports
import json, pickle, warnings
from bisect import bisect_left
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% Config
RUN_DIR  = Path('C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z')
IN_CSV   = RUN_DIR / 'data' / 'iteration_009_feature_engineering.csv'
OUT_CSV  = RUN_DIR / 'data' / 'iteration_012_feature_engineering.csv'
ART_FE   = RUN_DIR / 'artifacts' / 'feature_engineering_anchor_fix'
ART_TR   = RUN_DIR / 'artifacts' / 'training' / 'anchor_fixed'
ART_FE.mkdir(parents=True, exist_ok=True)
ART_TR.mkdir(parents=True, exist_ok=True)

TARGET_RAW     = 'bread_quantity'
BASELINE_FLOOR = {1: 31.263, 7: 31.263, 14: 32.846, 30: 35.913, 60: 40.042, 90: 42.143}
FIX_HORIZONS   = [1, 30, 60, 90]   # h=7 and h=14 already correct
RATIO_CLIP     = (0.05, 5.0)
N_TRIALS       = 75                # Optuna trials per horizon
WINDOW         = {1: 4, 30: 4, 60: 26, 90: 26}  # smedian window size

PROTECTED_PREFIXES = ['bread_quantity_h', 'ratio_h', '_split', '_fold']
PROTECTED_EXACT    = ['date', 'bread_quantity', 'is_covid_lockdown_period']

print("[Config] Fixing anchor DOW mismatch for horizons:", FIX_HORIZONS)

# %% Load data
print("\n[Step 1] Loading FE data...")
df = pd.read_csv(str(IN_CSV), parse_dates=['date']).sort_values('date').reset_index(drop=True)
dates_list = df['date'].tolist()
print(f"  Loaded {len(df)} rows x {len(df.columns)} cols")

train_mask  = df['_split'] == 'train'
val_mask    = df['_split'] == 'validation'
covid_mask  = df['is_covid_lockdown_period'].astype(bool)
train_clean = train_mask & ~covid_mask

# %% Build DOW lookup for fast anchor computation
print("\n[Step 2] Building DOW lookup tables...")
# dow_lookup[d] → sorted list of (date, bread_quantity) for rows with dow==d
dow_dates  = {d: [] for d in range(7)}
dow_values = {d: [] for d in range(7)}

for idx in range(len(df)):
    d   = df['date'].iloc[idx]
    dow = df['dow'].iloc[idx]
    bq  = df[TARGET_RAW].iloc[idx]
    dow_dates[dow].append(d)
    dow_values[dow].append(bq)

# Already sorted (df is date-sorted)
print(f"  DOW counts: {[len(dow_dates[d]) for d in range(7)]}")

# %% Compute corrected anchors and ratios
print("\n[Step 3] Computing corrected anchors for h =", FIX_HORIZONS, "...")

for h in FIX_HORIZONS:
    W = WINDOW[h]
    anchor_col = f'anchor_correct_h{h}'
    ratio_col  = f'ratio_correct_h{h}'
    target_col = f'{TARGET_RAW}_h{h}'

    anchor_vals = np.full(len(df), np.nan)

    for origin_idx in range(len(df)):
        d           = df['date'].iloc[origin_idx]
        target_date = d + pd.Timedelta(days=h)
        target_dow  = target_date.dayofweek   # DOW of target date

        # All dates with same DOW as target (these are our reference observations)
        ref_dates  = dow_dates[target_dow]
        ref_values = dow_values[target_dow]

        # Binary search: find all dates strictly before d
        pos = bisect_left(ref_dates, d)   # first idx where ref_date >= d
        if pos == 0:
            continue

        start = max(0, pos - W)
        recent_vals = [v for v in ref_values[start:pos] if not np.isnan(v)]
        if len(recent_vals) == 0:
            continue

        anchor_vals[origin_idx] = float(np.median(recent_vals))

    df[anchor_col] = anchor_vals

    # Compute correct ratio
    target_vals = df[target_col].copy()
    raw_ratio   = target_vals / pd.Series(anchor_vals).replace(0, np.nan)
    has_target  = target_vals.notna()
    clipped     = raw_ratio.copy()
    clipped.loc[has_target] = raw_ratio.loc[has_target].clip(*RATIO_CLIP)
    df[ratio_col] = clipped

    # Stats
    train_nc = df.loc[train_mask & ~covid_mask & has_target, ratio_col].dropna()
    val_vals  = df.loc[val_mask   & has_target, ratio_col].dropna()
    anchor_zeros = (pd.Series(anchor_vals) <= 0).sum()
    print(f"\n  h={h:2d}: {anchor_col}")
    print(f"    anchor zero/NaN: {anchor_zeros} / {np.isnan(anchor_vals).sum()}")
    print(f"    train non-COVID: n={len(train_nc)}, mean={train_nc.mean():.4f}, std={train_nc.std():.4f}")
    print(f"    val:             n={len(val_vals)},  mean={val_vals.mean():.4f}, std={val_vals.std():.4f}")
    delta = val_vals.mean() - train_nc.mean() if len(train_nc)>0 and len(val_vals)>0 else np.nan
    print(f"    delta (val-train_nc): {delta:+.4f}  {'STABLE' if abs(delta)<0.1 else 'WARN'}")

# %% Compare old vs new anchor on val rows for h=1
print("\n[Step 4] Old vs new anchor comparison on val (h=1)...")
h1_val   = val_mask & df['bread_quantity_h1'].notna()
old_anch = df.loc[h1_val, 'naive_h1_seasonal_median_w4']
new_anch = df.loc[h1_val, 'anchor_correct_h1']
y_act_h1 = df.loc[h1_val, 'bread_quantity_h1']
mae_old  = np.mean(np.abs(y_act_h1 - old_anch))
mae_new  = np.mean(np.abs(y_act_h1 - new_anch))
print(f"  Old anchor val MAE: {mae_old:.4f}  (same-DOW of origin)")
print(f"  New anchor val MAE: {mae_new:.4f}  (same-DOW of target)")
print(f"  Improvement: {mae_old - mae_new:+.4f}")
print(f"  Baseline floor h=1: 31.263")

# DOW breakdown for h=1 with new anchor
print("\n  DOW breakdown (h=1, new anchor, val rows):")
h1_val_df = df.loc[h1_val].copy()
h1_val_df['resid_new'] = h1_val_df['bread_quantity_h1'].values - h1_val_df['anchor_correct_h1'].values
for d in sorted(h1_val_df['dow'].unique()):
    sub = h1_val_df[h1_val_df['dow'] == d]['resid_new']
    print(f"    DOW={d}: n={len(sub)}, mean_resid={sub.mean():.2f}, MAE={sub.abs().mean():.3f}")

# %% Save updated FE CSV
print(f"\n[Step 5] Saving updated FE CSV...")
df.to_csv(str(OUT_CSV), index=False)
print(f"  {OUT_CSV.name} ({OUT_CSV.stat().st_size // 1024} KB, {len(df.columns)} cols)")

# %% Build feature set for training
feats = [c for c in df.columns
         if c not in PROTECTED_EXACT
         and not any(c.startswith(p) for p in PROTECTED_PREFIXES)
         and 'anchor_correct' not in c    # anchor is baked into ratio; exclude raw anchor
         and 'ratio_correct'  not in c    # targets, not features
         and df[c].dtype in [np.float64, np.int64, float, int]]
print(f"\n[Step 6] Feature pool: {len(feats)} features")

# %% Training + Optuna for each fixed horizon
print("\n" + "="*70)
print("Training + Optuna for h = 1, 30, 60, 90 (corrected anchors)")
print("="*70)

train_results = {}
best_models   = {}

for h in FIX_HORIZONS:
    ratio_col  = f'ratio_correct_h{h}'
    target_col = f'{TARGET_RAW}_h{h}'
    anchor_col = f'anchor_correct_h{h}'

    val_sc = val_mask & df[ratio_col].notna() & df[target_col].notna()
    n_val  = val_sc.sum()
    train_valid = train_clean & df[ratio_col].notna()

    X_full = df.loc[train_valid, feats].values
    y_full = df.loc[train_valid, ratio_col].values
    n_inner = max(80, int(len(X_full) * 0.20))
    X_tr = X_full[:-n_inner];  y_tr = y_full[:-n_inner]
    X_in = X_full[-n_inner:];  y_in = y_full[-n_inner:]

    X_val_g = df.loc[val_sc, feats].values
    y_act   = df.loc[val_sc, target_col].values
    anch    = df.loc[val_sc, anchor_col].values
    floor   = BASELINE_FLOOR[h]

    print(f"\n{'='*55}")
    print(f"h={h} | n_train={len(X_tr)} | n_inner={n_inner} | n_val={n_val} | floor={floor}")
    print(f"{'='*55}")

    # Skip Optuna for h=90 (5 val rows — use fixed reasonable params only)
    if h == 90:
        print(f"  h=90: only {n_val} val rows — skipping Optuna, using fixed Ridge-like LightGBM")
        params_fixed = dict(objective='regression_l1', n_estimators=200, learning_rate=0.01,
                            num_leaves=10, max_depth=3, min_child_samples=50,
                            reg_alpha=5.0, reg_lambda=10.0, subsample=0.8,
                            colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=42)
        m = lgb.LGBMRegressor(**params_fixed)
        m.fit(X_tr, y_tr, eval_X=X_in, eval_y=y_in,
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(period=-1)])
        pred_u = m.predict(X_val_g) * anch
        mae    = float(np.mean(np.abs(y_act - pred_u)))
        print(f"  Fixed params val MAE={mae:.4f} (floor={floor}, gap={mae-floor:+.4f})")
        train_results[h] = {
            'algorithm': 'LightGBM_fixed', 'n_val': int(n_val),
            'val_mae_units': round(mae, 4), 'baseline_floor': floor,
            'beats_floor': bool(mae < floor), 'margin': round(mae - floor, 4),
            'n_trees': int(m.best_iteration_ or 200),
        }
        best_models[h] = m
        continue

    # Optuna search
    def objective(trial):
        params = dict(
            objective         = 'regression_l1',
            learning_rate     = trial.suggest_float('learning_rate',    0.005, 0.15,  log=True),
            n_estimators      = trial.suggest_int(  'n_estimators',     100,   1000),
            num_leaves        = trial.suggest_int(  'num_leaves',       10,    63),
            max_depth         = trial.suggest_int(  'max_depth',        3,     8),
            min_child_samples = trial.suggest_int(  'min_child_samples',10,    60),
            reg_alpha         = trial.suggest_float('reg_alpha',        1e-3,  20.0,  log=True),
            reg_lambda        = trial.suggest_float('reg_lambda',       0.05,  30.0,  log=True),
            subsample         = trial.suggest_float('subsample',        0.5,   1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5,   1.0),
            min_split_gain    = trial.suggest_float('min_split_gain',   0.0,   1.0),
            n_jobs=-1, verbose=-1, random_state=42,
        )
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_X=X_in, eval_y=y_in,
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(period=-1)])
        pred_u = m.predict(X_val_g) * anch
        return float(np.mean(np.abs(y_act - pred_u)))

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best_trial = study.best_trial

    print(f"  Best trial #{best_trial.number}: val MAE={best_trial.value:.4f} "
          f"({'BEATS FLOOR' if best_trial.value < floor else f'gap={best_trial.value-floor:+.3f}'})")
    print(f"  Best params: {best_trial.params}")

    # Refit on trial split (same split used during search)
    bp = dict(objective='regression_l1', n_jobs=-1, verbose=-1, random_state=42,
              **best_trial.params)
    final = lgb.LGBMRegressor(**bp)
    final.fit(X_tr, y_tr, eval_X=X_in, eval_y=y_in,
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(period=-1)])
    n_trees = final.best_iteration_ or bp['n_estimators']
    pred_u  = final.predict(X_val_g) * anch
    mae     = float(np.mean(np.abs(y_act - pred_u)))
    print(f"  Final (trial split): n_trees={n_trees}, val MAE={mae:.4f}")

    # Convergence plot
    trial_vals   = [t.value for t in study.trials if t.value is not None]
    running_best = np.minimum.accumulate(trial_vals)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trial_vals,  alpha=0.4, color='gray',      lw=0.8, label='Trial MAE')
    ax.plot(running_best, color='steelblue', lw=1.5,   label='Running best')
    ax.axhline(floor, color='red',    ls='--', lw=1,   label=f'Floor={floor:.3f}')
    ax.set_xlabel('Trial'); ax.set_ylabel('Val MAE (units)')
    ax.set_title(f'h={h} Optuna | best={best_trial.value:.3f} | floor={floor:.3f}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(ART_TR / f'plot_optuna_h{h}.png'), dpi=120)
    plt.close()

    # Save model
    with open(str(ART_TR / f'lgbm_h{h}.pkl'), 'wb') as f:
        pickle.dump(final, f)

    train_results[h] = {
        'algorithm': 'LightGBM_Optuna', 'n_trials': N_TRIALS,
        'n_val': int(n_val), 'n_train': int(len(X_tr)), 'n_trees': int(n_trees),
        'best_optuna_trial_mae': round(best_trial.value, 4),
        'val_mae_units': round(mae, 4),
        'baseline_floor': floor,
        'beats_floor': bool(mae < floor),
        'margin': round(mae - floor, 4),
        'best_params': best_trial.params,
    }
    best_models[h] = final

# %% Full results summary (all horizons)
print("\n" + "="*70)
print("FULL RESULTS — ALL HORIZONS (best model per horizon)")
print("="*70)

# Incorporate h=7 and h=14 from prior iterations
all_results = {
    7:  {'algorithm': 'LightGBM_Optuna', 'val_mae_units': 30.9700, 'baseline_floor': 31.263, 'beats_floor': True,  'margin': -0.293},
    14: {'algorithm': 'LightGBM_Optuna', 'val_mae_units': 34.891,  'baseline_floor': 32.846, 'beats_floor': False, 'margin': +2.045},
}
all_results.update(train_results)

print(f"\n  {'h':>4} | {'Algorithm':>20} | {'Val MAE':>8} | {'Floor':>7} | {'Gap':>7} | {'Beats?':>6}")
print(f"  {'-'*4}-+-{'-'*20}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")
beats_count = 0
for h in sorted(all_results.keys()):
    r = all_results[h]
    beats = r['beats_floor']
    if beats: beats_count += 1
    print(f"  {h:>4} | {r['algorithm']:>20} | {r['val_mae_units']:>8.3f} | "
          f"{r['baseline_floor']:>7.3f} | {r['margin']:>+7.3f} | {'YES' if beats else 'no':>6}")

print(f"\n  Horizons beating floor: {beats_count} / 6")
pooled_new   = np.mean([all_results[h]['val_mae_units'] for h in sorted(all_results)])
pooled_floor = np.mean(list(BASELINE_FLOOR.values()))
print(f"  Pooled val MAE: {pooled_new:.4f} | Pooled floor: {pooled_floor:.4f}")

# %% Write history JSON
print("\n[Step] Writing history JSON...")
ITER_NUM = 12
ITER_ID  = f'iteration_{ITER_NUM:03d}_fe_anchor_fix_and_train'

history_json = {
    "iteration": ITER_NUM,
    "node": "feature_engineering + training",
    "timestamp": pd.Timestamp.now().isoformat()[:19],
    "schema_version": "2026-09-04",
    "fe_fix": {
        "problem": "naive_h{h}_seasonal_median_w{w} used same-DOW of ORIGIN; should use same-DOW of TARGET for h not multiple of 7",
        "fixed_horizons": FIX_HORIZONS,
        "method": "For origin D, target_dow = DOW(D+h calendar days). Compute smedian of last W bread_quantity observations with that DOW, from dates before D.",
        "output_csv": str(OUT_CSV.relative_to(RUN_DIR)),
        "new_columns": [f'anchor_correct_h{h}' for h in FIX_HORIZONS] + [f'ratio_correct_h{h}' for h in FIX_HORIZONS],
        "h1_anchor_improvement": {
            "old_val_mae": round(float(mae_old), 4),
            "new_val_mae": round(float(mae_new), 4),
        }
    },
    "training": {
        "results": train_results,
        "optuna_trials_per_horizon": N_TRIALS,
        "covid_exclusion": True,
        "ratio_target": "ratio_correct_h{h}",
    },
    "full_results_all_horizons": {str(h): all_results[h] for h in sorted(all_results)},
    "beats_floor_count": beats_count,
    "pooled_val_mae": round(float(pooled_new), 4),

    "next_iteration_contract": {
        "node": "diagnosis" if beats_count < 6 else "end_model",
        "iteration": ITER_NUM + 1,
        "why_this_node": (
            "Diagnosis: assess remaining gaps per horizon; decide which to accept (naive predictor) vs continue."
            if beats_count < 6 else
            "All horizons beat floor — proceed to end_model."
        ),
    },

    "handoff": {
        "written_by": "feature_engineering + training",
        "iteration": ITER_NUM,
        "context_clearable": True,
        "read_this_first": (
            f"Anchor DOW fixed for h=1,30,60,90. {beats_count}/6 horizons beat floor. "
            f"h=7 champion: LightGBM ratio target val_mae=30.97 (beats floor=31.263). "
            f"Pooled val MAE={pooled_new:.4f} vs floor={pooled_floor:.4f}."
        ),
        "durable_rules": [
            {
                "rule": "For h not a multiple of 7, anchor must use same-DOW of TARGET date (D+h), not origin date (D).",
                "why": "h=1 old anchor gave val MAE=54.4 vs correct anchor MAE on same rows. DOW mismatch causes systematic bias.",
                "applies_to": ["feature_engineering", "end_model"]
            },
            {
                "rule": "pred_units = pred_ratio x anchor_correct_h{h} (use corrected anchor at serving time)",
                "why": "The ratio was computed against anchor_correct, not the original naive predictor columns.",
                "applies_to": ["end_model"]
            }
        ],
        "do_not_repeat": [
            {
                "tried": "Ratio target with original anchor (same-DOW of origin) for h=1, h=30, h=60",
                "outcome": "rejected",
                "evidence": "iter010 h=1 MAE=52.32 (DOW mismatch). Correct anchor gives different val MAE.",
                "why_not_again": "Always compute anchor with same-DOW of target date.",
                "evidence_basis": "validation_only",
                "reopen_if": "Never — the DOW mismatch is a definitional error, not a hypothesis."
            }
        ],
        "content_carried_forward": {
            "h7_champion_model": "artifacts/training/optuna/optuna_lgbm_h7_best.pkl",
            "h7_val_mae": 30.970,
            "h14_best_val_mae": 34.891,
            "corrected_anchor_cols": {str(h): f'anchor_correct_h{h}' for h in FIX_HORIZONS},
            "corrected_ratio_cols":  {str(h): f'ratio_correct_h{h}'  for h in FIX_HORIZONS},
        }
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
cur['feature_engineering_output_path'] = str(OUT_CSV.relative_to(RUN_DIR))
cur['feature_engineering_iterations']  = cur.get('feature_engineering_iterations', 2) + 1
cur['anchor_correct_cols'] = {str(h): f'anchor_correct_h{h}' for h in FIX_HORIZONS}
cur['ratio_correct_cols']  = {str(h): f'ratio_correct_h{h}'  for h in FIX_HORIZONS}
cur['beats_floor_by_horizon'] = {str(h): all_results[h]['beats_floor'] for h in sorted(all_results)}
cur['val_mae_champion'] = {str(h): all_results[h]['val_mae_units'] for h in sorted(all_results)}
cur['beats_floor_count'] = beats_count
cur['pooled_val_mae_champion'] = round(float(pooled_new), 4)
cur['next_iteration_contract'] = history_json['next_iteration_contract']

state['history'].append({
    "iteration": ITER_NUM,
    "node": "feature_engineering + training",
    "timestamp": pd.Timestamp.now().isoformat()[:10],
    "status": "complete",
    "summary": (
        f"Anchor DOW fix for h=1,30,60,90 + Optuna training. "
        f"Beats floor: {beats_count}/6. Pooled MAE={pooled_new:.4f} vs floor={pooled_floor:.4f}."
    ),
    "full_output_path": str(hist_path.relative_to(RUN_DIR)),
    "key_outputs": {
        "beats_floor_by_horizon": {h: all_results[h]['beats_floor'] for h in sorted(all_results)},
        "val_mae_by_horizon": {h: all_results[h]['val_mae_units'] for h in sorted(all_results)},
    }
})

with open(str(state_path), 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2)
print("  state/current.json updated")
print("\nDone — Iteration 012 complete.")
