# %% [markdown]
# # Iteration 011 — Optuna Hyperparameter Search (h=7, h=14, ratio target)
# Run: run_20260908T075709Z
# Node: training (hyperparameter_search)
# Reads:  data/iteration_009_feature_engineering.csv
# Writes: artifacts/training/optuna/  (studies + plots)
#         history/iteration_011_optuna_h7_h14.json
#         state/current.json (updated)
#
# Scope: h=7 and h=14 only (ratio target, COVID excluded, inner-fold early stopping)
# These two horizons showed the strongest response to the ratio approach (iter010)
# and are closest to beating the floor.
# n_trials >= 50 per horizon (satisfies hyperparameter_search guardrail)

# %% Imports
import json, pickle, warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"LightGBM {lgb.__version__} | Optuna {optuna.__version__}")
except ImportError as e:
    raise ImportError(f"pip install lightgbm optuna — missing: {e}")

# %% Config
RUN_DIR  = Path('C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z')
FE_CSV   = RUN_DIR / 'data' / 'iteration_009_feature_engineering.csv'
ART_DIR  = RUN_DIR / 'artifacts' / 'training' / 'optuna'
ART_DIR.mkdir(parents=True, exist_ok=True)

TARGET_RAW     = 'bread_quantity'
BASELINE_FLOOR = {7: 31.263, 14: 32.846}
ANCHOR_COLS    = {7: 'naive_h7_seasonal_median_w4', 14: 'naive_h14_seasonal_median_w4'}
HORIZONS_OPT   = [7, 14]
N_TRIALS       = 75      # per horizon; satisfies guardrail (>=50)
RATIO_CLIP     = (0.05, 5.0)
ITER_NUM       = 11
ITER_ID        = f'iteration_{ITER_NUM:03d}_optuna_h7_h14'

# iter010 reference MAEs for comparison
ITER010_MAE    = {7: 32.604, 14: 35.586}

PROTECTED_PREFIXES = ['bread_quantity_h', 'ratio_h', '_split', '_fold']
PROTECTED_EXACT    = ['date', 'bread_quantity', 'is_covid_lockdown_period']

print(f"[Config] n_trials={N_TRIALS} per horizon | horizons={HORIZONS_OPT}")

# %% Load data
print("\n[Step 1] Loading FE data...")
df = pd.read_csv(str(FE_CSV), parse_dates=['date']).sort_values('date').reset_index(drop=True)

train_mask = df['_split'] == 'train'
val_mask   = df['_split'] == 'validation'
covid_mask = df['is_covid_lockdown_period'].astype(bool)
train_clean = train_mask & ~covid_mask

print(f"  Loaded {len(df)} rows x {len(df.columns)} cols")
print(f"  Clean train={train_clean.sum()}, val={val_mask.sum()}")

# Build feature set
feats = [c for c in df.columns
         if c not in PROTECTED_EXACT
         and not any(c.startswith(p) for p in PROTECTED_PREFIXES)
         and df[c].dtype in [np.float64, np.int64, float, int]]
print(f"  Feature pool: {len(feats)} features")

# %% Optuna per horizon
results_optuna = {}
best_models    = {}
best_params    = {}

for h in HORIZONS_OPT:
    ratio_col  = f'ratio_h{h}'
    target_col = f'{TARGET_RAW}_h{h}'
    anchor_col = ANCHOR_COLS[h]

    # Scoreable val rows
    val_sc = val_mask & df[ratio_col].notna() & df[target_col].notna()
    n_val  = val_sc.sum()

    # Training rows with valid ratio
    train_valid = train_clean & df[ratio_col].notna()

    X_tr_full = df.loc[train_valid, feats].values
    y_tr_full = df.loc[train_valid, ratio_col].values

    # Inner val = last 20% of clean train (time-ordered) for early stopping calibration
    n_inner = max(80, int(len(X_tr_full) * 0.20))
    X_tr    = X_tr_full[:-n_inner]
    y_tr    = y_tr_full[:-n_inner]
    X_in    = X_tr_full[-n_inner:]
    y_in    = y_tr_full[-n_inner:]

    X_val_g = df.loc[val_sc, feats].values
    y_val_r = df.loc[val_sc, ratio_col].values
    y_act   = df.loc[val_sc, target_col].values
    anch    = df.loc[val_sc, anchor_col].values

    print(f"\n{'='*60}")
    print(f"Optuna: h={h} | n_train={len(X_tr)} | n_inner={n_inner} | n_val={n_val}")
    print(f"  iter010 MAE={ITER010_MAE[h]:.4f} | floor={BASELINE_FLOOR[h]:.3f}")
    print(f"{'='*60}")

    def objective(trial):
        params = dict(
            objective          = 'regression_l1',
            learning_rate      = trial.suggest_float('learning_rate',    0.005, 0.15,  log=True),
            n_estimators       = trial.suggest_int(  'n_estimators',     100,   1000),
            num_leaves         = trial.suggest_int(  'num_leaves',       10,    63),
            max_depth          = trial.suggest_int(  'max_depth',        3,     8),
            min_child_samples  = trial.suggest_int(  'min_child_samples',10,    60),
            reg_alpha          = trial.suggest_float('reg_alpha',        1e-3,  20.0,  log=True),
            reg_lambda         = trial.suggest_float('reg_lambda',       0.05,  30.0,  log=True),
            subsample          = trial.suggest_float('subsample',        0.5,   1.0),
            colsample_bytree   = trial.suggest_float('colsample_bytree', 0.5,   1.0),
            min_split_gain     = trial.suggest_float('min_split_gain',   0.0,   1.0),
            n_jobs             = -1,
            verbose            = -1,
            random_state       = 42,
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_in, y_in)],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(period=-1)]
        )
        pred_r = model.predict(X_val_g)
        pred_u = pred_r * anch
        return float(np.mean(np.abs(y_act - pred_u)))

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_trial = study.best_trial
    print(f"\n  Best trial #{best_trial.number}: val MAE={best_trial.value:.4f}")
    print(f"  Improvement over iter010: {best_trial.value - ITER010_MAE[h]:+.4f}")
    print(f"  Gap to floor: {best_trial.value - BASELINE_FLOOR[h]:+.4f}")
    print(f"  Beats floor: {'YES' if best_trial.value < BASELINE_FLOOR[h] else 'no'}")
    print(f"  Best params: {best_trial.params}")

    # Refit best model on full clean training set
    bp = dict(
        objective         = 'regression_l1',
        learning_rate     = best_trial.params['learning_rate'],
        n_estimators      = best_trial.params['n_estimators'],
        num_leaves        = best_trial.params['num_leaves'],
        max_depth         = best_trial.params['max_depth'],
        min_child_samples = best_trial.params['min_child_samples'],
        reg_alpha         = best_trial.params['reg_alpha'],
        reg_lambda        = best_trial.params['reg_lambda'],
        subsample         = best_trial.params['subsample'],
        colsample_bytree  = best_trial.params['colsample_bytree'],
        min_split_gain    = best_trial.params['min_split_gain'],
        n_jobs=-1, verbose=-1, random_state=42,
    )
    final_model = lgb.LGBMRegressor(**bp)
    final_model.fit(
        X_tr_full, y_tr_full,
        eval_set=[(X_in, y_in)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(period=-1)]
    )
    n_trees_final = final_model.best_iteration_ or final_model.n_estimators

    pred_r_final = final_model.predict(X_val_g)
    pred_u_final = pred_r_final * anch
    mae_final = float(np.mean(np.abs(y_act - pred_u_final)))
    print(f"\n  Final model (full train): n_trees={n_trees_final}, val MAE={mae_final:.4f}")

    # Save model
    model_path = ART_DIR / f'optuna_lgbm_h{h}.pkl'
    with open(str(model_path), 'wb') as f:
        pickle.dump(final_model, f)

    # Optuna history plot
    trial_values = [t.value for t in study.trials if t.value is not None]
    running_best = np.minimum.accumulate(trial_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(trial_values,  alpha=0.5, color='gray',  lw=0.8, label='Trial MAE')
    axes[0].plot(running_best,  color='steelblue', lw=1.5, label='Running best')
    axes[0].axhline(BASELINE_FLOOR[h], color='red',    ls='--', lw=1, label=f'Floor={BASELINE_FLOOR[h]:.3f}')
    axes[0].axhline(ITER010_MAE[h],    color='orange', ls='--', lw=1, label=f'iter010={ITER010_MAE[h]:.3f}')
    axes[0].set_xlabel('Trial'); axes[0].set_ylabel('Val MAE (units)'); axes[0].set_title(f'h={h} Optuna convergence')
    axes[0].legend(fontsize=8)

    # Importances from final model
    imp = pd.Series(final_model.feature_importances_, index=feats).sort_values(ascending=False).head(15)
    axes[1].barh(imp.index[::-1], imp.values[::-1], color='cornflowerblue')
    axes[1].set_title(f'h={h} Feature importances (top 15)')
    axes[1].set_xlabel('Importance')

    plt.suptitle(f'Optuna h={h} | best MAE={best_trial.value:.3f} | floor={BASELINE_FLOOR[h]:.3f}')
    plt.tight_layout()
    plt.savefig(str(ART_DIR / f'plot_optuna_h{h}.png'), dpi=120)
    plt.close()

    # Store results
    results_optuna[h] = {
        'n_trials':        N_TRIALS,
        'best_trial_num':  best_trial.number,
        'best_val_mae_optuna': round(best_trial.value, 4),
        'final_val_mae':   round(mae_final, 4),
        'n_trees_final':   int(n_trees_final),
        'beats_floor':     bool(mae_final < BASELINE_FLOOR[h]),
        'baseline_floor':  BASELINE_FLOOR[h],
        'margin':          round(mae_final - BASELINE_FLOOR[h], 4),
        'iter010_mae':     ITER010_MAE[h],
        'improvement_over_iter010': round(mae_final - ITER010_MAE[h], 4),
        'best_params':     best_trial.params,
    }
    best_models[h] = final_model
    best_params[h] = best_trial.params

    print(f"  Saved: {model_path.name}, plot_optuna_h{h}.png")

# %% Summary
print("\n" + "="*70)
print("OPTUNA SUMMARY — h=7, h=14")
print("="*70)
print(f"\n  {'h':>4} | {'iter010 MAE':>11} | {'Optuna MAE':>10} | {'Floor':>7} | {'Gap':>7} | {'Beats?':>6}")
print(f"  {'-'*4}-+-{'-'*11}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")
for h in HORIZONS_OPT:
    r = results_optuna[h]
    print(f"  {h:>4} | {r['iter010_mae']:>11.4f} | {r['final_val_mae']:>10.4f} | "
          f"{r['baseline_floor']:>7.3f} | {r['margin']:>+7.3f} | {'YES' if r['beats_floor'] else 'no':>6}")

# %% Write history JSON
print("\n[Step] Writing history JSON...")
history_json = {
    "iteration": ITER_NUM,
    "node": "training",
    "timestamp": pd.Timestamp.now().isoformat()[:19],
    "schema_version": "2026-09-04",
    "type": "hyperparameter_search",
    "reads_from": str(FE_CSV.relative_to(RUN_DIR)),
    "horizons_searched": HORIZONS_OPT,
    "n_trials_per_horizon": N_TRIALS,
    "hyperparameter_search_done": True,
    "results": results_optuna,
    "methodology": {
        "optimizer": "Optuna TPE",
        "objective": "minimize val MAE in units (pred_ratio x anchor)",
        "early_stopping": "inner-fold (last 20% of clean train, 60 rounds patience)",
        "covid_exclusion": "is_covid_lockdown_period==1 excluded from training",
        "ratio_target": "ratio_h{h} = bread_quantity_h{h} / anchor_h{h}",
        "feature_set": f"{len(feats)} features from iter009 FE CSV",
    },
    "handoff": {
        "written_by": "training",
        "iteration": ITER_NUM,
        "context_clearable": True,
        "read_this_first": (
            f"Optuna complete for h=7 and h=14. "
            f"h=7 final MAE={results_optuna[7]['final_val_mae']:.4f} (floor=31.263, gap={results_optuna[7]['margin']:+.4f}). "
            f"h=14 final MAE={results_optuna[14]['final_val_mae']:.4f} (floor=32.846, gap={results_optuna[14]['margin']:+.4f}). "
            "hyperparameter_search_done=True."
        ),
        "parameters": {
            "optuna_best_params_h7":  {"value": best_params.get(7, {}),  "must_not_change": False},
            "optuna_best_params_h14": {"value": best_params.get(14, {}), "must_not_change": False},
        },
        "durable_rules": [
            {
                "rule": "pred_units = pred_ratio x anchor (always convert ratio to units for floor comparison)",
                "why": "Baseline floor is in bread_quantity units",
                "applies_to": ["training", "end_model"]
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
cur['hyperparameter_search_done'] = True
cur['optuna_results'] = {str(h): results_optuna[h] for h in HORIZONS_OPT}
cur['optuna_val_mae_by_horizon'] = {str(h): results_optuna[h]['final_val_mae'] for h in HORIZONS_OPT}
cur['optuna_beats_floor'] = {str(h): results_optuna[h]['beats_floor'] for h in HORIZONS_OPT}
cur['coverage_check']['hyperparameter_search_done'] = True

# Merge optuna MAEs into overall champion tracker
ratio_maes = cur.get('ratio_val_mae_by_horizon', {})
for h in HORIZONS_OPT:
    ratio_maes[str(h)] = results_optuna[h]['final_val_mae']
cur['ratio_val_mae_by_horizon'] = ratio_maes

cur['next_iteration_contract'] = {
    "node": "feature_engineering",
    "iteration": ITER_NUM + 1,
    "why_this_node": (
        "Fix anchor DOW mismatch for h=1 (and h=30, h=60): "
        "naive_h1_seasonal_median_w4 uses same-DOW of ORIGIN; should use same-DOW of TARGET (D+1). "
        "Then retrain ratio model for h=1, h=30, h=60 with corrected anchors."
    ),
}

state['history'].append({
    "iteration": ITER_NUM,
    "node": "training",
    "timestamp": pd.Timestamp.now().isoformat()[:10],
    "status": "complete",
    "summary": (
        f"Optuna h=7 and h=14 (ratio target, {N_TRIALS} trials each). "
        f"h=7: {results_optuna[7]['final_val_mae']:.4f} (floor=31.263, gap={results_optuna[7]['margin']:+.4f}). "
        f"h=14: {results_optuna[14]['final_val_mae']:.4f} (floor=32.846, gap={results_optuna[14]['margin']:+.4f}). "
        f"hyperparameter_search_done=True."
    ),
    "full_output_path": str(hist_path.relative_to(RUN_DIR)),
    "key_outputs": {
        "hyperparameter_search_done": True,
        "optuna_val_mae": {h: results_optuna[h]['final_val_mae'] for h in HORIZONS_OPT},
        "beats_floor": {h: results_optuna[h]['beats_floor'] for h in HORIZONS_OPT},
    }
})

with open(str(state_path), 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2)
print("  state/current.json updated")

print("\n" + "="*70)
print("DONE — Iteration 011 complete")
print("="*70)
