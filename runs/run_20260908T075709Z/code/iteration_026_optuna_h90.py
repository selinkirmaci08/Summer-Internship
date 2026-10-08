# %% [iteration_026_optuna_h90] Multi-fold Optuna for h=90 — run_20260908T075709Z
#
# h=90 has never had a proper hyperparameter search — only fixed conservative
# params from iter018 (num_leaves=16, lr=0.05, reg_alpha=2.0, reg_lambda=20.0).
#
# Feature CSV: FE4 (data/iteration_019_feature_engineering.csv, 90 cols, 62 feats)
# Target: bread_quantity_h90 (raw units)
# Current champion: pooled_mae=56.767  floor=75.937
# Eval: pooled multi-fold (fold-3 alone has only 5 val rows — cannot use alone)
#
# 200 Optuna trials, multi-fold weighted objective, inner-split early stopping

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RUN_DIR  = Path(__file__).parent.parent
FE4_CSV  = RUN_DIR / "data" / "iteration_019_feature_engineering.csv"
ART_OUT  = RUN_DIR / "artifacts" / "training" / "iter026"
HIST_OUT = RUN_DIR / "history" / "iteration_026_optuna_h90.json"
STATE_F  = RUN_DIR / "state" / "current.json"
ART_OUT.mkdir(parents=True, exist_ok=True)

N_TRIALS = 200
TGT      = "bread_quantity_h90"
PRIOR_POOLED_MAE  = 56.767
PRIOR_POOLED_FLOOR = 75.937

EXCLUDE_BASE = {
    "date", "bread_quantity", "is_covid_lockdown_period", "day_of_week",
    "_split", "_fold_1", "_fold_2", "_fold_3",
}

print("=== Iteration 026: Multi-fold Optuna for h=90 ===")
df = pd.read_csv(str(FE4_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
feats = [c for c in df.columns
         if c not in EXCLUDE_BASE
         and not c.startswith("bread_quantity_h") and not c.startswith("ratio_h")
         and not c.startswith("ratio_correct_") and not c.startswith("anchor_correct")
         and df[c].dtype != object]
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols  ({len(feats)} model features)")
print(f"Current champion: pooled_mae={PRIOR_POOLED_MAE}  floor={PRIOR_POOLED_FLOOR}")

# Per-fold val sizes for h=90
for fid in [1, 2, 3]:
    n = ((df[f"_fold_{fid}"] == "val") & df[TGT].notna()).sum()
    print(f"  fold-{fid} val rows with h90 notna: {n}")


def fit_es(params, X_tr, y_tr):
    cut = int(len(X_tr) * 0.8)
    m = lgb.LGBMRegressor(n_estimators=2000, random_state=42, **params)
    m.fit(X_tr[:cut], y_tr[:cut],
          eval_set=[(X_tr[cut:], y_tr[cut:])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m, m.best_iteration_


def eval_pooled(params, return_per_fold=False):
    fold_maes, fold_ns = [], []
    per_fold = {}
    for fid in [1, 2, 3]:
        fc = f"_fold_{fid}"
        tr_m = (df[fc] == "train") & (df["is_covid_lockdown_period"] == 0) & df[TGT].notna()
        vl_m = (df[fc] == "val") & df[TGT].notna()
        if vl_m.sum() == 0:
            continue
        X_tr = df.loc[tr_m, feats].fillna(0).values
        y_tr = df.loc[tr_m, TGT].values
        X_vl = df.loc[vl_m, feats].fillna(0).values
        y_vl = df.loc[vl_m, TGT].values
        model, n_est = fit_es(params, X_tr, y_tr)
        mae = float(np.mean(np.abs(model.predict(X_vl) - y_vl)))
        fold_maes.append(mae)
        fold_ns.append(int(vl_m.sum()))
        per_fold[fid] = {"mae": mae, "n": int(vl_m.sum()), "n_est": n_est}
    tot = sum(fold_ns)
    pooled = sum(m * n for m, n in zip(fold_maes, fold_ns)) / tot
    if return_per_fold:
        return pooled, per_fold
    return pooled


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial):
    p = {
        "num_leaves":        trial.suggest_int(  "num_leaves",        8, 128),
        "learning_rate":     trial.suggest_float("learning_rate",     0.005, 0.3, log=True),
        "max_depth":         trial.suggest_int(  "max_depth",         3, 8),
        "min_child_samples": trial.suggest_int(  "min_child_samples", 5, 60),
        "reg_alpha":         trial.suggest_float("reg_alpha",         1e-4, 20.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",        1e-4, 20.0, log=True),
        "subsample":         trial.suggest_float("subsample",         0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.4, 1.0),
        "min_split_gain":    trial.suggest_float("min_split_gain",    0.0, 1.0),
        "objective": "regression_l1", "verbose": -1,
    }
    return eval_pooled(p)


print(f"\nRunning Optuna ({N_TRIALS} trials)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_params = study.best_params.copy()
best_params.update({"objective": "regression_l1", "verbose": -1})
print(f"Best pooled MAE (Optuna): {study.best_value:.3f}")
print(f"Best params: {best_params}")

# ── Full eval with best params ────────────────────────────────────────────────
print("\nEvaluating best params (full per-fold detail)...")
pooled_mae, per_fold = eval_pooled(best_params, return_per_fold=True)

# Compute floor per fold
for fid in [1, 2, 3]:
    fc = f"_fold_{fid}"
    vl_m = (df[fc] == "val") & df[TGT].notna()
    if vl_m.sum() == 0:
        continue
    naive = df.loc[vl_m, "naive_h90_seasonal_median_w26"].values
    y_vl  = df.loc[vl_m, TGT].values
    per_fold[fid]["floor"] = float(np.mean(np.abs(naive - y_vl)))
    print(f"  fold-{fid}: MAE={per_fold[fid]['mae']:.3f}  floor={per_fold[fid]['floor']:.3f}  "
          f"n={per_fold[fid]['n']}  n_est={per_fold[fid]['n_est']}  beats={per_fold[fid]['mae'] < per_fold[fid]['floor']}")

fold_ns = [per_fold[fid]["n"] for fid in per_fold]
fold_floors = [per_fold[fid]["floor"] for fid in per_fold]
tot = sum(fold_ns)
pooled_floor = sum(f * n for f, n in zip(fold_floors, fold_ns)) / tot
improved = pooled_mae < PRIOR_POOLED_MAE

print(f"\nPooled MAE:   {pooled_mae:.3f}  (prior={PRIOR_POOLED_MAE}  floor={pooled_floor:.3f})")
print(f"Beats floor:  {pooled_mae < pooled_floor}")
print(f"Improved vs prior: {improved}")

# ── Save if improved ──────────────────────────────────────────────────────────
saved = {}
if improved:
    # Find avg n_est across folds for full retrain
    avg_nest = int(round(np.mean([per_fold[fid]["n_est"] for fid in per_fold])))
    full_m = df["_fold_3"].isin(["train", "val"]) & (df["is_covid_lockdown_period"] == 0) & df[TGT].notna()
    X_full = df.loc[full_m, feats].fillna(0).values
    y_full = df.loc[full_m, TGT].values
    final = lgb.LGBMRegressor(n_estimators=avg_nest, random_state=42, **best_params)
    final.fit(X_full, y_full)
    pkl026 = ART_OUT / "lgbm_h90_optuna_fe4.pkl"
    pickle.dump(final, open(pkl026, "wb"))
    pkl_final = RUN_DIR / "artifacts" / "training" / "final" / "lgbm_h90_final.pkl"
    pickle.dump(final, open(pkl_final, "wb"))
    saved[90] = str(pkl026.name)
    print(f"\nSaved: {pkl026.name}  n_est={avg_nest}  (updated final/)")
else:
    print(f"\nNo improvement -- keeping iter020 model (pooled_mae={PRIOR_POOLED_MAE})")

# ── Handoff JSON ──────────────────────────────────────────────────────────────
handoff = {
    "iteration": 26,
    "node": "training",
    "timestamp": "2026-09-09",
    "schema_version": "2026-09-04",
    "type": "optuna_h90",
    "summary": (
        f"Multi-fold Optuna ({N_TRIALS} trials) for h=90 (FE4). "
        "First proper hyperparameter search for h=90. "
        f"Pooled MAE={pooled_mae:.3f}  prior={PRIOR_POOLED_MAE}  floor={pooled_floor:.3f}  improved={improved}."
    ),
    "results": {
        "90": {
            "pooled_mae": pooled_mae,
            "pooled_floor": pooled_floor,
            "beats_floor": pooled_mae < pooled_floor,
            "prior_mae": PRIOR_POOLED_MAE,
            "improved": improved,
            "per_fold": {str(fid): per_fold[fid] for fid in per_fold},
            "best_params": best_params,
            "best_optuna_value": study.best_value,
        }
    },
    "saved_models": {str(k): v for k, v in saved.items()},
    "handoff": {
        "written_by": "training",
        "iteration": 26,
        "context_clearable": True,
        "durable_rules": [
            "All LightGBM models must use objective='regression_l1'.",
            "h=60 eval: 95 val target dates (Sep-Dec 2020), features from origin D-60.",
            "h=90 eval: pooled multi-fold (fold-3 alone has only 5 rows).",
        ],
    },
}
with open(HIST_OUT, "w", encoding="utf-8") as f:
    json.dump(handoff, f, indent=2)
print(f"Handoff written: {HIST_OUT.name}")

with open(STATE_F, encoding="utf-8") as f:
    state = json.load(f)
state["current"]["iteration"] = 26
state["current"]["run_status"] = "complete"
if improved:
    state["current"]["champions"]["90"]["source"] = "iter026_optuna_fe4"
    state["current"]["champions"]["90"]["pooled_mae"] = pooled_mae
with open(STATE_F, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
print("State updated: iteration=26")
print("\n=== Iteration 026 complete ===")
