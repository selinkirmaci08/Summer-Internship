# %% [iteration_025_optuna_h7_h14] Multi-fold Optuna for h=7 and h=14 — run_20260908T075709Z
#
# h=7:  FE5 CSV (92 cols, 64 feats), target=ratio_h7, anchor=naive_h7_seasonal_median_w4
#        Current champion: MAE=33.016 (iter024, FE5, pinned iter011 params)
# h=14: FE4 CSV (90 cols, 62 feats), target=bread_quantity_h14 (raw units)
#        Current champion: MAE=37.797 (iter020, FE4, fixed iter018 params) — only 0.32 above floor
#
# Both: 200 Optuna trials, multi-fold objective (avg MAE across folds 1-2-3),
#        inner-split early stopping, objective='regression_l1'

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
FE5_CSV  = RUN_DIR / "data" / "iteration_023_feature_engineering.csv"
ART_OUT  = RUN_DIR / "artifacts" / "training" / "iter025"
HIST_OUT = RUN_DIR / "history" / "iteration_025_optuna_h7_h14.json"
STATE_F  = RUN_DIR / "state" / "current.json"
ART_OUT.mkdir(parents=True, exist_ok=True)

RATIO_CLIP = (0.05, 5.0)
N_TRIALS   = 200

EXCLUDE_BASE = {
    "date", "bread_quantity", "is_covid_lockdown_period", "day_of_week",
    "_split", "_fold_1", "_fold_2", "_fold_3",
}

def get_feats(df):
    return [c for c in df.columns
            if c not in EXCLUDE_BASE
            and not c.startswith("bread_quantity_h") and not c.startswith("ratio_h")
            and not c.startswith("ratio_correct_") and not c.startswith("anchor_correct")
            and df[c].dtype != object]

print("=== Iteration 025: Multi-fold Optuna for h=7 and h=14 ===")

df4 = pd.read_csv(str(FE4_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df5 = pd.read_csv(str(FE5_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
feats4 = get_feats(df4)
feats5 = get_feats(df5)
print(f"FE4: {df4.shape[0]} rows x {df4.shape[1]} cols  ({len(feats4)} model features)")
print(f"FE5: {df5.shape[0]} rows x {df5.shape[1]} cols  ({len(feats5)} model features)")


# ── helper: inner-split early stopping ───────────────────────────────────────
def fit_es(params, X_tr, y_tr):
    n = len(X_tr)
    cut = int(n * 0.8)
    m = lgb.LGBMRegressor(n_estimators=2000, random_state=42, **params)
    m.fit(X_tr[:cut], y_tr[:cut],
          eval_set=[(X_tr[cut:], y_tr[cut:])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m, m.best_iteration_


# ═══════════════════════════════════════════════════════════════════════════════
# H = 7  (FE5 features, ratio_h7 target)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"H=7 Optuna ({N_TRIALS} trials, multi-fold, FE5)")
print(f"  Current champion: MAE=33.016  floor=34.125")

df_h7  = df5
feats7 = feats5
TGT7   = "ratio_h7"
ANC7   = "naive_h7_seasonal_median_w4"
ACT7   = "bread_quantity_h7"

def obj_h7(trial):
    p = {
        "num_leaves":       trial.suggest_int(  "num_leaves",       8, 128),
        "learning_rate":    trial.suggest_float("learning_rate",    0.005, 0.3, log=True),
        "max_depth":        trial.suggest_int(  "max_depth",        3, 8),
        "min_child_samples":trial.suggest_int(  "min_child_samples",5, 60),
        "reg_alpha":        trial.suggest_float("reg_alpha",        1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda",       1e-4, 10.0, log=True),
        "subsample":        trial.suggest_float("subsample",        0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_split_gain":   trial.suggest_float("min_split_gain",   0.0, 1.0),
        "objective": "regression_l1", "verbose": -1,
    }
    fold_maes = []
    for fid in [1, 2, 3]:
        fc = f"_fold_{fid}"
        tr_m = (df_h7[fc] == "train") & (df_h7["is_covid_lockdown_period"] == 0) & df_h7[TGT7].notna()
        vl_m = (df_h7[fc] == "val") & df_h7[TGT7].notna()
        if vl_m.sum() == 0:
            continue
        X_tr = df_h7.loc[tr_m, feats7].fillna(0).values
        y_tr = df_h7.loc[tr_m, TGT7].values
        X_vl = df_h7.loc[vl_m, feats7].fillna(0).values
        a_vl = df_h7.loc[vl_m, ANC7].values
        y_vl = df_h7.loc[vl_m, ACT7].values
        model, _ = fit_es(p, X_tr, y_tr)
        preds = np.clip(model.predict(X_vl), *RATIO_CLIP) * a_vl
        fold_maes.append(float(np.mean(np.abs(preds - y_vl))))
    return float(np.mean(fold_maes))

study7 = optuna.create_study(direction="minimize")
study7.optimize(obj_h7, n_trials=N_TRIALS, show_progress_bar=False)

best7 = study7.best_params.copy()
best7.update({"objective": "regression_l1", "verbose": -1})
print(f"  Best multi-fold MAE: {study7.best_value:.3f}  params: {best7}")

# Evaluate best h=7 params on fold-3 val
tr3_h7 = (df_h7["_fold_3"] == "train") & (df_h7["is_covid_lockdown_period"] == 0) & df_h7[TGT7].notna()
vl3_h7 = (df_h7["_fold_3"] == "val") & df_h7[TGT7].notna()
X_tr7  = df_h7.loc[tr3_h7, feats7].fillna(0).values
y_tr7  = df_h7.loc[tr3_h7, TGT7].values
X_vl7  = df_h7.loc[vl3_h7, feats7].fillna(0).values
a_vl7  = df_h7.loc[vl3_h7, ANC7].values
y_vl7  = df_h7.loc[vl3_h7, ACT7].values

m7, n7 = fit_es(best7, X_tr7, y_tr7)
preds7 = np.clip(m7.predict(X_vl7), *RATIO_CLIP) * a_vl7
mae7   = float(np.mean(np.abs(preds7 - y_vl7)))
improved7 = mae7 < 33.016
print(f"  Fold-3 val MAE: {mae7:.3f}  n_est={n7}  prior=33.016  improved={improved7}")

RES7 = {"mae": mae7, "n_est": n7, "floor": 34.125, "prior_mae": 33.016,
         "beats_floor": mae7 < 34.125, "improved": improved7,
         "best_params": best7, "best_multifold_mae": study7.best_value}


# ═══════════════════════════════════════════════════════════════════════════════
# H = 14  (FE4 features, raw units target)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"H=14 Optuna ({N_TRIALS} trials, multi-fold, FE4)")
print(f"  Current champion: MAE=37.797  floor=38.116  (margin=0.319)")

df_h14  = df4
feats14 = feats4
TGT14   = "bread_quantity_h14"
ACT14   = "bread_quantity_h14"

def obj_h14(trial):
    p = {
        "num_leaves":       trial.suggest_int(  "num_leaves",       8, 128),
        "learning_rate":    trial.suggest_float("learning_rate",    0.005, 0.3, log=True),
        "max_depth":        trial.suggest_int(  "max_depth",        3, 8),
        "min_child_samples":trial.suggest_int(  "min_child_samples",5, 60),
        "reg_alpha":        trial.suggest_float("reg_alpha",        1e-4, 20.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda",       1e-4, 20.0, log=True),
        "subsample":        trial.suggest_float("subsample",        0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_split_gain":   trial.suggest_float("min_split_gain",   0.0, 1.0),
        "objective": "regression_l1", "verbose": -1,
    }
    fold_maes = []
    for fid in [1, 2, 3]:
        fc = f"_fold_{fid}"
        tr_m = (df_h14[fc] == "train") & (df_h14["is_covid_lockdown_period"] == 0) & df_h14[TGT14].notna()
        vl_m = (df_h14[fc] == "val") & df_h14[TGT14].notna()
        if vl_m.sum() == 0:
            continue
        X_tr = df_h14.loc[tr_m, feats14].fillna(0).values
        y_tr = df_h14.loc[tr_m, TGT14].values
        X_vl = df_h14.loc[vl_m, feats14].fillna(0).values
        y_vl = df_h14.loc[vl_m, ACT14].values
        model, _ = fit_es(p, X_tr, y_tr)
        fold_maes.append(float(np.mean(np.abs(model.predict(X_vl) - y_vl))))
    return float(np.mean(fold_maes))

study14 = optuna.create_study(direction="minimize")
study14.optimize(obj_h14, n_trials=N_TRIALS, show_progress_bar=False)

best14 = study14.best_params.copy()
best14.update({"objective": "regression_l1", "verbose": -1})
print(f"  Best multi-fold MAE: {study14.best_value:.3f}  params: {best14}")

# Evaluate on fold-3 val
tr3_14 = (df_h14["_fold_3"] == "train") & (df_h14["is_covid_lockdown_period"] == 0) & df_h14[TGT14].notna()
vl3_14 = (df_h14["_fold_3"] == "val") & df_h14[TGT14].notna()
X_tr14 = df_h14.loc[tr3_14, feats14].fillna(0).values
y_tr14 = df_h14.loc[tr3_14, TGT14].values
X_vl14 = df_h14.loc[vl3_14, feats14].fillna(0).values
y_vl14 = df_h14.loc[vl3_14, ACT14].values

m14, n14 = fit_es(best14, X_tr14, y_tr14)
mae14     = float(np.mean(np.abs(m14.predict(X_vl14) - y_vl14)))
improved14 = mae14 < 37.797
print(f"  Fold-3 val MAE: {mae14:.3f}  n_est={n14}  prior=37.797  floor=38.116  improved={improved14}")

RES14 = {"mae": mae14, "n_est": n14, "floor": 38.116, "prior_mae": 37.797,
          "beats_floor": mae14 < 38.116, "improved": improved14,
          "best_params": best14, "best_multifold_mae": study14.best_value}


# ── Save improved models (full retrain on fold-3 train+val) ──────────────────
print(f"\n{'='*60}")
print("SAVING IMPROVED MODELS")
saved = {}
full_tr_mask = lambda df, tgt: df["_fold_3"].isin(["train", "val"]) & (df["is_covid_lockdown_period"] == 0) & df[tgt].notna()

for h, res, df_h, tgt, feats, anc, prior_pkl_name in [
    (7,  RES7,  df_h7,  TGT7,  feats7,  ANC7,  "lgbm_h7_final.pkl"),
    (14, RES14, df_h14, TGT14, feats14, None,  "lgbm_h14_final.pkl"),
]:
    if res["improved"]:
        cfg = res["best_params"].copy()
        n_est = res["n_est"]
        full_m = full_tr_mask(df_h, tgt)
        X_full = df_h.loc[full_m, feats].fillna(0).values
        y_full = df_h.loc[full_m, tgt].values
        final = lgb.LGBMRegressor(n_estimators=n_est, random_state=42, **cfg)
        final.fit(X_full, y_full)
        # iter025 copy
        pkl025 = ART_OUT / f"lgbm_h{h}_optuna_fe{'5' if h==7 else '4'}.pkl"
        pickle.dump(final, open(pkl025, "wb"))
        # overwrite final/
        pkl_final = RUN_DIR / "artifacts" / "training" / "final" / f"lgbm_h{h}_final.pkl"
        pickle.dump(final, open(pkl_final, "wb"))
        saved[h] = str(pkl025.name)
        print(f"  h={h}: IMPROVED ({res['prior_mae']:.3f} -> {res['mae']:.3f}) -- saved {pkl025.name} + updated final/")
    else:
        print(f"  h={h}: no improvement ({res['prior_mae']:.3f} -> {res['mae']:.3f}) -- keeping prior model")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'h':>3}  {'Prior MAE':>10}  {'New MAE':>10}  {'Floor':>8}  {'Beats?':>7}  {'Improved?':>10}")
print("-"*60)
for h, res in [(7, RES7), (14, RES14)]:
    print(f"  {h:>2}  {res['prior_mae']:>10.3f}  {res['mae']:>10.3f}  {res['floor']:>8.3f}  "
          f"{'YES' if res['beats_floor'] else 'NO':>7}  {'YES' if res['improved'] else 'no':>10}")


# ── Handoff JSON ──────────────────────────────────────────────────────────────
handoff = {
    "iteration": 25,
    "node": "training",
    "timestamp": "2026-09-09",
    "schema_version": "2026-09-04",
    "type": "optuna_h7_h14",
    "summary": (
        f"Multi-fold Optuna ({N_TRIALS} trials) for h=7 (FE5) and h=14 (FE4). "
        "Inner-split early stopping for n_estimators. Only updates champion if improved."
    ),
    "results": {"7": RES7, "14": RES14},
    "saved_models": saved,
    "handoff": {
        "written_by": "training",
        "iteration": 25,
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
print(f"\nHandoff written: {HIST_OUT.name}")

with open(STATE_F, encoding="utf-8") as f:
    state = json.load(f)
state["current"]["iteration"] = 25
state["current"]["run_status"] = "complete" if not saved else "in_progress"
if saved:
    for h_str, pkl_name in [(str(h), n) for h, n in saved.items()]:
        h = int(h_str)
        state["current"]["champions"][h_str]["source"] = f"iter025_optuna_fe{'5' if h==7 else '4'}"
        state["current"]["champions"][h_str]["fold3_mae"] = RES7["mae"] if h == 7 else RES14["mae"]
    state["current"]["run_status"] = "complete"
with open(STATE_F, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
print("State updated: iteration=25")
print("\n=== Iteration 025 complete ===")
