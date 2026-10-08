# %% [iteration_022_h60_fix] Fix h=60 evaluation + Optuna — run_20260908T075709Z
#
# Root cause: iter020 evaluated h=60 using origin-in-val rows (35 rows).
# The baseline floor 40.042 = anchor_correct_h60 MAE on the CORRECT scheme:
#   95 val TARGET dates (Sep-Dec 2020), features from origin D-60.
# Origins are in the gap (60 rows) + early val period (35 rows).
#
# Fix:
#   1. Use correct eval mask: origins where (O+60) falls in val period  →  95 rows
#   2. Try ratio_correct_h60 target (same approach as h=1 and h=30)
#   3. Optuna 100 trials, multi-fold objective (stable signal for short val)
#   4. If best model beats 40.042 on correct 95-row eval → update champion pkl

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

RUN_DIR  = Path(__file__).parent.parent
FE_CSV   = RUN_DIR / "data" / "iteration_019_feature_engineering.csv"
ART_H60  = RUN_DIR / "artifacts" / "training" / "h60_fix"
HIST_OUT = RUN_DIR / "history" / "iteration_022_h60_fix.json"
ART_H60.mkdir(parents=True, exist_ok=True)

BASELINE_FLOOR = 40.042   # anchor_correct_h60 MAE, 95 rows, correct scheme
RATIO_CLIP = (0.05, 5.0)

print("=== Iteration 022: h=60 Fix ===")
df = pd.read_csv(str(FE_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# ── Feature pool ────────────────────────────────────────────────────────────
EXCLUDE_BASE = {
    "date", "bread_quantity", "is_covid_lockdown_period", "day_of_week",
    "_split", "_fold_1", "_fold_2", "_fold_3",
}
ALL_FEATS = [c for c in df.columns
             if c not in EXCLUDE_BASE
             and not c.startswith("bread_quantity_h") and not c.startswith("ratio_h")
             and not c.startswith("ratio_correct_") and not c.startswith("anchor_correct")
             and df[c].dtype != object]
print(f"Feature pool: {len(ALL_FEATS)} features")

# ── Correct 95-row eval mask ─────────────────────────────────────────────────
# Origins where target date (O+60) falls in fold-3 val period
val_start = pd.Timestamp("2020-09-28")
val_end   = pd.Timestamp("2020-12-31")
df["_target_date_h60"] = df["date"] + pd.Timedelta(days=60)
eval95_mask = (
    (df["_target_date_h60"] >= val_start) &
    (df["_target_date_h60"] <= val_end) &
    df["bread_quantity_h60"].notna()
)
print(f"\nCorrect 95-row eval: {eval95_mask.sum()} rows")
print(f"  Eval origins: {df.loc[eval95_mask,'date'].min().date()} to {df.loc[eval95_mask,'date'].max().date()}")
print(f"  Eval targets: {df.loc[eval95_mask,'_target_date_h60'].min().date()} to {df.loc[eval95_mask,'_target_date_h60'].max().date()}")

# Verify anchor_correct_h60 MAE on correct eval = 40.042
anc95   = df.loc[eval95_mask, "anchor_correct_h60"].values
y95_raw = df.loc[eval95_mask, "bread_quantity_h60"].values
print(f"  anchor_correct_h60 MAE (should be 40.042): {np.mean(np.abs(anc95 - y95_raw)):.3f}")

X_eval95 = df.loc[eval95_mask, ALL_FEATS].fillna(0).values
anc_eval95 = df.loc[eval95_mask, "anchor_correct_h60"].values
y_eval95   = df.loc[eval95_mask, "bread_quantity_h60"].values

# ── Training mask (fold-3, non-COVID) ────────────────────────────────────────
tr_mask = (df["_fold_3"] == "train") & (df["is_covid_lockdown_period"] == 0) & df["ratio_correct_h60"].notna()
X_tr = df.loc[tr_mask, ALL_FEATS].fillna(0).values
y_tr = df.loc[tr_mask, "ratio_correct_h60"].values
print(f"\nTraining rows (non-COVID, ratio_correct_h60 target): {tr_mask.sum()}")

# Inner-split early stopping (last 20% of training rows)
INNER_PCT = 0.20
n_inner = max(20, int(INNER_PCT * len(X_tr)))

# ── Multi-fold helper (for stable Optuna objective) ─────────────────────────
fold_data = {}
for fid in [1, 2, 3]:
    fc = f"_fold_{fid}"
    tr_f = (df[fc] == "train") & (df["is_covid_lockdown_period"] == 0) & df["ratio_correct_h60"].notna()
    # Correct eval for each fold: origins where target falls in that fold's val period
    vl_dates = df.loc[df[fc] == "val", "date"]
    if len(vl_dates) == 0:
        continue
    vl_s, vl_e = vl_dates.min(), vl_dates.max()
    vl_f = (df["_target_date_h60"] >= vl_s) & (df["_target_date_h60"] <= vl_e) & df["bread_quantity_h60"].notna()

    Xf_tr  = df.loc[tr_f, ALL_FEATS].fillna(0).values
    yf_tr  = df.loc[tr_f, "ratio_correct_h60"].values
    Xf_vl  = df.loc[vl_f, ALL_FEATS].fillna(0).values
    af_vl  = df.loc[vl_f, "anchor_correct_h60"].values
    yf_raw = df.loc[vl_f, "bread_quantity_h60"].values
    n_in_f = max(20, int(INNER_PCT * len(Xf_tr)))
    fold_data[fid] = (Xf_tr, yf_tr, Xf_vl, af_vl, yf_raw, n_in_f)

    # Baseline on this fold
    fl = np.mean(np.abs(af_vl - yf_raw))
    print(f"  fold-{fid}: eval n={vl_f.sum()}  anchor_correct MAE={fl:.3f}")


# ── Optuna: multi-fold objective (units MAE, ratio_correct_h60) ───────────────
def objective_h60(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, 1.0, log=True),
        "objective": "regression_l1",
        "verbose": -1,
    }
    fold_maes = []
    for fid, (Xf_tr, yf_tr, Xf_vl, af_vl, yf_raw, n_in_f) in fold_data.items():
        model = lgb.LGBMRegressor(n_estimators=2000, random_state=42, **params)
        model.fit(
            Xf_tr[:-n_in_f], yf_tr[:-n_in_f],
            eval_set=[(Xf_tr[-n_in_f:], yf_tr[-n_in_f:])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        trial.set_user_attr(f"n_est_fold{fid}", model.best_iteration_)
        pred_ratio = np.clip(model.predict(Xf_vl), *RATIO_CLIP)
        preds_units = pred_ratio * af_vl
        fold_maes.append(float(np.mean(np.abs(preds_units - yf_raw))))
    return float(np.mean(fold_maes))


print("\nRunning Optuna (100 trials, multi-fold objective)...")
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective_h60, n_trials=100, show_progress_bar=False)

best = study.best_trial
best_params = best.params.copy()
best_params["objective"] = "regression_l1"
best_params["verbose"] = -1
best_n_est = int(np.mean([best.user_attrs.get(f"n_est_fold{fid}", 100) for fid in [1,2,3]]))
print(f"  Best multi-fold MAE: {best.value:.3f}  n_est (avg): {best_n_est}")
print(f"  Best params: {best_params}")


# ── Evaluate best model on correct 95-row eval ─────────────────────────────
final_model_h60 = lgb.LGBMRegressor(n_estimators=best_n_est, random_state=42, **best_params)
final_model_h60.fit(X_tr[:-n_inner], y_tr[:-n_inner],
    eval_set=[(X_tr[-n_inner:], y_tr[-n_inner:])],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
n_est_used = final_model_h60.best_iteration_

pred_ratio95 = np.clip(final_model_h60.predict(X_eval95), *RATIO_CLIP)
preds95_units = pred_ratio95 * anc_eval95
ml_mae95 = float(np.mean(np.abs(preds95_units - y_eval95)))
baseline_mae95 = float(np.mean(np.abs(anc_eval95 - y_eval95)))
beats = ml_mae95 < BASELINE_FLOOR

print(f"\n=== RESULT (correct 95-row eval) ===")
print(f"  ML (ratio_correct_h60, Optuna):   MAE = {ml_mae95:.3f}")
print(f"  anchor_correct_h60 floor:          MAE = {baseline_mae95:.3f}  (= documented baseline {BASELINE_FLOOR})")
print(f"  Beats baseline floor ({BASELINE_FLOOR}):    {'YES' if beats else 'NO'}")
print(f"  n_estimators used: {n_est_used}")

# ── Also try raw-units target ─────────────────────────────────────────────────
print("\n--- Also testing raw-units (bread_quantity_h60) ---")
tr_raw = (df["_fold_3"] == "train") & (df["is_covid_lockdown_period"] == 0) & df["bread_quantity_h60"].notna()
X_tr_raw = df.loc[tr_raw, ALL_FEATS].fillna(0).values
y_tr_raw = df.loc[tr_raw, "bread_quantity_h60"].values
n_inner_raw = max(20, int(INNER_PCT * len(X_tr_raw)))

fold_data_raw = {}
for fid, (Xf_tr, _, Xf_vl, af_vl, yf_raw, n_in_f) in fold_data.items():
    tr_f = (df[f"_fold_{fid}"] == "train") & (df["is_covid_lockdown_period"] == 0) & df["bread_quantity_h60"].notna()
    Xf_tr2 = df.loc[tr_f, ALL_FEATS].fillna(0).values
    yf_tr2 = df.loc[tr_f, "bread_quantity_h60"].values
    n_in_f2 = max(20, int(INNER_PCT * len(Xf_tr2)))
    fold_data_raw[fid] = (Xf_tr2, yf_tr2, Xf_vl, af_vl, yf_raw, n_in_f2)

def objective_h60_raw(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, 1.0, log=True),
        "objective": "regression_l1",
        "verbose": -1,
    }
    fold_maes = []
    for fid, (Xf_tr2, yf_tr2, Xf_vl, af_vl, yf_raw, n_in_f2) in fold_data_raw.items():
        model = lgb.LGBMRegressor(n_estimators=2000, random_state=42, **params)
        model.fit(
            Xf_tr2[:-n_in_f2], yf_tr2[:-n_in_f2],
            eval_set=[(Xf_tr2[-n_in_f2:], yf_tr2[-n_in_f2:])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        trial.set_user_attr(f"n_est_fold{fid}", model.best_iteration_)
        preds = model.predict(Xf_vl)
        fold_maes.append(float(np.mean(np.abs(preds - yf_raw))))
    return float(np.mean(fold_maes))

study_raw = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_raw.optimize(objective_h60_raw, n_trials=100, show_progress_bar=False)
best_raw = study_raw.best_trial
best_params_raw = best_raw.params.copy()
best_params_raw["objective"] = "regression_l1"
best_params_raw["verbose"] = -1
best_n_est_raw = int(np.mean([best_raw.user_attrs.get(f"n_est_fold{fid}", 100) for fid in [1,2,3]]))
print(f"  Best multi-fold MAE (raw-units): {best_raw.value:.3f}  n_est (avg): {best_n_est_raw}")

raw_model = lgb.LGBMRegressor(n_estimators=best_n_est_raw, random_state=42, **best_params_raw)
raw_model.fit(X_tr_raw[:-n_inner_raw], y_tr_raw[:-n_inner_raw],
    eval_set=[(X_tr_raw[-n_inner_raw:], y_tr_raw[-n_inner_raw:])],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
n_est_raw_used = raw_model.best_iteration_

preds95_raw = raw_model.predict(X_eval95)
ml_mae95_raw = float(np.mean(np.abs(preds95_raw - y_eval95)))
beats_raw = ml_mae95_raw < BASELINE_FLOOR
print(f"  Raw-units ML MAE (95-row eval): {ml_mae95_raw:.3f}  beats {BASELINE_FLOOR}: {'YES' if beats_raw else 'NO'}")


# ── Pick champion ────────────────────────────────────────────────────────────
print("\n=== CHAMPION SELECTION ===")
results = {
    "ratio_correct": {"mae": ml_mae95, "beats": beats, "n_est": n_est_used},
    "raw_units":     {"mae": ml_mae95_raw, "beats": beats_raw, "n_est": n_est_raw_used},
    "anchor_floor":  {"mae": baseline_mae95},
}
for k, v in results.items():
    print(f"  {k:18}: {v['mae']:.3f}" + (f"  beats={v['beats']}" if 'beats' in v else ""))

champion_mae = min(ml_mae95, ml_mae95_raw)
if champion_mae < BASELINE_FLOOR:
    if ml_mae95 <= ml_mae95_raw:
        champion_model = final_model_h60
        champion_target = "ratio_correct_h60"
        champion_params = best_params.copy()
        champion_params["n_estimators"] = n_est_used
        champion_mae_final = ml_mae95
        print(f"\n  Champion: ratio_correct_h60 (MAE={ml_mae95:.3f} < floor {BASELINE_FLOOR})")
    else:
        champion_model = raw_model
        champion_target = "bread_quantity_h60"
        champion_params = best_params_raw.copy()
        champion_params["n_estimators"] = n_est_raw_used
        champion_mae_final = ml_mae95_raw
        print(f"\n  Champion: raw_units (MAE={ml_mae95_raw:.3f} < floor {BASELINE_FLOOR})")

    # Retrain on full fold-3 train+val (non-COVID)
    if champion_target == "ratio_correct_h60":
        full_tr = (df["_fold_3"].isin(["train", "val"])) & (df["is_covid_lockdown_period"] == 0) & df["ratio_correct_h60"].notna()
        X_full = df.loc[full_tr, ALL_FEATS].fillna(0).values
        y_full = df.loc[full_tr, "ratio_correct_h60"].values
    else:
        full_tr = (df["_fold_3"].isin(["train", "val"])) & (df["is_covid_lockdown_period"] == 0) & df["bread_quantity_h60"].notna()
        X_full = df.loc[full_tr, ALL_FEATS].fillna(0).values
        y_full = df.loc[full_tr, "bread_quantity_h60"].values

    final_params = champion_params.copy()
    final_params.pop("n_estimators", None)
    final_model = lgb.LGBMRegressor(n_estimators=champion_params["n_estimators"],
                                     random_state=42, **final_params)
    final_model.fit(X_full, y_full)
    pkl_path = ART_H60 / "lgbm_h60_champion.pkl"
    pickle.dump(final_model, open(pkl_path, "wb"))
    print(f"  Final model saved: {pkl_path.name}  trained on {full_tr.sum()} rows")
else:
    print(f"\n  No ML model beats floor {BASELINE_FLOOR}. Champion remains naive.")
    champion_model = None
    champion_target = None
    champion_params = None
    champion_mae_final = BASELINE_FLOOR


# ── Handoff ─────────────────────────────────────────────────────────────────
handoff = {
    "iteration": 22,
    "node": "training",
    "timestamp": "2026-09-09",
    "schema_version": "2026-09-04",
    "type": "h60_fix",
    "summary": (
        "Fixes h=60 evaluation scheme. The baseline floor 40.042 = anchor_correct_h60 "
        "MAE on 95 val TARGET dates (Sep-Dec 2020) using origins D-60 (gap+early-val). "
        "iter020 incorrectly compared ML on 35 val-origin rows against 40.042 from 95-row scheme. "
        "This iteration trains with ratio_correct_h60 and raw-units targets, both evaluated "
        "on the correct 95-row scheme."
    ),
    "evaluation_scheme": {
        "n_rows": 95,
        "description": "Val target dates Sep 28 - Dec 31 2020, features from origin D-60 (gap/early-val rows)",
        "baseline_floor": BASELINE_FLOOR,
        "anchor_correct_mae_verified": round(baseline_mae95, 3),
    },
    "results": {
        "ratio_correct_h60": {
            "optuna_multifold_mae": round(best.value, 3),
            "correct_95row_mae": round(ml_mae95, 3),
            "n_estimators": n_est_used,
            "beats_floor": beats,
            "params": best_params,
        },
        "raw_units_h60": {
            "optuna_multifold_mae": round(best_raw.value, 3),
            "correct_95row_mae": round(ml_mae95_raw, 3),
            "n_estimators": n_est_raw_used,
            "beats_floor": beats_raw,
            "params": best_params_raw,
        },
    },
    "champion": {
        "h60": {
            "target": champion_target,
            "mae_95row": round(champion_mae_final, 3),
            "beats_floor": champion_mae_final < BASELINE_FLOOR,
            "model_path": f"artifacts/training/h60_fix/lgbm_h60_champion.pkl" if champion_model else "NAIVE",
            "params": champion_params,
        }
    },
    "handoff": {
        "written_by": "training",
        "iteration": 22,
        "context_clearable": True,
        "durable_rules": [
            "h=60 evaluation MUST use 95 val target dates (Sep-Dec 2020) with features from origin D-60.",
            "The 35-row val-origin evaluation scheme is NOT comparable to the 40.042 baseline floor.",
            "All LightGBM models must use objective='regression_l1'.",
        ],
    },
}

with open(HIST_OUT, "w", encoding="utf-8") as f:
    json.dump(handoff, f, indent=2, default=str)
print(f"\nHandoff written: {HIST_OUT.name}")
print("\n=== Iteration 022 h60_fix complete ===")
