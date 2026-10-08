# %% [iteration_020_training_with_fe4] Training with FE4 — run_20260908T075709Z
#
# Uses iteration_019_feature_engineering.csv (90 cols).
# CRITICAL fix vs first draft: all models use objective='regression_l1' (MAE loss).
# Without this, MSE is pulled toward outlier post-COVID ratio rows and fold-3 fails.
#
# Strategy per horizon:
#   h=1  : ratio_correct target, Optuna 75 trials (first Optuna for h=1)
#   h=7  : ratio target, keep existing config + regression_l1 + new features
#   h=14 : raw-units target, keep existing config + regression_l1 + new features
#   h=30 : ratio_correct target, Optuna 75 trials (first Optuna for h=30)
#   h=60 : raw-units target, Optuna 75 trials, evaluate on ALL val rows vs 40.042 floor
#   h=90 : raw-units target, keep existing config, multi-fold
#
# Evaluation: ALL fold-3 val rows where target.notna() (matches iter018 method).
# Feature definition: EXCLUDE_BASE approach with fillna(0).

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
ART_OUT  = RUN_DIR / "artifacts" / "training" / "iter020"
HIST_OUT = RUN_DIR / "history" / "iteration_020_training_with_fe4.json"
STATE_F  = RUN_DIR / "state" / "current.json"
ART_OUT.mkdir(parents=True, exist_ok=True)

BASELINE_FLOOR = {1: 31.263, 7: 31.263, 14: 32.846, 30: 35.913, 60: 40.042, 90: 42.143}
PRIOR_FOLD3_MAE = {1: 29.378, 7: 31.188, 14: 28.865, 30: 34.425, 60: 40.042, 90: 19.856}
RATIO_CLIP = (0.05, 5.0)
N_TRIALS   = 75
INNER_PCT  = 0.20

# ── Load data ─────────────────────────────────────────────────────────────
print("=== Iteration 020: Training with FE4 features ===")
df = pd.read_csv(str(FE_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# ── Feature pool: match iter018's EXCLUDE_BASE approach ───────────────────
# Exclude: date, bread_quantity, is_covid_lockdown_period, day_of_week,
#          _split, _fold_*, all bread_quantity_h*, ratio_h*, ratio_correct_h*
# Anchor columns are NOT excluded (they're predictors, not targets).
# anchor_correct_h* ARE excluded (baked into ratio_correct targets).

EXCLUDE_BASE = {
    "date", "bread_quantity", "is_covid_lockdown_period", "day_of_week",
    "_split", "_fold_1", "_fold_2", "_fold_3",
}

def make_feat_pool(df, also_exclude=None):
    also_exclude = set(also_exclude or [])
    feats = []
    for c in df.columns:
        if c in EXCLUDE_BASE or c in also_exclude:
            continue
        if c.startswith("bread_quantity_h") or c.startswith("ratio_h") \
           or c.startswith("ratio_correct_") or c.startswith("anchor_correct"):
            continue
        if df[c].dtype == object:
            continue
        feats.append(c)
    return feats

def get_feats(df, h, target_type):
    """Feature pool for a given horizon, excluding its own target columns."""
    extra_excl = set()
    if target_type == "ratio":
        # ratio_h{h} is target; keep anchors and ratio_correct for context
        extra_excl.add(f"ratio_h{h}")
    elif target_type == "ratio_correct":
        extra_excl.add(f"ratio_correct_h{h}")
    return make_feat_pool(df, also_exclude=extra_excl)

ALL_FEATS = make_feat_pool(df)
print(f"Full feature pool (no horizon exclusion): {len(ALL_FEATS)} features")


# ── Helper: get train/val arrays for a fold ────────────────────────────────
def get_fold_arrays(df, fold_col, target_col, feats, covid_filter=True):
    """Return X_tr, y_tr (full non-covid train) and X_vl, y_vl (all notna val)."""
    tr_mask = df[fold_col] == "train"
    if covid_filter:
        tr_mask = tr_mask & (df["is_covid_lockdown_period"] == 0)
    tr_mask = tr_mask & df[target_col].notna()
    vl_mask = (df[fold_col] == "val") & df[target_col].notna()
    X_tr = df.loc[tr_mask, feats].fillna(0).values
    y_tr = df.loc[tr_mask, target_col].values
    X_vl = df.loc[vl_mask, feats].fillna(0).values
    y_vl = df.loc[vl_mask, target_col].values
    return X_tr, y_tr, X_vl, y_vl, vl_mask


def compute_mae_units(model, X_vl, y_vl_units, ratio_col=None, anchor_vl=None):
    """Compute MAE in original units. If ratio model, convert pred*anchor."""
    raw_pred = model.predict(X_vl)
    if ratio_col is not None:
        preds = np.clip(raw_pred, *RATIO_CLIP) * anchor_vl
    else:
        preds = raw_pred
    return float(np.mean(np.abs(preds - y_vl_units))), preds


def fold_floor_mae(df, fold_col, h, vl_mask):
    """Naive floor MAE for a given fold using same-DOW seasonal median."""
    w = 4 if h in [1, 7, 14] else 26
    naive_col = f"naive_h{h}_seasonal_median_w{w}"
    tgt_col = f"bread_quantity_h{h}"
    y_vl = df.loc[vl_mask, tgt_col].values
    naive_preds = df.loc[vl_mask, naive_col].values
    return float(np.mean(np.abs(naive_preds - y_vl)))


RESULTS = {}

# ==========================================================================
# SECTION 1: h=1 — ratio_correct, Optuna 75 trials
# ==========================================================================
print("\n" + "="*70)
print("h=1: ratio_correct Optuna (75 trials) — first Optuna for h=1")
print("="*70)

H1_TARGET_RATIO = "ratio_correct_h1"
H1_ANCHOR       = "anchor_correct_h1"
H1_TARGET_UNITS = "bread_quantity_h1"
H1_FEATS = get_feats(df, 1, "ratio_correct")
print(f"  Feature pool: {len(H1_FEATS)}")

X1_tr, y1_tr, X1_vl, y1_vl_units, vl1 = get_fold_arrays(df, "_fold_3", H1_TARGET_RATIO, H1_FEATS)
anc1_vl = df.loc[vl1, H1_ANCHOR].values
y1_vl_raw = df.loc[vl1, H1_TARGET_UNITS].values

def objective_h1(trial):
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", 8, 64),
        "learning_rate":     trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 6),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 30.0, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
        "objective":         "regression_l1",
        "verbose":           -1,
    }
    n_inner = max(20, int(INNER_PCT * len(X1_tr)))
    model = lgb.LGBMRegressor(n_estimators=1000, random_state=42, **params)
    model.fit(
        X1_tr[:-n_inner], y1_tr[:-n_inner],
        eval_set=[(X1_tr[-n_inner:], y1_tr[-n_inner:])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    trial.set_user_attr("n_estimators", model.best_iteration_)
    pred_ratio = np.clip(model.predict(X1_vl), *RATIO_CLIP)
    preds_units = pred_ratio * anc1_vl
    return float(np.mean(np.abs(preds_units - y1_vl_raw)))

study_h1 = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
study_h1.optimize(objective_h1, n_trials=N_TRIALS, show_progress_bar=False)
best1 = study_h1.best_trial
h1_params = {**best1.params, "objective": "regression_l1", "verbose": -1}
h1_n_est = best1.user_attrs.get("n_estimators", 200)
print(f"  Best val MAE (units): {study_h1.best_value:.3f}  n_est={h1_n_est}")

model_h1 = lgb.LGBMRegressor(n_estimators=h1_n_est, random_state=42, **h1_params)
model_h1.fit(X1_tr, y1_tr)

mae1, _ = compute_mae_units(model_h1, X1_vl, y1_vl_raw, H1_TARGET_RATIO, anc1_vl)
floor1 = fold_floor_mae(df, "_fold_3", 1, vl1)
print(f"  Fold-3 MAE: {mae1:.3f}  floor={floor1:.3f}  baseline={BASELINE_FLOOR[1]:.3f}  prior={PRIOR_FOLD3_MAE[1]:.3f}  n={vl1.sum()}")
beats1 = mae1 < floor1

h1_mf = {}
for fid in [1, 2, 3]:
    fc = f"_fold_{fid}"
    Xf_tr, yf_tr, Xf_vl, yf_vl_u, vl_f = get_fold_arrays(df, fc, H1_TARGET_RATIO, H1_FEATS)
    anc_f = df.loc[vl_f, H1_ANCHOR].values
    y_f_raw = df.loc[vl_f, H1_TARGET_UNITS].values
    mf = lgb.LGBMRegressor(n_estimators=h1_n_est, random_state=42, **h1_params)
    mf.fit(Xf_tr, yf_tr)
    mae_f, _ = compute_mae_units(mf, Xf_vl, y_f_raw, H1_TARGET_RATIO, anc_f)
    fl_f = fold_floor_mae(df, fc, 1, vl_f)
    h1_mf[fid] = {"mae": mae_f, "floor": fl_f, "n": int(vl_f.sum()), "beats": mae_f < fl_f}
    print(f"  fold-{fid}: mae={mae_f:.3f} floor={fl_f:.3f} n={vl_f.sum()} beats={mae_f < fl_f}")

pkl1 = ART_OUT / "lgbm_h1_optuna_fe4.pkl"
pickle.dump(model_h1, open(pkl1, "wb"))
RESULTS[1] = {
    "fold3_mae": mae1, "floor_fold3": floor1, "baseline_floor": BASELINE_FLOOR[1],
    "beats_floor_fold3": beats1, "prior_fold3_mae": PRIOR_FOLD3_MAE[1],
    "config": {**h1_params, "n_estimators": h1_n_est},
    "per_fold": h1_mf, "model_path": str(pkl1.name),
    "n_val_rows": int(vl1.sum()),
}


# ==========================================================================
# SECTION 2: h=7 — ratio target, keep existing config, regression_l1
# ==========================================================================
print("\n" + "="*70)
print("h=7: ratio target, existing config + regression_l1 + new features")
print("="*70)

H7_TARGET_RATIO = "ratio_h7"
H7_ANCHOR_COL   = "naive_h7_seasonal_median_w4"
H7_TARGET_UNITS = "bread_quantity_h7"
H7_FEATS = get_feats(df, 7, "ratio")
print(f"  Feature pool: {len(H7_FEATS)}")

H7_PARAMS = {
    "num_leaves": 57, "learning_rate": 0.006877, "max_depth": 4,
    "min_child_samples": 17, "reg_alpha": 0.001348, "reg_lambda": 0.432,
    "subsample": 0.770, "colsample_bytree": 0.777, "min_split_gain": 0.0754,
    "objective": "regression_l1", "verbose": -1,
}
H7_N_EST = 522

X7_tr, y7_tr, X7_vl, y7_vl_u, vl7 = get_fold_arrays(df, "_fold_3", H7_TARGET_RATIO, H7_FEATS)
anc7_vl = df.loc[vl7, H7_ANCHOR_COL].values
y7_raw_vl = df.loc[vl7, H7_TARGET_UNITS].values

model_h7 = lgb.LGBMRegressor(n_estimators=H7_N_EST, random_state=42, **H7_PARAMS)
model_h7.fit(X7_tr, y7_tr)

mae7, _ = compute_mae_units(model_h7, X7_vl, y7_raw_vl, H7_TARGET_RATIO, anc7_vl)
floor7 = fold_floor_mae(df, "_fold_3", 7, vl7)
print(f"  Fold-3 MAE: {mae7:.3f}  floor={floor7:.3f}  baseline={BASELINE_FLOOR[7]:.3f}  prior={PRIOR_FOLD3_MAE[7]:.3f}  n={vl7.sum()}")
improved7 = mae7 < PRIOR_FOLD3_MAE[7] - 0.5

h7_mf = {}
for fid in [1, 2, 3]:
    fc = f"_fold_{fid}"
    Xf_tr, yf_tr, Xf_vl, _, vl_f = get_fold_arrays(df, fc, H7_TARGET_RATIO, H7_FEATS)
    anc_f = df.loc[vl_f, H7_ANCHOR_COL].values
    y_f_raw = df.loc[vl_f, H7_TARGET_UNITS].values
    mf = lgb.LGBMRegressor(n_estimators=H7_N_EST, random_state=42, **H7_PARAMS)
    mf.fit(Xf_tr, yf_tr)
    mae_f, _ = compute_mae_units(mf, Xf_vl, y_f_raw, H7_TARGET_RATIO, anc_f)
    fl_f = fold_floor_mae(df, fc, 7, vl_f)
    h7_mf[fid] = {"mae": mae_f, "floor": fl_f, "n": int(vl_f.sum()), "beats": mae_f < fl_f}
    print(f"  fold-{fid}: mae={mae_f:.3f} floor={fl_f:.3f} n={vl_f.sum()} beats={mae_f < fl_f}")

pkl7 = ART_OUT / "lgbm_h7_fe4.pkl"
pickle.dump(model_h7, open(pkl7, "wb"))
RESULTS[7] = {
    "fold3_mae": mae7, "floor_fold3": floor7, "baseline_floor": BASELINE_FLOOR[7],
    "beats_floor_fold3": mae7 < floor7, "prior_fold3_mae": PRIOR_FOLD3_MAE[7],
    "improved_vs_prior": improved7,
    "config": {**H7_PARAMS, "n_estimators": H7_N_EST},
    "per_fold": h7_mf, "model_path": str(pkl7.name),
}


# ==========================================================================
# SECTION 3: h=14 — raw-units, keep existing config, regression_l1
# ==========================================================================
print("\n" + "="*70)
print("h=14: raw-units, existing config + regression_l1 + new features")
print("="*70)

H14_TARGET = "bread_quantity_h14"
H14_FEATS = get_feats(df, 14, "raw")
print(f"  Feature pool: {len(H14_FEATS)}")

H14_PARAMS = {
    "num_leaves": 16, "learning_rate": 0.03, "max_depth": 5,
    "reg_alpha": 1.0, "reg_lambda": 10.0,
    "objective": "regression_l1", "verbose": -1,
}
H14_N_EST = 292

X14_tr, y14_tr, X14_vl, y14_vl, vl14 = get_fold_arrays(df, "_fold_3", H14_TARGET, H14_FEATS)

model_h14 = lgb.LGBMRegressor(n_estimators=H14_N_EST, random_state=42, **H14_PARAMS)
model_h14.fit(X14_tr, y14_tr)

mae14, _ = compute_mae_units(model_h14, X14_vl, y14_vl)
floor14 = fold_floor_mae(df, "_fold_3", 14, vl14)
print(f"  Fold-3 MAE: {mae14:.3f}  floor={floor14:.3f}  baseline={BASELINE_FLOOR[14]:.3f}  prior={PRIOR_FOLD3_MAE[14]:.3f}  n={vl14.sum()}")
improved14 = mae14 < PRIOR_FOLD3_MAE[14] - 0.5

h14_mf = {}
for fid in [1, 2, 3]:
    fc = f"_fold_{fid}"
    Xf_tr, yf_tr, Xf_vl, yf_vl, vl_f = get_fold_arrays(df, fc, H14_TARGET, H14_FEATS)
    mf = lgb.LGBMRegressor(n_estimators=H14_N_EST, random_state=42, **H14_PARAMS)
    mf.fit(Xf_tr, yf_tr)
    mae_f, _ = compute_mae_units(mf, Xf_vl, yf_vl)
    fl_f = fold_floor_mae(df, fc, 14, vl_f)
    h14_mf[fid] = {"mae": mae_f, "floor": fl_f, "n": int(vl_f.sum()), "beats": mae_f < fl_f}
    print(f"  fold-{fid}: mae={mae_f:.3f} floor={fl_f:.3f} n={vl_f.sum()} beats={mae_f < fl_f}")

pkl14 = ART_OUT / "lgbm_h14_raw_fe4.pkl"
pickle.dump(model_h14, open(pkl14, "wb"))
RESULTS[14] = {
    "fold3_mae": mae14, "floor_fold3": floor14, "baseline_floor": BASELINE_FLOOR[14],
    "beats_floor_fold3": mae14 < floor14, "prior_fold3_mae": PRIOR_FOLD3_MAE[14],
    "improved_vs_prior": improved14,
    "config": {**H14_PARAMS, "n_estimators": H14_N_EST},
    "per_fold": h14_mf, "model_path": str(pkl14.name),
}


# ==========================================================================
# SECTION 4: h=30 — ratio_correct, Optuna 75 trials
# ==========================================================================
print("\n" + "="*70)
print("h=30: ratio_correct Optuna (75 trials) — first Optuna for h=30")
print("="*70)

H30_TARGET_RATIO = "ratio_correct_h30"
H30_ANCHOR       = "anchor_correct_h30"
H30_TARGET_UNITS = "bread_quantity_h30"
H30_FEATS = get_feats(df, 30, "ratio_correct")
print(f"  Feature pool: {len(H30_FEATS)}")

X30_tr, y30_tr, X30_vl, y30_vl_u, vl30 = get_fold_arrays(df, "_fold_3", H30_TARGET_RATIO, H30_FEATS)
anc30_vl = df.loc[vl30, H30_ANCHOR].values
y30_raw_vl = df.loc[vl30, H30_TARGET_UNITS].values

def objective_h30(trial):
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", 8, 64),
        "learning_rate":     trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 6),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 30.0, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
        "objective":         "regression_l1",
        "verbose":           -1,
    }
    n_inner = max(20, int(INNER_PCT * len(X30_tr)))
    model = lgb.LGBMRegressor(n_estimators=1000, random_state=42, **params)
    model.fit(
        X30_tr[:-n_inner], y30_tr[:-n_inner],
        eval_set=[(X30_tr[-n_inner:], y30_tr[-n_inner:])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    trial.set_user_attr("n_estimators", model.best_iteration_)
    pred_ratio = np.clip(model.predict(X30_vl), *RATIO_CLIP)
    preds_units = pred_ratio * anc30_vl
    return float(np.mean(np.abs(preds_units - y30_raw_vl)))

study_h30 = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
study_h30.optimize(objective_h30, n_trials=N_TRIALS, show_progress_bar=False)
best30 = study_h30.best_trial
h30_params = {**best30.params, "objective": "regression_l1", "verbose": -1}
h30_n_est = best30.user_attrs.get("n_estimators", 200)
print(f"  Best val MAE (units): {study_h30.best_value:.3f}  n_est={h30_n_est}")

model_h30 = lgb.LGBMRegressor(n_estimators=h30_n_est, random_state=42, **h30_params)
model_h30.fit(X30_tr, y30_tr)

mae30, _ = compute_mae_units(model_h30, X30_vl, y30_raw_vl, H30_TARGET_RATIO, anc30_vl)
floor30 = fold_floor_mae(df, "_fold_3", 30, vl30)
print(f"  Fold-3 MAE: {mae30:.3f}  floor={floor30:.3f}  baseline={BASELINE_FLOOR[30]:.3f}  prior={PRIOR_FOLD3_MAE[30]:.3f}  n={vl30.sum()}")
beats30 = mae30 < floor30

h30_mf = {}
for fid in [1, 2, 3]:
    fc = f"_fold_{fid}"
    Xf_tr, yf_tr, Xf_vl, _, vl_f = get_fold_arrays(df, fc, H30_TARGET_RATIO, H30_FEATS)
    anc_f = df.loc[vl_f, H30_ANCHOR].values
    y_f_raw = df.loc[vl_f, H30_TARGET_UNITS].values
    mf = lgb.LGBMRegressor(n_estimators=h30_n_est, random_state=42, **h30_params)
    mf.fit(Xf_tr, yf_tr)
    mae_f, _ = compute_mae_units(mf, Xf_vl, y_f_raw, H30_TARGET_RATIO, anc_f)
    fl_f = fold_floor_mae(df, fc, 30, vl_f)
    h30_mf[fid] = {"mae": mae_f, "floor": fl_f, "n": int(vl_f.sum()), "beats": mae_f < fl_f}
    print(f"  fold-{fid}: mae={mae_f:.3f} floor={fl_f:.3f} n={vl_f.sum()} beats={mae_f < fl_f}")

pkl30 = ART_OUT / "lgbm_h30_optuna_fe4.pkl"
pickle.dump(model_h30, open(pkl30, "wb"))
RESULTS[30] = {
    "fold3_mae": mae30, "floor_fold3": floor30, "baseline_floor": BASELINE_FLOOR[30],
    "beats_floor_fold3": beats30, "prior_fold3_mae": PRIOR_FOLD3_MAE[30],
    "config": {**h30_params, "n_estimators": h30_n_est},
    "per_fold": h30_mf, "model_path": str(pkl30.name),
}


# ==========================================================================
# SECTION 5: h=60 — raw-units Optuna + evaluate on ALL val rows
# ==========================================================================
print("\n" + "="*70)
print("h=60: raw-units Optuna, ALL val rows vs baseline 40.042")
print("="*70)

H60_TARGET = "bread_quantity_h60"
H60_FEATS = get_feats(df, 60, "raw")
print(f"  Feature pool: {len(H60_FEATS)}")

X60_tr, y60_tr, X60_vl, y60_vl, vl60 = get_fold_arrays(df, "_fold_3", H60_TARGET, H60_FEATS)

def objective_h60(trial):
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", 8, 64),
        "learning_rate":     trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 6),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 30.0, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
        "objective":         "regression_l1",
        "verbose":           -1,
    }
    n_inner = max(20, int(INNER_PCT * len(X60_tr)))
    model = lgb.LGBMRegressor(n_estimators=1000, random_state=42, **params)
    model.fit(
        X60_tr[:-n_inner], y60_tr[:-n_inner],
        eval_set=[(X60_tr[-n_inner:], y60_tr[-n_inner:])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    trial.set_user_attr("n_estimators", model.best_iteration_)
    return float(np.mean(np.abs(model.predict(X60_vl) - y60_vl)))

study_h60 = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
study_h60.optimize(objective_h60, n_trials=N_TRIALS, show_progress_bar=False)
best60 = study_h60.best_trial
h60_params = {**best60.params, "objective": "regression_l1", "verbose": -1}
h60_n_est = best60.user_attrs.get("n_estimators", 200)

model_h60 = lgb.LGBMRegressor(n_estimators=h60_n_est, random_state=42, **h60_params)
model_h60.fit(X60_tr, y60_tr)

mae60_all, _ = compute_mae_units(model_h60, X60_vl, y60_vl)
floor60_all = fold_floor_mae(df, "_fold_3", 60, vl60)
print(f"  Fold-3 ALL rows: mae={mae60_all:.3f}  floor={floor60_all:.3f}  baseline={BASELINE_FLOOR[60]:.3f}  n={vl60.sum()}")
beats60 = mae60_all < BASELINE_FLOOR[60]

h60_mf = {}
for fid in [1, 2, 3]:
    fc = f"_fold_{fid}"
    Xf_tr, yf_tr, Xf_vl, yf_vl, vl_f = get_fold_arrays(df, fc, H60_TARGET, H60_FEATS)
    mf = lgb.LGBMRegressor(n_estimators=h60_n_est, random_state=42, **h60_params)
    mf.fit(Xf_tr, yf_tr)
    mae_f, _ = compute_mae_units(mf, Xf_vl, yf_vl)
    fl_f = fold_floor_mae(df, fc, 60, vl_f)
    h60_mf[fid] = {"mae_all": mae_f, "floor_all": fl_f, "n_all": int(vl_f.sum()),
                   "beats_floor_all": mae_f < fl_f}
    print(f"  fold-{fid}: mae={mae_f:.3f} floor={fl_f:.3f} n={vl_f.sum()} beats={mae_f < fl_f}")

pkl60 = ART_OUT / "lgbm_h60_optuna_fe4.pkl"
pickle.dump(model_h60, open(pkl60, "wb"))
RESULTS[60] = {
    "fold3_mae_all_rows": mae60_all, "floor_fold3_all": floor60_all,
    "baseline_floor_all_rows": BASELINE_FLOOR[60],
    "beats_baseline_all_rows": beats60, "prior_fold3_mae": PRIOR_FOLD3_MAE[60],
    "config": {**h60_params, "n_estimators": h60_n_est},
    "per_fold": h60_mf, "model_path": str(pkl60.name),
}


# ==========================================================================
# SECTION 6: h=90 — raw-units, keep config, multi-fold
# ==========================================================================
print("\n" + "="*70)
print("h=90: raw-units, existing config + regression_l1 + new features, multi-fold")
print("="*70)

H90_TARGET = "bread_quantity_h90"
H90_FEATS = get_feats(df, 90, "raw")
print(f"  Feature pool: {len(H90_FEATS)}")

H90_PARAMS = {
    "num_leaves": 16, "learning_rate": 0.05, "max_depth": 4,
    "reg_alpha": 2.0, "reg_lambda": 20.0,
    "objective": "regression_l1", "verbose": -1,
}
H90_N_EST = 68

X90_tr, y90_tr, X90_vl, y90_vl, vl90 = get_fold_arrays(df, "_fold_3", H90_TARGET, H90_FEATS)
model_h90 = lgb.LGBMRegressor(n_estimators=H90_N_EST, random_state=42, **H90_PARAMS)
model_h90.fit(X90_tr, y90_tr)

mae90, _ = compute_mae_units(model_h90, X90_vl, y90_vl)
floor90 = fold_floor_mae(df, "_fold_3", 90, vl90)
print(f"  Fold-3 MAE: {mae90:.3f}  floor={floor90:.3f}  n={vl90.sum()}  prior={PRIOR_FOLD3_MAE[90]:.3f}")

h90_mf = {}
for fid in [1, 2, 3]:
    fc = f"_fold_{fid}"
    Xf_tr, yf_tr, Xf_vl, yf_vl, vl_f = get_fold_arrays(df, fc, H90_TARGET, H90_FEATS)
    mf = lgb.LGBMRegressor(n_estimators=H90_N_EST, random_state=42, **H90_PARAMS)
    mf.fit(Xf_tr, yf_tr)
    mae_f, _ = compute_mae_units(mf, Xf_vl, yf_vl)
    fl_f = fold_floor_mae(df, fc, 90, vl_f)
    h90_mf[fid] = {"mae": mae_f, "floor": fl_f, "n": int(vl_f.sum()), "beats": mae_f < fl_f}
    print(f"  fold-{fid}: mae={mae_f:.3f} floor={fl_f:.3f} n={vl_f.sum()} beats={mae_f < fl_f}")

n_total90 = sum(h90_mf[f]["n"] for f in [1,2,3])
pooled90 = sum(h90_mf[f]["mae"] * h90_mf[f]["n"] for f in [1,2,3]) / n_total90
pooled_fl90 = sum(h90_mf[f]["floor"] * h90_mf[f]["n"] for f in [1,2,3]) / n_total90
print(f"  Pooled MAE: {pooled90:.3f}  pooled floor: {pooled_fl90:.3f}  beats: {pooled90 < pooled_fl90}")

pkl90 = ART_OUT / "lgbm_h90_fe4.pkl"
pickle.dump(model_h90, open(pkl90, "wb"))
RESULTS[90] = {
    "fold3_mae": mae90, "floor_fold3": floor90, "baseline_floor": BASELINE_FLOOR[90],
    "beats_floor_fold3": mae90 < floor90, "prior_fold3_mae": PRIOR_FOLD3_MAE[90],
    "pooled_mae": pooled90, "pooled_floor": pooled_fl90, "beats_pooled": pooled90 < pooled_fl90,
    "config": {**H90_PARAMS, "n_estimators": H90_N_EST},
    "per_fold": h90_mf, "model_path": str(pkl90.name),
}


# ==========================================================================
# SUMMARY
# ==========================================================================
print("\n" + "="*70)
print("SUMMARY: iter020 vs iter018 champions")
print("="*70)
for h in [1, 7, 14, 30, 60, 90]:
    r = RESULTS[h]
    if h == 60:
        mae = r["fold3_mae_all_rows"]
        fl = r["floor_fold3_all"]
    elif h == 90:
        mae = r["pooled_mae"]
        fl = r["pooled_floor"]
    else:
        mae = r["fold3_mae"]
        fl = r["floor_fold3"]
    prior = r["prior_fold3_mae"]
    floor_b = r.get("baseline_floor", BASELINE_FLOOR[h])
    print(f"  h={h:2d}: new={mae:.3f}  prior={prior:.3f}  delta={prior-mae:+.3f}  fold3_floor={fl:.3f}  beats_fold3_floor={mae<fl}")


# ==========================================================================
# CHAMPION SELECTION
# ==========================================================================
CHAMPIONS = {}

# h=1: FE4 Optuna vs prior (29.378)
if RESULTS[1]["beats_floor_fold3"]:
    if RESULTS[1]["fold3_mae"] < PRIOR_FOLD3_MAE[1] - 0.5:
        CHAMPIONS[1] = {"model_path": str(pkl1), "fold3_mae": RESULTS[1]["fold3_mae"],
                        "source": "iter020_optuna_fe4", "target": H1_TARGET_RATIO}
    else:
        CHAMPIONS[1] = {"model_path": "artifacts/training/anchor_fixed/lgbm_h1.pkl",
                        "fold3_mae": PRIOR_FOLD3_MAE[1], "source": "iter012",
                        "note": f"FE4 ({RESULTS[1]['fold3_mae']:.3f}) within 0.5 of prior ({PRIOR_FOLD3_MAE[1]:.3f})"}
else:
    CHAMPIONS[1] = {"model_path": "artifacts/training/anchor_fixed/lgbm_h1.pkl",
                    "fold3_mae": PRIOR_FOLD3_MAE[1], "source": "iter012",
                    "note": "FE4 Optuna did not beat fold-3 floor"}

# h=7: use FE4 if improved by >= 0.5 MAE
if RESULTS[7]["improved_vs_prior"] and RESULTS[7]["beats_floor_fold3"]:
    CHAMPIONS[7] = {"model_path": str(pkl7), "fold3_mae": RESULTS[7]["fold3_mae"],
                    "source": "iter020_fe4", "target": H7_TARGET_RATIO}
else:
    CHAMPIONS[7] = {"model_path": "artifacts/training/optuna/optuna_lgbm_h7_best.pkl",
                    "fold3_mae": PRIOR_FOLD3_MAE[7], "source": "iter011",
                    "note": f"FE4 delta={PRIOR_FOLD3_MAE[7]-RESULTS[7]['fold3_mae']:+.3f}"}

# h=14: use FE4 if improved by >= 0.5 MAE
if RESULTS[14]["improved_vs_prior"] and RESULTS[14]["beats_floor_fold3"]:
    CHAMPIONS[14] = {"model_path": str(pkl14), "fold3_mae": RESULTS[14]["fold3_mae"],
                     "source": "iter020_fe4", "target": H14_TARGET}
else:
    CHAMPIONS[14] = {"model_path": "artifacts/training/iter018/lgbm_h14_raw.pkl",
                     "fold3_mae": PRIOR_FOLD3_MAE[14], "source": "iter018",
                     "note": f"FE4 delta={PRIOR_FOLD3_MAE[14]-RESULTS[14]['fold3_mae']:+.3f}"}

# h=30: Optuna FE4
if RESULTS[30]["beats_floor_fold3"]:
    CHAMPIONS[30] = {"model_path": str(pkl30), "fold3_mae": RESULTS[30]["fold3_mae"],
                     "source": "iter020_optuna_fe4", "target": H30_TARGET_RATIO}
else:
    CHAMPIONS[30] = {"model_path": "artifacts/training/anchor_fixed/lgbm_h30.pkl",
                     "fold3_mae": PRIOR_FOLD3_MAE[30], "source": "iter012",
                     "note": "FE4 Optuna did not beat fold-3 floor"}

# h=60: use ML only if beats baseline floor on all rows
if RESULTS[60]["beats_baseline_all_rows"]:
    CHAMPIONS[60] = {"model_path": str(pkl60), "fold3_mae": RESULTS[60]["fold3_mae_all_rows"],
                     "source": "iter020_optuna_fe4", "target": H60_TARGET}
else:
    CHAMPIONS[60] = {"model_path": None, "fold3_mae": BASELINE_FLOOR[60], "source": "naive_baseline",
                     "note": "ML does not beat 40.042 floor on all rows — use naive_h60_seasonal_median_w26"}

# h=90: use FE4 if pooled MAE beats pooled floor
if RESULTS[90]["beats_pooled"]:
    CHAMPIONS[90] = {"model_path": str(pkl90), "fold3_mae": RESULTS[90]["pooled_mae"],
                     "source": "iter020_fe4", "target": H90_TARGET}
else:
    CHAMPIONS[90] = {"model_path": "artifacts/training/iter018/lgbm_h90_multifold_confirmed.pkl",
                     "fold3_mae": 40.796, "source": "iter018",
                     "note": f"FE4 pooled={RESULTS[90]['pooled_mae']:.3f} vs floor {RESULTS[90]['pooled_floor']:.3f}"}

print("\nFinal champions:")
for h, c in CHAMPIONS.items():
    print(f"  h={h}: fold3_mae={c['fold3_mae']:.3f}  source={c['source']}")


# ==========================================================================
# WRITE HANDOFF JSON
# ==========================================================================
handoff = {
    "iteration": 20,
    "node": "training",
    "timestamp": "2026-09-09",
    "schema_version": "2026-09-04",
    "type": "training_with_fe4",
    "summary": (
        "Training pass with FE4 (90 cols). All models use objective='regression_l1'. "
        "First Optuna for h=1 and h=30. Fixed-config re-train for h=7, h=14, h=90. "
        "h=60 Optuna with ALL-rows evaluation vs 40.042 floor."
    ),
    "critical_bug_fixed": "objective must be regression_l1 (MAE loss), NOT default MSE. "
                          "MSE is pulled toward post-COVID outlier ratios causing fold-3 failure.",
    "results": {str(h): RESULTS[h] for h in RESULTS},
    "champions": {str(h): CHAMPIONS[h] for h in CHAMPIONS},
    "handoff": {
        "written_by": "training",
        "iteration": 20,
        "context_clearable": True,
        "read_this_first": (
            "All models must use objective='regression_l1'. "
            "h=60: use ML only if beats_baseline_all_rows is True (MAE < 40.042). "
            "h=90: use pooled multi-fold for champion. "
            "FE CSV: data/iteration_019_feature_engineering.csv (90 cols)."
        ),
        "parameters": {
            "fe_csv": "data/iteration_019_feature_engineering.csv",
            "feature_count": len(ALL_FEATS),
        },
        "durable_rules": [
            {
                "rule": "ALL LightGBM models must use objective='regression_l1'.",
                "why": "Without this, MSE loss is pulled toward outlier post-COVID ratio rows, causing fold-3 predictions to be systematically biased low.",
                "applies_to": ["end_model"],
            },
            {
                "rule": "h=60: only declare ML champion if MAE < 40.042 on ALL fold-3 val rows.",
                "why": "Baseline floor 40.042 was computed on all rows including 2 zero-demand days.",
                "applies_to": ["end_model"],
            },
            {
                "rule": "h=90: use pooled multi-fold MAE for champion selection.",
                "why": "Fold-3 alone has only 4 non-NaN val rows for h=90.",
                "applies_to": ["end_model"],
            },
        ],
        "do_not_repeat": [
            {
                "tried": "LightGBM with default MSE objective (regression) for ratio models",
                "outcome": "catastrophic fold-3 failure — MAE 2-3x worse than prior champions",
                "evidence": "h=7 fold-3 MAE=77 with MSE vs 31 with MAE loss",
                "why_not_again": "Post-COVID ratio outliers (low ratio ~0.05 from lockdown naive) skew MSE toward predicting low ratios for all val rows.",
            }
        ],
        "content_carried_forward": {
            "champions_table": {str(h): CHAMPIONS[h] for h in CHAMPIONS},
            "h60_beats_floor_all_rows": RESULTS[60]["beats_baseline_all_rows"],
            "objective_must_be": "regression_l1",
        },
    },
}

with open(HIST_OUT, "w", encoding="utf-8") as f:
    json.dump(handoff, f, indent=2, default=str)
print(f"\nHandoff written: {HIST_OUT}")

with open(STATE_F, encoding="utf-8") as f:
    state = json.load(f)
state["iteration"] = 20
state["iter020_results"] = {
    "champions": {str(h): CHAMPIONS[h] for h in CHAMPIONS},
    "h60_beats_floor_all_rows": RESULTS[60]["beats_baseline_all_rows"],
    "objective_used": "regression_l1",
}
state["next_iteration_contract"] = {
    "node": "end_model",
    "iteration": 21,
    "read_first": [
        "history/iteration_020_training_with_fe4.json",
        "history/iteration_018_targeted_improvement.json",
    ],
    "fe_csv": "data/iteration_019_feature_engineering.csv",
    "must_use_champions_from": "history/iteration_020_training_with_fe4.json",
    "all_models_must_use": "objective='regression_l1'",
}
with open(STATE_F, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
print("State updated -> iter021 end_model")
print("\n=== Iteration 020 complete ===")
