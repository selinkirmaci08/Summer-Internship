# %% [iteration_024_training_with_fe5] Training with FE5 — run_20260908T075709Z
#
# Retrains all 6 horizon champions on the FE5 CSV (92 cols, 64 model features).
# Uses pinned hyperparams from iter020/iter022 — no new Optuna.
# Only updates champion if new MAE improves.
#
# Evaluation:
#   h=1,7,14,30,90: fold-3 val origins (all notna rows), units MAE
#   h=60: correct 95-row scheme (origins D-60, targets in val period)
# All models: objective='regression_l1', fillna(0)

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")

RUN_DIR  = Path(__file__).parent.parent
FE_CSV   = RUN_DIR / "data" / "iteration_023_feature_engineering.csv"
ART_OUT  = RUN_DIR / "artifacts" / "training" / "iter024"
HIST_OUT = RUN_DIR / "history" / "iteration_024_training_with_fe5.json"
STATE_F  = RUN_DIR / "state" / "current.json"
ART_OUT.mkdir(parents=True, exist_ok=True)

RATIO_CLIP = (0.05, 5.0)

print("=== Iteration 024: Training with FE5 ===")
df = pd.read_csv(str(FE_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# ── Feature pool ─────────────────────────────────────────────────────────────
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

# ── Shared masks ──────────────────────────────────────────────────────────────
cv_train  = (df["_fold_3"] == "train") & (df["is_covid_lockdown_period"] == 0)
val_mask  = df["_fold_3"] == "val"
full_tr   = df["_fold_3"].isin(["train", "val"]) & (df["is_covid_lockdown_period"] == 0)

# ── Prior champion results (from iter020 / iter022) ───────────────────────────
PRIOR = {
    1:  {"mae": 29.835, "floor": 54.415},
    7:  {"mae": 33.020, "floor": 34.125},
    14: {"mae": 37.797, "floor": 38.116},
    30: {"mae": 37.955, "floor": 40.262},
    60: {"mae": 36.229, "floor": 40.042},
    90: {"mae": 56.767, "floor": 75.937, "pooled": True},
}

# ── Hyperparams (pinned from iter020/iter022) ─────────────────────────────────
CONFIGS = {
    1: {"num_leaves": 27, "learning_rate": 0.056110292449094386, "max_depth": 3,
        "min_child_samples": 8, "reg_alpha": 0.00025599153998635853,
        "reg_lambda": 0.053810981737851815, "subsample": 0.6273109082811845,
        "colsample_bytree": 0.7880434602568575, "min_split_gain": 0.01908151398597356,
        "objective": "regression_l1", "verbose": -1, "n_estimators": 97},
    7: {"num_leaves": 57, "learning_rate": 0.006877, "max_depth": 4,
        "min_child_samples": 17, "reg_alpha": 0.001348, "reg_lambda": 0.432,
        "subsample": 0.77, "colsample_bytree": 0.777, "min_split_gain": 0.0754,
        "objective": "regression_l1", "verbose": -1, "n_estimators": 522},
    14: {"num_leaves": 16, "learning_rate": 0.03, "max_depth": 5,
         "reg_alpha": 1.0, "reg_lambda": 10.0,
         "objective": "regression_l1", "verbose": -1, "n_estimators": 292},
    30: {"num_leaves": 12, "learning_rate": 0.11305557316344139, "max_depth": 5,
         "min_child_samples": 22, "reg_alpha": 0.14017728443125377,
         "reg_lambda": 0.00038308665716819333, "subsample": 0.7217175301064572,
         "colsample_bytree": 0.5954830133876508, "min_split_gain": 0.47472027966527935,
         "objective": "regression_l1", "verbose": -1, "n_estimators": 54},
    60: {"num_leaves": 14, "learning_rate": 0.07938407816005191, "max_depth": 7,
         "min_child_samples": 23, "reg_alpha": 0.061338395620921464,
         "reg_lambda": 0.009118605447473642, "subsample": 0.6840196259574847,
         "colsample_bytree": 0.5097078382107525, "min_split_gain": 0.049351909487439935,
         "objective": "regression_l1", "verbose": -1, "n_estimators": 14},
    90: {"num_leaves": 16, "learning_rate": 0.05, "max_depth": 4,
         "reg_alpha": 2.0, "reg_lambda": 20.0,
         "objective": "regression_l1", "verbose": -1, "n_estimators": 68},
}
TARGETS   = {1: "ratio_correct_h1", 7: "ratio_h7", 14: "bread_quantity_h14",
             30: "ratio_correct_h30", 60: "ratio_correct_h60", 90: "bread_quantity_h90"}
ANCHORS   = {1: "anchor_correct_h1", 7: "naive_h7_seasonal_median_w4",
             14: None, 30: "anchor_correct_h30", 60: "anchor_correct_h60", 90: None}

# ── h=60 correct eval setup ───────────────────────────────────────────────────
val_start = pd.Timestamp("2020-09-28")
val_end   = pd.Timestamp("2020-12-31")
df["_target_date_h60"] = df["date"] + pd.Timedelta(days=60)
eval95 = (df["_target_date_h60"] >= val_start) & (df["_target_date_h60"] <= val_end) & df["bread_quantity_h60"].notna()

RESULTS = {}

for h in [1, 7, 14, 30, 60, 90]:
    print(f"\n{'='*55}")
    tgt = TARGETS[h]; anc = ANCHORS[h]
    cfg = CONFIGS[h].copy(); n_est = cfg.pop("n_estimators")
    print(f"h={h}: target={tgt}  n_est={n_est}")

    if h == 60:
        # Correct 95-row eval
        tr_m = cv_train & df[tgt].notna()
        X_tr = df.loc[tr_m, ALL_FEATS].fillna(0).values
        y_tr = df.loc[tr_m, tgt].values
        X_vl = df.loc[eval95, ALL_FEATS].fillna(0).values
        a_vl = df.loc[eval95, anc].values
        y_vl = df.loc[eval95, "bread_quantity_h60"].values

        model = lgb.LGBMRegressor(n_estimators=n_est, random_state=42, **cfg)
        model.fit(X_tr, y_tr)
        preds = np.clip(model.predict(X_vl), *RATIO_CLIP) * a_vl
        mae = float(np.mean(np.abs(preds - y_vl)))
        floor = PRIOR[60]["floor"]
        print(f"  h60 (95-row eval): MAE={mae:.3f}  floor={floor:.3f}  prior={PRIOR[60]['mae']:.3f}  beats={mae<floor}")
        RESULTS[60] = {"mae": mae, "floor": floor, "beats_floor": mae < floor,
                       "prior_mae": PRIOR[60]["mae"], "improved": mae < PRIOR[60]["mae"], "n": 95}

    elif h == 90:
        # Multi-fold pooled
        fold_maes, fold_ns, fold_floors = [], [], []
        for fid in [1, 2, 3]:
            fc = f"_fold_{fid}"
            tr_f = (df[fc] == "train") & (df["is_covid_lockdown_period"] == 0) & df[tgt].notna()
            vl_f = (df[fc] == "val") & df[tgt].notna()
            Xf_tr = df.loc[tr_f, ALL_FEATS].fillna(0).values; yf_tr = df.loc[tr_f, tgt].values
            Xf_vl = df.loc[vl_f, ALL_FEATS].fillna(0).values; yf_vl = df.loc[vl_f, tgt].values
            naive_f = df.loc[vl_f, "naive_h90_seasonal_median_w26"].values
            m = lgb.LGBMRegressor(n_estimators=n_est, random_state=42, **cfg)
            m.fit(Xf_tr, yf_tr)
            mae_f = float(np.mean(np.abs(m.predict(Xf_vl) - yf_vl)))
            fl_f  = float(np.mean(np.abs(naive_f - yf_vl)))
            fold_maes.append(mae_f); fold_ns.append(int(vl_f.sum())); fold_floors.append(fl_f)
            print(f"  fold-{fid}: MAE={mae_f:.3f}  floor={fl_f:.3f}  n={vl_f.sum()}  beats={mae_f<fl_f}")
        tot = sum(fold_ns)
        pooled_mae   = sum(m*n for m,n in zip(fold_maes, fold_ns)) / tot
        pooled_floor = sum(f*n for f,n in zip(fold_floors, fold_ns)) / tot
        print(f"  Pooled: MAE={pooled_mae:.3f}  floor={pooled_floor:.3f}  prior={PRIOR[90]['mae']:.3f}  beats={pooled_mae<pooled_floor}")
        RESULTS[90] = {"pooled_mae": pooled_mae, "pooled_floor": pooled_floor,
                       "beats_floor": pooled_mae < pooled_floor,
                       "prior_mae": PRIOR[90]["mae"], "improved": pooled_mae < PRIOR[90]["mae"],
                       "per_fold": {str(i+1): {"mae": fold_maes[i], "floor": fold_floors[i], "n": fold_ns[i]}
                                    for i in range(3)}}

    else:
        tr_m = cv_train & df[tgt].notna()
        vl_m = val_mask & df[tgt].notna()
        X_tr = df.loc[tr_m, ALL_FEATS].fillna(0).values; y_tr = df.loc[tr_m, tgt].values
        X_vl = df.loc[vl_m, ALL_FEATS].fillna(0).values; y_vl_raw = df.loc[vl_m, "bread_quantity_h1" if h==1 else f"bread_quantity_h{h}"].values

        model = lgb.LGBMRegressor(n_estimators=n_est, random_state=42, **cfg)
        model.fit(X_tr, y_tr)

        if anc:
            a_vl = df.loc[vl_m, anc].values
            preds = np.clip(model.predict(X_vl), *RATIO_CLIP) * a_vl
        else:
            preds = model.predict(X_vl)
        mae = float(np.mean(np.abs(preds - y_vl_raw)))

        naive_col = f"naive_h{h}_seasonal_median_w4" if h <= 30 else f"naive_h{h}_seasonal_median_w26"
        if naive_col not in df.columns:
            naive_col = next(c for c in df.columns if f"naive_h{h}" in c)
        naive_vals = df.loc[vl_m, naive_col].values
        floor = float(np.mean(np.abs(naive_vals - y_vl_raw)))
        print(f"  h={h} fold-3 val: MAE={mae:.3f}  floor={floor:.3f}  prior={PRIOR[h]['mae']:.3f}  beats={mae<floor}  improved={mae<PRIOR[h]['mae']}")
        RESULTS[h] = {"mae": mae, "floor": floor, "beats_floor": mae < floor,
                      "prior_mae": PRIOR[h]["mae"], "improved": mae < PRIOR[h]["mae"],
                      "n": int(vl_m.sum())}

# ── Save improved models ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("SAVING IMPROVED MODELS")
saved = {}
for h in [1, 7, 14, 30, 60, 90]:
    r = RESULTS[h]
    improved = r.get("improved", False)
    tgt = TARGETS[h]; anc = ANCHORS[h]
    cfg = CONFIGS[h].copy(); n_est = cfg.pop("n_estimators")

    if improved:
        # Retrain on full data
        if h == 60:
            full_m = full_tr & df[tgt].notna()
        else:
            full_m = full_tr & df[tgt].notna()
        X_full = df.loc[full_m, ALL_FEATS].fillna(0).values
        y_full = df.loc[full_m, tgt].values
        final_model = lgb.LGBMRegressor(n_estimators=n_est, random_state=42, **cfg)
        final_model.fit(X_full, y_full)
        pkl = ART_OUT / f"lgbm_h{h}_fe5.pkl"
        pickle.dump(final_model, open(pkl, "wb"))
        saved[h] = str(pkl.name)
        print(f"  h={h}: IMPROVED ({r.get('prior_mae', r.get('mae')):.3f} -> {r.get('mae', r.get('pooled_mae')):.3f}) — saved {pkl.name}")
    else:
        prior_mae = r.get("prior_mae", "?")
        cur_mae = r.get("mae") or r.get("pooled_mae")
        print(f"  h={h}: no improvement ({prior_mae:.3f} -> {cur_mae:.3f}) — keeping iter020/iter022 model")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print(f"{'h':>3}  {'Prior MAE':>10}  {'FE5 MAE':>10}  {'Floor':>8}  {'Beats?':>7}  {'Improved?':>10}")
print("-"*60)
for h in [1, 7, 14, 30, 60, 90]:
    r = RESULTS[h]
    prior = r.get("prior_mae", "?")
    cur   = r.get("mae") or r.get("pooled_mae")
    floor = r.get("floor") or r.get("pooled_floor")
    beats = r.get("beats_floor", False)
    imp   = r.get("improved", False)
    print(f"  {h:>2}  {prior:>10.3f}  {cur:>10.3f}  {floor:>8.3f}  {'YES' if beats else 'NO':>7}  {'YES' if imp else 'no':>10}")

# ── Handoff JSON ──────────────────────────────────────────────────────────────
handoff = {
    "iteration": 24,
    "node": "training",
    "timestamp": "2026-09-09",
    "schema_version": "2026-09-04",
    "type": "training_with_fe5",
    "summary": (
        "Retrains all 6 horizons with FE5 features (92 cols, 64 features). "
        "Pinned hyperparams from iter020/iter022. Only saves model if improved vs prior. "
        "FE5 adds days_since_last_holiday and temperature_zscore_seasonal."
    ),
    "fe_csv": "data/iteration_023_feature_engineering.csv",
    "results": {str(h): RESULTS[h] for h in RESULTS},
    "saved_models": saved,
    "handoff": {
        "written_by": "training",
        "iteration": 24,
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
state["current"]["iteration"] = 24
state["current"]["run_status"] = "in_progress"
with open(STATE_F, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
print("State updated: iteration=24")
print("\n=== Iteration 024 complete ===")
