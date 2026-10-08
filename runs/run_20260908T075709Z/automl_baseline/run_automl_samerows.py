# automl_baseline/run_automl_samerows.py
#
# AutoML with same training rows as our LightGBM pipeline:
#   - fold-3 train + fold-3 val, COVID lockdown excluded
#   - NO feature engineering: only date + weather + competitor features
#   - FLAML AutoML per horizon (time_budget=120s)
#   - Evaluate on same primary (2021) and stress (2022-2023) test sets
#
# This isolates the effect of feature engineering: same data, same AutoML tool,
# but without domain-driven features.

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML
warnings.filterwarnings("ignore")

OUT_DIR  = Path(__file__).parent
FE4_CSV  = Path(__file__).parent.parent / "data" / "iteration_019_feature_engineering.csv"

# Test feature CSVs from first AutoML run (already parsed + joined)
PRIMARY_CSV = OUT_DIR / "primary_features.csv"
STRESS_CSV  = OUT_DIR / "stress_features.csv"

HORIZONS    = [1, 7, 14, 30, 60, 90]
TIME_BUDGET = 300  # seconds per horizon

# Reference results
LGBM_PRIMARY  = {1: 39.313, 7: 41.187, 14: 44.892, 30: 42.950, 60: 46.303, 90: 64.924}
LGBM_STRESS   = {1: 102.397, 7: 112.560, 14: 129.852, 30: 105.400, 60: 111.956, 90: 134.741}
NAIVE_PRIMARY = {1: 64.118, 7: 41.355, 14: 42.436, 30: 71.149, 60: 73.019, 90: 85.154}
NAIVE_STRESS  = {1: 129.078, 7: 106.259, 14: 111.768, 30: 147.136, 60: 142.836, 90: 125.415}
AUTOML1_PRIMARY = {1: 67.308, 7: 66.998, 14: 69.432, 30: 59.141, 60: 63.826, 90: 61.439}
AUTOML1_STRESS  = {1: 128.840, 7: 124.008, 14: 123.117, 30: 123.612, 60: 125.372, 90: 125.709}

print("=== AutoML Baseline v2: Same Rows, No Feature Engineering ===")
print(f"FLAML time budget: {TIME_BUDGET}s per horizon\n")

# ── Load training data from FE4 CSV ──────────────────────────────────────────
print("Loading training data (FE4 CSV)...")
df = pd.read_csv(str(FE4_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# Same rows as our LightGBM pipeline
train_mask = df["_fold_3"].isin(["train", "val"]) & (df["is_covid_lockdown_period"] == 0)
df_tr = df[train_mask].copy().reset_index(drop=True)
print(f"Training rows: {len(df_tr)}  ({df_tr['date'].min().date()} to {df_tr['date'].max().date()})")

# ── Raw features only (no lags, no seasonal medians, no anchors) ─────────────
WX_COLS   = [c for c in df_tr.columns if c.startswith("wx_")]
COMP_COLS = ["comp_n_competitors_closed", "comp_any_closed", "comp_closure_density"]
COMP_COLS = [c for c in COMP_COLS if c in df_tr.columns]

# Basic date features (derive fresh — same as first AutoML run)
df_tr["year"]      = df_tr["date"].dt.year
df_tr["month"]     = df_tr["date"].dt.month
df_tr["day"]       = df_tr["date"].dt.day
df_tr["dayofweek"] = df_tr["date"].dt.dayofweek
df_tr["week"]      = df_tr["date"].dt.isocalendar().week.astype(int)
df_tr["dayofyear"] = df_tr["date"].dt.dayofyear

DATE_COLS = ["year", "month", "day", "dayofweek", "week", "dayofyear"]
FEAT_COLS = DATE_COLS + WX_COLS + COMP_COLS
print(f"Features ({len(FEAT_COLS)}): {FEAT_COLS}\n")

# ── Load test data (from first AutoML run CSVs — same features, same parsing) ─
df_primary = pd.read_csv(PRIMARY_CSV, parse_dates=["date"])
df_stress  = pd.read_csv(STRESS_CSV,  parse_dates=["date"])

# Align competitor col names (first run used comp_n_closed; we need same cols)
# Map our COMP_COLS to what exists in the test CSVs
FEAT_COLS_TEST = []
for c in FEAT_COLS:
    if c in df_primary.columns:
        FEAT_COLS_TEST.append(c)
    elif c == "comp_n_competitors_closed" and "comp_n_closed" in df_primary.columns:
        # rename for test alignment
        df_primary["comp_n_competitors_closed"] = df_primary["comp_n_closed"]
        df_stress["comp_n_competitors_closed"]  = df_stress["comp_n_closed"]
        FEAT_COLS_TEST.append(c)
    elif c in ["comp_any_closed", "comp_closure_density"]:
        # Not in test CSVs, fill with 0 (competitor coverage ended in 2020)
        df_primary[c] = 0.0
        df_stress[c]  = 0.0
        FEAT_COLS_TEST.append(c)
    else:
        df_primary[c] = 0.0
        df_stress[c]  = 0.0
        FEAT_COLS_TEST.append(c)

# ── FLAML AutoML per horizon ──────────────────────────────────────────────────
print(f"{'='*60}")
print(f"Running FLAML AutoML — same rows, no FE")
print(f"{'='*60}")

results    = {}
models_dir = OUT_DIR / "models_samerows"
models_dir.mkdir(exist_ok=True)

for h in HORIZONS:
    tgt     = f"target_h{h}"     # test CSVs use target_h{h}
    tgt_tr  = f"bread_quantity_h{h}"  # training CSV uses bread_quantity_h{h}

    tr = df_tr[FEAT_COLS + [tgt_tr]].dropna(subset=[tgt_tr])
    X_tr = tr[FEAT_COLS].fillna(0).values
    y_tr = tr[tgt_tr].values

    print(f"\nh={h}: training on {len(tr)} rows")

    automl = AutoML()
    automl.fit(X_tr, y_tr, task="regression", metric="mae",
               time_budget=TIME_BUDGET, verbose=0)

    best_model = automl.best_estimator
    best_mae   = automl.best_loss
    print(f"  Best model: {best_model}  best_val_MAE: {best_mae:.3f}")

    pkl_path = models_dir / f"automl_samerows_h{h}.pkl"
    pickle.dump(automl, open(pkl_path, "wb"))

    # Primary test
    te_p = df_primary[FEAT_COLS_TEST + [tgt]].dropna(subset=[tgt])
    if len(te_p) > 0:
        preds_p = automl.predict(te_p[FEAT_COLS_TEST].fillna(0).values)
        mae_p   = float(np.mean(np.abs(preds_p - te_p[tgt].values)))
    else:
        mae_p = np.nan

    # Stress test
    te_s = df_stress[FEAT_COLS_TEST + [tgt]].dropna(subset=[tgt])
    if len(te_s) > 0:
        preds_s = automl.predict(te_s[FEAT_COLS_TEST].fillna(0).values)
        mae_s   = float(np.mean(np.abs(preds_s - te_s[tgt].values)))
    else:
        mae_s = np.nan

    print(f"  Primary test MAE: {mae_p:.3f}  (LightGBM: {LGBM_PRIMARY[h]:.3f}  "
          f"AutoML-v1: {AUTOML1_PRIMARY[h]:.3f}  Naive: {NAIVE_PRIMARY[h]:.3f})")
    print(f"  Stress  test MAE: {mae_s:.3f}  (LightGBM: {LGBM_STRESS[h]:.3f}  "
          f"AutoML-v1: {AUTOML1_STRESS[h]:.3f}  Naive: {NAIVE_STRESS[h]:.3f})")

    results[h] = {
        "best_model": best_model, "best_val_mae": float(best_mae),
        "n_train": len(tr),
        "primary_mae": mae_p, "primary_n": len(te_p),
        "stress_mae": mae_s,  "stress_n":  len(te_s),
    }

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print("COMPARISON SUMMARY — Primary test (2021)")
print(f"{'h':>3}  {'Naive':>8}  {'AutoML-v1':>10}  {'AutoML-v2':>10}  {'LightGBM':>10}")
print("-"*50)
for h in HORIZONS:
    r = results[h]
    print(f"  {h:>2}  {NAIVE_PRIMARY[h]:>8.3f}  {AUTOML1_PRIMARY[h]:>10.3f}  "
          f"{r['primary_mae']:>10.3f}  {LGBM_PRIMARY[h]:>10.3f}")

print(f"\nCOMPARISON SUMMARY — Stress test (2022-2023)")
print(f"{'h':>3}  {'Naive':>8}  {'AutoML-v1':>10}  {'AutoML-v2':>10}  {'LightGBM':>10}")
print("-"*50)
for h in HORIZONS:
    r = results[h]
    print(f"  {h:>2}  {NAIVE_STRESS[h]:>8.3f}  {AUTOML1_STRESS[h]:>10.3f}  "
          f"{r['stress_mae']:>10.3f}  {LGBM_STRESS[h]:>10.3f}")

output = {
    "run": "automl_baseline_samerows",
    "tool": "FLAML", "flaml_version": "2.6.0",
    "time_budget_per_horizon_s": TIME_BUDGET,
    "training_rows": "fold-3 train+val, COVID excluded (same as LightGBM pipeline)",
    "features": FEAT_COLS, "n_features": len(FEAT_COLS),
    "results": {str(h): results[h] for h in HORIZONS},
}
with open(OUT_DIR / "automl_samerows_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: automl_baseline/automl_samerows_results.json")
print("\n=== AutoML v2 complete ===")
