# automl_baseline/run_automl.py
#
# Pure AutoML comparison baseline.
# No domain knowledge, no feature engineering, no custom preprocessing.
# Steps:
#   1. Parse raw XML sales -> daily bread_quantity
#   2. Join raw weather + competitor as-is
#   3. Add basic date columns (year, month, day, dow, week, doy)
#   4. For each horizon h: target = bread_quantity shifted -h days
#   5. FLAML AutoML per horizon (time_budget=120s each)
#   6. Evaluate on primary (2021) and stress (2022-2023) test sets
#   7. Compare vs our LightGBM pipeline results

import re
import glob
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from flaml import AutoML
warnings.filterwarnings("ignore")

OUT_DIR     = Path(__file__).parent
TRAIN_DIR   = Path("C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/data/train_set/train")
TEST_ROOT   = Path("C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/data/test_set")
PRIMARY_DIR = TEST_ROOT / "holdout_primary"
STRESS_DIR  = TEST_ROOT / "holdout_stress"

HORIZONS    = [1, 7, 14, 30, 60, 90]
TIME_BUDGET = 120  # seconds per horizon

# Reference results from our LightGBM pipeline (iter027)
LGBM_PRIMARY = {1: 39.313, 7: 41.187, 14: 44.892, 30: 42.950, 60: 46.303, 90: 64.924}
LGBM_STRESS  = {1: 102.397, 7: 112.560, 14: 129.852, 30: 105.400, 60: 111.956, 90: 134.741}

print("=== AutoML Baseline ===")
print(f"FLAML time budget: {TIME_BUDGET}s per horizon\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: BREAD ARTICLE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
prod_csv = TRAIN_DIR / "productdata" / "productdata.csv"
prod_df  = pd.read_csv(prod_csv, sep="|", encoding="latin-1", on_bad_lines="skip", low_memory=False)
prod_df.columns = [c.lstrip("\ufeff").strip() for c in prod_df.columns]
bread_articles = set(
    prod_df[prod_df["argcod"].astype(str).str.strip() == "1"]["artcod"]
    .astype(str).str.strip().str.zfill(8)
)
print(f"Bread articles: {len(bread_articles)}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: XML PARSING
# ═══════════════════════════════════════════════════════════════════════════════
def parse_xml(fpath, bread_set):
    try:
        root = ET.parse(fpath).getroot()
    except ET.ParseError:
        return None, None
    sd = root.get("salesdate", "")
    use_new = len(root.findall("sale")) > 0
    txn_tag = "sale" if use_new else "order"
    det_tag = "saledetail" if use_new else "orderdetail"
    bread_qty = 0.0
    for txn in root.findall(txn_tag):
        for det in txn.findall(det_tag):
            art = det.get("articlereference", "").strip().zfill(8)
            try:
                qty = float(det.get("quantity", "0").replace(",", "."))
            except Exception:
                qty = 0.0
            if art in bread_set:
                bread_qty += qty
    return sd, bread_qty


def load_xml_dir(xml_base):
    xmls = glob.glob(str(xml_base / "**" / "*.xml"), recursive=True)
    date_to_files = {}
    for fpath in xmls:
        m = re.search(r"PT_Sales_\d+_(\d{8})_(\d{14})\.xml", Path(fpath).name)
        if m:
            d, ts = m.group(1), m.group(2)
            if d not in date_to_files or ts > date_to_files[d][0]:
                date_to_files[d] = (ts, fpath)
    records = []
    for d, (_, fpath) in sorted(date_to_files.items()):
        sd, bq = parse_xml(fpath, bread_articles)
        if sd:
            records.append({"date": pd.Timestamp(sd), "bread_quantity": bq})
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


print("Parsing training XMLs...")
train_sales = load_xml_dir(TRAIN_DIR / "salesdata" / "raw_xml" / "01")
print(f"  Train: {len(train_sales)} days  {train_sales['date'].min().date()} to {train_sales['date'].max().date()}")

print("Parsing test XMLs (primary)...")
primary_sales = load_xml_dir(PRIMARY_DIR / "salesdata" / "raw_xml" / "01")
print(f"  Primary: {len(primary_sales)} days  {primary_sales['date'].min().date()} to {primary_sales['date'].max().date()}")

print("Parsing test XMLs (stress)...")
stress_sales = load_xml_dir(STRESS_DIR / "salesdata" / "raw_xml" / "01")
print(f"  Stress:  {len(stress_sales)} days  {stress_sales['date'].min().date()} to {stress_sales['date'].max().date()}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: WEATHER JOIN
# ═══════════════════════════════════════════════════════════════════════════════
def load_weather(wx_dir):
    wx_files = glob.glob(str(wx_dir / "*.parquet")) + glob.glob(str(wx_dir / "*.snappy.parquet"))
    if not wx_files:
        return pd.DataFrame(columns=["date"])
    wx_raw = pd.read_parquet(wx_files[0])
    wx_raw["_date"] = pd.to_datetime(wx_raw["Date_time"], errors="coerce").dt.normalize()
    drop_cols = {"Date_time","Location","Address","Conditions","Icon","Stations",
                 "Wind_Chill","Heat_Index","_date"}
    num_cols = [c for c in wx_raw.columns
                if c not in drop_cols and wx_raw[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    agg = {c: ("max" if "Maximum" in c else "min" if "Minimum" in c
               else "max" if "Wind_Gust" in c else "mean") for c in num_cols}
    wx_day = wx_raw.groupby("_date")[num_cols].agg(agg).reset_index()
    wx_day.rename(columns={"_date": "date", **{c: f"wx_{c}" for c in num_cols}}, inplace=True)
    return wx_day


wx_train   = load_weather(TRAIN_DIR   / "weatherdata")
wx_primary = load_weather(PRIMARY_DIR / "weatherdata")
wx_stress  = load_weather(STRESS_DIR  / "weatherdata")
print(f"\nWeather loaded: train={len(wx_train)}, primary={len(wx_primary)}, stress={len(wx_stress)} days")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: COMPETITOR JOIN
# ═══════════════════════════════════════════════════════════════════════════════
def load_competitor(comp_dir):
    vc_path = comp_dir / "VerlofConcurrent.csv"
    try:
        vc = pd.read_csv(vc_path, sep=";", encoding="latin-1")
        date_cols   = [c for c in vc.columns if "datum" in c.lower() or "date" in c.lower()]
        winkel_cols = [c for c in vc.columns if "winkel" in c.lower()]
        if not date_cols:
            return pd.DataFrame(columns=["date"])
        dc = date_cols[0]
        vc[dc] = pd.to_datetime(vc[dc], dayfirst=True, errors="coerce")
        vc = vc.dropna(subset=[dc])
        id_col = winkel_cols[0] if winkel_cols else vc.columns[0]
        comp = vc.groupby(dc).agg(comp_n_closed=(id_col, "count")).reset_index()
        comp.rename(columns={dc: "date"}, inplace=True)
        return comp
    except Exception:
        return pd.DataFrame(columns=["date"])


comp_train = load_competitor(TRAIN_DIR / "competitordata")
print(f"Competitor loaded: {len(comp_train)} days")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: BUILD FEATURE TABLE
# ═══════════════════════════════════════════════════════════════════════════════
def build_features(sales_df, wx_df, comp_df=None):
    df = sales_df.copy()
    # Basic date features — minimal, no domain reasoning
    df["year"]      = df["date"].dt.year
    df["month"]     = df["date"].dt.month
    df["day"]       = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["week"]      = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    # Weather
    if "date" in wx_df.columns and len(wx_df) > 0:
        df = df.merge(wx_df, on="date", how="left")
    # Competitor
    if comp_df is not None and "date" in comp_df.columns and len(comp_df) > 0:
        df = df.merge(comp_df, on="date", how="left")
        df["comp_n_closed"] = df["comp_n_closed"].fillna(0)
    return df


print("\nBuilding feature tables...")
df_train   = build_features(train_sales, wx_train, comp_train)
df_primary = build_features(primary_sales, wx_primary)
df_stress  = build_features(stress_sales, wx_stress)

# Add targets: for each horizon, shift bread_quantity back h days
bq_lookup_train   = dict(zip(df_train["date"],   df_train["bread_quantity"]))
bq_lookup_primary = dict(zip(df_primary["date"], df_primary["bread_quantity"]))
bq_lookup_stress  = dict(zip(df_stress["date"],  df_stress["bread_quantity"]))

for h in HORIZONS:
    df_train[f"target_h{h}"]   = df_train["date"].apply(
        lambda d: bq_lookup_train.get(d + pd.Timedelta(days=h), np.nan))
    df_primary[f"target_h{h}"] = df_primary["date"].apply(
        lambda d: {**bq_lookup_primary}.get(d + pd.Timedelta(days=h), np.nan))
    df_stress[f"target_h{h}"]  = df_stress["date"].apply(
        lambda d: {**bq_lookup_stress}.get(d + pd.Timedelta(days=h), np.nan))

# Feature columns (everything except date, bread_quantity, and target_h* columns)
FEAT_COLS = [c for c in df_train.columns
             if c not in ["date", "bread_quantity"] and not c.startswith("target_h")]

# Ensure all feature cols exist in test dataframes (fill missing with 0)
for col in FEAT_COLS:
    if col not in df_primary.columns:
        df_primary[col] = 0
    if col not in df_stress.columns:
        df_stress[col] = 0

print(f"Feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")
df_train.to_csv(OUT_DIR / "train_features.csv", index=False)
df_primary.to_csv(OUT_DIR / "primary_features.csv", index=False)
df_stress.to_csv(OUT_DIR / "stress_features.csv", index=False)
print("Feature CSVs saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: FLAML AUTOML PER HORIZON
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Running FLAML AutoML — {TIME_BUDGET}s per horizon")
print(f"{'='*60}")

results     = {}
models_dir  = OUT_DIR / "models"
models_dir.mkdir(exist_ok=True)

for h in HORIZONS:
    tgt = f"target_h{h}"
    tr  = df_train[FEAT_COLS + [tgt]].dropna(subset=[tgt])
    X_tr = tr[FEAT_COLS].fillna(0).values
    y_tr = tr[tgt].values

    print(f"\nh={h}: training on {len(tr)} rows  target={tgt}")

    automl = AutoML()
    automl.fit(
        X_tr, y_tr,
        task="regression",
        metric="mae",
        time_budget=TIME_BUDGET,
        verbose=0,
    )

    best_model = automl.best_estimator
    best_mae   = automl.best_loss
    print(f"  Best model: {best_model}  best_val_MAE: {best_mae:.3f}")

    # Save model
    pkl_path = models_dir / f"automl_h{h}.pkl"
    pickle.dump(automl, open(pkl_path, "wb"))

    # Evaluate on primary test
    te_p = df_primary[FEAT_COLS + [tgt]].dropna(subset=[tgt])
    if len(te_p) > 0:
        X_p = te_p[FEAT_COLS].fillna(0).values
        y_p = te_p[tgt].values
        preds_p = automl.predict(X_p)
        mae_p   = float(np.mean(np.abs(preds_p - y_p)))
    else:
        mae_p = np.nan

    # Evaluate on stress test
    te_s = df_stress[FEAT_COLS + [tgt]].dropna(subset=[tgt])
    if len(te_s) > 0:
        X_s = te_s[FEAT_COLS].fillna(0).values
        y_s = te_s[tgt].values
        preds_s = automl.predict(X_s)
        mae_s   = float(np.mean(np.abs(preds_s - y_s)))
    else:
        mae_s = np.nan

    print(f"  Primary test MAE: {mae_p:.3f}  (LightGBM pipeline: {LGBM_PRIMARY[h]:.3f})")
    print(f"  Stress  test MAE: {mae_s:.3f}  (LightGBM pipeline: {LGBM_STRESS[h]:.3f})")

    results[h] = {
        "best_model": best_model,
        "best_val_mae": float(best_mae),
        "n_train": len(tr),
        "primary_mae": mae_p,
        "primary_n":   len(te_p),
        "stress_mae":  mae_s,
        "stress_n":    len(te_s),
        "lgbm_primary_mae": LGBM_PRIMARY[h],
        "lgbm_stress_mae":  LGBM_STRESS[h],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FINAL COMPARISON SUMMARY")
print(f"\nPrimary test (2021):")
print(f"{'h':>3}  {'AutoML MAE':>12}  {'LightGBM MAE':>14}  {'Winner':>8}")
print("-"*45)
for h in HORIZONS:
    r = results[h]
    am = r["primary_mae"]
    lm = r["lgbm_primary_mae"]
    winner = "AutoML" if am < lm else "LightGBM"
    print(f"  {h:>2}  {am:>12.3f}  {lm:>14.3f}  {winner:>8}")

print(f"\nStress test (2022-2023):")
print(f"{'h':>3}  {'AutoML MAE':>12}  {'LightGBM MAE':>14}  {'Winner':>8}")
print("-"*45)
for h in HORIZONS:
    r = results[h]
    am = r["stress_mae"]
    lm = r["lgbm_stress_mae"]
    winner = "AutoML" if am < lm else "LightGBM"
    print(f"  {h:>2}  {am:>12.3f}  {lm:>14.3f}  {winner:>8}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
output = {
    "run": "automl_baseline",
    "tool": "FLAML",
    "flaml_version": "2.6.0",
    "time_budget_per_horizon_s": TIME_BUDGET,
    "features_used": FEAT_COLS,
    "n_features": len(FEAT_COLS),
    "results": {str(h): results[h] for h in HORIZONS},
}
with open(OUT_DIR / "automl_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: automl_baseline/automl_results.json")
print("\n=== AutoML Baseline complete ===")
