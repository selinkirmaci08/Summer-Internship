# %% [iteration_028_corrected_naive] Test evaluation with corrected naive baseline
#
# Same pipeline as iter027, but fixes the naive baseline comparison.
#
# Issue in iter027: naive_h{h}_seasonal_median_w{w} used D.dayofweek (origin DOW)
# as the DOW index, not (D+h).dayofweek (target DOW). For h=7,14 these are the
# same (7%7=0), so those were fine. For h=1,30,60,90 the naive was comparing
# origin-DOW history against a target with a different DOW — a weaker strawman.
#
# Fix: compute anchor_correct for ALL horizons (seasonal median of last N
# same-DOW-as-TARGET observations, causally from origin D), and use that as
# the naive comparison in evaluate().
#
# Definition used throughout:
#   naive(D, h) = median of last W observations of DOW=(D+h).dayofweek
#                 from strictly before origin date D
#   W: {1:4, 7:4, 14:4, 30:26, 60:26, 90:13}

import json
import pickle
import re
import glob
import datetime
import warnings
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from dateutil.easter import easter
import lightgbm as lgb
warnings.filterwarnings("ignore")

RUN_DIR     = Path(__file__).parent.parent
FE4_CSV     = RUN_DIR / "data" / "iteration_019_feature_engineering.csv"
FE5_CSV     = RUN_DIR / "data" / "iteration_023_feature_engineering.csv"
FINAL_DIR   = RUN_DIR / "artifacts" / "training" / "final"
HIST_OUT    = RUN_DIR / "history" / "iteration_028_corrected_naive.json"

TEST_ROOT   = Path("C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/data/test_set")
PRIMARY_DIR = TEST_ROOT / "holdout_primary"
STRESS_DIR  = TEST_ROOT / "holdout_stress"
PROD_CSV    = PRIMARY_DIR / "productdata" / "productdata.csv"

HORIZONS    = [1, 7, 14, 30, 60, 90]
RATIO_CLIP  = (0.05, 5.0)
ANCHOR_DATE = pd.Timestamp("2010-11-16")
REGIME_DATE = pd.Timestamp("2020-06-01")

# Correct naive window per horizon
NAIVE_WINDOWS = {1: 4, 7: 4, 14: 4, 30: 26, 60: 26, 90: 13}

print("=== Iteration 028: Test Evaluation with Corrected Naive Baseline ===")


# ── 1. Bread article map ───────────────────────────────────────────────────────
print("\n[1] Loading bread article map...")
prod_df = pd.read_csv(PROD_CSV, sep="|", encoding="latin-1", on_bad_lines="skip", low_memory=False)
prod_df.columns = [c.lstrip("\ufeff").strip() for c in prod_df.columns]
bread_articles = set(
    prod_df[prod_df["argcod"].astype(str).str.strip() == "1"]["artcod"]
    .astype(str).str.strip().str.zfill(8)
)
print(f"  Bread articles: {len(bread_articles)}")


# ── 2. XML parsing ─────────────────────────────────────────────────────────────
def parse_xml(fpath, bread_set):
    tree = ET.parse(fpath)
    root = tree.getroot()
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


def load_xml_folder(xml_dir):
    xmls = glob.glob(str(xml_dir / "**" / "*.xml"), recursive=True)
    date_to_files = {}
    for fpath in xmls:
        m = re.search(r"PT_Sales_\d+_(\d{8})_(\d{14})\.xml", Path(fpath).name)
        if m:
            d, ts = m.group(1), m.group(2)
            if d not in date_to_files or ts > date_to_files[d][0]:
                date_to_files[d] = (ts, fpath)
    records = []
    for d, (_, fpath) in sorted(date_to_files.items()):
        try:
            sd, bq = parse_xml(fpath, bread_articles)
            records.append({"date": pd.Timestamp(sd), "bread_quantity": bq})
        except Exception as e:
            print(f"  WARNING: failed {fpath}: {e}")
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


print("\n[2] Parsing test XMLs...")
df_primary = load_xml_folder(PRIMARY_DIR / "salesdata" / "raw_xml" / "01")
df_stress  = load_xml_folder(STRESS_DIR  / "salesdata" / "raw_xml" / "01")
print(f"  Primary: {len(df_primary)} days  {df_primary['date'].min().date()} to {df_primary['date'].max().date()}")
print(f"  Stress:  {len(df_stress)} days  {df_stress['date'].min().date()} to {df_stress['date'].max().date()}")
df_primary["_test_set"] = "primary"
df_stress["_test_set"]  = "stress"


# ── 3. Weather ─────────────────────────────────────────────────────────────────
def load_weather_parquet(wx_dir):
    wx_files = glob.glob(str(wx_dir / "*.parquet")) + glob.glob(str(wx_dir / "*.snappy.parquet"))
    if not wx_files:
        return pd.DataFrame(columns=["date"])
    wx_raw = pd.read_parquet(wx_files[0])
    wx_raw["_date"] = pd.to_datetime(wx_raw["Date_time"], errors="coerce").dt.normalize()
    drop_cols = {"Date_time", "Location", "Address", "Conditions", "Icon", "Stations",
                 "Wind_Chill", "Heat_Index", "_date"}
    num_cols = [c for c in wx_raw.columns
                if c not in drop_cols and wx_raw[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    agg_method = {c: ("max" if "Maximum" in c else "min" if "Minimum" in c
                      else "max" if "Wind_Gust" in c else "mean") for c in num_cols}
    wx_agg = wx_raw.groupby("_date")[num_cols].agg(agg_method).reset_index()
    wx_agg.rename(columns={"_date": "date"}, inplace=True)
    wx_agg.rename(columns={c: f"wx_{c}" for c in num_cols}, inplace=True)
    return wx_agg

print("\n[3] Loading test weather...")
wx_primary = load_weather_parquet(PRIMARY_DIR / "weatherdata")
wx_stress  = load_weather_parquet(STRESS_DIR  / "weatherdata")

WX_COLS = ["wx_Maximum_Temperature", "wx_Minimum_Temperature", "wx_Temperature",
           "wx_Precipitation", "wx_Snow_depth", "wx_Wind_Speed", "wx_Wind_Gust",
           "wx_Cloud_Cover", "wx_Relative_Humidity"]

def join_weather(df_test, wx_df):
    avail_wx = [c for c in WX_COLS if c in wx_df.columns]
    if avail_wx:
        df_test = df_test.merge(wx_df[["date"] + avail_wx], on="date", how="left")
    for c in WX_COLS:
        if c not in df_test.columns:
            df_test[c] = np.nan
    return df_test

df_primary = join_weather(df_primary, wx_primary)
df_stress  = join_weather(df_stress, wx_stress)


# ── 4. Training history + DOW lookup tables ────────────────────────────────────
print("\n[4] Loading training history...")
df_train = pd.read_csv(str(FE4_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df_train["_test_set"] = "train"

test_all = pd.concat([df_primary, df_stress], ignore_index=True).sort_values("date").reset_index(drop=True)
all_dates = pd.concat([
    df_train[["date", "bread_quantity", "_test_set"] + WX_COLS],
    test_all[["date", "bread_quantity", "_test_set"] + WX_COLS]
], ignore_index=True).sort_values("date").reset_index(drop=True)

print(f"  Combined timeline: {len(all_dates)} rows  "
      f"{all_dates['date'].min().date()} to {all_dates['date'].max().date()}")

bq_series   = all_dates["bread_quantity"].values
date_series = all_dates["date"].values
dow_series  = pd.to_datetime(date_series).dayofweek

dow_idx  = {d: [] for d in range(7)}
dow_vals = {d: [] for d in range(7)}
for i, (dt, bq, dw) in enumerate(zip(date_series, bq_series, dow_series)):
    dow_idx[dw].append(pd.Timestamp(dt))
    dow_vals[dw].append(bq)


# ── 5. Features for test rows ─────────────────────────────────────────────────
print("\n[5] Computing features for test rows...")
test_rows = all_dates[all_dates["_test_set"] != "train"].copy().reset_index(drop=True)

# Calendar
test_rows["year"]        = test_rows["date"].dt.year
test_rows["month"]       = test_rows["date"].dt.month
test_rows["dow"]         = test_rows["date"].dt.dayofweek
test_rows["week"]        = test_rows["date"].dt.isocalendar().week.astype(int)
test_rows["doy"]         = test_rows["date"].dt.dayofyear
test_rows["day_of_week"] = test_rows["date"].dt.dayofweek
test_rows["is_closed_monday"] = False
test_rows["year_trend"]  = (test_rows["date"] - ANCHOR_DATE).dt.days / 365.25
test_rows["regime_x_monday"] = (test_rows["dow"] == 0).astype(int)
test_rows["is_2014_sparse_period"] = 0
test_rows["weather_data_available"] = 1

# Holidays
YEARS_EXT = range(2009, 2025)

def build_holiday_set(years):
    h = set()
    for y in years:
        e = easter(y)
        fixed = [datetime.date(y,1,1), datetime.date(y,5,1), datetime.date(y,7,11),
                 datetime.date(y,7,21), datetime.date(y,8,15), datetime.date(y,11,1),
                 datetime.date(y,11,11), datetime.date(y,12,25)]
        floating = [e+datetime.timedelta(1), e+datetime.timedelta(39), e+datetime.timedelta(50)]
        h.update(fixed + floating)
    return h

HOLIDAYS = build_holiday_set(YEARS_EXT)
date_col = test_rows["date"].dt.date

test_rows["is_official_public_holiday"] = date_col.apply(lambda d: int(d in HOLIDAYS))
test_rows["is_day_before_holiday"]      = date_col.apply(lambda d: int((d + datetime.timedelta(1)) in HOLIDAYS))

def days_until_next(d, holidays, cap=30):
    for lag in range(0, cap+1):
        if (d + datetime.timedelta(days=lag)) in holidays:
            return lag
    return cap
test_rows["days_until_next_holiday"] = date_col.apply(lambda d: days_until_next(d, HOLIDAYS))

def days_since_last(d, holidays, cap=14):
    for lag in range(0, cap+1):
        if (d - datetime.timedelta(days=lag)) in holidays:
            return lag
    return cap
test_rows["days_since_last_holiday"] = date_col.apply(lambda d: days_since_last(d, HOLIDAYS))

def _get_monday(d):
    return d - datetime.timedelta(days=d.weekday())

def in_herfstvakantie(d):
    nov1 = datetime.date(d.year, 11, 1)
    start = _get_monday(nov1)
    return start <= d <= start + datetime.timedelta(6)

def in_kerstvakantie(d):
    dec25 = datetime.date(d.year, 12, 25)
    start = _get_monday(dec25)
    return start <= d <= start + datetime.timedelta(13)

def in_krokusvakantie(d):
    e = easter(d.year)
    start = e - datetime.timedelta(48)
    return start <= d <= start + datetime.timedelta(6)

def in_paasvakantie(d):
    e = easter(d.year)
    start = e - datetime.timedelta(13)
    return start <= d <= start + datetime.timedelta(13)

def in_zomervakantie(d):
    return datetime.date(d.year, 7, 1) <= d <= datetime.date(d.year, 8, 31)

GENTSE_FEESTEN = {
    2021: ("2021-07-10", "2021-07-19"),
    2022: ("2022-07-15", "2022-07-24"),
    2023: ("2023-07-14", "2023-07-23"),
}
GENTSE_DATES = set()
for yr, (s, e) in GENTSE_FEESTEN.items():
    d = pd.Timestamp(s)
    while d <= pd.Timestamp(e):
        GENTSE_DATES.add(d.date()); d += pd.Timedelta(days=1)

EASTER_DATES = set()
for y in YEARS_EXT:
    E = easter(y).timetuple()
    E = datetime.date(E.tm_year, E.tm_mon, E.tm_mday)
    for delta in range(-2, 2):
        EASTER_DATES.add(E + datetime.timedelta(days=delta))

test_rows["is_herfstvakantie"]       = date_col.apply(lambda d: int(in_herfstvakantie(d)))
test_rows["is_kerstvakantie"]        = date_col.apply(lambda d: int(in_kerstvakantie(d)))
test_rows["is_krokusvakantie"]       = date_col.apply(lambda d: int(in_krokusvakantie(d)))
test_rows["is_paasvakantie"]         = date_col.apply(lambda d: int(in_paasvakantie(d)))
test_rows["is_zomervakantie"]        = date_col.apply(lambda d: int(in_zomervakantie(d)))
test_rows["is_gentse_feesten"]       = date_col.apply(lambda d: int(d in GENTSE_DATES))
test_rows["is_sinterklaas_period"]   = (
    (test_rows["month"] == 12) & (test_rows["date"].dt.day <= 6)).astype(int)
test_rows["is_easter_period"]        = date_col.apply(lambda d: int(d in EASTER_DATES))
test_rows["is_covid_lockdown_period"] = 0

test_rows["is_hot_day"]    = (test_rows["wx_Temperature"] > 25).astype(int)
test_rows["is_heavy_rain"] = (test_rows["wx_Precipitation"] > 5).astype(int)

test_rows["comp_n_competitors_closed"] = 0.0
test_rows["comp_any_closed"]           = 0
test_rows["comp_closure_density"]      = 0.0
test_rows["comp_data_available"]       = 0

# Lag features
date_to_bq = dict(zip(pd.to_datetime(all_dates["date"]), all_dates["bread_quantity"]))

for lag in [1, 2, 3, 7, 14, 35, 63, 91]:
    test_rows[f"bread_lag_{lag}d"] = test_rows["date"].apply(
        lambda d: date_to_bq.get(d - pd.Timedelta(days=lag), np.nan))

# Naive seasonal medians (origin-DOW based — kept as model features, not naive comparison)
NAIVE_SPECS = {
    1:  [("smedian", 4), ("smedian", 13), ("smedian", 26)],
    7:  [("smedian", 4), ("smedian", 13), ("smedian", 26)],
    14: [("smedian", 4), ("smedian", 13), ("smedian", 26)],
    30: [("smedian", 4), ("smedian", 26), ("smedian", 8)],
    60: [("smedian", 26), ("smean",   52), ("smedian", 8)],
    90: [("smedian", 26), ("smean",   52), ("smean",   13)],
}

def lk_smedian(O, target_dow, w):
    idx = dow_idx[target_dow]
    pos = bisect_left(idx, O)
    if pos == 0:
        return np.nan
    lo = max(0, pos - w)
    return float(np.median(dow_vals[target_dow][lo:pos]))

def lk_smean(O, target_dow, w):
    idx = dow_idx[target_dow]
    pos = bisect_left(idx, O)
    if pos == 0:
        return np.nan
    lo = max(0, pos - w)
    return float(np.mean(dow_vals[target_dow][lo:pos]))

for h in HORIZONS:
    for kind, w in NAIVE_SPECS[h]:
        col = f"naive_h{h}_seasonal_median_w{w}" if kind == "smedian" else f"naive_h{h}_seasonal_mean_w{w}"
        vals = []
        for D in test_rows["date"]:
            O = D - pd.Timedelta(days=h)
            tgt_dow = D.dayofweek  # origin DOW — kept for model features
            v = lk_smedian(O, tgt_dow, w) if kind == "smedian" else lk_smean(O, tgt_dow, w)
            vals.append(v)
        test_rows[col] = vals
        test_rows[col] = test_rows[col].ffill().bfill()

# Forward-looking targets
for h in HORIZONS:
    test_rows[f"bread_quantity_h{h}"] = test_rows["date"].apply(
        lambda d: date_to_bq.get(d + pd.Timedelta(days=h), np.nan))

# Ratio targets (original naive anchor, for model features)
naive_anchor_col = {1: "naive_h1_seasonal_median_w4", 7: "naive_h7_seasonal_median_w4",
                    14: "naive_h14_seasonal_median_w4", 30: "naive_h30_seasonal_median_w4",
                    60: "naive_h60_seasonal_median_w26", 90: "naive_h90_seasonal_median_w26"}
for h in HORIZONS:
    anc = test_rows[naive_anchor_col[h]].replace(0, np.nan)
    test_rows[f"ratio_h{h}"] = (test_rows[f"bread_quantity_h{h}"] / anc).clip(*RATIO_CLIP)

# ── CORRECTED anchors: target DOW for ALL horizons ────────────────────────────
# This is the proper same-DOW-as-target seasonal median, used as naive comparison.
print("  Computing corrected naive (target-DOW anchor) for all horizons...")
for h in HORIZONS:
    W = NAIVE_WINDOWS[h]
    anchor_vals = []
    for D in test_rows["date"]:
        target_date = D + pd.Timedelta(days=h)
        target_dow  = target_date.dayofweek
        ref_idx     = dow_idx[target_dow]
        pos = bisect_left(ref_idx, D)  # causal cutoff at origin D
        if pos == 0:
            anchor_vals.append(np.nan)
            continue
        start  = max(0, pos - W)
        recent = [v for v in dow_vals[target_dow][start:pos] if not np.isnan(v)]
        anchor_vals.append(float(np.median(recent)) if recent else np.nan)
    test_rows[f"anchor_correct_h{h}"] = anchor_vals
    anc_s = pd.Series(anchor_vals).replace(0, np.nan)
    tgt_s = test_rows[f"bread_quantity_h{h}"]
    test_rows[f"ratio_correct_h{h}"] = (tgt_s / anc_s).clip(*RATIO_CLIP)

# Temperature z-score
df_train_wx = df_train[df_train["date"].dt.year >= 2015][["date", "wx_Temperature"]].dropna()
doy_stats = (df_train_wx.assign(doy=df_train_wx["date"].dt.dayofyear)
             .groupby("doy")["wx_Temperature"].agg(["mean", "std"])
             .rename(columns={"mean": "clim_mean", "std": "clim_std"}))
global_mean = df_train_wx["wx_Temperature"].mean()
global_std  = df_train_wx["wx_Temperature"].std()
doy_stats["clim_std"] = doy_stats["clim_std"].fillna(global_std).replace(0, global_std)

def calc_zscore(row):
    if pd.isna(row["wx_Temperature"]):
        return 0.0
    doy = row["date"].dayofyear
    if doy not in doy_stats.index:
        return 0.0
    return (row["wx_Temperature"] - doy_stats.loc[doy, "clim_mean"]) / doy_stats.loc[doy, "clim_std"]

test_rows["temperature_zscore_seasonal"] = test_rows.apply(calc_zscore, axis=1)

test_rows["_fold_1"] = None
test_rows["_fold_2"] = None
test_rows["_fold_3"] = None
test_rows["_split"]  = "test"
print(f"  Test features computed: {test_rows.shape}")


# ── 6. Feature sets ────────────────────────────────────────────────────────────
EXCLUDE_BASE = {
    "date", "bread_quantity", "is_covid_lockdown_period", "day_of_week",
    "_split", "_fold_1", "_fold_2", "_fold_3", "_test_set",
}

def get_feats_from_csv(fe_csv_path, test_df):
    ref = pd.read_csv(fe_csv_path, nrows=1)
    pool = [c for c in ref.columns
            if c not in EXCLUDE_BASE
            and not c.startswith("bread_quantity_h") and not c.startswith("ratio_h")
            and not c.startswith("ratio_correct_") and not c.startswith("anchor_correct")
            and ref[c].dtype != object]
    return [c for c in pool if c in test_df.columns]

feats4 = get_feats_from_csv(FE4_CSV, test_rows)
feats5 = get_feats_from_csv(FE5_CSV, test_rows)
print(f"\nFeature counts: FE4={len(feats4)}, FE5={len(feats5)}")


# ── 7. Load models ─────────────────────────────────────────────────────────────
print("\n[7] Loading final models...")
MODELS = {}
for h in HORIZONS:
    pkl = FINAL_DIR / f"lgbm_h{h}_final.pkl"
    MODELS[h] = pickle.load(open(pkl, "rb"))
    print(f"  h={h}: loaded {pkl.name}")

MODEL_CONFIG = {
    1:  {"feats": feats4, "tgt_col": "bread_quantity_h1",  "anchor": "anchor_correct_h1",  "ratio": True},
    7:  {"feats": feats5, "tgt_col": "bread_quantity_h7",  "anchor": "naive_h7_seasonal_median_w4", "ratio": True},
    14: {"feats": feats4, "tgt_col": "bread_quantity_h14", "anchor": None,                  "ratio": False},
    30: {"feats": feats5, "tgt_col": "bread_quantity_h30", "anchor": "anchor_correct_h30",  "ratio": True},
    60: {"feats": feats4, "tgt_col": "bread_quantity_h60", "anchor": "anchor_correct_h60",  "ratio": True},
    90: {"feats": feats4, "tgt_col": "bread_quantity_h90", "anchor": None,                  "ratio": False},
}
VAL_MAE = {1: 29.835, 7: 31.646, 14: 34.782, 30: 36.630, 60: 36.229, 90: 56.487}


# ── 8. Evaluate ────────────────────────────────────────────────────────────────
def evaluate(test_df, label):
    print(f"\n{'='*60}")
    print(f"TEST SET: {label.upper()}")
    print(f"  Naive = seasonal median of last W same-DOW-as-target observations")
    results = {}
    for h in HORIZONS:
        cfg   = MODEL_CONFIG[h]
        tgt   = cfg["tgt_col"]
        feats = cfg["feats"]

        mask = test_df[tgt].notna()
        if mask.sum() == 0:
            print(f"  h={h}: no scoreable rows")
            results[h] = None
            continue

        X = test_df.loc[mask, feats].fillna(0).values
        y = test_df.loc[mask, tgt].values

        preds = MODELS[h].predict(X)
        if cfg["ratio"] and cfg["anchor"]:
            a = test_df.loc[mask, cfg["anchor"]].values
            preds = np.clip(preds, *RATIO_CLIP) * a

        mae = float(np.mean(np.abs(preds - y)))
        n   = int(mask.sum())

        # CORRECTED naive: target-DOW seasonal median
        naive_vals = test_df.loc[mask, f"anchor_correct_h{h}"].values
        naive_mae  = float(np.mean(np.abs(naive_vals - y)))

        beats = mae < naive_mae
        delta_vs_val = mae - VAL_MAE[h]

        print(f"  h={h:>2}: MAE={mae:>7.3f}  naive(corrected)={naive_mae:>7.3f}  "
              f"beats={str(beats):>5}  n={n:>3}  vs_val={delta_vs_val:+.3f}")
        results[h] = {"mae": mae, "naive_mae": naive_mae, "beats_naive": beats,
                      "n": n, "val_mae": VAL_MAE[h], "delta_vs_val": delta_vs_val,
                      "naive_window": NAIVE_WINDOWS[h]}
    return results

primary_rows = test_rows[test_rows["_test_set"] == "primary"].copy()
stress_rows  = test_rows[test_rows["_test_set"] == "stress"].copy()

results_primary = evaluate(primary_rows, "holdout_primary (2021)")
results_stress  = evaluate(stress_rows,  "holdout_stress  (2022-2023)")


# ── 9. Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FULL SUMMARY — Corrected Naive (same DOW as target, last W weeks)")
print(f"{'h':>3}  {'W':>3}  {'Val MAE':>8}  {'Primary':>9}  {'P.Naive':>8}  "
      f"{'Stress':>8}  {'S.Naive':>8}")
print("-"*65)
for h in HORIZONS:
    rp = results_primary.get(h)
    rs = results_stress.get(h)
    pm = f"{rp['mae']:.3f}" if rp else "N/A"
    pn = f"{rp['naive_mae']:.3f}" if rp else "N/A"
    sm = f"{rs['mae']:.3f}" if rs else "N/A"
    sn = f"{rs['naive_mae']:.3f}" if rs else "N/A"
    W  = NAIVE_WINDOWS[h]
    print(f"  {h:>2}  {W:>3}  {VAL_MAE[h]:>8.3f}  {pm:>9}  {pn:>8}  {sm:>8}  {sn:>8}")


# ── 10. Save ───────────────────────────────────────────────────────────────────
handoff = {
    "iteration": 28,
    "node": "evaluation",
    "timestamp": "2026-09-09",
    "schema_version": "2026-09-04",
    "type": "test_evaluation_corrected_naive",
    "naive_definition": "seasonal median of last W same-DOW-as-target observations from before origin date",
    "naive_windows": NAIVE_WINDOWS,
    "results": {
        "holdout_primary": {str(h): results_primary.get(h) for h in HORIZONS},
        "holdout_stress":  {str(h): results_stress.get(h)  for h in HORIZONS},
    },
    "val_mae_reference": {str(h): VAL_MAE[h] for h in HORIZONS},
}
with open(HIST_OUT, "w", encoding="utf-8") as f:
    json.dump(handoff, f, indent=2)
print(f"\nResults saved: {HIST_OUT.name}")
print("\n=== Iteration 028 complete ===")
