# %% [iteration_029_open_days_evaluation]
#
# Re-evaluates stress test excluding zero-sales (closed) days from the target.
#
# Finding from iter028 diagnostic:
#   - 65/173 stress test days (37.6%) have zero bread sales
#   - These are almost exclusively weekdays (Mon-Fri: 38-65% zero, Sat: 0%, Sun: 11%)
#   - On open days, stress mean = 244/day — nearly identical to training (265/day)
#
# Predicting store-closure days is a classification problem, not a forecasting problem.
# Evaluating on open days only gives a fair picture of model forecasting capability.
#
# Outputs:
#   - MAE on ALL stress days (includes closed days)
#   - MAE on OPEN stress days only (target bread_quantity > 0)

import json, pickle, re, glob, datetime, warnings
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from dateutil.easter import easter
warnings.filterwarnings("ignore")

RUN_DIR   = Path(__file__).parent.parent
FE4_CSV   = RUN_DIR / "data" / "iteration_019_feature_engineering.csv"
FE5_CSV   = RUN_DIR / "data" / "iteration_023_feature_engineering.csv"
FINAL_DIR = RUN_DIR / "artifacts" / "training" / "final"
HIST_OUT  = RUN_DIR / "history" / "iteration_029_open_days_evaluation.json"

TEST_ROOT   = Path("C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/data/test_set")
PRIMARY_DIR = TEST_ROOT / "holdout_primary"
STRESS_DIR  = TEST_ROOT / "holdout_stress"
PROD_CSV    = PRIMARY_DIR / "productdata" / "productdata.csv"

HORIZONS      = [1, 7, 14, 30, 60, 90]
RATIO_CLIP    = (0.05, 5.0)
ANCHOR_DATE   = pd.Timestamp("2010-11-16")
NAIVE_WINDOWS = {1: 4, 7: 4, 14: 4, 30: 26, 60: 26, 90: 13}

print("=== Iteration 029: Open-Days Stress Test Evaluation ===\n")

# ── Bread articles ─────────────────────────────────────────────────────────────
prod_df = pd.read_csv(PROD_CSV, sep="|", encoding="latin-1", on_bad_lines="skip", low_memory=False)
prod_df.columns = [c.lstrip("\ufeff").strip() for c in prod_df.columns]
bread_articles = set(
    prod_df[prod_df["argcod"].astype(str).str.strip() == "1"]["artcod"]
    .astype(str).str.strip().str.zfill(8)
)

# ── XML parsing ────────────────────────────────────────────────────────────────
def parse_xml(fpath, bread_set):
    try:
        root = ET.parse(fpath).getroot()
    except Exception:
        return None, 0.0
    sd = root.get("salesdate", "")
    use_new = len(root.findall("sale")) > 0
    txn_tag = "sale" if use_new else "order"
    det_tag = "saledetail" if use_new else "orderdetail"
    bq = 0.0
    for txn in root.findall(txn_tag):
        for det in txn.findall(det_tag):
            art = det.get("articlereference", "").strip().zfill(8)
            try:
                qty = float(det.get("quantity", "0").replace(",", "."))
            except Exception:
                qty = 0.0
            if art in bread_set:
                bq += qty
    return sd, bq

def load_dir(xml_dir):
    xmls = glob.glob(str(xml_dir / "**" / "*.xml"), recursive=True)
    dtf = {}
    for fpath in xmls:
        m = re.search(r"PT_Sales_\d+_(\d{8})_(\d{14})\.xml", Path(fpath).name)
        if m:
            d, ts = m.group(1), m.group(2)
            if d not in dtf or ts > dtf[d][0]:
                dtf[d] = (ts, fpath)
    recs = []
    for d, (_, fpath) in sorted(dtf.items()):
        sd, bq = parse_xml(fpath, bread_articles)
        if sd:
            recs.append({"date": pd.Timestamp(sd), "bread_quantity": bq})
    return pd.DataFrame(recs).sort_values("date").reset_index(drop=True)

print("[1] Parsing test XMLs...")
df_primary = load_dir(PRIMARY_DIR / "salesdata" / "raw_xml" / "01")
df_stress  = load_dir(STRESS_DIR  / "salesdata" / "raw_xml" / "01")
df_primary["_test_set"] = "primary"
df_stress["_test_set"]  = "stress"

zero_p = (df_primary["bread_quantity"] == 0).sum()
zero_s = (df_stress["bread_quantity"]  == 0).sum()
print(f"  Primary: {len(df_primary)} days  ({zero_p} zero-sales days, {100*zero_p/len(df_primary):.0f}%)")
print(f"  Stress:  {len(df_stress)} days  ({zero_s} zero-sales days, {100*zero_s/len(df_stress):.0f}%)")

# ── Weather ────────────────────────────────────────────────────────────────────
def load_wx(wx_dir):
    wxf = glob.glob(str(wx_dir / "*.parquet")) + glob.glob(str(wx_dir / "*.snappy.parquet"))
    if not wxf:
        return pd.DataFrame(columns=["date"])
    wr = pd.read_parquet(wxf[0])
    wr["_date"] = pd.to_datetime(wr["Date_time"], errors="coerce").dt.normalize()
    drop = {"Date_time", "Location", "Address", "Conditions", "Icon",
            "Stations", "Wind_Chill", "Heat_Index", "_date"}
    nc = [c for c in wr.columns
          if c not in drop and wr[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    ag = {c: ("max" if "Maximum" in c else "min" if "Minimum" in c
              else "max" if "Wind_Gust" in c else "mean") for c in nc}
    wa = (wr.groupby("_date")[nc].agg(ag).reset_index()
          .rename(columns={"_date": "date", **{c: f"wx_{c}" for c in nc}}))
    return wa

WX_COLS = ["wx_Maximum_Temperature", "wx_Minimum_Temperature", "wx_Temperature",
           "wx_Precipitation", "wx_Snow_depth", "wx_Wind_Speed", "wx_Wind_Gust",
           "wx_Cloud_Cover", "wx_Relative_Humidity"]

def join_wx(df, wx):
    avail = [c for c in WX_COLS if c in wx.columns]
    if avail:
        df = df.merge(wx[["date"] + avail], on="date", how="left")
    for c in WX_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

df_primary = join_wx(df_primary, load_wx(PRIMARY_DIR / "weatherdata"))
df_stress  = join_wx(df_stress,  load_wx(STRESS_DIR  / "weatherdata"))

# ── Training history + DOW tables ─────────────────────────────────────────────
print("\n[2] Building DOW lookup tables...")
df_train = pd.read_csv(str(FE4_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df_train["_test_set"] = "train"

all_dates = pd.concat([
    df_train[["date", "bread_quantity", "_test_set"] + WX_COLS],
    pd.concat([df_primary, df_stress], ignore_index=True)
      .sort_values("date")[["date", "bread_quantity", "_test_set"] + WX_COLS]
], ignore_index=True).sort_values("date").reset_index(drop=True)

dow_idx  = {d: [] for d in range(7)}
dow_vals = {d: [] for d in range(7)}
for dt, bq, dw in zip(all_dates["date"].values,
                       all_dates["bread_quantity"].values,
                       pd.to_datetime(all_dates["date"].values).dayofweek):
    dow_idx[dw].append(pd.Timestamp(dt))
    dow_vals[dw].append(bq)

# ── Feature computation ────────────────────────────────────────────────────────
print("[3] Computing features...")
test_rows = all_dates[all_dates["_test_set"] != "train"].copy().reset_index(drop=True)

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

YEARS_EXT = range(2009, 2025)

def build_hols(yrs):
    h = set()
    for y in yrs:
        e = easter(y)
        h.update([datetime.date(y,1,1), datetime.date(y,5,1), datetime.date(y,7,11),
                  datetime.date(y,7,21), datetime.date(y,8,15), datetime.date(y,11,1),
                  datetime.date(y,11,11), datetime.date(y,12,25),
                  e+datetime.timedelta(1), e+datetime.timedelta(39), e+datetime.timedelta(50)])
    return h

HOLIDAYS = build_hols(YEARS_EXT)
dc = test_rows["date"].dt.date
test_rows["is_official_public_holiday"] = dc.apply(lambda d: int(d in HOLIDAYS))
test_rows["is_day_before_holiday"]      = dc.apply(lambda d: int((d + datetime.timedelta(1)) in HOLIDAYS))

def dun(d, h, cap=30):
    for i in range(cap + 1):
        if (d + datetime.timedelta(i)) in h:
            return i
    return cap

def dsl(d, h, cap=14):
    for i in range(cap + 1):
        if (d - datetime.timedelta(i)) in h:
            return i
    return cap

test_rows["days_until_next_holiday"] = dc.apply(lambda d: dun(d, HOLIDAYS))
test_rows["days_since_last_holiday"] = dc.apply(lambda d: dsl(d, HOLIDAYS))

def gm(d): return d - datetime.timedelta(days=d.weekday())
def in_herfst(d):  s=gm(datetime.date(d.year,11,1)); return s<=d<=s+datetime.timedelta(6)
def in_kerst(d):   s=gm(datetime.date(d.year,12,25)); return s<=d<=s+datetime.timedelta(13)
def in_krokus(d):  s=easter(d.year)-datetime.timedelta(48); return s<=d<=s+datetime.timedelta(6)
def in_paas(d):    s=easter(d.year)-datetime.timedelta(13); return s<=d<=s+datetime.timedelta(13)
def in_zomer(d):   return datetime.date(d.year,7,1)<=d<=datetime.date(d.year,8,31)

GF = set()
for yr, (s, e) in {2021:("2021-07-10","2021-07-19"),
                   2022:("2022-07-15","2022-07-24"),
                   2023:("2023-07-14","2023-07-23")}.items():
    dt = pd.Timestamp(s)
    while dt <= pd.Timestamp(e):
        GF.add(dt.date()); dt += pd.Timedelta(days=1)

ED = set()
for y in YEARS_EXT:
    E = easter(y)
    for delta in range(-2, 2):
        ED.add(E + datetime.timedelta(delta))

test_rows["is_herfstvakantie"]      = dc.apply(lambda d: int(in_herfst(d)))
test_rows["is_kerstvakantie"]       = dc.apply(lambda d: int(in_kerst(d)))
test_rows["is_krokusvakantie"]      = dc.apply(lambda d: int(in_krokus(d)))
test_rows["is_paasvakantie"]        = dc.apply(lambda d: int(in_paas(d)))
test_rows["is_zomervakantie"]       = dc.apply(lambda d: int(in_zomer(d)))
test_rows["is_gentse_feesten"]      = dc.apply(lambda d: int(d in GF))
test_rows["is_sinterklaas_period"]  = ((test_rows["month"] == 12) & (test_rows["date"].dt.day <= 6)).astype(int)
test_rows["is_easter_period"]       = dc.apply(lambda d: int(d in ED))
test_rows["is_covid_lockdown_period"] = 0
test_rows["is_hot_day"]    = (test_rows["wx_Temperature"] > 25).astype(int)
test_rows["is_heavy_rain"] = (test_rows["wx_Precipitation"] > 5).astype(int)
test_rows["comp_n_competitors_closed"] = 0.0
test_rows["comp_any_closed"]           = 0
test_rows["comp_closure_density"]      = 0.0
test_rows["comp_data_available"]       = 0

d2bq = dict(zip(pd.to_datetime(all_dates["date"]), all_dates["bread_quantity"]))
for lag in [1, 2, 3, 7, 14, 35, 63, 91]:
    test_rows[f"bread_lag_{lag}d"] = test_rows["date"].apply(
        lambda d: d2bq.get(d - pd.Timedelta(days=lag), np.nan))

def lk_sm(O, tdow, w):
    idx = dow_idx[tdow]
    pos = bisect_left(idx, O)
    if pos == 0: return np.nan
    lo = max(0, pos - w)
    return float(np.median(dow_vals[tdow][lo:pos]))

def lk_smn(O, tdow, w):
    idx = dow_idx[tdow]
    pos = bisect_left(idx, O)
    if pos == 0: return np.nan
    lo = max(0, pos - w)
    return float(np.mean(dow_vals[tdow][lo:pos]))

NAIVE_SPECS = {1:[("sm",4),("sm",13),("sm",26)], 7:[("sm",4),("sm",13),("sm",26)],
               14:[("sm",4),("sm",13),("sm",26)], 30:[("sm",4),("sm",26),("sm",8)],
               60:[("sm",26),("smn",52),("sm",8)], 90:[("sm",26),("smn",52),("smn",13)]}

for h in HORIZONS:
    for kind, w in NAIVE_SPECS[h]:
        col = f"naive_h{h}_seasonal_median_w{w}" if kind == "sm" else f"naive_h{h}_seasonal_mean_w{w}"
        vals = []
        for D in test_rows["date"]:
            O = D - pd.Timedelta(days=h)
            v = lk_sm(O, D.dayofweek, w) if kind == "sm" else lk_smn(O, D.dayofweek, w)
            vals.append(v)
        test_rows[col] = vals
        test_rows[col] = test_rows[col].ffill().bfill()

for h in HORIZONS:
    test_rows[f"bread_quantity_h{h}"] = test_rows["date"].apply(
        lambda d: d2bq.get(d + pd.Timedelta(days=h), np.nan))

nac = {1:"naive_h1_seasonal_median_w4", 7:"naive_h7_seasonal_median_w4",
       14:"naive_h14_seasonal_median_w4", 30:"naive_h30_seasonal_median_w4",
       60:"naive_h60_seasonal_median_w26", 90:"naive_h90_seasonal_median_w26"}
for h in HORIZONS:
    anc = test_rows[nac[h]].replace(0, np.nan)
    test_rows[f"ratio_h{h}"] = (test_rows[f"bread_quantity_h{h}"] / anc).clip(*RATIO_CLIP)

for h in HORIZONS:
    W = NAIVE_WINDOWS[h]
    av = []
    for D in test_rows["date"]:
        td = D + pd.Timedelta(days=h)
        tdow = td.dayofweek
        idx = dow_idx[tdow]
        pos = bisect_left(idx, D)
        if pos == 0:
            av.append(np.nan); continue
        st = max(0, pos - W)
        rc = [v for v in dow_vals[tdow][st:pos] if not np.isnan(v)]
        av.append(float(np.median(rc)) if rc else np.nan)
    test_rows[f"anchor_correct_h{h}"] = av
    ancs = pd.Series(av).replace(0, np.nan)
    test_rows[f"ratio_correct_h{h}"] = (test_rows[f"bread_quantity_h{h}"] / ancs).clip(*RATIO_CLIP)

df_tw = df_train[df_train["date"].dt.year >= 2015][["date", "wx_Temperature"]].dropna()
doy_stats = (df_tw.assign(doy=df_tw["date"].dt.dayofyear)
             .groupby("doy")["wx_Temperature"].agg(["mean", "std"])
             .rename(columns={"mean": "cm", "std": "cs"}))
gm_val = df_tw["wx_Temperature"].mean()
gs_val = df_tw["wx_Temperature"].std()
doy_stats["cs"] = doy_stats["cs"].fillna(gs_val).replace(0, gs_val)

def czs(row):
    if pd.isna(row["wx_Temperature"]): return 0.0
    doy = row["date"].dayofyear
    if doy not in doy_stats.index: return 0.0
    return (row["wx_Temperature"] - doy_stats.loc[doy, "cm"]) / doy_stats.loc[doy, "cs"]

test_rows["temperature_zscore_seasonal"] = test_rows.apply(czs, axis=1)
test_rows["_fold_1"] = None
test_rows["_fold_2"] = None
test_rows["_fold_3"] = None
test_rows["_split"]  = "test"

EXCL = {"date", "bread_quantity", "is_covid_lockdown_period", "day_of_week",
        "_split", "_fold_1", "_fold_2", "_fold_3", "_test_set"}

def get_feats(fe_csv, tdf):
    ref = pd.read_csv(fe_csv, nrows=1)
    pool = [c for c in ref.columns
            if c not in EXCL
            and not c.startswith("bread_quantity_h") and not c.startswith("ratio_h")
            and not c.startswith("ratio_correct_") and not c.startswith("anchor_correct")
            and ref[c].dtype != object]
    return [c for c in pool if c in tdf.columns]

feats4 = get_feats(FE4_CSV, test_rows)
feats5 = get_feats(FE5_CSV, test_rows)

# ── Load models ────────────────────────────────────────────────────────────────
print("[4] Loading models...")
MODELS = {}
for h in HORIZONS:
    MODELS[h] = pickle.load(open(FINAL_DIR / f"lgbm_h{h}_final.pkl", "rb"))

MC = {
    1:  {"feats": feats4, "tgt": "bread_quantity_h1",  "anc": "anchor_correct_h1",  "ratio": True},
    7:  {"feats": feats5, "tgt": "bread_quantity_h7",  "anc": "naive_h7_seasonal_median_w4", "ratio": True},
    14: {"feats": feats4, "tgt": "bread_quantity_h14", "anc": None, "ratio": False},
    30: {"feats": feats5, "tgt": "bread_quantity_h30", "anc": "anchor_correct_h30", "ratio": True},
    60: {"feats": feats4, "tgt": "bread_quantity_h60", "anc": "anchor_correct_h60", "ratio": True},
    90: {"feats": feats4, "tgt": "bread_quantity_h90", "anc": None, "ratio": False},
}
VAL_MAE = {1: 29.835, 7: 31.646, 14: 34.782, 30: 36.630, 60: 36.229, 90: 56.487}


# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(df, open_only=False):
    results = {}
    for h in HORIZONS:
        cfg  = MC[h]
        tgt  = cfg["tgt"]
        mask = df[tgt].notna()
        if open_only:
            mask = mask & (df[tgt] > 0)
        if mask.sum() == 0:
            results[h] = None; continue
        X = df.loc[mask, cfg["feats"]].fillna(0).values
        y = df.loc[mask, tgt].values
        preds = MODELS[h].predict(X)
        if cfg["ratio"] and cfg["anc"]:
            a = df.loc[mask, cfg["anc"]].values
            preds = np.clip(preds, *RATIO_CLIP) * a
        mae = float(np.mean(np.abs(preds - y)))
        nv  = df.loc[mask, f"anchor_correct_h{h}"].values
        naive_mae = float(np.mean(np.abs(nv - y)))
        results[h] = {"mae": mae, "naive_mae": naive_mae,
                      "n": int(mask.sum()), "beats": mae < naive_mae}
    return results

primary_rows = test_rows[test_rows["_test_set"] == "primary"].copy()
stress_rows  = test_rows[test_rows["_test_set"] == "stress"].copy()

rp       = evaluate(primary_rows, open_only=False)
rs_all   = evaluate(stress_rows,  open_only=False)
rs_open  = evaluate(stress_rows,  open_only=True)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("FINAL RESULTS — Corrected naive (same DOW as target)")
print(f"\n{'h':>3}  {'Val':>6}  "
      f"{'Prim MAE':>9}  {'P.Naive':>8}  "
      f"{'Stress(all)':>12}  {'S.Naive(all)':>13}  "
      f"{'Stress(open)':>13}  {'S.Naive(open)':>14}  {'n_open':>7}")
print("-"*100)
for h in HORIZONS:
    pp  = rp[h];    sa = rs_all[h];   so = rs_open[h]
    pm  = f"{pp['mae']:.1f}"      if pp else "N/A"
    pn  = f"{pp['naive_mae']:.1f}" if pp else "N/A"
    sam = f"{sa['mae']:.1f}"      if sa else "N/A"
    san = f"{sa['naive_mae']:.1f}" if sa else "N/A"
    som = f"{so['mae']:.1f}"      if so else "N/A"
    son = f"{so['naive_mae']:.1f}" if so else "N/A"
    no  = str(so['n'])            if so else "N/A"
    print(f"  {h:>2}  {VAL_MAE[h]:>6.1f}  "
          f"{pm:>9}  {pn:>8}  "
          f"{sam:>12}  {san:>13}  "
          f"{som:>13}  {son:>14}  {no:>7}")

print()
print(f"  Primary open days: {rp[1]['n'] if rp[1] else '?'} / {len(primary_rows)}")
print(f"  Stress  open days: {rs_open[1]['n'] if rs_open[1] else '?'} / {len(stress_rows)} "
      f"({100*(rs_open[1]['n'] if rs_open[1] else 0)/len(stress_rows):.0f}%)")
print()
print("NOTE: 'open days' = target date has bread_quantity > 0")
print("      Zero-sales days are closed-store events, not a forecasting problem")

# ── Save ───────────────────────────────────────────────────────────────────────
out = {
    "iteration": 29,
    "type": "test_evaluation_open_days",
    "timestamp": "2026-09-09",
    "naive_definition": "seasonal median, last W same-DOW-as-target weeks, causal from origin D",
    "naive_windows": NAIVE_WINDOWS,
    "zero_sales_days": {
        "primary": int(zero_p), "primary_total": len(df_primary),
        "stress":  int(zero_s), "stress_total":  len(df_stress),
    },
    "results": {
        "primary_all":   {str(h): rp[h]     for h in HORIZONS},
        "stress_all":    {str(h): rs_all[h]  for h in HORIZONS},
        "stress_open":   {str(h): rs_open[h] for h in HORIZONS},
    },
    "val_mae_reference": {str(h): VAL_MAE[h] for h in HORIZONS},
}
with open(HIST_OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {HIST_OUT.name}")
print("=== Iteration 029 complete ===")
