"""Rebuild the agentic run's feature frame over an arbitrary set of data roots.

The frozen boosters in run_20260904T140437Z are a function of 37 columns that were
produced by three scripts chained together (iteration_001 EDA -> iteration_002
preprocess -> iteration_003 feature engineering), each hard-wired to the training
partition. Scoring those boosters on the holdout needs the same 37 columns computed
over train + holdout together, because a lag of 364 rows at a 2021 origin reaches
back into 2020.

This module reproduces that chain. It is not a re-design: every rule below is copied
from the run's own code, and `verify` asserts that running it on the training
partition alone reproduces the run's `iteration_003_feature_engineering.csv`
column-for-column. Three departures are deliberate and are recorded in
DECLARED_DEPARTURES so they end up in the report rather than in a footnote:

  1. POS filter is applied BEFORE de-duplication. The run picks the newest export
     per date and then discards non-001 tills. That was equivalent on training data,
     where POS 001 is the only till. From 2021 a second till (POS ending 301) files
     its own exports, and on twelve 2021 dates its export is the newest -- verbatim
     replay would select it, fail the POS check, and drop the date entirely.
  2. Belgian public holidays are computed for 2010-2024 rather than the run's
     hard-coded range(2010, 2021). Same algorithm, more years. Replayed verbatim the
     flag is identically zero across the whole holdout.
  3. Weather and competitor imputation statistics stay pinned to the run's own train
     rows (date <= 2018-10-04). Recomputing them over holdout rows would let the
     holdout inform the model's inputs.
"""

from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RUN_DIR = r"C:\Users\selin\OneDrive\Desktop\All\KU Leuven\internship-project\General Plan\runs\run_20260904T140437Z"
TRAIN_ROOT = r"C:/Users/selin/OneDrive/Desktop/alldata/aernoudt_split/train"
HOLDOUT_ROOT = r"C:/Users/selin/OneDrive/Desktop/HOLDOUT_DO_NOT_GIVE_TO_PIPELINE"
PRIMARY_ROOT = os.path.join(HOLDOUT_ROOT, "holdout_primary")
STRESS_ROOT = os.path.join(HOLDOUT_ROOT, "holdout_stress")

TARGET = "bread_quantity"
HORIZONS = [1, 7, 14, 30, 60, 90]
REGIME_CHANGE = pd.Timestamp("2020-06-01")
LAST_TRAIN_ORIGIN = pd.Timestamp("2018-10-04")
FIRST_VAL_ORIGIN = pd.Timestamp("2019-01-02")

DECLARED_DEPARTURES = [
    {
        "id": "pos_before_dedup",
        "what": "POS 001 is selected before the newest-export-per-date rule, not after.",
        "why": "The run's order is equivalent on training data (POS 001 is the only till) "
               "but drops 12 dates in 2021 where the second till's export is newer.",
        "affects": "holdout_primary, holdout_stress",
    },
    {
        "id": "holiday_years",
        "what": "Belgian public holidays computed for 2010-2024 instead of the run's range(2010, 2021).",
        "why": "Same Gregorian-Easter algorithm, more years. Verbatim replay flags zero holidays "
               "anywhere in the holdout, which would misprice every holiday closure.",
        "affects": "holdout_primary, holdout_stress",
    },
    {
        "id": "imputation_pinned_to_train",
        "what": "Weather month-medians, competitor fills and lag/rolling NaN fills stay fitted on "
                "the run's own train rows (date <= 2018-10-04).",
        "why": "Refitting them over holdout rows would let holdout values reach the model's inputs.",
        "affects": "all",
    },
]

WX_COLS = [
    "wx_Maximum_Temperature", "wx_Minimum_Temperature", "wx_Temperature", "wx_Wind_Chill",
    "wx_Heat_Index", "wx_Precipitation", "wx_Snow_depth", "wx_Wind_Speed", "wx_Wind_Gust",
    "wx_Cloud_Cover", "wx_Relative_Humidity", "wx_Forecast",
]
COMP_COLS = [
    "comp_verlof_concurrent_1", "comp_verlof_concurrent_2", "comp_verlof_concurrent_3",
    "comp_verlof_concurrent_4", "comp_verlof_concurrent_5", "comp_n_holidays", "comp_any_holiday",
]
FEATURES = [
    "year", "doy", *WX_COLS, *COMP_COLS,
    "bq_lag1", "bq_lag7", "bq_lag14", "bq_lag28", "bq_lag30", "bq_lag91", "bq_lag364",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "is_weekend", "is_public_holiday", "year_centered",
]

LAGS = [1, 7, 14, 28, 30, 91, 364]
ROLLS = [(7, 4), (30, 15), (90, 45)]


# --------------------------------------------------------------------------- sales
def _article_sets(root):
    """argcod=1 is bread, argcod=3 is pistolets. Copied from iteration_001 step 1."""
    path = os.path.join(root, "productdata", "productdata.csv")
    prod = pd.read_csv(path, sep="|", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    prod.columns = [c.lstrip("\ufeff").strip() for c in prod.columns]
    def codes(argcod):
        sel = prod[prod["argcod"].astype(str).str.strip() == argcod]["artcod"]
        return set(sel.astype(str).str.strip().str.zfill(8))
    return codes("1"), codes("3")


def _parse_xml(fpath, bread_set, pist_set):
    """Dual-schema XML reader, copied verbatim from iteration_001 step 3."""
    root = ET.parse(fpath).getroot()
    if root.get("referencepointofsales", "") not in ("001", ""):
        return None
    use_new = len(root.findall("sale")) > 0
    txn_tag, det_tag, pay_tag = (
        ("sale", "saledetail", "salepayment") if use_new else ("order", "orderdetail", "orderpayment")
    )
    r = dict(date=root.get("salesdate", ""), bread_qty=0.0, pist_qty=0.0, total_qty=0.0,
             total_rev=0.0, bread_rev=0.0, num_txn=0, had_red=0, pm1=0.0, pm2=0.0, pm4=0.0)
    for txn in root.findall(txn_tag):
        r["num_txn"] += 1
        try:
            r["total_rev"] += float(txn.get("total", "0").replace(",", "."))
        except Exception:
            pass
        for det in txn.findall(det_tag):
            art = det.get("articlereference", "").strip().zfill(8)
            try:
                qty = float(det.get("quantity", "0").replace(",", "."))
            except Exception:
                qty = 0.0
            if det.get("detailtype", "0") != "0":
                r["had_red"] = 1
            r["total_qty"] += qty
            if art in bread_set:
                r["bread_qty"] += qty
                try:
                    r["bread_rev"] += float(det.get("total", "0").replace(",", "."))
                except Exception:
                    pass
            elif art in pist_set:
                r["pist_qty"] += qty
        for pay in txn.findall(pay_tag):
            try:
                amt = float(pay.get("total", "0").replace(",", "."))
            except Exception:
                amt = 0.0
            pm = pay.get("paymentmode", "0")
            if pm == "1":
                r["pm1"] += amt
            elif pm == "2":
                r["pm2"] += amt
            elif pm == "4":
                r["pm4"] += amt
    return r


def load_sales(roots, pos_first=True):
    """Daily sales table across every root. Newest export per date wins.

    `pos_first=False` reproduces the run's own order (dedup, then POS check) and is
    only used to measure what departure 1 costs.
    """
    bread_set, pist_set = _article_sets(roots[0])
    by_date = defaultdict(list)
    stats = {"files_seen": 0, "superseded_dropped": 0, "dropped_wrong_pos": 0}
    for root in roots:
        for fpath in glob.glob(os.path.join(root, "salesdata", "raw_xml", "01", "*", "*.xml")):
            m = re.match(r"PT_Sales_(\d+)_(\d{8})_(\d{14})\.xml", os.path.basename(fpath))
            if not m:
                continue
            stats["files_seen"] += 1
            pos = m.group(1)[-3:]
            if pos_first and pos != "001":
                stats["dropped_wrong_pos"] += 1
                continue
            by_date[m.group(2)].append((m.group(3), fpath))

    records = []
    stats["unparseable"] = []
    for sd in sorted(by_date):
        cand = sorted(by_date[sd], reverse=True)
        stats["superseded_dropped"] += len(cand) - 1
        try:
            rec = _parse_xml(cand[0][1], bread_set, pist_set)
        except Exception as exc:
            # The run's iteration_001 wraps this in a bare `except: pass`, so a malformed
            # export costs it the whole date. Same outcome here, but counted and named:
            # 2021-11-02's only export is truncated mid-file, so the date has no recoverable
            # total. Parsing the intact prefix would under-count bread and bias the target
            # low, which is worse than not scoring the date at all.
            stats["unparseable"].append({"date": sd, "file": os.path.basename(cand[0][1]),
                                         "error": str(exc)[:80]})
            continue
        if rec is None:
            stats["dropped_wrong_pos"] += 1
            continue
        records.append(rec)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(columns={
        "bread_qty": TARGET, "pist_qty": "pistolet_quantity", "total_qty": "total_quantity",
        "total_rev": "total_revenue", "bread_rev": "bread_revenue", "num_txn": "num_orders",
        "had_red": "had_reduction", "pm1": "payment_mode_1", "pm2": "payment_mode_2",
        "pm4": "payment_mode_4",
    })
    stats["dates"] = len(df)
    return df, stats


# ------------------------------------------------------------------------- exogenous
def load_weather(roots):
    frames = []
    for root in roots:
        for path in glob.glob(os.path.join(root, "weatherdata", "*.parquet")):
            frames.append(pq.read_table(path).to_pandas())
    raw = pd.concat(frames, ignore_index=True)
    dt_col = next(c for c in raw.columns if "date" in c.lower() or "time" in c.lower())
    raw[dt_col] = pd.to_datetime(raw[dt_col], errors="coerce")
    raw["date"] = raw[dt_col].dt.normalize()
    num_cols = [c for c in raw.columns if c not in (dt_col, "date") and raw[c].dtype != object]
    agg = {}
    for col in num_cols:
        lc = col.lower()
        if "max" in lc:
            agg[col] = "max"
        elif "min" in lc:
            agg[col] = "min"
        elif "precip" in lc or "snow" in lc:
            agg[col] = "sum"
        else:
            agg[col] = "mean"
    daily = raw.groupby("date")[num_cols].agg(agg).reset_index()
    daily.columns = ["date"] + [f"wx_{c}" for c in num_cols]
    return daily.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_competitor(roots):
    frames = []
    for root in roots:
        path = os.path.join(root, "competitordata", "VerlofConcurrent.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, sep=";", encoding="latin-1", on_bad_lines="skip"))
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = raw.columns.str.strip()
    date_col = [c for c in raw.columns if "date" in c.lower()][0]
    raw["date"] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["date"])
    verlof = [c for c in raw.columns if "verlof" in c.lower()]
    agg = raw.groupby("date")[verlof].max().reset_index()
    agg[verlof] = agg[verlof].fillna(0).clip(0, 1)
    agg["comp_n_holidays"] = agg[verlof].sum(axis=1)
    agg["comp_any_holiday"] = (agg["comp_n_holidays"] > 0).astype(int)
    return agg.rename(columns={c: f"comp_{c}" for c in verlof})


# -------------------------------------------------------------------------- calendar
def _easter(year):
    a, b, c = year % 19, year // 100, year % 100
    d, e, f = b // 4, b % 4, (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    return pd.Timestamp(year=year, month=(h + l - 7 * m + 114) // 31,
                        day=((h + l - 7 * m + 114) % 31) + 1)


def belgian_holidays(year_from=2010, year_to=2024):
    out = set()
    for year in range(year_from, year_to + 1):
        e = _easter(year)
        out.update([
            pd.Timestamp(year, 1, 1), pd.Timestamp(year, 5, 1), pd.Timestamp(year, 7, 21),
            pd.Timestamp(year, 8, 15), pd.Timestamp(year, 11, 1), pd.Timestamp(year, 11, 11),
            pd.Timestamp(year, 12, 25),
            e + pd.Timedelta(days=1), e + pd.Timedelta(days=39), e + pd.Timedelta(days=50),
        ])
    return out


# ------------------------------------------------------------------------------ build
def build(roots, holiday_year_to=2024, pos_first=True):
    """Full chain: sales -> trading filter -> joins -> imputation -> features."""
    df, sales_stats = load_sales(roots, pos_first=pos_first)

    # iteration_001 step 4: trading-day filter
    df["day_of_week"] = df["date"].dt.dayofweek
    closed_monday = (df["day_of_week"] == 0) & (df["date"] < REGIME_CHANGE)
    df = df[~closed_monday].copy().reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["doy"] = df["date"].dt.dayofyear

    # iteration_002 steps 0b/0c: exogenous joins
    df = df.merge(load_weather(roots), on="date", how="left")
    df = df.merge(load_competitor(roots), on="date", how="left")

    # iteration_002 step 3: the run's own split labels, extended with "holdout"
    def label(d):
        if d <= LAST_TRAIN_ORIGIN:
            return "train"
        if d < FIRST_VAL_ORIGIN:
            return "gap"
        if d <= pd.Timestamp("2020-12-31"):
            return "validation"
        return "holdout"
    df["_split"] = df["date"].apply(label)
    train_mask = df["_split"] == "train"

    # iteration_002 steps 5a/5b: imputation, statistics fitted on train rows only
    wx_cols = [c for c in df.columns if c.startswith("wx_")]
    for col in wx_cols:
        month_med = df[train_mask][["month", col]].dropna().groupby("month")[col].median()
        null_mask = df[col].isnull()
        if null_mask.any():
            df.loc[null_mask, col] = df.loc[null_mask, "month"].map(month_med)
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[train_mask][col].median())
    comp_cols = [c for c in df.columns if c.startswith("comp_")]
    for col in comp_cols:
        if df[col].isnull().sum() == 0:
            continue
        if "verlof" in col or "holiday" in col:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(df[train_mask][col].dropna().median())

    # iteration_003 steps 3a/3b: lags and rolling means, row-local, train-mean NaN fill
    for lag in LAGS:
        col = f"bq_lag{lag}"
        df[col] = df[TARGET].shift(lag)
        fill = df.loc[train_mask & df[col].notna(), col].mean()
        if np.isnan(fill):
            fill = df.loc[train_mask, TARGET].mean()
        df[col] = df[col].fillna(fill)
    for window, minp in ROLLS:
        col = f"bq_roll{window}_mean"
        df[col] = df[TARGET].rolling(window, min_periods=minp).mean()
        fill = df.loc[train_mask & df[col].notna(), col].mean()
        if np.isnan(fill):
            fill = df.loc[train_mask, TARGET].mean()
        df[col] = df[col].fillna(fill)

    # iteration_003 steps 3c-3f: calendar, holiday, trend
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_public_holiday"] = df["date"].isin(belgian_holidays(2010, holiday_year_to)).astype(int)
    df["year_centered"] = df["year"] - 2015

    # iteration_003 step 7: wide per-horizon targets (row offsets, as the run trained them)
    for h in HORIZONS:
        df[f"{TARGET}_h{h}"] = df[TARGET].shift(-h)

    df.attrs["sales_stats"] = sales_stats
    return df


# ----------------------------------------------------------------------- verification
def verify(tol=1e-9):
    """Assert the replay reproduces the run's own feature table on the train partition."""
    ref = pd.read_csv(os.path.join(RUN_DIR, "data", "iteration_003_feature_engineering.csv"),
                      parse_dates=["date"])
    got = build([TRAIN_ROOT], holiday_year_to=2020, pos_first=False)
    got = got[got["date"] <= pd.Timestamp("2020-12-31")].reset_index(drop=True)

    report = {"ref_rows": len(ref), "got_rows": len(got), "columns": {}}
    assert len(ref) == len(got), f"row count {len(ref)} != {len(got)}"
    assert (ref["date"].values == got["date"].values).all(), "date index differs"

    worst = 0.0
    for col in FEATURES + [TARGET] + [f"{TARGET}_h{h}" for h in HORIZONS]:
        a, b = ref[col].to_numpy(dtype=float), got[col].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.where(both_nan, 0.0, np.abs(a - b))
        mx = float(np.nanmax(diff)) if len(diff) else 0.0
        report["columns"][col] = mx
        worst = max(worst, mx)
    report["max_abs_diff"] = worst
    report["exact"] = bool(worst <= tol)
    return report


if __name__ == "__main__":
    import json
    rep = verify()
    print(json.dumps({k: v for k, v in rep.items() if k != "columns"}, indent=2))
    bad = {k: v for k, v in rep["columns"].items() if v > 1e-9}
    print("columns differing:", json.dumps(bad, indent=2) if bad else "none")
