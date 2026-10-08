# %% [iteration_019_feature_engineering_4] FE4 — run_20260908T075709Z
# Adds missing domain-recommended features to the iter012 FE CSV:
#   - Official Belgian/Flemish public holidays (binary + day-before)
#   - Flemish school holiday periods (herfst/kerst/krokus/paas/zomer)
#   - Gentse Feesten (exact dates from domain research iter002)
#   - Short lag features: bread_lag_1d, bread_lag_2d, bread_lag_3d
#
# Motivation: iter002 domain research recommended all of these; none made it
# into the FE CSV. Herfstvakantie and Kerstvakantie are directly in fold-3 val
# (Sep-Dec 2020), as are official holidays Nov 1, Nov 11, Dec 25.
# Short lags give the model recent demand signal missing below the 7d floor.

import json
import datetime
import numpy as np
import pandas as pd
from dateutil.easter import easter
from pathlib import Path

RUN_DIR = Path(__file__).parent.parent
FE_IN = RUN_DIR / "data" / "iteration_012_feature_engineering.csv"
FE_OUT = RUN_DIR / "data" / "iteration_019_feature_engineering.csv"
HISTORY_OUT = RUN_DIR / "history" / "iteration_019_feature_engineering_4.json"
STATE_FILE = RUN_DIR / "state" / "current.json"

print("=== Iteration 019: Feature Engineering 4 ===")

df = pd.read_csv(FE_IN, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols, {df['date'].min().date()} to {df['date'].max().date()}")

YEARS = range(2009, 2022)


# ── 1. OFFICIAL BELGIAN/FLEMISH PUBLIC HOLIDAYS ────────────────────────────

def _get_monday(d):
    return d - datetime.timedelta(days=d.weekday())

def build_holiday_set(years):
    h = set()
    for y in years:
        e = easter(y)
        fixed = [
            datetime.date(y, 1, 1),    # New Year
            datetime.date(y, 5, 1),    # Labour Day
            datetime.date(y, 7, 11),   # Flemish Community Day
            datetime.date(y, 7, 21),   # Belgian National Day
            datetime.date(y, 8, 15),   # Assumption
            datetime.date(y, 11, 1),   # All Saints
            datetime.date(y, 11, 11),  # Armistice
            datetime.date(y, 12, 25),  # Christmas
        ]
        floating = [
            e + datetime.timedelta(1),   # Easter Monday
            e + datetime.timedelta(39),  # Ascension Thursday
            e + datetime.timedelta(50),  # Whit Monday
        ]
        h.update(fixed + floating)
    return h

HOLIDAYS = build_holiday_set(YEARS)

date_col = df["date"].dt.date
df["is_official_public_holiday"] = date_col.apply(lambda d: int(d in HOLIDAYS))
df["is_day_before_holiday"] = date_col.apply(lambda d: int((d + datetime.timedelta(1)) in HOLIDAYS))

n_hol = df["is_official_public_holiday"].sum()
n_dbh = df["is_day_before_holiday"].sum()
print(f"  is_official_public_holiday: {n_hol} rows")
print(f"  is_day_before_holiday: {n_dbh} rows")


# ── 2. FLEMISH SCHOOL HOLIDAYS ──────────────────────────────────────────────

def build_herfstvakantie(years):
    """Monday of the week containing Nov 1, for 7 days."""
    periods = []
    for y in years:
        nov1 = datetime.date(y, 11, 1)
        start = _get_monday(nov1)
        periods.append((start, start + datetime.timedelta(6)))
    return periods

def build_kerstvakantie(years):
    """Monday of the week containing Dec 25, for ~13 days (through ~Jan 6)."""
    periods = []
    for y in years:
        dec25 = datetime.date(y, 12, 25)
        start = _get_monday(dec25)
        end = start + datetime.timedelta(13)
        periods.append((start, end))
    return periods

def build_krokusvakantie(years):
    """7th Monday before Easter Sunday, for 7 days. Easter-48 = 7th Monday before."""
    periods = []
    for y in years:
        e = easter(y)
        start = e - datetime.timedelta(48)
        if start.weekday() != 0:
            start = _get_monday(start)
        periods.append((start, start + datetime.timedelta(6)))
    return periods

def build_paasvakantie(years):
    """2-week spring break roughly centered on Easter; starts Monday 2 weeks before Easter Monday."""
    periods = []
    for y in years:
        e = easter(y)
        easter_mon = e + datetime.timedelta(1)
        start = _get_monday(easter_mon) - datetime.timedelta(weeks=2)
        periods.append((start, start + datetime.timedelta(13)))
    return periods

def is_in_periods(d, periods):
    for start, end in periods:
        if start <= d <= end:
            return 1
    return 0

herfst_periods = build_herfstvakantie(YEARS)
kerst_periods = build_kerstvakantie(YEARS)
krokus_periods = build_krokusvakantie(YEARS)
paas_periods = build_paasvakantie(YEARS)

df["is_herfstvakantie"] = date_col.apply(lambda d: is_in_periods(d, herfst_periods))
df["is_kerstvakantie"]  = date_col.apply(lambda d: is_in_periods(d, kerst_periods))
df["is_krokusvakantie"] = date_col.apply(lambda d: is_in_periods(d, krokus_periods))
df["is_paasvakantie"]   = date_col.apply(lambda d: is_in_periods(d, paas_periods))
df["is_zomervakantie"]  = ((df["date"].dt.month >= 7) & (df["date"].dt.month <= 8)).astype(int)

print(f"  is_herfstvakantie: {df['is_herfstvakantie'].sum()} rows")
print(f"  is_kerstvakantie:  {df['is_kerstvakantie'].sum()} rows")
print(f"  is_krokusvakantie: {df['is_krokusvakantie'].sum()} rows")
print(f"  is_paasvakantie:   {df['is_paasvakantie'].sum()} rows")
print(f"  is_zomervakantie:  {df['is_zomervakantie'].sum()} rows")

# Verify herfstvakantie 2020 in fold-3 val (Oct 26 - Nov 1)
val3 = df[df["_fold_3"] == "val"]
h20_in_val = val3[val3["is_herfstvakantie"] == 1]
print(f"  Herfstvakantie rows in fold-3 val: {len(h20_in_val)}")
k20_in_val = val3[val3["is_kerstvakantie"] == 1]
print(f"  Kerstvakantie rows in fold-3 val: {len(k20_in_val)}")
hol_in_val = val3[val3["is_official_public_holiday"] == 1]
print(f"  Official holiday rows in fold-3 val: {len(hol_in_val)} dates: {hol_in_val['date'].dt.date.tolist()}")


# ── 3. GENTSE FEESTEN ──────────────────────────────────────────────────────
# Exact dates from iter002 domain research content_carried_forward
GENTSE_DATES = {
    2010: ("2010-07-09", "2010-07-18"),
    2011: ("2011-07-15", "2011-07-24"),
    2012: ("2012-07-13", "2012-07-22"),
    2013: ("2013-07-12", "2013-07-21"),
    2014: ("2014-07-11", "2014-07-20"),
    2015: ("2015-07-17", "2015-07-26"),
    2016: ("2016-07-15", "2016-07-24"),
    2017: ("2017-07-14", "2017-07-23"),
    2018: ("2018-07-13", "2018-07-22"),
    2019: ("2019-07-12", "2019-07-21"),
    # 2020: cancelled (COVID)
}
gentse_periods = [
    (datetime.date.fromisoformat(s), datetime.date.fromisoformat(e))
    for s, e in GENTSE_DATES.values()
]
df["is_gentse_feesten"] = date_col.apply(lambda d: is_in_periods(d, gentse_periods))
print(f"  is_gentse_feesten: {df['is_gentse_feesten'].sum()} rows")


# ── 4. SHORT LAG FEATURES ──────────────────────────────────────────────────
# Use same calendar-day lookup as existing lags (date_to_val dict).
# NaN (e.g., Sundays not in data) are ffill/bfill.
ts = df.set_index("date")["bread_quantity"]
date_to_val = ts.to_dict()

for lag in [1, 2, 3]:
    col = f"bread_lag_{lag}d"
    df[col] = df["date"].apply(lambda d: date_to_val.get(d - pd.Timedelta(days=lag), np.nan))
    n_nan = df[col].isna().sum()
    df[col] = df[col].ffill().bfill()
    print(f"  {col}: {n_nan} NaN filled")


# ── 5. SANITY CHECKS ───────────────────────────────────────────────────────
print("\n=== Sanity checks ===")
# Check lag_1d — should match bread_quantity at D-1 for most rows
mask = (df["date"].dt.dayofweek != 0)  # skip Mondays (D-1 is Sunday, no data)
expected_match = df[mask].head(20).apply(
    lambda r: abs(r["bread_lag_1d"] - date_to_val.get(r["date"] - pd.Timedelta(1), r["bread_lag_1d"])) < 0.01,
    axis=1
).all()
print(f"  bread_lag_1d correct for non-Monday rows (sample): {expected_match}")

# Check official holiday Nov 11, 2020 in fold-3 val
nov11 = df[df["date"] == "2020-11-11"]
if len(nov11) > 0:
    print(f"  Nov 11, 2020: is_official_public_holiday={nov11['is_official_public_holiday'].values[0]}, is_day_before_holiday={nov11['is_day_before_holiday'].values[0]}")

# Check Dec 25, 2020
dec25 = df[df["date"] == "2020-12-25"]
if len(dec25) > 0:
    print(f"  Dec 25, 2020: is_official_public_holiday={dec25['is_official_public_holiday'].values[0]}")

new_features = [
    "is_official_public_holiday", "is_day_before_holiday",
    "is_herfstvakantie", "is_kerstvakantie", "is_krokusvakantie",
    "is_paasvakantie", "is_zomervakantie", "is_gentse_feesten",
    "bread_lag_1d", "bread_lag_2d", "bread_lag_3d",
]
print(f"\n  New features added: {len(new_features)}")
print(f"  Final shape: {df.shape}")
assert df.shape[1] == 79 + len(new_features), f"Expected {79 + len(new_features)} cols, got {df.shape[1]}"
assert not df[new_features].isnull().any().any(), "NaN in new features"

df.to_csv(FE_OUT, index=False)
print(f"\nSaved: {FE_OUT}")


# ── 6. WRITE HANDOFF JSON ─────────────────────────────────────────────────
handoff = {
    "iteration": 19,
    "node": "feature_engineering",
    "timestamp": "2026-09-09",
    "schema_version": "2026-09-04",
    "type": "feature_engineering_4",
    "summary": (
        "FE iteration 4. Added 11 new features from iter002 domain research that were never "
        "included: official Belgian/Flemish public holidays (is_official_public_holiday, "
        "is_day_before_holiday), all Flemish school holiday periods (herfstvakantie, "
        "kerstvakantie, krokusvakantie, paasvakantie, zomervakantie), Gentse Feesten, "
        "and short lag features (bread_lag_1d, bread_lag_2d, bread_lag_3d). "
        "Output: iteration_019_feature_engineering.csv (2897 x 90 cols). "
        "Herfstvakantie, Kerstvakantie, and official holidays appear directly in the "
        "fold-3 val period (Sep-Dec 2020), making them immediately testable."
    ),
    "new_features": new_features,
    "val_period_coverage": {
        "fold3_val_herfstvakantie_rows": int(len(h20_in_val)),
        "fold3_val_kerstvakantie_rows": int(len(k20_in_val)),
        "fold3_val_official_holiday_rows": int(len(hol_in_val)),
        "note": "Herfstvakantie and Kerstvakantie periods fall directly in all 3 fold val periods (Sep-Dec). Official holidays Nov 1, Nov 11, Dec 25 all appear in fold-3 val."
    },
    "output_csv": str(FE_OUT.name),
    "handoff": {
        "written_by": "feature_engineering",
        "iteration": 19,
        "context_clearable": True,
        "read_this_first": (
            "New FE CSV is iteration_019_feature_engineering.csv with 90 columns. "
            "All models in iter020 must use this new CSV. Key new features: "
            "is_official_public_holiday, is_herfstvakantie, is_kerstvakantie, "
            "bread_lag_1d/2d/3d. These appear in fold-3 val so they are directly testable."
        ),
        "parameters": {
            "fe_csv": "data/iteration_019_feature_engineering.csv",
            "new_feature_count": 11,
            "total_cols": 90,
        },
        "durable_rules": [
            {
                "rule": "All training from iter020 onward uses iteration_019_feature_engineering.csv (90 cols), not iter012 (79 cols).",
                "why": "11 domain-recommended features added, including holiday and school break features directly in the val period.",
                "applies_to": ["training", "end_model"],
            }
        ],
        "do_not_repeat": [],
        "content_carried_forward": {
            "fe_csv_path": "data/iteration_019_feature_engineering.csv",
            "new_features": new_features,
        },
    },
}

with open(HISTORY_OUT, "w") as f:
    json.dump(handoff, f, indent=2)
print(f"Handoff written: {HISTORY_OUT}")

# ── 7. UPDATE STATE ────────────────────────────────────────────────────────
with open(STATE_FILE, encoding="utf-8") as f:
    state = json.load(f)
state["iteration"] = 19
state["next_iteration_contract"] = {
    "node": "training",
    "iteration": 20,
    "read_first": ["history/iteration_019_feature_engineering_4.json"],
    "must_use_csv": "data/iteration_019_feature_engineering.csv",
    "tasks": [
        "Re-train h=1 with new features + Optuna (75 trials) — first Optuna run for h=1",
        "Re-train h=7 with new features, keep existing Optuna config",
        "Re-train h=14 with new features, keep raw-units config",
        "Re-train h=30 with new features + Optuna (75 trials) — first Optuna run for h=30",
        "Re-train h=60 with new features, evaluate on ALL val rows vs 40.042 floor",
        "Re-train h=90 with new features, multi-fold evaluation",
    ],
}
with open(STATE_FILE, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
print("State updated.")
print("\n=== Iteration 019 complete ===")
