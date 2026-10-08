# %% [iteration_023_feature_engineering_5] FE5 — run_20260908T075709Z
#
# Adds the two remaining documented gaps from iter002 domain research:
#   1. days_since_last_holiday (capped at 14) — post-holiday demand suppression
#      "Day AFTER a public holiday occasionally shows mild demand suppression"
#   2. temperature_zscore_seasonal — temperature deviation from same-DOY climatological
#      average (removes collinearity with month/season features; pre-2015 = 0.0)
#
# Input:  data/iteration_019_feature_engineering.csv (2897 x 90)
# Output: data/iteration_023_feature_engineering.csv (2897 x 92)

import json
import datetime
import numpy as np
import pandas as pd
from dateutil.easter import easter
from pathlib import Path

RUN_DIR  = Path(__file__).parent.parent
FE_IN    = RUN_DIR / "data" / "iteration_019_feature_engineering.csv"
FE_OUT   = RUN_DIR / "data" / "iteration_023_feature_engineering.csv"
HIST_OUT = RUN_DIR / "history" / "iteration_023_feature_engineering_5.json"
STATE_F  = RUN_DIR / "state" / "current.json"

print("=== Iteration 023: Feature Engineering 5 ===")
df = pd.read_csv(str(FE_IN), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
print(f"Input: {df.shape[0]} rows x {df.shape[1]} cols")

YEARS = range(2009, 2022)


# ── 1. DAYS_SINCE_LAST_HOLIDAY (capped at 14) ─────────────────────────────
def build_holiday_set(years):
    h = set()
    for y in years:
        e = easter(y)
        fixed = [
            datetime.date(y, 1, 1), datetime.date(y, 5, 1),
            datetime.date(y, 7, 11), datetime.date(y, 7, 21),
            datetime.date(y, 8, 15), datetime.date(y, 11, 1),
            datetime.date(y, 11, 11), datetime.date(y, 12, 25),
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

# days_since_last_holiday: for each date, find the most recent holiday on or before it
def days_since(d, holidays):
    for lag in range(0, 15):
        if (d - datetime.timedelta(days=lag)) in holidays:
            return lag
    return 14  # capped at 14

df["days_since_last_holiday"] = date_col.apply(lambda d: days_since(d, HOLIDAYS))

print(f"  days_since_last_holiday: min={df['days_since_last_holiday'].min()}  "
      f"max={df['days_since_last_holiday'].max()}  "
      f"mean={df['days_since_last_holiday'].mean():.2f}")

# Verify: day after Christmas should be 1
dec26 = df[df["date"].dt.strftime("%m-%d") == "12-26"].head(3)
print(f"  Dec 26 days_since_last_holiday: {dec26['days_since_last_holiday'].values}  (expected 1)")

# Verify: day-of-holiday should be 0
jan1 = df[df["date"].dt.strftime("%m-%d") == "01-01"].head(3)
print(f"  Jan 1 days_since_last_holiday: {jan1['days_since_last_holiday'].values}  (expected 0)")


# ── 2. TEMPERATURE_ZSCORE_SEASONAL ────────────────────────────────────────
# For each day-of-year, compute the climatological mean/std from AVAILABLE weather
# rows (2015-2020), then z-score each temperature observation.
# Pre-2015 rows: weather_data unavailable → set zscore = 0.0

wx_col = "wx_Temperature"
assert wx_col in df.columns, f"Missing column: {wx_col}"

weather_available = df["date"].dt.year >= 2015

# Compute DOY climatological mean and std from weather-available rows only
doy_stats = (
    df.loc[weather_available & df[wx_col].notna()]
    .assign(doy=df["date"].dt.dayofyear)
    .groupby("doy")[wx_col]
    .agg(["mean", "std"])
    .rename(columns={"mean": "clim_mean", "std": "clim_std"})
)
# Fill any missing DOY (e.g., Feb 29 in non-leap years) with global mean/std
global_mean = df.loc[weather_available & df[wx_col].notna(), wx_col].mean()
global_std  = df.loc[weather_available & df[wx_col].notna(), wx_col].std()
doy_stats["clim_std"] = doy_stats["clim_std"].fillna(global_std).replace(0, global_std)

df["_doy"] = df["date"].dt.dayofyear

def zscore_temp(row):
    if not weather_available[row.name] or pd.isna(row[wx_col]):
        return 0.0
    doy = row["_doy"]
    if doy not in doy_stats.index:
        return 0.0
    mu  = doy_stats.loc[doy, "clim_mean"]
    sig = doy_stats.loc[doy, "clim_std"]
    return (row[wx_col] - mu) / sig

df["temperature_zscore_seasonal"] = df.apply(zscore_temp, axis=1)

pre2015_check = df.loc[df["date"].dt.year < 2015, "temperature_zscore_seasonal"]
print(f"\n  temperature_zscore_seasonal:")
print(f"    Pre-2015 rows all zero: {(pre2015_check == 0.0).all()}")
print(f"    Post-2015 mean={df.loc[weather_available,'temperature_zscore_seasonal'].mean():.3f}  "
      f"std={df.loc[weather_available,'temperature_zscore_seasonal'].std():.3f}  (expected ~0, ~1)")

df.drop(columns=["_doy"], inplace=True)


# ── 3. SANITY CHECKS ───────────────────────────────────────────────────────
new_features = ["days_since_last_holiday", "temperature_zscore_seasonal"]
print(f"\nNew features: {new_features}")
print(f"Final shape: {df.shape}  (expected 2897 x 92)")
assert df.shape == (2897, 92), f"Shape mismatch: {df.shape}"
assert not df[new_features].isnull().any().any(), "NaN in new features"

# Check fold-3 val coverage
val3 = df[df["_fold_3"] == "val"]
print(f"\nFold-3 val (n={len(val3)}):")
print(f"  days_since_last_holiday: min={val3['days_since_last_holiday'].min()}  "
      f"max={val3['days_since_last_holiday'].max()}  "
      f"rows with value <= 3: {(val3['days_since_last_holiday'] <= 3).sum()}")
print(f"  temperature_zscore_seasonal (post-2015): "
      f"mean={val3['temperature_zscore_seasonal'].mean():.3f}  "
      f"std={val3['temperature_zscore_seasonal'].std():.3f}")

df.to_csv(str(FE_OUT), index=False)
print(f"\nSaved: {FE_OUT.name}")


# ── 4. HANDOFF JSON ────────────────────────────────────────────────────────
handoff = {
    "iteration": 23,
    "node": "feature_engineering",
    "timestamp": "2026-09-09",
    "schema_version": "2026-09-04",
    "type": "feature_engineering_5",
    "summary": (
        "FE iteration 5. Adds 2 remaining documented gaps from iter002 domain research: "
        "days_since_last_holiday (capped at 14, post-holiday demand suppression) and "
        "temperature_zscore_seasonal (temperature deviation from same-DOY climatological "
        "average; 0.0 for pre-2015 rows where weather unavailable). "
        "Output: iteration_023_feature_engineering.csv (2897 x 92 cols)."
    ),
    "new_features": new_features,
    "source_gaps": [
        "iter002 FR-04: days_since_last_holiday (medium priority, capped at 14)",
        "iter002 weather assessment: temperature_zscore_seasonal (removes collinearity with month/season)",
    ],
    "output_csv": "data/iteration_023_feature_engineering.csv",
    "handoff": {
        "written_by": "feature_engineering",
        "iteration": 23,
        "context_clearable": True,
        "parameters": {"total_cols": 92, "model_features": 64},
        "durable_rules": [
            "temperature_zscore_seasonal is 0.0 for pre-2015 rows (no weather data). "
            "This is intentional — do NOT impute with DOY average for pre-2015.",
        ],
    },
}

with open(HIST_OUT, "w", encoding="utf-8") as f:
    json.dump(handoff, f, indent=2)
print(f"Handoff written: {HIST_OUT.name}")

with open(STATE_F, encoding="utf-8") as f:
    state = json.load(f)
state["current"]["iteration"] = 23
state["current"]["next_iteration_contract"] = {
    "node": "training",
    "iteration": 24,
    "fe_csv": "data/iteration_023_feature_engineering.csv",
    "tasks": [
        "Retrain all 6 horizon champions with FE5 features (92 cols, 64 model features)",
        "h60: use correct 95-row eval scheme (origins D-60, targets in val period)",
        "All models use objective=regression_l1",
        "Compare vs iter022 results — only update champion if improved",
    ],
}
with open(STATE_F, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
print("State updated: iteration=23")
print("\n=== Iteration 023 complete ===")
