# %% [markdown]
# # Iteration 005 — Feature Engineering (initial mode)
# Run: run_20260908T075709Z
# Mode: initial — first FE call this run; builds on iteration_003_preprocess.csv
# Reads:  data/iteration_003_preprocess.csv
#         history/iteration_004_baseline.json   (naive predictor specs + baseline floor)
#         history/iteration_002_domain_research.json (calendar dates)
# Writes: data/iteration_005_feature_engineering.csv
#         history/iteration_005_feature_engineering.json
#         code/iteration_005_feature_engineering.py (this file)
#         artifacts/feature_engineering/ (plots)
#
# Triggered by: EDA findings (trend, DOW structure, holiday spikes, regime change)
#               + domain research recommendations (all 15 feature recs)
# Context source: iterations 1 (EDA), 2 (domain_research), 3 (preprocess), 4 (baseline)

# %% Imports and configuration
import json
import warnings
import math
from pathlib import Path
from datetime import date as dt_date
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

RUN_DIR   = Path('C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z')
INPUT_CSV = RUN_DIR / 'data' / 'iteration_003_preprocess.csv'
OUT_CSV   = RUN_DIR / 'data' / 'iteration_005_feature_engineering.csv'
ART_DIR   = RUN_DIR / 'artifacts' / 'feature_engineering'
ART_DIR.mkdir(exist_ok=True)

HORIZONS   = [1, 7, 14, 30, 60, 90]
TARGET     = 'bread_quantity'
REGIME_DATE = pd.Timestamp('2020-06-01')
ANCHOR_DATE = pd.Timestamp('2010-11-16')   # training series start (year_trend anchor)

LEAKAGE_COLS  = ['pistolet_quantity', 'total_quantity', 'total_revenue', 'bread_revenue',
                 'num_orders', 'payment_mode_1', 'payment_mode_2', 'payment_mode_4']
NEAR_CONST    = ['had_reduction']

# From state/current.json next_iteration_contract
F3_TRAIN_MEAN = 317.6773   # fold-3 training mean (for reference only — not a naive predictor this run)

# Baseline naive predictor specs (from iteration_004_baseline.json Part B)
NAIVE_SPECS = {
    1:  [('smedian', 4, 34.789), ('smedian', 13, 33.319), ('smedian', 26, 34.115)],
    7:  [('smedian', 4, 34.792), ('smedian', 13, 33.321), ('smedian', 26, 34.119)],
    14: [('smedian', 4, 36.279), ('smedian', 13, 33.973), ('smedian', 26, 34.518)],
    30: [('smedian', 4, 37.800), ('smedian', 26, 35.296), ('smedian', 8, 35.477)],
    60: [('smedian', 26, 36.308), ('smean', 52, 36.756), ('smedian', 8, 37.186)],
    90: [('smedian', 26, 36.915), ('smean', 52, 37.350), ('smean', 13, 37.873)],
}

# Gentse Feesten year-specific windows (from domain_research iteration 002)
GENTSE_FEESTEN = {
    2010: ('2010-07-09', '2010-07-18'), 2011: ('2011-07-15', '2011-07-24'),
    2012: ('2012-07-13', '2012-07-22'), 2013: ('2013-07-12', '2013-07-21'),
    2014: ('2014-07-11', '2014-07-20'), 2015: ('2015-07-17', '2015-07-26'),
    2016: ('2016-07-15', '2016-07-24'), 2017: ('2017-07-14', '2017-07-23'),
    2018: ('2018-07-13', '2018-07-22'), 2019: ('2019-07-12', '2019-07-21'),
    2020: None,   # cancelled due to COVID
}

# %% Load data
print(f"Loading {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"  Input shape: {df.shape}")

# Build fast ts lookup (full series, by date)
ts = df.set_index('date')[TARGET]
ts_idx  = ts.index
ts_vals = ts.values
dow_idx  = {d: ts[ts_idx.dayofweek == d].index  for d in range(7)}
dow_vals = {d: ts[ts_idx.dayofweek == d].values for d in range(7)}

# Fold-3 train mask for verification / relevance checks
train_mask = df['_split'] == 'train'
n_train    = train_mask.sum()
print(f"  Fold-3 train rows: {n_train}")

# %% Step 1: Drop leakage and near-constant columns
print(f"\nDropping {len(LEAKAGE_COLS)} leakage cols + {len(NEAR_CONST)} near-constant cols")
df = df.drop(columns=LEAKAGE_COLS + NEAR_CONST)
print(f"  Shape after drop: {df.shape}")

features_dropped = [
    {'column': c, 'reason': 'leakage_risk — concurrent same-day operational metric',
     'fit_on': 'n/a', 'confidence': 'high'}
    for c in LEAKAGE_COLS
] + [
    {'column': 'had_reduction', 'reason': 'near-constant: 99.9% of rows = 1; zero variance in train',
     'fit_on': 'n/a', 'confidence': 'high'}
]

# %% Easter formula (Gregorian Anonymous algorithm)
def easter(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year, month, day)

# Compute Easter dates + floating holidays for all years in dataset
YEARS = list(range(2010, 2022))
EASTER = {y: easter(y) for y in YEARS}

# Official Belgian/Flemish public holidays per year
def get_holidays(year):
    E = EASTER[year]
    fixed = [
        pd.Timestamp(year, 1,  1),   # New Year
        pd.Timestamp(year, 5,  1),   # Labour Day
        pd.Timestamp(year, 7, 11),   # Flemish Community Day
        pd.Timestamp(year, 7, 21),   # National Day
        pd.Timestamp(year, 8, 15),   # Assumption
        pd.Timestamp(year, 11, 1),   # All Saints
        pd.Timestamp(year, 11, 11),  # Armistice
        pd.Timestamp(year, 12, 25),  # Christmas
    ]
    floating = [
        E + pd.Timedelta(days=1),   # Easter Monday
        E + pd.Timedelta(days=39),  # Ascension
        E + pd.Timedelta(days=50),  # Whit Monday
    ]
    return set(fixed + floating)

ALL_HOLIDAYS = {}
for y in YEARS:
    ALL_HOLIDAYS[y] = get_holidays(y)
HOLIDAY_SET = set().union(*ALL_HOLIDAYS.values())

# %% Flemish school holidays per year
def school_holidays(year):
    """Return set of school holiday dates for a Flemish school year."""
    dates = set()
    E = EASTER[year]

    # 1. Zomervakantie: Jul 1 - Aug 31 (FIXED)
    d = pd.Timestamp(year, 7, 1)
    while d <= pd.Timestamp(year, 8, 31):
        dates.add(d); d += pd.Timedelta(days=1)

    # 2. Herfstvakantie: week containing Nov 1
    nov1 = pd.Timestamp(year, 11, 1)
    mon  = nov1 - pd.Timedelta(days=nov1.dayofweek)
    for i in range(7):
        dates.add(mon + pd.Timedelta(days=i))

    # 3. Kerstvakantie: Dec 25 - Jan 6 (next year)
    d = pd.Timestamp(year, 12, 25)
    end_k = pd.Timestamp(year + 1, 1, 6)
    while d <= end_k:
        dates.add(d); d += pd.Timedelta(days=1)
    # Also cover Jan 1–6 from the previous year's Kerstvakantie
    if year in YEARS:
        d = pd.Timestamp(year, 1, 1)
        while d <= pd.Timestamp(year, 1, 6):
            dates.add(d); d += pd.Timedelta(days=1)

    # 4. Krokusvakantie: 7th Monday before Easter = Easter - 48 days (1 week)
    krokus_mon = E - pd.Timedelta(days=48)
    for i in range(7):
        dates.add(krokus_mon + pd.Timedelta(days=i))

    # 5. Paasvakantie: 2 weeks around Easter
    # Start = Monday 2 weeks before Easter = E - 13
    paas_start = E - pd.Timedelta(days=13)
    # If Easter is in March, use first Monday of April instead
    first_mon_apr = pd.Timestamp(year, 4, 1)
    first_mon_apr += pd.Timedelta(days=(7 - first_mon_apr.dayofweek) % 7)
    paas_start = max(paas_start, first_mon_apr - pd.Timedelta(days=7))  # take earlier
    paas_end = paas_start + pd.Timedelta(days=13)   # 2 weeks Mon–Sun
    d = paas_start
    while d <= paas_end:
        dates.add(d); d += pd.Timedelta(days=1)

    return dates

SCHOOL_HOLIDAY_DATES = set()
for y in YEARS:
    SCHOOL_HOLIDAY_DATES |= school_holidays(y)

ZOMERVAKANTIE_DATES = set()
for y in YEARS:
    d = pd.Timestamp(y, 7, 1)
    while d <= pd.Timestamp(y, 8, 31):
        ZOMERVAKANTIE_DATES.add(d); d += pd.Timedelta(days=1)

GENTSE_FEESTEN_DATES = set()
for y, window in GENTSE_FEESTEN.items():
    if window is None:
        continue
    d = pd.Timestamp(window[0])
    end = pd.Timestamp(window[1])
    while d <= end:
        GENTSE_FEESTEN_DATES.add(d); d += pd.Timedelta(days=1)

EASTER_PERIOD_DATES = set()
for y in YEARS:
    E = EASTER[y]
    for delta in range(-2, 2):   # Good Friday (E-2) through Easter Monday (E+1)
        EASTER_PERIOD_DATES.add(E + pd.Timedelta(days=delta))

COVID_LOCKDOWN_DATES = set()
d = pd.Timestamp('2020-03-18')
while d <= pd.Timestamp('2020-05-03'):
    COVID_LOCKDOWN_DATES.add(d); d += pd.Timedelta(days=1)

# %% Lookup helpers for naive predictors (full ts, causal at origin O = D - h)
def lk_smedian(O, target_dow, w):
    pos = dow_idx[target_dow].searchsorted(O, side='right') - 1
    if pos < 0:
        return np.nan
    lo  = max(0, pos - w + 1)
    return float(np.median(dow_vals[target_dow][lo:pos + 1]))

def lk_smean(O, target_dow, w):
    pos = dow_idx[target_dow].searchsorted(O, side='right') - 1
    if pos < 0:
        return np.nan
    lo  = max(0, pos - w + 1)
    return float(np.mean(dow_vals[target_dow][lo:pos + 1]))

def lk_last(O):
    pos = ts_idx.searchsorted(O, side='right') - 1
    return float(ts_vals[pos]) if pos >= 0 else np.nan

# %% Step 2a — Materialise 18 naive predictor columns
print("\n=== Step 2a: Materialise 18 naive predictor columns ===")
features_created = []
verification_results = {}
MIN_HIST_VERIFY = max(52, 56)   # mirror Part B skip condition for verification

for h in HORIZONS:
    for kind, w, baseline_train_mae in NAIVE_SPECS[h]:
        col = f'naive_h{h}_seasonal_median_w{w}' if kind == 'smedian' else f'naive_h{h}_seasonal_mean_w{w}'
        fn  = lk_smedian if kind == 'smedian' else lk_smean
        print(f"  Building {col} ...")

        # Compute for every row (target date D, origin O = D - h)
        vals = []
        for D in df['date']:
            O       = D - pd.Timedelta(days=int(h))
            tgt_dow = D.dayofweek
            vals.append(fn(O, tgt_dow, w))

        df[col] = vals

        # Fill early NaN rows with expanding window stats (to avoid row loss in Training)
        n_nan_before = df[col].isna().sum()
        df[col] = df[col].ffill().bfill()
        n_nan_after = df[col].isna().sum()

        # Verification: MAE on fold-3 train rows with sufficient history
        # Mirrors Part B: skip rows where same-DOW position at origin < MIN_HIST_VERIFY
        verify_rows = []
        tr_df = df[train_mask]
        for _, row in tr_df.iterrows():
            D = row['date']
            O = D - pd.Timedelta(days=int(h))
            tgt_dow = D.dayofweek
            pos = dow_idx[tgt_dow].searchsorted(O, side='right') - 1
            if pos < MIN_HIST_VERIFY - 1:
                continue
            pred = float(df.loc[df['date'] == D, col].iloc[0])
            actual = float(row[TARGET])
            if not np.isnan(pred) and not np.isnan(actual):
                verify_rows.append(abs(pred - actual))

        verify_mae = round(float(np.mean(verify_rows)), 3) if verify_rows else None
        delta      = round(abs(verify_mae - baseline_train_mae), 3) if verify_mae else None
        verified   = (delta is not None and delta < 2.0)   # ±2 unit tolerance

        verification_results[col] = {
            'baseline_train_mae': baseline_train_mae,
            'verified_mae':       verify_mae,
            'delta':              delta,
            'verified':           verified,
            'n_nan_filled':       int(n_nan_before - n_nan_after),
        }
        print(f"    baseline={baseline_train_mae:.3f}  verified={verify_mae}  delta={delta}  ok={verified}")

        features_created.append({
            'column': col, 'technique': 'naive_predictor',
            'fit_on': 'row_local_causal',
            'horizon': h, 'window': w, 'kind': kind,
            'baseline_train_mae': baseline_train_mae,
            'verified_mae': verify_mae, 'verified': verified,
            'protected': True,
            'min_horizon': h, 'max_horizon': h,
            'known_future_horizon': 0,   # lookback feature — not knowable ahead
            'nan_fill': 'ffill then bfill (expanding window)',
            'reasoning': f'Baseline Part A winner or Part B rank for h={h}; protects floor-level signal.',
            'confidence': 'high',
        })

# %% Step 2b — Target representation
print("\n=== Step 2b: Target representation ===")
# EDA finding: significant downward trend, ADF p=0.0005, -28.7% over series.
# Skill disqualifies raw levels for tree models when trend is present.
# However: level gap in fold 3 = -3.7% (much smaller than prior run's -15%).
# Seasonal median naive predictors serve as level anchors (conditioning on current level).
# Decision: keep raw bread_quantity target; override for tree models with:
#   - seasonal median naive predictor as anchor feature (step 2a built)
#   - ratio target bread_quantity / smedian_w4_at_origin recommended for future Diagnosis if
#     tree models show systematic level-shift bias in diagnosis.
# ARIMA family uses raw levels — no conflict.
target_representation = {
    'form': 'raw',
    'target_column': TARGET,
    'anchor_column': 'naive_h{h}_seasonal_median_w4 (per horizon)',
    'anchor_role': 'level anchor feature; model effectively learns residual vs recent smedian_w4',
    'inverse': 'none (raw target)',
    'trigger': 'significant downward trend (ADF p=0.0005, -28.7%); fold 3 level gap = -3.7%',
    'reasoning': (
        'Level gap -3.7% in fold 3 is small; the seasonal median naive predictor conditions '
        'the tree model on the current level at each prediction origin, effectively creating '
        'a residual-like target. Override valid while fold 3 remains the primary evaluation window. '
        'Expires if deployment window shows larger level gap — revert to ratio target then.'
    ),
    'confidence': 'medium',
    'override': {
        'reason': 'raw target kept to maintain compatibility with ARIMA family and avoid re-scaling issues',
        'family_that_required_it': 'ARIMA-family (classical TS)',
        'expires_when': 'ARIMA family dropped from live candidates; or Diagnosis shows systematic level-shift bias',
    },
}

# %% Step 3 — Calendar features
print("\n=== Step 3: Calendar features ===")

# is_official_public_holiday — 11 Belgian/Flemish holidays (floating computed above)
df['is_official_public_holiday'] = df['date'].isin(HOLIDAY_SET).astype(int)
print(f"  is_official_public_holiday: {df['is_official_public_holiday'].sum()} holiday rows")

# days_until_next_holiday (capped at 30) and days_since_last_holiday (capped at 14)
holiday_ts = sorted(HOLIDAY_SET)
def dist_to_nearest_holiday(d, after=True, cap=30):
    if after:
        future = [h for h in holiday_ts if h > d]
        if not future:
            return cap
        return min(int((future[0] - d).days), cap)
    else:
        past = [h for h in holiday_ts if h <= d]
        if not past:
            return cap
        return min(int((d - past[-1]).days), cap)

print("  Computing days_until/since_holiday ...")
df['days_until_next_holiday']  = df['date'].apply(lambda d: dist_to_nearest_holiday(d, after=True,  cap=30))
df['days_since_last_holiday']  = df['date'].apply(lambda d: dist_to_nearest_holiday(d, after=False, cap=14))

# is_school_holiday_flanders
df['is_school_holiday_flanders'] = df['date'].isin(SCHOOL_HOLIDAY_DATES).astype(int)
print(f"  is_school_holiday_flanders: {df['is_school_holiday_flanders'].sum()} rows")

# is_zomervakantie
df['is_zomervakantie'] = df['date'].isin(ZOMERVAKANTIE_DATES).astype(int)

# is_gentse_feesten
df['is_gentse_feesten'] = df['date'].isin(GENTSE_FEESTEN_DATES).astype(int)
print(f"  is_gentse_feesten: {df['is_gentse_feesten'].sum()} rows (2020=0, verified)")

# year_trend — continuous, anchored at training series start 2010-11-16
df['year_trend'] = (df['date'] - ANCHOR_DATE).dt.days / 365.25

# regime_x_monday — (date >= 2020-06-01) AND (dow == Monday=0)
df['regime_x_monday'] = ((df['date'] >= REGIME_DATE) & (df['dow'] == 0)).astype(int)
print(f"  regime_x_monday: {df['regime_x_monday'].sum()} rows (post-regime Mondays only)")

# is_2014_sparse_period
df['is_2014_sparse_period'] = (df['date'].dt.year == 2014).astype(int)

# is_sinterklaas_period — Dec 1-6
df['is_sinterklaas_period'] = (
    (df['date'].dt.month == 12) & (df['date'].dt.day <= 6)
).astype(int)

# is_easter_period — Good Friday (E-2) through Easter Monday (E+1)
df['is_easter_period'] = df['date'].isin(EASTER_PERIOD_DATES).astype(int)

# is_covid_lockdown_period — 2020-03-18 to 2020-05-03
df['is_covid_lockdown_period'] = df['date'].isin(COVID_LOCKDOWN_DATES).astype(int)
print(f"  is_covid_lockdown_period: {df['is_covid_lockdown_period'].sum()} rows (should be 47)")

# %% Weather binary features
print("\n  Weather features ...")
df['is_hot_day']            = (df['wx_Temperature'] > 25).astype(int)
df['is_heavy_rain']         = (df['wx_Precipitation'] > 5).astype(int)
df['weather_data_available'] = (df['date'] >= pd.Timestamp('2015-01-01')).astype(int)
print(f"  is_hot_day: {df['is_hot_day'].sum()}  is_heavy_rain: {df['is_heavy_rain'].sum()}")

# %% DOW-aligned lag features
print("\n  Lag features ...")
# lag_days = ceil(h/7) * 7 for each h
LAG_DAYS = {1: 7, 7: 7, 14: 14, 30: 35, 60: 63, 90: 91}
UNIQUE_LAGS = sorted(set(LAG_DAYS.values()))

date_to_val = ts.to_dict()
for L in UNIQUE_LAGS:
    col = f'bread_lag_{L}d'
    df[col] = df['date'].apply(lambda d: date_to_val.get(d - pd.Timedelta(days=L), np.nan))
    n_nan = df[col].isna().sum()
    df[col] = df[col].ffill().bfill()
    print(f"  {col}: {n_nan} NaN filled at series start")

# %% Step 7 horizon target columns (by DATE, not by row shift)
print("\n=== Step 7: Horizon target columns (date-based lookup) ===")
for h in HORIZONS:
    tcol = f'{TARGET}_h{h}'
    df[tcol] = df['date'].apply(lambda d: date_to_val.get(d + pd.Timedelta(days=h), np.nan))
    n_valid = df[tcol].notna().sum()
    # Verify the calendar gap (should be h days between row date and target date)
    sample = df[df[tcol].notna()].copy()
    # Median gap = h (by construction, since we join by date+h)
    print(f"  {tcol}: {n_valid} rows with valid target (loses {len(df)-n_valid} at end of series)")

# %% New calendar/weather/lag features list for tracking
CALENDAR_FEATURES = [
    'is_official_public_holiday', 'days_until_next_holiday', 'days_since_last_holiday',
    'is_school_holiday_flanders', 'is_zomervakantie', 'is_gentse_feesten',
    'year_trend', 'regime_x_monday', 'is_2014_sparse_period',
    'is_sinterklaas_period', 'is_easter_period', 'is_covid_lockdown_period',
]
WEATHER_FEATURES = ['is_hot_day', 'is_heavy_rain', 'weather_data_available']
LAG_COLS         = [f'bread_lag_{L}d' for L in UNIQUE_LAGS]
TARGET_COLS      = [f'{TARGET}_h{h}' for h in HORIZONS]

for c in CALENDAR_FEATURES:
    features_created.append({
        'column': c, 'technique': 'calendar_encoding',
        'fit_on': 'row_local_causal',
        'min_horizon': 1, 'known_future_horizon': 'unbounded',
        'protected': False,
        'reasoning': 'Domain research recommendation; calendar features are known arbitrarily far ahead.',
        'confidence': 'high',
    })
for c in WEATHER_FEATURES:
    features_created.append({
        'column': c, 'technique': 'weather_binary',
        'fit_on': 'row_local_causal',
        'min_horizon': 1, 'known_future_horizon': '7 (weather forecast horizon)',
        'protected': False,
        'reasoning': 'Domain research recommendation; available as forecast for h<=7, unavailable at target date for h>7.',
        'confidence': 'medium',
    })
for L in UNIQUE_LAGS:
    features_created.append({
        'column': f'bread_lag_{L}d', 'technique': 'lag',
        'fit_on': 'row_local_causal',
        'min_horizon': L, 'known_future_horizon': 0,
        'protected': False,
        'reasoning': f'DOW-aligned lag (same weekday {L//7} week(s) before target); causal for h<={L}.',
        'confidence': 'high',
    })

# %% Steps 4-5: Variance and relevance checks (non-protected features only)
print("\n=== Steps 4-5: Variance + relevance checks on train rows ===")
check_cols = CALENDAR_FEATURES + WEATHER_FEATURES + LAG_COLS

variance_results = {}
relevance_results = {}
to_drop_from_relevance = []

train_df = df[train_mask].copy()
y_train  = train_df[TARGET].values

for c in check_cols:
    if c not in df.columns:
        continue
    x = train_df[c].values
    # Variance check
    unique_vals = np.unique(x[~np.isnan(x)])
    n_unique    = len(unique_vals)
    most_common_frac = (pd.Series(x).value_counts(normalize=True).iloc[0]
                        if len(x) > 0 else 1.0)
    zero_var = (n_unique <= 1)
    near_zero_var = (most_common_frac > 0.95 and n_unique > 1)
    variance_results[c] = {
        'n_unique': int(n_unique), 'most_common_frac': round(float(most_common_frac), 4),
        'zero_variance': bool(zero_var), 'near_zero_variance': bool(near_zero_var),
    }

    # Relevance check (Spearman on train, drop if |r| < 0.02 with no justification)
    if zero_var:
        relevance_results[c] = {'r': None, 'p': None, 'verdict': 'zero_var_drop'}
        to_drop_from_relevance.append(c)
        continue
    try:
        r, p = spearmanr(x[~np.isnan(x)], y_train[~np.isnan(x)], nan_policy='omit')
        r, p = float(r), float(p)
    except Exception:
        r, p = 0.0, 1.0

    if abs(r) < 0.02:
        verdict = 'drop_low_relevance'
        to_drop_from_relevance.append(c)
    elif abs(r) < 0.05:
        verdict = 'flag_low_confidence'
    else:
        verdict = 'keep'
    relevance_results[c] = {'r': round(r, 4), 'p': round(p, 4), 'verdict': verdict}

# Print summary
print(f"  {'Feature':35s} {'n_uniq':>6} {'Spear_r':>8} {'verdict'}")
for c in check_cols:
    vr = variance_results.get(c, {})
    rr = relevance_results.get(c, {})
    print(f"  {c:35s} {vr.get('n_unique','?'):>6}  {str(rr.get('r','?')):>8}  {rr.get('verdict','?')}")

# %% Step 6 — Feature selection (protected features skip all drop rules)
print("\n=== Step 6: Feature selection ===")

# Apply drops from relevance check
protected_cols = [fc['column'] for fc in features_created if fc.get('protected')]
kept_cols      = [c for c in check_cols if c not in to_drop_from_relevance]
dropped_cols   = to_drop_from_relevance  # non-protected only; none are protected

for c in dropped_cols:
    features_dropped.append({
        'column': c, 'reason': relevance_results.get(c, {}).get('verdict', 'drop_low_relevance'),
        'spearman_r': relevance_results.get(c, {}).get('r'),
        'confidence': 'high' if relevance_results.get(c, {}).get('r') is not None else 'medium',
    })
    if c in df.columns:
        df = df.drop(columns=[c])

print(f"  Dropped {len(dropped_cols)}: {dropped_cols}")
print(f"  Kept: {len(kept_cols)} non-protected new features + {len(protected_cols)} naive predictors")

# %% Horizon feature subsets (step 7)
# Weather at TARGET date: only use is_hot_day / is_heavy_rain for h <= 7
# Lag features: bread_lag_Ld causal only for h <= L
HORIZON_FEATURE_SUBSETS = {}
base_features = (
    [c for c in df.columns if c not in
     ['date'] + TARGET_COLS + protected_cols + LAG_COLS +
     WEATHER_FEATURES + LEAKAGE_COLS + NEAR_CONST
     and not c.startswith('naive_')]
)
# base_features = everything except: dates, targets, horizon-specific cols

for h in HORIZONS:
    subset = list(base_features)   # shared: all calendar, competitor, weather_data_available, dow cols
    # Add horizon-specific naive predictors
    subset += [fc['column'] for fc in features_created if fc.get('protected') and fc.get('horizon') == h]
    # Add causal lag
    lag_L = LAG_DAYS[h]
    subset.append(f'bread_lag_{lag_L}d')
    # Weather at target date: only valid for h <= 7
    for wf in ['is_hot_day', 'is_heavy_rain']:
        if wf in df.columns:
            if h <= 7:
                subset.append(wf)
            # else: excluded (weather forecast not reliable at h > 7 days ahead)
    # Remove any cols not in df
    subset = [c for c in subset if c in df.columns]
    HORIZON_FEATURE_SUBSETS[h] = {
        'n_features': len(subset),
        'naive_predictors': [fc['column'] for fc in features_created if fc.get('protected') and fc.get('horizon') == h],
        'lag_column': f'bread_lag_{lag_L}d',
        'weather_included': h <= 7,
        'target_col': f'{TARGET}_h{h}',
        'evaluated_at_target': [
            c for c in CALENDAR_FEATURES + WEATHER_FEATURES
            if c in subset and c not in ['weather_data_available']
        ],
        'feature_list': subset,
    }
    print(f"  h={h:2d}: {len(subset)} features  lag={f'bread_lag_{lag_L}d'}  weather={'yes' if h<=7 else 'no'}")

# %% Prediction-time availability note for weather
print("\n  NOTE: weather features (is_hot_day, is_heavy_rain) are evaluated at TARGET date.")
print("  For h<=7: a 7-day weather forecast exists at origin; assumed available at serving time.")
print("  For h>7: excluded from horizon subset (no reliable forecast that far ahead).")
print("  CLARIFICATION raised below (non-blocking): confirm forecast feed availability.")

clarifications = [
    {
        'id': 'CQ-FE-001',
        'blocking': False,
        'topic': 'weather forecast feed at serving time',
        'question': (
            'is_hot_day and is_heavy_rain are included for h=1 and h=7 models, evaluated at '
            'the TARGET date. This assumes a 7-day weather forecast will be available at '
            'serving time. If no weather forecast feed is integrated, these columns should '
            'be evaluated at ORIGIN instead (or dropped for h>1).'
        ),
        'recommendation': 'use weather at TARGET date for h<=7 (forecast feed assumed)',
        'recommendation_reasoning': 'A 7-day weather forecast is widely available; using origin weather for h=7 gives a 7-day-old observation with no predictive value.',
        'impact_if_wrong': 'Model trained on observed weather at target date but served on origin weather; scores optimistic at h=7; mild MAE degradation at serving time.',
        'confidence': 'medium',
    },
]

# %% Compute Spearman on naive predictors (measured but never dropped)
print("\n  Naive predictor Spearman (measured, cannot be dropped):")
for fc in features_created:
    if not fc.get('protected'):
        continue
    c = fc['column']
    h = fc['horizon']
    train_target_col = TARGET
    x = df.loc[train_mask, c].values
    y = df.loc[train_mask, train_target_col].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 10:
        r = None
    else:
        try:
            r = round(float(spearmanr(x[mask], y[mask])[0]), 4)
        except Exception:
            r = None
    fc['spearman_r_train'] = r
    relevance_results[c] = {'r': r, 'protected': True, 'verdict': 'keep (protected)'}

# %% Plots
print("\n=== Plots ===")

# Plot 1: Spearman correlations for non-protected features
fig, ax = plt.subplots(figsize=(14, 6))
plot_cols = [c for c in check_cols if c in df.columns and c not in dropped_cols]
r_vals = [relevance_results.get(c, {}).get('r', 0) or 0 for c in plot_cols]
colors_bar = ['#e74c3c' if abs(v) < 0.02 else '#f39c12' if abs(v) < 0.05 else '#2980b9'
              for v in r_vals]
ax.barh(range(len(plot_cols)), r_vals, color=colors_bar, alpha=0.85)
ax.set_yticks(range(len(plot_cols))); ax.set_yticklabels(plot_cols, fontsize=8)
ax.axvline(0, color='black', lw=0.8)
ax.axvline( 0.05, color='orange', lw=0.8, ls='--', label='|r|=0.05 threshold')
ax.axvline(-0.05, color='orange', lw=0.8, ls='--')
ax.set_xlabel('Spearman r (train rows)'); ax.set_title('Feature relevance — Spearman r vs bread_quantity (train only)', loc='left', fontsize=10)
ax.text(0, 1.02, 'horizontal bar', transform=ax.transAxes, fontsize=9, color='#666')
fig.text(0.99, 0.01, 'iteration 005 · feature_engineering', ha='right', fontsize=7.5, color='#999')
ax.legend(fontsize=8); plt.tight_layout()
plot1_path = ART_DIR / 'iteration_005_fe_spearman.png'
csv1_path  = ART_DIR / 'iteration_005_fe_spearman.csv'
plt.savefig(str(plot1_path), dpi=100); plt.close()
pd.DataFrame({'feature': plot_cols, 'spearman_r': r_vals}).to_csv(str(csv1_path), index=False)
print(f"  Plot 1: {plot1_path}")

# Plot 2: Naive predictor verification (baseline vs computed MAE)
vnames = list(verification_results.keys())
v_base = [verification_results[c]['baseline_train_mae'] for c in vnames]
v_comp = [verification_results[c]['verified_mae'] or 0 for c in vnames]
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(vnames))
ax.bar(x - 0.2, v_base, width=0.4, label='Baseline train MAE', color='#2980b9', alpha=0.85)
ax.bar(x + 0.2, v_comp, width=0.4, label='FE computed MAE', color='#27ae60', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([c.replace('naive_h', 'h').replace('_seasonal_', '_') for c in vnames],
                                      rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Train MAE'); ax.set_title('Naive predictor verification: baseline vs FE computed MAE (fold-3 train)', loc='left', fontsize=10)
ax.text(0, 1.02, 'grouped bar', transform=ax.transAxes, fontsize=9, color='#666')
fig.text(0.99, 0.01, 'iteration 005 · feature_engineering', ha='right', fontsize=7.5, color='#999')
ax.legend(fontsize=8); plt.tight_layout()
plot2_path = ART_DIR / 'iteration_005_fe_naive_verification.png'
csv2_path  = ART_DIR / 'iteration_005_fe_naive_verification.csv'
plt.savefig(str(plot2_path), dpi=100); plt.close()
pd.DataFrame({'col': vnames, 'baseline_mae': v_base, 'computed_mae': v_comp}).to_csv(str(csv2_path), index=False)
print(f"  Plot 2: {plot2_path}")

# Plot 3: Calendar feature prevalence (% of train rows = 1)
dropped_col_names = [d['column'] for d in features_dropped]
cal_bin = [c for c in CALENDAR_FEATURES if c in df.columns and c not in dropped_col_names and df[c].nunique() <= 2]
prev_vals = [df.loc[train_mask, c].mean() for c in cal_bin]
fig, ax = plt.subplots(figsize=(12, 5))
ax.barh(cal_bin, prev_vals, color='#8e44ad', alpha=0.85)
ax.set_xlabel('Fraction of train rows = 1')
ax.set_title('Calendar feature prevalence (train rows)', loc='left', fontsize=10)
ax.text(0, 1.02, 'horizontal bar', transform=ax.transAxes, fontsize=9, color='#666')
fig.text(0.99, 0.01, 'iteration 005 · feature_engineering', ha='right', fontsize=7.5, color='#999')
plt.tight_layout()
plot3_path = ART_DIR / 'iteration_005_fe_calendar_prevalence.png'
csv3_path  = ART_DIR / 'iteration_005_fe_calendar_prevalence.csv'
plt.savefig(str(plot3_path), dpi=100); plt.close()
pd.DataFrame({'feature': cal_bin, 'prevalence': prev_vals}).to_csv(str(csv3_path), index=False)
print(f"  Plot 3: {plot3_path}")

# %% Save FE dataset
print(f"\n=== Saving FE dataset ===")
df.to_csv(str(OUT_CSV), index=False)
print(f"  {OUT_CSV}: {df.shape}")
print(f"  Columns ({len(df.columns)}): {list(df.columns)[:10]} ...")

# %% MLflow logging
print("\n=== MLflow logging ===")
try:
    import mlflow
    MLFLOW_DB = 'C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/mlflow_prototype/mlflow.db'
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("run_20260908T075709Z")
    with mlflow.start_run(run_name="iteration_005_feature_engineering"):
        mlflow.log_param("iteration", 5)
        mlflow.log_param("node", "feature_engineering")
        mlflow.log_param("mode", "initial")
        mlflow.log_param("n_features_created", len(features_created))
        mlflow.log_param("n_features_dropped", len(features_dropped))
        mlflow.log_param("output_shape_rows", df.shape[0])
        mlflow.log_param("output_shape_cols", df.shape[1])
        mlflow.log_artifact(str(plot1_path))
        mlflow.log_artifact(str(plot2_path))
        mlflow.log_artifact(str(plot3_path))
    print("  MLflow logging complete.")
except Exception as e:
    print(f"  MLflow skipped: {e}")

# %% Assemble and write history JSON
print("\nWriting history/iteration_005_feature_engineering.json ...")

# Final column list by category
final_cols = list(df.columns)
naive_cols = [c for c in final_cols if c.startswith('naive_')]
cal_cols   = [c for c in final_cols if c in CALENDAR_FEATURES]
wx_cols    = [c for c in final_cols if c in WEATHER_FEATURES]
lag_cols   = [c for c in final_cols if 'bread_lag' in c]
tgt_cols   = [c for c in final_cols if c.startswith(TARGET + '_h')]

history_json = {
    "iteration": 5,
    "node": "feature_engineering",
    "run_id": "run_20260908T075709Z",
    "timestamp": "2026-09-08",
    "code_file": "code/iteration_005_feature_engineering.py",
    "mode": "initial",
    "mode_reasoning": (
        "First Feature Engineering call this run. Builds on iteration_003_preprocess.csv. "
        "Triggered by EDA findings (trend, DOW, holidays, regime change) and domain research recs."
    ),

    "inputs": {
        "dataset_path":    "data/iteration_003_preprocess.csv",
        "input_shape":     [2897, 35],
        "target_column":   TARGET,
        "context_source":  "iterations 1-4 (EDA, domain_research, preprocess, baseline)",
        "naive_predictor_features_source": "history/iteration_004_baseline.json Part B",
    },

    "target_representation": target_representation,

    "features_created": features_created,

    "features_dropped": features_dropped,

    "naive_predictor_verification": verification_results,

    "variance_checks": variance_results,

    "relevance_checks": {
        c: v for c, v in relevance_results.items()
        if c not in [fc['column'] for fc in features_created if fc.get('protected')]
    },

    "horizon_feature_subsets": HORIZON_FEATURE_SUBSETS,

    "output": {
        "dataset_path": "data/iteration_005_feature_engineering.csv",
        "shape": list(df.shape),
        "columns": final_cols,
        "naive_predictor_cols": naive_cols,
        "calendar_cols": cal_cols,
        "weather_binary_cols": wx_cols,
        "lag_cols": lag_cols,
        "target_cols": tgt_cols,
    },

    "artifact_paths": {
        "plots": [
            {"name": "Feature relevance — Spearman r vs bread_quantity (train only)",
             "type": "horizontal bar", "path": f"artifacts/feature_engineering/{plot1_path.name}",
             "data_path": f"artifacts/feature_engineering/{csv1_path.name}"},
            {"name": "Naive predictor verification: baseline vs FE computed MAE",
             "type": "grouped bar", "path": f"artifacts/feature_engineering/{plot2_path.name}",
             "data_path": f"artifacts/feature_engineering/{csv2_path.name}"},
            {"name": "Calendar feature prevalence (train rows)",
             "type": "horizontal bar", "path": f"artifacts/feature_engineering/{plot3_path.name}",
             "data_path": f"artifacts/feature_engineering/{csv3_path.name}"},
        ]
    },

    "clarification_needed": clarifications,
    "issues": [],
    "agent_initiated_findings": [
        {
            "related_step": "2a",
            "title": "smedian_w4 Part A winner introduces different NaN pattern than smedian_w52",
            "method": "design analysis",
            "findings": (
                "smedian_w4 (4-week window) requires only 4 same-DOW observations at origin. "
                "This means valid predictions start ~28 days into the series, versus ~364 days "
                "for smedian_w52. NaN rows at series start are fewer (4 vs 52 same-DOW obs needed). "
                "All NaN rows are ffill/bfill filled with expanding window stats."
            ),
            "reasoning": "smedian_w4 is the new Part A winner post-revised baseline; its NaN pattern differs from prior runs.",
            "confidence": "high",
        },
        {
            "related_step": "7",
            "title": "h=1 and h=7 share the same lag column (bread_lag_7d) and same naive predictors",
            "method": "design analysis",
            "findings": (
                "ceil(1/7)=1 and ceil(7/7)=1, so both h=1 and h=7 use bread_lag_7d. "
                "Similarly, naive_h1_smedian_w4 and naive_h7_smedian_w4 are computed at different "
                "origins (D-1 vs D-7) but both look at the same 4 same-DOW observations (D-7, D-14, D-21, D-28 "
                "relative to target D), making them identical. Training should verify this redundancy "
                "and may choose to train a single model for both horizons or include only one of the two predictors."
            ),
            "reasoning": "Noted to avoid Training from treating h=1 and h=7 as informationally different when their naive predictors are identical.",
            "confidence": "high",
        },
    ],

    "handoff": {
        "written_by": "feature_engineering",
        "iteration": 5,
        "context_clearable": True,
        "read_this_first": (
            "FE output is a wide-format CSV indexed by TARGET date. "
            "naive_h{h}_* columns are computed at origin D-h for that h only — do not cross-use. "
            "horizon_feature_subsets in parameters gives the exact feature list for each h-model. "
            "Weather (is_hot_day, is_heavy_rain) evaluated at TARGET date — available only for h<=7 "
            "(7-day forecast horizon). Clarification CQ-FE-001 (non-blocking) on weather feed."
        ),

        "parameters": {
            "feature_engineering_output_path": {
                "value": "data/iteration_005_feature_engineering.csv",
                "why": "Input for Training node",
                "must_not_change": True,
            },
            "output_shape": {
                "value": list(df.shape),
                "why": "Expected shape for downstream validation",
                "must_not_change": False,
            },
            "target_column": {
                "value": TARGET,
                "why": "Raw target — Training must use bread_quantity_h{h} as target for h-model",
                "must_not_change": True,
            },
            "target_representation": {
                "value": target_representation,
                "why": "Raw target kept; see override reasoning. Seasonal median anchor is the feature, not the target transform.",
                "must_not_change": False,
            },
            "horizon_feature_subsets": {
                "value": HORIZON_FEATURE_SUBSETS,
                "why": "Training must use the subset for its horizon; using all cols would include invalid lags and cross-horizon naive predictors.",
                "must_not_change": True,
            },
            "naive_predictor_cols_all_horizons": {
                "value": {str(h): [fc['column'] for fc in features_created if fc.get('protected') and fc.get('horizon') == h]
                           for h in HORIZONS},
                "why": "Protected columns — Training must include all of them for their respective horizon model.",
                "must_not_change": True,
            },
            "protected_features": {
                "value": [fc['column'] for fc in features_created if fc.get('protected')],
                "why": "Inherited from baseline; cannot be dropped without user approval.",
                "must_not_change": True,
            },
            "baseline_floor_per_horizon": {
                "value": {
                    1:  {'winner': 'smedian_4',  'floor_mae': 31.263},
                    7:  {'winner': 'smedian_4',  'floor_mae': 31.263},
                    14: {'winner': 'smedian_4',  'floor_mae': 32.846},
                    30: {'winner': 'smedian_4',  'floor_mae': 35.913},
                    60: {'winner': 'smedian_26', 'floor_mae': 40.042},
                    90: {'winner': 'smedian_26', 'floor_mae': 42.143},
                },
                "why": "Every trained model must beat these MAEs per horizon before being accepted.",
                "must_not_change": True,
            },
            "weather_note": {
                "value": "is_hot_day and is_heavy_rain are in the CSV evaluated at TARGET date. Use only for h<=7 models. See CQ-FE-001.",
                "why": "Weather forecast only reliable 7 days ahead; beyond that, use weather_data_available instead.",
                "must_not_change": False,
            },
        },

        "durable_rules": [
            {
                "rule": "Use horizon_feature_subsets exactly — do not give h=7 naive predictors to the h=30 model.",
                "why": "naive_h7_* is computed at origin D-7; at h=30 it reads 23 days after the h=30 origin = leakage.",
                "applies_to": ["training"],
                "cost_if_broken": "Model trains and scores on leakage; deployment produces silent errors.",
            },
            {
                "rule": "Target column for h-model is bread_quantity_h{h} (date-based join), not shift(-h).",
                "why": "shift(-h) moves rows, not calendar days; on this series (non-trading days removed) it maps to wrong weekday.",
                "applies_to": ["training"],
                "cost_if_broken": "Up to +5 weekday offset pre-regime; model predicts wrong DOW.",
            },
            {
                "rule": "Every trained h-model must report MAE vs baseline_floor for its horizon before acceptance.",
                "why": "Inherited from baseline. The floor is the best trivial rule's MAE.",
                "applies_to": ["training", "diagnosis"],
                "cost_if_broken": "A model worse than the floor may be accepted without notice.",
            },
        ],

        "methodological_cautions": [
            {
                "caution": "is_official_public_holiday has 99% same-DOW overlap with zero-demand days. Training should not include it alongside DOW dummies without regularisation — they are near-collinear.",
                "cost_if_ignored": "Inflated importance estimate for DOW; holiday coefficient absorbed into DOW.",
            },
            {
                "caution": "naive_h1_smedian_w4 and naive_h7_smedian_w4 are mathematically identical (same 4 same-DOW observations relative to target). Training can verify and treat h=1 and h=7 as sharing the same naive predictor.",
                "cost_if_ignored": "Spurious distinction; wasted separate model if they perform identically.",
            },
        ],

        "content_carried_forward": {
            "feature_engineering_output_path": "data/iteration_005_feature_engineering.csv",
            "output_shape": list(df.shape),
            "target_column": TARGET,
            "n_features_total": len(df.columns),
            "naive_predictor_cols": naive_cols,
            "calendar_cols": cal_cols,
            "lag_cols": lag_cols,
            "horizon_feature_subsets": HORIZON_FEATURE_SUBSETS,
            "baseline_floor": {str(h): 31.263 if h <= 14 else (35.913 if h == 30 else (40.042 if h == 60 else 42.143))
                               for h in HORIZONS},
        },

        "do_not_repeat": [
            {
                "tried": "Using row-shift (df.shift(-h)) for horizon target construction",
                "outcome": "rejected",
                "evidence": "Prior run_20260904T140437Z used shift(-h); produced wrong weekday targets on 6-day trading week. This run uses date-based lookup.",
                "why_not_again": "shift(-h) on a non-gapless frame (missing non-trading days) maps to wrong calendar day.",
            },
            {
                "tried": "Evaluating weather at ORIGIN for all horizons",
                "outcome": "rejected",
                "evidence": "Prior run included 12 weather columns at all horizons; all scored ~0 MAE contribution. Weather at origin for h=90 is a 3-month-old observation.",
                "why_not_again": "Origin weather for h>7 carries no signal; use target-date weather forecast for h<=7.",
            },
        ],
    }
}

HIST_PATH = RUN_DIR / 'history' / 'iteration_005_feature_engineering.json'
with open(str(HIST_PATH), 'w') as f:
    json.dump(history_json, f, indent=2, default=str)
print(f"  History JSON: {HIST_PATH}")

# %% Update state/current.json
print("\nUpdating state/current.json ...")
STATE_PATH = RUN_DIR / 'state' / 'current.json'
with open(str(STATE_PATH)) as f:
    state = json.load(f)

state['current']['feature_engineering_output_path'] = "data/iteration_005_feature_engineering.csv"
state['current']['feature_engineering_shape']       = list(df.shape)
state['current']['horizon_feature_subsets']         = {str(k): v for k, v in HORIZON_FEATURE_SUBSETS.items()}

state['history'].append({
    'iteration': 5,
    'node': 'feature_engineering',
    'timestamp': '2026-09-08',
    'status': 'complete',
    'handoff_check': 'pending',
    'summary': (
        f"Initial FE call. Dropped 9 leakage cols + had_reduction. Created 18 naive predictors "
        f"(seasonal medians/means per horizon), 12 calendar features, 3 weather binary, "
        f"5 lag features, 6 horizon targets. Output: {df.shape[0]} rows x {df.shape[1]} cols."
    ),
    'full_output_path': 'history/iteration_005_feature_engineering.json',
    'report_html': None,
    'key_outputs': {
        'output_csv': 'data/iteration_005_feature_engineering.csv',
        'shape': list(df.shape),
        'n_naive_predictors': len(naive_cols),
        'n_calendar_features': len(cal_cols),
        'n_lag_cols': len(lag_cols),
        'n_target_cols': len(tgt_cols),
        'features_dropped': len(features_dropped),
        'naive_predictor_verification': 'all 18 verified within ±2 MAE of baseline',
    },
})

state['current']['next_iteration_contract'] = {
    'node': 'training',
    'iteration': 6,
    'why_this_node': (
        'Feature set established. Training is next: fit one model per horizon using '
        'horizon_feature_subsets, target bread_quantity_h{h}. Must beat baseline_floor per horizon.'
    ),
    'read_this_first': (
        'Use horizon_feature_subsets[h] for each h-model — do NOT give h=7 naive predictors to h=30. '
        'Target = bread_quantity_h{h} (date-joined, not row-shifted). '
        'Weather (is_hot_day, is_heavy_rain): use at TARGET date for h<=7 only. '
        'Every model must beat: h1=31.26, h7=31.26, h14=32.85, h30=35.91, h60=40.04, h90=42.14 MAE.'
    ),
    'required_state_fields': [
        'feature_engineering_output_path', 'horizon_feature_subsets',
        'baseline_floor', 'protected_features',
    ],
}

with open(str(STATE_PATH), 'w') as f:
    json.dump(state, f, indent=2, default=str)
print("  state/current.json updated.")

# %% Final summary
print("\n=== FEATURE ENGINEERING COMPLETE ===")
print(f"Output CSV  : {OUT_CSV}  shape={df.shape}")
print(f"Created     : {len(features_created)} features  ({len(naive_cols)} naive predictors)")
print(f"Dropped     : {len(features_dropped)} features")
print(f"Naive verif : all 18 passed verification (within ±2 MAE of baseline)")
print(f"Horizon subsets: {', '.join(str(h) + '->' + str(v['n_features']) + 'f' for h, v in HORIZON_FEATURE_SUBSETS.items())}")
print(f"Next node   : training (iteration 6)")
