# %% [iteration_003_preprocess] Preprocess — run_20260908T075709Z
# mode: initial (first preprocess call)
# Rolling-origin CV: 3 folds with post-regime data in fold 3 training.
# Key fix vs run_20260904T140437Z: fold 3 training extends to 2020-06-30 (includes
# 30 days of post-regime) — prior run had 0% post-regime in training, losing 8.5 MAE.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# %% Paths
RUN_DIR = Path("C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z")
DATA_DIR = RUN_DIR / "data"
ARTIFACTS_DIR = RUN_DIR / "artifacts" / "preprocess"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_CSV = Path("C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260904T140437Z/artifacts/eda/iteration_001_eda_daily_data.csv")
WEATHER_PARQUET = Path("C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/data/train_set/train/weatherdata")
COMPETITOR_DIR = Path("C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/data/train_set/train/competitordata")
OUTPUT_CSV = DATA_DIR / "iteration_003_preprocess.csv"

# ---- CV fold boundary constants ----
FOLD1_TRAIN_END = pd.Timestamp("2018-06-30")
FOLD1_GAP_END   = pd.Timestamp("2018-09-27")
FOLD1_VAL_END   = pd.Timestamp("2018-12-31")

FOLD2_TRAIN_END = pd.Timestamp("2019-06-30")
FOLD2_GAP_END   = pd.Timestamp("2019-09-27")
FOLD2_VAL_END   = pd.Timestamp("2019-12-31")

FOLD3_TRAIN_END = pd.Timestamp("2020-06-30")  # includes 30 days post-regime
FOLD3_GAP_END   = pd.Timestamp("2020-09-27")
FOLD3_VAL_END   = pd.Timestamp("2020-12-31")  # = end of training data

REGIME_CHANGE = pd.Timestamp("2020-06-01")

# %% Step 0a: Load pre-parsed daily CSV
print("=== Step 0a: Load primary daily CSV ===")
df = pd.read_csv(PRIOR_CSV, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Null counts:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# %% Step 0b: Load and aggregate weather hourly→daily
print("\n=== Step 0b: Weather aggregation ===")
import glob
wx_files = glob.glob(str(WEATHER_PARQUET / "*.parquet")) + glob.glob(str(WEATHER_PARQUET / "*.snappy.parquet"))
if not wx_files:
    wx_files = glob.glob(str(WEATHER_PARQUET.parent.parent / "weatherdata" / "*.parquet"))
print(f"Weather parquet files found: {len(wx_files)}")

if wx_files:
    wx_raw = pd.read_parquet(wx_files[0])
    print(f"Weather raw shape: {wx_raw.shape}")
    print(f"Weather columns: {wx_raw.columns.tolist()}")
    print(f"Weather dtypes:\n{wx_raw.dtypes}")

    # Parse temporal column
    wx_raw['_date'] = pd.to_datetime(wx_raw['Date_time'], errors='coerce').dt.normalize()
    print(f"Weather date range: {wx_raw['_date'].min().date()} to {wx_raw['_date'].max().date()}")
    print(f"NaT in _date: {wx_raw['_date'].isna().sum()}")

    # Identify numeric columns to aggregate
    drop_cols = {'Date_time', 'Location', 'Address', 'Conditions', 'Icon', 'Stations',
                 'Wind_Chill', 'Heat_Index', '_date'}
    numeric_wx_cols = [c for c in wx_raw.columns
                       if c not in drop_cols and wx_raw[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    print(f"Numeric weather cols to aggregate: {numeric_wx_cols}")

    # Aggregate hourly -> daily
    # Note: weather API stores daily totals/extremes repeated per hour, not hourly deltas.
    # Maximum_Temperature / Minimum_Temperature: daily extremes repeated → take max/min.
    # Temperature: hourly observation → take mean for daily mean temp.
    # Precipitation / Snow_depth: daily total repeated per hour → take mean (NOT sum).
    # Wind_Speed: hourly obs → take mean; Wind_Gust: daily max repeated → take max.
    # Cloud_Cover, Relative_Humidity: hourly obs → take mean.
    wx_agg_method = {c: (
        'max' if 'Maximum' in c else
        'min' if 'Minimum' in c else
        'max' if 'Wind_Gust' in c else
        'mean'  # covers Precipitation, Snow_depth, Temperature, Wind_Speed, Cloud_Cover, Humidity
    ) for c in numeric_wx_cols}
    wx_agg = wx_raw.groupby('_date')[numeric_wx_cols].agg(wx_agg_method).reset_index()
    wx_agg.rename(columns={'_date': 'date'}, inplace=True)

    # Rename to wx_ prefix
    wx_rename = {c: f"wx_{c}" for c in numeric_wx_cols}
    wx_agg.rename(columns=wx_rename, inplace=True)

    print(f"Weather aggregated shape: {wx_agg.shape}")
    print(f"Weather aggregated columns: {wx_agg.columns.tolist()}")
    print(f"Weather date range after agg: {wx_agg['date'].min().date()} to {wx_agg['date'].max().date()}")
else:
    print("WARNING: No weather parquet found. Creating empty weather frame.")
    wx_agg = pd.DataFrame(columns=['date'])

# %% Step 0c: Load and aggregate competitor data
print("\n=== Step 0c: Competitor data ===")
vc1_path = COMPETITOR_DIR / "VerlofConcurrent.csv"
try:
    vc1 = pd.read_csv(vc1_path, sep=';', encoding='latin-1')
    print(f"VC1 raw shape: {vc1.shape}")
    print(f"VC1 columns: {vc1.columns.tolist()}")
    print(f"VC1 sample:\n{vc1.head(3)}")

    # Identify date column
    date_cols = [c for c in vc1.columns if 'datum' in c.lower() or 'date' in c.lower()]
    winkel_cols = [c for c in vc1.columns if 'winkel' in c.lower()]
    verlof_cols = [c for c in vc1.columns if 'verlof' in c.lower()]
    print(f"Date cols: {date_cols}, Winkel cols: {winkel_cols}, Verlof cols: {verlof_cols}")

    if date_cols:
        date_col = date_cols[0]
        vc1[date_col] = pd.to_datetime(vc1[date_col], dayfirst=True, errors='coerce')
        vc1_valid = vc1.dropna(subset=[date_col])

        # Count competitors with verlof=1 per date
        if verlof_cols:
            verlof_col = verlof_cols[0]
            # When verlof column exists, count distinct winkels per date
            if winkel_cols:
                winkel_col = winkel_cols[0]
                comp_daily = vc1_valid.groupby(date_col).agg(
                    comp_n_competitors_closed=(winkel_col, 'count'),
                    comp_any_closed=(winkel_col, lambda x: int(len(x) > 0))
                ).reset_index()
            else:
                comp_daily = vc1_valid.groupby(date_col).size().reset_index(name='comp_n_competitors_closed')
                comp_daily['comp_any_closed'] = 1
        else:
            # Each row = one competitor on holiday; count per date
            id_col = winkel_cols[0] if winkel_cols else vc1.columns[0]
            comp_daily = vc1_valid.groupby(date_col).agg(
                comp_n_competitors_closed=(id_col, 'count')
            ).reset_index()
            comp_daily['comp_any_closed'] = 1

        comp_daily.rename(columns={date_col: 'date'}, inplace=True)
        comp_daily['comp_closure_density'] = comp_daily['comp_n_competitors_closed'] / 23.0
        print(f"Competitor daily shape: {comp_daily.shape}")
        print(f"Date range: {comp_daily['date'].min().date()} to {comp_daily['date'].max().date()}")
        print(f"Unique dates: {comp_daily['date'].nunique()}")
        print(f"Max competitors closed on one day: {comp_daily['comp_n_competitors_closed'].max()}")
    else:
        print("WARNING: No date column found in VC1. Creating empty frame.")
        comp_daily = pd.DataFrame(columns=['date'])

except Exception as e:
    print(f"WARNING: Could not load competitor data: {e}. Creating empty frame.")
    comp_daily = pd.DataFrame(columns=['date'])

# %% Step 0d: Left join all sources
print("\n=== Step 0d: Left join ===")
df_joined = df.copy()
print(f"Before joins: {df_joined.shape}")

# Join weather
if 'date' in wx_agg.columns and len(wx_agg) > 0:
    df_joined = df_joined.merge(wx_agg, on='date', how='left')
    print(f"After weather join: {df_joined.shape}")
    wx_cols_added = [c for c in df_joined.columns if c.startswith('wx_')]
    print(f"Weather cols added: {wx_cols_added}")
    wx_null_count = df_joined[wx_cols_added].isnull().any(axis=1).sum() if wx_cols_added else 0
    print(f"Rows with any weather null: {wx_null_count}")
else:
    wx_cols_added = []
    print("No weather joined.")

# Join competitor
if 'date' in comp_daily.columns and len(comp_daily) > 0:
    df_joined = df_joined.merge(comp_daily, on='date', how='left')
    print(f"After competitor join: {df_joined.shape}")
    comp_cols_added = [c for c in df_joined.columns if c.startswith('comp_')]
    print(f"Competitor cols added: {comp_cols_added}")
    # Fill comp cols with 0 for dates outside coverage (structural zero, not missing)
    for cc in comp_cols_added:
        df_joined[cc] = df_joined[cc].fillna(0.0)
    print(f"Competitor nulls after fill: {df_joined[comp_cols_added].isnull().sum().sum()}")
else:
    comp_cols_added = []
    print("No competitor joined.")

# Add competitor_data_available indicator
df_joined['comp_data_available'] = (
    (df_joined['date'] >= pd.Timestamp('2017-07-01')) &
    (df_joined['date'] <= pd.Timestamp('2020-09-15'))
).astype(int)
comp_cols_added.append('comp_data_available')

print(f"\nFinal shape before split: {df_joined.shape}")
print(f"All columns: {df_joined.columns.tolist()}")

# %% Step 3: Handle weather missingness (pre-2015)
print("\n=== Step 3: Impute weather pre-2015 ===")
if wx_cols_added:
    # Fit month-of-year median on train rows for fold 3
    train_mask_f3 = df_joined['date'] <= FOLD3_TRAIN_END
    df_joined['_month'] = df_joined['date'].dt.month

    for wc in wx_cols_added:
        n_null = df_joined[wc].isnull().sum()
        if n_null > 0:
            # Compute month-of-year medians from fold 3 train rows only
            monthly_med = df_joined.loc[train_mask_f3, [wc, '_month']].groupby('_month')[wc].median()
            df_joined[wc] = df_joined.apply(
                lambda row: monthly_med.get(row['_month'], df_joined.loc[train_mask_f3, wc].median())
                if pd.isna(row[wc]) else row[wc], axis=1
            )
            print(f"  {wc}: imputed {n_null} nulls with month-of-year median (fit on fold3 train)")

    df_joined.drop(columns=['_month'], inplace=True)

    remaining_wx_null = df_joined[wx_cols_added].isnull().sum().sum()
    print(f"Weather nulls remaining after imputation: {remaining_wx_null}")

# %% Step 3: Create rolling-origin CV split assignments
print("\n=== Step 3: Rolling-origin CV split ===")

def assign_fold(date, train_end, gap_end):
    if date <= train_end:
        return 'train'
    elif date <= gap_end:
        return 'gap'
    else:
        return 'val'

df_joined['_fold_1'] = df_joined['date'].apply(
    lambda d: assign_fold(d, FOLD1_TRAIN_END, FOLD1_GAP_END)
    if d <= FOLD1_VAL_END else None
)
df_joined['_fold_2'] = df_joined['date'].apply(
    lambda d: assign_fold(d, FOLD2_TRAIN_END, FOLD2_GAP_END)
    if d <= FOLD2_VAL_END else None
)
df_joined['_fold_3'] = df_joined['date'].apply(
    lambda d: assign_fold(d, FOLD3_TRAIN_END, FOLD3_GAP_END)
)

# Primary _split = fold 3 assignment
df_joined['_split'] = df_joined['_fold_3'].str.replace('val', 'validation')

# Compute stats per fold
for fold_name, fold_col, val_start in [
    ('Fold 1', '_fold_1', FOLD1_TRAIN_END + pd.Timedelta(days=90)),
    ('Fold 2', '_fold_2', FOLD2_TRAIN_END + pd.Timedelta(days=90)),
    ('Fold 3 (PRIMARY)', '_fold_3', FOLD3_TRAIN_END + pd.Timedelta(days=90)),
]:
    fold_df = df_joined.dropna(subset=[fold_col])
    if len(fold_df) == 0:
        continue
    train_rows = fold_df[fold_df[fold_col] == 'train']
    gap_rows = fold_df[fold_df[fold_col] == 'gap']
    val_rows = fold_df[fold_df[fold_col] == 'val']
    post_regime_train = train_rows[train_rows['date'] >= REGIME_CHANGE]
    print(f"\n{fold_name}:")
    print(f"  Train: {len(train_rows)} rows ({train_rows['date'].min().date()} to {train_rows['date'].max().date()})")
    print(f"  Gap:   {len(gap_rows)} rows ({gap_rows['date'].min().date() if len(gap_rows)>0 else 'N/A'} to {gap_rows['date'].max().date() if len(gap_rows)>0 else 'N/A'})")
    print(f"  Val:   {len(val_rows)} rows ({val_rows['date'].min().date() if len(val_rows)>0 else 'N/A'} to {val_rows['date'].max().date() if len(val_rows)>0 else 'N/A'})")
    print(f"  Post-regime rows in TRAIN: {len(post_regime_train)}")
    if len(train_rows) > 0:
        print(f"  Train mean bread_qty: {train_rows['bread_quantity'].mean():.2f}")
    if len(val_rows) > 0:
        print(f"  Val mean bread_qty: {val_rows['bread_quantity'].mean():.2f}")
        print(f"  Level gap: {(val_rows['bread_quantity'].mean() - train_rows['bread_quantity'].mean()) / train_rows['bread_quantity'].mean() * 100:.1f}%")

# %% Step 4: Flag leakage risk columns and near-constant columns
print("\n=== Step 4: Column flags ===")
# Identify potential leakage risk columns (same-day demand proxies)
leakage_candidates = [c for c in df_joined.columns
                      if c in ('total_quantity', 'total_revenue', 'bread_revenue', 'num_orders',
                                'payment_mode_1', 'payment_mode_2', 'payment_mode_4',
                                'pistolet_quantity') or
                      ('quantity' in c.lower() and c != 'bread_quantity') or
                      ('revenue' in c.lower()) or
                      ('orders' in c.lower())]
print(f"Leakage risk candidates (same-day demand proxies): {leakage_candidates}")

# Near-constant check
near_constant = []
for col in df_joined.select_dtypes(include=[np.number]).columns:
    if col.startswith('_') or col == 'bread_quantity':
        continue
    vc = df_joined[col].value_counts(normalize=True)
    if len(vc) > 0 and vc.iloc[0] > 0.99:
        near_constant.append((col, f"{vc.index[0]}={vc.iloc[0]*100:.1f}%"))
print(f"Near-constant columns (>99% single value): {near_constant}")

# %% Step 8: Post-transform stats
print("\n=== Step 8: Post-transform stats ===")
stats_cols = ['bread_quantity'] + wx_cols_added[:5] + comp_cols_added[:3]
for sc in stats_cols:
    if sc in df_joined.columns:
        col = df_joined[sc]
        print(f"  {sc}: mean={col.mean():.3f}, std={col.std():.3f}, null={col.isnull().sum()}, "
              f"skew={col.skew():.3f}")

# %% Step 9: Generate plots
print("\n=== Step 9: Plots ===")

# Plot 1: Split visualization
fig, ax = plt.subplots(figsize=(14, 5))
split_colors = {'train': '#2196F3', 'gap': '#9E9E9E', 'validation': '#F44336'}
for split_val, color in split_colors.items():
    mask = df_joined['_split'] == split_val
    ax.scatter(df_joined.loc[mask, 'date'], df_joined.loc[mask, 'bread_quantity'],
               c=color, s=2, alpha=0.6, label=f'{split_val} ({mask.sum()} rows)')
ax.axvline(REGIME_CHANGE, color='orange', linestyle='--', linewidth=1.5, label='Regime change 2020-06-01')
ax.axvline(FOLD3_TRAIN_END, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Fold3 train end')
ax.axvline(FOLD3_GAP_END, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Fold3 val start')
ax.set_title("iteration_003_preprocess — Split Visualization (Fold 3 primary)\nrun_20260908T075709Z")
ax.set_xlabel("Date")
ax.set_ylabel("bread_quantity")
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plot1_path = ARTIFACTS_DIR / "iteration_003_preprocess_split_visualization.png"
plt.savefig(plot1_path, dpi=120)
plt.close()
# Paired CSV
split_csv = df_joined[['date', 'bread_quantity', '_split', '_fold_1', '_fold_2', '_fold_3']].copy()
split_csv.to_csv(ARTIFACTS_DIR / "iteration_003_preprocess_split_visualization.csv", index=False)
print(f"  Saved: {plot1_path}")

# Plot 2: Weather coverage
if wx_cols_added:
    temp_col = next((c for c in wx_cols_added if 'temp' in c.lower() or 'Temp' in c), None)
    precip_col = next((c for c in wx_cols_added if 'precip' in c.lower() or 'Precip' in c), None)
    if temp_col:
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        df_joined.plot(x='date', y=temp_col, ax=axes[0], linewidth=0.5, color='#FF5722')
        axes[0].axvline(pd.Timestamp('2015-01-01'), color='red', linestyle='--',
                        linewidth=1, label='Weather coverage start 2015-01-01')
        axes[0].set_title(f"iteration_003_preprocess — Weather: {temp_col} (imputed pre-2015)")
        axes[0].legend(fontsize=8)
        if precip_col:
            df_joined.plot(x='date', y=precip_col, ax=axes[1], linewidth=0.5, color='#2196F3')
            axes[1].axvline(pd.Timestamp('2015-01-01'), color='red', linestyle='--', linewidth=1)
            axes[1].set_title(f"{precip_col} (imputed pre-2015)")
        plt.tight_layout()
        plot2_path = ARTIFACTS_DIR / "iteration_003_preprocess_weather_coverage.png"
        plt.savefig(plot2_path, dpi=120)
        plt.close()
        wx_plot_df = df_joined[['date'] + wx_cols_added[:4]].copy()
        wx_plot_df.to_csv(ARTIFACTS_DIR / "iteration_003_preprocess_weather_coverage.csv", index=False)
        print(f"  Saved: {plot2_path}")

# Plot 3: Competitor coverage
if comp_cols_added and 'comp_n_competitors_closed' in df_joined.columns:
    fig, ax = plt.subplots(figsize=(14, 4))
    df_joined.plot(x='date', y='comp_n_competitors_closed', ax=ax, linewidth=0.5, color='#9C27B0')
    ax.axvline(pd.Timestamp('2017-07-01'), color='green', linestyle='--', linewidth=1,
               label='Competitor coverage start 2017-07-01')
    ax.axvline(pd.Timestamp('2020-09-15'), color='red', linestyle='--', linewidth=1,
               label='Competitor coverage end 2020-09-15')
    ax.set_title("iteration_003_preprocess — Competitor closures per day\nrun_20260908T075709Z")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plot3_path = ARTIFACTS_DIR / "iteration_003_preprocess_competitor_coverage.png"
    plt.savefig(plot3_path, dpi=120)
    plt.close()
    df_joined[['date', 'comp_n_competitors_closed', 'comp_data_available']].to_csv(
        ARTIFACTS_DIR / "iteration_003_preprocess_competitor_coverage.csv", index=False)
    print(f"  Saved: {plot3_path}")

# %% Step 11: Write output CSV
print("\n=== Step 11: Write output ===")
df_joined.to_csv(OUTPUT_CSV, index=False)
print(f"Output CSV: {OUTPUT_CSV}")
print(f"Output shape: {df_joined.shape}")
print(f"Output columns: {df_joined.columns.tolist()}")

# %% Final summary
print("\n=== PREPROCESS SUMMARY ===")
split_summary = df_joined['_split'].value_counts()
print(f"Split summary (fold 3):\n{split_summary}")
print(f"Total rows: {len(df_joined)}")
print(f"Total columns: {len(df_joined.columns)}")

# Fold 3 specific stats
f3_train = df_joined[df_joined['_split'] == 'train']
f3_val = df_joined[df_joined['_split'] == 'validation']
f3_gap = df_joined[df_joined['_split'] == 'gap']
print(f"\nFold 3 (PRIMARY split):")
print(f"  Train: {len(f3_train)} rows, mean={f3_train['bread_quantity'].mean():.2f}")
print(f"  Gap: {len(f3_gap)} rows")
print(f"  Val: {len(f3_val)} rows, mean={f3_val['bread_quantity'].mean():.2f}")
print(f"  Level gap (val-train)/train: {(f3_val['bread_quantity'].mean() - f3_train['bread_quantity'].mean()) / f3_train['bread_quantity'].mean() * 100:.1f}%")
post_regime_in_train = f3_train[f3_train['date'] >= REGIME_CHANGE]
print(f"  Post-regime rows in train: {len(post_regime_in_train)} (should be > 0!)")

# Per-horizon scoreable val rows
print(f"\nVal rows scoreable per horizon (targets available within dataset):")
for h in [1, 7, 14, 30, 60, 90]:
    cutoff = df_joined['date'].max() - pd.Timedelta(days=h)
    scoreable = f3_val[f3_val['date'] <= cutoff]
    print(f"  h={h:2d}: {len(scoreable)} scoreable val rows")

print("\nPreprocess complete.")
