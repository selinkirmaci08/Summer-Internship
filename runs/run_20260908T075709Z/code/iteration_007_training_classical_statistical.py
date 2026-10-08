# %% [markdown]
# # Iteration 007 — Training: Classical Statistical (SARIMA / ARIMA)
# Run: run_20260908T075709Z
# Mode: initial — classical_statistical ts_approach slot (required per training.md §2b-ts)
# ts_approach: classical_statistical
# Reads:  data/iteration_005_feature_engineering.csv
#         state/current.json (for baseline floors, LightGBM val_mae comparison)
# Writes: artifacts/training/sarima/  (model pkl, forecasts, plots)
#         history/iteration_007_training_classical_statistical.json
#         state/current.json (updated)
#
# Methodology:
#   - Fit on fold-3 training rows (bread_quantity as plain time series, sorted by date)
#   - Evaluate on fold-3 val rows: for val row i at horizon h, target =
#       full_series[n_train + n_gap + i + h], forecast index = n_gap + i + h (0-based)
#   - Seasonal period m=6 (6 trading days/week in pre-regime training; dominant_acf_lag=6 from EDA)
#   - ARIMA grid (ablation): (p,1,q) for p in {0,1,2}, q in {0,1,2} -> pick by AIC
#   - SARIMA grid (m=6):  4 candidates -> pick by AIC
#   - Pre-COVID SARIMA: same orders, fitted on data up to 2020-02-28 (COVID ablation)
#   - Champion = lowest pooled val MAE across ARIMA / SARIMA / pre-COVID variants
#   - MASE denominator: seasonal naive MAE on training (m=6)
#   - Compare all horizons to LightGBM val_mae_by_horizon from iteration_006

# %% Imports
import json, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import joblib

warnings.filterwarnings('ignore')

# %% Config
RUN_DIR  = Path('C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z')
FE_CSV   = RUN_DIR / 'data' / 'iteration_005_feature_engineering.csv'
ART_DIR  = RUN_DIR / 'artifacts' / 'training' / 'sarima'
ART_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS   = [1, 7, 14, 30, 60, 90]
TARGET     = 'bread_quantity'
ITER_NUM   = 7
ITER_ID    = f'iteration_{ITER_NUM:03d}_training_classical_statistical'
SEASONAL_M = 6          # weekly pattern in trading-day index (6 trading days/week pre-regime)
P_SEASONAL = 7          # calendar period for MASE denominator (same-DOW = 7 calendar days but 6 trading)

# Baseline floors (from iteration_004_baseline.json, fold-3 primary)
BASELINE_FLOOR = {1: 31.263, 7: 31.263, 14: 32.846, 30: 35.913, 60: 40.042, 90: 42.143}

# LightGBM val MAE from iteration_006 (best attempt: regression_l1, global val early stopping)
LGBM_VAL_MAE = {1: 39.360, 7: 39.391, 14: 40.189, 30: 43.227, 60: 50.080, 90: 79.218}

# COVID lockdown period (for pre-COVID ablation model cutoff)
COVID_LOCKDOWN_START = pd.Timestamp('2020-03-18')

print(f"[Config] RUN_DIR: {RUN_DIR}")
print(f"[Config] SEASONAL_M={SEASONAL_M}, HORIZONS={HORIZONS}")

# %% [Step 1] Load data
print("\n[Step 1] Loading data...")
df = pd.read_csv(FE_CSV, parse_dates=['date']).sort_values('date').reset_index(drop=True)

# Use fold-3 split columns
train_mask = df['_split'] == 'train'
gap_mask   = df['_split'] == 'gap'
val_mask   = df['_split'] == 'validation'

n_train = train_mask.sum()
n_gap   = gap_mask.sum()
n_val   = val_mask.sum()
print(f"  n_train={n_train}, n_gap={n_gap}, n_val={n_val}")

# Full series (all 2897 rows, sorted by date) for target lookup
full_series  = df[TARGET].values.astype(float)
train_series = full_series[:n_train]               # 2713 values
covid_cutoff_idx = (df['date'] < COVID_LOCKDOWN_START).sum()
precovid_series  = full_series[:covid_cutoff_idx]  # pre-COVID training series
print(f"  Full series length: {len(full_series)}")
print(f"  Train series length: {len(train_series)}")
print(f"  Pre-COVID series length: {len(precovid_series)} (up to {df.loc[covid_cutoff_idx-1,'date'].date()})")

# Forecast needs to cover: n_gap + n_val rows + max(HORIZONS) extra steps
N_FORECAST_FULL     = n_gap + n_val + max(HORIZONS) + 5   # from end of fold-3 training
N_FORECAST_PRECOVID = (n_train - covid_cutoff_idx) + n_gap + n_val + max(HORIZONS) + 5  # from COVID cutoff
print(f"  N_FORECAST (full training): {N_FORECAST_FULL}")
print(f"  N_FORECAST (pre-COVID):     {N_FORECAST_PRECOVID}")

# Scoreable val rows per horizon
scoreable = {h: int((df.loc[val_mask, f'{TARGET}_h{h}'].notna()).sum()) for h in HORIZONS}
print(f"  Scoreable val rows by horizon: {scoreable}")

# %% [Step 2] ADF stationarity test
print("\n[Step 2] ADF stationarity test on training series...")
adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(train_series, autolag='AIC')
print(f"  ADF statistic: {adf_stat:.4f}")
print(f"  ADF p-value:   {adf_pval:.6f}")
print(f"  Critical values: {adf_crit}")
d_recommendation = 0 if adf_pval < 0.05 else 1
stat_label = 'STATIONARY (p<0.05)' if adf_pval < 0.05 else 'NON-STATIONARY'
print(f'  Stationarity: {stat_label}  ->  d={d_recommendation}')
# Use d=1 regardless — downward level trend warrants first-difference for SARIMA stability
D_ORDER = 1
print(f"  Using d=1 (level trend visible over 10-year series)")

# %% [Step 3] MASE denominator (seasonal naive on training, m=6)
print("\n[Step 3] Computing MASE denominator...")
naive_errors = np.abs(train_series[SEASONAL_M:] - train_series[:-SEASONAL_M])
mase_denom   = np.mean(naive_errors)
print(f"  Seasonal naive MAE (m={SEASONAL_M}) on training: {mase_denom:.3f}")

# %% [Step 4] Helper: fit + forecast SARIMAX model
def fit_forecast(series, order, seasonal_order, n_forecast, label):
    """Fit SARIMAX on series, forecast n_forecast steps. Returns (result, forecast_arr, aic) or None on failure."""
    try:
        t0 = time.time()
        model  = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                         trend='n', enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False, method='lbfgs', maxiter=200)
        start  = int(result.nobs)
        end    = int(result.nobs + n_forecast - 1)
        preds  = result.predict(start=start, end=end)
        fc_arr = np.clip(np.asarray(preds, dtype=float), 0, None)
        print(f"  {label}: AIC={result.aic:.2f}, time={time.time()-t0:.1f}s")
        return result, fc_arr, result.aic
    except Exception as e:
        print(f"  {label}: FAILED ({e})")
        return None, None, np.inf

# %% [Step 5] Plain ARIMA grid search (ablation — no seasonal component)
print("\n[Step 5] Plain ARIMA grid search (d=1)...")
ARIMA_CANDIDATES = [(p, D_ORDER, q) for p in range(3) for q in range(3)]
best_arima = {'aic': np.inf, 'order': None, 'result': None, 'fc': None}
for order in ARIMA_CANDIDATES:
    result, fc, aic = fit_forecast(
        train_series, order=order, seasonal_order=(0, 0, 0, 0),
        n_forecast=N_FORECAST_FULL, label=f'ARIMA{order}'
    )
    if aic < best_arima['aic']:
        best_arima.update({'aic': aic, 'order': order, 'result': result, 'fc': fc})

print(f"\n  Best ARIMA: {best_arima['order']}, AIC={best_arima['aic']:.2f}")

# %% [Step 6] SARIMA grid search (m=6)
print(f"\n[Step 6] SARIMA grid search (m={SEASONAL_M})...")
SARIMA_CANDIDATES = [
    ((1, 1, 1), (0, 1, 1, SEASONAL_M)),
    ((1, 1, 1), (1, 0, 0, SEASONAL_M)),
    ((1, 1, 0), (0, 1, 1, SEASONAL_M)),
    ((0, 1, 1), (0, 1, 1, SEASONAL_M)),
    ((1, 1, 1), (1, 1, 0, SEASONAL_M)),
    ((2, 1, 1), (0, 1, 1, SEASONAL_M)),
]
best_sarima = {'aic': np.inf, 'order': None, 'seasonal_order': None, 'result': None, 'fc': None}
for (order, seasonal_order) in SARIMA_CANDIDATES:
    result, fc, aic = fit_forecast(
        train_series, order=order, seasonal_order=seasonal_order,
        n_forecast=N_FORECAST_FULL, label=f'SARIMA{order}x{seasonal_order}'
    )
    if aic < best_sarima['aic']:
        best_sarima.update({'aic': aic, 'order': order, 'seasonal_order': seasonal_order,
                            'result': result, 'fc': fc})

print(f"\n  Best SARIMA (full training): {best_sarima['order']}x{best_sarima['seasonal_order']}, AIC={best_sarima['aic']:.2f}")

# %% [Step 7] Pre-COVID SARIMA (ablation: exclude COVID lockdown from training)
print(f"\n[Step 7] Pre-COVID SARIMA ablation (training cut at {COVID_LOCKDOWN_START.date()})...")
best_precovid = {'aic': np.inf, 'order': None, 'seasonal_order': None, 'result': None, 'fc': None,
                 'n_precovid': len(precovid_series), 'n_forecast': N_FORECAST_PRECOVID}

SARIMA_ABLATION = [
    ((1, 1, 1), (0, 1, 1, SEASONAL_M)),
    ((1, 1, 0), (0, 1, 1, SEASONAL_M)),
    ((0, 1, 1), (0, 1, 1, SEASONAL_M)),
    ((1, 1, 1), (1, 1, 0, SEASONAL_M)),
]
for (order, seasonal_order) in SARIMA_ABLATION:
    result, fc, aic = fit_forecast(
        precovid_series, order=order, seasonal_order=seasonal_order,
        n_forecast=N_FORECAST_PRECOVID, label=f'PreCOVID-SARIMA{order}x{seasonal_order}'
    )
    if aic < best_precovid['aic']:
        best_precovid.update({'aic': aic, 'order': order, 'seasonal_order': seasonal_order,
                              'result': result, 'fc': fc})

print(f"\n  Best pre-COVID SARIMA: {best_precovid['order']}x{best_precovid['seasonal_order']}, AIC={best_precovid['aic']:.2f}")

# %% [Step 8] Per-horizon evaluation
print("\n[Step 8] Per-horizon evaluation on fold-3 val rows...")

def compute_metrics(actuals, preds, mase_denom):
    actuals = np.array(actuals, dtype=float)
    preds   = np.array(preds,   dtype=float)
    n = len(actuals)
    mae   = float(np.mean(np.abs(actuals - preds)))
    rmse  = float(np.sqrt(np.mean((actuals - preds)**2)))
    mask  = actuals != 0
    smape = float(200 * np.mean(np.abs(actuals[mask] - preds[mask]) / (np.abs(actuals[mask]) + np.abs(preds[mask])))) if mask.any() else np.nan
    mase  = mae / mase_denom if mase_denom > 0 else np.nan
    return {'n': n, 'mae': round(mae, 4), 'rmse': round(rmse, 4),
            'smape': round(smape, 4) if not np.isnan(smape) else None,
            'mase': round(mase, 4) if not np.isnan(mase) else None}

def evaluate_forecast(fc_arr, n_train_len, n_gap_len, n_val_len, full_series_arr, horizons, mase_denom, label):
    """
    For val row i (0-based) at horizon h:
      target_idx  = n_train_len + n_gap_len + i + h  (into full_series_arr)
      forecast_idx = n_gap_len  + i + h              (0-based into fc_arr)
    """
    results = {}
    for h in horizons:
        actuals, preds = [], []
        for i in range(n_val_len):
            target_idx  = n_train_len + n_gap_len + i + h
            forecast_idx = n_gap_len  + i + h
            if target_idx >= len(full_series_arr):
                continue
            if fc_arr is None or forecast_idx >= len(fc_arr):
                continue
            actuals.append(full_series_arr[target_idx])
            preds.append(fc_arr[forecast_idx])
        if len(actuals) == 0:
            results[h] = {'n': 0, 'mae': np.nan, 'rmse': np.nan, 'smape': None, 'mase': None}
        else:
            results[h] = compute_metrics(actuals, preds, mase_denom)
        print(f"  {label} h={h:2d}: n={results[h]['n']:3d}, MAE={results[h]['mae']:.3f}  "
              f"(floor={BASELINE_FLOOR[h]:.3f}, lgbm={LGBM_VAL_MAE[h]:.3f})")
    return results

print("\n--- ARIMA ---")
arima_metrics = evaluate_forecast(
    best_arima['fc'], n_train, n_gap, n_val, full_series, HORIZONS, mase_denom, 'ARIMA')

print("\n--- SARIMA (full training) ---")
sarima_metrics = evaluate_forecast(
    best_sarima['fc'], n_train, n_gap, n_val, full_series, HORIZONS, mase_denom, 'SARIMA')

print("\n--- SARIMA (pre-COVID) ---")
# For pre-COVID: the forecast starts from covid_cutoff_idx (not n_train)
# n_gap_precovid covers from COVID cutoff -> through real training end -> gap -> val
# offset = n_train - covid_cutoff_idx (rows between cutoff and fold-3 train end)
n_train_precovid  = len(precovid_series)
n_gap_precovid    = n_train - n_train_precovid + n_gap  # training tail + gap
precovid_metrics  = evaluate_forecast(
    best_precovid['fc'], n_train_precovid, n_gap_precovid, n_val, full_series, HORIZONS, mase_denom, 'PreCOVID-SARIMA')

# %% [Step 9] Summary table and champion selection
print("\n[Step 9] Summary table...")

header = f"{'h':>4} | {'ARIMA':>8} | {'SARIMA':>8} | {'PreCOVID':>8} | {'LGBM':>8} | {'Floor':>8} | {'Best-TS':>8} | BeatFloor"
print(header)
print('-' * len(header))

champion_by_h = {}
ts_val_mae = {}

for h in HORIZONS:
    mae_a = arima_metrics[h]['mae']
    mae_s = sarima_metrics[h]['mae']
    mae_p = precovid_metrics[h]['mae']
    mae_l = LGBM_VAL_MAE[h]
    floor = BASELINE_FLOOR[h]

    best_ts_mae = min(mae_a, mae_s, mae_p)
    beat_floor  = best_ts_mae < floor
    ts_val_mae[h] = best_ts_mae

    if best_ts_mae == mae_a:
        best_ts_name = f'ARIMA{best_arima["order"]}'
    elif best_ts_mae == mae_s:
        best_ts_name = f'SARIMA'
    else:
        best_ts_name = 'PreCOVID-SARIMA'

    champion_by_h[h] = {
        'model': best_ts_name,
        'val_mae': round(best_ts_mae, 4),
        'beats_floor': beat_floor,
        'beats_lgbm': best_ts_mae < mae_l
    }

    print(f"{h:>4} | {mae_a:>8.3f} | {mae_s:>8.3f} | {mae_p:>8.3f} | {mae_l:>8.3f} | {floor:>8.3f} | {best_ts_mae:>8.3f} | {'YES' if beat_floor else 'NO'}")

pooled_ts   = float(np.mean(list(ts_val_mae.values())))
pooled_lgbm = float(np.mean(list(LGBM_VAL_MAE.values())))
pooled_floor = float(np.mean(list(BASELINE_FLOOR.values())))
print(f"\n  Pooled TS MAE:   {pooled_ts:.4f}")
print(f"  Pooled LGBM MAE: {pooled_lgbm:.4f}")
print(f"  Pooled floor:    {pooled_floor:.4f}")
print(f"\n  TS beats floor  (per-horizon): {sum(v['beats_floor'] for v in champion_by_h.values())}/{len(HORIZONS)}")
print(f"  TS beats LGBM   (per-horizon): {sum(v['beats_lgbm']  for v in champion_by_h.values())}/{len(HORIZONS)}")

# %% [Step 10] Save best models (joblib)
print("\n[Step 10] Saving models...")
model_paths = {}

if best_arima['result'] is not None:
    p = ART_DIR / 'arima_best.pkl'
    joblib.dump({'result': best_arima['result'], 'order': best_arima['order'],
                 'seasonal_order': (0,0,0,0), 'model_type': 'ARIMA'}, str(p))
    model_paths['arima'] = str(p.relative_to(RUN_DIR))
    print(f"  Saved ARIMA -> {model_paths['arima']}")

if best_sarima['result'] is not None:
    p = ART_DIR / 'sarima_full_best.pkl'
    joblib.dump({'result': best_sarima['result'], 'order': best_sarima['order'],
                 'seasonal_order': best_sarima['seasonal_order'], 'model_type': 'SARIMA'}, str(p))
    model_paths['sarima_full'] = str(p.relative_to(RUN_DIR))
    print(f"  Saved SARIMA (full) -> {model_paths['sarima_full']}")

if best_precovid['result'] is not None:
    p = ART_DIR / 'sarima_precovid_best.pkl'
    joblib.dump({'result': best_precovid['result'], 'order': best_precovid['order'],
                 'seasonal_order': best_precovid['seasonal_order'],
                 'model_type': 'SARIMA_precovid', 'training_cutoff': str(COVID_LOCKDOWN_START.date())}, str(p))
    model_paths['sarima_precovid'] = str(p.relative_to(RUN_DIR))
    print(f"  Saved pre-COVID SARIMA -> {model_paths['sarima_precovid']}")

# Save forecast arrays
fc_df = pd.DataFrame({
    'step':       list(range(N_FORECAST_FULL)),
    'arima_fc':   best_arima['fc'] if best_arima['fc'] is not None else [np.nan]*N_FORECAST_FULL,
    'sarima_fc':  best_sarima['fc'] if best_sarima['fc'] is not None else [np.nan]*N_FORECAST_FULL,
})
fc_path = ART_DIR / 'forecasts_full_training.csv'
fc_df.to_csv(fc_path, index=False)
print(f"  Saved forecasts -> {fc_path.relative_to(RUN_DIR)}")

# %% [Step 11] Plots
print("\n[Step 11] Generating plots...")

# Plot 1: training series with COVID period highlighted
fig, ax = plt.subplots(figsize=(14, 5))
train_dates = df.loc[train_mask, 'date'].values
ax.plot(train_dates, train_series, lw=0.6, color='steelblue', label='Training series')
covid_dates = (df['date'] >= '2020-03-18') & (df['date'] <= '2020-05-03') & train_mask
if covid_dates.any():
    ax.axvspan(pd.Timestamp('2020-03-18'), pd.Timestamp('2020-05-03'),
               alpha=0.25, color='red', label='COVID lockdown')
ax.axvline(pd.Timestamp(str(COVID_LOCKDOWN_START.date())), color='orange', ls='--', lw=1, label='Pre-COVID cutoff')
ax.set_title('Training series — fold-3 train (with COVID highlight)')
ax.set_xlabel('Date'); ax.set_ylabel('bread_quantity')
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(str(ART_DIR / 'plot1_training_series.png'), dpi=120)
plt.close()
print("  Saved plot1_training_series.png")

# Plot 2: Forecast comparison (first 200 steps of full-training SARIMA and ARIMA)
n_plot = min(200, N_FORECAST_FULL)
fig, ax = plt.subplots(figsize=(14, 5))
steps = list(range(n_plot))
if best_arima['fc'] is not None:
    ax.plot(steps, best_arima['fc'][:n_plot], lw=1.0, label=f'ARIMA{best_arima["order"]}', color='darkorange')
if best_sarima['fc'] is not None:
    ax.plot(steps, best_sarima['fc'][:n_plot], lw=1.2, label=f'SARIMA (full)', color='green')
# Actual future values (gap + val)
future_actual = full_series[n_train:n_train + n_plot]
future_steps  = list(range(len(future_actual)))[:n_plot]
ax.plot(future_steps, future_actual, lw=0.8, color='steelblue', ls='--', label='Actual (gap+val)')
ax.axvline(n_gap, color='grey', ls=':', lw=1, label='Val start')
ax.set_title('Forecast vs Actual — first 200 steps from training end')
ax.set_xlabel('Steps ahead'); ax.set_ylabel('bread_quantity')
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(str(ART_DIR / 'plot2_forecast_vs_actual.png'), dpi=120)
plt.close()
print("  Saved plot2_forecast_vs_actual.png")

# Plot 3: Val MAE comparison across horizons
fig, ax = plt.subplots(figsize=(9, 5))
x = list(HORIZONS)
w = 0.2
positions = {k: [xi + i*w for xi in x] for i, k in enumerate(['arima', 'sarima', 'precovid', 'lgbm', 'floor'])}
vals = {
    'arima':   [arima_metrics[h]['mae'] for h in HORIZONS],
    'sarima':  [sarima_metrics[h]['mae'] for h in HORIZONS],
    'precovid':[precovid_metrics[h]['mae'] for h in HORIZONS],
    'lgbm':    [LGBM_VAL_MAE[h] for h in HORIZONS],
    'floor':   [BASELINE_FLOOR[h] for h in HORIZONS],
}
colors = {'arima': 'darkorange', 'sarima': 'green', 'precovid': 'purple', 'lgbm': 'cornflowerblue', 'floor': 'firebrick'}
labels = {
    'arima': f'ARIMA{best_arima["order"]}',
    'sarima': f'SARIMA (full)',
    'precovid': 'SARIMA (pre-COVID)',
    'lgbm': 'LightGBM (iter006)',
    'floor': 'Baseline floor',
}
for k in ['arima', 'sarima', 'precovid', 'lgbm', 'floor']:
    ax.bar([p + (list(['arima', 'sarima', 'precovid', 'lgbm', 'floor']).index(k) * w)
             for p in x], vals[k], width=w, label=labels[k], color=colors[k], alpha=0.8)
ax.set_xlabel('Horizon (days)'); ax.set_ylabel('Val MAE')
ax.set_xticks([xi + 2*w for xi in x]); ax.set_xticklabels([str(h) for h in HORIZONS])
ax.set_title('Val MAE by horizon: ARIMA vs SARIMA vs pre-COVID vs LGBM vs Floor')
ax.legend(fontsize=7); plt.tight_layout()
plt.savefig(str(ART_DIR / 'plot3_mae_comparison.png'), dpi=120)
plt.close()
print("  Saved plot3_mae_comparison.png")

# Plot 4: SARIMA residuals on training set (diagnostic)
if best_sarima['result'] is not None:
    try:
        resid = best_sarima['result'].resid
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(resid, lw=0.6, color='steelblue')
        axes[0].axhline(0, color='red', ls='--', lw=0.8)
        axes[0].set_title('SARIMA training residuals'); axes[0].set_xlabel('Time index')
        axes[1].hist(resid, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        axes[1].set_title('Residual distribution')
        plt.tight_layout()
        plt.savefig(str(ART_DIR / 'plot4_sarima_residuals.png'), dpi=120)
        plt.close()
        print("  Saved plot4_sarima_residuals.png")
    except Exception as e:
        print(f"  Plot 4 skipped: {e}")

# %% [Step 12] Build handoff JSON
print("\n[Step 12] Writing handoff JSON...")

# Determine best model for state update
best_overall_mae = min(pooled_ts, pooled_lgbm)
champion_approach = 'classical_statistical' if pooled_ts <= pooled_lgbm else 'ml_as_regression'

handoff = {
    "iteration": ITER_NUM,
    "node": "training",
    "timestamp": pd.Timestamp.now().isoformat(),
    "status": "complete",
    "ts_approach": "classical_statistical",
    "ts_approach_gap": None,  # both approaches now filled
    "mode": "initial",
    "input_csv": str(FE_CSV.relative_to(RUN_DIR)),
    "seasonal_period_m": SEASONAL_M,
    "adf_test": {
        "statistic": round(float(adf_stat), 4),
        "pvalue": round(float(adf_pval), 6),
        "d_order": D_ORDER,
        "conclusion": "stationary_with_trend_d1" if adf_pval < 0.05 else "non_stationary_d1"
    },
    "mase_denom": round(float(mase_denom), 4),
    "arima": {
        "best_order": list(best_arima['order']) if best_arima['order'] else None,
        "aic": round(float(best_arima['aic']), 2) if best_arima['aic'] < np.inf else None,
        "val_mae_by_horizon": {str(h): arima_metrics[h]['mae'] for h in HORIZONS},
        "model_path": model_paths.get('arima')
    },
    "sarima_full": {
        "best_order": list(best_sarima['order']) if best_sarima['order'] else None,
        "best_seasonal_order": list(best_sarima['seasonal_order']) if best_sarima['seasonal_order'] else None,
        "aic": round(float(best_sarima['aic']), 2) if best_sarima['aic'] < np.inf else None,
        "val_mae_by_horizon": {str(h): sarima_metrics[h]['mae'] for h in HORIZONS},
        "model_path": model_paths.get('sarima_full')
    },
    "sarima_precovid": {
        "best_order": list(best_precovid['order']) if best_precovid['order'] else None,
        "best_seasonal_order": list(best_precovid['seasonal_order']) if best_precovid['seasonal_order'] else None,
        "aic": round(float(best_precovid['aic']), 2) if best_precovid['aic'] < np.inf else None,
        "training_cutoff": str(COVID_LOCKDOWN_START.date()),
        "val_mae_by_horizon": {str(h): precovid_metrics[h]['mae'] for h in HORIZONS},
        "model_path": model_paths.get('sarima_precovid')
    },
    "champion_ts_by_horizon": {
        str(h): champion_by_h[h] for h in HORIZONS
    },
    "lgbm_val_mae_by_horizon": {str(h): LGBM_VAL_MAE[h] for h in HORIZONS},
    "baseline_floor_by_horizon": {str(h): BASELINE_FLOOR[h] for h in HORIZONS},
    "pooled_ts_val_mae": round(pooled_ts, 4),
    "pooled_lgbm_val_mae": round(pooled_lgbm, 4),
    "pooled_baseline_floor": round(pooled_floor, 4),
    "ts_beats_floor_count": sum(v['beats_floor'] for v in champion_by_h.values()),
    "ts_beats_lgbm_count":  sum(v['beats_lgbm']  for v in champion_by_h.values()),
    "champion_approach": champion_approach,
    "finding": (
        "SARIMA/ARIMA beats LightGBM pooled MAE and is the candidate champion" if pooled_ts < pooled_lgbm
        else "LightGBM beats SARIMA pooled MAE; classical TS slot fulfilled as ablation"
    ),
    "durable_rules": [
        "classical_statistical slot now filled — ts_approach_gap=null",
        "Both ml_as_regression (iter006) and classical_statistical (iter007) fitted — run may now stop if floor beat",
        "Pre-COVID SARIMA tested as ablation (training truncated at 2020-03-18)",
        "All SARIMA fitted on raw bread_quantity (no FE features) as pure time series models",
        "Seasonal period m=6 (6 trading days/week pre-regime; dominant_acf_lag=6 from EDA)"
    ],
    "clarifications": [
        {
            "id": "CQ-TS-001",
            "severity": "non_blocking",
            "question": "SARIMA uses m=6 (pre-regime 6 trading days/week). Post-regime validation (Oct-Dec 2020) has 7 trading days/week. Should SARIMA be retrained with m=7 or m=6.7 (post-regime period) in future iterations?",
            "recommendation": "Refit with m=7 if post-regime performance is unsatisfactory; current m=6 is the dominant historical pattern."
        },
        {
            "id": "CQ-TS-002",
            "severity": "non_blocking",
            "question": "Hyperparameter search for LightGBM not yet done (iter006 ts_approach_gap remained). Now that classical slot is filled, Optuna search (n_trials>=50) should run before run can close.",
            "recommendation": "Next iteration: LightGBM Optuna search. Champions from both approaches compete after search completes."
        }
    ],
    "next_iteration_contract": {
        "node": "training",
        "iteration": 8,
        "why_this_node": "Hyperparameter search (Optuna n_trials>=50) required for LightGBM before run may close per training.md guardrail. Both TS approaches now filled.",
        "read_this_first": "Run Optuna hyperparameter search for LightGBM on each horizon. Compare final LightGBM to SARIMA champion from iteration_007. Select per-horizon champions. Document stop recommendation in next_iteration_contract."
    },
    "artifacts": {
        "plots": [
            "artifacts/training/sarima/plot1_training_series.png",
            "artifacts/training/sarima/plot2_forecast_vs_actual.png",
            "artifacts/training/sarima/plot3_mae_comparison.png",
            "artifacts/training/sarima/plot4_sarima_residuals.png"
        ],
        "models": list(model_paths.values()),
        "forecasts": f"artifacts/training/sarima/forecasts_full_training.csv"
    }
}

hist_path = RUN_DIR / 'history' / f'{ITER_ID}.json'
with open(str(hist_path), 'w', encoding='utf-8') as f:
    json.dump(handoff, f, indent=2)
print(f"  Handoff JSON -> {hist_path.relative_to(RUN_DIR)}")

# %% [Step 13] Update state/current.json
print("\n[Step 13] Updating state/current.json...")
state_path = RUN_DIR / 'state' / 'current.json'
with open(str(state_path), 'r', encoding='utf-8') as f:
    state = json.load(f)

cur = state['current']

# Classical TS val MAE (best per horizon)
ts_mae_dict = {h: champion_by_h[h]['val_mae'] for h in HORIZONS}
cur['ts_val_mae_by_horizon'] = {str(h): ts_mae_dict[h] for h in HORIZONS}
cur['ts_approach_gap']       = None  # both approaches now filled

# Combined beats-floor: TS OR LightGBM beats floor per horizon
combined_beats = {
    h: (champion_by_h[h]['beats_floor'] or cur.get('beats_floor_by_horizon', {}).get(str(h), False))
    for h in HORIZONS
}
cur['ts_beats_floor_by_horizon']  = {str(h): champion_by_h[h]['beats_floor'] for h in HORIZONS}
cur['combined_beats_floor_by_horizon'] = {str(h): combined_beats[h] for h in HORIZONS}

# Update latest_model_paths with SARIMA champion paths
if model_paths:
    cur['latest_model_paths_sarima'] = model_paths

# Champion approach (pooled)
cur['champion_approach_so_far'] = champion_approach
cur['pooled_ts_val_mae']  = round(pooled_ts, 4)
cur['pooled_lgbm_val_mae'] = cur.get('pooled_val_mae', round(pooled_lgbm, 4))

# Update next_iteration_contract
cur['next_iteration_contract'] = handoff['next_iteration_contract']

# History entry
state['history'].append({
    "iteration": ITER_NUM,
    "node": "training",
    "timestamp": pd.Timestamp.now().isoformat()[:10],
    "status": "complete",
    "handoff_check": "pending",
    "summary": (
        f"Classical TS (SARIMA/ARIMA) initial pass. "
        f"Pooled val MAE: SARIMA={pooled_ts:.4f}, LGBM={pooled_lgbm:.4f}, Floor={pooled_floor:.4f}. "
        f"TS beats floor: {sum(v['beats_floor'] for v in champion_by_h.values())}/{len(HORIZONS)}. "
        f"TS beats LGBM: {sum(v['beats_lgbm'] for v in champion_by_h.values())}/{len(HORIZONS)}. "
        f"ts_approach_gap=None (both approaches filled). "
        f"Next: LightGBM Optuna hyperparameter search."
    ),
    "full_output_path": str(hist_path.relative_to(RUN_DIR)),
    "report_html": None,
    "key_outputs": {
        "arima_order": list(best_arima['order']) if best_arima['order'] else None,
        "sarima_order": list(best_sarima['order']) if best_sarima['order'] else None,
        "sarima_seasonal_order": list(best_sarima['seasonal_order']) if best_sarima['seasonal_order'] else None,
        "pooled_ts_val_mae": round(pooled_ts, 4),
        "pooled_lgbm_val_mae": round(pooled_lgbm, 4),
        "ts_beats_floor_count": sum(v['beats_floor'] for v in champion_by_h.values()),
        "ts_beats_lgbm_count":  sum(v['beats_lgbm'] for v in champion_by_h.values()),
        "champion_approach": champion_approach,
        "ts_approach_gap": None
    }
})

with open(str(state_path), 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2)
print(f"  state/current.json updated")

# %% [Step 14] Print final summary
print("\n" + "="*70)
print("SUMMARY — Iteration 007: Classical Statistical Training")
print("="*70)
print(f"{'h':>4} | {'ARIMA':>8} | {'SARIMA':>8} | {'PreCOVID':>8} | {'LGBM':>8} | {'Floor':>8} | BeatFloor | BeatLGBM")
print('-'*80)
for h in HORIZONS:
    ma = arima_metrics[h]['mae']
    ms = sarima_metrics[h]['mae']
    mp = precovid_metrics[h]['mae']
    ml = LGBM_VAL_MAE[h]
    mf = BASELINE_FLOOR[h]
    bt = champion_by_h[h]['beats_floor']
    bl = champion_by_h[h]['beats_lgbm']
    print(f"{h:>4} | {ma:>8.3f} | {ms:>8.3f} | {mp:>8.3f} | {ml:>8.3f} | {mf:>8.3f} | {'YES':>9} | {'YES' if bl else 'NO':>8}" if bt else
          f"{h:>4} | {ma:>8.3f} | {ms:>8.3f} | {mp:>8.3f} | {ml:>8.3f} | {mf:>8.3f} | {'NO':>9} | {'YES' if bl else 'NO':>8}")
print(f"\nPooled TS MAE:   {pooled_ts:.4f}")
print(f"Pooled LGBM MAE: {pooled_lgbm:.4f}")
print(f"Pooled Floor:    {pooled_floor:.4f}")
print(f"\nChampion approach so far: {champion_approach}")
print(f"ts_approach_gap: None  (both approaches filled)")
print(f"Next: LightGBM Optuna hyperparameter search (iteration 008)")
print("\nDone.")
