# %% [markdown]
# # Iteration 004 — Training (baseline mode) — Revised
# Run: run_20260908T075709Z
# Reads:  data/iteration_003_preprocess.csv (2897 rows × 35 cols)
# Writes: artifacts/training/iteration_004_baseline_mae_by_horizon.png  + .csv
#         artifacts/training/iteration_004_baseline_naive_predictors.png + .csv
#         history/iteration_004_baseline.json
#
# Part A: Score expanded candidate grid on validation rows → baseline_floor (per horizon)
#         Same 22-candidate set as Part B, scored on full ts (forecaster sees all data ≤ origin)
# Part B: Search same candidate grid on fold-3 train rows only → naive_predictor_features specs
# h=90 uses all 3 folds (fold 3 has only 5 scoreable val rows)

# %% Imports and configuration
import json
import warnings
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

RUN_DIR   = Path('C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z')
INPUT_CSV = RUN_DIR / 'data' / 'iteration_003_preprocess.csv'
ART_DIR   = RUN_DIR / 'artifacts' / 'training'
ART_DIR.mkdir(exist_ok=True)

HORIZONS = [1, 7, 14, 30, 60, 90]
P        = 7    # weekly seasonal period
S        = 52   # seasonal year: 52 same-DOW observations
TARGET   = 'bread_quantity'
EDA_PROV = {1: 38.01, 7: 40.73, 14: 42.67, 30: 44.14, 60: 45.14, 90: 47.54}

ZERO_HANDLING = (
    "zeros included as valid observations — recorded trading days with zero demand "
    "(closures, public holidays). Season-slot statistics include zero-demand rows."
)

# Expanded candidate grid — same for Part A (val rows) and Part B (train rows)
SEASON_WINDOWS  = [4, 8, 13, 26, 52]   # same-DOW window sizes
TRAILING_MEAN_W = [7, 14, 28, 56]       # trailing-day window sizes
TRAILING_MED_W  = [7, 28]
DAMPED_PHIS     = [0.25, 0.5]

# %% Load data and build lookup structures
print(f"Loading {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"Shape: {df.shape}")

ts       = df.set_index('date')[TARGET]
ts_idx   = ts.index            # full sorted DatetimeIndex
ts_vals  = ts.values           # full values array

# Full-series per-DOW structures (used in Part A — forecaster sees all data up to origin)
full_dow_idx  = {d: ts[ts_idx.dayofweek == d].index  for d in range(7)}
full_dow_vals = {d: ts[ts_idx.dayofweek == d].values for d in range(7)}

# %% Parametric core lookups (work for both full-ts and train-only structures)

def lk_last(O, idx, vals):
    """Last observed value at or before O."""
    pos = idx.searchsorted(O, side='right') - 1
    return float(vals[pos]) if pos >= 0 else np.nan


def lk_snaive(O, target_dow, dow_idx, dow_vals):
    pos = dow_idx[target_dow].searchsorted(O, side='right') - 1
    return float(dow_vals[target_dow][pos]) if pos >= 0 else np.nan


def lk_smedian(O, target_dow, w, dow_idx, dow_vals):
    pos = dow_idx[target_dow].searchsorted(O, side='right') - 1
    if pos < 0:
        return np.nan
    lo = max(0, pos - w + 1)
    return float(np.median(dow_vals[target_dow][lo:pos + 1]))


def lk_smean(O, target_dow, w, dow_idx, dow_vals):
    pos = dow_idx[target_dow].searchsorted(O, side='right') - 1
    if pos < 0:
        return np.nan
    lo = max(0, pos - w + 1)
    return float(np.mean(dow_vals[target_dow][lo:pos + 1]))


def lk_tmean(O, w, idx, vals):
    pos = idx.searchsorted(O, side='right') - 1
    if pos < 0:
        return np.nan
    lo = max(0, pos - w + 1)
    return float(np.mean(vals[lo:pos + 1]))


def lk_tmed(O, w, idx, vals):
    pos = idx.searchsorted(O, side='right') - 1
    if pos < 0:
        return np.nan
    lo = max(0, pos - w + 1)
    return float(np.median(vals[lo:pos + 1]))


def lk_drift(O, h, idx, vals):
    pos = idx.searchsorted(O, side='right') - 1
    if pos < 1:
        return np.nan
    v = vals[:pos + 1]
    avg_chg = (v[-1] - v[0]) / (len(v) - 1)
    return float(v[-1]) + h * avg_chg


def lk_ddrift(O, h, phi, idx, vals):
    pos = idx.searchsorted(O, side='right') - 1
    if pos < 1:
        return np.nan
    v = vals[:pos + 1]
    avg_chg = (v[-1] - v[0]) / (len(v) - 1)
    return float(v[-1]) + phi * h * avg_chg


# %% Build unified candidate list and column/formula lookups

def all_candidates():
    cands = ['last_value', 'seasonal_naive']
    for w in SEASON_WINDOWS:
        cands += [f'smedian_{w}', f'smean_{w}']
    cands += ['drift', 'train_mean']
    for w in TRAILING_MEAN_W:
        cands.append(f'tmean_{w}')
    for w in TRAILING_MED_W:
        cands.append(f'tmed_{w}')
    for phi in DAMPED_PHIS:
        cands.append(f'ddrift_{phi}')
    return cands


CANDS = all_candidates()
print(f"Candidate set: {len(CANDS)} rules per horizon")


def cand_col(cname, h):
    if cname == 'last_value':    return f'naive_h{h}_last_value'
    if cname == 'seasonal_naive': return f'naive_h{h}_seasonal_naive'
    if cname == 'drift':          return f'naive_h{h}_drift'
    if cname == 'train_mean':     return f'naive_h{h}_train_mean'
    if cname.startswith('smedian_'): w = cname[8:]; return f'naive_h{h}_seasonal_median_w{w}'
    if cname.startswith('smean_'):   w = cname[6:]; return f'naive_h{h}_seasonal_mean_w{w}'
    if cname.startswith('tmean_'):   w = cname[6:]; return f'naive_h{h}_trailing_mean_w{w}td'
    if cname.startswith('tmed_'):    w = cname[5:]; return f'naive_h{h}_trailing_median_w{w}td'
    if cname.startswith('ddrift_'):  phi = cname[7:]; return f'naive_h{h}_damped_drift_phi{phi}'
    return f'naive_h{h}_{cname}'


def cand_rule_display(cname, h, train_mean_val):
    if cname == 'last_value':    return f'last observed value at origin D-{h}'
    if cname == 'seasonal_naive': return f'seasonal naive (same DOW, last week at origin D-{h})'
    if cname == 'drift':          return f'linear drift over history at origin D-{h}, h={h}'
    if cname == 'train_mean':     return f'fold-3 train mean (constant {train_mean_val:.2f})'
    if cname.startswith('smedian_'): w = cname[8:]; return f'seasonal median w={w} same-DOW at origin D-{h}'
    if cname.startswith('smean_'):   w = cname[6:]; return f'seasonal mean w={w} same-DOW at origin D-{h}'
    if cname.startswith('tmean_'):   w = cname[6:]; return f'trailing mean w={w} trading-days at origin D-{h}'
    if cname.startswith('tmed_'):    w = cname[5:]; return f'trailing median w={w} trading-days at origin D-{h}'
    if cname.startswith('ddrift_'):  phi = cname[7:]; return f'damped drift phi={phi} at origin D-{h}, h={h}'
    return cname


def cand_formula(cname, h, train_mean_val):
    if cname == 'last_value':    return f'ts[ts.index <= (D-{h}days)].iloc[-1]'
    if cname == 'seasonal_naive': return f'ts[ts.index.dayofweek==target_dow & ts.index<=(D-{h}days)].iloc[-1]'
    if cname == 'drift':          return f'ts_at_O[-1] + {h}*(ts_at_O[-1]-ts_at_O[0])/(n-1); O=D-{h}days'
    if cname == 'train_mean':     return f'mean(fold3_train_series) -- constant {train_mean_val:.2f}; O=D-{h}days'
    if cname.startswith('smedian_'):
        w = cname[8:]
        return f'median(ts[dayofweek==target_dow].iloc[-{w}:]); O=D-{h}days'
    if cname.startswith('smean_'):
        w = cname[6:]
        return f'mean(ts[dayofweek==target_dow].iloc[-{w}:]); O=D-{h}days'
    if cname.startswith('tmean_'):
        w = cname[6:]
        return f'mean(ts.iloc[-{w}:]); O=D-{h}days'
    if cname.startswith('tmed_'):
        w = cname[5:]
        return f'median(ts.iloc[-{w}:]); O=D-{h}days'
    if cname.startswith('ddrift_'):
        phi = cname[7:]
        return f'ts_at_O[-1] + {phi}*{h}*(ts_at_O[-1]-ts_at_O[0])/(n-1); O=D-{h}days'
    return cname


# %% Compute candidate predictions for a single origin/target (full ts — Part A)
def preds_full(O, h, target_dow, train_mean_val):
    p = {}
    p['last_value']    = lk_last(O, ts_idx, ts_vals)
    p['seasonal_naive'] = lk_snaive(O, target_dow, full_dow_idx, full_dow_vals)
    for w in SEASON_WINDOWS:
        p[f'smedian_{w}'] = lk_smedian(O, target_dow, w, full_dow_idx, full_dow_vals)
        p[f'smean_{w}']   = lk_smean(O, target_dow, w, full_dow_idx, full_dow_vals)
    p['drift']      = lk_drift(O, h, ts_idx, ts_vals)
    p['train_mean'] = train_mean_val
    for w in TRAILING_MEAN_W:
        p[f'tmean_{w}'] = lk_tmean(O, w, ts_idx, ts_vals)
    for w in TRAILING_MED_W:
        p[f'tmed_{w}'] = lk_tmed(O, w, ts_idx, ts_vals)
    for phi in DAMPED_PHIS:
        p[f'ddrift_{phi}'] = lk_ddrift(O, h, phi, ts_idx, ts_vals)
    return p


# %% Part A — score expanded candidate set on validation rows

FOLDS = [
    ('_fold_1', 'train', 'val'),
    ('_fold_2', 'train', 'val'),
    ('_fold_3', 'train', 'val'),
]

print("\n=== Part A: expanded candidate set on validation rows ===")
fold_results = {}

for fold_col, tv, vv in FOLDS:
    fold_df  = df[df[fold_col].notna()]
    tr_dates = pd.DatetimeIndex(fold_df[fold_df[fold_col] == tv]['date'])
    vl_dates = pd.DatetimeIndex(fold_df[fold_df[fold_col] == vv]['date'])
    train_mean_val = float(ts.loc[tr_dates].mean())

    fold_h = {}
    for h in HORIZONS:
        rows = []
        for D in vl_dates:
            O = D - pd.Timedelta(days=int(h))
            tgt_val = ts.get(D)
            if tgt_val is None or np.isnan(float(tgt_val)):
                continue
            tgt_dow = D.dayofweek
            p = preds_full(O, h, tgt_dow, train_mean_val)
            p['_tgt'] = float(tgt_val)
            rows.append(p)
        preds_df = pd.DataFrame(rows)
        cand_maes = {}
        for c in CANDS:
            if c not in preds_df.columns:
                continue
            valid = preds_df[[c, '_tgt']].dropna()
            cand_maes[c] = round(float((valid[c] - valid['_tgt']).abs().mean()), 3) if len(valid) else None
        fold_h[h] = {
            'n': len(preds_df),
            'cand_maes': cand_maes,
            'train_mean': round(train_mean_val, 4),
        }
        # Print compact summary: top 5 candidates by MAE
        valid_sorted = sorted([(c, v) for c, v in cand_maes.items() if v is not None], key=lambda x: x[1])
        top5 = '  '.join(f"{c}={v:.2f}" for c, v in valid_sorted[:5])
        print(f"  {fold_col} h={h:2d}: n={len(preds_df):3d}  {top5}")
    fold_results[fold_col] = fold_h

# %% Aggregate baseline_floor (fold-3 primary; pooled 3 folds for h=90)
print("\n=== Part A: Aggregate baseline_floor ===")
baseline_floor_per_horizon = {}

for h in HORIZONS:
    if h == 90:
        # Pool all 3 folds weighted by n
        all_r = [fold_results[fc][h] for fc, _, _ in FOLDS if fold_results[fc][h]['n'] > 0]
        ns = [r['n'] for r in all_r]
        total_n = sum(ns)
        agg_maes = {}
        for c in CANDS:
            vals = [r['cand_maes'].get(c) for r in all_r]
            if any(v is None for v in vals):
                agg_maes[c] = None
            else:
                agg_maes[c] = round(float(sum(v * n for v, n in zip(vals, ns)) / total_n), 3)
        n_scored = total_n
        note = f"pooled 3 folds (fold 3 alone has only 5 scoreable h=90 val rows); n={n_scored}"
    else:
        r = fold_results['_fold_3'][h]
        agg_maes = r['cand_maes']
        n_scored = r['n']
        note = f"fold 3 primary; n={n_scored}"

    valid_maes = {c: v for c, v in agg_maes.items() if v is not None}
    best_c = min(valid_maes, key=lambda c: valid_maes[c])
    best_mae = valid_maes[best_c]
    snaive_mae = agg_maes.get('seasonal_naive')

    # All-rules summary (include only the original 6 + key expanded for compactness)
    key_rules = ['last_value', 'seasonal_naive', 'smedian_52', 'smean_52', 'drift', 'train_mean',
                 'smedian_13', 'smedian_26', 'smedian_8', 'smedian_4', 'tmean_28', 'tmed_28']
    all_rules_summary = {c: agg_maes.get(c) for c in key_rules if agg_maes.get(c) is not None}

    baseline_floor_per_horizon[h] = {
        'n': n_scored,
        'winner': best_c,
        'winner_col': cand_col(best_c, h),
        'winner_mae': best_mae,
        'seasonal_naive_mae': snaive_mae,
        'eda_provisional_mae': EDA_PROV[h],
        'delta_winner_vs_snaive': round(best_mae - snaive_mae, 3) if snaive_mae else None,
        'delta_winner_vs_eda': round(best_mae - EDA_PROV[h], 3),
        'all_rules': agg_maes,          # full expanded set
        'key_rules_summary': all_rules_summary,  # compact for display
        'note': note,
    }
    sn_str = f"{snaive_mae:.2f}" if snaive_mae else "N/A"
    print(f"  h={h:2d}: FLOOR={best_c:20s} MAE={best_mae:.2f}  snaive={sn_str}  EDA_prov={EDA_PROV[h]:.2f}  [{note}]")

# %% Part B — naive predictor feature search (fold-3 train rows only)
print("\n=== Part B: Naive predictor search (fold-3 train rows only) ===")

f3_train_dates = pd.DatetimeIndex(df[df['_fold_3'] == 'train']['date'])
f3_train_ts    = ts.loc[f3_train_dates]
f3_train_vals  = f3_train_ts.values
f3_train_idx   = f3_train_ts.index
f3_train_mean  = float(f3_train_ts.mean())

tr_dow_idx  = {d: f3_train_ts[f3_train_idx.dayofweek == d].index  for d in range(7)}
tr_dow_vals = {d: f3_train_ts[f3_train_idx.dayofweek == d].values for d in range(7)}

# Part A winner per horizon
PARTA_WINNER = {h: baseline_floor_per_horizon[h]['winner'] for h in HORIZONS}

MIN_HISTORY = max(S, max(TRAILING_MEAN_W))   # skip rows with insufficient lookback


def preds_train(O, h, target_dow, f3_tm):
    """Compute all candidate predictions using only fold-3 training data up to O."""
    p = {}
    p['last_value']    = lk_last(O, f3_train_idx, f3_train_vals)
    p['seasonal_naive'] = lk_snaive(O, target_dow, tr_dow_idx, tr_dow_vals)
    for w in SEASON_WINDOWS:
        p[f'smedian_{w}'] = lk_smedian(O, target_dow, w, tr_dow_idx, tr_dow_vals)
        p[f'smean_{w}']   = lk_smean(O, target_dow, w, tr_dow_idx, tr_dow_vals)
    p['drift']      = lk_drift(O, h, f3_train_idx, f3_train_vals)
    p['train_mean'] = f3_tm
    for w in TRAILING_MEAN_W:
        p[f'tmean_{w}'] = lk_tmean(O, w, f3_train_idx, f3_train_vals)
    for w in TRAILING_MED_W:
        p[f'tmed_{w}'] = lk_tmed(O, w, f3_train_idx, f3_train_vals)
    for phi in DAMPED_PHIS:
        p[f'ddrift_{phi}'] = lk_ddrift(O, h, phi, f3_train_idx, f3_train_vals)
    return p


naive_predictor_features = {}
part_b_search_grids      = {}

for h in HORIZONS:
    print(f"\n  h={h} — computing all {len(CANDS)} candidate predictions on fold-3 train rows...")
    all_preds   = {c: [] for c in CANDS}
    all_targets = []
    scored_n    = 0

    for D in f3_train_dates:
        O     = D - pd.Timedelta(days=int(h))
        tr_pos = f3_train_idx.searchsorted(O, side='right') - 1
        tgt_val = float(ts.get(D, np.nan))
        # Skip rows with insufficient history or missing target
        if tr_pos < MIN_HISTORY - 1 or np.isnan(tgt_val):
            for c in CANDS:
                all_preds[c].append(np.nan)
            all_targets.append(np.nan)
            continue
        tgt_dow = D.dayofweek
        p = preds_train(O, h, tgt_dow, f3_train_mean)
        for c in CANDS:
            all_preds[c].append(p.get(c, np.nan))
        all_targets.append(tgt_val)
        scored_n += 1

    targets_arr = np.array(all_targets)
    pred_arrs   = {c: np.array(all_preds[c]) for c in CANDS}

    cand_maes = {}
    for c in CANDS:
        pa   = pred_arrs[c]
        mask = ~(np.isnan(pa) | np.isnan(targets_arr))
        cand_maes[c] = round(float(np.mean(np.abs(pa[mask] - targets_arr[mask]))), 3) if mask.sum() > 0 else None

    valid_ranked = sorted([(c, cand_maes[c]) for c in CANDS if cand_maes[c] is not None], key=lambda x: x[1])

    parta_winner = PARTA_WINNER[h]
    kept         = [parta_winner]
    dedup_dropped = []

    for c, cmae in valid_ranked:
        if len(kept) >= 3:
            break
        if c in kept:
            continue
        pa = pred_arrs[c]
        max_corr, corr_with = 0.0, None
        for k in kept:
            ka   = pred_arrs[k]
            mask = ~(np.isnan(pa) | np.isnan(ka) | np.isnan(targets_arr))
            if mask.sum() < 10:
                continue
            r = abs(float(np.corrcoef(pa[mask], ka[mask])[0, 1]))
            if r > max_corr:
                max_corr, corr_with = r, k
        if max_corr > 0.98:
            dedup_dropped.append({
                'candidate': c,
                'column_name': cand_col(c, h),
                'train_mae': cmae,
                'max_corr': round(max_corr, 4),
                'corr_with': cand_col(corr_with, h) if corr_with else None,
                'reason': f'Pearson r={max_corr:.4f} > 0.98',
            })
            continue
        kept.append(c)

    # Cross-correlations among kept
    kept_corrs = {}
    for c in kept:
        pa    = pred_arrs[c]
        corrs = {}
        for k2 in kept:
            if k2 == c:
                continue
            ka   = pred_arrs[k2]
            mask = ~(np.isnan(pa) | np.isnan(ka) | np.isnan(targets_arr))
            if mask.sum() >= 10:
                corrs[cand_col(k2, h)] = round(abs(float(np.corrcoef(pa[mask], ka[mask])[0, 1])), 4)
        kept_corrs[c] = corrs

    naive_predictor_features[h] = []
    for rank_i, c in enumerate(kept):
        spec = {
            'horizon':       h,
            'column_name':   cand_col(c, h),
            'rule':          cand_rule_display(c, h, f3_train_mean),
            'formula':       cand_formula(c, h, f3_train_mean),
            'train_mae':     cand_maes[c],
            'train_primary': cand_maes[c],
            'rank':          rank_i + 1,
            'corr_with_kept': kept_corrs[c],
            'selected_by':   'part_a_winner' if c == parta_winner else 'train_search',
            'protected':     True,
            'zero_handling': ZERO_HANDLING,
        }
        naive_predictor_features[h].append(spec)
        print(f"    kept [{rank_i+1}] {spec['column_name']:45s}  train_mae={cand_maes[c]:.3f}"
              f"  ({'PART_A_WINNER' if c==parta_winner else 'train_search'})")
    if dedup_dropped:
        print(f"    dedup_dropped: {[d['column_name'] for d in dedup_dropped]}")

    part_b_search_grids[h] = {
        'n_train_scored': scored_n,
        'candidates_ranked': [
            {'name': c, 'column_name': cand_col(c, h), 'train_mae': cand_maes[c],
             'rank': i + 1, 'kept': c in kept}
            for i, (c, _) in enumerate(valid_ranked)
        ],
        'dedup_dropped': dedup_dropped,
        'selection_basis': 'train_rows_only (fold 3)',
    }

# %% Protected features list
protected_features = []
for h in HORIZONS:
    for spec in naive_predictor_features.get(h, []):
        protected_features.append({
            'column':                    spec['column_name'],
            'reason':                    f"naive predictor selected for h={h} (baseline iteration 004)",
            'protected_since_iteration': 4,
            'horizon':                   h,
        })
print(f"\nProtected features ({len(protected_features)}):")
for pf in protected_features:
    print(f"  {pf['column']}")

# %% Part A — Plot
print("\nGenerating Part A plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: original 6 rules for comparability with prior runs
ax = axes[0]
x = np.arange(len(HORIZONS))
bar_w = 0.13
key6 = ['last_value', 'seasonal_naive', 'smedian_52', 'smean_52', 'drift', 'train_mean']
labels6 = ['last_value', 'snaive', 'smedian_52w', 'smean_52w', 'drift', 'train_mean']
colors6 = ['#2c3e50', '#2980b9', '#27ae60', '#e67e22', '#e74c3c', '#8e44ad']
for i, (cname, col, lab) in enumerate(zip(key6, colors6, labels6)):
    maes = [baseline_floor_per_horizon[h]['all_rules'].get(cname, 0) or 0 for h in HORIZONS]
    ax.bar(x + i * bar_w, maes, width=bar_w, label=lab, color=col, alpha=0.85)
# Floor line
floor_maes = [baseline_floor_per_horizon[h]['winner_mae'] for h in HORIZONS]
ax.plot(x + 2.5 * bar_w, floor_maes, 'k--o', lw=2, ms=5, label='floor (best rule)')
ax.set_xticks(x + 2.5 * bar_w); ax.set_xticklabels([f'h={h}' for h in HORIZONS])
ax.set_ylabel('MAE'); ax.set_title('Part A: original 6 rules + best-rule floor', loc='left', fontsize=10)
ax.legend(loc='upper left', fontsize=8)

# Right: expanded set — just winners per horizon
ax = axes[1]
winners = [f"{baseline_floor_per_horizon[h]['winner']}={baseline_floor_per_horizon[h]['winner_mae']:.1f}"
           for h in HORIZONS]
bars = ax.bar(x, floor_maes, color='#2980b9', alpha=0.85)
ax.plot(x, [EDA_PROV[h] for h in HORIZONS], 'ro--', lw=1.5, ms=4, label='EDA provisional')
for bar, label in zip(bars, winners):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, label,
            ha='center', va='bottom', fontsize=7, rotation=45)
ax.set_xticks(x); ax.set_xticklabels([f'h={h}' for h in HORIZONS])
ax.set_ylabel('MAE'); ax.set_title('Part A: floor MAE per horizon (expanded rule set winner)', loc='left', fontsize=10)
ax.legend(fontsize=8)
fig.text(0.99, 0.01, 'iteration 004 · baseline', ha='right', fontsize=7.5, color='#999')
plt.tight_layout()

plot1_path = ART_DIR / 'iteration_004_baseline_mae_by_horizon.png'
csv1_path  = ART_DIR / 'iteration_004_baseline_mae_by_horizon.csv'
plt.savefig(str(plot1_path), dpi=100)
plt.close()

rows_csv1 = []
for h in HORIZONS:
    bf = baseline_floor_per_horizon[h]
    row = {'horizon': h, 'winner': bf['winner'], 'floor_mae': bf['winner_mae'],
           'eda_provisional': bf['eda_provisional_mae']}
    row.update({c: bf['all_rules'].get(c) for c in key6})
    rows_csv1.append(row)
pd.DataFrame(rows_csv1).to_csv(str(csv1_path), index=False)
print(f"Plot 1 saved: {plot1_path}")

# %% Part B — Plot
print("Generating Part B plot...")
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(HORIZONS))
n_kept = max(len(naive_predictor_features[h]) for h in HORIZONS)
bar_w_b = 0.22
colors_b = ['#2980b9', '#e67e22', '#27ae60']
for slot_i in range(n_kept):
    maes_slot = []
    for h in HORIZONS:
        specs = naive_predictor_features.get(h, [])
        maes_slot.append(specs[slot_i]['train_mae'] if slot_i < len(specs) else 0)
    ax.bar(x + slot_i * bar_w_b, maes_slot, width=bar_w_b,
           label=f'kept slot {slot_i+1}', color=colors_b[slot_i % 3], alpha=0.85)
ax.plot(x + (n_kept/2 - 0.5) * bar_w_b, floor_maes, 'k--o', lw=1.5, ms=4, label='Part A floor (val MAE)')
ax.set_xticks(x + (n_kept/2 - 0.5) * bar_w_b)
ax.set_xticklabels([f'h={h}' for h in HORIZONS])
ax.set_ylabel('MAE (TRAIN rows for predictors / VAL rows for floor)')
ax.set_title('Part B kept naive predictors — train MAE vs Part A validation floor', loc='left', fontsize=10)
ax.text(0, 1.02, 'grouped bar', transform=ax.transAxes, fontsize=9, color='#666')
fig.text(0.99, 0.01, 'iteration 004 · baseline', ha='right', fontsize=7.5, color='#999')
ax.legend(loc='upper left', fontsize=8)
plt.tight_layout()

plot2_path = ART_DIR / 'iteration_004_baseline_naive_predictors.png'
csv2_path  = ART_DIR / 'iteration_004_baseline_naive_predictors.csv'
plt.savefig(str(plot2_path), dpi=100)
plt.close()

rows_b = []
for h in HORIZONS:
    for spec in naive_predictor_features.get(h, []):
        rows_b.append({'horizon': h, 'column_name': spec['column_name'],
                       'rule': spec['rule'], 'train_mae': spec['train_mae'],
                       'rank': spec['rank'], 'selected_by': spec['selected_by']})
pd.DataFrame(rows_b).to_csv(str(csv2_path), index=False)
print(f"Plot 2 saved: {plot2_path}")

# %% Compute pooled floor MAE
winner_maes     = {h: baseline_floor_per_horizon[h]['winner_mae'] for h in HORIZONS}
pooled_floor_mae = round(float(np.mean(list(winner_maes.values()))), 3)
winner_names     = {h: baseline_floor_per_horizon[h]['winner'] for h in HORIZONS}
note_str = (
    f"Floor at h=1: {winner_names[1]} MAE={winner_maes[1]:.2f}. "
    f"Floor at h=90: {winner_names[90]} MAE={winner_maes[90]:.2f} (pooled 3 folds). "
    f"Every learned model must beat these floors per horizon. "
    f"Expanded candidate set ({len(CANDS)} rules) used in Part A."
)

baseline_floor_block = {
    "established_at_iteration": 4,
    "baseline_type": f"best of {len(CANDS)} candidates (winner per horizon, validation rows)",
    "n_candidates_tested": len(CANDS),
    "seasonal_period": P,
    "window": S,
    "window_source": f"one seasonal year = {S} same-DOW season-slots; P={P} from EDA dominant_acf_lag=6",
    "zero_handling": ZERO_HANDLING,
    "per_horizon": {
        str(h): {
            "winner":                 baseline_floor_per_horizon[h]["winner"],
            "winner_col":             baseline_floor_per_horizon[h]["winner_col"],
            "winner_mae":             baseline_floor_per_horizon[h]["winner_mae"],
            "n":                      baseline_floor_per_horizon[h]["n"],
            "key_rules_mae":          baseline_floor_per_horizon[h]["key_rules_summary"],
            "eda_provisional_mae":    baseline_floor_per_horizon[h]["eda_provisional_mae"],
            "delta_winner_vs_snaive": baseline_floor_per_horizon[h]["delta_winner_vs_snaive"],
            "note":                   baseline_floor_per_horizon[h]["note"],
        }
        for h in HORIZONS
    },
    "pooled": {"mae": pooled_floor_mae, "note": "simple mean of per-horizon floor MAEs"},
    "note": note_str,
}

# %% MLflow
print("\n=== MLflow logging ===")
try:
    import mlflow
    MLFLOW_DB = 'C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/mlflow_prototype/mlflow.db'
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("run_20260908T075709Z")
    with mlflow.start_run(run_name="iteration_004_baseline_revised"):
        mlflow.log_param("iteration", 4)
        mlflow.log_param("node", "baseline")
        mlflow.log_param("n_candidates", len(CANDS))
        mlflow.log_param("P", P); mlflow.log_param("S", S)
        for h in HORIZONS:
            mlflow.log_metric(f"floor_mae_h{h}", winner_maes[h])
            sn = baseline_floor_per_horizon[h].get('seasonal_naive_mae')
            if sn:
                mlflow.log_metric(f"snaive_mae_h{h}", sn)
        mlflow.log_metric("pooled_floor_mae", pooled_floor_mae)
        mlflow.log_artifact(str(plot1_path))
        mlflow.log_artifact(str(plot2_path))
    print("MLflow logging complete.")
except Exception as e:
    print(f"MLflow logging skipped: {e}")

# %% Write history JSON
print("\nWriting history/iteration_004_baseline.json...")
history_json = {
    "iteration": 4,
    "node": "baseline",
    "run_id": "run_20260908T075709Z",
    "timestamp": "2026-09-08",
    "code_file": "code/iteration_004_baseline.py",
    "mode": "baseline",
    "mode_reasoning": (
        "First Training call of the run. Part A scores an expanded {}-candidate grid on validation rows "
        "to establish baseline_floor. Part B searches the same grid on fold-3 train rows to select "
        "naive predictor feature specs. No model is fitted. Re-run only if split or trading-day rule changes."
    ).format(len(CANDS)),

    "seasonal_parameters": {
        "P": P, "S": S,
        "P_source": "EDA dominant_acf_lag=6 (weekly series)",
        "S_source": f"one seasonal year = {S} same-DOW season-slots",
        "zero_handling": ZERO_HANDLING,
    },

    "part_a_scores": {
        "n_candidates": len(CANDS),
        "candidate_set": CANDS,
        "fold_detail": {
            fc: {
                str(h): {
                    "n":       fold_results[fc][h]["n"],
                    "cand_maes": fold_results[fc][h]["cand_maes"],
                }
                for h in HORIZONS
            }
            for fc, _, _ in FOLDS
        },
        "h90_pooling_reason": (
            "Fold 3 alone has only 5 scoreable h=90 validation rows (dataset ends 2020-12-31, "
            "last h=90 origin approx 2020-10-02). Pooled 3-fold MAE is weighted by n per fold."
        ),
    },

    "baseline_floor": baseline_floor_block,

    "part_b_naive_predictors": {
        "selection_basis": f"fold-3 train rows only (n={len(f3_train_dates)}); same {len(CANDS)}-candidate grid as Part A",
        "search_grid": {str(h): part_b_search_grids[h] for h in HORIZONS},
        "kept_predictors": {str(h): naive_predictor_features.get(h, []) for h in HORIZONS},
        "note": (
            "Part A winner always kept regardless of train rank. Max 3 per horizon. "
            "Dedup: Pearson r > 0.98 on train predictions -> drop lower-ranked."
        ),
    },

    "protected_features_emitted": protected_features,

    "artifact_paths": {
        "plots": [
            {
                "name": "Baseline MAE by horizon — 6 key rules + best-rule floor",
                "type": "grouped bar + line",
                "path": f"artifacts/training/{plot1_path.name}",
                "data_path": f"artifacts/training/{csv1_path.name}",
                "reason": "Shows original-6 rule MAEs + expanded-set floor winner per horizon",
            },
            {
                "name": "Part B kept naive predictors — train MAE vs Part A validation floor",
                "type": "grouped bar",
                "path": f"artifacts/training/{plot2_path.name}",
                "data_path": f"artifacts/training/{csv2_path.name}",
                "reason": "Shows train MAEs for kept predictors per horizon, with Part A floor for reference",
            },
        ]
    },

    "handoff": {
        "written_by": "baseline",
        "iteration": 4,
        "context_clearable": True,
        "read_this_first": (
            f"baseline_floor established from {len(CANDS)}-candidate grid: per-horizon MAE floors "
            "every learned model must beat. "
            "naive_predictor_features carries the FULL SPEC (column_name + formula) that Feature Engineering "
            "must materialise as protected columns. "
            "h=90 floor uses pooled 3-fold val (fold 3 alone = 5 rows)."
        ),

        "parameters": {
            "baseline_floor": {
                "value": baseline_floor_block,
                "why": "Floor every later model must beat per horizon. Established on val rows.",
                "must_not_change": True,
            },
            "naive_predictor_features": {
                "value": {str(h): naive_predictor_features.get(h, []) for h in HORIZONS},
                "why": "FE must materialise these columns exactly. Protected — may NOT be dropped without user approval.",
                "must_not_change": True,
            },
            "protected_features": {
                "value": protected_features,
                "why": "Every subsequent FE, Training, and Diagnosis call must respect these.",
                "must_not_change": True,
            },
            "seasonal_period_P": {"value": P, "why": "Weekly seasonal period", "must_not_change": True},
            "seasonal_window_S": {"value": S, "why": "Season-slot window for median/mean rules", "must_not_change": True},
            "zero_handling": {"value": ZERO_HANDLING, "must_not_change": True, "why": "Closure/zero policy consistent across nodes"},
        },

        "durable_rules": [
            {
                "rule": "Every learned model must report MAE vs baseline_floor PER HORIZON before any stop decision.",
                "why": "A model below the naive floor is worse than doing nothing.",
                "applies_to": ["training", "diagnosis", "end_model"],
                "cost_if_broken": "A deployed model that loses to a trivial rule goes undetected.",
            },
            {
                "rule": "naive_predictor_features must be materialised in every FE call: exact column names, exact formulas, horizon-scoped.",
                "why": "These are the run's guaranteed floor-level signal.",
                "applies_to": ["feature_engineering"],
                "cost_if_broken": "Feature set missing the floor-level anchor.",
            },
            {
                "rule": "A node proposing to DROP any column in protected_features MUST raise a BLOCKING clarification and NOT drop on its own.",
                "why": "Naive predictor columns score near-zero in masked importance and flag in multicollinearity — known artefacts.",
                "applies_to": ["feature_engineering", "training", "diagnosis"],
                "cost_if_broken": "Naive predictor silently removed.",
            },
        ],

        "methodological_cautions": [
            {
                "caution": f"h=90 floor (pooled 3 folds, n={baseline_floor_per_horizon.get(90,{}).get('n','?')}) is modest.",
                "why": "Dataset ends 2020-12-31; origins for h=90 are early in the gap.",
                "cost_if_ignored": "h=90 champion selection based on fold-3 alone = 5 rows (random).",
            },
            {
                "caution": "Monday (DOW=0) has only 5 training observations post-regime (June 2020). Seasonal lookbacks for Mondays include pre-regime zero rows, biasing seasonal statistics downward.",
                "why": "Pre-regime Mondays are closed (zero demand); the seasonal window w=52 will include ~47 zero-demand pre-regime Mondays for any post-regime Monday origin.",
                "cost_if_ignored": "seasonal_mean_w52 is severely biased for Monday targets; FE must include regime_x_monday feature.",
            },
        ],

        "do_not_repeat": [
            {
                "tried": "Restricting Part A to only 6 fixed rules",
                "outcome": "rejected",
                "evidence": "Part B showed shorter windows (smedian_w13) have substantially lower train MAE; expanding Part A to the same grid finds better floor rules.",
                "why_not_again": "6 rules is an arbitrary restriction that misses better trivial predictors.",
            },
            {
                "tried": "Scoring h=90 baseline on fold 3 alone",
                "outcome": "rejected",
                "evidence": "Fold 3 has only 5 scoreable h=90 val rows; SE of MAE is ~45% of mean.",
                "why_not_again": "5 rows insufficient for reliable MAE estimation.",
            },
        ],

        "content_carried_forward": {
            "baseline_floor_per_horizon": {
                str(h): {"winner": baseline_floor_per_horizon[h]["winner"],
                          "winner_mae": baseline_floor_per_horizon[h]["winner_mae"],
                          "n": baseline_floor_per_horizon[h]["n"]}
                for h in HORIZONS
            },
            "naive_predictor_features_all_horizons": {
                str(h): [
                    {"column_name": s["column_name"], "rule": s["rule"],
                     "formula": s["formula"], "train_mae": s["train_mae"],
                     "selected_by": s["selected_by"], "protected": True}
                    for s in naive_predictor_features.get(h, [])
                ]
                for h in HORIZONS
            },
            "protected_features": protected_features,
            "f3_train_mean": round(f3_train_mean, 4),
            "seasonal_period_P": P,
            "seasonal_window_S": S,
        },
    }
}

HIST_PATH = RUN_DIR / 'history' / 'iteration_004_baseline.json'
with open(str(HIST_PATH), 'w') as f:
    json.dump(history_json, f, indent=2, default=str)
print(f"History JSON written: {HIST_PATH}")

# %% Summary table
print("\n=== BASELINE COMPLETE ===")
print(f"{'h':>4} | {'Floor rule':>25} | {'Floor MAE':>10} | {'Snaive MAE':>11} | {'EDA prov':>9}")
print("-" * 68)
for h in HORIZONS:
    bf  = baseline_floor_per_horizon[h]
    sn  = f"{bf['seasonal_naive_mae']:.2f}" if bf['seasonal_naive_mae'] else "N/A"
    print(f"  {h:2d} | {bf['winner']:>25} | {bf['winner_mae']:>10.2f} | {sn:>11} | {bf['eda_provisional_mae']:>9.2f}")
print(f"\nPooled floor MAE: {pooled_floor_mae:.3f}")
print(f"Protected features ({len(protected_features)}):")
for pf in protected_features:
    print(f"  {pf['column']}")
