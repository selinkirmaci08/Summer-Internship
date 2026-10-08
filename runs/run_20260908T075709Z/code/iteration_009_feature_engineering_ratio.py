# %% [markdown]
# # Iteration 009 — Feature Engineering (ratio target columns)
# Run: run_20260908T075709Z
# Node: feature_engineering (incremental — adds ratio columns to iter005 output)
# Reads:  data/iteration_005_feature_engineering.csv
# Writes: data/iteration_009_feature_engineering.csv
#         history/iteration_009_feature_engineering_ratio.json
#         artifacts/feature_engineering_ratio/  (distribution plots)
#         state/current.json (updated)
#
# What this adds:
#   ratio_h{h} = bread_quantity_h{h} / anchor_h{h}  (clipped to [0.05, 5.0])
#   anchor_h{h} = best naive predictor for that horizon (smedian_w4 for h<=30,
#                 smedian_w26 for h>=60)
#   NaN propagated where bread_quantity_h{h} is NaN
#   Rows where anchor <= 0 get ratio = NaN (pre-regime Monday anchor issue)

# %% Imports
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Config
RUN_DIR  = Path('C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z')
IN_CSV   = RUN_DIR / 'data' / 'iteration_005_feature_engineering.csv'
OUT_CSV  = RUN_DIR / 'data' / 'iteration_009_feature_engineering.csv'
ART_DIR  = RUN_DIR / 'artifacts' / 'feature_engineering_ratio'
ART_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 7, 14, 30, 60, 90]
TARGET   = 'bread_quantity'
RATIO_CLIP = (0.05, 5.0)

ANCHOR_COLS = {
    1:  'naive_h1_seasonal_median_w4',
    7:  'naive_h7_seasonal_median_w4',
    14: 'naive_h14_seasonal_median_w4',
    30: 'naive_h30_seasonal_median_w4',
    60: 'naive_h60_seasonal_median_w26',
    90: 'naive_h90_seasonal_median_w26',
}

print(f"[Config] Input:  {IN_CSV.name}")
print(f"[Config] Output: {OUT_CSV.name}")
print(f"[Config] Ratio clip: {RATIO_CLIP}")

# %% Load
print("\n[Step 1] Loading FE data...")
df = pd.read_csv(str(IN_CSV), parse_dates=['date']).sort_values('date').reset_index(drop=True)
print(f"  Loaded {len(df)} rows x {len(df.columns)} cols")

train_mask = df['_split'] == 'train'
gap_mask   = df['_split'] == 'gap'
val_mask   = df['_split'] == 'validation'
covid_mask = train_mask & df['is_covid_lockdown_period'].astype(bool)

print(f"  train={train_mask.sum()}, gap={gap_mask.sum()}, val={val_mask.sum()}")
print(f"  COVID rows in training: {covid_mask.sum()}")

# %% Verify anchor columns exist
print("\n[Step 2] Verifying anchor columns...")
for h, col in ANCHOR_COLS.items():
    assert col in df.columns, f"Missing anchor column: {col}"
    n_zero = (df[col] <= 0).sum()
    n_nan  = df[col].isna().sum()
    print(f"  h={h:2d}: {col} — zero/neg={n_zero}, NaN={n_nan}, "
          f"mean={df[col].mean():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")

# %% Compute ratio columns
print("\n[Step 3] Computing ratio_h{h} columns...")
ratio_stats = {}

for h in HORIZONS:
    target_col = f'{TARGET}_h{h}'
    anchor_col = ANCHOR_COLS[h]
    ratio_col  = f'ratio_h{h}'

    assert target_col in df.columns, f"Missing target column: {target_col}"

    anchor = df[anchor_col].copy()
    target = df[target_col].copy()

    # Compute raw ratio
    raw_ratio = target / anchor.replace(0, np.nan)  # anchor=0 → NaN

    # Clip only where target is not NaN (preserve NaN for rows beyond series end)
    clipped = raw_ratio.copy()
    has_target = target.notna()
    clipped.loc[has_target] = raw_ratio.loc[has_target].clip(lower=RATIO_CLIP[0], upper=RATIO_CLIP[1])

    df[ratio_col] = clipped

    # Stats by split
    def split_stats(mask, name):
        vals = clipped[mask & has_target]
        if len(vals) == 0:
            return f"{name}: n=0"
        return (f"{name}: n={len(vals)}, mean={vals.mean():.4f}, "
                f"std={vals.std():.4f}, min={vals.min():.3f}, max={vals.max():.3f}, "
                f"NaN={raw_ratio[mask & ~has_target].sum()}")

    train_vals_noncovid = clipped[train_mask & ~covid_mask & has_target]
    train_vals_covid    = clipped[covid_mask & has_target]
    val_vals            = clipped[val_mask & has_target]

    ratio_stats[h] = {
        'ratio_col': ratio_col,
        'anchor_col': anchor_col,
        'n_anchor_zero': int((anchor <= 0).sum()),
        'n_target_nan':  int((~has_target).sum()),
        'train_noncovid': {
            'n': int(len(train_vals_noncovid)),
            'mean': round(float(train_vals_noncovid.mean()), 4) if len(train_vals_noncovid) > 0 else None,
            'std':  round(float(train_vals_noncovid.std()), 4) if len(train_vals_noncovid) > 0 else None,
        },
        'train_covid': {
            'n': int(len(train_vals_covid)),
            'mean': round(float(train_vals_covid.mean()), 4) if len(train_vals_covid) > 0 else None,
            'std':  round(float(train_vals_covid.std()), 4) if len(train_vals_covid) > 0 else None,
        },
        'val': {
            'n': int(len(val_vals)),
            'mean': round(float(val_vals.mean()), 4) if len(val_vals) > 0 else None,
            'std':  round(float(val_vals.std()), 4) if len(val_vals) > 0 else None,
        },
    }

    print(f"\n  h={h:2d}: {ratio_col}")
    print(f"    anchor zero rows: {ratio_stats[h]['n_anchor_zero']}, target NaN rows: {ratio_stats[h]['n_target_nan']}")
    print(f"    train (non-COVID): n={ratio_stats[h]['train_noncovid']['n']}, "
          f"mean={ratio_stats[h]['train_noncovid']['mean']}, std={ratio_stats[h]['train_noncovid']['std']}")
    print(f"    train (COVID):     n={ratio_stats[h]['train_covid']['n']}, "
          f"mean={ratio_stats[h]['train_covid']['mean']}, std={ratio_stats[h]['train_covid']['std']}")
    print(f"    val:               n={ratio_stats[h]['val']['n']}, "
          f"mean={ratio_stats[h]['val']['mean']}, std={ratio_stats[h]['val']['std']}")

# %% Distribution stability check
print("\n[Step 4] Ratio distribution stability check (train non-COVID vs val)...")
print(f"\n  {'h':>4} | {'train_mean':>10} | {'val_mean':>8} | {'delta':>8} | {'within_0.1':>10}")
print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
for h in HORIZONS:
    tm = ratio_stats[h]['train_noncovid']['mean']
    vm = ratio_stats[h]['val']['mean']
    if tm is None or vm is None:
        continue
    delta = abs(vm - tm)
    ok = 'YES' if delta < 0.1 else 'WARN'
    print(f"  {h:>4} | {tm:>10.4f} | {vm:>8.4f} | {vm-tm:>+8.4f} | {ok:>10}")

# %% Distribution plots
print("\n[Step 5] Ratio distribution plots...")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, h in enumerate(HORIZONS):
    ratio_col  = f'ratio_h{h}'
    has_target = df[f'{TARGET}_h{h}'].notna()
    ax = axes[i]

    train_noncovid_vals = df.loc[train_mask & ~covid_mask & has_target, ratio_col].dropna()
    covid_vals          = df.loc[covid_mask & has_target, ratio_col].dropna()
    val_vals            = df.loc[val_mask & has_target, ratio_col].dropna()

    bins = np.linspace(RATIO_CLIP[0], RATIO_CLIP[1], 40)
    ax.hist(train_noncovid_vals, bins=bins, alpha=0.5, color='steelblue',
            edgecolor='white', label=f'Train non-COVID (n={len(train_noncovid_vals)})')
    if len(covid_vals) > 0:
        ax.hist(covid_vals, bins=bins, alpha=0.6, color='red',
                edgecolor='white', label=f'Train COVID (n={len(covid_vals)})')
    ax.hist(val_vals, bins=bins, alpha=0.5, color='green',
            edgecolor='white', label=f'Val (n={len(val_vals)})')

    ax.axvline(1.0, color='black', ls='--', lw=1, label='ratio=1')
    ax.set_title(f'ratio_h{h} distribution')
    ax.set_xlabel('Ratio (y / anchor)')
    ax.legend(fontsize=6)

plt.suptitle('Ratio target distributions: train non-COVID vs COVID vs val', fontsize=11)
plt.tight_layout()
plt.savefig(str(ART_DIR / 'plot1_ratio_distributions.png'), dpi=120)
plt.close()
print("  Saved plot1_ratio_distributions.png")

# Per-DOW ratio distribution for h=1 (key diagnostic)
fig, ax = plt.subplots(figsize=(10, 4))
dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
has_h1 = df['bread_quantity_h1'].notna()
for d in sorted(df['dow'].unique()):
    mask_d = train_mask & ~covid_mask & has_h1 & (df['dow'] == d)
    vals   = df.loc[mask_d, 'ratio_h1'].dropna()
    if len(vals) > 0:
        ax.scatter([d] * len(vals), vals, alpha=0.1, s=5,
                   label=f"{dow_names.get(d,d)} (n={len(vals)})")
        ax.errorbar(d, vals.mean(), yerr=vals.std(), fmt='o', color='black', ms=6, capsize=4)

val_has_h1 = val_mask & has_h1
for d in sorted(df.loc[val_has_h1, 'dow'].unique()):
    mask_d = val_has_h1 & (df['dow'] == d)
    vals   = df.loc[mask_d, 'ratio_h1'].dropna()
    if len(vals) > 0:
        ax.scatter([d + 0.25] * len(vals), vals, alpha=0.4, s=20, color='green')
        ax.errorbar(d + 0.25, vals.mean(), yerr=vals.std(), fmt='s', color='green', ms=6, capsize=4)

ax.axhline(1.0, color='red', ls='--', lw=1)
ax.set_xticks(range(7))
ax.set_xticklabels([dow_names[d] for d in range(7)])
ax.set_ylabel('ratio_h1 (y / anchor)')
ax.set_title('ratio_h1 by DOW — train non-COVID (black) vs val (green)')
plt.tight_layout()
plt.savefig(str(ART_DIR / 'plot2_ratio_h1_by_dow.png'), dpi=120)
plt.close()
print("  Saved plot2_ratio_h1_by_dow.png")

# %% Print column summary
print("\n[Step 6] New columns added:")
new_cols = [f'ratio_h{h}' for h in HORIZONS]
for c in new_cols:
    n_notna = df[c].notna().sum()
    print(f"  {c}: {n_notna} non-NaN rows out of {len(df)}")

print(f"\n  Total columns: {len(df.columns)} (was {len(df.columns) - len(new_cols)})")

# %% Save output CSV
print("\n[Step 7] Saving output CSV...")
df.to_csv(str(OUT_CSV), index=False)
print(f"  Saved: {OUT_CSV.name} ({OUT_CSV.stat().st_size // 1024} KB)")

# %% Write history JSON
print("\n[Step 8] Writing history JSON...")
ITER_NUM = 9
ITER_ID  = f'iteration_{ITER_NUM:03d}_feature_engineering_ratio'

history_json = {
    "iteration": ITER_NUM,
    "node": "feature_engineering",
    "timestamp": pd.Timestamp.now().isoformat()[:19],
    "schema_version": "2026-09-04",
    "incremental": True,
    "reads_from": "data/iteration_005_feature_engineering.csv",
    "writes_to": str(OUT_CSV.relative_to(RUN_DIR)),

    "what_was_added": {
        "ratio_columns": new_cols,
        "formula": "ratio_h{h} = bread_quantity_h{h} / anchor_h{h}, clipped to [0.05, 5.0]",
        "anchor_mapping": {str(h): ANCHOR_COLS[h] for h in HORIZONS},
        "nan_handling": "NaN where bread_quantity_h{h} is NaN; anchor=0 rows also get NaN",
        "clip_range": list(RATIO_CLIP),
    },

    "ratio_stats": ratio_stats,

    "stability_verdict": {
        h: (
            "STABLE" if (
                ratio_stats[h]['train_noncovid']['mean'] is not None and
                ratio_stats[h]['val']['mean'] is not None and
                abs(ratio_stats[h]['val']['mean'] - ratio_stats[h]['train_noncovid']['mean']) < 0.1
            ) else "WARN"
        )
        for h in HORIZONS
    },

    "do_not_repeat": [
        {
            "tried": "Additive residual target (y - anchor)",
            "outcome": "rejected",
            "evidence": "iter006: residual approach gave val_mae=49.20 for h=1 (worse than raw target 39.36). Train/val mean residual asymmetry persisted.",
            "why_not_again": "Additive residual does not remove level shift. Multiplicative ratio does."
        }
    ],

    "next_iteration_contract": {
        "node": "training",
        "iteration": 10,
        "why_this_node": "Train per-horizon models on ratio_h{h} target using non-COVID training rows. Select algorithm per horizon.",
        "read_this_first": (
            "For each horizon h: train on train_mask & ~is_covid_lockdown_period rows, "
            "predicting ratio_h{h}. Convert predictions to units: pred_units = pred_ratio x anchor. "
            "Compare pred_units val MAE to baseline floor. "
            "Algorithm selection per horizon: "
            "h=1,7 -> LightGBM (regression_l1 or huber); "
            "h=14,30 -> LightGBM with heavier regularization OR Ridge; "
            "h=60 -> Ridge/ElasticNet (only 35 val rows); "
            "h=90 -> Ridge or anchor multiplier only (5 val rows, tree models not learnable). "
            "Feature set: use all features from iter009 FE CSV except other ratio_h{h2} columns "
            "and bread_quantity_h{h2} columns for h2 != h."
        ),
        "required_state_fields": [
            "feature_engineering_output_path",
            "baseline_floor",
            "scoreable_val_rows",
        ],
        "context": {
            "feature_engineering_output_path": str(OUT_CSV.relative_to(RUN_DIR)),
            "ratio_target_cols": {str(h): f'ratio_h{h}' for h in HORIZONS},
            "anchor_cols": {str(h): ANCHOR_COLS[h] for h in HORIZONS},
            "covid_filter": "is_covid_lockdown_period == 0 (filter in Training)",
            "prediction_formula": "pred_units = pred_ratio x anchor_h{h}",
            "baseline_floor": {
                1: 31.263, 7: 31.263, 14: 32.846, 30: 35.913, 60: 40.042, 90: 42.143
            },
            "per_horizon_algorithm_guidance": {
                "h1":  "LightGBM regression_l1 or huber, Optuna or manual tuning",
                "h7":  "LightGBM regression_l1 or huber",
                "h14": "LightGBM with higher min_child_samples/L2, or Ridge",
                "h30": "LightGBM with heavier regularization, or Ridge/ElasticNet",
                "h60": "Ridge or ElasticNet with CV alpha (35 val rows, prefer linear)",
                "h90": "Ridge with 2-3 features only, or constant anchor (5 val rows)"
            },
            "clip_range": list(RATIO_CLIP),
        }
    },

    "handoff": {
        "written_by": "feature_engineering",
        "iteration": ITER_NUM,
        "context_clearable": True,
        "read_this_first": (
            "Ratio target columns added: ratio_h{h} = bread_quantity_h{h} / anchor_h{h} "
            "clipped to [0.05, 5.0]. Training filters COVID rows (is_covid_lockdown_period==1). "
            "Use per-horizon algorithm selection (LightGBM for h<=14, Ridge/ElasticNet for h>=60). "
            "Convert ratio predictions back to units for floor comparison."
        ),
        "parameters": {
            "ratio_clip": {"value": list(RATIO_CLIP), "must_not_change": False},
            "anchor_cols": {"value": {str(h): ANCHOR_COLS[h] for h in HORIZONS}, "must_not_change": True},
            "covid_filter_column": {"value": "is_covid_lockdown_period", "must_not_change": True},
        },
        "durable_rules": [
            {
                "rule": "Use multiplicative ratio (y/anchor), NOT additive residual (y-anchor).",
                "why": "Additive residual failed in iter006. Ratio removes level shift multiplicatively.",
                "applies_to": ["training"]
            },
            {
                "rule": "Convert ratio predictions back to bread_quantity units before floor comparison: pred_units = pred_ratio x anchor.",
                "why": "Baseline floor is in units (MAE of bread_quantity). Ratio MAE is not comparable.",
                "applies_to": ["training"]
            },
            {
                "rule": "Select algorithm per horizon; do not use LightGBM for h=60 and h=90.",
                "why": "h=60 has only 35 scoreable val rows; h=90 has only 5. Tree models cannot generalize on such small samples.",
                "applies_to": ["training"]
            }
        ],
    }
}

hist_path = RUN_DIR / 'history' / f'{ITER_ID}.json'
with open(str(hist_path), 'w', encoding='utf-8') as f:
    json.dump(history_json, f, indent=2)
print(f"  Saved: {hist_path.relative_to(RUN_DIR)}")

# %% Update state/current.json
print("\n[Step 9] Updating state/current.json...")
state_path = RUN_DIR / 'state' / 'current.json'
with open(str(state_path), 'r', encoding='utf-8') as f:
    state = json.load(f)

cur = state['current']
cur['feature_engineering_output_path'] = str(OUT_CSV.relative_to(RUN_DIR))
cur['feature_engineering_iterations']  = cur.get('feature_engineering_iterations', 1) + 1
cur['ratio_target_cols']   = {str(h): f'ratio_h{h}' for h in HORIZONS}
cur['anchor_cols']         = {str(h): ANCHOR_COLS[h] for h in HORIZONS}
cur['ratio_clip']          = list(RATIO_CLIP)
cur['ratio_stability']     = history_json['stability_verdict']
cur['next_iteration_contract'] = history_json['next_iteration_contract']

state['history'].append({
    "iteration": ITER_NUM,
    "node": "feature_engineering",
    "timestamp": pd.Timestamp.now().isoformat()[:10],
    "status": "complete",
    "summary": (
        f"Added ratio_h{{h}} = bread_quantity_h{{h}} / anchor_h{{h}} for h in {HORIZONS}. "
        f"Clipped to [{RATIO_CLIP[0]}, {RATIO_CLIP[1]}]. "
        "Output: data/iteration_009_feature_engineering.csv. "
        "Next: iter010 training on ratio target with per-horizon algorithm selection."
    ),
    "full_output_path": str(hist_path.relative_to(RUN_DIR)),
    "key_outputs": {
        "output_csv": str(OUT_CSV.relative_to(RUN_DIR)),
        "new_columns": new_cols,
        "ratio_stability": history_json['stability_verdict'],
    }
})

with open(str(state_path), 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2)
print("  state/current.json updated")

# %% Summary
print("\n" + "="*70)
print("SUMMARY - Iteration 009: Feature Engineering (ratio targets)")
print("="*70)
print(f"Added columns: {new_cols}")
print(f"Output CSV:    {OUT_CSV.name}")
print(f"\n  {'h':>4} | {'train_mean':>10} | {'covid_mean':>10} | {'val_mean':>8} | {'stable':>8}")
print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
for h in HORIZONS:
    tm = ratio_stats[h]['train_noncovid']['mean']
    cm = ratio_stats[h]['train_covid']['mean']
    vm = ratio_stats[h]['val']['mean']
    st = history_json['stability_verdict'][h]
    print(f"  {h:>4} | {str(tm):>10} | {str(cm):>10} | {str(vm):>8} | {st:>8}")

print("\nNext: iter010 training (ratio target, per-horizon algorithm selection)")
print("\nDone.")
