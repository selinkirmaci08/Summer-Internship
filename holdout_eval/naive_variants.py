"""Intuitive naive baselines — what a data scientist tries first.

Each uses only observations at or before origin (= target_date - h days).

Baselines
---------
  seasonal_naive_m7     last same-weekday value                       (reference)
  seasonal_median_52wk  52-week same-weekday median                   (current holdout winner)
  avg_lag7_lag14        avg(lag-7, lag-14)  — smooth noise over 2 weeks
  avg_lag1_lag7         avg(yesterday, same day last week)            — what the user described
  avg_lag7_lag364       avg(same day last week, same day last year)   — week + annual signal
  avg_lag1_lag7_lag364  avg(yesterday, same day last week, same day last year)
  avg_4wk               rolling 4-week same-weekday mean              — "one-month smoother"
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_frame as bf

OUT = os.path.join(HERE, "outputs")

WINDOWS = {
    "primary (2021)": (
        pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"),
        [bf.TRAIN_ROOT, bf.PRIMARY_ROOT],
    ),
    "stress (2022-2023-02-08)": (
        pd.Timestamp("2022-01-01"), pd.Timestamp("2023-02-08"),
        [bf.TRAIN_ROOT, bf.PRIMARY_ROOT, bf.STRESS_ROOT],
    ),
}
HORIZONS = [1, 7, 14, 30, 60, 90]

SYSTEMS = [
    "seasonal_naive_m7",
    "seasonal_median_52wk",
    "avg_lag7_lag14",
    "avg_lag1_lag7",
    "avg_lag7_lag364",
    "avg_lag1_lag7_lag364",
    "avg_4wk",
]


def load_frame(name, roots):
    path = os.path.join(OUT, f"frame_{name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return bf.build(roots)


def _pick(s_sorted, origin, dow, offset_weeks):
    """Return the float value offset_weeks same-weekday observations before origin."""
    candidates = s_sorted[(s_sorted.index <= origin) & (s_sorted.index.dayofweek == dow)]
    if len(candidates) < offset_weeks:
        return np.nan
    return float(candidates.iloc[-offset_weeks])


def _pick_by_calendar_days(s_sorted, origin, dow, approx_days):
    """Return the most recent same-weekday observation near origin - approx_days."""
    target_anchor = origin - pd.Timedelta(days=approx_days)
    # look for a same-weekday observation within ±10 days of the anchor
    window = s_sorted[
        (s_sorted.index.dayofweek == dow)
        & (s_sorted.index >= target_anchor - pd.Timedelta(days=10))
        & (s_sorted.index <= target_anchor + pd.Timedelta(days=10))
        & (s_sorted.index <= origin)
    ]
    if len(window) == 0:
        return np.nan
    return float(window.iloc[-1])


def compute_baselines(frame, truth):
    s = frame.set_index("date")[bf.TARGET].sort_index()
    idx = s.index

    results = {k: [] for k in SYSTEMS}

    for t, h in zip(truth["target_date"], truth["h"]):
        origin = t - pd.Timedelta(days=int(h))
        dow = t.dayofweek

        same_dow = s[(idx <= origin) & (idx.dayofweek == dow)]

        # reference baselines
        lag7  = float(same_dow.iloc[-1])  if len(same_dow) >= 1 else np.nan
        lag14 = float(same_dow.iloc[-2])  if len(same_dow) >= 2 else np.nan
        lag4w = float(np.mean(same_dow.iloc[-4:])) if len(same_dow) >= 1 else np.nan

        # yesterday: most recent trading day before or at origin
        any_day = s[idx <= origin]
        lag1 = float(any_day.iloc[-1]) if len(any_day) >= 1 else np.nan

        # same day approximately 52 weeks ago (±10 days, most recent match)
        lag364 = _pick_by_calendar_days(s, origin, dow, 364)

        # 52-week median
        med52 = float(np.median(same_dow.iloc[-52:])) if len(same_dow) >= 1 else np.nan

        results["seasonal_naive_m7"].append(lag7)
        results["seasonal_median_52wk"].append(med52)
        results["avg_lag7_lag14"].append(
            float(np.nanmean([lag7, lag14])) if not (np.isnan(lag7) and np.isnan(lag14)) else np.nan
        )
        results["avg_lag1_lag7"].append(
            float(np.nanmean([lag1, lag7])) if not (np.isnan(lag1) and np.isnan(lag7)) else np.nan
        )
        results["avg_lag7_lag364"].append(
            float(np.nanmean([lag7, lag364])) if not (np.isnan(lag7) and np.isnan(lag364)) else np.nan
        )
        results["avg_lag1_lag7_lag364"].append(
            float(np.nanmean([lag1, lag7, lag364]))
            if not all(np.isnan(v) for v in [lag1, lag7, lag364]) else np.nan
        )
        results["avg_4wk"].append(lag4w)

    out = truth.copy()
    for k, v in results.items():
        out[k] = v
    return out


def mae(y, p):
    mask = ~(np.isnan(y) | np.isnan(p)) & (y > 0)
    return float(np.mean(np.abs(y[mask] - p[mask]))) if mask.sum() else np.nan

def bias(y, p):
    mask = ~(np.isnan(y) | np.isnan(p)) & (y > 0)
    return float(np.mean(p[mask] - y[mask])) if mask.sum() else np.nan


def report(label, df):
    print(f"\n{'='*110}")
    print(f"  {label}")
    print(f"{'='*110}")

    scored = df[df["y_true"].notna() & (df["y_true"] > 0)].copy()
    y_s = scored["y_true"].to_numpy(dtype=float)
    naive_mae = mae(y_s, scored["seasonal_naive_m7"].to_numpy(dtype=float))

    rows = []
    for s in SYSTEMS:
        p = scored[s].to_numpy(dtype=float)
        m = mae(y_s, p)
        b = bias(y_s, p)
        skill = 100 * (1 - m / naive_mae) if naive_mae else np.nan
        rows.append((m, s, b, skill))

    # pooled
    print(f"\n  Pooled MAE (n={len(scored)}, reference = seasonal_naive_m7 MAE = {naive_mae:.2f})")
    print(f"  {'baseline':35s}  {'MAE':>8}  {'vs naive':>10}  {'bias':>8}")
    for m_, s, b, sk in sorted(rows):
        marker = " <--" if s == "seasonal_naive_m7" else ""
        print(f"  {s:35s}  {m_:>8.2f}  {sk:>+9.1f}%  {b:>+8.1f}{marker}")

    # per-horizon
    print(f"\n  MAE by horizon")
    print(f"  {'baseline':35s}" + "".join(f"{'h='+str(h):>9}" for h in HORIZONS))
    for m_, s, _, _ in sorted(rows):
        cells = []
        for h in HORIZONS:
            d = scored[scored["h"] == h]
            cells.append(f"{mae(d['y_true'].to_numpy(dtype=float), d[s].to_numpy(dtype=float)):>9.2f}")
        print(f"  {s:35s}" + "".join(cells))


def main():
    for win_label, (w_from, w_to, roots) in WINDOWS.items():
        name = "holdout_primary" if "primary" in win_label else "holdout_stress"
        frame = load_frame(name, roots)
        truth = pd.read_csv(
            os.path.join(OUT, f"truth_{name}.csv"), parse_dates=["target_date"]
        )
        truth["h"] = truth["h"].astype(int)
        truth = truth[(truth["target_date"] >= w_from) & (truth["target_date"] <= w_to)]

        df = compute_baselines(frame, truth)
        report(win_label, df)


if __name__ == "__main__":
    main()
