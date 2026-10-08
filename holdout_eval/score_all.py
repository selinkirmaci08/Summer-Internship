"""Score every system on identical holdout rows.

One scorer, one set of rules, applied once to all systems so no two sides can drift:

  trading day   a (target_date, h) pair is scored iff the ACTUAL quantity is > 0. Ground truth
                decides this, never a prediction. protocol.json requires the rule be declared and
                applied identically on both sides; this is the declaration.
  till scope    POS 001 only, on every side. The second till (POS ending 301) appears from 2021,
                sells essentially no bread, and was entirely unseen by every model's training.
  rows          the intersection of the pairs every system could produce, so the headline table
                compares systems and not coverage. Each system's own coverage is reported too.

The naive references are computed here rather than imported, because the reference shipped in
automl_baseline.baselines returns one row per horizon (n=1 at h=1) and cannot be scored.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

import build_frame as bf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
AUTOML = os.path.join(os.path.dirname(HERE), "automl-runs", "automl-baseline", "outputs")

WINDOWS = {
    "holdout_primary": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"),
                        [bf.TRAIN_ROOT, bf.PRIMARY_ROOT]),
    "holdout_stress": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-02-08"),
                       [bf.TRAIN_ROOT, bf.PRIMARY_ROOT, bf.STRESS_ROOT]),
}

AUTOML_SOURCES = {
    "automl_recipe": "contract_aernoudt_split",   # FLAML, 403 generic features, 900s
    "automl_no_fe": "naive_automl",               # FLAML on raw columns, shift(h) only
}

# k=52 is one year of the same weekday - a calendar constant, fixed the same way the seasonal
# period m=7 is. It is NOT the value a grid search picks (that is k=13 on the training rows), so
# it cannot be accused of having been tuned on anything.
SEASONAL_MEDIAN_K = 52


def cached_frame(name, roots):
    path = os.path.join(OUT, f"frame_{name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    frame = bf.build(roots)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")
    return frame


def naive_predictions(frame, grid):
    """Seasonal naive (m=7) and seasonal median over a 52-week same-weekday window.

    Both look only at observed values dated <= origin, where origin = target_date - h days.
    """
    s = frame.set_index("date")[bf.TARGET]
    idx = s.index
    sn, sm = [], []
    for t, h in zip(grid["target_date"], grid["h"]):
        o = t - pd.Timedelta(days=int(h))
        hist = s[(idx <= o) & (idx.dayofweek == t.dayofweek)].to_numpy()
        sn.append(hist[-1] if len(hist) else np.nan)
        sm.append(np.median(hist[-SEASONAL_MEDIAN_K:]) if len(hist) else np.nan)
    return np.array(sn), np.array(sm)


def load_preds(path, label):
    df = pd.read_csv(path, parse_dates=["target_date"])
    df["h"] = df["h"].astype(int)
    return df[["target_date", "h", "y_pred"]].rename(columns={"y_pred": label})


def metrics(y_true, y_pred, mase_denom):
    err = y_pred - y_true
    ae = np.abs(err)
    denom = np.abs(y_true) + np.abs(y_pred)
    return {
        "n": int(len(y_true)),
        "MAE": float(np.mean(ae)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "WAPE_pct": float(100 * ae.sum() / np.abs(y_true).sum()),
        "sMAPE_pct": float(100 * np.mean(np.where(denom == 0, 0.0, 2 * ae / np.where(denom == 0, 1, denom)))),
        "MASE": float(np.mean(ae) / mase_denom),
        "bias": float(np.mean(err)),
    }


def score_window(name, w_from, w_to, roots):
    frame = cached_frame(name, roots)

    truth = pd.read_csv(os.path.join(OUT, f"truth_{name}.csv"), parse_dates=["target_date"])
    truth["h"] = truth["h"].astype(int)

    # MASE denominator: mean |y_t - y_{t-7 days}| over the TRAINING rows, so the scale is the
    # same constant in every window and comparable with the run's own figures.
    tr = frame[frame["date"] <= bf.LAST_TRAIN_ORIGIN].set_index("date")[bf.TARGET]
    d7 = (tr - tr.reindex(tr.index - pd.Timedelta(days=7)).to_numpy()).abs()
    mase_denom = float(np.nanmean(d7.to_numpy()))

    table = truth.copy()
    for path in sorted(glob.glob(os.path.join(OUT, f"preds_{name}__*.csv"))):
        label = os.path.basename(path).split("__")[1][:-4]
        table = table.merge(load_preds(path, label), on=["target_date", "h"], how="left")
    for label, folder in AUTOML_SOURCES.items():
        p = os.path.join(AUTOML, folder, f"preds_{name}.csv")
        if os.path.exists(p):
            table = table.merge(load_preds(p, label), on=["target_date", "h"], how="left")

    sn, sm = naive_predictions(frame, table)
    table["seasonal_naive_m7"] = sn
    table[f"seasonal_median_{SEASONAL_MEDIAN_K}wk"] = sm

    systems = [c for c in table.columns if c not in ("target_date", "h", "y_true")]

    # trading-day rule, from ground truth, applied once to every system
    scored = table[table["y_true"].notna() & (table["y_true"] > 0)].copy()
    coverage = {s: int(scored[s].notna().sum()) for s in systems}
    complete = scored.dropna(subset=systems)

    results = {"window": [str(w_from.date()), str(w_to.date())],
               "mase_denominator": mase_denom,
               "rows": {"target_dates_in_window": int(truth["target_date"].nunique()),
                        "pairs_before_trading_rule": int(len(table)),
                        "excluded_zero_or_missing_actual": int(len(table) - len(scored)),
                        "pairs_after_trading_rule": int(len(scored)),
                        "pairs_all_systems_complete": int(len(complete))},
               "coverage_per_system": coverage,
               "overall": {}, "per_horizon": {}}

    for s in systems:
        results["overall"][s] = metrics(complete["y_true"].to_numpy(), complete[s].to_numpy(), mase_denom)
    for h in bf.HORIZONS:
        d = complete[complete["h"] == h]
        if not len(d):
            continue
        results["per_horizon"][h] = {s: metrics(d["y_true"].to_numpy(), d[s].to_numpy(), mase_denom)
                                     for s in systems}

    ref = "seasonal_naive_m7"
    for s in systems:
        base = results["overall"][ref]["MAE"]
        results["overall"][s]["skill_vs_naive_pct"] = round(100 * (1 - results["overall"][s]["MAE"] / base), 2)
        for h in results["per_horizon"]:
            b = results["per_horizon"][h][ref]["MAE"]
            results["per_horizon"][h][s]["skill_vs_naive_pct"] = round(
                100 * (1 - results["per_horizon"][h][s]["MAE"] / b), 2)

    complete.to_csv(os.path.join(OUT, f"scored_rows_{name}.csv"), index=False, date_format="%Y-%m-%d")
    return results, systems


def main():
    all_results = {}
    for name, (w_from, w_to, roots) in WINDOWS.items():
        res, systems = score_window(name, w_from, w_to, roots)
        all_results[name] = res

        print(f"\n{'='*100}\n{name}   {res['window'][0]} .. {res['window'][1]}")
        r = res["rows"]
        print(f"  target dates {r['target_dates_in_window']} | pairs {r['pairs_before_trading_rule']} "
              f"| excluded (zero/missing actual) {r['excluded_zero_or_missing_actual']} "
              f"| scored {r['pairs_after_trading_rule']} | all-systems-complete {r['pairs_all_systems_complete']}")
        print("  coverage:", res["coverage_per_system"])
        order = sorted(systems, key=lambda s: res["overall"][s]["MAE"])
        print(f"\n  {'system':32s}{'MAE':>8}{'RMSE':>8}{'WAPE%':>8}{'MASE':>7}{'bias':>8}{'skill%':>8}")
        for s in order:
            m = res["overall"][s]
            print(f"  {s:32s}{m['MAE']:>8.2f}{m['RMSE']:>8.2f}{m['WAPE_pct']:>8.1f}"
                  f"{m['MASE']:>7.2f}{m['bias']:>8.1f}{m['skill_vs_naive_pct']:>8.1f}")

        print(f"\n  MAE by horizon")
        print(f"  {'system':32s}" + "".join(f"{'h='+str(h):>9}" for h in res["per_horizon"]))
        for s in order:
            print(f"  {s:32s}" + "".join(f"{res['per_horizon'][h][s]['MAE']:>9.2f}" for h in res["per_horizon"]))

    with open(os.path.join(OUT, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\nscores -> outputs/scores.json")


if __name__ == "__main__":
    main()
