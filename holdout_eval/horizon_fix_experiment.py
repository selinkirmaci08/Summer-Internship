"""Quantify the two horizon fixes on the holdout, separately and together.

POST-HOC DIAGNOSTIC, NOT MODEL SELECTION. This re-runs a known defect with a pre-specified
correction to measure what the defect cost. All four configurations are reported, never only the
best, and none of them is a new champion: numbers produced after seeing the holdout cannot be used
to choose a model. What they can do is size a defect the run has already been shown to have.

Held fixed across all four: the 37-feature list, the frozen hyperparameters, the pinned
n_estimators, the training rows (both ends of every pair inside the training partition), the
holdout rows, and the trading-day rule. The only things that vary are the two axes below.

  targets   row_shift   df[target].shift(-h)      -- h ROWS ahead, what the run did
            date_join   value at row_date + h days -- h CALENDAR days ahead

  calendar  origin      calendar features describe the ORIGIN date, what the run did
            target      calendar features describe the TARGET date (legitimate: the date of a
                        future day is known at forecast time)
"""

from __future__ import annotations

import itertools
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

import build_frame as bf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
CUTOFF = pd.Timestamp("2020-12-31")
W_FROM, W_TO = pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")

LGBM_PARAMS = {
    "objective": "regression_l1", "metric": "mae", "learning_rate": 0.03, "num_leaves": 31,
    "min_child_samples": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.05, "reg_lambda": 0.1, "verbose": -1, "random_state": 42,
}
PINNED = {1: 185, 7: 141, 14: 274, 30: 119, 60: 52, 90: 69}

# Calendar features are a function of a date, so they can be recomputed for any date. Everything
# else (lags, weather, competitor) is an observation and stays anchored at the origin.
CAL = ["year", "doy", "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
       "is_weekend", "is_public_holiday", "year_centered"]
ORIGIN_ONLY = [f for f in bf.FEATURES if f not in CAL]
HOLIDAYS = bf.belgian_holidays(2010, 2024)


def calendar_block(dates: pd.Series) -> pd.DataFrame:
    """The run's own calendar recipe, evaluated on whatever dates it is handed."""
    d = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=range(len(d)))
    out["year"] = d.year
    out["doy"] = d.dayofyear
    out["dow_sin"] = np.sin(2 * np.pi * d.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * d.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * d.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * d.month / 12)
    out["doy_sin"] = np.sin(2 * np.pi * d.dayofyear / 365)
    out["doy_cos"] = np.cos(2 * np.pi * d.dayofyear / 365)
    out["is_weekend"] = (d.dayofweek >= 5).astype(int)
    out["is_public_holiday"] = pd.Series(d.isin(HOLIDAYS)).astype(int).to_numpy()
    out["year_centered"] = d.year - 2015
    return out


def build_pairs(frame, targets):
    """One row per (origin, target, h) with both dates, for every horizon."""
    pos = {d: i for i, d in enumerate(frame["date"])}
    dates = frame["date"]
    rows = []
    for h in bf.HORIZONS:
        if targets == "row_shift":
            tgt_date = dates.shift(-h)
        else:
            wanted = dates + pd.Timedelta(days=h)
            tgt_date = wanted.where(wanted.isin(pos))
        ok = tgt_date.notna()
        sub = pd.DataFrame({
            "origin_date": dates[ok].to_numpy(),
            "target_date": pd.DatetimeIndex(tgt_date[ok]),
            "h": h,
            "origin_idx": [pos[d] for d in dates[ok]],
        })
        sub["y"] = [frame[bf.TARGET].iloc[pos[d]] for d in sub["target_date"]]
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def matrix(frame, pairs, calendar):
    X = frame[ORIGIN_ONLY].iloc[pairs["origin_idx"]].reset_index(drop=True)
    src = pairs["target_date"] if calendar == "target" else pairs["origin_date"]
    return pd.concat([X, calendar_block(src)], axis=1)[bf.FEATURES]


def run_config(frame, targets, calendar):
    pairs = build_pairs(frame, targets)
    X = matrix(frame, pairs, calendar)

    fit = (pairs["origin_date"] <= CUTOFF) & (pairs["target_date"] <= CUTOFF) & pairs["y"].notna()
    score = (pairs["target_date"] >= W_FROM) & (pairs["target_date"] <= W_TO) & pairs["y"].notna()

    preds = {}
    for h in bf.HORIZONS:
        tr = fit & (pairs["h"] == h)
        te = score & (pairs["h"] == h)
        model = lgb.LGBMRegressor(n_estimators=PINNED[h], **LGBM_PARAMS)
        model.fit(X[tr.to_numpy()], pairs.loc[tr, "y"])
        p = pairs.loc[te, ["target_date", "h", "y"]].copy()
        p["y_pred"] = model.predict(X[te.to_numpy()])
        preds[h] = p
    return pd.concat(preds.values(), ignore_index=True)


def main():
    frame = pd.read_csv(os.path.join(OUT, "frame_holdout_primary.csv"), parse_dates=["date"])
    results, tables = {}, {}
    for targets, calendar in itertools.product(("row_shift", "date_join"), ("origin", "target")):
        name = f"targets={targets:9s} calendar={calendar}"
        tables[name] = run_config(frame, targets, calendar)
        print("fitted:", name)

    # score every configuration on the rows all of them share, and the trading rule
    key = ["target_date", "h"]
    common = None
    for t in tables.values():
        k = t[t["y"] > 0][key]
        common = k if common is None else common.merge(k, on=key)
    print(f"\nrows common to all four configurations, after the trading rule: {len(common)}\n")

    hdr = f"{'configuration':38s}" + "".join(f"{'h='+str(h):>8}" for h in bf.HORIZONS) + f"{'POOLED':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, t in tables.items():
        d = t.merge(common, on=key)
        row = [np.abs(d.loc[d.h == h, "y_pred"] - d.loc[d.h == h, "y"]).mean() for h in bf.HORIZONS]
        pooled = float(np.abs(d["y_pred"] - d["y"]).mean())
        results[name] = {"per_horizon": dict(zip(map(str, bf.HORIZONS), row)), "pooled": pooled,
                         "n": int(len(d))}
        print(f"{name:38s}" + "".join(f"{v:>8.2f}" for v in row) + f"{pooled:>9.2f}")

    with open(os.path.join(OUT, "horizon_fix_experiment.json"), "w", encoding="utf-8") as f:
        json.dump({"what_this_is": "post-hoc diagnostic sizing a known defect; not model selection",
                   "rows_scored": int(len(common)), "results": results}, f, indent=2)
    print("\n-> outputs/horizon_fix_experiment.json")


if __name__ == "__main__":
    main()
