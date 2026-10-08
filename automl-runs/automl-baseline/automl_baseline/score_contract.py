"""Score contract predictions against ground truth built from a holdout root.

Applies exactly the two rules confirmed for this evaluation, and no others:

  - till scope:      POS 001 only (the model's training scope; the second till
                      is entirely unseen by training and would charge the model
                      for a component it never saw)
  - trading-day rule: a date is scored iff the ACTUAL quantity is > 0 (ground
                      truth decides this, never the prediction)

Both are applied identically to every system being compared -- the model, and
the seasonal-naive reference `protocol.json` names as the floor.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines, metrics
from .features import prepare_series
from .sources import aernoudt


def build_actuals(holdout_root: Path, target: str, time_col: str = "date") -> pd.DataFrame:
    frame, _ = aernoudt.build(holdout_root, out_path=None, scope="chain",
                              stores=["001"], verbose=False)
    frame[time_col] = pd.to_datetime(frame[time_col])
    return frame[[time_col, target]].rename(columns={target: "y_true"})


def score(preds_path: Path, actuals: pd.DataFrame, history_daily: pd.DataFrame,
         target: str, period: int = 7) -> dict:
    preds = pd.read_csv(preds_path, parse_dates=["target_date"])
    merged = preds.merge(actuals, left_on="target_date", right_on="date", how="left")
    n_no_actual = int(merged["y_true"].isna().sum())
    merged = merged.dropna(subset=["y_true"])

    trading = merged[merged["y_true"] > 0].copy()
    excluded = len(merged) - len(trading)

    # Seasonal-naive reference, per protocol.json: last observed same-weekday value
    # at the origin. `history_daily` must span train + every holdout up to and
    # including this one, exactly like predict_contract's --data list -- a stress-
    # window origin in late 2021 needs 2021 on the calendar, or the naive silently
    # comes back NaN rather than reading a real week-ago value.
    full_series, freq = prepare_series(history_daily, "date", target, freq="D")
    y_full = full_series[target]

    results_by_h = []
    for h in sorted(trading["h"].unique()):
        sub = trading[trading["h"] == h]
        origins = pd.DatetimeIndex(sub["target_date"]) - pd.Timedelta(days=int(h))
        naive = baselines.seasonal_naive(y_full, origins, int(h), period).to_numpy()
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_pred"].to_numpy(dtype=float)

        scale = metrics.seasonal_naive_scale(
            y_full.loc[y_full.index < origins.min()] if len(origins) else y_full, period)

        m_model = metrics.all_metrics(y_true, y_pred, scale)
        m_naive = metrics.all_metrics(y_true, naive, scale)
        skill = 1 - m_model["mae"] / m_naive["mae"] if m_naive["mae"] else float("nan")

        results_by_h.append({
            "h": int(h), "n": int(len(sub)),
            "model": m_model, "seasonal_naive": m_naive,
            "skill_vs_seasonal_naive": round(float(skill), 4),
            "beats_seasonal_naive": bool(skill > 0),
        })

    overall_true = trading["y_true"].to_numpy(dtype=float)
    overall_pred = trading["y_pred"].to_numpy(dtype=float)
    # Same convention as the per-horizon scale: the denominator comes from data
    # strictly before the earliest origin scored, never from the window itself --
    # otherwise MASE would be measuring the window against its own history.
    all_origins = pd.DatetimeIndex(trading["target_date"]) - pd.to_timedelta(trading["h"], unit="D")
    overall_scale = metrics.seasonal_naive_scale(
        y_full.loc[y_full.index < all_origins.min()] if len(all_origins) else y_full, period)
    overall = metrics.all_metrics(overall_true, overall_pred, overall_scale)

    return {
        "rows_predicted": int(len(preds)),
        "rows_with_no_actual": n_no_actual,
        "rows_excluded_non_trading": excluded,
        "rows_scored": int(len(trading)),
        "overall": overall,
        "mean_skill_vs_seasonal_naive": round(
            float(np.mean([r["skill_vs_seasonal_naive"] for r in results_by_h])), 4),
        "horizons_beating_seasonal_naive": int(sum(r["beats_seasonal_naive"] for r in results_by_h)),
        "horizons_evaluated": len(results_by_h),
        "per_horizon": results_by_h,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="score-contract")
    p.add_argument("--preds", required=True)
    p.add_argument("--holdout-root", required=True)
    p.add_argument("--history-daily", action="append", required=True,
                   help="daily CSV(s) covering train + every holdout up to and including "
                        "this one; repeat for each. A stress-window origin can fall inside "
                        "an earlier holdout, and the naive reference needs it on the "
                        "calendar just as much as the model's lags do.")
    p.add_argument("--target", default="qty_g01_bread")
    p.add_argument("--period", type=int, default=7)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")
    actuals = build_actuals(Path(args.holdout_root), args.target)
    history = pd.concat(
        [pd.read_csv(p_, parse_dates=["date"]) for p_ in args.history_daily],
        ignore_index=True,
    ).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    result = score(Path(args.preds), actuals, history, args.target, args.period)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    o = result["overall"]
    print("scored {} / {} rows (excluded {} non-trading, {} with no actual)".format(
        result["rows_scored"], result["rows_predicted"],
        result["rows_excluded_non_trading"], result["rows_with_no_actual"]))
    print("overall  MAE {:.2f}  RMSE {:.2f}  WAPE {:.1f}%  MASE {:.3f}  bias {:+.2f}".format(
        o["mae"], o["rmse"], o["wape_pct"], o["mase"], o["bias"]))
    print("mean skill vs seasonal-naive: {:+.1%}  ({}/{} horizons beat it)".format(
        result["mean_skill_vs_seasonal_naive"],
        result["horizons_beating_seasonal_naive"], result["horizons_evaluated"]))
    print("-> {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
