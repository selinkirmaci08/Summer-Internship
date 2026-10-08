"""The floor: AutoML on raw columns, by someone who has not looked at the data.

This is deliberately the dumbest defensible thing. It exists so the agentic
pipeline (and the fixed-recipe baseline next door) can be measured against what
you get with *no* thinking at all -- not "generic feature engineering", but none.

What it does NOT do, and each omission is the point:

  * no lags, no rolling statistics, no calendar or Fourier terms, no exogenous
    lagging -- `features.py` is never imported
  * no calendar-aware alignment. The shift for horizon h is `DataFrame.shift(h)`,
    which moves h **rows**, not h days. The series is missing 579 calendar days;
    noticing that requires looking at the data, so this does not notice it. The
    resulting misalignment is real and is left in.
  * no handling of zeros, the negative value, missing days, or the 2020 regime
    change
  * no pooling across horizons, no horizon feature -- one independent model per
    horizon, which is what you write when you are not thinking about it
  * no feature selection, no target transform, no imputation

What it does: shift every raw column back by h rows, call that X, call the
unshifted target y, hand it to FLAML, and let FLAML do the only thinking in the
pipeline -- model choice and hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import search

# Bookkeeping, not measurements. Dropping these is the one concession, and it is
# the same concession every AutoML tutorial makes on its first line.
DROP_ALWAYS = ("date",)


def build_supervised(daily: pd.DataFrame, target: str, horizon: int,
                     time_col: str = "date") -> tuple:
    """X = every raw column shifted back h ROWS; y = the target where it stands.

    Row-shift, not day-shift. See the module docstring -- this is the naivety,
    not an oversight.
    """
    frame = daily.sort_values(time_col).reset_index(drop=True)
    feature_cols = [c for c in frame.columns if c not in DROP_ALWAYS]
    X = frame[feature_cols].shift(horizon)
    y = frame[target]
    keep = y.notna() & X.notna().any(axis=1)
    return X.loc[keep], y.loc[keep], frame.loc[keep, time_col]


def train(daily: pd.DataFrame, target: str, horizons, time_budget: int,
          out_dir: Path, metric: str = "mae", seed: int = 42,
          time_col: str = "date", label: str = "naive_automl") -> dict:
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    started = time.time()
    per_horizon = {}

    for h in sorted(int(x) for x in horizons):
        X, y, _ = build_supervised(daily, target, h, time_col)
        print("[h={:>2}]  {} rows x {} raw columns".format(h, len(X), X.shape[1]), flush=True)
        fitted = search.fit_regressor(
            X, y, time_budget=time_budget, metric=metric, seed=seed,
            log_file=str(out_dir / "models" / "flaml_{}_h{}.log".format(label, h)))
        print("        {} | validation {} {:.4f}".format(
            fitted["best_estimator"], metric, fitted["best_validation_loss"]), flush=True)

        joblib.dump({
            "model": fitted["model"],
            "feature_order": fitted["feature_order"],
            "target": target, "time_col": time_col, "horizon": h,
            "best_estimator": fitted["best_estimator"],
            "best_config": fitted["best_config"],
        }, out_dir / "models" / "{}_h{}.pkl".format(label, h))

        per_horizon[str(h)] = {
            "best_estimator": fitted["best_estimator"],
            "best_config": fitted["best_config"],
            "best_validation_loss": fitted["best_validation_loss"],
            "n_train_rows": fitted["n_train_rows"],
            "n_features": fitted["n_features"],
            "search_seconds": fitted["search_seconds"],
            "estimators_tried": fitted["estimators_tried"],
        }

    manifest = {
        "what_this_is": (
            "AutoML on raw columns with no feature engineering of any kind. The only "
            "transformation is DataFrame.shift(h), which moves h rows, not h days. "
            "This is the floor the other baselines and the agentic pipeline are "
            "measured against."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - started, 1),
        "deliberately_absent": [
            "lags", "rolling statistics", "calendar features", "Fourier terms",
            "exogenous lagging", "calendar-aware alignment (shift is by row, not day)",
            "zero/closed-day handling", "outlier handling", "imputation",
            "feature selection", "target transform", "horizon pooling",
        ],
        "target": target,
        "horizons": sorted(int(h) for h in horizons),
        "strategy": "one independent model per horizon",
        "budget_seconds_per_horizon": int(time_budget),
        "seed": seed,
        "per_horizon": per_horizon,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print("[out]  {}".format(out_dir / "manifest.json"), flush=True)
    return manifest


def predict(models_dir: Path, history: pd.DataFrame, target_dates, horizons,
            target: str, time_col: str = "date", label: str = "naive_automl") -> pd.DataFrame:
    """Same row-shift, applied to train+holdout concatenated in date order."""
    history = history.sort_values(time_col).reset_index(drop=True)
    target_dates = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(target_dates)).unique()))
    rows = []

    for h in sorted(int(x) for x in horizons):
        bundle = joblib.load(Path(models_dir) / "{}_h{}.pkl".format(label, h))
        feature_cols = [c for c in history.columns if c not in DROP_ALWAYS]
        X_all = history[feature_cols].shift(h)
        X_all.index = history[time_col]

        wanted = target_dates.intersection(X_all.index)
        Xh = X_all.loc[wanted].reindex(columns=bundle["feature_order"])
        preds = search.predict({"model": bundle["model"]}, Xh)
        rows.append(pd.DataFrame({"target_date": wanted, "h": h, "y_pred": preds}))

    return pd.concat(rows, ignore_index=True).sort_values(["target_date", "h"]).reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="naive-automl")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--daily", required=True)
    t.add_argument("--target", required=True)
    t.add_argument("--horizons", default="1,7,14,30,60,90")
    t.add_argument("--budget", type=int, default=150, help="seconds PER horizon")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out", default="outputs/naive_automl")

    q = sub.add_parser("predict")
    q.add_argument("--models", required=True)
    q.add_argument("--history", action="append", required=True,
                   help="daily CSVs, repeat for train + holdout(s)")
    q.add_argument("--target", required=True)
    q.add_argument("--horizons", default="1,7,14,30,60,90")
    q.add_argument("--from", dest="date_from", required=True)
    q.add_argument("--to", dest="date_to", required=True)
    q.add_argument("--out", required=True)

    args = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    if args.cmd == "train":
        daily = pd.read_csv(args.daily, parse_dates=["date"])
        train(daily, args.target, horizons, args.budget, Path(args.out), seed=args.seed)
    else:
        frames = [pd.read_csv(f, parse_dates=["date"]) for f in args.history]
        history = pd.concat(frames, ignore_index=True)
        history = history.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        preds = predict(Path(args.models), history,
                        pd.date_range(args.date_from, args.date_to, freq="D"),
                        horizons, args.target)
        preds["target_date"] = preds["target_date"].dt.strftime("%Y-%m-%d")
        preds.to_csv(args.out, index=False)
        print("[out]  {} rows -> {}".format(len(preds), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
