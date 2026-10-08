"""Use the trained model: give it a time, get a prediction.

    # one date
    python -m automl_baseline.predict --run outputs/<run> --at 2023-03-15

    # a horizon instead of a date
    python -m automl_baseline.predict --run outputs/<run> --horizon 30

    # everything from t+1 to t+90, one row per day
    python -m automl_baseline.predict --run outputs/<run> --through 90 --csv forecast.csv

The origin defaults to the last timestamp in the data that has an observed value --
"where we are now". Anything you ask for is measured forward from there.

With the pooled strategy this loads ONE model and calls it once per requested
timestamp, with `horizon` as an input. With the direct strategy it loads the
per-horizon model matching each request, and refuses horizons that were never
trained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig
from .features import HorizonDataset, prepare_series
from .ingest import load_dataset
from .search import predict as _predict


def _load_run(run_dir: Path) -> dict:
    with open(Path(run_dir) / "metrics.json", encoding="utf-8") as fh:
        return json.load(fh)


def _rebuild_dataset(result: dict):
    """Recreate the exact feature pipeline the run was trained with."""
    cfg = result["config"]
    raw, _ = load_dataset(
        cfg["data_path"], cfg.get("time_col"),
        series_id=cfg.get("series_id"), series=cfg.get("series"), aggregate=cfg.get("aggregate"),
    )
    time_col = result["task"]["time_column"]
    target = result["task"]["target"]
    frame, freq = prepare_series(raw, time_col, target, cfg.get("freq"))
    ds = HorizonDataset(frame, target, freq, known_future=cfg.get("known_future") or ())
    return ds, frame, freq, target


def _load_model(run_dir: Path, horizon: int, pooled: bool):
    import joblib
    name = "automl_pooled.pkl" if pooled else "automl_h{}.pkl".format(horizon)
    path = Path(run_dir) / "models" / name
    if not path.exists():
        raise FileNotFoundError(
            "no saved model at {}. With --strategy direct only the trained horizons have a "
            "model; re-run with --strategy pooled for a single model that answers any "
            "horizon.".format(path))
    return {"model": joblib.load(path)}


def forecast(run_dir, at=None, horizon=None, through=None, origin=None) -> pd.DataFrame:
    """Predict the target at the requested time(s).

    Exactly one of `at` (a timestamp), `horizon` (steps ahead) or `through` (every
    step from 1 to N) is used, in that order of precedence.
    """
    run_dir = Path(run_dir)
    result = _load_run(run_dir)
    ds, frame, freq, target = _rebuild_dataset(result)
    pooled = "pooled" in result["protocol"]["strategy"]
    offset = pd.tseries.frequencies.to_offset(freq)

    observed = frame[target].dropna()
    if observed.empty:
        raise ValueError("no observed values in the series")
    t0 = pd.Timestamp(origin) if origin else observed.index[-1]
    if t0 not in frame.index:
        raise ValueError("origin {} is not on the series grid ({} .. {}, freq {})".format(
            t0.date(), frame.index.min().date(), frame.index.max().date(), freq))

    if at is not None:
        target_time = pd.Timestamp(at)
        steps = (target_time - t0) / pd.Timedelta(offset) if freq.upper().startswith("D") else None
        if steps is None:
            # Non-daily frequencies: count grid steps rather than divide durations.
            future = pd.date_range(t0, target_time, freq=freq)
            steps = len(future) - 1
        steps = int(round(steps))
        if steps < 1:
            raise ValueError("{} is not after the origin {}".format(target_time.date(), t0.date()))
        horizons = [steps]
    elif horizon is not None:
        horizons = [int(horizon)]
    elif through is not None:
        horizons = list(range(1, int(through) + 1))
    else:
        horizons = [1]

    trained = result["protocol"].get("horizons_trained") or []
    lo, hi = (min(trained), max(trained)) if trained else (None, None)
    rows = []
    shared = _load_model(run_dir, horizons[0], True) if pooled else None
    for h in horizons:
        X_all = ds.build_features_at(h)
        if t0 not in X_all.index:
            raise ValueError("origin {} has no usable feature row at horizon {}".format(t0.date(), h))
        x = X_all.loc[[t0]].copy()
        if pooled:
            x["horizon"] = h
            fitted = shared
        else:
            fitted = _load_model(run_dir, h, False)
        pred = float(_predict(fitted, x)[0])
        rows.append({
            "origin_time": t0,
            "horizon": h,
            "target_time": t0 + h * offset,
            "prediction": pred,
            # Interpolating between trained horizons is what a numeric `horizon`
            # feature is for; going outside the trained range is the risky case.
            "interpolated_horizon": bool(trained and h not in trained and lo <= h <= hi),
            "extrapolated_horizon": bool(trained and (h < lo or h > hi)),
        })

    out = pd.DataFrame(rows)
    out.attrs["target"] = target
    out.attrs["model"] = "pooled" if pooled else "direct"
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="automl-baseline-predict",
        description="Give the trained model a time; it predicts.")
    p.add_argument("--run", required=True, help="a run directory produced by automl_baseline")
    p.add_argument("--at", default=None, help="predict this timestamp, e.g. 2023-03-15")
    p.add_argument("--horizon", type=int, default=None, help="predict this many steps ahead")
    p.add_argument("--through", type=int, default=None,
                   help="predict every step from 1 to N (gives the whole path, not just the endpoint)")
    p.add_argument("--origin", default=None,
                   help="forecast from this timestamp instead of the last observed one")
    p.add_argument("--csv", default=None, help="write the predictions to this CSV")
    args = p.parse_args(argv)

    out = forecast(args.run, at=args.at, horizon=args.horizon, through=args.through,
                   origin=args.origin)
    target = out.attrs.get("target", "target")
    print("forecasting {} from origin {} ({} model)".format(
        target, out.origin_time.iloc[0].date(), out.attrs.get("model")))
    show = out if len(out) <= 20 else pd.concat([out.head(10), out.tail(10)])
    for _, r in show.iterrows():
        flag = "  <- OUTSIDE the trained horizon range" if r["extrapolated_horizon"] else ""
        print("  t+{:<4} {}  {:>12.2f}{}".format(
            int(r["horizon"]), r["target_time"].date(), r["prediction"], flag))
    if len(out) > 20:
        print("  ... {} rows total".format(len(out)))
    n_interp = int(out["interpolated_horizon"].sum())
    if n_interp:
        print("  ({} of {} horizons sit between trained grid points - the model interpolates "
              "over `horizon`; raise --max-train-rows at training time for a denser grid)".format(
                  n_interp, len(out)))
    if args.csv:
        out.to_csv(args.csv, index=False)
        print("\nwritten: {}".format(args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
