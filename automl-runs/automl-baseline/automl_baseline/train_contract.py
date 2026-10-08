"""Train one frozen AutoML model against an external scoring contract.

This differs from `runner.py` in one way that matters: it carves out **no**
held-out window of its own. The whole training partition is fitted, because the
test rows live somewhere this process is never given a path to. Nothing this
module prints is a held-out number, and it does not pretend otherwise.

Hyperparameters are selected by FLAML's forward-chaining time-series CV *inside*
the training partition (split_type="time"), so the search never sees a fold that
sits after the fold it is scored on, and never sees a holdout row at all.

What it writes is designed to be inspectable after the fact: the full feature list,
the chosen estimator, its hyperparameters, and every protocol decision that was
applied -- not only the top-15 importances `runner.py` keeps.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import search
from .features import DEFAULT_LAGS, DEFAULT_WINDOWS, HorizonDataset, prepare_series


def blank_non_trading(frame: pd.DataFrame, target: str, rule: str = "none"):
    """Optionally blank the target on days the shop did not trade.

    **The default is `none`, and that is the point of this baseline.** Deciding that
    a zero means "closed" rather than "sold nothing", and withholding those rows
    from the model, is a judgement about the data. A baseline exists to answer what
    an off-the-shelf tool reaches when nobody makes such judgements for it, so the
    data goes in as the ETL produced it -- zeros, the one negative value, and all.
    Every judgement the model is compared against must be on the other side of the
    comparison, not quietly built into this one.

    `target_gt_zero` remains available so the choice can be *measured* rather than
    argued about, but it is opt-in and it is recorded in the manifest when used.
    When it is applied, blanking is used rather than deletion so a lag of 7 still
    means seven calendar days back.
    """
    if rule in (None, "none", ""):
        return frame.copy(), 0
    if rule != "target_gt_zero":
        raise ValueError("unknown trading-day rule: {}".format(rule))
    out = frame.copy()
    non_trading = out[target].notna() & (out[target] <= 0)
    out.loc[non_trading, target] = np.nan
    return out, int(non_trading.sum())


def train(
    daily: pd.DataFrame,
    target: str,
    time_col: str,
    horizons,
    time_budget: int,
    out_dir: Path,
    metric: str = "mae",
    seed: int = 42,
    estimators=(),
    dense_grid: bool = True,
    max_train_rows: int = 200_000,
    known_future=(),
    trading_rule: str = "none",
    label: str = "automl_contract",
    provenance=None,
) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    started = time.time()

    daily = daily.copy()
    daily[time_col] = pd.to_datetime(daily[time_col])
    daily, n_blanked = blank_non_trading(daily, target, trading_rule)

    prepared, freq = prepare_series(daily, time_col, target, freq="D")
    n_observed = int(prepared[target].notna().sum())
    print("[series] {} -> {} | {} calendar days | {} observed | trading-day rule: {} "
          "({} blanked)".format(
              prepared.index.min().date(), prepared.index.max().date(),
              len(prepared), n_observed, trading_rule, n_blanked), flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = HorizonDataset(prepared, target, freq, known_future=list(known_future))
        grid = ds.horizon_grid(horizons, dense=dense_grid)
        X, y, origin_time, target_time, horizon = ds.build_pooled(
            grid, max_rows=max_train_rows, must_keep=horizons)

    trained_grid = sorted({int(h) for h in horizon})
    print("[matrix] {} rows x {} features | horizons trained {}..{} ({} of them)".format(
        len(X), X.shape[1], min(trained_grid), max(trained_grid), len(trained_grid)), flush=True)
    print("[search] FLAML {}s, metric={}, time-ordered CV inside the training partition".format(
        time_budget, metric), flush=True)

    log_path = out_dir / "models" / "flaml_{}.log".format(label)
    fitted = search.fit_regressor(
        X, y, time_budget=time_budget, metric=metric, seed=seed,
        estimators=estimators, log_file=str(log_path))

    print("[model]  {} | validation {} {:.4f} | {} features".format(
        fitted["best_estimator"], metric, fitted["best_validation_loss"],
        fitted["n_features"]), flush=True)

    model_path = out_dir / "models" / "{}.pkl".format(label)
    joblib.dump({
        "model": fitted["model"],
        "feature_order": fitted["feature_order"],
        "target": target,
        "time_col": time_col,
        "freq": freq,
        "lags": list(DEFAULT_LAGS),
        "windows": list(DEFAULT_WINDOWS),
        "known_future": list(known_future),
        "trading_rule": trading_rule,
        "horizons_trained": trained_grid,
        "horizons_scored": sorted(int(h) for h in horizons),
        "best_estimator": fitted["best_estimator"],
        "best_config": fitted["best_config"],
    }, model_path)

    manifest = {
        "what_this_is": (
            "One frozen AutoML model fitted on the whole training partition. No held-out "
            "window was carved from it; the test rows are external and this process was "
            "never given a path to them. No number in this file is a held-out score."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "wall_seconds": round(time.time() - started, 1),
        "provenance": provenance or {},
        "protocol_decisions": {
            "principle": (
                "This baseline contains no judgements about the data. Every choice below "
                "is either mechanical (forced by a file format), inherited from the ETL's "
                "previously confirmed DECISIONS, or read from the scoring contract. "
                "Nothing was decided by reasoning about what the numbers appear to mean -- "
                "that work belongs on the other side of the comparison."
            ),
            "target": target,
            "trading_day_rule": (
                "none applied. Zeros, the negative value and every exported day go into "
                "training exactly as the ETL produced them. Treating a zero as 'closed' "
                "and withholding it would be a judgement about the data, which is the one "
                "thing a baseline must not contain. The scoring contract's trading-day "
                "rule is applied by the scorer, to both sides at once."
            ),
            "trading_rule_parameter": trading_rule,
            "non_trading_days_blanked": n_blanked,
            "duplicate_export_policy": "newest creation stamp (applied by the ETL)",
            "till_scope": (
                "chain -- every POS code summed per day, the ETL's confirmed DECISIONS "
                "value, unchanged. In this training partition that is POS 001 alone, "
                "because it is the only till present. Note that a second till (POS ending "
                "301) appears from 2021, so on the holdout side 'chain' and 'POS 001 only' "
                "are no longer the same thing; protocol.json requires that choice to be "
                "declared and applied identically on both sides."
            ),
            "exogenous_at_forecast_time": (
                "Every non-target column is lagged. Weather is NOT read at the target "
                "date: at forecast time you have a forecast, not an observation. "
                "known_future = {}".format(list(known_future) or "none")
            ),
            "recursion": (
                "None. Each row predicts t+h in one jump from the origin t = (t+h) - h."
            ),
        },
        "training_data": {
            "rows_in_daily_table": int(len(daily)),
            "calendar_days": int(len(prepared)),
            "observed_days": n_observed,
            "first_date": str(prepared.index.min().date()),
            "last_date": str(prepared.index.max().date()),
            "n_source_columns": int(daily.shape[1]),
        },
        "supervised_matrix": {
            "rows": int(len(X)),
            "n_features": int(X.shape[1]),
            "horizons_trained": trained_grid,
            "horizons_scored": sorted(int(h) for h in horizons),
            "dense_grid": dense_grid,
            "max_train_rows": max_train_rows,
            "recipe": {
                "target_lags": list(DEFAULT_LAGS),
                "rolling_windows": list(DEFAULT_WINDOWS),
                "rolling_stats": ["mean", "std", "min", "max"],
                "derived": ["diff_1", "diff_7", "ratio_lag1_roll7", "expanding_mean"],
                "calendar": ("dow, weekend, day, month, quarter, year, weekofyear, "
                             "dayofyear, month start/end, Fourier sin/cos k=1,2 weekly+yearly"),
                "calendar_evaluated_at": "the target date t+h (dates are known in advance)",
                "exogenous": ("every numeric non-target column lagged + rolling means; "
                              "low-cardinality categoricals passed through as category"),
                "horizon_as_feature": True,
            },
        },
        "search": {
            "engine": "FLAML",
            "best_estimator": fitted["best_estimator"],
            "best_config": fitted["best_config"],
            "best_validation_loss": fitted["best_validation_loss"],
            "metric": metric,
            "budget_seconds": int(time_budget),
            "search_seconds": fitted["search_seconds"],
            "cv_splits": fitted["cv_splits"],
            "validation": ("forward-chaining time-series CV inside the training partition "
                           "(split_type='time'); no holdout row was ever seen"),
            "estimators_tried": fitted["estimators_tried"],
            "seed": seed,
        },
        "features": {
            "n_features": fitted["n_features"],
            "feature_order": fitted["feature_order"],
            "top_importances": search.feature_importance(fitted, fitted["feature_order"], top_n=30),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": _versions(),
        },
        "artifacts": {
            "model": str(model_path),
            "flaml_log": str(log_path),
        },
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print("[out]    {}".format(out_dir / "manifest.json"), flush=True)
    print("[out]    {}".format(model_path), flush=True)
    return manifest


def _versions() -> dict:
    import flaml
    import lightgbm
    import sklearn
    import xgboost
    return {
        "flaml": flaml.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="train-contract",
        description="Fit one frozen AutoML model on a whole training partition.")
    p.add_argument("--daily", required=True, help="daily CSV produced by the ETL")
    p.add_argument("--target", required=True)
    p.add_argument("--time-col", default="date")
    p.add_argument("--horizons", default="1,7,14,30,60,90")
    p.add_argument("--budget", type=int, default=900, help="FLAML search seconds")
    p.add_argument("--metric", default="mae")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--estimators", default="")
    p.add_argument("--sparse-grid", action="store_true",
                   help="train only on the scored horizons, not every step 1..max")
    p.add_argument("--max-train-rows", type=int, default=200_000)
    p.add_argument("--known-future", default="",
                   help="columns whose value at t+h is genuinely known at the origin")
    p.add_argument("--trading-rule", default="none", choices=["none", "target_gt_zero"],
                   help="'none' (the default, and what makes this a baseline) feeds the data "
                        "in exactly as the ETL produced it; 'target_gt_zero' withholds "
                        "non-positive days, which is a judgement about the data")
    p.add_argument("--out", default="outputs/contract_train")
    p.add_argument("--label", default="automl_contract")
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")
    daily = pd.read_csv(args.daily, parse_dates=[args.time_col])
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    train(
        daily, args.target, args.time_col, horizons, args.budget, Path(args.out),
        metric=args.metric, seed=args.seed,
        estimators=[e for e in args.estimators.split(",") if e.strip()],
        dense_grid=not args.sparse_grid, max_train_rows=args.max_train_rows,
        known_future=[c for c in args.known_future.split(",") if c.strip()],
        trading_rule=args.trading_rule, label=args.label,
        provenance={"daily_table": str(Path(args.daily).resolve())},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
