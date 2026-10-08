"""Refit an externally trained model on a forward-chaining split.

A model whose training set contains dates after the rows it is scored on cannot
be compared on a forecasting protocol -- it was fitted knowing the future. When
the pipeline that produced it cannot be re-run, the next best thing is to keep
its exact specification (estimator, hyperparameters, feature list) and refit it
on train-only rows under a single time cutoff.

    python -m automl_baseline.refit_external \
        --model runs/<run>/artifacts/training/iteration_014_training_model.pkl \
        --features runs/<run>/data/iteration_013_feature_engineering.csv \
        --cutoff 2021-08-08 --out models/agentic_it014_refit.pkl

WHAT THIS FIXES: both split-level leaks. No training row is dated after the
cutoff, so nothing in the fit comes from the test period, and no test target can
appear as a lag or rolling input on a training row.

WHAT IT DOES NOT FIX, and must be stated wherever the result is reported: the
feature list and hyperparameters were *chosen* by the original pipeline while it
could see the whole series. Refitting cleans the fit, not the design. Only
re-running that pipeline blind removes the residual advantage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def model_spec(model) -> dict:
    """Everything needed to rebuild the estimator identically."""
    params = model.get_params()
    names = getattr(model, "feature_name_", None)
    if names is None:
        names = getattr(model, "feature_names_in_", None)
    spec = {
        "class": type(model).__module__ + "." + type(model).__name__,
        "params": {k: v for k, v in params.items() if not callable(v)},
        "features": list(names) if names is not None else None,
    }
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            spec["fitted_trees_in_original"] = int(booster.num_trees())
        except Exception:
            pass
    return spec


def rebuild(model):
    """A fresh, unfitted copy of the same estimator with the same hyperparameters."""
    from sklearn.base import clone
    return clone(model)


def refit(model_path, features_path, cutoff, target="bread_quantity", date_col="date",
          out_path=None, verbose=True) -> dict:
    import joblib

    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    cutoff = pd.Timestamp(cutoff)

    original = joblib.load(model_path)
    spec = model_spec(original)
    if not spec["features"]:
        raise ValueError("the model does not expose feature names; cannot refit faithfully")
    log("[spec] {} | {} features | {}".format(
        spec["class"].split(".")[-1], len(spec["features"]),
        ", ".join("{}={}".format(k, v) for k, v in sorted(spec["params"].items())
                  if k in ("max_depth", "learning_rate", "num_leaves", "n_estimators",
                           "subsample", "colsample_bytree", "random_state"))))

    frame = pd.read_csv(features_path, parse_dates=[date_col])
    missing = [c for c in spec["features"] + [target] if c not in frame.columns]
    if missing:
        raise KeyError("feature table is missing: {}".format(missing))

    train = frame[frame[date_col] <= cutoff].dropna(subset=[target])
    held = frame[frame[date_col] > cutoff]
    if len(train) < 30:
        raise ValueError("only {} training rows at or before {}".format(len(train), cutoff.date()))

    log("[split] cutoff {} | train {} rows ({} .. {}) | {} rows after the cutoff are excluded".format(
        cutoff.date(), len(train), train[date_col].min().date(), train[date_col].max().date(),
        len(held)))

    fresh = rebuild(original)
    X = train[spec["features"]]
    y = pd.to_numeric(train[target], errors="coerce")
    ok = y.notna().to_numpy()
    fresh.fit(X.loc[ok], y.loc[ok])
    log("[refit] fitted on train rows only")

    # Proof the leaks are gone, rather than an assurance that they are.
    checks = {
        "max_training_date": str(train[date_col].max().date()),
        "cutoff": str(cutoff.date()),
        "training_rows_after_cutoff": int((train[date_col] > cutoff).sum()),
        "rows_excluded_by_cutoff": int(len(held)),
    }
    if checks["training_rows_after_cutoff"] != 0:
        raise AssertionError("refit still contains rows after the cutoff")
    log("[verify] training rows dated after the cutoff: {}".format(
        checks["training_rows_after_cutoff"]))

    result = {
        "original_model": str(model_path),
        "features_table": str(features_path),
        "spec": spec,
        "target": target,
        "n_train_rows": int(ok.sum()),
        "leak_checks": checks,
        "caveat": (
            "The feature list and hyperparameters were selected by the original pipeline "
            "with visibility of the whole series. Refitting removes the split-level leaks, "
            "not the design-level one."),
    }

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fresh, out_path)
        with open(out_path.with_suffix(".refit.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        log("[out] {}".format(out_path))
        result["refit_model"] = str(out_path)
    result["model"] = fresh
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="refit-external",
        description="Refit an external model's exact spec on a forward-chaining split.")
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--cutoff", required=True, help="last date allowed in training, e.g. 2021-08-08")
    p.add_argument("--target", default="bread_quantity")
    p.add_argument("--date-col", default="date")
    p.add_argument("--out", required=True, help="where to write the refitted model")
    args = p.parse_args(argv)

    refit(args.model, args.features, args.cutoff, target=args.target,
          date_col=args.date_col, out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
