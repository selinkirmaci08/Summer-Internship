"""Score a model trained elsewhere on this baseline's protocol and metrics.

Built for the agentic pipeline's saved models, but nothing here is specific to
them: give it a pickled estimator with a `predict` method, the feature table it
was trained on, and a baseline run to compare against, and it scores both on the
identical rows.

    python -m automl_baseline.score_external \
        --model  runs/<run>/artifacts/training/iteration_014_training_model.pkl \
        --features runs/<run>/data/iteration_013_feature_engineering.csv \
        --baseline-run outputs/bread_cut2021 \
        --exclude-split train --name "agentic it_014"

CONTAMINATION IS CHECKED, NOT ASSUMED. A model may only be scored on rows it was
not trained on. Two separate checks run:

  * rows the external model trained on are excluded via `--exclude-split`;
  * rows dated at or before the baseline's cutoff are excluded, because the
    baseline trained on those.

A third condition cannot be enforced by excluding rows: if the external model's
training set contains dates *after* the rows being scored, it had access to the
future when it was fitted, which no forecasting protocol allows. That is reported
as `future_training_rows` and, when non-zero, every number in the output is an
optimistic bound on the external model rather than a like-for-like score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines, metrics


def load_estimator(path):
    import joblib
    model = joblib.load(path)
    if not hasattr(model, "predict"):
        raise TypeError("{} has no .predict(); got {}".format(path, type(model)))
    return model


def feature_names(model, frame: pd.DataFrame, target: str, date_col: str) -> list:
    for attr in ("feature_name_", "feature_names_in_"):
        names = getattr(model, attr, None)
        if names is not None:
            return list(names)
    raise ValueError(
        "the model does not expose feature names; pass --features-list to say which "
        "columns of {} it expects, in order".format(list(frame.columns)))


def score(model_path, features_path, baseline_run, target=None, date_col="date",
          exclude_split=None, split_col="_split", features_list=None,
          name="external", verbose=True):
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    baseline_run = Path(baseline_run)
    with open(baseline_run / "metrics.json", encoding="utf-8") as fh:
        base_result = json.load(fh)
    base_pred = pd.read_csv(baseline_run / "predictions.csv",
                            parse_dates=["origin_time", "target_time"])
    cutoff = pd.Timestamp(base_result["protocol"]["cutoff"])
    scale = float(base_result["protocol"]["mase_scale"])
    period = int(base_result["task"]["seasonal_period"])
    target = target or base_result["task"]["target"]

    model = load_estimator(model_path)
    frame = pd.read_csv(features_path, parse_dates=[date_col])
    cols = features_list or feature_names(model, frame, target, date_col)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError("feature table is missing {} columns the model needs: {}".format(
            len(missing), missing))

    total = len(frame)
    eligible = frame
    dropped = {}

    if exclude_split and split_col in eligible.columns:
        before = len(eligible)
        eligible = eligible[eligible[split_col].astype(str) != str(exclude_split)]
        dropped["trained_on_by_external_model"] = before - len(eligible)

    before = len(eligible)
    eligible = eligible[eligible[date_col] > cutoff]
    dropped["trained_on_by_baseline"] = before - len(eligible)

    before = len(eligible)
    eligible = eligible.dropna(subset=[target] + cols)
    dropped["incomplete_features"] = before - len(eligible)

    # The rows the baseline actually forecast, at the shortest horizon it ran.
    horizon = int(base_pred.horizon.min())
    base_h = base_pred[base_pred.horizon == horizon]
    shared = eligible.merge(
        base_h[["target_time", "y_true", "pred_automl", "pred_seasonal_naive"]],
        left_on=date_col, right_on="target_time", how="inner")
    dropped["not_forecast_by_baseline"] = len(eligible) - len(shared)

    log("[rows] {} in the feature table -> {} scored".format(total, len(shared)))
    for reason, n in dropped.items():
        if n:
            log("       -{:<5} {}".format(n, reason.replace("_", " ")))
    if shared.empty:
        raise ValueError(
            "no rows survive both exclusions: the external model trained on everything the "
            "baseline held out. Move the baseline cutoff earlier (--test-days) so some of the "
            "external model's non-training rows fall after it.")

    # Targets must agree or the two systems are not predicting the same thing.
    disagree = (shared[target] - shared["y_true"]).abs()
    mismatched = int((disagree > 1e-6 * shared["y_true"].abs().clip(lower=1)).sum())
    if mismatched:
        log("[warn] {} of {} rows have a different target value in the two tables; "
            "dropping them".format(mismatched, len(shared)))
        shared = shared[disagree <= 1e-6 * shared["y_true"].abs().clip(lower=1)]
    if shared.empty:
        raise ValueError("no rows left after target reconciliation")

    preds = np.asarray(model.predict(shared[cols]), dtype=float)
    y = shared["y_true"].to_numpy()

    # Look-ahead check: did the external model train on dates after the scored rows?
    future_training = 0
    if exclude_split and split_col in frame.columns:
        trained = frame[frame[split_col].astype(str) == str(exclude_split)]
        if not trained.empty:
            future_training = int((trained[date_col] > shared[date_col].min()).sum())

    result = {
        "external_model": str(model_path),
        "external_name": name,
        "baseline_run": str(baseline_run),
        "target": target,
        "cutoff": str(cutoff.date()),
        "horizon": horizon,
        "n_scored": int(len(shared)),
        "scored_window": [str(shared[date_col].min().date()), str(shared[date_col].max().date())],
        "rows_dropped": dropped,
        "target_mismatches_dropped": mismatched,
        "mase_scale": scale,
        name: metrics.all_metrics(y, preds, scale),
        "baseline": metrics.all_metrics(y, shared["pred_automl"].to_numpy(), scale),
        "seasonal_naive": metrics.all_metrics(y, shared["pred_seasonal_naive"].to_numpy(), scale),
        "future_training_rows": future_training,
    }
    ext_mae, base_mae = result[name]["mae"], result["baseline"]["mae"]
    result["winner"] = name if ext_mae < base_mae else "baseline"
    result["mae_improvement_pct"] = float((1 - ext_mae / base_mae) * 100) if base_mae > 0 else float("nan")
    if future_training:
        result["validity"] = (
            "OPTIMISTIC BOUND, NOT A LIKE-FOR-LIKE SCORE: {} rows in the external model's "
            "training set are dated after the start of the scored window, so it was fitted with "
            "access to the future of what it is being scored on. The baseline had no such "
            "access.".format(future_training))
    else:
        result["validity"] = "clean: neither model trained on the scored rows or on their future"
    return result


def to_text(result: dict) -> str:
    name = result["external_name"]
    lines = []
    lines.append("{} vs AutoML baseline on {} rows ({} .. {}), horizon t+{}".format(
        name, result["n_scored"], result["scored_window"][0], result["scored_window"][1],
        result["horizon"]))
    lines.append("")
    lines.append("  {:<18} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "", "MAE", "RMSE", "MASE", "SMAPE%", "WAPE%"))
    for label in (name, "baseline", "seasonal_naive"):
        m = result[label]
        lines.append("  {:<18} {:9.2f} {:9.2f} {:9.3f} {:9.2f} {:9.2f}".format(
            label, m["mae"], m["rmse"], m["mase"], m["smape_pct"], m["wape_pct"]))
    lines.append("")
    lines.append("  winner: {} ({:+.1f}% MAE vs the baseline)".format(
        result["winner"], result["mae_improvement_pct"]))
    lines.append("")
    lines.append("  validity: {}".format(result["validity"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="score-external",
        description="Score an externally trained model on this baseline's protocol.")
    p.add_argument("--model", required=True, help="pickled estimator with .predict()")
    p.add_argument("--features", required=True, help="CSV of the features it was trained on")
    p.add_argument("--baseline-run", required=True, help="a run directory from automl_baseline")
    p.add_argument("--target", default=None, help="target column (default: the baseline's)")
    p.add_argument("--date-col", default="date")
    p.add_argument("--exclude-split", default=None,
                   help="value of the split column marking the external model's TRAINING rows, "
                        "e.g. 'train'; those rows are excluded from scoring")
    p.add_argument("--split-col", default="_split")
    p.add_argument("--features-list", default=None,
                   help="comma-separated feature columns, if the model does not expose names")
    p.add_argument("--name", default="external", help="label for the external model")
    p.add_argument("--out", default=None, help="write the result JSON here")
    args = p.parse_args(argv)

    result = score(
        args.model, args.features, args.baseline_run, target=args.target,
        date_col=args.date_col, exclude_split=args.exclude_split, split_col=args.split_col,
        features_list=[c.strip() for c in args.features_list.split(",")] if args.features_list else None,
        name=args.name,
    )
    print("\n" + to_text(result))
    out = Path(args.out) if args.out else Path(args.baseline_run) / "external_comparison.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print("\nwritten: {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
