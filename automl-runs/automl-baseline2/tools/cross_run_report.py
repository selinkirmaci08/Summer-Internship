"""Turn the cross-run comparison into a readable record.

Two jobs beyond formatting:

  * it scores the runs `cross_run_h1.py` refused to rank, on the same frozen rows,
    and labels the result for what it is. A number that cannot be trusted is more
    useful shown next to the reason than left out, because "excluded" on its own
    invites someone to go and compute it later without the caveat;
  * it writes the inventory of every run, so the 13 that produced no model are
    visible as a fact about the project rather than a silent absence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent

TIME = "date"
TARGET = "bread_quantity"


def score_excluded(run_dir: Path, dates: pd.DatetimeIndex, y_true: np.ndarray) -> dict:
    """Score a run whose split makes its number meaningless, and say by how much."""
    import joblib

    fe_path = sorted(run_dir.glob("data/*feature_engineering.csv"))[-1]
    model_path = sorted(run_dir.glob("artifacts/training/*_training_model.pkl"))[-1]
    fe = pd.read_csv(fe_path, parse_dates=[TIME], low_memory=False)
    if "h" in fe.columns:
        fe = fe.loc[fe["h"] == 1]
    fe.columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(c)) for c in fe.columns]
    fe = fe.sort_values(TIME).drop_duplicates(TIME, keep="last").set_index(TIME)

    model = joblib.load(model_path)
    feats = list(getattr(model, "feature_name_", []) or getattr(model, "feature_names_in_", []))
    have = dates.intersection(fe.index)
    pred = np.asarray(model.predict(fe.loc[have, feats]), dtype=float)
    truth = pd.Series(y_true, index=dates).loc[have].to_numpy(dtype=float)

    train_dates = pd.DatetimeIndex(fe.index[fe["_split"] == "train"])
    seen = train_dates.intersection(have)
    return {
        "model": model_path.name,
        "n_scored": int(len(have)),
        "n_scored_dates_the_model_trained_on": int(len(seen)),
        "mae": round(float(np.mean(np.abs(truth - pred))), 4),
        "mae_on_dates_it_trained_on": (
            round(float(np.mean(np.abs(
                pd.Series(truth, index=have).loc[seen].to_numpy()
                - pd.Series(pred, index=have).loc[seen].to_numpy()))), 4) if len(seen) else None),
        "mae_on_dates_it_did_not_train_on": (
            round(float(np.mean(np.abs(
                pd.Series(truth, index=have).drop(seen).to_numpy()
                - pd.Series(pred, index=have).drop(seen).to_numpy()))), 4)
            if len(have) > len(seen) else None),
    }


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="report on the cross-run h=1 comparison")
    ap.add_argument("--dir", default=str(PKG / "outputs" / "cross_run_h1"))
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args = ap.parse_args()

    d = Path(args.dir)
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(d / "predictions.csv", parse_dates=[TIME]).set_index(TIME)
    dates = pd.DatetimeIndex(preds.index)
    y_true = preds["y_true"].to_numpy(dtype=float)

    contaminated = {}
    for run, why in m["excluded_runs"].items():
        contaminated[run] = score_excluded(Path(args.runs_dir) / run, dates, y_true)
        contaminated[run]["excluded_because"] = why
    m["excluded_runs_scored_anyway"] = contaminated
    (d / "metrics.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")

    fi = m["frozen_index"]
    scores, versus = m["scores"], m["versus_automl"]
    ranking = m["ranking_by_mae"]

    lines = []
    lines.append("# Every run against one AutoML baseline, on one set of rows\n")
    lines.append("Horizon 1, **{} dates**, {} to {}. Every system below predicts the same "
                 "{} days and is scored against the same truth.\n".format(
                     fi["n_scored_dates"], fi["window"][0], fi["window"][1], fi["n_scored_dates"]))

    lines.append("\n## What could be compared\n")
    inv = m["inventory"]
    lines.append(md_table(
        ["run", "feature table", "models", "in the comparison"],
        [[r["run"], r["feature_table"] or "--", r["n_models"],
          "yes" if r["run"] in m["runs_scored"] else
          ("excluded -- see below" if r["run"] in m["excluded_runs"] else "no model produced")]
         for r in inv]))
    lines.append("\n{} of {} runs reached a trained model. {} of those are scored here.\n".format(
        sum(1 for r in inv if r["scorable"]), len(inv), len(m["runs_scored"])))

    lines.append("\n## Results\n")
    lines.append(md_table(
        ["system", "MAE", "RMSE", "MASE", "WAPE %", "bias", "vs AutoML", "vs naive"],
        [[name, scores[name]["mae"], scores[name]["rmse"], scores[name]["mase"],
          scores[name]["wape_pct"], scores[name]["bias"],
          "{:+.2f}%".format(versus[name]["skill_vs_automl_pct"]),
          "{:+.2f}%".format(versus[name]["skill_vs_naive_pct"])] for name in ranking]))
    lines.append("\nPositive skill means better than that reference. The AutoML row is the "
                 "baseline: FLAML on target lags, rolling windows and calendar only, trained "
                 "on every day through {} and never on a scored day.\n".format(
                     m["automl_reference"]["trained_through"]))

    lines.append("\n## Why no model was scored on a row it had seen\n")
    lines.append(md_table(
        ["run", "trained through", "tuned through", "days before the first scored date"],
        [[k, v["train_through"], v["validation_through"], v["gap_days_to_first_scored_date"]]
         for k, v in m["leakage_check"].items()]))
    lines.append("\nThe frozen index starts the day after the latest validation window of any "
                 "run in it, and every scored date was asserted to be inside every run's own "
                 "test partition before anything was predicted.\n")
    drops = []
    if fi.get("dropped_for_duplicate_rows"):
        for k, v in fi["dropped_for_duplicate_rows"].items():
            drops.append([k, "duplicate rows for the same date", len(v)])
    if fi.get("n_dates_dropped_for_disagreeing_truth"):
        drops.append(["all", "runs disagree on the true value", fi["n_dates_dropped_for_disagreeing_truth"]])
    if drops:
        lines.append("\nDates removed from the window:\n")
        lines.append(md_table(["run", "reason", "dates"], drops))

    if contaminated:
        lines.append("\n## Runs that could not be ranked\n")
        for run, c in contaminated.items():
            lines.append("\n**{}** -- {}\n".format(run, c["excluded_because"]))
            lines.append(md_table(
                ["scored dates", "of which it trained on", "MAE (all)",
                 "MAE where it trained", "MAE where it did not"],
                [[c["n_scored"], c["n_scored_dates_the_model_trained_on"], c["mae"],
                  c["mae_on_dates_it_trained_on"], c["mae_on_dates_it_did_not_train_on"]]]))
            lines.append("\nThe gap between the last two columns is the size of the "
                         "contamination, not a modelling result.\n")

    (d / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote {}".format(d / "report.md"))
    for run, c in contaminated.items():
        print("{}: MAE {} overall / {} on trained dates / {} on unseen dates".format(
            run, c["mae"], c["mae_on_dates_it_trained_on"], c["mae_on_dates_it_did_not_train_on"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
