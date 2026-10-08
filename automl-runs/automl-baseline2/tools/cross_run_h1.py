"""Score every run's chosen model on ONE frozen set of rows.

`automl_baseline2` answers "how did *this* run do against AutoML, on its own
protocol". That is the right question per run, and the wrong one across runs: each
run picked its own split, its own trading-day filter, and -- as it turns out -- its
own ground truth on a few dozen days. Two MAEs measured that way are not on the
same scale.

This script builds the one comparison the runs can all legitimately appear in:

  * horizon 1 only, because four of the six model-bearing runs only ever modelled
    a one-step-ahead target and have no horizon column at all;
  * a date window that lies inside the *test* partition of every run scored, so no
    model has seen any scored row in training or in tuning;
  * only dates where every run agrees on the true value, so a single y is being
    predicted rather than six slightly different ones.

Every exclusion is asserted rather than assumed, and the reasons are written out.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
V1 = Path(os.environ.get("AUTOML_BASELINE_V1", ROOT / "automl-baseline"))
for _p in (str(PKG), str(V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from automl_baseline import metrics as v1metrics                      # noqa: E402
from automl_baseline.features import HorizonDataset, prepare_series   # noqa: E402
from automl_baseline.search import fit_regressor, predict as v1predict  # noqa: E402

TIME = "date"
TARGET = "bread_quantity"
TOL = 1e-6


# ------------------------------------------------------------------ inventory

def inventory(runs_dir):
    """What every run directory actually contains, and why it can or cannot be scored."""
    rows = []
    for d in sorted(runs_dir.glob("run_*")):
        fe = sorted(d.glob("data/*feature_engineering.csv"))
        models = sorted(d.glob("artifacts/training/*_training_model.pkl"))
        pp = sorted(d.glob("data/*preprocess.csv"))
        rec = {
            "run": d.name,
            "has_state": (d / "state" / "current.json").exists(),
            "feature_table": fe[-1].name if fe else None,
            "preprocess_table": pp[-1].name if pp else None,
            "n_models": len(models),
            "final_model": models[-1].name if models else None,
            "scorable": bool(fe and models),
        }
        if not rec["scorable"]:
            rec["reason"] = ("no trained model" if not models else "no feature table") + \
                            " -- the run stopped before producing a scoreable model"
        else:
            rec["reason"] = ""
        rows.append(rec)
    return rows


# ------------------------------------------------------------------ loading

def load_run(run_dir):
    """A run reduced to what a cross-run comparison needs: h=1 rows, a model, a split map."""
    import joblib

    fe_path = sorted(run_dir.glob("data/*feature_engineering.csv"))[-1]
    model_path = sorted(run_dir.glob("artifacts/training/*_training_model.pkl"))[-1]
    end = sorted(run_dir.glob("artifacts/end_model/*end_model_summary.json"))
    if end:
        es = json.loads(end[-1].read_text(encoding="utf-8"))
        rel = es.get("final_model_path")
        if rel and (run_dir / rel).exists():
            model_path = run_dir / rel

    fe = pd.read_csv(fe_path, parse_dates=[TIME], low_memory=False)
    horizoned = "h" in fe.columns
    if horizoned:
        fe = fe.loc[fe["h"] == 1].copy()
    # One run emitted two rows for some dates, with different targets and a lag chain
    # computed over row order rather than over the calendar -- so on those dates
    # `lag_1` is the same day's other row. The dates are recorded and later dropped
    # from the frozen index for every system, rather than being quietly deduplicated
    # into a number that looks clean.
    dup_dates = pd.DatetimeIndex(sorted(fe.loc[fe[TIME].duplicated(keep=False), TIME].unique()))
    fe = fe.sort_values(TIME).drop_duplicates(TIME, keep="last").reset_index(drop=True)

    # LightGBM rewrites characters outside [A-Za-z0-9_] in feature names and the
    # fitted booster remembers the rewritten name, so the CSV headers need the same
    # rewrite before a saved model can be pointed at them.
    fe.columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(c)) for c in fe.columns]

    model = joblib.load(model_path)
    feats = list(getattr(model, "feature_name_", []) or getattr(model, "feature_names_in_", []) or [])
    if not feats:
        raise ValueError("{} records no feature names".format(model_path))
    missing = [f for f in feats if f not in fe.columns]
    if missing:
        raise KeyError("{}: model features absent from feature table: {}".format(run_dir.name, missing))

    splits = {}
    for name in ("train", "validation", "test"):
        sub = fe.loc[fe["_split"] == name, TIME]
        if len(sub):
            splits[name] = (sub.min(), sub.max(), int(len(sub)))

    return {
        "run": run_dir.name,
        "dir": run_dir,
        "fe": fe,
        "fe_path": fe_path,
        "model": model,
        "model_path": model_path,
        "feats": feats,
        "horizoned": horizoned,
        "splits": splits,
        "duplicate_dates": dup_dates,
        "y": fe.set_index(TIME)[TARGET],
    }


def chronological(splits):
    """True when train precedes validation precedes test, which is what a forecast split means."""
    if not all(k in splits for k in ("train", "validation", "test")):
        return False
    return splits["train"][1] < splits["validation"][0] and splits["validation"][1] < splits["test"][0]


# ------------------------------------------------------------------ frozen index

def build_frozen_index(loaded):
    """The dates every scored run can be measured on without any of them cheating.

    Three filters, each recorded: inside every run's test window, present as an h=1
    row in every run, and carrying a target value all runs agree on.
    """
    notes = {}
    start = max(l["splits"]["test"][0] for l in loaded)
    end = min(l["splits"]["test"][1] for l in loaded)
    notes["window"] = [str(start.date()), str(end.date())]

    dates = None
    for l in loaded:
        te = l["fe"].loc[(l["fe"]["_split"] == "test") &
                         (l["fe"][TIME] >= start) & (l["fe"][TIME] <= end), TIME]
        s = set(pd.DatetimeIndex(te))
        dates = s if dates is None else (dates & s)
    dates = pd.DatetimeIndex(sorted(dates))
    notes["n_dates_in_all_test_partitions"] = int(len(dates))

    # A date any run recorded twice has an unreliable lag chain in that run.
    dup = set()
    for l in loaded:
        hit = [d for d in l["duplicate_dates"] if d in set(dates)]
        if hit:
            notes.setdefault("dropped_for_duplicate_rows", {})[l["run"]] = [str(d.date()) for d in hit]
            dup |= set(hit)
    dates = pd.DatetimeIndex([d for d in dates if d not in dup])

    # Gradient boosters take NaN as a value and every one of these models saw NaNs in
    # its own training rows, so a missing lag is in-distribution rather than
    # unscoreable. Requiring a fully populated row instead would have cut the window
    # at 2021-12-27 and thrown away the whole second year, because the shop's closed
    # days leave gaps in the lag chain. Rows with *no* readable feature at all are
    # dropped; partial NaN is recorded as a coverage note and scored.
    for l in loaded:
        sub = l["fe"].set_index(TIME).loc[dates, l["feats"]]
        usable = sub.notna().any(axis=1)
        n_partial = int((~sub.notna().all(axis=1) & usable).sum())
        if n_partial:
            notes.setdefault("dates_with_some_missing_features", {})[l["run"]] = n_partial
        if int((~usable).sum()):
            notes.setdefault("dropped_all_features_missing", {})[l["run"]] = int((~usable).sum())
        dates = dates[usable.to_numpy()]

    # Where two runs disagree about what actually happened, neither number means
    # anything next to the other, so those days leave the index.
    ys = pd.DataFrame({l["run"]: l["y"].reindex(dates) for l in loaded})
    spread = (ys.max(axis=1) - ys.min(axis=1)).abs()
    disagree = spread > TOL
    notes["n_dates_dropped_for_disagreeing_truth"] = int(disagree.sum())
    notes["disagreeing_dates"] = [str(d.date()) for d in dates[disagree.to_numpy()]]
    dates = dates[(~disagree).to_numpy()]
    notes["n_scored_dates"] = int(len(dates))

    return dates, ys.loc[dates].iloc[:, 0], notes


def assert_no_leakage(loaded, dates):
    """Every scored date must postdate everything each model was fitted or tuned on."""
    report = {}
    first = dates.min()
    for l in loaded:
        tr_max = l["splits"]["train"][1]
        va_max = l["splits"].get("validation", (None, None, 0))[1]
        fitted_through = max([d for d in (tr_max, va_max) if d is not None])
        if fitted_through >= first:
            raise AssertionError(
                "{}: fitted/tuned on data through {} but the frozen index starts {} -- "
                "scoring it here would be leakage".format(l["run"], fitted_through.date(), first.date()))
        te = set(pd.DatetimeIndex(l["fe"].loc[l["fe"]["_split"] == "test", TIME]))
        if not set(dates) <= te:
            raise AssertionError("{}: frozen index is not a subset of its test partition".format(l["run"]))
        report[l["run"]] = {
            "train_through": str(tr_max.date()),
            "validation_through": str(va_max.date()) if va_max is not None else None,
            "gap_days_to_first_scored_date": int((first - fitted_through).days),
        }
    return report


# ------------------------------------------------------------------ references

def consensus_series(loaded):
    """One daily target series the runs agree on, for the naive and AutoML references.

    Taken as the median across runs per date: where they agree it is that value, and
    on the minority of days that differ it does not follow any single run's cleaning.
    """
    frame = pd.DataFrame({l["run"]: l["y"] for l in loaded})
    return frame.median(axis=1).sort_index()


def seasonal_naive(series, dates):
    """Last observed same-weekday value at or before the origin t-1, on a real calendar."""
    grid = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="D"))
    out = []
    for d in dates:
        prior = grid.loc[:d - pd.Timedelta(days=1)]
        same_dow = prior[prior.index.dayofweek == d.dayofweek].dropna()
        out.append(float(same_dow.iloc[-1]) if len(same_dow) else np.nan)
    return np.asarray(out, dtype=float)


def automl_reference(series, dates, budget, seed, log_file=None):
    """An off-the-shelf h=1 AutoML that has never seen a scored day.

    Target lags, rolling windows and calendar only -- no exogenous columns, so it
    cannot accidentally inherit a same-day measurement the way one of the runs did.
    Trained on every day strictly before the frozen window; FLAML tunes on a
    forward-chaining split inside that history.
    """
    cutoff = dates.min()
    frame = series.rename(TARGET).to_frame()
    frame.index.name = TIME
    prepared, freq = prepare_series(frame.reset_index(), TIME, TARGET)
    ds = HorizonDataset(prepared, TARGET, freq)
    X, y, target_time = ds.build(1)

    tt = pd.Series(pd.DatetimeIndex(target_time.to_numpy()), index=X.index)
    train_mask = (tt < cutoff).to_numpy()
    if tt.loc[train_mask].max() >= cutoff:
        raise AssertionError("AutoML training rows reach into the scored window")

    fitted = fit_regressor(X.loc[train_mask], y.loc[train_mask], time_budget=budget,
                           metric="mae", seed=seed, log_file=log_file)

    wanted = tt.isin(dates).to_numpy()
    X_test = X.loc[wanted]
    preds = pd.Series(v1predict(fitted, X_test), index=pd.DatetimeIndex(tt.loc[wanted].to_numpy()))
    missing = [str(d.date()) for d in dates if d not in preds.index]
    if missing:
        raise AssertionError("AutoML reference cannot predict {} scored dates: {}".format(
            len(missing), missing[:5]))

    return {
        "predictions": preds.reindex(dates).to_numpy(),
        "search": {k: v for k, v in fitted.items() if k != "model"},
        "trained_through": str(tt.loc[train_mask].max().date()),
        "n_train_rows": int(train_mask.sum()),
    }


# ------------------------------------------------------------------ scoring

def score(y_true, y_pred, naive_scale):
    m = v1metrics.all_metrics(y_true, y_pred, naive_scale)
    out = {}
    for k, v in m.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            out[k] = v
            continue
        out[k] = round(fv, 4) if np.isfinite(fv) else None
    return out


def main():
    ap = argparse.ArgumentParser(description="cross-run h=1 comparison on one frozen index")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--budget", type=int, default=600, help="AutoML reference search seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=str(PKG / "outputs" / "cross_run_h1"))
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out = Path(args.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)

    inv = inventory(runs_dir)
    scorable = [r for r in inv if r["scorable"]]
    print("runs found: {}   with a trained model: {}".format(len(inv), len(scorable)))

    loaded, excluded = [], {}
    for rec in scorable:
        l = load_run(runs_dir / rec["run"])
        if not chronological(l["splits"]):
            s = l["splits"]
            excluded[l["run"]] = (
                "split is not chronological (train {}..{}, validation {}..{}, test {}..{}) -- "
                "training rows sit inside and after the test window, so any score it "
                "produces is contaminated".format(
                    s["train"][0].date(), s["train"][1].date(),
                    s["validation"][0].date(), s["validation"][1].date(),
                    s["test"][0].date(), s["test"][1].date()))
            continue
        loaded.append(l)

    for r, why in excluded.items():
        print("EXCLUDED {}: {}".format(r, why))
    if len(loaded) < 2:
        print("fewer than two comparable runs -- nothing to compare")
        return 1

    dates, y_true_s, notes = build_frozen_index(loaded)
    print("frozen index: {} dates, {} .. {}".format(len(dates), dates.min().date(), dates.max().date()))
    leak = assert_no_leakage(loaded, dates)
    print("leakage check passed for all {} runs".format(len(loaded)))

    y_true = y_true_s.to_numpy(dtype=float)
    canonical = consensus_series(loaded)
    naive = seasonal_naive(canonical, dates)
    naive_scale = float(np.nanmean(np.abs(y_true - naive)))

    preds = {"seasonal_naive_h1": naive}
    per_run = {}
    for l in loaded:
        sub = l["fe"].set_index(TIME).loc[dates]
        preds[l["run"]] = np.asarray(l["model"].predict(sub[l["feats"]]), dtype=float)
        per_run[l["run"]] = {
            "model": l["model_path"].name,
            "estimator": type(l["model"]).__name__,
            "n_features": len(l["feats"]),
            "horizon_conditioned": l["horizoned"],
            "feature_table": l["fe_path"].name,
            "features": l["feats"],
        }

    print("fitting the common AutoML reference ({}s)...".format(args.budget))
    ref = automl_reference(canonical, dates, args.budget, args.seed,
                           log_file=str(out / "models" / "flaml_common_h1.log"))
    preds["automl_common_h1"] = ref["predictions"]

    scores = {name: score(y_true, p, naive_scale) for name, p in preds.items()}
    ranking = sorted(scores, key=lambda k: scores[k]["mae"])
    base = scores["automl_common_h1"]["mae"]
    naive_mae = scores["seasonal_naive_h1"]["mae"]
    versus = {
        name: {
            "mae": scores[name]["mae"],
            "gap_vs_automl": round(scores[name]["mae"] - base, 4),
            "skill_vs_automl_pct": round(100 * (1 - scores[name]["mae"] / base), 2),
            "skill_vs_naive_pct": round(100 * (1 - scores[name]["mae"] / naive_mae), 2),
            "beats_automl": bool(scores[name]["mae"] < base),
        }
        for name in scores
    }

    pred_df = pd.DataFrame(preds, index=dates)
    pred_df.insert(0, "y_true", y_true)
    pred_df.index.name = TIME
    pred_df.to_csv(out / "predictions.csv")

    payload = {
        "what_this_is": "every comparable run's chosen model scored on one identical set of h=1 rows",
        "inventory": inv,
        "excluded_runs": excluded,
        "frozen_index": notes,
        "leakage_check": leak,
        "runs_scored": per_run,
        "automl_reference": {
            "features": "target lags/rolling windows + calendar, no exogenous columns",
            "trained_through": ref["trained_through"],
            "n_train_rows": ref["n_train_rows"],
            "budget_seconds": args.budget,
            "search": ref["search"],
        },
        "naive_reference": {
            "definition": "last observed same-weekday value at or before t-1",
            "mase_denominator": round(naive_scale, 4),
        },
        "scores": scores,
        "ranking_by_mae": ranking,
        "versus_automl": versus,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")

    print("\n{:<28} {:>8} {:>8} {:>8} {:>11} {:>10}".format(
        "system", "MAE", "RMSE", "MASE", "vs AutoML", "vs naive"))
    for name in ranking:
        s, v = scores[name], versus[name]
        print("{:<28} {:>8.3f} {:>8.3f} {:>8.3f} {:>10.2f}% {:>9.2f}%".format(
            name, s["mae"], s["rmse"], s["mase"], v["skill_vs_automl_pct"], v["skill_vs_naive_pct"]))
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
