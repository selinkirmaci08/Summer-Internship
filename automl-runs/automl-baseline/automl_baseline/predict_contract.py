"""Run the frozen contract model over a held-out window and emit the contract CSV.

This is the half that runs on *your* side, with the holdout attached. It rebuilds
features with the same code that built the training features, so the number it
produces measures the model rather than a second feature recipe that happens to
share column names.

Feed it every data root the window needs -- the training partition **and** the
holdout -- because a forecast made at an origin inside the holdout legitimately
knows the holdout's own past:

    python -m automl_baseline.predict_contract \
        --model outputs/contract_train/models/automl_contract.pkl \
        --data ".../aernoudt_split/train" \
        --data ".../holdout_primary" \
        --from 2021-01-01 --to 2021-12-31 \
        --out predictions_holdout_primary.csv

Output is one row per (target_date, h) with a `y_pred` column, which is the shape
`protocol.json` asks for. Non-trading days are predicted anyway and left for the
scorer to exclude: dropping them here would mean this side had applied the trading
rule and the other side had not.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import search
from .features import HorizonDataset, prepare_series
from .train_contract import blank_non_trading

MAX_LAG_NEEDED = 364  # the longest lag in the default recipe


def load_daily(roots, stores, cache_dir: Path | None = None, verbose: bool = True):
    """Run the Aernoudt ETL over each root and stack the results in date order."""
    from .sources import aernoudt

    frames, provenance = [], []
    for root in roots:
        root = Path(root)
        if verbose:
            print("[etl]    {}".format(root), flush=True)
        frame, stats = aernoudt.build(root, out_path=None, scope="chain",
                                      stores=list(stores) or None, verbose=verbose)
        frames.append(frame)
        provenance.append({
            "root": str(root.resolve()),
            "rows": int(len(frame)),
            "first_date": str(pd.to_datetime(frame["date"]).min().date()),
            "last_date": str(pd.to_datetime(frame["date"]).max().date()),
            "files_parsed": stats.get("files_parsed"),
            "superseded_exports_dropped": stats.get("superseded_dropped"),
            "duplicate_policy": stats.get("duplicate_policy"),
        })

    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    before = len(daily)
    daily = daily.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    daily = daily.reset_index(drop=True)
    if verbose and len(daily) != before:
        print("[etl]    {} overlapping dates across roots collapsed (kept the later root)".format(
            before - len(daily)), flush=True)
    return daily, provenance


def _warn_on_history_gaps(y: pd.Series, target_dates, horizons, verbose: bool = True,
                          gap_days: int = 14):
    """Shout about a hole in the history the lags will be reading through.

    The trap this exists for: scoring the 2022 window without also passing the 2021
    partition. Nothing errors -- the series still starts in 2010, so the origin rows
    exist -- but every lag shorter than a year reads into an empty 2021 and comes
    back NaN. The model tolerates NaN, so the run completes and the predictions are
    quietly much worse than the model actually is. A missing root should look like a
    mistake, not like a bad model.
    """
    lo = pd.Timestamp(min(target_dates)) - pd.Timedelta(days=max(horizons) + MAX_LAG_NEEDED)
    hi = pd.Timestamp(max(target_dates))
    window = y.loc[(y.index >= lo) & (y.index <= hi)]
    if window.empty:
        return
    missing = window.isna()
    # Longest run of consecutive absent days inside the span the lags draw on.
    runs = (missing != missing.shift()).cumsum()[missing]
    longest = int(runs.value_counts().max()) if len(runs) else 0
    if longest >= gap_days:
        block = runs.value_counts().idxmax()
        dates = runs.index[runs == block]
        print("[WARN]   {} consecutive days with no observation ({} -> {}) inside the history "
              "the lags read from. If that is a partition you did not pass as --data, the "
              "predictions for this window will be far worse than the model is -- pass every "
              "root between the training data and this window.".format(
                  longest, dates.min().date(), dates.max().date()), flush=True)
    elif verbose:
        print("[history] lag window {} -> {} | {:.0%} of days observed, longest gap {} days".format(
            lo.date(), hi.date(), 1 - missing.mean(), longest), flush=True)


def build_predictions(bundle: dict, daily: pd.DataFrame, target_dates, horizons,
                      verbose: bool = True) -> pd.DataFrame:
    """Predict every (target_date, h) pair from features built at origin = target_date - h."""
    target = bundle["target"]
    time_col = bundle["time_col"]
    feature_order = list(bundle["feature_order"])

    # Whatever the model was trained under, the history it forecasts from must match.
    # Taking the rule from the bundle rather than a default here is what stops the two
    # halves drifting apart if the trainer is ever re-run with a different choice.
    rule = bundle.get("trading_rule", "none")
    daily, n_blanked = blank_non_trading(daily, target, rule)
    if verbose:
        print("[trading] rule from the model bundle: {} ({} days blanked in the history)".format(
            rule, n_blanked), flush=True)

    prepared, freq = prepare_series(daily, time_col, target, freq="D")

    # Every origin we will be asked for must exist on the calendar. The earliest is
    # min(target_date) - max(h); pad back further still so its lags have somewhere to
    # read from rather than silently returning NaN for the first weeks of the window.
    target_dates = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(target_dates)).unique()))
    need_from = target_dates.min() - pd.Timedelta(days=max(horizons) + MAX_LAG_NEEDED)
    if prepared.index.min() > target_dates.min() - pd.Timedelta(days=max(horizons)):
        raise SystemExit(
            "history starts {} but an origin of {} is required for target {} at h={}. "
            "Pass the training partition as an extra --data root.".format(
                prepared.index.min().date(),
                (target_dates.min() - pd.Timedelta(days=max(horizons))).date(),
                target_dates.min().date(), max(horizons)))
    if prepared.index.min() > need_from and verbose:
        print("[warn]   history starts {}; the earliest targets have lags reaching before it "
              "and will carry NaNs (the model tolerates them, but they are weaker rows)".format(
                  prepared.index.min().date()), flush=True)

    full = pd.date_range(min(need_from, prepared.index.min()),
                         max(prepared.index.max(), target_dates.max()), freq=freq)
    prepared = prepared.reindex(full)
    prepared.index.name = time_col

    _warn_on_history_gaps(prepared[target], target_dates, horizons, verbose)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = HorizonDataset(prepared, target, freq, known_future=bundle.get("known_future", []))

    rows = []
    for h in sorted(int(x) for x in horizons):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = ds.build_features_at(h)
        origins = target_dates - pd.Timedelta(days=h)
        keep = origins.isin(X.index)
        if not keep.all() and verbose:
            print("[warn]   h={}: {} target dates have no origin row".format(
                h, int((~keep).sum())), flush=True)
        Xh = X.loc[origins[keep]].copy()
        Xh["horizon"] = h
        missing = [c for c in feature_order if c not in Xh.columns]
        if missing:
            if verbose:
                print("[warn]   h={}: {} training features absent from this data, filled NaN: "
                      "{}{}".format(h, len(missing), missing[:8],
                                    " ..." if len(missing) > 8 else ""), flush=True)
            for c in missing:
                Xh[c] = np.nan
        Xh = Xh.reindex(columns=feature_order)
        preds = search.predict({"model": bundle["model"]}, Xh)
        rows.append(pd.DataFrame({
            "target_date": target_dates[keep],
            "h": h,
            "origin_date": origins[keep],
            "y_pred": preds,
        }))

    out = pd.concat(rows, ignore_index=True).sort_values(["target_date", "h"])
    assert (out["origin_date"] < out["target_date"]).all(), "an origin is not before its target"
    return out.reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="predict-contract",
        description="Emit (target_date, h, y_pred) for a held-out window.")
    p.add_argument("--model", required=True, help="the frozen model bundle (.pkl)")
    p.add_argument("--data", action="append", required=True,
                   help="a raw Aernoudt data root; repeat for train + holdout")
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--horizons", default="", help="default: the grid the model was trained to score")
    p.add_argument("--stores", default="",
                   help="POS codes to include; empty means every code, which is the ETL's "
                        "confirmed 'chain' scope and what the model was trained on. Pass "
                        "'001' only if you have decided to score the retail till alone -- "
                        "and then apply that same choice on the scoring side")
    p.add_argument("--out", default="predictions_contract.csv")
    p.add_argument("--keep-origin", action="store_true",
                   help="keep the origin_date column (not part of the contract, useful for audit)")
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")
    bundle = joblib.load(args.model)
    horizons = ([int(h) for h in args.horizons.split(",") if h.strip()]
                or list(bundle["horizons_scored"]))
    print("[model]  {} | {} features | trained to score h={}".format(
        bundle["best_estimator"], len(bundle["feature_order"]), bundle["horizons_scored"]),
        flush=True)

    stores = [s.strip() for s in args.stores.split(",") if s.strip()]
    daily, provenance = load_daily(args.data, stores)
    print("[history] {} -> {} | {} rows".format(
        daily["date"].min().date(), daily["date"].max().date(), len(daily)), flush=True)

    target_dates = pd.date_range(args.date_from, args.date_to, freq="D")
    preds = build_predictions(bundle, daily, target_dates, horizons)

    out_path = Path(args.out)
    contract = preds if args.keep_origin else preds[["target_date", "h", "y_pred"]]
    contract = contract.copy()
    contract["target_date"] = pd.to_datetime(contract["target_date"]).dt.strftime("%Y-%m-%d")
    if "origin_date" in contract.columns:
        contract["origin_date"] = pd.to_datetime(contract["origin_date"]).dt.strftime("%Y-%m-%d")
    contract.to_csv(out_path, index=False)

    with open(out_path.with_suffix(".provenance.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "model": str(Path(args.model).resolve()),
            "best_estimator": bundle["best_estimator"],
            "best_config": bundle["best_config"],
            "n_features": len(bundle["feature_order"]),
            "data_roots": provenance,
            "till_scope": stores,
            "trading_rule": bundle.get("trading_rule"),
            "window": [args.date_from, args.date_to],
            "horizons": horizons,
            "rows_emitted": int(len(contract)),
            "note": ("Non-trading days are predicted and emitted; the scorer excludes them "
                     "using the rule in protocol.json, so the rule is applied once, to both "
                     "sides, rather than twice with a chance to drift."),
        }, fh, indent=2, default=str)

    print("[out]    {} rows -> {}".format(len(contract), out_path), flush=True)
    print("[out]    {}".format(out_path.with_suffix(".provenance.json")), flush=True)
    print("[range]  origins {} -> {}".format(
        preds["origin_date"].min().date(), preds["origin_date"].max().date()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
