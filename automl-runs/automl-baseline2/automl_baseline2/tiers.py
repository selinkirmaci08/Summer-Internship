"""The four reference points, each scored on the agentic run's exact test rows.

A single "AutoML baseline" number cannot say *where* an agentic pipeline earned its
keep. Three AutoML tiers can, because each one is given a different amount of the
agent's work and everything else is held fixed:

    naive               the seasonal-naive floor the run itself quotes
    B1  agent_features  AutoML on the agent's engineered matrix, unchanged
    B2  preprocessed    AutoML on the agent's cleaned daily frame, generic features
    B3  assembled       AutoML on the raw assembled frame, generic features

B1 minus the agentic model isolates **model selection and tuning**: same features,
same rows, so any gap is the estimator and its hyperparameters. B2 minus B1
isolates **feature engineering**: same cleaned data, but generic lags and calendar
terms instead of the agent's. B3 minus B2 isolates **preprocessing**: the same
generic recipe run against the frame as it came out of assembly.

Every tier is fitted on the run's train partition, tuned on its validation
partition and scored on its test partition. No tier ever sees a test row.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from automl_baseline.features import HorizonDataset, prepare_series

from .protocol import AgenticRun, lgbm_safe_columns
from .search2 import fit_with_holdout, predict

# Columns that are bookkeeping rather than data, and must never reach a model.
_NEVER_FEATURES = ("_split", "origin_date", "date", "h")


# ------------------------------------------------------------------ generic matrix


def _known_future_columns(frame: pd.DataFrame, run: AgenticRun) -> list:
    """Columns whose value at the target date is genuinely knowable at the origin.

    Taken from the run's own `known_at_origin_features` so the two systems are
    given the same information, plus the competitor-closure columns, which the run
    confirmed with the user are published in advance. Everything else is lagged.
    """
    declared = set(run.known_at_origin) | {
        "competitor_n_closed", "competitor_any_closed", "competitor_log_active",
    }
    # Calendar columns the agent's preprocessing already materialised; the generic
    # builder regenerates equivalents at the target date anyway, but leaving these
    # lagged would be actively wrong -- next month's month number is not a mystery.
    declared |= {"year", "month", "day_of_month", "week_of_year", "day_of_year",
                 "is_weekend", "dow", "day_name", "store_open_7_7",
                 "is_historical_closing_day"}
    cols = [c for c in frame.columns if c in declared]
    cols += [c for c in frame.columns if c.startswith("dow_")]
    return sorted(set(cols))


def build_generic_matrix(
    daily: pd.DataFrame,
    run: AgenticRun,
    drop_columns=(),
    lags=None,
    windows=None,
) -> dict:
    """Reframe a daily frame into the run's own (target_date, horizon) grid.

    The feature recipe is v1's `HorizonDataset` -- default lags, rolling statistics,
    calendar and Fourier terms, lagged exogenous columns. Nothing the agent
    discovered is used. What *is* borrowed from the run is only the shape of the
    problem: which dates count as trading days, which horizons are scored, and
    which partition each target date belongs to. Borrowing those is the point;
    they are the protocol, not the modelling.
    """
    time_col = run.time_col
    target = run.target

    frame = daily.copy()
    frame.columns = lgbm_safe_columns(frame.columns)
    drop = {c for c in drop_columns if c in frame.columns}
    drop |= {c for c in _NEVER_FEATURES if c in frame.columns and c != time_col}
    frame = frame.drop(columns=sorted(drop))

    frame[time_col] = pd.to_datetime(frame[time_col])
    modellable = set(run.modellable_dates())
    # Non-trading and unexported days are blanked rather than deleted, so a lag of 7
    # still means seven calendar days back. This is the same treatment the agentic
    # run applied, and it is what makes the two horizon grids line up.
    outside = ~frame[time_col].isin(modellable)
    frame.loc[outside, target] = np.nan

    known_future = _known_future_columns(frame, run)
    prepared, freq = prepare_series(frame, time_col, target, freq="D")

    # The earliest target dates have origins that fall before the data begins, so
    # without padding they would have no row to sit on and the tier would silently
    # be scored on a smaller training set than the agentic model was. The agentic
    # feature table keeps those rows with empty lags; pad the calendar so this one
    # does too. The padded dates carry no target, so nothing is invented.
    max_h = max(int(h) for h in run.horizons)
    pad_start = prepared.index.min() - pd.Timedelta(days=max_h)
    prepared = prepared.reindex(pd.date_range(pad_start, prepared.index.max(), freq=freq))
    prepared.index.name = time_col

    kwargs = {}
    if lags is not None:
        kwargs["lags"] = tuple(lags)
    if windows is not None:
        kwargs["windows"] = tuple(windows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = HorizonDataset(prepared, target, freq, known_future=known_future, **kwargs)
        X, y, origin_time, target_time, horizon = ds.build_pooled(run.horizons, max_rows=None)

    out = pd.DataFrame({
        time_col: pd.to_datetime(target_time),
        "origin_date": pd.to_datetime(origin_time),
        "h": np.asarray(horizon, dtype=int),
        target: np.asarray(y, dtype=float),
    })
    split_map = run.date_to_split()
    out["_split"] = out[time_col].map(split_map).astype("object")
    feature_cols = [c for c in X.columns if c not in out.columns]
    matrix = pd.concat([out.reset_index(drop=True), X[feature_cols].reset_index(drop=True)], axis=1)
    matrix = matrix.loc[matrix["_split"].isin(("train", "validation", "test"))]
    matrix = matrix.sort_values([time_col, "h"]).reset_index(drop=True)

    return {
        "matrix": matrix,
        "feature_cols": feature_cols,
        "known_future": known_future,
        "freq": freq,
        "n_lags": len(ds.past.columns),
    }


def assert_grid_matches(matrix: pd.DataFrame, run: AgenticRun, split: str, label: str) -> dict:
    """Fail loudly if a tier's rows are not the agentic run's rows.

    A comparison whose two sides are scored on slightly different rows is worse
    than no comparison, because it looks like a result. This is checked for every
    tier rather than trusted.
    """
    want = run.index(split)[[run.time_col, "h"]].apply(tuple, axis=1)
    got = matrix.loc[matrix["_split"] == split, [run.time_col, "h"]].apply(tuple, axis=1)
    want_set, got_set = set(want), set(got)
    missing, extra = want_set - got_set, got_set - want_set
    if missing or extra:
        raise ValueError(
            "{}: the {} grid does not match the agentic run -- {} rows missing, {} extra. "
            "Example missing: {}".format(
                label, split, len(missing), len(extra), sorted(missing)[:3]
            )
        )
    return {"n_rows": len(got_set), "matches_agentic_grid": True}


# ------------------------------------------------------------------ tier runners


def _split_frames(matrix: pd.DataFrame, feature_cols, target: str):
    parts = {}
    for name in ("train", "validation", "test"):
        sub = matrix.loc[matrix["_split"] == name]
        parts[name] = (sub[feature_cols], sub[target])
    return parts


def run_automl_tier(
    matrix: pd.DataFrame,
    feature_cols,
    run: AgenticRun,
    label: str,
    time_budget: int,
    metric: str = "mae",
    seed: int = 42,
    estimators=(),
    log_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Fit one AutoML tier and predict the run's test rows."""
    matrix = matrix.sort_values([run.time_col, "h"]).reset_index(drop=True)
    assert_grid_matches(matrix, run, "test", label)
    parts = _split_frames(matrix, feature_cols, run.target)

    log_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / "flaml_{}.log".format(label))

    if verbose:
        print("[{}] searching {}s over {} train rows x {} features".format(
            label, time_budget, len(parts["train"][1]), len(feature_cols)))

    fitted = fit_with_holdout(
        parts["train"][0], parts["train"][1],
        parts["validation"][0], parts["validation"][1],
        time_budget=time_budget, metric=metric, seed=seed,
        estimators=estimators, log_file=log_file,
    )

    preds = {}
    for name in ("validation", "test"):
        preds[name] = predict(fitted, parts[name][0])

    if verbose:
        print("[{}] picked {} | validation {} {:.3f} | {:.0f}s".format(
            label, fitted["estimator"], fitted["metric"], fitted["validation_loss"],
            fitted["search_seconds"]))

    test_rows = matrix.loc[matrix["_split"] == "test"]
    return {
        "label": label,
        "search": {k: v for k, v in fitted.items() if k != "model"},
        "model": fitted["model"],
        "predictions": {
            "test": pd.DataFrame({
                run.time_col: test_rows[run.time_col].to_numpy(),
                "h": test_rows["h"].to_numpy(),
                "pred": preds["test"],
            }),
        },
        "validation_predictions": pd.DataFrame({
            run.time_col: matrix.loc[matrix["_split"] == "validation", run.time_col].to_numpy(),
            "h": matrix.loc[matrix["_split"] == "validation", "h"].to_numpy(),
            "pred": preds["validation"],
        }),
    }


def tier_agent_features(run: AgenticRun, time_budget: int, **kw) -> dict:
    """B1 -- AutoML on the agent's engineered matrix, exactly as the agent left it.

    Same 56 features the chosen model used, same rows. The only difference from the
    agentic model is who chose the estimator and its hyperparameters.
    """
    model = run.load_final_model()
    feats = list(getattr(model, "feature_name_", []) or [])
    matrix = run.fe_frame
    feature_cols = [f for f in feats if f in matrix.columns]
    return run_automl_tier(matrix, feature_cols, run, "B1_agent_features", time_budget, **kw)


def tier_preprocessed(run: AgenticRun, time_budget: int, **kw) -> dict:
    """B2 -- AutoML on the agent's cleaned daily frame, with generic features."""
    built = build_generic_matrix(run.preprocess_frame, run)
    result = run_automl_tier(built["matrix"], built["feature_cols"], run,
                             "B2_preprocessed", time_budget, **kw)
    result["feature_build"] = {k: v for k, v in built.items() if k != "matrix"}
    return result


def tier_assembled(run: AgenticRun, time_budget: int, **kw) -> dict:
    """B3 -- AutoML on the raw assembled frame, with generic features.

    The run's `must_drop_columns` are removed here too. They are same-day
    measurements of the target (total revenue, bread revenue, transaction counts):
    an off-the-shelf tool would lag them and gain a genuine advantage, but then the
    two systems would no longer be forecasting from the same information and the
    gap would stop meaning what it appears to mean.
    """
    if run.assembled_path is None:
        raise FileNotFoundError("this run recorded no assembled frame, so B3 cannot be built")
    daily = pd.read_csv(run.assembled_path, parse_dates=[run.time_col], low_memory=False)
    built = build_generic_matrix(daily, run, drop_columns=run.must_drop)
    result = run_automl_tier(built["matrix"], built["feature_cols"], run,
                             "B3_assembled", time_budget, **kw)
    result["feature_build"] = {k: v for k, v in built.items() if k != "matrix"}
    result["dropped_columns"] = list(run.must_drop)
    return result


TIERS = {
    "B1_agent_features": tier_agent_features,
    "B2_preprocessed": tier_preprocessed,
    "B3_assembled": tier_assembled,
}
