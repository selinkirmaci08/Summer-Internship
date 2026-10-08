"""The check that decides whether the contract comparison means anything.

`predict_contract` is handed the training partition *and* the holdout, because a
forecast made at an origin inside the holdout legitimately knows the holdout's own
past. That is correct, and it is also exactly the shape of mistake that produces a
flattering number nobody catches: if any feature at origin t reads a value from
after t, the model is being scored on data it was shown.

So this test is constructive rather than name-based. It does not scan for
suspicious column names -- it blanks the series after a cutoff and asserts that
every prediction whose origin sits at or before that cutoff is bit-for-bit
unchanged. A feature that peeks forward cannot survive it.

    python tests/test_contract_leakage.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from automl_baseline.predict_contract import build_predictions  # noqa: E402
from automl_baseline.train_contract import blank_non_trading  # noqa: E402

TARGET = "y"
HORIZONS = [1, 7, 14, 30, 60, 90]


def _synthetic_daily(n_days: int = 1500, seed: int = 0) -> pd.DataFrame:
    """A series with weekly shape, trend, closed Mondays and an exogenous column."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", periods=n_days, freq="D")
    dow = np.asarray(idx.dayofweek, dtype=float)
    base = 300 + 40 * np.sin(2 * np.pi * dow / 7) + np.arange(n_days) * 0.02
    y = np.asarray(base, dtype=float) + rng.normal(0, 15, n_days)
    y[dow == 0] = 0.0  # closed Mondays -> blanked as non-trading
    return pd.DataFrame({
        "date": idx,
        TARGET: y.round(1),
        "exog_temp": rng.normal(12, 6, n_days).round(2),
        "exog_revenue": (y * 2.4 + rng.normal(0, 30, n_days)).round(2),
    })


class _EchoModel:
    """Stands in for a fitted estimator: returns a deterministic hash of each row.

    Any change in any feature value changes the output, so equality of predictions
    is equality of the whole feature matrix -- a far stricter assertion than
    comparing a real model's smoothed output.
    """

    def predict(self, X):
        values = np.nan_to_num(
            X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), nan=-999.0)
        weights = np.arange(1, values.shape[1] + 1, dtype=float)
        return values @ weights


def _bundle(feature_order):
    return {
        "model": _EchoModel(),
        "feature_order": feature_order,
        "target": TARGET,
        "time_col": "date",
        "trading_rule": "none",
        "known_future": [],
        "horizons_scored": HORIZONS,
    }


def _feature_order(daily):
    """The column order the 'model' was trained on, taken from a full-history build."""
    from automl_baseline.features import HorizonDataset, prepare_series
    frame, _ = blank_non_trading(daily, TARGET)
    prepared, freq = prepare_series(frame, "date", TARGET, freq="D")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = HorizonDataset(prepared, TARGET, freq)
        X = ds.build_features_at(HORIZONS[0])
    return list(X.columns) + ["horizon"]


def test_no_future_leaks_into_an_origin():
    daily = _synthetic_daily()
    bundle = _bundle(_feature_order(daily))

    cutoff = pd.Timestamp("2020-06-30")
    target_dates = pd.date_range("2020-04-01", "2020-09-30", freq="D")

    full = build_predictions(bundle, daily, target_dates, HORIZONS, verbose=False)

    # Everything strictly after the cutoff never happened, as far as this run knows.
    truncated = daily.copy()
    truncated.loc[truncated["date"] > cutoff, [TARGET, "exog_temp", "exog_revenue"]] = np.nan
    blinded = build_predictions(bundle, truncated, target_dates, HORIZONS, verbose=False)

    key = ["target_date", "h"]
    merged = full.merge(blinded, on=key, suffixes=("_full", "_blind"))
    past = merged[merged["origin_date_full"] <= cutoff]
    assert len(past) > 0, "the test window produced no origins at or before the cutoff"

    delta = (past["y_pred_full"] - past["y_pred_blind"]).abs()
    worst = float(delta.max())
    assert worst < 1e-9, (
        "{} of {} predictions changed when the future was blanked (worst {:.6g}) -- "
        "a feature at the origin is reading data from after it".format(
            int((delta > 1e-9).sum()), len(past), worst))

    print("  no-leak         {} predictions at origins <= {} unchanged when the future "
          "was blanked (worst delta {:.2g})".format(len(past), cutoff.date(), worst))

    # The counterpart: predictions whose origin sits *after* the cutoff must move,
    # otherwise the test passes trivially because nothing was blanked that mattered.
    future = merged[merged["origin_date_full"] > cutoff]
    moved = (future["y_pred_full"] - future["y_pred_blind"]).abs() > 1e-9
    assert moved.any(), "blanking changed nothing at all -- the test is not exercising anything"
    print("  sensitivity     {} of {} predictions at origins after the cutoff did move".format(
        int(moved.sum()), len(future)))


def test_origin_is_always_before_its_target():
    daily = _synthetic_daily()
    bundle = _bundle(_feature_order(daily))
    target_dates = pd.date_range("2020-04-01", "2020-06-30", freq="D")
    preds = build_predictions(bundle, daily, target_dates, HORIZONS, verbose=False)

    gap = (preds["target_date"] - preds["origin_date"]).dt.days
    assert (gap == preds["h"]).all(), "origin is not exactly h days before the target"
    expected = len(target_dates) * len(HORIZONS)
    assert len(preds) == expected, "expected {} rows, got {}".format(expected, len(preds))
    print("  grid            {} rows = {} dates x {} horizons, every origin exactly h days "
          "back".format(len(preds), len(target_dates), len(HORIZONS)))


def test_default_applies_no_trading_rule():
    """The baseline must not quietly withhold rows it has decided are 'closed'.

    Deciding a zero means the shop was shut is a judgement about the data. This
    asserts the default path makes no such judgement: every row the ETL produced
    reaches the model, zeros included.
    """
    daily = _synthetic_daily()
    passed_through, n = blank_non_trading(daily, TARGET)
    assert n == 0, "the default blanked {} rows; it must blank none".format(n)
    assert len(passed_through) == len(daily)
    zeros = int((passed_through[TARGET] == 0).sum())
    assert zeros > 0, "the fixture has no zeros, so this proves nothing"
    assert passed_through[TARGET].isna().sum() == 0
    print("  no judgement    default rule blanked 0 rows; {} zero days reach the model "
          "untouched".format(zeros))


def test_optin_rule_blanks_without_dropping():
    """The opt-in rule stays available so the choice can be measured, not argued."""
    daily = _synthetic_daily()
    blanked, n = blank_non_trading(daily, TARGET, rule="target_gt_zero")
    assert len(blanked) == len(daily), "blanking removed rows; lags would stop meaning calendar days"
    assert n > 0 and blanked[TARGET].isna().sum() == n
    assert (blanked[TARGET].dropna() > 0).all(), "a non-positive target survived blanking"
    print("  opt-in rule     {} days blanked, {} rows preserved".format(n, len(blanked)))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    failures = 0
    for fn in (test_no_future_leaks_into_an_origin,
               test_origin_is_always_before_its_target,
               test_default_applies_no_trading_rule,
               test_optin_rule_blanks_without_dropping):
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print("  FAIL {}: {}".format(fn.__name__, exc))
    print("\n{}".format("all contract checks passed" if not failures
                        else "{} check(s) FAILED".format(failures)))
    sys.exit(1 if failures else 0)
