"""Protocol guards: run with `python tests/test_protocol.py` (pytest optional).

These check the things that would silently invalidate a baseline comparison --
leakage across the horizon boundary, a mis-aligned target, a wrong naive
reference -- rather than the modelling itself.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from automl_baseline import baselines, metrics  # noqa: E402
from automl_baseline.features import HorizonDataset, prepare_series  # noqa: E402
from automl_baseline.ingest import load_dataset  # noqa: E402


def _series(n=400, start="2020-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    rng = np.random.default_rng(0)
    y = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 7) + np.arange(n) * 0.05 + rng.normal(0, 2, n)
    return pd.DataFrame({"date": idx, "sales": y, "temp": rng.normal(15, 5, n)})


def test_target_alignment():
    """y at origin t must be the true value at t+h, and target_time must say so."""
    df = _series()
    frame, freq = prepare_series(df, "date", "sales")
    ds = HorizonDataset(frame, "sales", freq)
    for h in (1, 7, 30):
        X, y, target_time = ds.build(h)
        origin = X.index[50]
        expected_time = origin + pd.Timedelta(days=h)
        assert target_time.loc[origin] == expected_time, (h, target_time.loc[origin], expected_time)
        assert math.isclose(y.loc[origin], frame["sales"].loc[expected_time], rel_tol=1e-9)
    print("ok  target alignment")


def test_no_future_leakage_in_past_features():
    """Every lag/rolling feature at origin t must come from times <= t.

    Proof by construction: blank out the series after an origin and check that the
    origin's past features are unchanged.
    """
    df = _series()
    frame, freq = prepare_series(df, "date", "sales")
    ds_full = HorizonDataset(frame, "sales", freq)
    origin = frame.index[300]

    truncated = frame.copy()
    truncated.loc[truncated.index > origin, "sales"] = np.nan
    truncated.loc[truncated.index > origin, "temp"] = np.nan
    ds_trunc = HorizonDataset(truncated, "sales", freq)

    past_cols = [c for c in ds_full.past.columns]
    a = ds_full.past.loc[origin, past_cols]
    b = ds_trunc.past.loc[origin, past_cols]
    for col in past_cols:
        va, vb = a[col], b[col]
        if pd.isna(va) and pd.isna(vb):
            continue
        assert math.isclose(float(va), float(vb), rel_tol=1e-9), (
            "feature '{}' at origin {} changed when the future was blanked -> it reads the "
            "future".format(col, origin.date()))
    print("ok  no future leakage in {} past features".format(len(past_cols)))


def test_calendar_features_describe_the_target_date():
    """Calendar features are the one legitimate look-ahead: they describe t+h."""
    df = _series()
    frame, freq = prepare_series(df, "date", "sales")
    ds = HorizonDataset(frame, "sales", freq)
    for h in (1, 14, 90):
        X, _, _ = ds.build(h)
        origin = X.index[100]
        assert X.loc[origin, "cal_dayofweek"] == (origin + pd.Timedelta(days=h)).dayofweek
        assert X.loc[origin, "cal_month"] == (origin + pd.Timedelta(days=h)).month
    print("ok  calendar features describe t+h")


def test_train_rows_never_cross_the_cutoff():
    df = _series()
    frame, freq = prepare_series(df, "date", "sales")
    ds = HorizonDataset(frame, "sales", freq)
    cutoff = frame.index[-181]
    for h in (1, 7, 14, 30, 90):
        _, _, target_time = ds.build(h)
        train_targets = target_time[target_time <= cutoff]
        test_targets = target_time[target_time > cutoff]
        assert train_targets.max() <= cutoff
        assert test_targets.min() > cutoff
        assert len(test_targets) > 0
    print("ok  train targets never cross the cutoff")


def test_pooled_matrix_targets_and_ordering():
    """One stacked matrix: y at (origin, h) is the real value at origin+h, rows time-ordered."""
    df = _series()
    frame, freq = prepare_series(df, "date", "sales")
    ds = HorizonDataset(frame, "sales", freq)
    X, y, origin_time, target_time, horizon = ds.build_pooled([1, 7, 30])

    assert len(X) == len(y) == len(origin_time) == len(target_time)
    assert "horizon" in X.columns
    assert pd.Series(target_time).is_monotonic_increasing, "pooled rows must be in target-time order"
    for i in (0, len(X) // 2, len(X) - 1):
        h = int(horizon.iloc[i])
        o = pd.Timestamp(origin_time.iloc[i])
        assert pd.Timestamp(target_time.iloc[i]) == o + pd.Timedelta(days=h)
        assert math.isclose(y.iloc[i], frame["sales"].loc[o + pd.Timedelta(days=h)], rel_tol=1e-9)
        assert int(X["horizon"].iloc[i]) == h
    print("ok  pooled matrix: targets aligned, rows in target-time order")


def test_pooled_grid_never_drops_a_scored_horizon():
    """Thinning the horizon grid must not leave a scored horizon to be extrapolated."""
    df = _series(n=400)
    frame, freq = prepare_series(df, "date", "sales")
    ds = HorizonDataset(frame, "sales", freq)
    scored = (1, 7, 14, 30, 90)
    grid = ds.horizon_grid(scored)
    assert grid == list(range(1, 91))
    # A cap tight enough to force heavy thinning.
    _, _, _, _, horizon = ds.build_pooled(grid, max_rows=5000, must_keep=scored)
    trained = set(int(v) for v in horizon.unique())
    for h in scored:
        assert h in trained, "scored horizon t+{} was thinned out of the training grid".format(h)
    print("ok  grid thinning keeps every scored horizon ({} trained)".format(len(trained)))


def test_inference_view_keeps_the_newest_origin():
    """`build` drops the last origins (no known y); inference must still reach them."""
    df = _series()
    frame, freq = prepare_series(df, "date", "sales")
    ds = HorizonDataset(frame, "sales", freq)
    last = frame.index[-1]
    for h in (1, 30, 90):
        X_train, _, _ = ds.build(h)
        X_infer = ds.build_features_at(h)
        assert last not in X_train.index, "training rows should not include an unknown target"
        assert last in X_infer.index, "inference must be able to forecast from the newest origin"
        # and its calendar features must still describe t+h, not be blank
        assert X_infer.loc[last, "cal_dayofweek"] == (last + pd.Timedelta(days=h)).dayofweek
        assert not pd.isna(X_infer.loc[last, "cal_month"])
    print("ok  inference view reaches the newest origin with correct t+h calendar")


def test_pooled_features_do_not_read_the_future():
    df = _series()
    frame, freq = prepare_series(df, "date", "sales")
    origin = frame.index[300]
    truncated = frame.copy()
    truncated.loc[truncated.index > origin, ["sales", "temp"]] = np.nan

    a = HorizonDataset(frame, "sales", freq).build_features_at(30).loc[origin]
    b = HorizonDataset(truncated, "sales", freq).build_features_at(30).loc[origin]
    past_cols = [c for c in a.index if not c.startswith("cal_")]
    for col in past_cols:
        va, vb = a[col], b[col]
        if pd.isna(va) and pd.isna(vb):
            continue
        assert math.isclose(float(va), float(vb), rel_tol=1e-9), (
            "pooled feature '{}' reads the future".format(col))
    print("ok  pooled features at t+30 read nothing after the origin")


def test_seasonal_naive_picks_the_right_source_point():
    idx = pd.date_range("2021-01-01", periods=200, freq="D")
    y = pd.Series(np.arange(200, dtype=float), index=idx)
    origins = idx[150:160]
    for h in (1, 7, 14, 30, 90):
        pred = baselines.seasonal_naive(y, origins, h, period=7)
        back = 7 * math.ceil(h / 7) - h
        for origin in origins:
            expected = y.loc[origin - pd.Timedelta(days=back)]
            assert math.isclose(pred.loc[origin], expected), (h, origin, pred.loc[origin], expected)
        # the source point is always at or before the origin, never after
        assert back >= 0
    print("ok  seasonal naive uses the correct in-sample source point")


def test_mase_matches_definition():
    y_true = np.array([10.0, 12.0, 14.0])
    y_pred = np.array([11.0, 12.0, 15.0])
    train = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 16.0])
    scale = metrics.seasonal_naive_scale(train, period=1)
    assert math.isclose(scale, np.mean(np.abs(np.diff(train.to_numpy()))))
    assert math.isclose(metrics.mase(y_true, y_pred, scale), metrics.mae(y_true, y_pred) / scale)
    print("ok  MASE matches the Hyndman definition")


def test_multi_source_merge(tmp_dir=None):
    """Two files, one period each plus an enrichment table -> one coalesced frame."""
    import tempfile
    tmp = Path(tmp_dir or tempfile.mkdtemp())
    (tmp / "part_a").mkdir(exist_ok=True)
    idx = pd.date_range("2021-01-01", periods=20, freq="D")
    pd.DataFrame({"date": idx[:10], "sales": range(10), "store": "A"}).to_csv(tmp / "part_a" / "y2021a.csv", index=False)
    pd.DataFrame({"date": idx[10:], "sales": range(10, 20), "store": "A"}).to_csv(tmp / "y2021b.csv", index=False)
    pd.DataFrame({"date": idx, "weather": np.arange(20) * 0.5}).to_csv(tmp / "weather.csv", index=False)

    df, report = load_dataset(tmp)
    assert report.time_column == "date"
    assert len(df) == 20, df
    assert "sales" in df.columns and "weather" in df.columns
    assert df["sales"].notna().all()
    assert df["weather"].notna().all()
    print("ok  multi-source merge (stack + join + coalesce)")


def test_overlapping_sources_are_coalesced():
    """A sparse source and a rich source for the same dates collapse to one row per date."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    rich = pd.DataFrame({"date": idx, "sales": np.arange(10.0), "promo": 1, "extra": 2.0})
    sparse = rich[["date", "sales", "promo", "extra"]].copy()
    sparse.loc[:, "extra"] = np.nan
    sparse.loc[3, "sales"] = np.nan
    rich.to_csv(tmp / "rich.csv", index=False)
    sparse.to_csv(tmp / "sparse.csv", index=False)

    df, report = load_dataset(tmp)
    assert len(df) == 10, len(df)
    assert df["extra"].notna().all(), "the richer source should fill the sparse one"
    assert df["sales"].notna().all()
    print("ok  overlapping sources coalesce to the most complete value")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print("FAIL {}: {}".format(fn.__name__, exc))
        except Exception as exc:
            failures += 1
            print("ERROR {}: {}: {}".format(fn.__name__, type(exc).__name__, exc))
    print("\n{}/{} passed".format(len(tests) - failures, len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
