"""The checks that would silently invalidate a comparison if they stopped holding.

None of these test the modelling. They test the three things that make a number
here mean anything at all:

  1. the harness reads the agentic run correctly, and rescoring its chosen model
     reproduces the MAE the run itself reported;
  2. every tier is scored on the agentic run's exact rows, not a near-match;
  3. no tier's features can see the target date.

Run with:  python tests/test_alignment.py [--run ../runs/run_<id>]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from automl_baseline2.evaluate import build_prediction_table, score  # noqa: E402
from automl_baseline2.protocol import load_agentic_run  # noqa: E402
from automl_baseline2.tiers import assert_grid_matches, build_generic_matrix  # noqa: E402

PASSED, FAILED = [], []


def check(name, fn):
    try:
        detail = fn()
        PASSED.append(name)
        print("  PASS  {}{}".format(name, " -- {}".format(detail) if detail else ""))
    except Exception as exc:  # noqa: BLE001 -- a failed check must report, not crash the suite
        FAILED.append((name, exc))
        print("  FAIL  {}\n          {}: {}".format(name, type(exc).__name__, exc))


# ---------------------------------------------------------------------------


def test_reproduces_reported_metrics(run):
    """Rescoring the run's chosen model must reproduce the run's own numbers.

    If this drifts, the harness is reading the feature table or the split wrong,
    and every comparison built on it is wrong in the same way.
    """
    reported = run.reported_metrics
    if not reported:
        return "run reported no test metrics; nothing to check against"
    pred = run.predict_final("test")
    truth = run.index("test")[run.target].to_numpy()
    mae = float(np.mean(np.abs(truth - pred)))
    rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
    assert abs(mae - reported["MAE"]) < 0.01, \
        "MAE {:.4f} != reported {}".format(mae, reported["MAE"])
    assert abs(rmse - reported["RMSE"]) < 0.01, \
        "RMSE {:.4f} != reported {}".format(rmse, reported["RMSE"])
    return "MAE {:.3f}, RMSE {:.3f}".format(mae, rmse)


def test_layout_is_horizon_conditioned(run):
    """origin_date must be exactly h days before the target date, on every row."""
    fe = run.fe_frame
    lag = (fe[run.time_col] - fe["origin_date"]).dt.days
    assert (lag == fe["h"]).all(), "origin is not h days before target on some rows"
    assert not fe.duplicated([run.time_col, "h"]).any(), "(date, h) is not unique"
    return "{} rows, {} horizons".format(len(fe), len(run.horizons))


def test_splits_are_chronological(run):
    """Train must end before validation begins, and validation before test."""
    s = run.splits
    assert s["train"].date_max < s["validation"].date_min, "train overlaps validation"
    assert s["validation"].date_max < s["test"].date_min, "validation overlaps test"
    return "train -> {} | val -> {} | test -> {}".format(
        s["train"].date_max.date(), s["validation"].date_max.date(), s["test"].date_max.date())


def test_generic_matrix_matches_grid(run):
    """The generic tiers must land on the agentic run's rows exactly."""
    built = build_generic_matrix(run.preprocess_frame, run)
    details = []
    for split in ("train", "validation", "test"):
        assert_grid_matches(built["matrix"], run, split, "B2")
        details.append("{} {}".format(split, int((built["matrix"]["_split"] == split).sum())))
    return " | ".join(details)


def test_no_target_leakage_in_generic_features(run):
    """Blank the series after an origin; that origin's features must not move.

    The constructive form of the leakage test: rather than inspecting names for
    suspicious-looking columns, it destroys the future and checks that nothing the
    model would see at the origin noticed.
    """
    frame = run.preprocess_frame.copy()
    cutoff = pd.Timestamp("2018-06-15")
    built_full = build_generic_matrix(frame, run)

    blanked = frame.copy()
    after = blanked[run.time_col] > cutoff
    numeric = blanked.select_dtypes(include=[np.number]).columns
    blanked.loc[after, numeric] = np.nan
    built_blank = build_generic_matrix(blanked, run)

    feats = [c for c in built_full["feature_cols"] if c in built_blank["feature_cols"]]
    key = [run.time_col, "h"]

    # Rows whose ORIGIN is at or before the cutoff: everything they may legitimately
    # read already existed before the blanking, so they must be bit-identical --
    # except for the columns that are deliberately evaluated at the target date,
    # which are the calendar terms and the declared known-in-advance columns.
    a = built_full["matrix"]
    b = built_blank["matrix"]
    a = a.loc[a["origin_date"] <= cutoff].set_index(key).sort_index()
    b = b.loc[b["origin_date"] <= cutoff].set_index(key).sort_index()
    # Blanking the target too means rows whose target date is after the cutoff drop
    # out of the second build. That is expected and is not what is under test, so
    # the comparison runs on the rows both builds still have.
    shared = a.index.intersection(b.index)
    assert len(shared) > 1000, "too few comparable rows ({}) to be a real check".format(len(shared))
    a, b = a.loc[shared], b.loc[shared]

    known = set(built_full["known_future"])
    past_only = [c for c in feats
                 if not c.startswith("cal_")
                 and not c.startswith("known_")
                 and c.split("exog_")[-1].rsplit("_lag", 1)[0] not in known]

    moved = []
    for col in past_only:
        left, right = a[col], b[col]
        if not pd.api.types.is_numeric_dtype(left):
            continue
        if not np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                           equal_nan=True):
            moved.append(col)
    assert not moved, "features changed when only the future was blanked: {}".format(moved[:6])
    return "{} past-only features unchanged across {} rows".format(len(past_only), len(a))


def test_scoring_uses_identical_rows(run):
    """Every system must be scored on the same mask, and a gap must be visible."""
    truth = run.index("test")[run.target].to_numpy()
    good = pd.DataFrame({run.time_col: run.index("test")[run.time_col],
                         "h": run.index("test")["h"],
                         "pred": truth})
    holed = good.iloc[10:].copy()   # a system that failed to predict 10 rows

    table = build_prediction_table(run, {"complete": good, "incomplete": holed})
    scored = score(run, {"complete": good, "incomplete": holed}, table=table)

    assert scored["coverage"]["complete"]["complete"] is True
    assert scored["coverage"]["incomplete"]["missing_rows"] == 10, "the gap was not reported"
    # Both systems scored on the intersection, so the complete one is not credited
    # for rows its rival never attempted.
    assert scored["n_scored_rows"] == len(truth) - 10, \
        "scoring did not fall back to the shared rows"
    assert scored["overall"]["complete"]["MAE"] == 0.0
    return "{} shared rows, 10-row gap reported".format(scored["n_scored_rows"])


def test_mase_denominators_match_run(run):
    """The MASE denominators must be the ones the run itself quotes."""
    d = run.mase_denominators()
    reported = run.reported_metrics
    if not reported or "MASE_train_denom" not in reported:
        return "run reported no MASE; denominators computed at {:.3f} / {:.3f}".format(
            d["train_denominator"], d["test_denominator"])
    pred = run.predict_final("test")
    truth = run.index("test")[run.target].to_numpy()
    mae = float(np.mean(np.abs(truth - pred)))
    assert abs(mae / d["train_denominator"] - reported["MASE_train_denom"]) < 0.001
    assert abs(mae / d["test_denominator"] - reported["MASE_test_denom"]) < 0.001
    return "train {:.3f}, test {:.3f}".format(d["train_denominator"], d["test_denominator"])


def main() -> int:
    default_run = Path(__file__).resolve().parents[1] / ".." / "runs" / "run_20260827T154407Z"
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(default_run.resolve()))
    args = ap.parse_args()

    # An explicit --run is resolved from the working directory, the way a path typed
    # at a shell prompt is expected to behave.
    run_dir = Path(args.run).expanduser().resolve()
    print("agentic run: {}\n".format(run_dir))
    run = load_agentic_run(run_dir)

    for name, fn in [
        ("reproduces the run's reported test metrics", test_reproduces_reported_metrics),
        ("layout is horizon-conditioned", test_layout_is_horizon_conditioned),
        ("splits are chronological and disjoint", test_splits_are_chronological),
        ("generic tiers land on the agentic grid", test_generic_matrix_matches_grid),
        ("no target leakage in generic features", test_no_target_leakage_in_generic_features),
        ("scoring uses identical rows and reports gaps", test_scoring_uses_identical_rows),
        ("MASE denominators match the run", test_mase_denominators_match_run),
    ]:
        check(name, lambda fn=fn: fn(run))

    print("\n{} passed, {} failed".format(len(PASSED), len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
