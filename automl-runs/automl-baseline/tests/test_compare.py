"""Guards for the head-to-head comparison: `python tests/test_compare.py`.

Builds a fake baseline run directory so the comparison logic can be checked
without paying for an AutoML search.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from automl_baseline.compare import compare  # noqa: E402


def _fake_run(tmp: Path, horizons=(1, 7, 90), n=40, scale=10.0) -> Path:
    run = tmp / "run"
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    blocks = []
    for h in horizons:
        targets = pd.date_range("2023-01-01", periods=n, freq="D")
        y = 100 + rng.normal(0, 5, n)
        blocks.append(pd.DataFrame({
            "horizon": h,
            "origin_time": targets - pd.Timedelta(days=int(h)),
            "target_time": targets,
            "y_true": y,
            "pred_automl": y + rng.normal(0, 8, n),
            "pred_seasonal_naive": y + rng.normal(0, 12, n),
            "pred_naive_last": y + rng.normal(0, 14, n),
        }))
    pd.concat(blocks, ignore_index=True).to_csv(run / "predictions.csv", index=False)
    with open(run / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"task": {"target": "sales"},
                   "protocol": {"cutoff": "2022-12-31", "mase_scale": scale}}, fh)
    return run


def test_perfect_candidate_wins_every_horizon():
    tmp = Path(tempfile.mkdtemp())
    run = _fake_run(tmp)
    base = pd.read_csv(run / "predictions.csv")
    base[["horizon", "target_time", "y_true"]].assign(y_pred=base["y_true"]).to_csv(
        tmp / "cand.csv", index=False)

    result = compare(run, tmp / "cand.csv", "agentic")
    assert result["summary"]["horizons_won_by_candidate"] == result["summary"]["horizons_compared"]
    assert all(r["winner"] == "agentic" for r in result["per_horizon"])
    assert result["summary"]["mean_mae_improvement_pct"] > 99
    assert result["ground_truth_mismatches"] == 0
    print("ok  a perfect candidate wins every horizon")


def test_metrics_are_computed_on_the_intersection_only():
    """A candidate covering half the rows is scored on those rows, and it is recorded."""
    tmp = Path(tempfile.mkdtemp())
    run = _fake_run(tmp)
    base = pd.read_csv(run / "predictions.csv")
    half = base.groupby("horizon", group_keys=False).head(20)
    half[["horizon", "target_time"]].assign(y_pred=half["pred_seasonal_naive"]).to_csv(
        tmp / "cand.csv", index=False)

    result = compare(run, tmp / "cand.csv", "agentic")
    for r in result["per_horizon"]:
        assert r["n_rows_compared"] == 20, r
        assert r["n_rows_baseline_only"] == 20, r
        # scoring the baseline on the intersection, not on all 40 rows
        assert r["baseline"]["n"] == 20
    print("ok  metrics use only the overlapping rows, and the gap is reported")


def test_ground_truth_mismatch_is_flagged():
    tmp = Path(tempfile.mkdtemp())
    run = _fake_run(tmp)
    base = pd.read_csv(run / "predictions.csv")
    cand = base[["horizon", "target_time", "y_true"]].copy()
    cand["y_pred"] = base["pred_automl"]
    cand.loc[0:4, "y_true"] = cand.loc[0:4, "y_true"] + 50  # a different ground truth
    cand.to_csv(tmp / "cand.csv", index=False)

    result = compare(run, tmp / "cand.csv", "agentic")
    assert result["ground_truth_mismatches"] == 5, result["ground_truth_mismatches"]
    assert result.get("warnings"), "a ground-truth disagreement must surface as a warning"
    print("ok  disagreeing ground truth is flagged, not silently averaged")


def test_no_overlap_raises_a_useful_error():
    tmp = Path(tempfile.mkdtemp())
    run = _fake_run(tmp)
    pd.DataFrame({"horizon": [1], "target_time": ["2019-01-01"], "y_pred": [100.0]}).to_csv(
        tmp / "cand.csv", index=False)
    try:
        compare(run, tmp / "cand.csv", "agentic")
    except ValueError as exc:
        assert "no overlapping" in str(exc)
        print("ok  a non-overlapping candidate raises a useful error")
        return
    raise AssertionError("expected a ValueError for a non-overlapping candidate")


def test_mase_uses_the_baseline_run_scale():
    tmp = Path(tempfile.mkdtemp())
    run = _fake_run(tmp, scale=25.0)
    base = pd.read_csv(run / "predictions.csv")
    base[["horizon", "target_time"]].assign(y_pred=base["pred_automl"]).to_csv(tmp / "cand.csv", index=False)
    result = compare(run, tmp / "cand.csv", "agentic")
    assert math.isclose(result["mase_scale"], 25.0)
    for r in result["per_horizon"]:
        assert math.isclose(r["agentic"]["mase"], r["agentic"]["mae"] / 25.0, rel_tol=1e-9)
    print("ok  MASE reuses the baseline run's scale, so both systems share a denominator")


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
