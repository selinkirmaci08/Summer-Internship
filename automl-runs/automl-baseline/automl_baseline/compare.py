"""Head-to-head: agentic pipeline vs the AutoML baseline, on identical rows.

    python -m automl_baseline.compare --baseline-run outputs/<run> \
        --candidate agentic_predictions.csv --name "agentic v3"

The candidate CSV needs `horizon`, `target_time` and a prediction column
(`y_pred`, `prediction` or `pred`); `y_true` is optional and only used as a
consistency check. Rows are joined on (horizon, target_time) and every metric is
recomputed on the intersection, so neither system gets scored on rows the other
never saw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics

PRED_ALIASES = ("y_pred", "prediction", "pred", "forecast", "yhat")


def _load_candidate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "horizon" not in df.columns:
        raise ValueError("candidate file needs a 'horizon' column")
    time_col = next((c for c in ("target_time", "target_date", "date", "ds") if c in df.columns), None)
    if time_col is None:
        raise ValueError("candidate file needs a 'target_time' (or target_date/date/ds) column")
    pred_col = next((c for c in PRED_ALIASES if c in df.columns), None)
    if pred_col is None:
        raise ValueError("candidate file needs one of {} as the prediction column".format(PRED_ALIASES))
    out = pd.DataFrame({
        "horizon": df["horizon"].astype(int),
        "target_time": pd.to_datetime(df[time_col]),
        "pred_candidate": pd.to_numeric(df[pred_col], errors="coerce"),
    })
    if "y_true" in df.columns:
        out["y_true_candidate"] = pd.to_numeric(df["y_true"], errors="coerce")
    return out


def compare(baseline_run: Path, candidate_path: Path, candidate_name: str = "candidate") -> dict:
    baseline_run = Path(baseline_run)
    with open(baseline_run / "metrics.json", encoding="utf-8") as fh:
        base_result = json.load(fh)
    base = pd.read_csv(baseline_run / "predictions.csv", parse_dates=["origin_time", "target_time"])
    cand = _load_candidate(Path(candidate_path))

    merged = base.merge(cand, on=["horizon", "target_time"], how="inner")
    if merged.empty:
        raise ValueError(
            "no overlapping (horizon, target_time) rows between the baseline and the candidate -- "
            "check that the candidate forecasts the same cutoff window ({} onward) and the same "
            "horizons {}".format(base_result["protocol"]["cutoff"], sorted(base.horizon.unique()))
        )

    mismatch = 0
    if "y_true_candidate" in merged.columns:
        diff = (merged["y_true"] - merged["y_true_candidate"]).abs()
        mismatch = int((diff > 1e-6 * merged["y_true"].abs().clip(lower=1)).sum())

    if candidate_name in ("baseline", "seasonal_naive"):
        candidate_name = candidate_name + "_candidate"
    scale = float(base_result["protocol"]["mase_scale"])
    rows = []
    for h in sorted(merged.horizon.unique()):
        block = merged.loc[merged.horizon == h]
        y = block["y_true"].to_numpy()
        entry = {
            "horizon": int(h),
            "n_rows_compared": int(len(block)),
            "n_rows_baseline_only": int((base.horizon == h).sum() - len(block)),
            "baseline": metrics.all_metrics(y, block["pred_automl"].to_numpy(), scale),
            candidate_name: metrics.all_metrics(y, block["pred_candidate"].to_numpy(), scale),
            "seasonal_naive": metrics.all_metrics(y, block["pred_seasonal_naive"].to_numpy(), scale),
        }
        b_mae, c_mae = entry["baseline"]["mae"], entry[candidate_name]["mae"]
        entry["mae_delta"] = float(c_mae - b_mae)
        entry["mae_improvement_pct"] = float((1 - c_mae / b_mae) * 100) if b_mae > 0 else float("nan")
        entry["winner"] = candidate_name if c_mae < b_mae else "baseline"
        rows.append(entry)

    wins = sum(1 for r in rows if r["winner"] == candidate_name)
    result = {
        "baseline_run": str(baseline_run),
        "candidate_file": str(candidate_path),
        "candidate_name": candidate_name,
        "target": base_result["task"]["target"],
        "cutoff": base_result["protocol"]["cutoff"],
        "mase_scale": scale,
        "ground_truth_mismatches": mismatch,
        "summary": {
            "horizons_compared": len(rows),
            "horizons_won_by_candidate": wins,
            "mean_mae_improvement_pct": float(np.nanmean([r["mae_improvement_pct"] for r in rows])),
            "mean_mase_baseline": float(np.nanmean([r["baseline"]["mase"] for r in rows])),
            "mean_mase_candidate": float(np.nanmean([r[candidate_name]["mase"] for r in rows])),
        },
        "per_horizon": rows,
    }
    if mismatch:
        result.setdefault("warnings", []).append(
            "{} rows where the candidate's y_true disagrees with the baseline's -- the two systems "
            "may not be reading the same ground truth".format(mismatch))
    return result


def to_markdown(result: dict) -> str:
    name = result["candidate_name"]
    s = result["summary"]
    lines = ["# {} vs AutoML baseline - {}".format(name, result["target"]), ""]
    lines.append("Same cutoff ({}), same held-out origins, same metrics.".format(result["cutoff"]))
    lines.append("")
    lines.append("- horizons won by {}: **{} / {}**".format(name, s["horizons_won_by_candidate"], s["horizons_compared"]))
    lines.append("- mean MAE improvement over the baseline: **{:+.1f}%**".format(s["mean_mae_improvement_pct"]))
    lines.append("- mean MASE: baseline {:.3f} vs {} {:.3f}".format(
        s["mean_mase_baseline"], name, s["mean_mase_candidate"]))
    for warn in result.get("warnings", []):
        lines.append("- warning: {}".format(warn))
    lines.append("")
    lines.append("| horizon | rows | baseline MAE | {} MAE | delta | improvement | baseline MASE | {} MASE | winner |".format(name, name))
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in result["per_horizon"]:
        lines.append("| t+{} | {} | {:.2f} | {:.2f} | {:+.2f} | {:+.1f}% | {:.3f} | {:.3f} | {} |".format(
            r["horizon"], r["n_rows_compared"], r["baseline"]["mae"], r[name]["mae"],
            r["mae_delta"], r["mae_improvement_pct"], r["baseline"]["mase"], r[name]["mase"], r["winner"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="automl-baseline-compare",
                                description="Compare a candidate model against a baseline run.")
    p.add_argument("--baseline-run", required=True, help="a run directory produced by automl_baseline")
    p.add_argument("--candidate", required=True, help="candidate predictions CSV")
    p.add_argument("--name", default="candidate", help="label for the candidate in the report")
    p.add_argument("--out", default=None, help="where to write comparison.json/.md (default: the run dir)")
    args = p.parse_args(argv)

    result = compare(Path(args.baseline_run), Path(args.candidate), args.name)
    out_dir = Path(args.out or args.baseline_run)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    md = to_markdown(result)
    (out_dir / "comparison.md").write_text(md, encoding="utf-8")
    print(md)
    print("\nwritten: {}".format(out_dir / "comparison.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
