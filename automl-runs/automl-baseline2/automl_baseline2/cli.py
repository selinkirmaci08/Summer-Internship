"""Command-line entry point.

    python -m automl_baseline2 --run ../runs/run_<id> --budget 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib

from .evaluate import head_to_head, build_prediction_table, score
from .protocol import load_agentic_run
from .report import write_all
from .tiers import TIERS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="automl-baseline2",
        description="AutoML baselines scored on an agentic run's exact held-out rows.",
    )
    p.add_argument("--run", required=True, help="agentic run directory (runs/run_<id>)")
    p.add_argument("--tiers", default="B1_agent_features,B2_preprocessed,B3_assembled",
                   help="comma-separated tiers to fit (default: all three)")
    p.add_argument("--budget", type=int, default=600,
                   help="AutoML search seconds per tier (default: 600)")
    p.add_argument("--metric", default="mae", help="what the search optimises (default: mae)")
    p.add_argument("--estimators", default="",
                   help="comma-separated FLAML estimators, e.g. lgbm,xgboost,rf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target", default=None, help="override the target read from the run")
    p.add_argument("--naive-col", default="lag_sw_1",
                   help="column holding the seasonal-naive reference (default: lag_sw_1)")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save-models", action="store_true", help="persist each fitted AutoML model")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet
    started = time.time()

    def log(msg):
        if verbose:
            print(msg)

    run = load_agentic_run(args.run, target=args.target, naive_col=args.naive_col)
    log("[run]    {} | target '{}' | {} horizons {}".format(
        run.run_dir.name, run.target, len(run.horizons), list(run.horizons)))
    for name, sp in run.splits.items():
        log("[split]  {:<11}{} -> {} | {} rows across {} dates".format(
            name, sp.date_min.date(), sp.date_max.date(), sp.n_rows, sp.n_dates))
    log("[model]  agentic choice: {} ({})".format(run.chosen_iteration, run.final_model_path.name))

    out_dir = Path(args.output_dir) / (args.run_name or run.run_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The two systems that need no search: the run's own model, and the naive floor.
    systems, meta = {}, {"n_features": {}, "estimator": {}, "budget": {}, "search": {}}

    agentic_pred = run.predict_final("test")
    systems["agentic"] = agentic_pred
    model = run.load_final_model()
    meta["n_features"]["agentic"] = len(getattr(model, "feature_name_", []) or [])
    meta["estimator"]["agentic"] = type(model).__name__
    meta["budget"]["agentic"] = "agent-directed"

    systems["naive"] = run.naive_reference("test")
    meta["n_features"]["naive"] = 1
    meta["estimator"]["naive"] = "seasonal naive"
    meta["budget"]["naive"] = "none"

    estimators = tuple(e.strip() for e in args.estimators.split(",") if e.strip())
    wanted = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown = [t for t in wanted if t not in TIERS]
    if unknown:
        print("unknown tier(s): {} (known: {})".format(unknown, sorted(TIERS)), file=sys.stderr)
        return 2

    for tier in wanted:
        result = TIERS[tier](
            run, args.budget, metric=args.metric, seed=args.seed,
            estimators=estimators, log_dir=out_dir / "models", verbose=verbose,
        )
        systems[tier] = result["predictions"]["test"]
        meta["n_features"][tier] = result["search"]["n_features"]
        meta["estimator"][tier] = result["search"]["estimator"]
        meta["budget"][tier] = "{}s".format(args.budget)
        meta["search"][tier] = result["search"]
        if args.save_models:
            (out_dir / "models").mkdir(parents=True, exist_ok=True)
            joblib.dump(result["model"], out_dir / "models" / "{}.pkl".format(tier))

    table = build_prediction_table(run, systems)
    scored = score(run, systems, table=table)
    verdict = head_to_head(scored, champion="agentic", reference="naive")

    meta["protocol_notes"] = _protocol_notes(run, args)
    paths = write_all(out_dir, scored, verdict, table, run, meta)

    # Reproduction check: the agentic model rescored here must match what the run
    # reported, or the harness is reading the run wrong and nothing else it says
    # can be trusted.
    reported = run.reported_metrics.get("MAE")
    recomputed = scored["overall"]["agentic"]["MAE"]
    if reported is not None:
        ok = abs(reported - recomputed) < 0.01
        log("[check]  agentic MAE reported {} | recomputed {:.3f} | {}".format(
            reported, recomputed, "match" if ok else "MISMATCH"))
        if not ok:
            print("\nWARNING: the agentic model rescored here does not match the MAE the run "
                  "reported. Every comparison below is suspect until that is explained.\n",
                  file=sys.stderr)

    if verbose:
        print("\n" + json.dumps({
            "scored_rows": scored["n_scored_rows"],
            "MAE": {k: v["MAE"] for k, v in scored["overall"].items()},
            "MASE_test_denom": {k: v["MASE_test_denom"] for k, v in scored["overall"].items()},
            "ranking": verdict["ranking_by_MAE"],
            "horizons_won": verdict["horizons_won"],
        }, indent=2))
        print("\nreport: {}".format(paths["report_html"]))
        print("elapsed: {:.0f}s".format(time.time() - started))
    return 0


def _protocol_notes(run, args) -> list:
    d = run.mase_denominators()
    return [
        "Rows: the {} (target date, horizon) pairs in the agentic run's test partition, "
        "checked pair-by-pair against every tier before scoring.".format(
            run.splits["test"].n_rows),
        "Split: the agentic run's own chronological 70/15/15 by target date. Train ends "
        "{}, validation ends {}, test runs {} to {}.".format(
            run.splits["train"].date_max.date(), run.splits["validation"].date_max.date(),
            run.splits["test"].date_min.date(), run.splits["test"].date_max.date()),
        "Tuning: every AutoML tier selects hyperparameters on the run's validation "
        "partition, never on test, and never on a split of its own choosing.",
        "No recursion: each row predicts t+h in one jump from data at the origin t = "
        "t+h minus h days, so long horizons do not inherit accumulated one-step error.",
        "Trading days: the {} dates the agentic run kept. The 242 sub-10-unit non-trading "
        "days it filtered are excluded from every system, and unexported dates stay absent "
        "on the calendar so a lag of 7 still means seven days.".format(
            len(run.modellable_dates())),
        "Exogenous columns are lagged unless the run declared them knowable in advance "
        "(calendar terms, opening pattern, competitor closures). Weather is lagged for "
        "the AutoML tiers, where the agentic features used observed weather as a forecast "
        "proxy at h&nbsp;&le;&nbsp;7 - a small advantage to the agentic side on a feature "
        "the run measured at Spearman -0.076.",
        "B3 drops the run's must_drop_columns (same-day revenue, transaction and line "
        "counts). Lagging them instead would give the AutoML tier information the agentic "
        "model was denied, and the gap would stop meaning what it appears to mean.",
        "MASE denominators: train {:.3f}, test {:.3f}, both from the run's own "
        "seasonal-naive column.".format(d["train_denominator"], d["test_denominator"]),
        "AutoML engine: FLAML, {}s per tier, optimising {}.".format(args.budget, args.metric),
    ]


if __name__ == "__main__":
    sys.exit(main())
