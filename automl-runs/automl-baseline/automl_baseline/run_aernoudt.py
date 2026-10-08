"""One command: raw Aernoudt path + goal -> trained model + horizon scores.

    python -m automl_baseline.run_aernoudt \
        --data "C:/Users/selin/OneDrive/Desktop/alldata/aernoudt" \
        --goal "predicting the bread demand"

Runs the ETL (cached after the first pass), then the standard baseline: one
pooled model with the horizon as an input feature, scored at t+1, t+7, t+14,
t+30 and t+90 by rolling origin against seasonal-naive.

The target column comes from the confirmed decisions in `sources.aernoudt`
(quantity of product group 1, 'Brood'), not from parsing the goal text -- the
goal is recorded in the run output for traceability. Everything downstream of
the ETL is the same code any other dataset gets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .cli import build_parser as base_parser
from .config import DEFAULT_HORIZONS, RunConfig
from .runner import run_forecast
from .sources import aernoudt


def prepare_dataset(raw_root, cache_path, scope, duplicate_policy, refresh=False, verbose=True):
    """Build the daily table, reusing the cached CSV unless --refresh is given."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        if verbose:
            print("[etl] reusing {} (pass --refresh to rebuild)".format(cache_path), flush=True)
        return cache_path
    started = time.time()
    aernoudt.build(raw_root, cache_path, scope=scope, duplicate_policy=duplicate_policy,
                   verbose=verbose)
    if verbose:
        print("[etl] built in {:.0f}s".format(time.time() - started), flush=True)
    return cache_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="run-aernoudt",
        description="Raw Aernoudt POS data + goal -> baseline model and horizon scores.")
    p.add_argument("--data", required=True, help="the raw aernoudt directory")
    p.add_argument("--goal", required=True, help="what to predict, in plain language")
    p.add_argument("--cache", default="data/aernoudt_daily.csv",
                   help="where the flattened daily table is written/read")
    p.add_argument("--refresh", action="store_true", help="rebuild the daily table from the XML")
    p.add_argument("--scope", default=aernoudt.DECISIONS["scope"], choices=["chain", "store"])
    p.add_argument("--duplicate-policy", default=aernoudt.DECISIONS["duplicate_policy"],
                   choices=["newest", "oldest", "error"])
    p.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    p.add_argument("--budget", type=int, default=240, help="AutoML search seconds")
    p.add_argument("--strategy", default="pooled", choices=["pooled", "direct"])
    p.add_argument("--test-days", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--mlflow", action="store_true")
    args = p.parse_args(argv)

    raw_root = Path(args.data)
    cache = prepare_dataset(raw_root, args.cache, args.scope, args.duplicate_policy,
                            refresh=args.refresh)

    target = aernoudt.target_column(raw_root)
    print("[target] {} -- {} of product group(s) {}, as confirmed".format(
        target, aernoudt.DECISIONS["target_measure"], aernoudt.DECISIONS["target_groups"]))

    cfg = RunConfig(
        data_path=str(cache),
        goal=args.goal,
        target=target,
        time_col="date",
        horizons=tuple(int(h) for h in args.horizons.split(",") if h.strip()),
        strategy=args.strategy,
        time_budget=args.budget,
        test_days=args.test_days,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name or "aernoudt_{}_{}".format(args.scope, args.strategy),
        mlflow=args.mlflow,
    )
    result = run_forecast(cfg)

    summary = result["summary"]
    print("\n" + "=" * 72)
    print("goal: {}".format(args.goal))
    print("target: {} | scope: {} | {} model(s)".format(
        target, args.scope, result["protocol"].get("models_fitted")))
    print("=" * 72)
    for entry in result["per_horizon"]:
        if entry["status"] != "ok":
            print("  t+{:<4} skipped - {}".format(entry["horizon"], entry.get("reason", "")))
            continue
        print("  t+{:<4} MASE {:.3f} | MAE {:8.2f} | WAPE {:5.1f}% | vs seasonal-naive {:+6.1%}".format(
            entry["horizon"], entry["automl"]["mase"], entry["automl"]["mae"],
            entry["automl"]["wape_pct"], entry["skill_vs_seasonal_naive"]))
    print("\n  mean MASE {:.3f} | beats seasonal-naive on {}/{} horizons".format(
        summary["mean_mase"], summary["horizons_beating_seasonal_naive"],
        summary["horizons_evaluated"]))
    print("\nreport: {}".format(result.get("artifact_paths", {}).get("report_html")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
