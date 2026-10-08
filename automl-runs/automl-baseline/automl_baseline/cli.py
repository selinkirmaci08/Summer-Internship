"""Command-line entry point.

    python -m automl_baseline --data <path> --goal "<goal>" [--target col] [options]
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import DEFAULT_HORIZONS, RunConfig
from .ingest import PanelDataError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="automl-baseline",
        description="AutoML baseline: dataset path + goal -> per-horizon forecast performance.",
    )
    p.add_argument("--data", required=True,
                   help="file or directory (searched recursively; multiple sources are merged)")
    p.add_argument("--goal", required=True, help="what to predict, in plain language")
    p.add_argument("--target", default=None, help="target column (inferred from the goal if omitted)")
    p.add_argument("--time-col", default=None, help="time index column (auto-detected if omitted)")
    p.add_argument("--series-id", default=None,
                   help="column that splits the data into several series (store_id, sku, region); "
                        "required when timestamps repeat across entities")
    p.add_argument("--series", default=None,
                   help="with --series-id: model this one series")
    p.add_argument("--aggregate", default=None, choices=["sum", "mean"],
                   help="with --series-id: roll every series up into one total and forecast that")
    p.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS),
                   help="comma-separated steps ahead to evaluate (default: 1,7,14,30,90)")
    p.add_argument("--strategy", default="pooled", choices=["pooled", "direct"],
                   help="'pooled' (default): ONE model with the horizon as an input feature - "
                        "hand it a time and it predicts. 'direct': a separate model per horizon.")
    p.add_argument("--sparse-grid", action="store_true",
                   help="pooled: train only on the scored horizons instead of every step 1..max "
                        "(faster, but the model can only answer those horizons)")
    p.add_argument("--max-train-rows", type=int, default=200000,
                   help="pooled: cap on stacked training rows before the horizon grid is thinned")
    p.add_argument("--budget", type=int, default=60,
                   help="AutoML search seconds (per horizon with --strategy direct; total for "
                        "the single model with --strategy pooled)")
    p.add_argument("--test-days", type=int, default=None,
                   help="held-out window length in periods (default: 2x the longest horizon)")
    p.add_argument("--train-window", type=int, default=None,
                   help="cap training to the last N periods before the cutoff (lag features still "
                        "use the full history); useful when the series has an old regime")
    p.add_argument("--freq", default=None, help="pandas frequency string, e.g. D, W, MS (inferred if omitted)")
    p.add_argument("--known-future", default="",
                   help="comma-separated columns whose future values are known at forecast time "
                        "(holiday flags, planned promotions, weather forecasts)")
    p.add_argument("--metric", default="mae", help="metric AutoML optimises during search (default: mae)")
    p.add_argument("--estimators", default="",
                   help="comma-separated FLAML estimators, e.g. lgbm,xgboost,rf (default: FLAML's list)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--mlflow", action="store_true", help="log params, metrics and artifacts to MLflow")
    p.add_argument("--mlflow-experiment", default="automl-baseline")
    p.add_argument("--tabular", action="store_true",
                   help="force plain tabular AutoML (no horizons) even if a time column exists")
    p.add_argument("--quiet", action="store_true")
    return p


def config_from_args(args) -> RunConfig:
    return RunConfig(
        data_path=args.data,
        goal=args.goal,
        target=args.target,
        time_col=args.time_col,
        series_id=args.series_id,
        series=args.series,
        aggregate=args.aggregate,
        horizons=tuple(int(h) for h in str(args.horizons).split(",") if str(h).strip()),
        strategy=args.strategy,
        dense_grid=not args.sparse_grid,
        max_train_rows=args.max_train_rows,
        time_budget=args.budget,
        test_days=args.test_days,
        train_window=args.train_window,
        freq=args.freq,
        known_future=tuple(c.strip() for c in args.known_future.split(",") if c.strip()),
        seed=args.seed,
        metric=args.metric,
        output_dir=args.output_dir,
        run_name=args.run_name,
        estimators=tuple(e.strip() for e in args.estimators.split(",") if e.strip()),
        mlflow=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    verbose = not args.quiet

    try:
        return _run(args, cfg, verbose)
    except PanelDataError as exc:
        # A choice the caller has to make, not a crash -- say so without a traceback.
        print("\n[panel data] {}\n".format(exc), file=sys.stderr)
        return 2


def _run(args, cfg: RunConfig, verbose: bool) -> int:
    if args.tabular:
        from .tabular import run_tabular
        result = run_tabular(cfg, verbose=verbose)
    else:
        from .runner import run_forecast
        try:
            result = run_forecast(cfg, verbose=verbose)
        except ValueError as exc:
            if "no time column" not in str(exc):
                raise
            if verbose:
                print("[fallback] {} -- running plain tabular AutoML instead".format(exc))
            from .tabular import run_tabular
            result = run_tabular(cfg, verbose=verbose)

    if verbose:
        print("\n" + json.dumps(result.get("summary", result.get("test_metrics", {})), indent=2, default=str))
        paths = result.get("artifact_paths", {})
        if paths.get("report_html"):
            print("\nreport: {}".format(paths["report_html"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
