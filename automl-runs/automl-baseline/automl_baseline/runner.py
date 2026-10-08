"""Orchestration: dataset path + goal -> per-horizon performance numbers.

Evaluation protocol (this is the part that has to match whatever the agentic
pipeline is scored with):

  * The series is cut at a single point in time. Everything whose *target*
    timestamp falls at or before the cutoff is training data; everything after is
    the held-out window. The split is on the target timestamp, not the origin, so
    a horizon-90 model can never have been fitted on a value it is later asked to
    predict.
  * Inside the held-out window every valid origin t is forecast, giving a
    rolling-origin evaluation: for h=90 and a 180-period window that is ~90
    independent forecasts, not one lucky shot.
  * By default ONE model is fitted, with the horizon as an ordinary input feature
    ("pooled"): hand it an origin and a time and it predicts, for any horizon in
    the training grid. `--strategy direct` fits a separate model per horizon
    instead, which scores each horizon independently at the cost of N models.
  * Models are fitted once and not refitted as the window advances, but each
    origin's features use the actuals up to that origin -- i.e. the setup assumes
    you know the past when you forecast, which is the realistic case.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines, metrics, search
from .config import RunConfig, infer_target
from .features import HorizonDataset, prepare_series, seasonal_period
from .ingest import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in str(text).lower()]
    return "".join(keep).strip("-")[:40] or "run"


def _resolve_run_dir(cfg: RunConfig, target: str) -> Path:
    name = cfg.run_name or "{}_{}".format(time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), _slug(target))
    out = Path(cfg.output_dir) / name
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def _choose_test_window(n_periods: int, horizons, explicit: int | None) -> tuple:
    """Pick the length of the held-out window."""
    max_h = max(horizons)
    if explicit:
        return int(explicit), "explicit --test-days"
    proposed = max(2 * max_h, 30)
    cap = max(1, int(n_periods * 0.35))
    if proposed > cap:
        # The cap wins: never hand the test window more than a third of the series,
        # even if that means the longest horizons cannot be evaluated at all.
        return cap, (
            "2x the longest horizon ({}) would consume more than 35% of a {}-period series; "
            "held-out window capped at {} periods, so horizons near or beyond the training "
            "length are skipped".format(max_h, n_periods, cap)
        )
    return proposed, "2x the longest horizon ({}), so every horizon gets multiple origins".format(max_h)


def run_forecast(cfg: RunConfig, verbose: bool = True) -> dict:
    started = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    # ---------------------------------------------------------------- ingest
    raw, ingest_report = load_dataset(
        cfg.data_path, cfg.time_col,
        series_id=cfg.series_id, series=cfg.series, aggregate=cfg.aggregate,
    )
    time_col = ingest_report.time_column
    if time_col is None:
        raise ValueError(
            "no time column found under {} -- forecasting horizons need one. "
            "Pass --time-col, or run in tabular mode.".format(cfg.data_path)
        )

    target = cfg.target
    target_candidates = []
    if not target:
        target, target_candidates = infer_target(cfg.goal, raw.columns, exclude=(time_col,))
        if not target:
            raise ValueError(
                "could not infer a target column from the goal '{}'. Pass --target explicitly. "
                "Columns: {}".format(cfg.goal, list(raw.columns))
            )
        log("[target] inferred '{}' from the goal (candidates: {})".format(
            target, ", ".join(c["column"] for c in target_candidates)))

    frame, freq = prepare_series(raw, time_col, target, cfg.freq)
    period = seasonal_period(freq)
    y = frame[target]
    n_periods = len(frame)
    log("[data] {} rows x {} cols | {} -> {} | freq={} | seasonal period={}".format(
        n_periods, frame.shape[1], frame.index.min().date(), frame.index.max().date(), freq, period))

    run_dir = _resolve_run_dir(cfg, target)

    # ------------------------------------------------------------ split point
    test_len, split_reason = _choose_test_window(n_periods, cfg.horizons, cfg.test_days)
    cutoff = frame.index[max(0, n_periods - test_len - 1)]
    train_y = y.loc[:cutoff]
    scale = metrics.seasonal_naive_scale(train_y, period)
    log("[split] cutoff {} | train {} periods | held-out {} periods ({})".format(
        cutoff.date(), len(train_y), n_periods - len(train_y), split_reason))

    # ----------------------------------------------------------- feature prep
    ds = HorizonDataset(frame, target, freq, known_future=cfg.known_future)

    per_horizon = []
    all_predictions = []

    # ------------------------------------------------- one pooled model (default)
    pooled = None
    if cfg.strategy == "pooled":
        grid = ds.horizon_grid(cfg.horizons, dense=cfg.dense_grid)
        Xp, yp, origin_p, target_p, horizon_p = ds.build_pooled(
            grid, max_rows=cfg.max_train_rows, must_keep=cfg.horizons)
        train_mask = (target_p <= cutoff).to_numpy()
        if cfg.train_window:
            earliest = cutoff - cfg.train_window * pd.tseries.frequencies.to_offset(freq)
            train_mask = train_mask & (target_p > earliest).to_numpy()
        n_train = int(train_mask.sum())
        if n_train < 30:
            raise ValueError(
                "only {} pooled training rows before the cutoff -- the series is too short "
                "for these horizons".format(n_train))
        trained_h = sorted(set(int(v) for v in horizon_p[train_mask]))
        log("[model] ONE model across horizons {}-{} ({} of them) | {} pooled train rows, "
            "{} features | {}s budget".format(
                min(trained_h), max(trained_h), len(trained_h), n_train, Xp.shape[1], cfg.time_budget))
        pooled = search.fit_regressor(
            Xp.loc[train_mask], yp.loc[train_mask],
            time_budget=cfg.time_budget,
            metric=cfg.metric,
            seed=cfg.seed,
            estimators=cfg.estimators,
            log_file=str(run_dir / "models" / "flaml_pooled.log"),
        )
        pooled["horizons_trained"] = trained_h
        missing = [h for h in cfg.horizons if h not in set(trained_h)]
        if missing:
            log("[model] WARNING: horizons {} are not in the training grid - the model is "
                "extrapolating for them".format(missing))
            pooled["horizons_extrapolated"] = missing
        log("[model] picked {} | validation {} {:.3f}".format(
            pooled["best_estimator"], cfg.metric, pooled["best_validation_loss"]))
        try:
            import joblib
            joblib.dump(pooled["model"], run_dir / "models" / "automl_pooled.pkl")
        except Exception as exc:
            log("[model] not saved: {}".format(exc))

    for h in cfg.horizons:
        X, y_h, target_time = ds.build(h)
        train_mask = (target_time <= cutoff).to_numpy()
        test_mask = ~train_mask
        if cfg.train_window:
            # Lags still see the full history; only the training rows are capped, which
            # is how you keep an old regime out of the fit without losing feature depth.
            earliest = cutoff - cfg.train_window * pd.tseries.frequencies.to_offset(freq)
            train_mask = train_mask & (target_time > earliest).to_numpy()
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())

        if (n_train < 30 and pooled is None) or n_test < 3:
            per_horizon.append({
                "horizon": h,
                "status": "skipped",
                "reason": "not enough data at this horizon (train rows={}, test origins={}); "
                          "need >=30 train and >=3 test".format(n_train, n_test),
                "n_train": n_train,
                "n_test": n_test,
            })
            log("[h={}] skipped - train={} test={}".format(h, n_train, n_test))
            continue

        X_tr, y_tr = X.loc[train_mask], y_h.loc[train_mask]
        X_te, y_te = X.loc[test_mask], y_h.loc[test_mask]
        origins_te = X_te.index
        targets_te = target_time.loc[test_mask]

        if pooled is not None:
            # The single model answers every horizon; `horizon` is just another input.
            fitted = pooled
            X_te_h = X_te.copy()
            X_te_h["horizon"] = h
            preds = search.predict(fitted, X_te_h[pooled["feature_order"]])
        else:
            log("[h={}] searching ({}s budget, {} train rows, {} features)...".format(
                h, cfg.time_budget, n_train, X.shape[1]))
            fitted = search.fit_regressor(
                X_tr, y_tr,
                time_budget=cfg.time_budget,
                metric=cfg.metric,
                seed=cfg.seed,
                estimators=cfg.estimators,
                log_file=str(run_dir / "models" / "flaml_h{}.log".format(h)),
            )
            preds = search.predict(fitted, X_te)

        base_preds = baselines.all_baselines(y, origins_te, h, period)
        model_metrics = metrics.all_metrics(y_te.to_numpy(), preds, scale)
        base_metrics = {
            name: metrics.all_metrics(y_te.to_numpy(), np.asarray(series, dtype=float), scale)
            for name, series in base_preds.items()
        }

        snaive_mae = base_metrics["seasonal_naive"]["mae"]
        skill = (
            float(1.0 - model_metrics["mae"] / snaive_mae)
            if np.isfinite(snaive_mae) and snaive_mae > 0 else float("nan")
        )

        # Zero actuals (closed days) dominate percentage metrics and are a different
        # prediction problem from "how much do we sell when we are open", so the
        # open-days-only view is reported alongside the full one.
        y_te_arr = y_te.to_numpy()
        open_mask = y_te_arr > 0
        open_metrics, open_skill = None, float("nan")
        if int(open_mask.sum()) >= 3:
            open_metrics = metrics.all_metrics(y_te_arr[open_mask], preds[open_mask], scale)
            open_snaive = metrics.mae(
                y_te_arr[open_mask],
                np.asarray(base_preds["seasonal_naive"], dtype=float)[open_mask],
            )
            if np.isfinite(open_snaive) and open_snaive > 0:
                open_skill = float(1.0 - open_metrics["mae"] / open_snaive)

        entry = {
            "horizon": h,
            "status": "ok",
            "n_train": n_train,
            "n_test_origins": n_test,
            "test_target_window": [str(targets_te.min().date()), str(targets_te.max().date())],
            "automl": model_metrics,
            "baselines": base_metrics,
            "skill_vs_seasonal_naive": skill,
            "beats_seasonal_naive": bool(np.isfinite(skill) and skill > 0),
            "open_days_only": open_metrics,
            "open_days_skill_vs_seasonal_naive": open_skill,
            "zero_share_of_test": float(np.mean(y_te_arr == 0)),
            "search": dict({k: v for k, v in fitted.items() if k not in ("model", "feature_order")},
                           shared_model=pooled is not None),
            "top_features": search.feature_importance(fitted, list(X.columns)),
        }
        per_horizon.append(entry)
        log("[h={}] MASE {:.3f} | MAE {:.2f} | WAPE {:.1f}% | vs seasonal-naive {:+.1%} "
            "(open days {:+.1%}) | {}".format(
                h, model_metrics["mase"], model_metrics["mae"], model_metrics["wape_pct"],
                skill, open_skill, fitted["best_estimator"]))

        block = pd.DataFrame({
            "horizon": h,
            "origin_time": origins_te,
            "target_time": targets_te.to_numpy(),
            "y_true": y_te.to_numpy(),
            "pred_automl": preds,
        })
        for name, series in base_preds.items():
            block["pred_" + name] = np.asarray(series, dtype=float)
        all_predictions.append(block)

        try:
            import joblib
            joblib.dump(fitted["model"], run_dir / "models" / "automl_h{}.pkl".format(h))
        except Exception as exc:
            entry.setdefault("warnings", []).append("model not saved: {}".format(exc))

    held_out_y = y.loc[y.index > cutoff]
    observed_held_out = int(held_out_y.notna().sum())
    data_quality = {
        "coverage_overall": round(float(y.notna().mean()), 4),
        "coverage_held_out": round(float(held_out_y.notna().mean()), 4),
        "zeros_overall": int((y == 0).sum()),
        "zeros_held_out": int((held_out_y == 0).sum()),
        "zero_share_held_out": round(float((held_out_y == 0).sum() / max(1, observed_held_out)), 4),
        "flags": [],
    }
    if data_quality["coverage_held_out"] < 0.7:
        data_quality["flags"].append(
            "only {:.0%} of the held-out window has an observed value - horizon scores rest on "
            "{} actual days".format(data_quality["coverage_held_out"], observed_held_out))
    if data_quality["zero_share_held_out"] > 0.05:
        data_quality["flags"].append(
            "{:.0%} of observed held-out days are exactly 0 (closed days) - these inflate SMAPE/MAPE "
            "for every model; read WAPE, MAE and the open-days-only table instead".format(
                data_quality["zero_share_held_out"]))

    ok_rows = [e for e in per_horizon if e["status"] == "ok"]
    if not ok_rows:
        raise ValueError("no horizon could be evaluated; see per-horizon reasons in the result")

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    if not predictions.empty:
        predictions.to_csv(run_dir / "predictions.csv", index=False)

    summary = {
        "mean_mase": float(np.nanmean([e["automl"]["mase"] for e in ok_rows])),
        "mean_smape_pct": float(np.nanmean([e["automl"]["smape_pct"] for e in ok_rows])),
        "mean_skill_vs_seasonal_naive": float(np.nanmean([e["skill_vs_seasonal_naive"] for e in ok_rows])),
        "horizons_beating_seasonal_naive": int(sum(e["beats_seasonal_naive"] for e in ok_rows)),
        "horizons_evaluated": len(ok_rows),
        "mean_wape_pct": float(np.nanmean([e["automl"]["wape_pct"] for e in ok_rows])),
        "mean_mase_open_days": float(np.nanmean(
            [e["open_days_only"]["mase"] for e in ok_rows if e.get("open_days_only")]
        )) if any(e.get("open_days_only") for e in ok_rows) else float("nan"),
        "mean_skill_open_days": float(np.nanmean(
            [e["open_days_skill_vs_seasonal_naive"] for e in ok_rows]
        )),
        "headline_metric": "skill_vs_seasonal_naive",
        "headline_note": "MASE is scaled by the in-sample seasonal-naive MAE of the training window "
                         "(period={}), so it is comparable across horizons; skill_vs_seasonal_naive is "
                         "the same-window verdict and is what decides whether a horizon was won.".format(period),
    }

    result = {
        "run": {
            "run_dir": str(run_dir),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": round(time.time() - started, 1),
            "baseline_system": "FLAML AutoML ({})".format(
                "single pooled model" if cfg.strategy == "pooled" else "direct multi-horizon"),
            "library_versions": _versions(),
        },
        "config": cfg.to_dict(),
        "task": {
            "type": "time_series_forecast",
            "goal": cfg.goal,
            "target": target,
            "target_inference": target_candidates,
            "time_column": time_col,
            "freq": freq,
            "seasonal_period": period,
            "series_start": str(frame.index.min().date()),
            "series_end": str(frame.index.max().date()),
            "n_periods": n_periods,
            "n_observed": int(y.notna().sum()),
            "missing_periods": int(y.isna().sum()),
            "data_quality": data_quality,
        },
        "ingest": ingest_report.to_dict(),
        "protocol": {
            "strategy": (
                "pooled (ONE model, horizon as an input feature)" if cfg.strategy == "pooled"
                else "direct (one model per horizon)"),
            "models_fitted": 1 if cfg.strategy == "pooled" else len(
                [e for e in per_horizon if e["status"] == "ok"]),
            "horizons_trained": (pooled or {}).get("horizons_trained"),
            "split": "single time cutoff on the target timestamp",
            "cutoff": str(cutoff.date()),
            "train_periods": int(len(train_y)),
            "test_periods": int(n_periods - len(train_y)),
            "test_window_reason": split_reason,
            "train_window_periods": cfg.train_window or int(len(train_y)),
            "train_window_capped": bool(cfg.train_window),
            "validation_inside_search": "forward-chaining time-series CV (FLAML split_type='time')",
            "mase_scale": scale,
            "refit_during_test": False,
        },
        "summary": summary,
        "per_horizon": per_horizon,
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    from . import report as report_mod
    plots = report_mod.make_plots(result, predictions, frame, target, run_dir)
    result["artifact_paths"] = {
        "metrics_json": str(run_dir / "metrics.json"),
        "predictions_csv": str(run_dir / "predictions.csv"),
        "plots": plots,
    }
    report_mod.write_reports(result, run_dir)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    if cfg.mlflow:
        _log_mlflow(cfg, result)

    log("\n[done] {:.1f}s -> {}".format(result["run"]["wall_seconds"], run_dir))
    return result


def _versions() -> dict:
    out = {}
    for mod in ("flaml", "sklearn", "lightgbm", "xgboost", "pandas", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def _log_mlflow(cfg: RunConfig, result: dict) -> None:
    try:
        import mlflow
        mlflow.set_experiment(cfg.mlflow_experiment)
        with mlflow.start_run(run_name=Path(result["run"]["run_dir"]).name):
            mlflow.log_params({
                "goal": cfg.goal[:250],
                "target": result["task"]["target"],
                "freq": result["task"]["freq"],
                "horizons": ",".join(str(h) for h in cfg.horizons),
                "time_budget_s": cfg.time_budget,
                "search_metric": cfg.metric,
                "seed": cfg.seed,
                "engine": "flaml",
            })
            for key, value in result["summary"].items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    mlflow.log_metric("summary_" + key, float(value))
            for entry in result["per_horizon"]:
                if entry["status"] != "ok":
                    continue
                h = entry["horizon"]
                for name, value in entry["automl"].items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        mlflow.log_metric("h{}_{}".format(h, name), float(value))
                mlflow.log_metric("h{}_skill_vs_snaive".format(h), float(entry["skill_vs_seasonal_naive"]))
            mlflow.log_artifacts(result["run"]["run_dir"])
    except Exception as exc:  # tracking must never break a run
        print("[mlflow] logging skipped: {}".format(exc))
