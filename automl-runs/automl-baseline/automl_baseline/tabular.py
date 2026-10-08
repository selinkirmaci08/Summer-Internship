"""Fallback path: plain tabular AutoML when the dataset has no usable time index.

Horizons do not apply here -- there is no t. This exists so the same
`path + goal` entry point still returns a defensible baseline number for a
classification or regression dataset (e.g. adult.csv) instead of erroring out.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RunConfig, infer_target
from .ingest import load_dataset
from .search import _prepare_matrix, _jsonable


def _infer_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def run_tabular(cfg: RunConfig, verbose: bool = True) -> dict:
    from flaml import AutoML
    from sklearn import metrics as skm

    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    started = time.time()

    raw, ingest_report = load_dataset(
        cfg.data_path, cfg.time_col,
        series_id=cfg.series_id, series=cfg.series, aggregate=cfg.aggregate,
    )
    target = cfg.target
    candidates = []
    if not target:
        target, candidates = infer_target(cfg.goal, raw.columns)
        if not target:
            raise ValueError("could not infer a target column from the goal; pass --target")
        log("[target] inferred '{}'".format(target))

    data = raw.dropna(subset=[target])
    y = data[target]
    X = _prepare_matrix(data.drop(columns=[target]))
    task = _infer_task(y)
    if task == "classification":
        y = y.astype(str)

    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=strat
    )

    metric = cfg.metric if cfg.metric not in ("mae",) else ("roc_auc" if task == "classification" else "mae")
    automl = AutoML()
    automl.fit(X_train=X_tr, y_train=y_tr, task=task, metric=metric,
               time_budget=cfg.time_budget, seed=cfg.seed, verbose=0)
    preds = automl.predict(X_te)

    if task == "classification":
        classes = sorted(pd.unique(y))
        scores = {
            "accuracy": float(skm.accuracy_score(y_te, preds)),
            "f1_macro": float(skm.f1_score(y_te, preds, average="macro")),
            "precision_macro": float(skm.precision_score(y_te, preds, average="macro", zero_division=0)),
            "recall_macro": float(skm.recall_score(y_te, preds, average="macro", zero_division=0)),
        }
        try:
            proba = automl.predict_proba(X_te)
            if len(classes) == 2:
                scores["roc_auc"] = float(skm.roc_auc_score((y_te == classes[1]).astype(int), proba[:, 1]))
            else:
                scores["roc_auc_ovr"] = float(skm.roc_auc_score(y_te, proba, multi_class="ovr"))
        except Exception:
            pass
        majority = y_tr.value_counts(normalize=True).max()
        scores["majority_class_accuracy"] = float(majority)
    else:
        scores = {
            "mae": float(skm.mean_absolute_error(y_te, preds)),
            "rmse": float(np.sqrt(skm.mean_squared_error(y_te, preds))),
            "r2": float(skm.r2_score(y_te, preds)),
            "mean_baseline_mae": float(skm.mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))),
        }

    run_dir = Path(cfg.output_dir) / (cfg.run_name or "{}_tabular_{}".format(
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), target))
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "run": {"run_dir": str(run_dir), "wall_seconds": round(time.time() - started, 1),
                "baseline_system": "FLAML AutoML (tabular)"},
        "config": cfg.to_dict(),
        "task": {"type": task, "goal": cfg.goal, "target": target, "target_inference": candidates,
                 "n_rows": int(len(data)), "n_features": int(X.shape[1]),
                 "note": "no time index found - horizon evaluation (t+1..t+90) does not apply"},
        "ingest": ingest_report.to_dict(),
        "protocol": {"split": "random 80/20 holdout", "seed": cfg.seed,
                     "validation_inside_search": "FLAML default CV"},
        "test_metrics": scores,
        "search": {"best_estimator": automl.best_estimator,
                   "best_config": _jsonable(automl.best_config),
                   "best_validation_loss": float(automl.best_loss),
                   "search_metric": metric},
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    log("[done] {} -> {}".format(json.dumps(scores, indent=None), run_dir))
    return result
