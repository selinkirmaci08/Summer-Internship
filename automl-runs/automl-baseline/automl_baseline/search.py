"""Thin wrapper around FLAML's AutoML search.

FLAML is the AutoML engine for this baseline: it is CPU-only, installs cleanly on
Windows, and its cost-frugal search gives a defensible "what a good off-the-shelf
tool finds in N seconds" number. The wrapper exists so the rest of the code never
depends on FLAML's API directly -- swapping in AutoGluon or H2O later means
reimplementing `fit_regressor` alone.
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd


def _prepare_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """FLAML wants clean dtypes: numerics as float, everything else as category."""
    out = X.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(float)
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(float)
        elif not isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype("category")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def fit_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    time_budget: int = 60,
    metric: str = "mae",
    seed: int = 42,
    estimators=(),
    n_splits: int = 5,
    log_file: str | None = None,
) -> dict:
    """Run AutoML search and return the fitted model plus what the search chose.

    Validation inside the search is a forward-chaining time split (`split_type="time"`),
    so hyperparameter selection never sees data from after the fold it is scored on.
    """
    from flaml import AutoML

    X_train = _prepare_matrix(X_train)
    y_train = pd.to_numeric(y_train, errors="coerce").astype(float)
    ok = y_train.notna().to_numpy()
    X_train, y_train = X_train.loc[ok], y_train.loc[ok]

    n = len(y_train)
    n_splits = int(max(2, min(n_splits, max(2, n // 30))))

    settings = {
        "time_budget": int(time_budget),
        "metric": metric,
        "task": "regression",
        "eval_method": "cv",
        "n_splits": n_splits,
        "split_type": "time",
        "seed": int(seed),
        "verbose": 0,
        "early_stop": True,
    }
    if estimators:
        settings["estimator_list"] = list(estimators)
    if log_file:
        settings["log_file_name"] = log_file

    automl = AutoML()
    started = time.time()
    flaml_logger = logging.getLogger("flaml")
    prior_level = flaml_logger.level
    flaml_logger.setLevel(logging.ERROR)
    try:
        with redirect_stdout(io.StringIO()):
            automl.fit(X_train=X_train, y_train=y_train, **settings)
    finally:
        flaml_logger.setLevel(prior_level)
    elapsed = time.time() - started

    return {
        "model": automl,
        "best_estimator": automl.best_estimator,
        "best_config": _jsonable(automl.best_config),
        "best_validation_loss": float(automl.best_loss),
        "search_metric": metric,
        "search_seconds": round(elapsed, 2),
        "n_train_rows": int(n),
        "n_features": int(X_train.shape[1]),
        "cv_splits": n_splits,
        "feature_order": list(X_train.columns),
        "estimators_tried": sorted(getattr(automl, "estimator_list", []) or []),
    }


def predict(fitted: dict, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(fitted["model"].predict(_prepare_matrix(X)), dtype=float)


def feature_importance(fitted: dict, feature_names, top_n: int = 15) -> list:
    """Top features of the winning model, when the estimator exposes them."""
    model = fitted["model"]
    try:
        booster = model.model.estimator
        importances = getattr(booster, "feature_importances_", None)
        if importances is None:
            return []
        names = list(getattr(model, "feature_names_in_", feature_names))
        pairs = sorted(zip(names, np.asarray(importances, dtype=float)), key=lambda p: -p[1])
        total = sum(abs(v) for _, v in pairs) or 1.0
        return [
            {"feature": str(n), "importance": float(v), "share": round(float(abs(v) / total), 4)}
            for n, v in pairs[:top_n]
        ]
    except Exception:
        return []


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
