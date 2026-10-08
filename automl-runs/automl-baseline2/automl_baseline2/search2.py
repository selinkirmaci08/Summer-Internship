"""AutoML search, with the agentic run's own validation block as the holdout.

The v1 wrapper (`automl_baseline.search.fit_regressor`) picks hyperparameters on a
forward-chaining split it cuts itself. That is right when the baseline owns the
protocol. Here it is not: the agentic run tuned against a specific validation
window, and if the baseline tunes on a different one the comparison stops being
about modelling and starts being about who got the friendlier tuning set.

So this wrapper hands FLAML the run's validation partition explicitly. The matrix
preparation is imported from v1 rather than rewritten -- it is the same dtype
contract, and two copies of it would drift.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from automl_baseline.search import _prepare_matrix  # noqa: F401  (deliberate reuse)


def fit_with_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    time_budget: int = 300,
    metric: str = "mae",
    seed: int = 42,
    estimators=(),
    log_file: str | None = None,
) -> dict:
    """Search hyperparameters against an explicit, caller-supplied validation set.

    Returns the fitted model and what the search chose. The model is the one FLAML
    refits on the training rows -- the validation block is used for selection only,
    never for fitting, so the test split stays the only untouched partition.
    """
    from flaml import AutoML

    X_train = _prepare_matrix(X_train)
    X_val = _prepare_matrix(X_val.reindex(columns=X_train.columns))
    # Categorical levels must agree across the two matrices or FLAML sees a
    # different feature space at validation time than the one it fitted on.
    for col in X_train.columns:
        if isinstance(X_train[col].dtype, pd.CategoricalDtype):
            levels = X_train[col].cat.categories.union(X_val[col].cat.categories)
            X_train[col] = X_train[col].cat.set_categories(levels)
            X_val[col] = X_val[col].cat.set_categories(levels)

    y_train = pd.to_numeric(y_train, errors="coerce").astype(float)
    y_val = pd.to_numeric(y_val, errors="coerce").astype(float)
    ok_tr = y_train.notna().to_numpy()
    ok_va = y_val.notna().to_numpy()
    X_train, y_train = X_train.loc[ok_tr], y_train.loc[ok_tr]
    X_val, y_val = X_val.loc[ok_va], y_val.loc[ok_va]

    settings = {
        "time_budget": int(time_budget),
        "metric": metric,
        "task": "regression",
        "eval_method": "holdout",
        "retrain_full": False,   # keep the model that was actually selected
        "seed": int(seed),
        "verbose": 0,
        "early_stop": True,
    }
    if estimators:
        settings["estimator_list"] = list(estimators)
    if log_file:
        settings["log_file_name"] = str(log_file)

    automl = AutoML()
    flaml_logger = logging.getLogger("flaml")
    prior = flaml_logger.level
    flaml_logger.setLevel(logging.ERROR)
    started = time.time()
    try:
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            **settings,
        )
    finally:
        flaml_logger.setLevel(prior)

    best = automl.best_config or {}
    return {
        "model": automl,
        "estimator": str(automl.best_estimator),
        "config": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                   for k, v in best.items()},
        "validation_loss": float(automl.best_loss),
        "metric": metric,
        "search_seconds": round(time.time() - started, 1),
        "budget_seconds": int(time_budget),
        "n_train_rows": int(len(y_train)),
        "n_val_rows": int(len(y_val)),
        "n_features": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
        "estimator_list": list(estimators) if estimators else "flaml default",
        "selection": "explicit holdout on the agentic run's validation partition",
    }


def predict(fitted: dict, X: pd.DataFrame) -> np.ndarray:
    """Predict with a model returned by `fit_with_holdout`, aligned to its columns."""
    cols = fitted["feature_names"]
    X = _prepare_matrix(X.reindex(columns=cols))
    return np.asarray(fitted["model"].predict(X), dtype=float)
