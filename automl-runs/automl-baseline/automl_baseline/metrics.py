"""Forecast accuracy metrics.

Metric choice follows the project's training-skill convention
(skills/training-skill/references/evaluation_metrics.md): MASE is the headline
scale-free metric because it is scaled by a seasonal-naive baseline rather than
by the actuals, so it survives near-zero values; MAE/RMSE stay as the
scale-dependent pair; SMAPE is reported instead of plain MAPE for symmetry.
"""

from __future__ import annotations

import numpy as np


def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred))) if y_true.size else float("nan")


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if y_true.size else float("nan")


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent. Zero actuals are excluded."""
    y_true, y_pred = _clean(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent, bounded 0-200."""
    y_true, y_pred = _clean(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|err| / sum|actual|, in percent."""
    y_true, y_pred = _clean(y_true, y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def bias(y_true, y_pred) -> float:
    """Mean signed error. Positive = model over-forecasts."""
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(y_pred - y_true)) if y_true.size else float("nan")


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def seasonal_naive_scale(train_series, period: int) -> float:
    """MAE of the in-sample one-step seasonal-naive forecast (the MASE denominator).

    Hyndman's definition: scale by the training-set error of the naive method, not
    by the test set, so MASE stays comparable across models and horizons.
    """
    y = np.asarray(train_series, dtype=float)
    y = y[np.isfinite(y)]
    if y.size <= period:
        return float("nan")
    diffs = np.abs(y[period:] - y[:-period])
    scale = float(np.mean(diffs))
    return scale if scale > 0 else float("nan")


def mase(y_true, y_pred, scale: float) -> float:
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return mae(y_true, y_pred) / scale


def all_metrics(y_true, y_pred, scale: float) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "smape_pct": smape(y_true, y_pred),
        "wape_pct": wape(y_true, y_pred),
        "mase": mase(y_true, y_pred, scale),
        "r2": r2(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n": int(len(_clean(y_true, y_pred)[0])),
    }
