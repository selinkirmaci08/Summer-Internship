"""Naive reference forecasts.

Every AutoML number is reported next to these. A forecast that does not beat
seasonal-naive is not a model, and the same applies to whatever the agentic
pipeline produces -- so these are the floor both systems are measured against.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def naive_last(y: pd.Series, origins: pd.Index, horizon: int) -> pd.Series:
    """Flat forecast: carry the last observed value at the origin forward."""
    filled = y.ffill()
    return filled.reindex(origins)


def seasonal_naive(y: pd.Series, origins: pd.Index, horizon: int, period: int) -> pd.Series:
    """Repeat the value from the same phase of the most recent complete cycle.

    For horizon h and period P the source point is t - (P*ceil(h/P) - h), which is
    the latest time at or before the origin sharing the target's phase.
    """
    if period <= 1:
        return naive_last(y, origins, horizon)
    back = period * math.ceil(horizon / period) - horizon
    filled = y.ffill()
    shifted = filled.shift(back)
    return shifted.reindex(origins)


def rolling_mean(y: pd.Series, origins: pd.Index, horizon: int, window: int = 28) -> pd.Series:
    """Flat forecast at the recent mean level."""
    roll = y.rolling(window, min_periods=max(2, window // 4)).mean()
    return roll.reindex(origins)


def all_baselines(y: pd.Series, origins: pd.Index, horizon: int, period: int) -> dict:
    return {
        "seasonal_naive": seasonal_naive(y, origins, horizon, period),
        "naive_last": naive_last(y, origins, horizon),
        "rolling_mean_28": rolling_mean(y, origins, horizon, 28),
    }
