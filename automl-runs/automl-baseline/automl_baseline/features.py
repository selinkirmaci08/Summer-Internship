"""Reframe a time series as a supervised learning problem.

Both supported strategies map information available at origin t to the value at
t+h, and neither feeds predictions back into itself, so nothing compounds the way
a recursive one-step model applied 90 times would:

  * **pooled** (default, `build_pooled`) -- every (origin, horizon) pair is stacked
    into ONE matrix with `horizon` as an input column, so a single fitted model
    answers any horizon it was trained across. This is the "give it a time and it
    predicts" model.
  * **direct** (`build`) -- one matrix, and one model, per horizon. More models, but
    each horizon gets hyperparameters tuned for it alone.

Leakage rule enforced here: every feature at origin t is computed from data at
times <= t, with the single exception of calendar features (and any column the
caller explicitly declares known-in-advance), which are evaluated at the target
timestamp t+h because they are genuinely knowable at forecast time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAGS = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 56, 91, 182, 364)
DEFAULT_WINDOWS = (7, 14, 28, 91, 182)
EXOG_LAGS = (0, 7)
EXOG_WINDOWS = (7, 28)


def infer_freq(index: pd.DatetimeIndex) -> str:
    """Best-effort frequency inference; falls back to the median observed gap."""
    if len(index) < 3:
        return "D"
    guess = pd.infer_freq(index)
    if guess:
        return guess
    delta = pd.Series(index).diff().dropna().median()
    if pd.isna(delta):
        return "D"
    days = delta / pd.Timedelta(days=1)
    if days <= 1.5:
        return "D"
    if days <= 8:
        return "W"
    if days <= 45:
        return "MS"
    if days <= 120:
        return "QS"
    return "YS"


def seasonal_period(freq: str) -> int:
    """Dominant seasonal cycle length, used for the MASE denominator and naive baselines."""
    f = freq.upper()
    if f.startswith("D") or f.startswith("B"):
        return 7
    if f.startswith("W"):
        return 52
    if f.startswith("M"):
        return 12
    if f.startswith("Q"):
        return 4
    if f.startswith("H"):
        return 24
    return 1


def prepare_series(df: pd.DataFrame, time_col: str, target: str, freq: str | None = None):
    """Sort by time, drop unusable rows, and reindex onto a regular grid.

    Returns (frame indexed by timestamp, freq string). Gaps in the calendar become
    all-NaN rows so that a "lag 7" really means seven periods back, not seven
    recorded observations back.
    """
    if target not in df.columns:
        raise KeyError("target column '{}' not in data (columns: {})".format(target, list(df.columns)))
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col)
    out = out.set_index(time_col)
    out = out[~out.index.duplicated(keep="first")]
    out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out.loc[out[target].first_valid_index():out[target].last_valid_index()]

    freq = freq or infer_freq(out.index)
    try:
        full = pd.date_range(out.index.min(), out.index.max(), freq=freq)
        out = out.reindex(full)
    except ValueError:
        freq = "D"
        full = pd.date_range(out.index.min(), out.index.max(), freq=freq)
        out = out.reindex(full)
    out.index.name = time_col
    return out, freq


def calendar_features(index: pd.DatetimeIndex, prefix: str = "cal_") -> pd.DataFrame:
    """Deterministic date features -- always known in advance, so safe at t+h."""
    idx = pd.DatetimeIndex(index)
    feats = pd.DataFrame(index=idx)
    feats[prefix + "dayofweek"] = idx.dayofweek
    feats[prefix + "is_weekend"] = (idx.dayofweek >= 5).astype(int)
    feats[prefix + "day"] = idx.day
    feats[prefix + "month"] = idx.month
    feats[prefix + "quarter"] = idx.quarter
    feats[prefix + "year"] = idx.year
    feats[prefix + "weekofyear"] = idx.isocalendar().week.to_numpy(dtype=int)
    feats[prefix + "dayofyear"] = idx.dayofyear
    feats[prefix + "is_month_start"] = idx.is_month_start.astype(int)
    feats[prefix + "is_month_end"] = idx.is_month_end.astype(int)
    # Fourier terms give the tree models a smooth handle on weekly/yearly cycles.
    for k in (1, 2):
        feats[prefix + "sin_week_{}".format(k)] = np.sin(2 * np.pi * k * idx.dayofweek / 7)
        feats[prefix + "cos_week_{}".format(k)] = np.cos(2 * np.pi * k * idx.dayofweek / 7)
        feats[prefix + "sin_year_{}".format(k)] = np.sin(2 * np.pi * k * idx.dayofyear / 365.25)
        feats[prefix + "cos_year_{}".format(k)] = np.cos(2 * np.pi * k * idx.dayofyear / 365.25)
    return feats


def _target_features(y: pd.Series, lags, windows, max_lag: int) -> pd.DataFrame:
    feats = pd.DataFrame(index=y.index)
    for lag in lags:
        if lag <= max_lag:
            feats["lag_{}".format(lag)] = y.shift(lag)
    for w in windows:
        if w > max_lag:
            continue
        base = y.shift(1).rolling(w, min_periods=max(2, w // 3))
        feats["roll_mean_{}".format(w)] = base.mean()
        feats["roll_std_{}".format(w)] = base.std()
        feats["roll_min_{}".format(w)] = base.min()
        feats["roll_max_{}".format(w)] = base.max()
    if 1 <= max_lag:
        feats["diff_1"] = y.shift(1) - y.shift(2)
    if 7 <= max_lag:
        feats["diff_7"] = y.shift(1) - y.shift(8)
        feats["ratio_lag1_roll7"] = y.shift(1) / feats.get("roll_mean_7", pd.Series(np.nan, index=y.index))
    feats["expanding_mean"] = y.shift(1).expanding(min_periods=5).mean()
    return feats


def _exog_features(df: pd.DataFrame, target: str, known_future) -> tuple:
    """Past-only features from the non-target columns, plus the categorical column list."""
    feats = pd.DataFrame(index=df.index)
    categoricals = []
    for col in df.columns:
        if col == target or col in known_future:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            if series.notna().sum() < 10 or series.nunique(dropna=True) <= 1:
                continue
            for lag in EXOG_LAGS:
                feats["exog_{}_lag{}".format(col, lag)] = series.shift(lag)
            for w in EXOG_WINDOWS:
                feats["exog_{}_rollmean{}".format(col, w)] = series.shift(0).rolling(
                    w, min_periods=max(2, w // 3)
                ).mean()
        else:
            nun = series.nunique(dropna=True)
            if nun <= 1 or nun > 50:
                continue
            name = "exog_{}_lag0".format(col)
            feats[name] = series.astype("category")
            categoricals.append(name)
    return feats, categoricals


class HorizonDataset:
    """Builds (X, y) pairs per horizon from one prepared series."""

    def __init__(
        self,
        frame: pd.DataFrame,
        target: str,
        freq: str,
        known_future=(),
        lags=DEFAULT_LAGS,
        windows=DEFAULT_WINDOWS,
    ):
        self.frame = frame
        self.target = target
        self.freq = freq
        self.known_future = list(known_future)
        n = len(frame)
        max_lag = max(4, n // 3)  # never ask for a lag longer than a third of history
        self.y = frame[target]
        past = _target_features(self.y, lags, windows, max_lag)
        exog, categoricals = _exog_features(frame, target, self.known_future)
        self.past = pd.concat([past, exog], axis=1)
        self.categoricals = categoricals
        self.cal = calendar_features(frame.index)

    def build_features_at(self, horizon: int) -> pd.DataFrame:
        """Feature matrix for every origin, with no target-based filtering.

        This is the inference view: at the newest origin the value at t+h has not
        happened yet, so `build` would drop exactly the row you want to predict from.
        """
        cal_future = self.cal.shift(-horizon)  # calendar of t+h, aligned back onto origin t
        if len(cal_future):
            # shift(-h) leaves the tail empty; the calendar of a future date is known,
            # so fill it in rather than losing the most recent origins.
            future_index = self.frame.index + horizon * pd.tseries.frequencies.to_offset(self.freq)
            cal_future = calendar_features(future_index)
            cal_future.index = self.frame.index
        parts = [self.past, cal_future]
        for col in self.known_future:
            if col in self.frame.columns:
                series = self.frame[col].shift(-horizon)
                if pd.api.types.is_numeric_dtype(series):
                    parts.append(series.rename("known_{}_at_target".format(col)))
                else:
                    parts.append(series.astype("category").rename("known_{}_at_target".format(col)))
        X = pd.concat(parts, axis=1)
        return X.loc[:, ~X.columns.duplicated()]

    def build(self, horizon: int):
        """Feature matrix and target for `horizon` steps ahead.

        Row index is the *origin* t; `target_time` records t+h. Rows are kept only
        where the true value at t+h exists.
        """
        y_future = self.y.shift(-horizon)
        X = self.build_features_at(horizon)
        mask = y_future.notna()
        # Drop origins with no usable history at all (start of the series).
        usable = X.notna().sum(axis=1) > 0
        mask = mask & usable
        X, y = X.loc[mask], y_future.loc[mask]
        offset = horizon * pd.tseries.frequencies.to_offset(self.freq)
        target_time = pd.Series(X.index, index=X.index) + offset
        return X, y, target_time

    def build_pooled(self, horizons, max_rows: int | None = None, must_keep=()):
        """Stack every (origin, horizon) pair into ONE training matrix.

        `horizon` becomes an ordinary input feature, so a single fitted model answers
        "what will the value be h steps from t" for any h it was trained across --
        you hand it a time, it predicts. Rows are returned in target-time order so a
        forward-chaining CV split still means what it says.

        Returns (X, y, origin_time, target_time, horizon) with a clean RangeIndex,
        because one origin now appears once per horizon.
        """
        horizons = sorted({int(h) for h in horizons if int(h) > 0})
        must_keep = sorted({int(h) for h in must_keep if int(h) in horizons})
        if max_rows:
            # Trim the grid rather than the origins: dropping origins would thin out the
            # history, dropping horizons only widens the spacing the model interpolates
            # over. The scored horizons are never dropped -- a model asked for t+90 must
            # have been trained on t+90, not left to extrapolate past the grid's edge.
            per_h = max(1, len(self.frame))
            budget = max(len(must_keep), 1, max_rows // per_h)
            if len(horizons) > budget:
                others = [h for h in horizons if h not in must_keep]
                slots = max(0, budget - len(must_keep))
                picked = []
                if slots and others:
                    idx = np.linspace(0, len(others) - 1, num=min(slots, len(others)))
                    picked = [others[int(round(i))] for i in idx]
                horizons = sorted(set(must_keep) | set(picked)) or must_keep or horizons[:1]

        xs, ys, origins, targets, hs = [], [], [], [], []
        for h in horizons:
            X, y, target_time = self.build(h)
            if X.empty:
                continue
            xs.append(X)
            ys.append(y)
            origins.append(pd.Series(X.index))
            targets.append(pd.Series(target_time.to_numpy()))
            hs.append(pd.Series(np.full(len(X), h, dtype=int)))

        if not xs:
            raise ValueError("no usable rows at any of the requested horizons")

        X = pd.concat(xs, ignore_index=True)
        X["horizon"] = pd.concat(hs, ignore_index=True)
        y = pd.concat(ys, ignore_index=True)
        origin_time = pd.concat(origins, ignore_index=True)
        target_time = pd.concat(targets, ignore_index=True)

        order = np.argsort(target_time.to_numpy(), kind="stable")
        return (
            X.iloc[order].reset_index(drop=True),
            y.iloc[order].reset_index(drop=True),
            origin_time.iloc[order].reset_index(drop=True),
            target_time.iloc[order].reset_index(drop=True),
            X["horizon"].iloc[order].reset_index(drop=True),
        )

    def horizon_grid(self, horizons, dense: bool = True) -> list:
        """Which horizons the single pooled model should be trained across.

        Dense (1..max) lets it answer any horizon in range, not only the five you
        happen to score on -- that is the point of a single model.
        """
        max_h = max(int(h) for h in horizons)
        if not dense:
            return sorted({int(h) for h in horizons})
        return list(range(1, max_h + 1))
