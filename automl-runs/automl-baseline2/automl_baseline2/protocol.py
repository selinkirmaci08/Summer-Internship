"""Read an agentic run and freeze its evaluation protocol.

Everything a fair comparison needs is already recorded inside a run directory --
which rows exist, which partition each belongs to, which horizons were scored,
which model was chosen. This module reads that out rather than restating it, so
the baseline cannot drift away from what the agentic run actually did.

The central object is the **frozen index**: the exact set of (target_date, horizon)
pairs the agentic run's test split contains. Every system in the comparison is
scored on that set and nothing else. A system that cannot produce a prediction for
a pair is reported as incomplete rather than quietly scored on a subset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# LightGBM rewrites any character outside [A-Za-z0-9_] in a feature name to '_',
# one underscore per character, and the fitted booster remembers the rewritten
# name. Reading a saved model's feature list back against the CSV it was trained
# from therefore needs the same rewrite applied to the CSV's headers.
_LGBM_SAFE = re.compile(r"[^A-Za-z0-9_]")


def lgbm_safe_columns(columns) -> list:
    return [_LGBM_SAFE.sub("_", str(c)) for c in columns]


@dataclass
class Split:
    name: str
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    n_rows: int
    n_dates: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "date_min": str(self.date_min.date()),
            "date_max": str(self.date_max.date()),
            "n_rows": self.n_rows,
            "n_dates": self.n_dates,
        }


@dataclass
class AgenticRun:
    """One agentic run, read as an evaluation protocol plus a model to beat."""

    run_dir: Path
    target: str
    time_col: str
    horizons: tuple
    splits: dict
    fe_frame: pd.DataFrame          # the agent's engineered matrix, LightGBM-safe column names
    fe_path: Path
    preprocess_frame: pd.DataFrame  # one row per date, after the agent's preprocessing
    preprocess_path: Path
    assembled_path: Path | None
    final_model_path: Path
    chosen_iteration: str
    naive_col: str
    must_drop: tuple = ()
    known_at_origin: tuple = ()
    reported_metrics: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- index

    def index(self, split: str) -> pd.DataFrame:
        """The (target_date, origin_date, horizon) triples in one partition."""
        sub = self.fe_frame.loc[self.fe_frame["_split"] == split,
                                [self.time_col, "origin_date", "h", self.target]]
        return sub.sort_values([self.time_col, "h"]).reset_index(drop=True)

    def split_dates(self, split: str) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.index(split)[self.time_col].unique()))

    def modellable_dates(self) -> pd.DatetimeIndex:
        """Every date the agentic run considered a usable trading day.

        Dates outside this set were either absent from the source export or removed
        as non-trading. A baseline must treat them identically, or it ends up
        forecasting a different series.
        """
        pp = self.preprocess_frame
        keep = pp.loc[pp["_split"].isin(("train", "validation", "test")), self.time_col]
        return pd.DatetimeIndex(sorted(pd.to_datetime(keep).unique()))

    def date_to_split(self) -> pd.Series:
        pp = self.preprocess_frame
        return pd.Series(pp["_split"].to_numpy(), index=pd.to_datetime(pp[self.time_col]))

    # ---------------------------------------------------------------- naive

    def naive_reference(self, split: str) -> np.ndarray:
        """The agentic run's own seasonal-naive column, on one partition's rows.

        `lag_sw_1` is the most recent observed same-weekday value at or before the
        origin -- a horizon-aware seasonal naive. Reusing the run's own column keeps
        the MASE denominators identical to every figure the run quotes.
        """
        sub = self.fe_frame.loc[self.fe_frame["_split"] == split].sort_values([self.time_col, "h"])
        return np.asarray(sub[self.naive_col].to_numpy(), dtype=float)

    def mase_denominators(self) -> dict:
        tr = self.fe_frame.loc[self.fe_frame["_split"] == "train"]
        te = self.fe_frame.loc[self.fe_frame["_split"] == "test"]
        return {
            "train_denominator": float(np.nanmean(np.abs(tr[self.target] - tr[self.naive_col]))),
            "test_denominator": float(np.nanmean(np.abs(te[self.target] - te[self.naive_col]))),
            "definition": "mean |y - {}|; {} is the last observed same-weekday value at "
                          "or before the origin".format(self.naive_col, self.naive_col),
        }

    # ---------------------------------------------------------------- model

    def load_final_model(self):
        import joblib
        return joblib.load(self.final_model_path)

    def predict_final(self, split: str) -> np.ndarray:
        model = self.load_final_model()
        feats = list(getattr(model, "feature_name_", []) or getattr(model, "feature_names_in_", []))
        if not feats:
            raise ValueError(
                "{} does not record its feature names, so it cannot be scored safely "
                "against the feature table".format(self.final_model_path)
            )
        missing = [f for f in feats if f not in self.fe_frame.columns]
        if missing:
            raise KeyError("model features absent from {}: {}".format(self.fe_path.name, missing))
        sub = self.fe_frame.loc[self.fe_frame["_split"] == split].sort_values([self.time_col, "h"])
        return np.asarray(model.predict(sub[feats]), dtype=float)

    def to_dict(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.run_dir.name,
            "target": self.target,
            "horizons": list(self.horizons),
            "splits": {k: v.to_dict() for k, v in self.splits.items()},
            "feature_table": str(self.fe_path.name),
            "preprocess_table": str(self.preprocess_path.name),
            "final_model": str(self.final_model_path.relative_to(self.run_dir)),
            "chosen_iteration": self.chosen_iteration,
            "seasonal_naive_column": self.naive_col,
            "mase_denominators": self.mase_denominators(),
            "reported_test_metrics": self.reported_metrics,
        }


# -------------------------------------------------------------------- loading


def _newest(paths):
    paths = sorted(paths)
    return paths[-1] if paths else None


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_agentic_run(
    run_dir,
    target: str | None = None,
    time_col: str = "date",
    naive_col: str = "lag_sw_1",
) -> AgenticRun:
    """Read a run directory into an evaluation protocol.

    Nothing here is configured by hand: the feature table, the preprocessed table,
    the chosen model and the split boundaries are all discovered from what the run
    wrote, so pointing this at a different run picks up that run's protocol instead.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError("no such run directory: {}".format(run_dir))

    state = _read_json(run_dir / "state" / "current.json")["current"]
    target = target or state.get("target_column")
    if not target:
        raise ValueError("could not determine the target column from state/current.json")

    fe_path = _newest((run_dir / "data").glob("*feature_engineering.csv"))
    pp_path = _newest((run_dir / "data").glob("*preprocess.csv"))
    if fe_path is None or pp_path is None:
        raise FileNotFoundError(
            "expected a *_feature_engineering.csv and a *_preprocess.csv under {}".format(run_dir / "data")
        )

    fe = pd.read_csv(fe_path, parse_dates=[time_col, "origin_date"], low_memory=False)
    fe.columns = lgbm_safe_columns(fe.columns)
    pp = pd.read_csv(pp_path, parse_dates=[time_col], low_memory=False)
    pp.columns = lgbm_safe_columns(pp.columns)

    for needed in ("_split", "h", "origin_date", target):
        if needed not in fe.columns:
            raise KeyError("{} has no '{}' column".format(fe_path.name, needed))
    if naive_col not in fe.columns:
        raise KeyError(
            "{} has no '{}' column to use as the seasonal-naive reference".format(fe_path.name, naive_col)
        )

    # The horizon-conditioned layout the comparison depends on: one row per
    # (target date, horizon), with the origin exactly h periods earlier. Checked
    # rather than assumed, because every downstream join relies on it.
    lag_days = (fe[time_col] - fe["origin_date"]).dt.days
    if not (lag_days == fe["h"]).all():
        raise ValueError(
            "{}: origin_date is not exactly h days before date on {} rows -- this "
            "harness assumes a horizon-conditioned direct layout".format(
                fe_path.name, int((lag_days != fe["h"]).sum())
            )
        )
    if fe.duplicated([time_col, "h"]).any():
        raise ValueError("{}: (date, h) is not unique".format(fe_path.name))

    splits = {}
    for name in ("train", "validation", "test"):
        sub = fe.loc[fe["_split"] == name]
        if sub.empty:
            continue
        splits[name] = Split(
            name=name,
            date_min=sub[time_col].min(),
            date_max=sub[time_col].max(),
            n_rows=int(len(sub)),
            n_dates=int(sub[time_col].nunique()),
        )
    if "test" not in splits:
        raise ValueError("{} contains no test rows".format(fe_path.name))

    end_summary = _newest((run_dir / "artifacts" / "end_model").glob("*end_model_summary.json"))
    chosen, model_path, reported = "unknown", None, {}
    if end_summary is not None:
        es = _read_json(end_summary)
        chosen = str(es.get("choice", {}).get("chosen", "unknown"))
        rel = es.get("final_model_path")
        if rel:
            model_path = run_dir / rel
        reported = es.get("final_test_metrics", {})
    if model_path is None or not model_path.exists():
        model_path = _newest((run_dir / "artifacts" / "training").glob("*_training_model.pkl"))
        if model_path is None:
            raise FileNotFoundError("no trained model found under {}".format(run_dir / "artifacts" / "training"))

    assembled = state.get("assembled_frame_path")
    assembled_path = (run_dir / assembled) if assembled else None
    if assembled_path is not None and not assembled_path.exists():
        assembled_path = None

    return AgenticRun(
        run_dir=run_dir,
        target=target,
        time_col=time_col,
        horizons=tuple(sorted(int(h) for h in fe["h"].unique())),
        splits=splits,
        fe_frame=fe,
        fe_path=fe_path,
        preprocess_frame=pp,
        preprocess_path=pp_path,
        assembled_path=assembled_path,
        final_model_path=model_path,
        chosen_iteration=chosen,
        naive_col=naive_col,
        must_drop=tuple(dict.fromkeys(state.get("must_drop_columns", []) or ())),
        known_at_origin=tuple(dict.fromkeys(state.get("known_at_origin_features", []) or ())),
        reported_metrics=reported,
    )
