"""Run configuration and goal parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

DEFAULT_HORIZONS = (1, 7, 14, 30, 90)

# Words that appear in goals but never help identify a target column.
_STOPWORDS = {
    "predict", "predicting", "forecast", "forecasting", "estimate", "model", "the", "a", "an",
    "of", "for", "in", "on", "at", "to", "next", "future", "daily", "weekly", "monthly", "days",
    "day", "week", "weeks", "month", "months", "how", "many", "much", "will", "be", "sold", "per",
    "and", "with", "using", "based", "value", "values", "amount", "number", "total", "want", "i",
}


@dataclass
class RunConfig:
    data_path: str
    goal: str
    target: str | None = None
    time_col: str | None = None
    series_id: str | None = None   # column splitting the data into several series
    series: str | None = None      # which series to model
    aggregate: str | None = None   # 'sum' or 'mean' to roll all series into one
    horizons: tuple = DEFAULT_HORIZONS
    strategy: str = "pooled"       # 'pooled' = ONE model with horizon as a feature;
                                  # 'direct' = a separate model per horizon
    dense_grid: bool = True        # pooled: train across every horizon 1..max, not only the scored ones
    max_train_rows: int = 200000   # pooled: row cap before the horizon grid is thinned
    time_budget: int = 60           # seconds of AutoML search per horizon
    test_days: int | None = None    # length of the held-out window, in periods
    train_window: int | None = None  # cap training lookback to the last N periods (None = all history)
    freq: str | None = None
    known_future: tuple = ()
    seed: int = 42
    metric: str = "mae"             # what AutoML optimises during search
    output_dir: str = "outputs"
    run_name: str | None = None
    estimators: tuple = ()          # empty -> FLAML's default estimator list
    mlflow: bool = False
    mlflow_experiment: str = "automl-baseline"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["horizons"] = list(self.horizons)
        d["known_future"] = list(self.known_future)
        d["estimators"] = list(self.estimators)
        return d


def _tokens(text: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t and t not in _STOPWORDS}


def infer_target(goal: str, columns, exclude=()) -> tuple:
    """Guess the target column from the goal text by token overlap with column names.

    Returns (best_column_or_None, ranked_candidates). Exact substring matches on the
    column name win outright; otherwise the column sharing the most goal tokens wins.
    """
    goal_tokens = _tokens(goal)
    goal_lower = str(goal).lower()
    scored = []
    for col in columns:
        if col in exclude:
            continue
        col_tokens = _tokens(col)
        if not col_tokens:
            continue
        exact = 1.0 if str(col).lower().replace("_", " ") in goal_lower.replace("_", " ") else 0.0
        overlap = len(col_tokens & goal_tokens)
        coverage = overlap / len(col_tokens)
        score = exact * 10 + overlap + coverage
        if score > 0:
            scored.append({"column": col, "score": round(score, 3), "matched": sorted(col_tokens & goal_tokens)})
    scored.sort(key=lambda d: -d["score"])
    best = scored[0]["column"] if scored else None
    return best, scored[:5]
