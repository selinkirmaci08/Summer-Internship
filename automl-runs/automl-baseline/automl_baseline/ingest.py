"""Multi-source ingestion.

A dataset path may be a single file, or a directory tree holding several files
from different sources (different years, different systems, enrichment tables).
This module discovers every tabular file under the path, clusters them by schema
similarity, stacks the files inside a cluster, and joins the clusters on the
time key -- coalescing overlapping columns so a partially-populated row from one
source is completed by another.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".xlsx", ".xls", ".json", ".jsonl"}
SOURCE_COL = "__source_file"

# Column names that look like a time index, in preference order.
TIME_NAME_HINTS = (
    "date", "datetime", "timestamp", "time", "ds", "day", "period", "month", "week",
)


@dataclass
class IngestReport:
    files: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    clusters: list = field(default_factory=list)
    time_column: str | None = None
    panel_candidates: list = field(default_factory=list)
    duplicate_keys_collapsed: int = 0
    rows: int = 0
    columns: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "skipped": self.skipped,
            "clusters": self.clusters,
            "time_column": self.time_column,
            "panel_candidates": self.panel_candidates,
            "duplicate_keys_collapsed": self.duplicate_keys_collapsed,
            "rows": self.rows,
            "columns": self.columns,
            "notes": self.notes,
        }


def discover_files(path) -> list:
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError("dataset path does not exist: {}".format(p))
    files = [
        f for f in sorted(p.rglob("*"))
        if f.is_file() and f.suffix.lower() in TABULAR_SUFFIXES and not f.name.startswith("~$")
    ]
    if not files:
        raise FileNotFoundError("no tabular files found under {}".format(p))
    return files


def load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:  # pragma: no cover - guarded by discover_files
        raise ValueError("unsupported file type: {}".format(path))
    df.columns = [str(c).strip() for c in df.columns]
    df[SOURCE_COL] = path.name
    return df


def detect_time_column(df: pd.DataFrame, hint: str | None = None) -> str | None:
    """Find the column that indexes the series: name hint first, then parseability."""
    if hint:
        if hint not in df.columns:
            raise KeyError("time column '{}' not in data (columns: {})".format(hint, list(df.columns)))
        return hint
    lowered = {str(c).lower(): c for c in df.columns}
    for name in TIME_NAME_HINTS:
        if name in lowered:
            return lowered[name]
    for col in df.columns:
        if col == SOURCE_COL:
            continue
        series = df[col].dropna().head(200)
        if series.empty or pd.api.types.is_numeric_dtype(series):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return col
    return None


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _cluster_frames(frames: list, threshold: float) -> list:
    """Group frames whose column sets overlap by at least `threshold` (single-link)."""
    sigs = [set(df.columns) - {SOURCE_COL} for _, df in frames]
    parent = list(range(len(frames)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if _jaccard(sigs[i], sigs[j]) >= threshold:
                parent[find(i)] = find(j)

    groups: dict = {}
    for i in range(len(frames)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


class PanelDataError(ValueError):
    """Raised when the data holds several series and the caller has not said which to use."""


def detect_panel_keys(df: pd.DataFrame, time_col: str, max_cardinality: int = 5000) -> list:
    """Find columns that split the rows into several series sharing the same timestamps.

    A repeated timestamp on its own means nothing -- the same day can arrive from four
    source files. It is a panel only if some column actually *varies* within a
    timestamp and largely resolves the duplication, e.g. store_id, sku, region.
    """
    if time_col not in df.columns:
        return []
    times = df[time_col]
    if not times.duplicated().any():
        return []

    candidates = []
    n = len(df)
    for col in df.columns:
        if col in (time_col, SOURCE_COL):
            continue
        series = df[col]
        nun = series.nunique(dropna=True)
        if nun < 2 or nun > max_cardinality or nun / max(1, n) > 0.5:
            continue
        per_time = df.groupby(times, observed=True)[col].nunique(dropna=True)
        # Does it vary inside a single timestamp?
        if per_time.max() < 2:
            continue
        # The decisive test: a key's values RECUR at every timestamp (store A appears on
        # every date), whereas a measurement takes fresh values each time. store_id has
        # 3 values per date out of 3 overall -> 1.0; sales has 3 per date out of 1013 -> 0.003.
        recurrence = float(per_time.median()) / nun
        if recurrence < 0.5:
            continue
        # Does adding it largely remove the duplication?
        dup_with = float(df.duplicated(subset=[time_col, col]).sum()) / max(1, n)
        if dup_with > 0.2:
            continue
        candidates.append({
            "column": col,
            "n_series": int(nun),
            "recurrence": round(recurrence, 4),
            "residual_duplicate_rate": round(dup_with, 4),
            "is_identifier_like": bool(not pd.api.types.is_float_dtype(series)),
            "values_preview": [str(v) for v in list(pd.unique(series.dropna()))[:10]],
        })
    candidates.sort(key=lambda c: (-c["recurrence"], -c["is_identifier_like"],
                                   c["residual_duplicate_rate"], c["n_series"]))
    return candidates


def _apply_series_selection(df, time_col, series_id, series, aggregate, report):
    """Reduce a panel to one series, either by picking one entity or by rolling up."""
    values = [str(v) for v in pd.unique(df[series_id].dropna())]
    if series is not None:
        keep = df[df[series_id].astype(str) == str(series)]
        if keep.empty:
            raise PanelDataError(
                "no rows with {}={!r}. Available: {}".format(series_id, series, values[:50]))
        report.notes.append(
            "panel column '{}': kept series {!r} only ({} of {} series, {} rows)".format(
                series_id, series, 1, len(values), len(keep)))
        return keep.drop(columns=[series_id])
    if aggregate:
        if aggregate not in ("sum", "mean"):
            raise ValueError("--aggregate must be 'sum' or 'mean', got {!r}".format(aggregate))
        numeric = [c for c in df.columns
                   if c not in (time_col, series_id) and pd.api.types.is_numeric_dtype(df[c])]
        dropped = [c for c in df.columns if c not in numeric + [time_col, series_id]]
        rolled = getattr(df.groupby(time_col, as_index=False)[numeric], aggregate)()
        report.notes.append(
            "panel column '{}': {} series rolled up with {} across {} numeric columns{}".format(
                series_id, len(values), aggregate, len(numeric),
                "; dropped non-numeric {}".format(dropped) if dropped else ""))
        return rolled
    raise PanelDataError(
        "'{}' splits the data into {} series ({}...). Pass --series to model one of them, "
        "or --aggregate sum|mean to roll them up.".format(series_id, len(values), values[:5]))


def _coalesce_duplicates(df: pd.DataFrame, key: str):
    """Collapse rows sharing a time key, taking the first non-null value per column.

    Rows are ordered most-complete-first so the richest source wins a conflict.
    """
    dupes = int(df.duplicated(subset=[key]).sum())
    if dupes == 0:
        return df, 0
    order = df.notna().sum(axis=1).rank(ascending=False, method="first")
    df = df.assign(__completeness=order).sort_values([key, "__completeness"])
    # GroupBy.first() takes the first non-null value per column, i.e. a coalesce.
    collapsed = df.groupby(key, as_index=False, sort=True).first()
    return collapsed.drop(columns="__completeness", errors="ignore"), dupes


def load_dataset(
    path,
    time_col_hint: str | None = None,
    cluster_threshold: float = 0.5,
    series_id: str | None = None,
    series: str | None = None,
    aggregate: str | None = None,
):
    """Load and merge every tabular file under `path` into a single DataFrame."""
    report = IngestReport()
    files = discover_files(path)

    frames = []
    for f in files:
        try:
            df = load_file(f)
        except Exception as exc:  # a broken file should not sink the run
            report.skipped.append({"file": str(f), "reason": "{}: {}".format(type(exc).__name__, exc)})
            continue
        if df.empty:
            report.skipped.append({"file": str(f), "reason": "empty file"})
            continue
        frames.append((f, df))
        report.files.append(str(f))

    if not frames:
        raise ValueError("no readable tabular files under {}".format(path))

    time_col = None
    for _, df in frames:
        time_col = detect_time_column(df, time_col_hint)
        if time_col:
            break
    report.time_column = time_col

    groups = _cluster_frames(frames, cluster_threshold)
    stacked = []
    for gi, idxs in enumerate(groups):
        parts = [frames[i][1] for i in idxs]
        merged = pd.concat(parts, ignore_index=True, sort=False)
        report.clusters.append({
            "cluster": gi,
            "files": [frames[i][0].name for i in idxs],
            "rows": int(len(merged)),
            "columns": int(merged.shape[1] - 1),
            "has_time_column": bool(time_col and time_col in merged.columns),
        })
        stacked.append(merged)

    if time_col is None:
        # Non-temporal data: keep the widest/longest cluster, drop the rest.
        out = max(stacked, key=lambda d: (d.shape[0] * d.shape[1]))
        if len(stacked) > 1:
            report.notes.append(
                "no time column found; kept the largest of {} schema clusters and dropped the "
                "others (no key to join them on)".format(len(stacked))
            )
    else:
        temporal = [d for d in stacked if time_col in d.columns]
        dropped = len(stacked) - len(temporal)
        if dropped:
            report.notes.append(
                "dropped {} schema cluster(s) with no '{}' column - nothing to join them "
                "on".format(dropped, time_col)
            )
        out = None
        total_dupes = 0
        for part in sorted(temporal, key=lambda d: -d.shape[0]):
            part = part.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                part[time_col] = pd.to_datetime(part[time_col], errors="coerce")
            part = part.dropna(subset=[time_col])

            # Before collapsing duplicate timestamps, check they are repeats of one
            # series rather than several entities. Collapsing a panel silently keeps
            # one entity and throws the rest away.
            panels = detect_panel_keys(part, time_col)
            if panels:
                report.panel_candidates = panels
                key = series_id or panels[0]["column"]
                if series_id and series_id not in part.columns:
                    raise KeyError("--series-id '{}' not in data (columns: {})".format(
                        series_id, list(part.columns)))
                if series_id is None and (series is None and aggregate is None):
                    raise PanelDataError(
                        "this dataset holds several series sharing the same timestamps: '{}' has "
                        "{} distinct values ({}...). Modelling it as one series would silently keep "
                        "one and discard the rest. Choose: --series-id {} --series <value> to model "
                        "one, or --series-id {} --aggregate sum to forecast the total.".format(
                            key, panels[0]["n_series"], panels[0]["values_preview"][:5], key, key))
                part = _apply_series_selection(part, time_col, key, series, aggregate, report)

            part, dupes = _coalesce_duplicates(part, time_col)
            total_dupes += dupes
            if out is None:
                out = part
                continue
            # Outer-join on the time key, then coalesce columns present in both.
            overlap = (set(out.columns) & set(part.columns)) - {time_col}
            out = out.merge(part, on=time_col, how="outer", suffixes=("", "__rhs"))
            for col in overlap:
                rhs = "{}__rhs".format(col)
                if rhs in out.columns:
                    out[col] = out[col].combine_first(out[rhs])
                    out = out.drop(columns=rhs)
        report.duplicate_keys_collapsed = total_dupes
        out = out.sort_values(time_col).reset_index(drop=True)

    if SOURCE_COL in out.columns:
        out = out.drop(columns=SOURCE_COL)
    report.rows, report.columns = int(out.shape[0]), int(out.shape[1])
    return out, report
