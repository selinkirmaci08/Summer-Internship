"""Route the Aernoudt raw drop into a training partition and hidden holdouts.

This tool SPLITS. It does not clean, aggregate, filter, deduplicate, or transform
anything. Every row and every column that goes in comes out, byte-identical, on
exactly one side of the boundary. The only judgement it makes is "which date does
this record belong to", and every rule it uses for that is declared in ROUTES
below and echoed into the split report.

Cleaning belongs downstream: the pipeline does it in `preprocess` on its own
partition, the scorer does it on the holdout, and both read the same rules out of
protocol.json so they cannot drift apart.

Usage
-----
    python tools/make_holdout.py \
        --source "C:/Users/selin/OneDrive/Desktop/alldata/aernoudt" \
        --out    "C:/Users/selin/OneDrive/Desktop/alldata/aernoudt_split"

    python tools/make_holdout.py --source ... --out ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

# --------------------------------------------------------------------------
# The split. These three windows are the whole contract; everything else in
# this file is mechanics for getting records into the right one.
# --------------------------------------------------------------------------

PARTITIONS = [
    ("train", None, date(2020, 12, 31)),
    ("holdout_primary", date(2021, 1, 1), date(2021, 12, 31)),
    ("holdout_stress", date(2022, 1, 1), date(2023, 2, 8)),
]

HOLDOUT_DIRNAME = "HOLDOUT_DO_NOT_GIVE_TO_PIPELINE"

HORIZON_GRID = [1, 7, 14, 30, 60, 90]

# The trailing group tolerates suffixes like " (2)" on filesystem-level duplicate
# copies. Such a file is still a dated record, so it routes like any other and the
# duplicate is left for preprocess to resolve.
XML_NAME = re.compile(
    r"^PT_Sales_(?P<pos>\d{6})_(?P<date>\d{8})_(?P<export>\d{14})(?P<suffix>[^.]*)\.xml$"
)


def partition_for(d: date | None) -> str | None:
    """Which partition a record dated `d` belongs to, or None if it belongs to none."""
    if d is None:
        return None
    for name, lo, hi in PARTITIONS:
        if (lo is None or d >= lo) and d <= hi:
            return name
    return None  # after the last window, or before the data starts


# --------------------------------------------------------------------------
# Date extraction, one function per source. Each returns a date or None.
# None means "this record could not be dated" -- it is withheld from every
# partition and listed in the report, never quietly dropped into train.
# --------------------------------------------------------------------------


def _parse_dmy(text: str) -> date | None:
    text = text.strip().strip('"')
    if not text:
        return None
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def date_from_xml_name(path: Path) -> date | None:
    m = XML_NAME.match(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("date"), "%Y%m%d").date()
    except ValueError:
        return None


def make_column_dater(column: str, how: str = "dmy"):
    """Row dater that reads one named column."""

    def dater(row: dict[str, str]) -> date | None:
        return _parse_dmy(row.get(column, ""))

    dater.columns = (column,)
    dater.description = f"column `{column}`"
    return dater


def make_latest_column_dater(*columns: str):
    """Row dater taking the LATEST of several date columns.

    Used for delivery headers, which carry both an order date and a delivery
    date. Taking the later of the two means a record describing anything that
    happened after the cutoff cannot land in train.
    """

    def dater(row: dict[str, str]) -> date | None:
        found = [d for d in (_parse_dmy(row.get(c, "")) for c in columns) if d]
        return max(found) if found else None

    dater.columns = columns
    dater.description = "latest of " + ", ".join(f"`{c}`" for c in columns)
    return dater


# --------------------------------------------------------------------------
# Routing table. Adding a source means adding a row here.
# --------------------------------------------------------------------------

ROUTES = [
    {
        "name": "sales_xml",
        "kind": "files",
        "glob": "salesdata/raw_xml/{stores}/**/*.xml",
        "dater": date_from_xml_name,
        "note": "one file per till-day; the date is the 8-digit field in the filename",
    },
    {
        "name": "competitor",
        "kind": "csv_rows",
        "glob": "competitordata/*.csv",
        "sep": ";",
        "dater": make_column_dater("date"),
        "note": "shop-day leave and promotion flags; semicolon-delimited, day-first dates",
    },
    {
        "name": "delivery_header",
        "kind": "csv_rows",
        "glob": "deliverydata/Header*.csv",
        "sep": "|",
        "dater": make_latest_column_dater("lbhbesdat", "lbhlevdat"),
        "note": "routed on the later of order date and delivery date",
    },
    {
        "name": "delivery_detail",
        "kind": "csv_rows_via_parent",
        "glob": "deliverydata/Detail.csv",
        "sep": "|",
        "parent": "delivery_header",
        "key": "bbhcod",
        "note": "line items inherit their header's date via bbhcod; unmatched lines are withheld",
    },
    {
        "name": "weather",
        "kind": "parquet_rows",
        "glob": "weatherdata/*.parquet",
        "column": "Date_time",
        "note": "extends past the end of sales; truncated at each boundary like every other source",
    },
    {
        "name": "product_reference",
        "kind": "reference",
        "glob": "productdata/*",
        "note": "lookup table, not observations; copied whole to every partition",
    },
    {
        "name": "store_reference",
        "kind": "reference",
        "glob": "storedata/*",
        "note": "lookup table, not observations; copied whole to every partition",
    },
]


# --------------------------------------------------------------------------

class Stats:
    def __init__(self) -> None:
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.spans: dict[str, dict[str, list[date]]] = defaultdict(lambda: defaultdict(list))
        self.notes: list[str] = []

    def record(self, source: str, part: str | None, d: date | None) -> None:
        self.counts[source][part or "UNROUTED"] += 1
        if part and d:
            span = self.spans[source][part]
            if not span:
                span.extend([d, d])
            else:
                span[0] = min(span[0], d)
                span[1] = max(span[1], d)


def out_dir(out_root: Path, part: str) -> Path:
    if part == "train":
        return out_root / "train"
    return out_root / HOLDOUT_DIRNAME / part


def _expand(src: Path, pattern: str):
    """glob, with brace alternation expanded (Path.glob has no {a,b} support)."""
    m = re.search(r"\{([^}]*)\}", pattern)
    if not m:
        return src.glob(pattern)
    out = []
    for alt in m.group(1).split(","):
        out.extend(_expand(src, pattern[: m.start()] + alt.strip() + pattern[m.end():]))
    return out


def split_files(src: Path, out_root: Path, route: dict, stats: Stats, dry: bool) -> None:
    dater = route["dater"]
    for path in sorted(_expand(src, route["glob"])):
        if not path.is_file():
            continue
        d = dater(path)
        part = partition_for(d)
        stats.record(route["name"], part, d)
        if part is None or dry:
            continue
        rel = path.relative_to(src)
        dest = out_dir(out_root, part) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def _read_rows(path: Path, sep: str):
    """Yield (header_line, raw_line, parsed_dict) preserving the original bytes."""
    with path.open("r", encoding="latin-1", newline="") as fh:
        header_line = fh.readline()
        cols = [c.strip().lstrip("\ufeff") for c in header_line.rstrip("\r\n").split(sep)]
        for raw in fh:
            if not raw.strip():
                continue
            values = raw.rstrip("\r\n").split(sep)
            yield header_line, raw, dict(zip(cols, values))


def split_csv_rows(src: Path, out_root: Path, route: dict, stats: Stats, dry: bool,
                   parent_dates: dict | None = None) -> None:
    sep = route["sep"]
    for path in sorted(src.glob(route["glob"])):
        if not path.is_file():
            continue
        buckets: dict[str, list[str]] = defaultdict(list)
        header_line = None
        for header_line, raw, row in _read_rows(path, sep):
            if parent_dates is not None:
                key = row.get(route["key"], "").strip().strip('"')
                d = parent_dates.get(key)
            else:
                d = route["dater"](row)
            part = partition_for(d)
            stats.record(route["name"], part, d)
            if part:
                buckets[part].append(raw)
        if dry or header_line is None:
            continue
        rel = path.relative_to(src)
        for part, lines in buckets.items():
            dest = out_dir(out_root, part) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w", encoding="latin-1", newline="") as fh:
                fh.write(header_line)
                fh.writelines(lines)


def collect_parent_dates(src: Path, route: dict, parent_route: dict) -> dict[str, date]:
    """Map each parent key to its date, so child rows can inherit it."""
    out: dict[str, date] = {}
    for path in sorted(src.glob(parent_route["glob"])):
        if not path.is_file():
            continue
        for _, _, row in _read_rows(path, parent_route["sep"]):
            key = row.get(route["key"], "").strip().strip('"')
            d = parent_route["dater"](row)
            if key and d:
                out[key] = max(d, out[key]) if key in out else d
    return out


def split_parquet_rows(src: Path, out_root: Path, route: dict, stats: Stats, dry: bool) -> None:
    import pandas as pd

    col = route["column"]
    for path in sorted(src.glob(route["glob"])):
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        dates = pd.to_datetime(df[col], errors="coerce").dt.date
        parts = dates.map(partition_for)
        for d, p in zip(dates, parts):
            stats.record(route["name"], p, d if isinstance(d, date) else None)
        if dry:
            continue
        rel = path.relative_to(src)
        for part in parts.dropna().unique():
            dest = out_dir(out_root, part) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.loc[parts == part].to_parquet(dest, index=False)


def copy_reference(src: Path, out_root: Path, route: dict, stats: Stats, dry: bool) -> None:
    for path in sorted(src.glob(route["glob"])):
        if not path.is_file():
            continue
        stats.record(route["name"], "reference", None)
        if dry:
            continue
        rel = path.relative_to(src)
        for part, _, _ in PARTITIONS:
            dest = out_dir(out_root, part) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def write_protocol(out_root: Path, source: Path) -> None:
    train_end = PARTITIONS[0][2]
    protocol = {
        "generated_by": "tools/make_holdout.py",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": str(source),
        "what_this_is": (
            "The scoring contract shared by the pipeline and whatever scores it afterwards. "
            "It carries no holdout data - only the shape the predictions must come back in, "
            "and the rules both sides must apply identically so their numbers are comparable."
        ),
        "target_column": "bread_quantity",
        "train_partition": {
            "path": "train/",
            "covers": f"through {train_end.isoformat()}",
            "note": (
                "This is what the Planner is given. It contains no holdout rows. The Planner "
                "carves its own train/validation split out of this and nothing it reports is "
                "a held-out number."
            ),
        },
        "holdouts": [
            {
                "name": name,
                "path": f"{HOLDOUT_DIRNAME}/{name}/",
                "from": lo.isoformat(),
                "to": hi.isoformat(),
            }
            for name, lo, hi in PARTITIONS[1:]
        ],
        "horizon_grid": HORIZON_GRID,
        "prediction_contract": {
            "key": ["target_date", "h"],
            "pred_column": "y_pred",
            "format": "csv",
            "note": (
                "One row per (target_date, h). The origin is target_date minus h days; origins "
                "before the holdout start legitimately fall inside the training partition, since "
                "the past is known at forecast time."
            ),
        },
        "scoring_rules": {
            "why_these_are_here": (
                "The split tool deliberately did NOT apply these. Both the pipeline and the "
                "scorer apply them to their own side, from this file, so the two sides cannot "
                "drift apart."
            ),
            "trading_day": (
                "A date is scored only if the shop traded. Days the shop was shut are excluded "
                "from every system's metrics. Declare the exact rule used before scoring."
            ),
            "second_till": (
                "A second till (POS ending 301) appears from 2021 and sells essentially no "
                "bread. Its rows sit on the same dates as the real ones. Decide explicitly "
                "whether the target is the retail till alone or the sum, and apply the same "
                "choice on both sides."
            ),
            "duplicate_exports": (
                "95 dates carry the same till exported more than once, usually one partial and "
                "one complete. Pick one rule (newest export, or largest) and state it."
            ),
            "known_data_defects": {
                "2014": "64 of ~310 expected trading days present; sits inside the training partition",
                "2022": "92 of ~310 expected trading days present; sits inside holdout_stress and is why it is reported separately",
                "regime_change_2020_06_01": "shop went 7/7; Mondays are closed before this date and open after",
                "system_migration_mid_2014": "till code and XML schema both change; levels are not directly comparable across it",
            },
        },
        "naive_baseline": {
            "definition": "seasonal naive - the last observed same-weekday value at the origin",
            "why": "lag-7 correlation dominates lag-1 in this series, so a last-value naive is trivially beatable and would flatter any model that only learned the weekday shape",
        },
    }
    (out_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_report(out_root: Path, source: Path, stats: Stats, dry: bool) -> str:
    lines = [
        "# Holdout split report",
        "",
        f"Source: `{source}`  ",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}"
        + ("  \n**DRY RUN — nothing was written.**" if dry else ""),
        "",
        "This tool routes records by date and changes nothing else. No cleaning, no",
        "aggregation, no deduplication, no filtering. Every value that went in came out",
        "unchanged on exactly one side of a boundary.",
        "",
        "## Partitions",
        "",
        "| partition | from | to | given to |",
        "|---|---|---|---|",
    ]
    for name, lo, hi in PARTITIONS:
        who = "the Planner" if name == "train" else "withheld"
        lines.append(f"| `{name}` | {lo.isoformat() if lo else 'start of data'} | {hi.isoformat()} | {who} |")

    lines += ["", "## Records routed", "", "| source | " + " | ".join(p for p, _, _ in PARTITIONS)
              + " | unrouted | rule |", "|---|---|---|---|---|---|"]
    for route in ROUTES:
        n = route["name"]
        c = stats.counts.get(n)
        if not c:
            continue
        if route["kind"] == "reference":
            cells = ["copied whole"] * len(PARTITIONS) + ["--"]
        else:
            cells = [f"{c.get(p, 0):,}" for p, _, _ in PARTITIONS] + [f"{c.get('UNROUTED', 0):,}"]
        lines.append(f"| `{n}` | " + " | ".join(cells) + f" | {route['note']} |")

    lines += ["", "## Date span actually landed in each partition", "",
              "| source | partition | first | last |", "|---|---|---|---|"]
    for route in ROUTES:
        for part, span in sorted(stats.spans.get(route["name"], {}).items()):
            lines.append(f"| `{route['name']}` | `{part}` | {span[0]} | {span[1]} |")

    unrouted = {n: c.get("UNROUTED", 0) for n, c in stats.counts.items() if c.get("UNROUTED", 0)}
    lines += ["", "## Unrouted records", ""]
    if unrouted:
        lines += ["Records that carry no usable date, or fall outside every window. They are",
                  "withheld from all three partitions rather than defaulted into training —",
                  "a record of unknown date is a leakage risk, not a training row.", ""]
        lines += [f"- `{n}`: {v:,}" for n, v in sorted(unrouted.items())]
    else:
        lines.append("None. Every record carried a usable date and fell inside a window.")

    lines += [
        "",
        "## What was deliberately not done",
        "",
        "- The second till (POS ending 301) is still present. It sells no bread and sits on",
        "  the same dates as the real till; deciding what to do about it is preprocessing.",
        "- Duplicate exports are still present, both copies.",
        "- Non-trading days are still present.",
        "- The one negative bread quantity is still present.",
        "- 2014 and 2022 are still present in full, thin as they are.",
        "",
        "Every one of these is a modelling decision, and `protocol.json` carries the rules so",
        "the pipeline and the scorer apply them identically to their own side.",
        "",
        "## One file is re-serialised",
        "",
        "The weather parquet is the only source not copied byte-for-byte: routing rows out of",
        "a columnar file means rewriting the container. Values and dtypes are unchanged; the",
        "file's internal layout is not.",
        "",
        "## Before pointing the pipeline at this",
        "",
        f"Move `{HOLDOUT_DIRNAME}/` somewhere the pipeline is never given a path to. It sits",
        "next to `train/` here only because this tool wrote both; leaving them as siblings",
        "means one `..` stands between an agent and the answers.",
    ]
    text = "\n".join(lines) + "\n"
    if not dry:
        (out_root / "split_report.md").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report what would move; write nothing")
    ap.add_argument(
        "--stores", default="01",
        help="branch folders under salesdata/raw_xml to include, comma-separated, "
             "or 'all'. Default '01' - the branch named in the goal. Widening this "
             "is a modelling choice: sibling branches may carry a useful chain-level "
             "signal, at roughly 2.2 GB for all five.",
    )
    args = ap.parse_args()

    stores = "*" if args.stores.strip().lower() == "all" else "{" + args.stores.strip() + "}"
    for route in ROUTES:
        if "{stores}" in route.get("glob", ""):
            route["glob"] = route["glob"].format(stores=stores)
            route["note"] += f"; branches included: {args.stores}"

    src: Path = args.source.resolve()
    out_root: Path = args.out.resolve()
    if not src.is_dir():
        print(f"source is not a directory: {src}", file=sys.stderr)
        return 2
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    by_name = {r["name"]: r for r in ROUTES}

    for route in ROUTES:
        kind = route["kind"]
        print(f"[{route['name']}] {kind} ...", flush=True)
        if kind == "files":
            split_files(src, out_root, route, stats, args.dry_run)
        elif kind == "csv_rows":
            split_csv_rows(src, out_root, route, stats, args.dry_run)
        elif kind == "csv_rows_via_parent":
            parent = by_name[route["parent"]]
            split_csv_rows(src, out_root, route, stats, args.dry_run,
                           parent_dates=collect_parent_dates(src, route, parent))
        elif kind == "parquet_rows":
            split_parquet_rows(src, out_root, route, stats, args.dry_run)
        elif kind == "reference":
            copy_reference(src, out_root, route, stats, args.dry_run)
        else:
            raise ValueError(f"unknown route kind: {kind}")

    if not args.dry_run:
        write_protocol(out_root, src)
    print()
    print(write_report(out_root, src, stats, args.dry_run))
    if not args.dry_run:
        print(f"written to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
