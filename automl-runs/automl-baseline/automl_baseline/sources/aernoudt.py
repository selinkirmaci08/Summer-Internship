"""Aernoudt bakery: raw POS XML -> a daily table the baseline can read.

The raw drop is not something the generic ingest can read: ~12 000 XML files, one
per store-day, whose element names change mid-history, plus a product master that
has to be joined before a product group means anything. This module does that
translation and nothing else.

DESIGN RULE: this file makes no judgements about the data. Every product group in
the master becomes its own set of columns, every weather column is carried
through, nothing is dropped for being small or uninteresting, and no ratios or
flags are invented. Selecting what matters is the model's job, not the ETL's.

The four choices that cannot be derived from the data are parameters, and their
confirmed values for this project are recorded in DECISIONS below.

Mechanical rules the code applies (forced by the file formats, not chosen):
  * Two schemas: files up to ~2014 use `<order>/<orderdetail>`, later ones
    `<sale>/<saledetail>`. Both carry articlereference/quantity/total.
  * Store identity is the POS code from the filename `PT_Sales_<hq><pos>_...`,
    not the folder: folder `01` holds POS 001 and POS 301.
  * Decimal commas, ISO-8859-1 XML, latin-1 CSVs.
  * Weather rows are exact duplicates (23 of every 24), so de-duplication is
    `drop_duplicates()`, not an aggregation.
  * The competitor files have no key that links their shop codes to POS codes, so
    they are not joined. Supply one with --competitor-map to include them.

Validated against the previously derived aernoudt_bread_daily_all_years.csv: for
POS 001 on 2010-11-16 this reproduces num_orders=473, num_items_sold=1175,
total_revenue=1933.80 and group-1 quantity=395 exactly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Confirmed with the user on 2026-08-27. Nothing else in this file is a choice.
DECISIONS = {
    "target_groups": [1],          # 'Brood' (bread) only, not Pistolets or Belegde broodjes
    "target_measure": "quantity",  # units sold, not revenue
    "scope": "chain",              # sum every POS code into one daily series
    "duplicate_policy": "newest",  # a re-exported store-day: keep the latest creation stamp
}
# ---------------------------------------------------------------------------

FILENAME_RE = re.compile(r"PT_Sales_(\d{3})(\d{3})_(\d{8})_(\d{8,14})", re.I)

# Dutch product-group names -> English, for column readability only. Translation
# is not selection: every group gets columns regardless of what it is called.
GROUP_TRANSLATIONS = {
    "brood": "bread",
    "pistolets": "bread_rolls",
    "slagroom klein": "cream_cake_small",
    "slagroom groot": "cream_cake_large",
    "drooggebak klein": "dry_pastry_small",
    "drooggebak groot": "dry_pastry_large",
    "ijs en ijstaarten": "ice_cream",
    "koffiekoeken": "breakfast_pastry",
    "belegde broodjes": "filled_sandwich",
    "confiserie": "confectionery",
    "cocks": "cocks",
    "dranken": "drinks",
    "verpakking en dergelijke": "packaging",
}


def _num(value):
    """Parse a decimal-comma number."""
    if value is None or value == "":
        return np.nan
    try:
        return float(value.replace(",", "."))
    except (ValueError, AttributeError):
        return np.nan


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


# --------------------------------------------------------------------- masters

def load_product_master(root: Path):
    """Return (ProductKey -> ProductGroupKey, ProductGroupKey -> column slug)."""
    root = Path(root)
    products = pd.read_csv(root / "productdata" / "AernoudtProductData.csv",
                           sep=";", encoding="latin-1")
    # The master lists each key once per POS 'Systeem' with the same group, so the
    # duplicates collapse to one lookup.
    key_to_group = (
        products.dropna(subset=["ProductKey", "ProductGroupKey"])
        .drop_duplicates(subset=["ProductKey"])
        .set_index("ProductKey")["ProductGroupKey"].astype(int)
    )
    groups = pd.read_csv(root / "productdata" / "ProductGroupData.csv",
                         sep=";", encoding="utf-8-sig")
    names = {}
    for key, dutch in zip(groups.ProductGroupKey, groups.ProductGroup):
        english = GROUP_TRANSLATIONS.get(str(dutch).strip().lower(), _slug(dutch))
        names[int(key)] = "g{:02d}_{}".format(int(key), english)
    return key_to_group, names


# ----------------------------------------------------------------- xml parsing

def parse_sales_file(path: Path) -> tuple:
    """Return (meta, detail rows, payment rows) for one store-day file."""
    root = ET.parse(path).getroot()
    meta = {
        "date": root.get("salesdate"),
        "hq": root.get("referenceheadquarters"),
        "pos": root.get("referencepointofsales"),
        "schema": None,
        "num_orders": 0,
        "tickets_printed": 0,
        "order_total": 0.0,
        "order_paid": 0.0,
    }
    details, payments = [], []
    for item in root:
        if item.tag not in ("order", "sale"):
            continue
        meta["schema"] = item.tag
        meta["num_orders"] += 1
        if str(item.get("ticketprinted", "")).lower() == "true":
            meta["tickets_printed"] += 1
        for attr, key in (("total", "order_total"), ("paid", "order_paid")):
            value = _num(item.get(attr))
            if np.isfinite(value):
                meta[key] += value
        for child in item:
            if child.tag.endswith("payment"):
                payments.append((child.get("paymentmode"), _num(child.get("total"))))
            elif child.tag.endswith("detail"):
                details.append((
                    child.get("articlereference"),
                    _num(child.get("quantity")),
                    _num(child.get("total")),
                    child.get("detailtype"),
                ))
    return meta, details, payments


def discover_sales_files(root: Path, duplicate_policy: str) -> pd.DataFrame:
    """Index every sales XML and resolve store-days exported more than once."""
    sales_root = Path(root) / "salesdata" / "raw_xml"
    if not sales_root.exists():
        raise FileNotFoundError("no salesdata/raw_xml under {}".format(root))

    records = []
    for path in sales_root.rglob("*.xml"):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        records.append({
            "path": str(path), "folder": path.parent.parent.name,
            "hq": match.group(1), "store": match.group(2),
            "date_str": match.group(3), "exported": match.group(4),
        })
    if not records:
        raise FileNotFoundError("no PT_Sales_*.xml files under {}".format(sales_root))

    index = pd.DataFrame(records)
    index["date"] = pd.to_datetime(index["date_str"], format="%Y%m%d", errors="coerce")
    index = index.dropna(subset=["date"]).sort_values(["store", "date", "exported"])

    duplicated = index.duplicated(subset=["store", "date"], keep=False)
    conflicts = index.loc[duplicated].copy()
    if duplicate_policy == "error" and not conflicts.empty:
        raise ValueError(
            "{} store-days appear in more than one export file. Re-run with "
            "--duplicate-policy newest|oldest, or resolve them manually.".format(
                conflicts.groupby(["store", "date"]).ngroups))
    keep = "last" if duplicate_policy == "newest" else "first"
    index = index.drop_duplicates(subset=["store", "date"], keep=keep)
    index.attrs["conflicts"] = conflicts
    index.attrs["superseded"] = int(duplicated.sum() - conflicts.groupby(["store", "date"]).ngroups) \
        if not conflicts.empty else 0
    return index.reset_index(drop=True)


def build_store_daily(root, duplicate_policy="newest", stores=None, verbose=True) -> tuple:
    """One row per (date, POS code), with columns for EVERY product group."""
    root = Path(root)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    key_to_group, group_names = load_product_master(root)
    log("[products] {} article keys -> {} groups: {}".format(
        len(key_to_group), len(group_names), ", ".join(sorted(group_names.values()))))

    index = discover_sales_files(root, duplicate_policy)
    if stores:
        wanted = {str(s).zfill(3) for s in stores}
        index = index[index.store.isin(wanted)]
    log("[files] {} store-days | POS codes {} | {} superseded exports dropped ({})".format(
        len(index), sorted(index.store.unique()), index.attrs.get("superseded", 0), duplicate_policy))

    rows = []
    total_lines = unmapped_lines = 0
    started = time.time()
    for i, rec in enumerate(index.itertuples(index=False), 1):
        try:
            meta, details, payments = parse_sales_file(Path(rec.path))
        except ET.ParseError as exc:
            rows.append({"date": rec.date, "store": rec.store, "parse_error": str(exc)})
            continue

        row = {
            "date": rec.date,
            "store": rec.store,
            "schema": meta["schema"],
            "num_orders": meta["num_orders"],
            "tickets_printed": meta["tickets_printed"],
            "order_total_sum": round(meta["order_total"], 2),
            "order_paid_sum": round(meta["order_paid"], 2),
            "num_detail_lines": len(details),
        }

        if details:
            detail = pd.DataFrame(details, columns=["artref", "quantity", "total", "detailtype"])
            group = pd.to_numeric(detail.artref, errors="coerce").map(key_to_group)
            total_lines += len(detail)
            unmapped_lines += int(group.isna().sum())

            row["num_items_sold"] = float(detail.quantity.sum())
            row["total_revenue"] = round(float(detail.total.sum()), 2)
            row["distinct_articles"] = int(detail.artref.nunique())

            # Every group in the master gets columns, present in the day or not.
            for key, name in group_names.items():
                mask = group == key
                row["qty_" + name] = float(detail.loc[mask, "quantity"].sum())
                row["rev_" + name] = round(float(detail.loc[mask, "total"].sum()), 2)
                row["lines_" + name] = int(mask.sum())
            unmapped = group.isna()
            row["qty_unmapped"] = float(detail.loc[unmapped, "quantity"].sum())
            row["rev_unmapped"] = round(float(detail.loc[unmapped, "total"].sum()), 2)
            row["lines_unmapped"] = int(unmapped.sum())

            for dtype, count in detail.detailtype.value_counts().items():
                row["detailtype_{}_lines".format(_slug(dtype))] = int(count)

        if payments:
            pay = pd.DataFrame(payments, columns=["mode", "total"])
            for mode, block in pay.groupby("mode"):
                row["paymode_{}_count".format(_slug(mode))] = int(len(block))
                row["paymode_{}_total".format(_slug(mode))] = round(float(block.total.sum()), 2)

        rows.append(row)
        if verbose and i % 3000 == 0:
            log("  ... {}/{} files ({:.0f}s)".format(i, len(index), time.time() - started))

    panel = pd.DataFrame(rows).sort_values(["store", "date"]).reset_index(drop=True)
    stats = {
        "files_parsed": int(len(index)),
        "superseded_dropped": int(index.attrs.get("superseded", 0)),
        "duplicate_policy": duplicate_policy,
        "detail_lines": int(total_lines),
        "unmapped_article_lines": int(unmapped_lines),
        "unmapped_share": round(unmapped_lines / max(1, total_lines), 4),
        "parse_errors": int(panel["parse_error"].notna().sum()) if "parse_error" in panel else 0,
        "seconds": round(time.time() - started, 1),
        "product_groups": group_names,
    }
    log("[sales] {} store-days | {:,} detail lines | {:.2%} articles not in the master | {:.0f}s".format(
        len(panel), total_lines, stats["unmapped_share"], stats["seconds"]))
    return panel, stats


# ------------------------------------------------------------------ enrichment

def aggregate_chain(panel: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Sum every POS code into one row per date.

    Numeric columns are summed. Per-store text attributes cannot survive a sum and
    are dropped; `n_stores_reporting` records how many contributed each day.
    """
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    numeric = [c for c in panel.columns
               if c not in ("date", "store", "schema") and pd.api.types.is_numeric_dtype(panel[c])]
    chain = panel.groupby("date", as_index=False)[numeric].sum(min_count=1)
    chain["n_stores_reporting"] = panel.groupby("date")["store"].nunique().to_numpy()
    schemas = panel.groupby("date")["schema"].agg(
        lambda s: "|".join(sorted(set(x for x in s if isinstance(x, str)))))
    chain["schema"] = schemas.to_numpy()
    log("[chain] {} store-days -> {} days | {} numeric columns summed".format(
        len(panel), len(chain), len(numeric)))
    return chain


def attach_weather(frame: pd.DataFrame, root: Path, verbose=True) -> pd.DataFrame:
    """Join every weather column on the date. No column selection, no aggregation."""
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    files = sorted((Path(root) / "weatherdata").glob("*.parquet"))
    if not files:
        log("[weather] no parquet found - skipped")
        return frame
    weather = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    before = len(weather)
    weather = weather.drop_duplicates()  # 23 of every 24 rows are byte-identical
    weather["date"] = pd.to_datetime(weather["Date_time"], errors="coerce")
    weather = weather.dropna(subset=["date"])

    still_duplicated = int(weather.duplicated(subset=["date"]).sum())
    if still_duplicated:
        # Deterministic tie-break so the join stays 1:1; observations before forecasts.
        sort_cols = ["date"] + (["Forecast"] if "Forecast" in weather.columns else [])
        weather = weather.sort_values(sort_cols).drop_duplicates(subset=["date"], keep="first")

    drop = [c for c in ("Date_time", "Location", "Address") if c in weather.columns]
    weather = weather.drop(columns=drop)
    weather = weather.rename(columns={c: "weather_" + _slug(c) for c in weather.columns if c != "date"})

    out = frame.merge(weather, on="date", how="left")
    probe = [c for c in out.columns if c.startswith("weather_")]
    coverage = out[probe[0]].notna().mean() if probe else 0.0
    log("[weather] {} rows -> {} unique dates ({} exact duplicates, {} same-date conflicts) | "
        "{} columns | covers {:.0%} of days".format(
            before, len(weather), before - len(weather) - still_duplicated, still_duplicated,
            len(probe), coverage))
    return out


def attach_store_master(frame: pd.DataFrame, root: Path, verbose=True) -> pd.DataFrame:
    """Join every store-master column on the POS code. Per-store scope only."""
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    path = Path(root) / "storedata" / "Winkelfile.csv"
    if not path.exists() or "store" not in frame.columns:
        log("[stores] not applicable at this scope - skipped")
        return frame
    stores = pd.read_csv(path, sep=";", encoding="latin-1")
    stores.columns = [str(c).replace("﻿", "").strip() for c in stores.columns]
    stores["store"] = stores["POSCOD"].astype(str).str.zfill(3)
    stores = stores.drop_duplicates(subset=["store"])
    stores = stores.rename(columns={c: "store_" + _slug(c) for c in stores.columns
                                    if c not in ("store",)})
    out = frame.merge(stores, on="store", how="left")
    log("[stores] {} columns joined on POS code".format(len(stores.columns) - 1))
    return out


def attach_competitor(frame, root, mapping_path=None, verbose=True):
    """Join competitor flags -- only if the caller supplies a shop-code -> POS map.

    The competitor files key on shop codes like GENTDEPINT. Nothing in the data
    links those to POS codes, so inventing the mapping would be a judgement about
    the data; without --competitor-map the files are left out and reported.
    """
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    files = sorted((Path(root) / "competitordata").glob("*.csv"))
    if not files:
        return frame
    if not mapping_path:
        log("[competitor] {} file(s) NOT joined: no key in the data links their shop codes "
            "to POS codes. Pass --competitor-map <csv with columns winkel,store> to include "
            "them.".format(len(files)))
        return frame

    mapping = pd.read_csv(mapping_path)
    if not {"winkel", "store"}.issubset(mapping.columns):
        raise ValueError("--competitor-map needs columns 'winkel' and 'store'")
    mapping["store"] = mapping["store"].astype(str).str.zfill(3)

    comp = pd.concat([pd.read_csv(f, sep=";", encoding="latin-1") for f in files], ignore_index=True)
    comp["date"] = pd.to_datetime(comp["date"], errors="coerce", dayfirst=True)
    comp = comp.dropna(subset=["date"]).merge(mapping, on="winkel", how="inner")
    value_cols = [c for c in comp.columns if c not in ("date", "winkel", "store", "StoreKey")]
    for c in value_cols:
        comp[c] = pd.to_numeric(comp[c], errors="coerce")
    comp = comp.groupby(["date", "store"], as_index=False)[value_cols].sum(min_count=1)
    comp = comp.rename(columns={c: "competitor_" + _slug(c) for c in value_cols})

    keys = ["date", "store"] if "store" in frame.columns else ["date"]
    if keys == ["date"]:
        numeric = [c for c in comp.columns if c.startswith("competitor_")]
        comp = comp.groupby("date", as_index=False)[numeric].sum(min_count=1)
    out = frame.merge(comp, on=keys, how="left")
    log("[competitor] joined on {} via the supplied mapping".format("+".join(keys)))
    return out


# ------------------------------------------------------------------------ main

def build(root, out_path=None, scope=DECISIONS["scope"],
          duplicate_policy=DECISIONS["duplicate_policy"], stores=None,
          competitor_map=None, verbose=True):
    root = Path(root)
    panel, stats = build_store_daily(root, duplicate_policy, stores, verbose)

    if scope == "chain":
        frame = aggregate_chain(panel, verbose)
    else:
        frame = attach_store_master(panel, root, verbose)

    frame = attach_weather(frame, root, verbose)
    frame = attach_competitor(frame, root, competitor_map, verbose)
    frame = frame.sort_values("date").reset_index(drop=True)

    stats["scope"] = scope
    stats["rows"] = int(len(frame))
    stats["columns"] = int(frame.shape[1])
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        with open(out_path.with_suffix(".etl.json"), "w", encoding="utf-8") as fh:
            json.dump({"decisions": DECISIONS, "stats": stats}, fh, indent=2, default=str)
        if verbose:
            print("[out] {} rows x {} cols -> {}".format(len(frame), frame.shape[1], out_path),
                  flush=True)
    return frame, stats


def target_column(root) -> str:
    """The confirmed target: quantity of the confirmed product group(s)."""
    _, group_names = load_product_master(root)
    measure = {"quantity": "qty", "revenue": "rev"}[DECISIONS["target_measure"]]
    keys = DECISIONS["target_groups"]
    if len(keys) != 1:
        raise ValueError("multi-group targets need a summed column; groups={}".format(keys))
    return "{}_{}".format(measure, group_names[int(keys[0])])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="aernoudt-etl",
        description="Flatten the raw Aernoudt POS XML drop into a daily table.")
    p.add_argument("--root", required=True, help="the aernoudt data directory")
    p.add_argument("--out", default="data/aernoudt_daily.csv")
    p.add_argument("--scope", default=DECISIONS["scope"], choices=["chain", "store"],
                   help="'chain': sum every POS code into one daily series; "
                        "'store': one row per (date, POS code)")
    p.add_argument("--duplicate-policy", default=DECISIONS["duplicate_policy"],
                   choices=["newest", "oldest", "error"],
                   help="how to resolve a store-day exported more than once")
    p.add_argument("--stores", default="", help="comma-separated POS codes (default: all)")
    p.add_argument("--competitor-map", default=None,
                   help="CSV with columns winkel,store mapping competitor shop codes to POS codes")
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")
    frame, stats = build(
        args.root, args.out, scope=args.scope, duplicate_policy=args.duplicate_policy,
        stores=[s for s in args.stores.split(",") if s.strip()] or None,
        competitor_map=args.competitor_map,
    )
    print("\ntarget column: {}".format(target_column(args.root)))
    print("date range: {} -> {} ({} rows)".format(
        frame.date.min().date(), frame.date.max().date(), len(frame)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
