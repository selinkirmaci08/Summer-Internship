"""Put the no-feature-engineering baseline next to the fixed-recipe one.

Both are scored by `score_contract.py` on identical rows, under identical rules
(POS 001 only, actual > 0), so the gap between them is attributable to one thing:
whether 403 engineered features existed or not. Same engine, same budget, same
seed, same holdout, same metric definitions.

The seasonal-naive column is the floor both are measured against, and it is the
same seasonal naive in both files -- so if one baseline beats it and the other
does not, that is a real difference and not two different naives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _row(label: str, scores: dict) -> dict:
    o = scores["overall"]
    return {
        "system": label,
        "MAE": o["mae"],
        "RMSE": o["rmse"],
        "WAPE%": o["wape_pct"],
        "MASE": o["mase"],
        "bias": o["bias"],
        "skill%": scores["mean_skill_vs_seasonal_naive"] * 100,
        "won": "{}/{}".format(scores["horizons_beating_seasonal_naive"],
                              scores["horizons_evaluated"]),
        "n": scores["rows_scored"],
    }


def _table(rows, cols, widths=None) -> str:
    widths = widths or {c: max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in cols}
    head = " | ".join(c.rjust(widths[c]) for c in cols)
    rule = "-+-".join("-" * widths[c] for c in cols)
    body = "\n".join(" | ".join(_fmt(r[c]).rjust(widths[c]) for c in cols) for r in rows)
    return "{}\n{}\n{}".format(head, rule, body)


def _fmt(v) -> str:
    if isinstance(v, float):
        return "{:+.1f}".format(v) if abs(v) < 1000 and v < 0 else "{:.2f}".format(v)
    return str(v)


def compare(naive_path: Path, recipe_path: Path, window: str) -> dict:
    naive, recipe = _load(naive_path), _load(recipe_path)

    assert naive["rows_scored"] == recipe["rows_scored"], (
        "the two systems were scored on different row counts ({} vs {}) -- the "
        "comparison is not valid".format(naive["rows_scored"], recipe["rows_scored"]))

    rows = [_row("no feature engineering", naive), _row("fixed recipe (403 feats)", recipe)]
    cols = ["system", "MAE", "RMSE", "WAPE%", "MASE", "bias", "skill%", "won", "n"]

    print("\n### {} -- {} rows, identical on both sides\n".format(window, naive["rows_scored"]))
    print(_table(rows, cols))

    d_mae = recipe["overall"]["mae"] - naive["overall"]["mae"]
    pct = 100 * d_mae / naive["overall"]["mae"] if naive["overall"]["mae"] else float("nan")
    print("\nfeature engineering moved MAE by {:+.2f} ({:+.1f}%)".format(d_mae, pct))

    print("\nper horizon (MAE):")
    ph_n = {r["h"]: r for r in naive["per_horizon"]}
    ph_r = {r["h"]: r for r in recipe["per_horizon"]}
    hrows = []
    for h in sorted(ph_n):
        n_mae = ph_n[h]["model"]["mae"]
        r_mae = ph_r[h]["model"]["mae"]
        hrows.append({
            "h": h,
            "none": n_mae,
            "recipe": r_mae,
            "delta": r_mae - n_mae,
            "naive_ref": ph_n[h]["seasonal_naive"]["mae"],
        })
    print(_table(hrows, ["h", "none", "recipe", "delta", "naive_ref"]))

    return {"window": window, "no_fe": rows[0], "with_fe": rows[1],
            "mae_delta": d_mae, "mae_delta_pct": pct, "per_horizon": hrows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="compare-baselines")
    p.add_argument("--naive", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--window", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    result = compare(Path(args.naive), Path(args.recipe), args.window)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print("\n-> {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
