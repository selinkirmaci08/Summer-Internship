"""Score every system on one shared table of rows.

The scoring functions are v1's (`automl_baseline.metrics`) so a number produced
here means the same thing as a number produced by the v1 baseline. What this module
adds is the discipline the comparison needs: one wide table, one row per
(target_date, horizon), one column per system, and every metric computed over the
identical row mask. If a system has a gap, the gap is reported -- it is never
filled and never silently dropped from the denominator.

Two MASE denominators are carried, matching the agentic run's own reporting:

  * **train denominator** -- the seasonal-naive error over the training window.
    Comparable with every MASE the run quotes elsewhere.
  * **test denominator** -- the seasonal-naive error over the scored rows
    themselves. The honest "better than doing nothing, here" number.

They differ whenever the test period is harder than the training period, which it
is here, so quoting only one of them would flatter or damn every system at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from automl_baseline import metrics as m1

# Horizon bands, so a 21-horizon table can be read at a glance. The boundaries are
# the ones the forecasting question actually turns on: within the week, within the
# month, within the quarter, beyond it.
HORIZON_BANDS = (
    ("h1-7 (within the week)", 1, 7),
    ("h8-30 (within the month)", 8, 30),
    ("h31-90 (within the quarter)", 31, 90),
    ("h91+ (beyond the quarter)", 91, 10**6),
)


def metric_set(y_true, y_pred, mase_train: float, mase_test: float) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = m1.mae(y_true, y_pred)
    return {
        "n": int(np.sum(np.isfinite(y_true) & np.isfinite(y_pred))),
        "MAE": round(mae, 4),
        "RMSE": round(m1.rmse(y_true, y_pred), 4),
        "R2": round(m1.r2(y_true, y_pred), 4),
        "SMAPE_pct": round(m1.smape(y_true, y_pred), 4),
        "WAPE_pct": round(m1.wape(y_true, y_pred), 4),
        "bias": round(m1.bias(y_true, y_pred), 4),
        "MASE_train_denom": round(mae / mase_train, 4) if mase_train else float("nan"),
        "MASE_test_denom": round(mae / mase_test, 4) if mase_test else float("nan"),
    }


def build_prediction_table(run, systems: dict) -> pd.DataFrame:
    """One row per scored (target_date, horizon); one `pred_<system>` column each.

    Joining on the pair rather than concatenating in order is deliberate: an
    ordering assumption that quietly breaks would misalign two systems' errors and
    the result would still look like a table of numbers.
    """
    idx = run.index("test").rename(columns={run.target: "y_true"})
    table = idx[[run.time_col, "origin_date", "h", "y_true"]].copy()

    for name, preds in systems.items():
        col = "pred_{}".format(name)
        if isinstance(preds, pd.DataFrame):
            got = preds.rename(columns={"pred": col})[[run.time_col, "h", col]]
            before = len(table)
            table = table.merge(got, on=[run.time_col, "h"], how="left", validate="one_to_one")
            if len(table) != before:
                raise ValueError("{}: merging predictions changed the row count".format(name))
        else:
            arr = np.asarray(preds, dtype=float)
            if len(arr) != len(table):
                raise ValueError(
                    "{}: {} predictions for {} scored rows".format(name, len(arr), len(table))
                )
            table[col] = arr

    return table.sort_values([run.time_col, "h"]).reset_index(drop=True)


def _coverage(table: pd.DataFrame, systems) -> dict:
    out = {}
    for name in systems:
        col = "pred_{}".format(name)
        missing = int(table[col].isna().sum())
        out[name] = {
            "predicted_rows": int(len(table) - missing),
            "missing_rows": missing,
            "complete": missing == 0,
        }
    return out


def score(run, systems: dict, table: pd.DataFrame | None = None) -> dict:
    """Overall, per-horizon and per-band metrics for every system."""
    table = build_prediction_table(run, systems) if table is None else table
    denoms = run.mase_denominators()
    d_train, d_test = denoms["train_denominator"], denoms["test_denominator"]

    names = list(systems)
    cols = {n: "pred_{}".format(n) for n in names}

    # Every system is scored on the rows where *all* of them predicted, so no system
    # is advantaged by having quietly skipped its hardest rows.
    common = table["y_true"].notna()
    for n in names:
        common &= table[cols[n]].notna()
    shared = table.loc[common]

    overall = {n: metric_set(shared["y_true"], shared[cols[n]], d_train, d_test) for n in names}

    per_horizon = []
    for h, grp in shared.groupby("h", sort=True):
        row = {"h": int(h), "n": int(len(grp)), "y_mean": round(float(grp["y_true"].mean()), 2)}
        for n in names:
            row["MAE_{}".format(n)] = round(m1.mae(grp["y_true"], grp[cols[n]]), 3)
        per_horizon.append(row)

    per_band = []
    for label, lo, hi in HORIZON_BANDS:
        grp = shared.loc[(shared["h"] >= lo) & (shared["h"] <= hi)]
        if grp.empty:
            continue
        row = {"band": label, "n": int(len(grp)),
               "horizons": sorted(int(x) for x in grp["h"].unique())}
        for n in names:
            row["MAE_{}".format(n)] = round(m1.mae(grp["y_true"], grp[cols[n]]), 3)
        per_band.append(row)

    per_year = []
    years = pd.DatetimeIndex(shared[run.time_col]).year
    for yr, grp in shared.groupby(years, sort=True):
        row = {"year": int(yr), "n": int(len(grp)), "y_mean": round(float(grp["y_true"].mean()), 2)}
        for n in names:
            row["MAE_{}".format(n)] = round(m1.mae(grp["y_true"], grp[cols[n]]), 3)
        per_year.append(row)

    return {
        "n_scored_rows": int(len(shared)),
        "n_rows_in_grid": int(len(table)),
        "scored_window": [str(pd.Timestamp(shared[run.time_col].min()).date()),
                          str(pd.Timestamp(shared[run.time_col].max()).date())],
        "mase_denominators": denoms,
        "coverage": _coverage(table, systems),
        "overall": overall,
        "per_horizon": per_horizon,
        "per_band": per_band,
        "per_year": per_year,
    }


def head_to_head(scored: dict, champion: str, reference: str = "naive") -> dict:
    """Rank the systems and state the gaps in the terms a reader will ask for."""
    overall = scored["overall"]
    ranked = sorted(overall, key=lambda n: overall[n]["MAE"])
    champ_mae = overall[champion]["MAE"]

    verdicts = {}
    for name, met in overall.items():
        if name == champion:
            continue
        gap = met["MAE"] - champ_mae
        verdicts[name] = {
            "MAE": met["MAE"],
            "MAE_gap_vs_{}".format(champion): round(gap, 4),
            "pct_worse_than_{}".format(champion): round(gap / champ_mae * 100.0, 2),
            "beats_{}".format(champion): bool(met["MAE"] < champ_mae),
        }

    per_h_wins = {n: 0 for n in overall}
    for row in scored["per_horizon"]:
        maes = {n: row["MAE_{}".format(n)] for n in overall}
        per_h_wins[min(maes, key=maes.get)] += 1

    ref_mae = overall.get(reference, {}).get("MAE")
    skill = {}
    if ref_mae:
        for name, met in overall.items():
            skill[name] = round((1.0 - met["MAE"] / ref_mae) * 100.0, 2)

    return {
        "ranking_by_MAE": ranked,
        "best": ranked[0],
        "champion": champion,
        "champion_is_best": ranked[0] == champion,
        "versus_champion": verdicts,
        "horizons_won": per_h_wins,
        "skill_vs_{}_pct".format(reference): skill,
        "skill_definition": "1 - MAE_system / MAE_{}, in percent, on the identical scored "
                            "rows. Positive means better than {}.".format(reference, reference),
    }
