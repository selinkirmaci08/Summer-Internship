"""Score the agentic run's frozen champions on the holdout, plus the contract-required refit.

Emits contract CSVs (target_date, h, y_pred) for four agentic variants:

  agentic_shipped_calendar  the six frozen boosters, origin = target_date - h CALENDAR days
                            (what protocol.json asks for)
  agentic_shipped_row       the same boosters, origin = h ROWS earlier in the trading-day frame
                            (how they were actually trained; reported as a sensitivity, because
                            pre-2020-06 Mondays are absent so a week is 6 rows and h=30 rows is
                            35 calendar days)
  agentic_refit_calendar    the same six specifications refit on train + validation through
                            2020-12-31 per contracts/end_model.json refit_champion_on_all_data,
                            n_estimators pinned to the early-stopped value
  agentic_refit_row         the refit under the row convention

Nothing here re-tunes anything. The refit changes how many rows the fit saw and nothing else.
"""

from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

import build_frame as bf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

WINDOWS = {
    "holdout_primary": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"),
                        [bf.TRAIN_ROOT, bf.PRIMARY_ROOT]),
    "holdout_stress": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-02-08"),
                       [bf.TRAIN_ROOT, bf.PRIMARY_ROOT, bf.STRESS_ROOT]),
}

# frozen at iteration_005 by early stopping; pinned for the refit, never re-derived
LGBM_PARAMS = {
    "objective": "regression_l1", "metric": "mae", "learning_rate": 0.03, "num_leaves": 31,
    "min_child_samples": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.05, "reg_lambda": 0.1, "verbose": -1, "random_state": 42,
}
PINNED_N_ESTIMATORS = {1: 185, 7: 141, 14: 274, 30: 119, 60: 52, 90: 69}


def load_shipped():
    return {h: lgb.Booster(model_file=os.path.join(bf.RUN_DIR, "artifacts", "training", "models",
                                                   f"lgbm_h{h}.txt"))
            for h in bf.HORIZONS}


CUTOFF = pd.Timestamp("2020-12-31")


def build_refit(frame):
    """Refit each horizon's spec on every row the run was given (through 2020-12-31).

    Both ends of a training pair must sit inside the training partition. Filtering only the
    origin is not enough: targets are shift(-h) over a frame that extends into the holdout, so
    an October 2020 origin at h=90 would otherwise be fitted against a January 2021 actual.
    """
    models, log = {}, {}
    for h in bf.HORIZONS:
        target = f"{bf.TARGET}_h{h}"
        target_date = frame["date"].shift(-h)
        d = frame[(frame["date"] <= CUTOFF) & (target_date <= CUTOFF) & frame[target].notna()]
        model = lgb.LGBMRegressor(n_estimators=PINNED_N_ESTIMATORS[h], **LGBM_PARAMS)
        model.fit(d[bf.FEATURES], d[target])
        models[h] = model
        log[h] = {"rows": int(len(d)), "n_estimators": PINNED_N_ESTIMATORS[h],
                  "last_origin": str(d["date"].max().date()),
                  "last_target": str(target_date.loc[d.index].max().date())}
    return models, log


def origin_index(frame, window_from, window_to):
    """Rows to predict: every observed date in the window, paired with both origin conventions."""
    dates = frame["date"].to_numpy()
    pos = {d: i for i, d in enumerate(frame["date"])}
    targets = frame[(frame["date"] >= window_from) & (frame["date"] <= window_to)]

    rows = []
    for _, t in targets.iterrows():
        ti = pos[t["date"]]
        for h in bf.HORIZONS:
            cal_origin = t["date"] - pd.Timedelta(days=h)
            row_origin = frame["date"].iloc[ti - h] if ti - h >= 0 else pd.NaT
            rows.append({
                "target_date": t["date"], "h": h, "y_true": t[bf.TARGET],
                "calendar_origin": cal_origin if cal_origin in pos else pd.NaT,
                "row_origin": row_origin,
            })
    return pd.DataFrame(rows), pos


def predict(frame, grid, pos, models, origin_col, kind):
    """kind 'booster' uses Booster.predict, kind 'sk' uses the sklearn wrapper."""
    out = np.full(len(grid), np.nan)
    feat = frame[bf.FEATURES]
    for h in bf.HORIZONS:
        m = (grid["h"] == h) & grid[origin_col].notna()
        if not m.any():
            continue
        idx = [pos[d] for d in grid.loc[m, origin_col]]
        X = feat.iloc[idx]
        out[m.to_numpy()] = models[h].predict(X if kind == "sk" else X.to_numpy())
    return out


def main():
    manifest = {"declared_departures": bf.DECLARED_DEPARTURES, "windows": {}}
    shipped = load_shipped()

    for name, (w_from, w_to, roots) in WINDOWS.items():
        print(f"\n=== {name} ===")
        frame = bf.build(roots)
        print(f"  frame: {len(frame)} trading rows, {frame.date.min().date()} -> {frame.date.max().date()}")
        print(f"  sales: {frame.attrs['sales_stats']}")

        refit, refit_log = build_refit(frame)
        grid, pos = origin_index(frame, w_from, w_to)

        preds = {
            "agentic_shipped_calendar": predict(frame, grid, pos, shipped, "calendar_origin", "booster"),
            "agentic_shipped_row": predict(frame, grid, pos, shipped, "row_origin", "booster"),
            "agentic_refit_calendar": predict(frame, grid, pos, refit, "calendar_origin", "sk"),
            "agentic_refit_row": predict(frame, grid, pos, refit, "row_origin", "sk"),
        }

        for system, y in preds.items():
            df = grid[["target_date", "h"]].copy()
            df["y_pred"] = y
            df = df[df["y_pred"].notna()]
            path = os.path.join(OUT, f"preds_{name}__{system}.csv")
            df.to_csv(path, index=False, date_format="%Y-%m-%d")
            print(f"  {system:26s} {len(df):5d} rows -> {os.path.basename(path)}")

        # ground truth for this window, from the same replay every prediction used
        truth = grid[["target_date", "h", "y_true"]].drop_duplicates()
        truth.to_csv(os.path.join(OUT, f"truth_{name}.csv"), index=False, date_format="%Y-%m-%d")

        manifest["windows"][name] = {
            "window": [str(w_from.date()), str(w_to.date())],
            "data_roots": roots,
            "frame_rows": int(len(frame)),
            "sales_stats": frame.attrs["sales_stats"],
            "target_dates_in_window": int(grid["target_date"].nunique()),
            "grid_rows": int(len(grid)),
            "calendar_origin_missing": int(grid["calendar_origin"].isna().sum()),
            "row_origin_missing": int(grid["row_origin"].isna().sum()),
            "refit": refit_log,
        }

    manifest["pinned_n_estimators"] = PINNED_N_ESTIMATORS
    manifest["hyperparameters"] = LGBM_PARAMS
    manifest["refit_note"] = ("per contracts/end_model.json refit_champion_on_all_data: identical "
                              "features, identical hyperparameters, n_estimators pinned to the "
                              "early-stopped value, fitted through 2020-12-31. Never scored on "
                              "validation - only on the holdout.")
    with open(os.path.join(OUT, "agentic_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> outputs/agentic_manifest.json")


if __name__ == "__main__":
    main()
