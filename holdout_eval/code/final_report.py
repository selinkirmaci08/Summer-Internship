# %% [Setup]
"""Holdout evaluation report: agentic pipeline vs AutoML vs naive, on the external holdout.

Follows skills/final_report-skill/final_report.md. Emits:
  ../report.html                       self-contained, inline SVG, no network
  ../final_report.json                 full metric set + alignment record
  ../artifacts/final_report/*.csv      one CSV per chart; the HTML is a rendering, these are the record
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
OUT = os.path.join(EVAL, "outputs")
ART = os.path.join(EVAL, "artifacts", "final_report")
os.makedirs(ART, exist_ok=True)
RUN_DIR = (r"C:\Users\selin\OneDrive\Desktop\All\KU Leuven\internship-project"
           r"\General Plan\runs\run_20260904T140437Z")
AUTOML = os.path.join(os.path.dirname(EVAL), "automl-runs", "automl-baseline", "outputs")

HORIZONS = [1, 7, 14, 30, 60, 90]
WINDOWS = ["holdout_primary", "holdout_stress"]
NAIVE = "seasonal_naive_m7"

SYSTEMS = {
    "agentic_shipped_calendar": dict(label="Agentic (as shipped)", slot=1, chart=True,
        what="the run's six frozen LightGBM boosters, one per horizon", feats=37,
        est="LightGBM", budget="agent-directed, 8 iterations"),
    NAIVE: dict(label="Seasonal naive (m=7)", slot=2, chart=True,
        what="last observed same-weekday value at the origin; zero parameters", feats=0,
        est="none", budget="none"),
    "automl_recipe": dict(label="AutoML (403 features)", slot=3, chart=True,
        what="FLAML over a fixed generic feature recipe, fitted on the whole training partition",
        feats=403, est="FLAML → lgbm", budget="900 s"),
    "seasonal_median_52wk": dict(label="Seasonal median (52wk)", slot=4, chart=True,
        what="median of the same weekday over the last year; window is a calendar constant", feats=0,
        est="none", budget="none"),
    "agentic_refit_calendar": dict(label="Agentic (refit on train+val)", slot=5, chart=True,
        what="same six specifications, refit through 2020-12-31 per the end_model contract", feats=37,
        est="LightGBM", budget="no re-tuning"),
    "automl_no_fe": dict(label="AutoML (no feature eng.)", slot=6, chart=False,
        what="FLAML on raw columns, shift(h) only — the floor", feats=94, est="FLAML", budget="150 s/horizon"),
    "agentic_shipped_row": dict(label="Agentic (as shipped, row-offset h)", slot=7, chart=False,
        what="the frozen boosters under the h convention they were trained on", feats=37,
        est="LightGBM", budget="—"),
    "agentic_refit_row": dict(label="Agentic (refit, row-offset h)", slot=8, chart=False,
        what="the refit under the training h convention", feats=37, est="LightGBM", budget="—"),
}
PRED_FILE = {s: (f"outputs/preds_{{w}}__{s}.csv" if s.startswith("agentic")
                 else {"automl_recipe": "automl-runs/.../contract_aernoudt_split/preds_{w}.csv",
                       "automl_no_fe": "automl-runs/.../naive_automl/preds_{w}.csv"}.get(s, "computed in score_all.py"))
             for s in SYSTEMS}
COLORS = {1: ("#2a78d6", "#3987e5"), 2: ("#eb6834", "#d95926"), 3: ("#1baf7a", "#199e70"),
          4: ("#eda100", "#c98500"), 5: ("#e87ba4", "#d55181")}

# %% [Load]
scores = json.load(open(os.path.join(OUT, "scores.json"), encoding="utf-8"))
agentic_manifest = json.load(open(os.path.join(OUT, "agentic_manifest.json"), encoding="utf-8"))
rows = {w: pd.read_csv(os.path.join(OUT, f"scored_rows_{w}.csv"), parse_dates=["target_date"])
        for w in WINDOWS}


# %% [Metrics] per-horizon MASE against naive AT THE SAME HORIZON, plus level/shape split
def metric_block(d, sys):
    y, p = d["y_true"].to_numpy(), d[sys].to_numpy()
    e = p - y
    ae = np.abs(e)
    denom = np.abs(y) + np.abs(p)
    naive_mae = float(np.mean(np.abs(d[NAIVE].to_numpy() - y)))
    return {
        "n": int(len(d)), "MAE": float(ae.mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
        "sMAPE_pct": float(100 * np.mean(np.where(denom == 0, 0.0, 2 * ae / np.where(denom == 0, 1, denom)))),
        "bias": float(e.mean()),
        "MAE_debiased": float(np.mean(np.abs(e - e.mean()))),
        "MASE_vs_naive_same_h": float(ae.mean() / naive_mae) if naive_mae else float("nan"),
    }


analysis = {}
for w in WINDOWS:
    d = rows[w]
    per_h = {h: {s: metric_block(d[d.h == h], s) for s in SYSTEMS} for h in HORIZONS if (d.h == h).any()}
    pooled = {s: metric_block(d, s) for s in SYSTEMS}
    wins = {s: sum(1 for h in per_h if min(per_h[h], key=lambda x: per_h[h][x]["MAE"]) == s) for s in SYSTEMS}
    beats_naive = {s: sum(1 for h in per_h if per_h[h][s]["MAE"] < per_h[h][NAIVE]["MAE"]) for s in SYSTEMS}
    analysis[w] = {"pooled": pooled, "per_horizon": per_h, "wins": wins,
                   "beats_naive_h": beats_naive, "n_horizons": len(per_h),
                   "rows": scores[w]["rows"], "coverage": scores[w]["coverage_per_system"],
                   "window": scores[w]["window"]}

# per-year bias, across both windows
allrows = pd.concat([rows[w].assign(_w=w) for w in WINDOWS], ignore_index=True)
allrows["year"] = allrows["target_date"].dt.year
per_year = {}
for yr, d in allrows.groupby("year"):
    per_year[int(yr)] = {"n": int(len(d)), "mean_actual": float(d["y_true"].mean()),
                         **{s: {"bias": float((d[s] - d["y_true"]).mean()),
                                "MAE": float(np.abs(d[s] - d["y_true"]).mean())}
                            for s in SYSTEMS if d[s].notna().any()}}

# %% [Explainability] grouped permutation importance on VALIDATION, never the holdout
import lightgbm as lgb

GROUPS = {
    "lags": [f"bq_lag{k}" for k in (1, 7, 14, 28, 30, 91, 364)],
    "fourier": ["dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"],
    "calendar": ["year", "doy", "is_weekend", "is_public_holiday"],
    "trend": ["year_centered"],
    "weather": [c for c in json.load(open(os.path.join(RUN_DIR, "artifacts", "end_model",
                "iteration_008_prediction_manifest.json"), encoding="utf-8"))["feature_order"]
                if c.startswith("wx_")],
    "competitor": [c for c in json.load(open(os.path.join(RUN_DIR, "artifacts", "end_model",
                   "iteration_008_prediction_manifest.json"), encoding="utf-8"))["feature_order"]
                   if c.startswith("comp_")],
}
FEATS = json.load(open(os.path.join(RUN_DIR, "artifacts", "end_model",
                 "iteration_008_prediction_manifest.json"), encoding="utf-8"))["feature_order"]

lf = pd.read_csv(os.path.join(RUN_DIR, "data", "iteration_003_feature_engineering_long.csv"),
                 parse_dates=["date", "origin_date"])
val = lf[(lf._split == "test") & lf.bread_quantity.notna()].copy()
boosters = {h: lgb.Booster(model_file=os.path.join(RUN_DIR, "artifacts", "training", "models", f"lgbm_h{h}.txt"))
            for h in HORIZONS}
rng = np.random.default_rng(42)


def val_mae(Xmap):
    tot, n = 0.0, 0
    for h in HORIZONS:
        d = val[val.h == h]
        if not len(d):
            continue
        X = Xmap[h]
        tot += np.abs(boosters[h].predict(X) - d.bread_quantity.to_numpy()).sum()
        n += len(d)
    return tot / n


base_X = {h: val[val.h == h][FEATS].to_numpy() for h in HORIZONS}
base = val_mae(base_X)


def permuted(cols, reps=5):
    idxs = [FEATS.index(c) for c in cols]
    vals = []
    for _ in range(reps):
        Xm = {}
        for h in HORIZONS:
            X = base_X[h].copy()
            perm = rng.permutation(len(X))
            X[:, idxs] = X[perm][:, idxs]
            Xm[h] = X
        vals.append(val_mae(Xm) - base)
    return float(np.mean(vals)), float(np.std(vals))


importance = {}
for g, cols in GROUPS.items():
    grouped, sd = permuted(cols)
    solo = sum(permuted([c], reps=3)[0] for c in cols)
    importance[g] = {"grouped_dMAE": grouped, "sd": sd, "sum_solo": solo,
                     "masking_gap": grouped - solo, "n_features": len(cols)}

# %% [CSV record]
pd.DataFrame([{"window": w, "h": h, "system": s, **analysis[w]["per_horizon"][h][s]}
              for w in WINDOWS for h in analysis[w]["per_horizon"] for s in SYSTEMS]
             ).to_csv(os.path.join(ART, "mae_by_horizon.csv"), index=False)
pd.DataFrame([{"window": w, "system": s, **analysis[w]["pooled"][s]} for w in WINDOWS for s in SYSTEMS]
             ).to_csv(os.path.join(ART, "pooled_metrics.csv"), index=False)
pd.DataFrame([{"year": y, "system": s, **v} for y, d in per_year.items()
              for s, v in d.items() if isinstance(v, dict)]
             ).to_csv(os.path.join(ART, "bias_by_year.csv"), index=False)
pd.DataFrame([{"group": g, **v} for g, v in importance.items()]
             ).to_csv(os.path.join(ART, "grouped_importance.csv"), index=False)

json.dump({"analysis": analysis, "per_year": per_year, "importance": importance,
           "validation_base_mae": base, "agentic_manifest": agentic_manifest,
           "sources": {s: PRED_FILE[s] for s in SYSTEMS}},
          open(os.path.join(EVAL, "final_report.json"), "w", encoding="utf-8"), indent=2, default=str)
print("metrics + CSVs written")
print(json.dumps({w: {s: round(analysis[w]['pooled'][s]['MAE'], 2) for s in SYSTEMS} for w in WINDOWS}, indent=1))
print(json.dumps(importance, indent=1))
