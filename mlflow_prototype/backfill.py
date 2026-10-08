"""
Prototype: backfill MLflow tracking from the existing agentic pipeline's
history/ and artifacts/ records for run_20260821T130000Z.

READ-ONLY with respect to the original project: every source file is read
from the staged uploads copy. Nothing in the original "General Plan" folder
or its runs/ directory is modified. All output goes to a brand new
./mlruns store inside this prototype folder.
"""
import json
import os
import pickle

import mlflow

SRC = "/mnt/user-data/uploads/General Plan/runs/run_20260821T130000Z"
PROTO_DIR = "/home/claude/mlflow_prototype"
# MLflow 3.x deprecated the raw filesystem store; sqlite is the modern
# recommended local backend and is what we'd actually want long-term anyway.
TRACKING_URI = f"sqlite:///{PROTO_DIR}/mlflow.db"

ITERATIONS = [6, 9, 10, 12, 14]

mlflow.set_tracking_uri(TRACKING_URI)
experiment_name = "aernoudt_bakery_demand_forecast"
mlflow.set_experiment(experiment_name)


def load_history(iter_num):
    path = f"{SRC}/history/iteration_{iter_num:03d}_training.json"
    with open(path) as f:
        return json.load(f)


def flatten_hparams(hparams: dict) -> dict:
    """hyperparameters in the history json are {name: {value, reasoning}} -- pull just values."""
    out = {}
    for k, v in hparams.items():
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


parent_run_name = "run_20260821T130000Z (bread_quantity forecast, branch 01)"
with mlflow.start_run(run_name=parent_run_name) as parent_run:
    mlflow.set_tags(
        {
            "pipeline_run_id": "run_20260821T130000Z",
            "goal": "forecast daily bread demand for Aernoudt Bakery branch 01",
            "target_column": "bread_quantity",
            "task_type": "time_series_forecasting",
            "source": "backfilled from history/*.json + artifacts/training/* (read-only prototype)",
        }
    )

    best_val_mae = None
    best_iter = None

    for it in ITERATIONS:
        hist = load_history(it)
        out = hist["output"]
        inp = hist["input"]

        with mlflow.start_run(run_name=f"iteration_{it:03d}_training", nested=True) as child:
            # --- tags: narrative context an MLflow run doesn't normally carry ---
            mlflow.set_tags(
                {
                    "iteration_id": f"iteration_{it:03d}",
                    "mode": inp.get("mode", ""),
                    "mode_reasoning": (inp.get("mode_reasoning") or "")[:250],
                    "model_class": out.get("model_class", ""),
                    "feature_subset_type": out.get("feature_subset_type", ""),
                    "deployment_scenario": out.get("deployment_scenario", ""),
                    "overfitting_assessment": out.get("overfitting_check", {}).get("assessment", ""),
                    "vs_prior_iteration_note": (out.get("vs_prior_iteration", {}) or {}).get("note", "")[:250],
                }
            )

            # --- params: hyperparameters + shapes ---
            hparams = flatten_hparams(out.get("hyperparameters", {}))
            mlflow.log_params(hparams)
            mlflow.log_param("n_features", out.get("training_summary", {}).get("n_features"))
            mlflow.log_param("n_train_rows", out.get("training_summary", {}).get("n_train_rows"))
            mlflow.log_param("n_val_rows", out.get("training_summary", {}).get("n_val_rows"))
            if out.get("hyperparameter_search"):
                hs = out["hyperparameter_search"]
                mlflow.log_param("hp_search_method", hs.get("method"))
                mlflow.log_param("hp_search_n_iter", hs.get("n_iter"))

            # --- metrics: train/val MAE, RMSE, R2, MAPE, MASE + overfit ratio ---
            metrics = out.get("metrics", {})
            for split in ("train", "validation"):
                split_metrics = metrics.get(split, {})
                for name, val in split_metrics.items():
                    if isinstance(val, dict) and "value" in val:
                        v = val["value"]
                    else:
                        v = val
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{split}_{name}", v)

            of = out.get("overfitting_check", {})
            if "ratio" in of:
                mlflow.log_metric("overfit_ratio", of["ratio"])

            baseline = out.get("baseline_model", {})
            if "val_MAE" in baseline:
                mlflow.log_metric("baseline_val_MAE", baseline["val_MAE"])

            # --- artifacts: feature importance table/plot, diagnostic plots, model ---
            art_dir = f"{SRC}/artifacts/training"
            for fname_suffix in (
                "feature_importance.csv",
                "feature_importance.png",
                "actual_vs_predicted_val.png",
                "residuals_over_time.png",
            ):
                fpath = f"{art_dir}/iteration_{it:03d}_training_{fname_suffix}"
                if os.path.exists(fpath):
                    mlflow.log_artifact(fpath, artifact_path="diagnostics")

            # log the trained model itself as a proper MLflow model (enables the registry)
            model_pkl = f"{art_dir}/iteration_{it:03d}_training_model.pkl"
            if os.path.exists(model_pkl):
                with open(model_pkl, "rb") as f:
                    model = pickle.load(f)
                mlflow.lightgbm.log_model(model, name="model")

            val_mae = metrics.get("validation", {}).get("MAE")
            if isinstance(val_mae, dict):
                val_mae = val_mae.get("value")
            if isinstance(val_mae, (int, float)) and (best_val_mae is None or val_mae < best_val_mae):
                best_val_mae = val_mae
                best_iter = it

    mlflow.set_tag("best_iteration", f"iteration_{best_iter:03d}" if best_iter else "n/a")
    mlflow.set_tag("best_val_mae", str(best_val_mae))
    mlflow.log_metric("best_val_mae_across_run", best_val_mae)

print(f"Done. Best iteration: {best_iter} (val MAE {best_val_mae})")
print(f"Tracking store: {TRACKING_URI}")
