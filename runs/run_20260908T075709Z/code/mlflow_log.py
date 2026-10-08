"""
Log run_20260908T075709Z to MLflow.

Creates one experiment with one parent run (final metrics + artifacts)
and child runs for each major skill iteration that has scoreable metrics.
"""
import json
import pathlib
import mlflow
import mlflow.tracking

RUN_DIR = pathlib.Path(__file__).parent.parent
HISTORY = RUN_DIR / "history"
ARTIFACTS = RUN_DIR / "artifacts"
MLRUNS = RUN_DIR.parent.parent / "mlruns"   # General Plan/mlruns
DB_PATH = RUN_DIR.parent.parent / "mlflow.db"   # General Plan/mlflow.db

EXPERIMENT_NAME = "aernoudt-bakery-branch01"
PARENT_RUN_NAME = "run_20260908T075709Z"


def load(filename):
    return json.loads((HISTORY / filename).read_text())


def log_child(parent_run_id, name, metrics, params=None, tags=None):
    with mlflow.start_run(
        run_name=name,
        nested=True,
        tags={"parent_run_id": parent_run_id, **(tags or {})},
    ):
        mlflow.log_metrics(metrics)
        if params:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})


def main():
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ------------------------------------------------------------------ #
    #  Load history JSONs                                                  #
    # ------------------------------------------------------------------ #
    eda_raw    = load("iteration_001_eda.json")
    eda        = eda_raw.get("target_analysis", eda_raw)  # normalise nesting
    baseline   = load("iteration_004_baseline.json")
    champ13    = load("iteration_013_final_pass_champion_selection.json")
    test27     = load("iteration_027_test_evaluation.json")
    naive28    = load("iteration_028_corrected_naive.json")

    horizons = [1, 7, 14, 30, 60, 90]

    # baseline floors from part_a_scores best per horizon
    floor = {}
    for h_str, hdata in baseline.get("per_horizon_floor", {}).items():
        floor[int(h_str)] = hdata.get("fold3_mae") or hdata.get("pooled_mae") or hdata.get("mae")

    # fallback: read from champion selection margin + val_mae
    if not floor:
        for h_str, hdata in champ13["per_horizon_champion"].items():
            h = int(h_str)
            floor[h] = hdata.get("baseline_floor")

    primary   = test27["results"]["holdout_primary"]
    stress    = test27["results"]["holdout_stress"]
    naive_pri = naive28["results"]["holdout_primary"]
    val_ref   = test27.get("val_mae_reference", {})

    # ------------------------------------------------------------------ #
    #  Parent run                                                          #
    # ------------------------------------------------------------------ #
    with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent:
        parent_run_id = parent.info.run_id

        # --- tags ---
        mlflow.set_tags({
            "run_id":          "run_20260908T075709Z",
            "task":            "time_series_forecasting",
            "target":          "bread_quantity",
            "bakery":          "Aernoudt branch 01",
            "training_cutoff": "2020-12-31",
            "holdout_primary": "2021-01-01 to 2021-12-31",
            "n_trading_days":  str(eda.get("n_trading_days", 2897)),
            "n_iterations":    "29",
            "metric":          "MAE",
        })

        # --- params: champion algorithm per horizon ---
        champion_params = {}
        for h_str, hdata in champ13["per_horizon_champion"].items():
            h = int(h_str)
            champion_params[f"champion_h{h}"] = hdata.get("algorithm", "unknown")
        mlflow.log_params(champion_params)

        # --- EDA params ---
        ts     = eda.get("target_stats", {})
        trend  = eda.get("trend", {})
        acf    = eda.get("acf_at_horizons", {})
        regime = eda.get("regime_change", {})
        mlflow.log_params({
            "eda_mean":           round(float(ts.get("mean", 315.89)), 2),
            "eda_std":            round(float(ts.get("std", 84.12)), 2),
            "eda_trend_pct":      float(trend.get("total_decline_pct", -28.7)),
            "eda_acf_lag6":       float(acf.get("lag_6", 0.582)),
            "eda_zero_days":      int(ts.get("zero_demand_days", 50)),
            "regime_change_date": str(regime.get("date", "2020-06-01")),
        })

        # --- metrics: test holdout primary ---
        test_metrics = {}
        for h_str, hd in primary.items():
            h = int(h_str)
            test_metrics[f"test_mae_h{h}"]       = round(hd["mae"], 3)
            test_metrics[f"naive_mae_h{h}"]       = round(naive_pri[h_str]["naive_mae"], 3)
            test_metrics[f"beats_naive_h{h}"]     = int(hd["beats_naive"])
            test_metrics[f"val_mae_h{h}"]         = round(float(val_ref.get(h_str, 0)), 3)
            if floor.get(h) is not None:
                test_metrics[f"baseline_floor_h{h}"] = round(floor[h], 3)

        # aggregate: mean test MAE across horizons
        test_metrics["test_mae_mean"] = round(
            sum(primary[str(h)]["mae"] for h in horizons) / len(horizons), 3
        )
        test_metrics["horizons_beating_naive"] = sum(
            int(primary[str(h)]["beats_naive"]) for h in horizons
        )

        # stress holdout
        for h_str, hd in stress.items():
            h = int(h_str)
            test_metrics[f"stress_mae_h{h}"] = round(hd["mae"], 3)

        mlflow.log_metrics(test_metrics)

        # --- artifacts: EDA plots ---
        eda_plot_dir = ARTIFACTS / "eda"
        if eda_plot_dir.exists():
            for png in sorted(eda_plot_dir.glob("*.png")):
                mlflow.log_artifact(str(png), artifact_path="eda_plots")

        # --- artifacts: end model pickles ---
        end_model_dir = ARTIFACTS / "end_model"
        if end_model_dir.exists():
            for pkl in sorted(end_model_dir.glob("*.pkl")):
                mlflow.log_artifact(str(pkl), artifact_path="models")

        # --- artifacts: diagnosis plots ---
        diag_dir = ARTIFACTS / "diagnosis"
        if diag_dir.exists():
            for png in sorted(diag_dir.glob("*.png")):
                mlflow.log_artifact(str(png), artifact_path="diagnosis_plots")

        # ------------------------------------------------------------------ #
        #  Child run: EDA                                                     #
        # ------------------------------------------------------------------ #
        eda_metrics = {}
        for lag_key, val in acf.items():
            try:
                eda_metrics[f"acf_{lag_key}"] = float(val)
            except (TypeError, ValueError):
                pass
        eda_metrics["mean_demand"]       = round(float(ts.get("mean", 315.89)), 3)
        eda_metrics["std_demand"]        = round(float(ts.get("std", 84.12)), 3)
        eda_metrics["trend_decline_pct"] = float(trend.get("total_decline_pct", -28.7))
        eda_metrics["zero_demand_days"]  = int(ts.get("zero_demand_days", 50))
        log_child(parent_run_id, "01_eda", eda_metrics,
                  tags={"skill": "eda", "iteration": "001"})

        # ------------------------------------------------------------------ #
        #  Child run: baseline                                                #
        # ------------------------------------------------------------------ #
        if floor:
            baseline_metrics = {f"floor_h{h}": v for h, v in floor.items() if v}
            log_child(parent_run_id, "04_baseline", baseline_metrics,
                      tags={"skill": "baseline", "iteration": "004"})

        # ------------------------------------------------------------------ #
        #  Child run: champion selection (val MAE)                           #
        # ------------------------------------------------------------------ #
        champ_metrics = {}
        for h_str, hd in champ13["per_horizon_champion"].items():
            h = int(h_str)
            champ_metrics[f"val_mae_h{h}"]    = round(hd["val_mae"], 3)
            champ_metrics[f"beats_floor_h{h}"] = int(hd["beats_floor"])
        log_child(parent_run_id, "13_champion_selection", champ_metrics,
                  params={f"algo_h{h_str}": hd["algorithm"]
                          for h_str, hd in champ13["per_horizon_champion"].items()},
                  tags={"skill": "training", "iteration": "013"})

        # ------------------------------------------------------------------ #
        #  Child run: test evaluation                                         #
        # ------------------------------------------------------------------ #
        eval_metrics = {f"test_mae_h{h}": round(primary[str(h)]["mae"], 3) for h in horizons}
        eval_metrics.update({f"naive_mae_h{h}": round(naive_pri[str(h)]["naive_mae"], 3) for h in horizons})
        eval_metrics["mean_test_mae"] = test_metrics["test_mae_mean"]
        log_child(parent_run_id, "27_test_evaluation", eval_metrics,
                  tags={"skill": "evaluation", "iteration": "027",
                        "holdout": "2021-01-01 to 2021-12-31"})

        print(f"\nOK MLflow run logged.")
        print(f"   Experiment : {EXPERIMENT_NAME}")
        print(f"   Run name   : {PARENT_RUN_NAME}")
        print(f"   Run ID     : {parent_run_id}")
        print(f"   Tracking   : {DB_PATH}")
        print(f"\n   Test MAE (primary holdout):")
        for h in horizons:
            mae   = primary[str(h)]["mae"]
            naive = naive_pri[str(h)]["naive_mae"]
            beat  = "OK" if primary[str(h)]["beats_naive"] else "X"
            print(f"     h={h:2d}: {mae:6.2f}  (naive {naive:6.2f})  {beat}")


if __name__ == "__main__":
    main()
