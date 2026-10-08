# MLflow Logging Reference

Source: this pipeline's own MLflow prototype (`mlflow_prototype/` at the project root, produced
while evaluating this integration against `runs/run_20260821T130000Z`) — not third-party docs,
since this is pipeline-specific convention rather than general ML knowledge.

## Purpose

Mirror this iteration's params/metrics/artifacts into MLflow so they're queryable and comparable
across iterations *and across separate pipeline runs* tackling the same goal. The file-based
`history/`/`artifacts/` output remains the source of truth — MLflow is a queryable index on top of
it, never a replacement. If MLflow logging fails for any reason, that failure must never block
writing the required outputs (history JSON, report, state update): record it as a non-blocking
`issues` entry instead (see Guardrails in `training.md`).

## Tracking store location (shared across every run)

Use one tracking store per project, not one per `runs/run_<run_id>` folder, so iterations from
different pipeline runs tackling the same modeling problem stay comparable over time:

```
tracking_uri = f"sqlite:///{PROJECT_ROOT}/mlflow_tracking/mlflow.db"
```

`PROJECT_ROOT` is the folder containing `runs/`, `contracts/`, `skills/` — the same root every node
already resolves paths against. Create `mlflow_tracking/` if it doesn't exist yet; this is the one
piece of state that persists outside any single `runs/run_<run_id>` folder.

## Experiment naming

One MLflow experiment per modeling problem, not per run, so retrains of the same goal across
multiple `runs/run_<run_id>` folders land in the same experiment:

```
experiment_name = slugify(current.goal)
```

Read `current.goal` from `state/current.json`. If the goal changes meaningfully mid-run (rare),
keep the experiment name fixed to whatever it was when the run started.

## Run nesting, and how to find the parent run again

- One **parent run** per `runs/run_<run_id>`, named `run_<run_id>`, created by the *first* Training
  call in that run.
- One **nested child run** per training iteration, named `iteration_<NNN>_training`, created fresh
  every Training call — never reused, even in `incremental` mode.

Don't search MLflow for the parent run by name on every call — cache it instead. The first Training
call in a run creates the parent run and writes its id to a new `state/current.json` field,
`mlflow_parent_run_id`. Every subsequent Training call in the same run reads that field and resumes
the parent run (`mlflow.start_run(run_id=mlflow_parent_run_id)`) before starting its own nested
child run underneath it. If `mlflow_parent_run_id` is absent (first call, or a prior call's MLflow
step failed and left it unset), create a new parent run and write the id.

## What to log, mapped from this skill's own output fields

| MLflow call | Source field(s) in this call's output |
|---|---|
| `log_params` | `hyperparameters.*.value` (flattened — the paired `reasoning` stays in tags/history only, never as a param value), `training_summary.n_features`, `training_summary.n_train_rows`, `training_summary.n_val_rows`, `hyperparameter_search.method` / `.n_iter` (only if `hyperparameter_search` is not null) |
| `log_metrics` | every value under `metrics.train.*` and `metrics.validation.*`, prefixed `train_` / `validation_`; `overfitting_check.ratio` as `overfit_ratio`; `baseline_model.val_<primary_metric>` as `baseline_val_<primary_metric>` (only present when `mode == "initial"`) |
| `set_tags` | `iteration_id`, `mode`, `mode_reasoning` (truncate to ~250 chars), `model_class`, `feature_subset_type`, `deployment_scenario`, `overfitting_check.assessment`, `vs_prior_iteration.note` (truncate to ~250 chars; only present in `incremental` mode) |
| `log_artifact` (under `artifact_path="diagnostics"`) | every path in `artifact_paths.plots[]` — both the `.png` and its paired `.csv` |
| `log_model` (flavor matches `model_class`: `mlflow.lightgbm` / `mlflow.xgboost` / `mlflow.sklearn`) | the fitted estimator object still in memory in step 6 — log it directly there, don't reload it from the `.pkl` afterward |

Only log numeric values as metrics. Never put free-text reasoning into `log_metric` — it belongs
in tags (truncated) or stays in `history/` only.

## Guardrails specific to this reference

- Never let steps 1–13 of `training.md` depend on anything MLflow-side succeeding — this logging
  happens after the model and metrics already exist, purely as an additional mirror.
- Wrap the whole block in one `try/except`. On any exception, add one `issues` entry
  (`"mlflow logging failed: <message>"`, non-blocking) and continue to step 15 as normal. Leave
  `mlflow_parent_run_id` in `state/current.json` unset (or unchanged from its prior value) if the
  parent-run step itself failed — never write a run id that doesn't actually exist in the store.
- Don't invent a different tracking-URI or experiment-naming scheme per node — if this is ever
  extended to Preprocess/EDA/Feature Engineering, they should follow this same `PROJECT_ROOT` /
  experiment-name convention rather than each choosing their own (open question — see
  `mlflow_prototype/proposal/CHANGES.md`).
