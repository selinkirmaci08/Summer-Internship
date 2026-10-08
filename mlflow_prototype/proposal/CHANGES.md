# Proposal: wire MLflow into the training skill

Nothing here is live. These are new files, reviewed side-by-side with your originals:

```
proposal/
  skills/training-skill/training_PROPOSED.md            <- vs skills/training-skill/training.md
  skills/training-skill/references/mlflow_logging.md      <- new file, no original
  contracts/training_PROPOSED.json                        <- vs contracts/training.json
```

If you like it, the adoption step is just renaming `training_PROPOSED.md` → `training.md` and
`training_PROPOSED.json` → `training.json` (overwriting the originals) plus keeping
`mlflow_logging.md` where it is. Nothing else in your project needs to change for this scope.

## What actually changed in `training.md`

Five small, additive touch points — no existing step was reworded or renumbered:

1. **Reference list** — one new line pointing at `references/mlflow_logging.md`, same pattern as
   every other reference file.
2. **Tools available** — `mlflow` (+ the matching model-flavor module) added to the sandbox
   library list, and `mlflow_tracking/` added to the writable-paths list.
3. **Step 14** — `state/current.json` now also carries `mlflow_parent_run_id`, and a new sub-step
   **14a** logs the iteration to MLflow, mirroring the `6a`/`6b`/`2a`/`2b` sub-step pattern already
   used elsewhere in this file. 14a just points at `mlflow_logging.md` for the mechanics, the same
   way step 5 points at `hyperparameter_tuning.md` rather than inlining it.
4. **Guardrails** — one new bullet: MLflow logging is best-effort bookkeeping and must never block
   the required outputs; a failure there is a non-blocking `issues` entry, never a stopped call.
5. **Input/output contract summary** — the new `mlflow` output block is listed alongside the
   existing ones.

## What actually changed in `contracts/training.json`

One new key added to `output`: `mlflow` (`tracking_uri`, `experiment_name`, `parent_run_id`,
`run_id`), populated here as if logging succeeded, exactly like every other field in this contract
is a worked example rather than a schema stub. The `note` at the top was extended by one sentence
to mention it.

## Design decisions baked into `mlflow_logging.md` — worth double-checking these match what you want

- **One tracking store for the whole project**, at `mlflow_tracking/mlflow.db` next to `runs/`,
  `contracts/`, `skills/` — not one per `runs/run_<run_id>` folder. This is the one piece of state
  that outlives a single run, on purpose: it's what lets iteration 014 of
  `run_20260821T130000Z` get compared against the best iteration of an earlier run tackling the
  same goal, months later. If you'd rather keep every run fully self-contained (nothing living
  outside `runs/run_<run_id>/`), this is the one decision to push back on — everything else in the
  proposal follows from it.
- **One MLflow experiment per goal** (slugified `current.goal`), not per run — same reasoning.
- **Parent run id cached in `state/current.json`**, not re-discovered by searching MLflow by name
  on every call — cheaper, and avoids ambiguity if two runs ever happened to get named alike.
- **Failure is always non-blocking.** Every one of the five touch points above treats MLflow as a
  mirror, never a dependency — consistent with what we agreed a few messages back: the JSON
  history stays the source of truth, MLflow is a queryable index on top of it.

## Deliberately left out of this proposal (from the earlier menu)

- **EDA / preprocess / feature_engineering logging** — those iterations don't have clean scalar
  metrics the way training does; extending this same convention to them is a separate design
  question (tags + artifacts, probably no metrics), not folded in here.
- **Model Registry / "Champion" staging** — this proposal logs the model as an MLflow model
  (`mlflow.lightgbm.log_model`, unlocking the registry technically) but doesn't touch
  `state/current.json`'s existing `latest_model_path` logic or wire up registry promotion. That's
  a reasonable next step once this lands, not before.
- **Agent tracing (Traces/Sessions)** — ruled out for now, since there's no coded orchestration
  harness to instrument (per your answer: skills run as manual Claude Code sessions).
