# Storage & Naming Conventions

This is the single source of truth for where any node writes a file and what it's named. Every
node builds its paths the same way — nothing is node-specific except the `node_slug`.

## Static (not per-run)

```
skills/
  <node_slug>-skill/
    <node_slug>.md          — the skill itself: decisions, procedure, guardrails
    references/             — lookup tables and catalogs the skill points at, loaded on demand
      index.md              — what each reference file holds
  shared-references/        — reference material owned jointly by more than one node
contracts/    — one <node_slug>.json per node (input/output example), written once
conventions/  — rules that hold across every node (this file, handoff_contract.md,
                code_and_plot_conventions.md)
```

**Why a skill is a directory and not one file.** A skill file mixes three kinds of content:
decisions (what to choose and why), reference (lookup tables that are identical on every run),
and examples. Only the first has to be in context for the node to reason. Splitting reference
material into `references/` keeps the skill file about decisions and lets the bulky lookup
tables be opened at the step that needs them. The skill file keeps a pointer at each extraction
point, so nothing is hidden — it is one level of indirection, not a removal.

**This layout governs static files only.** Nothing about where a *run* writes its output changes:
`runs/run_<run_id>/` and every path under it is exactly as specified in the rest of this file.

## Per-run root

```
runs/run_<run_id>/
```

- `run_id` — UTC timestamp, compact ISO 8601: `20260811T143200Z`. One folder per full
  execution, from Start to Done (or to an abandoned/paused run).

## Node slug vocabulary

Use exactly these strings everywhere a node is named in a path — never a variant, abbreviation,
or different casing:

```
eda | preprocess | feature_engineering | train | diagnosis | end_model
```

## Global iteration counter

One counter per run, incremented every time *any* node executes — not a separate counter per
node. This means iteration numbers are globally chronological and a file listing sorts into the
exact order things happened.

- Zero-padded to 3 digits: `001`, `002`, ... `099`, `100`. Always 3 digits even if the run never
  reaches 100 — this guarantees correct alphabetical sort regardless of run length.

## The path stem

Every file produced by a node's execution shares one stem:

```
iteration_<NNN>_<node_slug>
```

e.g. `iteration_004_train`. This stem is reused, with a different folder and extension, across
every artifact type that iteration produces:

| What | Path template | Example |
|---|---|---|
| Structured output (always) | `history/<stem>.json` | `history/iteration_004_train.json` |
| Generated code (always) | `code/<stem>.py` | `code/iteration_004_train.py` |
| Processed/engineered dataset (Preprocess, Feature Engineering) | `data/<stem>.csv` | `data/iteration_002_preprocess.csv` |
| Trained model object (Train; also End model, for the champion refit on train + validation) | `models/<stem>.pkl` | `models/iteration_004_train.pkl`, `models/iteration_016_end_model.pkl` |
| Plots / non-JSON artifacts | `artifacts/<node_slug>/<stem>_<descriptive_slug>.<ext>` | `artifacts/eda/iteration_001_eda_hist_blood_sugar.png` |
| Full correlation matrix, other large tabular artifacts | `artifacts/<node_slug>/<stem>_<descriptive_slug>.csv` | `artifacts/eda/iteration_001_eda_full_corr_matrix.csv` |
| Human-readable execution report | `reports/<stem>.md` | `reports/iteration_001_eda.md` |

`<descriptive_slug>` is a short, lowercase, underscore-separated description of the specific
plot/file content (e.g. `hist_blood_sugar`, `corr_heatmap`, `segment_recall_breakdown`) — this is
what keeps multiple artifacts from the same iteration distinguishable, and keeps artifacts from
different iterations of the same node from colliding.

## Non-iteration files (one per run, always overwritten to latest)

```
state/current.json   — the Planner's current snapshot, overwritten every iteration
report/final_summary.md — written once, by End model, at the end of the run
```

`current.json` is never versioned by iteration — it's always "the latest state." History is
what's versioned; current is what's live.

**Note the naming is deliberately distinct:** `report/` (singular) holds the one final summary
End model writes at the end of a run. `reports/` (plural, see the path template table above)
holds one human-readable file per iteration, written by whichever node just ran — every node
produces one of these, not just End model. Don't confuse the two folders when implementing this.

## Full example trace

```
runs/run_20260811T090000Z/
  state/current.json
  history/
    iteration_001_eda.json
    iteration_002_preprocess.json
    iteration_003_feature_engineering.json
    iteration_004_train.json
    iteration_005_diagnosis.json
  reports/
    iteration_001_eda.md
    iteration_002_preprocess.md
    iteration_003_feature_engineering.md
    iteration_004_train.md
    iteration_005_diagnosis.md
  code/
    iteration_001_eda.py
    iteration_002_preprocess.py
    iteration_003_feature_engineering.py
    iteration_004_train.py
  data/
    iteration_002_preprocess.csv
    iteration_003_feature_engineering.csv
  models/
    iteration_004_train.pkl
  artifacts/
    eda/
      iteration_001_eda_hist_blood_sugar.png
      iteration_001_eda_class_balance.png
      iteration_001_eda_full_corr_matrix.csv
    preprocess/
      iteration_002_preprocess_blood_sugar_before_after.png
    train/
      iteration_004_train_feature_importance.png
  report/
    final_summary.md
```

## Read access for any node

Every node's input includes one field for prior context: `context`, a list of entries. Each
entry is tagged with why it's there, so the shape is the same everywhere even though the reason
for including something varies:

```json
"context": [
  {
    "type": "previous_step",
    "source_node": "diagnosis",
    "source_iteration": 6,
    "content": { "...": "diagnosis's reasoning/recommendation fields only" }
  },
  {
    "type": "planner_selected",
    "source_node": "eda",
    "source_iteration": 1,
    "reason": "flags diagnosis_code as high cardinality — relevant to choosing an encoding method",
    "content": { "...": "whatever specific fields the Planner judged relevant" }
  }
]
```

**`previous_step` entry — always present, automatic.** Whichever node ran immediately before
this call (not necessarily the one a step conceptually depends on most), with only its
reasoning/recommendation-type fields included — e.g. EDA's `recommended_strategy`, `key_findings`,
`limitations`; Diagnosis's `diagnosis` and `hypothesis`; Train's `reasoning` and
`metrics.metric_assessment`; Evaluate's `stopping_check.reasoning`. Never the node's full stats,
correlations, or artifact paths — those are excluded to keep this entry cheap. This entry always
reflects exactly the one immediately preceding step, never an accumulation across several past
steps — growing it over the course of a run would reintroduce the same unbounded-context problem
that `current` vs. `history` was designed to avoid.

**`planner_selected` entries — zero or more, added when the Planner judges something specific
from further back is relevant.** Unlike the `previous_step` entry, these can pull from any past
iteration and any field, not just reasoning — e.g. a specific flag or missingness note a node
needs regardless of how many steps ago it was produced. Each one carries a `reason` explaining
why it was included. When curating one of these, the Planner must copy the data's actual shape as
it exists in the source — never reshape or flatten it along the way (e.g. a list of
`{name, note}` objects must stay a list of objects, not get flattened into plain strings).
Reshaping on the way through is a common source of silent inconsistency between what a node
originally reported and what a later node receives.

Beyond `context`, no node is limited to only what it was given: every node has read-only access
to the whole run folder, specifically `state/current.json` and every file under `history/`,
regardless of which node produced them or how long ago. This matters because the Planner can
route to any node at any point in the loop — Preprocess might run right after EDA, or at
iteration 7 after Feature Engineering, Training, and Diagnosis have each already run once.

**This broader access should stay the exception, not the default.** Reading a full past iteration
costs meaningfully more than what's already sitting in `context`, and that cost compounds across
every node in the loop. The order of preference is: (1) `context` (both entry types), (2) a
targeted lookup in `history/` for one specific iteration if there's a clear reason to believe
it's needed — never an open-ended scan through many past iterations "just to check."

**How a node looks something up:**

1. Check whether what it needs is already in `context` (either entry type). This is the common
   case and needs no extra lookup.
2. Otherwise, list `history/` — filenames sort chronologically by construction (the zero-padded
   `iteration_<NNN>_<node_slug>` stem), so finding "the most recent EDA output" or "everything
   Diagnosis has concluded so far" is a filter on the filename pattern, not a search — then read
   only the specific file(s) actually needed, never an open-ended scan.

**Boundaries on this access:**

- Read-only. A node never edits another iteration's `history/` file — history is append-only and
  immutable once written, by any node, including the one that originally wrote it.
- A node writes its own `history/<stem>.json` at the end of its own turn, and nothing else's.
- Pulling extra context this way should be the exception, not the default — a node that reads
  most or all of `history/` on every call has defeated the purpose of the Planner curating a
  focused input in the first place, and will bloat its own context unnecessarily.

## Confidence on reasoning fields

Every field that carries a `reasoning`, a `note` explaining a judgment call, or a
`suggested_action` must also carry a sibling `confidence`: `low`, `medium`, or `high`. This
applies to every node, not just one — any place a node is interpreting or recommending rather
than reporting a directly measured number. Use qualitative levels, not a percentage — a number
like "73% confident" implies a precision these judgments don't actually have. Raw measured stats
(correlations, counts, percentages, quartiles) don't need a confidence — they're computed, not
inferred; confidence is only for the interpretation layered on top of them.

## Rule for a node writing its own output

Every node's output JSON must set its own `*_path` fields (e.g. `processed_dataset_path`,
`model_path`, `artifact_paths.plots[].path`) using this exact scheme — a node never invents its
own naming pattern. If a node needs to know its own iteration number, it reads it from the
Planner's call (the Planner owns and increments the global counter, then passes `iteration` as
part of every node's input).