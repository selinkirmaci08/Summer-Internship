# Code & Plot Conventions

Two rules that apply to **every** node that writes a script or draws a figure. They are here
rather than repeated in eight skill files because they are the same rule everywhere, and a
convention that holds in some nodes and not others is not a convention.

Both exist for the same reason: an artifact is read later, by someone who does not have the
context that produced it. A script is read as a diff against the previous iteration's script; a
plot is read after being pulled out of a report and pasted into a slide. Neither survives that
unless it carries its own identity.

---

## 1. Generated code is a cell-marked script

Every generated script — `code/iteration_<NNN>_<node_slug>.py` and its `_output.py` companion —
is a **plain `.py` file with `# %%` cell markers**. Not a notebook.

```python
# %% [markdown]
# # Iteration 011 - Training
# Fits the horizon-conditioned LightGBM on the iteration-008 feature table.
# Reads:  data/iteration_008_feature_engineering.csv
# Writes: artifacts/training/iteration_011_training_model.pkl
#         history/iteration_011_training.json

# %% Imports and configuration
import pandas as pd
...

# %% Load the feature table and check split integrity
...

# %% Fit
...
```

**Why a script and not a notebook.** The `.ipynb` format is JSON carrying embedded outputs and
execution counts, and that costs three things this pipeline depends on:

- **Diffs.** Comparing iteration 006's training script against iteration 011's is how a reader
  sees what an iteration actually changed. In notebook JSON that diff is unreadable.
- **One source of truth per number.** A notebook stores its outputs inside itself, so the file
  starts carrying results that can drift out of agreement with `artifacts/` and `history/`. In
  this pipeline every number lives in exactly one place and the script regenerates it.
- **Reproducibility.** Notebook cells can be executed out of order and leave state the file does
  not reproduce. A script cannot.

**Why cell markers anyway.** `# %%` costs nothing, is ignored by the Python interpreter, and
makes the same file openable as a notebook whenever a human wants to re-run it piece by piece:

```bash
jupytext --to notebook code/iteration_001_eda.py
```

The notebook is an **export**, never the artifact. Nothing in the pipeline reads it, and it is
not written to the run directory.

**What the markers should mark.** One cell per logical stage, not per line. A `# %% [markdown]`
header cell at the top naming the iteration, the node, what the script does, and which files it
reads and writes. Then one `# %%` cell per stage a reader would want to stop at — load, check,
transform, fit, evaluate, save. Any cell worth stopping at is worth a short label after the
marker, as above.

The existing split between the compute script and its `_output.py` companion is unchanged and
still required: the compute script produces the JSON, the output script renders from the JSON
without re-running the computation. Cell markers do not blur that boundary.

---

## 2. Every plot states what it is, inside the plot

A figure must be identifiable from the image alone, with no surrounding report, no filename and
no JSON. Three things go **in the rendered figure**:

| Element | Content | Where |
|---|---|---|
| **Title** | the plot's declared `name`, written for a human — "Validation residuals over time", not `resid_over_time` | top of the figure |
| **Plot type** | the declared `type` — "scatter with LOESS trend", "grouped bar", "horizontal bar", "boxplot by weekday" | subtitle, or a small annotation under the title |
| **Provenance** | `iteration <NNN> · <node_slug>` | small, muted, top-right or bottom-left |

The title and type in the image must be **the same strings** recorded in the artifact's JSON
entry. If the JSON says `"type": "grouped bar, validation residual mean by year x COVID era"`,
that is what the reader sees on the figure. Two different descriptions of one plot is a defect,
not a stylistic choice.

```python
# %% Plot: validation residuals over time
fig, ax = plt.subplots(figsize=(9, 4))
...
ax.set_title("Validation residuals over time", loc="left", fontsize=12)
ax.text(0, 1.02, "scatter with LOESS trend", transform=ax.transAxes,
        fontsize=9, color="#666")
fig.text(0.99, 0.01, "iteration 011 · training", ha="right",
         fontsize=7.5, color="#999")
```

**Why.** Plots are the part of a run that travels. They get pulled into slides, pasted into
messages, and looked at months later next to a plot from a different iteration that looks almost
identical. A figure whose only identity is its filename loses that identity the first time
someone drags it somewhere. Naming the plot type as well as the subject also forces the choice to
be deliberate — "grouped bar" written on the figure is a claim that a grouped bar was the right
form for this comparison.

This is required for every figure, including ones raised under `agent_initiated_findings`. Axis
labels and units remain required as before; this rule adds to them and replaces nothing.
