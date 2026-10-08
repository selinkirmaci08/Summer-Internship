# Final Report — Layout and Chart Specs

The section list, what each section must contain, and the rules that keep the HTML self-contained.
Open this at step 8 of `final_report.md`, not before — steps 1–7 produce the numbers this file
arranges.

## Document shape

One vertical document, single column, max width ~1100px. **Not a slideshow** — this is read end to
end and printed. Sections in the order below; a section with no data is omitted entirely rather
than shown empty.

| § | Section | Must contain |
|---|---|---|
| 1 | **Verdict** | Three plain statements (vs AutoML, vs naive, why), each with its number and horizon count. No methodology yet. |
| 2 | **What was compared** | One row per system: label, what it is, feature count, estimator, search budget, and the exact prediction file. Plus the alignment record from step 1 — test window, `n` before and after the join, dropped horizons. |
| 3 | **Headline metrics** | Pooled table: MAE, RMSE, bias, MASE, n. Champion and naive visually distinguished. |
| 4 | **Error by horizon** | Line chart + full table. Win counts per system. |
| 5 | **Skill against naive** | MASE chart with the 1.00 reference line. |
| 6 | **Where the error comes from** | The level/shape decomposition table (MAE, bias, debiased MAE) and the per-year bias table. |
| 7 | **What the model uses** | Grouped importance chart, masking-gap table, split-count-vs-permutation rank disagreement table. |
| 8 | **Limits** | What the test set can and cannot support. Any system not scored, and why. |
| 9 | **Reproduction** | Every path consumed, library versions, the `final_report.json` location. |

Sections 3–7 each carry the two-to-four-sentence explanation required by step 7 of the skill,
placed **below** the table or chart it explains.

## Chart specs

Three charts, all inline SVG, drawn to one scale each.

**§4 — MAE by horizon.** X = lead time, log scale (the grid is dense at 1–7 days and sparse past
three months; linear x wastes 80% of the width). Y = MAE, linear, starting at a value below the
lowest point — **never at zero if that flattens the differences**, but say so in the axis label.
One line per system, 2px. Direct-label every line at its right end; this is also the accessibility
relief for any series colour below 3:1 contrast. Mark paired horizons (`91~90`) in the table, not
on the chart.

**§5 — MASE by horizon.** Same x scale. Y centred on 1.00 with the reference line drawn as a 2px
dashed rule in a neutral ink, labelled `naive parity`. Only the learned systems are plotted — the
naive system is the line at 1.00 by definition, and plotting it as a series implies otherwise.

**§7 — Grouped importance.** Horizontal bars, one per feature group, sorted by importance. Each bar
carries two segments: the sum of its members' solo importances, then the **masking gap** in a
lighter tint of the same hue. The gap is the point of the chart, so it must be visible as area, not
only as a number in a table.

### Colour

Assign in fixed order and never cycle. The champion always takes slot 1, the naive baseline always
takes slot 2 (and is dashed, so it reads as a reference rather than a competitor), AutoML arms take
slots 3+. Colour follows the system, not its rank — a system's colour is the same in every chart in
the document.

Validated 5-slot categorical palette, light / dark:

| slot | role | light | dark |
|---|---|---|---|
| 1 | champion | `#2a78d6` | `#3987e5` |
| 2 | naive (dashed) | `#eb6834` | `#d95926` |
| 3 | baseline A | `#1baf7a` | `#199e70` |
| 4 | baseline B | `#eda100` | `#c98500` |
| 5 | baseline C | `#e87ba4` | `#d55181` |

Past five systems, fold the weakest into "other baselines (worst of n)" rather than inventing a
sixth hue. Three of the light-mode hues sit below 3:1 against a light surface, which is why direct
labels and the full table are mandatory rather than optional.

## Self-contained HTML rules

The file must render with no network access, from any folder, years later.

- **No CDN, no external stylesheet, no remote font, no `<img src>` to a separate PNG.** Charts are
  inline `<svg>`, authored to the scale computed in step 8. If a raster image is genuinely needed,
  embed it as a `data:` URI.
- **Fonts** are a system stack (`ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif`),
  with a monospace stack for every figure. Digits use `font-variant-numeric: tabular-nums` wherever
  they line up in a column.
- **Both themes.** Define the complete light palette as CSS custom properties on `:root`, redefine
  only those properties under `@media (prefers-color-scheme: dark)`. Every colour comes from a
  token; a colour defined only inside the media block renders one theme's text on the other theme's
  ground. `body` sets an explicit background.
- **Print.** A `@media print` block: white ground, black ink, no page-break inside a table row or
  between a chart and its explanation.
- **Wide tables** get `overflow-x: auto` on their own container so the page body never scrolls
  sideways.
- Every chart's numbers also exist as a CSV under `artifacts/final_report/`, named for the chart.
  The HTML is a rendering; the CSVs are the record.

## Writing

Titles state findings, not topics: "Naive beats every learned system out to 91 days", not
"Horizon analysis". Plain professional register — a technical report, not a narrative. Expand each
metric the first time it appears. Never use a number in prose without saying what it means.
