---
name: report-generator-skill
description: Reads a completed skill's history JSON and artifact images, then generates a self-contained, element61-branded HTML slideshow report (one dashboard slide per major finding group), writes it to reports/, updates report_html in the history JSON, and regenerates index.html. Triggered immediately after any EDA, Domain Research, Preprocess, Feature Engineering, Training, or Diagnosis call. Re-runnable at any time.
---

# Report Generator Skill

## Purpose

Produce a rich, self-contained HTML **slideshow** from a completed skill's history JSON and the
PNG artifacts it wrote to disk. One report = one slide deck: full-height slides in a vertical
scroll-snap flow (scroll, arrow keys, dot nav, or prev/next buttons — never a hard-cut fixed-height
track), a real Cover and Agenda opening it, each slide clearly section-labeled — numbers as tiles
and badges, plots shown with a real explanatory paragraph plus a "Full analysis" disclosure for
the exhaustive version, and the single most decision-relevant fact on each slide made visually
unmissable. Wording throughout is plain and professional — a consulting deliverable, not a
narrated story: titles state findings directly, no taglines, wordplay, or dramatized framing.
Every section must render real
numbers from the JSON; every image must be verified on disk before being included.

This skill reads and renders. It never re-runs analysis or invents a number, a finding, or a fact
that isn't in the JSON. Within that constraint, it **is** allowed — and expected — to choose and
condense: pick which numbers/plots/tables earn a slide's limited space, and write a short,
faithful synthesis of a long `reasoning`/`insight`/`narrative` field instead of quoting it in
full. A summary must not invent, soften, or drop a material qualifier (a confidence level, a
caveat, a "but") — compressing "left-skewed, same-day operational measure, not available at
forecast time, correlates 0.606 with the target but unusable directly, propose a rolling-mean
proxy" into "Leaky same-day measure — use a rolling-mean proxy instead" is the intended kind of
compression; dropping the leakage reason entirely would not be.

**Content priority: numbers and plots lead, but a real explanatory paragraph is the default, not
a one-liner.** Every slide's numbers (KPI tiles, stat chips, tables) come first and are never
diluted. Below them, the *default-visible* prose for a plot or a finding is a genuine 2-4 sentence
paragraph in the writer's own voice — what the shape/number means, why it matters, what it implies
— not squeezed to a single clause. What's compressed is not the visible text's substance but its
*exhaustiveness*: the full four-layer analysis (`eda.md` step 10's shape → status → mechanism →
action, or the equivalent for other skills) sits one click away behind a **"Full analysis"**
toggle (see the `.details-btn`/`.details-panel` component below) — expanding in place, not a
one-line takeaway hiding almost everything. A reader skimming gets a complete thought per slide
without clicking anything; a reader who wants the whole reasoning chain clicks once.

---

## Inputs

- `history_json_path` — absolute path to the skill's history JSON
- `run_dir` — root of the run (e.g. `runs/run_20260820T110000Z/`)

## Tools

- Read access to history JSON and all files under `run_dir/history/`
- Read access to `run_dir/artifacts/**` — discover every PNG that actually exists
- Write access to `run_dir/reports/`
- Write access to the history JSON (update `report_html` only)

---

## File Placement Constraints

All file paths are governed by `conventions/naming_conventions.md`. The rules below are the
binding subset for this skill. A wrong path is a hard error — do not place a file in the wrong
location and note it as a caveat.

### Stem convention

Every file produced in iteration NNN of node SKILL shares one stem:

```
iteration_NNN_SKILL
```

`NNN` is zero-padded to 3 digits (`001`, `002`, … `099`, `100`).
`SKILL` is the exact node slug — never a variant or abbreviation:

```
eda | domain_research | preprocess | feature_engineering | training | diagnosis | end_model
```

(Corrected from an earlier draft of this file, which said `train` — every real run and
`training.md` itself use `training`. `domain_research` was also missing from this list despite
already being a real, reported-on skill.)

### Files this skill writes

| File | Required path (relative to run root) |
|---|---|
| Skill HTML report (slideshow) | `reports/iteration_NNN_SKILL.html` |
| Run navigator (homepage) | `reports/index.html` |
| History JSON field | `history/iteration_NNN_SKILL.json` — update `report_html` field only |

### Files the preceding skill must have written (pre-flight check)

Before generating the report, verify that the preceding skill placed every output correctly.
If a file is missing or misplaced, log the violation in the console output and render a visible
warning banner on the first slide of the generated report — do not silently ignore it.

| Artifact type | Required path (relative to run root) | Skills that write it |
|---|---|---|
| History JSON | `history/iteration_NNN_SKILL.json` | all |
| Generated code | `code/iteration_NNN_SKILL.py` | all |
| Planner state snapshot | `state/current.json` (single file, always overwritten to latest) | planner / any skill |
| PNG plots | `artifacts/<node_slug>/iteration_NNN_SKILL_<descriptive_slug>.png` | all |
| Large tabular / JSON artifacts | `artifacts/<node_slug>/iteration_NNN_SKILL_<descriptive_slug>.json` | all |
| Processed dataset | `data/iteration_NNN_SKILL.csv` | preprocess, feature_engineering |
| Trained model | `artifacts/training/iteration_NNN_SKILL_model.pkl` | training |

`<node_slug>` in artifact paths equals the `SKILL` slug above (`eda`, `preprocess`, etc.).

Common violations to catch and flag:

- Code file in run root (`run_eda.py`) instead of `code/iteration_001_eda.py`
- Intermediate computed stats JSON in run root instead of `artifacts/eda/iteration_001_eda_computed_stats.json`
- `state/current.json` absent entirely
- `data/` folder absent after Preprocess or Feature Engineering ran
- Any `<img src>` pointing to an absolute path or a path that does not exist on disk

### Paths inside HTML

All `<img src="...">` must be relative from `reports/` to the artifact. There is exactly one
correct prefix for every skill's images:

```
../artifacts/eda/                  ← from reports/iteration_001_eda.html
../artifacts/domain_research/      ← from reports/iteration_001b_domain_research.html
../artifacts/preprocess/           ← from reports/iteration_002_preprocess.html
../artifacts/feature_engineering/  ← from reports/iteration_003_feature_engineering.html
../artifacts/training/             ← from reports/iteration_004_training.html
../artifacts/diagnosis/            ← from reports/iteration_NNN_diagnosis.html
```

Absolute paths (`C:/…`, `file:///…`) are forbidden. The page must open under `file://` with no
server.

### Write scope

This skill writes **only**:
- `reports/iteration_NNN_SKILL.html`
- `reports/index.html`
- The single field `report_html` in `history/iteration_NNN_SKILL.json`

It must never create or modify any file outside `reports/`, except that one JSON field.

---

## Design System

The source of truth for every color, font, spacing, radius, and shadow value is the real brand
package at `element61 Design System/colors_and_type.css` in the project root — **not** a palette
invented in this file. Read that file and copy its `:root` custom properties verbatim into the
report's inline `<style>` block (reports are self-contained, so the tokens are copy-pasted, not
`<link>`ed — the only external dependency stays the Google Fonts `@import` already at the top of
that same CSS file). Do not hand-roll substitute hex values — if a color is needed that isn't in
that file, treat it as a documented extension (see "Accent colors" below), not a guess.

**Reference copy of the tokens actually used by this skill** (keep in sync with
`colors_and_type.css` — that file wins on any conflict):

```css
@import url("https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Open+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap");

:root {
  --e61-orange-red: #FC5B00;
  --e61-orange:     #FC9808;
  --e61-yellow:     #FCBA08;
  --e61-magenta:    #9D1D56;
  --e61-purple:     #301D56;

  --e61-ink:        #1A1230;
  --e61-ink-2:      #301D56;
  --e61-ink-3:      #5B5070;
  --e61-ink-4:      #8B8298;
  --e61-line:       #E6E2EC;
  --e61-line-2:     #D8D2E0;
  --e61-surface:    #FFFFFF;
  --e61-bg:         #FAF8F5;
  --e61-bg-warm:    #FFF6EC;
  --e61-bg-cool:    #F2EEF6;

  --e61-success: #2F8F5C;
  --e61-warning: #FCBA08;
  --e61-danger:  #C0294A;
  --e61-info:    #2F6BB3;

  --font-display: "Montserrat", "Helvetica Neue", system-ui, sans-serif;
  --font-body:    "Open Sans", "Helvetica Neue", Arial, system-ui, sans-serif;
  --font-mono:    "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;

  --space-1:4px; --space-2:8px; --space-3:12px; --space-4:16px; --space-5:24px;
  --space-6:32px; --space-7:48px;

  --radius-sm:4px; --radius-md:8px; --radius-lg:14px; --radius-xl:22px; --radius-pill:999px;

  --shadow-1: 0 1px 2px rgba(48,29,86,.06), 0 1px 1px rgba(48,29,86,.04);
  --shadow-2: 0 4px 16px rgba(48,29,86,.08), 0 1px 2px rgba(48,29,86,.04);
  --shadow-3: 0 10px 28px rgba(48,29,86,.12), 0 2px 6px rgba(48,29,86,.06);
  --shadow-focus: 0 0 0 3px rgba(252,91,0,.28);

  --ease-standard: cubic-bezier(0.2, 0.7, 0.2, 1);
  --dur-fast: 120ms;
  --dur-base: 200ms;
}
```

### Accent colors (per skill)

Four of the six accents are the design system's own base-palette tokens; two (`diagnosis`,
`domain_research`) are documented extensions beyond the 5-color base palette, kept because six
skill types need six distinguishable accents and the base palette only defines five hues:

```
eda                  → var(--e61-orange-red)   #FC5B00   (base palette)
preprocess           → var(--e61-magenta)      #9D1D56   (base palette)
feature_engineering  → var(--e61-info)         #2F6BB3   (base palette, semantic "info")
training             → var(--e61-success)      #2F8F5C   (base palette, semantic "success")
diagnosis            → #0E7C86  (teal)                    — EXTENSION, not in colors_and_type.css
domain_research      → #5C4EA8  (medium purple)            — EXTENSION, not in colors_and_type.css,
                                                              matches what domain_research.md's own
                                                              step 9 already specifies
```

Each report sets `--accent: <its color>` once at the top of its `<style>` block and every
component below reads `var(--accent)` rather than hardcoding a hex — this is what lets one shared
CSS/JS spec serve all six skill types.

### Logo

Use `element61 Design System/assets/logo-mark-white.svg` (inlined as a data URI or raw `<svg>`,
never an external file reference) as the small mark in the top bar. Never redraw it.

---

## Slide Deck Shell (all skill types)

**Mechanism: vertical scroll-snap, not a fixed-height slide track.** Every report is a sequence of
full-height `<section>`s (`.cover` for the first, `.slide` for the rest) placed in normal document
flow, `scroll-snap-align:start` on each. The reader scrolls (wheel, trackpad, arrow keys, dot
nav, prev/next buttons) and the browser snaps to the nearest slide boundary — this is a real
`<body>` scroll, so nothing can ever be clipped or unreachable the way a fixed-height flex track
risked: if one slide's content runs a little tall on a given device/zoom, the reader simply
scrolls a little further before the snap catches the next slide. No `overflow:hidden` anywhere,
no per-slide height budgeting arithmetic — the browser's native scroll handles it.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>element61 · [Skill] · Iteration NNN</title>
  <style>/* tokens (above) + shell + component CSS — no external files but Google Fonts */</style>
</head>
<body>
  <div class="topbar"><div class="topbar-fill" id="topbarFill"></div></div>
  <div class="dotnav" id="dotnav"></div>
  <div class="navbar">
    <button id="prevBtn" title="Previous (←)">‹</button>
    <span class="counter" id="counter">1 / N</span>
    <button id="nextBtn" title="Next (→)">›</button>
  </div>

  <!-- SLIDE 0 — COVER (every report opens with one) -->
  <section class="cover slide-item" data-title="Cover"> ... see Cover Slide below ... </section>

  <!-- SLIDE 1 — AGENDA (every report's 2nd slide) -->
  <section class="slide slide-item" data-title="Agenda"> ... see Agenda Slide below ... </section>

  <!-- one <section class="slide slide-item" data-title="..."> per entry in that skill's slide
       table below. Add class "dark" or "warm" (see Emphasis Slides) on the ONE slide per
       report that carries the single most significant finding — never more than one. -->
  <section class="slide slide-item" data-title="Key Findings"> ... </section>

  <script>/* deck engine (verbatim, see below) + per-skill JS (HISTS grid, etc.) */</script>
</body>
</html>
```

**Top bar** — a thin fixed progress rail, not the old header band (fixed, all reports):
```css
.topbar{position:fixed;top:0;left:0;right:0;height:3px;background:var(--e61-line);z-index:201}
.topbar-fill{height:100%;background:var(--e61-gradient-warm, linear-gradient(135deg,var(--e61-yellow) 0%,var(--e61-orange) 50%,var(--e61-orange-red) 100%));
  width:0;transition:width .25s}
```

**Dot nav** (right edge, one dot per slide including cover) and **navbar** (bottom-center pill,
prev/next + counter):
```css
.dotnav{position:fixed;right:20px;top:50%;transform:translateY(-50%);display:flex;
  flex-direction:column;gap:9px;z-index:200}
.dotnav button{width:9px;height:9px;border-radius:50%;border:none;background:var(--e61-line-2);
  cursor:pointer;padding:0;transition:background .2s,transform .2s}
.dotnav button.active{background:var(--e61-orange-red);transform:scale(1.5)}
.dotnav button:hover{background:var(--e61-orange)}
.navbar{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);display:flex;
  align-items:center;gap:14px;background:rgba(26,18,48,.88);backdrop-filter:blur(6px);
  border-radius:999px;padding:9px 18px;z-index:200;box-shadow:var(--shadow-3)}
.navbar button{background:none;border:none;color:#fff;font-size:1rem;cursor:pointer;width:30px;
  height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  transition:background .15s}
.navbar button:hover{background:rgba(255,255,255,.15)}
.navbar button:disabled{opacity:.3;cursor:default}
.navbar .counter{font-family:var(--font-mono);font-size:.72rem;color:rgba(255,255,255,.85);
  min-width:52px;text-align:center}
```

**Raw-JSON / back-to-index links** move into the footer of the last slide (see per-skill
breakdowns) rather than a persistent header — the top bar's job now is only the progress rail.

---

## Deck Engine (copy verbatim into every report)

```css
html,body{margin:0;padding:0;background:var(--e61-bg);color:var(--e61-ink-2);
  font-family:var(--font-body);line-height:1.6;
  scroll-snap-type:y proximity;height:100%;overflow-y:auto;scroll-behavior:smooth}
.slide{min-height:100vh;scroll-snap-align:start;display:flex;flex-direction:column;
  justify-content:center;padding:88px 64px 96px;position:relative;
  border-bottom:1px solid var(--e61-line);box-sizing:border-box}
.slide-inner{max-width:1180px;margin:0 auto;width:100%}
.slide-num{position:absolute;top:28px;left:64px;font-family:var(--font-mono);font-size:.72rem;
  letter-spacing:.1em;color:var(--e61-ink-4);text-transform:uppercase}
.slide-tag{position:absolute;top:28px;right:64px;font-family:var(--font-mono);font-size:.72rem;
  color:var(--e61-ink-4)}
@media(max-width:700px){.slide{padding:72px 24px 80px}.slide-num,.slide-tag{left:24px;right:24px}}
```

```javascript
const slides = Array.from(document.querySelectorAll('.cover, .slide'));
const dotnav = document.getElementById('dotnav');
slides.forEach((s,i)=>{
  const dot = document.createElement('button');
  dot.title = s.dataset.title || ('Slide '+(i+1));
  dot.onclick = ()=> s.scrollIntoView({behavior:'smooth'});
  dotnav.appendChild(dot);
});
const dots = Array.from(dotnav.children);
const counter = document.getElementById('counter');
const topbarFill = document.getElementById('topbarFill');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
let current = 0;

function setActive(idx){
  current = idx;
  dots.forEach((d,i)=> d.classList.toggle('active', i===idx));
  counter.textContent = (idx+1) + ' / ' + slides.length;
  topbarFill.style.width = (((idx+1)/slides.length)*100) + '%';
  prevBtn.disabled = idx===0;
  nextBtn.disabled = idx===slides.length-1;
}
setActive(0);

// IntersectionObserver drives the active-slide state from real scroll position — no fixed-height
// bookkeeping. threshold 0.55 needs a slide to be just past half-visible before it "wins"; a slide
// taller than the viewport still resolves correctly once its dominant portion is in view.
const io = new IntersectionObserver((entries)=>{
  entries.forEach(e=>{
    if(e.isIntersecting && e.intersectionRatio>0.55){
      const idx = slides.indexOf(e.target);
      if(idx>-1) setActive(idx);
    }
  });
}, {threshold:[0.55]});
slides.forEach(s=>io.observe(s));

function goTo(idx){
  idx = Math.max(0, Math.min(slides.length-1, idx));
  slides[idx].scrollIntoView({behavior:'smooth'});
}
prevBtn.onclick = ()=> goTo(current-1);
nextBtn.onclick = ()=> goTo(current+1);
document.addEventListener('keydown', e=>{
  if(['ArrowDown','ArrowRight','PageDown',' '].includes(e.key)){ e.preventDefault(); goTo(current+1); }
  if(['ArrowUp','ArrowLeft','PageUp'].includes(e.key)){ e.preventDefault(); goTo(current-1); }
});
```

---

## Cover Slide (every report opens with one)

The reader's first impression should read like the cover of a real deck, not a dashboard's
top-left corner — but the background stays light, the same calm surface as every other slide.
Distinction comes from generous whitespace, larger type, and a thin accent-colored top bar, never
from a full-bleed dark or saturated background. The wording stays plain and professional
throughout: a factual headline describing what this iteration found or did, not a tagline, a pun,
or marketing-style copy.

```html
<section class="cover slide-item" data-title="Cover">
  <div class="cover-inner">
    <div class="logo">element61 &nbsp;·&nbsp; pipeline</div>
    <span class="eyebrow">[SKILL] REPORT &nbsp;·&nbsp; ITERATION NNN &nbsp;·&nbsp; [run_id]</span>
    <h1>[One factual sentence: what this iteration analyzed/produced — e.g. "Exploratory Data
      Analysis — Daily Bread Demand, Branch 01". No wordplay, no rhetorical framing.]</h1>
    <p class="sub">[1-2 plain sentences: scope, row count, date range — whatever grounds the
      report — plus one line noting every number that follows traces to the history JSON, nothing
      here is invented for presentation.]</p>
    <div class="kpi-row kpi-row-lt"><!-- 4-5 top-line KPIs, light surface tiles --></div>
  </div>
</section>
```

**Title-writing rule (applies everywhere in the deck, not just the cover):** every heading —
cover `h1`, slide `h2`, `.slide-tag`, agenda card labels — is a plain, descriptive statement of
what that slide contains or found, written the way a consulting deliverable would write it.
Specific numbers and findings are good ("A ~30% Level Shift Between Two Measurement Regimes" is
fine, that's a real finding stated plainly); rhetorical devices are not — no puns, no
"plot twist"/"headline"/"what next"-style framing devices, no questions-as-titles, no exclamation
marks, no informal asides. If a title reads like something a journalist or copywriter would write
rather than an analyst, rewrite it.

```css
.cover{min-height:100vh;scroll-snap-align:start;background:var(--e61-bg);color:var(--e61-ink);
  display:flex;flex-direction:column;justify-content:center;padding:64px 64px 64px 68px;
  position:relative;overflow:hidden;box-sizing:border-box;border-left:6px solid var(--accent)}
.cover-inner{position:relative;max-width:980px;margin:0 auto;width:100%}
.cover .logo{font-family:var(--font-display);font-weight:800;font-size:1rem;letter-spacing:.02em;
  color:var(--accent);margin-bottom:40px}
.cover h1{font-family:var(--font-display);font-weight:700;font-size:2.6rem;color:var(--e61-purple);
  line-height:1.15;margin:0 0 18px}
.cover .sub{font-size:1.05rem;color:var(--e61-ink-3);max-width:680px;margin-bottom:34px}
@media(max-width:700px){.cover h1{font-size:1.9rem}}
```

`.eyebrow` already renders in `var(--accent)` (see Dashboard Component Vocabulary below) — no
extra dark-background override is needed since the cover no longer uses one. The 6px left accent
border is the cover's one distinguishing mark; everything else reads exactly like the rest of the
deck, just larger and more spacious.

---

## Agenda Slide (every report's 2nd slide)

A one-screen table of contents so the reader knows what's coming and can jump straight to it —
reuse `.card` in a `.grid-3`, one card per upcoming slide (or group of slides): the slide number,
a plain section name matching its `.slide-tag` (e.g. "Key Findings", "Missingness"), and one
short factual line underneath describing its content. This is a judgment call on grouping per
skill, grounded in that run's actual slide list — never a fixed template of card text, since the
slide count and content vary call to call. Wording follows the same rule as everywhere else:
literal section names, not evocative or narrative ones.

```html
<div class="grid-3">
  <div class="card"><b class="mono" style="color:var(--accent)">03</b><br><b>Key Findings</b><br>
    <span style="font-size:.85rem;color:var(--e61-ink-3)">Five findings from this iteration's analysis</span></div>
  <!-- one card per remaining slide/group, numbered to match slide-num -->
</div>
```

---

## Emphasis Slides (`dark` / `warm`, at most one per report)

Give the **single most significant finding** in the run — the one thing that, if the reader saw
only this slide, would tell them what mattered most (a regime break, a blocking leakage verdict,
a stop/continue call) — a visually distinct but still calm treatment: a light tinted background
and a thick accent-colored left border, not a full-bleed dark or saturated one. Every other slide
sits on plain `--e61-bg`; this slide sits on a faint tint of the same palette. It's a deliberate
rarity used for emphasis, not decoration — the wording on it follows the same plain, factual rule
as every other slide, only the tint and border signal importance. (The class names `dark`/`warm`
are kept from an earlier, more saturated version of this component — both are light today.)

```css
.slide.dark{background:var(--e61-bg-cool);border-left:6px solid var(--e61-purple)}
.slide.dark h1,.slide.dark h2,.slide.dark h3,.slide.dark h4{color:var(--e61-purple)}
.slide.warm{background:var(--e61-bg-warm);border-left:6px solid var(--e61-orange-red)}
.slide.warm h1,.slide.warm h2,.slide.warm h3{color:var(--e61-ink)}
```

Pick `dark` (cool purple tint) for a data-quality finding that needs attention (a regime break, a
leakage verdict) and `warm` (orange tint) for a forward-looking recommendation (strategy, next
steps). This mirrors the Training/Diagnosis accent split conceptually — `dark` signals a finding
the reader must account for, `warm` signals a recommendation going forward. Because both are now
light, no `.kpi`/`.details-btn`/`.card` variant is needed for them — every component already
looks right on a light background regardless of which slide it's on (see the simplified component
CSS below, no more `:not(.dark):not(.warm)` branching).

---

## Layout Rule: Presentation Feel, Not a Uniform Grid

A slide someone actually presents from doesn't tile every element at the same size — the one
number or plot that matters most is bigger, everything supporting it is visibly secondary. Equal-
size grids (`.grid-2`, `.grid-3`, `.grid-fill`) are still the right default for genuinely
parallel items (a table's rows, a set of equally-weighted feature cards) — but a slide's *primary*
content should almost never share a size class with its supporting content. Concretely:

- **Asymmetric splits over equal columns** when one side is the point and the other is context —
  e.g. a 2fr/1fr grid (`grid-template-columns:2fr 1fr`) with the lead plot or the biggest number
  on the wide side, supporting stats/chips on the narrow side, instead of two equal-width cards
  that imply equal importance.
- **`.plot-card` size is itself this principle** — the plot IS the finding on most slides; give it
  room (see the Plot Card component below) rather than shrinking it to fit beside equal-weight text.
- Vary vertical rhythm too — a slide doesn't need every section to start with an `<h3>` at the
  same weight; `.slide-inner` centers vertically by default (`.slide{justify-content:center}`), so
  a slide with little content already reads spacious rather than cramped at the top.
- Every slide carries `.slide-num` (top-left, "04 / 15") and `.slide-tag` (top-right, a short
  plain section label — e.g. "Target Analysis", "Data Quality", "Recommendations"). This is a
  section marker, the way a slide deck's running header names the current chapter — not a
  narrative device. Keep it literal: the label should read like a table-of-contents entry, never
  a headline, a question, or a hook.

---

## Dashboard Component Vocabulary (shared across every skill)

Copy this block verbatim into every report's `<style>` tag. Never invent class names — use
exactly these. Extend only by adding new rules; never change existing ones.

```css
*{box-sizing:border-box}
h1,h2,h3,h4{font-family:var(--font-display);color:var(--e61-ink);margin:0 0 .5em;font-weight:700;text-wrap:balance}
p{margin:0 0 1em}
a{color:var(--e61-orange-red)}
.mono{font-family:var(--font-mono)}
img{max-width:100%;display:block}
.eyebrow{font-family:var(--font-body);font-size:.8125rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.14em;color:var(--e61-orange-red);margin-bottom:10px;display:block}
/* ── KPI TILES — every slide is a light surface now, so one definition covers all of them,
   cover included (`.kpi-row-lt` is kept as a harmless alias for older markup) ── */
.kpi-row,.kpi-row-lt{display:flex;gap:14px;flex-wrap:wrap;margin-top:20px}
.kpi{background:#fff;border:1px solid var(--e61-line);border-radius:14px;box-shadow:var(--shadow-1);
  padding:16px 22px;min-width:150px}
.kpi-val{font-family:var(--font-display);font-size:1.7rem;font-weight:700;color:var(--e61-purple)}
.kpi-lbl{font-size:.72rem;margin-top:4px;color:var(--e61-ink-3)}
/* ── PILLS / BADGES ── */
.pill{display:inline-flex;align-items:center;gap:8px;padding:8px 22px;border-radius:999px;
  font-family:var(--font-display);font-weight:700;font-size:.9rem;margin-bottom:10px}
.pill-ok{background:var(--e61-success);color:#fff}
.pill-danger{background:var(--e61-danger);color:#fff}
.badge{display:inline-block;font-family:var(--font-display);font-size:.62rem;font-weight:700;
  padding:3px 9px;border-radius:5px;letter-spacing:.05em;text-transform:uppercase}
.b-hi{background:var(--e61-success);color:#fff} .b-med{background:var(--e61-warning);color:var(--e61-ink)}
.b-lo{background:var(--e61-magenta);color:#fff} .b-lk{background:var(--e61-danger);color:#fff}
.b-tl{background:var(--e61-warning);color:var(--e61-ink)} .b-nc{background:var(--e61-line);color:var(--e61-ink-3)}
.b-sk{background:var(--e61-warning);color:var(--e61-ink)} .b-block{background:var(--e61-danger);color:#fff}
.b-nb{background:var(--e61-warning);color:var(--e61-ink)} .b-mnar{background:var(--e61-magenta);color:#fff}
.b-mar{background:var(--e61-info);color:#fff} .b-mcar{background:var(--e61-success);color:#fff}
.b-amb{background:var(--e61-info);color:#fff} .b-reg{background:var(--e61-purple);color:#fff}
.b-ok{background:#D1FAE5;color:#065F46}
/* ── CARDS & GRIDS ── */
.card{background:var(--e61-surface);border:1px solid var(--e61-line);border-radius:16px;
  padding:22px 24px;box-shadow:var(--shadow-1)}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.grid-lead{display:grid;grid-template-columns:2fr 1fr;gap:18px;align-items:start}
@media(max-width:900px){.grid-2,.grid-3,.grid-4,.grid-lead{grid-template-columns:1fr}}
.finding-card{background:#fff;border-left:4px solid var(--accent);border-radius:0 14px 14px 0;
  padding:18px 20px;box-shadow:var(--shadow-1);font-size:.92rem;color:var(--e61-ink-2);line-height:1.6}
/* ── PLOT CARD — the most important visual unit in every report. Image sits above its caption,
   full slide width by default; use grid-lead to sit a plot beside supporting text/stats. ── */
.plot-card{background:#fff;border-radius:16px;border:1px solid var(--e61-line);overflow:hidden;
  box-shadow:var(--shadow-2)}
.plot-card img{width:100%}
.plot-cap{padding:14px 18px;font-size:.8rem;color:var(--e61-ink-3);border-top:1px solid var(--e61-line)}
.plot-cap b{color:var(--e61-ink-2)}
/* ── HERO / KPI-MINI / HERO-STAT / CHIPS — supporting-number components (unchanged from prior
   drafts; still correct for a slide's secondary stats) ── */
.kpi-mini-row{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:900px){.kpi-mini-row{grid-template-columns:1fr}}
.kpi-mini{background:#fff;border:1px solid var(--e61-line);border-radius:12px;box-shadow:var(--shadow-1);padding:16px 18px}
.kpi-mini-label{font-size:.7rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--e61-ink-3);margin-bottom:4px}
.kpi-mini-val{font-family:var(--font-display);font-size:1.25rem;font-weight:700;color:var(--accent)}
.kpi-mini-note{font-size:.72rem;color:var(--e61-ink-3);margin-top:4px;line-height:1.4}
.hero-stat-lbl{font-size:.78rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--e61-ink-3);margin-bottom:4px}
.hero-stat-val{font-family:var(--font-display);font-size:2.4rem;font-weight:700;color:var(--accent);line-height:1.05}
.hero-stat-note{font-size:.86rem;color:var(--e61-ink-3);margin-top:6px;max-width:520px;line-height:1.5}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}
.chip{background:var(--e61-bg-cool);border:1px solid var(--e61-line-2);border-radius:8px;
  padding:5px 12px;font-size:.78rem;font-family:var(--font-mono);color:var(--e61-ink-3)}
.chip span{color:var(--e61-purple);font-weight:500}
/* ── TABLE ── */
table{width:100%;border-collapse:collapse;font-size:.84rem}
th{text-align:left;padding:9px 12px;font-family:var(--font-display);font-size:.7rem;color:var(--e61-ink-3);
  border-bottom:2px solid var(--e61-line);font-weight:600;background:var(--e61-bg-cool);
  text-transform:uppercase;letter-spacing:.04em}
td{padding:9px 12px;border-bottom:1px solid var(--e61-line);color:var(--e61-ink-2);vertical-align:top}
tr:last-child td{border-bottom:none}
/* ── MISSINGNESS BAR ── */
.miss-bar-wrap{background:var(--e61-line);border-radius:99px;height:8px;margin:10px 0}
.miss-bar-fill{height:8px;background:var(--accent);border-radius:99px}
/* ── DETAILS DISCLOSURE — the ONE progressive-disclosure component, used everywhere: histogram
   full insight, a plot's full four-layer analysis, a limitation's full sentence, a strategy's
   full narrative. Default-visible text is already a real paragraph (see Purpose above) — this
   reveals the *exhaustive* version, not the only version. Smooth height animation, no modal. ── */
.details-btn{font-family:var(--font-display);font-weight:600;font-size:.78rem;color:var(--e61-orange-red);
  background:var(--e61-bg-warm);border:1px solid #FCE0BE;border-radius:999px;padding:8px 16px;
  cursor:pointer;display:inline-flex;align-items:center;gap:6px;transition:background .15s}
.details-btn:hover{background:#FFE8CC}
.details-btn::after{content:"▾";font-size:.65rem;transition:transform .2s}
.details-btn.open::after{transform:rotate(180deg)}
.details-panel{max-height:0;overflow:hidden;transition:max-height .35s ease;margin-top:0}
.details-panel.open{max-height:900px;margin-top:14px}
.details-panel .inner{background:#fff;border-left:3px solid var(--e61-purple);
  border-radius:0 10px 10px 0;padding:14px 18px;font-size:.86rem;color:var(--e61-ink-2);line-height:1.7}
/* ── HISTOGRAM GRID — every feature visible at once in a bounded, self-scrolling panel; each
   card expands its own insight in place with a "+ insight" toggle — no modal ── */
.hist-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;
  max-height:56vh;overflow-y:auto;padding:4px 4px 4px 0;margin-top:6px}
.hist-card{background:#fff;border:1px solid var(--e61-line);border-radius:12px;padding:10px;box-shadow:var(--shadow-1)}
.hist-card img{border-radius:7px;margin-bottom:7px}
.hist-name{font-family:var(--font-mono);font-size:.68rem;color:var(--e61-purple);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hist-sub{font-size:.63rem;color:var(--e61-ink-3);margin:3px 0 6px}
.hist-badges{display:flex;gap:3px;flex-wrap:wrap;margin-bottom:6px}
.hist-toggle{font-size:.62rem;color:var(--e61-orange-red);cursor:pointer;font-weight:600;background:none;border:none;padding:0}
.hist-insight{display:none;font-size:.68rem;color:var(--e61-ink-3);line-height:1.5;margin-top:6px;border-top:1px dashed var(--e61-line);padding-top:6px}
.hist-insight.open{display:block}
/* ── LIMITATIONS LIST ── */
.limits-list{list-style:none;padding:0;margin:0}
.limits-list li{position:relative;padding-left:22px;margin-bottom:10px;font-size:.88rem;color:var(--e61-ink-2)}
.limits-list li::before{content:"●";position:absolute;left:0;color:var(--e61-yellow);font-size:.6rem;top:.45em}
/* ── CLARIFICATIONS ── */
.clar-card{border-left:4px solid var(--accent);padding:16px 20px;border-radius:0 12px 12px 0;margin-bottom:14px}
.clar-card.blocking{border-color:var(--e61-danger);background:var(--e61-bg-warm)}
.clar-card.non-blocking{border-color:var(--e61-line-2);background:var(--e61-surface)}
@media print{.dotnav,.navbar,.topbar{display:none}.slide,.cover{min-height:0;page-break-after:always}}
```

Plots are the highest-priority content type in this skill (right under raw numbers). Every
`.plot-card` gets a **caption**, always:

```html
<div class="plot-card">
  <img src="../artifacts/eda/iteration_001_eda_target_hist.png" alt="bread_quantity distribution" loading="lazy">
  <div class="plot-cap"><b>target_hist</b> — main mode ~293, distinct near-zero cluster</div>
</div>
```

- The `.plot-cap` line ← a short synthesis of `…plots[i].reason` if it adds context, else the
  plot's slug + the headline of `…plots[i].insight`.
- The slide's own prose paragraph beside/below the plot (see Purpose's content-priority rule) is
  where the real explanation lives — 2-4 sentences, `…plots[i].insight`'s substance in the
  writer's own words, not compressed to a caption-length clause.
- The exhaustive version — `…plots[i].insight` verbatim if it's the full four-layer analysis —
  goes behind a `.details-btn`/`.details-panel` right after that paragraph (see the Details
  Disclosure component above), labeled `Full analysis` (or `Full analysis: <slug>` when a slide
  has more than one).
- confidence badge ← `…plots[i].confidence`, rendered as `<span class="badge b-hi/b-med/b-lo">`.

If `reason` or `insight` is missing/empty, do not invent content — render a red inline banner:
"Missing `reason`/`insight` in history JSON" and skip that label.

**EDA gets extra plot prominence** — its single most significant finding (a regime break, a
structural collection-boundary, whatever the data actually surfaces) earns its own `dark`/`warm`
emphasis slide with the plot large and centered, not sharing space with a `.grid-2` of others.

---

## Component: Headline Callout — usage rule

Every slide **may** include exactly one prominent callout — a `.pill-ok`/`.pill-danger` verdict
pill, or a bordered `.card` with an accent-left-border — carrying the single fact a user glancing
at only this slide must not miss. Never more than one strongly emphasized element per slide — if
two facts compete, pick the more decision-relevant one and let the other live in its normal card.

---

## Component: Histogram HISTS Array

Histograms render as a **grid** (`.hist-grid`, bounded to `max-height:56vh` with its own
scrollbar — see the Details Disclosure/Histogram Grid CSS above) — every feature's card visible
by scrolling that one panel, never paginated. Each card shows its thumbnail + skew/n + badges
always, and a `+ insight` toggle that expands the full insight text in place, inline, with no
modal. The grid is driven by a single `HISTS` JS array populated directly from
`artifact_paths.histograms` in the history JSON — **not** from `artifact_paths.plots`. These are
two separate arrays in the JSON.

**Source mapping** (read verbatim, do not rewrite):

| HISTS field | JSON source |
|---|---|
| `img` | `"../" + histograms[i].path` (prepend `../` for relative path from `reports/`) |
| `name` | `histograms[i].name` |
| `skew` | `String(histograms[i].skew)` — always 2 decimal places with sign |
| `n` | `histograms[i].n` — comma-formatted non-missing row count |
| `badges` | `histograms[i].badges` — array of badge strings; `[]` if none |
| `insight` | `histograms[i].insight` — verbatim, never rewritten |

Every entry must have all 6 fields — never omit any:

```javascript
const HISTS = [
  {name:"feature_name", img:"../artifacts/eda/iteration_001_eda_hist_FEATURE_NAME.png",
   skew:"-1.76", n:"3,453", badges:["HARD LEAKAGE"], insight:"…verbatim from histograms[i].insight…"},
  // ... one entry per histogram, in the same order as artifact_paths.histograms
];
```

**Badge vocabulary for histogram entries** — use exactly these strings, mapped via `BADGE_CLASS`:

| String | Badge class | When to use |
|---|---|---|
| `"HARD LEAKAGE"` | `b-lk` | algebraic or identity leakage |
| `"TEMPORAL LEAKAGE"` | `b-tl` | same-day operational measure |
| `"NEAR-CONSTANT"` | `b-nc` | near-zero variance |
| `"HIGH SKEW"` | `b-sk` | \|skew\| > 2 |
| `"DEPLOYMENT AMBIGUOUS"` | `b-amb` | availability at forecast time unclear |

```javascript
const BADGE_CLASS = {"HARD LEAKAGE":"b-lk","TEMPORAL LEAKAGE":"b-tl","NEAR-CONSTANT":"b-nc",
  "HIGH SKEW":"b-sk","DEPLOYMENT AMBIGUOUS":"b-amb"};
function renderHists(){
  const grid = document.getElementById('histGrid');
  grid.innerHTML = HISTS.map((h,i)=>`
    <div class="hist-card">
      <img src="${h.img}" alt="${h.name}" loading="lazy">
      <div class="hist-name">${h.name}</div>
      <div class="hist-sub">skew ${h.skew} &middot; n=${h.n}</div>
      <div class="hist-badges">${h.badges.map(b=>`<span class="badge ${BADGE_CLASS[b]||'b-nc'}" style="font-size:.56rem">${b}</span>`).join('')}</div>
      <button class="hist-toggle" onclick="toggleHist(this)">+ insight</button>
      <div class="hist-insight">${h.insight}</div>
    </div>`).join('');
}
function toggleHist(btn){
  const panel = btn.nextElementSibling;
  panel.classList.toggle('open');
  btn.textContent = panel.classList.contains('open') ? '− insight' : '+ insight';
}
renderHists();
```

**Histogram insight quality standard** — insights are read verbatim, never composed. This
standard is for **verifying** the EDA skill produced adequate content; if an insight fails these
checks, render a yellow warning inside the expanded panel rather than silently displaying a weak
insight.

A valid histogram insight covers four layers in one coherent paragraph: (1) shape, with skew
value if > |1|; (2) modelling status matching the `badges` array; (3) mechanism/why; (4)
recommended action. Minimum 2 sentences.

Example of a valid insight:
> "Left-skewed. Same-day operational count — not available at forecast time. The correlation with
> bread_quantity (0.606) is real but unusable directly. Historical rolling mean of num_orders
> would be a valid Feature Engineering candidate as a proxy for operational pace."

Cross-reference: `eda.md` step 10 governs what EDA writes in these fields. If histogram insights
are consistently weak, the fix is there, not here.

---

## Component: Clarification Block

```html
<!-- BLOCKING -->
<div class="clar-card blocking">
  <span class="badge b-block">BLOCKING</span> <span class="badge b-hi">HIGH CONFIDENCE</span>
  <h4 style="margin-top:10px">Question as a complete sentence ending with a question mark?</h4>
  <p style="font-size:.85rem;color:var(--e61-ink-3)">Context: why this is blocking and what
  decision depends on the answer. Ends with which downstream step is affected.</p>
  <p style="font-size:.85rem"><b>Recommended answer:</b> the node's `recommendation`, verbatim —
  followed by `recommendation_reasoning`. <b>If wrong:</b> `impact_if_wrong`.</p>
</div>

<!-- NON-BLOCKING: class "non-blocking" -->
<div class="clar-card non-blocking">
  <span class="badge b-nb">NON-BLOCKING</span> <span class="badge b-med">MEDIUM CONFIDENCE</span>
  <h4 style="margin-top:10px">Question?</h4>
  <p style="font-size:.85rem;color:var(--e61-ink-3)">Context.</p>
</div>
```

---

## Component: Details Disclosure JS (copy verbatim into every report)

The one progressive-disclosure mechanism, shared by every "Full analysis" / "Full reasoning" /
"Full strategy narrative" button in the deck — plain toggle, smooth `max-height` transition, no
modal, no overlay, nothing to "not open."

```javascript
function toggleDetails(btn){
  btn.classList.toggle('open');
  const panel = btn.nextElementSibling;
  panel.classList.toggle('open');
  btn.textContent = panel.classList.contains('open')
    ? btn.textContent.replace(/^Full/, 'Hide')
    : (btn.dataset.orig || btn.textContent.replace(/^Hide/, 'Full'));
}
document.querySelectorAll('.details-btn').forEach(b=>{ b.dataset.orig = b.textContent; });
```

```html
<button class="details-btn" onclick="toggleDetails(this)">Full analysis</button>
<div class="details-panel"><div class="inner">…verbatim full text…</div></div>
```

---

## Component: Plot Modal (every `.plot-card` image that isn't already on the Histograms grid)

The histogram grid has its own expand-in-place mechanism (`+ insight`, see above) because it's
already a dedicated page of nothing but plots. Every other plot in the deck — the `.plot-card`s
scattered across Overview, Data Quality, Target Analysis, Feature Analysis, and every other
skill's slides — sits in a slide-width card and its full explanation may be cut down to fit
alongside other content. **Every one of those images must be clickable**, opening a larger view
with the complete explanation, not just the on-slide version. This is the "no dedicated page"
case the histogram grid already solved differently; this component solves it for everything else.

```css
.img-modal{display:none;position:fixed;inset:0;background:rgba(26,18,48,.6);z-index:500;
  align-items:center;justify-content:center;padding:24px;backdrop-filter:blur(3px)}
.img-modal.open{display:flex}
.img-modal-inner{background:#fff;border-radius:18px;box-shadow:var(--shadow-3);
  display:grid;grid-template-columns:1.4fr 1fr;max-width:1040px;width:100%;max-height:88vh;
  overflow:hidden;position:relative}
.img-modal-close{position:absolute;top:12px;right:12px;background:var(--e61-line);border:none;
  border-radius:50%;width:30px;height:30px;cursor:pointer;font-size:.85rem;color:var(--e61-ink-2);
  z-index:10}
.img-modal-close:hover{background:var(--e61-danger);color:#fff}
.img-modal-img-wrap{background:var(--e61-bg);display:flex;align-items:center;justify-content:center;
  padding:24px;overflow:hidden}
.img-modal-img-wrap img{max-width:100%;max-height:76vh;height:auto;border-radius:10px;object-fit:contain}
.img-modal-meta{padding:28px 26px;display:flex;flex-direction:column;gap:12px;overflow-y:auto;
  border-left:1px solid var(--e61-line)}
.img-modal-name{font-family:var(--font-display);font-size:1.05rem;font-weight:700;color:var(--e61-purple)}
.img-modal-body{font-size:.9rem;color:var(--e61-ink-2);line-height:1.7}
@media(max-width:700px){.img-modal-inner{grid-template-columns:1fr;max-height:92vh}
  .img-modal-meta{border-left:none;border-top:1px solid var(--e61-line)}}
```

**Wiring** — every `<img>` inside a `.plot-card` carries `data-modal-title` and `data-modal-body`.
`data-modal-body` is the same full explanation the card already presents (the plot's `insight`/
finding text verbatim, plus its `reason`/why-this-plot line if useful context) — never a shorter
or different text than what a reader already has access to elsewhere in the card; the modal exists
to show it bigger and uninterrupted, not to add new content:

```html
<div class="plot-card">
  <img src="../artifacts/eda/iteration_001_eda_target_hist.png" alt="bread_quantity distribution"
       data-modal-title="target_hist"
       data-modal-body="Bimodal: spike near 0 from the 350-row anomalous cluster (9.2%) vs. the genuine-demand distribution.">
  <div class="plot-cap"><b>target_hist</b> — main mode ~293, distinct near-zero cluster</div>
</div>
```

```javascript
function openImgModal(imgEl){
  document.getElementById('imgModalImg').src = imgEl.currentSrc || imgEl.src;
  document.getElementById('imgModalImg').alt = imgEl.alt;
  document.getElementById('imgModalName').textContent = imgEl.dataset.modalTitle || imgEl.alt;
  document.getElementById('imgModalBody').textContent = imgEl.dataset.modalBody || '';
  document.getElementById('imgModal').classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeImgModal(){
  document.getElementById('imgModal').classList.remove('open');
  document.body.style.overflow = '';
}
function wireImgModals(){
  document.querySelectorAll('.plot-card img').forEach(img=>{
    img.style.cursor = 'zoom-in';
    img.addEventListener('click', ()=>openImgModal(img));
  });
}
document.addEventListener('keydown', e=>{
  if(document.getElementById('imgModal')?.classList.contains('open') && e.key==='Escape') closeImgModal();
});
wireImgModals();
```

```html
<div class="img-modal" id="imgModal" onclick="if(event.target.id==='imgModal')closeImgModal()">
  <div class="img-modal-inner">
    <button class="img-modal-close" onclick="closeImgModal()">✕</button>
    <div class="img-modal-img-wrap"><img id="imgModalImg" src="" alt=""></div>
    <div class="img-modal-meta">
      <div class="img-modal-name" id="imgModalName"></div>
      <div class="img-modal-body" id="imgModalBody"></div>
    </div>
  </div>
</div>
```

This markup goes once per report, right before the closing `</body>`, same as the histogram
modal pattern this replaces the need for on every other plot. `wireImgModals()` runs after the
page body is written (end of script, alongside `renderHists()`), so it picks up every `.plot-card`
image on every slide in one pass — no per-image inline handler.

---

## Procedure

**Step 0 — Pre-flight: verify convention compliance.**
Before generating anything, check that the preceding skill placed its files correctly using the
table in "File Placement Constraints" above:
- `history/iteration_NNN_SKILL.json` exists and is valid JSON.
- `code/iteration_NNN_SKILL.py` exists. If absent, log: "CONVENTION VIOLATION: code file missing."
- `state/current.json` exists. If absent, log: "CONVENTION VIOLATION: state/current.json missing."
- For Preprocess / FE: `data/iteration_NNN_SKILL.csv` exists.
- For Training: `artifacts/training/iteration_NNN_SKILL_model.pkl` exists (or note simulated run if not real).
- No skill outputs sitting in the run root or in any non-convention path.

If any violation is found, render this banner as the first thing inside slide 1 (above the hero):

```html
<div style="background:#FEF2F2;border-left:4px solid var(--e61-danger);padding:10px 16px;
  margin-bottom:14px;font-family:var(--font-body);font-size:.8rem;color:#7F1D1D;
  border-radius:0 var(--radius-md) var(--radius-md) 0">
  ⚠ Convention violation detected: [description]. See naming_conventions.md.
</div>
```

**Step 1** — Read and parse `history_json_path`. Identify `skill_type` (from JSON or filename).

**Step 2** — Glob `run_dir/artifacts/<skill_type>/*.png` — build a verified set of image paths.
Relative path from `reports/`: `../artifacts/<skill_type>/filename.png`. Never include any path
not confirmed to exist on disk.

**Step 3** — Build the slide deck per that skill's table below: one `<section class="slide">` per
row, in order, using the shared dashboard component vocabulary.

**Step 4** — Write to `run_dir/reports/iteration_NNN_SKILL.html`. Overwrite if it exists.

**Step 5** — Edit `report_html` in the history JSON: `"report_html": "reports/iteration_NNN_SKILL.html"`.

**Step 6** — Regenerate `run_dir/reports/index.html` (see Index section at end of this skill).

**Step 7 — Post-flight verification:**
- HTML file exists at the correct path.
- `report_html` is set in the history JSON.
- `index.html` is updated and links to the new report.
- Every `<img src="...">` points to a path that exists on disk.
- No absolute paths anywhere in the HTML.
- The deck engine initializes (slide 1 visible, dots match slide count, arrows correctly disabled at the ends).
- `state/current.json` exists (create a minimal one if missing, do not silently skip it).

---

---

# EDA Report

**Accent: `#FC5B00`** (`var(--e61-orange-red)`)

Read every field from `output`. Use the canonical reference data at
`runs/run_20260819T090000Z/history/iteration_001_eda.json` to sanity-check field paths.

Slide numbers below are sequential positions, not fixed counts — the Plots section spans 3-4
themed slides on its own (see Slide 6+ below). 10 is a typical total.

## Slide 1 — Overview & Feasibility (`data-title="Overview"`)

**Hero** (background `var(--accent)`):
- Feasibility pill: green `pill-ok` "✓ TRAINABLE" if `feasibility.trainable=true`, else red
  `pill-warn` "✗ NOT TRAINABLE".
- `.hero-reason` = a one-sentence synthesis of `feasibility.reasoning` (the verdict-driving fact —
  e.g. "3,459 genuine rows after excluding the 350-row anomalous cluster — comfortably above the
  row floor"), with the full original text in a `.details-btn`/`.details-panel` disclosure right below it.
- KPI row: Rows (raw) · Columns · Overall missing % · Recommended primary metric.

**Below the hero, two-column grid (and nothing else on this slide — Key Findings gets its own
slide next; a hero + two cards + a 5-item grid together don't fit one screen without scrolling):**
- Left card — **Schema**: table of dtype category | count from `dataset_overview.dtype_breakdown`.
- Right card — **Time Coverage**: date range as the card's own headline (large, mono), then the
  rest of `time_coverage` as `.chips` — one chip per measured number (e.g. `18.2% day-gap`,
  `weather 2015+`, `competitor 2017+`, `n gaps: 3`) — never a bullet list of full sentences. Only
  `collection_shift_note` (if it names something structurally important — a schema change, a
  regime break) gets a single short `.callout-pu` line; anything else in `gaps` that's just a
  number becomes a chip, not a sentence.

## Slide 2 — Key Findings (`data-title="Findings"`)

`.grid-fill` of `.finding-card`s from `output.key_findings`, one per finding, its own slide (split
out from Overview — see the note on Slide 1). Each card is a **short bold headline (≤8 words) +
one short clause**, not the multi-sentence source text — e.g. source "This dataset is properly a
time-series/forecasting task... lag-7 autocorrelation=0.38... ADF test found it likely stationary
(p=0.0039)..." becomes **"Time-series, not plain regression"** — confirmed trend + weekly
seasonality (lag-7 ACF 0.38) + stationary (p=0.004). Numbers that matter stay; connective prose is
cut. `key_findings` is typically 4-6 entries, which fits one screen at 2-3 columns comfortably —
if a call ever produces meaningfully more, split alphabetically/thematically into a second slide
rather than shrinking the cards.

## Slide 3 — Target Analysis (`data-title="Target"`)

`.grid-2` of cards:
1. **Distribution** — task-type badge (REGRESSION / BINARY CLASSIFICATION / etc.) + a compact
   monospace stat table (mean, median, std, skew, min, max) from `target.distribution` (or
   `target.range` for the time-series shape this JSON actually uses — read whichever the contract
   provides, never invent missing stats).
2. **Temporal drift** — ✓/⚠ icon by whether drift was detected + a one-clause synthesis of
   `target.temporal_drift.note` (the number, not the method).
3. **Segment breakdown** — if present: compact table of segment | value | n, amber row highlight
   on any segment carrying a `note` (anomalous) — the table's numbers are the content here, no
   extra prose needed.
4. **Recommended metric** — primary metric large in `var(--accent)`, confidence badge, secondary
   metrics as small pill chips, one-sentence synthesis of `reasoning` + `.details-btn`/`.details-panel` for the full text.

Add a prominent callout if `time_series_analysis.trend.detected` is true and the trend is large —
e.g. "Downward trend: -28% over the series (high confidence)" — this is exactly the kind of fact
a user must not miss when skimming.

## Slide 4 — Top Features, Interactions & Multicollinearity (`data-title="Features"`)

- **Table** of `output.top_features`: Feature (mono) | Spearman | MI | Missing % | Skew |
  Confidence badge (high→`b-hi`, medium→`b-med`, low→`b-lo`).
- If `flags.possible_temporal_leakage` non-empty: `.callout` listing those features.
- **Feature Interactions** — `.grid-fill` of compact cards, one per `feature_interactions` entry:
  `feature × segment_dimension` header (mono), method badge, small shift table, `note`, confidence
  badge.
- **Multicollinearity** — table of `multicollinearity.high_pairs`: Feature pair | Pearson r
  (r>0.9 → red-tinted cell, 0.7–0.9 → amber-tinted) | Suggested action.

If a scatter/interaction plot PNG exists for a top feature, it belongs on Slide 5 (Plots), not
duplicated here — this slide stays table/stat-dense.

## Slide 5 — Missingness & Data Quality (`data-title="Missingness & Flags"`)

- `.grid-2` (or `.grid-fill` for 5+) of `.card`s, one per `missingness.columns_with_missing`
  entry — if there are more than ~6, wrap the grid in `style="max-height:40vh;overflow-y:auto"`
  inline (a run with 8-11 affected columns needs it; fewer doesn't). Each card: feature name
  (mono) + mechanism badge (`b-mnar`/`b-mar`/`b-mcar`), `.miss-bar-wrap`/`.miss-bar-fill` (width =
  missing %), `missing_pct`%, a short paragraph on `mechanism_evidence`, `suggested_action` in a
  bordered callout (green-left-border, `background:#F0FBF4`).
- Correlated-missingness pairs as a one-line note below the grid.
- **Flags** — three compact groups below: Leakage/Temporal Leakage (red-bordered),
  Near-Constant/High-Skew (amber-bordered), Out-of-range/Duplicates/Label noise (compact table).
  PII columns (if any) as red pill badges.

## Slide 6+ — Generated Plots (`data-title="Plots"`, split by theme across multiple slides)

This is the flagship slide type for EDA. Group `artifact_paths.plots` entries (excluding any path
containing `_hist_`, which belongs on the Histograms slide) into themed slides — the exact
grouping is a judgment call grounded in what the plots actually show, but a typical split:

- **Target & Trend** — the target distribution + target-over-time plots (these two carry the
  dataset's central story), each a full-width `.plot-card` with its caption and a real
  explanatory paragraph beside/below it (see the Plot Card component above). Trend/stationarity/
  seasonality numbers that don't need their own image go in a `.kpi-mini-row` rather than a third
  plot (e.g. "Trend −0.0345/day", "ADF p=0.0039", "Seasonality: weekly + mild annual").
- **Seasonality & Store Regime** — the store-regime × day-of-week plot (direct evidence for a
  top_feature) plus the weekly-pattern plot, `.grid-2` of `.plot-card`s; supporting plots
  (schema-era split, data-coverage-over-time) as a second `.grid-2` row below if they add a
  distinct finding, otherwise fold their one-line takeaway into the surrounding prose instead of
  forcing another image onto the slide.
- **Feature Relationships** — the two or three most important scatter plots (the strongest
  leakage flag and the strongest genuine driver, at minimum) as `.plot-card`s; the correlation
  heatmap and any multicollinearity scatters either join this slide's grid or get their own
  themed slide if that would otherwise crowd more than ~4 images onto one screen — with
  scroll-snap, a slide running slightly taller than the viewport is fine (the reader just scrolls
  a little further before the next snap point), but a wall of 6+ images on one slide is still a
  content-budgeting problem, not a layout one.

Every plot's finding gets a real paragraph, not just a caption — see the Plot Card component's
rule above. If a PNG doesn't exist on disk, render a text-only card with the same paragraph +
`.details-btn`/`.details-panel` plus a small grey note: "Plot available in artifacts/ after code
execution."

## Next — Feature Histograms (`data-title="Histograms"`)

`.hist-grid` exactly as specified in "Component: Histogram HISTS Array" above, built from
`artifact_paths.histograms` (never from `plots`) — the grid is already self-bounded
(`max-height:56vh;overflow-y:auto`) so this is the one slide type that keeps its own internal
scrollbar regardless of feature count; the rest of the slide (title, intro line) stays still
above it. If `artifact_paths.histograms` is absent or empty (simulation mode), render a styled
placeholder instead of an empty grid. Each card's expanded insight text stays as-is (it's already
the compact per-feature summary `eda.md` step 10 asks for) — no further compression needed.

## Multi-horizon runs

When the training output is keyed by horizon, every metric table renders **one row per horizon**,
in ascending lead-time order, with the pooled figure last and labelled as pooled — never first and
never alone. Model family, feature count and the margin over the baseline floor are per-horizon
columns, because they legitimately differ between lead times. A report that shows one number for a
run that trained four models has hidden three of them.

## Last — Clarifications, Strategy & Limitations (`data-title="Clarifications"`)

- `.grid-2`: left column — one `.clar-card` per `clarification_needed` entry (`blocking` class for
  blocking, `non-blocking` for the rest; `context` already short enough to render as-is). Every
  card also renders the entry's `recommendation`, `recommendation_reasoning` and `impact_if_wrong`
  — a question shown without the answer its own node proposed is an incomplete card, and the
  recommendation is the part a reader can actually act on. Where an entry lacks those fields, say
  so on the card rather than omitting the line silently. Right
  column — **Strategy** card: primary metric + posture (e.g. "incremental, evidence-driven") as a
  real paragraph on `recommended_strategy.narrative`, confidence badge, the full narrative behind
  a `.details-btn`/`.details-panel` labeled "Full strategy narrative", then **Limitations** below
  it as a `.limits-list` — `output.limitations` rendered as list items verbatim (they're already
  single well-written sentences in the reference data; if a real run's limitations run
  meaningfully longer, keep the full sentence in the list item rather than truncating it — this
  list doesn't need a disclosure the way a four-layer plot insight does).

---

---

# Domain Research Report

**Accent: `#5C4EA8`**

This section did not exist in the pre-slideshow version of this file — `domain_research.md` step 9
already fully specifies this report's structure and its own accent color, and its own step 8 has
it write a reproducible `code/iteration_<NNN>_domain_research.py` that renders the report directly
from the history JSON, independent of this file. That means this section is usually redundant with
what domain_research already builds itself — it exists so that when the Planner's unconditional
"report_generator always runs next" rule (see `planner.md`) invokes this skill after a Domain
Research call, it has a real slide deck to fall back on instead of an empty one, rather than to
replace domain_research's own reproducible rendering path. Content/field mapping below is ported
directly from `domain_research.md` step 9 and cross-checked against a real report already on disk
(`runs/run_20260821T130000Z/reports/iteration_001b_domain_research.html`) — only the container
(slides instead of scroll sections) is new; the `data-title` values below reuse that report's own
existing section ids (`identity`, `triggers`, `covid`, `datasources`, `drivers`, `context`,
`recommendations`, `questions`) so any external references to those anchors stay meaningful.

## Slide 1 — Identity & Trigger Responses (`data-title="identity"` + `data-title="triggers"` combined)

**Hero**: KPI row with `meta.triggers_addressed`, `meta.feature_recommendations` (count of
`feature_engineering_recommendations`), `meta.data_sources_assessed`, plus up to 3 headline domain
facts as small stat chips below the KPI row (a judgment call on which facts are most
headline-worthy — e.g. largest event attendance, the key structural-shock date, the primary demand
driver — not a fixed JSON path).

Below the hero: `.grid-fill` of compact cards, one per `eda_trigger_responses` entry (`trigger`,
`finding`, confidence badge, `source` as a small mono label, `feature_implication`). One
a prominent accent-bordered callout surfacing whichever single entry is most actionable — a judgment
call, not a separate JSON field.

## Slide 2 — Structural Shocks & Demand Drivers (`data-title="covid"` + `"drivers"`)

Only render the shocks half if `structured_findings.structural_shocks` is non-empty — otherwise
this slide is drivers-only. Shocks: compact timeline + bullet list of demand-shock effects + a note
on train/val/test overlap if known. Drivers: table from
`structured_findings.demand_drivers.identified_demand_drivers_ranked` — driver | type | strength
badge (very_high/high→green, medium→amber, low/low_confounded→grey) | evidence.

## Slide 3 — Data Source Assessment (`data-title="datasources"`)

`.grid-fill` of compact panels, one per key in `data_source_assessment`: verdict badge (USEFUL→
green, CONDITIONALLY USEFUL→blue, REAL BUT SPARSE/WEAK SIGNAL→amber, DROP CANDIDATE→red),
`availability_window`, `domain_context` (short), a small per-feature signal table from
`signal_assessment_per_feature`, `known_confounds` as a compact callout,
`recommended_additional_features` as chip list.

## Slide 4 — Context & Recommendations (`data-title="context"` + `"recommendations"`)

Whichever of `public_holidays`, `school_holidays`, `city_demographics` are non-empty, each as its
own compact table/card group — omit any empty ones. Below: `.grid-fill` of recommendation cards
from `feature_engineering_recommendations` (feature name mono, type, priority badge, `description`,
`rationale`, `data_source` label, `precondition` in amber callout if present).

## Slide 5 — Open Questions (`data-title="questions"`)

Same visual pattern as EDA's Clarifications (`.clar-card`), one per `open_questions` entry
(`question`, `reason`, priority badge). `open_questions` doesn't carry a `blocking` field, so don't
invent one — use the `non-blocking` styling for all of them, accent-colored to this report's own
`#5C4EA8` rather than EDA's blocking/non-blocking red-vs-amber split.

---

---

# Preprocess Report

**Accent: `#9D1D56`** (`var(--e61-magenta)`)

## Slide 1 — Overview (`data-title="Overview"`)

**Hero**: mode badge pill (INITIAL / INCREMENTAL / REBUILD_FROM_RAW), `.hero-reason` = first
sentence of `input.mode_reasoning`. KPI row: rows filtered · cols dropped
(`len(columns_dropped)`) · features transformed (`len(transformations)`) · split created
(green YES / grey NO).

Below: `.grid-2` —
- **Call Context** card: mode badge, features in scope (mono chips) — the chips are the content
  here; `mode_reasoning`/context summary reduce to one short paragraph + `.details-btn`/`.details-panel` for the full text.
  `already_processed` list if non-empty, deployment-scenario note ("All transformations are fit on
  train rows only").
- **Rows Filtered** card: if `rows_filtered > 0`, compact table (Condition | Rows removed | Before
  | After | Source) plus a before→after visual (`3,809 → 3,459 (350 filtered)`); if 0, muted
  italic "No rows filtered this call."

## Slide 2 — Train/Validation Split & Columns Dropped (`data-title="Split & Drops"`)

- If `split_created=true`: card with method badge (TIME_BASED/STRATIFIED/RANDOM), the cutoff table
  (Split | Start | End | Rows | Level gap vs train, row-tinted train=green / validation=blue)
  leading — it's the actual content — followed by the fixed warning note about
  scaler/imputation/encoding fit on train only. There are only ever two partitions; if a
  partition named `test` appears in the data, render it as a red error callout rather than a
  table row, because it should not exist. Where `split_info.cv_design` or
  `split_info.selection_noise_band` are present, render both — the noise band is what tells a
  reader which of this run's comparisons were real.

  **`split.reasoning` is rendered in full, not collapsed.** Unlike other prose fields, this one
  does not reduce to a one-line synthesis behind a `.details-btn` — every number the run reports
  is conditional on the split, and a reader who never opens a disclosure would never learn what
  those numbers are conditional on. Give each of step 3's questions its own labelled line: what
  deployment situation the split simulates, whether validation is a fair stand-in for the unseen
  holdout, whether either boundary lands somewhere meaningful, and what a reasonable person might
  have done differently.

  When Preprocess reported that validation sits closer to train than the unseen holdout is likely
  to (`split_info.holdout_representativeness`), render that verdict as a **prominent warning
  callout** on this slide — amber, above the fold, reading to the effect of *"validation can rank
  models on accuracy but not on robustness; these numbers are optimistic by an unknown margin."*
  Downstream slides quote validation metrics as if they generalise; this is the one place the
  report can say when they do not.

  **Every metric in every report this pipeline writes is a validation metric.** The held-out
  score is produced after the run, outside the pipeline. Never label a figure "test", "final" or
  "held-out" anywhere in a report.
- **Columns Dropped** table: Column (mono) | Category | Reason. Category + row tint derived from
  the `reason` text: leakage→red, temporal leakage→amber, near-constant/redundant→grey.

## Slide 3 — Per-Feature Transformations (`data-title="Transformations"`)

`.grid-fill` of compact cards, one per `output.transformations` entry: header strip (feature name
mono + confidence badge), then for each step: a small colored type pill (LOG_TRANSFORM orange /
SCALE blue / MISSING_INDICATOR purple / FILL green / CLIP amber), formula/parameters in mono if
present, `reasoning` (short), skew before→after as a compact two-number row with an arrow if
present (`14.27 → 1.12 (↓92%)`), `rows_affected` as a muted footnote. If a before/after PNG exists
for the feature, embed it small at the bottom of that card.

## Slide 4 — Before/After, Output & Issues (`data-title="Results"`)

- **Before/After Comparison** table from `output.before_after_comparison`: Feature | Metric |
  Before | After | Δ | What changed (Δ green if improvement, grey if neutral).
- **Output Dataset** compact card: path (mono), rows, cols, retained columns as chip list, NaN
  note if applicable.
- **Issues**: green check "No issues" if empty, else one red-bordered card per issue.

---

---

# Feature Engineering Report

**Accent: `#2F6BB3`** (`var(--e61-info)`)

## Slide 1 — Overview (`data-title="Overview"`)

**Hero**: headline "**N Features Created**" (`len(output.features_created)`), subtitle from mode +
deployment scenario. KPI row: features created · features rejected · highest Spearman (top new
feature) · leakage audit PASSED (green).

Below: **Call Context** card — mode badge, features already in model (mono chips, the content),
one-line a short paragraph on the context (with `.details-btn`/`.details-panel` if the source is more than ~2 sentences),
deployment scenario note.

## Slide 2 — Features Created (`data-title="Created"`)

Grouped by technique keyword into named subsections (**Naive Predictors** / Lag Features /
Rolling-Window Features / Calendar Features / Interaction Features / Other), each with a small
colored header strip tinted to `var(--accent)`.

**Naive Predictors comes first when `naive_predictors_built` is non-empty**, one card per horizon
group: the column name (mono), the rule and its parameters, `selected_by` as a badge
(`part_a_winner` / `train_search`), the train score, the verification result against the
baseline's own number for that rule (green ✓ / red ✗ — a ✗ is the headline of the whole slide,
not a footnote), and a **PROTECTED** pill. These are the columns the run is not allowed to drop
silently, so a reader must be able to see at a glance that they exist and that they were built
correctly. `.grid-fill` of compact cards per feature: name (mono) + confidence badge,
technique (small italic), formula/window if present (mono, this is a number — lead with it),
**Why:** one clause synthesizing the field (`.details-btn`/`.details-panel` for the full text), **Leakage rule:** green ✓
if future-safe/row-local-causal else amber ⚠ + one clause (`.details-btn`/`.details-panel` for the full text),
**Relevance:** spearman stat badge (the number leads) + short note. `first_valid_row` muted
footnote. Small embedded distribution/scatter PNG at the bottom of the card if one exists for that
feature — these plots matter, don't drop them for space.

## Slide 3 — Rejected Features & Leakage Audit (`data-title="Audit"`)

- If `features_rejected` non-empty: `.grid-fill` of small cards — feature name (mono,
  strikethrough), one-clause reason (`.details-btn`/`.details-panel` if the source reason is more than a sentence),
  confidence badge.
- **Protected Features** callout, whenever `protected_features_status` is present: one row per
  protected column — present in the output dataset ✓/✗, and any blocking clarification raised
  against it, quoted with the raising evidence and the recommendation. A protected column missing
  from the dataset is a red callout at the top of this slide: it means something removed it
  without the user answering, which is the failure the protection exists to prevent.
- **Leakage Audit** full-width prominent `.pill-ok`/accent-bordered `.card` callout: green if verdict contains "No leakage"/
  "passed", red otherwise; `all_features_audited` ✓/✗; `verdict` text bold; deployment-scenario
  badge (LIVE_FORECAST/RETROSPECTIVE); caveat note in amber callout if present.

## Slide 4 — Output & Issues (`data-title="Results"`)

**Output Dataset** card: path, rows, cols, new columns as chip list, NaN handling note if
applicable. **Issues**: same green-check/red-card pattern as Preprocess.

---

---

# Training Report

**Accent: `#2F8F5C`** (`var(--e61-success)`)

## Slide 1 — Overview & Configuration (`data-title="Overview"`)

**Hero**: model family badge pill (e.g. "LightGBM"), primary metric very large (e.g. "Val MAE
31.2"). KPI row: Train metric · Val metric · N features · early-stopping round (if applicable).

Below: `.grid-2` —
- **Deployment Scenario** card (`.callout-gr` background): LIVE_FORECAST/RETROSPECTIVE badge,
  one-line a short paragraph on `deployment_scenario_reasoning` + `.details-btn`/`.details-panel`; amber callout listing any
  features with unresolved deployment-time-availability caveats.
- **Model Configuration** card: hyperparameter table (Value column mono, "(tuned)" tag on tuned
  params) + compact sub-table of training details (train rows, validation rows, training time,
  early stopping).

## Slide 2 — Metrics (`data-title="Metrics"`)

`.grid-2`: **Train** panel (`.callout-gr` tint, `var(--e61-success)` border) and **Validation**
panel (light-blue tint, `var(--e61-info)` border), each a compact stat list (MAE, RMSE, R²,
MAPE...). Below both, full-width **Gap Analysis** card from `metrics.gap_analysis`: assessment
badge ("moderate overfitting"→amber, "severe overfitting"→red, "well-generalised"→green), MAE gap
+ gap% as large stat numbers (these lead), one-clause synthesis of `note` + `.details-btn`/`.details-panel` for the full
text.

## Slide 3 — Feature Importance (`data-title="Importance"`)

If `artifacts.feature_importance_plot` exists on disk, embed it full-width first. Table below
(always rendered regardless of plot): Rank | Feature (mono) | Importance | visual bar
(`width:{importance/max*100}%`, `var(--accent)`) | Note. Rank 1–3 rows lightly highlighted. Any
feature flagged in `flags_for_diagnosis` as a leakage risk gets an amber row + ⚠ badge.

## Slide 4 — Residuals & Flags (`data-title="Diagnostics"`)

- Residual plot PNG if it exists; stat tiles (mean_residual | std_residual); pattern notes as small
  cards with a colored left border (amber underprediction, blue temporal, red systematic bias).
- **Flags for Diagnosis** — `.grid-fill` of cards, one per `flags_for_diagnosis` entry: priority
  high→red border+badge, medium→amber, low→grey; title as heading, one-clause synthesis of
  `detail` + `.details-btn`/`.details-panel` for the full text.
- **Issues** — same pattern as Preprocess.

---

---

# Diagnosis Report

**Accent: `#0E7C86`**

Read every field from `output` per `contracts/diagnosis.json`. Always has a stop/continue verdict
and root-cause/evidence; the Hypothesis slide only renders content when the verdict is `continue`.

## Slide 1 — Verdict & Assessment (`data-title="Verdict"`)

**Hero**: verdict pill — green `pill-ok` "✓ STOP — GOAL MET" if
`stop_recommendation.recommendation=="stop"`, else white-on-accent "→ CONTINUE". `.hero-reason` =
one-sentence synthesis of `stop_recommendation.reasoning` (the deciding fact), full text in a
`.details-btn`/`.details-panel` disclosure. KPI row: Confidence · Suggested next skill
(`hypothesis.suggested_action.node`, or "—" if stopping) · Hypothesis confidence (or "—" if
stopping).

Below: `.grid-fill` of 5 compact cards, one per key in `stop_recommendation.checks`
(`explicit_target`, `diminishing_returns`, `residual_or_segment_flags`, `baseline_comparison`,
`unresolved_clarifications`) — check name title-cased as `<h3>`, value as body. Left-border amber
if the text implies a flag (phrasing like "not given", "borderline", "remains open", "worse",
"flag"), green otherwise.

## Slide 2 — Root Cause (`data-title="Root Cause"`)

`diagnosis.primary_issue` as `<h3>` in a card with `border-left:4px solid var(--accent)`.
`diagnosis.evidence` as `.grid-fill` of `.finding-card`s (accent-colored border, these are already
short and numeric — keep as-is). `root_cause` as a `.callout-pu` block: one-sentence `.summary`
synthesis leading, full text in `.details-btn`/`.details-panel`.

## Slide 3 — Hypothesis (`data-title="Hypothesis"`)

Only rendered with content when `stop_recommendation.recommendation=="continue"`. When `stop`,
render a single `.callout-gr` box instead: "✓ Goal met this iteration — see the Verdict &
Assessment slide." and nothing else on this slide.

When continuing: `hypothesis.statement` as a `.summary` lead line (+ `.details-btn`/`.details-panel` if long) with
confidence badge beside the heading. **Suggested action** as a prominent `.pill-ok`/accent-bordered `.card` callout (the most
visually prominent element on this slide): node badge for `suggested_action.node` reusing that
node's own accent color (orange=eda, medium purple=domain_research, magenta=preprocess,
blue=feature_engineering, green=training), `action` text (short, this is an instruction, keep it
as-is), one-clause synthesis of `reasoning` + `.details-btn`/`.details-panel`. `alternative_if_wrong` as a muted card
below, labeled "IF THIS IS WRONG", one clause + `.details-btn`/`.details-panel`.

## Slide 4 — Trend, Clarifications & Issues (`data-title="Trend & Clarifications"`)

- If `artifact_paths.plots` present and non-empty: each as a standard `.plot-card` (reason/insight/
  confidence, same as every other report type — if the JSON calls it something other than
  `insight`, follow whatever `diagnosis.md` actually produced). If absent (fewer than 2 Training
  iterations exist yet), omit this block entirely rather than an empty placeholder.
- **Clarifications** — identical `.clar-card` component to EDA's, one per
  `clarification_needed` entry. Every entry is surfaced regardless of `blocking` (per the
  Planner's own rules) — this report doesn't change what gets asked, it renders what Diagnosis
  recorded.
- **Issues** — same green-check/red-card pattern as every other report type.

---

---

# Index (Run Navigator / Homepage)

After writing the skill report HTML, regenerate `run_dir/reports/index.html`. This page is **not**
a slideshow — it's the run's homepage, a normal scrolling dashboard page linking out to every
report deck. Structure and logic are unchanged from before; only the token/component styling
changes (use the Design System tokens and card/badge components above instead of the old inline
palette).

Read all JSON files under `run_dir/history/` (glob `iteration_*.json`). Extract per iteration:
- `skill_type`, `iteration`, `timestamp`, `report_html`
- Summary label:
  - EDA → `"TRAINABLE"` or `"NOT TRAINABLE"` + recommended metric
  - Domain Research → `"N triggers addressed · M recommendations"` (from `meta.triggers_addressed`
    and `meta.feature_recommendations`)
  - Preprocess → mode + "N cols dropped"
  - Feature Engineering → "N features created"
  - Training → `"Val [metric] [value]"`
  - Diagnosis → `"✓ STOP"` if `stop_recommendation.recommendation == "stop"`, else
    `"→ CONTINUE · [primary_issue truncated to ~50 chars]"`
- Blocking flag: any skill with a `clarification_needed[].blocking = true` entry — not only EDA.
  Diagnosis in particular can carry forward an unresolved blocking clarification from an earlier
  skill (see `diagnosis.md` step 5).

Since the Planner can run skills in any order and revisit any of them (see `planner.md`), the
"By Skill Type" tab is not a fixed 4 columns — one column per skill type actually present in this
history, in stable order (`eda, domain_research, preprocess, feature_engineering, training,
diagnosis`), each holding however many iter-cards that skill produced, including repeats.

Bake all data as a hardcoded JS array (no `fetch()` — file:// protocol).

**Page structure:**
- Fixed header bar (`background:var(--e61-purple)`, `height:52px`) — this is a normal navigator
  page, not a slide deck, so it uses its own real header, not the report decks' thin 3px progress
  rail: element61 mark, run ID label, logo links to `index.html`.
- Hero band (`var(--e61-purple)` background): run ID, iteration count, last-updated timestamp, KPI
  tiles.
- Tab bar: "By Skill Type" | "Timeline" — pure JS switching via `switchTab(name)`.
- By Skill Type tab: one column per skill type present in this run's history, colored header per
  that skill's accent, iter-cards below in run order.
- Timeline tab: horizontal scrollable track, nodes colored by skill type with connectors,
  click-to-jump to the card in By Skill Type view.
- Footer: thin, dark purple, run metadata.

**Iter card:**
```css
.iter-card{border-left:4px solid var(--accent);border-radius:var(--radius-md);padding:12px 14px;
  background:var(--e61-bg);margin-bottom:8px;box-shadow:var(--shadow-1)}
```
Top line: "Iteration NNN" (mono) + status badge (DONE green / NO REPORT grey / BLOCKING ⚠ red for
any skill with an unresolved blocking clarification — not EDA-specific). Middle: summary label.
Timestamp: muted mono. Bottom: "Open Report ↗" link if `report_html` set (opens that report's
deck on its cover slide), else greyed-out "No report".

---

## Guardrails

- **Every plot names itself** (`conventions/code_and_plot_conventions.md`). Each figure carries
  its declared `name` as the title, its declared `type` stated in the figure, and its
  `iteration - node` provenance, using the same strings recorded in the JSON. A plot that is
  unidentifiable once separated from its report is not finished.
- Never alter any history JSON field other than `report_html`.
- Never fabricate a number, finding, or fact — every number rendered must trace to the JSON.
- Summarizing prose is allowed and expected (see "Content priority" in the Purpose section) —
  but a summary may never drop a material qualifier (a confidence level, a caveat, a scope limit)
  and the full original text must always still be reachable via a `.details-btn`/`.details-panel` disclosure. Numbers,
  tables, and plots are never summarized/altered — only prose fields are.
- Never include `<img>` for a path not verified on disk. Show a text-only placeholder instead.
- No absolute filesystem paths in HTML — all paths relative to the reports/ directory.
- Every report deck must open on slide 1 with working sliding navigation (arrows/dots/keys) and
  correct slide count at `file://` with no server.
- Google Fonts `@import`/`<link>` is the only external dependency.
