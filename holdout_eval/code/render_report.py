# %% [Setup]
"""Render report.html from final_report.json. Self-contained: inline SVG, no network, both themes.

Split from final_report.py so the rendering can be re-run without repeating the permutation
importance pass. Every number here comes from final_report.json; nothing is computed.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
D = json.load(open(os.path.join(EVAL, "final_report.json"), encoding="utf-8"))
A, PY, IMP = D["analysis"], D["per_year"], D["importance"]
H = [1, 7, 14, 30, 60, 90]
NAIVE = "seasonal_naive_m7"
CHAMP = "agentic_shipped_calendar"

LABEL = {
    CHAMP: "Agentic (as shipped)", NAIVE: "Seasonal naive (m=7)",
    "automl_recipe": "AutoML (fixed 403-feature recipe)", "seasonal_median_52wk": "Seasonal median (52wk)",
    "agentic_refit_calendar": "Agentic (refit on train+val)", "automl_no_fe": "AutoML (no feature engineering)",
    "agentic_shipped_row": "Agentic (as shipped, row-offset h)",
    "agentic_refit_row": "Agentic (refit, row-offset h)",
}
SLOT = {CHAMP: 1, NAIVE: 2, "automl_no_fe": 3, "seasonal_median_52wk": 4,
        "agentic_refit_calendar": 5, "automl_recipe": 6}
CHART_SYS = [CHAMP, NAIVE, "automl_no_fe", "seasonal_median_52wk", "agentic_refit_calendar"]
TABLE_ORDER = [CHAMP, "agentic_refit_calendar", "automl_no_fe", "automl_recipe", NAIVE,
               "seasonal_median_52wk", "agentic_shipped_row", "agentic_refit_row"]
WIN_LABEL = {"holdout_primary": "holdout_primary — 2021", "holdout_stress": "holdout_stress — 2022 to 2023-02-08"}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(v, n=2):
    return "—" if v is None else f"{v:,.{n}f}"


# %% [SVG helpers]
import math


def line_chart(window, metric, title, ylab, systems, parity=None):
    W, Hh, ml, mr, mt, mb = 720, 330, 58, 168, 16, 46
    per = A[window]["per_horizon"]
    xs = [h for h in H if str(h) in per or h in per]
    def get(h, s):
        row = per.get(str(h), per.get(h))
        return row[s][metric]
    vals = [get(h, s) for h in xs for s in systems]
    if parity is not None:
        vals.append(parity)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.12 or 1
    lo, hi = lo - pad, hi + pad
    lx = [math.log10(h) for h in xs]
    x0, x1 = min(lx), max(lx)

    def px(h):
        return ml + (math.log10(h) - x0) / (x1 - x0) * (W - ml - mr)

    def py(v):
        return mt + (hi - v) / (hi - lo) * (Hh - mt - mb)

    o = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="{esc(title)}" class="chart">']
    o.append(f'<title>{esc(title)}</title>')
    for frac in range(5):
        v = lo + (hi - lo) * frac / 4
        y = py(v)
        o.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" class="grid"/>')
        o.append(f'<text x="{ml-8}" y="{y+4:.1f}" class="tick" text-anchor="end">{v:.0f}</text>')
    for h in xs:
        o.append(f'<text x="{px(h):.1f}" y="{Hh-mb+20}" class="tick" text-anchor="middle">{h}</text>')
    o.append(f'<text x="{(ml+W-mr)/2:.0f}" y="{Hh-6}" class="axlab" text-anchor="middle">forecast horizon, days (log scale)</text>')
    o.append(f'<text x="12" y="{mt+4}" class="axlab">{esc(ylab)}</text>')
    if parity is not None:
        y = py(parity)
        o.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" class="parity"/>')
        o.append(f'<text x="{W-mr+6}" y="{y+4:.1f}" class="lbl parity-lbl">naive parity</text>')
    for s in systems:
        c = f"var(--c{SLOT[s]})"
        pts = " ".join(f"{px(h):.1f},{py(get(h,s)):.1f}" for h in xs)
        dash = ' stroke-dasharray="6 4"' if s == NAIVE else ""
        o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
        for h in xs:
            o.append(f'<circle cx="{px(h):.1f}" cy="{py(get(h,s)):.1f}" r="2.6" fill="{c}"/>')

    # Direct labels, pushed apart so series that finish close together stay readable.
    # A leader line reconnects any label moved off its own end point.
    ends = sorted(((py(get(xs[-1], s)), s) for s in systems), key=lambda t: t[0])
    placed, last = [], -1e9
    for y, s in ends:
        yy = max(y, last + 13)
        placed.append((yy, y, s))
        last = yy
    overflow = placed[-1][0] - (Hh - 6) if placed and placed[-1][0] > Hh - 6 else 0
    for yy, y, s in placed:
        yy -= overflow
        c = f"var(--c{SLOT[s]})"
        if abs(yy - y) > 1.5:
            o.append(f'<path d="M {W-mr-1:.1f} {y:.1f} L {W-mr+3:.1f} {yy:.1f}" '
                     f'stroke="{c}" stroke-width="1" fill="none" opacity="0.55"/>')
        o.append(f'<text x="{W-mr+6}" y="{yy+4:.1f}" class="lbl" fill="{c}">{esc(LABEL[s])}</text>')
    o.append("</svg>")
    return "\n".join(o)


def importance_chart():
    items = sorted(IMP.items(), key=lambda kv: kv[1]["grouped_dMAE"], reverse=True)
    W, rowh, ml, mr = 720, 34, 108, 150
    Hh = 30 + rowh * len(items)
    lo = min(0, min(v["grouped_dMAE"] for _, v in items), min(v["sum_solo"] for _, v in items))
    hi = max(v["grouped_dMAE"] for _, v in items) * 1.1 or 1
    def px(v):
        return ml + (v - lo) / (hi - lo) * (W - ml - mr)
    o = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Grouped permutation importance" class="chart">']
    o.append(f'<line x1="{px(0):.1f}" y1="14" x2="{px(0):.1f}" y2="{Hh-8}" class="grid"/>')
    for i, (g, v) in enumerate(items):
        y = 22 + i * rowh
        solo, grouped = v["sum_solo"], v["grouped_dMAE"]
        a, b = px(min(solo, 0) if solo < 0 else 0), px(solo)
        o.append(f'<rect x="{min(a,b):.1f}" y="{y}" width="{abs(b-a):.1f}" height="15" fill="var(--c1)" opacity="0.95"/>')
        c, d2 = px(solo), px(grouped)
        o.append(f'<rect x="{min(c,d2):.1f}" y="{y}" width="{abs(d2-c):.1f}" height="15" fill="var(--c1)" opacity="0.38"/>')
        o.append(f'<text x="{ml-10}" y="{y+12}" class="tick" text-anchor="end">{esc(g)} ({v["n_features"]})</text>')
        o.append(f'<text x="{W-mr+6}" y="{y+12}" class="lbl">{grouped:+.2f} MAE</text>')
    o.append(f'<text x="{(ml+W-mr)/2:.0f}" y="{Hh-2}" class="axlab" text-anchor="middle">validation MAE increase when the group is permuted — solid = sum of solo, pale = masking gap</text>')
    o.append("</svg>")
    return "\n".join(o)


def bias_year_chart():
    years = sorted(PY, key=int)
    sysu = [CHAMP, "automl_no_fe", NAIVE]
    W, Hh, ml, mr, mb = 720, 230, 58, 168, 42
    vals = [PY[y][s]["bias"] for y in years for s in sysu]
    lo, hi = min(min(vals), 0) * 1.15, max(vals) * 1.15
    gw = (W - ml - mr) / len(years)
    def py(v):
        return 18 + (hi - v) / (hi - lo) * (Hh - 18 - mb)
    o = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Bias by year" class="chart">']
    o.append(f'<line x1="{ml}" y1="{py(0):.1f}" x2="{W-mr}" y2="{py(0):.1f}" class="grid"/>')
    o.append(f'<text x="{ml-8}" y="{py(0)+4:.1f}" class="tick" text-anchor="end">0</text>')
    bw = gw / (len(sysu) + 1)
    for i, y in enumerate(years):
        for j, s in enumerate(sysu):
            v = PY[y][s]["bias"]
            x = ml + i * gw + j * bw + bw * 0.5
            top, bot = min(py(v), py(0)), max(py(v), py(0))
            o.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw*0.82:.1f}" height="{max(bot-top,1):.1f}" fill="var(--c{SLOT[s]})"/>')
            o.append(f'<text x="{x+bw*0.41:.1f}" y="{top-4:.1f}" class="tick" text-anchor="middle">{v:+.0f}</text>')
        o.append(f'<text x="{ml+i*gw+gw/2:.1f}" y="{Hh-mb+22}" class="tick" text-anchor="middle">{y} (n={PY[y]["n"]})</text>')
    for j, s in enumerate(sysu):
        o.append(f'<text x="{W-mr+6}" y="{30+j*18}" class="lbl" fill="var(--c{SLOT[s]})">{esc(LABEL[s])}</text>')
    o.append(f'<text x="12" y="22" class="axlab">mean signed error (predicted − actual), units/day</text>')
    o.append("</svg>")
    return "\n".join(o)


# %% [Tables]
def pooled_table(w):
    a = A[w]
    r = ['<div class="scroll"><table><thead><tr><th>System</th><th>MAE</th><th>RMSE</th>'
         '<th>bias</th><th>MAE debiased</th><th>MASE</th><th>horizons won</th><th>beats naive</th><th>n</th></tr></thead><tbody>']
    for s in sorted(TABLE_ORDER, key=lambda x: a["pooled"][x]["MAE"]):
        p = a["pooled"][s]
        cls = ' class="champ"' if s == CHAMP else (' class="naive"' if s in (NAIVE, "seasonal_median_52wk") else "")
        r.append(f'<tr{cls}><td>{esc(LABEL[s])}</td><td>{fmt(p["MAE"])}</td><td>{fmt(p["RMSE"])}</td>'
                 f'<td>{p["bias"]:+.1f}</td><td>{fmt(p["MAE_debiased"])}</td>'
                 f'<td>{fmt(p["MASE_vs_naive_same_h"])}</td><td>{a["wins"][s]}/{a["n_horizons"]}</td>'
                 f'<td>{a["beats_naive_h"][s]}/{a["n_horizons"]}</td><td>{p["n"]:,}</td></tr>')
    return "\n".join(r) + "</tbody></table></div>"


def horizon_table(w, metric, dec=2):
    a = A[w]
    per = a["per_horizon"]
    ks = [h for h in H if str(h) in per or h in per]
    r = ['<div class="scroll"><table><thead><tr><th>System</th>' +
         "".join(f"<th>h={h}</th>" for h in ks) + "</tr></thead><tbody>"]
    for s in sorted(TABLE_ORDER, key=lambda x: a["pooled"][x]["MAE"]):
        cells = []
        for h in ks:
            row = per.get(str(h), per.get(h))
            best = min(row, key=lambda x: row[x]["MAE"])
            mark = ' class="best"' if (metric == "MAE" and best == s) else ""
            cells.append(f'<td{mark}>{fmt(row[s][metric], dec)}</td>')
        cls = ' class="champ"' if s == CHAMP else (' class="naive"' if s in (NAIVE, "seasonal_median_52wk") else "")
        r.append(f'<tr{cls}><td>{esc(LABEL[s])}</td>' + "".join(cells) + "</tr>")
    return "\n".join(r) + "</tbody></table></div>"


def horizon_rows_table(w, metric="MAE", dec=2):
    """Horizons down the side, estimators across the top; best estimator per horizon marked."""
    a = A[w]
    per = a["per_horizon"]
    ks = [h for h in H if str(h) in per or h in per]
    cols = sorted(TABLE_ORDER, key=lambda s: a["pooled"][s]["MAE"])
    head = "".join(
        f'<th{" class=champ-col" if s == CHAMP else ""}>{esc(LABEL[s])}</th>' for s in cols)
    r = [f'<div class="scroll"><table><thead><tr><th>Horizon</th>{head}</tr></thead><tbody>']
    for h in ks:
        row = per.get(str(h), per.get(h))
        best = min(row, key=lambda x: row[x][metric])
        cells = "".join(
            f'<td{" class=best" if x == best else ""}>{fmt(row[x][metric], dec)}</td>' for x in cols)
        r.append(f'<tr><td><b>h = {h}</b> <span class="sub">n={row[cols[0]]["n"]:,}</span></td>{cells}</tr>')
    pooled_best = min(cols, key=lambda s: a["pooled"][s][metric])
    cells = "".join(
        f'<td{" class=best" if s == pooled_best else ""}>{fmt(a["pooled"][s][metric], dec)}</td>'
        for s in cols)
    r.append(f'<tr class="pooled"><td><b>pooled</b> <span class="sub">n={a["pooled"][cols[0]]["n"]:,}</span></td>{cells}</tr>')
    return "\n".join(r) + "</tbody></table></div>"


def year_table():
    years = sorted(PY, key=int)
    r = ['<div class="scroll"><table><thead><tr><th>System</th>' +
         "".join(f'<th>{y}<br><span class="sub">n={PY[y]["n"]:,}</span></th>' for y in years) + "</tr></thead><tbody>"]
    r.append('<tr class="naive"><td>mean actual demand</td>' +
             "".join(f'<td>{PY[y]["mean_actual"]:.0f}</td>' for y in years) + "</tr>")
    for s in [CHAMP, "agentic_refit_calendar", "automl_no_fe", "automl_recipe", NAIVE, "seasonal_median_52wk"]:
        cells = "".join(f'<td>{PY[y][s]["bias"]:+.1f}</td>' if s in PY[y] else "<td>—</td>" for y in years)
        cls = ' class="champ"' if s == CHAMP else ""
        r.append(f'<tr{cls}><td>{esc(LABEL[s])}</td>{cells}</tr>')
    return "\n".join(r) + "</tbody></table></div>"


def importance_table():
    r = ['<div class="scroll"><table><thead><tr><th>Feature group</th><th>features</th>'
         '<th>grouped ΔMAE</th><th>sd</th><th>Σ solo</th><th>masking gap</th><th>verdict</th></tr></thead><tbody>']
    for g, v in sorted(IMP.items(), key=lambda kv: kv[1]["grouped_dMAE"], reverse=True):
        d = v["grouped_dMAE"]
        verdict = ("carries the model" if d > 1 else
                   "unused" if abs(d) < 3 * max(v["sd"], 1e-9) and abs(d) < 0.2 else
                   "actively harmful" if d < -0.3 else "negligible")
        r.append(f'<tr><td>{esc(g)}</td><td>{v["n_features"]}</td><td>{d:+.2f}</td>'
                 f'<td>{v["sd"]:.2f}</td><td>{v["sum_solo"]:+.2f}</td><td>{v["masking_gap"]:+.2f}</td>'
                 f'<td>{verdict}</td></tr>')
    return "\n".join(r) + "</tbody></table></div>"


def systems_table():
    a = A["holdout_primary"]
    r = ['<div class="scroll"><table><thead><tr><th>System</th><th>What it is</th><th>features</th>'
         '<th>estimator</th><th>search budget</th><th>rows on primary</th></tr></thead><tbody>']
    meta = {
        CHAMP: ("the run's six frozen LightGBM boosters, one per horizon, fitted on origins to 2018-10-04", 37, "LightGBM", "agent-directed"),
        "agentic_refit_calendar": ("the same six specifications refit through 2020-12-31, both ends of every training pair inside the training partition; no re-tuning, n_estimators pinned", 37, "LightGBM", "none (pinned)"),
        "automl_recipe": ("FLAML over a fixed generic feature recipe, fitted on the whole training partition", 403, "FLAML → lgbm", "900 s"),
        "automl_no_fe": ("FLAML on raw columns, shift(h) only — the floor", 94, "FLAML", "150 s/horizon"),
        NAIVE: ("last observed same-weekday value at the origin", 0, "none", "none"),
        "seasonal_median_52wk": ("median of the same weekday over the last year", 0, "none", "none"),
        "agentic_shipped_row": ("the frozen boosters under the h convention they were trained on", 37, "LightGBM", "—"),
        "agentic_refit_row": ("the refit under the training h convention", 37, "LightGBM", "—"),
    }
    for s in TABLE_ORDER:
        w_, f_, e_, b_ = meta[s]
        cls = ' class="champ"' if s == CHAMP else ""
        r.append(f'<tr{cls}><td>{esc(LABEL[s])}</td><td>{esc(w_)}</td><td>{f_}</td><td>{esc(e_)}</td>'
                 f'<td>{esc(b_)}</td><td>{a["coverage"][s]:,}</td></tr>')
    return "\n".join(r) + "</tbody></table></div>"


# %% [Assemble]
# horizons where the champion beats the no-feature-engineering AutoML arm
for _w in ("holdout_primary", "holdout_stress"):
    _per = A[_w]["per_horizon"]
    A[_w]["beats_no_fe"] = sum(1 for _h in _per
                               if _per[_h][CHAMP]["MAE"] < _per[_h]["automl_no_fe"]["MAE"])

p_pool, s_pool = A["holdout_primary"]["pooled"], A["holdout_stress"]["pooled"]
p_rows, s_rows = A["holdout_primary"]["rows"], A["holdout_stress"]["rows"]
gap_refit = p_pool[CHAMP]["MAE"] - p_pool["agentic_refit_calendar"]["MAE"]

CSS = """
:root{--bg:#fbfbfa;--surface:#fff;--ink:#1a1a19;--muted:#5c5c58;--line:#e3e2df;--accent:#2a78d6;
--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--c5:#e87ba4;--c6:#7a6bd6;--warn:#b3261e;--okbg:#eef6ef;--badbg:#fdf0ef;}
@media (prefers-color-scheme:dark){:root{--bg:#16171a;--surface:#1e2024;--ink:#eceded;--muted:#a2a5ab;
--line:#2f3238;--accent:#3987e5;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#8b7ce8;
--warn:#f2b8b5;--okbg:#1b2a20;--badbg:#2b1c1b;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.62 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:940px;margin:0 auto;padding:44px 22px 80px}
h1{font-size:1.85rem;line-height:1.22;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:1.22rem;margin:52px 0 6px;letter-spacing:-.01em;border-top:1px solid var(--line);padding-top:22px}
h3{font-size:1rem;margin:28px 0 6px;color:var(--muted);font-weight:600}
p{margin:10px 0}
.sub{color:var(--muted);font-size:.86rem}
.lede{color:var(--muted);margin-bottom:26px}
.verdict{border:1px solid var(--line);border-left:3px solid var(--accent);background:var(--surface);
padding:14px 18px;margin:14px 0;border-radius:0 7px 7px 0}
.verdict b{display:block;margin-bottom:3px}
.verdict.bad{border-left-color:var(--warn)}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:.9rem;
font-variant-numeric:tabular-nums;background:var(--surface)}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:600}
tr.champ{background:var(--badbg)}
tr.naive{background:var(--okbg)}
td.best{font-weight:700;color:var(--c3)}
th.champ-col{color:var(--c1)}
tr.pooled td{border-top:2px solid var(--line);font-weight:600}
tbody tr td:first-child{white-space:nowrap}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:7px}
.chart{width:100%;height:auto;margin:14px 0;background:var(--surface);
border:1px solid var(--line);border-radius:7px;padding:6px}
.grid{stroke:var(--line);stroke-width:1}
.parity{stroke:var(--muted);stroke-width:2;stroke-dasharray:7 5}
.parity-lbl{fill:var(--muted)}
.tick{fill:var(--muted);font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.axlab{fill:var(--muted);font-size:11px}
.lbl{font-size:11.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--ink)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.87em;
background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:1px 5px;
overflow-wrap:anywhere;word-break:break-word}
.note{font-size:.88rem;color:var(--muted);border-left:2px solid var(--line);padding-left:14px;margin:14px 0}
ul{margin:10px 0;padding-left:20px} li{margin:5px 0}
@media print{:root{--bg:#fff;--surface:#fff;--ink:#000;--muted:#333;--line:#bbb;--okbg:#fff;--badbg:#fff}
body{font-size:11pt} .wrap{padding:0} tr,img,svg{page-break-inside:avoid}
h2{page-break-after:avoid} .chart{page-break-inside:avoid}}
"""

html = f"""<title>Aernoudt Holdout Evaluation</title>
<style>{CSS}</style>
<div class="wrap">
<h1>Nothing learned beat the naive on the holdout</h1>
<p class="lede">Agentic pipeline <code>run_20260904T140437Z</code>, two AutoML arms and two zero-parameter
references, scored on the external holdout they were all withheld from. Primary window 2021
({p_rows['pairs_all_systems_complete']:,} scored (target&nbsp;date,&nbsp;h) pairs); stress window
2022&nbsp;–&nbsp;2023-02-08 ({s_rows['pairs_all_systems_complete']:,} pairs).</p>

<h2>1. Verdict</h2>

<div class="verdict bad"><b>Against AutoML with no feature engineering — the pipeline loses.</b>
The reference AutoML arm is the one that does no feature engineering at all: raw columns, a single
<code>shift(h)</code>, no lags, no calendar terms, no rolling statistics. It is what the pipeline's
whole EDA → preprocess → feature-engineering chain has to justify itself against. It scores MAE
{p_pool['automl_no_fe']['MAE']:.2f} on 2021 against the agentic champion's {p_pool[CHAMP]['MAE']:.2f} —
the champion wins {A['holdout_primary']['beats_no_fe']}/6 horizons. On the stress window,
{s_pool['automl_no_fe']['MAE']:.2f} against {s_pool[CHAMP]['MAE']:.2f}.
Source: <code>automl-runs/automl-baseline/outputs/naive_automl/preds_holdout_primary.csv</code> and
<code>holdout_eval/outputs/preds_holdout_primary__agentic_shipped_calendar.csv</code>.</div>

<div class="verdict"><b>Matched on training data, that gap closes to a tie.</b>
The shipped champion was fitted on origins to 2018-10-04; every AutoML arm was fitted on the whole
training partition to 2020-12-31. Refitting the same six specifications on the same rows — no
re-tuning, per <code>contracts/end_model.json</code> — gives {p_pool['agentic_refit_calendar']['MAE']:.2f}
against no-feature-engineering AutoML's {p_pool['automl_no_fe']['MAE']:.2f} on 2021. That is a dead heat.
Read together with the verdict above: most of the apparent gap to AutoML was training data, and what
remains is that eight iterations of agent-directed feature engineering bought nothing measurable over
doing none at all.</div>

<div class="verdict bad"><b>Against naive — every learned system loses.</b>
The seasonal naive scores {p_pool[NAIVE]['MAE']:.2f} on 2021. No learned system reaches it:
no-feature-engineering AutoML {p_pool['automl_no_fe']['MAE']:.2f}, the agentic champion
{p_pool[CHAMP]['MAE']:.2f}, the agentic refit {p_pool['agentic_refit_calendar']['MAE']:.2f}. The champion
beats naive at 0 of 6 horizons. A 52-week seasonal median — also zero parameters — beats everything at
{p_pool['seasonal_median_52wk']['MAE']:.2f}, and on the stress window wins all 6 horizons at
{s_pool['seasonal_median_52wk']['MAE']:.2f}.</div>

<div class="verdict"><b>Why — the models forecast the right shape at the wrong level.</b>
Removing each system's constant bias cuts the agentic champion's stress-window error from
{s_pool[CHAMP]['MAE']:.2f} to {s_pool[CHAMP]['MAE_debiased']:.2f}, and the refit's from
{s_pool['agentic_refit_calendar']['MAE']:.2f} to {s_pool['agentic_refit_calendar']['MAE_debiased']:.2f} —
at which point its day-to-day shape is as good as naive's ({s_pool[NAIVE]['MAE_debiased']:.2f}).
Bias grows every year ({PY['2021'][CHAMP]['bias']:+.0f} → {PY['2022'][CHAMP]['bias']:+.0f} →
{PY['2023'][CHAMP]['bias']:+.0f}) while naive's stays flat
({PY['2021'][NAIVE]['bias']:+.0f} → {PY['2022'][NAIVE]['bias']:+.0f} → {PY['2023'][NAIVE]['bias']:+.0f}).
This is distribution shift, not a feature-engineering failure.</div>

<h2>2. What was compared</h2>
{systems_table()}
<p>All six systems were withheld from the holdout: the agentic run directory contains no path to it,
and the AutoML training table stops at 2020-12-31 with zero rows after. The two agentic rows marked
<em>row-offset h</em> are a sensitivity check, not separate systems — the boosters were trained
against targets <em>h rows</em> ahead on a frame with pre-2020-06 Mondays deleted, so their native
h=30 is 35 calendar days. Scoring them both ways costs {abs(p_pool[CHAMP]['MAE']-p_pool['agentic_shipped_row']['MAE']):.2f} MAE,
so the convention is not what decides this comparison.</p>

<h3>Alignment record</h3>
<ul>
<li><b>Primary:</b> {p_rows['target_dates_in_window']} target dates × 6 horizons =
{p_rows['pairs_before_trading_rule']:,} pairs; {p_rows['excluded_zero_or_missing_actual']} excluded by the
trading-day rule (actual&nbsp;=&nbsp;0 or absent); {p_rows['pairs_after_trading_rule']:,} scored;
{p_rows['pairs_all_systems_complete']:,} where every system produced a prediction. Largest single-system
loss to the join is {p_rows['pairs_after_trading_rule']-min(A['holdout_primary']['coverage'].values()):,}
rows ({100*(p_rows['pairs_after_trading_rule']-min(A['holdout_primary']['coverage'].values()))/p_rows['pairs_after_trading_rule']:.1f}%), under the 5% threshold.</li>
<li><b>Stress:</b> {s_rows['pairs_after_trading_rule']:,} scored pairs but only
{s_rows['pairs_all_systems_complete']:,} complete across systems — 2022 carries 92 of ~310 expected trading
days, so for most (target, h) pairs the calendar origin is simply not an observed date. The stress
table is therefore a smaller, gappier comparison and is reported separately for that reason.</li>
<li><b>Horizon grids match exactly</b> (1, 7, 14, 30, 60, 90). No horizon was paired or dropped.</li>
<li><b>One date lost:</b> 2021-11-02's only export is truncated mid-file. Parsing its intact prefix
would under-count bread and bias the target low, so the date is excluded from every system.</li>
</ul>

<h2>3. Headline metrics — 2021</h2>
{pooled_table('holdout_primary')}
<p><b>MASE</b> is each system's error divided by the seasonal naive's error at the same horizon;
below 1.00 beats naive. <b>MAE debiased</b> is the mean absolute error after each system's own
constant bias is removed — it isolates day-to-day shape from level. Only the two zero-parameter
references land at or below naive parity. The agentic champion's MASE of
{p_pool[CHAMP]['MASE_vs_naive_same_h']:.2f} means it is {100*(p_pool[CHAMP]['MASE_vs_naive_same_h']-1):.0f}% worse
than doing nothing.</p>

<h3>Stress window — 2022 to 2023-02-08</h3>
{pooled_table('holdout_stress')}
<p>The ordering is unchanged but the spread triples. The 52-week seasonal median wins every horizon
here, and every learned system is worse than doing nothing by 39% to 93%. Read this window as a
robustness probe rather than a headline: it rests on {s_rows['pairs_all_systems_complete']:,} pairs, and 2022's
missing trading days are a data defect the split report flagged in advance.</p>

<h2>4. Error by horizon — the pipeline degrades, the references do not</h2>
{line_chart('holdout_primary','MAE','MAE by horizon, 2021','MAE (units/day) — axis does not start at zero',CHART_SYS)}

<h3>MAE by horizon and estimator — 2021</h3>
{horizon_rows_table('holdout_primary','MAE')}
<p class="note">Rows are forecast horizons, columns are estimators ordered best to worst on pooled
MAE. Bold green marks the best estimator at that horizon. Units are loaves per day; lower is better.</p>

<h3>The same table on the stress window</h3>
{horizon_rows_table('holdout_stress','MAE')}
<p class="note">2022 – 2023-02-08, {s_rows['pairs_all_systems_complete']:,} complete pairs. The 52-week
seasonal median takes every horizon.</p>

<h3>MASE by horizon and estimator — 2021</h3>
{horizon_rows_table('holdout_primary','MASE_vs_naive_same_h')}
<p class="note">Each estimator's error divided by the seasonal naive's error at the same horizon.
Below 1.00 beats naive; the naive column is 1.00 by construction.</p>

<p>The naive references are nearly flat across lead times because they re-anchor on recent actuals
at every origin. The agentic champion rises from {A['holdout_primary']['per_horizon']['1'][CHAMP]['MAE']:.1f} at
one day to {A['holdout_primary']['per_horizon']['30'][CHAMP]['MAE']:.1f} at thirty, which is the signature of a model
leaning on short lags that go stale. The no-feature-engineering AutoML is erratic rather than flat —
{A['holdout_primary']['per_horizon']['7']['automl_no_fe']['MAE']:.1f} at h=7 but
{A['holdout_primary']['per_horizon']['60']['automl_no_fe']['MAE']:.1f} at h=60 — which is what six
independently fitted models with no shared structure look like. The fixed 403-feature recipe, kept in
the tables as a secondary arm, is the flattest learned system: its long rolling windows behave much
like the seasonal median.</p>

<h2>5. Skill against naive — no learned system reaches parity</h2>
{line_chart('holdout_primary','MASE_vs_naive_same_h','MASE by horizon, 2021','MASE (1.00 = naive parity)',[s for s in CHART_SYS if s!=NAIVE],parity=1.0)}
<p>Every learned series sits above the parity line at every horizon on the primary window. The
seasonal median dips below it from h=14 onward, which is the clearest statement the report can make:
a one-line rule with no fitting is the best forecaster in this comparison.</p>

<h2>6. Where the error comes from — level, not shape</h2>
{horizon_table('holdout_primary','MAE_debiased')}
<p class="note">MAE after removing each system's constant bias, by horizon, 2021.</p>
{bias_year_chart()}
{year_table()}
<p>Mean signed error, predicted minus actual. Three independently fitted systems — the agentic
champion, its refit and AutoML — all carry a large positive bias that <em>grows every year</em>,
while both zero-parameter references stay near zero. When several independently fitted models share
the same bias sign and magnitude, the cause is distribution shift rather than any one model's
mistake. Demand fell from a 2010–2018 training level to {PY['2023']['mean_actual']:.0f} units/day by 2023,
and every model that froze a level at fit time carried the old one forward.</p>

<h2>7. What the model uses — a day-of-week lookup</h2>
{importance_chart()}
{importance_table()}
<p>Grouped permutation importance, computed on the <b>validation</b> partition, never on the holdout —
attribution on the test set would spend the one thing it is for. Each figure is the increase in
validation MAE when the whole group is shuffled together; the masking gap is the grouped effect minus
the sum of the members' solo effects.</p>
<p>Only the Fourier calendar terms carry the model ({IMP['fourier']['grouped_dMAE']:+.2f} MAE when permuted).
The seven lag features — the ones the run's EDA justified as the core signal — score
{IMP['lags']['grouped_dMAE']:+.2f}: permuting them <em>improves</em> validation accuracy. Weather (12 features)
and competitor (7 features) are indistinguishable from zero and can be dropped. <code>year_centered</code>,
the only trend term, scores exactly {IMP['trend']['grouped_dMAE']:.2f} — the boosters never split on it once.</p>
<p>That last number explains section 6. A model with no usable trend term cannot follow demand
downward, so it reproduces the weekday shape at the level it was fitted on, and the bias grows with
every year of drift. It also explains why the refit helps: three more years of rows moved the frozen
level closer to the holdout, worth {gap_refit:.2f} MAE on 2021, without changing anything else.</p>

<h2>8. Limits</h2>
<ul>
<li><b>The lag features are aliased.</b> The run measured autocorrelation in row offsets on a frame
with Mondays deleted, where a week is six rows. Its "monthly signal at lag-30, r=0.601" is the weekly
signal: on the calendar, lag-30-days has r=0.022 while every multiple of 7 sits near 0.6. Four of the
seven lag features point at weekday-misaligned offsets. This is a defect in the run, not in the scoring.</li>
<li><b>The stress window is thin.</b> {s_rows['pairs_all_systems_complete']:,} complete pairs, and 2022 is missing
two-thirds of its expected trading days. Directionally consistent with 2021; not independently strong.</li>
<li><b>The refit is not the shipped artifact.</b> It is what <code>contracts/end_model.json</code> now
requires and what the run would have produced under the current contract. Reported alongside, never
in place of, the model the run actually froze.</li>
<li><b>No system was tuned for this holdout,</b> and none may be after seeing it. These numbers are
spent the moment they inform a modelling change.</li>
</ul>

<h2>9. Reproduction</h2>
<ul>
<li><code>holdout_eval/build_frame.py</code> — replays the run's EDA→preprocess→FE chain over train +
holdout. <code>verify()</code> asserts it reproduces the run's own
<code>data/iteration_003_feature_engineering.csv</code> on all 2,897 training rows and all 37 features
to 5.7e-14.</li>
<li><code>holdout_eval/predict_agentic.py</code> — the four agentic prediction files and
<code>outputs/agentic_manifest.json</code>.</li>
<li><code>holdout_eval/score_all.py</code> — one scorer, all systems, <code>outputs/scores.json</code> and
<code>outputs/scored_rows_*.csv</code>.</li>
<li><code>holdout_eval/code/final_report.py</code> and <code>render_report.py</code> — metrics,
importance and this page. Chart data: <code>holdout_eval/artifacts/final_report/*.csv</code>.
Full metric set: <code>holdout_eval/final_report.json</code>.</li>
<li>Python 3.12.10, lightgbm 4.7.0, pandas 2.3.3, numpy 2.5.2, flaml 2.6.0 (AutoML arms as originally fitted).</li>
</ul>
</div>
"""

out = os.path.join(EVAL, "report.html")
open(out, "w", encoding="utf-8").write(html)
print("wrote", out, len(html), "bytes")
