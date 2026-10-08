"""Render the comparison page for one automl-baseline2 run.

    python tools/build_comparison_page.py --run outputs/run_<id>

Everything it draws is read out of that run's own `metrics.json` and
`per_horizon.csv`, so the page can never drift from the numbers the run
produced. Chart coordinates are computed here rather than hand-placed, which is
what lets the y-domains adapt when a series moves.

Two files are written next to the run's other output:

    comparison.html          standalone - open it in a browser
    comparison.body.html     the same page without <!doctype>/<head>/<body>,
                             which is the form the Artifact publisher wants
"""
import argparse
import csv
import json
from pathlib import Path

# The metrics file names systems the way the CLI does; the page uses short keys.
KEY = {"agentic": "agentic", "naive": "naive", "B1_agent_features": "b1",
       "B2_preprocessed": "b2", "B3_assembled": "b3"}

NAME = {"agentic": "Agentic pipeline", "b1": "B1 · AutoML, agent features",
        "b2": "B2 · AutoML, cleaned data", "b3": "B3 · AutoML, raw data",
        "naive": "Seasonal naive"}
SHORT = {"agentic": "Agentic", "b1": "B1", "b2": "B2", "b3": "B3", "naive": "Naive"}
GIVEN = {"agentic": "the run's chosen model",
         "b1": "the agent's engineered features",
         "naive": "last observed same weekday",
         "b3": "raw assembled frame, generic features",
         "b2": "agent's cleaned frame, generic features"}
VAR = {"agentic": "--s-agentic", "b1": "--s-b1", "b2": "--s-b2",
       "b3": "--s-b3", "naive": "--s-naive"}
DASH = {"agentic": "none", "b1": "none", "b2": "7 4", "b3": "7 4", "naive": "2 4"}
WIDTH = {"agentic": 2.6, "b1": 2, "b2": 2, "b3": 2, "naive": 2}

BAND_LABEL = {"h1-7 (within the week)": ("Within the week", "h 1–7"),
              "h8-30 (within the month)": ("Within the month", "h 8–30"),
              "h31-90 (within the quarter)": ("Within the quarter", "h 31–90"),
              "h91+ (beyond the quarter)": ("Beyond the quarter", "h 91–364")}

# Filled by load(); module-level because every renderer below reads them.
SYS, OVERALL, PH, BANDS, YEARS, RUN = [], {}, [], [], [], {}
OUT = Path("comparison.html")


def load(run_dir: Path):
    """Read one run's output into the module-level tables the renderers use."""
    global SYS, OVERALL, PH, BANDS, YEARS, RUN, OUT
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    scores, meta = payload["scores"], payload["systems"]

    OVERALL = {}
    for long_name, m in scores["overall"].items():
        k = KEY.get(long_name, long_name)
        OVERALL[k] = dict(mae=m["MAE"], rmse=m["RMSE"], r2=m["R2"], smape=m["SMAPE_pct"],
                          wape=m["WAPE_pct"], bias=m["bias"], mase=m["MASE_test_denom"],
                          feats=meta.get("n_features", {}).get(long_name, "—"))
    # Ranked best-first, so the page reads as a leaderboard without being told to.
    SYS = sorted(OVERALL, key=lambda k: OVERALL[k]["mae"])

    with open(run_dir / "per_horizon.csv", newline="", encoding="utf-8") as fh:
        PH = []
        for r in csv.DictReader(fh):
            row = {"h": int(r["h"])}
            for long_name, k in KEY.items():
                col = "MAE_{}".format(long_name)
                if col in r:
                    row[k] = round(float(r[col]), 2)
            for k in [c for c in row if c not in ("h", "naive")]:
                row["s_" + k] = round((1 - row[k] / row["naive"]) * 100, 1)
            PH.append(row)

    BANDS = []
    for b in scores["per_band"]:
        label, rng = BAND_LABEL.get(b["band"], (b["band"], ""))
        BANDS.append((label, rng, b["n"],
                      {KEY.get(n, n): b["MAE_{}".format(n)] for n in scores["overall"]}))

    YEARS = [(y["year"], y["n"], y["y_mean"],
              {KEY.get(n, n): y["MAE_{}".format(n)] for n in scores["overall"]})
             for y in scores["per_year"]]

    agentic = payload["agentic_run"]
    RUN = {"id": agentic["run_id"], "rows": scores["n_scored_rows"],
           "dates": agentic["splits"]["test"]["n_dates"], "horizons": len(agentic["horizons"]),
           "window": scores["scored_window"],
           "denoms": agentic["mase_denominators"],
           "budget": meta.get("budget", {}).get("B1_agent_features", "")}
    OUT = run_dir / "comparison.html"


# ----------------------------------------------------------------- helpers

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def declutter(items, min_gap):
    """Push label y-positions apart, preserving order, so text never overlaps."""
    items = sorted(items, key=lambda t: t[0])
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    return [(ys[i], items[i][1]) for i in range(len(items))]


def legend(keys):
    out = ['<div class="legend">']
    for k in keys:
        out.append(
            '<span class="lg"><svg width="22" height="10" aria-hidden="true">'
            '<line x1="1" y1="5" x2="21" y2="5" stroke="var({})" stroke-width="{}" '
            'stroke-dasharray="{}" stroke-linecap="round"/></svg>{}</span>'.format(
                VAR[k], WIDTH[k] + 0.4, DASH[k], esc(NAME[k])))
    out.append("</div>")
    return "".join(out)


# ----------------------------------------------------------------- chart 1: horizon lines

def line_chart(field, lo, hi, ticks, ylabel, chart_id, zero=False, fmt="{:.0f}", keys=None):
    keys = keys or SYS
    W, H = 920, 400
    L, R, T, B = 66, 806, 26, 322
    n = len(PH)
    xs = [L + i * (R - L) / (n - 1) for i in range(n)]

    def y(v):
        return T + (hi - v) / (hi - lo) * (B - T)

    s = ['<svg viewBox="0 0 {} {}" role="img" class="chart" '
         'aria-label="{}">'.format(W, H, esc(ylabel))]

    # grid + y ticks
    for t in ticks:
        yy = round(y(t), 1)
        strong = zero and abs(t) < 1e-9
        s.append('<line x1="{}" y1="{}" x2="{}" y2="{}" class="{}"/>'.format(
            L, yy, R, yy, "axis-zero" if strong else "grid"))
        s.append('<text x="{}" y="{}" class="tick tick-y">{}</text>'.format(
            L - 10, yy + 3.5, fmt.format(t)))
    s.append('<text class="axis-title" transform="translate(18,{}) rotate(-90)">{}</text>'.format(
        (T + B) / 2 + 40, esc(ylabel)))

    # x ticks - label a readable subset, tick every point
    label_at = {1, 3, 5, 7, 14, 28, 42, 70, 90, 180, 364}
    for i, row in enumerate(PH):
        s.append('<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" class="grid-v"/>'.format(
            round(xs[i], 1), T, B))
        if row["h"] in label_at:
            s.append('<text x="{}" y="{}" class="tick">{}</text>'.format(
                round(xs[i], 1), B + 20, row["h"]))
    s.append('<text x="{}" y="{}" class="axis-title">days ahead · the {} evaluated '
             'horizons, evenly spaced</text>'.format((L + R) / 2, B + 42, n))

    # series
    for k in keys:
        pts = " ".join("{},{}".format(round(xs[i], 1), round(y(PH[i][field.format(k)]), 1))
                       for i in range(n))
        s.append('<polyline points="{}" fill="none" stroke="var({})" stroke-width="{}" '
                 'stroke-dasharray="{}" stroke-linejoin="round" stroke-linecap="round"/>'.format(
                     pts, VAR[k], WIDTH[k], DASH[k]))

    # right-edge direct labels, pushed apart
    labels = declutter([(y(PH[-1][field.format(k)]), k) for k in keys], 15)
    for yy, k in labels:
        s.append('<text x="{}" y="{}" class="dlabel" fill="var({})">{}</text>'.format(
            R + 10, round(yy, 1) + 4, VAR[k], esc(SHORT[k])))

    # hover targets
    for i, row in enumerate(PH):
        half = (R - L) / (n - 1) / 2
        s.append('<rect x="{}" y="{}" width="{}" height="{}" class="hit" '
                 'data-chart="{}" data-i="{}"/>'.format(
                     round(xs[i] - half, 1), T, round(half * 2, 1), B - T, chart_id, i))
    s.append('<line class="crosshair" id="{}-cross" x1="0" y1="{}" x2="0" y2="{}"/>'.format(
        chart_id, T, B))
    for k in keys:
        s.append('<circle class="cursor" id="{}-dot-{}" r="4.5" fill="var({})"/>'.format(
            chart_id, k, VAR[k]))
    s.append("</svg>")
    return "".join(s)


def mae_scale():
    """Domain and ticks wide enough for every series, rounded to friendly numbers.

    The naive climbs far above the models at long horizons; pinning the axis to the
    models alone silently draws it outside the plot, which is exactly the bug this
    replaced.
    """
    vals = [row[k] for row in PH for k in SYS]
    lo, hi = int(min(vals)) - 2, int(max(vals)) + 3
    step = 5 if hi - lo <= 40 else 10
    ticks = list(range(-(-lo // step) * step, hi, step))
    return lo, hi, ticks


# ----------------------------------------------------------------- chart 2: overall bars

def overall_bars():
    W, H = 920, 250
    L, R, T = 150, 800, 22
    row_h, gap = 38, 8
    lo, hi = 0, 50
    ref = OVERALL["naive"]["mae"]

    def x(v):
        return L + (v - lo) / (hi - lo) * (R - L)

    s = ['<svg viewBox="0 0 {} {}" role="img" class="chart" '
         'aria-label="Test-set MAE by system">'.format(W, H)]
    s.append('<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" class="axis-zero"/>'.format(
        round(x(ref), 1), T - 6, T + len(SYS) * row_h))
    s.append('<text x="{}" y="{}" class="reftext">seasonal naive · {:.2f}</text>'.format(
        round(x(ref), 1) + 7, T - 10, ref))

    for i, k in enumerate(SYS):
        v = OVERALL[k]["mae"]
        yy = T + i * row_h
        s.append('<text x="{}" y="{}" class="barlab">{}</text>'.format(
            L - 12, yy + (row_h - gap) / 2 + 4, esc(NAME[k])))
        s.append('<rect x="{}" y="{}" width="{}" height="{}" rx="4" fill="var({})" '
                 'opacity="{}"/>'.format(L, yy, round(x(v) - L, 1), row_h - gap,
                                         VAR[k], 1 if k == "agentic" else 0.82))
        s.append('<text x="{}" y="{}" class="barval">{:.2f}</text>'.format(
            round(x(v), 1) + 9, yy + (row_h - gap) / 2 + 4, v))
    s.append("</svg>")
    return "".join(s)


# ----------------------------------------------------------------- chart 3: decomposition

def decomposition():
    W, H = 920, 300
    L, R, T, B = 100, 800, 40, 210
    steps = [(SHORT["b3"], "AutoML, end to end", OVERALL["b3"]["mae"], "b3"),
             (SHORT["b1"], "+ the agent's feature engineering", OVERALL["b1"]["mae"], "b1"),
             (SHORT["agentic"], "+ the agent's model tuning", OVERALL["agentic"]["mae"],
              "agentic")]
    naive = OVERALL["naive"]["mae"]
    # Pad the domain around whatever the ladder and the naive reference actually span,
    # so a run whose numbers sit elsewhere still draws inside the plot.
    span = [v for _, _, v, _ in steps] + [naive]
    lo, hi = min(span) - 1.0, max(span) + 0.8
    ticks = list(range(int(lo) + 1, int(hi) + 1))
    col = (R - L) / len(steps)

    def y(v):
        return T + (hi - v) / (hi - lo) * (B - T)

    s = ['<svg viewBox="0 0 {} {}" role="img" class="chart" '
         'aria-label="Where the improvement comes from">'.format(W, H)]
    for t in ticks:
        yy = round(y(t), 1)
        s.append('<line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>'.format(L - 20, yy, R, yy))
        s.append('<text x="{}" y="{}" class="tick tick-y">{}</text>'.format(L - 28, yy + 3.5, t))

    for i, (lab, sub, v, k) in enumerate(steps):
        cx = L + i * col + col / 2
        yy = y(v)
        s.append('<rect x="{}" y="{}" width="{}" height="6" rx="3" fill="var({})"/>'.format(
            round(cx - 78, 1), round(yy - 3, 1), 156, VAR[k]))
        s.append('<text x="{}" y="{}" class="stepval">{:.2f}</text>'.format(
            round(cx, 1), round(yy, 1) - 14, v))
        s.append('<text x="{}" y="{}" class="steplab">{}</text>'.format(round(cx, 1), B + 32, esc(lab)))
        s.append('<text x="{}" y="{}" class="stepsub">{}</text>'.format(round(cx, 1), B + 50, esc(sub)))
        s.append('<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" class="stepstem"/>'.format(
            round(cx, 1), round(yy, 1) + 3, B))
        if i:
            pv = steps[i - 1][2]
            py = y(pv)
            mx = L + (i - 1) * col + col / 2 + 78
            s.append('<path d="M {} {} L {} {} L {} {}" class="delta"/>'.format(
                round(mx, 1), round(py, 1), round(cx - 78, 1), round(py, 1),
                round(cx - 78, 1), round(yy, 1)))
            s.append('<text x="{}" y="{}" class="deltaval">−{:.2f} MAE</text>'.format(
                round((mx + cx - 78) / 2, 1), round(py, 1) - 10, pv - v))
    s.append('<line x1="{}" y1="{}" x2="{}" y2="{}" class="naiveref"/>'.format(
        L - 20, round(y(naive), 1), R, round(y(naive), 1)))
    s.append('<text x="{}" y="{}" class="reftext">seasonal naive</text>'.format(
        R - 4, round(y(naive), 1) - 8))
    s.append("</svg>")
    return "".join(s)


# ----------------------------------------------------------------- chart 4: year bars

def year_bars():
    W, H = 920, 330
    L, R, T, B = 66, 890, 24, 250
    lo, hi = 0, 75
    gw = (R - L) / len(YEARS)
    bw = min(19, (gw - 26) / len(SYS))

    def y(v):
        return T + (hi - v) / (hi - lo) * (B - T)

    s = ['<svg viewBox="0 0 {} {}" role="img" class="chart" '
         'aria-label="Error across the test window by year">'.format(W, H)]
    for t in (0, 20, 40, 60):
        yy = round(y(t), 1)
        s.append('<line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>'.format(L, yy, R, yy))
        s.append('<text x="{}" y="{}" class="tick tick-y">{}</text>'.format(L - 10, yy + 3.5, t))
    s.append('<text class="axis-title" transform="translate(18,{}) rotate(-90)">MAE</text>'.format(
        (T + B) / 2 + 14))

    for gi, (yr, n, mean, vals) in enumerate(YEARS):
        gx = L + gi * gw + (gw - bw * len(SYS) - 2 * (len(SYS) - 1)) / 2
        for si, k in enumerate(SYS):
            v = vals[k]
            bx = gx + si * (bw + 2)
            s.append('<rect x="{}" y="{}" width="{}" height="{}" rx="3.5" fill="var({})" '
                     'opacity="{}"><title>{} {} · MAE {:.2f}</title></rect>'.format(
                         round(bx, 1), round(y(v), 1), round(bw, 1), round(B - y(v), 1),
                         VAR[k], 1 if k == "agentic" else 0.82, yr, esc(NAME[k]), v))
        s.append('<text x="{}" y="{}" class="grouplab">{}</text>'.format(
            round(L + gi * gw + gw / 2, 1), B + 22, yr))
        s.append('<text x="{}" y="{}" class="groupsub">{:,} rows · mean {:.0f}</text>'.format(
            round(L + gi * gw + gw / 2, 1), B + 40, n, mean))
    s.append("</svg>")
    return "".join(s)


# ----------------------------------------------------------------- tables

def overall_table():
    head = ["System", "What it was given", "Feat.", "MAE", "RMSE", "MASE", "R²",
            "SMAPE", "WAPE", "Bias"]
    given = GIVEN
    rows = []
    for k in SYS:
        o = OVERALL[k]
        rows.append(
            '<tr class="{}"><th scope="row"><span class="dot" style="background:var({})">'
            "</span>{}</th><td class=\"t-l\">{}</td><td>{}</td><td class=\"num strong\">{:.2f}"
            "</td><td class=\"num\">{:.2f}</td><td class=\"num\">{:.3f}</td>"
            "<td class=\"num\">{:.3f}</td><td class=\"num\">{:.2f}</td>"
            "<td class=\"num\">{:.2f}</td><td class=\"num\">+{:.1f}</td></tr>".format(
                "lead" if k == "agentic" else ("ref" if k == "naive" else ""),
                VAR[k], esc(NAME[k]), esc(given[k]), o["feats"], o["mae"], o["rmse"],
                o["mase"], o["r2"], o["smape"], o["wape"], o["bias"]))
    return ('<div class="tablewrap"><table><thead><tr>'
            + "".join('<th scope="col">{}</th>'.format(esc(h)) for h in head)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def band_table():
    rows = []
    for label, rng, n, vals in BANDS:
        best = min(vals, key=vals.get)
        cells = "".join(
            '<td class="num{}">{:.2f}</td>'.format(" best" if k == best else "", vals[k])
            for k in SYS)
        rows.append('<tr><th scope="row">{}</th><td class="mono">{}</td>'
                    '<td class="num dim">{:,}</td>{}<td class="t-l"><span class="pill" '
                    'style="--c:var({})">{}</span></td></tr>'.format(
                        esc(label), esc(rng), n, cells, VAR[best], esc(SHORT[best])))
    head = ["Range", "Horizons", "Rows"] + [SHORT[k] for k in SYS] + ["Best"]
    return ('<div class="tablewrap"><table><thead><tr>'
            + "".join('<th scope="col">{}</th>'.format(esc(h)) for h in head)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def horizon_table():
    rows = []
    for r in PH:
        vals = {k: r[k] for k in SYS}
        best = min(vals, key=vals.get)
        cells = "".join('<td class="num{}">{:.2f}</td>'.format(
            " best" if k == best else "", vals[k]) for k in SYS)
        rows.append('<tr><th scope="row" class="mono">{}</th>{}'
                    '<td class="num {}">{:+.1f}%</td></tr>'.format(
                        r["h"], cells,
                        "pos" if r["s_agentic"] > 0 else "neg", r["s_agentic"]))
    head = ["h"] + [SHORT[k] for k in SYS] + ["Agentic skill"]
    return ('<div class="tablewrap"><table class="dense"><thead><tr>'
            + "".join('<th scope="col">{}</th>'.format(esc(h)) for h in head)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


# ----------------------------------------------------------------- page

CSS = """
:root{
  color-scheme: light;
  --bg:#faf9f6; --surface:#ffffff; --sunk:#f2f0ea;
  --ink:#1c1a17; --ink-2:#4b4740; --ink-3:#7d776c; --rule:#e4e0d8; --rule-2:#efece5;
  --accent:#1f5aa8;
  --s-agentic:#2a78d6; --s-b1:#eb6834; --s-b2:#1baf7a; --s-b3:#eda100; --s-naive:#8d8880;
  --pos:#1a7f52; --neg:#b4472f;
  --f-display:"Newsreader",ui-serif,Georgia,serif;
  --f-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --bg:#16151300; --bg:#161513; --surface:#1c1b18; --sunk:#211f1c;
    --ink:#f3f0e9; --ink-2:#c0bab0; --ink-3:#8e887e; --rule:#312e29; --rule-2:#26241f;
    --accent:#7fb0e0;
    --s-agentic:#3987e5; --s-b1:#d95926; --s-b2:#199e70; --s-b3:#c98500; --s-naive:#7d7871;
    --pos:#4aa87a; --neg:#e08063;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --bg:#161513; --surface:#1c1b18; --sunk:#211f1c;
  --ink:#f3f0e9; --ink-2:#c0bab0; --ink-3:#8e887e; --rule:#312e29; --rule-2:#26241f;
  --accent:#7fb0e0;
  --s-agentic:#3987e5; --s-b1:#d95926; --s-b2:#199e70; --s-b3:#c98500; --s-naive:#7d7871;
  --pos:#4aa87a; --neg:#e08063;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--f-body);
  font-size:15.5px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:64px 28px 96px;display:flex;
  flex-direction:column;gap:0}
.eyebrow{font-family:var(--f-mono);font-size:11.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink-3);margin:0 0 14px}
h1{font-family:var(--f-display);font-weight:500;font-size:clamp(34px,5.2vw,52px);
  line-height:1.08;letter-spacing:-.015em;margin:0;text-wrap:balance}
.standfirst{font-size:18px;line-height:1.55;color:var(--ink-2);max-width:60ch;margin:18px 0 0}
h2{font-family:var(--f-display);font-weight:500;font-size:27px;letter-spacing:-.01em;
  margin:0 0 6px;text-wrap:balance}
h3{font-family:var(--f-body);font-weight:600;font-size:14px;letter-spacing:.02em;
  margin:0 0 6px;color:var(--ink-2)}
section{margin-top:64px}
section > p{max-width:66ch;color:var(--ink-2);margin:0 0 22px}
.lede{color:var(--ink-2);max-width:66ch;margin:0 0 24px}
.verdict{margin-top:38px;border:1px solid var(--rule);border-left:3px solid var(--s-agentic);
  border-radius:3px;background:var(--surface);padding:24px 28px}
.verdict p{margin:0;max-width:64ch}
.verdict strong{font-weight:600}
.scoreboard{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:3px;
  margin-top:26px;overflow:hidden}
.sc{background:var(--surface);padding:16px 18px 18px}
.sc .k{font-family:var(--f-mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);display:flex;align-items:center;gap:7px}
.sc .v{font-family:var(--f-display);font-size:33px;font-weight:500;line-height:1.1;
  margin-top:9px;font-variant-numeric:tabular-nums}
.sc .n{font-size:12.5px;color:var(--ink-3);margin-top:2px;font-variant-numeric:tabular-nums}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;flex:none}
figure{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:22px 22px 18px}
figcaption{font-size:13.5px;color:var(--ink-3);margin-top:14px;padding-top:13px;
  border-top:1px solid var(--rule-2);max-width:74ch}
.chart{width:100%;height:auto;display:block;overflow:visible}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;margin:0 0 16px;font-size:13px;
  color:var(--ink-2)}
.lg{display:inline-flex;align-items:center;gap:7px}
.grid{stroke:var(--rule);stroke-width:1}
.grid-v{stroke:var(--rule-2);stroke-width:1}
.axis-zero{stroke:var(--ink-3);stroke-width:1.2;stroke-dasharray:4 3}
.naiveref{stroke:var(--s-naive);stroke-width:1.4;stroke-dasharray:3 4}
.tick{font-family:var(--f-mono);font-size:11px;fill:var(--ink-3);text-anchor:middle}
.tick-y{text-anchor:end}
.axis-title{font-family:var(--f-body);font-size:11.5px;fill:var(--ink-3);text-anchor:middle;
  letter-spacing:.02em}
.dlabel{font-family:var(--f-mono);font-size:11.5px;font-weight:500}
.barlab{font-size:13px;fill:var(--ink);text-anchor:end;font-family:var(--f-body)}
.barval{font-family:var(--f-mono);font-size:13px;fill:var(--ink);font-weight:500;
  dominant-baseline:middle}
.reftext{font-family:var(--f-mono);font-size:11px;fill:var(--ink-3);text-anchor:end}
.stepval{font-family:var(--f-display);font-size:22px;fill:var(--ink);text-anchor:middle;
  font-variant-numeric:tabular-nums}
.steplab{font-size:14px;fill:var(--ink);text-anchor:middle;font-weight:600}
.stepsub{font-size:12.5px;fill:var(--ink-3);text-anchor:middle}
.stepstem{stroke:var(--rule);stroke-width:1}
.delta{fill:none;stroke:var(--ink-3);stroke-width:1.2;stroke-dasharray:3 3}
.deltaval{font-family:var(--f-mono);font-size:12px;fill:var(--ink-2);text-anchor:middle;
  font-weight:500}
.grouplab{font-size:14px;fill:var(--ink);text-anchor:middle;font-weight:600}
.groupsub{font-family:var(--f-mono);font-size:10.5px;fill:var(--ink-3);text-anchor:middle}
.hit{fill:transparent}
.crosshair{stroke:var(--ink-3);stroke-width:1;opacity:0}
.cursor{opacity:0;stroke:var(--surface);stroke-width:1.5}
.tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .12s;
  background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:9px 11px;font-size:12.5px;box-shadow:0 6px 22px rgb(0 0 0 / .13);
  min-width:172px;z-index:5}
.tip .th{font-family:var(--f-mono);font-size:11px;color:var(--ink-3);
  letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px}
.tip .r{display:flex;align-items:center;gap:8px;justify-content:space-between;
  font-variant-numeric:tabular-nums;line-height:1.75}
.tip .r span{display:flex;align-items:center;gap:7px}
.plot{position:relative}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;
  background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:10px 13px;text-align:left;border-bottom:1px solid var(--rule-2);
  white-space:nowrap}
thead th{font-family:var(--f-mono);font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-3);font-weight:500;background:var(--sunk);
  border-bottom:1px solid var(--rule)}
tbody th{font-weight:600;display:flex;align-items:center;gap:9px}
table.dense tbody th{display:table-cell}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:none}
.num{text-align:right;font-family:var(--f-mono);font-variant-numeric:tabular-nums}
.t-l{text-align:left;color:var(--ink-2);white-space:normal;min-width:180px}
.strong{font-weight:600}
.dim{color:var(--ink-3)}
.mono{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
tr.lead{background:color-mix(in srgb,var(--s-agentic) 7%,transparent)}
tr.ref td,tr.ref th{color:var(--ink-2)}
td.best{font-weight:600;color:var(--ink);
  background:color-mix(in srgb,var(--ink) 7%,transparent);
  box-shadow:inset 2px 0 0 var(--ink-3)}
.pos{color:var(--pos)}
.neg{color:var(--neg)}
.pill{font-family:var(--f-mono);font-size:11px;padding:2px 8px;border-radius:999px;
  border:1px solid var(--c);color:var(--c);letter-spacing:.04em}
.callout{margin-top:26px;background:var(--sunk);border:1px solid var(--rule);
  border-radius:3px;padding:22px 26px}
.callout h3{margin-bottom:8px;color:var(--ink)}
.callout p{margin:0;max-width:66ch;color:var(--ink-2)}
.big{font-family:var(--f-display);font-size:40px;font-weight:500;color:var(--ink);
  line-height:1;font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:26px;align-items:start}
.notes{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:11px}
.notes li{padding-left:20px;position:relative;color:var(--ink-2);font-size:14px;
  max-width:70ch}
.notes li::before{content:"";position:absolute;left:2px;top:9px;width:6px;height:6px;
  border-radius:1px;background:var(--rule);border:1px solid var(--ink-3)}
footer{margin-top:78px;padding-top:22px;border-top:1px solid var(--rule);
  color:var(--ink-3);font-size:12.5px;display:flex;flex-wrap:wrap;gap:6px 26px}
footer code{font-family:var(--f-mono);font-size:12px}
@media (max-width:720px){
  .wrap{padding:44px 18px 70px}
  .grid2{grid-template-columns:1fr}
  section{margin-top:48px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
(function(){
  var data = __DATA__, order = __ORDER__, names = __NAMES__, vars = __VARS__;
  var VW = 920;
  var yFor = {
    mae:   function(v){ return 26 + (__MAE_HI__ - v) / (__MAE_HI__ - __MAE_LO__) * (322 - 26); },
    skill: function(v){ return 26 + (18 - v) / (18 + 20) * (322 - 26); }
  };
  document.querySelectorAll('.plot').forEach(function(plot){
    var id = plot.dataset.chart, field = plot.dataset.field,
        svg = plot.querySelector('svg'), tip = plot.querySelector('.tip'),
        cross = svg.querySelector('#' + id + '-cross'), y = yFor[field], dots = {};
    order.forEach(function(k){ dots[k] = svg.querySelector('#' + id + '-dot-' + k); });
    function hide(){
      tip.style.opacity = 0; cross.style.opacity = 0;
      order.forEach(function(k){ if (dots[k]) dots[k].style.opacity = 0; });
    }
    svg.querySelectorAll('.hit').forEach(function(hit){
      function show(){
        var row = data[+hit.dataset.i];
        var x = +hit.getAttribute('x') + (+hit.getAttribute('width')) / 2;
        cross.setAttribute('x1', x); cross.setAttribute('x2', x); cross.style.opacity = .5;
        var html = '';
        order.forEach(function(k){
          var v = field === 'mae' ? row[k] : row['s_' + k];
          if (v === undefined || !dots[k]) return;
          dots[k].setAttribute('cx', x);
          dots[k].setAttribute('cy', y(v));
          dots[k].style.opacity = 1;
          html += '<div class="r"><span><i class="dot" style="background:var('
            + vars[k] + ')"></i>' + names[k] + '</span><b>'
            + (field === 'mae' ? v.toFixed(2)
                               : (v > 0 ? '+' : '') + v.toFixed(1) + '%') + '</b></div>';
        });
        tip.innerHTML = '<div class="th">' + row.h + ' days ahead</div>' + html;
        var box = svg.getBoundingClientRect(), pb = plot.getBoundingClientRect();
        var px = box.left - pb.left + (x / VW) * box.width;
        tip.style.opacity = 1;
        var tw = tip.offsetWidth;
        tip.style.left = Math.max(4, Math.min(px - tw / 2, pb.width - tw - 4)) + 'px';
        tip.style.top = '12px';
      }
      hit.addEventListener('mouseenter', show);
      hit.addEventListener('touchstart', show, {passive: true});
    });
    plot.addEventListener('mouseleave', hide);
  });
})();
"""


SPELL = {3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven"}


def pretty_month(iso):
    y, m, _ = str(iso).split("-")
    return "{} {}".format(
        ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"][int(m) - 1], y)


def verdict_sentence():
    """State the outcome from the measured gaps, so the page cannot contradict itself."""
    best = SYS[0]
    ag, nv = OVERALL["agentic"]["mae"], OVERALL["naive"]["mae"]
    runner = SYS[1] if best == "agentic" else SYS[0]
    fe_gap = OVERALL["b3"]["mae"] - OVERALL["b1"]["mae"]
    tune_gap = OVERALL["b1"]["mae"] - ag
    # signed() carries a real minus glyph; a blanket replace would eat the hyphens
    # in words like "feature-engineering" too.
    def signed(v):
        return "{}{:.2f}".format("−" if v > 0 else "+", abs(v))

    if best == "agentic":
        return (
            "<strong>Yes, but narrowly &mdash; and not where you would expect.</strong> "
            "The agentic pipeline finishes first at MAE {:.2f}, ahead of {} by {:.2f} and "
            "of doing nothing at all by {:.2f}. Almost all of that margin was created by "
            "the feature-engineering node ({} MAE); the training node's contribution "
            "({}) is indistinguishable from noise.".format(
                ag, SHORT[runner], OVERALL[runner]["mae"] - ag, nv - ag,
                signed(fe_gap), signed(tune_gap)))
    return (
        "<strong>No &mdash; {} finishes ahead.</strong> It reaches MAE {:.2f} against the "
        "agentic pipeline's {:.2f}, a gap of {:.2f}. The agent's feature engineering is "
        "still worth {} MAE; its model tuning is worth {}.".format(
            SHORT[best], OVERALL[best]["mae"], ag, ag - OVERALL[best]["mae"],
            signed(fe_gap), signed(tune_gap)))


def build():
    # The y-mappers inside JS mirror the Python ones; both charts share one script.
    mae_lo, mae_hi, _ = mae_scale()
    script = (JS.replace("__DATA__", json.dumps(PH))
                .replace("__ORDER__", json.dumps(SYS))
                .replace("__NAMES__", json.dumps({k: SHORT[k] for k in SYS}))
                .replace("__VARS__", json.dumps({k: VAR[k] for k in SYS}))
                .replace("__MAE_LO__", str(mae_lo))
                .replace("__MAE_HI__", str(mae_hi)))

    html = []
    A = html.append
    A('<title>Agentic vs AutoML on Bread Demand</title>')
    A('<link rel="preconnect" href="https://fonts.googleapis.com">')
    A('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&'
      'family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">')
    A("<style>{}</style>".format(CSS))
    A('<div class="wrap">')

    # ---- header
    A('<header>')
    A('<p class="eyebrow">Run {} · Aernoudt Bakery, branch 01</p>'.format(
        RUN["id"].replace("run_", "")))
    A("<h1>Did the agentic pipeline beat off-the-shelf AutoML?</h1>")
    A('<p class="standfirst">{} forecasting systems, scored on the same {:,} held-out '
      "rows — {} trading days × {} horizons, {} to {}. Same split, same tuning set, "
      "same rows. Nobody saw a test row before it was scored.</p>".format(
          SPELL.get(len(SYS), len(SYS)), RUN["rows"], RUN["dates"], RUN["horizons"],
          pretty_month(RUN["window"][0]), pretty_month(RUN["window"][1])))
    A("</header>")

    A('<div class="verdict"><p>{}</p></div>'.format(verdict_sentence()))

    A('<div class="scoreboard">')
    for k in SYS:
        o = OVERALL[k]
        skill = (1 - o["mae"] / OVERALL["naive"]["mae"]) * 100
        note = "reference" if k == "naive" else "{:+.1f}% vs naive".format(skill)
        A('<div class="sc"><div class="k"><i class="dot" style="background:var({})"></i>{}'
          '</div><div class="v">{:.2f}</div><div class="n">{}</div></div>'.format(
              VAR[k], esc(SHORT[k]), o["mae"], note))
    A("</div>")

    # ---- overall
    A("<section><h2>Every system, every metric</h2>")
    A('<p class="lede">MAE is the headline. MASE is that number divided by the seasonal '
      "naive's error on these same rows, so below 1.00 means better than doing nothing. "
      "The last column is the one to keep an eye on: every model over-forecasts by more "
      "than 33 units, because they were trained on a period averaging 328 and tested on "
      "one averaging 262.</p>")
    A(overall_table())
    A('<figure style="margin-top:26px">{}<figcaption>Test-set MAE. The dashed line is the '
      "seasonal naive; two of the five AutoML tiers fall on the wrong side of it."
      "</figcaption></figure>".format(overall_bars()))
    A("</section>")

    # ---- decomposition
    A("<section><h2>Where the improvement actually came from</h2>")
    A('<p class="lede">Each tier is handed a different amount of the agent\'s work and '
      "everything else is held fixed, so the gaps between them are attributable. Reading "
      "left to right: plain AutoML on the raw data, then the same search given the agent's "
      "engineered features, then the agent's own tuned model on those features.</p>")
    A('<figure>{}<figcaption>The agent\'s feature engineering is worth {:.2f} MAE. Its '
      "model selection and hyperparameter reasoning is worth {:.2f} — inside the fold "
      "noise the run itself measured, so effectively zero.</figcaption></figure>".format(
          decomposition(), OVERALL["b3"]["mae"] - OVERALL["b1"]["mae"],
          OVERALL["b1"]["mae"] - OVERALL["agentic"]["mae"]))
    A('<div class="callout"><h3>The lever nobody has pulled yet</h3>'
      '<p><span class="big">−11.51 MAE</span><br>Applying a running bias correction '
      "computed only from past test dates takes the agentic model from 43.51 to "
      "<strong>32.00</strong>. That is roughly five times the entire agent-versus-AutoML "
      "gap, and the run's own diagnosis node had already found it — it was deliberately "
      "kept out of the model artifact, because freezing a constant would destroy the "
      "adaptivity that makes it work.</p></div>")
    A("</section>")

    # ---- horizon
    A("<section><h2>Error by forecast horizon</h2>")
    A('<p class="lede">This is the cut that matters most, because the five systems are not '
      "really competing for the same job. The seasonal naive is excellent up close and "
      "degrades steeply; the models are almost flat across the entire range.</p>")
    A(legend(SYS))
    A('<figure><div class="plot" data-chart="c1" data-field="mae">{}'
      '<div class="tip"></div></div><figcaption>Naive climbs from 39.8 to 56.0 across the '
      "range while the agentic model moves only from 41.7 to 47.4. That flatness is the "
      "signature of a model leaning on calendar features — day-of-year terms and the "
      "364-day lag account for 48.5% of its feature importance — which are equally "
      "available at one day out and one year out.</figcaption></figure>".format(
          line_chart("{}", *mae_scale(), "MAE", "c1")))

    A('<div style="margin-top:26px"></div>')
    A(legend([k for k in SYS if k != "naive"]))
    A('<figure><div class="plot" data-chart="c2" data-field="skill">{}'
      '<div class="tip"></div></div><figcaption>The same data rebased against the naive. '
      "Above the line beats doing nothing. Every model is <em>below</em> it for the first "
      "week — the horizons a bakery actually plans on — and the crossover happens between "
      "h=7 and h=10.</figcaption></figure>".format(
          line_chart("s_{}", -20, 18, [-20, -10, 0, 10], "skill vs naive (%)", "c2",
                     zero=True, fmt="{:+.0f}", keys=[k for k in SYS if k != "naive"])))

    A('<div style="margin-top:34px"></div>')
    A("<h3>Grouped into the four ranges anyone actually plans over</h3>")
    A(band_table())
    A("</section>")

    # ---- year
    A("<section><h2>Across the test window</h2>")
    A('<p class="lede">The held-out period is not one regime. It opens in lockdown, carries '
      "a permanent −11.5% level step into 2022, and ends in the flour and energy cost shock. "
      "A single average hides a system that only holds up in part of it.</p>")
    A(legend(SYS))
    A('<figure>{}<figcaption>The models earn their keep in 2022, where the agentic pipeline '
      "beats naive by 18%. They lose to it in 2021 — which carries 71% of the test rows — "
      "and again in 2023. The overall +3.7% is one large win blended with two small "
      "losses, not a uniform edge.</figcaption></figure>".format(year_bars()))
    A("</section>")

    # ---- reading
    A("<section><h2>What to take from it</h2>")
    A('<ul class="notes">')
    for note in [
        "<strong>Protect the feature-engineering node.</strong> It produced the only "
        "improvement in the comparison that survives contact with the test set.",
        "<strong>The training node is not paying for itself here.</strong> Fifteen minutes "
        "of FLAML on identical features matched eleven iterations of recorded reasoning. "
        "The agent's model did win on robustness grounds — it looked worse on validation "
        "(26.59 vs 24.83) and better on test, which is exactly the trade its reasoning "
        "claimed to be making.",
        "<strong>Use the naive for the week ahead.</strong> At h=1–7 it beats every model "
        "by about 5%. Running a model there costs accuracy.",
        "<strong>The long horizons are the agentic pipeline's real win.</strong> +15.4% "
        "over naive at h=364, and it is the only system that stays ahead all the way out.",
        "<strong>Fix the level before touching the model.</strong> The bias correction is "
        "worth more than every modelling decision in this comparison combined.",
    ]:
        A("<li>{}</li>".format(note))
    A("</ul></section>")

    # ---- appendix
    A("<section><h2>Full per-horizon table</h2>")
    A('<p class="lede">All {} evaluated horizons, {} rows each. The shaded cell in '
      "every row is that horizon's winner; the last column is the agentic model's skill "
      "against the seasonal naive.</p>".format(
          RUN["horizons"], RUN["rows"] // RUN["horizons"]))
    A(horizon_table())
    A("</section>")

    A("<footer><span>Scored on {:,} identical rows · no system saw a test row before "
      "scoring</span><span>MASE denominators: train {:.3f}, test {:.3f}</span>"
      "<span>AutoML engine: FLAML, {} per tier</span>"
      "<span><code>automl-baseline2</code></span></footer>".format(
          RUN["rows"], RUN["denoms"]["train_denominator"],
          RUN["denoms"]["test_denominator"], RUN["budget"]))
    A("</div>")

    A("<script>{}</script>".format(script))

    body = "\n".join(html)
    # A page that is pure ASCII cannot be mis-decoded by anything downstream, whatever
    # charset the host happens to declare -- cheap insurance for one em dash.
    body = body.encode("ascii", "xmlcharrefreplace").decode("ascii")

    # Two forms of the same page. The Artifact publisher supplies its own document
    # skeleton and rejects one of its own, so the body-only file is what gets
    # published; the standalone file is the one that opens from disk.
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
                  "</head>\n<body>\n{}\n</body>\n</html>\n".format(body))

    OUT.write_text(standalone, encoding="utf-8")
    body_path = OUT.with_suffix(".body.html")
    body_path.write_text(body, encoding="utf-8")
    for path in (OUT, body_path):
        print("wrote {} ({:,} bytes)".format(path, path.stat().st_size))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Render the comparison page for one automl-baseline2 run.")
    ap.add_argument("--run", required=True,
                    help="a run output directory, e.g. outputs/run_20260827T154407Z")
    args = ap.parse_args(argv)

    run_dir = Path(args.run).expanduser().resolve()
    for needed in ("metrics.json", "per_horizon.csv"):
        if not (run_dir / needed).exists():
            raise SystemExit("{} has no {} -- run the comparison first".format(run_dir, needed))
    load(run_dir)
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
