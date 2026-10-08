"""Plots, Markdown and a self-contained HTML report for one comparison."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEM_ORDER = ("agentic", "B1_agent_features", "B2_preprocessed", "B3_assembled", "naive")

SYSTEM_LABEL = {
    "agentic": "Agentic pipeline",
    "B1_agent_features": "B1 - AutoML on agent features",
    "B2_preprocessed": "B2 - AutoML on cleaned data",
    "B3_assembled": "B3 - AutoML on assembled data",
    "naive": "Seasonal naive",
}

SYSTEM_BLURB = {
    "agentic": "the run's chosen model",
    "B1_agent_features": "identical features and rows; AutoML chose the estimator",
    "B2_preprocessed": "the agent's cleaned frame, generic lags and calendar terms",
    "B3_assembled": "the assembled frame, generic lags and calendar terms",
    "naive": "last observed same-weekday value at the origin",
}

# A colour per system, stable across every plot so the eye can track one line.
COLOURS = {
    "agentic": "#1f4e79",
    "B1_agent_features": "#2e8b57",
    "B2_preprocessed": "#c47f17",
    "B3_assembled": "#8b5a9f",
    "naive": "#9a9a9a",
}


def _order(names):
    known = [n for n in SYSTEM_ORDER if n in names]
    return known + [n for n in names if n not in known]


def _label(name: str) -> str:
    return SYSTEM_LABEL.get(name, name)


# --------------------------------------------------------------------- plots


def _figure(plot_dir: Path, name: str, fig) -> dict:
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / "{}.png".format(name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return {"name": name, "path": str(path),
            "b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def make_plots(scored: dict, table: pd.DataFrame, run, plot_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = _order(list(scored["overall"]))
    plots = []

    # 1. Overall MAE -- the headline, as a bar chart so the gaps are read as sizes
    #    rather than decoded from a table.
    fig, ax = plt.subplots(figsize=(8, 4))
    maes = [scored["overall"][n]["MAE"] for n in names]
    bars = ax.bar([_label(n) for n in names], maes,
                  color=[COLOURS.get(n, "#666") for n in names])
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v + max(maes) * 0.01, "{:.2f}".format(v),
                ha="center", fontsize=9)
    ax.set_ylabel("MAE (units of bread)")
    ax.set_title("Test-set MAE on {} identical rows".format(scored["n_scored_rows"]))
    ax.tick_params(axis="x", labelrotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.grid(axis="y", alpha=0.3)
    plots.append(_figure(plot_dir, "mae_overall", fig))

    # 2. MAE by horizon -- where in the forecast range each system wins or loses.
    ph = pd.DataFrame(scored["per_horizon"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n in names:
        ax.plot(ph["h"], ph["MAE_{}".format(n)], marker="o", markersize=3.5,
                label=_label(n), color=COLOURS.get(n, "#666"),
                linewidth=2.0 if n == "agentic" else 1.4,
                linestyle="--" if n == "naive" else "-")
    ax.set_xscale("log")
    ax.set_xticks(ph["h"])
    ax.set_xticklabels([str(int(h)) for h in ph["h"]], fontsize=7)
    ax.set_xlabel("horizon h (days ahead, log scale)")
    ax.set_ylabel("MAE")
    ax.set_title("Error by forecast horizon")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plots.append(_figure(plot_dir, "mae_by_horizon", fig))

    # 3. Skill against the seasonal naive -- the same data as (2), but rebased so
    #    the question "is this worth more than doing nothing" is answered directly.
    if "naive" in names:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for n in names:
            if n == "naive":
                continue
            skill = (1 - ph["MAE_{}".format(n)] / ph["MAE_naive"]) * 100
            ax.plot(ph["h"], skill, marker="o", markersize=3.5, label=_label(n),
                    color=COLOURS.get(n, "#666"),
                    linewidth=2.0 if n == "agentic" else 1.4)
        ax.axhline(0, color="#9a9a9a", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xticks(ph["h"])
        ax.set_xticklabels([str(int(h)) for h in ph["h"]], fontsize=7)
        ax.set_xlabel("horizon h (days ahead, log scale)")
        ax.set_ylabel("skill vs seasonal naive (%)")
        ax.set_title("Above the line beats doing nothing")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plots.append(_figure(plot_dir, "skill_by_horizon", fig))

    # 4. Error over the test window by year -- the test period spans a regime the
    #    training data never saw, so a single average can hide a system that only
    #    holds up in part of it.
    py = pd.DataFrame(scored["per_year"])
    if len(py) > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.8 / len(names)
        x = np.arange(len(py))
        for i, n in enumerate(names):
            ax.bar(x + i * width, py["MAE_{}".format(n)], width, label=_label(n),
                   color=COLOURS.get(n, "#666"))
        ax.set_xticks(x + width * (len(names) - 1) / 2)
        ax.set_xticklabels(py["year"].astype(int))
        ax.set_ylabel("MAE")
        ax.set_title("Error across the test window")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plots.append(_figure(plot_dir, "mae_by_year", fig))

    # 5. Forecast against actuals at h=7, the horizon a bakery actually plans on.
    sub = table.loc[table["h"] == 7].sort_values(run.time_col)
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.plot(sub[run.time_col], sub["y_true"], color="#333", linewidth=1.2, label="actual")
        for n in names:
            col = "pred_{}".format(n)
            if col in sub.columns:
                ax.plot(sub[run.time_col], sub[col], linewidth=1.0, alpha=0.85,
                        label=_label(n), color=COLOURS.get(n, "#666"))
        ax.set_ylabel("bread units")
        ax.set_title("Test window at h=7 (one week ahead)")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)
        plots.append(_figure(plot_dir, "forecast_h7", fig))

    # 6. Residual distributions -- the run's chosen model carries a large negative
    #    mean residual, and a boxplot says whether that is a shift or a few tails.
    fig, ax = plt.subplots(figsize=(8, 4))
    data, labels = [], []
    for n in names:
        col = "pred_{}".format(n)
        if col in table.columns:
            resid = (table["y_true"] - table[col]).dropna()
            data.append(resid.to_numpy())
            labels.append(_label(n))
    # matplotlib renamed this keyword in 3.9; support both so the report does not
    # depend on which version happens to be installed.
    try:
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLOURS.get(n, "#666"))
        patch.set_alpha(0.55)
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_ylabel("residual (actual - predicted)")
    ax.set_title("Residual spread and shift")
    ax.tick_params(axis="x", labelrotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.grid(axis="y", alpha=0.3)
    plots.append(_figure(plot_dir, "residual_spread", fig))

    return plots


# ------------------------------------------------------------------ markdown


def _md_table(rows, columns=None) -> str:
    if not rows:
        return "_(no rows)_\n"
    columns = columns or list(rows[0])
    out = ["| " + " | ".join(str(c) for c in columns) + " |",
           "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    return "\n".join(out) + "\n"


def build_markdown(scored: dict, verdict: dict, run, meta: dict) -> str:
    names = _order(list(scored["overall"]))
    d = scored["mase_denominators"]

    lines = []
    lines.append("# AutoML baseline vs agentic pipeline\n")
    lines.append("Run `{}` | target `{}` | scored on **{} rows** "
                 "({} target dates x {} horizons), {} to {}.\n".format(
                     run.run_dir.name, run.target, scored["n_scored_rows"],
                     run.splits["test"].n_dates, len(run.horizons),
                     scored["scored_window"][0], scored["scored_window"][1]))
    lines.append("Every system below was fitted on the same training partition, tuned on the "
                 "same validation partition, and scored on the same held-out rows. No system "
                 "saw a test row before it was scored.\n")

    lines.append("\n## Headline\n")
    best = verdict["best"]
    champ = verdict["champion"]
    if verdict["champion_is_best"]:
        lines.append("**The agentic pipeline wins.** MAE {:.3f} against {:.3f} for the closest "
                     "AutoML tier ({}).\n".format(
                         scored["overall"][champ]["MAE"],
                         min(scored["overall"][n]["MAE"] for n in names if n != champ),
                         _label(verdict["ranking_by_MAE"][1])))
    else:
        gap = scored["overall"][champ]["MAE"] - scored["overall"][best]["MAE"]
        lines.append("**{} wins.** MAE {:.3f} against the agentic pipeline's {:.3f} -- "
                     "a gap of {:.3f} ({:.1f}%).\n".format(
                         _label(best), scored["overall"][best]["MAE"],
                         scored["overall"][champ]["MAE"], gap,
                         gap / scored["overall"][best]["MAE"] * 100))

    lines.append("\n## What each system was given\n")
    lines.append(_md_table([
        {"system": _label(n), "what it is": SYSTEM_BLURB.get(n, ""),
         "features": meta.get("n_features", {}).get(n, "-"),
         "estimator": meta.get("estimator", {}).get(n, "-")}
        for n in names
    ], ["system", "what it is", "features", "estimator"]))

    lines.append("\n## Test-set metrics\n")
    lines.append(_md_table([
        dict(system=_label(n), **{k: v for k, v in scored["overall"][n].items() if k != "n"})
        for n in names
    ], ["system", "MAE", "RMSE", "R2", "SMAPE_pct", "WAPE_pct", "bias",
        "MASE_train_denom", "MASE_test_denom"]))
    lines.append("\nMASE against the **train** denominator ({:.3f}) is comparable with every "
                 "figure the agentic run quotes. MASE against the **test** denominator "
                 "({:.3f}) is the same-window verdict: below 1.0 means better than the "
                 "seasonal naive on these rows.\n".format(
                     d["train_denominator"], d["test_denominator"]))

    lines.append("\n## Skill against the seasonal naive\n")
    skill = verdict.get("skill_vs_naive_pct", {})
    lines.append(_md_table([
        {"system": _label(n), "MAE": scored["overall"][n]["MAE"],
         "skill vs naive (%)": skill.get(n, "-"),
         "horizons won (of {})".format(len(scored["per_horizon"])): verdict["horizons_won"].get(n, 0)}
        for n in names
    ], ["system", "MAE", "skill vs naive (%)",
        "horizons won (of {})".format(len(scored["per_horizon"]))]))

    lines.append("\n## By horizon band\n")
    lines.append(_md_table([
        {"band": r["band"], "n": r["n"],
         **{_label(n): r["MAE_{}".format(n)] for n in names}}
        for r in scored["per_band"]
    ], ["band", "n"] + [_label(n) for n in names]))

    lines.append("\n## By horizon\n")
    lines.append(_md_table([
        {"h": r["h"], "n": r["n"], "actual mean": r["y_mean"],
         **{_label(n): r["MAE_{}".format(n)] for n in names}}
        for r in scored["per_horizon"]
    ], ["h", "n", "actual mean"] + [_label(n) for n in names]))

    lines.append("\n## Across the test window\n")
    lines.append(_md_table([
        {"year": r["year"], "n": r["n"], "actual mean": r["y_mean"],
         **{_label(n): r["MAE_{}".format(n)] for n in names}}
        for r in scored["per_year"]
    ], ["year", "n", "actual mean"] + [_label(n) for n in names]))

    lines.append("\n## Protocol\n")
    for note in meta.get("protocol_notes", []):
        lines.append("- {}".format(note))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- html

_CSS = """
:root{--ink:#16191d;--muted:#5b6470;--line:#e2e6ea;--bg:#ffffff;--accent:#1f4e79;--soft:#f6f8fa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:48px 24px 80px}
h1{font-size:30px;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:20px;margin:44px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--line)}
p{margin:10px 0}
.sub{color:var(--muted);font-size:14px;margin-bottom:26px}
.verdict{background:var(--soft);border-left:4px solid var(--accent);
  padding:16px 20px;border-radius:0 6px 6px 0;margin:22px 0}
.verdict strong{color:var(--accent)}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}
.card{flex:1 1 170px;border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.card .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:26px;font-weight:600;margin-top:4px}
.card .n{font-size:12px;color:var(--muted);margin-top:2px}
.tablewrap{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:8px 11px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--soft);font-weight:600;color:var(--muted);
  text-transform:uppercase;font-size:11.5px;letter-spacing:.04em}
tbody tr:hover{background:var(--soft)}
tr.win td{background:#eef6ef;font-weight:600}
figure{margin:22px 0}
figure img{width:100%;border:1px solid var(--line);border-radius:8px}
figcaption{color:var(--muted);font-size:13px;margin-top:7px}
ul{margin:10px 0 10px 20px;padding:0}
li{margin:5px 0}
code{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:13px}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12.5px}
@media (prefers-color-scheme:dark){
  :root{--ink:#e6e9ee;--muted:#9aa4b0;--line:#2b313a;--bg:#14171b;--accent:#7fb0e0;--soft:#1c2027}
  figure img{background:#fff}
  tr.win td{background:#1d2a20}
}
"""


def _html_table(rows, columns, highlight_min=()) -> str:
    if not rows:
        return "<p><em>no rows</em></p>"
    head = "".join("<th>{}</th>".format(c) for c in columns)
    body = []
    for r in rows:
        cls = ' class="win"' if r.get("_win") else ""
        cells = "".join("<td>{}</td>".format(r.get(c, "")) for c in columns)
        body.append("<tr{}>{}</tr>".format(cls, cells))
    return ('<div class="tablewrap"><table><thead><tr>{}</tr></thead>'
            "<tbody>{}</tbody></table></div>".format(head, "".join(body)))


def build_html(scored: dict, verdict: dict, run, meta: dict, plots: list) -> str:
    names = _order(list(scored["overall"]))
    best = verdict["best"]
    d = scored["mase_denominators"]
    by_name = {p["name"]: p for p in plots}

    cards = []
    for n in names:
        met = scored["overall"][n]
        cards.append(
            '<div class="card"><div class="k">{}</div><div class="v">{:.2f}</div>'
            '<div class="n">MAE &middot; MASE {:.3f}</div></div>'.format(
                _label(n), met["MAE"], met["MASE_test_denom"]))

    if verdict["champion_is_best"]:
        runner = verdict["ranking_by_MAE"][1]
        headline = (
            "<strong>The agentic pipeline wins.</strong> MAE {:.3f} against {:.3f} for the "
            "closest AutoML tier ({}) &mdash; {:.1f}% better, on {} identical held-out rows."
        ).format(scored["overall"]["agentic"]["MAE"], scored["overall"][runner]["MAE"],
                 _label(runner),
                 (scored["overall"][runner]["MAE"] - scored["overall"]["agentic"]["MAE"])
                 / scored["overall"][runner]["MAE"] * 100,
                 scored["n_scored_rows"])
    else:
        gap = scored["overall"]["agentic"]["MAE"] - scored["overall"][best]["MAE"]
        headline = (
            "<strong>{} wins.</strong> MAE {:.3f} against the agentic pipeline's {:.3f} "
            "&mdash; a gap of {:.3f} ({:.1f}%), on {} identical held-out rows."
        ).format(_label(best), scored["overall"][best]["MAE"],
                 scored["overall"]["agentic"]["MAE"], gap,
                 gap / scored["overall"][best]["MAE"] * 100, scored["n_scored_rows"])

    def metric_rows():
        rows = []
        champ_mae = min(scored["overall"][n]["MAE"] for n in names)
        for n in names:
            m = scored["overall"][n]
            rows.append({"system": _label(n), "MAE": "{:.3f}".format(m["MAE"]),
                         "RMSE": "{:.3f}".format(m["RMSE"]), "R&sup2;": "{:.4f}".format(m["R2"]),
                         "SMAPE %": "{:.2f}".format(m["SMAPE_pct"]),
                         "WAPE %": "{:.2f}".format(m["WAPE_pct"]),
                         "bias": "{:.2f}".format(m["bias"]),
                         "MASE (train)": "{:.4f}".format(m["MASE_train_denom"]),
                         "MASE (test)": "{:.4f}".format(m["MASE_test_denom"]),
                         "_win": m["MAE"] == champ_mae})
        return rows

    skill = verdict.get("skill_vs_naive_pct", {})
    won_col = "horizons won (of {})".format(len(scored["per_horizon"]))

    sections = []
    sections.append("<h2>What each system was given</h2>")
    sections.append(_html_table([
        {"system": _label(n), "what it is": SYSTEM_BLURB.get(n, ""),
         "features": meta.get("n_features", {}).get(n, "&mdash;"),
         "estimator": meta.get("estimator", {}).get(n, "&mdash;"),
         "search budget": meta.get("budget", {}).get(n, "&mdash;")}
        for n in names
    ], ["system", "what it is", "features", "estimator", "search budget"]))

    sections.append("<h2>Test-set metrics</h2>")
    sections.append(_html_table(metric_rows(),
                                ["system", "MAE", "RMSE", "R&sup2;", "SMAPE %", "WAPE %",
                                 "bias", "MASE (train)", "MASE (test)"]))
    sections.append(
        "<p>MASE against the <strong>train</strong> denominator ({:.3f}) is comparable with "
        "every figure the agentic run quotes. MASE against the <strong>test</strong> "
        "denominator ({:.3f}) is the same-window verdict &mdash; below 1.0 means better than "
        "the seasonal naive on these rows.</p>".format(d["train_denominator"], d["test_denominator"]))

    if "mae_overall" in by_name:
        sections.append('<figure><img src="data:image/png;base64,{}" alt="MAE by system">'
                        "<figcaption>Test-set MAE. Every bar is computed over the same "
                        "{} rows.</figcaption></figure>".format(
                            by_name["mae_overall"]["b64"], scored["n_scored_rows"]))

    sections.append("<h2>Skill against the seasonal naive</h2>")
    sections.append(_html_table([
        {"system": _label(n), "MAE": "{:.3f}".format(scored["overall"][n]["MAE"]),
         "skill vs naive": "{:+.2f}%".format(skill[n]) if n in skill else "&mdash;",
         won_col: verdict["horizons_won"].get(n, 0),
         "_win": n == best}
        for n in names
    ], ["system", "MAE", "skill vs naive", won_col]))

    for key, caption in (
        ("skill_by_horizon", "Percent better than the seasonal naive at each horizon. "
                             "Above zero beats doing nothing."),
        ("mae_by_horizon", "Absolute error at each of the {} scored horizons.".format(
            len(scored["per_horizon"]))),
    ):
        if key in by_name:
            sections.append('<figure><img src="data:image/png;base64,{}" alt="{}">'
                            "<figcaption>{}</figcaption></figure>".format(
                                by_name[key]["b64"], key, caption))

    sections.append("<h2>By horizon band</h2>")
    sections.append(_html_table([
        {"band": r["band"], "n": r["n"],
         **{_label(n): "{:.2f}".format(r["MAE_{}".format(n)]) for n in names}}
        for r in scored["per_band"]
    ], ["band", "n"] + [_label(n) for n in names]))

    sections.append("<h2>Across the test window</h2>")
    sections.append(_html_table([
        {"year": r["year"], "n": r["n"], "actual mean": r["y_mean"],
         **{_label(n): "{:.2f}".format(r["MAE_{}".format(n)]) for n in names}}
        for r in scored["per_year"]
    ], ["year", "n", "actual mean"] + [_label(n) for n in names]))

    for key, caption in (
        ("mae_by_year", "The test window spans a regime the training data never saw, so a "
                        "single average can hide a system that only holds up in part of it."),
        ("forecast_h7", "One week ahead, across the whole held-out window."),
        ("residual_spread", "Residual spread and shift. A box centred below zero means the "
                            "system over-forecasts."),
    ):
        if key in by_name:
            sections.append('<figure><img src="data:image/png;base64,{}" alt="{}">'
                            "<figcaption>{}</figcaption></figure>".format(
                                by_name[key]["b64"], key, caption))

    sections.append("<h2>By horizon</h2>")
    sections.append(_html_table([
        {"h": r["h"], "n": r["n"], "actual mean": r["y_mean"],
         **{_label(n): "{:.2f}".format(r["MAE_{}".format(n)]) for n in names}}
        for r in scored["per_horizon"]
    ], ["h", "n", "actual mean"] + [_label(n) for n in names]))

    sections.append("<h2>Protocol</h2><ul>{}</ul>".format(
        "".join("<li>{}</li>".format(n) for n in meta.get("protocol_notes", []))))

    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoML baseline vs agentic pipeline</title><style>{css}</style></head>
<body><div class="wrap">
<h1>AutoML baseline vs agentic pipeline</h1>
<p class="sub">Run <code>{run}</code> &middot; target <code>{target}</code> &middot;
{n} scored rows ({dates} target dates &times; {h} horizons) &middot; {w0} to {w1}</p>
<div class="verdict">{headline}</div>
<div class="cards">{cards}</div>
<p>Every system was fitted on the same training partition, tuned on the same validation
partition, and scored on the same held-out rows. No system saw a test row before it was
scored.</p>
{sections}
<footer>Generated by automl-baseline2, which reuses the v1 package
(<code>automl_baseline.features</code>, <code>.metrics</code>, <code>.search</code>)
for feature construction and scoring.</footer>
</div></body></html>""".format(
        css=_CSS, run=run.run_dir.name, target=run.target,
        n=scored["n_scored_rows"], dates=run.splits["test"].n_dates, h=len(run.horizons),
        w0=scored["scored_window"][0], w1=scored["scored_window"][1],
        headline=headline, cards="".join(cards), sections="\n".join(sections))


def write_all(out_dir: Path, scored: dict, verdict: dict, table: pd.DataFrame,
              run, meta: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = make_plots(scored, table, run, out_dir / "plots")
    table.to_csv(out_dir / "predictions.csv", index=False)

    payload = {"agentic_run": run.to_dict(), "systems": meta, "scores": scored, "verdict": verdict}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    with open(out_dir / "report.md", "w", encoding="utf-8") as fh:
        fh.write(build_markdown(scored, verdict, run, meta))
    with open(out_dir / "report.html", "w", encoding="utf-8") as fh:
        fh.write(build_html(scored, verdict, run, meta, plots))

    pd.DataFrame(scored["per_horizon"]).to_csv(out_dir / "per_horizon.csv", index=False)

    return {
        "metrics_json": str(out_dir / "metrics.json"),
        "predictions_csv": str(out_dir / "predictions.csv"),
        "per_horizon_csv": str(out_dir / "per_horizon.csv"),
        "report_md": str(out_dir / "report.md"),
        "report_html": str(out_dir / "report.html"),
        "plots": [p["path"] for p in plots],
    }
