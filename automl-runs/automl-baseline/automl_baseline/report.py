"""Plots and reports for a baseline run."""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACCENT = "#2563eb"
MUTED = "#94a3b8"
WARN = "#f97316"
GOOD = "#16a34a"


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "-"
    return "{:,.{d}f}".format(f, d=digits)


def make_plots(result: dict, predictions: pd.DataFrame, frame: pd.DataFrame, target: str, run_dir: Path) -> list:
    plots = []
    plot_dir = Path(run_dir) / "plots"
    ok = [e for e in result["per_horizon"] if e["status"] == "ok"]
    cutoff = pd.Timestamp(result["protocol"]["cutoff"])

    # 1. The series itself, with the evaluation cutoff marked.
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(frame.index, frame[target], color=ACCENT, lw=0.9)
    ax.axvline(cutoff, color=WARN, ls="--", lw=1.4)
    ax.annotate("cutoff\n{}".format(cutoff.date()), xy=(cutoff, ax.get_ylim()[1]),
                xytext=(-6, -28), textcoords="offset points", ha="right", fontsize=8, color=WARN)
    ax.set_title("{} - full series (train | held-out)".format(target))
    ax.set_ylabel(target)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "series_overview.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    plots.append({"path": str(path), "reason": "shows the series and where the held-out window starts"})

    # 2. MASE by horizon, AutoML vs the naive references.
    horizons = [e["horizon"] for e in ok]
    x = np.arange(len(horizons))
    series = {
        "AutoML": [e["automl"]["mase"] for e in ok],
        "seasonal naive": [e["baselines"]["seasonal_naive"]["mase"] for e in ok],
        "naive last": [e["baselines"]["naive_last"]["mase"] for e in ok],
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.26
    for i, (label, values) in enumerate(series.items()):
        ax.bar(x + (i - 1) * width, values, width,
               label=label, color=[ACCENT, MUTED, "#cbd5e1"][i])
    ax.axhline(1.0, color="#334155", ls=":", lw=1)
    ax.text(len(horizons) - 0.4, 1.02, "MASE = 1 (naive floor)", fontsize=8, color="#334155")
    ax.set_xticks(x)
    ax.set_xticklabels(["t+{}".format(h) for h in horizons])
    ax.set_ylabel("MASE (lower is better)")
    ax.set_title("Forecast accuracy by horizon")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "mase_by_horizon.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    plots.append({"path": str(path), "reason": "headline comparison - the number the agentic pipeline has to beat"})

    if predictions.empty:
        return plots

    # 3. Absolute error spread per horizon.
    fig, ax = plt.subplots(figsize=(9, 4))
    data = [
        np.abs(predictions.loc[predictions.horizon == h, "y_true"] -
               predictions.loc[predictions.horizon == h, "pred_automl"]).dropna().to_numpy()
        for h in horizons
    ]
    ticks = ["t+{}".format(h) for h in horizons]
    try:
        bp = ax.boxplot(data, tick_labels=ticks, patch_artist=True, showfliers=False)
    except TypeError:  # matplotlib < 3.9
        bp = ax.boxplot(data, labels=ticks, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#dbeafe")
        patch.set_edgecolor(ACCENT)
    for median in bp["medians"]:
        median.set_color(WARN)
    ax.set_ylabel("|error| in units of {}".format(target))
    ax.set_title("Error spread across held-out origins")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "error_spread_by_horizon.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    plots.append({"path": str(path), "reason": "how much accuracy varies origin to origin, not just on average"})

    # 4. Actual vs predicted inside the held-out window, per horizon.
    for h in horizons:
        block = predictions.loc[predictions.horizon == h].sort_values("target_time")
        fig, ax = plt.subplots(figsize=(11, 3.4))
        ax.plot(block.target_time, block.y_true, color="#0f172a", lw=1.3, label="actual")
        ax.plot(block.target_time, block.pred_automl, color=ACCENT, lw=1.3, label="AutoML")
        if "pred_seasonal_naive" in block:
            ax.plot(block.target_time, block.pred_seasonal_naive, color=MUTED, lw=1.0, ls="--",
                    label="seasonal naive")
        ax.set_title("Held-out forecasts at horizon t+{}".format(h))
        ax.set_ylabel(target)
        ax.legend(frameon=False, fontsize=9, ncol=3)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = plot_dir / "forecast_h{}.png".format(h)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        plots.append({"path": str(path), "reason": "horizon t+{} predictions against truth".format(h)})

    return plots


def _summary_rows(result: dict) -> list:
    rows = []
    for entry in result["per_horizon"]:
        if entry["status"] != "ok":
            rows.append({"horizon": entry["horizon"], "status": entry["status"], "reason": entry.get("reason", "")})
            continue
        m = entry["automl"]
        rows.append({
            "horizon": entry["horizon"],
            "status": "ok",
            "n_test_origins": entry["n_test_origins"],
            "mase": m["mase"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "smape_pct": m["smape_pct"],
            "wape_pct": m["wape_pct"],
            "r2": m["r2"],
            "bias": m["bias"],
            "snaive_mase": entry["baselines"]["seasonal_naive"]["mase"],
            "skill": entry["skill_vs_seasonal_naive"],
            "estimator": entry["search"]["best_estimator"],
            "open": entry.get("open_days_only"),
            "open_skill": entry.get("open_days_skill_vs_seasonal_naive", float("nan")),
            "zero_share": entry.get("zero_share_of_test", 0.0),
        })
    return rows


def write_reports(result: dict, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    md = _markdown(result)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    html = _html(result)
    (run_dir / "report.html").write_text(html, encoding="utf-8")
    paths = {"report_md": str(run_dir / "report.md"), "report_html": str(run_dir / "report.html")}
    result.setdefault("artifact_paths", {}).update(paths)
    return paths


def _markdown(result: dict) -> str:
    task, proto, summary = result["task"], result["protocol"], result["summary"]
    rows = _summary_rows(result)
    lines = []
    lines.append("# AutoML baseline - {}".format(task["target"]))
    lines.append("")
    lines.append("**Goal:** {}".format(task["goal"]))
    lines.append("")
    lines.append("- Engine: {} | {} model(s) fitted | {}s search budget, optimising {}".format(
        result["run"]["baseline_system"], proto.get("models_fitted", "?"),
        result["config"]["time_budget"], result["config"]["metric"]))
    lines.append("- Strategy: {}{}".format(
        proto["strategy"],
        "; trained across horizons {}-{} ({} of them)".format(
            min(proto["horizons_trained"]), max(proto["horizons_trained"]),
            len(proto["horizons_trained"])) if proto.get("horizons_trained") else ""))
    lines.append("- Series: {} -> {} ({} periods at freq {}, {} missing)".format(
        task["series_start"], task["series_end"], task["n_periods"], task["freq"], task["missing_periods"]))
    lines.append("- Split: train through {} ({} periods), held out {} periods; {}".format(
        proto["cutoff"], proto["train_periods"], proto["test_periods"], proto["test_window_reason"]))
    lines.append("- Sources merged: {} file(s) -> {} rows x {} cols".format(
        len(result["ingest"]["files"]), result["ingest"]["rows"], result["ingest"]["columns"]))
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append("| mean MASE across horizons | {} |".format(_fmt(summary["mean_mase"])))
    lines.append("| mean WAPE across horizons | {}% |".format(_fmt(summary.get("mean_wape_pct"), 2)))
    lines.append("| mean SMAPE across horizons | {}% |".format(_fmt(summary["mean_smape_pct"], 2)))
    lines.append("| mean skill vs seasonal naive | {}% |".format(_fmt(summary["mean_skill_vs_seasonal_naive"] * 100, 1)))
    lines.append("| horizons beating seasonal naive | {} / {} |".format(
        summary["horizons_beating_seasonal_naive"], summary["horizons_evaluated"]))
    if summary.get("mean_skill_open_days") is not None:
        lines.append("| mean MASE, open days only | {} |".format(_fmt(summary.get("mean_mase_open_days"))))
        lines.append("| mean skill, open days only | {}% |".format(
            _fmt((summary.get("mean_skill_open_days") or 0) * 100, 1)))
    lines.append("")
    lines.append("## Per-horizon results")
    lines.append("")
    lines.append("| horizon | origins | MASE | MAE | RMSE | SMAPE % | WAPE % | R2 | bias | s-naive MASE | skill | winner |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        if r["status"] != "ok":
            lines.append("| t+{} | - | skipped: {} | | | | | | | | | |".format(r["horizon"], r.get("reason", "")))
            continue
        lines.append("| t+{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}% | {} |".format(
            r["horizon"], r["n_test_origins"], _fmt(r["mase"]), _fmt(r["mae"], 2), _fmt(r["rmse"], 2),
            _fmt(r["smape_pct"], 2), _fmt(r["wape_pct"], 2), _fmt(r["r2"]), _fmt(r["bias"], 2),
            _fmt(r["snaive_mase"]), _fmt(r["skill"] * 100, 1), r["estimator"]))
    lines.append("")
    dq = task.get("data_quality") or {}
    if dq.get("flags"):
        lines.append("## Data quality flags")
        lines.append("")
        for flag in dq["flags"]:
            lines.append("- {}".format(flag))
        lines.append("")
    if any(r.get("open") for r in rows if r["status"] == "ok"):
        lines.append("## Open days only (actual > 0)")
        lines.append("")
        lines.append("Closed days sit at exactly 0 and blow up every percentage metric. This is the "
                     "same evaluation restricted to days the shop actually traded.")
        lines.append("")
        lines.append("| horizon | rows | MASE | MAE | SMAPE % | WAPE % | skill |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            if r["status"] != "ok" or not r.get("open"):
                continue
            o = r["open"]
            lines.append("| t+{} | {} | {} | {} | {} | {} | {}% |".format(
                r["horizon"], o["n"], _fmt(o["mase"]), _fmt(o["mae"], 2), _fmt(o["smape_pct"], 2),
                _fmt(o["wape_pct"], 2), _fmt(r["open_skill"] * 100, 1)))
        lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- **MASE** is the scale-free accuracy number (Hyndman): MAE divided by the "
                 "in-sample one-step seasonal-naive MAE over the training window. It is comparable "
                 "across horizons and across models on this series, but its denominator comes from "
                 "the training period, so MASE < 1 is not by itself proof of beating naive *here*.")
    lines.append("- **skill** is the honest same-window comparison: `1 - MAE_model / MAE_seasonal_naive` "
                 "computed on the identical held-out rows. Positive skill = genuinely better than "
                 "repeating last week's value.")
    lines.append("- Each horizon has its own model and its own held-out origins, so the numbers "
                 "are independent - a rise in MASE with horizon is real degradation, not compounding.")
    lines.append("")
    lines.append("## Comparing the agentic pipeline against this")
    lines.append("")
    lines.append("Score the agentic model on the same cutoff ({}), the same origins, and the same "
                 "metric set. `predictions.csv` holds every (origin, target, truth, prediction) row, "
                 "so the agentic system's predictions can be joined on `(horizon, target_time)` and "
                 "diffed directly.".format(proto["cutoff"]))
    return "\n".join(lines)


def _embed(path: str) -> str:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ""
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _html(result: dict) -> str:
    task, proto, summary = result["task"], result["protocol"], result["summary"]
    rows = _summary_rows(result)
    plots = result.get("artifact_paths", {}).get("plots", [])

    def cells(r):
        if r["status"] != "ok":
            return "<tr><td>t+{}</td><td colspan='11' class='muted'>skipped - {}</td></tr>".format(
                r["horizon"], r.get("reason", ""))
        good = "good" if r["mase"] < 1 else "bad"
        return (
            "<tr><td class='h'>t+{h}</td><td>{n}</td><td class='{g}'>{mase}</td><td>{mae}</td>"
            "<td>{rmse}</td><td>{smape}</td><td>{wape}</td><td>{r2}</td><td>{bias}</td>"
            "<td class='muted'>{sn}</td><td class='{g}'>{skill}%</td><td class='muted'>{est}</td></tr>"
        ).format(
            h=r["horizon"], n=r["n_test_origins"], g=good, mase=_fmt(r["mase"]), mae=_fmt(r["mae"], 2),
            rmse=_fmt(r["rmse"], 2), smape=_fmt(r["smape_pct"], 2), wape=_fmt(r["wape_pct"], 2),
            r2=_fmt(r["r2"]), bias=_fmt(r["bias"], 2), sn=_fmt(r["snaive_mase"]),
            skill=_fmt(r["skill"] * 100, 1), est=r["estimator"])

    plot_html = "".join(
        "<figure><img src='{}' alt='{}'/><figcaption>{}</figcaption></figure>".format(
            _embed(p["path"]), Path(p["path"]).stem, p["reason"])
        for p in plots
    )

    features_html = ""
    for entry in result["per_horizon"]:
        if entry["status"] != "ok" or not entry.get("top_features"):
            continue
        items = "".join(
            "<li><code>{}</code><span>{:.1%}</span></li>".format(f["feature"], f["share"])
            for f in entry["top_features"][:8]
        )
        features_html += "<div class='fcard'><h4>t+{}</h4><ol>{}</ol></div>".format(entry["horizon"], items)

    dq = task.get("data_quality") or {}
    flags_html = ""
    if dq.get("flags"):
        flags_html = (
            "<section class='flags'><h2>Data quality flags</h2><ul>"
            + "".join("<li>{}</li>".format(f) for f in dq["flags"])
            + "</ul></section>"
        )

    open_rows = [r for r in rows if r["status"] == "ok" and r.get("open")]
    open_html = ""
    if open_rows:
        body = "".join(
            "<tr><td class='h'>t+{h}</td><td>{n}</td><td>{mase}</td><td>{mae}</td><td>{smape}</td>"
            "<td>{wape}</td><td class='{g}'>{skill}%</td></tr>".format(
                h=r["horizon"], n=r["open"]["n"], mase=_fmt(r["open"]["mase"]),
                mae=_fmt(r["open"]["mae"], 2), smape=_fmt(r["open"]["smape_pct"], 2),
                wape=_fmt(r["open"]["wape_pct"], 2), skill=_fmt(r["open_skill"] * 100, 1),
                g="good" if (r["open_skill"] or 0) > 0 else "bad")
            for r in open_rows
        )
        open_html = (
            "<section><h2>Open days only (actual &gt; 0)</h2><div class='scroll'><table>"
            "<thead><tr><th>horizon</th><th>rows</th><th>MASE</th><th>MAE</th><th>SMAPE %</th>"
            "<th>WAPE %</th><th>skill</th></tr></thead><tbody>" + body + "</tbody></table></div>"
            "<p class='note'>Closed days sit at exactly 0 and blow up every percentage metric. "
            "This is the same evaluation restricted to days the shop actually traded.</p></section>"
        )

    stat = lambda label, value, sub: (
        "<div class='stat'><span class='label'>{}</span><span class='value'>{}</span>"
        "<span class='sub'>{}</span></div>".format(label, value, sub))

    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AutoML baseline - {target}</title>
<style>
  :root {{ --bg:#f8fafc; --card:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
           --accent:#2563eb; --good:#16a34a; --bad:#dc2626; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
          font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1120px; margin:0 auto; padding:40px 24px 80px; }}
  header h1 {{ margin:0 0 4px; font-size:26px; letter-spacing:-0.02em; }}
  header p {{ margin:0; color:var(--muted); }}
  .badge {{ display:inline-block; margin-bottom:14px; padding:3px 10px; border-radius:999px;
            background:#eff6ff; color:var(--accent); font-size:12px; font-weight:600; }}
  section {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
             padding:22px 24px; margin-top:22px; }}
  h2 {{ font-size:15px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
        margin:0 0 16px; font-weight:700; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }}
  .stat {{ display:flex; flex-direction:column; gap:2px; padding:14px 16px; border:1px solid var(--line);
           border-radius:10px; background:#fbfdff; }}
  .stat .label {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }}
  .stat .value {{ font-size:26px; font-weight:650; letter-spacing:-0.02em; }}
  .stat .sub {{ font-size:12px; color:var(--muted); }}
  table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:14px; }}
  th, td {{ padding:9px 10px; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); font-weight:600; }}
  td.h {{ font-weight:650; }}
  .good {{ color:var(--good); font-weight:600; }}
  .bad {{ color:var(--bad); font-weight:600; }}
  .muted {{ color:var(--muted); }}
  .scroll {{ overflow-x:auto; }}
  figure {{ margin:0 0 22px; }}
  figure img {{ width:100%; border:1px solid var(--line); border-radius:10px; background:#fff; }}
  figcaption {{ font-size:12.5px; color:var(--muted); margin-top:6px; }}
  ul.meta {{ list-style:none; padding:0; margin:0; columns:2; font-size:13.5px; }}
  ul.meta li {{ margin-bottom:6px; break-inside:avoid; }}
  ul.meta b {{ color:var(--muted); font-weight:600; }}
  .fcards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; }}
  .fcard {{ border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  .fcard h4 {{ margin:0 0 8px; font-size:13px; }}
  .fcard ol {{ margin:0; padding-left:18px; font-size:12.5px; color:var(--muted); }}
  .fcard li {{ display:flex; justify-content:space-between; gap:8px; margin-bottom:3px; }}
  .fcard code {{ font-size:11.5px; color:var(--ink); }}
  .flags {{ border-color:#fed7aa; background:#fffbeb; }}
  .flags h2 {{ color:#b45309; }}
  .flags ul {{ margin:0; padding-left:18px; font-size:13.5px; color:#7c2d12; }}
  .flags li {{ margin-bottom:6px; }}
  .note {{ font-size:13.5px; color:var(--muted); }}
  .note code {{ background:#f1f5f9; padding:1px 5px; border-radius:4px; color:var(--ink); }}
</style></head><body><div class="wrap">
<header>
  <div class="badge">AutoML baseline &middot; {engine}</div>
  <h1>{target} &mdash; forecast accuracy at t+{horizons}</h1>
  <p>{goal}</p>
</header>

<section>
  <h2>Headline</h2>
  <div class="stats">{stats}</div>
</section>

<section>
  <h2>Per-horizon results</h2>
  <div class="scroll"><table>
    <thead><tr><th>horizon</th><th>origins</th><th>MASE</th><th>MAE</th><th>RMSE</th>
    <th>SMAPE %</th><th>WAPE %</th><th>R&sup2;</th><th>bias</th><th>s-naive MASE</th>
    <th>skill</th><th>winner</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <p class="note"><b>MASE</b> = MAE / in-sample seasonal-naive MAE on the training window (Hyndman) &mdash;
  scale-free and comparable across horizons. <b>skill</b> = <code>1 - MAE_model / MAE_seasonal_naive</code> on
  the identical held-out rows, which is the honest same-window verdict; positive means genuinely better than
  repeating last week's value. Each horizon has its own model and its own origins, so the numbers are independent.</p>
</section>

{flags_html}
{open_html}
<section>
  <h2>Evaluation protocol</h2>
  <ul class="meta">
    <li><b>Strategy:</b> {strategy}</li>
    <li><b>Models fitted:</b> {nmodels}</li>
    <li><b>Split:</b> {split} at {cutoff}</li>
    <li><b>Train / held-out periods:</b> {train_p} / {test_p}</li>
    <li><b>Search validation:</b> {val}</li>
    <li><b>Series:</b> {start} &rarr; {end} ({n} periods, freq {freq}, {missing} missing)</li>
    <li><b>Seasonal period:</b> {period}</li>
    <li><b>MASE scale:</b> {scale}</li>
    <li><b>Sources merged:</b> {nfiles} file(s) &rarr; {irows} rows &times; {icols} cols</li>
    <li><b>Search budget:</b> {budget}s per horizon, optimising {metric}</li>
    <li><b>Wall time:</b> {wall}s</li>
  </ul>
</section>

<section>
  <h2>What the winning models leaned on</h2>
  <div class="fcards">{features}</div>
</section>

<section>
  <h2>Plots</h2>
  {plots}
</section>

<section>
  <h2>Comparing the agentic pipeline against this</h2>
  <p class="note">Score the agentic model on the same cutoff (<code>{cutoff}</code>), the same origins and the
  same metrics. <code>predictions.csv</code> in the run directory holds every
  (horizon, origin_time, target_time, y_true, prediction) row, so the agentic system's output can be
  joined on <code>(horizon, target_time)</code> and diffed row by row.</p>
</section>
</div></body></html>""".format(
        target=task["target"],
        goal=task["goal"],
        engine=result["run"]["baseline_system"],
        horizons="/t+".join(str(h) for h in result["config"]["horizons"]),
        stats=(
            stat("mean MASE", _fmt(summary["mean_mase"]), "MAE / in-sample s-naive MAE")
            + stat("mean WAPE", _fmt(summary.get("mean_wape_pct"), 1) + "%",
                   "error as a share of total volume")
            + stat("mean skill", _fmt(summary["mean_skill_vs_seasonal_naive"] * 100, 1) + "%",
                   "MAE reduction vs seasonal naive")
            + stat("horizons won", "{} / {}".format(summary["horizons_beating_seasonal_naive"],
                                                    summary["horizons_evaluated"]),
                   "beat the naive floor")
            + stat("mean skill, open days", _fmt((summary.get("mean_skill_open_days") or 0) * 100, 1) + "%",
                   "excluding closed (zero) days")
        ),
        rows="".join(cells(r) for r in rows),
        flags_html=flags_html,
        open_html=open_html,
        strategy=proto["strategy"],
        nmodels=proto.get("models_fitted", "-"),
        split=proto["split"],
        cutoff=proto["cutoff"],
        train_p=proto["train_periods"],
        test_p=proto["test_periods"],
        val=proto["validation_inside_search"],
        start=task["series_start"], end=task["series_end"], n=task["n_periods"],
        freq=task["freq"], missing=task["missing_periods"], period=task["seasonal_period"],
        scale=_fmt(proto["mase_scale"], 2),
        nfiles=len(result["ingest"]["files"]), irows=result["ingest"]["rows"],
        icols=result["ingest"]["columns"],
        budget=result["config"]["time_budget"], metric=result["config"]["metric"],
        wall=result["run"]["wall_seconds"],
        features=features_html or "<p class='note'>winning estimator exposed no feature importances</p>",
        plots=plot_html or "<p class='note'>no plots generated</p>",
    )
