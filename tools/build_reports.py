"""
build_reports -- per-skill slide builders + driver for report_generator.

Implements the per-skill slide tables in skills/report_generator-skill/report_generator.md. Shared shell,
tokens, component CSS and deck engine come from tools/report_generator.py.

Write scope: reports/*.html and the single `report_html` field in each history JSON.

Usage:
    python tools/build_reports.py runs/run_<id>
    python tools/build_reports.py runs/run_<id> 001_eda 014_training
"""
import os
import sys
import json
import glob
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_generator import (E, _s, kpi, kpis, mini, details, table, conf_badge, plot_card,
                              clar, issues_block, slide, cover, agenda, page, ACCENT, LABEL,
                              WARN_BANNER, TOKENS, COMPONENT_CSS)


def plots_on_disk(run, doc):
    out = {}
    for p in (doc.get("artifact_paths", {}) or {}).get("plots", []) or []:
        path = p.get("path")
        if not path:
            continue
        slug = p.get("name") or os.path.basename(path).rsplit(".", 1)[0]
        out[slug] = {"rel": "../" + path,
                     "exists": os.path.exists(os.path.join(run, path)),
                     "reason": p.get("reason", ""),
                     "insight": p.get("insight") or p.get("what_it_shows") or p.get("reason", ""),
                     "conf": p.get("confidence"), "type": p.get("type", "")}
    return out


def pc(P, slug, cap_len=130):
    d = P.get(slug)
    if not d:
        return ""
    return plot_card(d["rel"], d["exists"], slug, _s(d["reason"], cap_len), d["insight"])


# ============================================================ EDA
def build_eda(run, doc):
    ov, fe, tg = doc["dataset_overview"], doc["feasibility"], doc["target"]
    met, P = doc["recommended_metric"], plots_on_disk(run, doc)
    T = 10
    S = [cover("eda", "001", doc["run_id"],
               "Exploratory Data Analysis — Daily Bread Demand, Aernoudt Branch 01",
               "One row per calendar date assembled from six raw sources for point-of-sale 01, "
               "%s to %s. Every number in this deck traces to history/iteration_001_eda.json."
               % (ov["date_min"], ov["date_max"]),
               [(format(ov["n_rows"], ","), "Rows assembled"), (ov["n_columns"], "Columns"),
                ("%.1f%%" % ov["overall_missing_pct"], "Overall missing"),
                ("%.2f" % ov["span_years"], "Years covered"), ("MASE", "Primary metric")]),
         agenda([(3, "Overview & Feasibility", "Trainability verdict, schema, time coverage"),
                 (4, "Key Findings", "%d findings from this iteration" % len(doc["key_findings"])),
                 (5, "Target Analysis", "Distribution and the recommended metric"),
                 (6, "Features", "Top features, leakage flags"),
                 (7, "Missingness & Flags", "Per-column mechanisms"),
                 (8, "Target & Trend", "The dataset's central story"),
                 (9, "Seasonality & Structure", "Weekly, annual and horizon structure"),
                 (10, "Clarifications", "Open questions, strategy, limitations")], T)]
    chips = "".join('<div class="chip">%s</div>' % E(c) for c in [
        "span %d days" % ov["span_days"], "duplicate rows %d" % ov["duplicate_rows"],
        "duplicate dates %d" % ov["duplicate_dates"], "memory %.2f MB" % ov["memory_mb"],
        "observed target days %s" % format(tg["n_observations"], ",")])
    S.append(slide(3, T, "Overview",
                   '<span class="pill pill-ok">&#10003; Trainable</span>'
                   "<h2>All seven hard-blocker checks pass</h2><p>%s</p>%s%s"
                   '<div class="grid-2" style="margin-top:22px">'
                   "<div class=\"card\"><h3>Schema</h3>%s</div>"
                   "<div class=\"card\"><h3>Time coverage</h3>"
                   '<div class="mono" style="font-size:1.15rem;color:var(--accent);margin-bottom:12px">'
                   "%s &rarr; %s</div><div class=\"chips\">%s</div>"
                   '<div class="callout-pu" style="font-size:.82rem">%s</div></div></div>'
                   % (E(_s(fe["reasoning"], 340)), details("Full feasibility reasoning", fe["reasoning"]),
                      kpis([(format(ov["n_rows"], ","), "Rows"), (ov["n_columns"], "Columns"),
                            ("%.1f%%" % ov["overall_missing_pct"], "Missing"), ("MASE", "Metric")]),
                      table(["Type", "Count"], [["numeric", ov["n_numeric"]],
                                                ["categorical", ov["n_categorical"]],
                                                ["total", ov["n_columns"]]]),
                      ov["date_min"], ov["date_max"], chips, E(_s(ov.get("assembly_note"), 300)))))
    S.append(slide(4, T, "Key Findings", "<h2>Key findings</h2><div class=\"grid-fill\">%s</div>"
                   % "".join('<div class="finding-card">%s%s</div>'
                             % (E(_s(k, 260)),
                                details("Full finding", k) if len(str(k)) > 260 else "")
                             for k in doc["key_findings"])))
    stat = table(["Statistic", "Value"],
                 [[k, ("%.3f" % tg[k]) if isinstance(tg.get(k), float) else tg.get(k)]
                  for k in ["mean", "median", "std", "min", "max", "skewness", "kurtosis"]])
    S.append(slide(5, T, "Target Analysis",
                   "<h2>Target: bread_quantity</h2><div class=\"grid-lead\">"
                   '<div class="card"><h3>Distribution</h3><span class="badge b-reg">%s</span>%s'
                   '<p style="font-size:.82rem;color:var(--e61-ink-3);margin-top:10px">%s observed '
                   "days; %s below 10 units (%.1f%%) — the closure cluster Preprocess later "
                   "filtered.</p></div><div><div class=\"card\"><h3>Recommended metric</h3>"
                   '<div class="hero-stat-val" style="font-size:1.3rem">MASE</div>'
                   '<p style="font-size:.82rem;margin-top:8px">%s</p>%s'
                   '<div class="chips" style="margin-top:10px">%s</div></div></div></div>'
                   % (E(tg["task_type"]), stat, format(tg["n_observations"], ","),
                      format(tg.get("n_below_10", 0), ","), tg.get("pct_below_10", 0),
                      E(_s(met["reasoning"], 240)), details("Full metric reasoning", met["reasoning"]),
                      "".join('<div class="chip">%s</div>' % E(m) for m in met["secondary"]))))
    rows = [['<span class="mono">%s</span>' % E(f["name"]),
             ("%.4f" % f["target_corr"]) if f.get("target_corr") is not None else "—",
             ("%.4f" % f["target_mi"]) if f.get("target_mi") is not None else "—",
             '<span class="badge b-lk">leakage</span>' if f.get("is_leakage_flagged") else "",
             conf_badge(f.get("confidence"))] for f in doc["top_features"][:12]]
    lk = doc["flags"].get("possible_leakage", []) or []
    S.append(slide(6, T, "Features",
                   "<h2>Top features and leakage</h2>%s"
                   '<div class="callout-rd" style="margin-top:18px"><b>Possible leakage (%d '
                   "columns)</b><div class=\"chips\" style=\"margin:8px 0 0\">%s</div></div>"
                   % (table(["Feature", "Spearman", "MI", "Flag", "Confidence"], rows), len(lk),
                      "".join('<div class="chip">%s</div>'
                              % E(x if isinstance(x, str) else x.get("name", str(x)))
                              for x in lk[:16]))))
    cols = sorted(doc["missingness"]["columns"], key=lambda c: -c["missing_pct"])[:8]
    cards = []
    for c in cols:
        mech = (c.get("mechanism") or "").lower()
        b = "b-mnar" if "mnar" in mech else ("b-mar" if "mar" in mech else "b-mcar")
        cards.append('<div class="card"><span class="mono">%s</span> <span class="badge %s">%s</span>'
                     '<div class="miss-bar-wrap"><div class="miss-bar-fill" style="width:%.1f%%">'
                     "</div></div><b>%.2f%%</b> missing (%s rows)</div>"
                     % (E(c["name"]), b, E(_s(c.get("mechanism"), 42)), c["missing_pct"],
                        c["missing_pct"], format(c["n_missing"], ",")))
    S.append(slide(7, T, "Missingness & Flags",
                   "<h2>Missingness by column</h2>"
                   '<p style="font-size:.86rem;color:var(--e61-ink-3)">%s</p>'
                   '<div class="grid-fill" style="max-height:50vh;overflow-y:auto">%s</div>'
                   '<p style="font-size:.78rem;color:var(--e61-ink-4);margin-top:12px">Showing the '
                   "8 highest of %d columns carrying missing values.</p>"
                   % (E(_s(doc["missingness"].get("note"), 300)), "".join(cards),
                      len(doc["missingness"]["columns"]))))
    a = P.get("target_over_time")
    S.append(slide(8, T, "Target & Trend",
                   "<h2>The series and its distribution</h2><div class=\"grid-2\">%s%s</div>"
                   '<p style="margin-top:16px;font-size:.9rem">%s</p>%s'
                   % (pc(P, "target_over_time"), pc(P, "target_histogram"),
                      E(_s(a["insight"], 420)) if a else "",
                      details("Full analysis: target_over_time", a["insight"]) if a else ""),
                   cls="dark"))
    S.append(slide(9, T, "Seasonality & Structure",
                   "<h2>Weekly, annual and horizon structure</h2><div class=\"grid-2\">%s</div>"
                   % "".join(pc(P, n) for n in
                             ["weekday_profile", "acf_pacf", "horizon_decay", "yearly_level"])))
    strat = doc["recommended_strategy"]
    S.append(slide(10, T, "Clarifications",
                   '<div class="grid-2"><div><h2>Open questions</h2>%s</div>'
                   "<div><h2>Strategy</h2><div class=\"card\">"
                   '<p style="font-size:.88rem">%s</p>%s<div style="margin-top:10px">%s '
                   '<span class="badge b-nc">%s</span></div></div>'
                   '<h3 style="margin-top:20px">Limitations</h3><ul class="limits-list">%s</ul>'
                   "</div></div>"
                   % ("".join(clar(c) for c in doc.get("clarification_needed", [])),
                      E(_s(strat.get("text"), 320)),
                      details("Full strategy narrative", strat.get("text")),
                      conf_badge(strat.get("confidence")),
                      E(strat.get("feature_selection_posture", "")),
                      "".join("<li>%s</li>" % E(l) for l in doc.get("limitations", [])))))
    return S


# ============================================================ Domain Research
def build_domain_research(run, doc):
    meta = doc.get("meta", {}) or {}
    sf = doc.get("structured_findings", {}) or {}
    T = 7
    S = [cover("domain_research", "002", "run_20260827T154407Z",
               "Domain Research — Catchment, Demand Drivers and Data Sources",
               "External research answering the triggers EDA raised for Aernoudt branch 01. "
               "Every number traces to history/iteration_002_domain_research.json.",
               [(meta.get("triggers_addressed", len(doc.get("eda_trigger_responses", []))),
                 "Triggers addressed"),
                (len(doc.get("feature_engineering_recommendations", [])), "Feature recommendations"),
                (len(doc.get("data_source_assessment", {}) or {}), "Data sources assessed"),
                (len(doc.get("open_questions", [])), "Open questions")]),
         agenda([(3, "Identity & Triggers", "The catchment, and EDA's questions answered"),
                 (4, "Demand Drivers", "Ranked drivers and structural shocks"),
                 (5, "Data Sources", "Per-source verdicts"),
                 (6, "Recommendations", "Features recommended downstream"),
                 (7, "Open Questions", "What research could not settle")], T)]
    tr = doc.get("eda_trigger_responses", []) or []
    S.append(slide(3, T, "identity",
                   "<h2>Catchment identity and trigger responses</h2>"
                   '<div class="callout-pu" style="margin-bottom:18px"><b>Catchment:</b> '
                   "institutional — UZ Gent hospital and Ghent University campus, not "
                   "residential. This single fact independently explains several EDA findings."
                   "</div><div class=\"grid-fill\">%s</div>"
                   % "".join('<div class="card"><b>%s</b>'
                             '<p style="font-size:.85rem;margin:.5em 0">%s</p>%s'
                             '<div style="margin-top:8px">%s</div></div>'
                             % (E(_s(t.get("trigger"), 90)), E(_s(t.get("finding"), 250)),
                                details("Full finding", t.get("finding")
                                        if len(str(t.get("finding", ""))) > 250 else None),
                                conf_badge(t.get("confidence"))) for t in tr)))
    dd = (sf.get("demand_drivers", {}) or {}).get("identified_demand_drivers_ranked", []) or []
    rows = [[E(_s(d.get("driver"), 60)), E(d.get("type", "")),
             '<span class="badge %s">%s</span>'
             % ("b-hi" if str(d.get("strength", "")).startswith(("very", "high"))
                else ("b-med" if d.get("strength") == "medium" else "b-nc"),
                E(d.get("strength", ""))),
             E(_s(d.get("evidence"), 170))] for d in dd]
    S.append(slide(4, T, "drivers", "<h2>Ranked demand drivers</h2>"
                   + (table(["Driver", "Type", "Strength", "Evidence"], rows) if rows else
                      '<div class="callout-am">No ranked drivers recorded in the history JSON.</div>')))
    cards = []
    for k, v in (doc.get("data_source_assessment", {}) or {}).items():
        vd = (v.get("verdict") or "") if isinstance(v, dict) else ""
        u = vd.upper()
        b = ("b-hi" if "USEFUL" in u and "CONDITION" not in u else
             "b-amb" if "CONDITION" in u else "b-lk" if "DROP" in u else "b-med")
        cards.append('<div class="card"><b class="mono">%s</b> <span class="badge %s">%s</span>'
                     '<p style="font-size:.82rem;margin:.6em 0">%s</p>%s</div>'
                     % (E(k), b, E(_s(vd, 40)),
                        E(_s(v.get("domain_context") if isinstance(v, dict) else v, 230)),
                        ('<div class="chip">%s</div>' % E(v["availability_window"]))
                        if isinstance(v, dict) and v.get("availability_window") else ""))
    S.append(slide(5, T, "datasources", "<h2>Data source assessment</h2>"
                   '<div class="grid-fill">%s</div>' % "".join(cards)))
    fr = doc.get("feature_engineering_recommendations", []) or []
    S.append(slide(6, T, "recommendations",
                   "<h2>Features recommended downstream</h2><div class=\"grid-fill\">%s</div>"
                   % "".join('<div class="finding-card"><b class="mono">%s</b> '
                             '<span class="badge %s">%s</span>'
                             '<p style="font-size:.84rem;margin:.5em 0">%s</p>%s</div>'
                             % (E(r.get("feature", r.get("name", ""))),
                                "b-hi" if r.get("priority") == "high"
                                else ("b-med" if r.get("priority") == "medium" else "b-nc"),
                                E(r.get("priority", "")),
                                E(_s(r.get("description") or r.get("rationale"), 200)),
                                details("Full rationale", r.get("rationale")))
                             for r in fr), cls="warm"))
    S.append(slide(7, T, "questions", "<h2>What research could not settle</h2>"
                   + "".join('<div class="clar-card non-blocking">'
                             '<span class="badge b-nb">%s</span>'
                             '<h4 style="margin-top:10px">%s</h4>'
                             '<p style="font-size:.85rem;color:var(--e61-ink-3)">%s</p></div>'
                             % (E(q.get("priority", "open")), E(_s(q.get("question"), 220)),
                                E(_s(q.get("reason"), 300)))
                             for q in (doc.get("open_questions", []) or []))))
    return S
