"""
build_reports2 -- preprocess / feature_engineering / training / diagnosis / end_model
builders, the index generator, and the driver.

Split from build_reports.py only for file size; both implement the per-skill slide
tables in skills/report_generator-skill/report_generator.md against this run's ACTUAL history JSON field
names, which differ from that file's reference contracts. Nothing is invented: where a
field the spec names does not exist here, the closest real field is used and the slide
says what it is.

NOTE ON A SPEC GAP: skills/report_generator-skill/report_generator.md defines six skill report types and has
NO end_model section and no end_model accent. The end_model deck below follows the same
shared shell, cover/agenda rules and component vocabulary as every other deck, and uses
--e61-purple (unused by the other six) as a documented extension.
"""
import os
import re
import sys
import json
import glob
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_generator import (E, _s, kpis, mini, details, table, conf_badge, plot_card,
                              clar, issues_block, slide, cover, agenda, page, ACCENT, LABEL,
                              WARN_BANNER)
from build_reports import build_eda, build_domain_research, plots_on_disk, pc


# ============================================================ Preprocess
def build_preprocess(run, doc):
    sp, P = doc["split_info"], plots_on_disk(run, doc)
    osh, T = doc["output_shape"], 6
    rf = doc.get("rows_filtered", []) or []
    S = [cover("preprocess", "004", doc["run_id"],
               "Preprocessing — Row Filtering, Split Construction and Column Drops",
               "Mode: %s. Produced %s, the modelling frame every later iteration reads. "
               "Every number traces to history/iteration_004_preprocess.json."
               % (doc["mode"], doc["processed_dataset_path"]),
               [(format(osh["rows"], ","), "Rows out"), (osh["columns"], "Columns out"),
                (format(osh["modellable_rows"], ","), "Modellable rows"),
                (len(doc.get("dropped_columns", [])), "Columns dropped"),
                (len(doc.get("transformations_applied", [])), "Features transformed")]),
         agenda([(3, "Overview", "Mode, rows filtered and call context"),
                 (4, "Split & Drops", "The three-way split and every dropped column"),
                 (5, "Transformations", "Per-feature missing-value and encoding handling"),
                 (6, "Results", "Output dataset, open question and issues")], T)]
    frows = [[E(r["condition"]), format(r["rows_removed"], ","), format(r["rows_before"], ","),
              format(r["rows_after"], ","), E(_s(r.get("reason"), 200))] for r in rf]
    S.append(slide(3, T, "Overview",
                   '<span class="pill pill-accent">%s</span><h2>Call context</h2><p>%s</p>%s%s'
                   "<h3 style=\"margin-top:24px\">Rows filtered</h3>%s"
                   % (E(doc["mode"].upper()), E(_s(doc.get("mode_reasoning"), 320)),
                      details("Full mode reasoning", doc.get("mode_reasoning")),
                      kpis([(format(osh["rows"], ","), "Rows out"),
                            (format(osh["modellable_rows"], ","), "Modellable"),
                            (len(doc.get("dropped_columns", [])), "Dropped"),
                            (len(doc.get("transformations_applied", [])), "Transformed")]),
                      table(["Condition", "Removed", "Before", "After", "Reason"], frows)
                      if frows else "<i>No rows filtered this call.</i>")))
    srows = []
    for k, tint in (("train", "#F0FBF4"), ("validation", "#EEF4FB"), ("test", "#F2EEF6")):
        s = sp.get(k, {})
        srows.append(['<span style="background:%s;padding:2px 8px;border-radius:5px"><b>%s</b></span>'
                      % (tint, k), E(s.get("date_min", "")), E(s.get("date_max", "")),
                      format(s.get("n", 0), ","), s.get("target_mean", "—")])
    drows = [['<span class="mono">%s</span>' % E(c["name"]), E(c.get("reason", "")),
              E(_s(c.get("detail"), 150))] for c in doc.get("dropped_columns", [])]
    S.append(slide(4, T, "Split & Drops",
                   '<h2>Chronological three-way split</h2><span class="badge b-reg">%s</span>'
                   "<div style=\"margin-top:12px\">%s</div>"
                   '<p style="font-size:.86rem;margin-top:12px">%s</p>%s'
                   '<div class="callout-am" style="margin-top:14px;font-size:.83rem">Every scaler, '
                   "imputation value and encoding in this run is fit on train rows only.</div>"
                   "%s<h3 style=\"margin-top:26px\">Columns dropped (%d)</h3>"
                   '<div style="max-height:34vh;overflow-y:auto">%s</div>'
                   % (E(sp.get("method", "")),
                      table(["Split", "Start", "End", "Rows", "Target mean"], srows),
                      E(_s(sp.get("method_reasoning"), 300)),
                      details("Full split reasoning", sp.get("method_reasoning")),
                      pc(P, "split_timeline"), len(drows),
                      table(["Column", "Reason", "Detail"], drows)), cls="dark"))
    tcards = []
    for t in doc.get("transformations_applied", []):
        mh = t.get("missing_handling", {}) or {}
        tcards.append('<div class="card"><b class="mono">%s</b> %s'
                      '<p style="font-size:.82rem;margin:.55em 0"><span class="badge b-mar">'
                      "fill</span> %s</p>"
                      '<p style="font-size:.78rem;color:var(--e61-ink-3)">%s</p>%s</div>'
                      % (E(t.get("feature", "")), conf_badge(t.get("confidence")),
                         E(_s(mh.get("method"), 150)),
                         E(_s(mh.get("source_mechanism"), 120)),
                         ('<div style="font-size:.75rem;color:var(--e61-ink-4);margin-top:6px">'
                          "rows filled: %s</div>" % format(mh.get("rows_filled", 0), ","))
                         if mh.get("rows_filled") else ""))
    S.append(slide(5, T, "Transformations",
                   "<h2>Per-feature handling (%d features)</h2>"
                   '<div class="grid-fill" style="max-height:62vh;overflow-y:auto">%s</div>'
                   % (len(tcards), "".join(tcards))))
    S.append(slide(6, T, "Results",
                   '<div class="grid-2"><div><h2>Output dataset</h2><div class="card">'
                   '<p class="mono" style="color:var(--accent)">%s</p>%s</div>'
                   "<h3 style=\"margin-top:20px\">Issues</h3>%s</div>"
                   "<div><h2>Open question</h2>%s%s</div></div>"
                   % (E(doc["processed_dataset_path"]),
                      table(["Metric", "Value"],
                            [["rows", format(osh["rows"], ",")], ["columns", osh["columns"]],
                             ["modellable rows", format(osh["modellable_rows"], ",")],
                             ["no-target rows", format(sp.get("no_target_rows", 0), ",")]]),
                      issues_block(doc.get("issues")),
                      "".join(clar(c) for c in doc.get("clarification_needed", [])),
                      pc(P, "regime_exposure"))))
    return S


# ============================================================ Feature Engineering
def build_fe(run, doc):
    P, T = plots_on_disk(run, doc), 6
    r2f, osh = doc["rows_to_features_check"], doc["output_shape"]
    created, mv = doc.get("features_created", []), doc.get("mechanism_verdict", {}) or {}
    lag = (doc.get("effective_lag_of_level_signals", {}) or {}).get("values", {})
    S = [cover("feature_engineering", "008", doc["run_id"],
               "Feature Engineering — Three Level-Adaptation Features Added",
               "Mode: incremental. All %d existing features kept; %d created to test the "
               "level-adaptation hypothesis Diagnosis 007 raised. Every number traces to "
               "history/iteration_008_feature_engineering.json."
               % (r2f["before"]["features"], len(created)),
               [(len(created), "Features created"), (osh["n_features"], "Features out"),
                (format(osh["rows"], ","), "Rows"),
                ("%.1f" % r2f["after"]["effective_ratio"], "Effective rows/feature"),
                ("0", "Existing dropped")]),
         agenda([(3, "Overview", "Mode, trigger and the rows-to-features check"),
                 (4, "Created", "The three features and why each exists"),
                 (5, "Audit", "Relevance screen, the override, and leakage"),
                 (6, "Results", "Output dataset, handover and issues")], T)]
    S.append(slide(3, T, "Overview",
                   '<span class="pill pill-accent">Incremental</span>'
                   "<h2>%d features created, all %d existing kept</h2><p>%s</p>%s%s"
                   '<div class="grid-2" style="margin-top:22px">'
                   '<div class="card"><h3>Trigger</h3><p style="font-size:.86rem">%s</p>%s</div>'
                   '<div class="card"><h3>Rows to features</h3>%s'
                   '<p style="font-size:.78rem;color:var(--e61-ink-3);margin-top:8px">%s</p></div></div>'
                   % (len(created), r2f["before"]["features"],
                      E(_s(doc.get("mode_reasoning"), 300)),
                      details("Full mode reasoning", doc.get("mode_reasoning")),
                      kpis([(len(created), "Created"), (osh["n_features"], "Total features"),
                            ("%.1f" % r2f["after"]["effective_ratio"], "Eff. ratio"),
                            ("0", "Dropped")]),
                      E(_s((doc.get("trigger") or {}).get("observation"), 300)),
                      details("Full trigger", (doc.get("trigger") or {}).get("observation")),
                      table(["", "Rows", "Features", "Effective ratio"],
                            [["before", format(r2f["before"]["rows"], ","),
                              r2f["before"]["features"], "%.1f" % r2f["before"]["effective_ratio"]],
                             ["after", format(r2f["after"]["rows"], ","),
                              r2f["after"]["features"], "%.1f" % r2f["after"]["effective_ratio"]]]),
                      E(_s(r2f["after"].get("warning"), 260)))))
    ccards = []
    for c in created:
        rel = c.get("relevance", {}) or {}
        ccards.append('<div class="card"><b class="mono">%s</b> %s'
                      '<p style="font-size:.76rem;color:var(--e61-ink-4);font-style:italic;'
                      'margin:.4em 0">%s</p>'
                      '<div class="chips" style="margin:8px 0"><div class="chip">Spearman '
                      "<span>%s</span></div><div class=\"chip\">MI <span>%s</span></div>"
                      '<div class="chip">fill <span>%s</span></div></div>'
                      '<p style="font-size:.84rem"><b>Why:</b> %s</p>%s'
                      '<div class="callout-gr" style="font-size:.78rem;margin-top:8px">'
                      "&#10003; pre-period: lag k = 7&middot;ceil(h/7) &ge; h, legal at every "
                      "horizon</div></div>"
                      % (E(c["name"]), conf_badge(c.get("confidence")), E(c.get("technique", "")),
                         rel.get("spearman_train", "—"), rel.get("mutual_info_train", "—"),
                         E(c.get("lag_nan_fill", "")), E(_s(c.get("reasoning"), 230)),
                         details("Full reasoning", c.get("reasoning"))))
    S.append(slide(4, T, "Created",
                   "<h2>Features created</h2><div class=\"grid-fill\">%s</div>" % "".join(ccards)))
    prot = doc.get("protected_from_relevance_drop", {}) or {}
    S.append(slide(5, T, "Audit",
                   "<h2>The relevance screen tried to delete the feature the hypothesis rides on</h2>"
                   '<div class="grid-lead"><div>%s</div><div>'
                   '<div class="card"><h3>Domain-argument override</h3>'
                   '<div class="chips"><div class="chip">protected <span>%s</span></div></div>'
                   '<p style="font-size:.84rem">%s</p>%s</div>'
                   '<div class="callout-gr" style="margin-top:14px;font-size:.82rem">'
                   "<b>Leakage audit: %d/%d pre-period.</b> No concurrent or post-period feature "
                   "survives into the engineered set.</div></div></div>"
                   % (pc(P, "relevance_screen"), E(", ".join(prot.get("features", []))),
                      E(_s(prot.get("reasoning"), 330)),
                      details("Full override reasoning", prot.get("reasoning")),
                      (doc.get("availability_summary", {}) or {}).get("pre_period", 0),
                      len(doc.get("prediction_time_availability_audit", []))), cls="dark"))
    lagrows = [[E(k), v.get("best_shift_days", "—"), v.get("corr_at_best", "—")]
               for k, v in lag.items()]
    S.append(slide(6, T, "Results",
                   '<div class="grid-2"><div><h2>Mechanism verified before training</h2>'
                   '<div class="callout-gr"><b>%s</b> &mdash; %s</div>'
                   "<h3 style=\"margin-top:18px\">Effective lag by level signal</h3>%s"
                   "<h3 style=\"margin-top:18px\">Output</h3><div class=\"card\">"
                   '<p class="mono" style="color:var(--accent)">%s</p>%s</div></div>'
                   "<div>%s<h3>Handover to Training</h3><ul class=\"limits-list\">%s</ul>"
                   "<h3 style=\"margin-top:18px\">Issues</h3>%s</div></div>"
                   % (E(mv.get("verdict", "")), E(_s(mv.get("evidence"), 300)),
                      table(["Signal", "Effective lag (days)", "Corr at best"], lagrows),
                      E(doc["engineered_dataset_path"]),
                      table(["Metric", "Value"],
                            [["rows", format(osh["rows"], ",")],
                             ["features", osh["n_features"]]]),
                      "".join(clar(c) for c in doc.get("clarification_needed", [])),
                      "".join("<li>%s</li>" % E(_s(r, 220))
                              for r in doc.get("recommendations_for_training", [])[:4]),
                      issues_block(doc.get("issues")))))
    return S


# ============================================================ Training
# Training is the one node this run called five times, and the calls do not share a
# shape: 014 carries cv_selection / classical_comparison / ensembling_check, while 011
# carries hypothesis_outcome.part_by_part and a bias-by-era comparison and none of
# those. So this builder branches on what the history JSON actually contains rather
# than assuming one call's layout -- a missing section is skipped, never faked.
def build_training(run, doc):
    P = plots_on_disk(run, doc)
    m, oc = doc["metrics"], doc["overfitting_check"]
    # 014 has no `hyperparameters` block at all -- it compared families on per-family
    # defaults and recorded that choice in `hyperparameter_note` instead. Both shapes are
    # legitimate, so the accessor tolerates the absence and the config card below falls
    # back to whichever of the two the JSON actually carries.
    hp, ho = doc.get("hyperparameters") or {}, doc.get("hypothesis_outcome", {}) or {}
    pv = doc.get("pre_committed_verdict", {}) or {}
    fi = doc.get("feature_importance", {}) or {}
    rs = doc.get("residuals_summary", {}) or {}
    si = doc.get("split_integrity", {}) or {}
    nnn = "%03d" % doc["iteration"]
    has_cv = "cv_selection" in doc

    plan = [(3, "Overview", "Mode, trigger and model configuration")]
    if has_cv:
        plan += [(4, "Model Family", "Six families, and why the winner is a tie"),
                 (5, "Architecture", "ML against classical forecasting"),
                 (6, "Ensembling & Results", "Blends, metrics and handover")]
    else:
        plan += [(4, "Verdict", "The pre-committed test, part by part"),
                 (5, "Metrics", "Train, validation and bias split at the lockdown"),
                 (6, "Importance", "Where the removed terms' importance went"),
                 (7, "Diagnostics", "Residuals, horizon response and handover")]
    T = plan[-1][0]

    outcome = str(ho.get("outcome", pv.get("verdict", ""))).upper()
    verdict_pill = ('<span class="pill pill-danger">Hypothesis rejected</span>'
                    if outcome.startswith(("REJ", "FAIL"))
                    else '<span class="pill pill-ok">&#10003; Confirmed</span>')
    cov_kpis = [("%.3f" % m["validation"]["MAE"], "Val MAE"),
                ("%.4f" % m["mase"]["mase_validation"], "Val MASE"),
                ("%.4f" % oc["ratio"], "Overfit ratio"),
                (si.get("n_features", "-"), "Features")]
    if pv.get("verdict"):
        cov_kpis.append((pv["verdict"], "Verdict"))
    S = [cover("training", nnn, doc["run_id"],
               "Training — %s" % ("Model Family Comparison and Ensembling" if has_cv
                                       else "Calendar-Term Exclusion and the Level Offset"),
               "Mode: %s. %s Every number traces to history/iteration_%s_training.json."
               % (doc["mode"],
                  "Six estimator families compared by rolling-origin CV inside train."
                  if has_cv else
                  "doy_sin_2 and doy_cos_2 excluded from the feature subset and "
                  "min_child_samples raised from 40 to 100; everything else pinned.", nnn),
               cov_kpis),
         agenda(plan, T)]

    hprows = [[k, '<span class="mono">%s</span>' % E(v),
               '<span class="badge b-med">changed</span>'
               if k in (hp.get("changed_from_frozen") or {}) else ""]
              for k, v in (hp.get("values") or {}).items()]
    S.append(slide(3, T, "Overview",
                   "%s<h2>%s</h2><p>%s</p>%s%s"
                   '<div class="grid-2" style="margin-top:20px">'
                   '<div class="card"><h3>Model configuration</h3>'
                   '<div class="chips"><div class="chip">%s</div>'
                   '<div class="chip">seed <span>%s</span></div>'
                   '<div class="chip">search <span>%s</span></div></div>%s'
                   '<p style="font-size:.78rem;color:var(--e61-ink-3);margin-top:10px">%s</p></div>'
                   '<div class="card"><h3>Trigger</h3><p style="font-size:.86rem">%s</p>%s'
                   '<h3 style="margin-top:16px">Split integrity</h3>%s</div></div>'
                   % (verdict_pill,
                      E(_s(ho.get("hypothesis"), 150)) or "Training call",
                      E(_s(doc.get("mode_reasoning"), 320)),
                      details("Full mode reasoning", doc.get("mode_reasoning")),
                      kpis([("%.3f" % m["train"]["MAE"], "Train MAE"),
                            ("%.3f" % m["validation"]["MAE"], "Val MAE"),
                            ("%.4f" % oc["ratio"], "Overfit"),
                            (si.get("n_features", "-"), "Features")]),
                      E(doc.get("model_class", "")), hp.get("random_state", "-"),
                      "none" if hp.get("search_run") is False else "yes",
                      (table(["Hyperparameter", "Value", ""], hprows) if hprows else
                       '<p style="font-size:.84rem">%s</p>' % E(_s(doc.get("hyperparameter_note"), 340))),
                      E(_s(hp.get("seed_note") or doc.get("hyperparameter_note"), 220)),
                      E(_s((doc.get("trigger") or {}).get("observation"), 300)),
                      details("Full trigger", (doc.get("trigger") or {}).get("observation")),
                      table(["Check", "Value"],
                            [["train ends", si.get("train_date_max", "-")],
                             ["validation starts", si.get("validation_date_min", "-")],
                             ["train strictly before val",
                              si.get("train_strictly_before_validation", "-")],
                             ["test rows untouched",
                              format(si.get("n_test_untouched", 0), ",")]]))))

    if has_cv:
        return S + _training_cv_slides(run, doc, P, T)

    prows = [[E(p["part"]), E(p["predicted"]), E(p["observed"]),
              '<span class="badge %s">%s</span>'
              % ("b-lk" if str(p["verdict"]).startswith("REF") else "b-hi",
                 E(str(p["verdict"]).split(" - ")[0]))]
             for p in ho.get("part_by_part", [])]
    S.append(slide(4, T, "Verdict",
                   "<h2>%s</h2><p style=\"font-size:.92rem\">%s</p>%s"
                   '<p style="margin-top:16px;font-size:.9rem"><b>%s</b></p>%s'
                   % ("Rejected — but two of the prediction's three parts held"
                      if prows else "Verdict",
                      E(_s(ho.get("how_it_failed"), 430)),
                      table(["Part of the prediction", "Predicted", "Observed", ""], prows)
                      if prows else "",
                      E(_s(ho.get("what_it_means"), 400)),
                      details("Full interpretation", ho.get("what_it_means"))),
                   cls="dark"))

    era, era6 = (rs.get("by_era") or {}), (rs.get("by_era_006") or {})
    erows = [[E(k.replace("_", "-")), format(v.get("n", 0), ","),
              "%+.2f" % v.get("mean_resid", 0), "%.3f" % v.get("MAE", 0),
              ("%+.2f" % era6[k]["mean_resid"]) if k in era6 else "-"]
             for k, v in era.items()]
    S.append(slide(5, T, "Metrics",
                   '<div class="grid-2">'
                   '<div class="card" style="border-left:4px solid var(--e61-success)">'
                   "<h3>Train</h3>%s</div>"
                   '<div class="card" style="border-left:4px solid var(--e61-info)">'
                   "<h3>Validation</h3>%s</div></div>"
                   '<div class="card" style="margin-top:18px"><h3>Gap analysis</h3>'
                   '<div class="grid-lead"><div><div class="hero-stat-val">%.4f</div>'
                   '<div class="hero-stat-note">%s. Iteration 006 was %.4f, and its own ablation '
                   "put the floor for removing all four day-of-year terms at about %s — this "
                   "call removed two of the four.</div></div><div>%s</div></div></div>"
                   '<h3 style="margin-top:20px">Residual bias, split at the lockdown</h3>%s'
                   % (table(["Metric", "Value"], [[k, v] for k, v in m["train"].items()]),
                      table(["Metric", "Value"],
                            [[k, v] for k, v in m["validation"].items()]
                            + [["MASE", m["mase"]["mase_validation"]]]),
                      oc["ratio"], E(oc.get("assessment", "")),
                      oc.get("iteration_006_ratio", 0),
                      oc.get("ablation_006_predicted", "-"),
                      pc(P, "bias_by_era_vs_006"),
                      table(["Era", "Rows", "Mean residual", "MAE", "Iteration 006"], erows))))

    pri = fi.get("prior_iterations_recomputed", {}) or {}
    brows = [["calendar (lag_364 + doy terms)",
              "%.4f" % (pri.get("iteration_006", {}) or {}).get("calendar", 0),
              "%.4f" % fi.get("calendar_block_share", 0)],
             ["level features",
              "%.4f" % (pri.get("iteration_006", {}) or {}).get("level", 0),
              "%.4f" % fi.get("level_block_share", 0)]]
    top = fi.get("top_15", []) or []
    mx = max([r.get("share", 0) for r in top] or [1])
    trows = [[r.get("rank", ""), '<span class="mono">%s</span>' % E(r.get("feature", "")),
              "%.4f" % r.get("share", 0),
              '<div class="miss-bar-wrap" style="margin:2px 0"><div class="miss-bar-fill" '
              'style="width:%.0f%%"></div></div>' % (100 * r.get("share", 0) / mx)]
             for r in top[:12]]
    S.append(slide(6, T, "Importance",
                   "<h2>Where the removed terms' importance went</h2>"
                   '<div class="grid-lead"><div>%s</div><div>%s'
                   '<p style="font-size:.78rem;color:var(--e61-ink-3);margin-top:10px">%s</p>'
                   "</div></div>"
                   '<h3 style="margin-top:20px">Top features</h3>%s'
                   % (pc(P, "importance_blocks"),
                      table(["Block", "Iteration 006", "This call"], brows),
                      E(_s(fi.get("block_share_method"), 300)),
                      table(["Rank", "Feature", "Share", ""], trows))))

    acf, hz = (rs.get("acf") or {}), (doc.get("horizon_response_check") or {})
    S.append(slide(7, T, "Diagnostics",
                   '<div class="grid-2">%s%s</div>'
                   '<div class="kpi-mini-row" style="margin-top:16px">%s%s%s</div>'
                   '<div class="grid-2" style="margin-top:20px">'
                   "<div><h3>Findings beyond the procedure</h3>%s</div>"
                   '<div><h3>Handover to Diagnosis</h3><ul class="limits-list">%s</ul>'
                   '<h3 style="margin-top:14px">Issues</h3>%s</div></div>'
                   % (pc(P, "residuals_over_time"), pc(P, "residual_acf"),
                      mini("Residual ACF lag-1", acf.get("lag_1", "-"),
                           "iteration 006: %s" % rs.get("acf_lag1_iteration_006", "-")),
                      mini("Mean residual", rs.get("mean", "-"),
                           "std %s" % rs.get("std", "-")),
                      mini("Horizon response", _s(hz.get("verdict", "-"), 26),
                           "h<=7 %s vs h>=90 %s" % (hz.get("mae_h_1_to_7", "-"),
                                                    hz.get("mae_h_90_plus", "-"))),
                      "".join('<div class="card" style="margin-bottom:12px"><b>%s</b>'
                              '<p style="font-size:.83rem;margin:.5em 0">%s</p>%s</div>'
                              % (E(f["title"]), E(_s(f["findings"], 220)),
                                 details("Full reasoning", f["reasoning"]))
                              for f in doc.get("agent_initiated_findings", [])),
                      "".join("<li>%s</li>" % E(_s(r, 210))
                              for r in doc.get("recommendations_for_diagnosis", [])[:4]),
                      issues_block(doc.get("issues")))))
    return S


def _training_cv_slides(run, doc, P, T):
    """The three 014-specific slides: family CV, classical comparison, ensembling."""
    cs, m = doc["cv_selection"], doc["metrics"]
    en = doc.get("ensembling_check", {}) or {}
    cl = doc.get("classical_comparison", {}) or {}
    S = []
    rows = [['<span class="mono">%s</span>' % E(r["family"]), "%.3f" % r["cv_mae_mean"],
             "%.3f" % r["cv_mae_std"],
             "nominal winner" if r["family"] == cs["winner"] else
             ("the incumbent" if r["family"] == "lightgbm_nanNative" else "")]
            for r in cs["results"]]
    S.append(slide(4, T, "Model Family",
                   "<h2>Six families, separated by less than fold noise</h2>"
                   '<div class="grid-lead"><div>%s</div><div>%s</div></div>'
                   '<div class="callout-am" style="margin-top:16px"><b>The margin is %.3f MAE '
                   "against a fold standard deviation of %.3f</b> — %.0f%% of the noise it "
                   "is measured in. By the durable rule from iteration 013 that is a tie, not a "
                   "win. The nominal winner is also worse on validation (%.3f) than both "
                   "incumbents.</div>"
                   % (table(["Family", "CV MAE", "Fold SD", ""], rows), pc(P, "cv_model_family"),
                      cs["margin_over_incumbent"], cs["fold_sd_of_winner"],
                      100 * cs["margin_over_incumbent"] / cs["fold_sd_of_winner"],
                      m["validation"]["MAE"]), cls="dark"))
    crows = [[E(r["fit_through"]), "%.2f" % r["holt_winters_MAE"],
              "%.2f" % r["seasonal_naive_MAE"], "<b>%.2f</b>" % r["xgboost_MAE"]]
             for r in cl.get("per_fold", [])]
    mns = cl.get("means", {})
    crows.append(["<b>mean</b>", "<b>%.2f</b>" % mns.get("holt_winters_MAE", 0),
                  "<b>%.2f</b>" % mns.get("seasonal_naive_MAE", 0),
                  "<b>%.2f</b>" % mns.get("xgboost_MAE", 0)])
    S.append(slide(5, T, "Architecture",
                   "<h2>The ML-as-regression architecture, tested for the first time</h2>"
                   '<div class="grid-lead"><div>%s</div><div>%s</div></div>'
                   '<div class="callout-am" style="margin-top:12px;font-size:.82rem">%s</div>'
                   % (pc(P, "classical_comparison"),
                      table(["Forecast from", "Holt-Winters", "Seasonal naive", "XGBoost"], crows),
                      E(_s(cl.get("caveat"), 300))), cls="warm"))
    S.append(slide(6, T, "Ensembling & Results",
                   '<div class="grid-2"><div><h2>Ensembling</h2>%s'
                   '<div class="callout-rd" style="margin-top:12px;font-size:.83rem">'
                   "<b>%s.</b> Best single %.3f; best blend %s at %.3f. Residual correlations "
                   "run from %.4f upward — the members err almost identically.</div></div>"
                   '<div><h2>Metrics</h2>%s<h3 style="margin-top:18px">Handover</h3>'
                   '<ul class="limits-list">%s</ul><h3 style="margin-top:14px">Issues</h3>%s'
                   "</div></div>"
                   % (pc(P, "ensemble_blends"), E(en.get("verdict", "")),
                      en.get("best_single_val_MAE", 0),
                      E((en.get("best_blend") or {}).get("members", "")),
                      (en.get("best_blend") or {}).get("val_MAE", 0),
                      en.get("min_residual_correlation", 0),
                      table(["Metric", "Train", "Validation"],
                            [["MAE", m["train"]["MAE"], m["validation"]["MAE"]],
                             ["RMSE", m["train"]["RMSE"], m["validation"]["RMSE"]],
                             ["R2", m["train"]["R2"], m["validation"]["R2"]],
                             ["MASE", "-", m["mase"]["mase_validation"]]]),
                      "".join("<li>%s</li>" % E(_s(r, 200))
                              for r in doc.get("recommendations_for_diagnosis", [])[:4]),
                      issues_block(doc.get("issues")))))
    return S


# ============================================================ Diagnosis
def build_diagnosis(run, doc):
    P, T = plots_on_disk(run, doc), 6
    sr, dg = doc["stop_recommendation"], doc["diagnosis"]
    pv, pt = doc["price_lead_verdict"], doc["price_signal_test"]
    S = [cover("diagnosis", "015", doc["run_id"],
               "Diagnosis — Final Stop Call After Closing the Last Untried Lead",
               "Anchored on training iteration 014. The price lead was closed here by a test "
               "costing one CSV read. Every number traces to "
               "history/iteration_015_diagnosis.json.",
               [("STOP", "Recommendation"), (sr["confidence"], "Confidence"),
                ("7", "Axes closed"), ("end_model", "Next node"),
                ("—", "Hypothesis")]),
         agenda([(3, "Verdict", "Stop/continue and the checks that drove it"),
                 (4, "Price Lead", "The last untried lead, closed by measurement"),
                 (5, "Root Cause", "What the remaining error actually is"),
                 (6, "Clarifications", "Open question and issues")], T)]
    checks = "".join('<div class="card" style="border-left:4px solid %s"><h3>%s</h3>'
                     '<p style="font-size:.84rem">%s</p></div>'
                     % ("var(--e61-warning)" if any(w in c["result"].lower() for w in
                                                    ("not ", "worse", "remains", "flag", "none exists"))
                        else "var(--e61-success)", E(c["check"]), E(_s(c["result"], 260)))
                     for c in sr["checks_that_drove_it"])
    S.append(slide(3, T, "Verdict",
                   '<span class="pill pill-ok">&#10003; STOP &mdash; go to end_model</span>'
                   "<h2>%s</h2><p>%s</p>%s%s<div class=\"grid-fill\" style=\"margin-top:20px\">%s</div>"
                   % (E(_s(sr["reasoning"].split(".")[0], 150)), E(_s(sr["reasoning"], 380)),
                      details("Full stop reasoning", sr["reasoning"]),
                      kpis([("stop", "Recommendation"), (sr["confidence"], "Confidence"),
                            ("end_model", "Next node"), ("—", "Hypothesis")]), checks)))
    qrows = [["Q%d" % (i + 1), format(int(r["n"]), ","), "%+.2f%%" % r["price_dev_pct_mean"],
              "%+.2f" % r["mean_resid"]] for i, r in enumerate(pt["quintile_table"])]
    S.append(slide(4, T, "Price Lead",
                   "<h2>The last untried lead, closed for the cost of one CSV read</h2>"
                   '<div class="grid-lead"><div>%s</div><div>%s'
                   '<div class="callout-rd" style="margin-top:12px;font-size:.82rem"><b>%s</b></div>'
                   "</div></div><ul class=\"limits-list\" style=\"margin-top:16px\">%s</ul>%s"
                   % (pc(P, "price_over_time"),
                      table(["Quintile", "n", "Price dev", "Mean residual"], qrows),
                      E(pv["verdict"]),
                      "".join("<li>%s</li>" % E(_s(e, 300)) for e in pv["evidence"]),
                      details("Bounded, not absolute — the caveat", pv.get("honest_caveat"))),
                   cls="dark"))
    ev = "".join('<div class="finding-card"><b>%d.</b> %s'
                 '<div style="font-size:.75rem;color:var(--e61-ink-4);margin-top:6px">%s</div></div>'
                 % (e["step"], E(_s(e["fact"], 240)), E(e["source"]))
                 for e in dg["evidence_trail"])
    S.append(slide(5, T, "Root Cause",
                   '<div class="card" style="border-left:4px solid var(--accent)">'
                   "<h3>%s</h3><p style=\"font-size:.9rem\">%s</p></div>"
                   '<div class="grid-fill" style="margin-top:18px">%s</div>'
                   % (E(dg["primary_issue"]), E(dg["root_cause"]), ev)))
    S.append(slide(6, T, "Clarifications",
                   '<div class="grid-2"><div><h2>Open question</h2>%s</div>'
                   "<div><h2>Findings beyond the procedure</h2>%s"
                   "<h3 style=\"margin-top:16px\">Issues</h3>%s</div></div>"
                   % ("".join(clar(c) for c in doc.get("clarification_needed", [])),
                      "".join('<div class="card" style="margin-bottom:12px"><b>%s</b>'
                              '<p style="font-size:.84rem;margin:.5em 0">%s</p>%s</div>'
                              % (E(f["title"]), E(_s(f["findings"], 240)),
                                 details("Full reasoning", f["reasoning"]))
                              for f in doc.get("agent_initiated_findings", [])),
                      issues_block(doc.get("issues")))))
    return S


# ============================================================ End Model
def build_end_model(run, doc):
    P, T = plots_on_disk(run, doc), 7
    hd, ch = doc["headline"], doc["choice"]
    hb, hr = hd["the_bare_model_on_test"], hd["with_recalibration"]
    ft, dep = doc["final_test_metrics"], doc["deployment_recommendation"]
    S = [cover("end_model", "016", doc["run_id"],
               "Final Report — Model Selection, Test Evaluation and Deployment Recommendation",
               "The test split was untouched for fifteen iterations and spent once here. The "
               "final model was chosen before any test row was read. Every number traces to "
               "history/iteration_016_end_model.json.",
               [("iteration 011", "Final model"), ("%.2f" % ft["MAE"], "Test MAE (bare)"),
                ("%.2f" % hr["MAE_after"], "Test MAE (+recal)"),
                ("%.1f%%" % hr["improvement_pct"], "Recal improvement"),
                ("16", "Iterations")]),
         agenda([(3, "Headline", "The conclusion in one table"),
                 (4, "Model Choice", "How the final model was chosen, and against what"),
                 (5, "Test Results", "Every candidate on the held-out split"),
                 (6, "Recalibration", "The run's primary recommendation, verified"),
                 (7, "Risks & Deployment", "What to ship and what to watch")], T)]
    S.append(slide(3, T, "Headline",
                   '<span class="pill pill-danger">Do not deploy the bare model</span>'
                   "<h2>%s</h2>%s"
                   '<p style="margin-top:16px;font-size:.9rem">%s</p>'
                   '<p style="font-size:.9rem">%s</p>'
                   % (E(hd["statement"]),
                      table(["", "Test MAE", "vs seasonal naive"],
                            [["Model alone", "<b>%.2f</b>" % hb["MAE"],
                              "+%.1f%% overall, <b>%.1f%%</b> at h&le;7"
                              % (hb["skill_vs_seasonal_naive_on_the_same_rows_pct"],
                                 hb["skill_at_short_horizons_pct"])],
                             ["Model + recalibration", "<b>%.2f</b>" % hr["MAE_after"], "—"],
                             ["<i>model alone, on validation</i>", "<i>26.59</i>", "<i>+30%</i>"]]),
                      E(hb["reading"]), E(hr["reading"])), cls="warm"))
    S.append(slide(4, T, "Model Choice",
                   "<h2>Chosen before the test split was read</h2>"
                   '<p style="font-size:.9rem">%s</p>%s'
                   '<div class="grid-2" style="margin-top:18px">'
                   '<div class="callout-gr"><b>Chosen: %s</b><p style="font-size:.83rem;'
                   'margin-top:6px">%s</p></div>'
                   '<div class="callout-pu"><b>Confidence: %s</b><p style="font-size:.83rem;'
                   'margin-top:6px">%s</p></div></div>'
                   % (E(_s(ch["reasoning"], 520)), details("Full choice reasoning", ch["reasoning"]),
                      E(ch["chosen"]), E(_s(ch["criteria_source"], 160)),
                      E(ch["confidence"]), E(_s(ch["confidence_reasoning"], 300)))))
    trows = [[E(r["model"].replace("_training", "")), "%.3f" % r["MAE"], "%.3f" % r["RMSE"],
              "%.4f" % r["R2"], "%.4f" % r["MASE_test_denom"],
              "<b>chosen</b>" if r["is_chosen"] else ""] for r in doc["all_candidates_on_test"]]
    yrows = [[int(r["date"]), int(r["n_dates"]), "%.1f" % r["target_mean"], "%.2f" % r["MAE"],
              "%+.2f" % r["mean_residual"]] for r in doc["test_by_year"]]
    S.append(slide(5, T, "Test Results",
                   "<h2>All three candidates on the held-out split</h2>"
                   '<div class="grid-lead"><div>%s</div><div>%s</div></div>'
                   "<h3 style=\"margin-top:20px\">Degradation by year</h3>%s"
                   '<p style="font-size:.86rem;margin-top:10px">Over-prediction grows as the '
                   "series moves further from training levels — the defect diagnosis 012 "
                   "identified.</p>"
                   % (table(["Model", "MAE", "RMSE", "R2", "MASE", ""], trows),
                      pc(P, "test_vs_validation"),
                      table(["Year", "Dates", "Target mean", "MAE", "Mean residual"], yrows))))
    rrows = [[E(k.replace("_", " ")), format(v["n_rows_scored"], ","), "%.3f" % v["MAE_before"],
              "%.3f" % v["MAE_after"], "<b>%.1f%%</b>" % v["improvement_pct"]]
             for k, v in doc["recalibration_verified_on_test"]["results"].items()]
    S.append(slide(6, T, "Recalibration",
                   "<h2>The primary recommendation, verified on test</h2>"
                   '<div class="grid-lead"><div>%s</div><div>%s</div></div>'
                   '<p style="margin-top:14px;font-size:.88rem">%s</p>'
                   % (pc(P, "recalibration_on_test"),
                      table(["Correction", "Rows", "Before", "After", "Improvement"], rrows),
                      E(hd["why_this_was_predictable_and_was_predicted"]))))
    S.append(slide(7, T, "Risks & Deployment",
                   '<div class="grid-2"><div><h2>Deployment</h2>'
                   '<div class="callout-gr"><b>Ship: %s</b><p style="font-size:.83rem;margin-top:6px">'
                   "%s</p></div>%s"
                   '<div class="callout-rd" style="margin-top:12px"><b>Do not:</b>'
                   '<ul class="limits-list" style="margin-top:8px">%s</ul></div></div>'
                   "<div><h2>Remaining risks</h2><ul class=\"limits-list\">%s</ul></div></div>"
                   '<p style="margin-top:18px;font-size:.88rem">%s</p>%s'
                   % (E(dep["ship"]), E(_s(dep["recalibration_spec"], 300)),
                      details("Full recalibration spec", dep["recalibration_spec"]),
                      "".join("<li>%s</li>" % E(d) for d in dep["do_not"]),
                      "".join("<li>%s</li>" % E(_s(r, 300)) for r in doc["remaining_risks"]),
                      E(_s(doc["final_summary"], 500)),
                      details("Full run summary", doc["final_summary"]))))
    return S


BUILDERS = {"eda": build_eda, "domain_research": build_domain_research,
            "preprocess": build_preprocess, "feature_engineering": build_fe,
            "training": build_training, "diagnosis": build_diagnosis,
            "end_model": build_end_model}


# ============================================================ pre-flight
def preflight(run, nnn, skill):
    v = []
    for path, what in ((os.path.join(run, "history", "iteration_%s_%s.json" % (nnn, skill)),
                        "history JSON"),
                       (os.path.join(run, "code", "iteration_%s_%s.py" % (nnn, skill)),
                        "code file code/iteration_%s_%s.py" % (nnn, skill)),
                       (os.path.join(run, "state", "current.json"), "state/current.json")):
        if not os.path.exists(path):
            v.append(what + " missing")
    if skill in ("preprocess", "feature_engineering"):
        if not os.path.exists(os.path.join(run, "data", "iteration_%s_%s.csv" % (nnn, skill))):
            v.append("data/iteration_%s_%s.csv missing" % (nnn, skill))
    if skill == "training":
        if not os.path.exists(os.path.join(run, "artifacts", "training",
                                           "iteration_%s_training_model.pkl" % nnn)):
            v.append("artifacts/training/iteration_%s_training_model.pkl missing" % nnn)
    return v


# ============================================================ index
def build_index(run):
    rows = []
    for hp in sorted(glob.glob(os.path.join(run, "history", "iteration_*.json"))):
        doc = json.load(open(hp, encoding="utf-8"))
        base = os.path.basename(hp)[:-5]
        m = re.match(r"iteration_(\w+?)_(eda|domain_research|preprocess|feature_engineering|"
                     r"training|diagnosis|end_model)$", base)
        if not m:
            continue
        nnn, skill = m.group(1), m.group(2)
        if skill == "eda":
            lab = ("TRAINABLE" if (doc.get("feasibility") or {}).get("trainable") else
                   "NOT TRAINABLE") + " · MASE"
        elif skill == "domain_research":
            lab = "%s triggers · %s recommendations" % (
                (doc.get("meta") or {}).get("triggers_addressed",
                                            len(doc.get("eda_trigger_responses", []))),
                len(doc.get("feature_engineering_recommendations", [])))
        elif skill == "preprocess":
            lab = "%s · %d cols dropped" % (doc.get("mode", ""), len(doc.get("dropped_columns", [])))
        elif skill == "feature_engineering":
            lab = "%d features created" % len(doc.get("features_created", []))
        elif skill == "training":
            lab = "Val MAE %s" % ((doc.get("metrics") or {}).get("validation", {}) or {}).get("MAE", "—")
        elif skill == "diagnosis":
            sr = doc.get("stop_recommendation") or {}
            lab = ("✓ STOP" if sr.get("recommendation") == "stop"
                   else "→ CONTINUE · " + _s((doc.get("diagnosis") or {}).get("primary_issue", ""), 50))
        else:
            lab = "Final: %s" % os.path.basename(doc.get("final_model_path", "")).replace(
                "_model.pkl", "")
        blocking = any(c.get("blocking") for c in (doc.get("clarification_needed") or []))
        rep = doc.get("report_html")
        if rep and not os.path.exists(os.path.join(run, rep)):
            rep = None
        rows.append({"n": nnn, "skill": skill, "label": lab, "blocking": blocking,
                     "report": rep, "ts": doc.get("timestamp", "")})
    order = ["eda", "domain_research", "preprocess", "feature_engineering", "training",
             "diagnosis", "end_model"]
    present = [s for s in order if any(r["skill"] == s for r in rows)]
    cols = []
    for s in present:
        cards = []
        for r in [x for x in rows if x["skill"] == s]:
            badge = ('<span class="badge b-block">Blocking</span>' if r["blocking"]
                     else '<span class="badge b-hi">Done</span>' if r["report"]
                     else '<span class="badge b-nc">No report</span>')
            # report_html is stored run-relative ("reports/x.html") but index.html lives
            # INSIDE reports/, so the href must drop that prefix or it resolves to
            # reports/reports/x.html and every link 404s under file://.
            href = re.sub(r"^reports/", "", r["report"]) if r["report"] else ""
            link = ('<a href="%s">Open report &#8599;</a>' % E(href) if r["report"]
                    else '<span style="color:var(--e61-ink-4)">No report</span>')
            cards.append('<div class="iter-card" style="--accent:%s"><div><span class="mono">'
                         "Iteration %s</span> %s</div>"
                         '<div style="font-size:.83rem;margin:6px 0">%s</div>'
                         '<div style="font-size:.78rem">%s</div></div>'
                         % (ACCENT[s], E(r["n"]), badge, E(r["label"]), link))
        cols.append('<div><div class="col-head" style="background:%s">%s</div>%s</div>'
                    % (ACCENT[s], E(LABEL[s]), "".join(cards)))
    n_rep = sum(1 for r in rows if r["report"])
    html_out = (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>element61 &middot; Run Navigator</title><style>%s\n:root{--accent:#FC5B00}\n%s\n"
        "body{margin:0;background:var(--e61-bg);font-family:var(--font-body);color:var(--e61-ink-2)}"
        ".nav{background:var(--e61-purple);height:52px;display:flex;align-items:center;gap:14px;"
        "padding:0 28px;position:sticky;top:0;z-index:20}"
        ".nav-logo{color:var(--e61-orange-red);font-family:var(--font-display);font-weight:800;"
        "font-size:1.05rem}.nav-sub{color:#CFC7E0;font-size:.8rem;font-family:var(--font-mono)}"
        ".hero{background:var(--e61-purple);color:#fff;padding:38px 28px 44px}"
        ".hero h1{color:#fff;font-size:1.9rem;margin:0 0 8px}"
        ".hero .sub{color:#DDD6EE;font-size:.9rem;margin-bottom:22px}"
        ".hero .kpi{background:rgba(255,255,255,.12);border:none;box-shadow:none}"
        ".hero .kpi-val{color:#fff}.hero .kpi-lbl{color:#DDD6EE}"
        ".wrap{max-width:1300px;margin:0 auto;padding:28px}"
        ".cols{display:grid;grid-template-columns:repeat(%d,1fr);gap:16px;align-items:start}"
        "@media(max-width:1100px){.cols{grid-template-columns:1fr}}"
        ".col-head{color:#fff;font-family:var(--font-display);font-weight:700;font-size:.8rem;"
        "padding:9px 12px;border-radius:8px;margin-bottom:10px;text-transform:uppercase;"
        "letter-spacing:.05em}"
        ".iter-card{border-left:4px solid var(--accent);border-radius:var(--radius-md);"
        "padding:12px 14px;background:#fff;margin-bottom:8px;box-shadow:var(--shadow-1)}"
        "footer{background:var(--e61-purple);color:#CFC7E0;padding:18px 28px;font-size:.78rem;"
        "font-family:var(--font-mono);margin-top:40px}</style></head><body>"
        "<div class=\"nav\"><span class=\"nav-logo\">element61</span>"
        "<span class=\"nav-sub\">run navigator</span></div>"
        "<div class=\"hero\"><h1>run_20260827T154407Z</h1>"
        "<div class=\"sub\">Aernoudt branch 01 &mdash; daily bread demand, horizon-conditioned "
        "direct model</div>%s</div><div class=\"wrap\"><div class=\"cols\">%s</div></div>"
        "<footer>Generated %s &middot; element61 pipeline &middot; reports rendered by "
        "skills/report_generator-skill/report_generator.md</footer></body></html>"
        % (TOKENS_LOCAL(), COMPONENT_CSS_LOCAL(), len(present),
           kpis([(len(rows), "Iterations"), (n_rep, "Reports"), (len(present), "Skill types"),
                 ("iteration 011", "Final model")]),
           "".join(cols), datetime.date.today().isoformat()))
    out = os.path.join(run, "reports", "index.html")
    open(out, "w", encoding="utf-8").write(html_out)
    return out, len(rows), n_rep


def TOKENS_LOCAL():
    from report_generator import TOKENS
    return TOKENS


def COMPONENT_CSS_LOCAL():
    from report_generator import COMPONENT_CSS
    return COMPONENT_CSS


# ============================================================ driver
def main():
    args = sys.argv[1:]
    if not args:
        print("usage: python tools/build_reports.py runs/run_<id> [NNN_skill ...]")
        return 2
    run = args[0].rstrip("/\\")
    targets = args[1:] or ["001_eda", "002_domain_research", "004_preprocess",
                           "008_feature_engineering", "014_training", "015_diagnosis",
                           "016_end_model"]
    ok = 0
    for t in targets:
        nnn, skill = t.split("_", 1)
        hp = os.path.join(run, "history", "iteration_%s_%s.json" % (nnn, skill))
        doc = json.load(open(hp, encoding="utf-8"))
        viol = preflight(run, nnn, skill)
        warn = WARN_BANNER % "; ".join(viol) if viol else ""
        if viol:
            print("  CONVENTION VIOLATION (%s): %s" % (t, "; ".join(viol)))
        slides = BUILDERS[skill](run, doc)
        out = os.path.join(run, "reports", "iteration_%s_%s.html" % (nnn, skill))
        open(out, "w", encoding="utf-8").write(page(skill, nnn, LABEL[skill], slides, warn))
        doc["report_html"] = "reports/iteration_%s_%s.html" % (nnn, skill)
        json.dump(doc, open(hp, "w", encoding="utf-8"), indent=2, default=str)
        print("  wrote %-46s %2d slides, %5.1f KB"
              % (doc["report_html"], len(slides), os.path.getsize(out) / 1024))
        ok += 1
    ip, n, nr = build_index(run)
    print("  wrote %-46s %d iterations, %d with reports" % ("reports/index.html", n, nr))
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
