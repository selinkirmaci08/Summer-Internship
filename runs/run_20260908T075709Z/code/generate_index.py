import json, glob, os

RUN_DIR = r"C:\Users\selin\OneDrive\Desktop\All\KU Leuven\internship-project\General Plan\runs\run_20260908T075709Z"
os.chdir(RUN_DIR)

files = sorted(glob.glob("history/iteration_*.json"))
iters = []
for f in files:
    with open(f, encoding="utf-8") as fh:
        d = json.load(fh)
    st = d.get("skill_type", d.get("node", d.get("type", "unknown")))
    if st in ("?", "test_evaluation_open_days", None):
        st = "evaluation"
    it = int(d.get("iteration", 0))
    ts = str(d.get("timestamp", ""))[:10]
    rh = d.get("report_html", "")
    if rh and not os.path.exists(rh):
        rh = ""
    # index.html lives inside reports/ — strip the "reports/" prefix for relative links
    if rh.startswith("reports/"):
        rh = rh[len("reports/"):]
    blocking = any(isinstance(cq, dict) and cq.get("blocking") for cq in d.get("clarification_needed", []))
    if st == "eda":
        summary = "FEASIBLE — metric: MAE | 2,897 trading days | 2010-2020"
    elif st == "domain_research":
        meta = d.get("meta", {})
        summary = str(meta.get("triggers_addressed", "4")) + " triggers addressed · " + str(meta.get("feature_recommendations", "15")) + " feature recommendations"
    elif st == "preprocess":
        summary = "INITIAL — 3-fold rolling-origin CV | 4 cols dropped | weather + competitor joined"
    elif st == "feature_engineering":
        s = d.get("summary", "")
        summary = (s[:80] if s else "Feature engineering iteration")
    elif st == "training":
        s = d.get("summary", "")
        s = s if isinstance(s, str) else str(s)
        summary = (s[:80] if s else "Training iteration")
    elif st == "diagnosis":
        stop = d.get("stop_recommendation", {})
        rec = stop.get("recommendation", "continue")
        if rec == "stop":
            summary = "STOP — goal met"
        else:
            issue = d.get("diagnosis", {}).get("primary_issue", "")
            summary = ("CONTINUE — " + issue)[:80]
    elif st == "end_model":
        s = d.get("summary", "")
        summary = (s if isinstance(s, str) else "Final 6-horizon models saved")[:80]
    elif st == "evaluation":
        s = d.get("summary", "Holdout test evaluation")
        summary = (s if isinstance(s, str) else "Holdout test evaluation")[:80]
    elif st == "baseline":
        summary = "Provisional naive baseline computed for all 6 horizons"
    else:
        summary = st
    iters.append({"it": it, "st": st, "ts": ts, "rh": rh, "summary": summary, "blocking": blocking})

SKILL_ORDER  = ["eda","domain_research","preprocess","baseline","feature_engineering","training","diagnosis","end_model","evaluation"]
SKILL_ACCENT = {"eda":"#FC5B00","domain_research":"#5C4EA8","preprocess":"#9D1D56","baseline":"#888888",
                "feature_engineering":"#2F6BB3","training":"#2F8F5C","diagnosis":"#0E7C86",
                "end_model":"#301D56","evaluation":"#FC9808"}
SKILL_LABEL  = {"eda":"EDA","domain_research":"Domain Research","preprocess":"Preprocess",
                "baseline":"Baseline","feature_engineering":"Feature Engineering","training":"Training",
                "diagnosis":"Diagnosis","end_model":"End Model","evaluation":"Evaluation"}
present = [s for s in SKILL_ORDER if any(i["st"] == s for i in iters)]

def iter_card(i):
    num = f"{i['it']:03d}"
    has = bool(i["rh"])
    acc = SKILL_ACCENT.get(i["st"], "#888")
    if i["blocking"]:
        badge = '<span style="background:#C0294A;color:#fff;font-size:.58rem;font-weight:700;padding:2px 7px;border-radius:4px;text-transform:uppercase">BLOCKING</span>'
        bg = "#FEF2F2"
    elif has:
        badge = '<span style="background:#D1FAE5;color:#065F46;font-size:.58rem;font-weight:700;padding:2px 7px;border-radius:4px;text-transform:uppercase">DONE</span>'
        bg = "#fff"
    else:
        badge = '<span style="background:#E5E7EB;color:#6B7280;font-size:.58rem;font-weight:700;padding:2px 7px;border-radius:4px;text-transform:uppercase">NO REPORT</span>'
        bg = "#FAFAFA"
    link = (f'<a href="{i["rh"]}" style="font-size:.78rem;color:#FC5B00;font-weight:600;text-decoration:none">Open Report &#8599;</a>' if has
            else '<span style="font-size:.78rem;color:#9CA3AF">No report</span>')
    summ = (i["summary"][:60] + "...") if len(i["summary"]) > 60 else i["summary"]
    return (
        f'<div style="border-left:4px solid {acc};border-radius:0 8px 8px 0;padding:9px 10px;background:{bg};margin-bottom:6px;box-shadow:0 1px 2px rgba(48,29,86,.06);overflow:hidden">'
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;flex-wrap:wrap">'
        f'<span style="font-family:monospace;font-size:.68rem;color:{acc};font-weight:700;white-space:nowrap">ITER {num}</span> {badge}</div>'
        f'<div style="font-size:.76rem;color:#301D56;margin-bottom:4px;line-height:1.35;word-break:break-word">{summ}</div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:4px">'
        f'<span style="font-family:monospace;font-size:.6rem;color:#9CA3AF;white-space:nowrap">{i["ts"]}</span> {link}</div>'
        f'</div>'
    )

skill_cols_html = ""
for s in present:
    acc = SKILL_ACCENT.get(s, "#888")
    lbl = SKILL_LABEL.get(s, s)
    cards = "".join(iter_card(i) for i in iters if i["st"] == s)
    skill_cols_html += (
        f'<div style="min-width:0;overflow:hidden">'
        f'<div style="background:{acc};color:#fff;padding:10px 14px;border-radius:8px 8px 0 0;font-family:sans-serif;font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em">{lbl}</div>'
        f'<div style="padding:8px 0">{cards}</div>'
        f'</div>'
    )

timeline_html = ""
for i in iters:
    acc = SKILL_ACCENT.get(i["st"], "#888")
    has = bool(i["rh"])
    lbl = SKILL_LABEL.get(i["st"], i["st"])[:3].upper()
    itnum = i["it"]
    open_tag  = f'<a href="{i["rh"]}" title="{lbl} iter {itnum}" style="text-decoration:none">' if has else "<span>"
    close_tag = "</a>" if has else "</span>"
    border = "#fff" if has else acc
    timeline_html += (
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:4px;min-width:56px">'
        f'{open_tag}<div style="width:38px;height:38px;border-radius:50%;background:{acc};display:flex;align-items:center;justify-content:center;color:#fff;font-family:monospace;font-size:.58rem;font-weight:700;border:2px solid {border};box-shadow:0 2px 6px rgba(0,0,0,.15)">{i["it"]:03d}</div>{close_tag}'
        f'<span style="font-size:.56rem;color:#9CA3AF;text-align:center">{lbl}</span>'
        f'</div>'
        f'<div style="width:16px;height:2px;background:#E6E2EC;align-self:center;margin-top:-14px;flex-shrink:0"></div>'
    )

legend_html = " ".join(
    f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:.72rem"><span style="width:11px;height:11px;border-radius:50%;background:{SKILL_ACCENT.get(s,"#888")};display:inline-block"></span>{SKILL_LABEL.get(s,s)}</span>'
    for s in present
)

n_reports = sum(1 for i in iters if i["rh"])
last_ts = max((i["ts"] for i in iters if i["ts"]), default="2026-09-09")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>run_20260908T075709Z — Aernoudt Bakery Bread Demand Forecast</title>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&family=Open+Sans:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Open Sans","Helvetica Neue",Arial,sans-serif;background:#FAF8F5;color:#1A1230}}
a{{color:#FC5B00;text-decoration:none}}
a:hover{{text-decoration:underline}}
.hdr{{position:sticky;top:0;background:#301D56;height:52px;display:flex;align-items:center;padding:0 32px;gap:16px;z-index:100;box-shadow:0 2px 8px rgba(48,29,86,.3)}}
.hdr-logo{{font-family:"Montserrat",sans-serif;font-weight:800;font-size:1.05rem;color:#FC5B00;letter-spacing:.02em}}
.hdr-sub{{font-size:.75rem;color:rgba(255,255,255,.65)}}
.hero{{background:linear-gradient(135deg,#1A1230 0%,#301D56 70%,#4a2d7a 100%);color:#fff;padding:40px 32px 32px;border-bottom:4px solid #FC9808}}
.hero h1{{font-family:"Montserrat",sans-serif;font-size:1.65rem;font-weight:700;margin-bottom:5px}}
.hero .meta{{font-size:.75rem;color:rgba(255,255,255,.55);margin-bottom:14px}}
.kpi-row{{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}}
.kpi{{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);border-radius:12px;padding:13px 18px;min-width:130px;backdrop-filter:blur(4px)}}
.kpi-val{{font-family:"Montserrat",sans-serif;font-size:1.55rem;font-weight:700;color:#FC9808}}
.kpi-lbl{{font-size:.64rem;color:rgba(255,255,255,.65);margin-top:3px;text-transform:uppercase;letter-spacing:.07em}}
.tabs{{display:flex;background:#231545;border-bottom:1px solid rgba(255,255,255,.08)}}
.tab{{padding:11px 22px;color:rgba(255,255,255,.55);cursor:pointer;font-size:.78rem;font-family:"Montserrat",sans-serif;font-weight:600;border-bottom:3px solid transparent;transition:all .2s;text-transform:uppercase;letter-spacing:.06em}}
.tab:hover{{color:rgba(255,255,255,.9)}}
.tab.active{{color:#FC9808;border-bottom-color:#FC9808}}
.content{{padding:26px 32px}}
@media(max-width:700px){{.content{{padding:14px}}.hdr{{padding:0 14px}}}}
.section-title{{font-family:"Montserrat",sans-serif;font-size:.68rem;font-weight:700;color:#8B8298;letter-spacing:.12em;text-transform:uppercase;margin-bottom:12px;margin-top:22px}}
footer{{background:#1A1230;color:rgba(255,255,255,.45);font-size:.7rem;padding:18px 32px;margin-top:30px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}}
</style>
</head>
<body>
<div class="hdr">
  <span class="hdr-logo">element61</span>
  <span class="hdr-sub">&nbsp;ML Pipeline &mdash; Run Navigator</span>
</div>
<div class="hero">
  <div class="meta">Run ID: run_20260908T075709Z &nbsp;&middot;&nbsp; Dataset: train_set/train &nbsp;&middot;&nbsp; Last updated: {last_ts}</div>
  <h1>Aernoudt Bakery &mdash; Bread Demand Forecast</h1>
  <div style="font-size:.88rem;color:rgba(255,255,255,.75);max-width:680px;margin-top:5px">Multi-horizon LightGBM &mdash; 6 horizons [1, 7, 14, 30, 60, 90 days] &mdash; Primary test (2021) + Stress test (2022&ndash;2023)</div>
  <div class="kpi-row">
    <div class="kpi"><div class="kpi-val">{len(iters)}</div><div class="kpi-lbl">Iterations</div></div>
    <div class="kpi"><div class="kpi-val">{n_reports}</div><div class="kpi-lbl">Reports</div></div>
    <div class="kpi"><div class="kpi-val">6</div><div class="kpi-lbl">Horizons</div></div>
    <div class="kpi"><div class="kpi-val">29.8</div><div class="kpi-lbl">Best Val MAE (h=1)</div></div>
    <div class="kpi"><div class="kpi-val">6 / 6</div><div class="kpi-lbl">Beats Naive (stress open)</div></div>
  </div>
</div>
<div class="tabs">
  <div class="tab active" onclick="switchTab('skill')">By Skill Type</div>
  <div class="tab" onclick="switchTab('timeline')">Timeline</div>
</div>
<div class="content">
  <div id="tab-skill">
    <div class="section-title">Latest Iteration per Skill Type</div>
    <div style="display:grid;grid-template-columns:repeat({len(present)},1fr);gap:10px;align-items:start">
      {skill_cols_html}
    </div>
  </div>
  <div id="tab-timeline" style="display:none">
    <div class="section-title">Run Timeline &mdash; All {len(iters)} Iterations</div>
    <div style="overflow-x:auto;padding-bottom:8px">
      <div style="display:flex;align-items:center;min-width:max-content;padding:20px 4px">
        {timeline_html}
      </div>
    </div>
    <div style="margin-top:12px;display:flex;flex-wrap:wrap;gap:10px">{legend_html}</div>
    <div class="section-title" style="margin-top:22px">All Iterations</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(275px,1fr));gap:10px">
      {"".join(iter_card(i) for i in iters)}
    </div>
  </div>
</div>
<footer>
  <span>run_20260908T075709Z &nbsp;&middot;&nbsp; Aernoudt Bakery branch 01 &nbsp;&middot;&nbsp; {len(iters)} iterations &nbsp;&middot;&nbsp; Generated {last_ts}</span>
  <span>element61 ML Pipeline</span>
</footer>
<script>
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach(function(t,i){{ t.classList.toggle('active',(name==='skill'&&i===0)||(name==='timeline'&&i===1)); }});
  document.getElementById('tab-skill').style.display = name==='skill' ? '' : 'none';
  document.getElementById('tab-timeline').style.display = name==='timeline' ? '' : 'none';
}}
</script>
</body>
</html>"""

out_path = "reports/index.html"
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(html)
print(f"Wrote {out_path} ({len(html)} chars)")
print(f"  {len(iters)} iterations | {n_reports} with reports | {len(present)} skill types: {present}")
