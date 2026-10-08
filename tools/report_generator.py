"""
report_generator -- renders a run's history JSON into element61 slide-deck HTML reports.

Implements skills/report_generator-skill/report_generator.md. Design tokens are copied from
"element61 Design System/colors_and_type.css" (that file wins on any conflict); the
component CSS, deck engine, details-disclosure and plot-modal blocks are the verbatim
blocks that skill specifies.

Write scope, per the skill: reports/*.html and the single `report_html` field in each
history JSON. Nothing else in the run directory is created or modified. This file itself
lives in tools/ alongside handoff_check.py because it is a reusable tool, not an
iteration artifact.

Usage:
    python tools/report_generator.py runs/run_<id> 001_eda 004_preprocess ...
    python tools/report_generator.py runs/run_<id> --index-only
"""
import os
import re
import sys
import json
import glob
import html
import datetime

# ---------------------------------------------------------------- design tokens
# Copied verbatim from element61 Design System/colors_and_type.css :root.
TOKENS = """
@import url("https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Open+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap");
:root{
--e61-orange-red:#FC5B00;--e61-orange:#FC9808;--e61-yellow:#FCBA08;
--e61-magenta:#9D1D56;--e61-purple:#301D56;
--e61-ink:#1A1230;--e61-ink-2:#301D56;--e61-ink-3:#5B5070;--e61-ink-4:#8B8298;
--e61-line:#E6E2EC;--e61-line-2:#D8D2E0;--e61-surface:#FFFFFF;--e61-bg:#FAF8F5;
--e61-bg-warm:#FFF6EC;--e61-bg-cool:#F2EEF6;
--e61-gradient-warm:linear-gradient(135deg,#FCBA08 0%,#FC9808 50%,#FC5B00 100%);
--e61-success:#2F8F5C;--e61-warning:#FCBA08;--e61-danger:#C0294A;--e61-info:#2F6BB3;
--font-display:"Montserrat","Helvetica Neue",system-ui,sans-serif;
--font-body:"Open Sans","Helvetica Neue",Arial,system-ui,sans-serif;
--font-mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,monospace;
--space-1:4px;--space-2:8px;--space-3:12px;--space-4:16px;--space-5:24px;--space-6:32px;--space-7:48px;
--radius-sm:4px;--radius-md:8px;--radius-lg:14px;--radius-xl:22px;--radius-pill:999px;
--shadow-1:0 1px 2px rgba(48,29,86,.06),0 1px 1px rgba(48,29,86,.04);
--shadow-2:0 4px 16px rgba(48,29,86,.08),0 1px 2px rgba(48,29,86,.04);
--shadow-3:0 10px 28px rgba(48,29,86,.12),0 2px 6px rgba(48,29,86,.06);
--ease-standard:cubic-bezier(0.2,0.7,0.2,1);--dur-fast:120ms;--dur-base:200ms;
}
"""

SHELL_CSS = """
html,body{margin:0;padding:0;background:var(--e61-bg);color:var(--e61-ink-2);
 font-family:var(--font-body);line-height:1.6;scroll-snap-type:y proximity;height:100%;
 overflow-y:auto;scroll-behavior:smooth}
.slide{min-height:100vh;scroll-snap-align:start;display:flex;flex-direction:column;
 justify-content:center;padding:88px 64px 96px;position:relative;
 border-bottom:1px solid var(--e61-line);box-sizing:border-box}
.slide-inner{max-width:1180px;margin:0 auto;width:100%}
.slide-num{position:absolute;top:28px;left:64px;font-family:var(--font-mono);font-size:.72rem;
 letter-spacing:.1em;color:var(--e61-ink-4);text-transform:uppercase}
.slide-tag{position:absolute;top:28px;right:64px;font-family:var(--font-mono);font-size:.72rem;
 color:var(--e61-ink-4)}
@media(max-width:700px){.slide{padding:72px 24px 80px}.slide-num,.slide-tag{left:24px;right:24px}}
.topbar{position:fixed;top:0;left:0;right:0;height:3px;background:var(--e61-line);z-index:201}
.topbar-fill{height:100%;background:var(--e61-gradient-warm);width:0;transition:width .25s}
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
.cover{min-height:100vh;scroll-snap-align:start;background:var(--e61-bg);color:var(--e61-ink);
 display:flex;flex-direction:column;justify-content:center;padding:64px 64px 64px 68px;
 position:relative;overflow:hidden;box-sizing:border-box;border-left:6px solid var(--accent)}
.cover-inner{position:relative;max-width:980px;margin:0 auto;width:100%}
.cover .logo{font-family:var(--font-display);font-weight:800;font-size:1rem;letter-spacing:.02em;
 color:var(--accent);margin-bottom:40px;display:flex;align-items:center;gap:10px}
.cover .logo svg{width:26px;height:26px}
.cover h1{font-family:var(--font-display);font-weight:700;font-size:2.6rem;color:var(--e61-purple);
 line-height:1.15;margin:0 0 18px}
.cover .sub{font-size:1.05rem;color:var(--e61-ink-3);max-width:680px;margin-bottom:34px}
@media(max-width:700px){.cover h1{font-size:1.9rem}}
.slide.dark{background:var(--e61-bg-cool);border-left:6px solid var(--e61-purple)}
.slide.dark h1,.slide.dark h2,.slide.dark h3,.slide.dark h4{color:var(--e61-purple)}
.slide.warm{background:var(--e61-bg-warm);border-left:6px solid var(--e61-orange-red)}
.slide.warm h1,.slide.warm h2,.slide.warm h3{color:var(--e61-ink)}
"""

COMPONENT_CSS = """
*{box-sizing:border-box}
h1,h2,h3,h4{font-family:var(--font-display);color:var(--e61-ink);margin:0 0 .5em;font-weight:700;text-wrap:balance}
p{margin:0 0 1em}
a{color:var(--e61-orange-red)}
.mono{font-family:var(--font-mono)}
img{max-width:100%;display:block}
.eyebrow{font-family:var(--font-body);font-size:.8125rem;font-weight:600;text-transform:uppercase;
 letter-spacing:.14em;color:var(--e61-orange-red);margin-bottom:10px;display:block}
.kpi-row,.kpi-row-lt{display:flex;gap:14px;flex-wrap:wrap;margin-top:20px}
.kpi{background:#fff;border:1px solid var(--e61-line);border-radius:14px;box-shadow:var(--shadow-1);
 padding:16px 22px;min-width:150px}
.kpi-val{font-family:var(--font-display);font-size:1.7rem;font-weight:700;color:var(--e61-purple)}
.kpi-lbl{font-size:.72rem;margin-top:4px;color:var(--e61-ink-3)}
.pill{display:inline-flex;align-items:center;gap:8px;padding:8px 22px;border-radius:999px;
 font-family:var(--font-display);font-weight:700;font-size:.9rem;margin-bottom:10px}
.pill-ok{background:var(--e61-success);color:#fff}
.pill-danger{background:var(--e61-danger);color:#fff}
.pill-accent{background:var(--accent);color:#fff}
.badge{display:inline-block;font-family:var(--font-display);font-size:.62rem;font-weight:700;
 padding:3px 9px;border-radius:5px;letter-spacing:.05em;text-transform:uppercase}
.b-hi{background:var(--e61-success);color:#fff}.b-med{background:var(--e61-warning);color:var(--e61-ink)}
.b-lo{background:var(--e61-magenta);color:#fff}.b-lk{background:var(--e61-danger);color:#fff}
.b-tl{background:var(--e61-warning);color:var(--e61-ink)}.b-nc{background:var(--e61-line);color:var(--e61-ink-3)}
.b-sk{background:var(--e61-warning);color:var(--e61-ink)}.b-block{background:var(--e61-danger);color:#fff}
.b-nb{background:var(--e61-warning);color:var(--e61-ink)}.b-mnar{background:var(--e61-magenta);color:#fff}
.b-mar{background:var(--e61-info);color:#fff}.b-mcar{background:var(--e61-success);color:#fff}
.b-amb{background:var(--e61-info);color:#fff}.b-reg{background:var(--e61-purple);color:#fff}
.b-ok{background:#D1FAE5;color:#065F46}
.card{background:var(--e61-surface);border:1px solid var(--e61-line);border-radius:16px;
 padding:22px 24px;box-shadow:var(--shadow-1)}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.grid-fill{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.grid-lead{display:grid;grid-template-columns:2fr 1fr;gap:18px;align-items:start}
@media(max-width:900px){.grid-2,.grid-3,.grid-4,.grid-lead,.grid-fill{grid-template-columns:1fr}}
.finding-card{background:#fff;border-left:4px solid var(--accent);border-radius:0 14px 14px 0;
 padding:18px 20px;box-shadow:var(--shadow-1);font-size:.92rem;color:var(--e61-ink-2);line-height:1.6}
.plot-card{background:#fff;border-radius:16px;border:1px solid var(--e61-line);overflow:hidden;
 box-shadow:var(--shadow-2)}
.plot-card img{width:100%}
.plot-cap{padding:14px 18px;font-size:.8rem;color:var(--e61-ink-3);border-top:1px solid var(--e61-line)}
.plot-cap b{color:var(--e61-ink-2)}
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
table{width:100%;border-collapse:collapse;font-size:.84rem}
th{text-align:left;padding:9px 12px;font-family:var(--font-display);font-size:.7rem;color:var(--e61-ink-3);
 border-bottom:2px solid var(--e61-line);font-weight:600;background:var(--e61-bg-cool);
 text-transform:uppercase;letter-spacing:.04em}
td{padding:9px 12px;border-bottom:1px solid var(--e61-line);color:var(--e61-ink-2);vertical-align:top}
tr:last-child td{border-bottom:none}
.miss-bar-wrap{background:var(--e61-line);border-radius:99px;height:8px;margin:10px 0}
.miss-bar-fill{height:8px;background:var(--accent);border-radius:99px}
.details-btn{font-family:var(--font-display);font-weight:600;font-size:.78rem;color:var(--e61-orange-red);
 background:var(--e61-bg-warm);border:1px solid #FCE0BE;border-radius:999px;padding:8px 16px;
 cursor:pointer;display:inline-flex;align-items:center;gap:6px;transition:background .15s}
.details-btn:hover{background:#FFE8CC}
.details-btn::after{content:"\\25BE";font-size:.65rem;transition:transform .2s}
.details-btn.open::after{transform:rotate(180deg)}
.details-panel{max-height:0;overflow:hidden;transition:max-height .35s ease;margin-top:0}
.details-panel.open{max-height:1400px;margin-top:14px}
.details-panel .inner{background:#fff;border-left:3px solid var(--e61-purple);
 border-radius:0 10px 10px 0;padding:14px 18px;font-size:.86rem;color:var(--e61-ink-2);line-height:1.7}
.limits-list{list-style:none;padding:0;margin:0}
.limits-list li{position:relative;padding-left:22px;margin-bottom:10px;font-size:.88rem;color:var(--e61-ink-2)}
.limits-list li::before{content:"\\25CF";position:absolute;left:0;color:var(--e61-yellow);font-size:.6rem;top:.45em}
.clar-card{border-left:4px solid var(--accent);padding:16px 20px;border-radius:0 12px 12px 0;margin-bottom:14px}
.clar-card.blocking{border-color:var(--e61-danger);background:var(--e61-bg-warm)}
.clar-card.non-blocking{border-color:var(--e61-line-2);background:var(--e61-surface)}
.callout-gr{background:#F0FBF4;border-left:4px solid var(--e61-success);padding:14px 18px;border-radius:0 10px 10px 0}
.callout-am{background:var(--e61-bg-warm);border-left:4px solid var(--e61-warning);padding:14px 18px;border-radius:0 10px 10px 0}
.callout-pu{background:var(--e61-bg-cool);border-left:4px solid var(--e61-purple);padding:14px 18px;border-radius:0 10px 10px 0}
.callout-rd{background:#FEF2F2;border-left:4px solid var(--e61-danger);padding:14px 18px;border-radius:0 10px 10px 0}
.img-modal{display:none;position:fixed;inset:0;background:rgba(26,18,48,.6);z-index:500;
 align-items:center;justify-content:center;padding:24px;backdrop-filter:blur(3px)}
.img-modal.open{display:flex}
.img-modal-inner{background:#fff;border-radius:18px;box-shadow:var(--shadow-3);
 display:grid;grid-template-columns:1.4fr 1fr;max-width:1040px;width:100%;max-height:88vh;
 overflow:hidden;position:relative}
.img-modal-close{position:absolute;top:12px;right:12px;background:var(--e61-line);border:none;
 border-radius:50%;width:30px;height:30px;cursor:pointer;font-size:.85rem;color:var(--e61-ink-2);z-index:10}
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
@media print{.dotnav,.navbar,.topbar{display:none}.slide,.cover{min-height:0;page-break-after:always}}
"""

ENGINE_JS = """
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
function toggleDetails(btn){
  btn.classList.toggle('open');
  const panel = btn.nextElementSibling;
  panel.classList.toggle('open');
  btn.textContent = panel.classList.contains('open')
    ? btn.textContent.replace(/^Full/, 'Hide')
    : (btn.dataset.orig || btn.textContent.replace(/^Hide/, 'Full'));
}
document.querySelectorAll('.details-btn').forEach(b=>{ b.dataset.orig = b.textContent; });
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
  const m = document.getElementById('imgModal');
  if(m && m.classList.contains('open') && e.key==='Escape') closeImgModal();
});
wireImgModals();
"""

MODAL_HTML = """
<div class="img-modal" id="imgModal" onclick="if(event.target.id==='imgModal')closeImgModal()">
  <div class="img-modal-inner">
    <button class="img-modal-close" onclick="closeImgModal()">&#10005;</button>
    <div class="img-modal-img-wrap"><img id="imgModalImg" src="" alt=""></div>
    <div class="img-modal-meta">
      <div class="img-modal-name" id="imgModalName"></div>
      <div class="img-modal-body" id="imgModalBody"></div>
    </div>
  </div>
</div>
"""

LOGO_SVG = ('<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<circle cx="12" cy="12" r="10" fill="currentColor"/></svg>')

ACCENT = {
    "eda": "#FC5B00", "preprocess": "#9D1D56", "feature_engineering": "#2F6BB3",
    "training": "#2F8F5C", "diagnosis": "#0E7C86", "domain_research": "#5C4EA8",
    # EXTENSION: skills/report_generator-skill/report_generator.md defines six skill accents and has no
    # end_model section at all. Deep purple is the design system's own --e61-purple,
    # unused by any other skill, so the six existing accents stay distinguishable.
    "end_model": "#301D56",
}
LABEL = {"eda": "EDA", "domain_research": "Domain Research", "preprocess": "Preprocess",
         "feature_engineering": "Feature Engineering", "training": "Training",
         "diagnosis": "Diagnosis", "end_model": "End Model"}


# ---------------------------------------------------------------- helpers
def E(x):
    return html.escape(str(x), quote=True)


def _s(x, n=None):
    """Text of a field that may be missing -- never invents content."""
    if x is None:
        return ""
    t = str(x).replace("\n\n", " ").replace("\n", " ").strip()
    if n and len(t) > n:
        cut = t[:n]
        sp = cut.rfind(". ")
        return (cut[:sp + 1] if sp > n * 0.4 else cut.rstrip() + "…")
    return t


def kpi(val, lbl):
    return '<div class="kpi"><div class="kpi-val">%s</div><div class="kpi-lbl">%s</div></div>' % (
        E(val), E(lbl))


def kpis(pairs):
    return '<div class="kpi-row">%s</div>' % "".join(kpi(v, l) for v, l in pairs)


def mini(lbl, val, note=""):
    n = '<div class="kpi-mini-note">%s</div>' % E(note) if note else ""
    return ('<div class="kpi-mini"><div class="kpi-mini-label">%s</div>'
            '<div class="kpi-mini-val">%s</div>%s</div>' % (E(lbl), E(val), n))


def details(label, text):
    if not text:
        return ""
    return ('<button class="details-btn" onclick="toggleDetails(this)">%s</button>'
            '<div class="details-panel"><div class="inner">%s</div></div>'
            % (E(label), E(str(text)).replace("\n", "<br>")))


def table(headers, rows, tint=None):
    th = "".join("<th>%s</th>" % E(h) for h in headers)
    body = []
    for i, r in enumerate(rows):
        style = ' style="background:%s"' % tint(r) if tint and tint(r) else ""
        body.append("<tr%s>%s</tr>" % (style, "".join("<td>%s</td>" % c for c in r)))
    return "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (th, "".join(body))


def conf_badge(c):
    m = {"high": "b-hi", "medium": "b-med", "low": "b-lo"}
    c = (c or "").lower()
    return '<span class="badge %s">%s confidence</span>' % (m.get(c, "b-nc"), E(c or "n/a"))


def plot_card(img_rel, exists, slug, caption, modal_body):
    if not exists:
        return ('<div class="card"><b>%s</b><p style="font-size:.85rem;color:var(--e61-ink-3)">%s'
                '</p><p style="font-size:.75rem;color:var(--e61-ink-4)">Plot not present on disk; '
                'text-only card rendered.</p></div>' % (E(slug), E(caption)))
    return ('<div class="plot-card"><img src="%s" alt="%s" loading="lazy" '
            'data-modal-title="%s" data-modal-body="%s">'
            '<div class="plot-cap"><b>%s</b> &mdash; %s</div></div>'
            % (E(img_rel), E(slug), E(slug), E(_s(modal_body)), E(slug), E(caption)))


def clar(entry, accent_only=False):
    blocking = bool(entry.get("blocking"))
    cls = "blocking" if blocking and not accent_only else "non-blocking"
    bb = ('<span class="badge b-block">Blocking</span>' if blocking
          else '<span class="badge b-nb">Non-blocking</span>')
    q = entry.get("question") or entry.get("description") or entry.get("topic", "")
    ctx = entry.get("recommendation") or entry.get("why_asked") or entry.get("context") or ""
    extra = ""
    if entry.get("why_asked") and entry.get("recommendation"):
        extra = '<p style="font-size:.8rem;color:var(--e61-ink-4)"><b>Why asked:</b> %s</p>' % E(
            _s(entry["why_asked"]))
    return ('<div class="clar-card %s">%s %s<h4 style="margin-top:10px">%s</h4>'
            '<p style="font-size:.85rem;color:var(--e61-ink-3)">%s</p>%s%s</div>'
            % (cls, bb, conf_badge(entry.get("confidence")),
               E(entry.get("topic", "")), E(_s(q, 320)),
               details("Full question", q if len(str(q)) > 320 else None) or "",
               ('<p style="font-size:.85rem"><b>Recommendation:</b> %s</p>%s' %
                (E(_s(ctx, 300)), details("Full recommendation", ctx if len(str(ctx)) > 300 else None))
                if ctx else "") + extra))


def issues_block(issues):
    if not issues:
        return '<div class="callout-gr"><b>&#10003; No issues recorded.</b></div>'
    return "".join('<div class="callout-rd">%s</div>' % E(_s(json.dumps(i) if isinstance(i, dict)
                                                            else i, 400)) for i in issues)


def findings_cards(fs):
    out = []
    for f in fs or []:
        out.append('<div class="finding-card"><b>%s</b><p style="margin:.5em 0 0;font-size:.85rem">'
                   '%s</p>%s<div style="margin-top:8px">%s</div></div>'
                   % (E(f.get("title", "")), E(_s(f.get("findings"), 300)),
                      details("Full reasoning", f.get("reasoning")),
                      conf_badge(f.get("confidence"))))
    return '<div class="grid-fill">%s</div>' % "".join(out) if out else ""


# ---------------------------------------------------------------- page shell
def slide(num, total, tag, inner, cls=""):
    return ('<section class="slide %s slide-item" data-title="%s">'
            '<div class="slide-num">%02d / %02d</div><div class="slide-tag">%s</div>'
            '<div class="slide-inner">%s</div></section>'
            % (cls, E(tag), num, total, E(tag), inner))


def cover(skill, nnn, run_id, h1, sub, kpi_pairs):
    return ('<section class="cover slide-item" data-title="Cover"><div class="cover-inner">'
            '<div class="logo">%s element61 &nbsp;&middot;&nbsp; pipeline</div>'
            '<span class="eyebrow">%s report &nbsp;&middot;&nbsp; iteration %s '
            '&nbsp;&middot;&nbsp; %s</span><h1>%s</h1><p class="sub">%s</p>%s'
            '</div></section>'
            % (LOGO_SVG, E(LABEL[skill].upper()), E(nnn), E(run_id), E(h1), E(sub),
               kpis(kpi_pairs)))


def agenda(items, total):
    cards = "".join(
        '<div class="card"><b class="mono" style="color:var(--accent)">%02d</b><br><b>%s</b><br>'
        '<span style="font-size:.85rem;color:var(--e61-ink-3)">%s</span></div>' % (n, E(t), E(d))
        for n, t, d in items)
    return slide(2, total, "Agenda",
                 "<h2>Contents</h2><div class=\"grid-3\">%s</div>" % cards)


def page(skill, nnn, title, body_slides, warning=""):
    return ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>element61 &middot; %s &middot; Iteration %s</title><style>%s\n:root{--accent:%s}\n%s\n%s</style>"
            "</head><body><div class=\"topbar\"><div class=\"topbar-fill\" id=\"topbarFill\"></div></div>"
            "<div class=\"dotnav\" id=\"dotnav\"></div>"
            "<div class=\"navbar\"><button id=\"prevBtn\" title=\"Previous\">&lsaquo;</button>"
            "<span class=\"counter\" id=\"counter\">1 / N</span>"
            "<button id=\"nextBtn\" title=\"Next\">&rsaquo;</button></div>%s%s%s"
            "<script>%s</script></body></html>"
            % (E(LABEL[skill]), E(nnn), TOKENS, ACCENT[skill], SHELL_CSS, COMPONENT_CSS,
               warning, "".join(body_slides), MODAL_HTML, ENGINE_JS))


WARN_BANNER = ('<div style="background:#FEF2F2;border-left:4px solid var(--e61-danger);'
               'padding:10px 16px;margin-bottom:14px;font-family:var(--font-body);font-size:.8rem;'
               'color:#7F1D1D;border-radius:0 var(--radius-md) var(--radius-md) 0">'
               '&#9888; Convention violation detected: %s. See naming_conventions.md.</div>')
