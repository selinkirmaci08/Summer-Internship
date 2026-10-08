# %% [iteration_002_domain_research] Domain Research — run_20260908T075709Z
# Renders HTML report from history/iteration_002_domain_research.json.
# Report generation skipped per user instruction (report_generator closed).
# Re-run this script to regenerate HTML from the JSON without re-running web searches.

import json
from pathlib import Path

RUN_DIR = Path(__file__).parent.parent
HISTORY_JSON = RUN_DIR / "history" / "iteration_002_domain_research.json"
REPORT_OUT = RUN_DIR / "reports" / "iteration_002_domain_research.html"

# %% Load domain research JSON
with open(HISTORY_JSON, encoding="utf-8") as f:
    dr = json.load(f)

handoff = dr["handoff"]
meta = dr["meta"]
triggers = dr["eda_trigger_responses"]
sources = dr["data_source_assessment"]
findings = dr["structured_findings"]
features = dr["feature_engineering_recommendations"]
open_q = dr["open_questions"]

# %% Summarise to console (report_generator is closed; HTML not written)
print(f"Domain Research — {dr['run_id']} iteration {dr['iteration']}")
print(f"Location: {meta['location']} | Domain: {meta['domain']}")
print(f"Triggers addressed: {meta['triggers_addressed']}")
print(f"Feature recommendations: {meta['feature_recommendations']} "
      f"(high={sum(1 for f in features if f['priority']=='high')}, "
      f"medium={sum(1 for f in features if f['priority']=='medium')}, "
      f"low={sum(1 for f in features if f['priority']=='low')})")
print(f"Open questions: {meta['open_questions']}")
print(f"Data sources assessed: {meta['data_sources_assessed']}")

print("\n--- EDA Trigger Responses ---")
for t in triggers:
    print(f"  [{t['confidence'].upper()}] Trigger {t['trigger_id']}: {t['trigger'][:80]}...")
    print(f"    Feature implication: {t['feature_implication'][:100]}...")

print("\n--- Data Source Verdicts ---")
for src, data in sources.items():
    print(f"  {src}: {data['overall_assessment']} — {data['overall_reasoning'][:100]}...")

print("\n--- Top Feature Recommendations (HIGH priority) ---")
for feat in features:
    if feat["priority"] == "high":
        print(f"  {feat['feature']} ({feat['type']}) — {feat['rationale'][:80]}...")

print("\n--- Open Questions ---")
for q in open_q:
    print(f"  [{q['priority'].upper()}] {q['question'][:100]}...")

print("\nHandoff read_this_first:", handoff["read_this_first"])
print("\nNOTE: HTML report not generated — report_generator closed per user instruction.")
print("Domain Research complete.")
