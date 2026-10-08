---
name: domain-research-skill
description: Gathers domain intelligence for a dataset after EDA. Reads EDA-raised triggers, does targeted web search + knowledge-base synthesis, assesses each non-target data source for usefulness, and produces structured feature recommendations grounded in cultural, geographic, operational, and event context. Runs between EDA and Preprocess when EDA sets domain_research_recommended to true.
---

# Domain Research Skill

## Purpose

After EDA has identified statistical patterns, this skill answers the "why" — providing cultural,
geographic, event-calendar, and domain-industry context that pure statistics cannot supply. It
converts EDA's `domain_research_triggers` into actionable findings, assesses each non-target data
source in the dataset (weather, competitor, external enrichment) for genuine domain utility, and
surfaces feature engineering ideas grounded in domain knowledge rather than in-sample correlation
alone.

This skill never touches the data statistically — it reads only enough of the dataset to orient
web searches (column names, date range, a few sample values). Everything it produces is reasoning
from external knowledge sources applied to what EDA already found. The goal is a structured output
that Preprocess, Feature Engineering, and Training can reference by name — not a prose document
that gets ignored after the first read.

## Inputs

- `eda_history_path` — path to `history/iteration_<NNN>_eda.json` from the most recent EDA run.
  The skill reads `domain_research_triggers`, `key_findings`, `target`, and `dataset_overview`
  from this file.
- `dataset_path` — path to the raw dataset, for a minimal scan only (column names, dtypes, date
  range, a few representative values). Never load full rows into context.
- `goal` — the stated modeling objective, in plain language.
- `iteration` — the global iteration number assigned by the Planner, used to build every output
  path. Domain research is typically labeled with a `b` suffix on the preceding EDA iteration
  (e.g., EDA was iteration 001, domain research is iteration 001b) to indicate it slots between
  EDA and Preprocess without displacing the numeric sequence.
- `context` — prior reasoning relevant to this call, per `conventions/naming_conventions.md`.
  Usually contains the EDA summary.

## Tools available

- Web search — for targeted lookups grounded in the EDA triggers (local events, holiday calendars,
  cultural patterns, industry norms, regulatory or operational context, COVID or other structural
  shock timelines). Each search should be a precise query, not a broad sweep.
- Knowledge base — internal domain knowledge about the industry, region, and dataset context.
  Use this for facts that don't require a web lookup (e.g. well-established cultural norms,
  known public holiday structures, standard industry practices).
- File read access to `history/` and the raw dataset (minimal scan only).
- File write access to `history/`, `reports/`, and `code/`.

## Procedure

**Every step below is a floor, not a ceiling.** If something about this specific domain calls for
a check or search not listed here, pursue it. Record anything beyond what's specified in
`agent_initiated_findings`, never blended into the standard output fields.

1. **Read EDA triggers and orient the research scope.**
   Load `history/iteration_<NNN>_eda.json`. Extract:
   - `domain_research_triggers` — the specific questions EDA flagged for domain input.
   - `key_findings` — what the data already showed, to avoid re-deriving statistics here.
   - `dataset_overview.time_coverage` — date range and any noted gaps.
   - `target.time_series_analysis` (if present) — seasonality and trend signals EDA found.

   If `domain_research_triggers` is absent or empty, derive triggers from `key_findings`
   directly: any finding that names an unexplained pattern, anomalous period, or unusual
   correlation is a domain research question. Record the derived triggers in `meta.triggers_derived`
   rather than `meta.triggers_addressed`, so the output is honest about which triggers came from
   EDA versus which were inferred here.

2. **Do a minimal dataset scan.**
   Load just the column names, dtypes, and ten representative rows (not raw values into context —
   compute summaries: date range, unique values per categorical column, min/max per numeric). This
   orients web searches without passing raw data into reasoning. Identify:
   - Which columns are non-target data sources (weather, competitor, external enrichment) that
     need a domain usefulness assessment in step 4.
   - The dataset's geographic location, industry, and time period — these anchor every search.
   - Any column names or value patterns that hint at context not yet covered by EDA's triggers
     (e.g. a `schema` column with values that suggest an operational regime change).

3. **Address each EDA trigger.**
   For each entry in `domain_research_triggers` (or derived triggers from step 1), formulate a
   targeted search query, search, and synthesize findings into a structured response:

   ```json
   {
     "trigger_id": 1,
     "trigger": "the observation text from EDA",
     "finding": "synthesized domain explanation, specific enough to act on",
     "confidence": "high | medium | low",
     "source": "web_search | knowledge_base | inference",
     "feature_implication": "what Preprocess or Feature Engineering should do differently"
   }
   ```

   `confidence` levels:
   - `high` — found explicit, authoritative sources or well-established domain knowledge.
   - `medium` — plausible and consistent with multiple sources but not definitively confirmed;
     or confirmed from one source without independent corroboration.
   - `low` — inferred from indirect evidence, single-source, or genuinely uncertain. Always pair
     `low` with a flag in `open_questions` asking for stakeholder confirmation.

   Source must be stated — never leave it implicit. If a finding blends web search and knowledge
   base, state both: `"web_search + knowledge_base"`.

4. **Assess each non-target data source.**
   For every column group that represents an external data source (weather features, competitor
   features, any third-party enrichment): produce a structured assessment covering:
   - **Availability window** — when does this data begin? Is there a structural boundary (e.g.
     weather only from 2015+)?
   - **Domain context** — what does this source actually measure in the real world? What does the
     domain say about its relationship to the target?
   - **Signal assessment per feature** — for each column in this source, what is the expected
     direction and strength of its relationship with the target, *before looking at any in-sample
     statistics*? This is a prior from domain knowledge, not a post-hoc rationalization.
   - **Known confounds** — are there known structural confounds that would make the in-sample
     statistics misleading? (E.g. a pre-availability fill-0 strategy creating spurious negative
     correlations with a target that was historically higher during the missing-data period.)
   - **Overall verdict** — one of: `USEFUL`, `CONDITIONALLY USEFUL`, `REAL BUT SPARSE`,
     `WEAK SIGNAL`, or `DROP CANDIDATE`. Include reasoning.
   - **Recommended additional features** — domain-grounded feature ideas specific to this source
     (e.g. extreme weather indicators, seasonal competitor closure proxy).

   **The point of this step is to form a domain prior *before* checking in-sample importance.**
   Training's feature importance is a posterior. A feature the domain says should matter but
   shows low importance is a signal to investigate (confound? missingness? encoding issue?) —
   not automatic grounds for dropping. Conversely, a feature with high in-sample importance but
   no domain rationale is a leakage or spurious-correlation risk to investigate. This step gives
   you the prior; Training gives the posterior; Diagnosis reconciles them.

5. **Broad domain sweep.**
   Even without explicit triggers, search for domain context in the following categories. Skip
   any that clearly do not apply to this dataset's domain and location. Record skipped categories
   in `meta.categories_skipped` with a one-line reason.

   | Category | What to research |
   |---|---|
   | Local events and festivals | Major annual events near the dataset location that would create demand spikes or suppressions not captured by the calendar. Include dates, duration, attendance scale, and any cancellation years (e.g. COVID). |
   | Public holidays | Official holiday calendar for the country/region. Fixed and moveable dates, cultural significance, typical retail closure vs. open patterns, anticipation-day effect (day before a holiday often matters more for demand than the holiday itself). |
   | School holidays | Annual break structure (names, typical windows, year-to-year variance). School holidays shift household routines and can create weekday demand at weekend-like levels. |
   | Demographic structure | Population, major demographic segments (students, tourists, commuters) and their seasonal presence patterns. Any segment that is reliably absent for a part of the year creates a latent seasonal pattern not captured by calendar features alone. |
   | Cultural demand patterns | Industry-specific cultural norms (e.g. Sunday bread-buying ritual in Flemish Belgium for a bakery dataset). Per-capita consumption statistics where available. Weekend vs. weekday behavioral differences. |
   | Structural shocks | COVID-19 lockdowns, economic crises, major regulatory changes, competitor market entries or exits in the dataset's time range. Include specific dates, scope, and whether the domain (e.g. food retail) was exempt from restrictions. |
   | Industry and competitive norms | Typical seasonal closure patterns for the industry (e.g. Belgian artisan bakeries closing in August), standard operating hours, pricing seasonality, or competitive dynamics that would explain demand variation. |

6. **Generate feature engineering recommendations.**
   For each recommended feature — combining the trigger responses (step 3), data source
   assessment (step 4), and broad domain sweep (step 5) — produce a structured entry:

   ```json
   {
     "feature": "feature_name",
     "type": "binary | continuous | categorical",
     "description": "exactly what values this feature takes and how it is computed",
     "priority": "high | medium | low",
     "rationale": "domain reasoning — why this feature would carry signal for the target",
     "data_source": "calendar | weather | competitor | external | domain_knowledge",
     "precondition": "any stakeholder confirmation or data access needed before implementing (omit if none)"
   }
   ```

   Priority:
   - `high` — domain strongly predicts a large demand effect; the feature is currently absent from
     the feature set; and implementation is straightforward.
   - `medium` — meaningful expected signal but either partially captured by an existing feature,
     requires a precondition to be confirmed, or the domain effect size is uncertain.
   - `low` — plausible signal but small expected effect, partially confounded, or difficult to
     implement reliably without additional data.

   **Do not recommend features that are:** (a) already present and correctly encoded in the
   dataset, (b) variants of an existing feature that would be redundant without a clear marginal
   value argument, or (c) features that require data not available at inference time (leakage).
   For any feature where inference-time availability is uncertain, add a `precondition` asking
   for clarification.

7. **Surface open questions.**
   Any finding that a stakeholder or data owner could resolve better than domain research alone
   should become an `open_questions` entry:

   ```json
   {
     "question": "specific, answerable question",
     "reason": "why the answer would change recommendations",
     "priority": "high | medium | low"
   }
   ```

   Priority `high` means the answer could materially change whether a recommended feature is
   included or how a data source is treated (e.g. null interpretation for a sparse source). Do
   not raise questions the data itself already answers — check the EDA output first.

8. **Write the output.**
   Save the full structured JSON to `history/iteration_<NNN>_domain_research.json` per
   `conventions/naming_conventions.md`. Update `state/current.json` with a history entry.
   Generate the Python script to `code/iteration_<NNN>_domain_research.py` — this script reads
   the history JSON and renders the HTML report, so the report is reproducible without re-running
   the agent search step. Save the HTML report to `reports/iteration_<NNN>_domain_research.html`.

9. **Generate an HTML report.**
   Follow the element61 design system (`element61 Design System/SKILL.md`). Use the domain
   research accent color `#5C4EA8` (medium purple). The report must include, in order:

   - **Hero** with 6 KPIs: triggers addressed, feature recommendations, data sources assessed,
     and 3 top domain facts (e.g. largest event attendance, key structural shock date,
     primary demand driver).
   - **EDA Trigger Responses** — one card per trigger with the finding, confidence badge, and
     feature implication.
   - **Highest-impact finding callout** — a visually distinct block for the single most
     actionable finding (typically a major missing feature or structural shock explanation).
   - **COVID or structural shock section** (if applicable) — timeline + demand shock bullet list
     + impact on the dataset's split/validation window.
   - **Data Source Assessment** — side-by-side panel per source with verdict badge, per-feature
     signal table, and overall assessment.
   - **Demand Drivers ranked table** — all identified demand drivers with type, strength, and
     dataset evidence.
   - **Public holidays table** (if applicable).
   - **School holidays table** (if applicable).
   - **City / demographic statistics** (if applicable).
   - **Feature Engineering Recommendations** — one card per recommendation with priority badge,
     type, description, and rationale.
   - **Open Questions** — one card per question with priority.

   The report is self-contained and file:// compatible. Google Fonts is the only external
   dependency permitted.


## Handoff block (required)

Before writing the output, add a top-level `handoff` block to this node's
`history/iteration_<NNN>_<node>.json`. **The field rules, the required fields, and the
externalisation pass that must be run before finishing are all in
`conventions/handoff_contract.md`** — that file is the specification, and it is what
`tools/handoff_check.py` validates against.

Two things worth restating here because they are the failures this node can actually cause:

- A finding recorded in `agent_initiated_findings` that changes what a later node should do must
  be duplicated into the handoff. The two serve different readers: one is the honest record of
  what was explored, the other is the instruction set for whoever runs next.
- A node that genuinely changes nothing downstream still writes the block, with empty lists and
  `context_clearable: true`. An absent block is indistinguishable from a forgotten one.

## Guardrails

- **Generated code is a cell-marked `.py`, and every plot names itself**
  (`conventions/code_and_plot_conventions.md`). Scripts carry `# %%` markers so the same file
  stays diffable and converts to a notebook on demand — never write a `.ipynb` as the artifact.
  Every figure carries its declared `name` as the title, its declared `type` stated in the
  figure, and its `iteration - node` provenance, using the same strings recorded in the JSON. A
  plot that is unidentifiable once separated from its report is not finished.
- **Never end a call without a `handoff` block** (`conventions/handoff_contract.md`). Context may
  be cleared immediately after this node returns; anything load-bearing that exists only in prose,
  only in a code file, or only in `agent_initiated_findings` is lost at that point. Never store a
  count where a later node needs the content.
- Never pass raw dataset rows into your reasoning context — only column names, dtypes, date
  range, and summary statistics. The "no raw data in context" rule from EDA applies here equally.
- Never state a domain fact without a source (`web_search`, `knowledge_base`, or `inference`).
  Inference is permitted only for logical deductions from confirmed facts — never for invented
  specifics. A made-up attendance figure is worse than saying "attendance data not found."
- Every finding must carry a `confidence` — never leave it implicit. A specific, sourced fact
  gets `high`; a plausible but single-sourced finding gets `medium`; a reasoned inference gets
  `low`. A `low`-confidence finding that would materially change recommendations must also appear
  in `open_questions`.
- Never derive feature recommendations from in-sample correlations. This skill's recommendations
  must be grounded in domain reasoning — what *should* matter, not what happens to correlate in
  this specific training window. In-sample evidence belongs to EDA and Training; domain rationale
  belongs here.
- Never recommend dropping a data source based solely on low Training feature importance. Form
  the domain prior (step 4) first; if the domain says a source should matter but importance is
  low, flag it for Diagnosis to investigate — do not recommend dropping it unilaterally here.
- Never recommend features that require data unavailable at inference time. If inference-time
  availability is uncertain for a feature, always add a `precondition` asking for confirmation
  rather than silently assuming it's available.
- Do not re-run or replicate EDA statistics. If a statistic from EDA is referenced in a finding
  (e.g. a Spearman r value), cite it from the EDA history JSON rather than recomputing it.
- Never decide to pause the workflow — raise genuine ambiguity via `open_questions` and let the
  Planner decide whether to surface it to a stakeholder. Same boundary as EDA and Preprocess.
- Every search query and its result must be summarized into a specific, citable finding — never
  carry raw search results into the output JSON or report. The output must be synthesized, not
  a search dump.
- If a web search returns no useful result, record it in `meta.search_misses` with the query
  and what was attempted. A `search_miss` is never a reason to silently omit the corresponding
  trigger response — fill it with `confidence: low` and a note that domain knowledge was the
  only source.
- The HTML report must only restate what's in the JSON output — no finding, recommendation, or
  statistic in the report that isn't also present in the structured JSON. The JSON is the source
  of truth; the report is a rendered view of it.
- The Python script in `code/` must be reproducible: given the same history JSON, re-running it
  must regenerate the same HTML report without re-running any web searches. The search step is
  the agent's work; the script's work is rendering.

## Input / output contract

Full output JSON structure:

```json
{
  "report_html": "reports/iteration_<NNN>_domain_research.html",
  "skill_type": "domain_research",
  "iteration": "<NNN or NNNb>",
  "timestamp": "ISO 8601",
  "triggered_by": {
    "iteration": "<EDA iteration number>",
    "skill": "eda",
    "trigger_count": 6
  },
  "meta": {
    "domain": "industry domain, e.g. artisan_bakery_retail",
    "location": "city, country",
    "dataset_date_range": "YYYY-MM-DD to YYYY-MM-DD",
    "search_sources": ["web_search", "knowledge_base"],
    "triggers_addressed": 6,
    "triggers_derived": 0,
    "feature_recommendations": 10,
    "open_questions": 4,
    "data_sources_assessed": 2,
    "categories_skipped": [],
    "search_misses": []
  },
  "eda_trigger_responses": [
    {
      "trigger_id": 1,
      "trigger": "...",
      "finding": "...",
      "confidence": "high | medium | low",
      "source": "web_search | knowledge_base | inference | web_search + knowledge_base",
      "feature_implication": "..."
    }
  ],
  "data_source_assessment": {
    "<source_name>": {
      "columns_in_dataset": ["col1", "col2"],
      "availability_window": "...",
      "domain_context": "...",
      "signal_assessment_per_feature": {
        "col1": "expected direction and strength, with domain reasoning"
      },
      "known_confounds": ["..."],
      "overall_assessment": "USEFUL | CONDITIONALLY USEFUL | REAL BUT SPARSE | WEAK SIGNAL | DROP CANDIDATE",
      "overall_reasoning": "...",
      "recommended_additional_features": ["feature_name (description)"]
    }
  },
  "structured_findings": {
    "local_events": {},
    "public_holidays": {},
    "school_holidays": {},
    "structural_shocks": {},
    "demand_drivers": {
      "identified_demand_drivers_ranked": [
        {
          "driver": "...",
          "type": "calendar | local_event | time_series | operational | demographic | external_shock | environmental",
          "strength": "very_high | high | medium | low | low_confounded",
          "evidence": "dataset evidence or 'not yet captured'"
        }
      ]
    },
    "city_demographics": {}
  },
  "feature_engineering_recommendations": [
    {
      "feature": "feature_name",
      "type": "binary | continuous | categorical",
      "description": "exact computation rule",
      "priority": "high | medium | low",
      "rationale": "domain reasoning",
      "data_source": "calendar | weather | competitor | external | domain_knowledge",
      "precondition": "optional — stakeholder confirmation needed before implementing"
    }
  ],
  "open_questions": [
    {
      "question": "...",
      "reason": "why the answer changes recommendations",
      "priority": "high | medium | low"
    }
  ]
}
```

Top-level output sections: `meta` (including `triggers_derived`, `search_misses`, and
`categories_skipped`), `eda_trigger_responses` (one entry per trigger, each with `confidence`,
`source`, and `feature_implication`), `data_source_assessment` (one entry per non-target data
source with per-feature signal assessment and an overall verdict), `structured_findings`
(organized by category: local events, public holidays, school holidays, structural shocks,
demand drivers ranked, city demographics — omit categories with no findings), and
`feature_engineering_recommendations` (each with `priority`, `type`, `rationale`, `data_source`,
and an optional `precondition`). A human-readable HTML report is written to
`reports/iteration_<NNN>_domain_research.html` per step 9, and a reproducible Python script to
`code/iteration_<NNN>_domain_research.py` that renders the HTML from the JSON without re-running
searches.
