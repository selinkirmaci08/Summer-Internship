# Handoff Contract

**Purpose: make context disposable after *every* skill run, not just between runs.**

The pipeline is designed so an agent can clear its context at any node boundary and a fresh agent
can pick up from disk. That only works if each skill's history JSON is self-sufficient for
whoever runs next. It currently is not, and the failure mode is specific and repeatable:

> **Counts and paths get promoted into state and handoffs. Content and rules do not.**

Observed in `run_20260827T154407Z`: domain research's twelve feature recommendations were carried
forward as the integer `12`; the lag-construction rule the whole run depends on
(`k = 7*ceil(h/7)`) existed only inside a code file and a prose paragraph; the reason the horizon
feature `h` had to be exempted from the relevance screen existed only inside an
`agent_initiated_findings` entry. Each of those is load-bearing. A fresh agent missing the third
would re-run the relevance screen, see `h` score exactly 0.0000, drop it, and silently destroy
the horizon-conditioned design — the model would still train and still report a metric.

## The rule

**Every skill's `history/iteration_<NNN>_<node>.json` MUST carry a top-level `handoff` block.**
It is the last thing a skill writes and the first thing the next agent reads. If a fact is needed
downstream, it goes here as a value — not as prose elsewhere in the file, and not only in code.

```json
"handoff": {
  "written_by": "<node_slug>",
  "iteration": 5,
  "context_clearable": true,
  "read_this_first": "one sentence: what the next node most needs to know",

  "parameters": {
    "<name>": {"value": <machine-readable>, "why": "...", "must_not_change": true}
  },

  "durable_rules": [
    {"rule": "...", "why": "...", "applies_to": ["feature_engineering", "training"],
     "cost_if_broken": "what actually goes wrong"}
  ],

  "methodological_cautions": [
    {"caution": "...", "why": "...", "cost_if_ignored": "..."}
  ],

  "do_not_repeat": [
    {"tried": "...", "outcome": "rejected", "evidence": "...", "why_not_again": "..."}
  ],

  "content_carried_forward": {
    "<thing>": "the actual content, never a count of it"
  }
}
```

## Field rules

- **`parameters`** — values the next node must *use*, not read about. Anything a downstream node
  would otherwise have to re-derive, re-tune, or guess. `must_not_change: true` marks values that
  exist to keep a comparison valid; changing one silently invalidates the next result.
- **`durable_rules`** — constraints that outlive this iteration. A rule belongs here if breaking
  it produces a *plausible-looking wrong answer* rather than an error. State `cost_if_broken` in
  terms of what goes wrong, not in terms of the rule being violated.
- **`methodological_cautions`** — where a standard procedure gives a misleading answer on this
  data. These are the most expensive things to rediscover, because rediscovering them requires
  first being suspicious.
- **`do_not_repeat`** — hypotheses tested and rejected, with the evidence. Prevents a fresh agent
  spending an iteration re-deriving a known negative. Each entry carries `evidence_basis` — one of
  `validation_only`, `train_cv_only`, `cross_run`, `domain_reasoning`, or `external_holdout` — and
  `reopen_if`. The tier matters: this pipeline can only ever produce the first four, because it
  holds no held-out data. `external_holdout` labels a finding inherited from an effort that was
  scored against a real holdout, and such a finding may motivate a hypothesis but may never
  justify a decision — re-establish it on this run's own validation evidence or carry it forward
  as untested. See `skills/diagnosis-skill/diagnosis.md`, "Reading findings from other runs".
- **`content_carried_forward`** — the specific antidote to the counts-not-content failure. If a
  node produced a list a downstream node needs, the list goes here in full. Never its length.

## Before you finish: the externalisation pass

Every node runs this before writing its handoff. It lived, byte-identical, in all six skill
files; it is here once because it is one rule, not six.

Everything above describes what you already know you need to record. The expensive losses are the
things you did **not** recognise as load-bearing while you were working — and by the time a later
node needs them, the context that held them is gone.

So after drafting the handoff, re-read your own reasoning from this call and check it against the
block you just wrote. For each conclusion you reached, ask whether it appears in `parameters`,
`durable_rules`, `methodological_cautions`, `do_not_repeat`, or `content_carried_forward`.

Three questions catch what the drafting pass misses:

1. **What did I notice that changed how I proceeded?** Anything that redirected you mid-call is a
   finding, even if it never became a formal result. If it would redirect the next node too, it is
   a `methodological_caution`.
2. **What did I look at that I was not asked to look at, and why?** The suspicion that sent you
   there is worth more than what you found. A later agent cannot rediscover it without first
   becoming suspicious again, which is the single most expensive thing to reproduce.
3. **What would I warn the next node about if I could only say one sentence?** If that sentence is
   not already `read_this_first`, replace `read_this_first` with it.

Assume the next agent starts with **no memory of this call** — not a summary of it, none of it.
Anything you are relying on it "just knowing" is either in this block or it is lost.

## Guardrails

- **Never store a count where the content is what is needed.** `"n_recommendations": 12` is a
  reporting statistic; the twelve recommendations are the handoff.
- **Never leave a load-bearing rule only in a code file.** Code records *what was done*; the
  handoff records *what must keep being done*. A fresh agent reads the handoff before it reads
  code, and may never read the code at all.
- **Never leave a load-bearing finding only in `agent_initiated_findings`.** That section is the
  honest record of what was explored beyond the procedure. If a finding changes what a later node
  should do, it is *also* a caution or a rule, and must be duplicated into `handoff`. Duplication
  is correct here — the two sections serve different readers.
- **A skill that changes nothing downstream still writes a `handoff` block**, with empty lists and
  `context_clearable: true`. An absent block is indistinguishable from a forgotten one.
- **`context_clearable: false`** is the escape hatch: set it when this node genuinely could not
  externalise everything the next node needs, and say what is missing in `read_this_first`. It
  should be rare, and it is a defect to be fixed rather than a state to live in.

## Where this sits relative to `state/current.json`

`state/current.json` stays the Planner's live snapshot — the cheap read that answers "where are
we". The `handoff` block is the per-iteration detail the snapshot deliberately does not carry.
The Planner should copy a handoff's `parameters` and `do_not_repeat` entries up into `current`
when they govern more than the immediately-following node.

## The resume contract: `next_iteration_contract`

A `handoff` block makes an iteration self-sufficient. It does **not**, on its own, tell a fresh
agent *what runs next* — and an agent that has just cleared its context has no other way to find
out. That is what `state/current.json` -> `current.next_iteration_contract` is for.

The Planner writes it as the last action of every loop pass (`skills/planner-skill/planner.md` Step 1g), and it
is the **first** thing a resuming agent reads — before, and usually instead of, the full state
snapshot.

```json
"next_iteration_contract": {
  "node": "training",
  "iteration": 12,
  "why_this_node": "iteration 010 root-caused the over-prediction to the day-of-year terms; this call tests dropping them, which is a training change, not a feature-engineering one",
  "read_this_first": "hyperparameters are frozen from iteration 009 so the comparison stays valid -- do not re-tune",
  "required_state_fields": ["fe_output_path", "feature_list", "recommended_metric"],
  "context": {
    "...": "the curated bundle this node needs, content and not counts"
  }
}
```

### Field rules

- **`node`** — the skill that runs next. `null` **only** when `current.run_complete` is true; an
  absent or null node on a live run means a fresh agent must re-derive routing from prose, which
  is the failure this field exists to prevent.
- **`iteration`** — the global iteration number this next call will take.
- **`why_this_node`** — the Planner's routing reason, in its own words. Without it a resuming
  Planner can see *what* ran but not *why*, and will re-consider options that were already weighed
  and rejected. This is the Planner's own handoff to itself.
- **`read_this_first`** — one sentence: the single thing that would most change how the next node
  behaves. Same role as the per-iteration `handoff.read_this_first`, aimed at the next node.
- **`required_state_fields`** — the fields the named node cannot run without. Mirrors
  `NODE_REQUIREMENTS` in `tools/handoff_check.py`, so the validator checks the contract against
  what the contract itself declares rather than against a hardcoded map.
- **`context`** — the curated bundle the next node needs, written by the Planner in Step 1d.
  Same curation discipline as any other context (see planner.md Context Curation Rules): exact
  column names, split boundaries, content never counts.

### Why the curated context lives here

The outgoing pass already knows what the incoming node needs — it just finished reasoning about
exactly that. Writing the bundle at that moment, rather than making the next agent re-derive it,
is what keeps the resume read small: the fresh agent reads one contract instead of the whole
`current` snapshot. It also means the context that was actually used is on disk and auditable,
instead of being reconstructed slightly differently on every resume.

### On a completed run

When the run is finished, set `node` and `iteration` to `null`, state that in `status`, and use
`if_resumed` to say what a later agent would need to know before adding anything to the run:

```json
"next_iteration_contract": {
  "iteration": null,
  "node": null,
  "status": "RUN COMPLETE - no further iteration contracted",
  "if_resumed": "no held-out score exists inside this run; every metric here is validation-only, and the axes closed in do_not_repeat should be read before adding anything"
}
```

---

## Validation

`tools/handoff_check.py` verifies that a run's history files carry conformant handoff blocks and
that the next node's required parameters are present. Run it before clearing context; a clean
result is what makes clearing safe, instead of an agent judging it by eye.
