"""
handoff_check.py -- can context be cleared safely right now?

Validates that every history JSON in a run carries a conformant `handoff` block
per conventions/handoff_contract.md, and that nothing load-bearing exists only as
prose or only inside a code file.

Usage:
    python tools/handoff_check.py runs/run_<id>                  # full: safe to clear?
    python tools/handoff_check.py runs/run_<id> --iteration-only  # just-finished skill only
    python tools/handoff_check.py runs/run_<id> --brief

Two modes, because they answer different questions at different moments in the loop:

  --iteration-only  validates the history handoff blocks alone. Use at planner Step 1f-bis,
                    right after a skill returns and BEFORE state is updated -- at that point
                    next_iteration_contract is still the previous pass's and is expected to
                    be stale.
  (default)         also validates state: the resume contract, the fields the next node
                    needs, and the Planner's own decision trail. This is the Step 1h gate.
                    Run it only AFTER Step 1g has written the contract.

Exit code 0 means safe to clear. Non-zero means a fresh agent would be missing
something, and the report says what.
"""
import os
import sys
import json
import glob
import re

REQUIRED = ["written_by", "iteration", "context_clearable", "read_this_first",
            "parameters", "durable_rules", "methodological_cautions",
            "do_not_repeat", "content_carried_forward"]

RULE_FIELDS = ["rule", "why", "applies_to", "cost_if_broken"]
CAUTION_FIELDS = ["caution", "why", "cost_if_ignored"]
REPEAT_FIELDS = ["tried", "outcome", "evidence", "why_not_again"]

# What state.current.next_iteration_contract must carry for a resuming agent to act on it
# alone, per conventions/handoff_contract.md.
CONTRACT_FIELDS = ["node", "iteration", "why_this_node", "read_this_first",
                   "required_state_fields", "context"]

# Fields state must carry for the named node to run without re-deriving anything.
# Two horizon fields exist and they are NOT the same thing:
#   forecast_horizons -- the lead times that each get their own trained model
#   horizon_grid      -- the (usually denser) lead times predictions are scored at
# Conflating them is how a run ends up with one model answering for four horizons.
NODE_REQUIREMENTS = {
    # data_sources: EDA is measured against the full file list, not just dataset_path.
    "eda": ["goal", "dataset_path", "target_column", "data_sources"],
    "domain_research": ["goal", "eda_summary_ref"],
    # join_plan: Preprocess physically combines the sources; it is always present after EDA,
    # empty when the run genuinely has one source.
    "preprocess": ["task_type", "must_drop_columns", "confirmed_decisions", "join_plan"],
    "feature_engineering": ["preprocess_output_path", "split_info", "horizon_grid"],
    "training": ["fe_output_path", "feature_list", "recommended_metric"],
    "diagnosis": ["latest_model_path", "latest_metrics"],
    # end_model compares every candidate, so the LIST of saved models is load-bearing --
    # latest_model_path alone would silently reduce it to a one-model run.
    # stop_justification/baseline_floor: what stopped the run, and what the champion beat.
    "end_model": ["latest_model_path", "all_saved_models", "split_info", "feature_list",
                  "stop_justification", "baseline_floor"],
}

# Conventions introduced on 2026-09-04: the baseline floor before any learned model, the
# written stop justification, blocking clarifications halting the run, one model per horizon,
# and both non-automatic time-series approaches. Runs started before that date cannot satisfy
# them retroactively, so the checks below apply only to runs whose state declares the version.
# A run opts in via current.run_config.conventions_version, written by the Planner at Step 0.
NEW_RULES_FROM = "2026-09-04"

TIME_SERIES_TASKS = ("time_series", "time_series_regression", "time_series_forecasting",
                     "forecasting")

# Both non-automatic approaches must have been fitted before a time-series run may end.
TS_APPROACHES_REQUIRED = ("classical_statistical", "ml_as_regression")

# Every clarification carries the answer its own node would give, and what a wrong answer costs.
CLARIFICATION_FIELDS = ["recommendation", "impact_if_wrong"]

# A count where the content is what a downstream node needs. The pattern that
# motivated this whole contract.
COUNT_SMELL = re.compile(r"^(n_|num_|count_)|_count$|_recommendations$|_n$")


def check_run(run_dir, brief=False, iteration_only=False):
    problems, warnings, notes = [], [], []
    hist = sorted(glob.glob(os.path.join(run_dir, "history", "iteration_*.json")))
    if not hist:
        problems.append("no history files found in %s/history" % run_dir)
        return problems, warnings, notes

    for path in hist:
        name = os.path.basename(path)
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            problems.append("%s: unreadable (%s)" % (name, e))
            continue

        h = doc.get("handoff")
        if h is None:
            problems.append("%s: NO handoff block. A fresh agent gets only prose from this "
                            "iteration." % name)
            continue

        for f in REQUIRED:
            if f not in h:
                problems.append("%s: handoff missing required field '%s'" % (name, f))

        if h.get("context_clearable") is False:
            problems.append("%s: context_clearable is false -- %s"
                            % (name, h.get("read_this_first", "no reason given")))

        for i, r in enumerate(h.get("durable_rules", [])):
            for f in RULE_FIELDS:
                if f not in r:
                    problems.append("%s: durable_rules[%d] missing '%s'" % (name, i, f))
        for i, c in enumerate(h.get("methodological_cautions", [])):
            for f in CAUTION_FIELDS:
                if f not in c:
                    problems.append("%s: methodological_cautions[%d] missing '%s'" % (name, i, f))
        for i, d in enumerate(h.get("do_not_repeat", [])):
            for f in REPEAT_FIELDS:
                if f not in d:
                    problems.append("%s: do_not_repeat[%d] missing '%s'" % (name, i, f))

        for k, v in h.get("parameters", {}).items():
            if not isinstance(v, dict) or "value" not in v:
                problems.append("%s: parameters['%s'] must be an object with a 'value'"
                                % (name, k))

        # counts-not-content smell inside content_carried_forward
        for k, v in h.get("content_carried_forward", {}).items():
            if isinstance(v, int) and COUNT_SMELL.search(k):
                warnings.append("%s: content_carried_forward['%s'] is the integer %d -- store "
                                "the content, not its length" % (name, k, v))

        # a node that changed nothing downstream should say so, not stay silent
        if not h.get("durable_rules") and not h.get("content_carried_forward"):
            notes.append("%s: handoff carries no rules and no content -- confirm that istrue "
                         "rather than forgotten" % name)

    if iteration_only:
        notes.append("--iteration-only: handoff blocks checked; state and resume contract "
                     "NOT checked. Run without this flag before clearing context.")
        return problems, warnings, notes

    # ---- state completeness for whatever runs next
    sp = os.path.join(run_dir, "state", "current.json")
    if not os.path.exists(sp):
        problems.append("state/current.json missing")
        return problems, warnings, notes
    st = json.load(open(sp, encoding="utf-8"))
    cur = st.get("current", {})

    contract = cur.get("next_iteration_contract") or {}
    nxt = contract.get("node")
    if not nxt and cur.get("run_complete"):
        # A finished run has no next node by design. That is a terminal state, not a gap --
        # but only when it is stated explicitly, so a run that merely stopped being updated
        # still trips the block below.
        notes.append("run is COMPLETE (state.current.run_complete) -- no next node expected. "
                     "Final model: %s" % cur.get("final_model_path", "not recorded"))
        if not cur.get("final_model_path"):
            problems.append("run_complete is true but no final_model_path is recorded")
    elif not nxt:
        # Clearing context without this is the failure the contract exists to prevent: the
        # fresh agent has no resume point at all. Blocking, not advisory.
        problems.append("state.current.next_iteration_contract has no 'node' and the run is not "
                        "marked complete -- a fresh agent has no resume point and would have to "
                        "re-derive routing from history prose")
    else:
        for f in CONTRACT_FIELDS:
            if f not in contract:
                problems.append("next_iteration_contract missing '%s' -- see "
                                "conventions/handoff_contract.md" % f)

        # The contract declares what the next node needs; NODE_REQUIREMENTS is the floor.
        declared = contract.get("required_state_fields") or []
        floor = NODE_REQUIREMENTS.get(nxt, [])
        under = [f for f in floor if f not in declared]
        if declared and under:
            warnings.append("next_iteration_contract.required_state_fields for '%s' omits %s, "
                            "which NODE_REQUIREMENTS treats as mandatory -- one of the two is "
                            "out of date" % (nxt, ", ".join(under)))

        needed = list(declared) + [f for f in floor if f not in declared]
        missing = [f for f in needed if f not in cur]
        if missing:
            problems.append("next node '%s' needs state fields not present: %s"
                            % (nxt, ", ".join(missing)))
        else:
            notes.append("next node '%s': all required state fields present" % nxt)

        if not (contract.get("context") or {}):
            warnings.append("next_iteration_contract.context is empty -- the resuming agent will "
                            "fall back to reading the full state snapshot")

    # ---- checks added with the 2026-09-04 conventions, skipped for runs that predate them
    conventions = (cur.get("run_config") or {}).get("conventions_version")
    new_rules = bool(conventions) and str(conventions) >= NEW_RULES_FROM
    if not new_rules:
        notes.append("run_config.conventions_version is %s -- the 2026-09-04 convention checks "
                     "(baseline floor, stop justification, blocking-clarification halt, one "
                     "model per horizon) are skipped for this run"
                     % (repr(conventions) if conventions else "absent"))

    # ---- a blocking clarification halts the run: nothing runs, nothing clears
    if new_rules and cur.get("run_status") == "halted_awaiting_answer":
        problems.append("state.current.run_status is 'halted_awaiting_answer' -- a blocking "
                        "clarification is outstanding. No node may run and context may not be "
                        "cleared until the user answers it")

    pending = cur.get("pending_clarifications") or []
    blocking = [c for c in pending if c.get("blocking")]
    if new_rules and blocking and cur.get("run_status") != "halted_awaiting_answer":
        problems.append("%d blocking clarification(s) in pending_clarifications (%s) but "
                        "run_status is '%s' -- a blocking question must halt the run, not be "
                        "carried past it"
                        % (len(blocking), ", ".join(str(c.get("topic", "?")) for c in blocking),
                           cur.get("run_status", "unset")))
    for i, c in enumerate(pending if new_rules else []):
        missing_c = [f for f in CLARIFICATION_FIELDS if not c.get(f)]
        if missing_c:
            warnings.append("pending_clarifications[%d] ('%s') has no %s -- a question raised "
                            "without the answer its own node would give hands the user that "
                            "node's reasoning to do"
                            % (i, c.get("topic", "?"), " or ".join(missing_c)))

    # ---- the floor must exist before any learned model does
    task = (cur.get("task_type") or "").lower()
    is_ts = any(t in task for t in TIME_SERIES_TASKS)
    floor_recorded = cur.get("baseline_floor")
    if new_rules and cur.get("latest_model_path") and not floor_recorded:
        problems.append("a trained model exists but state.current.baseline_floor does not -- the "
                        "run's first Training call is mode='baseline' and establishes the floor "
                        "every later iteration is measured against")
    elif new_rules and nxt == "training" and not floor_recorded:
        notes.append("no baseline_floor yet -- this Training call should be mode='baseline'")

    # ---- multi-horizon runs owe one model per horizon
    fh = cur.get("forecast_horizons") or {}
    fh_values = fh.get("values") if isinstance(fh, dict) else fh
    if new_rules and fh_values and len(fh_values) > 1:
        by_h = cur.get("models_by_horizon") or {}
        if cur.get("latest_model_path") and not by_h:
            problems.append("forecast_horizons has %d values (%s) but state records no "
                            "models_by_horizon -- a multi-horizon forecast trains one model per "
                            "horizon, and a single latest_model_path cannot say which horizon it "
                            "answers for" % (len(fh_values), fh_values))
        else:
            unbuilt = [h for h in fh_values if str(h) not in {str(k) for k in by_h}]
            if by_h and unbuilt:
                warnings.append("horizons with no model yet: %s -- the run is not comparable "
                                "across horizons until every one has its own"
                                % ", ".join(str(h) for h in unbuilt))

    # ---- a finished run has to say what stopped it
    if new_rules and cur.get("run_complete"):
        sj = cur.get("stop_justification") or {}
        if not sj:
            problems.append("run_complete is true but no stop_justification is recorded -- the "
                            "pre-stop question (what was the binding constraint, what did each "
                            "axis buy, what was left untried and why) has not been answered")
        elif not sj.get("binding_constraint"):
            problems.append("stop_justification has no binding_constraint -- 'gains got smaller' "
                            "is the observation that prompted the question, not an answer to it")
        if is_ts:
            tried = cur.get("ts_approaches_tried") or []
            missing_ts = [a for a in TS_APPROACHES_REQUIRED if a not in tried]
            if missing_ts:
                warnings.append("time-series run completed without fitting %s -- both "
                                "non-automatic approaches are required before a stop"
                                % ", ".join(missing_ts))

    # ---- the Planner's own decision trail must survive a clear too
    unexplained = [h.get("iteration") for h in st.get("history", [])
                   if not h.get("why_this_node")]
    if unexplained:
        warnings.append("%d history %s no 'why_this_node' (iterations: %s) -- after a "
                        "clear the Planner can see what ran but not why, and may re-consider "
                        "options already rejected"
                        % (len(unexplained),
                           "entry carries" if len(unexplained) == 1 else "entries carry",
                           ", ".join(str(i) for i in unexplained[:8])
                           + (", ..." if len(unexplained) > 8 else "")))

    for k, v in cur.items():
        if isinstance(v, int) and COUNT_SMELL.search(k) and k != "n_features":
            warnings.append("state.current['%s'] is the integer %d -- if a downstream node needs "
                            "the content, this is not enough" % (k, v))

    frozen = cur.get("frozen_hyperparameters")
    if nxt in ("training", "feature_engineering") and not frozen:
        warnings.append("no frozen_hyperparameters in state -- a fresh Training call may re-tune "
                        "and invalidate the comparison it was asked to make")

    return problems, warnings, notes


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    brief = "--brief" in sys.argv
    iteration_only = "--iteration-only" in sys.argv
    if not args:
        print("usage: python tools/handoff_check.py runs/run_<id> "
              "[--iteration-only] [--brief]")
        return 2
    run_dir = args[0]
    problems, warnings, notes = check_run(run_dir, brief, iteration_only)

    print("=" * 68)
    print("HANDOFF CHECK  %s" % run_dir)
    print("=" * 68)
    if notes and not brief:
        for n in notes:
            print("  ok      %s" % n)
    for w in warnings:
        print("  WARN    %s" % w)
    for p in problems:
        print("  BLOCK   %s" % p)
    print("-" * 68)
    if iteration_only:
        if problems:
            print("ITERATION INCOMPLETE -- %d blocking, %d warnings"
                  % (len(problems), len(warnings)))
            return 1
        print("ITERATION OK -- handoff blocks conformant (%d warning(s)). "
              "Not a clear-safety verdict." % len(warnings))
        return 0
    if problems:
        print("NOT SAFE TO CLEAR -- %d blocking, %d warnings" % (len(problems), len(warnings)))
        return 1
    if warnings:
        print("SAFE TO CLEAR, with %d warning(s) worth reading first" % len(warnings))
        return 0
    print("SAFE TO CLEAR -- every iteration is self-sufficient for the next agent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
