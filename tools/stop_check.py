"""
stop_check.py -- is this run allowed to stop yet?

Evaluates the coverage conditions in skills/diagnosis-skill/diagnosis.md ("Coverage
(breadth)") mechanically, from state/current.json and history/, so the gate is a fact
on disk rather than a paragraph weighed against a metric that looks finished.

Usage:
    python tools/stop_check.py runs/run_<id>            # may this run call end_model?
    python tools/stop_check.py runs/run_<id> --brief    # blockers only

Run it at planner Step 1c, BEFORE invoking end_model. Exit code 0 means the coverage
gate is clear and the stop rests on the Planner's own reasoning. Non-zero means at
least one axis has never been tried and the report says which.

Why this exists: run_20260904T140437Z stopped at iteration 8 with eda, preprocess,
feature_engineering and diagnosis each visited exactly once, no hyperparameter search
ever run, and the feature set never revised after iteration 003. Its Diagnosis call
recorded the visit counts that prove condition (a) was violated, then recommended
`stop` anyway. Every number the run reported was arithmetically correct and the
conclusion did not survive the holdout.

A blocker is not a verdict on the model. It names an axis nobody has measured.

On runs that predate a field (older runs carry no `ts_approach`, no
`hyperparameter_search`), the tool reports the axis as unmeasured rather than assuming
it was covered. That is the intended reading: not recorded is not demonstrated. It does
mean this tool is a gate for new runs, not a retrospective grade for old ones.
"""
import glob
import json
import os
import sys

# Node slugs that carry their own iteration in history/. `end_model` is exempt from the
# single-visit rule: it runs once by design, at the end.
EXEMPT_FROM_SINGLE_VISIT = {"end_model"}

# The two non-automatic time-series approaches. An automatic tool (Auto ARIMA, Prophet)
# fills neither slot; the classical slot needs an ARIMA-family fit.
TS_APPROACHES = {"classical_statistical", "ml_as_regression"}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def node_of(filename):
    """iteration_005_training_ml_regression.json -> training_ml_regression"""
    stem = os.path.basename(filename)[:-5]
    parts = stem.split("_", 2)
    return parts[2] if len(parts) > 2 else stem


def base_node(node):
    """training_ml_regression -> training; everything else unchanged."""
    return "training" if node.startswith("training") else node


def check_run(run_dir):
    blockers, notes = [], []

    state_path = os.path.join(run_dir, "state", "current.json")
    if not os.path.exists(state_path):
        return ["no state/current.json at %s" % run_dir], []
    cur = load_json(state_path).get("current", {})

    hist_files = sorted(glob.glob(os.path.join(run_dir, "history", "iteration_*.json")))
    if not hist_files:
        return ["no history/ iterations found -- nothing to check"], []

    histories = []
    for f in hist_files:
        try:
            histories.append((node_of(f), load_json(f)))
        except Exception as exc:
            blockers.append("history file unreadable: %s (%s)" % (os.path.basename(f), exc))

    visits = {}
    for node, _ in histories:
        visits[base_node(node)] = visits.get(base_node(node), 0) + 1
    notes.append("node visits: " + ", ".join("%s=%d" % kv for kv in sorted(visits.items())))

    # (a) every node other than end_model visited more than once
    once = sorted(n for n, c in visits.items() if c == 1 and n not in EXEMPT_FROM_SINGLE_VISIT)
    if once:
        blockers.append("(a) visited only once: %s -- a node run a single time has not been "
                        "revised in light of anything the run learned afterwards" % ", ".join(once))

    # (b) more than one model family fitted. Families are recorded under several different
    # names across the pipeline -- a narrow reader raises a false blocker, and a gate that
    # cries wolf is a gate that gets argued past.
    families = set()
    for _, h in histories:
        for key in ("model_family", "model_class", "baseline_type"):
            v = h.get(key)
            if isinstance(v, str):
                families.add(v)
            elif isinstance(v, dict):
                families.update(str(x) for x in v.values() if x)
        cc = h.get("coverage_check")
        if isinstance(cc, dict):
            families.update(str(x) for x in (cc.get("model_families") or []))
    families |= {str(m.get("model")) for m in (cur.get("champion_per_horizon") or {}).values()
                 if isinstance(m, dict) and m.get("model")}
    families |= {str(m.get("model")) for m in (cur.get("classical_champion_per_h") or {}).values()
                 if isinstance(m, dict) and m.get("model")}
    if isinstance(cur.get("baseline_floor"), dict) and cur.get("baseline_floor"):
        families.add("baseline")
    # per-family metric blocks state writes directly, e.g. arima_val_mae / sarima_val_mae
    for key in cur:
        if key.endswith("_val_mae") and isinstance(cur[key], dict) and cur[key]:
            families.add(key[:-len("_val_mae")])
    families = {f for f in families if f and f not in ("None", "none")}
    if len(families) < 2:
        blockers.append("(b) only %d model family fitted (%s)"
                        % (len(families), ", ".join(sorted(families)) or "none recorded"))
    else:
        notes.append("model families: " + ", ".join(sorted(families)))

    # (c) EVERY ranked hypothesis tested or formally retired. A hypothesis is retired only by a
    # recorded outcome, by an existing measurement cited, or by named infeasibility -- never by an
    # estimate of its expected gain. `what_was_left_untried` is therefore evidence AGAINST, not for.
    proposed, retired = [], set()
    for node, h in histories:
        for hyp in (h.get("ranked_hypotheses") or h.get("hypotheses") or []):
            if isinstance(hyp, dict):
                label = hyp.get("statement") or hyp.get("hypothesis") or hyp.get("name")
                if label:
                    proposed.append(str(label))
        for done in (h.get("hypotheses_tested") or []):
            if isinstance(done, dict):
                lab = done.get("statement") or done.get("hypothesis")
                if lab and (done.get("outcome") or done.get("result")):
                    retired.add(str(lab))
    for h_ in (cur.get("hypotheses_summary") or []):
        if isinstance(h_, dict) and h_.get("statement") and h_.get("result"):
            retired.add(str(h_["statement"]))
    for h_ in (cur.get("hypotheses_retired") or []):
        if isinstance(h_, dict) and h_.get("statement") and h_.get("retired_by") in (
                "tested", "answered_by_existing_measurement", "infeasible"):
            retired.add(str(h_["statement"]))

    def norm(x):
        return "".join(ch for ch in str(x).lower() if ch.isalnum())[:60]

    retired_n = {norm(r) for r in retired}
    untested = [p for p in proposed if norm(p) not in retired_n]
    if untested:
        blockers.append("(c) %d ranked hypothesis/es never tested, first: %r"
                        % (len(untested), untested[0][:90]))

    # An explicit "left untried" list is a declined hypothesis, which this gate does not allow.
    left = (cur.get("stop_justification") or {}).get("what_was_left_untried") or []
    if left:
        blockers.append("(c2) %d item(s) recorded in stop_justification.what_was_left_untried -- a "
                        "hypothesis is retired by a measurement, not by declining it. First: %r"
                        % (len(left), str(left[0])[:90]))

    # (d) an external reference has been run
    ref_markers = ("automl_b1_test_mae", "automl_baseline2", "external_reference",
                   "beats_automl_baseline2")
    has_ref = any(k in cur for k in ref_markers) or any(
        any(k in h for k in ref_markers) for _, h in histories)
    if not has_ref:
        blockers.append("(d) no cheap external reference has ever been run "
                        "(automl-baseline2 over the run's own features)")

    # Field-name drift: the contract says `stop_recommendation`, at least one run wrote
    # `stop_continue_verdict`. Read both, and say so when only the non-contract name is present.
    verdicts = [h.get("stop_recommendation") or h.get("stop_continue_verdict")
                for _, h in histories if h.get("stop_recommendation") or h.get("stop_continue_verdict")]
    if verdicts:
        notes.append("latest diagnosis verdict: %s" % verdicts[-1])
    if any("stop_continue_verdict" in h and "stop_recommendation" not in h for _, h in histories):
        notes.append("note: diagnosis wrote `stop_continue_verdict`; contracts/diagnosis.json "
                     "specifies `stop_recommendation` -- align the field name")

    # (e) at least one hyperparameter search
    searched = False
    for node, h in histories:
        hs = h.get("hyperparameter_search")
        if isinstance(hs, dict) and hs.get("method"):
            searched = True
        elif isinstance(hs, dict) and any(hs.values()):
            searched = True
    if not searched:
        blockers.append("(e) no hyperparameter search on any Training call "
                        "-- hyperparameter_search is null throughout. An axis closed by "
                        "argument rather than measurement is untried; a fixed baseline does "
                        "not move when a model is tuned")

    # (f) feature_engineering revised at least once
    if visits.get("feature_engineering", 0) <= 1:
        blockers.append("(f) feature_engineering ran %d time(s) -- no feature rationale has "
                        "been tested against a Training result"
                        % visits.get("feature_engineering", 0))

    # (g) both non-automatic time-series approaches, when the task is a forecast
    if str(cur.get("task_type", "")).startswith("time_series"):
        tried_ts = set(cur.get("ts_approaches_tried") or [])
        for _, h in histories:
            if h.get("ts_approach"):
                tried_ts.add(str(h["ts_approach"]))
        missing = TS_APPROACHES - tried_ts
        if missing:
            blockers.append("(g) time-series approach never fitted: %s" % ", ".join(sorted(missing)))
        else:
            notes.append("ts approaches: " + ", ".join(sorted(tried_ts)))

    return blockers, notes


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    brief = "--brief" in sys.argv
    if not args:
        print("usage: python tools/stop_check.py runs/run_<id> [--brief]")
        return 2
    run_dir = args[0]
    blockers, notes = check_run(run_dir)

    print("=" * 68)
    print("STOP CHECK  %s" % run_dir)
    print("=" * 68)
    if notes and not brief:
        for n in notes:
            print("  ok      %s" % n)
    for b in blockers:
        print("  BLOCK   %s" % b)
    print("-" * 68)
    if blockers:
        print("STOP NOT AVAILABLE -- %d axis/axes never measured" % len(blockers))
        print("Fix by running the named node, not by arguing the metric looks finished.")
        return 1
    print("COVERAGE CLEAR -- the stop now rests on the Planner's own reasoning, "
          "which this tool does not check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
