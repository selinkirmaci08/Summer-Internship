"""
state_prune.py -- keep state/current.json a live snapshot rather than a ledger.

`current` is read on every resume, so every dead key in it is paid for again on every
iteration. This reports keys that look superseded -- iteration-stamped values a later
iteration has replaced -- and, with --apply, moves them to a sibling `archive` root key.
Nothing is deleted: reports and the audit trail still reference these values.

Duplicate-looking key FAMILIES (three_model_comparison / four_model_comparison /
model_family_comparison) are reported but never moved automatically. Which one is
authoritative is a judgement call, and guessing it wrong is worse than a large file.

Usage:
    python tools/state_prune.py runs/run_<id>
    python tools/state_prune.py runs/run_<id> --apply

Exit code 0 means nothing to do (or --apply succeeded); 1 means there are keys worth moving.
"""
import os
import re
import sys
import json
import shutil

try:
    from handoff_check import NODE_REQUIREMENTS
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from handoff_check import NODE_REQUIREMENTS

# Keys any node may need to run. Never moved, whatever they look like.
PROTECTED = set()
for _fields in NODE_REQUIREMENTS.values():
    PROTECTED.update(_fields)
PROTECTED.update([
    "goal", "dataset_path", "target_column", "task_type", "stop_condition_met",
    "pending_clarifications", "next_iteration_contract", "run_config", "run_complete",
    "latest_metrics", "latest_model_path", "latest_stop_recommendation",
    "prior_hypotheses_tried", "frozen_hyperparameters", "confirmed_decisions",
    "final_model_path", "handoff_contract", "handoff_validator",
])

# Values stamped with the iteration that produced them. The per-iteration record already
# lives in history/ and in that iteration's handoff block.
ITERATION_STAMPED = [
    re.compile(r"^iteration_\d+[a-z]?_"),
    re.compile(r"_iteration_\d+[a-z]?$"),
    re.compile(r"_\d{3}_vs_\d{3}$"),
    re.compile(r"^pre_registered_.*_\d+$"),
]

# Append-only records. They grow without bound and describe what HAPPENED, not what IS.
# The decisions they led to live in `confirmed_decisions`; the narrative lives in history/.
LOG_KEYS = ["resolved_clarifications"]

NUMBER_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def is_iteration_stamped(key):
    return any(p.search(key) for p in ITERATION_STAMPED)


def stem(key):
    """Normalise a key so accreted variants of one fact collapse to the same stem."""
    k = key.lower()
    k = re.sub(r"\d+", "", k)
    for w in NUMBER_WORDS:
        k = re.sub(r"(^|_)%s(_|$)" % w, "_", k)
    k = re.sub(r"(^|_)(now|latest|current|final|all)(_|$)", "_", k)
    return re.sub(r"_+", "_", k).strip("_")


def analyse(cur):
    stamped, families = [], {}
    for k, v in cur.items():
        size = len(json.dumps(v, default=str))
        if k in PROTECTED:
            continue
        if is_iteration_stamped(k) or k in LOG_KEYS:
            stamped.append((k, size))
        families.setdefault(stem(k), []).append((k, size))
    dupes = {s: ks for s, ks in families.items() if len(ks) > 1}
    return sorted(stamped, key=lambda x: -x[1]), dupes


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply_it = "--apply" in sys.argv
    if not args:
        print("usage: python tools/state_prune.py runs/run_<id> [--apply]")
        return 2

    path = os.path.join(args[0], "state", "current.json")
    if not os.path.exists(path):
        print("no state file at %s" % path)
        return 2

    st = json.load(open(path, encoding="utf-8"))
    cur = st.get("current", {})
    total = len(json.dumps(st))
    stamped, dupes = analyse(cur)

    print("=" * 68)
    print("STATE PRUNE  %s" % path)
    print("=" * 68)
    print("  current.json is %.1f KB (~%d tokens), read on every resume"
          % (total / 1024.0, total // 4))
    print("  %d keys in `current`" % len(cur))
    print("-" * 68)

    if stamped:
        moved_bytes = sum(s for _, s in stamped)
        print("  SUPERSEDED -- iteration-stamped or append-only log (%d keys, %.1f KB):"
              % (len(stamped), moved_bytes / 1024.0))
        for k, size in stamped:
            print("      %-52s %6d B" % (k, size))
    else:
        print("  no superseded keys found")

    if dupes:
        print("-" * 68)
        print("  DUPLICATE FAMILIES -- one fact, several keys. Not moved automatically;")
        print("  decide which is authoritative, fold the rest into it:")
        for s, ks in sorted(dupes.items()):
            print("      %s:" % s)
            for k, size in sorted(ks, key=lambda x: -x[1]):
                print("          %-48s %6d B" % (k, size))

    print("-" * 68)
    if not stamped:
        print("nothing to move")
        return 0

    if not apply_it:
        print("run again with --apply to move %d key(s) into `archive`" % len(stamped))
        return 1

    shutil.copyfile(path, path + ".bak")
    archive = st.setdefault("archive", {})
    for k, _ in stamped:
        archive[k] = cur.pop(k)
    json.dump(st, open(path, "w", encoding="utf-8"), indent=2, default=str)
    new_total = len(json.dumps(st))
    new_cur = len(json.dumps(st.get("current", {})))
    print("moved %d key(s) to `archive`. backup: %s" % (len(stamped), path + ".bak"))
    print("  current: %.1f KB -> %.1f KB (~%d tokens off every resume read)"
          % (total / 1024.0, new_cur / 1024.0, (total - new_cur) // 4))
    print("  file total unchanged at %.1f KB -- nothing deleted" % (new_total / 1024.0,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
