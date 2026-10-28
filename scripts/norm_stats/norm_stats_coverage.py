#!/usr/bin/env python3
"""Coverage stamp for a norm-stats file: which corpora went into the fit.

WHY THIS EXISTS
    `num_episodes` is the only trace a pooled stats file keeps of what it
    covers, and it does not identify the episode set. A run once trained on 30
    YAM corpora against a fit that held 23 of them (9,567 of 11,525 episodes);
    nothing in the file, the launcher or the training log said so. The merge
    arithmetic was exact -- the bookkeeping was what rotted.

    A merge of per-corpus summaries reproduces a joint fit bit for bit (min/max
    and q01/q99 reduce as elementwise min/max; mean/std close through the law of
    total variance given num_episodes). So the merge is never the problem. What
    a summary cannot say for itself is WHICH episodes it saw, and that is what
    `coverage` records.

SCHEMA
    stats["coverage"] = {"<corpus name>": <episodes>, ...}
    stats["coverage_sha"] = sha256 of the sorted "name:episodes" lines

    A file with no `coverage` key predates this and its extent is unknown; the
    checker says so rather than assuming the file is complete.
"""
from __future__ import annotations

import hashlib
import json
import sys


def coverage_sha(coverage: dict) -> str:
    body = "\n".join(f"{k}:{int(v)}" for k, v in sorted(coverage.items()))
    return hashlib.sha256(body.encode()).hexdigest()


def stamp(stats_path: str, coverage: dict, derived: bool = False) -> dict:
    """Add `coverage` (+ sha) to a stats json by TEXT insertion, in place.

    Deliberately not a json.load/json.dump round-trip: these files are ~7k lines
    of floats, and re-serialising them reflows every line, so a one-key addition
    lands as a whole-file rewrite in review. Insert before the final brace and
    the diff is the handful of lines actually added.
    """
    with open(stats_path) as fh:
        raw = fh.read()
    cov = {k: int(v) for k, v in coverage.items()}
    # The stamp has to account for every episode the fit pooled: a stamp that
    # names fewer episodes than the fit silently drops corpora that WERE in it.
    declared = json.loads(raw).get("num_episodes")
    if declared is not None and sum(cov.values()) != int(declared):
        raise ValueError(
            f"{stats_path}: coverage sums to {sum(cov.values()):,} but the file "
            f"pooled {int(declared):,} episodes. A coverage stamp must account "
            f"for every episode in the fit."
        )
    body = json.dumps(cov, indent=1)
    tail = raw.rstrip()
    if not tail.endswith("}"):
        raise ValueError(f"{stats_path} does not end in a json object")
    if '"coverage"' in raw:
        raise ValueError(f"{stats_path} already carries a coverage stamp")
    add = f',\n "coverage": {body},\n "coverage_sha": "{coverage_sha(cov)}"'
    if derived:
        add += ',\n "coverage_derived": true'
    with open(stats_path, "w") as fh:
        fh.write(tail[:-1].rstrip() + add + "\n}\n")
    return json.loads(open(stats_path).read())


def check(stats_path: str, required: list) -> tuple:
    """(missing, covered_total, note) for the corpora a run actually selects."""
    with open(stats_path) as fh:
        stats = json.load(fh)
    cov = stats.get("coverage")
    if cov is None:
        return (list(required), 0,
                "stats file carries no `coverage` — its extent is unknown")
    missing = [c for c in required if c not in cov]
    note = "coverage reconstructed after the fit" if stats.get("coverage_derived") else ""
    return (missing, sum(int(v) for v in cov.values()), note)


_USAGE = "usage: norm_stats_coverage.py check <stats.json> <corpus> [<corpus> ...]"


def main() -> int:
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
        print(_USAGE)
        return 0
    if len(sys.argv) < 3 or sys.argv[1] != "check":
        print(__doc__)
        print(_USAGE, file=sys.stderr)
        return 2
    stats_path, required = sys.argv[2], sys.argv[3:]
    missing, total, note = check(stats_path, required)
    print(f"[norm-stats] {stats_path}: covers {total:,} episodes"
          + (f" ({note})" if note else ""))
    if not missing:
        print(f"[norm-stats] all {len(required)} selected corpora are in the fit ✓")
        return 0
    print(f"[norm-stats] {len(missing)} of {len(required)} selected corpora are "
          f"NOT in the fit:", file=sys.stderr)
    for m in missing:
        print(f"    {m}", file=sys.stderr)
    print("[norm-stats] Their actions are normalised against a range fitted "
          "without them.\n"
          "             A partial per-episode fit's q01/q99 range is a SUBSET of "
          "the full fit's —\n"
          "             adding a corpus can only widen it — so under the default "
          "clamp the missing\n"
          "             corpora are clipped MORE than a full fit would clip them, "
          "not less. A\n"
          "             frame-pooled run with action_norm_clamp .inf passes their "
          "out-of-fit\n"
          "             targets beyond ±1 instead of clipping them.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
