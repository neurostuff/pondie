#!/usr/bin/env python3
"""What fraction of a record set can answer the questions a meta-analysis actually asks?

Validator errors count defects; this counts capability. A record can be free of validator
errors and still be unqueryable -- a cell whose term reaches no ModelTerm breaks the
traversal a design question needs, and an analysis with no Measure cannot be restricted to
one modality however well-formed it is.

Each check is one join a real query performs, reported as the share of analyses (or
records) that can complete it.

    python audit_queryability.py 'data/runs/schiz/final2/*.extraction.json'
"""
from __future__ import annotations
import argparse
import glob as globlib
import json
from collections import Counter
from pathlib import Path


def val(node):
    return node.get("value") if isinstance(node, dict) and "value" in node else node


def ids(body, key, holder=None):
    source = (body.get(holder) or {}) if holder else body
    return {val(e.get("local_id")) for e in (source.get(key) or [])
            if isinstance(e, dict) and val(e.get("local_id"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", nargs="+")
    args = ap.parse_args()

    paths = sorted({p for pattern in args.records for p in globlib.glob(pattern)})
    paths = [Path(p) for p in paths if not p.endswith(".raw.json")]

    c = Counter()
    for path in paths:
        body = json.loads(path.read_text())
        body = body.get("study") or body
        c["records"] += 1

        tables = ids(body, "tables")
        measures = ids(body, "measures")
        groups = ids(body, "groups")
        arms = ids(body, "arms", "design")
        timepoints = ids(body, "timepoints", "design")
        regions = ids(body, "regions")
        # Conditions are nested under the Task that defines them, not a top-level list, so
        # a pool built from the record's own top level silently omits them -- and a level
        # naming a task condition then looks unresolvable when it is not.
        conditions = {val(cnd.get("local_id"))
                      for t in (body.get("tasks") or []) if isinstance(t, dict)
                      for cnd in (t.get("conditions") or []) if isinstance(cnd, dict)
                      if val(cnd.get("local_id"))}
        models = {val(m.get("local_id")): m for m in (body.get("model_estimations") or [])
                  if isinstance(m, dict)}
        terms = {}
        # term local_id -> (owning model, {folded level string: the FactorLevel node}).
        # A Cell holds a term reference and a bare level STRING; the references that make
        # the level queryable -- groups, arms, timepoints -- hang off the FactorLevel on
        # the term, reached by matching that string.
        levels_of = {}
        for m in models.values():
            for t in (m.get("terms") or []):
                if not (isinstance(t, dict) and val(t.get("local_id"))):
                    continue
                tid = val(t.get("local_id"))
                terms[tid] = val(m.get("local_id"))
                levels_of[tid] = {
                    str(val(fl.get("level")) or "").strip().lower(): fl
                    for fl in (t.get("levels") or []) if isinstance(fl, dict)}

        if any(val(g.get("species")) for g in (body.get("groups") or [])
               if isinstance(g, dict)):
            c["records that state a species"] += 1

        for a in body.get("analyses") or []:
            if not isinstance(a, dict):
                continue
            c["analyses"] += 1

            if val(a.get("source_table_analysis")):
                c["  reach their coordinates (parse key)"] += 1
            cited = [t for t in (val(a.get("tables")) or []) if isinstance(t, str)]
            if cited and all(t in tables for t in cited):
                c["  reach every Table they cite"] += 1
            if val(a.get("measure")) in measures:
                c["  reach a Measure"] += 1
            if val(a.get("coordinate_space")):
                c["  state a coordinate space"] += 1

            cells = (a.get("effect") or {}).get("cells") or []
            if not cells:
                continue
            c["analyses with cells"] += 1
            if all(val(x.get("direction")) for x in cells):
                c["  every cell carries a direction"] += 1
            if all(val(x.get("term")) in terms for x in cells):
                c["  every cell's term reaches a ModelTerm"] += 1
            if all(terms.get(val(x.get("term"))) == val(a.get("model_estimation"))
                   for x in cells):
                c["  every cell's term is in THIS analysis's model"] += 1

            # a level that names a cohort or an arm is what a cross-paper query selects on
            reached = 0
            for x in cells:
                node = levels_of.get(val(x.get("term")), {}).get(
                    str(val(x.get("level")) or "").strip().lower()) or {}
                refs = []
                for key, pool in (("groups", groups), ("arms", arms),
                                  ("timepoints", timepoints),
                                  ("conditions", conditions), ("regions", regions)):
                    refs += [r for r in (val(node.get(key)) or []) if r in pool]
                reached += bool(refs)
            if reached == len(cells):
                c["  every cell's level resolves to a named entity"] += 1
            elif reached:
                c["  SOME cells' levels resolve"] += 1

    an, cellsn, rec = c["analyses"], c["analyses with cells"], c["records"]
    print(f"{rec} records, {an} analyses, {cellsn} of them with cells\n")
    print(f"  {c['records that state a species']:5d}/{rec:<5d} "
          f"{c['records that state a species']/rec:6.1%}  records that state a species")
    for label in ("  reach their coordinates (parse key)", "  reach every Table they cite",
                  "  reach a Measure", "  state a coordinate space"):
        print(f"  {c[label]:5d}/{an:<5d} {c[label]/an:6.1%}  analyses{label}")
    for label in ("  every cell carries a direction",
                  "  every cell's term reaches a ModelTerm",
                  "  every cell's term is in THIS analysis's model",
                  "  every cell's level resolves to a named entity",
                  "  SOME cells' levels resolve"):
        print(f"  {c[label]:5d}/{cellsn:<5d} {c[label]/cellsn:6.1%}  analyses where{label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
