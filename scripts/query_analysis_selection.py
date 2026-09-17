#!/usr/bin/env python3
"""Can a query select the ANALYSIS a meta-analysis pools, not just the paper?

    python scripts/query_analysis_selection.py --records '<dir>/*/*.extraction.json' \
        --gold analysis_gold.csv

Screening picks papers; a coordinate meta-analysis pools *contrasts*, and the benchmark names
them: "non-PTSD > PTSD", "bvFTD < HC (all & by modality)", "users < non-users". The gold for
that is `nimads/<project>/merged/`, whose annotation carries one boolean per key per analysis.

Each key below is a directional between-group contrast plus, sometimes, a modality. Both are
things the record encodes: `Effect.cells` with a `Cell.direction` on a `FactorLevel` that
reaches a `Group`, and `Analysis.measure` -> `Measure.family`. So the question is not whether
the schema can say it -- it can -- but whether the records do.

Counted three ways per paper, because "found the contrast" and "found the right number of
contrasts" are different claims and only the second is what a pooled map needs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from pondie.formats import values

#: manual key -> (a regex for the patient cohort, a regex for the measure family or None).
#: The direction is always the same: the meta-analyses pool the contrast in which the
#: patient or user group is LOWER, which is `Cell.direction: negative` on that group's level.
KEYS = {
    ("vbm_of_ptsd", "nonptsdgtptsd_merged"): (r"PTSD|post.?traumatic", None),
    ("dementia", "decrease"): (r"bvFTD|frontotemporal|FTD", None),
    ("dementia", "structural"): (r"bvFTD|frontotemporal|FTD", r"morphometry|structural"),
    ("dementia", "functional"): (r"bvFTD|frontotemporal|FTD", r"bold|functional|perfusion|molecular"),
    ("vbm_of_substance_use", "all_drug_classes"): (
        r"alcohol|nicotine|tobacco|smok|cocaine|cannabis|opioid|heroin|methamphetamine|"
        r"substance|depend|abuse|addict", None),
}


def read(node: object) -> object:
    if not isinstance(node, dict) or "extraction_status" not in node:
        return node
    return node.get("value") if node.get("extraction_status") == "extracted" else None


def texts(node: object) -> list[str]:
    value = read(node)
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(x) for x in items if str(x).strip()]


def ids(node: object) -> list[str]:
    value = read(node)
    items = value if isinstance(value, list) else [value]
    return [x for x in items if isinstance(x, str) and x]


def selected_analyses(record: dict, cohort: str, family: str | None) -> list[str]:
    """The analyses whose cells encode "cohort lower than the other group"."""
    groups = {
        g["local_id"]: " ".join(texts(g.get("name")) + texts(g.get("medical_condition")))
        for g in record.get("groups") or []
        if isinstance(g, dict) and isinstance(g.get("local_id"), str)
    }
    patient = {gid for gid, blurb in groups.items() if re.search(cohort, blurb, re.I)}
    if not patient:
        return []
    levels: dict[str, set[str]] = {}
    for model in record.get("model_estimations") or []:
        if not isinstance(model, dict):
            continue
        for term in model.get("terms") or []:
            if not isinstance(term, dict):
                continue
            for level in term.get("levels") or []:
                if isinstance(level, dict):
                    name = read(level.get("level"))
                    if isinstance(name, str):
                        levels.setdefault(f"{term.get('local_id')}|{name}", set()).update(
                            ids(level.get("groups")))
    measures = {
        m["local_id"]: " ".join(texts(m.get("family")) + texts(m.get("type")))
        for m in record.get("measures") or []
        if isinstance(m, dict) and isinstance(m.get("local_id"), str)
    }
    out = []
    for analysis in record.get("analyses") or []:
        if not isinstance(analysis, dict):
            continue
        if family is not None:
            blurb = measures.get(str(analysis.get("measure")) or "", "")
            if not re.search(family, blurb, re.I):
                continue
        effect = analysis.get("effect")
        if not isinstance(effect, dict):
            continue
        lower, higher = False, False
        for cell in effect.get("cells") or []:
            if not isinstance(cell, dict):
                continue
            direction = str(read(cell.get("direction")) or "").lower()
            name = read(cell.get("level"))
            reached = levels.get(f"{cell.get('term')}|{name}", set()) if isinstance(name, str) else set()
            if not reached:
                continue
            if direction == "negative" and reached & patient:
                lower = True
            elif direction == "positive" and reached - patient:
                higher = True
        if lower and higher:
            out.append(str(analysis.get("local_id")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--gold", required=True, type=Path)
    args = ap.parse_args()

    records: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(args.records)):
        body = json.loads(Path(path).read_text())
        records[Path(path).parent.name][Path(path).name.split(".")[0]] = body.get("study") or body

    gold: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with args.gold.open() as fh:
        for row in csv.DictReader(fh):
            gold[(row["project"], row["manual_key"])].append(row)

    print(f"{'project / key':46} {'papers':>7} {'auto':>20} {'query':>20}")
    print(f"{'':46} {'':>7} {'>=1':>6} {'=gold':>6} {'foci':>6} "
          f"{'>=1':>6} {'=gold':>6} {'foci':>6}")
    for (project, manual), rows in sorted(gold.items()):
        spec = KEYS.get((project, manual))
        if spec is None:
            continue
        cohort, family = spec
        here = records.get(project) or {}
        tally = Counter()
        n = 0
        for row in rows:
            pmid = row["pmid"]
            body = here.get(pmid)
            if body is None:
                continue
            n += 1
            want, want_foci = int(row["gold_analyses"]), int(row["gold_foci"])
            auto, auto_foci = int(row["auto_analyses"]), int(row["auto_foci"])
            picked = selected_analyses(body, cohort, family)
            if auto >= 1:
                tally["auto found"] += 1
            if auto == want:
                tally["auto count"] += 1
            if auto_foci == want_foci:
                tally["auto foci"] += 1
            if picked:
                tally["query found"] += 1
            if len(picked) == want:
                tally["query count"] += 1
        if not n:
            continue
        print(f"{project + ' / ' + manual:46} {n:7d} "
              f"{tally['auto found']/n:5.0%} {tally['auto count']/n:6.0%} "
              f"{tally['auto foci']/n:6.0%} "
              f"{tally['query found']/n:5.0%} {tally['query count']/n:6.0%} {'-':>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
