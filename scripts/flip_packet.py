"""Assemble everything needed to explain one arm-vs-arm disagreement.

Per discordant paper: the gold label, both arms' decision, reason and applied criteria, the
record's own account of itself (what it carried, what it marked unsupported, what it said the
paper never reported), and the rendered text each arm actually read. The point is to separate
three things that look alike in a metrics table -- the record never carried the fact, the
record carried it wrongly, or the record carried it and the screener still erred.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

EXP = Path("/data/james/pondie-vs-fulltext")
REPO = EXP / "repos" / "autonima-results"
INCLUDE = {"included_fulltext", "included"}


def screening(run_dir: Path) -> dict[str, dict]:
    data = json.loads((run_dir / "outputs" / "fulltext_screening_results.json").read_text())
    return {str(r["study_id"]): r for r in data.get("screening_results", [])}


def gold_pmids(project: str) -> set[str]:
    mapping = json.loads((REPO / "projects" / project / "nmb_mappings.json").read_text())
    meta = str(mapping.get("meta_pmid"))
    with (REPO.parent / "neurometabench" / "data" / "included_studies.csv").open() as fh:
        return {r["study_pmid"].strip() for r in csv.DictReader(fh)
                if r["meta_pmid"] == meta}


def record_summary(pmid: str, run: str) -> dict:
    path = EXP / "pondie-data" / "runs" / run / "records" / f"{pmid}.extraction.json"
    if not path.is_file():
        return {}
    rec = json.loads(path.read_text())
    out: dict = {"n": {}, "not_found": [], "not_reported": []}
    for key in ("analyses", "groups", "tables", "inference_settings", "model_estimations",
                "measures", "regions", "acquisitions"):
        val = rec.get(key)
        if isinstance(val, list):
            out["n"][key] = len(val)

    def walk(node, path_str=""):
        if isinstance(node, dict):
            if "extraction_status" in node:
                ev = (node.get("evidence") or {}).get("status")
                if node.get("extraction_status") == "not_reported":
                    out["not_reported"].append(path_str)
                elif ev == "not_found":
                    out["not_found"].append(path_str)
                return
            for k, v in node.items():
                walk(v, f"{path_str}.{k}" if path_str else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path_str}[{i}]")

    walk(rec)
    # the fields this project's criteria actually turn on
    out["key_fields"] = {}
    for analysis in (rec.get("analyses") or []):
        if not isinstance(analysis, dict):
            continue
        local = analysis.get("local_id", "?")
        def leaf(node, *keys):
            cur = node
            for k in keys:
                cur = (cur or {}).get(k) if isinstance(cur, dict) else None
            if isinstance(cur, dict) and "extraction_status" in cur:
                return cur.get("value"), (cur.get("evidence") or {}).get("status")
            return None, None
        name, _ = leaf(analysis, "name")
        scope, scope_ev = leaf(analysis, "spatial_scope")
        out["key_fields"][local] = {
            "name": name, "spatial_scope": scope, "spatial_scope_evidence": scope_ev,
            "n_groups": len(analysis.get("groups") or []),
            "n_cells": len(((analysis.get("effect") or {}).get("cells")) or []),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--text-arm", required=True)
    ap.add_argument("--record-arm", required=True)
    ap.add_argument("--run", default="pondie-907")
    ap.add_argument("--pmids", nargs="*")
    args = ap.parse_args()

    gold = gold_pmids(args.project)
    text = screening(REPO / "projects" / args.project / args.text_arm)
    rec = screening(REPO / "projects" / args.project / args.record_arm)
    both = set(text) & set(rec)
    flips = args.pmids or sorted(
        p for p in both if (text[p].get("decision") in INCLUDE) != (rec[p].get("decision") in INCLUDE))

    criteria = json.loads(
        (REPO / "projects" / args.project / args.record_arm / "outputs" / "criteria_mapping.json").read_text())

    for pmid in flips:
        t, r = text[pmid], rec[pmid]
        is_gold = pmid in gold
        correct = "record" if (r.get("decision") in INCLUDE) == is_gold else "text"
        print("=" * 100)
        print(f"PMID {pmid}   gold={'IN the meta-analysis' if is_gold else 'not in it'}"
              f"   -> {correct.upper()} arm matches gold")
        for label, row in ((args.text_arm, t), (args.record_arm, r)):
            print(f"\n  [{label}] {row.get('decision')}  conf={row.get('confidence')}")
            print(f"    inc_applied={row.get('inclusion_criteria_applied')} "
                  f"exc_applied={row.get('exclusion_criteria_applied')}")
            reason = re.sub(r"\s+", " ", str(row.get("reason") or ""))
            print(f"    reason: {reason[:700]}")
        summary = record_summary(pmid, args.run)
        if summary:
            print(f"\n  [record] entities={summary['n']}")
            print(f"    not_found fields: {len(summary['not_found'])}   "
                  f"not_reported: {len(summary['not_reported'])}")
            for local, info in summary["key_fields"].items():
                print(f"    {local}: scope={info['spatial_scope']!r} "
                      f"(ev={info['spatial_scope_evidence']}) groups={info['n_groups']} "
                      f"cells={info['n_cells']} name={str(info['name'])[:60]!r}")
        else:
            print("\n  [record] NONE - this paper had no record")
        print()
    print("criteria vocabulary:")
    for k, v in (criteria.get("fulltext") or criteria).items():
        if isinstance(v, str):
            print(f"  {k}: {v[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
