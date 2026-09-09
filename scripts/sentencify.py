"""Turn every extracted leaf in a record into a standalone claim a fact-checker can judge.

`evidence.status` already says whether a supporting sentence was *located*. It does not say
whether that sentence *supports the value* -- the two come apart, and pondie's own hand
audit of 70 field instances found the automated agreement number "wrong about what it
claims to measure". A claim plus the document is the question an entailment model can
actually answer, so this renders each leaf as one.

Each claim carries the dotted path it came from, so a verdict maps back to a field and a
re-extraction can be scoped to the fields that failed rather than the whole paper.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXP = Path("/data/james/pondie-vs-fulltext")

#: Entity lists whose members get named in the subject of the claim, so "the analysis
#: `ana_vbm_group`" reads as a referring expression rather than an opaque id.
ENTITY_NOUN = {
    "analyses": "analysis", "groups": "participant group", "tasks": "task",
    "acquisitions": "acquisition", "devices": "device", "measures": "measure",
    "regions": "region", "preprocessings": "preprocessing procedure",
    "model_estimations": "model estimation", "inference_settings": "inference setting",
    "tables": "table", "assessments": "assessment",
}


def humanize(path: str) -> str:
    return path.replace("_", " ")


def flatten(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float, str)):
        return re.sub(r"\s+", " ", str(value)).strip()
    if isinstance(value, list):
        return "; ".join(x for x in (flatten(v) for v in value) if x)
    if isinstance(value, dict):
        return "; ".join(f"{humanize(k)} {flatten(v)}" for k, v in value.items() if flatten(v))
    return ""


def claims(record: dict, pmid: str) -> list[dict]:
    """One claim per extracted leaf, with the subject it belongs to."""
    out: list[dict] = []

    def label_of(entity: dict) -> str | None:
        for key in ("name", "local_id"):
            node = entity.get(key)
            if isinstance(node, dict) and "extraction_status" in node:
                text = flatten(node.get("value"))
                if text:
                    return text
            elif isinstance(node, str):
                return node
        return None

    def walk(node, path: str, subject: str) -> None:
        if isinstance(node, dict):
            if "extraction_status" in node:
                if node.get("extraction_status") != "extracted":
                    return
                value = flatten(node.get("value"))
                if not value:
                    return
                field = humanize(path.rsplit(".", 1)[-1].split("[")[0])
                evidence = (node.get("evidence") or {}).get("status")
                quotes = [sp.get("text", "")
                          for s in ((node.get("evidence") or {}).get("sets") or [])
                          for sp in (s.get("spans") or [])]
                out.append({
                    "pmid": pmid, "path": path, "field": field,
                    "claim": f"{subject}, the {field} is {value}."
                             if subject else f"The {field} of the study is {value}.",
                    "value": value, "evidence_status": evidence,
                    "quote": " ".join(quotes)[:1200], "value_source": node.get("value_source"),
                })
                return
            for key, val in node.items():
                if key in ("extraction_metadata", "local_id"):
                    continue
                walk(val, f"{path}.{key}" if path else key, subject)
        elif isinstance(node, list):
            noun = ENTITY_NOUN.get(path.split(".")[-1])
            for index, item in enumerate(node):
                sub = subject
                if noun and isinstance(item, dict):
                    lab = label_of(item)
                    sub = f"For the {noun} “{lab}”" if lab else f"For {noun} {index + 1}"
                walk(item, f"{path}[{index}]", sub)

    walk(record, "", "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=EXP / "pondie-data/runs/pondie-907/records")
    ap.add_argument("--out", type=Path, default=EXP / "reports" / "claims.jsonl")
    ap.add_argument("--pmids", nargs="*")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    from collections import Counter
    per_paper, by_ev = [], Counter()
    with args.out.open("w") as fh:
        for path in sorted(args.records.glob("*.extraction.json")):
            pmid = path.name.split(".")[0]
            if args.pmids and pmid not in args.pmids:
                continue
            rows = claims(json.loads(path.read_text()), pmid)
            per_paper.append(len(rows))
            for row in rows:
                by_ev[row["evidence_status"]] += 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(rows)
    print(f"{total} claims from {len(per_paper)} records "
          f"(median {sorted(per_paper)[len(per_paper)//2] if per_paper else 0}/paper)")
    print("by evidence status:", dict(by_ev))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
