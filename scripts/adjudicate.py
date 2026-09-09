"""Last pass: put the contradictions nothing else could settle to a model, with the paper.

Everything upstream is deterministic or local. The repair loop rewrites what it can ground,
the schema rules report what stays inconsistent, and a refutation pass removes the subset the
record itself decides. What is left is genuinely undecidable from the record: two fields that
cannot both be true, no arithmetic or reference that picks between them, and the answer
sitting in a sentence of the paper that nobody has looked at for this purpose.

Three properties this pass has to have, because an adjudicator that lacks them is worse than
leaving the contradiction visible:

  * It is told the options and may not invent a third. Every case ships its own permissible
    values, drawn from the schema, plus `unresolved`.
  * It must quote. A resolution is applied only when its quote resolves to a span of this
    paper, by the same `spans.resolve`/`verify` the extractor is held to.
  * It may decline. `unresolved` is a first-class answer and is recorded as such, because a
    forced choice on 12% of cases is noise dressed as a decision.

    python adjudicate.py --records reports/repair --corpus corpus --out reports/adjudicated
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

MODEL = "@psyc-aid338-ope-333f18/gpt-5.6-luna"

SYSTEM = """\
You resolve contradictions in a structured record extracted from a neuroimaging paper.

You are given the paper's methods and results, and a list of cases. Each case names two or
more fields of the record that cannot all be true, and lists the values each field may take.

For each case, answer with the value the paper supports and one verbatim sentence from the
paper that shows it. Copy the sentence exactly; do not paraphrase, join, or trim it.

Answer "unresolved" whenever the paper does not settle the case. That is the correct answer
when the paper is silent, ambiguous, or describes something the options do not cover. A wrong
confident answer is worse than an honest "unresolved", because the record already flags the
contradiction and a reviewer can see it."""


def txt(node):
    return str(node.get("value")) if isinstance(node, dict) and "value" in node else ""


def permissible(schema, class_name: str, slot: str) -> list[str]:
    """The values a slot may take, from the schema, or [] if it is open."""
    cls = schema.classes.get(class_name)
    attribute = (getattr(cls, "attributes", None) or {}).get(slot) if cls else None
    rng = getattr(attribute, "range", None)
    enum = schema.enums.get(rng) if rng else None
    return list(getattr(enum, "permissible_values", {}) or {})


def scope_cases(record: dict, schema) -> list[dict]:
    """The paired scope/regions contradictions the schema rules report.

    Stated here rather than read from the validator's message text because a case needs the
    *fields*, not a sentence about them -- which slot to write, and what it may be set to.
    """
    cases = []
    pairs = (("analyses", "Analysis", "spatial_scope", "regions"),
             ("inference_settings", "InferenceSettings", "correction_scope",
              "correction_regions"))
    for key, class_name, scope_slot, region_slot in pairs:
        for entity in record.get(key) or []:
            if not isinstance(entity, dict):
                continue
            scope = txt(entity.get(scope_slot)).strip().lower()
            regions = entity.get(region_slot) or []
            if scope in ("whole_brain", "whole brain", "searchlight") and regions:
                names = [region_name(record, r) for r in regions]
                cases.append({
                    "id": f"{key}/{entity.get('local_id')}/{scope_slot}",
                    "question": (
                        f"The record says {scope_slot} is '{scope}' while {region_slot} names "
                        f"{', '.join(names)}. A whole-brain or searchlight "
                        f"{'analysis models' if scope_slot == 'spatial_scope' else 'correction covers'}"
                        f" no single region, so at most one of these is right."),
                    "options": permissible(schema, class_name, scope_slot) or
                               ["whole_brain", "roi", "searchlight"],
                    "target": (key, entity.get("local_id"), scope_slot, region_slot),
                })
    return cases


def region_name(record: dict, local_id: str) -> str:
    for r in record.get("regions") or []:
        if r.get("local_id") == local_id:
            return txt(r.get("name")) or local_id
    return local_id


def unresolved_findings(record: dict, schema) -> list[str]:
    """Whatever the validator still reports, for the record of what was left."""
    from pondie.extraction.record.validate import Validator

    v = Validator(schema, None)
    v.check_record(record)
    return v.errors + v.warnings


def ask(caller, pmid: str, premise: str, cases: list[dict]) -> dict:
    from pondie.extraction.models import ModelCall

    listing = "\n\n".join(
        f"case {i + 1} (id {c['id']}):\n  {c['question']}\n"
        f"  permissible values: {', '.join(c['options'])}, or unresolved"
        for i, c in enumerate(cases))
    prompt = (f"## Paper (methods and results)\n\n{premise}\n\n"
              f"## Cases\n\n{listing}\n\n"
              'Reply as {"resolutions": [{"id": ..., "value": ..., "quote": ...}]}. '
              'Use the case id verbatim. Use "unresolved" as the value when the paper does '
              "not settle it, and give an empty quote for those.")
    reply = caller(ModelCall(model=MODEL, system=SYSTEM, prompt=prompt,
                             effort="low", max_output_tokens=4000),
                   paper=pmid, stage="adjudicate")
    body = getattr(reply, "body", reply)
    return body if isinstance(body, dict) else json.loads(str(body))


def apply(record: dict, case: dict, value: str, quote: str, text: str,
          stats: Counter) -> str:
    """Write a resolution, but only when its quote is really in this paper."""
    from pondie.extraction.record import spans as span_tools

    key, local, scope_slot, region_slot = case["target"]
    if value == "unresolved" or value not in case["options"]:
        stats["left unresolved"] += 1
        return "unresolved"
    cleaned = re.sub(r"\s+", " ", quote or "").strip()
    span = None
    if len(cleaned) >= 20:
        try:
            span = span_tools.resolve(text, cleaned).as_record()
            span_tools.verify(text, span)
        except Exception:
            span = None
    if span is None:
        # No quote, no write. The failure mode this guards against is a plausible value with
        # an invented sentence, which reads exactly like a resolved case.
        stats["rejected: quote not in the paper"] += 1
        return "unquoted"
    entity = next((e for e in record.get(key) or []
                   if isinstance(e, dict) and e.get("local_id") == local), None)
    if entity is None:
        stats["entity vanished"] += 1
        return "missing"
    entity[scope_slot] = {"extraction_status": "extracted", "value": value,
                          "value_source": "reported",
                          "evidence": {"status": "present",
                                       "sets": [{"source": "adjudication",
                                                 "spans": [span]}]}}
    if value in ("whole_brain", "searchlight"):
        # Resolving the scope resolves the pair: the regions were the other half of the
        # contradiction, and leaving them would re-report it on the next validation.
        entity[region_slot] = []
    stats[f"resolved to {value}"] += 1
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suffix", default=".repaired.json")
    ap.add_argument("--env", type=Path, default=Path("/home/james/pondie/.env"))
    ap.add_argument("--dry-run", action="store_true",
                    help="collect and print the cases without calling the model")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from pondie import schema as ps
    from pondie.extraction.llm import GatewayCaller, load_env
    from pondie.schema import reader

    if args.env.is_file():
        load_env(args.env)
    schema = reader.load(ps.STORAGE)
    caller = None if args.dry_run else GatewayCaller()

    stats, report, papers = Counter(), [], 0
    for path in sorted(args.records.glob(f"*{args.suffix}")):
        pmid = path.name.split(".")[0]
        source = args.corpus / pmid / "processed/local/text.tables.txt"
        if not source.is_file():
            continue
        record = json.loads(path.read_text())
        text = source.read_text(errors="replace")
        cases = scope_cases(record, schema)
        papers += 1
        if not cases:
            (args.out / path.name).write_text(json.dumps(record, indent=1))
            stats["no contradiction"] += 1
            continue
        stats["papers with a contradiction"] += 1
        if args.dry_run:
            for c in cases:
                print(f"  {pmid} {c['id']}\n     {c['question']}")
            continue

        # The methods and results, not the whole paper: the same slice the repair loop
        # checks claims against, so the adjudicator sees what the grounder saw.
        from repair_loop import sections
        premise = sections(text)
        try:
            answer = ask(caller, pmid, premise, cases)
        except Exception as exc:
            stats[f"call failed: {type(exc).__name__}"] += 1
            (args.out / path.name).write_text(json.dumps(record, indent=1))
            continue
        by_id = {c["id"]: c for c in cases}
        for row in answer.get("resolutions") or []:
            case = by_id.get(str(row.get("id", "")).strip())
            if case is None:
                stats["reply named an unknown case"] += 1
                continue
            outcome = apply(record, case, str(row.get("value", "")).strip(),
                            str(row.get("quote", "")), text, stats)
            report.append({"pmid": pmid, "case": case["id"], "outcome": outcome,
                           "quote": str(row.get("quote", ""))[:200]})
        (args.out / path.name).write_text(json.dumps(record, indent=1))
        left = unresolved_findings(record, schema)
        if left:
            report.append({"pmid": pmid, "remaining_findings": left[:10]})

    (args.out / "adjudication.json").write_text(json.dumps(report, indent=1))
    print(f"{papers} records -> {args.out}")
    for k, v in stats.most_common():
        print(f"  {v:>5}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
