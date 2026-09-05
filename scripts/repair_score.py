"""R5: is what repair wrote RIGHT? Scored against `benchmarks/repair_truth/`.

M1-M3 and R1-R4 say whether the pass damaged the record. None of them can say it helped.
This one can, and only for the papers someone has read.

Four verdicts, not two. `wrong` and `invented` are different failures and the second is the
one repair was built to avoid: a field the paper does not report, filled anyway.

    correct      the value matches the truth, or one of its `also_acceptable` readings
    inferred     the paper is silent but the value is one the truth lists as defensible --
                 a reading, not a reading-off. Not damage, but it must be `generated`.
    wrong        the truth reports a value and this is not it
    invented     the truth says the paper is silent (`support: absent`, `value: null`)
                 and the record carries a value the truth does not even allow
    missed       the truth reports a value and the record is empty
    unverifiable the truth has no entry, or no entity in the record matches

Two headline numbers, both per record:

    damage rate  (wrong + invented) / fields this pass changed or created
    yield        fields that went empty -> a correct value

Precision over all writes is not one of them: most writes re-write what was already there,
so a pass that changes nothing and re-states 80 correct fields would score 100%.

Entities are matched on the truth file's `match` list -- substrings from the article, not
from any record -- so the scoring does not inherit the record's own segmentation. An entity
the record does not have is reported as unmatched rather than counted as correct.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from pondie import schema
from pondie.formats import values
from pondie.schema import reader

TRUTH = Path(__file__).resolve().parent.parent / "benchmarks" / "repair_truth"


def bare(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def same(expected: Any, got: Any) -> bool:
    """Whether `got` says what `expected` says, at the tolerance a reader would allow."""
    if expected is None:
        return got in (None, "", [])
    if isinstance(expected, list) or isinstance(got, list):
        want = expected if isinstance(expected, list) else [expected]
        have = got if isinstance(got, list) else [got]
        return bool(have) and all(any(same(w, h) for w in want) for h in have)
    if isinstance(expected, bool) or isinstance(got, bool):
        return bool(expected) == bool(got)
    try:
        return float(expected) == float(got)
    except (TypeError, ValueError):
        pass
    left, right = bare(expected), bare(got)
    return bool(left) and bool(right) and (left in right or right in left)


def acceptable(field: Mapping[str, Any], got: Any) -> bool:
    if same(field.get("value"), got):
        return True
    return any(same(alt, got) for alt in field.get("also_acceptable") or [])


def containers_for(sch, kind: str) -> list[str]:
    return [c for c, cls in sch.classes_by_container().items() if cls == kind
            or (kind in sch and cls in sch and sch.resolves_to(kind, cls))]


def label_of(entity: Mapping[str, Any]) -> str:
    from pondie.extraction.record import edit as edit_module

    return edit_module.label_of(entity)


def find(record: Mapping[str, Any], sch, kind: str, patterns: list[str]) -> list[tuple[str, dict]]:
    out = []
    for container in containers_for(sch, kind.split("[")[0]):
        for entity in record.get(container) or []:
            if not isinstance(entity, Mapping):
                continue
            hay = f"{entity.get('local_id', '')} {label_of(entity)}".lower()
            if any(p.lower() in hay for p in patterns):
                out.append((f"{container}/{entity.get('local_id')}", dict(entity)))
    return out


def score(pmid: str, before: Mapping, after: Mapping, sch) -> tuple[Counter, list[dict]]:
    truth = json.loads((TRUTH / f"{pmid}.json").read_text(encoding="utf-8"))
    tally: Counter = Counter()
    rows: list[dict] = []

    for spec in truth.get("must_not_exist") or []:
        kind = spec["kind"].split("[")[0]
        for container in containers_for(sch, kind):
            present = [e.get("local_id") for e in after.get(container) or []
                       if isinstance(e, Mapping)]
            was = {e.get("local_id") for e in before.get(container) or []
                   if isinstance(e, Mapping)}
            new = [p for p in present if p not in was]
            if new:
                tally["invented"] += len(new)
                rows.append({"verdict": "invented", "path": f"{container}/*",
                             "expected": "no such entity", "got": new,
                             "note": spec["why"][:160]})

    for spec in truth["entities"]:
        found = find(after, sch, spec["kind"], spec.get("match") or [])
        if not found:
            tally["unmatched_entities"] += 1
            rows.append({"verdict": "unverifiable", "path": f"{spec['kind']}/{spec['key']}",
                         "expected": "-", "got": "no matching entity in the record",
                         "note": spec.get("identified_by", "")})
            continue
        for key, entity in found:
            old_entity = next((e for e in before.get(key.split("/")[0]) or []
                               if isinstance(e, Mapping)
                               and e.get("local_id") == key.split("/")[1]), {})
            for slot, field in spec["fields"].items():
                if slot not in entity and slot not in old_entity:
                    got, was = None, None
                else:
                    got = values.read(entity.get(slot))
                    was = values.read(old_entity.get(slot))
                changed = not same(was, got) or (was in (None, "", []) and got not in (None, "", []))
                empty_now = got in (None, "", [])
                if field.get("value") is None and field.get("support") == "absent":
                    # `also_acceptable` on an absent field names the readings a careful
                    # reviewer would defend. Filling one is inference, not invention, and
                    # scoring it as invention would punish the pass for being right.
                    verdict = ("correct" if empty_now
                               else "inferred" if any(same(alt, got) for alt
                                                      in field.get("also_acceptable") or [])
                               else "invented")
                elif empty_now:
                    verdict = "missed"
                else:
                    verdict = "correct" if acceptable(field, got) else "wrong"
                tally[verdict] += 1
                if changed:
                    tally[f"changed_{verdict}"] += 1
                    tally["changed"] += 1
                    if was in (None, "", []) and verdict in ("correct", "inferred"):
                        tally["yield"] += 1
                if verdict not in ("correct", "inferred"):
                    rows.append({"verdict": verdict, "path": f"{key}.{slot}",
                                 "expected": field.get("value"), "got": got,
                                 "changed_by_repair": changed,
                                 "note": (field.get("note") or field.get("quote", ""))[:160]})
    return tally, rows


def pairs(run: Path) -> list[tuple[str, Path, Path]]:
    out = []
    for after in sorted((run / "records").glob("*.extraction.json")):
        pmid = after.name.split(".")[0]
        if not (TRUTH / f"{pmid}.json").is_file():
            continue
        before = run / "unrepaired" / after.name
        out.append((pmid, before if before.is_file() else after, after))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, nargs="+", help="run dir(s) with records/ and unrepaired/")
    ap.add_argument("--detail", action="store_true", help="every non-correct field")
    args = ap.parse_args()
    sch = reader.load(schema.EXTRACTION)

    for run in args.run:
        found = pairs(run)
        print(f"\n=== {run.name}: {len(found)} of "
              f"{len(list(TRUTH.glob('*.json')))} truth papers present")
        if not found:
            print("    nothing to score here")
            continue
        grand: Counter = Counter()
        for pmid, before_path, after_path in found:
            tally, rows = score(pmid, json.loads(before_path.read_text()),
                                json.loads(after_path.read_text()), sch)
            grand.update(tally)
            scored = sum(tally[v] for v in ("correct", "inferred", "wrong", "invented",
                                            "missed"))
            damage = tally["changed_wrong"] + tally["changed_invented"]
            rate = f"{damage / tally['changed']:.0%}" if tally["changed"] else "n/a"
            print(f"  {pmid}: {scored} fields scored | correct {tally['correct']} "
                  f"inferred {tally['inferred']} "
                  f"wrong {tally['wrong']} invented {tally['invented']} "
                  f"missed {tally['missed']} | changed {tally['changed']} "
                  f"| damage {rate} | yield {tally['yield']} "
                  f"| unmatched entities {tally['unmatched_entities']}")
            if args.detail:
                for row in rows:
                    print(f"      {row['verdict'].upper():12} {row['path']}: "
                          f"got {row['got']!r}, truth {row['expected']!r}")
                    if row.get("note"):
                        print(f"          {row['note']}")
        damage = grand["changed_wrong"] + grand["changed_invented"]
        rate = f"{damage / grand['changed']:.0%}" if grand["changed"] else "n/a"
        print(f"  TOTAL: correct {grand['correct']} inferred {grand['inferred']} "
              f"wrong {grand['wrong']} "
              f"invented {grand['invented']} missed {grand['missed']} "
              f"| changed {grand['changed']} | damage rate {rate} "
              f"| yield {grand['yield']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
