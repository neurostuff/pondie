"""R4: what a repair pass did to the reference slots, which no other measure can see.

`scripts/repair_delta.py` walks `ExtractedValue` wrappers. A reference slot does not hold
one -- it holds a bare list of `local_id` -- so it carries no evidence, no `value_source`
and nothing a span metric can count. Every write to one is invisible to M1, M2, M3 and M5.

On 18823721 the pass made six reference writes and at least four are wrong:

    groups/grp_controls.diagnostic_instrument         <- 4 assessments
    groups/grp_opioid_patients.diagnostic_instrument  <- the same 4 assessments
    inference_settings/inf_interaction.correction_regions  <- [reg_stn]
    inference_settings/inf_stn.correction_regions          <- [reg_stn]

`Group.diagnostic_instrument` is "The study assessment that established this group's
defining condition"; the four are craving, anhedonia and drug-history questionnaires, and
two of them were given to the patients only. The subthalamic nucleus is the *result* of the
interaction contrast, and "region of interest", "small volume" and word-boundary ROI occur
zero times in that paper.

Two things here, and they are different jobs:

  * **accounting** -- reference writes per record, added and removed, so the pass's
    reference half is on the same table as its value half.
  * **the shared-target signal** -- one target list written to several entities of a class
    in one pass. Both errors above have that shape. Measured over the twelve-paper baseline
    it is *not* a general error signal: 15 of 15 shared-target writes there are analyses
    sharing the paper's one task, six analyses sharing its one diagnostic interview, and two
    model estimations sharing its one preprocessing -- all correct. What discriminates is
    the slot, not the pattern, so only slots whose schema description ties the target to the
    entity are gated. See `EXCLUSIVE` and `--explain`.

With `benchmarks/repair_truth/` present the reference slots recorded there are scored
correct / wrong / invented, which is the half of R5 that applies to references.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from pondie import schema
from pondie.schema import reader

TRUTH = Path(__file__).resolve().parent.parent / "benchmarks" / "repair_truth"

#: Slots where one target belongs to one entity, so writing the same list to two of them in
#: one pass is a copy rather than a reading. Named, not derived: the schema describes
#: `diagnostic_instrument` as "The study assessment that established THIS group's defining
#: condition", and no rule reads that. Everything not here is measured and not gated,
#: because sharing is how the other reference slots are meant to work -- twelve baseline
#: papers produced 15 shared-target writes and every one of them is correct.
EXCLUSIVE: frozenset[tuple[str, str]] = frozenset({
    ("groups", "diagnostic_instrument"),
})

#: Shared here is suspicious and not decidable: two contrasts CAN share a correction mask,
#: and on 18823721 neither of them did -- the STN is the result the contrast found. Counted
#: separately so the number is visible without failing a run on it.
SUSPECT: frozenset[tuple[str, str]] = frozenset({
    ("inference_settings", "correction_regions"),
    ("analyses", "regions"),
})

EXPLAIN = """\
Where the shared-target refusal belongs
---------------------------------------
In `edit.py`, as a refusal, but ONLY for the slots in `EXCLUSIVE`, and it cannot be written
as one of the existing `GUARDS`.

The blanket rule does not survive contact with the data. Across `runs/repair-baseline` the
pass made 15 shared-target writes and all 15 are right: six analyses of 11296095 sharing the
one SCID, two analyses of 12860777 sharing the one picture task, three analyses of 14667419
sharing the one cue-reactivity task and two sharing its three tables, two model estimations
of 14679386 sharing the one preprocessing. A paper with one task and six contrasts is the
normal case, not the error. Refusing on the pattern alone would have blocked all of it and
caught nothing.

A `Check` sees one `Edit`: one entity, one slot, one value. "The same targets are going to
two entities of this class" is a property of the sweep, not of the edit, and the second
write is the one that has to be refused -- by which time the first has already landed. The
refusal therefore needs sweep-scoped state: `_sweep` in `repair.py` already iterates one
class at a time and holds `by_id`, so the natural home is a per-class record of
`(slot, tuple(sorted(targets)))` already written, with `apply` refusing a repeat.

What survives is the slot. `Group.diagnostic_instrument` is described as the assessment that
established THIS group's condition, so one list on two groups is a copy: on 18823721 the same
four questionnaires were written to the patients and the controls, and two of the four were
administered to the patients only. That is `EXCLUSIVE`, and it is what to refuse.

A `Check` sees one `Edit`: one entity, one slot, one value. "The same targets already went to
another entity of this class" is a property of the sweep, and the second write is the one to
refuse -- by which time the first has landed. So it needs sweep-scoped state rather than a
new `GUARDS` entry: `_sweep` in `repair.py` already iterates one class at a time and holds
`by_id`, so the home is a per-class set of `(slot, tuple(sorted(targets)))` already written,
passed into `apply` and refused on a repeat, for slots in `EXCLUSIVE` only.

The gate is `exclusive_shared == 0`. `suspect_shared` and `shared_target_writes` are reported
and not gated: the first needs the ground truth to adjudicate, the second is normal.
"""


def entities(record: Mapping[str, Any]) -> dict[str, tuple[str, dict]]:
    """`container/local_id` -> (container, entity), for every addressable entity."""
    out: dict[str, tuple[str, dict]] = {}
    for container, items in record.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, Mapping) and item.get("local_id"):
                out[f"{container}/{item['local_id']}"] = (container, dict(item))
    return out


def reference_slots(sch) -> dict[str, set[str]]:
    """container -> the slots of its class that hold references."""
    out: dict[str, set[str]] = {}
    for container, class_name in sch.classes_by_container().items():
        out[container] = {name for name, _slot, kind in sch.iter_slots(class_name)
                          if kind == "reference"}
    return out


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def label(record: Mapping[str, Any], local_id: str) -> str:
    from pondie.extraction.record import edit as edit_module

    for items in record.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, Mapping) and item.get("local_id") == local_id:
                return edit_module.label_of(item)
    return local_id


def diff(before: Mapping[str, Any], after: Mapping[str, Any], sch) -> dict:
    """Every reference slot that changed, and the shared-target writes among them."""
    slots = reference_slots(sch)
    was, now = entities(before), entities(after)
    changes: list[dict] = []
    for key, (container, entity) in now.items():
        for slot in sorted(slots.get(container, ())):
            old = as_list((was.get(key) or (None, {}))[1].get(slot))
            new = as_list(entity.get(slot))
            if old == new:
                continue
            changes.append({
                "path": f"{key}.{slot}", "container": container, "local_id": key.split("/")[1],
                "slot": slot,
                "added": [t for t in new if t not in old],
                "removed": [t for t in old if t not in new],
                "before": old, "after": new,
                "created_entity": key not in was,
            })

    # One target list written to several entities of a class: the shape of both known errors.
    by_target: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
    for change in changes:
        if change["added"]:
            key = (change["container"], change["slot"], tuple(sorted(change["added"])))
            by_target[key].append(change["local_id"])
    shared = [{"container": c, "slot": s, "targets": list(t), "entities": ids,
               "kind": "exclusive" if (c, s) in EXCLUSIVE
                       else "suspect" if (c, s) in SUSPECT else "normal"}
              for (c, s, t), ids in sorted(by_target.items()) if len(ids) > 1]
    return {"changes": changes, "shared": shared}


def truth_for(pmid: str) -> dict | None:
    path = TRUTH / f"{pmid}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def matches(entity_key: str, entity_label: str, patterns: Iterable[str]) -> bool:
    haystack = f"{entity_key} {entity_label}".lower()
    return any(p.lower() in haystack for p in patterns)


def score_links(pmid: str, before: Mapping[str, Any], after: Mapping[str, Any],
                sch) -> list[dict]:
    """Every link this pass ADDED, scored against `links` in the truth file.

    Per target and per write, because they answer different questions. A write is right only
    if every target it names is; a target is one entity chosen. The first is what a reader
    sees, the second is what the pass actually decided, and on 18823721 one wrong write
    carries four wrong targets while one right write carries one right target -- so quoting
    either alone flatters or damns the pass by a factor of four.
    """
    truth = truth_for(pmid)
    if truth is None:
        return []
    wanted = truth.get("links") or []
    if not wanted:
        return []
    patterns = {e["key"]: e.get("match") or [] for e in truth["entities"]}
    result = diff(before, after, sch)
    out: list[dict] = []
    for change in result["changes"]:
        if not change["added"]:
            continue
        here = f"{change['container']}/{change['local_id']}"
        label = label_of_id(after, change["local_id"])
        spec = next((l for l in wanted
                     if l["slot"] == change["slot"]
                     and matches(here, label, patterns.get(l["entity"], []))), None)
        if spec is None:
            out.append({"pmid": pmid, "path": change["path"], "verdict": "unverifiable",
                        "targets": change["added"], "allowed": None, "quote": "",
                        "note": "no truth link for this entity and slot"})
            continue
        allowed = spec.get("targets") or []
        good, bad = [], []
        for target in change["added"]:
            name = f"{target} {label_of_id(after, target)}".lower()
            (good if any(a.lower() in name for a in allowed) else bad).append(target)
        out.append({"pmid": pmid, "path": change["path"],
                    "verdict": "correct" if not bad else "wrong",
                    "targets": change["added"], "right": good, "wrong": bad,
                    "allowed": allowed, "support": spec.get("support"),
                    "note": (spec.get("note") or "")[:200]})
    return out


def label_of_id(record: Mapping[str, Any], local_id: str) -> str:
    return label(record, local_id)


def score(pmid: str, after: Mapping[str, Any], sch) -> list[dict]:
    """Reference slots of the ground truth, scored against the repaired record.

    Entities are matched on `match` -- substrings drawn from the article, not from any
    record -- so the scoring does not inherit the record's own idea of what an entity is.
    """
    truth = truth_for(pmid)
    if truth is None:
        return []
    slots = reference_slots(sch)
    now = entities(after)
    verdicts: list[dict] = []
    for spec in truth["entities"]:
        patterns = spec.get("match") or []
        wanted = {slot: field for slot, field in spec["fields"].items()
                  if slot in {s for group in slots.values() for s in group}}
        if not wanted or not patterns:
            continue
        found = [(key, entity) for key, (container, entity) in now.items()
                 if slot_owner(container, wanted, slots)
                 and matches(key, label(after, key.split("/")[1]), patterns)]
        for slot, field in wanted.items():
            expected = field.get("value")
            for key, entity in found:
                got = as_list(entity.get(slot))
                names = [label(after, t) for t in got]
                if expected is None and got:
                    verdict = "invented"
                elif expected is None:
                    verdict = "correct"
                elif not got:
                    verdict = "missed"
                else:
                    ok = [n for n in names
                          if any(_bare(str(e)) in _bare(n) or _bare(n) in _bare(str(e))
                                 for e in (expected if isinstance(expected, list)
                                           else [expected]))]
                    verdict = "correct" if len(ok) == len(names) else "wrong"
                verdicts.append({"pmid": pmid, "entity": spec["key"], "record": key,
                                 "slot": slot, "expected": expected, "got": names,
                                 "verdict": verdict, "quote": field.get("quote", "")[:120]})
    return verdicts


def slot_owner(container: str, wanted: Mapping[str, Any], slots: Mapping[str, set]) -> bool:
    return any(slot in slots.get(container, ()) for slot in wanted)


def _bare(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def pairs(run: Path) -> list[tuple[str, Path, Path]]:
    out = []
    for after in sorted((run / "records").glob("*.extraction.json")):
        before = run / "unrepaired" / after.name
        if before.is_file():
            out.append((after.name.split(".")[0], before, after))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, nargs="?", help="a run dir with records/ and unrepaired/")
    ap.add_argument("--detail", action="store_true", help="every reference change, by path")
    ap.add_argument("--explain", action="store_true",
                    help="where the shared-target refusal belongs, and why not in GUARDS")
    args = ap.parse_args()
    if args.explain:
        print(EXPLAIN)
        return 0
    if args.run is None:
        ap.error("a run directory is required unless --explain")

    sch = reader.load(schema.EXTRACTION)
    rows, all_shared, all_verdicts, all_links = [], [], [], []
    for pmid, before_path, after_path in pairs(args.run):
        before = json.loads(before_path.read_text())
        after = json.loads(after_path.read_text())
        result = diff(before, after, sch)
        counts = Counter()
        for change in result["changes"]:
            counts["slots_changed"] += 1
            counts["targets_added"] += len(change["added"])
            counts["targets_removed"] += len(change["removed"])
            if change["created_entity"]:
                counts["on_new_entities"] += 1
        for entry in result["shared"]:
            counts["shared_target_writes"] += len(entry["entities"])
            if entry["kind"] == "exclusive":
                counts["exclusive_shared"] += len(entry["entities"])
            elif entry["kind"] == "suspect":
                counts["suspect_shared"] += len(entry["entities"])
        rows.append((pmid, counts))
        for entry in result["shared"]:
            all_shared.append((pmid, entry))
        all_verdicts += score(pmid, after, sch)
        all_links += score_links(pmid, before, after, sch)
        if args.detail:
            for change in result["changes"]:
                print(f"  {pmid} {change['path']}: {change['before']} -> {change['after']}")

    cols = [("ref slots changed", "slots_changed"), ("targets +", "targets_added"),
            ("targets -", "targets_removed"), ("on new entities", "on_new_entities"),
            ("shared-target", "shared_target_writes"),
            ("exclusive", "exclusive_shared"), ("suspect", "suspect_shared")]
    print(f"{'pmid':12}" + "".join(f"{label:>18}" for label, _k in cols))
    total = Counter()
    for pmid, counts in rows:
        print(f"{pmid:12}" + "".join(f"{counts[k]:>18}" for _l, k in cols))
        total.update(counts)
    print("-" * (12 + 18 * len(cols)))
    print(f"{'TOTAL':12}" + "".join(f"{total[k]:>18}" for _l, k in cols))

    if all_shared:
        print(f"\nR4 shared-target writes ({len(all_shared)} group(s)) -- one target list, "
              f"several entities of a class:")
        for pmid, entry in all_shared:
            print(f"  [{entry['kind']:9}] {pmid} {entry['container']}.{entry['slot']} -> "
                  f"{entry['targets']}")
            print(f"      written to {entry['entities']}")

    if all_verdicts:
        tally = Counter(v["verdict"] for v in all_verdicts)
        print(f"\nR5 on reference slots ({len(all_verdicts)} scored): {dict(tally)}")
        for v in all_verdicts:
            if v["verdict"] != "correct":
                print(f"  {v['verdict'].upper():9} {v['pmid']} {v['record']}.{v['slot']}: "
                      f"got {v['got']}, truth {v['expected']!r}")
                print(f"      {v['quote']}")
    else:
        print("\nR5 on reference slots: no ground truth for these papers "
              f"(have: {sorted(p.stem for p in TRUTH.glob('*.json'))})")

    scored = [v for v in all_links if v["verdict"] != "unverifiable"]
    if scored:
        writes_bad = sum(1 for v in scored if v["verdict"] == "wrong")
        targets = sum(len(v["targets"]) for v in scored)
        targets_bad = sum(len(v["wrong"]) for v in scored)
        print(f"\nR5 on LINKS: {len(scored)} link writes scored "
              f"({len(all_links) - len(scored)} unverifiable)")
        print(f"  writes  : {len(scored) - writes_bad} right, {writes_bad} wrong "
              f"({writes_bad / len(scored):.0%} wrong)")
        print(f"  targets : {targets - targets_bad} right, {targets_bad} wrong "
              f"({targets_bad / max(1, targets):.0%} wrong)")
        for v in scored:
            if v["verdict"] == "wrong":
                print(f"  WRONG {v['pmid']} {v['path']}")
                print(f"      wrote {v['wrong']}, truth allows {v['allowed'] or 'nothing'}")
                print(f"      {v['note']}")
    elif all_links:
        print(f"\nR5 on LINKS: {len(all_links)} writes, none matched a truth link")

    print(f"\n{len(rows)} record(s). R4 gate: exclusive_shared == 0 "
          f"(shared-target and suspect are reported, not gated).")
    failed = [p for p, c in rows if c["exclusive_shared"]]
    print(f"records failing R4: {len(failed)}" + (f" -> {failed}" if failed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
