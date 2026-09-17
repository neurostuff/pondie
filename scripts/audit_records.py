#!/usr/bin/env python3
"""What is systematically wrong with a corpus of extraction records.

Six sweeps, each measuring a defect class that a deterministic fix could close. Every
number in docs/record-defects.md comes from here; rerun it to reproduce them.

    python scripts/audit_records.py --records '<dir>/*/*.extraction.json'

The sweeps are separate because the fixes are: one is a check with a false positive, one
is a join the record already holds the two halves of, one is a shape, one is a label, one
is a contradiction between two slots, and one is an entity nothing points at. Reporting
them as one count would hide that.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

from pondie import schema
from pondie.extraction.record import rules
from pondie.extraction.record.rules import _model_index, terms_in_scope
from pondie.formats import values
from pondie.formats.values import iter_fields
from pondie.normalization._records import iter_records
from pondie.schema import reader

fold = lambda s: re.sub(r"[^a-z0-9]+", "", str(s or "").lower())  # noqa: E731

#: A `Cell.level` on a continuous term that says which way the effect went. It is not a
#: level -- `Cell.direction` is the slot for it, and 211 of the 214 already hold it.
DIRECTION_WORDS = {
    "positive", "negative", "higher", "lower", "greater", "less", "increase", "decrease",
    "increased", "decreased", "up", "down", "more", "fewer", "activation", "deactivation",
    "positivecorrelation", "negativecorrelation",
}

#: The entity slots a `FactorLevel` can reach. A categorical level reaching none of them is
#: a bare string, and the mapper joins these on the string.
LEVEL_JOINS = ("conditions", "arms", "timepoints", "groups", "regions")

NUMERIC = {"integer", "float", "double", "decimal"}


#: The FactorLevel slots a name match may fill, and the kind of entity each names. The
#: fixer is enumerated rather than applied to every reference slot because name identity is
#: evidence only where the reference means "is that entity". See docs/record-defects.md:
#: `ModelTerm.interaction_with` means "crossed with", and matching it on a name links the
#: group factor to the group factor 708 times.
IDENTITY_SLOTS = ("conditions", "arms", "timepoints", "groups", "regions")


def name_catalogue(body: dict) -> dict[str, dict[str, list[str]]]:
    """folded name -> {FactorLevel slot: [local_id]} over the kinds a level can name."""
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    def add(slot: str, entity: object) -> None:
        if isinstance(entity, dict) and isinstance(entity.get("local_id"), str):
            name = fold(values.read(entity.get("name")))
            if name:
                out[name][slot].append(entity["local_id"])

    for group in body.get("groups") or []:
        add("groups", group)
    for task in body.get("tasks") or []:
        if isinstance(task, dict):
            for condition in task.get("conditions") or []:
                add("conditions", condition)
    for region in body.get("regions") or []:
        add("regions", region)
    design = body.get("design") or {}
    for slot in ("arms", "timepoints"):
        for entity in design.get(slot) or []:
            add(slot, entity)
    return out


def held(node: object, slot: str) -> bool:
    if not isinstance(node, dict):
        return False
    return bool(node.get(slot)) and bool([x for x in (values.read(node.get(slot)) or []) if x])


def link_by_name(body: dict) -> Counter:
    """Write the links a name match settles. Returns what it wrote, by slot.

    Only an exact fold match to exactly one candidate, only where the slot is empty, and
    never to the entity itself -- the three guards, each of which a measured failure earned.
    """
    catalogue = name_catalogue(body)
    wrote: Counter = Counter()
    for model in body.get("model_estimations") or []:
        if not isinstance(model, dict):
            continue
        for term in model.get("terms") or []:
            if not isinstance(term, dict):
                continue
            for level in term.get("levels") or []:
                if not isinstance(level, dict):
                    continue
                name = fold(values.read(level.get("level")))
                for slot, local_ids in (catalogue.get(name) or {}).items():
                    if len(local_ids) != 1 or held(level, slot):
                        continue
                    level[slot] = {
                        "value": [local_ids[0]],
                        "extraction_status": "extracted",
                        "value_source": "generated",
                        "evidence": None,
                    }
                    wrote[slot] += 1
    for group in body.get("groups") or []:
        if not isinstance(group, dict) or group.get("arm"):
            continue
        local_ids = (catalogue.get(fold(values.read(group.get("name")))) or {}).get("arms") or []
        if len(local_ids) == 1:
            group["arm"] = local_ids[0]
            wrote["Group.arm"] += 1
    return wrote


def queryability(body: dict, out: Counter) -> None:
    """Can a query reconstruct each analysis's contrast from the entity graph?

    Counted per (analysis, level) pair rather than per distinct level, because that is what
    a query traverses: one unwritten link costs every analysis whose model reaches the term.
    """
    models = _model_index(body)
    for analysis in body.get("analyses") or []:
        if not isinstance(analysis, dict):
            continue
        terms = terms_in_scope(analysis.get("model_estimation"), models)
        cells = [
            level
            for term in terms.values()
            if str(values.read(term.get("type"))) == "categorical"
            for level in (term.get("levels") or [])
            if isinstance(level, dict)
        ]
        if not cells:
            continue
        out["analyses with a categorical contrast"] += 1
        resolved = [any(held(level, slot) for slot in IDENTITY_SLOTS) for level in cells]
        out["levels"] += len(resolved)
        out["levels resolved"] += sum(resolved)
        if all(resolved):
            out["analyses fully resolvable from the entity graph"] += 1
        for slot, label in (
            ("conditions", "analyses that can say which condition"),
            ("arms", "analyses that can say which arm"),
            ("timepoints", "analyses that can say which occasion"),
        ):
            if any(held(level, slot) for level in cells):
                out[label] += 1


class Sink:
    def __init__(self):
        self.errors: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []

    def error(self, path, message):
        self.errors.append((path, message))

    def warn(self, path, message):
        self.warnings.append((path, message))


def template(message: str) -> str:
    """Collapse a finding to its shape, so instances aggregate rather than listing."""
    out = re.sub(r"'[^']*'", "'X'", message)
    out = re.sub(r'"[^"]*"', '"X"', out)
    return re.sub(r"\b\d+\b", "N", out)[:110]


def entity_index(body: dict) -> dict[str, tuple[str, str]]:
    """folded name -> (the FactorLevel slot that reaches it, its local_id).

    Built per record, because a level joins to an entity of the same paper or to nothing.
    """
    out: dict[str, tuple[str, str]] = {}

    def add(slot: str, entity: object) -> None:
        if isinstance(entity, dict) and entity.get("local_id"):
            out.setdefault(fold(values.read(entity.get("name"))), (slot, entity["local_id"]))

    for group in body.get("groups") or []:
        add("groups", group)
    for task in body.get("tasks") or []:
        if isinstance(task, dict):
            for condition in task.get("conditions") or []:
                add("conditions", condition)
    for region in body.get("regions") or []:
        add("regions", region)
    design = body.get("design") or {}
    for slot in ("arms", "timepoints"):
        for entity in design.get(slot) or []:
            add(slot, entity)
    out.pop("", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True, help="glob reaching *.extraction.json")
    args = ap.parse_args()

    ext, sto = reader.load(schema.EXTRACTION), reader.load(schema.STORAGE)
    errs, warns = Counter(), Counter()
    err_papers, warn_papers = defaultdict(set), defaultdict(set)
    purpose_when_cited = Counter()
    levels = Counter()
    level_recovery = Counter()
    listed = Counter()
    list_len = Counter()
    multi_slot = Counter()
    evidence_tot, evidence_missing = Counter(), Counter()
    continuous_level = Counter()
    declared_cls, referenced_cls = Counter(), Counter()
    records = 0
    wrote: Counter = Counter()
    before, after = Counter(), Counter()

    def walk_shapes(node, cls):
        if not isinstance(node, dict) or values.is_field(node):
            return
        cls = ext.designated_type(node, cls)
        for key, attribute in ext.attributes(cls).items():
            if key not in node:
                continue
            kind = ext.classify(key, attribute)
            if kind == "nested" and isinstance(attribute.range, str):
                child = node[key]
                for item in child if isinstance(child, list) else [child]:
                    walk_shapes(item, attribute.range)
                continue
            wrapper = node[key]
            if not isinstance(wrapper, dict) or "value" not in wrapper:
                continue
            stored = sto.attributes(cls).get("id" if key == "local_id" else key)
            if stored is None:
                continue
            value = wrapper["value"]
            if not stored.multivalued and isinstance(value, list):
                listed[f"{cls}.{key}"] += 1
                list_len[len(value)] += 1
                if len(value) > 1:
                    multi_slot[(f"{cls}.{key}", tuple(str(x)[:18] for x in value[:3]))] += 1
            base = str(stored.range or "")
            for item in value if isinstance(value, list) else [value]:
                if item is None:
                    continue
                if base in NUMERIC and (isinstance(item, bool) or not isinstance(item, (int, float))):
                    listed[f"{cls}.{key} [{type(item).__name__} where {base} declared]"] += 1

    def index_ids(node, cls, out):
        if not isinstance(node, dict):
            return
        cls = ext.designated_type(node, cls)
        attributes = ext.attributes(cls)
        if not attributes:
            return
        if isinstance(node.get("local_id"), str):
            out[node["local_id"]] = cls
        for name, attribute in attributes.items():
            if name not in node or ext.classify(name, attribute) != "nested":
                continue
            if not isinstance(attribute.range, str):
                continue
            child = node[name]
            for item in child if isinstance(child, list) else [child]:
                index_ids(item, attribute.range, out)

    def collect_refs(node, cls, hit):
        if not isinstance(node, dict) or values.is_field(node):
            return
        cls = ext.designated_type(node, cls)
        for key, attribute in ext.attributes(cls).items():
            if key not in node:
                continue
            kind = ext.classify(key, attribute)
            child = node[key]
            if kind == "reference":
                for item in child if isinstance(child, list) else [child]:
                    if isinstance(item, str):
                        hit.add(item)
            elif kind == "nested" and isinstance(attribute.range, str):
                for item in child if isinstance(child, list) else [child]:
                    collect_refs(item, attribute.range, hit)

    for study, body in iter_records((args.records,)):
        records += 1
        queryability(body, before)

        # 1 -- what the existing rules already say
        sink = Sink()
        try:
            rules.check_all(body, sink)
        except Exception as exc:  # noqa: BLE001 -- a crash is a finding
            errs[f"RULE CRASHED: {type(exc).__name__}"] += 1
        for _path, message in sink.errors:
            errs[template(message)] += 1
            err_papers[template(message)].add(study)
        for _path, message in sink.warnings:
            warns[template(message)] += 1
            warn_papers[template(message)].add(study)

        # 2 -- Table.purpose on a table an analysis cites
        cited = {
            name
            for analysis in body.get("analyses") or []
            if isinstance(analysis, dict)
            for name in (analysis.get("tables") or [])
            if isinstance(name, str)
        }
        for table in body.get("tables") or []:
            if isinstance(table, dict) and table.get("local_id") in cited:
                marked = values.read(table.get("purpose"))
                if marked:
                    purpose_when_cited[str(marked)] += 1

        # 3 -- FactorLevel joins, and whether an unjoined one is recoverable
        candidates = entity_index(body)
        for model in body.get("model_estimations") or []:
            if not isinstance(model, dict):
                continue
            for term in model.get("terms") or []:
                if not isinstance(term, dict):
                    continue
                categorical = str(values.read(term.get("type"))) == "categorical"
                for level in term.get("levels") or []:
                    if not isinstance(level, dict):
                        continue
                    levels["total"] += 1
                    reached = [
                        slot
                        for slot in LEVEL_JOINS
                        if level.get(slot) and [x for x in (values.read(level.get(slot)) or []) if x]
                    ]
                    for slot in reached:
                        levels[slot] += 1
                    if reached or not categorical:
                        continue
                    levels["categorical, no entity"] += 1
                    name = fold(values.read(level.get("level")))
                    if not name:
                        level_recovery["the level is empty"] += 1
                    elif name in candidates:
                        level_recovery[f"exact match -> {candidates[name][0]}"] += 1
                    else:
                        near = [v for k, v in candidates.items() if name in k or k in name]
                        level_recovery[
                            "substring match, ambiguous"
                            if len(near) > 1
                            else f"substring match -> {near[0][0]}"
                            if len(near) == 1
                            else "matches no declared entity"
                        ] += 1

        # 4 -- shapes
        walk_shapes(body, "Study")

        # 5 -- evidence against value_source, per slot
        for path, node in iter_fields(body):
            if not isinstance(node, dict) or node.get("extraction_status") != "extracted":
                continue
            if node.get("value_source") != "reported":
                continue
            slot = path.rsplit(".", 1)[-1].split("[")[0]
            if path.startswith("tables[") or slot in rules.IDENTIFIERS:
                continue
            owner = ".".join(part.split("[")[0] for part in path.split(".")[-2:])
            evidence_tot[owner] += 1
            if (node.get("evidence") or {}).get("status") == "not_found":
                evidence_missing[owner] += 1

        # 6 -- a cell naming a level on a term that declares none
        models = _model_index(body)
        for analysis in body.get("analyses") or []:
            if not isinstance(analysis, dict):
                continue
            terms = terms_in_scope(analysis.get("model_estimation"), models)
            effect = analysis.get("effect")
            if not isinstance(effect, dict):
                continue
            for cell in effect.get("cells") or []:
                if not isinstance(cell, dict):
                    continue
                term_id, level = cell.get("term"), values.read(cell.get("level"))
                if not isinstance(term_id, str) or not isinstance(level, str):
                    continue
                term = terms.get(term_id)
                if term is None or str(values.read(term.get("type"))) != "continuous":
                    continue
                if [
                    n
                    for n in (
                        values.read(e.get("level")) for e in (term.get("levels") or []) if isinstance(e, dict)
                    )
                    if isinstance(n, str)
                ]:
                    continue
                name, want = fold(values.read(term.get("name"))), fold(level)
                if want and (want == name or want in name or name in want):
                    continuous_level["restates the term's own name"] += 1
                elif want in DIRECTION_WORDS:
                    continuous_level["a direction word, not a level"] += 1
                elif want in candidates:
                    continuous_level["categorical, and the entity is declared"] += 1
                else:
                    continuous_level["categorical, entity not declared"] += 1

        # 8 -- what a name-match fixer would write, and what it buys
        wrote += link_by_name(body)
        queryability(body, after)

        # 7 -- orphans
        ids: dict[str, str] = {}
        index_ids(body, "Study", ids)
        hit: set[str] = set()
        collect_refs(body, "Study", hit)
        for local_id, cls in ids.items():
            declared_cls[cls] += 1
            if local_id in hit:
                referenced_cls[cls] += 1

    def section(title):
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    print(f"{records:,} records")

    section("1. What the existing rules already report")
    print(f"errors {sum(errs.values()):,} over {len(set().union(*err_papers.values())):,} papers")
    for message, count in errs.most_common(10):
        print(f"  {count:6d}  [{len(err_papers[message]):4d} papers]  {message}")
    print(f"\nwarnings {sum(warns.values()):,}")
    for message, count in warns.most_common(5):
        print(f"  {count:6d}  [{len(warn_papers[message]):4d} papers]  {message}")

    section("2. Table.purpose on a table an analysis cites")
    for value, count in purpose_when_cited.most_common():
        print(f"  {count:6d}  {value}")

    section("3. FactorLevel joins")
    total = max(1, levels["total"])
    for slot in LEVEL_JOINS:
        print(f"  {slot:14} {levels[slot]:6d}  {levels[slot] / total:6.1%}")
    print(f"  {'no entity':14} {levels['categorical, no entity']:6d}  "
          f"{levels['categorical, no entity'] / total:6.1%}  (categorical only)")
    recoverable = sum(v for k, v in level_recovery.items() if k.startswith("exact match"))
    print(f"\n  of those, recoverable by an exact name match: {recoverable} "
          f"({recoverable / max(1, levels['categorical, no entity']):.0%})")
    for kind, count in level_recovery.most_common():
        print(f"    {count:6d}  {kind}")

    section("4. Shape: a scalar slot holding a list")
    print(f"  {sum(list_len.values()):,} instances; lengths " +
          ", ".join(f"{n}:{c}" for n, c in sorted(list_len.items())))
    print(f"  one-item, trivially unwrappable: {list_len[1]:,} "
          f"({list_len[1] / max(1, sum(list_len.values())):.1%})")
    for slot, count in listed.most_common(10):
        print(f"    {count:6d}  {slot}")
    print("\n  genuinely multi-valued, which is a decision and not a fix:")
    for (slot, sample), count in multi_slot.most_common(6):
        print(f"    {count:6d}  {slot:38} {list(sample)}")

    section("5. value_source 'reported' with no sentence, by slot")
    tot_m, tot_t = sum(evidence_missing.values()), max(1, sum(evidence_tot.values()))
    print(f"  {tot_m:,} of {tot_t:,} reported values ({tot_m / tot_t:.0%})")
    print(f"\n  {'slot':40} {'reported':>9} {'no sentence':>12} {'rate':>6}")
    for slot, count in evidence_missing.most_common(14):
        print(f"  {slot:40} {evidence_tot[slot]:9d} {count:12d} {count / evidence_tot[slot]:6.0%}")

    section("6. A cell naming a level on a continuous term")
    total = max(1, sum(continuous_level.values()))
    for kind, count in continuous_level.most_common():
        print(f"  {count:6d}  ({count / total:4.0%})  {kind}")

    section("8. What linking on an exact name match would write, and buy")
    print("  links written:")
    for slot, count in wrote.most_common():
        print(f"    {count:6d}  {slot}")
    print(f"    {sum(wrote.values()):6d}  total\n")
    print(f"  {'':52} {'before':>8} {'after':>8}")
    for key, denom in (
        ("levels resolved", "levels"),
        ("analyses fully resolvable from the entity graph", "analyses with a categorical contrast"),
        ("analyses that can say which condition", "analyses with a categorical contrast"),
        ("analyses that can say which arm", "analyses with a categorical contrast"),
        ("analyses that can say which occasion", "analyses with a categorical contrast"),
    ):
        print(f"  {key:52} {before[key] / max(1, before[denom]):7.1%} "
              f"{after[key] / max(1, after[denom]):7.1%}")

    section("7. Entities nothing references")
    print(f"  {'class':24} {'declared':>9} {'referenced':>11} {'orphaned':>9} {'rate':>6}")
    for cls, count in declared_cls.most_common(14):
        orphaned = count - referenced_cls[cls]
        print(f"  {cls:24} {count:9d} {referenced_cls[cls]:11d} {orphaned:9d} {orphaned / count:6.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
