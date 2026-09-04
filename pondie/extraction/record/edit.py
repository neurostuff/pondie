"""Write a proposed change into a record, or say why not.

One entry point, `apply`, so that every write a repair pass makes goes past the same guards.
It returns what it did and what it refused rather than logging, because a pass that declines
silently is indistinguishable from one that was never asked -- which is how an admission gate
turned away an unknown number of candidates for the life of a run.

The order matters and is stated here rather than left to the caller: reference slots are
written before value slots, because the paired-slot guards read a sibling. Doing it the other
way round is what let `correction_scope: whole_brain` land beside a named `correction_regions`
-- the guard on the regions side ran while the scope was still unset, and the scope was then
set with nothing left to check it against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Mapping, MutableMapping

from pondie.extraction.record import guards
from pondie.formats import values
from pondie.schema.reader import Schema


@dataclass
class EditLog:
    """What one pass did to one entity."""

    written: list[tuple[str, Any]] = dataclass_field(default_factory=list)
    refused: list[guards.Refusal] = dataclass_field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.written)


def label_of(entity: Mapping[str, Any], _class_name: str = "") -> str:
    """What a source would call this entity: its name, else its definition, else its id."""
    for slot in ("name", "definition", "model_type", "type"):
        text = values.read(entity.get(slot))
        if isinstance(text, str) and text.strip():
            return text.strip()
    return str(entity.get("local_id") or "")


def same_entity(one: str, other: str, abbreviations: Any = None) -> bool:
    """Do two labels name one thing, once the paper's abbreviations are spelled out?

    "CAPS total score" and "clinician-administered PTSD scale (CAPS)" are one instrument
    written twice, and normalized equality alone minted a second copy of it that analyses
    then linked to. Expansion is repeated because an expansion can itself contain an
    abbreviation, and compared as word sets because expanding duplicates text.

    One name containing the other's words is enough; "PTSD checklist" and "PTSD symptom
    scale" share their expansion and still differ on the rest, so they stay two instruments.
    That discrimination is what a hand-kept list of "generic" acronyms was standing in for,
    and got right only by being told PTSD in advance.
    """
    left, right = _words(one, abbreviations), _words(other, abbreviations)
    small, large = (left, right) if len(left) <= len(right) else (right, left)
    # Two words at least: "scale" alone is a subset of half the instruments in any paper.
    return len(small) >= 2 and small <= large


def _words(label: str, abbreviations: Any, rounds: int = 3) -> set[str]:
    text = label or ""
    for _ in range(rounds):
        expanded, grew = [], False
        for token in re.findall(r"[A-Za-z0-9-]+", text):
            full = abbreviations.expand(token) if abbreviations and token.isupper() else None
            expanded.append(full or token)
            grew |= bool(full)
        text = " ".join(expanded)
        if not grew:
            break
    return set(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def resolve(record: Mapping[str, Any], sch: Schema, target_class: str, named: Any,
            abbreviations: Any = None) -> list[str]:
    """Names the model gave -> local_ids of that class, dropping what does not resolve.

    Matching is on the label because a name is what the paper prints and a local_id is not.
    It is also on the class: without that, `inputs_from` -- which the schema says targets a
    ModelEstimation -- resolved to a Condition nested under a Task because the name matched,
    and a type-violating link is worse than a missing one.
    """
    from pondie.extraction.recall import CONTAINER

    container = CONTAINER.get(target_class, "")
    pool = {entity.get("local_id"): label_of(entity)
            for entity in record.get(container) or [] if isinstance(entity, Mapping)}
    out: list[str] = []
    for raw in (named if isinstance(named, list) else [named]):
        if not isinstance(raw, str) or not raw.strip():
            continue
        want = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
        hit = raw if raw in pool else next(
            (lid for lid, label in pool.items()
             if re.sub(r"[^a-z0-9]+", " ", label.lower()).strip() == want), None)
        if hit is None:
            hit = next((lid for lid, label in pool.items()
                        if same_entity(label, raw, abbreviations)), None)
        if hit and hit not in out:
            out.append(hit)
    return out


def create(sch: Schema, record: MutableMapping[str, Any], class_name: str,
           proposal: Mapping[str, Any], text: str = "",
           abbreviations: Any = None) -> tuple[dict | None, str]:
    """A new entity from `proposal`, or `(None, why not)`.

    Two conditions, both read from the schema rather than chosen here.

    An entity must be constructible as *valid*: every required slot the class declares has
    to be fillable from the proposal. A Region needs `definition_method`, which the template
    asks for and the model answers; an Analysis needs eight slots including `effect`, a
    nested structure no flat template can carry, so an Analysis proposal is refused. That is
    a derivation, not a policy -- and it is reported by slot, so making analyses creatable
    is a matter of supplying what the message names.

    And an id must be one this pass may choose. `ids.DERIVED` says Analysis and Table ids
    come from the table parse, so an invented one would not match the analysis the parse
    produced.
    """
    from pondie.extraction.record import ids

    taken = {e.get("local_id") for e in record.get(_container(class_name)) or []
             if isinstance(e, Mapping)}
    label = str(proposal.get("name") or proposal.get("definition") or "").strip()
    if not label:
        return None, f"the proposed {class_name} has no name to build an id from"
    # Before minting, not after: id stems differ where labels agree, so "CAPS total score"
    # and "clinician-administered PTSD scale (CAPS)" collided nowhere and became two records
    # that analyses then linked to separately.
    existing = next((e for e in record.get(_container(class_name)) or []
                     if isinstance(e, Mapping)
                     and same_entity(label_of(e), label, abbreviations)), None)
    if existing is not None:
        return None, (f"the record already holds this {class_name} as "
                      f"{existing.get('local_id')!r}")
    local_id = ids.mint(class_name, label, taken)
    if local_id is None:
        return None, f"{class_name} ids come from the table parse, not from a proposal"

    entity: dict[str, Any] = {"local_id": local_id}
    for name, _slot, kind in sch.iter_slots(class_name):
        if name in ("local_id", "id") or name not in proposal or kind == "reference":
            continue
        value = values.shape(sch, class_name, name, proposal[name])
        if value is not None:
            entity[name] = _wrap(value, text)

    entity.update(_nested_defaults(sch, record, class_name, proposal, text))

    required = {name for name, slot, _kind in sch.iter_slots(class_name)
                if slot.required and name not in ("local_id", "id")}
    missing = sorted(required - set(entity) - _referenced_slots(sch, class_name))
    if missing:
        return None, f"{class_name} would be missing {', '.join(missing)}"
    return entity, ""


def _referenced_slots(sch: Schema, class_name: str) -> set[str]:
    """Required slots that are references, which `apply` fills afterwards from the proposal."""
    return {name for name, _slot, kind in sch.iter_slots(class_name) if kind == "reference"}


def _nested_defaults(sch: Schema, record: Mapping[str, Any], class_name: str,
                     proposal: Mapping[str, Any], text: str) -> dict:
    """The nested required slots a proposal can honestly supply.

    Only two, and only for Analysis. `groups` is a list of AnalysisGroup, each of which needs
    nothing but the Group it names -- so an analysis that says which groups it compared can
    have them. `details` has a declared escape hatch: `NotStructurableDetails` is what the
    schema provides for an analysis whose method has no stable structured decomposition, and
    a contrast reported in a sentence is exactly that.

    `effect` and `groups` are not here and are not invented. A Cell needs a ModelTerm to
    point at and a direction, and an AnalysisGroup needs the Group it names; a flat template
    carries neither, and guessing would put a fabricated contrast structure in the record.
    So an Analysis proposal is refused today, by name -- which is what turns "should the pass
    write analyses" into "have the proposer return cells and groups, and it will".
    """
    if class_name != "Analysis":
        return {}
    out: dict[str, Any] = {}
    # `groups` is not built here. It is a nested AnalysisGroup list, so `nu_type` never asks
    # for it and a proposal cannot carry it -- an earlier version read a key the proposer
    # cannot emit, which made this branch dead and the refusal below the only real outcome.
    # Filling it needs the proposer to return cells and groups, which a flat template cannot.
    if proposal.get("definition"):
        out["details"] = {
            "details_type": "NotStructurableDetails",
            "reason": "insufficient_standard_structure",
            "explanation": _wrap(
                "Reported in prose with no table row group to decompose.", text),
        }
    return out


def _container(class_name: str) -> str:
    from pondie.extraction.recall import CONTAINER

    return CONTAINER.get(class_name, class_name.lower())


def apply(sch: Schema, record: MutableMapping[str, Any], class_name: str,
          entity: MutableMapping[str, Any], proposal: Mapping[str, Any],
          text: str = "", abbreviations: Any = None) -> EditLog:
    """Write the slots of `proposal` this entity may take. Returns what happened."""
    log = EditLog()
    kinds = {name: kind for name, _slot, kind in sch.iter_slots(class_name)}
    ranges = {name: slot.range for name, slot, _kind in sch.iter_slots(class_name)}

    # References first: the paired-slot guards read a sibling, and a sibling written after
    # the guard has run is a sibling the guard did not see.
    ordered = sorted(proposal, key=lambda name: kinds.get(name) != "reference")
    for name in ordered:
        if name in ("local_id", "id") or name not in kinds:
            continue
        proposed = proposal[name]
        if proposed in (None, "", []):
            continue
        if kinds[name] == "nested":
            # A nested slot holds objects with their own fields -- AnalysisGroup, Cell,
            # ModelTerm. `cast` would stringify one, so it is left to whatever builds it.
            continue
        if kinds[name] == "reference":
            resolved = resolve(record, sch, str(ranges.get(name) or ""), proposed,
                               abbreviations)
            if not resolved:
                continue
            existing = entity.get(name) or []
            existing = list(existing) if isinstance(existing, list) else [existing]
            # Union, never replacement: a model naming more entities is not a model saying
            # the existing link was wrong. Replacing dropped `asm_caps` from the analysis
            # that correlates with the CAPS total score.
            merged = existing + [r for r in resolved if r not in existing]
            value: Any = merged if _multivalued(sch, class_name, name) else merged[0]
            if value == entity.get(name):
                continue
        else:
            value = values.shape(sch, class_name, name, proposed)
            if value is None:
                log.refused.append(guards.Refusal(name, "will not fit the slot", proposed))
                continue

        edit = guards.Edit(sch, record, entity, class_name, name, value)
        if refusals := guards.refusals(edit):
            log.refused.extend(refusals)
            continue
        entity[name] = value if kinds[name] == "reference" else _wrap(value, text)
        log.written.append((name, value))
    return log


def _multivalued(sch: Schema, class_name: str, slot: str) -> bool:
    attribute = sch.attributes(class_name).get(slot)
    return bool(attribute is not None and attribute.multivalued)


def _wrap(value: Any, text: str) -> dict:
    """A wrapper whose evidence says what was actually established.

    `not_found` rather than `not_applicable`: a sentence should exist for a value read off a
    paper, and saying none applies would be a claim the pass has not earned.
    """
    from pondie.extraction.record import spans as span_tools

    evidence: dict[str, Any] = {"status": "not_found"}
    quote = str(value)
    if text and len(quote) >= 20:
        try:
            span = span_tools.resolve(text, quote).as_record()
            span_tools.verify(text, span)
            evidence = {"status": "present",
                        "sets": [{"source": "repair_pass", "spans": [span]}]}
        except Exception:
            pass
    return {"extraction_status": "extracted", "value": value,
            "value_source": "reported", "evidence": evidence}
