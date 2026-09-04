"""Write a proposed change into a record, or say why not -- and what stops a bad write.

One entry point, `apply`, so every write a repair pass makes goes past the same refusals.
They live here rather than in a module of their own because they have exactly one caller and
share its vocabulary: a `Refusal` is what an `EditLog` carries, and an `Edit` is `apply`'s
own arguments named.

One entry point, `apply`, so every write a repair pass makes goes past the same refusals.
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
from typing import Any, Callable, Mapping, MutableMapping

from pondie.formats import values
from pondie.schema.reader import Schema


@dataclass(frozen=True)
class Edit:
    """One proposed write, and everything a guard needs to judge it."""

    record: Mapping[str, Any]
    #: The entity being edited, as it stands before the write.
    entity: Mapping[str, Any]
    slot: str
    #: What the pass wants to put there: a scalar for a value slot, a list of local_ids for
    #: a reference slot.
    value: Any

    @property
    def current(self) -> Any:
        return self.entity.get(self.slot)

    @property
    def current_value(self) -> Any:
        return values.read(self.current)


@dataclass(frozen=True)
class Refusal:
    """Why a write did not happen, in terms a reviewer can act on."""

    slot: str
    why: str
    value: Any = None


Check = Callable[[Edit], "Refusal | None"]


# ------------------------------------------------------------------------------- helpers


def _bare(text: Any) -> str:
    """Letters and digits only, for asking whether one string says what another says."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _scope_pair(slot: str) -> str:
    """The slot on the other side of a scope/regions pair, or ''.

    `Analysis.spatial_scope` says what volume was modelled and `Analysis.regions` names what
    it was restricted to; `InferenceSettings.correction_scope` and `correction_regions` say
    the same about the correction. Each pair is meaningless one half at a time.
    """
    return {"spatial_scope": "regions", "regions": "spatial_scope",
            "correction_scope": "correction_regions",
            "correction_regions": "correction_scope"}.get(slot, "")


UNRESTRICTED = frozenset({"whole_brain", "whole brain", "searchlight"})


# -------------------------------------------------------------------------------- guards


def refuses_truncation(edit: Edit) -> Refusal | None:
    """Shortening a value is not correcting it.

    22952599: `definition` went from "compared to traumatized controls." to "compared to
    traumatized", and lost its evidence doing so. The same path had correctly *extended* a
    definition on 23021615, so the direction is what distinguishes them -- an edit has to
    add something.
    """
    old, new = edit.current_value, edit.value
    if not isinstance(old, str) or not isinstance(new, str):
        return None
    if _bare(new) and _bare(new) in _bare(old) and _bare(new) != _bare(old):
        return Refusal(edit.slot, "shortens the value it replaces", new)
    return None


def refuses_shortening_a_list(edit: Edit) -> Refusal | None:
    """A single value does not replace several.

    16701903 acquires two sequences -- MP-RAGE at TE 4.4 ms and FLASH at TE 5 ms -- and
    `echo_time_seconds` held both. A one-value correction dropped the FLASH echo time, which
    is a truncated definition wearing a different shape.

    That record should be two acquisitions rather than one (`rules.check_one_protocol_per_
    acquisition` reports it), so this guard defends an encoding that is itself wrong. It is
    still the right refusal: a repair pass is not where that is decided.
    """
    old = edit.current_value
    if isinstance(old, list) and len(old) > 1 and not isinstance(edit.value, list):
        return Refusal(edit.slot, "replaces several values with one", edit.value)
    return None


def refuses_losing_the_warrant(edit: Edit) -> Refusal | None:
    """An edit keeps its evidence or says more; it does not quietly drop both.

    12853571: `correction_scope` went from "whole volume analyzed and a priori small
    volumes" -- cited, and true, the paper did both -- to the bare enum "whole_brain", which
    drops the small-volume half and has no sentence behind it. Coercion to a permissible
    value is not a correction.

    The old spans are kept when they still contain the new value, which is what lets a
    genuine extension through: on 23021615 the restored full sentence was already the span.
    """
    node = edit.current
    if not isinstance(node, Mapping):
        return None
    if (node.get("evidence") or {}).get("status") != "present":
        return None
    want = _bare(edit.value)
    for group in (node.get("evidence") or {}).get("sets") or []:
        for span in group.get("spans") or []:
            if want and want in _bare(span.get("text", "")):
                return None
    return Refusal(edit.slot, "loses the span that warranted the value it replaces",
                   edit.value)


def refuses_an_unrestricted_scope_beside_regions(edit: Edit) -> Refusal | None:
    """Whole-brain and searchlight restrict to no region, in both directions.

    11950456, an STG volumetric study, acquired `correction_scope: whole_brain` beside
    `correction_regions: [r_stg]`; 19996042 and 16038682 acquired `roi` beside nothing. The
    pair states one procedure, so half of it alone is not a claim.
    """
    other = _scope_pair(edit.slot)
    if not other:
        return None
    if edit.slot.endswith("scope"):
        scope, regions = str(edit.value).strip().lower(), edit.entity.get(other)
        if scope in UNRESTRICTED and regions:
            return Refusal(edit.slot, f"{other} is not empty", edit.value)
        if scope in {"roi", "region of interest"} and not regions:
            return Refusal(edit.slot, f"{other} is empty, so the restriction is unnamed",
                           edit.value)
        return None
    # The regions side reads the *scope* slot, not its own: what makes naming a region
    # wrong is the scope beside it saying the search was not restricted.
    scope = str(values.read(edit.entity.get(other)) or "").strip().lower()
    if scope in UNRESTRICTED and edit.value:
        return Refusal(edit.slot, f"{other} is not restricted to a region", edit.value)
    return None


def refuses_a_self_reference(edit: Edit) -> Refusal | None:
    """Nothing is its own input, its own mirror, or a term it is a product of.

    27082610 and 19942229: `inputs_from` resolved to the model being edited, because the
    label the reply gave matched the entity whose slot it was filling. The three slots whose
    range is their own class all mean a *different* instance of it.
    """
    own = edit.entity.get("local_id")
    named = edit.value if isinstance(edit.value, list) else [edit.value]
    if own and own in named:
        return Refusal(edit.slot, "names the entity it is written on", own)
    return None


def refuses_orphaning_cell_terms(edit: Edit) -> Refusal | None:
    """An analysis keeps reaching the terms its cells name.

    19942229: `a_793_1` was repointed at a model that does not reach `trm_group_r_nr`, so
    two of its cells named a term nothing could resolve. The cells are the analysis's own
    structure and the model reference is a pointer, so the pointer is what gives way.
    """
    if edit.slot != "model_estimation" or not edit.value:
        return None
    target = edit.value[0] if isinstance(edit.value, list) else edit.value
    named = {cell.get("term")
             for cell in ((edit.entity.get("effect") or {}).get("cells") or [])
             if isinstance(cell, Mapping) and isinstance(cell.get("term"), str)}
    if not named:
        return None
    from pondie.extraction.record.effect import terms_in_scope

    models = {m.get("local_id"): m for m in edit.record.get("model_estimations") or []
              if isinstance(m, Mapping) and m.get("local_id")}
    reachable = set(terms_in_scope(target, models))
    missing = named - reachable
    if missing:
        return Refusal(edit.slot,
                       f"would leave cells naming {', '.join(sorted(missing))}, which this "
                       f"model does not reach", target)
    return None


#: Every refusal, enumerated so a reviewer can read the whole list, and run in full so one
#: write reports every reason it was rejected rather than the first. Ordering is not a
#: property of the guards: `apply` writes reference slots before value slots, which is what
#: the paired-slot guards depend on, and that ordering is stated there.
GUARDS: tuple[Check, ...] = (
    refuses_truncation,
    refuses_shortening_a_list,
    refuses_losing_the_warrant,
    refuses_an_unrestricted_scope_beside_regions,
    refuses_a_self_reference,
    refuses_orphaning_cell_terms,
)


def refusals(edit: Edit, checks: tuple[Check, ...] = GUARDS) -> list[Refusal]:
    """Every reason this write should not happen. Empty means write it."""
    return [refusal for check in checks if (refusal := check(edit)) is not None]


@dataclass
class EditLog:
    """What one pass did to one entity."""

    written: list[tuple[str, Any]] = dataclass_field(default_factory=list)
    refused: list[Refusal] = dataclass_field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.written)


def label_of(entity: Mapping[str, Any]) -> str:
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
    container = sch.containers().get(target_class, "")
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

    taken = {e.get("local_id") for e in record.get(_container(sch, class_name)) or []
             if isinstance(e, Mapping)}
    label = str(proposal.get("name") or proposal.get("definition") or "").strip()
    if not label:
        return None, f"the proposed {class_name} has no name to build an id from"
    # Before minting, not after: id stems differ where labels agree, so "CAPS total score"
    # and "clinician-administered PTSD scale (CAPS)" collided nowhere and became two records
    # that analyses then linked to separately.
    existing = next((e for e in record.get(_container(sch, class_name)) or []
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
            # `generated`: this sentence is the pass's, not the paper's, and stamping it
            # `reported` would claim the paper said it -- and would then send it to the
            # checker as a claim about the paper.
            "explanation": _wrap(
                "Reported in prose with no table row group to decompose.", text,
                source="generated"),
        }
    return out


def _container(sch: Schema, class_name: str) -> str:
    return sch.containers().get(class_name, class_name.lower())


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
                log.refused.append(Refusal(name, "will not fit the slot", proposed))
                continue

        if refused := refusals(Edit(record, entity, name, value)):
            log.refused.extend(refused)
            continue
        entity[name] = value if kinds[name] == "reference" else _wrap(value, text)
        log.written.append((name, value))
    return log


def _multivalued(sch: Schema, class_name: str, slot: str) -> bool:
    attribute = sch.attributes(class_name).get(slot)
    return bool(attribute is not None and attribute.multivalued)


def _wrap(value: Any, text: str, source: str = "reported") -> dict:
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
            "value_source": source, "evidence": evidence}
