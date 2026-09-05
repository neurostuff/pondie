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

import functools
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


#: Reference slots where one target belongs to one entity, so the same list arriving on a
#: second entity of the class in one sweep is a copy rather than a reading.
#:
#: Named rather than derived, because what makes a slot exclusive is what its description
#: says and no rule reads that. `Group.diagnostic_instrument` is "The study assessment that
#: established THIS group's defining condition" -- on 18823721 the pass wrote the same four
#: questionnaires to the patients and the controls, and two of the four were administered to
#: the patients only.
#:
#: Sharing is the normal case everywhere else and must stay legal: over twelve papers the
#: pass made fifteen shared-target writes -- six analyses on one SCID, three on one cue task,
#: two model estimations on one preprocessing -- and every one of them is correct. A blanket
#: rule would have refused all fifteen and caught neither of the two real errors.
EXCLUSIVE_REFERENCES: frozenset[tuple[str, str]] = frozenset({
    ("Group", "diagnostic_instrument"),
})


# -------------------------------------------------------------------------------- guards


def refuses_truncation(edit: Edit) -> Refusal | None:
    """Shortening a value is not correcting it.

    22952599: `definition` went from "compared to traumatized controls." to "compared to
    traumatized", and lost its evidence doing so. The same path had correctly *extended* a
    definition on 23021615, so the direction is what distinguishes them -- an edit has to
    add something.
    """
    old, new = _one(edit.current_value), _one(edit.value)
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
    if not isinstance(old, list):
        return None
    # Lengths, not shapes. The guard used to ask whether the new value was a list; once
    # `shape` began resolving multiplicity through the wrapper it always is, which switched
    # this off -- on `MRI.echo_time_seconds` among others, the slot it was written for.
    new = edit.value if isinstance(edit.value, list) else [edit.value]
    if len(new) < len(old):
        return Refusal(edit.slot, "drops values the record already held", edit.value)
    return None


def _one(value: Any) -> Any:
    """A one-element list unwrapped, so a list/string comparison is still a comparison."""
    return value[0] if isinstance(value, list) and len(value) == 1 else value


def refuses_losing_the_warrant(edit: Edit) -> Refusal | None:
    """An edit keeps its evidence or says more; it does not quietly drop both.

    12853571: `correction_scope` went from "whole volume analyzed and a priori small
    volumes" -- cited, and true, the paper did both -- to the bare enum "whole_brain", which
    drops the small-volume half and has no sentence behind it. Coercion to a permissible
    value is not a correction.

    The old spans are kept when they still contain the new value, which is what lets a
    genuine extension through: on 23021615 the restored full sentence was already the span.
    """
    if _inherited(edit.current, edit.value) is not None:
        return None
    node = edit.current
    if not isinstance(node, Mapping):
        return None
    if (node.get("evidence") or {}).get("status") != "present":
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


@functools.lru_cache(maxsize=1)
def ADDRESSABLE() -> frozenset[str]:
    """Classes a record addresses by `local_id`, read from the extraction projection."""
    from pondie.extraction.record.validate import EXTRACTION_SCHEMA
    from pondie.schema import reader

    schema = reader.load(EXTRACTION_SCHEMA)
    return frozenset(name for name in schema.classes
                     if "local_id" in schema.attributes(name))


#: Below this a derived label is a fragment, not a name. `mea_fa` would otherwise offer "fa",
#: which appears inside "factor" and "surface" -- and `same_entity` merging on that is the
#: opposite of the duplicate it exists to prevent.
_SHORTEST_DERIVED = 4


def from_local_id(local_id: str) -> str:
    """A readable label out of a minted id: `dev_siemens_trio` -> "siemens trio".

    `Measure`, `Acquisition`, `Device` and `ModelEstimation` declare no `name`, and only
    `Device` has no usable fallback either, so `label_of` fell through to the raw id for a
    third of all entities -- 336 of 1,032 over eighty papers. Nothing matches
    `dev_siemens_trio` in a paper, so every mechanism that reads a label was working blind
    on those: `resolve` turning a proposed name into an id, `same_entity` deduplicating, and
    the locator's bonus for a sentence that mentions the entity.

    The ids are minted from content, so the content is recoverable. Of the 333 whose label
    was absent from their paper, 57.7% match verbatim once derived and a further 35.7% have
    every token present. Short results are refused rather than guessed at.
    """
    from pondie.extraction.record.ids import PREFIX

    text = local_id.strip()
    for prefix in sorted(PREFIX.values(), key=len, reverse=True):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = re.sub(r"[_\-]+", " ", text).strip()
    return text if len(text) >= _SHORTEST_DERIVED else ""


def label_of(entity: Mapping[str, Any]) -> str:
    """What a source would call this entity: its name, else its definition, else its id.

    The id is read through `from_local_id` rather than returned raw, because a minted id is
    not a string any paper contains. It is a label for matching and never a name to store:
    `acq_fmri` yields "fmri", which is the modality rather than what the paper calls that
    acquisition.
    """
    for slot in ("name", "definition", "model_type", "type"):
        text = values.read(entity.get(slot))
        if isinstance(text, str) and text.strip():
            return text.strip()
    local_id = str(entity.get("local_id") or "")
    return from_local_id(local_id) or local_id


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
    if class_name not in ADDRESSABLE():
        # `local_id` is the extraction projection's, not storage's, so the question is asked
        # of the schema a record is written against. `ExternalDataset` has no local_id there,
        # and an entity minted for it carries an attribute its class does not declare.
        return None, f"{class_name} takes no local_id, so a new one cannot be addressed"
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
        # `nested` alongside `reference`: a nested slot holds objects, and `shape` renders
        # one as its own repr. A Task minted on 12860777 came out with `conditions` as a
        # wrapper whose value was a list of stringified dicts -- valid JSON, nothing the
        # schema declares, and the one finding repair still introduced across fifteen
        # records. `_nested_defaults` supplies the two nested slots a proposal can honestly
        # fill, and `apply` writes the rest through `_nested` once the entity exists.
        if name in ("local_id", "id") or name not in proposal \
                or kind in ("reference", "nested"):
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
          text: str = "", abbreviations: Any = None,
          claimed: MutableMapping[tuple[str, str, tuple[str, ...]], str] | None = None
          ) -> EditLog:
    """Write the slots of `proposal` this entity may take. Returns what happened.

    `claimed` is the sweep's memory of which entity already took a set of targets on an
    `EXCLUSIVE_REFERENCES` slot. It is a caller's dict rather than state here because the
    thing being refused is a property of the sweep and not of the edit: a `Check` sees one
    entity and one value, and what is wrong with the second write is only visible beside the
    first. `repair._sweep` holds one per class.
    """
    log = EditLog()
    from pondie.extraction import recall

    quotes = proposal.get(recall.QUOTES) or {}
    if not isinstance(quotes, Mapping):
        quotes = {}
    # The class the entity says it is, not the one its container declares. An acquisition is
    # an `MRI` by type designator, and `magnetic_field_strength_tesla` is a slot of that
    # subclass -- written against the base class it is an attribute `Acquisition` does not
    # have. `Validator` resolves the designator for the same reason.
    class_name = sch.designated_type(entity, class_name)
    # `class_name` was resolved from the designator, so every other slot in this proposal is
    # checked against that subclass. Writing the designator would re-type the entity after
    # the fact, and it is declared a plain string -- the wrap below put a dict where the
    # schema wants "MRI", which then read back as no subclass at all.
    designator = sch.type_designator(class_name)
    kinds = {name: kind for name, _slot, kind in sch.iter_slots(class_name)}
    ranges = {name: slot.range for name, slot, _kind in sch.iter_slots(class_name)}

    # References first: the paired-slot guards read a sibling, and a sibling written after
    # the guard has run is a sibling the guard did not see.
    ordered = sorted(proposal, key=lambda name: kinds.get(name) != "reference")
    for name in ordered:
        if name in ("local_id", "id", designator, recall.QUOTES) or name not in kinds:
            continue
        cited = str(quotes.get(name) or "")
        proposed = proposal[name]
        if proposed in (None, "", []):
            continue
        if kinds[name] == "nested":
            # A nested slot holds objects with their own fields. Most are structures a flat
            # reply cannot express -- `Analysis.effect` carries cells carrying statistics --
            # and `cast` would stringify one, so those are still left to whatever builds
            # them. `Task.conditions` is not that: a Condition is an id, a name, a kind and
            # a description, and `recall.flat` says which classes are of that shape.
            #
            # Reached late. The template began offering conditions before this could write
            # them, so the proposer was asked and its answer discarded -- and the one field
            # that says a state was a control could still only come from the extraction pass.
            written = _nested(sch, str(ranges.get(name) or ""), entity, name, proposed,
                              text, log)
            if written:
                log.written.append((name, written))
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
            # On what is added, not on the whole slot: an entity that already held a target
            # is not claiming it again, and the copy this refuses is an id arriving for the
            # first time on a second entity.
            added = tuple(sorted(r for r in resolved if r not in existing))
            added, taken = _unclaimed(class_name, name, added, entity, claimed)
            if taken:
                log.refused.append(Refusal(
                    name, f"{', '.join(taken)} already belongs to "
                          f"{', '.join(sorted({claimed[(class_name, name, t)] for t in taken}))}"
                          f", and this slot names what belongs to one entity", list(taken)))
                merged = [r for r in merged if r not in taken]
                if not merged:
                    continue
                value = merged if _multivalued(sch, class_name, name) else merged[0]
                if value == entity.get(name):
                    continue
        else:
            value = values.shape(sch, class_name, name, proposed)
            if value is None:
                log.refused.append(Refusal(name, "will not fit the slot", proposed))
                continue

        current = entity.get(name)
        if kinds[name] != "reference" and values.is_field(current) \
                and _same(sch, class_name, name, values.read(current), value):
            # Re-proposing what is already there is not an edit. Writing it anyway rebuilt
            # the wrapper and lost its warrant: 26 fields on 18823721 kept their value and
            # went `present` -> `not_found`, `reported` -> `generated`.
            log.refused.append(Refusal(name, "already recorded with this value", value))
            continue
        if refused := refusals(Edit(record, entity, name, value)):
            log.refused.extend(refused)
            continue
        if kinds[name] == "reference":
            entity[name] = value
            if claimed is not None and (class_name, name) in EXCLUSIVE_REFERENCES:
                for target in added:
                    claimed[(class_name, name, target)] = str(entity.get("local_id") or "")
        else:
            written = _wrap(value, text, quote=cited)
            if (kept := _inherited(current, value)) is not None:
                written["evidence"], written["value_source"] = kept
            elif text and _closed_vocabulary(sch, class_name, name, value) \
                    and written["evidence"]["status"] != "present" \
                    and not _in_document(value, text):
                # Only where the slot holds a closed vocabulary. Measured over 139 refusals:
                # the 46 on `non_analysis_content` carry the whole gain -- 16 of the pass's
                # 56 introduced findings -- while the free-text and numeric refusals cost
                # real values and no findings, and the six on `is_healthy` cost two correct
                # writes and took a paper from passing to failing.
                #
                # The distinction is the one `_nested` already draws: an enum term is
                # vocabulary, not a quote. A token chosen from a list is either in the paper
                # or invented, so "not in the document" decides it. A number and a sentence
                # are things the paper says in its own words -- "7.5 minutes" for 450
                # seconds, a definition reworded -- and the same test is unsound for them.
                log.refused.append(Refusal(
                    name, "nothing in the paper places this vocabulary term", value))
                continue
            entity[name] = written
        log.written.append((name, value))
    return log


def _unclaimed(class_name: str, slot: str, added: tuple[str, ...],
               entity: Mapping[str, Any],
               claimed: Mapping[tuple[str, str, str], str] | None
               ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`added`, split into what this entity may take and what already belongs elsewhere.

    Not a `Check`: the guards judge one edit against the record, and this judges an edit
    against what the same sweep already wrote. The first write is legitimate and only the
    later one is a copy, which needs the caller's memory rather than the record.

    Per target rather than per list, because the copy does not arrive as a copy. On 21118656
    three groups took overlapping subsets of the same three interviews -- [CAPS, MINI, vivo],
    [CAPS, MINI], [MINI] -- and a rule keyed on the whole list saw three different lists and
    let all three through. On 18823721, where the list was identical, it fired.

    The unclaimed part is still written rather than the whole write refused: whichever entity
    the sweep happens to reach first takes the shared target, and refusing everything after
    it would cost a correct link for an accident of ordering.
    """
    if claimed is None or not added or (class_name, slot) not in EXCLUSIVE_REFERENCES:
        return added, ()
    own = str(entity.get("local_id") or "")
    taken = tuple(t for t in added
                  if claimed.get((class_name, slot, t)) not in (None, own))
    return tuple(t for t in added if t not in taken), taken


def _multivalued(sch: Schema, class_name: str, slot: str) -> bool:
    return sch.is_multivalued(class_name, slot)


def _nested(sch: Schema, inner: str, entity: MutableMapping[str, Any], slot: str,
            proposed: Any, text: str, log: Log) -> int:
    """Merge a list of nested objects into `entity[slot]`. Returns how many fields landed.

    Merge and never replace, for the reason the reference path unions rather than assigns: a
    model returning three conditions is not a model saying the fourth was wrong. An object
    already present is matched by `local_id`, then by label, and only its *empty* fields are
    filled -- an extracted value with a sentence behind it outranks a proposal without one,
    and that ordering is what stops a second pass quietly rewriting the first.
    """
    from pondie.extraction.recall import flat

    if not inner or inner not in sch or not flat(sch, inner):
        return 0
    items = proposed if isinstance(proposed, list) else [proposed]
    items = [x for x in items if isinstance(x, Mapping)]
    if not items:
        return 0

    current = entity.get(slot)
    current = list(current) if isinstance(current, list) else []
    by_id = {str(x.get("local_id") or ""): x for x in current if isinstance(x, Mapping)}
    by_label = {label_of(x).lower(): x for x in current if isinstance(x, Mapping)}

    landed = 0
    for item in items:
        local_id = str(item.get("local_id") or "").strip()
        named = str(values.read(item.get("name")) or "").strip()
        target = by_id.get(local_id) or by_label.get(named.lower())
        if target is None:
            # A nested object is created only where the record has none of that name. The
            # sweep's job here is to complete what `satisfy` left thin, not to invent a
            # condition the paper never ran.
            continue
        for field_name, raw in item.items():
            if field_name in ("local_id", "id") or raw in (None, "", []):
                continue
            if values.read(target.get(field_name)) not in (None, "", []):
                continue                      # what is already there, with its evidence
            value = values.shape(sch, inner, field_name, raw)
            if value is None:
                log.refused.append(Refusal(f"{slot}.{field_name}",
                                           "will not fit the slot", raw))
                continue
            written = _wrap(value, text)
            # A nested field has to be placeable in the paper, which in practice admits
            # prose and refuses classifications: an enum term is vocabulary, not a quote, and
            # `_wrap` will not even look for one under twenty characters. That is the line
            # this should draw. `satisfy` classifies, having read the whole document -- it
            # got `Neutral` right on 16038771 -- and this sweep, asked the same question from
            # a template, answered `fixation` for three picture-viewing conditions. Honestly
            # labelled `generated` with no sentence, and wrong three times in four.
            if written["evidence"]["status"] != "present":
                log.refused.append(Refusal(
                    f"{slot}.{field_name}",
                    "nothing in the paper places this value, and a nested guess is not "
                    "worth the risk of being wrong", value))
                continue
            target[field_name] = written
            landed += 1
    return landed


def _same(sch: Schema, class_name: str, slot: str, old: Any, new: Any) -> bool:
    """Whether these are the same value for this slot, not the same repr.

    `40` and `40.0` are one value in a float slot and two different strings, so a raw
    string comparison let an int-for-float rewrite through -- and a rewrite is what loses
    the warrant. Both sides go through `shape`, which is what the write itself would do.
    """
    try:
        return values.shape(sch, class_name, slot, old) == \
            values.shape(sch, class_name, slot, new)
    except Exception:                     # a value that will not shape is not the same one
        return False


def _inherited(current: Any, value: Any) -> tuple[dict, str] | None:
    """The evidence and provenance `current` holds, when they still warrant `value`.

    `refuses_losing_the_warrant` allows an edit exactly when a surviving span contains the
    new value, and says the old spans are then kept. They were not: the write rebuilt the
    wrapper through `_wrap`, which searches for a span only when the value is twenty
    characters or more, so every count and every mean age replaced a verified span with
    `not_found` and demoted `reported` to `generated`.

    Substring containment is the wrong test for the shapes a record actually holds. `"1"`
    sits inside `"consisted of 12 opioid-dependent patients"`, so `acquired_count: 12 -> 1`
    inherited a span that says the opposite; every integer and every short enum token was
    exposed the same way. A list is worse: `_bare` stringifies the repr, so extending
    `["SPM2"]` to `["SPM2", "FSL"]` -- both named in the paper -- looked unwarranted while
    dropping FSL looked fine.

    So each element is asked for separately, numbers are compared as numbers against the
    span's own tokens, and a short string needs a word boundary. Measured over the 218
    fields the unfixed pass destroyed across 14 records, this inherits 216 where naive
    containment inherits 217 -- the one difference being `spatial_scope: "roi"` matched
    inside a longer word, which it should not have been.
    """
    if not isinstance(current, Mapping):
        return None
    if (current.get("evidence") or {}).get("status") != "present":
        return None
    wanted = value if isinstance(value, list) else [value]
    if not wanted:
        return None
    held = values.read(current)
    if isinstance(value, list) and isinstance(held, list) and held \
            and all(item in value for item in held):
        # A strict superset removes nothing, so whatever warranted the old elements still
        # does. Asking the new ones to appear in the *same* span refuses most real
        # extensions -- a second value is usually named in a second sentence -- which
        # would leave `template_for`'s list templates unable to show any yield at all.
        return current["evidence"], str(current.get("value_source") or "reported")
    texts = [str(span.get("text", ""))
             for group in (current.get("evidence") or {}).get("sets") or []
             for span in group.get("spans") or []]
    if not texts or not all(any(_warrants(text, item) for text in texts) for item in wanted):
        return None
    return current["evidence"], str(current.get("value_source") or "reported")


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _closed_vocabulary(sch: Schema, class_name: str, slot: str, value: Any = None) -> bool:
    """Whether this WRITE is a vocabulary term rather than something the paper worded.

    Asked of the value and not of the slot. `non_analysis_content` is
    `any_of: [TableContent, string]` -- deliberately open, so the source can say something
    the vocabulary has no token for -- and it carries 46 of the 139 refusals and the whole
    of their gain. A slot-level test excludes it and keeps the slots that cost values.

    A term chosen from the list is either in the paper or invented, so "not in the document"
    decides it. The paper's own wording is not: it can be a paraphrase of what the source
    said, or a unit conversion of it.
    """
    attribute = sch.attributes(class_name).get(slot)
    if attribute is None:
        return False
    terms = {str(token).lower()
             for candidate in sch.value_ranges(attribute)
             for token in ((sch.enums[candidate].permissible_values or {})
                           if candidate in sch.enums else {})}
    if not terms:
        return False
    written = value if isinstance(value, list) else [value]
    return bool(written) and all(str(v).lower() in terms for v in written)


def _in_document(value: Any, text: str) -> bool:
    """Whether the paper states this value somewhere, span or no span.

    The locator answers "is there a sentence I can cite for this field", which is a harder
    question than "does the paper say this", and it fails on values the document plainly
    contains. Separating the two is what lets an ungrounded write be refused without
    throwing away the software names and field strengths the retriever merely missed.
    """
    if not text:
        return False
    return all(_warrants(text, item)
               for item in (value if isinstance(value, list) else [value]))


def _warrants(text: str, value: Any) -> bool:
    """Whether this span says this one value, rather than merely containing its characters."""
    if isinstance(value, bool):
        return False                       # "true" is not a thing a sentence states
    if isinstance(value, (int, float)):
        return any(float(token) == float(value) for token in _NUMBER.findall(text))
    wanted = str(value).strip().lower()
    if not _bare(wanted):
        return False
    if len(_bare(wanted)) < 4:
        # Against the sentence, not `_bare` of it: `_bare` removes the spaces, so no word
        # boundary survives to anchor on. "roi" must not match inside "heroin", and "FSL"
        # must match in "used SPM2 and FSL."
        return re.search(rf"\b{re.escape(wanted)}\b", text.lower()) is not None
    return _bare(wanted) in _bare(text)


def _wrap(value: Any, text: str, source: str = "reported", quote: str = "") -> dict:
    """A wrapper whose evidence says what was actually established.

    `not_found` rather than `not_applicable`: a sentence should exist for a value read off a
    paper, and saying none applies would be a claim the pass has not earned.
    """
    from pondie.extraction.record import spans as span_tools

    evidence: dict[str, Any] = {"status": "not_found"}
    # The proposer's own citation when it gave one, the value itself otherwise. A cited
    # sentence retires the twenty-character floor, which exists only because a bare value
    # is too short to search for safely -- and it is what makes a numeric groundable at all.
    quote = str(quote or value)
    if text and len(quote) >= 20:
        try:
            span = span_tools.resolve(text, quote).as_record()
            span_tools.verify(text, span)
            evidence = {"status": "present",
                        "sets": [{"source": "repair_pass", "spans": [span]}]}
        except Exception:
            pass
    # A value this pass could not place in the paper is one it reasoned to, and the schema
    # has a word for that. Marked `reported` regardless, it asserted the source said things
    # the source may not have: nine of thirteen findings on the first paper where the
    # proposer could write values at all were exactly that pairing -- `groups[].species`,
    # `recruitment_method`, `is_healthy`, `spatial_scope`, each `reported` with no sentence.
    honest = source if evidence["status"] == "present" else "generated"
    return {"extraction_status": "extracted", "value": value,
            "value_source": honest, "evidence": evidence}
