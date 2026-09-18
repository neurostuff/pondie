"""Making the references resolve.

The third kind of fix, and the one with the most judgement in it, so it is also where the
most is refused. A record addresses its own entities by `local_id`, and a reference that
resolves nowhere costs an analysis its inference settings or a level its cohort -- which
downstream reads as a missing field rather than a naming slip.

The rule throughout is that a fix decides nothing. `align_cell_levels` rewrites a level that
folds to exactly one declared level and `check_cell_terms` reports one that does not;
`repair_references` repoints a transcription slip and refuses a guess. Where two answers are
possible the record keeps its defect and a human is told.

Each of these is a `Repair` in `repairs.build_sequence`, which holds the order and the reason
each one sits where it does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pondie.extraction.record import direction
from pondie.extraction.record import ids
from pondie.extraction.record import spans as span_tools
from pondie.extraction.record import walk
from pondie.extraction.record.effect import terms_in_scope
from pondie.formats import values
from pondie.schema import reader
from pondie.schema.reader import Schema
from pondie.vocabularies import abbreviations
from typing import Any
import json
import re


#: The slots an entity's identity can be read off. `description` is not among them:
#: matching prose against an entity name is a substring operation, and
#: `normalize_open_fields.py` measured what containment does -- it merged `emotion
#: regulation`, the corpus's most frequent task term, into a rarer variant, with 38
#: candidate hosts.
NAMING_SLOTS = ("name", "level", "label")


#: Words a `Cell.level` carries when it is really stating which way the effect went, by
#: polarity. `Cell.direction` is the slot for that, and on a continuous term there is no
#: level for it to be.
_LEVEL_POLARITY: dict[str, str] = {
    **{w: "positive" for w in (
        "positive", "higher", "greater", "more", "increase", "increased", "up", "activation",
        "positive correlation", "positively correlated",
    )},
    **{w: "negative" for w in (
        "negative", "lower", "less", "fewer", "decrease", "decreased", "down", "deactivation",
        "negative correlation", "negatively correlated",
    )},
}


#: The minted part of a local_id, once its class prefix is off. `ids.mint` builds it from
#: the paper's own wording, which is what lets a dangling id be compared to a real name.
_PREFIXES = tuple(prefix for prefix in ids.PREFIX.values() if prefix)


#: Words that name no entity, so sharing one is not agreement.
_EMPTY_WORDS = frozenset(
    {"the", "of", "a", "and", "main", "effect", "condition", "conditions", "group",
     "groups", "task", "tasks", "all", "1", "2", "3"}
)


def align_cell_levels(body: dict[str, Any]) -> list[str]:
    """Rewrite a `Cell.level` to the declared `FactorLevel.level` it folds to.

    The join is on the string (extraction-readme.md §3 invariant 3), so `Healthy controls`
    against a declared `healthy controls` is a broken join that no reader would call a
    disagreement. Repaired only where exactly one declared level folds to it: two would make
    the choice a guess, and a guess about which condition was compared is the one thing this
    field must not contain.

    Deliberately narrow. `AD` against a declared `AD group` does *not* fold, and is left for
    `Validator.check_cell_terms` to report -- shortening a level is a claim about the paper,
    not a transcription slip.
    """

    fixed: list[str] = []
    models = {
        model.get("local_id"): model
        for model in body.get("model_estimations") or []
        if isinstance(model, Mapping)
    }

    for index, analysis in enumerate(body.get("analyses") or []):
        if not isinstance(analysis, Mapping):
            continue
        scope = terms_in_scope(analysis.get("model_estimation"), models)
        effect = analysis.get("effect")
        cells = effect.get("cells") if isinstance(effect, Mapping) else None
        for position, cell in enumerate(cells or []):
            if not isinstance(cell, Mapping) or not values.is_field(cell.get("level")):
                continue
            level = cell["level"].get("value")
            term = scope.get(cell.get("term"))
            if not isinstance(level, str) or not isinstance(term, Mapping):
                continue
            declared = [
                values.read(entry.get("level"))
                for entry in (term.get("levels") or [])
                if isinstance(entry, Mapping)
            ]
            declared = [name for name in declared if isinstance(name, str)]
            if level in declared:
                continue
            folded = span_tools.fold_label(level)
            matches = [name for name in declared if span_tools.fold_label(name) == folded]
            if len(matches) == 1:
                cell["level"]["value"] = matches[0]
                fixed.append(
                    f"analyses[{index}].effect.cells[{position}].level: "
                    f"{level!r} -> {matches[0]!r}"
                )
    return fixed


def link_entities_by_name(body: dict[str, Any], sch: Schema) -> list[str]:
    """Write a reference where a name settles which entity it means.

    1,713 of 6,860 `FactorLevel`s are on a categorical term and reach no entity at all -- a
    bare string, and `check_cell_terms` says why that costs: "the mapper joins these on the
    string". 715 of those fold to the exact name of an entity the same record already
    declares, `local_id` and all: `level 'smoking cue'` beside `cond_smoking_cue`,
    `level 'predose'` beside `tp_predose`, `level 'Exercise'` beside `arm_exercise`. Both
    halves of the join are in the record and nothing wrote it down.

    THE SCHEMA DECIDES ALL OF IT, which is the only reason this is safe to run over every
    reference slot rather than a list:

      which entities may connect   `attribute.range`, through `Schema.resolves_to` so a
                                   subclass satisfies a supertype -- a slot declaring
                                   `Acquisition` is satisfied by an `MRI`.

      which references a name may   a reference whose range is the owner's OWN kind is a
      settle                        structural relation, not an identity. Exactly three
                                    slots in the schema are -- `Analysis.mirror_of`,
                                    `ModelEstimation.inputs_from`,
                                    `ModelTerm.interaction_with` -- and they mean
                                    *sign-reversed twin of*, *fitted on the output of* and
                                    *crossed with*. A name says what a thing is and nothing
                                    about how two things relate. Matched on a name these
                                    three propose 9,151 self-links, and after excluding
                                    self, `interaction_with` still proposes 708 of which
                                    97% are a term named `group` in one model matching a
                                    term named `group` in another -- links that would
                                    fabricate interactions `check_crossings` then reports.

      what shape to write           `attribute.multivalued`: a bare id string or a bare
                                    list of them, never an `ExtractedValue`. A reference is
                                    an address inside the record, not a claim about the
                                    paper, which is why `rules.IDENTIFIERS` exempts these
                                    slots from the evidence checks.

    Exact fold match to exactly one candidate, and never to the entity itself. A further 966
    links are reachable on a single substring match and are not taken, for `NAMING_SLOTS`'
    reason. A name matching two candidates of the same kind is left alone; matching two
    *kinds* is written to both, which is the parallel-group case -- 112 of 122 are a level
    naming both the arm and the cohort allocated to it, and `Group.arm` exists to say so.
    """

    index: dict[str, list[tuple[str, str]]] = {}

    def collect(node: Any, class_name: str) -> None:
        if not isinstance(node, dict):
            return
        class_name = sch.designated_type(node, class_name)
        attributes = sch.attributes(class_name)
        if not attributes:
            return
        local_id = node.get("local_id")
        if isinstance(local_id, str) and local_id:
            for slot in NAMING_SLOTS:
                name = _fold_name(values.read(node.get(slot)))
                if name:
                    index.setdefault(name, []).append((class_name, local_id))
        for key, attribute in attributes.items():
            if key not in node or sch.classify(key, attribute) != "nested":
                continue
            if not isinstance(attribute.range, str):
                continue
            child = node[key]
            for item in child if isinstance(child, list) else [child]:
                collect(item, attribute.range)

    def is_relation(owner: str, target: str) -> bool:
        return owner == target or sch.resolves_to(owner, target) or sch.resolves_to(target, owner)

    fixed: list[str] = []

    def visit(node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict) or values.is_field(node):
            return
        class_name = sch.designated_type(node, class_name)
        mine = node.get("local_id") if isinstance(node.get("local_id"), str) else None
        names = [n for n in (_fold_name(values.read(node.get(s))) for s in NAMING_SLOTS) if n]
        for key, attribute in sch.attributes(class_name).items():
            kind = sch.classify(key, attribute)
            if kind == "nested" and isinstance(attribute.range, str) and key in node:
                child = node[key]
                for order, item in enumerate(child if isinstance(child, list) else [child]):
                    suffix = f"[{order}]" if isinstance(child, list) else ""
                    visit(item, attribute.range, f"{path}.{key}{suffix}")
                continue
            if kind != "reference" or not isinstance(attribute.range, str):
                continue
            target = attribute.range
            if not names or is_relation(class_name, target):
                continue
            current = node.get(key)
            if [x for x in (current if isinstance(current, list) else [current]) if x]:
                continue
            found = {
                local_id
                for name in names
                for owner, local_id in index.get(name, ())
                if local_id != mine and (owner == target or sch.resolves_to(owner, target))
            }
            if len(found) != 1:
                continue
            settled = found.pop()
            node[key] = [settled] if attribute.multivalued else settled
            fixed.append(f"{path}.{key}: {settled!r} from its name")

    collect(body, "Study")
    visit(body, "Study", "Study")
    return fixed


def _fold_name(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _restates(one: Any, other: Any) -> bool:
    """Whether either name says no more than the other: `BMI` against term `BMI`."""
    left, right = _fold_name(one), _fold_name(other)
    return bool(left and right) and (left in right or right in left)


def drop_redundant_cell_levels(body: dict[str, Any]) -> list[str]:
    """Drop a `Cell.level` that a continuous term cannot have and that says nothing new.

    A continuous term declares no levels, so `check_cell_terms` errors on any cell naming
    one: 1,204 errors over 527 papers, the largest error class in the corpus, and 1,185 of
    the 1,205 are on a term typed continuous. Two of the three shapes carry no information
    and are removed here.

      restates the term  547 (46%). `BMI` on term `BMI`, `age` on `age`, `pack-years` on
                         `pack-years`. A regressor's cell has no level, and naming it after
                         the term says nothing a reader did not already have.

      duplicates the     185 (16%). `positive` where `direction` already says positive.
      direction          `Cell.direction` is where the sign lives and it is already right.

    The third shape is left alone: 424 cells name a genuinely categorical level on a term
    whose `type` is wrong. Fixing that means flipping the type *and* synthesising the levels
    the term should have declared, which is a claim about the model rather than a tidy-up, so
    `check_cell_terms` keeps reporting it.

    So is the case where the two disagree. 26 cells read `direction='negative'` with
    `level='positive'`, and dropping the level there would resolve a contradiction about the
    sign of an effect by picking one side silently -- and the sign is what decides whether a
    coordinate enters an increase map or a decrease map. `check_cell_level_polarity` reports
    those instead.
    """

    from pondie.extraction.record.effect import terms_in_scope

    models = {
        model["local_id"]: model
        for model in (body.get("model_estimations") or [])
        if isinstance(model, Mapping) and isinstance(model.get("local_id"), str)
    }
    fixed: list[str] = []
    for index, analysis in enumerate(body.get("analyses") or []):
        if not isinstance(analysis, dict):
            continue
        terms = terms_in_scope(analysis.get("model_estimation"), models)
        effect = analysis.get("effect")
        if not isinstance(effect, dict):
            continue
        for position, cell in enumerate(effect.get("cells") or []):
            if not isinstance(cell, dict):
                continue
            term_id = cell.get("term")
            level = values.read(cell.get("level"))
            if not isinstance(term_id, str) or not isinstance(level, str) or not level.strip():
                continue
            term = terms.get(term_id)
            if term is None or values.read(term.get("type")) != "continuous":
                continue
            declared = [
                name
                for name in (
                    values.read(entry.get("level"))
                    for entry in (term.get("levels") or [])
                    if isinstance(entry, Mapping)
                )
                if isinstance(name, str)
            ]
            if declared:
                continue
            path = f"analyses[{index}].effect.cells[{position}].level"
            if _restates(level, values.read(term.get("name"))):
                cell.pop("level", None)
                fixed.append(f"{path}: {level!r} restated term {term_id!r} -- dropped")
                continue
            polarity = _LEVEL_POLARITY.get(str(level).strip().lower())
            if polarity is None:
                continue
            # A sign word that another term in this model declares as a LEVEL is a level.
            # 29935441 puts `level: 'negative'` on the product column
            # `PCL-by-emotion-by-task`, and `negative` is one of `trm_emotion`'s three
            # declared levels alongside `positive` and `neutral` -- an emotion, not a sign.
            # Read as a direction it became `direction: negative` and turned a factor level
            # into a polarity, which `check_crossings` then reported as a signed cell on a
            # product column. The level belongs to a term the cell does not name, which is
            # `check_cell_terms`' finding and not something to resolve here.
            if any(
                _fold_name(level) == _fold_name(values.read(entry.get("level")))
                for other in terms.values()
                for entry in (other.get("levels") or [])
                if isinstance(entry, Mapping)
            ):
                continue
            held_raw = values.read(cell.get("direction"))
            held = _LEVEL_POLARITY.get(str(held_raw or "").strip().lower())
            if held == polarity:
                cell.pop("level", None)
                fixed.append(f"{path}: {level!r} duplicated direction -- dropped")
            elif held is None and (held_raw is None or not str(held_raw).strip()):
                # Only where `direction` is genuinely absent. `undirected` is an answer, not
                # a gap, and overwriting it would replace a stated fact with an inference.
                cell["direction"] = values.wrap(
                    polarity, source="generated", evidence="not_applicable"
                )
                cell.pop("level", None)
                fixed.append(f"{path}: {level!r} moved to direction")

    return fixed


def scope_duplicate_terms(body: dict[str, Any]) -> list[str]:
    """Make two models' identically-named terms distinguishable, by their model.

    A term's id is only meaningful inside the model that declares it, but the record's id
    namespace is flat, so two estimations that both control for `term_age` collide and
    every reference naming it becomes ambiguous. `check_local_ids` reports that and is
    right to -- nothing downstream can tell which one a cell meant.

    Nothing is being picked. A cell sits in an analysis, the analysis names its model
    estimation, and a reference from that analysis can only have meant that model's term.
    The scope is information the record already carries; this writes it into the id as
    `<model>.<term>`.

    All or nothing per id. A rename that leaves one reference pointing at the old name
    turns an ambiguous reference into a dangling one, which is worse: ambiguity is
    reported against a term that exists, and a dangling reference is a slot that resolves
    to nothing. Every reference is therefore repointed first. An id with a reference that
    cannot be scoped -- one reached from no analysis, or from an analysis whose model does
    not declare it -- is reverted and left for the report.
    """

    models = [m for m in body.get("model_estimations") or [] if isinstance(m, Mapping)]
    declared_by: dict[str, set[str]] = {}
    for model in models:
        model_id = model.get("local_id")
        if not isinstance(model_id, str):
            continue
        for term in model.get("terms") or []:
            if isinstance(term, Mapping) and isinstance(term.get("local_id"), str):
                declared_by.setdefault(term["local_id"], set()).add(model_id)

    counts: dict[str, int] = {}

    def count(node: Any) -> None:
        if isinstance(node, Mapping):
            if values.is_field(node):
                return
            name = node.get("local_id")
            if isinstance(name, str):
                counts[name] = counts.get(name, 0) + 1
            for value in node.values():
                count(value)
        elif isinstance(node, list):
            for value in node:
                count(value)

    count(body)
    # Only ids whose every declaration is a term under a model. An id shared between a
    # term and a group is a conflict rather than a scope, and renaming it would hide that.
    collisions = {
        name
        for name, total in counts.items()
        if total > 1 and len(declared_by.get(name, ())) == total
    }
    if not collisions:
        return []

    def rewrite(node: Any, mapping: Mapping[str, str]) -> None:
        """Repoint every reference-shaped string, whatever slot it sits in."""
        if isinstance(node, Mapping):
            for key, value in list(node.items()):
                if key == "local_id":
                    continue
                if isinstance(value, str) and value in mapping:
                    node[key] = mapping[value]
                elif isinstance(value, list):
                    node[key] = [mapping.get(v, v) if isinstance(v, str) else v for v in value]
                    for item in node[key]:
                        rewrite(item, mapping)
                else:
                    rewrite(value, mapping)
        elif isinstance(node, list):
            for value in node:
                rewrite(value, mapping)

    scoped: list[str] = []
    for name in sorted(collisions):
        owners = declared_by[name]
        # Use each analysis's model to determine which copy it could mean.
        reachable = {
            analysis_index: values.read(analysis.get("model_estimation"))
            for analysis_index, analysis in enumerate(body.get("analyses") or [])
            if isinstance(analysis, Mapping)
        }
        snapshot = json.loads(json.dumps(body))

        for model in models:
            model_id = model.get("local_id")
            if model_id not in owners:
                continue
            mapping = {name: f"{model_id}.{name}"}
            for term in model.get("terms") or []:
                if isinstance(term, Mapping) and term.get("local_id") == name:
                    term["local_id"] = mapping[name]
            rewrite(model.get("terms"), mapping)
            for index, analysis in enumerate(body.get("analyses") or []):
                if reachable.get(index) == model_id:
                    rewrite(analysis, mapping)

        # Any surviving mention of the bare name in a non-declaration position is a
        # reference this could not scope. Put the record back rather than leave it
        # dangling.
        if json.dumps(body).count(f'"{name}"') > 0:
            body.clear()
            body.update(snapshot)
            continue
        scoped.append(f"{name!r} declared by {len(owners)} models -> scoped per model")
    return scoped


def repoint_out_of_scope_terms(body: dict[str, Any]) -> list[str]:
    """Repoint a cell at the same-named term its analysis's model can actually reach.

    A cell must name a term in its analysis's model scope -- that model's own terms plus
    those of the models it reaches through `inputs_from`. When it names a term from
    somewhere else the cell is a sign of nothing, and the validator says so.

    Repaired only where the choice is forced: exactly one term in scope carries the same
    name as the one the cell named. That is the same rule `align_cell_levels` follows for
    levels, and it covers 24 of the 105 out-of-scope references measured over 30 records.
    The other 81 are left reported -- 73 have no same-named term in scope at all, which
    means something larger is wrong than a mistyped identifier.

    A second, narrower case: the cell names an id nothing declares, and exactly one term in
    scope declares that id under a model prefix -- `trm_modality` against
    `mod_mass_univariate.trm_modality`. That is not a mistyped identifier but a disagreement
    between two passes. `demands` declares a term `trm_modality` and writes cells naming it;
    `satisfy` re-declares it once per model estimation that needs it, prefixing each with the
    model to keep them apart, and the cells written by the earlier pass are left pointing at
    an id no longer present. 21 of 183 cells over the fifteen benchmark papers, against 0 of
    186 in the reviewer's reference.

    It is worth repairing because the loss is silent and large. Nothing points at the
    surviving term, so it has no incoming edges; the benchmark's aligner reads that as
    structural disagreement and scores the pair 0.421 against a 0.45 threshold even though
    the names agree and the parent model matches. The term goes unaligned, every cell on it
    is dropped from the polarity score, and one paper loses 19 cells that way.

    Scoped by the analysis, which is what makes it unambiguous: `trm_modality` may be
    declared under two models, but only one of those is in this analysis's scope. Measured
    over the same fifteen papers, all 21 resolve to exactly one term and none to several.
    """

    models = {
        str(values.read(m.get("local_id"))): m
        for m in body.get("model_estimations") or []
        if isinstance(m, Mapping)
    }

    everywhere = {
        str(values.read(t.get("local_id"))): t
        for m in models.values()
        for t in (m.get("terms") or [])
        if isinstance(t, Mapping) and values.read(t.get("local_id"))
    }

    def name_of(term: Any) -> str:
        return str(values.read((term or {}).get("name")) or "").strip().lower()

    fixed: list[str] = []
    for index, analysis in enumerate(body.get("analyses") or []):
        if not isinstance(analysis, Mapping):
            continue
        in_scope = terms_in_scope(str(values.read(analysis.get("model_estimation")) or ""), models)
        for position, cell in enumerate((analysis.get("effect") or {}).get("cells") or []):
            if not isinstance(cell, Mapping):
                continue
            named = values.read(cell.get("term"))
            if not isinstance(named, str) or named in in_scope:
                continue
            if named not in everywhere:
                # Declared nowhere: the id may be the unprefixed half of a term `satisfy`
                # re-declared under its model.
                under = [k for k in in_scope if k.endswith(f".{named}")]
                if len(under) == 1:
                    cell["term"] = under[0]
                    fixed.append(
                        f"analyses[{index}].effect.cells[{position}].term: "
                        f"{named!r} -> {under[0]!r} (declared under its model)"
                    )
                continue
            wanted = name_of(everywhere.get(named))
            if not wanted:
                continue
            same = [k for k, term in in_scope.items() if name_of(term) == wanted]
            if len(same) == 1:
                cell["term"] = same[0]
                fixed.append(
                    f"analyses[{index}].effect.cells[{position}].term: "
                    f"{named!r} -> {same[0]!r} (same name, in scope)"
                )
    return fixed


def _minted_part(local_id: str) -> str:
    for prefix in _PREFIXES:
        if local_id.startswith(prefix):
            return local_id[len(prefix):]
    return local_id


def _words(text: Any) -> frozenset[str]:
    found = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split()
    return frozenset(word for word in found if word not in _EMPTY_WORDS)


def _same_id(one: str, other: str) -> bool:
    squash = lambda name: re.sub(r"[^a-z0-9]+", "", name.lower())  # noqa: E731
    return squash(one) == squash(other)


def names_agree(dangling: str, target: str, target_name: Any) -> bool:
    """Does the name a dangling id carries refer to what the target is called?

    A local_id is minted from the paper's wording, so `trm_three_way_interaction` says the
    model meant a term it called "three way interaction". Measured over the 120 references
    `the_only_candidate` proposes: 70% share a word, 8% more are an initialism of the
    target's name -- `asm_scid` against "Structured Clinical Interview for DSM-V" -- and the
    18% that share nothing are the wrong repoints, `reg_caudate` and `reg_insula` and
    `reg_thalamus` each onto one "Gain versus nongain reward-processing regions".
    """
    asked = _minted_part(dangling)
    offered = _words(_minted_part(target)) | _words(target_name)

    if _words(asked) & offered:
        return True
    name = str(target_name or "")
    return bool(name and abbreviations.matches(asked.replace("_", ""), name))


@dataclass(frozen=True, eq=False)
class _Reference:
    """One id in one slot, and where to write its replacement."""

    slot: walk.Slot
    position: int
    id: str

    def repoint_to(self, target: str) -> str:
        if isinstance(self.slot.value, list):
            self.slot.value[self.position] = target
            return f"{self.slot.path}[{self.position}]: {self.id!r} -> {target!r}"
        self.slot.owner[self.slot.key] = target
        return f"{self.slot.path}: {self.id!r} -> {target!r}"


def _dangling(body: dict[str, Any], sch: Schema, declared: Mapping[str, str]):
    for slot in walk.references(body, sch):
        for position, local_id in enumerate(walk.ids_of(slot.value)):
            if local_id not in declared:
                yield _Reference(slot=slot, position=position, id=local_id)


def _transcription_slip(reference: _Reference, declared: Mapping[str, str]) -> str | None:
    """The one declared id this differs from only in case and punctuation."""
    same = [name for name in declared if _same_id(name, reference.id) and name != reference.id]
    return same[0] if len(same) == 1 else None


def _the_only_candidate(
    reference: _Reference, declared: Mapping[str, str], sch: Schema
) -> str | None:
    """The one declared entity of the kind this slot holds, where there is only one."""
    wanted = reference.slot.range
    if not isinstance(wanted, str):
        return None
    of_that_kind = [
        name
        for name, class_name in declared.items()
        if class_name == wanted or sch.resolves_to(class_name, wanted)
    ]
    return of_that_kind[0] if len(of_that_kind) == 1 else None


def _without_collapses(proposed: dict[_Reference, str]) -> dict[_Reference, str]:
    """Drop any target that two differently-named references both want.

    An interaction term shares a word with the main effect inside it, so
    `trm_smoking_opportunity_cue` and `trm_quitting_motivation_cue` both pass the name test
    against a term called "cue". A record declaring one term whose cells name three is a
    record missing two, and repointing all three at the survivor invents an effect.
    """
    wanted_by = {}
    for reference, target in proposed.items():
        wanted_by.setdefault(target, set()).add(reference.id)
    return {
        reference: target
        for reference, target in proposed.items()
        if len(wanted_by[target]) == 1
    }


def repair_references(body: dict[str, Any], sch: Schema) -> list[str]:
    """Repoint a cross-reference that names a local_id nothing declares, where forced.

    A dangling reference is the commonest reason a build reports a defect, and it is always
    the same shape: the model wrote `inf_baseline` for an entity it declared as
    `inference_wholebrain_dti`. The record is written either way -- the exit code says the
    record has a fault, not that the paper was skipped -- but a reference resolving nowhere
    costs an analysis its inference settings, and downstream that reads as a missing field
    rather than a naming slip.

    A transcription slip is repaired outright. Anything else is repaired only when all three
    hold, and each condition was earned by a wrong repoint the previous version made:

      the only candidate    exactly one declared entity of the slot's kind exists. Alone
                            this repointed `asm_mini` to `asm_ftnd`, a psychiatric interview
                            onto a nicotine-dependence scale.
      the names agree       and it is called something the dangling id names. This is what
                            makes the rest safe, and it is deterministic because a local_id
                            is minted from the paper's own wording.
      no collapse           and no differently-named reference wants the same target.

    `declared` is read schema-guided. Sweeping the Study-level lists, as this did, misses
    10,867 ids over 1,817 records -- every ModelTerm under `model_estimations[].terms`, every
    Condition under `tasks[].conditions` -- so it repaired neither of the two slots that
    dangle most, while `validate.index_ids` descended and disagreed with it all along.
    """

    declared = walk.declared_ids(body, sch)
    names = {
        entity.local_id: values.read(entity.node.get("name"))
        for entity in walk.entities(body, sch)
        if entity.local_id
    }

    proposed: dict[_Reference, str] = {}
    slips: dict[_Reference, str] = {}
    for reference in _dangling(body, sch, declared):
        slip = _transcription_slip(reference, declared)
        if slip:
            slips[reference] = slip
            continue
        candidate = _the_only_candidate(reference, declared, sch)
        if candidate and candidate != reference.slot.owner.get("local_id"):
            if names_agree(reference.id, candidate, names.get(candidate)):
                proposed[reference] = candidate

    settled = {**slips, **_without_collapses(proposed)}
    return [reference.repoint_to(target) for reference, target in settled.items()]


def check_local_ids(body: dict[str, Any], sch: Schema) -> list[str]:
    """Verify that every cross-reference resolves to a declared local_id.

    The schema identifies cross-reference slots; they are not hardcoded. A slot is a
    reference when its range is a native string and it is
    not local_id. That distinction matters because sibling slots differ --
    PredictorSource.group is a local_id string while PredictorSource.other is an
    ExtractedString, and Predictor.source is a nested object, not a reference.
    """

    # Collected by walking, not by iterating the Study lists: ModelTerm lives under
    # `model_estimations[].terms` and Condition under `tasks[].conditions`, so a
    # top-level sweep declares neither and every `Cell.term` and
    # `FactorLevel.conditions` reference reads as dangling when it is in fact fine.
    declared: set[str] = set()
    # Counted as well as collected: a reference resolves to a *set* membership, so two
    # entities sharing a local_id both "resolve" and nothing downstream can tell which
    # one a Cell.term meant. Sibling model estimations that share covariate names --
    # term_age, term_sex -- produce this without anything looking wrong.
    times: dict[str, int] = {}

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            if values.is_field(node):
                return
            if isinstance(node.get("local_id"), str):
                declared.add(node["local_id"])
                times[node["local_id"]] = times.get(node["local_id"], 0) + 1
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(body)
    problems: list[str] = [
        f"local_id {name!r} is declared {count} times; every reference to it is ambiguous"
        for name, count in sorted(times.items())
        if count > 1
    ]

    def visit(node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict) or values.is_field(node):
            return
        # Without resolving the designator, every reference declared on a payload subclass
        # -- the seed and target regions of a ConnectivityDetails -- is never visited, so a
        # dangling one reads as fine.
        class_name = sch.designated_type(node, class_name)
        attributes = sch.attributes(class_name)
        for key, value in node.items():
            attribute = attributes.get(key)
            if attribute is None:
                continue
            here = f"{path}.{key}"
            kind = sch.classify(key, attribute)
            if kind == "reference":
                refs = (
                    [value] if isinstance(value, str) else value if isinstance(value, list) else []
                )
                for ref in refs:
                    if isinstance(ref, str) and ref and ref not in declared:
                        problems.append(f"{here} -> unknown local_id {ref!r}")
            elif kind == "nested":
                target = attribute.range
                if isinstance(target, str):
                    for index, item in enumerate(value if isinstance(value, list) else [value]):
                        suffix = f"[{index}]" if isinstance(value, list) else ""
                        visit(item, target, f"{here}{suffix}")

    # From Study rather than from the entity lists, so that references living under a
    # non-list slot -- `design.arms[].`, and anything added there later -- are checked
    # too. `_entity_lists()` holds dotted paths that `body.get()` cannot resolve.
    visit(body, "Study", "Study")
    return problems
