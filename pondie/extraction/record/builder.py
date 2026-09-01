"""Assemble an extraction record from extractor payloads, resolving quotes to offsets.

The extractor emits verbatim quotes rather than character offsets, because a
language model cannot count characters reliably. This module locates each quote
in the normalized source text and rewrites

    {"evidence": {"status": "present", "sets": [{"quotes": ["..."]}]}}

into the schema's shape

    {"evidence": {"status": "present", "sets": [{"spans": [{text, start_char, end_char}]}]}}

Every emitted span is verified against the source, so a record that builds is a
record whose offsets are correct by construction. Quotes that cannot be located
are reported and their field is downgraded rather than silently dropped.

Usage:
    pondie extract --pmids papers.pmids --run <run> --model <model> --stages build
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from linkml_runtime.linkml_model.meta import SlotDefinition

from pondie import schema
from pondie.extraction.record import direction, repairs
from pondie.extraction.record import spans as span_tools

# Imported as the function rather than the module: `effect` is a local name
# throughout this file, and the module would be shadowed on first assignment.
from pondie.extraction.record.effect import terms_in_scope
from pondie.formats import parse_keys, text_index, values
from pondie.schema import reader
from pondie.schema.reader import Schema

#: The schema is a submodule of this repository, not the parent directory this
#: module used to sit in.
EXTRACTION_SCHEMA = schema.EXTRACTION


def _entity_lists() -> dict[str, str]:
    """Payload key -> dotted path to the Study attribute holding the inlined list.

    Derived from the schema rather than written out, because a hardcoded list
    silently drops whatever it has not caught up with: `arms` and `timepoints`
    were added to Study and every intervention and longitudinal paper's payload
    lost them, reported only as an "unexpected payload key" note.

    An extractor emits one file per entity kind, so the payload key is a bare
    entity name however deep the schema puts the list. Most sit directly on Study
    and the mapping is the identity; `arms` and `timepoints` sit one level down
    under `design`, so a single level of inlined singleton is followed. Deeper
    nesting is not: the payload would need a path of its own to be unambiguous.
    """

    sch = reader.load(EXTRACTION_SCHEMA)
    study = sch.attributes("Study")

    def lists_on(attributes: Mapping[str, SlotDefinition], prefix: str = "") -> dict[str, str]:
        return {
            name: f"{prefix}{name}"
            for name, attribute in attributes.items()
            if attribute is not None and attribute.multivalued
        }

    found = lists_on(study)
    for name, attribute in study.items():
        if attribute is None or attribute.multivalued:
            continue
        if attribute.inlined is not True:
            continue
        nested = attribute.range
        if nested not in sch:
            continue
        for key, path in lists_on(sch.attributes(nested), f"{name}.").items():
            found.setdefault(key, path)
    return found


_ENTITY_LISTS = _entity_lists()

# Keys that are extractor scaffolding, not schema content.
_SCAFFOLDING = {"cross_reference_notes"}

# Payload filename holding local_id reconciliation, excluded from the merge.
_ALIAS_FILE = "aliases.json"

#: Payload keys that are a pass's output but not the record's. `required_entities` is the
#: demands pass's shopping list, read by `Satisfy.context`; reporting it as an unexpected
#: key on every paper is what made the genuine signal unreadable.
_CONSUMED_ELSEWHERE = {"required_entities"}


@dataclass
class BuildReport:
    resolved_exact: int = 0
    resolved_tolerant: int = 0
    failures: list[str] = field(default_factory=list)
    fields_total: int = 0
    fields_extracted: int = 0
    fields_not_reported: int = 0
    fields_evidence_present: int = 0
    fields_evidence_not_found: int = 0
    downgraded: list[str] = field(default_factory=list)

    #: The repairs, one list each, and the two hard faults. Counted rather than printed and
    #: forgotten: the counts are how a prompt regression becomes visible, and nothing
    #: downstream could read them while `build()` wrote them straight to stdout.
    #:
    #: They used to be sixteen `list[str]` fields here, each hand-copied from the repair
    #: of the same meaning under a DIFFERENT name -- `acquisition_type` arrived as
    #: `derived_acquisition_types`, `cell_levels` as `aligned_levels`, `references` as
    #: `repointed_references`. Two naming schemes for one set of sixteen things, mapped by
    #: hand, and it had drifted twice: `repairs` summed fourteen of them, and `summary()`
    #: printed a fifteenth set. The log below already holds all of it, under the repairs'
    #: own names, so there is nothing to keep in step.
    dangling: list[str] = field(default_factory=list)
    payload_notes: list[str] = field(default_factory=list)

    #: What each repair did, under the repair's own name. The one record of it.
    repair_log: repairs.RepairLog | None = None

    #: Every repair the builder performed, for the one-line report and the threshold.
    @property
    def repairs(self) -> int:
        return self.repair_log.total if self.repair_log else 0

    def summary(self) -> str:
        return (
            f"fields={self.fields_total} "
            f"(extracted={self.fields_extracted}, not_reported={self.fields_not_reported})\n"
            f"evidence: present={self.fields_evidence_present}, "
            f"not_found={self.fields_evidence_not_found}\n"
            f"spans: exact={self.resolved_exact}, whitespace-tolerant={self.resolved_tolerant}, "
            f"unresolved={len(self.failures)}\n"
            f"downgraded fields={len(self.downgraded)}\n"
            f"repairs={self.repairs} ("
            + ", ".join(
                f"{name}={len(self.repair_log.changes(name))}"
                for name in (self.repair_log.fired() if self.repair_log else ())
            )
            + ")"
        )


#: The only two things `extraction_status` may say.
_STATUSES = ("extracted", "not_reported")


def repair_wrappers(node: Any, path: str = "") -> list[str]:
    """Put a collapsed ExtractedValue back together, and report every one.

    The wrapper is `{"extraction_status": "extracted", "value": X}`, and the model
    intermittently writes X into the status slot instead -- `"extraction_status":
    "undirected"` for a direction, `"extraction_status": 0.05` for an alpha level. Every
    such field is invalid against the schema, and the numeric ones additionally break
    `ls.py export`, whose task contract requires `llm_status` to be a string.

    The repair is unambiguous, which is why it is done rather than reported: the status
    slot has exactly two legal strings, so anything else in it was never a status. A
    value already sitting in `value` is kept and only the status is corrected; otherwise
    the misplaced payload is moved into `value`.

    Deliberately here and not in `render.normalize`: it has to hold for payloads
    already on disk, so a rebuild fixes them without paying for the extraction again.
    """

    repaired: list[str] = []
    if isinstance(node, dict):
        status = node.get("extraction_status") if "extraction_status" in node else None

        if "extraction_status" in node and status not in _STATUSES:
            if "value" not in node or node["value"] in (None, ""):
                node["value"] = status
            node["extraction_status"] = "extracted"
            node.setdefault("value_source", "reported")
            repaired.append(f"{path or '<root>'}: status held {status!r}")
        for key, value in node.items():
            repaired += repair_wrappers(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            repaired += repair_wrappers(value, f"{path}[{index}]")
    return repaired


def derive_acquisition_types(body: dict[str, Any]) -> list[str]:
    """Fill `Acquisition.acquisition_type` from each record's modality.

    The schema says of this slot: "Derived by the mapper from the modality value's
    `instantiates`, never extracted" (`acquisition.yaml`) -- so the builder owes it, and
    an extraction that leaves it out is not wrong, it is being taken at its word. Without
    it the record resolves only to the base `Acquisition`, where every modality-specific
    parameter the model *did* extract is undeclared: one omission, seven errors.

    The mapping is read out of the `Modality` enum rather than written down here, so a
    new permissible value arrives with its own subclass already attached.
    """

    modality_enum = reader.load(EXTRACTION_SCHEMA).enums.get("Modality")
    modality = (modality_enum.permissible_values or {}) if modality_enum else {}
    instantiates = {
        name: str(spec["instantiates"][0]).split(":")[-1]
        for name, spec in modality.items()
        if spec.instantiates
    }

    filled: list[str] = []
    for index, acquisition in enumerate(body.get("acquisitions") or []):
        if not isinstance(acquisition, dict) or acquisition.get("acquisition_type"):
            continue
        value = acquisition.get("modality")
        if isinstance(value, Mapping):
            value = value.get("value")
        target = instantiates.get(value)
        if target:
            acquisition["acquisition_type"] = target
            filled.append(f"acquisitions[{index}]: {value} -> {target}")
    return filled


def _resolve_field(
    node: dict[str, Any], normalized: str, folded: str, path: str, report: BuildReport
) -> None:
    """Rewrite one FIELD object's evidence quotes into verified spans, in place."""

    report.fields_total += 1
    status = node.get("extraction_status")
    if status == "extracted":
        report.fields_extracted += 1
    elif status == "not_reported":
        report.fields_not_reported += 1

    evidence = node.get("evidence")
    if not isinstance(evidence, dict):
        # A field with no `evidence` object at all is the same defect as one claiming the
        # wrong status: `evidence` is REQUIRED on every ExtractedValue, so returning here
        # left the record to fail validation at the end of the run rather than be fixed and
        # counted. `repair_wrappers` produces exactly this shape.
        #
        # Branching on the status, because the two answers are different claims and an
        # earlier version of this wrote `not_found` for both. A `not_reported` field is one
        # the paper never mentioned; saying its support could not be located is the exact
        # conflation `values.py` exists to prevent, and the schema rejects the pair outright
        # ("not_reported fields must have evidence.status not_applicable"). That shape is
        # what `--no-evidence` instructs the model to emit, so every such run produced it.
        located = status == "extracted"
        node["evidence"] = evidence = {"status": "not_found" if located else "not_applicable"}
        report.downgraded.append(f"{path}: {status} with no evidence block")

    raw_sets = evidence.get("sets")
    if not isinstance(raw_sets, list):
        # An extracted field may not claim its evidence is not_applicable: the value is
        # asserted, so support for it was either found or not. The branch below enforces
        # this once a set has been tried; a field that arrived with no `sets` at all has
        # never been through it, and used to keep the contradiction all the way into the
        # record.
        if status == "extracted" and evidence.get("status") == "not_applicable":
            evidence["status"] = "not_found"
            report.downgraded.append(path)
        if evidence.get("status") == "not_found":
            report.fields_evidence_not_found += 1
        return

    rebuilt: list[dict[str, Any]] = []
    for index, evidence_set in enumerate(raw_sets):
        quotes = evidence_set.get("quotes") if isinstance(evidence_set, dict) else None
        if not isinstance(quotes, list):
            continue
        resolved: list[dict[str, object]] = []
        for quote in quotes:
            try:
                found = span_tools.resolve(normalized, quote, folded_text=folded)
            except span_tools.SpanResolutionError as error:
                report.failures.append(f"{path} set[{index}]: {error}")
                continue
            if found.exact:
                report.resolved_exact += 1
            else:
                report.resolved_tolerant += 1
            resolved.append(found.as_record())

        # An EvidenceSet requires at least one span (minimum_cardinality: 1), so a
        # set whose every quote failed to resolve cannot be emitted at all.
        if resolved:
            rebuilt.append({"spans": resolved})

    if rebuilt:
        evidence["sets"] = rebuilt
        evidence["status"] = "present"
        report.fields_evidence_present += 1
    else:
        # No usable evidence survived. The value may still be right, so keep it
        # and record that support was not located rather than deleting the field.
        evidence.pop("sets", None)
        if status == "extracted":
            evidence["status"] = "not_found"
            report.fields_evidence_not_found += 1
            report.downgraded.append(path)
        else:
            evidence["status"] = "not_applicable"


def _walk(node: Any, normalized: str, folded: str, path: str, report: BuildReport) -> None:
    if values.is_field(node):
        _resolve_field(node, normalized, folded, path, report)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, normalized, folded, f"{path}.{key}" if path else str(key), report)
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, normalized, folded, f"{path}[{index}]", report)


def merge_payloads(payload_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Merge extractor payloads into a single Study body.

    Each agent covers a disjoint set of classes, so entity lists concatenate and
    the `study` mapping is a shallow union. Overlapping study attributes are a
    prompt bug; the later payload wins and the collision is reported.
    """

    study: dict[str, Any] = {}
    lists: dict[str, list[Any]] = {}
    collisions: list[str] = []

    for path in sorted(payload_dir.glob("*.json")):
        if path.name == _ALIAS_FILE:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            if key in _SCAFFOLDING:
                continue
            if key == "study" and isinstance(value, dict):
                for attr, attr_value in value.items():
                    if attr in study:
                        collisions.append(f"{attr} (from {path.name})")
                    study[attr] = attr_value
            elif key in _ENTITY_LISTS and isinstance(value, list):
                lists.setdefault(_ENTITY_LISTS[key], []).extend(value)
            elif key not in _CONSUMED_ELSEWHERE:
                # The one signal that an entity list was silently lost. `arms` and
                # `timepoints` were added to Study, every intervention and longitudinal
                # paper's payload dropped them, and this note was the only trace -- printed
                # to stdout, outside the report, and buried under a false alarm that fired
                # on every paper.
                collisions.append(f"unexpected payload key {key!r} in {path.name}")

    body = dict(study)
    for attr, items in lists.items():
        if not items:
            continue
        holder = body
        *ancestors, leaf = attr.split(".")
        for step in ancestors:
            holder = holder.setdefault(step, {})
        holder[leaf] = items
    return body, collisions


def load_aliases(payload_dir: Path) -> dict[str, str]:
    path = payload_dir / _ALIAS_FILE
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    aliases = document.get("aliases", {})
    return {k: v for k, v in aliases.items() if isinstance(k, str) and isinstance(v, str)}


def apply_aliases(body: dict[str, Any], sch: Schema, aliases: dict[str, str]) -> int:
    """Rewrite cross-reference slots through the alias map, in place.

    Only slots the schema classifies as references are touched, so an alias can
    never corrupt an extracted value that happens to share a string with an id.
    """

    if not aliases:
        return 0
    rewrites = 0

    def visit(node: Any, class_name: str) -> None:
        nonlocal rewrites
        if not isinstance(node, dict) or values.is_field(node):
            return
        # A reference inside a self-naming payload -- ConnectivityDetails.seed_regions --
        # is declared on the subclass, so recursing on the declared range leaves it
        # unrewritten and a merge silently keeps pointing at the absorbed local_id.
        class_name = sch.designated_type(node, class_name)
        attributes = sch.attributes(class_name)
        for key, value in list(node.items()):
            attribute = attributes.get(key)
            if attribute is None:
                continue
            kind = sch.classify(key, attribute)
            if kind == "reference":
                if isinstance(value, str) and value in aliases:
                    node[key] = aliases[value]
                    rewrites += 1
                elif isinstance(value, list):
                    for index, ref in enumerate(value):
                        if isinstance(ref, str) and ref in aliases:
                            value[index] = aliases[ref]
                            rewrites += 1
            elif kind == "nested":
                target = attribute.range
                if isinstance(target, str):
                    for item in value if isinstance(value, list) else [value]:
                        visit(item, target)

    # From Study, the way `check_local_ids`, `listify_scalars` and `unwrap_plain_slots`
    # already do. This walked `_ENTITY_LISTS.values()` and looked each up on Study by name
    # -- and three of those values are dotted paths (`design.arms`, `design.timepoints`,
    # `extraction_metadata.paper_sections`) that `Study` has no attribute for. Three
    # iterations resolved to nothing, silently, every build. Harmless only for as long as
    # Arm and Timepoint declare no reference slots; the first one added -- an Arm pointing
    # at a Group is the obvious candidate -- would stop being aliased with no signal.
    visit(body, "Study")
    return rewrites


def derive_coordinate_spaces(
    body: dict[str, Any], stage1: Path | None, table_map: Path | None
) -> list[str]:
    """Fill `Analysis.coordinate_space` from the space stage 1 read off the table.

    Stage 1 parses a space for every coordinate it extracts and is right about it: across
    ten papers it returned MNI for all 197 points and disagreed with itself on none. The
    stage-3 prompt already injects that space, but as a hint "to confirm, not values to
    copy" -- and the model declines to confirm, leaving the slot unreported on 50 of 57
    analyses. A fact this deterministic should not be routed through a model that has been
    told to distrust it, any more than `Table.caption` is.

    Only fills what the model left empty, and only when every point behind the analysis
    agrees, so a genuinely mixed-space paper still reaches a human.
    """

    if not (stage1 and stage1.is_file() and table_map and table_map.is_file()):
        return []

    parsed = json.loads(stage1.read_text(encoding="utf-8")).get("analyses") or []
    mapping = json.loads(table_map.read_text(encoding="utf-8"))

    spaces_by_table: dict[str, set[str]] = {}
    for analysis in parsed:
        local = mapping.get(analysis.get("table_id"))
        if not local:
            continue
        seen = {p.get("space") for p in analysis.get("points") or [] if p.get("space")}
        spaces_by_table.setdefault(local, set()).update(seen)

    filled: list[str] = []
    for index, analysis in enumerate(body.get("analyses") or []):
        slot = analysis.get("coordinate_space")
        if isinstance(slot, Mapping) and slot.get("extraction_status") == "extracted":
            continue
        spaces = set()
        for table in analysis.get("tables") or []:
            spaces |= spaces_by_table.get(table, set())
        if len(spaces) != 1:
            continue
        space = spaces.pop()
        # `not_found` and not `not_applicable`, which is the state this used to synthesise.
        # An extracted field claiming `not_applicable` is one of the shape errors
        # `_resolve_field` repairs, so the walk rewrote every one of these to `not_found`
        # anyway -- and counted it in `report.downgraded`, which meant that number was
        # dominated by the builder's own output and could not be thresholded on. There is no
        # quote to find for a value read off the table parse, so `not_found` is simply true.
        analysis["coordinate_space"] = values.wrap(
            space, source="reported", evidence="not_found"
        )
        filled.append(f"analyses[{index}] -> {space}")
    return filled


def listify_nested(body: dict[str, Any], sch: Schema) -> list[str]:
    """Wrap a lone object in a list wherever the schema declares a multivalued slot.

    `model_estimations[].terms` is the one that recurs: an estimation with a single
    ModelTerm comes back as the object rather than a list of one. Which of the three
    shapes a slot takes is the most confusable thing in this schema -- the prompt states
    it per line for that reason -- and this is the benign half of getting it wrong, so it
    is repaired here instead of costing a re-extraction.

    Schema-driven, and reference slots are included: a multivalued reference given one
    bare id has the same shape problem.

    Two things the walk has to get right to reach `ConnectivityDetails.seed_regions`, which
    is where 40 of the corpus's shape errors sat. A single-valued nested slot is descended
    into even though it needs no repair itself, because the slots that do are inside it --
    `Analysis.details`, `.effect` and `.inference_settings` are all single-valued. The
    recursion resolves the type designator, because `details` ranges on the abstract
    AnalysisDetails, whose only attribute is `details_type`; recursing on the declared range
    finds no `seed_regions` to repair and reports nothing.
    """

    fixed: list[str] = []

    def visit(node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict) or values.is_field(node):
            return
        class_name = sch.designated_type(node, class_name)
        attributes = sch.attributes(class_name)
        for key, value in list(node.items()):
            attribute = attributes.get(key)
            if attribute is None:
                continue
            kind = sch.classify(key, attribute)
            if kind not in ("nested", "reference"):
                continue
            if not attribute.multivalued:
                # Nothing to repair on the slot itself, but its contents may need it.
                if kind == "nested" and isinstance(attribute.range, str):
                    visit(value, attribute["range"], f"{path}.{key}")
                continue
            if values.is_field(value):
                # A nested record list is not a multivalued scalar and has no
                # `not_reported` form: "no terms" is the empty list, not a wrapper
                # saying so. Where the model wrapped one anyway, take its `value` if it
                # carried the list and drop to empty if it only carried the excuse.
                inner = value.get("value")
                node[key] = (
                    inner
                    if isinstance(inner, list)
                    else ([inner] if isinstance(inner, dict) else [])
                )
                fixed.append(f"{path}.{key} (unwrapped {value.get('extraction_status')})")
            elif isinstance(value, dict) or (kind == "reference" and isinstance(value, str)):
                node[key] = [value]
                fixed.append(f"{path}.{key}")
            if kind == "nested":
                target = attribute.range
                if isinstance(target, str):
                    for index, item in enumerate(
                        node[key] if isinstance(node[key], list) else []
                    ):
                        visit(item, target, f"{path}.{key}[{index}]")

    study_attributes = sch.attributes("Study")
    for attr in _ENTITY_LISTS.values():
        attribute = study_attributes.get(attr)
        target = attribute.range if attribute is not None else None
        if isinstance(target, str):
            for index, entity in enumerate(body.get(attr, []) or []):
                visit(entity, target, f"{attr}[{index}]")
    return fixed


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


def fill_directions(body: dict[str, Any]) -> list[str]:
    """Give a cell its direction from the contrast's own name, where the model gave none.

    Only fills `absent` and only where the name states a comparison the level appears on
    one side of. Measured over 328 reviewed cells: it answers 17% of them at 98%, and in
    every case where it disagreed with the extraction pass the pass had said `absent` --
    it recovered five and lost none. See docs/deterministic-direction.md.

    It never overrides a direction the model committed to. A rule that answers a sixth of
    the cells has no standing to overturn the pass on the rest, and a silent overwrite
    would hide a disagreement worth reading.
    """

    filled: list[str] = []
    for analysis in body.get("analyses") or []:
        if not isinstance(analysis, Mapping):
            continue
        contrast = " . ".join(
            filter(
                None,
                [
                    str(values.read(analysis.get("name")) or ""),
                    str(values.read(analysis.get("definition")) or ""),
                ],
            )
        )
        if not contrast:
            continue
        for cell in (analysis.get("effect") or {}).get("cells") or []:
            if not isinstance(cell, Mapping):
                continue
            node = cell.get("direction")
            current = values.read(node)
            if current not in (None, "", "absent"):
                continue
            level = str(values.read(cell.get("level")) or "")
            derived = direction.direction_of(level, contrast)
            if derived is None:
                continue
            if isinstance(node, dict):
                node["value"] = derived
                node["extraction_status"] = "extracted"
                node["value_source"] = "generated"
            else:
                cell["direction"] = values.wrap(
                    derived, source="generated", evidence="not_found"
                )
            filled.append(
                f"{values.read(analysis.get('local_id'))}: "
                f"{level or '(unnamed level)'} -> {derived}"
            )
    return filled


def mirror_withheld(body: dict[str, Any], stage1: Path | None) -> list[str]:
    """Rebuild the reversed half of every sign-split contrast, from the corrected record.

    `tables.split_opposite_signs` hands the extraction pass only the half the paper
    describes and marks the other `withhold`. This runs last, on the assembled record, so
    the mirror is taken from the contrast the model actually settled on -- including
    whatever the wrapper repairs, level alignment and direction fill changed about it.
    Mirroring the raw payload would copy mistakes the builder had already fixed.
    """

    if not (stage1 and stage1.is_file()):
        return []
    parsed = json.loads(stage1.read_text(encoding="utf-8")).get("analyses") or []
    # Keyed alongside the parse so each withheld entry carries the address of its own row
    # group -- the mirrored analysis's only route to the rows it is about.
    keyed = list(zip(parse_keys.parse_keys(parsed), parsed))
    withheld = [(key, a) for key, a in keyed if a.get("withhold") and a.get("mirror_of")]
    if not withheld:
        return []

    analyses = body.setdefault("analyses", [])

    # Joined on the parse key, not on the name. `entry["mirror_of"]` is the *parse's* name
    # for the described half, and the record's name for that analysis is whatever the model
    # wrote -- which the prompt actively tells it to change: "Each part is its own entry,
    # named `<given name> (<level>)`". The most common case, a SPLIT, was therefore the
    # case the join could not survive, and the reversed half of a mixed-sign table was
    # silently never built. A trailing space did it too.
    #
    # `source_links` is repair 13 and this is repair 16, so every analysis that can carry a
    # key already does. A SPLIT leaves several record analyses sharing one key; each is a
    # contrast in its own right and each gets its mirror, which is the right semantics
    # rather than a tie to break.
    # Keyed on `(table_id, name)`, because a name alone is not unique and `stage1_block`
    # says so: "the same analysis name recurs across tables in the same paper (an ROI table
    # and a whole-brain table reporting one contrast), and the table is the only thing that
    # tells those apart." Keyed on the name alone this was last-wins, and the failure was
    # silent in three directions at once -- the earlier table's half was never mirrored and
    # no MISSING note fired because *a* described half had been found; the mirror carried
    # one table's cells against the other's coordinates; and both mirrors were minted with
    # the same `local_id`.
    #
    # The pair is exact rather than a heuristic: `split_opposite_signs` emits the two halves
    # of one entry consecutively, so they always share a `table_id`.
    described_key = {
        (a.get("table_id"), str(a.get("name") or "")): key
        for key, a in keyed
        if not a.get("withhold")
    }
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for analysis in analyses:
        if isinstance(analysis, Mapping):
            key = values.read(analysis.get("source_table_analysis"))
            if isinstance(key, str) and key:
                by_key.setdefault(key, []).append(analysis)

    # The fallback, for the analyses that carry no key at all -- `derive_analysis_ids`
    # measures that at about a quarter of them. Folded, which is the discipline
    # `resolve_source_table_analysis` already uses for names in this module; the raw `==`
    # this replaced lost a match to a trailing space.
    by_folded_name: dict[str, list[Mapping[str, Any]]] = {}
    for analysis in analyses:
        if isinstance(analysis, Mapping):
            name = span_tools.fold_label(str(values.read(analysis.get("name")) or ""))
            if name:
                by_folded_name.setdefault(name, []).append(analysis)

    made: list[str] = []
    for parse_key, entry in withheld:
        target = described_key.get((entry.get("table_id"), entry["mirror_of"]))
        described = by_key.get(target or "", [])
        route = "parse key"
        if not described:
            described = by_folded_name.get(span_tools.fold_label(entry["mirror_of"]), [])
            route = "name"
        if not described:
            made.append(
                f"MISSING {entry['mirror_of']}: no analysis carries the described half's "
                f"parse key ({target or 'unknown'}) and none matches its name, so its "
                f"reversal cannot be built"
            )
            continue
        for half in described:
            mirrored = direction.mirror_analysis(half, entry, parse_key)
            analyses.append(mirrored)
            made.append(
                f"{entry['mirror_of']} -> {mirrored.get('local_id')} "
                f"(rows at {parse_key}, signs negated, joined on {route})"
            )
    return made


def listify_scalars(body: dict[str, Any], sch: Schema) -> list[str]:
    """Wrap a lone scalar in a list inside an `Extracted<T>List` wrapper.

    The other half of the shape confusion `listify_nested` repairs, one level down. A
    multivalued *source-derived* field is one wrapper whose `value` is a list -- the
    convention extraction-readme.md §2 leads with -- and a model that has just been told
    a wrapper holds "the value" writes the string. `interpretations` is where it recurs.

    Distinct from `listify_nested`, which repairs the slot: here the slot is right and its
    `value` is not, so the wrapper class's own `value` declaration is what decides. Walks
    from Study rather than the entity lists so a wrapper under `design.arms[]` is reached.
    """

    fixed: list[str] = []

    def visit(node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict) or values.is_field(node):
            return
        class_name = sch.designated_type(node, class_name)
        attributes = sch.attributes(class_name)
        for key, value in node.items():
            attribute = attributes.get(key)
            if attribute is None:
                continue
            kind = sch.classify(key, attribute)
            if kind == "evidence" and values.is_field(value):
                wrapper = attribute.range
                declared = (
                    (sch.attributes(wrapper) or {}).get("value", {})
                    if isinstance(wrapper, str)
                    else {}
                )
                if not declared.multivalued or "value" not in value:
                    continue
                inner = value["value"]
                # A missing value is a different fault and stays visible as one; only a
                # present scalar is the shape this repairs.
                if inner is not None and not isinstance(inner, list):
                    value["value"] = [inner]
                    fixed.append(f"{path}.{key}")
            elif kind == "nested":
                target = attribute.range
                if isinstance(target, str):
                    for index, item in enumerate(
                        value if isinstance(value, list) else [value]
                    ):
                        suffix = f"[{index}]" if isinstance(value, list) else ""
                        visit(item, target, f"{path}.{key}{suffix}")

    visit(body, "Study", "Study")
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


def unwrap_plain_slots(body: dict[str, Any], sch: Schema) -> list[str]:
    """Unwrap an ExtractedValue the model put in a slot that holds a bare scalar.

    The record has two kinds of slot and they look alike from inside a model: a
    source-derived value carries `{"value": ..., "evidence": ...}`, and a cross-reference
    or native scalar carries the bare thing. Having just written twenty wrappers, the
    model writes a twenty-first, and the validator reports `must be a string, got dict`.

    Repaired rather than reported because there is nothing to decide: the wrapper's own
    `value` is the answer, and the evidence it carried was never a slot the schema has a
    place for. Only `reference` and `native` slots are touched -- an `evidence` slot is
    supposed to hold a wrapper and unwrapping it would destroy the value.
    """

    fixed: list[str] = []

    def visit(node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict) or values.is_field(node):
            return
        class_name = sch.designated_type(node, class_name)
        attributes = sch.attributes(class_name)
        for key, value in list(node.items()):
            attribute = attributes.get(key)
            if attribute is None:
                continue
            here = f"{path}.{key}"
            kind = sch.classify(key, attribute)
            if kind in ("reference", "native"):
                if values.is_field(value) and "value" in value:
                    node[key] = value["value"]
                    fixed.append(f"{here}: unwrapped a wrapper into a {kind} slot")
                elif values.is_field(value):
                    # A wrapper with no `value` says `not_reported`, which is the right
                    # encoding for an evidence slot and meaningless here: a reference has
                    # no wrapper form, so "not reported" is simply absence. Dropped
                    # rather than left, because the validator reads it as a malformed
                    # cross-reference and the paper said nothing either way.
                    del node[key]
                    fixed.append(
                        f"{here}: dropped an empty "
                        f"{value.get('extraction_status', 'valueless')!r} wrapper "
                        f"from a {kind} slot"
                    )
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        if values.is_field(item) and "value" in item:
                            value[index] = item["value"]
                            fixed.append(
                                f"{here}[{index}]: unwrapped a wrapper into a " f"{kind} slot"
                            )
                continue
            if (
                kind == "evidence"
                and value is not None
                and not isinstance(value, (dict, list))
            ):
                # The inverse slip: a bare scalar in a slot that holds an ExtractedValue.
                # The value is the model's answer and it offered no span for it, so the
                # evidence is honestly `not_found` rather than invented.
                node[key] = values.wrap(value, source="reported", evidence="not_found")
                fixed.append(
                    f"{here}: wrapped a bare {type(value).__name__} into an " f"ExtractedValue"
                )
                continue
            if kind == "nested":
                target = attribute.range or class_name
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        visit(item, target, f"{here}[{index}]")
                else:
                    visit(value, target, here)

    visit(body, "Study", "Study")
    return fixed


#: Slot suffixes whose ExtractedValue must hold a number. Read from the wrapper's own
#: declared range rather than guessed, but the suffix is what makes the intent legible
#: at the call site.
def coerce_numeric_values(body: dict[str, Any], sch: Schema) -> list[str]:
    """Turn `"4.5"` into `4.5` where the wrapper declares a number.

    A number read off a table arrives as text and the model passes it through. The
    schema says what the slot holds, so the conversion is arithmetic; a value that is
    not a number after stripping units is left alone and stays a validator finding,
    because inventing a number is worse than reporting a string.
    """

    fixed: list[str] = []

    def visit(node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict):
            return
        class_name = sch.designated_type(node, class_name)
        attributes = sch.attributes(class_name)
        for key, value in node.items():
            attribute = attributes.get(key)
            if attribute is None:
                continue
            here = f"{path}.{key}"
            wrapper = attribute.range
            if values.is_field(value) and isinstance(wrapper, str):
                declared = (sch.attributes(wrapper) or {}).get("value")
                # `getattr`, not `.range`: the fallback for a wrapper class that declares
                # no `value` slot was a bare `{}`, which has no `.range` and took the whole
                # build down with an AttributeError -- one paper in 89, at the last stage,
                # after every model call it needed had been paid for.
                wants = getattr(declared, "range", None)
                inner = value.get("value")
                if wants in ("float", "double", "decimal", "integer") and isinstance(
                    inner, str
                ):
                    cleaned = re.sub(r"[^0-9eE.+-]", "", inner.strip())
                    try:
                        number = float(cleaned)
                    except ValueError:
                        continue
                    value["value"] = int(number) if wants == "integer" else number
                    fixed.append(f"{here}: {inner!r} -> {value['value']}")
                continue
            if isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, attribute.range or class_name, f"{here}[{index}]")
            elif isinstance(value, dict):
                visit(value, attribute.range or class_name, here)

    visit(body, "Study", "Study")
    return fixed


def rehome_stray_tables(body: dict[str, Any], sch: Schema) -> list[str]:
    """Move a Table the model wrote as a Study attribute into `tables`.

    Seen as `Study: attribute 'tab4' is not declared on Study` while every analysis
    referenced `tab4`. The cost is not cosmetic: the stray key is dropped on load, so
    every analysis pointing at it loses the table its coordinates are joined through,
    and the paper contributes nothing to a coordinate query.

    Only keys that some analysis actually references are moved, and only when they carry
    no `local_id` of their own -- anything else is a slot the schema does not know
    about, which is a different fault and stays reported.
    """

    declared = set(sch.attributes("Study") or {})
    referenced: set[str] = set()
    for analysis in body.get("analyses") or []:
        if isinstance(analysis, Mapping):
            cited = values.read(analysis.get("tables")) or []
            referenced |= {
                t
                for t in (cited if isinstance(cited, list) else [cited])
                if isinstance(t, str)
            }

    moved: list[str] = []
    for key in [k for k in body if k not in declared and k in referenced]:
        stray = body.pop(key)
        entry = dict(stray) if isinstance(stray, Mapping) else {}
        entry.setdefault("local_id", key)
        body.setdefault("tables", []).append(entry)
        moved.append(f"Study.{key}: moved into tables[] as {key!r}")
    return moved


def resolve_source_table_analysis(body: dict[str, Any], stage1: Path | None) -> list[str]:
    """Verify, or deterministically fill, each analysis's link to its parsed row group.

    `Analysis.source_table_analysis` is the only exact route from an analysis to its
    coordinates: the schema stores none, `Table.coordinate_count` says only how many
    exist, and `tables` cannot disambiguate because a table usually reports several
    contrasts and several analyses usually cite the same table.

    Not left to the model. A key it invents resolves to nothing, and a key it omits
    leaves the join to a later string match -- which is what this slot exists to replace.
    A present key is checked against the parse and dropped if it names no row group,
    and an absent one is filled when exactly one parsed entry under the cited tables
    carries the same name. Where neither holds, the slot stays empty and the analysis is
    honestly unjoinable rather than joined to a guess.
    """

    if not (stage1 and stage1.is_file()):
        return []
    parsed = json.loads(stage1.read_text(encoding="utf-8")).get("analyses") or []
    if not parsed:
        return []

    # The same keying the prompt prints, from the same function, so the two cannot drift.
    keys = dict(zip(parse_keys.parse_keys(parsed), parsed))

    def fold(text: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())

    notes: list[str] = []
    for index, analysis in enumerate(body.get("analyses") or []):
        if not isinstance(analysis, Mapping):
            continue
        node = analysis.get("source_table_analysis")
        claimed = values.read(node)
        path = f"analyses[{index}].source_table_analysis"

        if isinstance(claimed, str) and claimed in keys:
            continue
        if isinstance(claimed, str) and claimed:
            analysis.pop("source_table_analysis", None)
            notes.append(f"{path}: {claimed!r} names no parsed row group -- dropped")

        cited = [t for t in (values.read(analysis.get("tables")) or []) if isinstance(t, str)]
        wanted = fold(values.read(analysis.get("name")))
        same = [
            key
            for key, entry in keys.items()
            if fold(entry.get("name")) == wanted
            and (not cited or str(entry.get("table_id")) in cited)
        ]
        if len(same) == 1 and wanted:
            analysis["source_table_analysis"] = values.wrap(
                same[0], source="generated", evidence="not_applicable"
            )
            notes.append(
                f"{path}: filled {same[0]!r} from the parsed analysis of the " f"same name"
            )
    return notes


def derive_analysis_ids(body: dict[str, Any]) -> list[str]:
    """Rename each analysis to an id the parse determines, not one the model chose.

    A model-chosen `local_id` is unstable. Over the same sixteen papers extracted twice,
    only four produced identical analysis ids: `a_ic25/a_ic30/a_ic35` one run against
    `a_independent_component_spatial_maps` the next, `a_fa_group` against `a_fa`. That
    matters outside the record -- the review layer addresses an analysis as
    `paper|value|Analysis|<local_id>|<path>`, so a re-extraction orphans every answer a
    reviewer gave.

    `source_table_analysis` is already a deterministic paper-scoped key, so the id is
    derived from it: `a_<table id>_<ordinal>`. Safe to do here because nothing in the
    schema references an Analysis by id -- `Study.analyses` is the only slot with that
    range and it inlines them -- so the sole pointer to follow is `mirror_of`.

    An analysis with no key keeps the model's id. That is 25% of them and it is the
    honest outcome: the parse does not determine an id for a row group it cannot identify,
    and inventing a stable-looking one would claim otherwise.

    Idempotent, and collision-safe: a SPLIT emits several analyses against one listing
    entry, so they share a key and are numbered apart in the order they appear.
    """

    renamed: dict[str, str] = {}
    used: set[str] = set()
    notes: list[str] = []

    for analysis in body.get("analyses") or []:
        if isinstance(analysis, Mapping) and isinstance(analysis.get("local_id"), str):
            used.add(analysis["local_id"])

    seen: dict[str, int] = {}
    for analysis in body.get("analyses") or []:
        if not isinstance(analysis, Mapping):
            continue
        key = values.read(analysis.get("source_table_analysis"))
        if not isinstance(key, str) or "#" not in key:
            continue
        table_id, _, ordinal = key.partition("#")
        stem = f"a_{re.sub(r'[^A-Za-z0-9]+', '_', table_id).strip('_')}_{ordinal}"
        seen[stem] = seen.get(stem, 0) + 1
        derived = stem if seen[stem] == 1 else f"{stem}_{seen[stem]}"
        old = analysis.get("local_id")
        if old == derived:
            continue
        if derived in used and derived != old:
            # Another analysis already answers to this. Leave both alone rather than
            # collapse two analyses into one id.
            notes.append(f"{old!r}: derived id {derived!r} is already taken -- left as is")
            continue
        analysis["local_id"] = derived
        used.discard(old)
        used.add(derived)
        if isinstance(old, str):
            renamed[old] = derived
        notes.append(f"{old!r} -> {derived!r} (from {key!r})")

    # `mirror_of` is the only pointer at an analysis anywhere in the record.
    for analysis in body.get("analyses") or []:
        if isinstance(analysis, Mapping) and analysis.get("mirror_of") in renamed:
            analysis["mirror_of"] = renamed[analysis["mirror_of"]]
    return notes


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
        in_scope = terms_in_scope(
            str(values.read(analysis.get("model_estimation")) or ""), models
        )
        for position, cell in enumerate((analysis.get("effect") or {}).get("cells") or []):
            if not isinstance(cell, Mapping):
                continue
            named = values.read(cell.get("term"))
            if not isinstance(named, str) or named in in_scope:
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


def repair_references(body: dict[str, Any], sch: Schema) -> list[str]:
    """Repoint a cross-reference that names a local_id nothing declares, where forced.

    A dangling reference is the commonest reason a build reports a defect, and it is
    always the same shape: the model wrote `inf_baseline` for an entity it declared as
    `inference_wholebrain_dti`. The record is written either way -- the exit code says the
    record has a fault, not that the paper was skipped -- but a reference that resolves
    nowhere costs an analysis its inference settings, and downstream that reads as a
    missing field rather than a naming slip.

    Repaired only where the choice is not a choice, which is the rule
    `align_cell_levels` already follows for levels:

      fold      the dangling id folds to exactly one declared id -- case, underscores and
                hyphens removed. A transcription slip, and nothing is being decided.
      sole      the slot's own Study-level list holds exactly one entity. There is only
                one thing the reference could have meant.

    Anything else is left dangling and reported. Two candidates and a guess about which
    inference settings an analysis used is a claim about the paper, and the point of the
    report is that a human sees it.
    """

    def fold(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    declared: dict[str, str] = {}
    for key, value in body.items():
        if not isinstance(value, list):
            continue
        for entity in value:
            if isinstance(entity, Mapping) and isinstance(entity.get("local_id"), str):
                declared[entity["local_id"]] = key

    by_fold: dict[str, list[str]] = {}
    for name in declared:
        by_fold.setdefault(fold(name), []).append(name)

    def pool_for(slot: str) -> list[str]:
        # `model_estimation` is a reference to something in `model_estimations`; the
        # slot is singular and the Study list is not.
        for candidate in (slot, f"{slot}s", slot.rstrip("s")):
            if candidate in body and isinstance(body[candidate], list):
                return [n for n, owner in declared.items() if owner == candidate]
        return []

    fixed: list[str] = []

    def repoint(dangling: str, slot: str) -> str | None:
        same = [n for n in by_fold.get(fold(dangling), []) if n != dangling]
        if len(same) == 1:
            return same[0]
        pool = pool_for(slot)
        return pool[0] if len(pool) == 1 else None

    def visit(node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict) or values.is_field(node):
            return
        class_name = sch.designated_type(node, class_name)
        attributes = sch.attributes(class_name)
        for key, value in list(node.items()):
            attribute = attributes.get(key)
            if attribute is None:
                continue
            here = f"{path}.{key}"
            if sch.classify(key, attribute) == "reference":
                if isinstance(value, str) and value not in declared:
                    target = repoint(value, key)
                    if target:
                        node[key] = target
                        fixed.append(f"{here}: {value!r} -> {target!r}")
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, str) and item not in declared:
                            target = repoint(item, key)
                            if target:
                                value[index] = target
                                fixed.append(f"{here}[{index}]: {item!r} -> {target!r}")
                continue
            if isinstance(value, Mapping):
                visit(value, attribute.range or class_name, here)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, attribute.range or class_name, f"{here}[{index}]")

    visit(body, "Study", "Study")
    return fixed


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
                    [value]
                    if isinstance(value, str)
                    else value if isinstance(value, list) else []
                )
                for ref in refs:
                    if isinstance(ref, str) and ref and ref not in declared:
                        problems.append(f"{here} -> unknown local_id {ref!r}")
            elif kind == "nested":
                target = attribute.range
                if isinstance(target, str):
                    for index, item in enumerate(
                        value if isinstance(value, list) else [value]
                    ):
                        suffix = f"[{index}]" if isinstance(value, list) else ""
                        visit(item, target, f"{here}{suffix}")

    # From Study rather than from the entity lists, so that references living under a
    # non-list slot -- `design.arms[].`, and anything added there later -- are checked
    # too. `_ENTITY_LISTS` holds dotted paths that `body.get()` cannot resolve.
    visit(body, "Study", "Study")
    return problems


def build(
    paper_id: str,
    text_path: Path,
    payload_dir: Path,
    extractor_model: str,
    extractor_version: str,
    extraction_date: str,
    stage1: Path | None = None,
    table_map: Path | None = None,
) -> tuple[dict[str, Any], BuildReport]:
    normalized, digest, sections = text_index.load(text_path)
    folded = span_tools.fold(normalized)
    report = BuildReport()

    sch = reader.load(EXTRACTION_SCHEMA)
    body, merge_notes = merge_payloads(payload_dir)
    report.payload_notes += merge_notes
    # Nothing here prints. Every repair is the model getting the wrapper shape wrong or the
    # builder paying a debt the schema assigned it, and the counts are how a prompt
    # regression becomes visible -- which they cannot be while they go straight to stdout
    # for a reader to notice or not. `main()` is the only formatter, and `--strict`
    # thresholds on these same numbers.
    rewrites = apply_aliases(body, sch, load_aliases(payload_dir))
    if rewrites:
        report.payload_notes.append(
            f"reconciled {rewrites} cross-reference(s) through aliases.json"
        )

    # The order and its constraints live in `record/repairs.py` as data, and are
    # checked before anything runs. They were nine consecutive statements with the
    # constraints in comments beside them, which states an ordering without enforcing it.
    log = repairs.apply_all(
        body, repairs.Context(schema=sch, stage1=stage1, table_map=table_map)
    )
    report.repair_log = log

    _walk(body, normalized, folded, "", report)

    # The body first, then the builder's own fields over the top -- not the reverse.
    # `_ENTITY_LISTS` maps `paper_sections` to `extraction_metadata.paper_sections`, and
    # `merge_payloads` materialises a dotted path into a nested dict, so a payload carrying
    # a top-level `paper_sections` produces a `body["extraction_metadata"]` that
    # `record.update(body)` substituted wholesale -- taking `source_text_hash` with it.
    # Every evidence offset in the record then addresses a document nothing can identify,
    # and the validator's hash check is `if declared_hash and ...`, so a missing one passes.
    record: dict[str, Any] = dict(body)
    # Required on Study, and not something a model reads off the page: the paper's corpus
    # id is what the record is keyed by, so the builder supplies it.
    record["local_id"] = paper_id
    metadata = record.setdefault("extraction_metadata", {})
    metadata.update(
        {
            "extractor_model": extractor_model,
            "extractor_version": extractor_version,
            "source_text_hash": digest,
            "extraction_date": extraction_date,
            "paper_sections": [section.as_record() for section in sections],
        }
    )

    # A check, not a repair, which is why it is here and not in `repairs.build_sequence()`:
    # the sequence's contract is that every entry may change the record, and this one only
    # looks. It runs after the sequence because `repoint_dangling_references` resolves the
    # ones it can, so what survives is what a human has to settle.
    #
    # It was written, documented, and cited by three other modules as the thing that catches
    # an undeclared or doubly-declared id -- and had no caller at all, so `report.dangling`
    # was always empty and `Build`'s "N cross-reference(s) need a human" could never print.
    report.dangling = check_local_ids(record, sch)

    # Integrity gate: nothing leaves this function unless every span addresses the
    # document it claims to.
    for evidence_set in _iter_sets(record):
        for span in evidence_set.get("spans", []):
            span_tools.verify(normalized, span)

    return record, report


def _iter_sets(node: Any):
    if isinstance(node, dict):
        if "spans" in node and isinstance(node["spans"], list):
            yield node
        for value in node.values():
            yield from _iter_sets(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_sets(value)
