#!/usr/bin/env python3
"""Does this record conform to the extraction schema?

Structure only, and only what LinkML can state: does every slot exist on its class, does
every value match its declared range, does a wrapper carry the evidence block the schema
requires, does a span address the document it claims to. It knows nothing about
neuroimaging.

The thirteen things a record can be that are structurally legal and scientifically wrong --
an interaction the prose names but the encoding does not, a contrast whose name carries a
sign its cells do not -- are `rules.py`, and `check_record` runs them from a registry rather
than by hand. The two halves have different readers: this one needs LinkML, that one needs
to know what a crossover is.

Enforces LinkML structure (declared attributes, required slots, ranges,
multivalued shape), the class `rules` the storage schema states, and the
invariants in the extraction schema header:

  * extraction_status: not_reported  =>  value omitted, evidence.status not_applicable
  * evidence.status: present         =>  at least one set, each with at least one span
  * every span satisfies text == source[start_char:end_char]

Structure is read from the YAML directly; only the rules need linkml, imported
where they are loaded so the rest runs without it.

Usage:
    python -m pondie.extraction.record.validate \
        --record data/runs/<run>/records/2abntY3hQSyq.extraction.json \
        --text data/corpus/2abntY3hQSyq/processed/pubget/text.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from copy import copy
from pathlib import Path
from typing import Any

from linkml_runtime.linkml_model.meta import EnumDefinition, SlotDefinition

from pondie import schema
from pondie.extraction.record import rules
from pondie.extraction.record import spans as span_tools
from pondie.formats import text_index, values
from pondie.schema import reader
from pondie.schema.reader import Schema

#: The schema is a submodule of this repository, not the parent directory this
#: module used to sit in.
EXTRACTION_SCHEMA = schema.EXTRACTION
#: Rules are read from storage: `gen_extraction_schema.py` drops `rules` on the way
#: to extraction, so this is the only statement of them.
STORAGE_SCHEMA = schema.STORAGE

_RULES: dict[str, list[Mapping[str, Any]]] | None = None


def storage_rules() -> Mapping[str, list[Mapping[str, Any]]]:
    """Class name -> its rules, resolved through linkml so imports are followed.

    Cached: SchemaView takes about a second to walk the eleven modules, and a record
    has hundreds of instances to check.
    """

    global _RULES
    if _RULES is None:
        from linkml_runtime.dumpers import json_dumper
        from linkml_runtime.utils.schemaview import SchemaView

        view = SchemaView(str(STORAGE_SCHEMA))
        _RULES = {
            name: [json_dumper.to_dict(rule) for rule in definition.rules]
            for name, definition in view.all_classes().items()
            if definition.rules
        }
    return _RULES


_EXTRACTION_STATUS = {"extracted", "not_reported"}
_VALUE_SOURCE = {"reported", "generated"}
_EVIDENCE_STATUS = {"present", "not_found", "not_applicable"}

# LinkML native ranges of the ExtractedValue subclasses, and the Python types
# that satisfy them. bool is excluded from integer deliberately: True would
# otherwise pass as 1.
_SCALAR_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "float": (int, float),
    "date": (str,),
    "boolean": (bool,),
}

#: What makes an analysis's own text a claim about a crossing. Short on purpose:
#: `interaction` and `moderation` are the words papers use for one, and `×` is how a
#: term name spells it. `-by-` is a crossing in "group-by-stage" and a reduplication
#: in "voxel-by-voxel", so it counts only when the words it joins differ.
class Validator:
    def __init__(
        self,
        sch: Schema,
        normalized: str | None,
        enums: Mapping[str, EnumDefinition] | None = None,
    ) -> None:
        self.schema = sch
        self.enums = dict(enums) if enums is not None else dict(sch.enums)
        self.normalized = normalized
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.fields = 0
        self.spans = 0
        #: local_id -> the class that declared it, filled by `index_ids` before the walk.
        #: Empty until then, and a reference check with an empty index is skipped rather
        #: than failed, so validating a fragment stays possible.
        self.declared: dict[str, str] = {}

    def error(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    def warn(self, path: str, message: str) -> None:
        self.warnings.append(f"{path}: {message}")

    def index_ids(self, node: Any, class_name: str) -> None:
        """Record every `local_id` in the record against the class that declared it.

        Schema-guided rather than a blind walk: descending only into slots the schema calls
        nested, with the declared range as the class, is what makes the class known. A type
        designator is followed through `Schema.designated_type` rather than `resolve_type`
        so indexing reports nothing -- the walk proper will report it once.
        """
        if not isinstance(node, dict):
            return
        class_name = self.schema.designated_type(node, class_name)
        attributes = self.schema.attributes(class_name)
        if not attributes:
            return
        local = node.get("local_id")
        if isinstance(local, str) and local:
            self.declared[local] = class_name
        for name, attribute in attributes.items():
            if name not in node or self.schema.classify(name, attribute) != "nested":
                continue
            target = attribute.range
            if not isinstance(target, str):
                continue
            value = node[name]
            for item in value if isinstance(value, list) else [value]:
                self.index_ids(item, target)

    def check_reference(self, ref: str, attribute: SlotDefinition, path: str) -> None:
        """A cross-reference must resolve, and resolve to the class the slot declares.

        Checking only that the value is a string let two different faults through. A
        reference naming an id nothing declares costs an analysis its inference settings and
        reads downstream as a missing field; `repair_references` repoints those where the
        choice is forced and reports the rest, so they are warned here rather than failed.

        The second fault nothing saw at all: a reference that resolves perfectly well to the
        *wrong kind of thing*. `ModelEstimation.inputs_from` declares `range:
        ModelEstimation` and its description says "the models whose fitted output this model
        was estimated on", and a record still pointed it at a Condition nested under a Task.
        That reference is not dangling, so no repair notices it, and every reader downstream
        follows it to something that cannot answer the question the slot asks. Measured over
        903 records: 26,897 references, 593 dangling, 22 pointing at the wrong class.

        Subclasses resolve: a slot declaring `Acquisition` is satisfied by an `MRI`, which is
        what `Schema.resolves_to` is for.
        """
        target = attribute.range
        if not isinstance(target, str) or not self.declared:
            return
        actual = self.declared.get(ref)
        if actual is None:
            self.warn(path, f"cross-reference {ref!r} names no declared local_id")
            return
        if actual != target and not self.schema.resolves_to(actual, target):
            self.error(
                path,
                f"cross-reference {ref!r} is a {actual}, but this slot declares {target}",
            )

    # -- class instances ---------------------------------------------------

    def resolve_type(self, node: Mapping[str, Any], class_name: str, path: str) -> str:
        """Follow a type designator to the subclass the record says it is.

        `Analysis.details` declares range AnalysisDetails and `details_type` says which of
        the eight it really is; the fields that make the payload useful live on the
        subclass, so validating against the declared range rejects every one of them.

        `Schema.type_designator` identifies the type slot by reading
        `designates_type: true`, rather than from a list here -- the same resolution four
        walkers in `build_record` and `extract_record` need, and a list in five places is
        how they came to disagree.
        """

        designator = self.schema.type_designator(class_name)
        if designator is None:
            return class_name
        # An extraction record wraps the designator in an ExtractedValue, a storage record
        # states it bare; `values.read` is the one place that difference is handled.
        named = values.read(node.get(designator))
        if not isinstance(named, str):
            return class_name
        if named not in self.schema:
            self.error(path, f"{designator} names unknown class {named!r}")
            return class_name
        if not self.schema.resolves_to(named, class_name):
            self.error(path, f"{designator} {named!r} is not a {class_name}")
            return class_name
        return named

    def check_instance(self, node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict):
            self.error(path, f"expected an object for {class_name}, got {type(node).__name__}")
            return

        class_name = self.resolve_type(node, class_name, path)
        attributes = self.schema.attributes(class_name)
        if not attributes:
            self.error(path, f"class {class_name!r} is not defined in the schema")
            return

        for key in node:
            if key not in attributes:
                self.error(path, f"attribute {key!r} is not declared on {class_name}")

        for name, attribute in attributes.items():
            if attribute.required and name not in node:
                self.error(path, f"required attribute {name!r} is missing on {class_name}")

        for key, value in node.items():
            attribute = attributes.get(key)
            if attribute is None:
                continue
            self.check_slot(value, key, attribute, f"{path}.{key}",
                            owner=node.get("local_id"), owner_class=class_name)

        self.check_rules(node, class_name, path)

    # -- class rules -------------------------------------------------------

    def check_rules(self, node: Mapping[str, Any], class_name: str, path: str) -> None:
        """Apply the storage schema's `rules` for this class.

        They are read from storage because the projection drops them: an extraction
        record is the only thing there is to check them against, so a rule stated on
        storage and never evaluated is prose. Reading them rather than restating them
        is what keeps a rule added later enforced without code.
        """

        for rule in storage_rules().get(class_name, []):
            if not self.conditions_hold(node, rule.get("preconditions"), path):
                continue
            if self.conditions_hold(node, rule.get("postconditions"), path):
                continue
            self.error(path, rule.get("description") or f"violates a rule on {class_name}")

    def conditions_hold(self, node: Mapping[str, Any], conditions: Any, path: str) -> bool:
        """Whether a pre- or postcondition block holds of this instance."""

        if not isinstance(conditions, Mapping):
            return True
        for keyword, body in conditions.items():
            if keyword == "slot_conditions":
                if not all(
                    self.slot_condition_holds(node.get(slot), condition, f"{path}.{slot}")
                    for slot, condition in body.items()
                ):
                    return False
            elif keyword == "any_of":
                if not any(self.conditions_hold(node, one, path) for one in body):
                    return False
            elif keyword == "none_of":
                if any(self.conditions_hold(node, one, path) for one in body):
                    return False
            elif keyword == "all_of":
                if not all(self.conditions_hold(node, one, path) for one in body):
                    return False
            else:
                self.unsupported(path, keyword)
                return True
        return True

    def slot_condition_holds(self, value: Any, condition: Any, path: str) -> bool:
        if not isinstance(condition, Mapping):
            return True
        for keyword, body in condition.items():
            if keyword == "name":
                continue
            if keyword == "equals_string":
                if values.read(value) != body:
                    return False
            elif keyword == "pattern":
                # A standard LinkML metaslot, and unsupported keywords pass silently -- so
                # without this a `pattern` condition is a rule that never fires, which is
                # worse than no rule at all.
                text = values.read(value)
                if text is None or not re.search(str(body), str(text)):
                    return False
            elif keyword == "value_presence":
                present = value not in (None, "", [], {})
                if present != (str(body).upper() == "PRESENT"):
                    return False
            elif keyword == "any_of":
                if not any(self.slot_condition_holds(value, one, path) for one in body):
                    return False
            elif keyword == "none_of":
                if any(self.slot_condition_holds(value, one, path) for one in body):
                    return False
            elif keyword == "all_of":
                if not all(self.slot_condition_holds(value, one, path) for one in body):
                    return False
            else:
                self.unsupported(path, keyword)
        return True

    def unsupported(self, path: str, keyword: str) -> None:
        """A construct this evaluator cannot read is reported, never skipped.

        Silently ignoring one turns the rule into a check that always passes, which
        is worse than the prose it replaced.
        """

        self.error(
            path, f"rule construct {keyword!r} is not implemented; the rule was not checked"
        )

    def check_slot(
        self, value: Any, name: str, attribute: SlotDefinition, path: str,
        owner: Any = None, owner_class: str | None = None,
    ) -> None:
        kind = self.schema.classify(name, attribute)
        multivalued = bool(attribute.multivalued)

        if multivalued and not isinstance(value, list):
            self.error(path, f"{name!r} is multivalued but got {type(value).__name__}")
            return
        if not multivalued and isinstance(value, list) and kind in {"evidence", "nested"}:
            self.error(path, f"{name!r} is not multivalued but got a list")
            return

        items = value if multivalued and isinstance(value, list) else [value]
        for index, item in enumerate(items):
            here = f"{path}[{index}]" if multivalued else path
            if kind == "identifier":
                if not isinstance(item, str) or not item:
                    self.error(here, "local_id must be a non-empty string")
            elif kind == "reference":
                if not isinstance(item, str):
                    self.error(
                        here, f"cross-reference must be a string, got {type(item).__name__}"
                    )
                else:
                    self.check_reference(item, attribute, here)
                    # A slot whose range is its own class points at a *different* instance of
                    # it: nothing is its own input, its own mirror, or one of the terms it is
                    # a product of. `inputs_from` had this as a stage-chain rule; `mirror_of`
                    # and `interaction_with` had nothing, so a self-loop on either passed.
                    # Derived from the range rather than named here, so a self-referential
                    # slot added later is covered the day it is added.
                    if (isinstance(owner, str) and owner == item
                            and owner_class is not None
                            and attribute.range == owner_class):
                        self.error(
                            here,
                            f"{name!r} names its own instance {item!r}; a slot whose range "
                            f"is {owner_class} refers to a different one",
                        )
            elif kind == "native":
                self.check_native(item, attribute, here)
            elif kind == "nested":
                target = attribute.range
                if isinstance(target, str):
                    self.check_instance(item, target, here)
            elif kind == "evidence":
                target = attribute.range
                self.check_field(
                    item, target if isinstance(target, str) else "ExtractedValue", here
                )

    def check_native(self, value: Any, attribute: SlotDefinition, path: str) -> None:
        """Type-check a pipeline scalar. Enum ranges are checked by their callers."""

        declared = attribute.range or "string"
        if not isinstance(declared, str) or declared not in _SCALAR_TYPES:
            return
        if declared == "integer" and isinstance(value, bool):
            self.error(path, "must be an integer, got a boolean")
            return
        if not isinstance(value, _SCALAR_TYPES[declared]):
            self.error(path, f"must be a {declared}, got {type(value).__name__}")
            return
        minimum = attribute.minimum_value
        if isinstance(minimum, int) and isinstance(value, (int, float)) and value < minimum:
            self.error(path, f"must be >= {minimum}, got {value}")

    # -- ExtractedValue fields --------------------------------------------

    def check_field(self, node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict):
            self.error(path, f"expected an ExtractedValue object, got {type(node).__name__}")
            return

        self.fields += 1
        attributes = self.schema.attributes(class_name)

        for key in node:
            if key not in attributes:
                self.error(path, f"attribute {key!r} is not declared on {class_name}")

        status = node.get("extraction_status")
        if status not in _EXTRACTION_STATUS:
            self.error(
                path,
                f"extraction_status must be one of {sorted(_EXTRACTION_STATUS)}, got {status!r}",
            )

        source = node.get("value_source")
        if source is not None and source not in _VALUE_SOURCE:
            self.error(
                path, f"value_source must be one of {sorted(_VALUE_SOURCE)}, got {source!r}"
            )

        evidence = node.get("evidence")
        if evidence is None:
            self.error(path, "evidence is required")
        else:
            self.check_evidence(evidence, status, f"{path}.evidence")

        # Header invariant: not_reported means no value at all.
        if status == "not_reported":
            if "value" in node:
                self.error(path, "not_reported fields must omit value")
        elif status == "extracted":
            if "value" not in node:
                self.error(path, "extracted fields must carry a value")
            else:
                self.check_value_type(node["value"], class_name, attributes, path)

    def vocabulary_of(self, value_slot: SlotDefinition) -> tuple[set[str] | None, bool]:
        """(permissible values, closed) for a wrapper's `value` slot, or (None, False).

        Closed is readable off the range: a bare enum admits nothing else, while
        `any_of: [<Enum>, string]` is storage's declared escape hatch.
        """

        for name in self.schema.ranges(value_slot):
            enum = self.enums.get(name)
            if enum is not None:
                return set(enum.permissible_values or {}), value_slot.range == name
        return None, False

    def check_value_type(
        self, value: Any, class_name: str, attributes: Mapping[str, SlotDefinition], path: str
    ) -> None:
        value_slot = attributes.get("value", {})

        # A vocabulary is checked before the scalar branch, because an enum range is not in
        # _SCALAR_TYPES and would otherwise fall straight through -- which is how a
        # `statistic.family` of "T" passed while the vocabulary says "t". A closed field is
        # enforced; an open one (`any_of: [<Enum>, string]`) keeps its escape hatch and the
        # off-vocabulary value is reported as a warning, since those accumulating are the
        # evidence for whether the vocabulary is short a value.
        vocabulary, closed = self.vocabulary_of(value_slot)
        if vocabulary is not None:
            for item in (
                value if value_slot.multivalued and isinstance(value, list) else [value]
            ):
                if not isinstance(item, str):
                    continue
                # Missingness has one encoding, and no vocabulary offers `unstated` any
                # more. Named here anyway: on an open field it would otherwise pass as free
                # text with only a warning, and on a closed one the membership error would
                # name the wrong defect.
                if item == "unstated":
                    self.error(
                        path,
                        "'unstated' is not a value: a fact the source does "
                        "not report is `extraction_status: not_reported`, "
                        "which is the one encoding of missingness. Drop "
                        "`value` and set `evidence.status` to not_applicable",
                    )
                elif item not in vocabulary:
                    message = (
                        f"{item!r} is not a permissible value "
                        f"({', '.join(sorted(vocabulary))})"
                    )
                    if closed:
                        self.error(path, message)
                    else:
                        self.warn(path, message + "; open vocabulary, kept as free text")

        declared = value_slot.range or "Any"

        # The shape assertion comes before the scalar branch, for the same reason the
        # vocabulary check does. An Extracted<Enum>List declares its `value` with `any_of`
        # and no `range`, so `declared` is "Any" and the early return below skips it --
        # which let a `Task.response_mode` of "button_press" pass where the schema wants
        # ["button_press"]. Only the per-item recursion needs a native range.
        if value_slot.multivalued and not isinstance(value, list):
            # `declared` is "Any" for an any_of slot, which reads as nonsense in the
            # message, so name what the slot actually accepts.
            accepts = (
                declared
                if declared in _SCALAR_TYPES
                else " or ".join(
                    r for r in self.schema.ranges(value_slot) if r != "Any"
                )
                or "value"
            )
            self.error(
                path,
                f"{class_name}.value must be a list of {accepts}, "
                f"got {type(value).__name__}",
            )
            return

        if declared in {"Any", None} or declared not in _SCALAR_TYPES:
            return  # ExtractedValue holds lists and free-form structures by design.
        expected = _SCALAR_TYPES[declared]

        # An Extracted<T>List declares `multivalued: true` on its own `value`, and a list
        # there is the whole point: extraction-readme.md §2 makes "one wrapper holding a
        # list" the headline convention, so rejecting it rejected every correctly shaped
        # inclusion_criteria, preprocessing step and echo time in the record.
        if value_slot.multivalued:
            # A copy with `multivalued` cleared, so the recursion checks one item against
            # the slot's declared type. This was a dict comprehension over `value_slot`
            # -- correct when a slot was a dict and an `AttributeError` once it became a
            # `SlotDefinition`, which is to say: on every record carrying a multivalued
            # extracted value. `Group.medical_condition` alone is 237 groups with two or
            # more, and this is the branch that exists to accept them.
            item_slot = copy(value_slot)
            item_slot.multivalued = False
            for index, item in enumerate(value):
                self.check_value_type(
                    item, class_name, {"value": item_slot}, f"{path}.value[{index}]"
                )
            return

        # A multivalued concept expressed as a list inside a scalar ExtractedValue
        # subtype is a real shape problem: the mapper would fail to parse it.
        if isinstance(value, list):
            self.error(path, f"{class_name}.value must be a {declared}, got a list")
            return
        if declared == "integer" and isinstance(value, bool):
            self.error(path, f"{class_name}.value must be an integer, got a boolean")
            return
        if not isinstance(value, expected):
            self.error(
                path, f"{class_name}.value must be a {declared}, got {type(value).__name__}"
            )

    def check_evidence(self, node: Any, status: Any, path: str) -> None:
        if not isinstance(node, dict):
            self.error(path, f"expected an Evidence object, got {type(node).__name__}")
            return

        evidence_status = node.get("status")
        if evidence_status not in _EVIDENCE_STATUS:
            self.error(
                path,
                f"status must be one of {sorted(_EVIDENCE_STATUS)}, got {evidence_status!r}",
            )

        if status == "not_reported" and evidence_status != "not_applicable":
            self.error(path, "not_reported fields must have evidence.status not_applicable")
        if status == "extracted" and evidence_status == "not_applicable":
            self.error(path, "extracted fields must not have evidence.status not_applicable")

        sets = node.get("sets")
        if evidence_status == "present":
            if not isinstance(sets, list) or not sets:
                self.error(path, "evidence.status present requires at least one set")
                return
        elif sets:
            self.error(path, f"evidence.status {evidence_status} must not carry sets")
            return

        for index, evidence_set in enumerate(sets or []):
            self.check_set(evidence_set, f"{path}.sets[{index}]")

    def check_set(self, node: Any, path: str) -> None:
        """The attributes `EvidenceSet` declares, read from the schema rather than named here.

        This used to hardcode `spans`, so the day `source` was added to the schema the
        validator began rejecting the one field the extractor had just started writing --
        the schema said yes and the checker said no. Reading the class keeps them from
        drifting apart again.
        """
        if not isinstance(node, dict):
            self.error(path, f"expected an EvidenceSet object, got {type(node).__name__}")
            return
        declared = self.schema.classes.get("EvidenceSet")
        allowed = set(getattr(declared, "attributes", None) or {}) or {"spans"}
        for key in node:
            if key not in allowed:
                self.error(path, f"attribute {key!r} is not declared on EvidenceSet")
        source = node.get("source")
        permissible = getattr(self.enums.get("EvidenceSource"), "permissible_values", None)
        if source is not None and permissible and source not in permissible:
            self.error(path, f"evidence set source {source!r} is not a permissible value "
                             f"({', '.join(sorted(permissible))})")
        spans = node.get("spans")
        if not isinstance(spans, list) or not spans:
            self.error(path, "EvidenceSet requires at least one span (minimum_cardinality: 1)")
            return
        for index, span in enumerate(spans):
            self.check_span(span, f"{path}.spans[{index}]")

    def check_span(self, node: Any, path: str) -> None:
        if not isinstance(node, dict):
            self.error(path, f"expected an EvidenceSpan object, got {type(node).__name__}")
            return
        self.spans += 1
        for key in node:
            if key not in {"text", "start_char", "end_char"}:
                self.error(path, f"attribute {key!r} is not declared on EvidenceSpan")
        for key in ("text", "start_char", "end_char"):
            if key not in node:
                self.error(path, f"required attribute {key!r} is missing on EvidenceSpan")
                return
        if self.normalized is not None:
            try:
                span_tools.verify(self.normalized, node)
            except span_tools.SpanResolutionError as error:
                self.error(path, str(error))

    # -- crossings and the columns that carry them -------------------------


    def diff(self, before: Any, after: Any) -> list[str]:
        """Findings `after` has that `before` did not, most frequent first.

        A record is repaired by editing it, and an edit can damage what it did not touch.
        Validating only the result cannot separate a defect the pass caused from one it
        inherited; validating both and subtracting can. Array indices are collapsed so that
        the same fault on two analyses counts as one kind rather than two.

        Written because the alternative was demonstrated: a repair pass put 665 findings into
        fifteen records over several weeks and nobody saw it, since every one of those records
        already had findings of its own.
        """
        from collections import Counter

        def kinds(record: Any) -> Counter:
            checker = Validator(self.schema, self.normalized, self.enums)
            checker.check_record(record)
            return Counter(
                re.sub(r"\[\d+\]", "[]", message)
                for message in checker.errors + checker.warnings
            )

        added = kinds(after) - kinds(before)
        return [f"{count}  {message}" for message, count in added.most_common()]

    def check_record(self, record: Any) -> None:
        # Before the walk: a reference can name an entity declared later in the document,
        # so every local_id has to be known before the first one is checked.
        self.declared = {}
        self.index_ids(record, "Study")
        self.check_instance(record, "Study", "Study")

        # The domain rules, from the registry rather than named one by one here.
        if isinstance(record, dict):
            rules.check_all(record, self)

        metadata = record.get("extraction_metadata") if isinstance(record, dict) else None
        if isinstance(metadata, dict) and self.normalized is not None:
            declared_hash = metadata.get("source_text_hash")
            actual = text_index.text_hash(self.normalized)
            if declared_hash and declared_hash != actual:
                self.error(
                    "Study.extraction_metadata.source_text_hash",
                    f"does not match the supplied text ({declared_hash[:12]}... != {actual[:12]}...)",
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument(
        "--text", type=Path, help="normalized source text; enables offset checks"
    )
    parser.add_argument("--paper", help="neurostore id, for the report header")
    args = parser.parse_args()

    normalized = None
    if args.text:
        normalized = text_index.normalize(args.text.read_text(encoding="utf-8"))

    validator = Validator(reader.load(EXTRACTION_SCHEMA), normalized)
    validator.check_record(json.loads(args.record.read_text(encoding="utf-8")))

    paper = args.paper or args.record.name.split(".")[0]
    print(f"{paper}: {validator.fields} fields, {validator.spans} spans checked")
    if validator.warnings:
        print(f"\nwarnings ({len(validator.warnings)}):")
        for warning in validator.warnings:
            print(f"  - {warning}")
    if validator.errors:
        print(f"\nerrors ({len(validator.errors)}):")
        for error in validator.errors:
            print(f"  - {error}")
        return 1
    print("valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
