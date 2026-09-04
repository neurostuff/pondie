"""Read the LinkML schema, through LinkML.

The schema is authored as an entrypoint plus a directory of modules, and answering "what
attributes does `Analysis` have" means following the imports, walking `is_a`, and applying
`slot_usage` on the way down. `pondie.schema.authoring` did all three by hand. LinkML's own
`SchemaView` does them, and it does them better: it applies the schema's `default_range`,
which the hand-rolled walk did not -- `EvidenceSpan.text` came back with no range at all.

Equivalence was measured before the swap, not assumed. Over the 92 classes and 508 slots of
the extraction schema, `class_induced_slots` and the hand-rolled `attributes_for` agree on
**every class**, `class_descendants` and `subclasses_of` agree everywhere, and of 508 slots
exactly one range differs -- the `default_range` case above, where LinkML is right. The only
other divergence, `inlined` on 32 slots, does not reach a decision: those slots carry
`inlined_as_list` and classify as `nested` under both readers.

What LinkML cannot answer is what a slot MEANS here, because that is this project's
convention rather than the language's:

    nested vs reference   a reference slot holds only the target's `local_id`. LinkML has
                          no such concept; `classify` is the rule and it stays ours.
    evidence              a slot whose range resolves to `ExtractedValue` carries a value
                          and the span that warrants it.

This class is therefore a reader with a vocabulary, not a wrapper. It answers the questions this
codebase asks, and delegates the ones LinkML already answers.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Literal, Mapping

import yaml
from linkml_runtime.linkml_model.meta import ClassDefinition, EnumDefinition, SlotDefinition
from linkml_runtime.utils.schemaview import SchemaView

from pondie.formats import values

#: Base class of every source-derived attribute in the extraction schema.
EXTRACTED_VALUE = "ExtractedValue"

#: The generated document-local identifier, which is never a cross-reference.
LOCAL_ID = "local_id"

#: How a reviewer must treat one slot. See `Schema.classify`.
SlotKind = Literal["identifier", "evidence", "nested", "reference", "native"]


class Schema:
    """One schema, loaded once, answering the questions this codebase asks of it.

    Loading is not cheap -- `SchemaView` materialises the whole tree, about 0.6s for the
    extraction schema -- so an instance is built per path and cached. Every method is a
    read; nothing here mutates the schema, which is what makes the cache safe.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.view = SchemaView(str(self.path))
        # Per instance, not `@lru_cache` on the method. A method-level cache keys on `self`
        # and, at `maxsize=None`, holds it forever -- every `Schema` ever built stays alive
        # with a materialised `SchemaView` behind it. These die with the instance.
        self._attributes: dict[str, Mapping[str, SlotDefinition]] = {}
        self._subclasses: dict[str, frozenset[str]] = {}
        self._designator: dict[str, str | None] = {}

    # -- what is in the schema ---------------------------------------------------------

    @functools.cached_property
    def classes(self) -> dict[str, ClassDefinition]:
        return dict(self.view.all_classes())

    @functools.cached_property
    def enums(self) -> dict[str, EnumDefinition]:
        return dict(self.view.all_enums())

    @functools.cached_property
    def declaration_order(self) -> tuple[str, ...]:
        """Class names in the order the YAML declares them, imports first.

        LinkML does not reproduce this and should not be asked to: none of its four
        `OrderedBy` modes matches, because the order is a property of how *this* schema is
        split across files rather than of the language. It is load-bearing anyway -- the
        prompt renders one block per class in this order so a reader comparing prompt to
        schema meets them in the same sequence, and the gateway caches whole prompts, so a
        reordering that changed nothing semantically would still cost a full re-cache.

        Only the ordering is read from the YAML here. Every value comes from `SchemaView`.
        """

        order: list[str] = []
        seen_files: set[Path] = set()

        def visit(path: Path) -> None:
            path = path.resolve()
            if path in seen_files:
                return
            seen_files.add(path)
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for name in document.get("imports") or ():
                if not isinstance(name, str) or ":" in name:
                    continue
                target = path.parent / name
                if target.suffix != ".yaml":
                    target = target.with_suffix(".yaml")
                if target.is_file():
                    visit(target)
            for name in document.get("classes") or ():
                if name not in order:
                    order.append(name)

        visit(self.path)
        # Anything SchemaView knows about that no local file declared -- imported from the
        # LinkML metamodel, say -- goes last rather than being dropped.
        return tuple(order) + tuple(n for n in self.classes if n not in set(order))

    def __contains__(self, class_name: str) -> bool:
        return class_name in self.classes

    def __iter__(self) -> Iterator[str]:
        """Class names, in the order the schema declares them."""
        return iter(self.declaration_order)

    def definition(self, class_name: str) -> ClassDefinition | None:
        return self.classes.get(class_name)

    # -- what a class holds ------------------------------------------------------------

    def attributes(self, class_name: str) -> Mapping[str, SlotDefinition]:
        """Every slot of `class_name`, inherited and narrowed ones included.

        `class_induced_slots` is what makes this one line rather than thirty: it walks
        `is_a` and applies `slot_usage`, which is how `ExtractedString` narrows `value` to
        a string on top of the `Any` it inherits.
        """
        if class_name not in self.classes:
            return {}
        if class_name in self._attributes:
            return self._attributes[class_name]
        induced = {slot.name: slot for slot in self.view.class_induced_slots(class_name)}

        # Root-first, and LinkML is not: `class_induced_slots` puts a subclass's own slots
        # ahead of the ones it inherits, so `MRI` would lead with
        # `magnetic_field_strength_tesla` and reach `modality` twelve lines later. The
        # prompt renders these in order, and reading "what every Acquisition has, then what
        # an MRI adds" is the order that matches how the schema is written. Ordering only --
        # every slot itself is LinkML's, `slot_usage` and all.
        order: list[str] = []
        ancestry: list[str] = []
        current: str | None = class_name
        while current:
            ancestry.append(current)
            parent = self.classes[current].is_a if current in self.classes else None
            current = parent if isinstance(parent, str) and parent in self.classes else None
        for ancestor in reversed(ancestry):
            for name in self.classes[ancestor].attributes or {}:
                if name in induced and name not in order:
                    order.append(name)
        order += [name for name in induced if name not in order]
        # Read-only, because this is shared: a caller that filtered the returned dict in
        # place -- and two callers build filtered copies of it today -- would corrupt the
        # schema for every later reader in the process.
        self._attributes[class_name] = MappingProxyType({n: induced[n] for n in order})
        return self._attributes[class_name]

    def containers(self) -> Mapping[str, str]:
        """Class -> the top-level list of `Study` its instances live in.

        Derived, not listed. A thirteenth entity list added to the schema appears here the
        day it is added; a hand-kept table is one a new list is silently missing from, and
        anything keyed on it -- a recall sweep, a reference resolver -- then skips it without
        saying so.
        """
        out: dict[str, str] = {}
        for name, slot in self.attributes("Study").items():
            if isinstance(slot.range, str) and slot.range in self.classes and slot.multivalued:
                out[slot.range] = name
        return out

    def subclasses(self, class_name: str) -> frozenset[str]:
        """Every class inheriting from `class_name`, directly or transitively.

        A slot whose range is an abstract class holds an instance of one of its subclasses,
        so anything walking ranges has to follow this edge: `Analysis.details` ranges on the
        abstract `AnalysisDetails` and never on `AnalysisDetails` itself.
        """
        if class_name not in self.classes:
            return frozenset()
        if class_name not in self._subclasses:
            self._subclasses[class_name] = (
                frozenset(self.view.class_descendants(class_name)) - {class_name}
            )
        return self._subclasses[class_name]

    def resolves_to(self, class_name: str, ancestor: str) -> bool:
        """Whether `class_name` is, or inherits from, `ancestor`."""
        if class_name not in self.classes:
            return class_name == ancestor
        return ancestor in self.view.class_ancestors(class_name)

    # -- what a slot means -------------------------------------------------------------

    @staticmethod
    def ranges(slot: SlotDefinition) -> list[str]:
        """Every class or type this slot may hold.

        `any_of` as well as `range`, because a slot that keeps an escape hatch --
        `any_of: [SomeEnum, string]` -- ranges over both, and a consumer that reads only
        `range` silently treats the open vocabulary as closed.
        """
        out = [slot.range] if slot.range else []
        out += [option.range for option in (slot.any_of or []) if option.range]
        seen: dict[str, None] = {}
        for name in out:
            seen.setdefault(str(name), None)
        return list(seen)

    def classify(self, name: str, slot: SlotDefinition) -> SlotKind:
        """Classify one slot by how a reviewer must treat it.

        identifier -- the generated `local_id`; not reviewable.
        evidence   -- range resolves to `ExtractedValue`, so it carries value + evidence.
        nested     -- range is another class the record OWNS; recurse into it.
        reference  -- range is another class the record only points AT, carried as that
                      class's `local_id`. Reviewable, but the schema records no evidence.
        native     -- a plain scalar that is not source-derived, e.g. the
                      `ExtractionMetadata` and `PaperSection` pipeline fields.

        Ownership separates nested from reference, and this reads the schema's own
        `inlined` / `inlined_as_list` rather than LinkML's *inference* of them -- which is
        the opposite answer for every reference here. `SchemaView.is_inlined` returns True
        for 38 slots this calls references, because LinkML inlines a class range whose
        target declares no `identifier` slot, and `local_id` is deliberately not one: it is
        a document-local address, unique within a record and meaningless outside it, which
        is a different thing from a LinkML identifier.

        The distinction therefore rests on a schema property that nothing else states and
        the failure mode is quiet: adding `inlined_as_list: true` to `Analysis.tables` -- a
        plausible edit to satisfy a generator -- reclassifies it `nested`, and the prompt
        then asks the model for whole Table records where a list of id strings belongs.
        `tests/test_schema_reader.py` pins the reference set so that edit is a red test.

        It used to be read off the *description*, because the hand-written extraction schema
        declared every cross-reference as `range: string` and only the prose ("local_id of
        the Task record...") told `Analysis.tasks` apart from
        `ExtractionMetadata.extractor_model`. Since the extraction schema became a projection
        of storage, a reference keeps its real range and the prose is no longer load-bearing.
        """
        if name == LOCAL_ID:
            return "identifier"

        slot_range = slot.range
        if not isinstance(slot_range, str):
            return "native"
        if slot_range not in self.classes:
            return "native"
        if self.resolves_to(slot_range, EXTRACTED_VALUE):
            return "evidence"
        if slot.inlined or slot.inlined_as_list:
            return "nested"
        return "reference"

    def iter_slots(self, class_name: str) -> Iterator[tuple[str, SlotDefinition, SlotKind]]:
        """Every slot of a class with its kind, which is what most callers actually want."""
        for name, slot in self.attributes(class_name).items():
            yield name, slot, self.classify(name, slot)

    # -- self-naming payloads ----------------------------------------------------------

    def type_designator(self, class_name: str) -> str | None:
        """The slot naming the concrete subclass, for a class with a self-naming payload.

        Read off LinkML's own `designates_type` rather than from a list here, so a third
        such family costs nothing to support.
        """
        if class_name not in self._designator:
            self._designator[class_name] = next(
                (n for n, slot in self.attributes(class_name).items() if slot.designates_type),
                None,
            )
        return self._designator[class_name]

    def designated_type(self, node: Mapping[str, object], class_name: str) -> str:
        """The concrete class `node` says it is, or `class_name` when it says nothing.

        The record states its own variant because nothing downstream could infer it: an
        `AnalysisDetails` payload is only a `DecodingDetails` because it says so.
        """
        designator = self.type_designator(class_name)
        if not designator:
            return class_name
        stated = values.read(node.get(designator)) if isinstance(node, Mapping) else None
        if isinstance(stated, str) and self.resolves_to(stated, class_name):
            return stated
        return class_name


def load(path: Path | str) -> Schema:
    """The schema at `path`, loaded once per process.

    Resolved before the cache is consulted: keyed on the raw argument, `load(Path(p))` and
    `load(str(p))` are two entries and two `SchemaView`s, which is the 0.6s this cache
    exists to spend once.
    """
    return _load(str(Path(path).expanduser().resolve()))


@functools.lru_cache(maxsize=4)
def _load(resolved: str) -> Schema:
    return Schema(resolved)
