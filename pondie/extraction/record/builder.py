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
from pondie.extraction.evidence import warrant as evidence
from pondie.extraction.record import fix
from pondie.extraction.record import spans as span_tools

# Imported as the function rather than the module: `effect` is a local name
# throughout this file, and the module would be shadowed on first assignment.
from pondie.extraction.record.effect import terms_in_scope
from pondie.formats import parse_keys, text_index, values
from pondie.vocabularies import abbreviations
from pondie.schema import reader
from pondie.schema.reader import Schema

#: The schema is a submodule of this repository, not the parent directory this
#: module used to sit in.
EXTRACTION_SCHEMA = schema.EXTRACTION




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
    """What `build` did to one paper. Warranting keeps its own tally, which it owns.

    Thirteen of the sixteen counters here belonged to span resolution and three to the
    build, and `warrant` now holds the thirteen: `report.warrant.exact` rather than
    `report.resolved_exact`, and `report.warrant.unresolved` rather than
    `report.failures`.
    """

    warrant: evidence.Warrant = field(default_factory=evidence.Warrant)

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
    repair_log: fix.RepairLog | None = None

    #: Every repair the builder performed, for the one-line report and the threshold.
    @property
    def repairs(self) -> int:
        return self.repair_log.total if self.repair_log else 0

    def summary(self) -> str:
        return (
            f"{self.warrant.summary()}\n"
            f"repairs={self.repairs} ("
            + ", ".join(
                f"{name}={len(self.repair_log.changes(name))}"
                for name in (self.repair_log.fired() if self.repair_log else ())
            )
            + ")"
        )






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
            elif key in schema.entity_lists() and isinstance(value, list):
                # An entity list holds objects. 16023086's `analyses` came back as
                # [{...}, "required_entities"] -- one stray token from the reply, which
                # `derive_coordinate_spaces` then called `.get` on, losing a paper that had
                # already paid for all six stages. Dropped here rather than guarded at each
                # consumer, because every walker downstream assumes the same shape.
                kept = [v for v in value if isinstance(v, dict)]
                if len(kept) != len(value):
                    collisions.append(
                        f"{key}: dropped {len(value) - len(kept)} non-object "
                        f"entr{'y' if len(value) - len(kept) == 1 else 'ies'} "
                        f"(from {path.name})"
                    )
                lists.setdefault(schema.entity_lists()[key], []).extend(kept)
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
    # already do. This walked `schema.entity_lists().values()` and looked each up on Study by name
    # -- and three of those values are dotted paths (`design.arms`, `design.timepoints`,
    # `extraction_metadata.paper_sections`) that `Study` has no attribute for. Three
    # iterations resolved to nothing, silently, every build. Harmless only for as long as
    # Arm and Timepoint declare no reference slots; the first one added -- an Arm pointing
    # at a Group is the obvious candidate -- would stop being aliased with no signal.
    visit(body, "Study")
    return rewrites











































































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

    # The order and its constraints live in `record/fix/sequence.py` as data, and are
    # checked before anything runs. They were nine consecutive statements with the
    # constraints in comments beside them, which states an ordering without enforcing it.
    #
    # `AT_MERGE` and not everything: a repair reading one payload has already run beside the
    # pass that wrote it, so what is left here is the group that needs analyses, entities and
    # tables together. Running the earlier groups again would be harmless for the idempotent
    # ones and wrong for `mirrored`, which appends.
    log = fix.apply_all(
        body,
        fix.Context(schema=sch, stage1=stage1, table_map=table_map),
        stage=fix.AT_MERGE,
    )
    report.repair_log = log

    report.warrant = evidence.warrant(body, normalized)

    # The body first, then the builder's own fields over the top -- not the reverse.
    # `schema.entity_lists()` maps `paper_sections` to `extraction_metadata.paper_sections`, and
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

    # A check, not a repair, which is why it is here and not in `fix.build_sequence()`:
    # the sequence's contract is that every entry may change the record, and this one only
    # looks. It runs after the sequence because `repoint_dangling_references` resolves the
    # ones it can, so what survives is what a human has to settle.
    #
    # It was written, documented, and cited by three other modules as the thing that catches
    # an undeclared or doubly-declared id -- and had no caller at all, so `report.dangling`
    # was always empty and `Build`'s "N cross-reference(s) need a human" could never print.
    report.dangling = fix.check_local_ids(record, sch)

    # Integrity gate: nothing leaves this function unless every span addresses the
    # document it claims to.
    evidence.verify(record, normalized)

    return record, report
