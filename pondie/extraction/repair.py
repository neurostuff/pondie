"""Repair a built record: propose, ground, guard, and put what is left to a model.

Runs after `build`, on a record that already exists, and changes it in place. Four steps,
narrowing at each one:

  1. **propose** -- a local model reads the methods and results and returns entities of one
     class at a time, with the entities it may point at listed per reference slot.
  2. **ground** -- a local entailment model scores each proposal against the passage offered
     for it, so a proposal with no warrant is not written.
  3. **guard** -- `record.edit.apply` refuses the writes that would damage the record, and
     says why.
  4. **adjudicate** -- what is left is a contradiction the record cannot settle from its own
     contents. That goes to the extraction model, once, with the paper.

Every step is optional. With no proposer and no checker there is nothing to propose and
nothing to ground, and the stage does only step 4; with `adjudicate` off it does nothing at
all. This is deliberate: the local models want a GPU, and a run without one should still be
able to resolve what the paper plainly answers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence

from pondie.extraction.evidence.grounding import Checker, Claim
from pondie.extraction.record import edit as edit_module
from pondie.extraction.record import guards
from pondie.extraction.record.validate import Validator
from pondie.formats import values
from pondie.schema.reader import Schema

ADJUDICATION_SYSTEM = """\
You resolve contradictions in a structured record extracted from a neuroimaging paper.

Each case names fields of the record that cannot all be true, and lists the values each may
take. Answer with the value the paper supports and one verbatim sentence from the paper that
shows it. Copy the sentence exactly; do not paraphrase, join, or trim it.

Answer "unresolved" whenever the paper does not settle the case -- when it is silent,
ambiguous, or describes something the options do not cover. The record already reports the
contradiction, so a reviewer can see it; a confident wrong answer removes that."""


@dataclass
class Report:
    """What one pass did to one record."""

    written: list[str] = field(default_factory=list)
    refused: list[guards.Refusal] = field(default_factory=list)
    adjudicated: list[str] = field(default_factory=list)
    #: Findings this pass introduced, from `Validator.diff`. Should be empty.
    introduced: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"wrote {len(self.written)}, refused {len(self.refused)}, "
                f"adjudicated {len(self.adjudicated)}, introduced {len(self.introduced)}")


@dataclass(frozen=True)
class Case:
    """One contradiction, with the values it may be resolved to."""

    id: str
    question: str
    options: tuple[str, ...]
    container: str
    local_id: str
    slot: str
    #: Cleared when the answer makes the slot beside it inapplicable.
    clears: str = ""


def contradictions(record: Mapping[str, Any], sch: Schema) -> list[Case]:
    """The scope/regions pairs that disagree.

    Only this family, and deliberately: a case is adjudicable when it can be put as "choose
    one of these and quote the sentence". Of the findings a repaired record carries, the
    largest group is dangling references caused by a deletion -- for which the answer is not
    to delete, not to ask. Measured over 42 records: 209 findings, of which 8 are these.
    """
    out: list[Case] = []
    pairs = (("analyses", "Analysis", "spatial_scope", "regions"),
             ("inference_settings", "InferenceSettings", "correction_scope",
              "correction_regions"))
    for container, class_name, scope_slot, region_slot in pairs:
        attribute = sch.attributes(class_name).get(scope_slot)
        options = tuple(
            getattr(sch.enums.get(r), "permissible_values", {}) or {}
            for r in (sch.ranges(attribute) if attribute else [])
        )
        allowed = tuple(v for group in options for v in group) or ("whole_brain", "roi")
        for entity in record.get(container) or []:
            if not isinstance(entity, Mapping):
                continue
            scope = str(values.read(entity.get(scope_slot)) or "").strip().lower()
            regions = entity.get(region_slot) or []
            if scope not in guards.UNRESTRICTED or not regions:
                continue
            named = ", ".join(_label(record, r) for r in regions)
            out.append(Case(
                id=f"{container}/{entity.get('local_id')}/{scope_slot}",
                question=(f"{scope_slot} is '{scope}' while {region_slot} names {named}. "
                          f"A whole-brain or searchlight procedure is restricted to no "
                          f"region, so at most one of these is right."),
                options=allowed, container=container,
                local_id=str(entity.get("local_id")), slot=scope_slot,
                clears=region_slot))
    return out


def _label(record: Mapping[str, Any], local_id: str) -> str:
    for entities in record.values():
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if isinstance(entity, Mapping) and entity.get("local_id") == local_id:
                return edit_module.label_of(entity)
    return local_id


def adjudicate(record: MutableMapping[str, Any], sch: Schema, text: str, caller: Any,
               *, study_id: str, model: str, report: Report) -> None:
    """Put the unresolved contradictions to the extraction model, once, with the paper.

    A resolution is applied only when its quote resolves to a span of this paper, by the same
    `spans.resolve`/`verify` the extractor is held to. A plausible value with an invented
    sentence reads exactly like a resolved case, so the quote is the gate rather than a
    courtesy.
    """
    from pondie.extraction.models import ModelCall
    from pondie.extraction.record import spans as span_tools

    cases = contradictions(record, sch)
    if not cases:
        return
    listing = "\n\n".join(
        f"case {i + 1} (id {c.id}):\n  {c.question}\n"
        f"  permissible values: {', '.join(c.options)}, or unresolved"
        for i, c in enumerate(cases))
    reply = caller(
        ModelCall(model=model, system=ADJUDICATION_SYSTEM, effort="low",
                  max_output_tokens=4_000,
                  prompt=(f"## Paper\n\n{text}\n\n## Cases\n\n{listing}\n\n"
                          'Reply as {"resolutions": [{"id": ..., "value": ..., '
                          '"quote": ...}]}, using the case id verbatim and an empty quote '
                          'for anything unresolved.')),
        paper=study_id, stage="repair")
    body = getattr(reply, "body", reply)
    answers = body if isinstance(body, Mapping) else json.loads(str(body))

    by_id = {c.id: c for c in cases}
    for row in answers.get("resolutions") or []:
        case = by_id.get(str(row.get("id", "")).strip())
        value = str(row.get("value", "")).strip()
        if case is None or value == "unresolved" or value not in case.options:
            report.adjudicated.append(f"{row.get('id')}: unresolved")
            continue
        quote = re.sub(r"\s+", " ", str(row.get("quote", ""))).strip()
        try:
            span = span_tools.resolve(text, quote).as_record()
            span_tools.verify(text, span)
        except Exception:
            report.adjudicated.append(f"{case.id}: rejected, the quote is not in the paper")
            continue
        entity = next((e for e in record.get(case.container) or []
                       if isinstance(e, dict) and e.get("local_id") == case.local_id), None)
        if entity is None:
            continue
        entity[case.slot] = {
            "extraction_status": "extracted", "value": value, "value_source": "reported",
            "evidence": {"status": "present",
                         "sets": [{"source": "repair_pass", "spans": [span]}]}}
        if case.clears and value in guards.UNRESTRICTED:
            entity[case.clears] = []
        report.adjudicated.append(f"{case.id}: {value}")


def run(record: MutableMapping[str, Any], text: str, sch: Schema, *, study_id: str,
        proposer: Any = None, checker: Checker | None = None, caller: Any = None,
        model: str = "", threshold: float = 0.5) -> Report:
    """Repair `record` in place. Returns what happened, including anything it broke."""
    from copy import deepcopy

    before = deepcopy(record)
    report = Report()

    if proposer is not None:
        _sweep(record, text, sch, proposer, checker, threshold, report)
    if caller is not None and model:
        adjudicate(record, sch, text, caller, study_id=study_id, model=model, report=report)

    report.introduced = Validator(sch, None).diff(before, record)
    return report


def _sweep(record: MutableMapping[str, Any], text: str, sch: Schema, proposer: Any,
           checker: Checker | None, threshold: float, report: Report) -> None:
    """Ask the proposer per class, targets first, and write what survives the guards."""
    from pondie.extraction.recall import CLASS_OF, candidates, sweep_order

    for container in sweep_order(sch, [k for k in CLASS_OF if record.get(k)]):
        class_name = CLASS_OF[container]
        proposals = proposer.propose(
            sch, class_name, text,
            candidates(sch, record, class_name, lambda e, _c="": edit_module.label_of(e)))
        by_id = {e.get("local_id"): e for e in record.get(container) or []
                 if isinstance(e, Mapping)}
        for proposal in proposals:
            entity = by_id.get(str(proposal.get("local_id") or "").strip())
            if entity is None:
                entity, why = edit_module.create(sch, record, class_name, proposal, text)
                if entity is None:
                    report.refused.append(guards.Refusal(container, why))
                    continue
                record.setdefault(container, []).append(entity)
                report.written.append(f"{container}/{entity['local_id']} created")
            log = edit_module.apply(sch, record, class_name, entity, proposal, text)
            report.written += [f"{container}/{entity['local_id']}.{s}" for s, _v in log.written]
            report.refused += log.refused
