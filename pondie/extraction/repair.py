"""Repair a built record: propose, ground, guard, and put what is left to a model.

Runs after `build`, on a record that already exists, and changes it in place. Four steps,
narrowing at each one:

  1. **propose** -- a local model reads the methods and results and returns entities of one
     class at a time, with the entities it may point at listed per reference slot.
  2. **ground** -- a local entailment model scores each proposal against the passage offered
     for it, so a proposal with no warrant is not written.
  3. **guard** -- `record.edit` refuses the writes that would damage the record, and says
     why. Every write goes past it, step 4 included.
  4. **adjudicate** -- what is left is a contradiction the record cannot settle from its own
     contents. That goes to the extraction model, once, with the paper, and its answer is
     written through the same guards as everything else.

Every step is optional. With no proposer and no checker there is nothing to propose and
nothing to ground, and the stage does only step 4; with `adjudicate` off it does nothing at
all. This is deliberate: the local models want a GPU, and a run without one should still be
able to resolve what the paper plainly answers.
"""

from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence

from pondie.extraction.evidence import grounding
from pondie.extraction.evidence.grounding import Checker
from pondie.extraction.record import edit as edit_module
from pondie.extraction.record.edit import Edit, Refusal, UNRESTRICTED, refusals
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


@functools.lru_cache(maxsize=1)
def models(visible_devices: str, proposer_device: int) -> tuple[Any, Any]:
    """The two local models, built once per process and shared by every paper.

    Cached because they are ~10 GB of weights and the stage runs per paper under a thread
    pool: constructing them in `run` loaded and freed them for each of 52 papers, and two
    workers put two proposers on one card. `schema.reader` caches the same way and for the
    same reason -- the objects are immutable readers, not mutable state.

    Visibility is set here, before either import, because MiniCheck places itself from
    `CUDA_VISIBLE_DEVICES` and takes no device argument. It has to happen before torch
    initialises CUDA, which an earlier stage may already have done -- so a run that wants a
    specific placement sets it in the environment, and this is the fallback rather than the
    mechanism.
    """
    if visible_devices and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    from pondie.extraction.evidence.grounding import MiniCheck
    from pondie.extraction.recall import NuExtract

    return NuExtract(device=proposer_device), MiniCheck()


@dataclass
class Report:
    """What one pass did to one record."""

    written: list[str] = field(default_factory=list)
    refused: list[Refusal] = field(default_factory=list)
    adjudicated: list[str] = field(default_factory=list)
    #: What the adjudication spent, so a run can sum it. Every other stage returns its cost
    #: rather than logging it, for the reason `llm.py` gives: a stage that has to scrape its
    #: own spend out of its own logging cannot be summed.
    cost: Any = None
    traces: tuple = ()
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
            if scope not in UNRESTRICTED or not regions:
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
               *, study_id: str, model: str, report: Report) -> Any:
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
        return None
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
    # `payload`, which is what a ModelReply carries. `body` is an attribute of
    # `MalformedReply` -- the exception -- so a getattr for it fell through to the reply
    # itself and json.loads got "payload={...} cost=Cost(...)".
    answers = reply.payload

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
        # Through the guards, like every other write. Coercing a cited scope to a bare enum
        # is exactly the shape `refuses_losing_the_warrant` exists for, and step 4 was the
        # one path that bypassed it.
        edit = Edit(record, entity, case.slot, value)
        if refused := refusals(edit):
            report.refused.extend(refused)
            report.adjudicated.append(f"{case.id}: refused, {refused[0].why}")
            continue
        entity[case.slot] = {
            "extraction_status": "extracted", "value": value, "value_source": "reported",
            "evidence": {"status": "present",
                         "sets": [{"source": "repair_pass", "spans": [span]}]}}
        if case.clears and value in UNRESTRICTED:
            entity[case.clears] = []
        report.adjudicated.append(f"{case.id}: {value}")
    # Returned, not logged. `llm.py`: "Cost is returned rather than logged, because a stage
    # that has to scrape its own spend out of its own logging cannot be summed."
    return reply


def run(record: MutableMapping[str, Any], text: str, sch: Schema, *, study_id: str,
        proposer: Any = None, checker: Checker | None = None, caller: Any = None,
        model: str = "", threshold: float = 0.5) -> Report:
    """Repair `record` in place. Returns what happened, including anything it broke."""
    from copy import deepcopy

    before = deepcopy(record)
    report = Report()

    # The paper's own abbreviation table, built once. Without it `same_entity` has nothing
    # to expand and cannot tell "CAPS total score" from "clinician-administered PTSD scale
    # (CAPS)" -- the check exists for that case and was unreachable in production while
    # every caller passed None.
    abbreviations = _abbreviations(text)
    # Methods and results, not the whole paper. Both models take this as their premise, and
    # a proposer truncating the first 45,000 characters of a full document sees title,
    # abstract and introduction before it sees a method. `sectionize` falls back to the whole
    # text when it finds nothing, which is the honest behaviour for a paper it cannot split.
    premise = _premise(text)
    if proposer is not None:
        _sweep(record, premise, sch, proposer, checker, threshold, report, abbreviations)
    if checker is not None:
        grounding.drop_unsupported_spans(record, checker, report.refused)
    if caller is not None and model:
        reply = adjudicate(record, sch, text, caller, study_id=study_id, model=model,
                           report=report)
        if reply is not None:
            report.cost = reply.cost
            report.traces = ((reply.trace_id, reply.cache_status),) if reply.trace_id else ()

    # The extraction schema, not `sch`. A record is extraction-shaped -- every value in an
    # `ExtractedValue` wrapper -- while `sch` is storage, where `name` is a plain string.
    # Validating one against the other reported every wrapper as "must be a string, got
    # dict": twenty findings on one record, none of them real.
    #
    # (The pass itself reasons with storage, because that is where `required`, `multivalued`
    # and the vocabularies live. Only the validation needs the projection.)
    from pondie.extraction.record.validate import EXTRACTION_SCHEMA
    from pondie.schema import reader

    # `text`, not None: passing None disables `check_span`'s span verification, which is the
    # one check that catches a bad offset written by this pass -- switched off inside the
    # function whose job is to report what the attempt broke.
    checker_schema = reader.load(EXTRACTION_SCHEMA)
    report.introduced = Validator(checker_schema, text or None).diff(before, record)
    return report


#: Sections whose prose describes what was done and what was found. An entity is judged to
#: exist against these; a paper's introduction describes other people's studies.
PREMISE_SECTIONS = ("method", "material", "result")


def _premise(text: str) -> str:
    """The methods and results, or the whole text where they cannot be found.

    Measured over 40 papers: 30 slice, 10 fall back because the sectioniser finds no method
    heading in them -- which is the honest answer for those, and why the fallback exists.
    """
    from pondie.extraction.evidence.retrieval import sectionize

    spans = [text[start:end] for start, end, label in sectionize(text)
             if any(word in label.lower() for word in PREMISE_SECTIONS)]
    joined = "\n\n".join(spans)
    return joined if len(joined) >= max(2_000, len(text) // 10) else text


def _abbreviations(text: str) -> Any:
    """The paper's own expansions, or None where the vocabulary package is unavailable.

    Without it `same_entity` has nothing to expand and cannot tell "CAPS total score" from
    "clinician-administered PTSD scale (CAPS)".
    """
    try:
        from pondie.vocabularies.abbreviations import Abbreviations

        return Abbreviations.load().for_paper(text)
    except Exception:  # noqa: BLE001 -- an optional vocabulary, not a failure
        return None


def _sweep(record: MutableMapping[str, Any], text: str, sch: Schema, proposer: Any,
           checker: Checker | None, threshold: float, report: Report,
           abbreviations: Any = None) -> None:
    """Ask the proposer per class, targets first, and write what survives the guards."""
    from pondie.extraction.recall import candidates, sweep_order

    # Every class, not only the populated ones. Sweeping what the record already has asks
    # the model to improve what was found and never to find what was missed -- and an empty
    # container is where recall matters most: 16508348 declares no regions at all while four
    # of its analyses search the hippocampus.
    by_container = {key: cls for cls, key in sch.containers().items()}
    for container in sweep_order(sch, list(by_container)):
        class_name = by_container[container]
        proposals = proposer.propose(
            sch, class_name, text,
            candidates(sch, record, class_name))
        by_id = {e.get("local_id"): e for e in record.get(container) or []
                 if isinstance(e, Mapping)}
        graded = grounding.supported(proposals, class_name, text, checker, threshold,
                                     report.refused)
        for proposal in graded:
            entity = by_id.get(str(proposal.get("local_id") or "").strip())
            if entity is None:
                entity, why = edit_module.create(sch, record, class_name, proposal, text,
                                                 abbreviations)
                if entity is None:
                    report.refused.append(Refusal(container, why))
                    continue
                record.setdefault(container, []).append(entity)
                report.written.append(f"{container}/{entity['local_id']} created")
            log = edit_module.apply(sch, record, class_name, entity, proposal, text,
                                    abbreviations)
            report.written += [f"{container}/{entity['local_id']}.{s}" for s, _v in log.written]
            report.refused += log.refused
