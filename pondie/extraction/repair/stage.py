"""Repair a built record: propose, guard, and put what is left to a model.

    __init__   the three steps, and the model calls two of them make
    guard      write a proposed change into a record, or say why not

Runs after `build`, on a record that already exists, and changes it in place. Four steps,
narrowing at each one:

  1. **propose** -- the model reads the methods and results and returns entities of one
     class at a time, with the entities it may point at listed per reference slot.
  2. **guard** -- `repair.guard` refuses the writes that would damage the record, and says
     why. Every write goes past it, step 3 included.
  3. **adjudicate** -- what is left is a contradiction the record cannot settle from its own
     contents (`contradictions` lists the kinds). That goes to the model, once, with the
     paper, and its answer is written through the same guards as everything else.

There used to be a **ground** step between 1 and 2: a local entailment model scored each
proposal against the passage offered for it. It went with the local models, and the guards
are now the only thing standing between a proposal and the record -- which is why
`refuses_an_unwarranted_replacement` exists.

Both steps are optional. With no proposer there is nothing to propose and the stage does
only the adjudication; with `adjudicate` off it does neither.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping

from pondie.extraction.evidence.retrieval import sectionize
from pondie.extraction.models import ModelCall
from pondie.extraction.prompt import render
from pondie.extraction.record import direction, rules
from pondie.extraction.record import spans as span_tools
from pondie.extraction.record.effect import (
    NO_LABEL,
    UNDETERMINED_VARIATION,
    UNRESOLVED_TERM,
    derive_effect_kind,
    levels_a_cell_may_name,
    terms_in_scope,
)
from pondie.extraction.record.validate import EXTRACTION_SCHEMA, Validator
from pondie.extraction.repair import guard as edit_module
from pondie.extraction.repair.guard import UNRESTRICTED, Edit, Refusal, refusals
from pondie.extraction.repair.propose import candidates, existing, sweep_order
from pondie.extraction.record.fix.link import merge_restated_terms
from pondie.extraction.record.fix.reachable import drop_unreachable
from pondie.formats import parse_keys, values
from pondie.schema import reader
from pondie.schema.reader import Schema
from pondie.vocabularies.abbreviations import Abbreviations

#: Stamped on a record this pass changed. Bump it when a change would make two repaired
#: records incomparable, which is the rule `EXTRACTOR_VERSION` states for the extractor.
REPAIRER = "pondie-repair-1"

ADJUDICATION_SYSTEM = """\
You resolve contradictions in a structured record extracted from a neuroimaging paper.

Each case names fields of the record that cannot all be true, and lists the values each may
take. Answer with the value the paper supports and one verbatim sentence from the paper that
shows it. Copy the sentence exactly; do not paraphrase, join, or trim it.

A case that asks for regions is a region-of-interest procedure with none named. If it was
restricted to regions, answer "roi" and name each region as the paper names it -- no
numbering, no description of where it lies. If the paper does not name them one by one,
answer "unresolved". If the procedure was not restricted, answer the scope it had.

Answer "unresolved" whenever the paper does not settle the case -- when it is silent,
ambiguous, or describes something the options do not cover. The record already reports the
contradiction, so a reviewer can see it; a confident wrong answer removes that."""


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
    #: Entities removed from the record because no analysis reached them. Reported rather
    #: than silent: the entity is out of the record, not deleted from the audit trail.
    dropped: list[str] = field(default_factory=list)
    #: Terms dropped as restatements of a lower stage's (`link.merge_restated_terms`).
    merged: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"wrote {len(self.written)}, "
            f"refused {len(self.refused)}, adjudicated {len(self.adjudicated)}, "
            f"introduced {len(self.introduced)}"
        )


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
    #: Where `slot` sits below the entity: `("effect", "cells", 0)` for a cell's level.
    within: tuple[str | int, ...] = ()
    #: The reference slot an answer of `roi` fills with the regions it names.
    names: str = ""
    #: What else an answer writes, by option, on the same owner as `slot`. An answer
    #: settling one half of a contradiction must make the other half agree with it.
    then: Mapping[str, Callable[[MutableMapping[str, Any], Any], list[str]]] = field(
        default_factory=dict
    )
    #: False for a case answered only through `then`: `slot` names what it is about, and
    #: no option is a value to write there.
    writes: bool = True


#: The scope/regions pairs: what volume was searched, and the regions it was restricted to.
_SCOPE_PAIRS = (
    ("analyses", "Analysis", "spatial_scope", "regions"),
    ("inference_settings", "InferenceSettings", "correction_scope", "correction_regions"),
)


def contradictions(record: Mapping[str, Any], sch: Schema) -> list[Case]:
    """The validator's findings that can be put as "choose one of these and quote it".

    - a whole-brain or searchlight scope beside named regions, and an ROI scope beside none;
    - a cell level that none of its term's declared levels spells;
    - an `effect.kind` its own cells contradict;
    - an analysis named "A < B" whose cells sign A above B, or the reverse.

    Not dangling references, the largest group a repaired record carries: those come from a
    deletion, and the answer is not to delete rather than to ask.
    """
    return _scopes(record, sch) + _levels(record) + _kinds(record) + _named_directions(record)


def _scopes(record: Mapping[str, Any], sch: Schema) -> list[Case]:
    out: list[Case] = []
    for container, class_name, scope_slot, region_slot in _SCOPE_PAIRS:
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
            where = f"{container}/{entity.get('local_id')}"
            if scope in UNRESTRICTED and regions:
                named = ", ".join(_label(record, r) for r in regions)
                out.append(
                    Case(
                        id=f"{where}/{scope_slot}",
                        question=(
                            f"{scope_slot} is '{scope}' while {region_slot} names {named}. "
                            f"A whole-brain or searchlight procedure is restricted to no "
                            f"region, so at most one of these is right."
                        ),
                        options=allowed,
                        container=container,
                        local_id=str(entity.get("local_id")),
                        slot=scope_slot,
                        clears=region_slot,
                    )
                )
            elif scope == "roi" and not regions:
                out.append(
                    Case(
                        id=f"{where}/{region_slot}",
                        question=(
                            f"{class_name} {edit_module.label_of(entity)!r} has {scope_slot} "
                            f"'roi' and names no {region_slot}. Which regions was it "
                            f"restricted to?"
                        ),
                        options=allowed,
                        container=container,
                        local_id=str(entity.get("local_id")),
                        slot=scope_slot,
                        names=region_slot,
                    )
                )
    return out


def _levels(record: Mapping[str, Any]) -> list[Case]:
    models = rules.model_index(record)
    out: list[Case] = []
    for analysis in record.get("analyses") or []:
        if not isinstance(analysis, Mapping):
            continue
        terms = terms_in_scope(analysis.get("model_estimation"), models)
        cells = (analysis.get("effect") or {}).get("cells") or []
        for position, cell in enumerate(cells):
            if not isinstance(cell, Mapping):
                continue
            term = terms.get(cell.get("term"))
            level = values.read(cell.get("level"))
            if term is None or not isinstance(level, str):
                continue
            declared = levels_a_cell_may_name(term, terms, cell.get("direction"))
            if not declared or level in declared:
                continue  # no levels at all is the term's type, not a spelling to choose
            out.append(
                Case(
                    id=f"analyses/{analysis.get('local_id')}/effect.cells[{position}].level",
                    question=(
                        f"A cell of analysis {edit_module.label_of(analysis)!r} names level "
                        f"{level!r} of term {edit_module.label_of(term)!r}, which may name "
                        f"only the levels listed. Which one does the cell mean?"
                    ),
                    options=declared,
                    container="analyses",
                    local_id=str(analysis.get("local_id")),
                    slot="level",
                    within=("effect", "cells", position),
                )
            )
    return out


def _kinds(record: Mapping[str, Any]) -> list[Case]:
    models = rules.model_index(record)
    out: list[Case] = []
    for analysis in record.get("analyses") or []:
        if not isinstance(analysis, Mapping):
            continue
        effect = analysis.get("effect")
        if not isinstance(effect, Mapping):
            continue
        terms = terms_in_scope(analysis.get("model_estimation"), models)
        derived, why = derive_effect_kind(effect.get("cells"), terms)
        stated = values.read(effect.get("kind"))
        if derived in (NO_LABEL, UNDETERMINED_VARIATION, UNRESOLVED_TERM) or stated in (
            None,
            derived,
        ):
            continue
        question = (
            f"Analysis {edit_module.label_of(analysis)!r} says its effect is "
            f"{stated!r}, but its cells describe {derived!r} ({why}). Which is the "
            f"test the paper reports?"
        )
        then = {}
        if _withheld(effect.get("cells"), terms) == stated:
            question += (
                " Its cells are all `undirected`, which marks a test with no per-level sign"
                " (an F or chi-square over the factor). A directional test (t, z) that found"
                f" nothing, or whose direction the paper does not print, is still {stated!r};"
                " answering that marks the cells' sign as not reported."
            )
            then = {str(stated): _withhold_signs}
        out.append(
            Case(
                id=f"analyses/{analysis.get('local_id')}/effect.kind",
                question=question,
                options=(derived, str(stated)),
                container="analyses",
                local_id=str(analysis.get("local_id")),
                slot="kind",
                within=("effect",),
                then=then,
            )
        )
    return out


_NOT_REPORTED = values.wrap(None, source="reported", evidence="not_applicable")


def _undirected(cell: Any) -> bool:
    return isinstance(cell, Mapping) and values.read(cell.get("direction")) == "undirected"


def _withheld(cells: Any, terms: Mapping[str, Any]) -> str | None:
    """The kind `cells` would derive with every `undirected` sign marked not reported, when
    every cell is undirected; else None."""
    if not cells or not all(_undirected(c) for c in cells):
        return None
    withheld = [{**c, "direction": _NOT_REPORTED} for c in cells]
    return derive_effect_kind(withheld, terms)[0]


def _withhold_signs(effect: MutableMapping[str, Any], _span: Any = None) -> list[str]:
    """Mark each `undirected` cell's sign as not reported: a directional test whose
    direction the paper did not print (extraction-readme.md §2)."""
    changed = []
    for index, cell in enumerate(effect.get("cells") or []):
        if _undirected(cell):
            cell["direction"] = values.wrap(None, source="reported", evidence="not_applicable")
            changed.append(f"cells[{index}].direction: undirected -> not_reported")
    return changed


_SIGN = {"positive": 1, "negative": -1}


def _named_against_cells(name: Any, cells: Any) -> bool:
    """Whether the name's stated comparison (`direction.direction_of`, which `fill_directions`
    signs cells by) is the reverse of every signed cell's sign, on at least two cells."""
    if not isinstance(name, str) or not isinstance(cells, list):
        return False
    verdicts = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        sign, level = values.read(cell.get("direction")), values.read(cell.get("level"))
        if sign not in _SIGN or not isinstance(level, str) or not level:
            continue
        named = direction.direction_of(level, name)
        if named is None:
            return False
        verdicts.append(named == sign)
    return len(verdicts) >= 2 and not any(verdicts)


def _named_directions(record: Mapping[str, Any]) -> list[Case]:
    out: list[Case] = []
    for analysis in record.get("analyses") or []:
        if not isinstance(analysis, Mapping) or not isinstance(analysis.get("effect"), Mapping):
            continue
        name = values.read(analysis.get("name"))
        if not _named_against_cells(name, analysis["effect"].get("cells")):
            continue
        out.append(
            Case(
                id=f"analyses/{analysis.get('local_id')}/effect.cells.direction",
                question=(
                    f"Analysis {name!r} is named for one direction of comparison, and its "
                    f"cells sign the levels the other way.{_located(record, analysis)} "
                    f"Which does the paper report: "
                    f"'name' (the cells' signs are reversed) or 'cells' (the name is)? Quote "
                    f"what states the direction of THIS comparison -- its table row or "
                    f"heading, or its sentence -- not another contrast's result. A contrast "
                    f"that found nothing keeps the signs of what was tested."
                ),
                options=("name", "cells"),
                container="analyses",
                local_id=str(analysis.get("local_id")),
                slot="effect",
                then={"name": _reverse_signs, "cells": _reverse_name},
                writes=False,
            )
        )
    return out


def _reverse_signs(analysis: MutableMapping[str, Any], span: Any = None) -> list[str]:
    """Swap every signed cell's direction, cited by the adjudicated sentence."""
    changed = []
    for index, cell in enumerate((analysis.get("effect") or {}).get("cells") or []):
        sign = values.read(cell.get("direction")) if isinstance(cell, Mapping) else None
        if sign not in _SIGN:
            continue
        flipped = direction.reverse(sign)
        cell["direction"] = edit_module.cited(flipped, span)
        changed.append(f"cells[{index}].direction: {sign} -> {flipped}")
    return changed


def _reverse_name(analysis: MutableMapping[str, Any], _span: Any = None) -> list[str]:
    """Swap the comparison in the analysis's name: the cells are right and it is not."""
    text = values.read(analysis.get("name"))
    swapped = direction.reverse_comparison(text) if isinstance(text, str) else None
    if swapped is None:
        return []
    analysis["name"] = values.wrap(swapped, source="generated", evidence="not_found")
    return [f"name: {text!r} -> {swapped!r}"]


def _located(record: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    """" It is row group 3 of Table 3." when the analysis is linked to one; else ""."""
    key = values.read(analysis.get("source_table_analysis"))
    tables = {t.get("local_id"): t for t in record.get("tables") or [] if isinstance(t, Mapping)}
    linked = [tables[t] for t in analysis.get("tables") or [] if t in tables]
    if not isinstance(key, str) or "#" not in key or len(linked) != 1:
        return ""
    label = values.read(linked[0].get("table_number"))
    if not label:
        return ""
    label = label if str(label).lower().startswith("table") else f"Table {label}"
    ordinal = parse_keys.split(key)[1]
    # A CoordinateParse key ends in a digest, which numbers nothing.
    return f" It is row group {ordinal} of {label}." if ordinal.isdigit() else f" It is from {label}."


def _consequences(case: Case, value: str, owner: MutableMapping[str, Any], span: Any) -> list[str]:
    """What an answer writes besides its slot -- whether the slot changed or was kept. A
    kept `whole_brain` that left its regions beside it settled nothing (33169525)."""
    done = []
    if case.clears and value in UNRESTRICTED and owner.get(case.clears):
        owner[case.clears] = []
        done.append(f"{case.clears} cleared")
    return done + (case.then[value](owner, span) if value in case.then else [])


def _descend(entity: Any, path: tuple[str | int, ...]) -> MutableMapping[str, Any] | None:
    node = entity
    for step in path:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return None
    return node if isinstance(node, MutableMapping) else None


def _name_regions(
    record: MutableMapping[str, Any],
    sch: Schema,
    entity: MutableMapping[str, Any],
    case: Case,
    row: Mapping[str, Any],
    text: str,
    quote: str,
    abbreviations: Any,
    report: Report,
) -> None:
    """Point `case.names` at the regions an answer named, declaring the ones not yet held."""
    named: list[str] = []
    for proposal in row.get("regions") or []:
        if isinstance(proposal, str):
            proposal = {"name": proposal}
        if not isinstance(proposal, Mapping):
            continue
        held = edit_module.resolve(record, sch, "Region", proposal.get("name"), abbreviations)
        if held:
            named += [r for r in held if r not in named]
            continue
        region, why = edit_module.create(
            sch, record, "Region", proposal, text, abbreviations, quote=quote
        )
        if region is None:
            report.refused.append(Refusal("regions", why, proposal.get("name")))
            continue
        record.setdefault("regions", []).append(region)
        report.written.append(f"regions/{region['local_id']} created")
        named.append(region["local_id"])
    if not named:
        report.adjudicated.append(f"{case.id}: unresolved, no region could be named")
        return
    class_name = sch.classes_by_container()[case.container]
    log = edit_module.apply(sch, record, class_name, entity, {case.names: named}, text)
    report.written += [f"{case.container}/{case.local_id}.{s}" for s, _v in log.written]
    report.refused += log.refused
    if log.written:
        report.adjudicated.append(f"{case.id}: roi, {', '.join(named)}")
    else:
        report.adjudicated.append(f"{case.id}: refused, {log.refused[0].why}")


def _label(record: Mapping[str, Any], local_id: str) -> str:
    for entities in record.values():
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if isinstance(entity, Mapping) and entity.get("local_id") == local_id:
                return edit_module.label_of(entity)
    return local_id


def adjudicate(
    record: MutableMapping[str, Any],
    sch: Schema,
    text: str,
    caller: Any,
    *,
    study_id: str,
    model: str,
    report: Report,
    service_tier: str = "",
    effort: str = "low",
    abbreviations: Any = None,
) -> Any:
    """Put the unresolved contradictions to the extraction model, once, with the paper.

    A resolution is applied only when its quote resolves to a span of this paper, by the same
    `spans.resolve`/`verify` the extractor is held to. A plausible value with an invented
    sentence reads exactly like a resolved case, so the quote is the gate rather than a
    courtesy.
    """
    cases = contradictions(record, sch)
    if not cases:
        return None
    listing = "\n\n".join(
        f"case {i + 1} (id {c.id}):\n  {c.question}\n"
        f"  permissible values: {', '.join(repr(o) for o in c.options)}, or unresolved"
        for i, c in enumerate(cases)
    )
    methods = getattr(sch.enums.get("RegionDefinition"), "permissible_values", {}) or {}
    reply = caller(
        ModelCall(
            model=model,
            # Paper in the system half; only the cases vary between calls. See the note
            # in `propose_with_extractor._generate`.
            system=f"{ADJUDICATION_SYSTEM}\n\n{render.paper_block(text)}",
            effort=effort,
            # Room for reasoning: at medium effort it is spent from the same budget.
            max_output_tokens=16_000,
            service_tier=service_tier,
            prompt=(
                f"## Cases\n\n{listing}\n\n"
                'Reply with JSON, {"resolutions": [{"id": ..., "value": ..., '
                '"quote": ...}]}, using the case id verbatim and an empty quote '
                "for anything unresolved. For a case that asks for regions, answered "
                '"roi", add "regions": [{"name": ..., "definition_method": ...}], where '
                "definition_method is how the region was delimited: one of "
                f"{', '.join(methods)}; else a few words of the paper's saying how "
                '(e.g. "manually traced"); else "not_reported".'
            ),
        ),
        paper=study_id,
        stage="repair",
    )
    # `payload`, which is what a ModelReply carries. `body` is an attribute of
    # `MalformedReply` -- the exception -- so a getattr for it fell through to the reply
    # itself and json.loads got "payload={...} cost=Cost(...)".
    answers = reply.payload

    by_id = {c.id: c for c in cases}
    answered: list[str] = []
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
            report.adjudicated.append(
                f"{case.id}: rejected, the quote is not in the paper ({quote[:80]!r})"
            )
            continue
        entity = next(
            (
                e
                for e in record.get(case.container) or []
                if isinstance(e, dict) and e.get("local_id") == case.local_id
            ),
            None,
        )
        owner = _descend(entity, case.within)
        if owner is None:
            continue
        if case.names and value not in UNRESTRICTED:
            _name_regions(record, sch, entity, case, row, text, quote, abbreviations, report)
            continue
        if not case.writes or values.read(owner.get(case.slot)) == value:
            also = _consequences(case, value, owner, span)
            said = f"kept {value}" if case.writes else value
            report.adjudicated.append(f"{case.id}: {said}" + "".join(f"; {a}" for a in also))
            answered.append(case.id)
            continue
        # Through the guards, like every other write. Coercing a cited free-text scope to a
        # bare enum is exactly the shape `refuses_losing_the_warrant` exists for; swapping
        # one of the case's options for another is the choice it asked for (`choices`). The
        # quote goes with it: it was resolved against this paper just above, and without it
        # `refuses_an_unwarranted_replacement` would judge a cited edit as a bare one.
        edit = Edit(record, owner, case.slot, value, text, quote, choices=case.options)
        if refused := refusals(edit):
            report.refused.extend(refused)
            report.adjudicated.append(f"{case.id}: refused, {refused[0].why}")
            continue
        owner[case.slot] = edit_module.cited(value, span)
        also = _consequences(case, value, owner, span)
        report.adjudicated.append(f"{case.id}: {value}" + "".join(f"; {a}" for a in also))
        answered.append(case.id)
    # An answer counts only if the contradiction it answered is gone. "kept contrast" on
    # cells that still derive `omnibus` read as resolved while the record still disagreed.
    standing = {c.id for c in contradictions(record, sch)}
    for case_id in answered:
        if case_id in standing:
            report.adjudicated.append(f"{case_id}: still contradicted after the answer")
    # Returned, not logged. `llm.py`: "Cost is returned rather than logged, because a stage
    # that has to scrape its own spend out of its own logging cannot be summed."
    return reply


def run(
    record: MutableMapping[str, Any],
    text: str,
    sch: Schema,
    *,
    study_id: str,
    proposer: Any = None,
    caller: Any = None,
    model: str = "",
    service_tier: str = "",
    iterations: int = 2,
    effort: str = "low",
) -> Report:
    """Repair `record` in place. Returns what happened, including anything it broke."""
    from copy import deepcopy

    before = deepcopy(record)
    report = Report()

    # The paper's own abbreviation table, built once. Without it `same_entity` has nothing
    # to expand and cannot tell "CAPS total score" from "clinician-administered PTSD scale
    # (CAPS)" -- the check exists for that case and was unreachable in production while
    # every caller passed None.
    abbreviations = _abbreviations(text, study_id)
    # Methods and results, not the whole paper. Both models take this as their premise, and
    # a proposer truncating the first 45,000 characters of a full document sees title,
    # abstract and introduction before it sees a method. `sectionize` falls back to the whole
    # text when it finds nothing, which is the honest behaviour for a paper it cannot split.
    premise = _premise(text)
    # No gate: the proposer answers over the network, so the sweep is a wait rather than
    # work this process does. This was a semaphore back when the proposer held a card.
    for _pass in range(iterations if proposer is not None else 0):
        before_pass = len(report.written)
        _sweep(record, premise, text, sch, proposer, report, abbreviations, study_id)
        if len(report.written) == before_pass:
            break  # nothing changed, so a further pass sees the same
    if caller is not None and model:
        reply = adjudicate(
            record,
            sch,
            text,
            caller,
            study_id=study_id,
            model=model,
            report=report,
            service_tier=service_tier,
            effort=effort,
            abbreviations=abbreviations,
        )
        if reply is not None:
            report.cost = reply.cost
            report.traces = ((reply.trace_id, reply.cache_status),) if reply.trace_id else ()

    # Here as well as in `build`: a link this pass wrote to `inputs_from` can put a model's
    # term above the same term at the stage it now reaches, which `build` never saw.
    report.merged += merge_restated_terms(record, sch)

    # LAST, because it judges what everything above produced. The proposer creates an entity
    # whenever the model proposes one and never asks whether anything will point at it:
    # audited over 126 papers, 66 of 69 orphans appear in no payload at all and none was
    # ever referenced. See `reachable` for what that catches and why tables are exempt.
    report.dropped += drop_unreachable(record)

    # The extraction schema, not `sch`. A record is extraction-shaped -- every value in an
    # `ExtractedValue` wrapper -- while `sch` is storage, where `name` is a plain string.
    # Validating one against the other reported every wrapper as "must be a string, got
    # dict": twenty findings on one record, none of them real.
    #
    # (The pass itself reasons with storage, because that is where `required`, `multivalued`
    # and the vocabularies live. Only the validation needs the projection.)
    # `text`, not None: passing None disables `check_span`'s span verification, which is the
    # one check that catches a bad offset written by this pass -- switched off inside the
    # function whose job is to report what the attempt broke.
    checker_schema = reader.load(EXTRACTION_SCHEMA)
    report.introduced = Validator(checker_schema, text or None).diff(before, record)
    if report.written or report.adjudicated:
        # A repaired record is not the record the extractor produced, and saying otherwise
        # makes two records that differ look comparable. `tools/adjudicate` re-stamps for the
        # same reason -- "so the corrected record is honestly a different extractor's output
        # rather than a doctored copy of the model's".
        record.setdefault("extraction_metadata", {})["repaired_by"] = REPAIRER
    return report


#: Sections whose prose describes what was done and what was found. An entity is judged to
#: exist against these; a paper's introduction describes other people's studies.
PREMISE_SECTIONS = ("method", "material", "result")


def _premise(text: str) -> str:
    """The methods and results, or the whole text where they cannot be found.

    Measured over 40 papers: 30 slice, 10 fall back because the sectioniser finds no method
    heading in them -- which is the honest answer for those, and why the fallback exists.
    """
    spans = [
        text[start:end]
        for start, end, label in sectionize(text)
        if any(word in label.lower() for word in PREMISE_SECTIONS)
    ]
    joined = "\n\n".join(spans)
    return joined if len(joined) >= max(2_000, len(text) // 10) else text


def _abbreviations(text: str, study_id: str) -> Any:
    """The paper's own expansions, or None where the vocabulary package is unavailable.

    Without it `same_entity` has nothing to expand and cannot tell "CAPS total score" from
    "clinician-administered PTSD scale (CAPS)".

    Scoped to the study rather than to the text alone. `for_paper` needs the paper's name
    to reach the store's own rows for it, and passing only the text left it re-mining and
    nothing else.
    """
    if not str(study_id or "").strip():
        return None
    try:
        return Abbreviations.load().for_paper(text, study_id)
    except Exception:  # noqa: BLE001 -- an optional vocabulary, not a failure
        return None


def _sweep(
    record: MutableMapping[str, Any],
    premise: str,
    document: str,
    sch: Schema,
    proposer: Any,
    report: Report,
    abbreviations: Any = None,
    study_id: str = "",
) -> None:
    """Ask the proposer per class, targets first, and write what survives the guards.

    Two texts, and conflating them writes spans that address the wrong string. The models see
    `premise` -- the methods and results -- because that is where an entity is described. A
    span is resolved against `document`, the whole normalized text, because that is what
    every offset in the record is measured from and what `source_text_hash` covers. Passing
    the premise to both put offsets into the slice: "span text disagrees with source at
    2180-2225" on three of three spot-checked papers.
    """
    # Every class, not only the populated ones. Sweeping what the record already has asks
    # the model to improve what was found and never to find what was missed -- and an empty
    # container is where recall matters most: 16508348 declares no regions at all while four
    # of its analyses search the hippocampus.
    by_container = sch.classes_by_container()
    order = sweep_order(sch, list(by_container))
    context = {
        by_container[c]: existing(sch, record, by_container[c])
        + candidates(sch, record, by_container[c])
        for c in order
    }
    # A proposer that can answer about several classes at once is asked once rather than
    # per class: the premise is the same paper every time, and a network proposer paid to
    # read it twenty-eight times a paper.
    batched: dict[str, list] | None = None
    if hasattr(proposer, "propose_many"):
        try:
            batched = proposer.propose_many(
                sch, [by_container[c] for c in order], premise, context
            )
        except Exception as error:  # noqa: BLE001 -- fall back to the per-class sweep
            report.refused.append(
                Refusal(
                    "sweep",
                    f"batched proposal failed ({type(error).__name__}); "
                    f"asked class by class instead",
                )
            )
    for container in order:
        class_name = by_container[container]
        proposals = (
            batched.get(class_name, [])
            if batched is not None
            else proposer.propose(sch, class_name, premise, context[class_name])
        )
        by_id = {
            e.get("local_id"): e for e in record.get(container) or [] if isinstance(e, Mapping)
        }
        # One per class sweep, and passed to every `apply` in it. An exclusive reference
        # slot names what belongs to one entity, so the same target list arriving on a
        # second entity of this class is a copy: on 18823721 the pass wrote the same four
        # questionnaires to `grp_opioid_patients` and `grp_controls` as their
        # `diagnostic_instrument`, and two of the four were administered to the patients
        # only. Held here rather than in `edit` because the first write is right and only
        # the second is wrong, which nothing looking at one edit can see.
        # Only what would be created is asked to justify its existence. A proposal naming an
        # entity the record already holds is an *edit*, and the extractor established that
        # entity already -- re-asking whether the paper describes it rejects corrections to
        # things that are plainly there. On 26424424 that cost 61 refusals and all but one of
        # the links: the model returned every ROI of all three analyses, by their exact ids,
        # and the existence gate threw the proposals away before the edit was attempted.
        edits = [p for p in proposals if str(p.get("local_id") or "").strip() in by_id]
        news = [p for p in proposals if p not in edits]
        for proposal in edits + news:
            entity = by_id.get(str(proposal.get("local_id") or "").strip())
            if entity is None:
                entity, why = edit_module.create(
                    sch, record, class_name, proposal, document, abbreviations
                )
                if entity is None:
                    report.refused.append(Refusal(container, why))
                    continue
                record.setdefault(container, []).append(entity)
                report.written.append(f"{container}/{entity['local_id']} created")
            log = edit_module.apply(
                sch, record, class_name, entity, proposal, document, abbreviations
            )
            report.written += [f"{container}/{entity['local_id']}.{s}" for s, _v in log.written]
            report.refused += log.refused
