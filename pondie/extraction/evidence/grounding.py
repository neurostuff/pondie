"""Does this passage support this claim, and which claims can be asked at all?

`quote.py` decides which passages *warrant* a value -- the model's own citation, and the
cross-encoder's second opinion. Neither judges entailment: a retriever returns its best match
whether or not the match says anything, and on one field that was the acknowledgements
section offered as the warrant for a model term's type.

A grounding model answers the question the locators cannot. Measured on that case: 0.041 for
the acknowledgements sentence, 0.919 for a sentence that really says it, and 0.025 for a
sentence naming the wrong term against 0.952 for the right one. Wide enough that the
retriever can stay permissive, which is why it recovers the warrant on fields the extracting
model never quoted (5 of 14 sampled) without those recoveries being taken on trust.

`Checker` is a protocol, and the pass that uses it takes `None`, because the weights are a
heavyweight optional dependency and the rest of a repair is deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Protocol, Sequence

from pondie.extraction.record.edit import Refusal
from pondie.formats import values


@dataclass(frozen=True)
class Claim:
    """One thing the record asserts, and the text it is to be judged against."""

    #: The assertion in prose, e.g. "In the model estimation, the term X type is continuous."
    claim: str
    #: What it is checked against: the span that warrants it, or the methods and results
    #: when asking whether an entity exists at all.
    premise: str


class Checker(Protocol):
    """Scores each claim against its own premise, 0 to 1."""

    def score(self, claims: Sequence[Claim]) -> Sequence[float]: ...



#: subset on the slot would say it beside the definition instead of here.
#:
#: Slots whose value is a conclusion rather than a quotation. A paper states its scanner and
#: its sample size; it does not state that an analysis was `exploratory`, that a contrast
#: `direction` is negative, or that a scope is `whole_brain` -- those are read off the method
#: by whoever encodes it. Asking a checker for the sentence that supports one is asking for a
#: sentence that does not exist, and scoring its absence as unsupported marks a correct
#: reading wrong.
REASONED = frozenset({
    "spatial_scope", "correction_scope", "prespecification", "direction", "variation_level",
    "assignment_structure", "allocation", "blinding", "stage", "spatial_unit", "family",
    "inference_level", "region_type", "definition_method", "acquisition_type",
    "details_type", "value_source", "is_healthy", "type",
})


#: Addresses, not claims. A local_id is how the record refers to something internally, so a
#: checker asked whether the paper says "reg_hippocampus" always says no.
#:
#: `local_id` and `id` are here for completeness rather than reach: `iter_fields` yields only
#: `ExtractedValue` wrappers, and those two are plain strings that never appear as one. The
#: slots that do reach here are the reference-shaped ones a projection wraps.
IDENTIFIERS = frozenset({"local_id", "id", "source_table_analysis", "table_id"})


def groundable(slot: str, node: Mapping[str, Any]) -> bool:
    """Whether a sentence could support this field at all.

    Three exclusions. `value_source: generated` is the record's own: it marks a value the
    extraction system produced rather than read -- a mirrored contrast, a derived direction
    -- so there is no sentence behind it by construction. `REASONED` slots hold a judgement
    about the method rather than a thing the paper says. `IDENTIFIERS` hold an address.

    Getting this wrong is not neutral: a field wrongly called ungroundable keeps a bad
    citation, and a field wrongly called groundable loses a good one.
    """
    if values.read(node) is None:
        return False
    if (node or {}).get("value_source") == "generated":
        return False
    return slot not in REASONED and slot not in IDENTIFIERS


def supported(proposals: Sequence[Mapping[str, Any]], class_name: str, premise: str,
              checker: Checker | None, threshold: float,
              refused: list) -> list[Mapping[str, Any]]:
    """Proposals the paper is judged to support, or all of them when nothing can judge.

    An entity is scored by what it *is*, not by its name alone: "The paper fits this
    statistical model: group VBM t-tests" was scored unsupported for a paper whose methods
    say "t-tests with statistical parametric mapping (SPM5)" and "Total brain volume was
    treated as a confounding variable". The phrase was the extractor's, not the paper's, so
    judging the entity by it judged the wrong thing.

    With no checker every proposal passes, which is honest: the pass is then proposing
    without grounding and says so by writing what it was given.
    """
    if checker is None:
        return list(proposals)
    claims = [Claim(claim=describe(class_name, proposal), premise=premise)
              for proposal in proposals]
    kept = []
    for proposal, score in zip(proposals, checker.score(claims)):
        if score >= threshold:
            kept.append(proposal)
        else:
            refused.append(Refusal(
                class_name, f"the paper does not support it ({score:.2f})",
                proposal.get("name")))
    return kept


def describe(class_name: str, proposal: Mapping[str, Any], limit: int = 5) -> str:
    """The proposal as a sentence, its own field values included.

    A label alone is a thin thing to ask a checker about, and the fields are what say which
    thing is meant.
    """
    label = str(proposal.get("name") or proposal.get("definition") or "").strip()
    parts = [f"{name.replace('_', ' ')} {value}"
             for name, value in proposal.items()
             if name not in ("name", "local_id") and isinstance(value, str) and value.strip()]
    said = f"The paper describes a {class_name}: {label}."
    return said + (f" It is described as: {'; '.join(parts[:limit])}." if parts else "")


#: Sections whose prose describes what was done and what was found. An entity is judged to
#: exist against these; a paper's introduction describes other people's studies.
PREMISE_SECTIONS = ("method", "material", "result")


def _premise(text: str) -> str:
    """The methods and results, or the whole text where they cannot be found."""
    from pondie.extraction.evidence.retrieval import sectionize

    spans = [text[start:end] for start, end, label in sectionize(text)
             if any(word in label.lower() for word in PREMISE_SECTIONS)]
    joined = "\n\n".join(spans)
    return joined if len(joined) >= max(2_000, len(text) // 10) else text


def _abbreviations(text: str) -> Any:
    """The paper's own expansions, or None where the vocabulary package is unavailable."""
    try:
        from pondie.vocabularies.abbreviations import Abbreviations

        return Abbreviations.load().for_paper(text)
    except Exception:  # noqa: BLE001 -- an optional vocabulary, not a failure
        return None


#: Below this a citation is worth a second look. Nothing is removed on it -- `review_spans`
#: reports and `relocate` compares -- because the score cannot support a verdict at any
#: threshold, and this is what that looks like. Measured over 2,025 sentences the extraction
#: model chose as a warrant and 2,025 it did not, from the same papers:
#:
#:     cut   keeps true   drops false
#:    0.02       89%          34%
#:    0.05       66%          69%
#:    0.10       52%          84%
#:    0.20       42%          90%
#:    0.50       31%          96%
#:
#: Removing 90% of the bad citations costs 58% of the good ones, and keeping 90% of the good
#: ones means keeping two thirds of the bad. The distributions overlap that far -- AUC 0.75,
#: and 0.747 on the cases where the value appears nowhere in the sentence, so it is not a
#: paraphrase problem but a uniformly weak signal. Three alternatives measured worse:
#: deberta-v3-large-mnli-fever at 0.676, nli-deberta-v3-base at 0.658.
#:
#: 0.02 rather than the 0.2 first chosen from three hand-read examples, because this now
#: only decides what to *report*: at 0.02 three quarters of what it flags is genuinely
#: unsupported, against three fifths at 0.2, and it flags a quarter as many things. A
#: reviewer can read 20 doubts per paper; 83 of them, 58% of which are correct citations,
#: is a list nobody opens.
DOUBT_BELOW = 0.02

#: The old name, kept because nothing prunes on it any more and the new one says so.
PRUNE_BELOW = DOUBT_BELOW


#: A normalized number reads as unsupported against prose that states it differently:
#: "echo time seconds is 0.004" against "TE = 4 ms". Measured on one paper by the pass this
#: was ported from -- prose claims mean 0.571, numeric claims 0.114 -- so scoring them
#: together buries the signal. Reintroducing them cost 100% of `echo_time_seconds`,
#: `height_threshold_value` and `clusterwise_threshold_value` on a six-paper sample.
NUMERIC = re.compile(r"^[-+0-9.,;:\s]+$")

#: What a nested container is, said in words. Without it a claim about
#: `effect.cells[0].level` reads "level is PTSD" with nothing saying which cell, and no
#: checker can fairly judge an unanchored fragment -- 13% of claims were unanchored that way
#: before the trail was added, and 44% of `level` spans were being discarded after it was
#: dropped in the port.
CONTAINER = {
    "cells": "contrast cell", "terms": "model term", "levels": "factor level",
    "groups": "analysis group", "conditions": "task condition", "arms": "trial arm",
    "timepoints": "timepoint", "sex_distribution": "sex breakdown entry",
    "race_distribution": "race breakdown entry", "steps": "preprocessing step",
    "effect": "reported effect", "statistic": "test statistic",
    "details": "method detail", "design": "study design", "mediation": "mediation path",
}

#: The top-level containers, said in words, so a claim has a subject.
SUBJECT = {
    "analyses": "analysis", "groups": "group", "tasks": "task", "measures": "measure",
    "regions": "brain region", "acquisitions": "acquisition", "devices": "device",
    "preprocessings": "preprocessing procedure", "model_estimations": "statistical model",
    "inference_settings": "statistical threshold", "tables": "table",
    "assessments": "assessment",
}

_INDEX = re.compile(r"^([a-z_]+)(?:\[(\d+)\])?$")


def is_numeric(value: Any) -> bool:
    """A value that is only digits and separators, however the paper chose to write it.

    A list of numbers counts. `str([6, 6, 6])` is "[6, 6, 6]", whose brackets fail the
    pattern, so a smoothing kernel, a voxel size and an echo-time pair all read as prose and
    went to a checker that cannot judge a number. On 26424424 that cost the one citation
    that stated the value -- "smoothing of normalized gray matter (GM) tissue maps with a
    6 mm 3 FWHM Gaussian filter." -- replaced on score noise by "Data were preprocessed
    according to default toolbox settings: bias correction;", which does not mention
    smoothing at all.
    """
    if isinstance(value, (list, tuple)):
        return bool(value) and all(is_numeric(item) for item in value)
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return bool(NUMERIC.match(str(value).strip()))


def claim_for(record: Mapping[str, Any], path: str, value: Any) -> str:
    """The subject, where in it the leaf sits, then the assertion.

    `analyses[1].effect.cells[0].level` becomes "The analysis 'AA versus CC smokers', in the
    reported effect, in contrast cell 1, the level is African American." -- not "level is
    African American.", which names nothing and entails from nothing.
    """
    from pondie.extraction.record.edit import label_of

    parts = path.split(".")
    subject, trail, cursor = "", [], record
    for i, part in enumerate(parts[:-1]):
        matched = _INDEX.match(part)
        if not matched:
            break
        name, index = matched.group(1), matched.group(2)
        step = cursor.get(name) if isinstance(cursor, Mapping) else None
        if index is not None and isinstance(step, list) and int(index) < len(step):
            step = step[int(index)]
        if i == 0 and name in SUBJECT:
            label = label_of(step) if isinstance(step, Mapping) else None
            subject = f"The {SUBJECT[name]}" + (f" {label!r}" if label else "")
        elif name in CONTAINER:
            trail.append(f"in {CONTAINER[name]} {int(index) + 1}" if index is not None
                         else f"in the {CONTAINER[name]}")
        cursor = step
    field = parts[-1].split("[")[0].replace("_", " ")
    head = subject or "The study"
    where = f", {', '.join(trail)}," if trail else ","
    return f"{head}{where} the {field} is {value}."


#: Two to six capitals is what a paper's own short forms look like. Anything longer is a
#: word in caps, and a single capital is an initial.
ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")


def expand(text: str, abbreviations: Any, paper: str = "") -> str:
    """Write the paper's own expansion beside each acronym it defines.

    The value says "African American" and the sentence says "AA", so the checker is asked to
    entail a phrase the premise never contains and scores 0.016. Resolved per paper, because
    `AD` is axial diffusivity in a DTI paper and Alzheimer's disease in a dementia one --
    `Abbreviations.expand` already takes the paper for exactly that reason.
    """
    if abbreviations is None:
        return text
    for short in dict.fromkeys(ACRONYM.findall(text)):
        long = abbreviations.expand(short, paper)
        if long and long.lower() not in text.lower():
            text = re.sub(rf"\b{re.escape(short)}\b", f"{short} ({long})", text, count=1)
    return text


def review_spans(record: MutableMapping[str, Any], checker: Checker, refused: list,
                 abbreviations: Any = None, paper: str = "") -> list[tuple[str, float]]:
    """Score every citation and report the weak ones. Never delete one.

    The pass this was ported from asked a proposer for a better sentence and swapped only on
    a strict improvement, keeping the original when none was found -- so total support could
    only rise. The port replaced that with an unconditional delete below a threshold, which
    destroyed 46% of all spans across a six-paper sample, 36% of them sentences containing
    the value verbatim: "Experimental stimuli were controlled by computer (NeuroStim)" cited
    for `presentation_software` = NeuroStim, scored 0.021 and deleted.

    Deleting is the one thing this cannot do. A low score means the citation is worth a
    second look, not that the extractor was wrong, and nothing here knows which.
    """
    from pondie.formats.values import iter_fields

    scored: list[tuple[str, dict, Claim]] = []
    for path, node in iter_fields(record):
        slot = path.rsplit(".", 1)[-1]
        value = values.read(node)
        if not groundable(slot, node) or is_numeric(value):
            continue
        for group in (node.get("evidence") or {}).get("sets") or []:
            for span in group.get("spans") or []:
                if span.get("text"):
                    scored.append((path, span, Claim(
                        claim=claim_for(record, path, value),
                        premise=expand(span["text"], abbreviations, paper))))
    if not scored:
        return []
    weak = []
    for (path, span, _claim), score in zip(scored, checker.score([c for *_x, c in scored])):
        if score >= DOUBT_BELOW:
            continue
        weak.append((path, float(score)))
        refused.append(Refusal(
            "evidence", f"the span may not support the value ({score:.2f}); left in place",
            span.get("text", "")[:80]))
    return weak
