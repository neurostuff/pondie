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


class MiniCheck:
    """`bespokelabs/minicheck` behind the protocol.

    Two departures from the package's own entry point, both measured.

    It scores through `Inferencer.inference`, not `MiniCheck.score`. The latter routes to
    `inference_example_batch`, which loops one (premise, claim) pair at a time -- its
    `batch_size` batches *chunks within* a pair, so with a one-sentence premise every pair is
    a separate forward pass of a 770M encoder-decoder and the batch size does nothing.
    `inference` chunks the list of pairs instead and returns one probability per pair, in
    order. Safe here because `batch_tokenize` truncates at `max_model_len` rather than
    chunking, and an evidence span is a sentence; a premise long enough to truncate would
    lose its tail, which is why `LONG` falls back to the per-example path.

    It loads in bfloat16, never float16. The checkpoint is float32 and MiniCheck passes no
    dtype. T5 was trained in bfloat16 and overflows float16's range in `T5DenseReluDense`,
    which is a decade-old class of NaN reports; Ampere has native bf16, so this halves both
    the footprint and the work with no such risk.
    """

    #: Characters beyond which a premise may truncate at `max_model_len` and needs the
    #: package's own chunk-and-max path. A sentence is far below this.
    LONG = 1_200

    def __init__(self, model_name: str = "flan-t5-large", batch_size: int = 64,
                 cache_dir: str | None = None) -> None:
        import torch
        from minicheck.minicheck import MiniCheck as _MiniCheck

        self._batch = batch_size
        self._model = _MiniCheck(model_name=model_name, enable_prefix_caching=False,
                                 batch_size=batch_size, cache_dir=cache_dir)
        inner = getattr(self._model, "model", None)
        weights = getattr(inner, "model", None)
        if weights is not None and torch.cuda.is_available():
            inner.model = weights.to(torch.bfloat16)
            # The package does `label_probs[:, 1].cpu().numpy()`, and numpy has no bfloat16.
            # Upcasting at the head rather than after softmax keeps the whole encoder in
            # bf16 and hands the library the float32 it expects.
            inner.model.lm_head.register_forward_hook(
                lambda _module, _inputs, output: output.float())

    def score(self, claims: Sequence[Claim]) -> Sequence[float]:
        if not claims:
            return []
        inner = getattr(self._model, "model", None)
        if inner is None or any(len(c.premise) > self.LONG for c in claims):
            return self._per_example(claims)
        out = inner.inference([c.premise for c in claims], [c.claim for c in claims])
        return [float(x) for x in out["support_prob_per_chunk"]]

    def _per_example(self, claims: Sequence[Claim]) -> Sequence[float]:
        """The package's own path, for premises long enough that truncation would bite."""
        import torch

        out: list[float] = []
        for start in range(0, len(claims), self._batch):
            batch = claims[start:start + self._batch]
            _, probabilities, _, _ = self._model.score(
                docs=[c.premise for c in batch], claims=[c.claim for c in batch]
            )
            out.extend(probabilities)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return out


#: A stopgap, and named as one. The durable answer is `value_source: generated` -- the
#: schema's own vocabulary for a value the pipeline reasoned to rather than read -- and where
#: a pass stamps that correctly, `groundable` already exempts the field for free and this
#: list is unnecessary. Until every writer does, these are matched by bare slot name, which
#: catches `Measure.type` and `ModelEstimation.stage` alongside the ones meant. A LinkML
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


#: Below this a span is not weak evidence, it is evidence for something else. Set low on
#: purpose: the checker rejected an acknowledgements sentence offered as a warrant at 0.041
#: and a sentence naming the wrong model term at 0.025, while accepting real ones at 0.92 and
#: 0.95. Anything between is a judgement call, and a repair pass should not be making those
#: -- it should be removing the citations that are plainly about something else.
PRUNE_BELOW = 0.2


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
    """A value that is only digits and separators, however the paper chose to write it."""
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
        if score >= PRUNE_BELOW:
            continue
        weak.append((path, float(score)))
        refused.append(Refusal(
            "evidence", f"the span may not support the value ({score:.2f}); left in place",
            span.get("text", "")[:80]))
    return weak
