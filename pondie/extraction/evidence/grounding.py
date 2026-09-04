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

    The batch is small by measurement, not by default: sixteen claims against a
    methods-and-results premise exhausts an 8 GB card, and MiniCheck chunks the document per
    claim, so the batch is the only thing that can shrink.

    It takes no device, and deliberately does not set one. MiniCheck loads with
    `device_map="auto"` and reads the visible devices, so the only way to place it is
    `CUDA_VISIBLE_DEVICES` -- which is process-wide. Setting it here restricted the process
    to one card and the proposer, asked for the second, got "invalid device ordinal".
    Visibility is the caller's to set, once, before either model loads.
    """

    def __init__(self, model_name: str = "flan-t5-large", batch_size: int = 4,
                 cache_dir: str | None = None) -> None:
        from minicheck.minicheck import MiniCheck as _MiniCheck

        self._batch = batch_size
        self._model = _MiniCheck(model_name=model_name, enable_prefix_caching=False,
                                 batch_size=batch_size, cache_dir=cache_dir)

    def score(self, claims: Sequence[Claim]) -> Sequence[float]:
        """Scored in slices, freeing between them, so one long premise cannot end a run."""
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


#: Addresses, not claims. A local_id is how the record refers to something internally; the
#: paper never says "reg_hippocampus", so a checker asked whether it does will always say no
#: and a pass that prunes on that answer deletes every citation on an identifier.
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


def drop_unsupported_spans(record: MutableMapping[str, Any], checker: Checker,
                           refused: list) -> None:
    """Drop a span that does not support what it is cited for.

    Only the span. The value stays and its evidence becomes `not_found`, which is the honest
    state: something was extracted and no sentence has been found for it. Removing the value
    too would be deleting a reading on the strength of a citation being wrong about it.
    """
    from pondie.formats.values import iter_fields

    scored: list[tuple[dict, dict, Claim]] = []
    for path, node in iter_fields(record):
        slot = path.rsplit(".", 1)[-1]
        if not groundable(slot, node):
            continue
        for group in (node.get("evidence") or {}).get("sets") or []:
            for span in group.get("spans") or []:
                if span.get("text"):
                    scored.append((node, span, Claim(
                        claim=f"{slot.replace('_', ' ')} is {values.read(node)}.",
                        premise=span["text"])))
    if not scored:
        return
    for (node, span, _claim), score in zip(scored, checker.score([c for *_x, c in scored])):
        if score >= PRUNE_BELOW:
            continue
        evidence = node.get("evidence") or {}
        for group in evidence.get("sets") or []:
            group["spans"] = [s for s in group.get("spans") or [] if s is not span]
        evidence["sets"] = [g for g in evidence.get("sets") or [] if g.get("spans")]
        if not evidence["sets"]:
            node["evidence"] = {"status": "not_found"}
        refused.append(Refusal(
            "evidence", f"the span does not support the value ({score:.2f})",
            span.get("text", "")[:80]))
