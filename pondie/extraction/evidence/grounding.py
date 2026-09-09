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






#: Two to six capitals is what a paper's own short forms look like. Anything longer is a
#: word in caps, and a single capital is an initial.
ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")




