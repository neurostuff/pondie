"""`InferenceSettings.correction_scope` -> the volume the correction was applied over.

217 surface forms over 760 values. The distinction a meta-analysis needs is whole-brain
against a restricted volume: a small-volume-corrected result survived a much lower bar than a
whole-brain one, and pooling them treats the two as equal evidence.

`cluster level` is not an answer to this question -- it names the unit a threshold applied to,
not the volume searched -- so it is OTHER rather than being forced onto the scale.
"""
from __future__ import annotations

from . import OTHER, UNKNOWN
from ._lexicon import ClosedField, Rule

WHOLE_BRAIN, RESTRICTED = "WHOLE_BRAIN", "RESTRICTED"
VALUES = (WHOLE_BRAIN, RESTRICTED, OTHER, UNKNOWN)

RULES = (
    #: Tested first: "whole brain and a priori ROIs" restricts somewhere, and the restricted
    #: half is the one that changes how the result should be weighed.
    Rule.of(RESTRICTED, r"\bROI\b|region[s]?[\s-]of[\s-]interest|small[\s-]volume|\bSVC\b|"
                        r"a priori|\bmask(?:ed|s)?\b|search volume|anatomically[\s-]defined|"
                        r"\bsphere\b|\bseed\b|volume[s]? of interest|\bVOI\b"),
    Rule.of(WHOLE_BRAIN, r"whole[\s-]?brain|entire brain|\bglobal\b|across the brain|"
                         r"grey matter mask|gray matter mask|whole[\s-]?volume|"
                         r"brain[\s-]?wise"),
    Rule.of(OTHER, r"\bcluster[\s-]?(?:level|wise|extent)\b|\bvoxel[\s-]?(?:level|wise)\b|"
                   r"\bvertex\b"),
)

FIELD = ClosedField("inference_settings.correction_scope", RULES, VALUES)
normalize = FIELD.normalize


if __name__ == "__main__":
    print(FIELD.report())
