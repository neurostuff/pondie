"""`InferenceSettings.correction_scope` -> the volume the correction was applied over.

217 surface forms over 760 values. The distinction a meta-analysis needs is whole-brain
against a restricted volume: a small-volume-corrected result survived a much lower bar than a
whole-brain one, and pooling them treats the two as equal evidence.

`cluster level` is not an answer to this question -- it names the unit a threshold applied to,
not the volume searched -- so it is OTHER rather than being forced onto the scale. `searchlight`
is the same kind of answer and lands in the same place.

Every separator class here admits an underscore as well as a space and a hyphen, because the
value this field holds most often is the schema's own `whole_brain`. Measured over 1,817
records it is 212 of 433 values, and a class of space-or-hyphen matched none of them: the
field answered 51% of what it saw, and every miss was the permissible value spelled exactly
as the enum spells it.
"""

from __future__ import annotations

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

WHOLE_BRAIN, RESTRICTED = "WHOLE_BRAIN", "RESTRICTED"
VALUES = (WHOLE_BRAIN, RESTRICTED, OTHER, UNKNOWN)

RULES = (
    #: Tested first: "whole brain and a priori ROIs" restricts somewhere, and the restricted
    #: half is the one that changes how the result should be weighed.
    Rule.of(
        RESTRICTED,
        r"\bROI\b|region[s]?[\s_-]of[\s_-]interest|small[\s_-]volume|\bSVC\b|"
        r"a priori|\bmask(?:ed|s)?\b|search volume|anatomically[\s_-]defined|"
        r"\bsphere\b|\bseed\b|volume[s]? of interest|\bVOI\b",
    ),
    Rule.of(
        WHOLE_BRAIN,
        r"whole[\s_-]?brain|entire brain|\bglobal\b|across the brain|"
        r"grey matter mask|gray matter mask|whole[\s_-]?volume|"
        r"brain[\s_-]?wise",
    ),
    Rule.of(
        OTHER,
        r"\bcluster[\s_-]?(?:level|wise|extent)\b|\bvoxel[\s_-]?(?:level|wise)\b|"
        r"\bvertex\b|\bsearchlight\b",
    ),
)

FIELD = ClosedField("inference_settings.correction_scope", RULES, VALUES)
normalize = FIELD.normalize
#: The residual, for `pondie normalize <field>`. `__init__` states the contract --
#: "Every module exposes `normalize(...)` ... and `report(...)`" -- and five of the eight
#: closed-target modules bound only the first, so the CLI verb raised for them.
report = FIELD.report


if __name__ == "__main__":
    print(FIELD.report())
