"""`InferenceSettings.correction_scope` -> the volume the correction covered.

WHOLE_BRAIN against RESTRICTED is the distinction a meta-analysis needs; `cluster level`
and `searchlight` name a unit rather than a volume and are OTHER.

Why, with the measurements: docs/normalization-rationale.md, "correction_scope".
"""

from __future__ import annotations

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

WHOLE_BRAIN, RESTRICTED = "WHOLE_BRAIN", "RESTRICTED"
VALUES = (WHOLE_BRAIN, RESTRICTED, OTHER, UNKNOWN)

RULES = (
    #: Tested first; see the doc named above.
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
report = FIELD.report
