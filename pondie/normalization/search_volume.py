"""`InferenceSettings.search_volume` -> how much of the brain the search covered.

The field `Analysis.spatial_scope` cannot answer. That enum offers `whole_brain`, `roi` and
`searchlight`, and over 1,817 records the models pick one of the first two 5,841 times out of
6,032 -- so an analysis masked to a tissue compartment is filed as whole-brain and the mask
survives only as the free text this module reads.

COMPARTMENT is the value `spatial_scope` is missing, and the reason it is measured here
first: a grey-matter-masked VBM search is not the whole brain and is not a region of
interest. Whether it should become a fourth `SpatialScope` value depends on whether
separating it changes a published criterion's verdict, which a free-text field cannot test
and a normalized one can.

PARTIAL is kept apart because it is not a choice at all -- it is what the scanner sampled,
which `Acquisition.coverage` already owns as "a third axis distinct from the two the schema
already carries". A value landing here is a routing error, not a scope.

A size is not a space. `64 x 64 x 30 voxels`, `600 cm3` and `6,146 uL` say how big the
volume was and nothing about which volume it was, so no rule claims them: they stay in the
residual, where they report the slot being used for something it does not mean.

Why, with the measurements: docs/normalization-rationale.md, "search_volume".
"""

from __future__ import annotations

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

WHOLE_BRAIN, COMPARTMENT, REGIONS, PARTIAL = (
    "WHOLE_BRAIN",
    "COMPARTMENT",
    "REGIONS",
    "PARTIAL",
)
VALUES = (WHOLE_BRAIN, COMPARTMENT, REGIONS, PARTIAL, OTHER, UNKNOWN)

#: Order matters, and the first three are `decisive`; see the doc named above.
RULES = (
    # First, and decisive: a volume the scanner never sampled is not a scope the authors
    # chose. `Twenty axial slices covering the whole brain` is whole-brain, so this asks
    # for a statement of INCOMPLETENESS rather than for the word "slices".
    Rule.of(
        PARTIAL,
        r"not fully covered|did not cover|partial[\s-]?(?:brain|volume|coverage)|"
        r"anterior third|\d+ of \d+\s+\w*\s*slices|limited (?:coverage|field)",
        decisive=True,
    ),
    # Decisive over COMPARTMENT because a named structure is the more specific claim --
    # the same reason `gestational_weeks` beats `weeks`. `Nine gray matter regions of
    # interest defined with WFU PickAtlas` is a region list that happens to say which
    # tissue, not a tissue compartment.
    #
    # The cost, stated because it is real: one `InferenceSettings` can serve several
    # analyses, so a paper reporting a whole-brain search AND a small-volume correction
    # writes both into one value and this reads it as REGIONS. That undercounts
    # COMPARTMENT and WHOLE_BRAIN rather than overcounting them, and it is an argument
    # for the field being per-analysis.
    Rule.of(
        REGIONS,
        r"\bROIs?\b|regions?[\s-]of[\s-]interest|\bVOIs?\b|small[\s-]volume|\ba priori\b|"
        r"amygdala|hippocamp|striat|insula|prefrontal|\bACC\b|cingulate|\bOFC\b|"
        r"nucleus accumbens|thalam|\bVTA\b|cerebellum mask|\bspheres?\b|\bseeds?\b|"
        r"\blobar\b|\blobes?\b|\bnode\b|\d+[\s-]node",
        decisive=True,
    ),
    # The value this module exists to count: a tissue class or a cortical sheet, which is
    # most of the brain and none of it a region of interest.
    Rule.of(
        COMPARTMENT,
        r"gr[ae]y[\s-]?matter|white[\s-]?matter|skeleton|cerebrum and cerebellum|"
        r"cortical (?:ribbon|surface|volume)|cortical surface vertices|"
        r"entire cortical|supratentorial|\btissue\b",
        decisive=True,
    ),
    Rule.of(
        WHOLE_BRAIN,
        r"whole[\s-]?brain|entire brain|full brain|across the brain|brain[\s-]?wide|"
        r"entire volume|all brain voxels|^\s*brain\s*$",
    ),
    Rule.of(OTHER, r"\bsearchlight\b|\bvertex[\s-]?wise\b"),
)

FIELD = ClosedField("inference_settings.search_volume", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report
