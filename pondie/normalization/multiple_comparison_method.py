"""`InferenceSettings.multiple_comparison_method` -> the family of correction used.

UNCORRECTED is not UNKNOWN: a paper stating it did not correct has told us something.

Why, with the measurements: docs/normalization-rationale.md, "multiple_comparison_method".
"""

from __future__ import annotations

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

FWE, FDR, PERMUTATION, UNCORRECTED = "FWE", "FDR", "PERMUTATION", "UNCORRECTED"
VALUES = (FWE, FDR, PERMUTATION, UNCORRECTED, OTHER, UNKNOWN)

#: Order matters; see the doc named above.
RULES = (
    # Decisive, and first, for the same reason `gestational_weeks` and `fMRI` are: the
    # specific claim beats the generic one. AFNI's AlphaSim and its successor 3dClustSim
    # ARE Monte Carlo simulations, so every paper that names both -- 40 values here,
    # "AlphaSim Monte Carlo simulation" and six other orderings -- matched the PERMUTATION
    # rule and the FWE rule at once and `classify` read one tool as an ambiguity between
    # two answers. Naming the implementation settles which of the two the paper meant.
    Rule.of(FWE, r"\bAlphaSim\b|\b3?d?ClustSim\b", decisive=True),
    # Decisive for the reason the section below already gave for putting it first, which
    # first alone could not deliver: `classify` consults rule ORDER only when one of the
    # matches is OTHER, so "family-wise error correction using permutations" matched two
    # answers and was read as an ambiguity -- 28 values, the whole of what was left after
    # the rule above. A family-wise threshold derived by resampling is written as both
    # names and the resampling is the specific claim.
    Rule.of(
        PERMUTATION,
        r"permut|randomi[sz]|monte[\s-]?carlo|bootstrap|\bTFCE\b|threshold[\s-]free",
        decisive=True,
    ),
    Rule.of(FDR, r"\bFDR\b|false[\s-]discovery"),
    Rule.of(
        FWE,
        # `random field` without `gaussian` is 9 values in the corpus and the same
        # theory; `3dClustSim` is AFNI's successor to `AlphaSim`, 7 more, and is placed
        # beside it rather than under PERMUTATION so the two spellings of one tool agree.
        r"\bFWE\b|family[\s-]?wise|\bbonferroni\b|\bholm\b|\bsidak\b|"
        r"random[\s-](?:gaussian[\s-])?field|gaussian[\s-]random|\bGRF\b|\bFWER\b|"
        r"small[\s-]volume|\bSVC\b",
    ),
    Rule.of(
        UNCORRECTED, r"\b(un|non)[\s-]?corrected\b|\bno correction\b|^\s*none\b|" r"\buncorr\b"
    ),
    Rule.of(
        OTHER,
        # `clusterwise correction` and `cluster thresholding` are 8 values the earlier
        # alternation missed by one word each.
        r"\bcluster[\s-]?(?:level|extent|size|based|correct|wise|thresh|filter)|"
        r"\bROI[\s-]?based\b|\bcluster correction\b|\bjoint probability\b",
    ),
)

FIELD = ClosedField("inference_settings.multiple_comparison_method", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report
