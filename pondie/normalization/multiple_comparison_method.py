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
    Rule.of(
        PERMUTATION,
        r"permut|randomi[sz]|monte[\s-]?carlo|bootstrap|\bTFCE\b|" r"threshold[\s-]free",
    ),
    Rule.of(FDR, r"\bFDR\b|false[\s-]discovery"),
    Rule.of(
        FWE,
        r"\bFWE\b|family[\s-]?wise|\bbonferroni\b|\bholm\b|\bsidak\b|"
        r"gaussian[\s-]random[\s-]field|\bGRF\b|\bAlphaSim\b|small[\s-]volume|\bSVC\b",
    ),
    Rule.of(
        UNCORRECTED, r"\b(un|non)[\s-]?corrected\b|\bno correction\b|^\s*none\b|" r"\buncorr\b"
    ),
    Rule.of(
        OTHER,
        r"\bcluster[\s-]?(?:level|extent|size|based|correct)|\bROI[\s-]?based\b|"
        r"\bcluster correction\b",
    ),
)

FIELD = ClosedField("inference_settings.multiple_comparison_method", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report
