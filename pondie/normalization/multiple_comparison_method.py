"""`InferenceSettings.multiple_comparison_method` -> the family of correction used.

293 surface forms over 921 values in this corpus, for four answers. "Corrected results only"
is a standard meta-analysis inclusion criterion, and it cannot be applied against
`FWE`, `family-wise error (FWE)`, `Family-Wise Error` and `FWE correction` as four values.

UNCORRECTED is not UNKNOWN: a paper stating it did not correct has told us something, and a
paper silent on the matter has not. Only the first is safely excludable.
"""

from __future__ import annotations

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

FWE, FDR, PERMUTATION, UNCORRECTED = "FWE", "FDR", "PERMUTATION", "UNCORRECTED"
VALUES = (FWE, FDR, PERMUTATION, UNCORRECTED, OTHER, UNKNOWN)

#: Order matters. `permutation` is tested before `FWE` because a permutation-derived
#: family-wise threshold is usually written as both and the resampling is the specific claim.
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
    #: Cluster-level thresholding names a unit, not a family: it says where the correction
    #: was applied and not which error rate it controls, so it is outside the four.
    Rule.of(
        OTHER,
        r"\bcluster[\s-]?(?:level|extent|size|based|correct)|\bROI[\s-]?based\b|"
        r"\bcluster correction\b",
    ),
)

FIELD = ClosedField("inference_settings.multiple_comparison_method", RULES, VALUES)
normalize = FIELD.normalize
#: The residual, for `pondie normalize <field>`. `__init__` states the contract --
#: "Every module exposes `normalize(...)` ... and `report(...)`" -- and five of the eight
#: closed-target modules bound only the first, so the CLI verb raised for them.
report = FIELD.report


if __name__ == "__main__":
    print(FIELD.report())
