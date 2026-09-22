"""`Acquisition.modality` -> the Modality value the source's wording names.

The targets are the schema's own permissible values. No OTHER: the vocabulary already
has `other`.

Why, with the measurements: docs/normalization-rationale.md, "modality".
"""

from __future__ import annotations

from pondie.normalization import UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

FMRI, SMRI, DMRI, MRI = "fMRI", "sMRI", "dMRI", "MRI"
EEG, FNIRS, PET, MEG, SPECT, OTHER_MODALITY = "EEG", "fNIRS", "PET", "MEG", "SPECT", "other"
VALUES = (FMRI, SMRI, DMRI, MRI, EEG, FNIRS, PET, MEG, SPECT, OTHER_MODALITY, UNKNOWN)

#: Order matters; see the doc named above.
RULES = (
    Rule.of(FMRI, r"\bfmri\b|functional (?:mri|magnetic resonance|imaging)|\bbold\b"),
    Rule.of(
        SMRI,
        r"\bsmri\b|structural (?:mri|magnetic resonance|imaging|scan)|"
        r"anatomical (?:mri|scan|imaging)|\bt1[\s-]?weighted\b|"
        r"voxel[\s-]?based morphometry|\bvbm\b|cortical thickness",
    ),
    Rule.of(DMRI, r"\bdmri\b|\bdti\b|\bdwi\b|diffusion[\s-](?:mri|tensor|weighted)|tractograph"),
    Rule.of(
        MRI,
        r"(?<!functional )(?<!structural )(?<!anatomical )(?<!diffusion )\bmri\b|"
        r"(?<!functional )(?<!structural )\bmagnetic resonance\b",
    ),
    Rule.of(EEG, r"\beeg\b|electroencephalograph"),
    Rule.of(MEG, r"\bmeg\b|magnetoencephalograph"),
    Rule.of(FNIRS, r"\bf?nirs\b|near[\s-]?infrared"),
    Rule.of(PET, r"\bpet\b|positron emission"),
    Rule.of(SPECT, r"\bspect\b|single[\s-]photon emission"),
    Rule.of(OTHER_MODALITY, r"^\s*other\s*$"),
)

FIELD = ClosedField("acquisitions.modality", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report
