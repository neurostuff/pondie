"""`Acquisition.modality` -> the Modality value the source's wording names.

The field was a closed enum and is now open, so a paper writing "functional magnetic
resonance imaging" keeps its words instead of costing the whole acquisition. This maps them
back on, and the mapping is load-bearing rather than cosmetic: only a vocabulary value
carries an `instantiates`, so `Acquisition.acquisition_type` cannot be derived from anything
else, and without the designator the record resolves to the base class where every
modality-specific parameter is undeclared. `rules.check_acquisition_subclass` reports each
one that lands there.

The targets are the schema's own permissible values rather than new names, because the
question here is "which vocabulary value is this" -- unlike `coordinate_space`, whose MNI and
TAL are a downstream answer the record does not store.

No `OTHER`. The vocabulary already has `other` for a modality outside the named ones, and a
second value meaning the same thing would be two spellings of one answer. `UNKNOWN` keeps its
usual sense: the wording does not say.

The rules are seeded from the vocabulary's own synonyms, not from measured drift -- at the
time of writing no committed record holds an off-vocabulary modality, because the field was
closed and `values.cast` refused them before they could be counted. `report()` is how the
real surface forms arrive: an input no rule matches is UNKNOWN with `reason="unmatched"` and
is listed, so the first paper that writes something new forces a rule rather than vanishing.
"""

from __future__ import annotations

from pondie.normalization import UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

FMRI, SMRI, DMRI, MRI = "fMRI", "sMRI", "dMRI", "MRI"
EEG, FNIRS, PET, MEG, SPECT, OTHER_MODALITY = "EEG", "fNIRS", "PET", "MEG", "SPECT", "other"
VALUES = (FMRI, SMRI, DMRI, MRI, EEG, FNIRS, PET, MEG, SPECT, OTHER_MODALITY, UNKNOWN)

#: The three qualified MRI values are tested before the bare one, and `MRI` carries a
#: lookbehind for each qualifier. Without it "functional MRI" matches both `fMRI` and `MRI`,
#: which `classify` reads as an ambiguity and answers UNKNOWN -- the commonest wording in the
#: corpus would have been the one value this cannot resolve. The acronyms need no lookbehind:
#: "fMRI" is one token, so `\bmri\b` does not reach the MRI inside it.
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
    #: Only the bare word. "other imaging" is a description, and reading it as the
    #: vocabulary's `other` asserts the paper placed itself outside the named modalities.
    Rule.of(OTHER_MODALITY, r"^\s*other\s*$"),
)

FIELD = ClosedField("acquisitions.modality", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report


if __name__ == "__main__":
    print(FIELD.report())
