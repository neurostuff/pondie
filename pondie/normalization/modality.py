"""`Acquisition.modality` -> the Modality value the source's wording names.

The targets are study_schema's own `Modality` values, read from it rather than retyped, and
their parents (`is_a`) come from it too. No OTHER: the vocabulary already has `other`.

- A schema value maps to itself.
- `other` (or the class name `OtherModality`) is read through the acquisition's
  `modality_label`; a label no rule reaches leaves it `other`.
- A wording that names no acquisition (a database, sequencing, histology) is unmapped, one
  that names two modalities is ambiguous, and one no rule reaches is unmatched.
- A bare `MRI` takes fMRI, sMRI, dMRI or MRS from the measures of the analyses that use it,
  when they agree (`refine_mri`).

`annotate` is the writer of the `Modality` entry on `acquisitions[].modality`.

Why, with the measurements: docs/normalization-rationale.md, "modality".
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable

from study_schema.jsonschema import value_parents
from study_schema.models.storage import MeasureType, Modality

from pondie.normalization import UNKNOWN, _sidecar
from pondie.normalization._lexicon import ClosedField, Decision, Rule, field_report
from pondie.normalization._records import NOT_REPORTED, value_of

M = Modality
FMRI, SMRI, DMRI, MRS, MRI = M.fMRI.value, M.sMRI.value, M.dMRI.value, M.MRS.value, M.MRI.value
EEG, IEEG, MEG, FNIRS = M.EEG.value, M.iEEG.value, M.MEG.value, M.fNIRS.value
PET, SPECT, CT, PHYSIOLOGICAL = M.PET.value, M.SPECT.value, M.CT.value, M.physiological.value
OTHER_MODALITY = M.other.value
VALUES = (*(m.value for m in Modality), UNKNOWN)

#: Each refined value's coarser one: fMRI -> MRI, iEEG -> other, ...
PARENT: dict[str, str] = value_parents("storage-study", "Modality")

PATH = "acquisitions.modality"
VOCABULARY = "Modality"

#: The schema's values, by lower case, so `fmri` is fMRI. They map to themselves.
SCHEMA = {m.value.lower(): m.value for m in Modality}
#: What the extractor writes when the wording is in `modality_label`.
UNNAMED = {"other", "othermodality"}

#: First match wins unless several match; see `OUTRANKS`.
RULES = (
    Rule.of(
        FMRI,
        r"\bfmri\b|functional (?:mri|magnetic resonance|imaging)|\bbold\b|arterial spin|\basl\b",
    ),
    Rule.of(
        SMRI,
        r"\bsmri\b|structural (?:mri|magnetic resonance|imaging|scan)|"
        r"anatomical (?:mri|scan|imaging)|\bt1[\s-]?weighted\b|"
        r"voxel[\s-]?based morphometry|\bvbm\b|cortical thickness",
    ),
    Rule.of(DMRI, r"\bdmri\b|\bdti\b|\bdwi\b|diffusion[\s-](?:mri|tensor|weighted)|tractograph"),
    Rule.of(MRS, r"\bf?mrs\b|\b1h[\s-]?mrs|\b31p\b|(?:magnetic resonance|mr) spectroscop"),
    Rule.of(MRI, r"\bmri\b|magnetic resonance"),
    Rule.of(
        IEEG,
        r"\becogs?\b|electrocortico|\bseegs?\b|stereo-?electroenceph|"
        r"stereotactic (?:eeg|electroenceph)|\bieeg\b|intracranial|intra-?cerebral|"
        r"depth electrode|subdural|\blfps?\b|local field potential|single[\s-]unit|multi-?unit|"
        r"micro-?(?:wire|electrode)|extracellular|\bifps?\b",
    ),
    Rule.of(EEG, r"\beeg\b|electroenceph|(?<!motor )(?<!motor-)evoked potential|\berps?\b"),
    Rule.of(MEG, r"\bmeg\b|magneto-?enceph"),
    Rule.of(
        FNIRS,
        r"\bf?nirs\b|near[\s-]?infrared spectroscop|diffuse optical|\bhd-dot\b|\bdot\b|"
        r"event-related optical|\beros\b",
    ),
    Rule.of(PET, r"\bpet\b|positron emission"),
    Rule.of(SPECT, r"\bspect\b|single[\s-]photon emission"),
    Rule.of(
        CT,
        r"(?<![a-z])(?:micro-?)?ct(?![a-z])|\bc?cbct\b|\bcct\b|(?<!emission )computed tomograph|"
        r"computeri[sz]ed (?:axial )?tomograph|x-ray (?:micro)?tomograph|microtomograph",
    ),
    Rule.of(
        PHYSIOLOGICAL,
        r"\bemgs?\b|electromyo|\becgs?\b|\bekg\b|electrocardio|"
        r"eye[\s-]?(?:track|movement|position)|eyetrack|\beyelink|pupil|\bgaze\b|oculo|"
        r"\bocular|\beogs?\b|saccad|skin conduct|electroderm|\bscrs?\b|\beda\b|\bgsr\b|"
        r"plethysmo|\bppgs?\b|\bpulse\b|respirat|breath|"
        r"heart|cardi|\begg\b|electrogastro|electrointestin|accelero|\bimu\b|kinemat|"
        r"motion (?:capture|track)|body tracking|sympathetic nerve|blood pressure|polysomno|"
        r"nasal pressure|tumescence|motor[\s-]evoked|\bmeps?\b",
    ),
)

#: A value matched beside one it refines is the one meant: each value over its schema parent
#: ("functional MRI" is fMRI), iEEG over EEG ("intracranial EEG"), and PET-CT is PET.
OUTRANKS: dict[str, set[str]] = {child: {parent} for child, parent in PARENT.items()}
OUTRANKS[IEEG].add(EEG)
OUTRANKS.update({PET: {CT}, SPECT: {CT}})

#: Names something that is not an acquisition; it settles the reading over every rule above.
NOT_ACQUISITION = re.compile(
    r"meta[\s-]?analy|brainmap|neurosynth|\bale\b|database|coordinate|literature|atlas|simulat|"
    r"sequenc|genom|exome|rna-?seq|\b[dr]na\b|methylation|genotyp|genechip|microarray|chip-chip|"
    r"histolog|immuno|microscop|confocal|autoradiograph|tracer stud|\btracing\b|docking|stain|"
    r"in situ hybrid|mass spectrometr|\bgc-ms\b|questionnaire|rainfall|meteorolog|water sample|"
    r"lidar|seismic|seafloor|geographic|density functional|dipole moment|particle trajector",
    re.I,
)

#: "MRI-compatible EMG" is EMG: the scanner it was made to work in is not what it records.
COMPATIBLE = re.compile(
    r"\b(?:f?mri|mr|magnetic resonance)[\s-](?:compatible|conditional)\b", re.I
)

FIELD = ClosedField(PATH, RULES, VALUES, outranks=OUTRANKS)


def normalize(text: object) -> Decision:
    """One wording -> its Modality value, and why."""
    raw = text if isinstance(text, str) else ""
    named = SCHEMA.get(raw.strip().lower())
    if named:
        return Decision(named, "schema value", raw)
    found = NOT_ACQUISITION.search(raw)
    if found:
        return Decision(UNKNOWN, f"not an acquisition: {found.group().lower()}", raw)
    return dataclasses.replace(FIELD.normalize(COMPATIBLE.sub(" ", raw)), text=raw)


def report(patterns: tuple[str, ...] | None = None) -> str:
    return field_report(PATH, normalize, VALUES, patterns)


def _stated(slot: object) -> str | None:
    value = value_of(slot)
    return value if isinstance(value, str) and value.strip() else None


def read_acquisition(acquisition: dict) -> tuple[Decision, tuple[str, ...]] | None:
    """One acquisition's modality, read through its label where it has one: (decision, the
    other slots read). None when the modality is unstated."""
    value = _stated(acquisition.get("modality"))
    if value is None:
        return None
    label = _stated(acquisition.get("modality_label"))
    unnamed = value.strip().lower() in UNNAMED
    if unnamed and label is None:
        return Decision(OTHER_MODALITY, "schema value", value), ()
    decision = normalize(value) if not unnamed else None
    if decision is not None and decision.reason != "unmatched":
        return decision, ()
    if label is not None:
        by_label = normalize(label)
        if by_label.reason != "unmatched":
            return by_label, ("modality_label",)
    if unnamed:
        return Decision(OTHER_MODALITY, "schema value; no rule names its label", value), ()
    return decision, ()


#: A measure type's MRI modality. A MeasureType value missing here decides nothing.
MEASURE = {
    **dict.fromkeys(
        (
            "bold_response", "bold_derived_metric", "cerebral_blood_flow",
            "cerebral_blood_volume", "connectivity_strength", "decoding_performance",
            "representational_similarity",
        ),
        FMRI,
    ),
    **dict.fromkeys(
        (
            "gray_matter_volume", "gray_matter_density", "white_matter_volume",
            "cortical_thickness", "surface_area", "structural_morphometry_other",
        ),
        SMRI,
    ),
    "diffusion_metric": DMRI,
}
MEASURE_TYPES = frozenset(m.value for m in MeasureType)
#: The same for a free-text measure type.
MEASURE_TEXT = tuple(
    (value, re.compile(pattern, re.I))
    for value, pattern in (
        (MRS, r"metabolite|spectroscop|\bgaba\b|glutam|\bnaa\b"),
        (DMRI, r"diffusion|anisotrop|tractograph"),
        (SMRI, r"susceptib|relax|myelin|magneti[sz]ation transfer|lesion|atroph|morpholog"),
        (FMRI, r"perfusion|blood flow|\bcbf\b|\bbold\b|\balff\b|\breho\b"),
    )
)


def measure_class(measure_type: object) -> str | None:
    """The MRI modality a measure type implies, or None."""
    if not isinstance(measure_type, str):
        return None
    if measure_type in MEASURE_TYPES:
        return MEASURE.get(measure_type)
    return next((value for value, rx in MEASURE_TEXT if rx.search(measure_type)), None)


def refine_mri(measure_types: Iterable[str], text: str = MRI) -> Decision:
    """A bare MRI acquisition's modality from its analyses' measure types."""
    types = sorted({t for t in measure_types if isinstance(t, str)})
    classes = {measure_class(t) for t in types} - {None}
    if len(classes) == 1:
        return Decision(classes.pop(), "measure: " + ", ".join(types), text)
    if classes:
        return Decision(MRI, "measures disagree: " + ", ".join(sorted(classes)), text)
    return Decision(MRI, "no measure decides", text)


def _ids(slot: object) -> list[str]:
    value = value_of(slot, True)
    return [str(x) for x in value if isinstance(x, str)] if isinstance(value, list) else []


def read(record: dict) -> dict[str, tuple[Decision, tuple[str, ...]]]:
    """Each acquisition's reading, by local id, with a bare MRI refined by `refine_mri`.

    An analysis whose other linked acquisition already has the measure's modality is not
    evidence: the extractor links every acquisition of a session to each analysis, so a gray
    matter volume beside an sMRI acquisition says nothing about the bare MRI next to it.
    """
    acquisitions = [a for a in record.get("acquisitions") or [] if isinstance(a, dict)]
    base = {str(value_of(a.get("local_id"))): read_acquisition(a) for a in acquisitions}
    base = {aid: r for aid, r in base.items() if r is not None}
    measures = {
        str(value_of(m.get("local_id"))): _stated(m.get("type"))
        for m in record.get("measures") or []
        if isinstance(m, dict)
    }
    evidence: dict[str, list[str]] = {aid: [] for aid, (d, _) in base.items() if d.value == MRI}
    for analysis in record.get("analyses") or []:
        if not isinstance(analysis, dict):
            continue
        linked = _ids(analysis.get("acquisitions"))
        measure = value_of(analysis.get("measure"))
        measure = measure[0] if isinstance(measure, list) and measure else measure
        mtype = measures.get(str(measure))
        if mtype is None:
            continue
        implied = measure_class(mtype)
        for aid in linked:
            if aid not in evidence:
                continue
            siblings = {base[o][0].value for o in linked if o != aid and o in base}
            if implied is None or implied not in siblings:
                evidence[aid].append(mtype)
    for aid, types in evidence.items():
        base[aid] = (refine_mri(types, base[aid][0].text), ())
    return base


def annotate(record: dict) -> dict[str, int]:
    """Write the `Modality` entry on every acquisition's `modality`. Tally of what was written."""
    mapping_tool = _sidecar.tool("modality")
    readings = read(record)
    tally = {"set": 0, "unmapped": 0, "refined": 0}
    for acquisition in record.get("acquisitions") or []:
        if not isinstance(acquisition, dict) or not isinstance(acquisition.get("modality"), dict):
            continue
        wrapper = acquisition["modality"]
        reading = readings.get(str(value_of(acquisition.get("local_id"))))
        if reading is None or value_of(wrapper) in (None, NOT_REPORTED):
            _sidecar.replace(wrapper, mapping_tool, [])
            continue
        decision, also_read = reading
        entry = _sidecar.from_decision(
            decision, VOCABULARY, mapping_tool, also_read=list(also_read)
        )
        _sidecar.replace(wrapper, mapping_tool, [entry])
        tally["set"] += 1
        tally["unmapped"] += entry["status"] != _sidecar.MAPPED
        tally["refined"] += decision.reason.startswith("measure: ")
    return tally
