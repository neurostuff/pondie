"""`Acquisition.modality` onto study_schema's `Modality`, on real values.

The wordings are `Acquisition.modality` values and OtherModality `modality_label`s as
production records carry them (ns-pond's beast catalog, 2026-11-19).
`fixtures/normalization/modality_records.json` holds nine real records trimmed to their
acquisitions, the analyses' acquisition and measure links, and those measures' types.
"""

import json
from pathlib import Path

import pytest
from study_schema.jsonschema import value_parents
from study_schema.models.storage import Modality

from pondie.normalization import apply_normalized, modality

DATA = json.loads(
    (Path(__file__).parent / "fixtures" / "normalization" / "modality_records.json").read_text()
)


def acquisition(value, label=None):
    out = {"local_id": "acq", "modality": {"value": value, "extraction_status": "extracted"}}
    if label is not None:
        out["modality_label"] = {"value": label, "extraction_status": "extracted"}
    return out


def reading(value, label=None):
    decision, _ = modality.read_acquisition(acquisition(value, label))
    return decision


@pytest.mark.parametrize("value, label, expected", [
    # the schema's own values, as the extractor writes them
    ("fMRI", None, "fMRI"), ("sMRI", None, "sMRI"), ("dMRI", None, "dMRI"),
    ("EEG", None, "EEG"), ("fNIRS", None, "fNIRS"), ("PET", None, "PET"),
    ("MRS", None, "MRS"), ("iEEG", None, "iEEG"), ("physiological", None, "physiological"),
    ("MEG", "magnetoencephalography (MEG)", "MEG"),
    ("SPECT", "single photon emission computed tomography (SPECT)", "SPECT"),
    ("SPECT", "123I-FP-CIT SPECT", "SPECT"),
    ("CT", None, "CT"), ("CT", "computerized tomography (CT)", "CT"),
    # free text in `modality`
    ("ECoG", None, "iEEG"), ("electrocorticography", None, "iEEG"), ("sEEG", None, "iEEG"),
    ("ECG", None, "physiological"), ("EOG", None, "physiological"),
    ("EMG", None, "physiological"), ("eye-tracking", None, "physiological"),
    ("eye tracking", None, "physiological"), ("nasal pressure recording", None, "physiological"),
    ("OtherModality", "eye tracking", "physiological"),
    # `other`, read through its label
    ("other", "CT", "CT"), ("other", "computed tomography (CT)", "CT"),
    ("other", "whole-brain research CT scans without contrast", "CT"), ("other", "CBCT", "CT"),
    ("other", "X-ray micro-CT", "CT"), ("other", "postoperative CT", "CT"),
    ("other", "subdural electrocorticography (ECoG)", "iEEG"),
    ("other", "stereoelectroencephalography (SEEG)", "iEEG"),
    ("other", "intracranial EEG (iEEG)", "iEEG"),
    ("other", "local field potentials (LFPs)", "iEEG"),
    ("other", "extracellular single-unit recordings", "iEEG"),
    ("other", "stereotactically placed depth electrodes (sEEGs)", "iEEG"),
    ("other", "1H-MRS", "MRS"), ("other", "fMRS", "MRS"), ("other", "MRS", "MRS"),
    ("other", "high-density diffuse optical tomography (HD-DOT)", "fNIRS"),
    ("other", "event-related optical signal (EROS)", "fNIRS"),
    ("other", "electromyography (EMG)", "physiological"),
    ("other", "MRI-compatible EMG", "physiological"),
    ("other", "MR-compatible infrared EYElink 1000 eye-tracker system", "physiological"),
    ("other", "fMRI-compatible high-speed eye-tracker", "physiological"),
    ("other", "electrocardiogram (ECG)", "physiological"),
    ("other", "skin conductance response (SCR)", "physiological"),
    ("other", "Pupil diameter", "physiological"),
    ("other", "photoplethysmography (PPG)", "physiological"),
    ("other", "respiration", "physiological"),
    ("other", "magnetoencephalography (MEG)", "MEG"),
    # an acquisition outside the vocabulary, said to be one, stays `other`
    ("other", "transcranial magnetic stimulation", "other"),
    ("other", "functional transcranial Doppler", "other"),
    ("other", "swept-source optical coherence tomography angiography (SS-OCTA)", "other"),
    ("other", None, "other"),
])
def test_wording_maps(value, label, expected):
    assert reading(value, label).value == expected


def test_a_label_that_decided_is_named_in_also_read():
    record = {"acquisitions": [acquisition("other", "electrocorticography (ECoG)")]}
    modality.annotate(record)
    (entry,) = record["acquisitions"][0]["modality"]["normalized"]
    assert (entry["vocabulary"], entry["status"], entry["value"]) == ("Modality", "mapped", "iEEG")
    assert entry["also_read"] == ["modality_label"]


def test_free_text_no_rule_reaches_is_unmatched_not_other():
    """ns-pond answered `other` here. The paper did not place itself outside the named
    modalities; the reading failed, so it is reported (docs: `_lexicon`, `modality`)."""
    decision = reading("human developmental anatomy")
    assert (decision.value, decision.reason) == ("UNKNOWN", "unmatched")


@pytest.mark.parametrize("value, label", [
    ("other", "BrainMap® functional database"), ("other", "Neurosynth database"),
    ("other", "coordinate-based meta-analysis (CBMA)"),
    ("meta-analytic connectivity modeling", None), ("coordinate-based ALE meta-analysis", None),
    ("other", "whole-exome sequencing (WES)"), ("other", "RNA-seq"),
    ("other", "Immunohistochemistry"), ("other", "confocal laser scanning microscopy"),
    ("other", "Simulated MEG data"), ("other", "Molecular docking"),
    ("other", "Quantitative autoradiographic receptor binding"),
])
def test_not_an_acquisition_is_unmapped(value, label):
    decision = reading(value, label)
    assert decision.value == "UNKNOWN"
    assert decision.reason.startswith("not an acquisition")


@pytest.mark.parametrize("value, label", [
    ("CT or MRI", None),
    ("other", "fMRI or positron emission tomography (PET)"),
    ("OtherModality", "fMRI and positron emission tomography (PET) studies"),
])
def test_two_modalities_are_ambiguous(value, label):
    decision = reading(value, label)
    assert decision.value == "UNKNOWN" and decision.reason.startswith("matches ")


@pytest.mark.parametrize("wording, expected", [
    ("BOLD MRI", "fMRI"),                     # an ambiguity before OUTRANKS
    ("T1-weighted MRI", "sMRI"),              # likewise
    ("magnetic resonance spectroscopy", "MRS"),
    ("intracranial EEG", "iEEG"),
    ("PET-CT", "PET"),
])
def test_a_refinement_outranks_the_value_it_refines(wording, expected):
    assert modality.normalize(wording).value == expected


def test_unreported_modality_is_not_read():
    assert modality.read_acquisition({"local_id": "a", "modality": {"value": None}}) is None


def readings(case):
    return {k: d.value for k, (d, _) in modality.read(DATA[case]["record"]).items()}


def test_bare_mri_takes_its_analyses_measure():
    # an ASL acquisition whose one analysis is cerebral blood flow: fMRI, as the schema defines it
    assert readings("cbf") == {"acq_smri": "sMRI", "acq_asl": "fMRI", "acq_fmri": "fMRI"}
    assert readings("mrs") == {f"acq_mrs_{t}": "MRS" for t in ("15t", "3t", "4t")}


def test_measure_a_sibling_explains_is_no_evidence():
    # gray matter volume is the linked sMRI acquisition's; the MRI acquisition (ASL) is not refined
    assert readings("gmv")["acq_asl"] == "MRI"
    # diffusion is the dMRI acquisition's; the bare MRI beside it (spectroscopy) stays MRI
    assert readings("diff")["acq_mrs"] == "MRI"
    assert readings("mixed")["acq_asl"] == "MRI"


def test_a_sibling_of_another_modality_leaves_the_evidence():
    # lesions traced on MRI beside CT: the CT does not explain a structural measure away
    assert readings("ct") == {"acq_mri": "sMRI", "acq_ct": "CT"}


def test_other_acquisitions_in_real_records():
    assert readings("ecog") == {"acq_ecog_nihon": "iEEG", "acq_ecog_blackrock": "iEEG"}
    assert readings("notacq") == {"acq_brainmap": "UNKNOWN"}
    assert readings("physio_compat")["acq_emg"] == "physiological"


@pytest.mark.parametrize("measures, expected", [
    ({"bold_response"}, "fMRI"), ({"cerebral_blood_flow"}, "fMRI"),
    ({"connectivity_strength"}, "fMRI"),
    ({"gray_matter_volume", "cortical_thickness"}, "sMRI"),
    ({"magnetic susceptibility"}, "sMRI"),
    ({"myelin water fraction"}, "sMRI"), ({"diffusion_metric"}, "dMRI"),
    ({"31P metabolite concentration"}, "MRS"),
    ({"bold_response", "gray_matter_volume"}, "MRI"),        # disagree
    ({"component_loading"}, "MRI"), ({"postural stability"}, "MRI"), (set(), "MRI"),
])
def test_refine_mri(measures, expected):
    assert modality.refine_mri(measures).value == expected


def test_apply_normalized_writes_one_modality_entry_per_acquisition():
    """pondie is the one writer of `Modality` on this slot: a rerun replaces its own entry."""
    import copy

    record = copy.deepcopy(DATA["cbf"]["record"])
    apply_normalized(record)
    apply_normalized(record)
    for acq in record["acquisitions"]:
        entries = [e for e in acq["modality"]["normalized"] if e["vocabulary"] == "Modality"]
        assert len(entries) == 1
    asl = next(a for a in record["acquisitions"] if a["local_id"] == "acq_asl")
    (entry,) = asl["modality"]["normalized"]
    assert (entry["value"], entry["method"]) == ("fMRI", "measure: cerebral_blood_flow")


def test_values_and_parents_are_study_schemas():
    assert modality.VALUES[:-1] == tuple(m.value for m in Modality)
    assert modality.PARENT == value_parents("storage-study", "Modality")
    targets = {rule.value for rule in modality.RULES} | set(modality.OUTRANKS)
    assert targets <= set(modality.VALUES)
