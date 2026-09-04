"""What the schema reader must not get wrong.

It is the one thing between the LinkML files and everything that reads a record, so a change
here moves the prompt, the builder and the validator at once, silently and in the same
direction.
"""

from __future__ import annotations

import pytest

from pondie import schema as schema_paths
from pondie.schema import reader


@pytest.fixture(scope="module")
def sch():
    return reader.load(schema_paths.EXTRACTION)


#: Every slot that holds another entity's `local_id` rather than the entity itself.
#:
#: Pinned as a list, because the distinction rests on a property of the schema that nothing
#: else states. LinkML's own `is_inlined` says True for all 38 of these -- it inlines a class
#: range whose target declares no `identifier`, and `local_id` deliberately is not one. A
#: one-line `inlined_as_list: true` on any of them would reclassify it `nested` and the
#: prompt would start asking the model for whole records where a list of ids belongs. That
#: edit should turn this test red, not re-cache a prompt.
REFERENCE_SLOTS = frozenset({
    "Acquisition.device",
    "Analysis.acquisitions",
    "Analysis.assessments",
    "Analysis.defines_regions",
    "Analysis.inference_settings",
    "Analysis.measure",
    "Analysis.mirror_of",
    "Analysis.model_estimation",
    "Analysis.regions",
    "Analysis.tables",
    "Analysis.tasks",
    "AnalysisGroup.group",
    "Cell.term",
    "ConnectivityDetails.seed_regions",
    "ConnectivityDetails.target_regions",
    "ConnectivityEdge.source_region",
    "ConnectivityEdge.target_region",
    "DecodingClass.condition",
    "EEG.device",
    "FNIRS.device",
    "FactorLevel.arms",
    "FactorLevel.conditions",
    "FactorLevel.groups",
    "FactorLevel.regions",
    "FactorLevel.timepoints",
    "Group.arm",
    "InferenceSettings.correction_regions",
    "Group.diagnostic_instrument",
    "LatentDecompositionDetails.second_block_assessments",
    "MRI.device",
    "Mediation.mediator",
    "ModelEstimation.inputs_from",
    "ModelEstimation.preprocessing",
    "ModelTerm.assessment",
    "ModelTerm.interaction_with",
    "ModelTerm.region",
    "OtherModality.device",
    "PET.device",
    "Task.acquisitions",
})


def test_the_reference_slots_are_exactly_these(sch):
    found = frozenset(
        f"{name}.{slot}"
        for name in sch
        for slot, _spec, kind in sch.iter_slots(name)
        if kind == "reference"
    )
    assert found == REFERENCE_SLOTS, (
        "a slot changed between holding a local_id and holding the record itself; "
        f"newly nested: {sorted(REFERENCE_SLOTS - found)}, "
        f"newly a reference: {sorted(found - REFERENCE_SLOTS)}"
    )


def test_attributes_cannot_be_mutated_by_a_caller(sch):
    """The mapping is shared by every later reader in the process."""
    with pytest.raises(TypeError):
        sch.attributes("Group")["injected"] = None  # type: ignore[index]


def test_the_schema_is_loaded_once_however_the_path_is_spelled(sch):
    assert reader.load(schema_paths.EXTRACTION) is reader.load(str(schema_paths.EXTRACTION))
