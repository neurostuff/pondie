"""What the schema reader must not get wrong.

It is the one thing between the LinkML files and everything that reads a record, so a change
here moves the prompt, the builder and the validator at once, silently and in the same
direction.
"""

from __future__ import annotations

import pytest

from pondie import schema as schema_paths
from pondie.schema import reader
from pondie import schema

#: Every slot that holds another entity's `local_id` rather than the entity itself.
#:
#: Pinned as a list, because the distinction rests on a property of the schema that nothing
#: else states. LinkML's own `is_inlined` says True for all 38 of these -- it inlines a class
#: range whose target declares no `identifier`, and `local_id` deliberately is not one. A
#: one-line `inlined_as_list: true` on any of them would reclassify it `nested` and the
#: prompt would start asking the model for whole records where a list of ids belongs. That
#: edit should turn this test red, not re-cache a prompt.
REFERENCE_SLOTS = frozenset(
    {
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
    }
)


def test_the_reference_slots_are_exactly_these(extraction_schema):
    found = frozenset(
        f"{name}.{slot}"
        for name in extraction_schema
        for slot, _spec, kind in extraction_schema.iter_slots(name)
        if kind == "reference"
    )
    assert found == REFERENCE_SLOTS, (
        "a slot changed between holding a local_id and holding the record itself; "
        f"newly nested: {sorted(REFERENCE_SLOTS - found)}, "
        f"newly a reference: {sorted(found - REFERENCE_SLOTS)}"
    )


def test_attributes_cannot_be_mutated_by_a_caller(extraction_schema):
    """The mapping is shared by every later reader in the process."""
    with pytest.raises(TypeError):
        extraction_schema.attributes("Group")["injected"] = None  # type: ignore[index]


def test_the_schema_is_loaded_once_however_the_path_is_spelled(extraction_schema):
    assert reader.load(schema_paths.EXTRACTION) is reader.load(str(schema_paths.EXTRACTION))


# -- what the reader answers about a slot ----------------------------------
#
# These four were in a file named for the review layer while the three above were here,
# so `schema.reader` had seven tests in two places and one of the two was named for it.


def test_classify_slot_separates_references_from_pipeline_scalars(extraction_schema: dict) -> None:
    analysis = extraction_schema.attributes("Analysis")
    metadata = extraction_schema.attributes("ExtractionMetadata")
    analysis_group = extraction_schema.attributes("AnalysisGroup")

    def kind(attrs: dict, name: str) -> str:
        return extraction_schema.classify(name, attrs[name])

    # Both range on a class; only the inlined one is owned rather than pointed at.
    assert kind(analysis, "model_estimation") == "reference"
    assert kind(analysis, "effect") == "nested"

    assert kind(metadata, "extractor_model") == "native"
    assert kind(analysis, "local_id") == "identifier"
    assert kind(analysis, "name") == "evidence"

    # Sibling slots on one class differ in kind.
    assert kind(analysis_group, "group") == "reference"
    assert kind(analysis_group, "n") == "evidence"


def test_attributes_for_includes_is_a_ancestors_and_slot_usage(extraction_schema: dict) -> None:
    extracted_string = extraction_schema.attributes("ExtractedString")
    # inherited from ExtractedValue
    assert "extraction_status" in extracted_string
    # narrowed by slot_usage from Any to string
    assert extracted_string["value"]["range"] == "string"
    assert extraction_schema.attributes("ExtractedValue")["value"]["range"] == "Any"


def test_entity_lists_cover_every_study_entity_list(extraction_schema: dict) -> None:
    """The payload merge has to accept every entity list Study declares.

    A hardcoded list does not fail loudly when the schema grows: an unlisted key
    is reported as an "unexpected payload key" note and the entities are dropped
    from the record. `arms` and `timepoints` were lost that way, which cost every
    intervention and longitudinal paper its arms and occasions.
    """

    study = extraction_schema.attributes("Study")
    declared = {name for name, attribute in study.items() if attribute.multivalued}
    assert declared, "Study should declare multivalued entity lists"
    assert declared <= set(schema.entity_lists())
    # A list directly on Study maps to itself.
    assert all(schema.entity_lists()[name] == name for name in declared)

    # A list one level down keeps its bare payload key and gains a dotted path, so an
    # extractor that emits arms.json does not have to know where the schema puts them.
    nested = extraction_schema.attributes(study["design"]["range"])
    for name in (n for n, a in nested.items() if a.multivalued):
        assert schema.entity_lists()[name] == f"design.{name}"


def test_resolves_to_follows_is_a(extraction_schema: dict) -> None:
    assert extraction_schema.resolves_to("ExtractedInteger", "ExtractedValue")
    assert extraction_schema.resolves_to("ExtractedValue", "ExtractedValue")
    assert not extraction_schema.resolves_to("Group", "ExtractedValue")
