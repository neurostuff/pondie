"""What `cast` must refuse, and what it must let through.

Every case is a value a model actually produced against a slot that could not hold it, taken
from the repair pass over the neurometabench corpus. The wrapper rejects them at validation;
the point of casting is to not write them in the first place, because a field that fails
validation is a field a reviewer has to adjudicate.
"""

from __future__ import annotations

import pytest

from pondie.formats import values


@pytest.mark.parametrize(
    "class_name,slot,value,expected",
    [
        # 28416565: `is_healthy` was given the word, and `bool("false")` is True
        ("Group", "is_healthy", "true", True),
        ("Group", "is_healthy", "No", False),
        ("Group", "is_healthy", "mostly", None),
        # 29740753: counts and means arrived as strings
        ("Group", "acquired_count", "31", 31),
        ("Group", "acquired_count", "about twenty", None),
        ("Group", "age_mean", "33.4", 33.4),
        ("Analysis", "prespecification", "exploratory", "exploratory"),
        # 28888350: "post-hoc" is accurate about the analysis and is not one of the two
        # values. It used to be refused, which cost the whole Analysis, since the slot is
        # required; the vocabulary is open now and `normalization.prespecification` maps it.
        ("Analysis", "prespecification", "post-hoc", "post-hoc"),
        # A closed enum is still closed, and this is the case that says so: `Cell.direction`
        # is filled by `derive`, so an off-vocabulary value there is not a synonym to keep.
        ("Cell", "direction", "positive", "positive"),
        ("Cell", "direction", "increase", None),
        ("Region", "definition_method", "atlas", "atlas"),
        # an open vocabulary takes the source's own wording. `definition_method` is required,
        # so a closed range discarded the whole Region over the one unrecognised word --
        # `create` returned "Region would be missing definition_method" and the name,
        # description and atlas the proposal also carried went with it.
        ("Region", "definition_method", "hand drawn by an expert", "hand drawn by an expert"),
        ("Region", "region_type", "gray matter", "gray matter"),
        ("Region", "name", "hippocampus", "hippocampus"),
    ],
)
def test_a_value_fits_its_slot_or_is_refused(storage_schema, class_name, slot, value, expected):
    assert values.cast(storage_schema, class_name, slot, value) == expected


def test_a_slot_the_class_does_not_declare_takes_nothing(storage_schema):
    """23021615: `correction_scope` was written onto three analyses; it belongs to
    InferenceSettings, which those analyses already referenced."""

    assert values.cast(storage_schema, "Analysis", "correction_scope", "roi") is None


def test_a_multivalued_slot_gets_a_list(storage_schema):
    """`Task.response_modality` is multivalued, and the scalar produced
    "ExtractedResponseModalityList.value must be a list of ResponseModality or string, got str"."""

    assert values.shape(storage_schema, "Task", "response_modality", "button press") == [
        "button press"
    ]
    assert values.shape(storage_schema, "Group", "acquired_count", "31") == 31


def test_shape_refuses_what_cast_refuses(storage_schema):
    assert values.shape(storage_schema, "Group", "is_healthy", "mostly") is None
