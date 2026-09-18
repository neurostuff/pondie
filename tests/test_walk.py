"""One schema-guided traversal, and the designator the walkers read to do it.

`record/walk.py` exists because ten functions in the builder each walked the record
themselves and disagreed at the edges -- one passed no path, one followed lists where
another did not. `designated_type` is what all of them consult to know a node's class, so
what it does when a declaration is missing decides what every walker does.
"""

from __future__ import annotations

import pytest

from pondie import schema
from pondie import schema
from pondie.extraction.record import builder, fix
from pondie.extraction.record import validate as validate_record


# `Analysis.details` ranges on the abstract AnalysisDetails, whose only attribute is
# `details_type`; `seed_regions` is declared on ConnectivityDetails. A walker recursing on
# the declared range therefore never sees it, which on the corpus hid 40 shape errors.
def test_designated_type_follows_the_declaration(extraction_schema: dict) -> None:
    payload = {"details_type": "ConnectivityDetails"}
    assert extraction_schema.designated_type(payload, "AnalysisDetails") == "ConnectivityDetails"
    assert extraction_schema.type_designator("AnalysisDetails") == "details_type"
    assert extraction_schema.type_designator("Group") is None


@pytest.mark.parametrize("named", [None, "", "NotAClass", "Group", 7])
def test_designated_type_falls_back_rather_than_raising(extraction_schema: dict, named) -> None:
    """Silent by contract: a repair pass wants the best available answer, and `Group` is
    not an AnalysisDetails so naming it must not smuggle Group's slots in."""

    assert (
        extraction_schema.designated_type({"details_type": named}, "AnalysisDetails")
        == "AnalysisDetails"
    )


def test_listify_reaches_a_slot_declared_on_a_payload_subclass(extraction_schema: dict) -> None:
    """The 40-error regression guard. `seed_regions` is multivalued and lives on
    ConnectivityDetails, two hops down through a single-valued nested slot."""

    body = {
        "analyses": [
            {
                "local_id": "a1",
                "details": {"details_type": "ConnectivityDetails", "seed_regions": "reg_1"},
            }
        ]
    }
    fixed = fix.listify_nested(body, extraction_schema)
    assert body["analyses"][0]["details"]["seed_regions"] == ["reg_1"]
    assert any("seed_regions" in line for line in fixed)


def test_a_scalar_in_a_multivalued_wrapper_is_listified(extraction_schema: dict) -> None:
    """`interpretations` is an ExtractedStringList: one wrapper holding a list."""

    body = {
        "analyses": [
            {
                "local_id": "a1",
                "interpretations": {
                    "extraction_status": "extracted",
                    "value": "one finding",
                    "value_source": "reported",
                    "evidence": {"status": "not_found"},
                },
            }
        ]
    }
    fixed = fix.listify_scalars(body, extraction_schema)
    assert body["analyses"][0]["interpretations"]["value"] == ["one finding"]
    assert fixed == ["Study.analyses[0].interpretations"]


def test_a_missing_value_is_left_for_the_validator(extraction_schema: dict) -> None:
    """`extracted` with no value is a different fault and stays visible as one."""

    body = {
        "analyses": [
            {
                "local_id": "a1",
                "interpretations": {
                    "extraction_status": "extracted",
                    "value": None,
                    "evidence": {"status": "not_found"},
                },
            }
        ]
    }
    assert fix.listify_scalars(body, extraction_schema) == []


def test_a_scalar_where_an_enum_list_belongs_is_an_error(extraction_schema: dict) -> None:
    """`ExtractedResponseModalityList` declares its `value` with `any_of` and no `range`, so
    the shape check used to be unreachable and a bare string passed silently."""

    validator = validate_record.Validator(extraction_schema, None)
    validator.check_field(
        {
            "extraction_status": "extracted",
            "value": "button_press",
            "value_source": "reported",
            "evidence": {"status": "not_found"},
        },
        "ExtractedResponseModalityList",
        "Study.tasks[0].response_modality",
    )
    assert [e for e in validator.errors if "must be a list of ResponseModality" in e]
