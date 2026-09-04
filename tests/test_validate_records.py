"""The validator runs over the records this repository ships.

432 tests passed while `check_value_type` raised `AttributeError` on every record carrying a
multivalued extracted value -- `Group.medical_condition` alone is 237 groups with two or more
conditions -- because no test put one through the validator. A checker with no test over real
data checks nothing.
"""

from __future__ import annotations

import json

import pytest

from pondie import paths, schema
from pondie.extraction.record import validate
from pondie.schema import reader

RECORDS = sorted((paths.REPO / "benchmarks" / "candidate").glob("*.extraction.json"))
RECORDS += sorted((paths.REPO / "benchmarks" / "gold").glob("*.extraction.json"))


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.name.split(".")[0])
def test_the_validator_runs_over_every_shipped_record(path):
    """Not "is valid" -- these are real extractions and some carry real defects. Only that
    the checker completes, so a defect is reported as a defect and not as a traceback."""
    validator = validate.Validator(reader.load(schema.EXTRACTION), None)
    validator.check_record(json.loads(path.read_text()))
    for message in validator.errors + validator.warnings:
        assert isinstance(message, str) and message.strip()


def test_a_multivalued_extracted_value_is_checked_item_by_item():
    """The shape `extraction-readme.md` leads with: one wrapper over a list, not a list of
    wrappers. The branch that accepts it recursed with a dict comprehension over the slot,
    which stopped being a dict when the schema reader started returning `SlotDefinition`."""
    validator = validate.Validator(reader.load(schema.EXTRACTION), None)
    node = {
        "extraction_status": "extracted",
        "value": ["depression", "anxiety"],
        "value_source": "reported",
        "evidence": {"status": "not_found"},
    }
    validator.check_field(node, "ExtractedStringList", "Group.medical_condition")
    assert validator.errors == [], validator.errors

    # A wrong item type inside the list is still caught.
    validator = validate.Validator(reader.load(schema.EXTRACTION), None)
    validator.check_field(
        {**node, "value": ["depression", {"nested": "object"}]},
        "ExtractedStringList",
        "Group.medical_condition",
    )
    assert validator.errors, "a non-string item in a string list must be reported"


def test_a_wrapped_type_designator_still_names_the_subclass():
    """26424424, 19914045 and 22952599 each reported three "attribute is not declared on
    Acquisition" violations for slots that are declared on `MRI`. The repair pass had
    wrapped `acquisition_type`, which is declared a plain string; the validator then read a
    dict where it wanted a class name and fell back to the declared class, so one real
    violation became four. `edit.apply` no longer writes the designator -- this is the
    second line of defence, so a bad write is reported as itself and not as a cascade."""
    sch = reader.load(schema.EXTRACTION)
    designator = sch.type_designator("Acquisition")
    node = {
        "local_id": "acq_mri",
        designator: {"extraction_status": "extracted", "value": "MRI",
                     "value_source": "reported", "evidence": {"status": "not_found"}},
        "magnetic_field_strength_tesla": {
            "extraction_status": "extracted", "value": 3.0,
            "value_source": "reported", "evidence": {"status": "not_found"}},
    }
    validator = validate.Validator(sch, None)
    assert validator.resolve_type(node, "Acquisition", "Study.acquisitions[]") == "MRI"

    validator = validate.Validator(sch, None)
    validator.check_instance(node, "Acquisition", "Study.acquisitions[]")
    assert not [e for e in validator.errors if "is not declared" in e], validator.errors
