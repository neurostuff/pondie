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

from conftest import (  # the shared paper harness
    requires_current_record,
    requires_paper,
)
import copy
from pondie.extraction.record import fix
from pondie.extraction.record import spans as span_tools
from pondie.formats import text_index, values
from pondie.extraction.record import walk
from pondie.extraction.record import validate as validate_record
from pondie.extraction.evidence import warrant

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


def test_a_blank_may_say_why_and_an_extracted_value_may_not():
    """`unreported_reason` qualifies a blank; on a filled slot there is nothing to explain.

    The vocabulary deliberately has no `not_applicable`: extraction-readme.md reserves that
    as a *value* -- "the concept does not apply", which an observational study's allocation
    genuinely is -- and a blank claiming it would be silence recorded where an answer belongs.
    """
    sch = reader.load(schema.EXTRACTION)
    blank = {
        "extraction_status": "not_reported",
        "unreported_reason": "outside_text",
        "evidence": {"status": "not_applicable"},
    }

    validator = validate.Validator(sch, None)
    validator.check_field(blank, "ExtractedString", "Group.name")
    assert validator.errors == [], validator.errors

    # Absent is the ordinary case: `not_reported` already says the attribute was examined
    # and the source carries no value, so plain silence adds no reason at all.
    validator = validate.Validator(sch, None)
    validator.check_field(
        {k: v for k, v in blank.items() if k != "unreported_reason"},
        "ExtractedString",
        "Group.name",
    )
    assert validator.errors == [], validator.errors

    validator = validate.Validator(sch, None)
    validator.check_field(
        {**blank, "unreported_reason": "silent"}, "ExtractedString", "Group.name"
    )
    assert validator.errors, "`silent` was removed from the vocabulary and must be refused"

    validator = validate.Validator(sch, None)
    validator.check_field(
        {
            "extraction_status": "extracted",
            "value": "controls",
            "value_source": "reported",
            "unreported_reason": "outside_text",
            "evidence": {"status": "not_found"},
        },
        "ExtractedString",
        "Group.name",
    )
    assert validator.errors, "a reason on a filled slot must be reported"


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
        designator: {
            "extraction_status": "extracted",
            "value": "MRI",
            "value_source": "reported",
            "evidence": {"status": "not_found"},
        },
        "magnetic_field_strength_tesla": {
            "extraction_status": "extracted",
            "value": 3.0,
            "value_source": "reported",
            "evidence": {"status": "not_found"},
        },
    }
    validator = validate.Validator(sch, None)
    assert validator.resolve_type(node, "Acquisition", "Study.acquisitions[]") == "MRI"

    validator = validate.Validator(sch, None)
    validator.check_instance(node, "Acquisition", "Study.acquisitions[]")
    assert not [e for e in validator.errors if "is not declared" in e], validator.errors


# -- a real record, and records the validator must refuse ------------------
#
# The positive cases run against the shipped paper: every span addresses the text, the hash
# matches, nothing dangles. The negative ones matter as much and are easier to forget -- a
# validator that accepts a corrupted record reports the same thing as one that works.


@requires_current_record
@requires_paper
def test_example_record_validates(record: dict, normalized: str, extraction_schema: dict) -> None:
    validator = validate_record.Validator(extraction_schema, normalized)
    validator.check_record(record)
    assert validator.errors == []
    assert validator.fields > 0
    assert validator.spans > 0


@requires_paper
def test_every_span_addresses_the_source_text(record: dict, normalized: str) -> None:
    checked = 0
    for evidence_set in warrant._iter_sets(record):
        for span in evidence_set["spans"]:
            span_tools.verify(normalized, span)
            checked += 1
    assert checked > 0


@requires_paper
def test_recorded_hash_matches_the_text(record: dict, normalized: str) -> None:
    declared = record["extraction_metadata"]["source_text_hash"]
    assert declared == text_index.text_hash(normalized)


@requires_paper
def test_no_dangling_cross_references(record: dict, extraction_schema: dict) -> None:
    assert fix.check_local_ids(record, extraction_schema) == []


@requires_paper
def test_section_index_covers_every_span(record: dict, normalized: str) -> None:
    """Every span must fall inside an indexed section, or reviewers get no hint."""

    sections = text_index.build_sections(normalized)
    for evidence_set in warrant._iter_sets(record):
        for span in evidence_set["spans"]:
            assert text_index.section_path(sections, span["start_char"]) is not None


@requires_paper
@pytest.mark.parametrize(
    "mutate, expected",
    [
        pytest.param(
            lambda r: r["groups"][0].update({"not_a_real_attribute": 1}),
            "is not declared",
            id="undeclared-attribute",
        ),
        pytest.param(
            lambda r: r["groups"][0].pop("local_id"),
            "required attribute 'local_id' is missing",
            id="missing-required",
        ),
        pytest.param(
            lambda r: r["groups"][0]["age_mean"].update({"value": ["not", "a", "number"]}),
            "must be a float, got a list",
            id="list-in-scalar",
        ),
        pytest.param(
            lambda r: r["groups"][0]["age_mean"].update({"extraction_status": "maybe"}),
            "extraction_status must be one of",
            id="bad-enum",
        ),
        pytest.param(
            # Any slot the extractor marked not_reported: the rule is about the
            # wrapper, not about which field it happens to wrap.
            lambda r: next(
                node
                for node in r["groups"][0].values()
                if isinstance(node, dict) and node.get("extraction_status") == "not_reported"
            ).update({"value": "smuggled in"}),
            "not_reported fields must omit value",
            id="not-reported-with-value",
        ),
        pytest.param(
            lambda r: r["extraction_metadata"].update({"source_text_hash": "0" * 64}),
            "does not match the supplied text",
            id="wrong-hash",
        ),
        pytest.param(
            lambda r: r["extraction_metadata"]["paper_sections"][0].update({"ordinal": -1}),
            "must be >= 0",
            id="negative-minimum",
        ),
        pytest.param(
            lambda r: r["extraction_metadata"]["paper_sections"][0].update({"level": "one"}),
            "must be a integer, got str",
            id="wrong-native-type",
        ),
    ],
)
def test_validator_rejects_corrupted_record(
    record: dict, normalized: str, extraction_schema: dict, mutate, expected: str
) -> None:
    broken = copy.deepcopy(record)
    mutate(broken)

    validator = validate_record.Validator(extraction_schema, normalized)
    validator.check_record(broken)

    assert validator.errors, f"expected an error containing {expected!r}"
    assert any(expected in error for error in validator.errors), validator.errors


@requires_paper
def test_validator_rejects_shifted_span_offset(
    record: dict, normalized: str, extraction_schema: dict
) -> None:
    broken = copy.deepcopy(record)
    for evidence_set in warrant._iter_sets(broken):
        evidence_set["spans"][0]["start_char"] += 3
        break

    validator = validate_record.Validator(extraction_schema, normalized)
    validator.check_record(broken)
    assert any("disagrees with source" in error for error in validator.errors), validator.errors


@requires_paper
def test_validator_rejects_evidence_set_without_spans(
    record: dict, normalized: str, extraction_schema: dict
) -> None:
    broken = copy.deepcopy(record)
    for evidence_set in warrant._iter_sets(broken):
        evidence_set["spans"] = []
        break

    validator = validate_record.Validator(extraction_schema, normalized)
    validator.check_record(broken)
    assert any("at least one span" in error for error in validator.errors), validator.errors


def test_a_quote_that_is_not_text_is_unresolved_not_a_crash():
    """dem_p_effort_medium: a JSON-mode reply put an object where a quote goes, and the
    whole build raised."""
    record = {"analyses": [{"name": {
        "extraction_status": "extracted", "value": "x", "value_source": "reported",
        "evidence": {"status": "present", "sets": [{"quotes": [{"text": "x"}]}]}}}]}
    report = warrant.warrant(record, "x marks the spot.")
    assert report.unresolved and "not text" in report.unresolved[0]


# -- explicit null ---------------------------------------------------------------------
#
# study_schema's layouts write an empty optional slot as `null`, so a record that went
# through them is the pondie record with every absent slot spelled out.

NULLED_EXAMPLE = paths.REPO / "tests" / "fixtures" / "x7Ex4mpleA01.extraction.json"
GOLD = sorted((paths.REPO / "benchmarks" / "gold").glob("*.extraction.json"))


def _errors(record):
    validator = validate.Validator(reader.load(schema.EXTRACTION), None)
    validator.check_record(record)
    return validator.errors


def _as_layouts_write(record):
    """What `Run.records` writes for this record: the study_schema model, nulls included."""
    from study_schema.models.extraction import Study

    return Study.model_validate(record).model_dump(mode="json")


def test_the_example_study_schema_writes_is_valid():
    """The fixture is study_schema's own `examples/run/records` file, null for every
    optional slot, which is how its layouts serialize it."""
    record = json.loads(NULLED_EXAMPLE.read_text())
    assert record["groups"] is None
    assert record == _as_layouts_write(record)
    assert _errors(record) == []


def _wrappers(record, status):
    return [w for _, w in values.iter_fields(record) if w.get("extraction_status") == status]


def _with_faults(record):
    """A copy with one fault of each kind a wrapper can have, all of which the study_schema
    model still loads, so the round trip can be run on it."""
    record = copy.deepcopy(record)
    extracted = _wrappers(record, "extracted")
    blank = _wrappers(record, "not_reported")
    extracted[0]["evidence"] = {"status": "not_applicable"}
    extracted[1]["unreported_reason"] = "ambiguous"
    extracted[2]["evidence"] = {"status": "present"}
    blank[0]["value"] = "a value on a blank"
    blank[1]["evidence"] = {"status": "not_found"}
    return record


@pytest.mark.parametrize("path", GOLD, ids=lambda p: p.name.split(".")[0])
@pytest.mark.parametrize("faulty", [False, True], ids=["as_shipped", "with_faults"])
def test_a_record_validates_the_same_with_its_absent_slots_written_as_null(path, faulty):
    """Null is absence: the record and its layouts round trip report the same errors. The
    round trip fills in wrapper slots too (`value`, `unreported_reason`), and the faulted
    copy makes the comparison one of real errors rather than of two empty lists."""
    record = json.loads(path.read_text())
    if faulty:
        record = _with_faults(record)
    nulled = _as_layouts_write(record)
    assert json.dumps(nulled).count("null") > 100
    errors = _errors(record)
    assert bool(errors) is faulty
    assert _errors(nulled) == errors


def _extracted_wrapper():
    record = json.loads(GOLD[0].read_text())
    owner = next(
        e.node for e in walk.entities(record, reader.load(schema.EXTRACTION))
        if values.is_field(e.node.get("name") or {}) and e.node["name"]["extraction_status"] == "extracted"
    )
    return record, owner["name"]


@pytest.mark.parametrize("slot", ["value", "evidence", "extraction_status"])
def test_an_extracted_wrapper_with_a_null_slot_is_an_error(slot):
    record, wrapper = _extracted_wrapper()
    assert _errors(record) == []
    wrapper[slot] = None
    assert _errors(record), f"a null {slot} on an extracted wrapper went unreported"


@pytest.mark.parametrize("omit", [False, True], ids=["null", "omitted"])
def test_an_extracted_wrapper_without_a_value_says_so(omit):
    record, wrapper = _extracted_wrapper()
    if omit:
        del wrapper["value"]
    else:
        wrapper["value"] = None
    assert any("extracted fields must carry a value" in e for e in _errors(record))


def test_value_of_reads_a_null_valued_blank_as_not_reported():
    omitted = {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}}
    nulled = {**omitted, "value": None, "unreported_reason": None}
    assert values.value_of(omitted) is values.NOT_REPORTED
    assert values.value_of(nulled) is values.NOT_REPORTED
    assert values.value_of({**omitted, "value": 0}) == 0


def test_null_does_not_satisfy_a_required_slot():
    record = json.loads(NULLED_EXAMPLE.read_text())
    record["local_id"] = None
    assert any("'local_id' is null" in error for error in _errors(record))


def test_a_multivalued_slot_still_rejects_a_wrong_type():
    record = json.loads(NULLED_EXAMPLE.read_text())
    record["groups"] = "not a list"
    assert any("'groups' is multivalued but got str" in e for e in _errors(record))
