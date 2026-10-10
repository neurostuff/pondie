from pondie.schema import EXTRACTION, STORAGE
from pondie.schema.authoring import load_imported_classes
from pondie.schema.checks.extraction_to_storage_map import check_conditional_fields

STORAGE_CLASSES = load_imported_classes(STORAGE)
EXTRACTION_CLASSES = load_imported_classes(EXTRACTION)


def run(fields):
    return check_conditional_fields(
        STORAGE_CLASSES, EXTRACTION_CLASSES, {"conditional_fields": dict.fromkeys(fields, {})}
    )


def test_existing_field_passes():
    assert run(["Analysis.outcome"]) == []


def test_missing_section_passes():
    assert check_conditional_fields(STORAGE_CLASSES, EXTRACTION_CLASSES, {}) == []


def test_misspelled_field_is_reported():
    problems = run(["Analysis.outcom"])
    assert len(problems) == 2
    assert all("Analysis.outcom" in p for p in problems)


def test_unknown_class_is_reported():
    assert run(["Analyses.outcome"])
