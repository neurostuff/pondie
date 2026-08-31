"""One test per shape, on the cases that were wrong before they were rules."""

import pytest

from pondie.normalization import (
    coordinate_space,
    handedness_distribution,
    medication_status,
    multiple_comparison_method,
)


def test_a_space_naming_both_is_unknown_rather_than_a_guess():
    assert coordinate_space.normalize("MNI152").value == "MNI"
    ambiguous = coordinate_space.normalize("MNI/TAL")
    assert ambiguous.value == "UNKNOWN" and "MNI and TAL" in ambiguous.reason


def test_other_and_unknown_are_not_the_same_claim():
    assert coordinate_space.normalize("fsaverage").value == "OTHER", "a third space"
    assert coordinate_space.normalize("").value == "UNKNOWN", "no information"


def test_negation_decides_medication_status():
    for text in (
        "not medicated",
        "no longer receiving medication",
        "unmedicated",
        "free of psychotropic medication",
    ):
        assert medication_status.normalize(text).value == "FREE", text
    assert (
        medication_status.normalize("on stable antipsychotic medication").value == "MEDICATED"
    )


def test_a_negation_in_a_later_clause_does_not_invert_the_cohort():
    text = "Most patients were taking antidepressant medication; no medication changes"
    assert medication_status.normalize(text).value == "MEDICATED"


def test_non_left_handed_is_not_left_handed():
    assert handedness_distribution.normalize("non-left-handed").value == "RIGHT"
    assert handedness_distribution.normalize("left-handed").value == "LEFT"


def test_cluster_level_names_a_unit_not_an_error_family():
    assert multiple_comparison_method.normalize("cluster correction").value == "OTHER"
    assert multiple_comparison_method.normalize("family-wise error (FWE)").value == "FWE"
    assert multiple_comparison_method.normalize("uncorrected").value == "UNCORRECTED"


def test_a_missing_parser_is_an_error_not_an_unreported_field(monkeypatch):
    """Without a parse the field read UNKNOWN, which is what a silent paper reads too.

    So a broken environment was indistinguishable from a corpus that stopped reporting
    medication. It raises now, naming the package and the install.
    """

    from pondie import _deps
    from pondie.normalization import _negation

    _negation._parser.cache_clear()
    monkeypatch.setattr(
        _deps.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name))
    )
    with pytest.raises(_deps.MissingDependency, match="spacy"):
        medication_status.normalize("patients were medicated")
    _negation._parser.cache_clear()
