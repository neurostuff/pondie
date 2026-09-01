"""One test per shape, on the cases that were wrong before they were rules."""

import pytest

from pondie.normalization import (
    _clustering,
    coordinate_space,
    handedness_distribution,
    medication_status,
    multiple_comparison_method,
)
from pondie.vocabularies import folding


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

    A broken environment was therefore indistinguishable from a corpus that stopped reporting
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


# ---------------------------------------------------------------------------
# The audit in docs/regex-audit.md, finding 6.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("naïve", "naive"),
    ("Étude", "etude"),
    ("Möbitz II", "mobitzii"),
    ("Müllerian", "mullerian"),
    ("gray matter", "graymatter"),
    ("Alzheimer's disease", "alzheimersdisease"),
])
def test_squash_folds_an_accent_rather_than_deleting_it(value, expected):
    """`squash` said it was "`fold` with the spaces removed" and skipped `fold`'s NFKD step,
    so an accented letter fell out of `[a-z0-9]` and was dropped: `naïve` squashed to
    `nave`. 245 of 160,565 MONDO and Cognitive Atlas surface forms are affected."""

    assert folding.squash(value) == expected


def test_squash_is_exactly_fold_without_the_spaces():
    for value in ["naïve", "Étude", "first-episode schizophrenia", "ADHD"]:
        assert folding.squash(value) == folding.fold(value).replace(" ", "")


def test_one_name_spelled_two_ways_clusters_as_one():
    """The single caller is exact-key clustering, so the deleted accent turned one name
    into two clusters -- the failure clustering exists to prevent."""

    assert _clustering.name_links(["drug naïve patients", "drug naive patients"]) == [(0, 1)]
