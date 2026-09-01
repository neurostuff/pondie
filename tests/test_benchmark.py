"""The benchmark runs from a clean clone, and its number does not silently move."""



import pytest

from pondie.benchmark import CANDIDATE, DIRECTION_GOLD, REFERENCE, run


@pytest.fixture(scope="module")
def result():
    """Scored once. Every field of the gold record goes through the entity matcher, which
    is the expensive part, and no test here mutates what it gets back."""
    return run()


def test_the_gold_ships_with_the_benchmark_that_reads_it():
    assert len(list(DIRECTION_GOLD.glob("*.direction.json"))) == 16
    assert REFERENCE.is_dir() and CANDIDATE.is_dir()


def test_direction_polarity_does_not_regress(result):
    """A floor, not a target. Raise it when a change earns it; never lower it silently."""
    direction = result.direction
    assert direction.papers >= 14, direction
    assert direction.scored_cells >= 55, direction
    assert direction.accuracy is not None and direction.accuracy >= 0.94, direction


def test_coverage_is_reported_so_the_headline_cannot_omit(result):
    """Polarity is measured on the cells that carry weight; the rest must stay visible."""
    direction = result.direction
    assert (
        direction.gold_cells > direction.scored_cells
    ), "if every reviewed cell were scored, coverage would not be worth reporting"
    assert 0.0 < direction.coverage < 1.0


def test_the_benchmark_discriminates_between_extraction_runs():
    """Not a tautology: gold is a third party, so a set scored against itself is still
    scored against the reviewers. The deployed extraction the reviewers saw reaches 96.6%
    that way; the candidate run shipped here does not."""
    deployed = run(candidate=REFERENCE, reference=REFERENCE).direction
    shipped = run().direction
    assert (
        deployed.accuracy is not None and deployed.accuracy > shipped.accuracy
    ), f"deployed {deployed} vs candidate {shipped}"
    assert (
        deployed.accuracy < 1.0
    ), "a third-party gold means even the reviewed set can be wrong"


# --- what the field half must report ------------------------------------------


def test_every_field_gets_precision_recall_and_f1(result):
    """Per field, not only per entity type.

    "Analysis is 94% accurate" does not say which of its thirty fields to go and fix, and
    the two questions have different answers: a type scores well while one field inside it
    is always wrong.
    """
    assert result.records_scored >= 1
    scored = [f for f in result.fields if f.scored]
    assert len(scored) > 50, "one row per field the gold record actually exercises"
    for field in scored:
        assert "." in field.field, "a field is named Class.field, not just Class"
        assert 0.0 <= field.precision <= 1.0
        assert 0.0 <= field.recall <= 1.0
        assert 0.0 <= field.f1 <= 1.0


def test_presence_and_value_are_scored_apart(result):
    """P/R/F1 answer "was the field filled"; accuracy answers "was the value right".

    Conflating them hides the commoner defect: a field filled everywhere it should be,
    with the wrong value in it, would otherwise read as a perfect score.
    """
    filled_but_wrong = [
        f for f in result.fields if f.f1 == 1.0 and f.accuracy is not None and f.accuracy < 1.0
    ]
    assert filled_but_wrong, "the shipped candidate has fields it fills correctly but fills wrongly"


def test_a_field_both_sides_left_empty_is_not_scored_as_a_failure(result):
    """With tp=fp=fn=0 the F1 formula yields 0.0, which reads as total failure.

    Both sides agreeing the paper is silent is a correct answer, so those fields are counted
    and kept out of the scored set rather than dragging the macro average down.
    """
    unscored = [f for f in result.fields if not f.scored]
    assert unscored, "the gold record has fields neither side fills"
    for field in unscored:
        assert field.both_filled == 0 and field.missed == 0 and field.spurious == 0
        assert field.agreed_absent > 0


def test_the_report_puts_the_worst_field_first(result):
    """The report exists to say what to fix, so it is ordered by what is most broken."""
    fields_section = result.report().split("FIELDS")[1]
    f1s = [
        float(line.split()[3].rstrip("%"))
        for line in fields_section.splitlines()
        if line.startswith("  ") and "%" in line
    ]
    assert f1s, "the table has rows"
    assert f1s == sorted(f1s), "worst F1 first, so the fixable thing is at the top"
