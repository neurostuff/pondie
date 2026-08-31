"""The benchmark runs from a clean clone, and its number does not silently move."""
import json

from pondie.benchmark.run import CANDIDATE, GOLD, REFERENCE, run


def test_the_gold_ships_with_the_benchmark_that_reads_it():
    assert len(list(GOLD.glob("*.direction.json"))) == 16
    assert REFERENCE.is_dir() and CANDIDATE.is_dir()


def test_direction_polarity_does_not_regress():
    """A floor, not a target. Raise it when a change earns it; never lower it silently."""
    result = run()
    assert result.papers >= 14, result.summary()
    assert result.scored_cells >= 55, result.summary()
    assert result.accuracy is not None and result.accuracy >= 0.94, result.summary()


def test_coverage_is_reported_so_the_headline_cannot_omit():
    """Polarity is measured on cells both sides signed; the rest must stay visible."""
    result = run()
    assert result.gold_cells > result.scored_cells, (
        "if every reviewed cell were scored, coverage would not be worth reporting")
    assert 0.0 < result.coverage < 1.0


def test_the_benchmark_discriminates_between_extraction_runs():
    """Not a tautology: gold is a third party, so a set scored against itself is still
    scored against the reviewers. The deployed extraction the reviewers saw reaches 96.6%
    that way; the candidate run shipped here does not."""
    deployed = run(candidate=REFERENCE, reference=REFERENCE)
    shipped = run()
    assert deployed.accuracy is not None and deployed.accuracy > shipped.accuracy, (
        f"deployed {deployed.summary()} vs candidate {shipped.summary()}")
    assert deployed.accuracy < 1.0, "a third-party gold means even the reviewed set can be wrong"
