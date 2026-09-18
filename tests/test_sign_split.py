"""A table reporting both signs is two contrasts, and partitioning it is arithmetic.

The rule runs before the model passes because this is the only point in the pipeline that
sees a row's values -- the extraction passes are shown captions, never cells. Checked here:
that it splits only on a total partition and refuses a partial one, that a p-value carries
no direction where a z-statistic does, and that running it twice changes nothing.
"""

from __future__ import annotations

from pondie.formats import table_parse as tables


def _point(*values: tuple[float, str]) -> dict:
    return {"coordinates": [0.0, 0.0, 0.0], "values": [{"value": v, "kind": k} for v, k in values]}


def _split(points: list[dict], name: str = "contrast"):
    from pondie.extraction.sign_split import split_opposite_signs

    split = split_opposite_signs([{"name": name, "points": points}])
    return list(split.analyses), list(split.notes)


# The schema wants one Analysis per normalized direction, and stage 1 is the only pass
# that can see the row values to enforce it. These pin the partition rule, and in
# particular the two shapes that look like defects and are not.
def test_opposite_signs_become_one_analysis_per_direction() -> None:
    out, notes = _split([_point((0.42, "correlation")), _point((-0.41, "correlation"))])
    assert [a["split_direction"] for a in out] == ["positive", "negative"]
    assert all(a["split_from"] == "contrast" for a in out)
    assert notes and notes[0].startswith("SPLIT")

    # The described half keeps the parsed name, because the extraction prompt tells the
    # model to quote that name verbatim and the paper's prose is about this half.
    assert out[0]["name"] == "contrast"
    assert not out[0].get("withhold")

    # The reversed half is never shown to the model: the paper does not describe it, so
    # a name and definition for it could only be invented. It is rebuilt from the
    # described half after extraction -- docs/deterministic-direction.md.
    assert out[1]["withhold"] is True
    assert out[1]["mirror_of"] == "contrast"


def test_a_single_row_in_the_minority_direction_still_splits() -> None:
    """One surviving cluster is an ordinary result of thresholding, not a bad parse.

    Gating the split on group size would leave a negative statistic sitting inside a
    contrast the paper named `>`, which is the contradiction the rule exists to remove.
    """
    out, _ = _split(
        [_point((3.0, "t-statistic")), _point((4.0, "t-statistic")), _point((-2.9, "t-statistic"))]
    )
    assert [len(a["points"]) for a in out] == [2, 1]


def test_an_all_positive_analysis_is_left_alone() -> None:
    out, notes = _split([_point((3.0, "t-statistic")), _point((4.0, "t-statistic"))])
    assert len(out) == 1 and "split_from" not in out[0] and not notes


def test_a_p_value_carries_no_direction() -> None:
    """The only kind that cannot be negative, so it can never trigger or block a split."""
    out, notes = _split([_point((0.01, "p-value")), _point((0.03, "p-value"))])
    assert len(out) == 1 and not notes


def test_a_z_statistic_is_directional() -> None:
    """No kind is exempt for want of an observed negative: most tables print |z|."""
    out, _ = _split([_point((2.0, "z-statistic")), _point((-2.0, "z-statistic"))])
    assert len(out) == 2


def test_an_unsignable_row_flags_rather_than_splitting_part_of_the_table() -> None:
    """A partial partition files some rows and strands the rest, which is worse than none."""
    out, notes = _split(
        [_point((3.0, "t-statistic")), _point((-3.0, "t-statistic")), _point((0.01, "p-value"))]
    )
    assert len(out) == 1
    assert notes and notes[0].startswith("FLAG")


def test_statistics_disagreeing_within_one_row_give_no_sign() -> None:
    from pondie.extraction.sign_split import _point_sign

    assert _point_sign(_point((3.0, "t-statistic"), (-0.4, "correlation"))) is None
    assert _point_sign(_point((0.0, "t-statistic"))) is None


def test_splitting_twice_changes_nothing() -> None:
    """Re-running over a corpus already split must be a no-op, since `--resplit` is rerun."""
    from pondie.extraction.sign_split import split_opposite_signs

    once, _ = _split([_point((1.0, "beta")), _point((-1.0, "beta"))])
    split = split_opposite_signs(once)
    twice, notes = list(split.analyses), list(split.notes)
    assert [a["name"] for a in twice] == [a["name"] for a in once]
    assert not notes
