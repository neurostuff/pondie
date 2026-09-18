"""The nineteen things a record can be that are structurally legal and scientifically wrong.

`rules.py` is 1,458 lines and had no test file of its own; these were spread through a file
named for the review layer. They are separate from `test_validate_records.py` for the reason
`rules.py` is separate from `validate.py`: that module asks whether a record conforms to
LinkML and knows only the language, while these know what a crossover is, what a product
column means, and why an analysis contrasting two timepoints is not a treatment contrast.

`RULES` carries no ordering constraint -- measured: no rule mutates the record, none reads
validator state, and twelve random orderings give identical finding sets -- so nothing here
depends on running in a particular sequence.

Three cases check the repair side of an invariant next to the report side, because that
boundary is the design: `fix.align_cell_levels` repairs a level that folds to a declared
one, `rules.check_cell_terms` reports one that does not, and testing them apart would hide
which of the two owns a given input.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pondie import schema
from pondie import schema
from pondie.extraction.record import builder, fix, rules
from pondie.extraction.record import validate as validate_record
from pondie.formats import table_parse as tables
from pondie.schema import reader


#: Fixtures rather than a real record, because the defect being checked is one a
#: real record wears invisibly: every reference resolves and every level agrees with
#: its term, so the only way to test it is to build the shape by hand.
def _text(value: str) -> dict:
    return {"extraction_status": "extracted", "value": value}


def _cell(term: str, direction: str, level: str | None = None) -> dict:
    cell = {"term": term, "direction": _text(direction)}
    if level is not None:
        cell["level"] = _text(level)
    return cell


GROUP = {"local_id": "t_group", "type": _text("categorical")}
STAGE = {"local_id": "t_stage", "type": _text("categorical")}
PRODUCT = {
    "local_id": "t_gxs",
    "type": _text("categorical"),
    "interaction_with": ["t_group", "t_stage"],
}

#: The two group cells of an interaction reported as an unsigned chi-square: the test
#: yields no per-level sign, which crosses nothing.
UNSIGNED_GROUP = [
    _cell("t_group", "undirected", "patients"),
    _cell("t_group", "undirected", "controls"),
]


def _record(terms: list[dict], analyses: list[tuple[str, list[dict]]], model: str = "m1") -> dict:
    return {
        "model_estimations": [{"local_id": model, "terms": terms}],
        "analyses": [
            {
                "local_id": f"a{index}",
                "name": _text(name),
                "definition": _text(name),
                "model_estimation": model,
                "effect": {"cells": cells},
            }
            for index, (name, cells) in enumerate(analyses)
        ],
    }


def _flags(record: dict, extraction_schema: dict) -> list[str]:
    validator = validate_record.Validator(extraction_schema, None)
    rules.check_crossings(record, validator)
    rules.check_product_columns(record, validator)
    rules.check_unsigned_cells(record, validator)
    rules.check_occasion_factors(record, validator)
    rules.check_arm_reachability(record, validator)
    rules.check_derived_columns(record, validator)
    assert validator.errors == []  # these checks route to review, never reject
    return validator.warnings


#: A factor whose levels are declared, which is what the held-constant reading is read
#: against: `held` says one of these sat on both sides and the rest were weighted out,
#: so a record celling all of them that way is claiming something else.
LOAD = {
    "local_id": "t_load",
    "type": _text("categorical"),
    "levels": [{"level": _text("high")}, {"level": _text("low")}],
}


#: representing-models.md §5.6: one categorical term whose levels name the occasions.
#: Only the slots these checks read are populated, per the fixture note above.
TIME = {
    "local_id": "t_time",
    "type": _text("categorical"),
    "variation_level": _text("within_subject"),
    "levels": [
        {"level": _text("pre"), "timepoints": ["tp_base"]},
        {"level": _text("post"), "timepoints": ["tp_post"]},
    ],
}

#: TgcHKMRfrVog's defect: the same axis collapsed into one column named for the
#: contrast it was the subject of, so nothing says which occasions were compared.
COLLAPSED = {
    "local_id": "t_prepost",
    "type": _text("continuous"),
    "name": _text("pre > post rsFC change"),
    "variation_level": _text("within_subject"),
}

#: The exception the term half must not flag: one number per participant, named for the
#: subtraction it came from, varying across the sample rather than within anyone. Sourced,
#: so it is correct in every respect but the one under test.
DIFFERENCE_SCORE = {
    "local_id": "t_dbdi",
    "type": _text("continuous"),
    "name": _text("percent change in BDI"),
    "variation_level": _text("between_subject"),
    "source_definition": _text("Percent reduction in BDI, (post-pre)/pre."),
}

TWO_OCCASIONS = {"timepoints": [{"local_id": "tp_base"}, {"local_id": "tp_post"}]}


def _derived(**over) -> dict:
    """A percent-change covariate, complete, with `over` knocking pieces out."""

    term = {
        "local_id": "t_dbdi",
        "type": _text("continuous"),
        "name": _text("percent change in BDI"),
        "variation_level": _text("between_subject"),
        "assessment": "as_bdi",
        "source_definition": _text(
            "Percent reduction in BDI, (post-pre)/pre, from " "baseline to post-treatment."
        ),
    }
    term.update(over)
    return {k: v for k, v in term.items() if v is not None}


def _derived_record(term: dict) -> dict:
    record = _record(
        [term], [("CBT change: rsFC and percent reduction in BDI", [_cell("t_dbdi", "positive")])]
    )
    record["assessments"] = [{"local_id": "as_bdi", "name": _text("Beck Depression Inventory")}]
    return record


#: xevP8UDRAVh9's design: a crossover whose two arms are the whole of what its
#: analyses differ by, so an analysis that cannot reach one states its subject in
#: prose alone.
TWO_ARMS = {
    "arms": [
        {"local_id": "arm_heroin", "name": _text("heroin"), "agent": _text("heroin")},
        {"local_id": "arm_placebo", "name": _text("placebo"), "agent": _text("saline")},
    ]
}

#: The factor a crossover compares its arms with. Its levels are worded as the
#: analysis section words them, which is what a cell has to match.
ARM_FACTOR = {
    "local_id": "t_arm",
    "type": _text("categorical"),
    "levels": [
        {"level": _text("heroin-associated perfusion"), "arms": ["arm_heroin"]},
        {"level": _text("placebo-associated perfusion"), "arms": ["arm_placebo"]},
    ],
}


def _arm_record(analyses: list[tuple[str, list[dict]]], **extra) -> dict:
    record = _record([ARM_FACTOR], analyses)
    record["design"] = TWO_ARMS
    record.update(extra)
    return record


def _rule_errors(node: dict, extraction_schema: dict) -> list[str]:
    validator = validate_record.Validator(extraction_schema, None)
    validator.check_rules(node, "Analysis", "Study.analyses[0]")
    return validator.errors


def _cell_errors(record: dict, extraction_schema: dict) -> list[str]:
    validator = validate_record.Validator(extraction_schema, None)
    rules.check_cell_terms(record, validator)
    return validator.errors


def _levelled(*names: str) -> dict:
    return {
        "local_id": "t_group",
        "type": _text("categorical"),
        "levels": [{"level": _text(name)} for name in names],
    }


def _purpose_flags(record: dict, extraction_schema: dict) -> tuple[list[str], list[str]]:
    validator = validate_record.Validator(extraction_schema, None)
    rules.check_table_purpose(record, validator)
    return validator.errors, validator.warnings


def _vocabulary_flags(
    wrapper: str, value: str, extraction_schema: dict, enums: dict
) -> tuple[list, list]:
    validator = validate_record.Validator(extraction_schema, None, enums)
    validator.check_field(
        {
            "extraction_status": "extracted",
            "value": value,
            "value_source": "generated",
            "evidence": {"status": "not_found"},
        },
        wrapper,
        "Study.somewhere",
    )
    return validator.errors, validator.warnings


def test_reduplication_is_not_a_crossing() -> None:
    assert rules.names_a_crossing(_text("Group-by-stage interaction"))
    assert rules.names_a_crossing(_text("age × diagnosis"))
    assert rules.names_a_crossing(_text("the moderated slope"))
    assert not rules.names_a_crossing(_text("voxel-by-voxel comparison"))
    assert not rules.names_a_crossing(_text("main effect of group"))
    assert not rules.names_a_crossing(None, {"extraction_status": "not_reported"})


def test_interaction_without_a_product_column_is_flagged(extraction_schema: dict) -> None:
    """QQCjAAT6SwwQ's defect: an unsigned interaction test with nowhere to sit."""

    flags = _flags(
        _record([GROUP, STAGE], [("Group-by-stage interaction", UNSIGNED_GROUP)]),
        extraction_schema,
    )

    assert len(flags) == 1
    assert "interaction_with" in flags[0]
    assert "Study.analyses[0].effect.cells" in flags[0]


def test_an_unsigned_cell_on_the_product_column_satisfies_it(extraction_schema: dict) -> None:
    flags = _flags(
        _record(
            [GROUP, STAGE, PRODUCT], [("Group-by-stage interaction", [_cell("t_gxs", "unstated")])]
        ),
        extraction_schema,
    )

    assert flags == []


def test_crossed_levels_need_no_product_column(extraction_schema: dict) -> None:
    """extraction-readme.md's converse: two crossed categorical factors say it themselves."""

    cells = [
        _cell("t_group", "positive", "patients"),
        _cell("t_group", "negative", "controls"),
        _cell("t_stage", "positive", "wake"),
        _cell("t_stage", "negative", "n3"),
    ]

    assert (
        _flags(_record([GROUP, STAGE], [("Group-by-stage interaction", cells)]), extraction_schema)
        == []
    )


def test_a_simple_effect_within_one_level_is_not_flagged(extraction_schema: dict) -> None:
    """representing-models.md §5.5's last row, named after the interaction it came from."""

    cells = UNSIGNED_GROUP + [_cell("t_stage", "held", "wake")]

    assert (
        _flags(
            _record([GROUP, STAGE], [("Group-by-stage interaction at wake", cells)]),
            extraction_schema,
        )
        == []
    )


def test_a_levelless_cell_may_not_be_held(extraction_schema: dict) -> None:
    """§4's first corollary: a product column or a slope has no level, so it has nothing
    to put on both sides of the comparison. An undirected test of one is `undirected`."""

    flags = _flags(
        _record(
            [GROUP, STAGE, PRODUCT], [("Group-by-stage interaction", [_cell("t_gxs", "held")])]
        ),
        extraction_schema,
    )

    assert len(flags) == 1
    assert "names no level" in flags[0]
    assert "undirected" in flags[0]


def test_a_factor_held_at_every_level_is_flagged(extraction_schema: dict) -> None:
    """An omnibus F miscoded. Celling every level `held` says the factor was held on
    both sides of its own test."""

    cells = [_cell("t_load", "held", "high"), _cell("t_load", "held", "low")]

    flags = _flags(_record([LOAD], [("Main effect of load", cells)]), extraction_schema)

    assert len(flags) == 1
    assert "every declared level" in flags[0]


def test_the_same_factor_undirected_at_every_level_is_not(extraction_schema: dict) -> None:
    """Which is the shape that replaced it, and the one §5.8 now writes down."""

    cells = [_cell("t_load", "undirected", "high"), _cell("t_load", "undirected", "low")]

    assert _flags(_record([LOAD], [("Main effect of load", cells)]), extraction_schema) == []


def test_a_held_level_leaves_the_others_absent_and_is_not_flagged(extraction_schema: dict) -> None:
    """The one shape `held` has: one level celled, the rest weighted out."""

    cells = [
        _cell("t_group", "positive", "patients"),
        _cell("t_group", "negative", "controls"),
        _cell("t_load", "held", "high"),
    ]

    assert (
        _flags(_record([GROUP, LOAD], [("Group effect at high load", cells)]), extraction_schema)
        == []
    )


def test_identical_cells_disagreeing_about_a_crossing_is_flagged(extraction_schema: dict) -> None:
    """The visible cost: an interaction and a main effect became the same record."""

    flags = _flags(
        _record(
            [GROUP, STAGE, PRODUCT],
            [
                ("Group-by-stage interaction", UNSIGNED_GROUP),
                ("Group effect", list(UNSIGNED_GROUP)),
            ],
        ),
        extraction_schema,
    )

    # All three fire, which is the raw QQCjAAT6SwwQ record in miniature: the cells
    # record no crossing, the two analyses are therefore indistinguishable, and the
    # column that would have separated them carries nothing.
    assert [flag for flag in flags if "identical to Study.analyses[1]" in flag]
    assert [flag for flag in flags if "interaction_with" in flag]
    assert [flag for flag in flags if "carries no cell" in flag]


def test_a_product_column_may_name_a_lower_stage_term(extraction_schema: dict) -> None:
    """A group stage crossing a cohort factor with a first-level condition."""

    record = {
        "model_estimations": [
            {"local_id": "first", "terms": [{"local_id": "t_cond", "type": _text("categorical")}]},
            {
                "local_id": "group",
                "inputs_from": ["first"],
                "terms": [
                    GROUP,
                    {
                        "local_id": "t_x",
                        "type": _text("categorical"),
                        "interaction_with": ["t_group", "t_cond"],
                    },
                ],
            },
        ],
        "analyses": [
            {
                "local_id": "a0",
                "name": _text("Group × condition"),
                "definition": _text("Group × condition"),
                "model_estimation": "group",
                "effect": {"cells": [_cell("t_x", "positive")]},
            }
        ],
    }

    assert _flags(record, extraction_schema) == []


def test_a_component_in_a_sibling_model_is_flagged(extraction_schema: dict) -> None:
    """What check_local_ids cannot see: the reference resolves, to the wrong model."""

    record = {
        "model_estimations": [
            {"local_id": "other", "terms": [GROUP]},
            {
                "local_id": "mine",
                "terms": [
                    {"local_id": "t_mi", "type": _text("continuous")},
                    {
                        "local_id": "t_x",
                        "type": _text("continuous"),
                        "interaction_with": ["t_group", "t_mi"],
                    },
                ],
            },
        ],
        "analyses": [
            {
                "local_id": "a0",
                "name": _text("Group × MI"),
                "definition": _text("Group × MI"),
                "model_estimation": "mine",
                "effect": {"cells": [_cell("t_x", "positive")]},
            }
        ],
    }

    flags = _flags(record, extraction_schema)

    assert len(flags) == 1
    assert "'t_group' is not a term of 'mine'" in flags[0]


def test_a_product_column_no_cell_names_is_flagged(extraction_schema: dict) -> None:
    """A declared crossing whose analysis was never extracted."""

    flags = _flags(
        _record([GROUP, STAGE, PRODUCT], [("Group effect", UNSIGNED_GROUP)]), extraction_schema
    )

    assert len(flags) == 1
    assert "carries no cell" in flags[0]


def test_the_checks_survive_a_cyclic_stage_chain(extraction_schema: dict) -> None:
    """Invariant 6's violation is an error elsewhere; here it must not hang."""

    record = {
        "model_estimations": [
            {"local_id": "a", "inputs_from": ["b"], "terms": [GROUP]},
            {"local_id": "b", "inputs_from": ["a"], "terms": [STAGE]},
        ],
        "analyses": [],
    }

    assert _flags(record, extraction_schema) == []


def test_names_a_comparison_reads_contrast_syntax() -> None:
    assert rules.names_a_comparison(_text("pre > post rsFC change"))
    assert rules.names_a_comparison(_text("patients versus controls"))
    assert rules.names_a_comparison(_text("faces vs houses"))
    assert rules.names_a_comparison(_text("difference between sessions"))
    # A threshold is not an axis: the operator wants a word character on both sides.
    assert not rules.names_a_comparison(_text("p < .001 uncorrected"))
    assert not rules.names_a_comparison(_text("aSCC seed connectivity"))
    assert not rules.names_a_comparison(_text("age"))
    assert not rules.names_a_comparison(None, {"extraction_status": "not_reported"})


def test_a_contrast_shaped_continuous_term_is_flagged(extraction_schema: dict) -> None:
    """TgcHKMRfrVog's defect: the occasion axis recorded as one continuous column."""

    record = _record(
        [COLLAPSED], [("CBT change: rsFC with aSCC, pre > post", [_cell("t_prepost", "positive")])]
    )

    flags = _flags(record, extraction_schema)

    assert len(flags) == 1
    assert "Study.model_estimations[0].terms[0].name" in flags[0]
    assert "states a comparison" in flags[0]


def test_an_occasion_factor_satisfies_it(extraction_schema: dict) -> None:
    """§5.6's encoding of the same result raises nothing."""

    record = _record(
        [TIME],
        [
            (
                "CBT change: rsFC with aSCC, pre > post",
                [_cell("t_time", "positive", "pre"), _cell("t_time", "negative", "post")],
            )
        ],
    )
    record["design"] = TWO_OCCASIONS

    assert _flags(record, extraction_schema) == []


def test_a_per_participant_difference_score_is_not_flagged(extraction_schema: dict) -> None:
    """ModelTerm.type's stated exception. Its name says `change in` and it is right:
    one number per participant, entered across the sample, is a slope."""

    record = _record(
        [DIFFERENCE_SCORE],
        [("CBT change: rsFC with aSCC, percent reduction in BDI", [_cell("t_dbdi", "positive")])],
    )

    assert _flags(record, extraction_schema) == []


def test_a_product_column_named_for_its_crossing_is_not_flagged(extraction_schema: dict) -> None:
    """A product column has no levels either, and is named for what it multiplies."""

    term = {
        "local_id": "t_x",
        "type": _text("continuous"),
        "name": _text("age × diagnosis"),
        "interaction_with": ["t_group"],
    }
    record = _record([GROUP, term], [("Age × diagnosis", [_cell("t_x", "positive")])])

    assert [
        flag for flag in _flags(record, extraction_schema) if "states a comparison" in flag
    ] == []


def test_declared_occasions_that_no_level_names_are_flagged(extraction_schema: dict) -> None:
    """The defect from the design end: the scans are recorded, the comparison is not."""

    record = _record([GROUP], [("CBT change in rsFC, pre > post", UNSIGNED_GROUP)])
    record["design"] = TWO_OCCASIONS

    flags = _flags(record, extraction_schema)

    assert len(flags) == 1
    assert "Study.design.timepoints" in flags[0]
    assert "the comparison between them is not" in flags[0]


def test_a_baseline_only_record_is_not_flagged(extraction_schema: dict) -> None:
    """A study that scanned twice and reported once is the legitimate reading, which
    is why the trigger needs prose claiming a change and `baseline` is not it."""

    record = _record([GROUP], [("Baseline rsFC with aSCC", UNSIGNED_GROUP)])
    record["design"] = TWO_OCCASIONS

    assert _flags(record, extraction_schema) == []


def test_one_declared_occasion_cannot_be_compared(extraction_schema: dict) -> None:
    """Nothing to flag: a single occasion has no second side to have lost."""

    record = _record([GROUP], [("Change in rsFC after treatment", UNSIGNED_GROUP)])
    record["design"] = {"timepoints": [{"local_id": "tp_base"}]}

    assert _flags(record, extraction_schema) == []


def test_names_a_derivation_reads_construction_not_measurement() -> None:
    assert rules.names_a_derivation(_text("percent change in BDI"))
    assert rules.names_a_derivation(_text("difference in reaction time"))
    assert rules.names_a_derivation(_text("improvement in HDRS"))
    # A measurement that merely contains "percent" is not a construction from several.
    assert not rules.names_a_derivation(
        _text("percentage methylation at CpG sites 11-12 around AKT1 rs1130233")
    )
    # A collapsed occasion factor is check_occasion_factors' finding, not this one.
    assert not rules.names_a_derivation(_text("pre > post rsFC change"))
    assert not rules.names_a_derivation(_text("BDI"))


def test_a_fully_sourced_derived_column_is_not_flagged(extraction_schema: dict) -> None:
    assert _flags(_derived_record(_derived()), extraction_schema) == []


def test_a_derived_column_with_no_derivation_recorded_is_flagged(extraction_schema: dict) -> None:
    """TgcHKMRfrVog's `term_bdi_percent_change`: the occasions it spans are nowhere."""

    flags = _flags(_derived_record(_derived(source_definition=None)), extraction_schema)

    assert len(flags) == 1
    assert "Study.model_estimations[0].terms[0].source_definition" in flags[0]
    assert "derivation is not recorded" in flags[0]


def test_a_derived_column_still_names_its_instrument(extraction_schema: dict) -> None:
    """Deriving a column does not break the link to what supplied it."""

    flags = _flags(_derived_record(_derived(assessment=None)), extraction_schema)

    assert len(flags) == 1
    assert "Study.model_estimations[0].terms[0].assessment" in flags[0]
    assert "names no assessment" in flags[0]


def test_a_derived_column_with_no_assessment_to_name_is_not_flagged(
    extraction_schema: dict,
) -> None:
    """A record declaring no instrument has none for the column to have dropped, so
    the assessment half stays quiet and only the derivation is asked for."""

    record = _record(
        [_derived(assessment=None, source_definition=None)],
        [("Change score", [_cell("t_dbdi", "positive")])],
    )

    flags = _flags(record, extraction_schema)

    assert len(flags) == 1
    assert "derivation is not recorded" in flags[0]


def test_a_factor_over_occasions_is_not_a_derived_column(extraction_schema: dict) -> None:
    """`TIME` compares occasions rather than being computed across them, so it needs
    no source_definition however its levels are labelled."""

    record = _record(
        [TIME],
        [
            (
                "Change in rsFC, pre > post",
                [_cell("t_time", "positive", "pre"), _cell("t_time", "negative", "post")],
            )
        ],
    )
    record["design"] = TWO_OCCASIONS

    assert _flags(record, extraction_schema) == []


def test_an_analysis_naming_an_arm_it_cannot_reach_is_flagged(extraction_schema: dict) -> None:
    """xevP8UDRAVh9's defect: the cell says `heroin`, the level says
    `heroin-associated perfusion`, and the join to the arm breaks on the string."""

    flags = _flags(
        _arm_record(
            [
                (
                    "Positive correlation with heroin-associated perfusion",
                    [_cell("t_arm", "positive", "heroin")],
                )
            ]
        ),
        extraction_schema,
    )

    assert len(flags) == 1
    assert "arm_heroin" in flags[0]


def test_a_cell_reaching_the_level_that_names_the_arm_satisfies_it(
    extraction_schema: dict,
) -> None:
    flags = _flags(
        _arm_record(
            [
                (
                    "Positive correlation with heroin-associated perfusion",
                    [_cell("t_arm", "positive", "heroin-associated perfusion")],
                )
            ]
        ),
        extraction_schema,
    )

    assert flags == []


def test_an_analysed_cohort_assigned_to_the_arm_satisfies_it(extraction_schema: dict) -> None:
    """The parallel-group route: no cell names the arm, but the cohort was assigned
    to it, so `Group.arm` carries what the contrast does not."""

    record = _arm_record(
        [("Perfusion under heroin", [_cell("t_arm", "positive", "heroin")])],
        groups=[{"local_id": "g1", "arm": "arm_heroin"}],
    )
    record["analyses"][0]["groups"] = [{"group": "g1"}]

    assert _flags(record, extraction_schema) == []


def test_an_analysis_naming_no_arm_is_left_alone(extraction_schema: dict) -> None:
    """A baseline contrast in a study that has arms is not about either of them,
    which is what keeps 84rGLhCbUJTh's four pre-medication analyses silent."""

    flags = _flags(
        _arm_record(
            [("Areas of abnormal FA before medication", [_cell("t_arm", "positive", "heroin")])]
        ),
        extraction_schema,
    )

    assert flags == []


def test_a_short_arm_name_does_not_match_everything(extraction_schema: dict) -> None:
    """A two-character arm name would appear inside unrelated prose, so it is not
    vocabulary. The arm is then unreachable in the same way and silently so."""

    record = _arm_record(
        [("Positive correlation in the striatum", [_cell("t_arm", "positive", "heroin")])]
    )
    record["design"] = {"arms": [{"local_id": "arm_iv", "name": _text("IV")}]}

    assert _flags(record, extraction_schema) == []


#: `rules` is dropped by the projection to extraction, so a rule is only ever
#: evaluated against an extraction record. These check that it is evaluated at all:
#: the failure mode is a rule that reads correctly and never fires.
def test_the_storage_rules_are_found(extraction_schema: dict) -> None:
    """The inventory is pinned because `rules` is the one thing the projection drops: a rule
    added to storage and not reaching `check_rules` is a constraint that reads correctly and
    never fires, which is the failure this whole section exists to catch."""

    found = validate_record.storage_rules()
    assert sorted(found) == ["Analysis", "Effect", "InferenceSettings", "Region"]
    assert len(found["Analysis"]) == 2, "the two spatial_scope/regions rules"
    assert len(found["Effect"]) == 1, "cells cannot be empty"


def test_an_effect_with_no_cells_is_rejected(extraction_schema: dict) -> None:
    """`required: true` on `cells` catches an absent key and nothing else -- LinkML has no
    minimum cardinality here, so an effect that compared nothing used to validate."""

    validator = validate_record.Validator(extraction_schema, None)
    validator.check_rules({"cells": []}, "Effect", "Study.analyses[0].effect")
    assert len(validator.errors) == 1 and "cells cannot be empty" in validator.errors[0]

    ok = validate_record.Validator(extraction_schema, None)
    ok.check_rules({"cells": [{"term": "t1"}]}, "Effect", "Study.analyses[0].effect")
    assert ok.errors == []


@pytest.mark.parametrize(
    "scope,regions,fails",
    [
        ("roi", ["r1"], False),
        ("roi", [], True),
        ("roi", None, True),
        ("whole_brain", None, False),
        ("whole_brain", ["r1"], True),
        ("searchlight", ["r1"], True),
        # No rule has `unstated` as a precondition, so neither shape is constrained.
        ("unstated", ["r1"], False),
        ("unstated", None, False),
    ],
)
def test_spatial_scope_and_regions_agree(
    extraction_schema: dict, scope: str, regions: list | None, fails: bool
) -> None:
    node = {"spatial_scope": {"extraction_status": "extracted", "value": scope}}
    if regions is not None:
        node["regions"] = regions

    assert bool(_rule_errors(node, extraction_schema)) is fails


def test_a_rule_construct_the_evaluator_cannot_read_is_reported(
    extraction_schema: dict, monkeypatch
) -> None:
    """Silently skipping one turns the rule into a check that always passes."""

    monkeypatch.setattr(
        validate_record,
        "_RULES",
        {
            "Analysis": [
                {
                    "description": "invented",
                    "preconditions": {
                        "slot_conditions": {"spatial_scope": {"equals_string": "roi"}}
                    },
                    "postconditions": {"slot_conditions": {"regions": {"maximum_cardinality": 3}}},
                }
            ]
        },
    )
    errors = _rule_errors(
        {"spatial_scope": {"extraction_status": "extracted", "value": "roi"}, "regions": ["r1"]},
        extraction_schema,
    )

    assert any("maximum_cardinality" in error and "not implemented" in error for error in errors)


# The join is on the string, and nothing checked it. On the 16-record corpus 55 of 140
# levelled cells named a level their term does not declare, across 8 papers -- and 45 of
# those survived a careful hand review, so this is the class a reader cannot see.
def test_a_cell_level_naming_a_declared_level_is_accepted(extraction_schema: dict) -> None:
    record = _record(
        [_levelled("patients", "controls")], [("dx", [_cell("t_group", "positive", "patients")])]
    )
    assert _cell_errors(record, extraction_schema) == []


def test_a_cell_level_naming_no_declared_level_is_an_error(extraction_schema: dict) -> None:
    """`AD` against a declared `AD group`: the mapper's join finds nothing, and the record
    looks like it recorded which cohort was compared."""

    record = _record(
        [_levelled("AD group", "HC group")], [("dx", [_cell("t_group", "positive", "AD")])]
    )
    errors = _cell_errors(record, extraction_schema)
    assert len(errors) == 1 and "matches none of term" in errors[0]
    assert "'AD group'" in errors[0], "the declared levels are offered, not just refused"


def test_a_cell_naming_a_term_of_another_model_is_an_error(extraction_schema: dict) -> None:
    """Invariant 2. The term exists, so `check_local_ids` is satisfied and the record is
    structurally fine -- it is the *scope* that is wrong, and the message says whose."""

    record = _record(
        [_levelled("patients", "controls")],
        [("dx", [_cell("t_elsewhere", "positive", "patients")])],
    )
    record["model_estimations"].append(
        {"local_id": "m2", "terms": [{"local_id": "t_elsewhere", "type": _text("categorical")}]}
    )
    errors = _cell_errors(record, extraction_schema)
    assert len(errors) == 1 and "'m2'" in errors[0] and "inputs_from" in errors[0]


def test_a_cell_naming_a_term_of_a_lower_stage_is_accepted(extraction_schema: dict) -> None:
    """The converse, and the reason the walk follows `inputs_from`: a group contrast of a
    first-level column is a cell on that stage's term, not a copy hoisted upward."""

    record = _record([], [("dx", [_cell("t_first", "positive", "task")])])
    record["model_estimations"][0]["inputs_from"] = ["m_first"]
    record["model_estimations"].append(
        {
            "local_id": "m_first",
            "terms": [
                {
                    "local_id": "t_first",
                    "type": _text("categorical"),
                    "levels": [{"level": _text("task")}],
                }
            ],
        }
    )
    assert _cell_errors(record, extraction_schema) == []


def test_a_term_naming_nothing_at_all_is_an_error(extraction_schema: dict) -> None:
    record = _record([], [("dx", [_cell("t_missing", "positive", "patients")])])
    errors = _cell_errors(record, extraction_schema)
    assert len(errors) == 1 and "names no ModelTerm anywhere" in errors[0]


def test_a_level_differing_only_in_case_is_repaired_not_reported(extraction_schema: dict) -> None:
    """A transcription slip, not a claim about the paper, so the builder settles it and
    says so -- and `check_cell_terms` then has nothing to report."""

    record = _record(
        [_levelled("healthy controls")],
        [("dx", [_cell("t_group", "positive", "Healthy controls")])],
    )
    fixed = fix.align_cell_levels(record)
    assert len(fixed) == 1 and "'Healthy controls' -> 'healthy controls'" in fixed[0]
    assert record["analyses"][0]["effect"]["cells"][0]["level"]["value"] == "healthy controls"
    assert _cell_errors(record, extraction_schema) == []


def test_a_level_that_merely_shortens_a_declared_one_is_not_repaired(
    extraction_schema: dict,
) -> None:
    """`AD` is not a folding of `AD group`. Shortening a level is a claim, and guessing
    which cohort was meant is the one thing this field must not contain."""

    record = _record(
        [_levelled("AD group", "HC group")], [("dx", [_cell("t_group", "positive", "AD")])]
    )
    assert fix.align_cell_levels(record) == []


def test_an_ambiguous_fold_is_left_alone(extraction_schema: dict) -> None:
    """Two declared levels folding to the same string makes the rewrite a coin toss."""

    record = _record(
        [_levelled("Controls", "controls")], [("dx", [_cell("t_group", "positive", "CONTROLS")])]
    )
    assert fix.align_cell_levels(record) == []


# `Table.purpose` is the only field that can say a table's rows are locations
# rather than findings. Without it, a table deliberately not encoded and a table the
# extraction missed are the same silence -- and `6oTrCJA43Jcd`'s ICA component peaks were
# encoded as an analysis with a fabricated cell rather than left unowned.
def test_a_table_an_analysis_names_needs_no_purpose(extraction_schema: dict) -> None:
    record = {
        "tables": [{"local_id": "tbl1"}],
        "analyses": [{"local_id": "a1", "tables": ["tbl1"]}],
    }
    assert _purpose_flags(record, extraction_schema) == ([], [])


def test_a_table_nobody_names_and_nothing_explains_is_flagged(extraction_schema: dict) -> None:
    """The missed-analysis case, and the one this field exists to separate."""

    record = {"tables": [{"local_id": "tbl4"}], "analyses": []}
    errors, warnings = _purpose_flags(record, extraction_schema)
    assert errors == []
    assert len(warnings) == 1 and "deliberately not encoded or missed" in warnings[0]


def test_a_table_that_says_what_it_reports_is_accepted(extraction_schema: dict) -> None:
    record = {
        "tables": [{"local_id": "tbl4", "purpose": _text("component_peaks")}],
        "analyses": [],
    }
    assert _purpose_flags(record, extraction_schema) == ([], [])


def test_a_table_cannot_both_be_an_analysis_and_not_one(extraction_schema: dict) -> None:
    record = {
        "tables": [{"local_id": "tbl4", "purpose": _text("component_peaks")}],
        "analyses": [{"local_id": "a1", "tables": ["tbl4"]}],
    }
    errors, warnings = _purpose_flags(record, extraction_schema)
    assert warnings == []
    assert len(errors) == 1 and "an analysis names it" in errors[0]


def test_the_purpose_vocabulary_is_open(extraction_schema: dict, enums: dict) -> None:
    """An unanticipated purpose is written down rather than forced into the nearest value,
    which is what `any_of: [TablePurpose, string]` buys."""

    validator = validate_record.Validator(extraction_schema, None, enums)
    validator.check_field(
        {
            "extraction_status": "extracted",
            "value": "a genotyping panel",
            "value_source": "reported",
            "evidence": {"status": "not_found"},
        },
        "ExtractedTablePurpose",
        "Study.tables[0].purpose",
    )
    assert validator.errors == [], "an open vocabulary must not reject a free-text answer"
    assert any("open vocabulary" in w for w in validator.warnings), (
        "and it must still be reported, because off-vocabulary answers accumulating are "
        "the evidence for whether the vocabulary is short a value"
    )


def test_unstated_is_rejected_on_a_closed_vocabulary(extraction_schema: dict, enums: dict) -> None:
    """`Prespecification` is closed, so an off-vocabulary value is already an error. The
    check still has to fire, because the membership error would name the wrong defect."""

    errors, _ = _vocabulary_flags(
        "ExtractedPrespecification", "unstated", extraction_schema, enums
    )
    assert len(errors) == 1 and "not_reported" in errors[0]


def test_unstated_is_rejected_on_an_open_vocabulary(extraction_schema: dict, enums: dict) -> None:
    """The case a membership check cannot catch: an open field keeps a free-text escape
    hatch, so `unstated` would pass with a warning rather than be rejected."""

    errors, warnings = _vocabulary_flags(
        "ExtractedSpatialScope", "unstated", extraction_schema, enums
    )
    assert len(errors) == 1 and "not_reported" in errors[0]
    assert warnings == [], "it is rejected as missingness, not reported as off-vocabulary"


def test_not_reported_is_how_a_silent_source_is_recorded(
    extraction_schema: dict, enums: dict
) -> None:
    """The other half: the encoding the check sends people to has to pass."""

    validator = validate_record.Validator(extraction_schema, None, enums)
    validator.check_field(
        {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}},
        "ExtractedPrespecification",
        "Study.somewhere",
    )
    assert validator.errors == []


def test_a_reported_value_that_is_not_a_sign_still_passes(
    extraction_schema: dict, enums: dict
) -> None:
    """`undirected` is a test that yields no sign, which the source does report, and
    `not_applicable` is a concept that does not apply. Neither is missingness, and folding
    them in would undo the Direction re-cut."""

    for value in ("undirected", "held", "positive"):
        errors, _ = _vocabulary_flags("ExtractedDirection", value, extraction_schema, enums)
        assert errors == [], f"{value} is a reported fact, not a silence"


def test_no_vocabulary_offers_unstated(enums: dict) -> None:
    """The schema half of the rule, mirroring `check_schema.check_no_unstated_member`.

    These descriptions are the extraction prompt, so a vocabulary declaring `unstated`
    is an instruction to produce exactly what the record check rejects.
    """

    live = [name for name, body in enums.items() if "unstated" in (body.permissible_values or {})]
    assert live == [], f"these vocabularies still offer `unstated` as an answer: {live}"


def test_a_cyclic_inputs_from_is_reported_and_not_merely_survived(extraction_schema: dict) -> None:
    """`_terms_in_scope` already guards against the hang. Surviving bad input is not
    reporting it, and a model fitted on its own output is not a stage order."""

    record = {
        "model_estimations": [
            {"local_id": "m1", "inputs_from": ["m2"], "terms": []},
            {"local_id": "m2", "inputs_from": ["m1"], "terms": []},
        ],
        "analyses": [],
    }
    validator = validate_record.Validator(extraction_schema, None)
    rules.check_model_stages(record, validator)
    assert any("cyclic" in error for error in validator.errors)


def test_one_term_name_twice_in_a_stage_chain_is_reported(extraction_schema: dict) -> None:
    """A first-level `motion` and a group-level `motion` are two columns with one name in
    one term list, and a reader cannot tell a refit from a mistake."""

    record = {
        "model_estimations": [
            {
                "local_id": "m_group",
                "inputs_from": ["m_first"],
                "terms": [{"local_id": "t_a", "name": _text("motion")}],
            },
            {"local_id": "m_first", "terms": [{"local_id": "t_b", "name": _text("Motion")}]},
        ],
        "analyses": [],
    }
    validator = validate_record.Validator(extraction_schema, None)
    rules.check_model_stages(record, validator)
    assert any("appears on both" in error for error in validator.errors)


def test_the_same_name_on_one_model_is_not_a_chain_collision(extraction_schema: dict) -> None:
    """The invariant is about a *chain*. Two same-named terms on one record are a different
    fault, and `unique_keys` is what would catch it."""

    record = {
        "model_estimations": [
            {
                "local_id": "m1",
                "terms": [
                    {"local_id": "t_a", "name": _text("motion")},
                    {"local_id": "t_b", "name": _text("motion")},
                ],
            },
        ],
        "analyses": [],
    }
    validator = validate_record.Validator(extraction_schema, None)
    rules.check_model_stages(record, validator)
    assert validator.errors == []


def test_two_protocols_in_one_acquisition_are_reported() -> None:
    """A repetition time belongs to the sequence, not to an echo, so several of both is two
    acquisitions fused into one -- 16701903 acquires MP-RAGE at TE 4.4 ms and FLASH at TE
    5 ms, and `pulse_sequence_type` reads "3D MP-RAGE and 3D FLASH" to match.

    A multi-echo sequence -- several echoes, one TR -- is the case the list exists for and
    must not be reported.
    """

    def field(value: object) -> dict:
        return {
            "extraction_status": "extracted",
            "value": value,
            "value_source": "reported",
            "evidence": {"status": "not_found"},
        }

    class Collected:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def error(self, path: str, message: str) -> None:
            self.messages.append(f"{path}: {message}")

        warn = error

    fused = {
        "acquisitions": [
            {
                "local_id": "acq_mri",
                "echo_time_seconds": field([0.0044, 0.005]),
                "repetition_time_seconds": field([0.0114, 0.015]),
                "pulse_sequence_type": field("3D MP-RAGE and 3D FLASH"),
            }
        ]
    }
    found = Collected()
    rules.check_one_protocol_per_acquisition(fused, found)
    assert found.messages and "two protocols" in found.messages[0]

    multi_echo = {
        "acquisitions": [
            {
                "local_id": "acq_me",
                "echo_time_seconds": field([0.012, 0.028, 0.045]),
                "repetition_time_seconds": field([2.0]),
            }
        ]
    }
    quiet = Collected()
    rules.check_one_protocol_per_acquisition(multi_echo, quiet)
    assert quiet.messages == []


def test_a_stray_token_in_an_entity_list_does_not_lose_the_paper(tmp_path: Path) -> None:
    """16023086's `analyses` came back as [{...}, "required_entities"]. Every walker
    downstream assumes an entity list holds objects, so `derive_coordinate_spaces` called
    `.get` on the string and the build raised -- discarding a paper that had already paid
    for all six stages over one stray token in one reply."""
    (tmp_path / "agent.json").write_text(
        json.dumps({"analyses": [{"local_id": "an_1"}, "required_entities"]}),
        encoding="utf-8",
    )

    body, notes = builder.merge_payloads(tmp_path)

    assert [a["local_id"] for a in body["analyses"]] == ["an_1"]
    assert any("dropped 1 non-object" in n for n in notes), notes
