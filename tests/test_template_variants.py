"""Template arms for the D4 residue: what the proposer is shown, and in what shape.

Repair is wrong on most of what it changes, and every one of those errors is a fact the
paper genuinely contains put on the wrong entity or in the wrong slot. The template is one
of the two places that could be responsible -- the other is the model -- so the shape is
made switchable and measured rather than argued about.

`PONDIE_TEMPLATE` and not a `Settings` field: these are arms of an experiment, and the one
that wins should become the only shape rather than a fifth thing to configure.
"""

from __future__ import annotations

import json

import pytest

from pondie import schema
from pondie.extraction import recall
from pondie.schema import reader


@pytest.fixture(scope="module")
def sch():
    return reader.load(schema.EXTRACTION)


@pytest.fixture
def style(monkeypatch):
    def choose(value: str):
        monkeypatch.setenv("PONDIE_TEMPLATE", value)
    return choose


def test_no_style_is_the_shape_every_measurement_was_taken_with(sch, style):
    style("")
    body = recall.template_for(sch, "Group")["groups"][0]
    assert body["enrolled_count"] == "integer"
    assert body["medications"] == ["string"]
    assert recall.styles() == frozenset()


def test_a_quoted_template_asks_for_the_sentence_beside_the_value(sch, style):
    style("quoted")
    body = recall.template_for(sch, "Group")["groups"][0]
    assert body["enrolled_count"] == {"value": "integer", "quote": "verbatim-string"}
    # The multiplicity survives inside the wrapper: a list slot still asks for a list.
    assert body["medications"] == {"value": ["string"], "quote": "verbatim-string"}


def test_a_quoted_reply_splits_into_values_and_citations():
    reply = {"local_id": "grp_a",
             "acquired_count": {"value": 12, "quote": "the final sample consisted of 12"},
             "medications": {"value": ["haloperidol"], "quote": "One patient was excluded"}}
    out = recall.unquote(reply)
    assert out["acquired_count"] == 12
    assert out["medications"] == ["haloperidol"]
    assert out[recall.QUOTES]["acquired_count"] == "the final sample consisted of 12"
    # `_quotes` is not a slot of any class, so `apply` skips it on the way past.
    assert recall.QUOTES.startswith("_")


def test_a_reply_in_the_old_shape_is_still_usable():
    """A model asked for an object sometimes returns the scalar, and a half-migrated reply
    should not be thrown away."""
    out = recall.unquote({"local_id": "grp_a", "acquired_count": 12})
    assert out == {"local_id": "grp_a", "acquired_count": 12}


def test_descriptions_say_what_distinguishes_two_slots_of_one_type(sch):
    """`enrolled_count` and `acquired_count` are both integers on `Group`, and the template
    cannot tell them apart. The schema can, and never showed the model."""
    said = recall.descriptions(sch, "Group")
    assert "before acquisition" in said
    assert "were acquired or who were scanned" in said
    assert "established this group's defining condition" in said


def test_nothing_is_dropped_from_any_class_at_the_default_cap(sch):
    """A smaller cap dropped `diagnostic_instrument` off the end of `Group` -- one of the
    slots this exists for -- in `iter_slots` order."""
    for class_name in sch.classes_by_container().values():
        assert "further fields are not described" not in recall.descriptions(sch, class_name)


def test_a_class_over_the_cap_says_so_rather_than_being_quietly_shortened(sch):
    said = recall.descriptions(sch, "Group", limit=200)
    assert "further fields are not described" in said


def test_the_scoped_rule_names_the_failure_it_exists_for():
    assert "excluded participant" in recall.SCOPED
    assert "subgroup" in recall.SCOPED


def test_every_style_still_projects_a_template_for_every_class(sch, style):
    style("described,quoted,scoped")
    for class_name in sch.classes_by_container().values():
        template = recall.template_for(sch, class_name)
        assert len(template) == 1
        assert json.dumps(template)          # serialisable, which is what the model is sent
