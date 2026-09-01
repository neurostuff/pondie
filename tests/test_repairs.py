"""The deterministic half of the build: repairs, the sign split, and the parse joins.

Nothing here calls a model. What is checked is what the builder does to a payload after the
model has gone: the repair sequence and its ordering constraints, the conversion of a corpus
partitioned by an earlier rule, the fixes added for each validator finding, and the join from
an analysis back to the parsed row group it came from.

The stages themselves are `test_stages.py`; the contracts they pass are `test_models.py`.
"""

from __future__ import annotations

import json

import pytest

from pondie.extraction.record import repairs as repair_module


def test_the_declared_repair_order_holds():
    assert repair_module.check_order(repair_module.build_sequence()) == []


def test_an_order_that_violates_its_own_constraint_is_refused():
    late = repair_module.Repair("a", "", lambda body, ctx: [], after="b")
    early = repair_module.Repair("b", "", lambda body, ctx: [])
    assert repair_module.check_order((late, early))
    with pytest.raises(ValueError):
        repair_module.apply_all({}, repair_module.Context(schema={}), (late, early))


def test_the_log_says_which_repairs_fired():
    sequence = (
        repair_module.Repair("noisy", "", lambda body, ctx: ["did a thing"]),
        repair_module.Repair("quiet", "", lambda body, ctx: []),
    )
    log = repair_module.apply_all({}, repair_module.Context(schema={}), sequence)
    assert log.fired() == ["noisy"]
    assert log.total == 1
    assert "did a thing" in log.explain()


def test_the_mirror_runs_after_the_direction_fill():
    # The mirror copies the corrected contrast; taking it before the fill would copy a
    # cell the builder was about to sign.
    names = [r.name for r in repair_module.build_sequence()]
    assert names.index("mirrored") > names.index("directions")
    assert names.index("directions") > names.index("cell_levels")


# --- adopting a corpus split by the earlier rule -----------------------------

from pondie.extraction.corpus.tables import adopt_withholding  # noqa: E402


def _part(parent, direction, points=1):
    return {
        "name": f"{parent} ({direction})",
        "split_from": parent,
        "split_direction": direction,
        "split_rule": "sign-of-directional-statistic",
        "points": [{"values": [{"kind": "t", "value": 1.0}]}] * points,
    }


def test_an_old_pair_becomes_a_described_half_and_a_withheld_one():
    result = adopt_withholding([_part("A > B", "positive"), _part("A > B", "negative")])
    analyses, converted = list(result.analyses), list(result.notes)
    described = [a for a in analyses if not a.get("withhold")]
    withheld = [a for a in analyses if a.get("withhold")]
    # The described half takes the paper's own name back: the prompt tells the model to
    # quote the parsed name verbatim, and "(positive)" is not what the paper called it.
    assert described[0]["name"] == "A > B"
    assert withheld[0]["mirror_of"] == "A > B"
    assert converted


def test_an_entry_split_on_something_else_as_well_is_left_alone():
    # Three parts sharing a parent means a band or a session was also split on, and
    # which one the paper describes is not answerable from the sign.
    parts = [
        _part("A > B", "positive"),
        _part("A > B", "negative"),
        _part("A > B", "positive"),
    ]
    result = adopt_withholding(parts)
    analyses, converted = list(result.analyses), list(result.notes)
    assert converted == []
    assert not any(a.get("withhold") for a in analyses)


def test_an_unsplit_analysis_is_untouched():
    entry = {"name": "plain", "points": []}
    result = adopt_withholding([entry])
    analyses, converted = list(result.analyses), list(result.notes)
    assert converted == [] and analyses == [entry]


def test_adoption_is_idempotent():
    parts = [_part("A > B", "positive"), _part("A > B", "negative")]
    once = list(adopt_withholding(parts).analyses)
    twice = list(adopt_withholding(once).analyses)
    assert [a["name"] for a in once] == [a["name"] for a in twice]
    assert sum(1 for a in twice if a.get("withhold")) == 1


# --- the deterministic repairs added for validator findings -------------------

from pondie.extraction.record import builder as br


@pytest.fixture(scope="module")
def classes():
    from pondie.extraction.record.builder import EXTRACTION_SCHEMA
    from pondie.schema import reader

    return reader.load(EXTRACTION_SCHEMA)


def test_a_wrapper_in_a_reference_slot_is_unwrapped(classes):
    # The model has just written twenty wrappers and writes a twenty-first into a slot
    # that holds a bare local_id. The wrapper's own value is the answer.
    body = {
        "analyses": [
            {
                "local_id": "a1",
                "model_estimation": {
                    "extraction_status": "extracted",
                    "value": "m1",
                    "evidence": {"status": "present"},
                },
            }
        ]
    }
    changed = br.unwrap_plain_slots(body, classes)
    assert body["analyses"][0]["model_estimation"] == "m1"
    assert changed


def test_a_wrapper_in_an_evidence_slot_is_left_alone(classes):
    # An evidence slot is supposed to hold a wrapper; unwrapping it destroys the value
    # and the span that warrants it.
    body = {
        "groups": [
            {
                "local_id": "g1",
                "name": {
                    "extraction_status": "extracted",
                    "value": "controls",
                    "evidence": {"status": "present"},
                },
            }
        ]
    }
    br.unwrap_plain_slots(body, classes)
    assert isinstance(body["groups"][0]["name"], dict)
    assert body["groups"][0]["name"]["value"] == "controls"


def test_a_numeric_string_becomes_the_number_its_slot_declares(classes):
    body = {
        "acquisitions": [
            {
                "local_id": "acq1",
                "acquisition_duration_seconds": {
                    "extraction_status": "extracted",
                    "value": "252 s",
                    "evidence": {"status": "present"},
                },
            }
        ]
    }
    changed = br.coerce_numeric_values(body, classes)
    assert body["acquisitions"][0]["acquisition_duration_seconds"]["value"] == 252.0
    assert changed


def test_a_value_that_is_not_a_number_is_left_for_the_validator(classes):
    # Inventing a number is worse than reporting a string.
    body = {
        "acquisitions": [
            {
                "local_id": "acq1",
                "acquisition_duration_seconds": {
                    "extraction_status": "extracted",
                    "value": "not stated",
                    "evidence": {"status": "present"},
                },
            }
        ]
    }
    assert br.coerce_numeric_values(body, classes) == []
    assert body["acquisitions"][0]["acquisition_duration_seconds"]["value"] == "not stated"


def test_a_table_written_as_a_study_attribute_is_rehomed(classes):
    # The stray key is dropped on load, so every analysis pointing at it loses the table
    # its coordinates join through and the paper contributes nothing.
    body = {
        "analyses": [{"local_id": "a1", "tables": ["tab4"]}],
        "tab4": {"table_number": {"extraction_status": "extracted", "value": 4}},
    }
    moved = br.rehome_stray_tables(body, classes)
    assert "tab4" not in body
    assert [t["local_id"] for t in body["tables"]] == ["tab4"]
    assert moved


def test_an_unreferenced_stray_key_is_left_reported(classes):
    body = {"analyses": [], "somethingElse": {"x": 1}}
    assert br.rehome_stray_tables(body, classes) == []
    assert "somethingElse" in body


def _scoped(cell_term):
    return {
        "model_estimations": [
            {"local_id": "m_low", "terms": [{"local_id": "t_low", "name": _wrapped("group")}]},
            {
                "local_id": "m_high",
                "inputs_from": ["m_low"],
                "terms": [{"local_id": "t_high", "name": _wrapped("condition")}],
            },
            {
                "local_id": "m_other",
                "terms": [{"local_id": "t_other", "name": _wrapped("group")}],
            },
        ],
        "analyses": [
            {
                "local_id": "a1",
                "model_estimation": "m_high",
                "effect": {"cells": [{"term": cell_term, "level": _wrapped("x")}]},
            }
        ],
    }


def test_a_cell_is_repointed_to_the_same_named_term_in_scope():
    # `t_other` names "group" but sits in a model the analysis does not reach; `t_low`
    # names "group" and is reachable through inputs_from.
    body = _scoped("t_other")
    changed = br.repoint_out_of_scope_terms(body)
    assert body["analyses"][0]["effect"]["cells"][0]["term"] == "t_low"
    assert changed


def test_a_cell_already_in_scope_is_untouched():
    body = _scoped("t_high")
    assert br.repoint_out_of_scope_terms(body) == []
    assert body["analyses"][0]["effect"]["cells"][0]["term"] == "t_high"


def test_no_same_named_term_in_scope_is_left_reported():
    body = _scoped("t_other")
    # Rename the reachable term so nothing in scope matches.
    body["model_estimations"][0]["terms"][0]["name"]["value"] = "timepoint"
    assert br.repoint_out_of_scope_terms(body) == []
    assert body["analyses"][0]["effect"]["cells"][0]["term"] == "t_other"


def test_two_same_named_terms_in_scope_are_not_guessed_between():
    body = _scoped("t_other")
    body["model_estimations"][1]["terms"].append(
        {"local_id": "t_dup", "name": _wrapped("group")}
    )
    assert br.repoint_out_of_scope_terms(body) == []


def test_a_valueless_wrapper_in_a_reference_slot_is_dropped(classes):
    # Every valueless wrapper in the corpus says `not_reported`, which is correct in an
    # evidence slot and meaningless in a reference slot: a reference has no wrapper form,
    # so "not reported" is simply absence.
    body = {
        "model_estimations": [
            {
                "local_id": "m1",
                "terms": [
                    {
                        "local_id": "t1",
                        "name": {"value": "age"},
                        "assessment": {
                            "extraction_status": "not_reported",
                            "evidence": {"status": "not_applicable"},
                        },
                    }
                ],
            }
        ]
    }
    changed = br.unwrap_plain_slots(body, classes)
    assert "assessment" not in body["model_estimations"][0]["terms"][0]
    assert changed and "dropped" in changed[0]


def test_a_bare_scalar_in_an_evidence_slot_is_wrapped(classes):
    body = {
        "analyses": [
            {
                "local_id": "a1",
                "effect": {
                    "cells": [{"term": "t1", "level": {"value": "x"}, "direction": "held"}]
                },
            }
        ]
    }
    changed = br.unwrap_plain_slots(body, classes)
    cell = body["analyses"][0]["effect"]["cells"][0]
    assert cell["direction"]["value"] == "held"
    assert cell["direction"]["extraction_status"] == "extracted"
    # No span was offered, so the evidence is honestly not_found rather than invented.
    assert cell["direction"]["evidence"]["status"] == "not_found"
    assert changed


# --- the link from an analysis to its parsed row group ------------------------


def _wrapped(value):
    """A realistic ExtractedValue. `_is_field` keys on `extraction_status`, so a fixture
    of bare `{"value": ...}` never gets unwrapped and tests nothing."""
    return {
        "extraction_status": "extracted",
        "value": value,
        "value_source": "reported",
        "evidence": {"status": "not_applicable"},
    }


def _stage1(tmp_path, entries):
    path = tmp_path / "analyses.json"
    path.write_text(json.dumps({"analyses": entries}))
    return path


def test_a_present_key_that_names_a_row_group_is_left_alone(tmp_path):
    stage1 = _stage1(
        tmp_path,
        [{"table_id": "t1", "name": "SZ > HC"}, {"table_id": "t1", "name": "HC > SZ"}],
    )
    body = {
        "analyses": [
            {
                "local_id": "a1",
                "tables": ["t1"],
                "name": _wrapped("HC > SZ"),
                "source_table_analysis": _wrapped("t1#2"),
            }
        ]
    }
    assert br.resolve_source_table_analysis(body, stage1) == []
    assert body["analyses"][0]["source_table_analysis"]["value"] == "t1#2"


def test_an_invented_key_is_dropped(tmp_path):
    # A key that resolves to nothing is worse than none: it looks like a working join.
    stage1 = _stage1(tmp_path, [{"table_id": "t1", "name": "SZ > HC"}])
    body = {
        "analyses": [
            {
                "local_id": "a1",
                "tables": ["t1"],
                "name": _wrapped("something else"),
                "source_table_analysis": _wrapped("t9#7"),
            }
        ]
    }
    notes = br.resolve_source_table_analysis(body, stage1)
    assert "source_table_analysis" not in body["analyses"][0]
    assert notes and "names no parsed row group" in notes[0]


def test_a_missing_key_is_filled_from_a_unique_name_match(tmp_path):
    stage1 = _stage1(
        tmp_path,
        [{"table_id": "t1", "name": "SZ > HC"}, {"table_id": "t1", "name": "HC > SZ"}],
    )
    body = {"analyses": [{"local_id": "a1", "tables": ["t1"], "name": _wrapped("HC>SZ")}]}
    notes = br.resolve_source_table_analysis(body, stage1)
    assert body["analyses"][0]["source_table_analysis"]["value"] == "t1#2"
    # Filled, not read off the page, so the provenance says so.
    assert body["analyses"][0]["source_table_analysis"]["value_source"] == "generated"
    assert notes


def test_an_ambiguous_name_leaves_the_analysis_honestly_unjoinable(tmp_path):
    # Two row groups with the same name: the join is a guess, and the slot stays empty.
    stage1 = _stage1(
        tmp_path,
        [{"table_id": "t1", "name": "SZ > HC"}, {"table_id": "t2", "name": "SZ > HC"}],
    )
    body = {
        "analyses": [{"local_id": "a1", "tables": ["t1", "t2"], "name": _wrapped("SZ > HC")}]
    }
    assert br.resolve_source_table_analysis(body, stage1) == []
    assert "source_table_analysis" not in body["analyses"][0]


def test_the_key_is_scoped_to_the_tables_the_analysis_cites(tmp_path):
    stage1 = _stage1(
        tmp_path,
        [{"table_id": "t1", "name": "SZ > HC"}, {"table_id": "t2", "name": "SZ > HC"}],
    )
    body = {"analyses": [{"local_id": "a1", "tables": ["t2"], "name": _wrapped("SZ > HC")}]}
    br.resolve_source_table_analysis(body, stage1)
    assert body["analyses"][0]["source_table_analysis"]["value"] == "t2#1"


def test_no_parse_means_nothing_is_invented(tmp_path):
    body = {"analyses": [{"local_id": "a1", "name": _wrapped("x")}]}
    assert br.resolve_source_table_analysis(body, tmp_path / "missing.json") == []
    assert "source_table_analysis" not in body["analyses"][0]


# --- derived analysis ids -----------------------------------------------------


def test_an_analysis_id_is_derived_from_its_parse_key():
    # A model-chosen id is unstable: over the same 16 papers extracted twice, only four
    # produced identical analysis ids.
    body = {
        "analyses": [
            {
                "local_id": "a_independent_component_spatial_maps",
                "source_table_analysis": _wrapped("t0035#2"),
            }
        ]
    }
    notes = br.derive_analysis_ids(body)
    assert body["analyses"][0]["local_id"] == "a_t0035_2"
    assert notes and "t0035#2" in notes[0]


def test_deriving_ids_is_idempotent():
    body = {"analyses": [{"local_id": "a_x", "source_table_analysis": _wrapped("t1#1")}]}
    br.derive_analysis_ids(body)
    first = body["analyses"][0]["local_id"]
    assert br.derive_analysis_ids(body) == []
    assert body["analyses"][0]["local_id"] == first


def test_analyses_sharing_a_key_are_numbered_apart():
    # A SPLIT emits several analyses against one listing entry, so they share a key.
    body = {
        "analyses": [
            {"local_id": "a1", "source_table_analysis": _wrapped("t1#1")},
            {"local_id": "a2", "source_table_analysis": _wrapped("t1#1")},
        ]
    }
    br.derive_analysis_ids(body)
    assert [a["local_id"] for a in body["analyses"]] == ["a_t1_1", "a_t1_1_2"]


def test_an_analysis_with_no_key_keeps_the_models_id():
    # 25% of analyses cannot be tied to a row group; inventing a stable-looking id for
    # them would claim the parse identifies something it does not.
    body = {"analyses": [{"local_id": "a_hand_named", "name": _wrapped("x")}]}
    assert br.derive_analysis_ids(body) == []
    assert body["analyses"][0]["local_id"] == "a_hand_named"


def test_mirror_of_is_repointed_to_the_new_id():
    # `mirror_of` is the only pointer at an analysis anywhere in the record.
    body = {
        "analyses": [
            {"local_id": "a_described", "source_table_analysis": _wrapped("t1#1")},
            {
                "local_id": "a_described-reversed",
                "mirror_of": "a_described",
                "source_table_analysis": _wrapped("t1#2"),
            },
        ]
    }
    br.derive_analysis_ids(body)
    assert body["analyses"][0]["local_id"] == "a_t1_1"
    assert body["analyses"][1]["mirror_of"] == "a_t1_1"


def test_a_derived_id_already_taken_leaves_both_alone():
    # Collapsing two analyses into one id is worse than an unstable id.
    body = {
        "analyses": [
            {"local_id": "a_t1_1", "name": _wrapped("already answers to it")},
            {"local_id": "a_other", "source_table_analysis": _wrapped("t1#1")},
        ]
    }
    notes = br.derive_analysis_ids(body)
    assert body["analyses"][1]["local_id"] == "a_other"
    assert notes and "already taken" in notes[0]
