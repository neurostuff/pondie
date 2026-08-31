"""What the direction derivation must not get wrong.

Two rules, and they are not the same rule. `polarity` reads a contrast's own name and
gives each named level a direction -- worth 17% coverage at 98% accuracy against the
reviewed gold, and in every case where it disagreed with the extraction pass the pass had
said `absent` and the rule was right. `mirror_analysis` rebuilds the half of a sign-split
contrast that the paper never describes, by reversing the half it does.

Neither may guess. A cell the contrast name does not mention, and a direction with no
opposite, must come back unchanged rather than plausible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

import derive_direction as dd  # noqa: E402
from parse_tables import split_opposite_signs  # noqa: E402


# --- reading a contrast's own name ------------------------------------------

@pytest.mark.parametrize("contrast,level,expected", [
    ("FESZ>NC", "FESZ", "positive"),
    ("FESZ>NC", "NC", "negative"),
    ("AD < HC reduced GM volume", "AD", "negative"),
    ("AD < HC reduced GM volume", "HC", "positive"),
    ("greater activation in patients than controls", "patients", "positive"),
    ("greater activation in patients than controls", "controls", "negative"),
])
def test_polarity_directs_each_named_level(contrast, level, expected):
    assert dd.direction_of(level, contrast) == expected


def test_a_level_the_contrast_does_not_name_gets_no_answer():
    # A cell the name does not mention is one the model still has to be asked about.
    assert dd.direction_of("age", "FESZ > NC") is None


def test_the_comparison_stops_at_the_clause_boundary():
    # "7d > 28d . ALFF differences in the CCD group" ran the right-hand side to the end
    # of the definition and directed a level the contrast holds constant.
    contrast = "7d > 28d . ALFF differences in the CCD group, 7 days after surgery."
    assert dd.direction_of("CCD", contrast) is None
    assert dd.direction_of("7d", contrast) == "positive"


def test_levels_are_matched_by_word_set_and_never_by_similarity():
    # `men` is a substring of `women`; `synchronous` scores 0.96 against `asynchronous`.
    assert not dd.same_level("men", "women")
    assert not dd.same_level("synchronous", "asynchronous")
    assert dd.same_level("ASD", "ASD children")


def test_a_contrast_that_is_not_a_comparison_yields_nothing():
    assert dd.polarity("Mean dwell time of connectivity states") is None


# --- mirroring a withheld half ----------------------------------------------

def test_only_the_positive_half_is_offered_for_extraction():
    analyses = [{"name": "FESZ > NC", "points": [
        {"values": [{"kind": "t", "value": 3.1}]},
        {"values": [{"kind": "t", "value": -2.9}]}]}]
    out, notes = split_opposite_signs(analyses)
    described = [a for a in out if not a.get("withhold")]
    withheld = [a for a in out if a.get("withhold")]
    assert len(described) == 1 and len(withheld) == 1
    # The described half keeps the paper's own name; renaming it would break the
    # instruction to quote the parsed name verbatim.
    assert described[0]["name"] == "FESZ > NC"
    assert withheld[0]["mirror_of"] == "FESZ > NC"
    assert notes and "withheld" in notes[0]


def test_each_half_keeps_only_its_own_rows():
    analyses = [{"name": "A > B", "points": [
        {"values": [{"kind": "t", "value": 3.1}]},
        {"values": [{"kind": "t", "value": 2.2}]},
        {"values": [{"kind": "t", "value": -2.9}]}]}]
    out, _ = split_opposite_signs(analyses)
    described = next(a for a in out if not a.get("withhold"))
    withheld = next(a for a in out if a.get("withhold"))
    assert len(described["points"]) == 2
    assert len(withheld["points"]) == 1


def test_mirroring_flips_directions_and_addresses_the_withheld_rows():
    described = {"local_id": "analysis_01", "effect": {"cells": [
        {"level": {"value": "FESZ"}, "direction": {"value": "positive",
                                                   "value_source": "reported"}},
        {"level": {"value": "NC"}, "direction": {"value": "negative",
                                                 "value_source": "reported"}}]}}
    withheld = {"points": [{"values": [{"kind": "t", "value": -2.9}]}]}
    mirrored = dd.mirror_analysis(described, withheld, "t3#2")
    got = [c["direction"]["value"] for c in mirrored["effect"]["cells"]]
    assert got == ["negative", "positive"]
    assert mirrored["mirror_of"] == "analysis_01"
    # The rows live in the parse, addressed by the withheld entry's key. Nothing about the
    # coordinates changes between the two readings of one contrast; what changes is the
    # polarity, and that is on the cells.
    assert mirrored["source_table_analysis"]["value"] == "t3#2"
    assert "points" not in mirrored
    assert withheld["points"][0]["values"][0]["value"] == -2.9, "mutated its input"


def test_the_mirror_is_named_for_the_half_it_holds_not_the_half_it_came_from():
    described = {"local_id": "a1",
                 "name": {"extraction_status": "extracted", "value": "FESZ > NC"},
                 "effect": {"cells": [{"level": {"value": "FESZ"},
                                       "direction": {"value": "positive"}}]}}
    withheld = {"name": "FESZ > NC (reversed)", "points": []}
    mirrored = dd.mirror_analysis(described, withheld, "t3#2")
    # The cells say NC > FESZ, so a name saying "FESZ > NC" would contradict them -- and
    # would collide with the real analysis of that name on the same table.
    assert mirrored["name"]["value"] == "FESZ > NC (reversed)"
    assert mirrored["name"]["value"] != described["name"]["value"]
    assert mirrored["name"]["value_source"] == "generated"
    assert mirrored["name"]["evidence"]["status"] == "not_applicable"


def test_a_mirror_with_no_parse_label_keeps_the_name_it_was_copied_from():
    described = {"local_id": "a1",
                 "name": {"extraction_status": "extracted", "value": "A > B"},
                 "effect": {"cells": []}}
    # Nothing better is available, and inventing one would assert a contrast the paper
    # never wrote. The collision is reported by the validator rather than papered over.
    mirrored = dd.mirror_analysis(described, {"points": []})
    assert mirrored["name"]["value"] == "A > B"


def test_mirroring_does_not_mutate_the_analysis_it_was_built_from():
    described = {"local_id": "a", "effect": {"cells": [
        {"level": {"value": "X"}, "direction": {"value": "positive"}}]}}
    dd.mirror_analysis(described, {"points": []})
    assert described["effect"]["cells"][0]["direction"]["value"] == "positive"


@pytest.mark.parametrize("direction", ["undirected", "held", "absent"])
def test_a_direction_with_no_opposite_survives_the_mirror(direction):
    # A level held constant is held from either side of the contrast, and an undirected
    # effect has no sign to flip. Reversing them would invent a claim.
    described = {"local_id": "a", "effect": {"cells": [
        {"level": {"value": "X"}, "direction": {"value": direction,
                                                "value_source": "reported"}}]}}
    cell = dd.mirror_analysis(described, {"points": []})["effect"]["cells"][0]
    assert cell["direction"]["value"] == direction
    # It kept the warrant it was read from; only a flipped direction is generated.
    assert cell["direction"]["value_source"] == "reported"


def test_a_flipped_direction_is_marked_generated():
    described = {"local_id": "a", "effect": {"cells": [
        {"level": {"value": "X"}, "direction": {"value": "positive",
                                                "value_source": "reported"}}]}}
    cell = dd.mirror_analysis(described, {"points": []})["effect"]["cells"][0]
    assert cell["direction"]["value_source"] == "generated"


# --- wired into the build ---------------------------------------------------

import build_record  # noqa: E402


def _cell(level, direction):
    return {"level": {"extraction_status": "extracted", "value": level},
            "direction": {"extraction_status": "extracted", "value": direction}}


def test_the_build_fills_only_the_cells_the_model_gave_up_on():
    body = {"analyses": [{
        "local_id": "a1",
        "name": {"extraction_status": "extracted", "value": "FESZ > NC"},
        "effect": {"cells": [_cell("FESZ", "absent"), _cell("NC", "positive")]}}]}
    filled = build_record.fill_directions(body)
    cells = body["analyses"][0]["effect"]["cells"]
    assert cells[0]["direction"]["value"] == "positive"
    assert cells[0]["direction"]["value_source"] == "generated"
    # A rule that answers a sixth of the cells has no standing to overturn the pass on
    # one it committed to -- even when the rule disagrees.
    assert cells[1]["direction"]["value"] == "positive"
    assert cells[1]["direction"].get("value_source") != "generated"
    assert len(filled) == 1


def test_the_build_leaves_a_level_the_contrast_does_not_name():
    body = {"analyses": [{
        "local_id": "a1",
        "name": {"extraction_status": "extracted", "value": "FESZ > NC"},
        "effect": {"cells": [_cell("age", "absent")]}}]}
    assert build_record.fill_directions(body) == []
    assert body["analyses"][0]["effect"]["cells"][0]["direction"]["value"] == "absent"


def test_the_mirror_is_built_from_the_corrected_record(tmp_path):
    stage1 = tmp_path / "analyses.json"
    stage1.write_text(json.dumps({"analyses": [
        {"name": "A > B", "split_from": "A > B"},
        {"name": "A > B (reversed)", "mirror_of": "A > B", "withhold": True,
         "coordinates": [{"statistic_value": -3.4, "statistic_type": "T"}]}]}))
    body = {"analyses": [{
        "local_id": "a1",
        "name": {"extraction_status": "extracted", "value": "A > B"},
        "effect": {"cells": [_cell("A", "positive"), _cell("B", "negative")]}}]}
    made = build_record.mirror_withheld(body, stage1)
    assert len(made) == 1 and len(body["analyses"]) == 2
    mirrored = body["analyses"][1]
    assert [c["direction"]["value"] for c in mirrored["effect"]["cells"]] == \
        ["negative", "positive"]
    assert mirrored["mirror_of"] == "a1"
    # The mirror reaches its rows by the withheld entry's parse key, like every other
    # analysis. Carrying them inline would put an attribute on Analysis that no class
    # declares, because the schema stores no coordinates at all.
    assert mirrored["source_table_analysis"]["value"] == "#2"
    assert "points" not in mirrored and "coordinates" not in mirrored


def test_a_withheld_half_whose_partner_vanished_is_reported_not_invented(tmp_path):
    stage1 = tmp_path / "analyses.json"
    stage1.write_text(json.dumps({"analyses": [
        {"name": "A > B (reversed)", "mirror_of": "A > B", "withhold": True}]}))
    body = {"analyses": []}
    made = build_record.mirror_withheld(body, stage1)
    assert body["analyses"] == []
    assert made and made[0].startswith("MISSING")
