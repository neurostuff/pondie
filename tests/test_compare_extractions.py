"""What has to hold for a comparison score to mean anything.

Every test here perturbs the gold record in one known way and asserts the metric that
should move -- and, just as importantly, the ones that should not. A scorer that reports
0.9 for everything passes no test in this file.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

import pytest

ROOT = Path(__file__).resolve().parent.parent

from pondie.benchmark import compare_extractions as ce  # noqa: E402

GOLD = ROOT / "benchmarks" / "gold" / "xevP8UDRAVh9.extraction.json"
FLIP = {"positive": "negative", "negative": "positive"}


@pytest.fixture(scope="module")
def schema():
    return ce.Schema()


@pytest.fixture(scope="module")
def gold():
    return json.loads(GOLD.read_text(encoding="utf-8"))


def run(gold_doc, candidate, schema):
    return ce.compare(gold_doc, candidate, schema, ce.Semantics(False), "test")


def rename_ids(record):
    """Rewrite every local_id, leaving the references that point at them consistent."""

    blob = json.dumps(record)
    ids = set(re.findall(r'"local_id": "([^"]+)"', blob)) - {record["local_id"]}
    for name in sorted(ids, key=len, reverse=True):
        blob = blob.replace(f'"{name}"', '"z%s"' % hashlib.md5(name.encode()).hexdigest()[:8])
    return json.loads(blob)


def test_identity_is_perfect(gold, schema):
    result = run(gold, json.loads(json.dumps(gold)), schema)
    assert result["entities"]["micro"]["f1"] == 1.0
    assert result["relationships"]["micro"]["f1"] == 1.0
    assert result["fields"]["overall"]["value_accuracy"] == 1.0
    primary = result["direction"]["primary"]
    assert primary["accuracy_term_grounded"] == 1.0
    assert primary["cell_prf"]["f1"] == 1.0
    assert primary["contrast"]["counts"] == {"exact": len(gold["analyses"])}
    assert result["composite"]["score"] == 1.0


def test_renaming_every_identifier_changes_nothing(gold, schema):
    """The point of matching rather than joining: two extractors never agree on names."""

    result = run(gold, rename_ids(json.loads(json.dumps(gold))), schema)
    assert result["entities"]["micro"]["f1"] == 1.0
    assert result["relationships"]["micro"]["f1"] == 1.0
    assert result["direction"]["primary"]["accuracy_term_grounded"] == 1.0


def test_flipped_signs_are_reported_as_reversed(gold, schema):
    candidate = json.loads(json.dumps(gold))
    target = candidate["analyses"][4]
    for cell in target["effect"]["cells"]:
        if cell["direction"].get("value") in FLIP:
            cell["direction"]["value"] = FLIP[cell["direction"]["value"]]

    primary = run(gold, candidate, schema)["direction"]["primary"]
    assert primary["contrast"]["counts"].get("reversed") == 1
    assert primary["sign_flip_rate"] > 0
    assert primary["accuracy_term_grounded"] < 1.0
    # A flipped sign is a wrong direction, not a missing object or a broken link.
    assert primary["cells"]["term_grounded"] == primary["cells"]["gold"]


def test_a_right_sign_on_the_wrong_term_earns_nothing(gold, schema):
    """The grounding requirement, which is the whole reason direction is scored this way."""

    candidate = json.loads(json.dumps(gold))
    for analysis in candidate["analyses"]:
        cells = analysis["effect"]["cells"]
        if len(cells) == 2:
            cells[0]["term"], cells[1]["term"] = cells[1]["term"], cells[0]["term"]

    primary = run(gold, candidate, schema)["direction"]["primary"]
    assert primary["cell_prf"]["f1"] < 1.0
    assert primary["cells"]["term_grounded"] < primary["cells"]["aligned"]


def test_a_cell_missing_its_level_still_aligns_on_its_term(gold, schema):
    """A missing `level` is a field error, not grounds for refusing to score the cell.

    If it prevented alignment, the cell would vanish from the direction metrics instead
    of being counted -- and a pipeline could improve its apparent accuracy by omitting
    the levels of the cells it was going to get wrong.
    """

    candidate = json.loads(json.dumps(gold))
    stripped = 0
    for analysis in candidate["analyses"]:
        for cell in analysis["effect"]["cells"]:
            if cell.pop("level", None) is not None:
                stripped += 1
    assert stripped, "fixture has no levelled cells to strip"

    primary = run(gold, candidate, schema)["direction"]["primary"]
    assert primary["cells"]["term_grounded"] == primary["cells"]["gold"]
    assert primary["accuracy_term_grounded"] == 1.0


def test_dropping_cells_costs_recall_not_just_accuracy(gold, schema):
    """Accuracy alone would let an extractor win by emitting only the easy cell."""

    candidate = json.loads(json.dumps(gold))
    for analysis in candidate["analyses"]:
        analysis["effect"]["cells"] = analysis["effect"]["cells"][:1]

    primary = run(gold, candidate, schema)["direction"]["primary"]
    assert primary["accuracy_term_grounded"] == 1.0
    assert primary["cell_prf"]["recall"] < 1.0
    assert primary["cell_prf"]["f1"] < 1.0


def test_a_deleted_entity_is_a_recall_miss_and_takes_its_edges_with_it(gold, schema):
    candidate = json.loads(json.dumps(gold))
    dropped = candidate["regions"].pop()["local_id"]

    result = run(gold, candidate, schema)
    region = result["entities"]["per_type"]["Region"]
    assert region["recall"] < 1.0
    assert dropped in region["missed"]
    assert any(edge[2] == dropped for edge in result["relationships"]["false_negatives"])


def test_an_invented_entity_is_a_precision_hit(gold, schema):
    candidate = json.loads(json.dumps(gold))
    candidate["regions"].append({
        "local_id": "region_invented",
        "name": {"extraction_status": "extracted", "value": "nucleus of nowhere"},
    })

    entities = run(gold, candidate, schema)["entities"]["per_type"]["Region"]
    assert entities["precision"] < 1.0
    assert entities["spurious"] == ["region_invented"]


def test_a_reference_to_an_unmatched_entity_can_never_be_a_true_positive(gold, schema):
    candidate = json.loads(json.dumps(gold))
    candidate["analyses"][0]["regions"] = ["region_that_does_not_exist"]

    rel = run(gold, candidate, schema)["relationships"]
    assert rel["unmatched_endpoint_edges"] >= 1
    assert any(t.startswith("?") for _, _, t in rel["false_positives"])


def test_edges_out_of_a_hallucinated_entity_are_false_positives_too(gold, schema):
    """Precision and recall have to describe the same graph difference."""

    candidate = json.loads(json.dumps(gold))
    # A seventh analysis has nothing left to pair with: the six copies take the six gold ones.
    candidate["analyses"].append({
        "local_id": "analysis_invented",
        "name": {"extraction_status": "extracted", "value": "an analysis nobody ran"},
        "measure": "measure_cbf",
        "effect": {"cells": [], "statistic": {}},
    })

    rel = run(gold, candidate, schema)["relationships"]
    assert ("?analysis_invented", "measure", "measure_cbf") in rel["false_positives"]


def test_numbers_are_coerced_before_they_are_compared(gold, schema):
    candidate = json.loads(json.dumps(gold))
    candidate["groups"][0]["age_mean"]["value"] = "40.7"
    candidate["acquisitions"][0]["magnetic_field_strength_tesla"]["value"] = 3.0

    fields = run(gold, candidate, schema)["fields"]["overall"]
    assert fields["value_accuracy"] == 1.0
    assert fields["numeric"]["mae"] == 0.0


def test_a_number_outside_tolerance_is_wrong_and_measured(gold, schema):
    candidate = json.loads(json.dumps(gold))
    candidate["groups"][0]["age_mean"]["value"] = 44.0

    numeric = run(gold, candidate, schema)["fields"]["per_type"]["Group"]["numeric"]
    assert numeric["within_tolerance"] < 1.0
    assert numeric["mae"] == pytest.approx(3.3 / numeric["n"], abs=1e-6)
    assert numeric["bias"] > 0


def test_missingness_is_scored_apart_from_value(gold, schema):
    """`not_reported` is a claim about the paper, and getting it wrong is its own defect."""

    candidate = json.loads(json.dumps(gold))
    group = candidate["groups"][0]
    group["age_mean"] = {"extraction_status": "not_reported"}
    group["age_minimum"] = {"extraction_status": "extracted", "value": 19}

    stats = run(gold, candidate, schema)["fields"]["per_type"]["Group"]
    assert stats["presence"]["fn"] >= 1  # a value the paper had and the candidate dropped
    assert stats["presence"]["fp"] >= 1  # a value the candidate supplied from nowhere
    assert stats["presence"]["f1"] < 1.0


def test_entities_match_across_different_names(schema):
    """`dev_verio` and `dev_magnetom_verio` are the same scanner."""

    gold_doc = {
        "local_id": "p", "devices": [{
            "local_id": "dev_verio",
            "manufacturer": {"extraction_status": "extracted", "value": "Siemens Healthcare"},
            "model": {"extraction_status": "extracted", "value": "Magnetom Verio"}}]}
    candidate = {
        "local_id": "p", "devices": [{
            "local_id": "scanner_1",
            "manufacturer": {"extraction_status": "extracted", "value": "Siemens"},
            "model": {"extraction_status": "extracted", "value": "MAGNETOM Verio 3T"}}]}

    result = run(gold_doc, candidate, schema)
    assert result["entities"]["per_type"]["Device"]["f1"] == 1.0


def test_subclass_mismatch_is_a_wrong_field_not_an_unmatched_object(gold, schema):
    candidate = json.loads(json.dumps(gold))
    candidate["analyses"][0]["details"]["details_type"] = "OtherAnalysisDetails"

    result = run(gold, candidate, schema)
    assert result["entities"]["per_type"]["Analysis"]["f1"] == 1.0
    wrong = [w for row in result["fields"]["per_entity"] for w in row["wrong"]]
    assert any(w["path"] == "details.details_type" for w in wrong)


# -- the matcher ------------------------------------------------------------

def test_a_term_is_matched_by_who_references_it_not_only_by_its_name(gold, schema):
    """The failure this matcher was built for.

    A continuous ModelTerm declares no levels and so makes no references at all. Matched on
    attributes alone it is matched on its name, and an extractor that calls the same term
    "cerebral perfusion" where gold says "perfusion condition" loses every cell hanging off
    it -- silently, as unaligned rather than as wrong.
    """

    candidate = json.loads(json.dumps(gold))
    for model in candidate["model_estimations"]:
        for term in model.get("terms") or []:
            if term["local_id"] == "term_perfusion_condition":
                term["name"]["value"] = "cerebral perfusion"

    result = run(gold, candidate, schema)
    matched = [r for r in result["structure"]["per_entity"]
               if r["gold_id"] == "term_perfusion_condition"]
    assert matched, "the renamed term was not matched at all"
    assert matched[0]["evidence"]["incoming"] > 0.9
    assert result["direction"]["primary"]["cell_prf"]["f1"] == 1.0


def test_incoming_references_include_those_held_by_inline_objects(gold, schema):
    """`Cell.term` and `AnalysisGroup.group` are the references that identify a term."""

    record = ce.flatten(gold, schema, "g")
    sources = {source for source, path in record.incoming["term_gray_matter_volume"]}
    assert sources, "no incoming references found for a term every cell names"
    assert all(record.entities[s].etype == "Analysis" for s in sources)
    assert any(path.startswith("effect.cells[]")
               for _, path in record.incoming["term_gray_matter_volume"])


def test_an_object_wired_into_the_wrong_place_is_reported_as_misplaced(gold, schema):
    """Right attributes, wrong neighbourhood -- the failure a flat edge count averages out."""

    candidate = json.loads(json.dumps(gold))
    for analysis in candidate["analyses"]:
        if analysis.get("measure") == "measure_cbf":
            analysis["measure"] = "measure_gray_matter_volume"

    result = run(gold, candidate, schema)
    misplaced = {r["gold_id"] for r in result["structure"]["misplaced"]}
    assert "measure_cbf" in misplaced
    assert result["structure"]["mean_neighbourhood_f1"] < 1.0


def test_containment_separates_terms_of_different_models(gold, schema):
    record = ce.flatten(gold, schema, "g")
    parents = {t: record.entities[t].parent
               for t in ("term_perfusion_condition", "term_treatment_condition")}
    assert parents["term_perfusion_condition"] == "model_bpm_correlation"
    assert parents["term_treatment_condition"] == "model_glm_paired"


def test_discriminative_weight_is_zero_for_a_field_every_instance_agrees_on(gold, schema):
    record = ce.flatten(gold, schema, "g")
    weights = ce.discriminative_weights(record)
    # Both gold regions are atlas-defined anatomical regions, and both have distinct names.
    assert weights[("Region", "region_type")] == 0.0
    assert weights[("Region", "name")] == 1.0


def test_identical_records_still_align_perfectly_under_the_structural_matcher(gold, schema):
    result = run(gold, json.loads(json.dumps(gold)), schema)
    assert result["structure"]["mean_neighbourhood_f1"] == 1.0
    assert result["structure"]["misplaced"] == []


# -- the primitives ---------------------------------------------------------

def test_hungarian_finds_the_optimum_a_greedy_pass_would_miss():
    # Greedy takes (0,0)=0.9 and is then forced onto (1,1)=0.1, for 1.0; the optimum is 1.6.
    score = [[0.9, 0.8], [0.8, 0.1]]
    assert ce.hungarian(score) == {0: 1, 1: 0}


@pytest.mark.parametrize("rows,cols", [(3, 5), (5, 3), (1, 1), (4, 4)])
def test_hungarian_matches_brute_force(rows, cols):
    from itertools import permutations

    rng = random.Random(rows * 10 + cols)
    score = [[rng.random() for _ in range(cols)] for _ in range(rows)]
    pairs = ce.hungarian(score)
    got = sum(score[r][c] for r, c in pairs.items())

    n = min(rows, cols)
    best = max(
        sum(score[r][c] for r, c in zip(range(rows), perm))
        for perm in permutations(range(cols), n)
    ) if rows <= cols else max(
        sum(score[r][c] for r, c in zip(perm, range(cols)))
        for perm in permutations(range(rows), n)
    )
    assert got == pytest.approx(best)
    assert len(set(pairs.values())) == len(pairs)


def test_fuzzy_carries_an_abbreviation_against_its_expansion():
    assert ce.fuzzy("SCID-II", "Structured Clinical Interview for DSM-IV Axis II (SCID-II)") > 0.6
    assert ce.fuzzy("frontal lobe", "lobe frontal") > 0.9
    assert ce.fuzzy("frontal lobe", "cerebellum") < 0.4


@pytest.mark.parametrize("raw,expected", [
    (2, 2.0), (2.0, 2.0), ("2", 2.0), ("2.5 s", 2.5), ("-0.01", -0.01), (True, None), ("n/a", None),
])
def test_number_coercion(raw, expected):
    assert ce.as_number(raw) == expected


def test_kappa_punishes_a_constant_answer():
    pairs = [("positive", "positive")] * 8 + [("negative", "positive")] * 2
    assert sum(1 for a, b in pairs if a == b) / len(pairs) == 0.8
    assert ce.cohen_kappa(pairs) == 0.0


def test_semantic_similarity_uses_the_vectors_when_it_has_them():
    sem = ce.Semantics(True)
    sem.vectors = {"cerebral blood flow": [1.0, 0.0], "perfusion": [0.96, 0.28]}
    assert sem.similarity("cerebral blood flow", "perfusion") > ce.fuzzy(
        "cerebral blood flow", "perfusion")
    # An unembedded string still gets a score rather than an exception.
    assert 0.0 <= sem.similarity("cerebral blood flow", "unseen text") <= 1.0
