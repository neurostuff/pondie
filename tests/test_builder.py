"""What `build` promises: the same payloads give the same record, and offsets survive.

The two `merge_payloads` cases were filed under the sign-split banner, which is where they
had drifted rather than where they belong -- they are about what the builder reports when a
payload holds something that is not record content.
"""

from __future__ import annotations

import json

import pytest

from pondie.extraction.record import builder

from conftest import PAPER, PAYLOADS, TEXT, requires_paper
from pathlib import Path


@requires_paper
def test_build_is_reproducible_and_gated_on_offsets() -> None:
    """Rebuilding from the same payloads yields the same record, every span verified.

    `build` verifies every span it emits before returning, so reaching the assertions
    at all is most of the guarantee. What it does NOT promise is that a given paper's
    payloads resolve completely: a quote the extractor paraphrased is reported as a
    failure and left out, and the record beside it carries a hand correction for
    exactly those. That is a fact about the payloads, not about the builder, so it is
    checked as "reported with a reason" rather than as "never happens".
    """

    if not PAYLOADS.is_dir():
        pytest.skip("payloads are not present")

    first, report = builder.build(
        PAPER, TEXT, PAYLOADS, "test-model", "test-version", "2026-08-02"
    )
    second, _ = builder.build(PAPER, TEXT, PAYLOADS, "test-model", "test-version", "2026-08-02")
    assert first == second
    assert report.warrant.exact + report.warrant.whitespace_tolerant > 0
    for failure in report.warrant.unresolved:
        assert ":" in failure and "quote" in failure, failure


@requires_paper
def test_aliases_only_rewrite_reference_slots(extraction_schema: dict) -> None:
    """An alias must never touch an extracted value that shares a string with an id."""

    body = {
        "analyses": [
            {
                "local_id": "a1",
                "model_estimation": "old_id",
                "name": {
                    "extraction_status": "extracted",
                    "value": "old_id",
                    "evidence": {"status": "not_found"},
                },
            }
        ]
    }
    rewrites = builder.apply_aliases(body, extraction_schema, {"old_id": "new_id"})

    assert rewrites == 1
    assert body["analyses"][0]["model_estimation"] == "new_id"
    assert body["analyses"][0]["name"]["value"] == "old_id"


def test_a_pass_output_that_is_not_record_content_is_not_reported_as_a_fault(tmp_path):
    """`required_entities` is the demands pass's shopping list, read by the satisfy pass.

    Reported as an unexpected key it fired on every paper, and buried the one note that says
    an entity list was silently dropped -- the failure that lost `arms` and `timepoints` from
    every intervention and longitudinal paper.
    """
    (tmp_path / "demands.json").write_text(
        json.dumps({"analyses": [], "required_entities": [{"local_id": "g1"}]})
    )
    _body, notes = builder.merge_payloads(tmp_path)
    assert notes == []


def test_a_dropped_entity_list_is_still_reported(tmp_path):
    (tmp_path / "satisfy.json").write_text(json.dumps({"widgets": [{"local_id": "w1"}]}))
    _body, notes = builder.merge_payloads(tmp_path)
    assert any("widgets" in note for note in notes)


# -- which payload keys survive the merge ----------------------------------
#
# `merge_payloads` accepts the keys `schema.entity_lists()` derives, which is why `arms` and
# `timepoints` appearing in the schema did not need a code change here. A hardcoded list is
# what lost them once.


def test_merge_payloads_keeps_arms_and_timepoints(tmp_path: Path) -> None:
    """The two lists the hardcoded mapping missed, end to end through the merge.

    They now live under Study.design rather than on Study, so this also covers the
    payload key staying flat while the path it lands at does not.
    """

    (tmp_path / "trial.json").write_text(
        json.dumps(
            {
                "arms": [{"local_id": "active", "name": {"extraction_status": "extracted"}}],
                "timepoints": [
                    {"local_id": "baseline", "name": {"extraction_status": "extracted"}}
                ],
            }
        ),
        encoding="utf-8",
    )

    body, notes = builder.merge_payloads(tmp_path)

    assert [arm["local_id"] for arm in body["design"]["arms"]] == ["active"]
    assert [tp["local_id"] for tp in body["design"]["timepoints"]] == ["baseline"]
