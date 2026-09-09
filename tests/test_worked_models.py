"""The worked models are projected from the records, and cannot restate them.

`representing-models.md` §5 used to hold a hand-written transcription of records this package
already ships. The transcription drifted -- an invented `FactorLevel.order`, a decrease encoded
on the wrong model's term, and two places where the document silently substituted a level label
its record's cells did not carry. These tests pin the property that made those possible
impossible: no encoding is written down, so none can disagree with its record.
"""

from __future__ import annotations

import copy
import re

import pytest

from pondie import schema
from pondie.extraction.prompt import render, worked


def test_the_committed_document_is_what_the_composer_produces():
    """`representing-models.md` §5 is generated output, committed so it renders on the forge.

    A file edited by hand is a file that has drifted by the next commit, which is the whole
    history of this section. Regenerate with `pondie.extraction.prompt.worked.document()`.
    """
    committed = re.search(
        r"^## 5\. Worked models$.*?(?=^## 6\.)",
        (schema.ROOT / "representing-models.md").read_text(encoding="utf-8"),
        re.M | re.S,
    )
    assert committed, "representing-models.md has no §5 to compare against"
    assert committed.group(0).rstrip() == worked.document().rstrip()


def test_every_worked_model_resolves_against_its_record():
    """Each example names a record, a model and an analysis that exist, and renders."""
    for entry in worked.manifest():
        rendered = worked.example(entry)
        assert rendered.startswith(f"### {entry['id']} ")
        assert "{{block}}" not in rendered, f"{entry['id']}: an unfilled block marker"
        assert "```yaml" in rendered, f"{entry['id']}: no encoding was rendered"


def test_the_prose_holds_no_encoding_of_its_own():
    """A YAML fence in a prose fragment would be a hand-written encoding, which is the thing
    this module exists to prevent -- and it would render, silently, right next to a real one.
    """
    for entry in worked.manifest():
        assert "```yaml" not in worked.prose(entry), (
            f"{entry['id']}: prose contains a YAML block. Encodings come from the record; "
            "the prose marks where one goes with {{block}}."
        )


def test_a_cell_naming_a_level_its_term_does_not_declare_is_refused(monkeypatch):
    """The failure §5.10 used to paper over: cells reading `HC` under a term declaring
    `healthy controls`. The document showed the declared label, so the record's inconsistency
    was invisible to every reader of both."""
    entry = next(e for e in worked.manifest() if e["id"] == "5.10")
    rec = copy.deepcopy(worked.record(entry["referent"]))
    for analysis in rec["analyses"]:
        for cell in (analysis.get("effect") or {}).get("cells") or []:
            if isinstance(cell.get("level"), dict):
                cell["level"]["value"] = "a level no term declares"
    monkeypatch.setattr(worked, "record", lambda _referent: rec)
    with pytest.raises(ValueError, match="does not declare"):
        worked.block(entry["referent"], entry["blocks"][0])


def test_a_manifest_naming_a_model_the_record_lacks_is_refused():
    entry = next(e for e in worked.manifest() if e["id"] == "5.1")
    spec = {"models": [{"id": "me-does-not-exist", "show": ["stage"], "terms": []}]}
    with pytest.raises(LookupError, match="no model_estimation"):
        worked.block(entry["referent"], spec)


def test_the_prompt_carries_every_worked_model():
    """`worked_models()` is what both passes are sent; a lost example is a lost shape."""
    prompt = render.worked_models()
    for entry in worked.manifest():
        assert f"### {entry['id']} " in prompt, f"{entry['id']} missing from the prompt"


def test_a_not_reported_wrapper_contributes_nothing():
    """A `not_reported` wrapper can still carry a stale `value`. Reading it would put a figure
    in a worked model that the record declines to state."""
    assert worked._unwrap({"extraction_status": "not_reported", "value": "3 T"}) is None
    assert worked._unwrap({"extraction_status": "extracted", "value": "3 T"}) == "3 T"
