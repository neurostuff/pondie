"""What `prompt/preprocess.py`'s sentence splitter must not get wrong.

Indexed evidence numbers a paper's sentences with `sentence_spans` and reads a cited
number back with it, so a bad boundary is a citation to the wrong sentence.
"""

from __future__ import annotations

import pytest

from pondie.extraction.prompt import preprocess

# ---------------------------------------------------------------------------
# The audit in docs/regex-audit.md, findings 1, 3 and 7.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk,mid",
    [
        ("Additionally, Li et al.", True),
        ("as reported previously (e.g.", True),
        ("opaque idioms (vs.", True),
        ("shown in Fig.", True),
        ("The scan lasted 12 min.", True),
        ("a whole sentence that ends here.", False),
        ("threshold was p < 0.001.", False),
        ("", False),
    ],
)
def test_a_period_after_an_abbreviation_is_not_a_sentence_end(chunk, mid):
    """The one abbreviation list in the repo, public so a second one cannot drift from it."""

    assert preprocess.ends_mid_sentence(chunk) is mid


def test_the_sentence_splitter_still_uses_the_shared_guard():
    text = "Activation was reported by Li et al. The cluster survived correction."
    assert preprocess.sentence_spans(text) == [(0, len(text))]


def test_a_decimal_point_is_not_a_boundary_and_a_full_stop_is():
    text = "The threshold was p < 0.001. Results survived."
    first, second = preprocess.sentence_spans(text)
    assert text[slice(*first)] == "The threshold was p < 0.001."
    assert text[slice(*second)] == "Results survived."
