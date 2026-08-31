"""Scoring an extraction against the reviewer gold that lives beside it.

The gold tables in `benchmarks/gold/` are a reviewer artefact, generated elsewhere. The
benchmark that reads them is here, so a change to extraction is scored by the repository that
made the change rather than by whichever checkout happens to hold the tables.

One question is asked: for a term both sides agree is in the contrast, did the extractor put
it on the right side? Missing terms and mislabelled other terms are reported as coverage
rather than penalised, so the headline cannot be a lie by omission.

Read the headline against the right thing. Two reviewers scoring the same 239 cells agree
**78.2%** read naively; the 95.8% sometimes quoted is that figure weighed by provenance tier,
and the narrowest defensible number is 44 cells at 95.5% where both chose a sign. None of
those share a denominator with a polarity score over this gold, so none of them is a ceiling
for it. What the doubly-reviewed set does show is that of 52 disputed cells only 2 are
`positive` vs `negative`: humans agree about polarity and argue about membership.
"""
from .direction import load_gold, score  # noqa: F401

__all__ = ["load_gold", "score"]
