"""Warranting a value: which characters of the paper say so.

    quote      ask the model for a supporting quote, and put a block on every field
    retrieval  sentence and section machinery; only `sectionize` still has callers

`quote` is the locator. There was a second one that ranked sentences locally and was unioned
with it -- handing the model a retrieved shortlist instead of the whole paper cost 21 points,
so the union was the measured answer. It went with the local models, and what is left of
`retrieval` is the text-splitting `repair` and `grounding` still use.
See docs/evidence-union-design.md.
"""
