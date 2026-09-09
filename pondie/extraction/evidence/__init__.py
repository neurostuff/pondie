"""Warranting a value: which characters of the paper say so.

    quote      ask the model for a supporting quote, and put a block on every field
    retrieval  sentence and section machinery; only `sectionize` still has a caller

`quote` is the locator. There was a second one that ranked sentences locally and was unioned
with it -- handing the model a retrieved shortlist instead of the whole paper cost 21 points,
so the union was the measured answer. It went with the local models, and what is left of
`retrieval` is the text-splitting `repair` still uses.

There was a third module, `grounding`, that scored a proposal against the passage offered for
it before `record.edit` was allowed to write it. It went with the same models, and for a while
survived as a holder for one constant.
See docs/evidence-union-design.md.
"""
