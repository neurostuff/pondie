"""Warranting a value: which characters of the paper say so.

    quote      ask the model for a supporting quote, and put a block on every field
    retrieval  a second locator that runs locally, unioned with the first

Two locators rather than one because they fail differently, and the union was measured:
handing the model a retrieved shortlist instead of the whole paper cost 21 points. The
retriever is an enhancement and stays optional -- a host without torch does the quote pass
and says so, rather than taking the stage down. See docs/evidence-union-design.md.
"""
