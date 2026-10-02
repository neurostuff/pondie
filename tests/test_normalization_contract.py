"""The contract `pondie/normalization/__init__.py` states, enforced.

"Every module exposes `normalize(...)` returning a value plus the reason it was chosen, and
`report(...)` for the residual." Five of the eight bound only the first, so `pondie normalize
coordinate_space` raised `AttributeError` -- on a field the CLI's own help text offered as an
example. A contract stated in a docstring and checked nowhere is a suggestion.
"""

from __future__ import annotations

import importlib

import pytest

from pondie import normalization
from pondie.cli import DERIVED, _normalizable

FIELDS = normalization.fields()


@pytest.mark.parametrize("field", FIELDS)
def test_every_field_module_exposes_the_whole_contract(field: str) -> None:
    module = importlib.import_module(f"pondie.normalization.{field}")
    assert callable(getattr(module, "normalize", None)), f"{field} exposes no normalize()"
    assert callable(getattr(module, "report", None)), f"{field} exposes no report()"


def test_the_cli_offers_exactly_the_fields_that_exist():
    """Derived from the package rather than written down, so the two cannot drift.

    Plus `derived`, which runs the slot fills rather than reporting a field. It is the one
    choice that is not a module name, and it is checked against the field list here so it
    can never shadow one.
    """
    assert _normalizable() == [*FIELDS, DERIVED]
    assert DERIVED not in FIELDS
