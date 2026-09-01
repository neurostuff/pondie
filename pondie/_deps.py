"""Refuse to answer without the package the answer depends on.

Three of this package's fields are decided by a model or a parser rather than by a rule, and
each one has an obvious-looking value to return when the dependency is absent: no negation
found, no abbreviation found, no evidence found. Every one of those is also a real answer some
paper genuinely warrants, so a missing install is indistinguishable from a paper that said
nothing -- and it is indistinguishable in the direction that looks like data rather than like
a broken environment.

This module raises an error when a missing dependency would change an answer. Code may fall
back when the dependency provides only an enhancement, such as a second evidence locator or
a better abbreviation miner, but it reports which path it used.
"""

from __future__ import annotations

import importlib


class MissingDependency(ImportError):
    """A package the answer depends on is not installed."""


def require(module: str, extra: str, because: str):
    """Import `module`, or say what is missing, why it matters, and how to install it."""

    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise MissingDependency(
            f"{module} is not installed, and {because}. "
            f"Install it with: pip install 'pondie[{extra}]'"
        ) from error
