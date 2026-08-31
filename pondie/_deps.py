"""Refuse to answer without the package the answer depends on.

Three of this package's fields are decided by a model or a parser rather than by a rule, and
each one has an obvious-looking value to return when the dependency is absent: no negation
found, no abbreviation found, no evidence found. Every one of those is also a real answer some
paper genuinely warrants, so a missing install is indistinguishable from a paper that said
nothing -- and it is indistinguishable in the direction that looks like data rather than like
a broken environment.

So a dependency whose absence would change an answer raises here instead. A dependency whose
absence only costs an *enhancement* -- a second evidence locator, a better abbreviation miner
-- may still fall back, but it says which path it took rather than degrading in silence.
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
