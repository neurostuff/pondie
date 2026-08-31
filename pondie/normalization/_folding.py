"""Orthographic folding: the language-general part of matching a surface form.

Deliberately not domain rules. `use disorder -> dependence` and `affective -> mood` are
claims about psychiatry and belong to a vocabulary or an encoder; case, punctuation, plurals
and hyphenation are claims about English and belong here.
"""

from __future__ import annotations

import re

_PLURAL_NOUNS = (
    "disorders",
    "diseases",
    "syndromes",
    "deficits",
    "symptoms",
    "episodes",
    "controls",
    "patients",
    "subjects",
    "participants",
)


def fold(value: object) -> str:
    """Lowercase, alphanumeric, single-spaced. The comparison key for an exact match."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def squash(value: object) -> str:
    """`fold` with the spaces removed, for containment tests where spacing is noise."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def singular(text: str) -> str:
    """Plurals only. A four-letter minimum keeps `bias` and `axis` intact."""
    out = re.sub(r"\b(" + "|".join(_PLURAL_NOUNS) + r")\b", lambda m: m.group(1)[:-1], text)
    return re.sub(r"\b(\w{4,})s\b", r"\1", out)


def variants(value: object) -> list[str]:
    """Every orthographic form worth trying, most faithful first, without duplicates."""
    base = fold(value)
    return list(dict.fromkeys(v for v in (base, singular(base)) if v))
