"""Orthographic folding: the language-general part of matching a surface form.

Deliberately not domain rules. `use disorder -> dependence` and `affective -> mood` are
claims about psychiatry and belong to a vocabulary or an encoder; case, punctuation, plurals
and hyphenation are claims about English and belong here.
"""

from __future__ import annotations

import functools
import re
import unicodedata

from pondie._deps import MissingDependency, require


def fold(value: object) -> str:
    """Lowercase, unaccented, alphanumeric, single-spaced. The key an exact match compares on.

    NFKD first, so a combining mark is separated from its letter and dropped rather than
    taking the letter with it. Without that step `naive` and `naive` are different keys and
    `Etude` folds to `tude` -- the accented letter is not in `[a-z0-9]`, so it becomes a
    space and splits the word. This module claims the language-general job, and it was the
    copy that got accents wrong while a second one in `onvoc` got them right.
    """
    stripped = unicodedata.normalize("NFKD", str(value or ""))
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", stripped.lower()).strip()


def squash(value: object) -> str:
    """`fold` with the spaces removed, for containment tests where spacing is noise.

    Built on `fold` rather than repeating its body, which is what it used to do without the
    NFKD step: `squash("naive")` returned "nave" where `fold` returns "naive", because an
    accented letter is not in `[a-z0-9]` and was deleted rather than decomposed. The one
    caller is exact-key clustering, so that turned one name spelled two ways into two
    clusters -- the failure clustering exists to prevent.
    """
    return fold(value).replace(" ", "")


@functools.lru_cache(maxsize=1)
def _lemmatizer():
    """WordNet's noun lemmatizer, built once.

    Raise rather than fall back. The variant this produces is one of the two keys an exact
    vocabulary lookup tries, so without it a plural head term misses its concept and drops
    through to the embedding path, where it is scored against a threshold and may be
    silently queued for review or rejected. That reads as a term the vocabulary does not
    cover rather than as a missing install.
    """
    nltk = require("nltk", "nlp", "a plural surface form cannot be matched without a lemma")
    try:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatizer.lemmatize("disorders", "n")
    except LookupError as error:
        raise MissingDependency(
            "nltk is installed but its wordnet corpus is not. "
            "Install it with: python -m nltk.downloader wordnet"
        ) from error
    return lemmatizer


@functools.lru_cache(maxsize=4096)
def singular(text: str) -> str:
    """Each word reduced to its WordNet noun lemma. `gyri -> gyrus`, `foci -> focus`.

    Nouns only, and the part of speech is asserted rather than tagged: every caller arrives
    through `variants`, whose input is a disorder or anatomy label, and a tagger would be a
    parse per lookup to decide a question the caller has already answered.

    A dictionary rather than a suffix rule, because the words here are the ones a suffix
    rule gets wrong. The `\\b(\\w{4,})s\\b` this replaced made `corpus` into `corpu`,
    `sclerosis` into `sclerosi` and `stress` into `stres`, and left every irregular alone --
    `stimuli`, `children`, `indices`. It was defended as symmetric, on the grounds that the
    same mangling applied to the vocabulary key would cancel, but `Vocabulary.surface` is
    built from `fold` alone (`mondo.py`), so nothing ever cancelled: a mangled variant was
    simply a key that could not exist. Over Mondo's 90,244 surface forms the two disagree
    on 22,337, and the lemma lands on a real vocabulary key 128 times against the suffix
    rule's 53.
    """
    lemmatize = _lemmatizer().lemmatize
    return " ".join(lemmatize(word, "n") for word in text.split())


def variants(value: object) -> list[str]:
    """Every orthographic form worth trying, most faithful first, without duplicates."""
    base = fold(value)
    return list(dict.fromkeys(v for v in (base, singular(base)) if v))
