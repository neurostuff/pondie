"""Do two labels name one thing? The domain-general part of deciding.

A layer between `folding` and a vocabulary, and it exists because both `onvoc` and
`abbreviations` need it. `folding` claims the language-general job -- case, accents,
plurals -- and says so: "`use disorder -> dependence` and `affective -> mood` are claims
about psychiatry and belong to a vocabulary". These are the third thing, and neither rule
covers them: `group`, `scale`, `questionnaire` and `disorder` carry no identity in *this*
corpus, which is a claim about clinical writing rather than about English or about any
one ontology.

They were in `onvoc` and `abbreviations` reached `stems` through a deferred import to
compare two expansions -- while `onvoc` deferred an import back the other way, for the
paper's own abbreviations. A mutual cycle, suppressed at three call sites. Both sides
depend on this instead.
"""

from __future__ import annotations

from pondie.vocabularies.folding import fold

#: Words that carry no identity. Dropped only when comparing token sets, never when
#: deciding whether a phrase exists -- "usual care" is made entirely of weak words.
_WEAK = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "and",
        "or",
        "for",
        "with",
        "group",
        "groups",
        "patients",
        "subjects",
        "participants",
        "condition",
        "conditions",
        "task",
        "tasks",
        "test",
        "tests",
        "scale",
        "inventory",
        "questionnaire",
        "disorder",
        "arm",
    }
)


#: True function words. Distinct from `_WEAK`, which also drops domain nouns for the
#: purpose of comparing token sets; those nouns still carry a letter in an acronym.
_FUNCTION = frozenset({"the", "a", "an", "of", "in", "and", "or", "for", "with", "on"})


def tokens(text: str) -> frozenset[str]:
    return frozenset(fold(text).split())


def content(text: str, stop: frozenset[str] = _WEAK) -> frozenset[str]:
    """Content tokens, or every token when the phrase is nothing but weak ones.

    `stop` is a parameter because there are two defensible lists and they should be
    readable against each other. Vocabulary matching drops the domain nouns in `_WEAK` --
    `scale`, `questionnaire`, `disorder` -- because an ONVOC label is made of them.
    `record.direction` keeps those and drops `children` and `adults` instead, because a
    cell level naming a cohort is the identity there. The fallback is what makes either
    list safe: a phrase of nothing but stopwords keeps all of them, so "usual care"
    survives `_WEAK` and "adults" survives direction's.
    """
    every = tokens(text)
    return (every - stop) or every


#: Suffixes stripped to relate `depression` to `Depressive Disorder`. ONVOC carries the
#: clinical noun phrase and papers write the everyday noun, and no amount of containment
#: bridges the two: neither string contains the other. Ordered longest first so
#: `-ational` is tried before `-al`.
_SUFFIXES = (
    "ational",
    "iveness",
    "ically",
    "ation",
    "ities",
    # After `ation`, so `agitation` stems to `agitat` and not `agitati`. Without it the
    # nominalisation of a clinical noun does not reach its adjective -- `depression` stayed
    # whole while `depressive` folded to `depress`, so a paper writing the commoner of the two
    # missed `Depressive Disorder` entirely. 45 ONVOC labels contain a word this strips, and
    # the resolvable stem count is unchanged at 744, so it collapses no concept into another.
    "ion",
    "ive",
    "ity",
    "ies",
    "ing",
    "ed",
    "al",
    "s",
)


def stem(word: str) -> str:
    """A crude suffix strip. Crude on purpose: it only has to make two surface forms of
    the same clinical noun collide, and a real stemmer would be a dependency for that."""
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def stems(text: str) -> frozenset[str]:
    return frozenset(stem(word) for word in content(text))


def acronym(text: str) -> str:
    """The initials of a multi-word label, or "" when it is not that kind of label.

    Two words is too few -- `Drug Use` would claim `DU`. Single-letter tokens are dropped
    before counting, because folding `Alzheimer's Disease` leaves a stray `s` that turns
    a two-word name into the three-letter `ASD`, which is a different disorder entirely.
    """
    # Function words only, never the domain nouns `_WEAK` drops. `disorder`, `scale` and
    # `test` are precisely the words a clinical acronym is built from -- dropping them
    # turns `Autism Spectrum Disorder` into two words and no acronym at all.
    words = [w for w in fold(text).split() if w not in _FUNCTION and len(w) > 1]
    if len(words) < 3 or len(words) > 6:
        return ""
    return "".join(word[0] for word in words)
