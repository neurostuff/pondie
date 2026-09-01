"""Whether a mention falls inside the scope of a negation, from the sentence's own syntax.

Written because the alternative does not generalize. A proximity regex -- a negation cue
within N characters of a concept word -- has to be retuned for every new phrasing, and it
cannot tell `no` the negator from `no` the determiner: in "taking antidepressant medication;
no medication changes" the cohort *is* medicated, and a proximity rule reads it as the
opposite. A dependency parse distinguishes them because they are different relations.

The domain part shrinks to a lexicon of concept words, which is small and stable. The
linguistic part is a general parser, so a phrasing nobody anticipated is handled by syntax
rather than by another rule.

Scope follows the standard clinical-NLP treatment (Chapman et al., NegEx, 2001): a negation
governs its syntactic subtree, so a mention is negated when the negation attaches to it or to
any of its ancestors.
"""

from __future__ import annotations

import functools
import re

from pondie._deps import MissingDependency, require

#: Negation expressed as a modifier rather than a `neg` dependency: "drug-free", "off
#: medication", "absence of treatment". Closed and short by design.
_ADJECTIVAL = re.compile(r"\b(?:free|naive|na[iï]ve|off|absent)\b|\bfree$|-free\b", re.I)
_DETERMINERS = {"no", "neither", "none"}
#: Negating words wherever they sit in the clause. "no longer receiving medication" attaches
#: `no` to `longer`, two steps from the mention, so a direct-children test misses it.
_NEGATORS = {"no", "not", "never", "nor", "neither", "without", "none", "n't"}


@functools.lru_cache(maxsize=1)
def _parser():
    """The blank-parse pipeline, built once.

    Raise an error if parsing is unavailable. Without scope, "not medicated" and
    "medicated" contain the same words, so the field would read UNKNOWN. Papers that never
    mention medication also read UNKNOWN, making a missing model look like missing data.
    """
    spacy = require("spacy", "nlp", "negation scope cannot be read without a parse")
    try:
        return spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
    except OSError as error:
        raise MissingDependency(
            "spaCy is installed but the en_core_web_sm model is not. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from error


def available() -> bool:
    """Whether a parse is possible. For reporting the state of a run, not for deciding."""
    try:
        _parser()
    except MissingDependency:
        return False
    return True


def mentions(text: str, concepts: re.Pattern) -> list[tuple[str, bool]]:
    """(mention, is negated) for every concept word the text contains.

    A caller decides what to do with a mixture. For a status field the usual reading is that
    one unnegated mention settles it: a cohort described as taking something is taking it,
    whatever else the sentence goes on to deny.
    """
    nlp = _parser()
    found = []
    for token in nlp(text):
        if not concepts.search(token.text):
            continue
        found.append((token.text, _negated(token)))
    return found


def _negated(token) -> bool:
    """A mention is negated when a negation attaches to it or governs one of its ancestors.

    Ancestors are searched over their LEFT subtree only. English negation precedes what it
    scopes over, and the restriction is what stops a later clause reaching back: in
    "taking medication; no changes" the `no` is to the right of the mention and does not
    negate it.
    """
    for node in (token, *token.ancestors):
        for left in node.lefts:
            for candidate in left.subtree:
                if candidate.i >= token.i:
                    continue
                if candidate.dep_ == "neg" or candidate.lower_ in _NEGATORS:
                    return True
                if candidate.dep_ == "det" and candidate.lower_ in _DETERMINERS:
                    return True
        for child in node.children:
            if child.dep_ in {"amod", "acomp", "prep"} and _ADJECTIVAL.search(child.text):
                return True
        if _ADJECTIVAL.search(node.text) and node is not token:
            return True
    return False
