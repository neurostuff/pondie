"""Whether a mention falls inside the scope of a negation, from the sentence's own syntax.

A negation governs its syntactic subtree, so a mention is negated when the negation
attaches to it or to any of its ancestors (Chapman et al., NegEx, 2001). Raises when no
parser is installed rather than answering as though the text said nothing.

Why, and what the alternatives got wrong: docs/normalization-rationale.md, "_negation".
"""

from __future__ import annotations

import functools
import re

from pondie._deps import MissingDependency, require

#: Negation expressed as a modifier rather than a `neg` dependency: "drug-free", "off
#: medication", "HIV-negative". Closed and short by design.
_ADJECTIVAL = re.compile(
    r"\b(?:free|naive|na[iï]ve|off|absent|negative)\b|\bfree$|-(?:free|negative)\b", re.I
)
_DETERMINERS = {"no", "neither", "none"}
#: Negating words wherever they sit in the clause. "no longer receiving medication" attaches
#: `no` to `longer`, two steps from the mention, so a direct-children test misses it.
#:
#: `absence` and `lack` are nouns and the rest function words. In "absence of major
#: depressive disorder" the negation IS the head noun and the disorder its `of`-complement,
#: so nothing else in this module could see it.
_NEGATORS = {
    "no",
    "not",
    "never",
    "nor",
    "neither",
    "without",
    "none",
    "n't",
    "absence",
    "lack",
}


@functools.lru_cache(maxsize=1)
def _parser():
    """The blank-parse pipeline, built once. Raises if parsing is unavailable."""
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

    A caller decides what to do with a mixture.
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

    Ancestors are searched over their LEFT subtree only, and an ancestor that IS the
    negation is checked before its subtree. Why: docs/normalization-rationale.md, "_negation".
    """
    for node in (token, *token.ancestors):
        if node is not token and (node.dep_ == "neg" or node.lower_ in _NEGATORS):
            return True
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


#: A negation cue as a standalone word, for the residue pass and the no-parser fallback.
#: Built from `_NEGATORS` so the two layers cannot drift.
_CUE = re.compile(
    r"(?<![\w-])(?:"
    + "|".join(sorted(_NEGATORS - {"n't"}, key=len, reverse=True))
    + r"|free|absent|negative|unaffected)(?![\w])|-(?:free|negative)\b",
    re.I,
)


def cue_forward_scope(text: str) -> str:
    """`text` with the first negation cue and everything after it removed.

    NegEx's forward scope (Chapman et al. 2001).
    """
    found = _CUE.search(text or "")
    if not found:
        return text or ""
    cut = found.start()
    # A suffixed cue negates what PRECEDES it: `HIV-negative` and `drug-free` are absences
    # of the thing they name, so the word in front of the hyphen goes too.
    if text[cut] == "-":
        while cut > 0 and (text[cut - 1].isalnum() or text[cut - 1] == "-"):
            cut -= 1
    return text[:cut].strip(" -,;/")


@functools.lru_cache(maxsize=8192)
def scope(text: str) -> tuple[str, str] | None:
    """(what this phrase asserts, what it denies), or None with no parser installed.

    The whole value is parsed BEFORE the caller splits it into heads, and what survives is
    cut out of the ORIGINAL string by character offset. Why: docs/normalization-rationale.md, "_negation".
    """
    try:
        nlp = _parser()
    except MissingDependency:
        return None
    parsed = nlp(text)
    in_scope = [False] * len(text)
    for token in parsed:
        if _negated(token):
            for i in range(token.idx, min(token.idx + len(token.text), len(text))):
                in_scope[i] = True
    kept = "".join(c for i, c in enumerate(text) if not in_scope[i])
    denied = "".join(c if in_scope[i] else " " for i, c in enumerate(text))
    return _tidy(cue_forward_scope(_tidy(kept))), _tidy(denied)


def _tidy(text: str) -> str:
    """Collapse the whitespace and dangling punctuation a cut-out span leaves behind."""
    return re.sub(r"\s{2,}", " ", re.sub(r"\s*([,;/])\s*", r"\1 ", text)).strip(" -,;/")
