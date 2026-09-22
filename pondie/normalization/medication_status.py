"""`Group.medication_status` -> whether the cohort was on medication when scanned.

Negation, not vocabulary, is the discriminating feature: scope comes from a dependency
parse (`_negation`) and the domain part is the `CONCEPTS` lexicon. An unnegated mention
settles it. NAIVE is kept apart from FREE.

Why, with the measurements: docs/normalization-rationale.md, "medication_status".
"""

from __future__ import annotations

import re

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import Decision, Rule, classify
from pondie.normalization._negation import mentions

MEDICATED, FREE, NAIVE, MIXED = "MEDICATED", "FREE", "NAIVE", "MIXED"
VALUES = (MEDICATED, FREE, NAIVE, MIXED, OTHER, UNKNOWN)

#: Where the field lives. Named once; `report` and every caller read it from here.
PATH = "groups.medication_status"

CONCEPTS = re.compile(
    r"medicat|drug|antipsychot|antidepress|psychotrop|psychoactiv|neurolept|lithium|"
    r"stimulant|\bSSRI|\bSNRI|benzodiazep|anxiolytic|mood stabili[sz]|prescri|"
    r"pharmacolog|pharmacotherap|\bmedic\b",
    re.I,
)

MORPHOLOGICAL = re.compile(r"\b(?:un|non)[\s-]?(medicat|treated|prescribed)", re.I)

DENIED_NAIVE = re.compile(r"\b(?:not|never|non)\b[\s\w-]{0,16}?na[iï]ve", re.I)

#: Read before scope; see the doc named above.
MARKERS = (
    Rule.of(
        NAIVE,
        r"\bna[iï]ve\b|never (?:been )?(?:medicated|treated|prescribed)|"
        r"no (?:prior|previous|lifetime) (?:medication|treatment|exposure)",
    ),
    Rule.of(
        MIXED,
        r"\bmixed\b|\bsome (?:were|of (?:them|the|whom))|partially medicated|"
        r"\bboth medicated and\b|\bvaried\b|\bheterogeneous\b",
    ),
    Rule.of(OTHER, r"^\s*not applicable\s*$|^\s*n/?a\s*$"),
)


def normalize(text: object) -> Decision:
    raw = text if isinstance(text, str) else ""
    if not raw.strip():
        return Decision(UNKNOWN, "empty", raw)

    text_ = MORPHOLOGICAL.sub(r"not \1", raw)

    marked = classify(text_, MARKERS)
    if marked and not (marked.value == NAIVE and DENIED_NAIVE.search(text_)):
        return marked
    # No `available()` guard: without a parse this returned UNKNOWN, which is also what a
    # paper that never mentions medication returns, so an uninstalled model read as a corpus
    # that stopped reporting. `mentions` raises instead, naming the package.
    found = mentions(text_, CONCEPTS)
    if not found:
        return Decision(UNKNOWN, "no medication mention", raw)
    if any(not negated for _word, negated in found):
        return Decision(MEDICATED, "affirmative mention", raw)
    return Decision(FREE, "negation scope", raw)


def report(patterns: tuple[str, ...] | None = None) -> str:
    from pondie.normalization._lexicon import field_report

    return field_report(PATH, normalize, VALUES, patterns)
