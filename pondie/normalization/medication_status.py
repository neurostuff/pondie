"""`Group.medication_status` -> whether the cohort was on medication when scanned.

493 surface forms over 711 values, the messiest field measured, and the standard moderator in
schizophrenia and depression meta-analyses.

The discriminating feature is **negation**, not vocabulary. Every affirmative cue appears
inside its own negation -- "not medicated" contains "medicated", "free of antipsychotics"
contains "antipsychotics" -- so a keyword rule inverts the cohort it describes unless it
models scope, and a proximity rule cannot tell `no` the negator from `no` the determiner.
Scope therefore comes from a dependency parse (`_negation`), and the domain part shrinks to a
lexicon of concept words. A phrasing nobody anticipated is then handled by syntax rather than
by adding another regex.

An unnegated mention settles it. A cohort described as taking something is taking it, whatever
else the sentence goes on to deny -- "taking antidepressants; no medication changes" is a
medicated cohort.

NAIVE is kept apart from FREE deliberately: never-medicated and withdrawn-before-scanning are
different populations, and a moderator analysis that merges them cannot see a treatment-history
effect.
"""

from __future__ import annotations

import re

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import Decision, Rule, classify
from pondie.normalization._negation import mentions

MEDICATED, FREE, NAIVE, MIXED = "MEDICATED", "FREE", "NAIVE", "MIXED"
VALUES = (MEDICATED, FREE, NAIVE, MIXED, OTHER, UNKNOWN)

#: The domain lexicon: what counts as a medication mention. Small and stable -- growing it is
#: how this field is extended, not by adding phrasings.
CONCEPTS = re.compile(
    r"medicat|drug|antipsychot|antidepress|psychotrop|psychoactiv|neurolept|lithium|"
    r"stimulant|\bSSRI|\bSNRI|benzodiazep|anxiolytic|mood stabili[sz]|prescri|"
    r"pharmacolog|pharmacotherap|\bmedic\b",
    re.I,
)

#: Morphological negation. A parse cannot see it -- "unmedicated" is one token with no
#: syntactic negation to attach to -- so it is stripped to its scope-visible form first.
MORPHOLOGICAL = re.compile(r"\b(?:un|non)[\s-]?(medicat|treated|prescribed)", re.I)

#: The marker denied within a few words of itself, which is the only reading that inverts it.
DENIED_NAIVE = re.compile(r"\b(?:not|never|non)\b[\s\w-]{0,16}?na[iï]ve", re.I)

#: Read before scope, because each names a status that negation alone cannot express. Each is
#: itself checked for negation: "not drug-naive" contains "naive" and means the opposite.
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
    #: "not drug-naive" carries the marker and denies it. Checked with an adjacency pattern
    #: rather than parse scope, because scope reaches across clauses: in "never-medicated;
    #: antipsychotic naive" the negation belongs to the first clause and not to the marker.
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
    from pondie.normalization._lexicon import summarize
    from pondie.normalization._records import DEFAULT, iter_records, strings_at

    decisions = [
        normalize(s)
        for _study, body in iter_records(patterns or DEFAULT)
        for s in strings_at(body, "groups.medication_status")
    ]
    return f"groups.medication_status: {len(decisions)} values\n" + summarize(
        decisions, VALUES
    )


if __name__ == "__main__":
    print(report())
