"""Read a contrast's polarity off its own name, and give each cell its direction.

The statistic cannot do this. 76% of reviewed analyses carry a statistic with an
unambiguous sign, but the sign is the same either way round: a table of "FESZ > NC" and
a table of "NC > FESZ" both print positive t-values, and in the gold two analyses with
identical `sign=+1` assign opposite directions to the same two groups. Sign is enough to
split a mixed-sign table -- which is what `sign_split.split_opposite_signs` uses it for
-- and not enough to direct a cell.

The polarity is written in the contrast's name: `FESZ>NC`, `AD < HC reduced GM volume`,
`greater activation in patients than controls`. That is a regex, and this is it.

Levels are matched to the sides of the comparison by word-set containment and never by a
similarity ratio: `men` is a substring of `women`, and `synchronous` scores 0.96 against
`asynchronous`.
"""

from __future__ import annotations

import json
import re

from pondie.formats import values
from pondie.vocabularies import labels

#: Ordered: the first pattern that matches wins, so the explicit operators are tried
#: before the prose forms that could also match inside them.
_COMPARISONS: tuple[tuple[re.Pattern[str], int], ...] = (
    # The sides stop at a sentence or clause boundary. Without that the right-hand side
    # runs to the end of the definition and swallows a group name held constant across
    # the contrast -- "7d > 28d . ALFF differences in the CCD group" matched CCD to the
    # right side and directed a `held` cell.
    # A parenthesis ends one too: "PTSD > HC (p < 0.001)" is HC, not "HC (p".
    (re.compile(r"(?P<a>[^<>.,;(]+?)\s*>\s*(?P<b>[^<>.,;(]+)"), +1),
    (re.compile(r"(?P<a>[^<>.,;(]+?)\s*<\s*(?P<b>[^<>.,;(]+)"), -1),
    (
        re.compile(
            r"(?P<a>[^.,;]+?)\s+(?:greater|higher|larger|stronger|increased|more)\s+"
            r"than\s+(?P<b>[^.,;]+)",
            re.I,
        ),
        +1,
    ),
    (
        re.compile(
            r"(?P<a>[^.,;]+?)\s+(?:less|lower|smaller|weaker|decreased|reduced)\s+"
            r"than\s+(?P<b>[^.,;]+)",
            re.I,
        ),
        -1,
    ),
    (
        re.compile(
            r"(?:greater|higher|increased|stronger|more)\s+(?:\w+\s+){0,4}?in\s+"
            r"(?P<a>[^.,;]+?)\s+(?:compared (?:with|to)|relative to|versus|vs\.?|than)"
            r"\s+(?P<b>[^.,;]+)",
            re.I,
        ),
        +1,
    ),
    (
        re.compile(
            r"(?:lower|reduced|decreased|weaker|less)\s+(?:\w+\s+){0,4}?in\s+"
            r"(?P<a>[^.,;]+?)\s+(?:compared (?:with|to)|relative to|versus|vs\.?|than)"
            r"\s+(?P<b>[^.,;]+)",
            re.I,
        ),
        -1,
    ),
)

#: Dropped when comparing, so `ASD` matches `ASD group`. Never used to decide whether a
#: side of a comparison exists: a contrast named "patients than controls" is made
#: entirely of these words, and treating it as empty loses the commonest phrasing there
#: is.
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "for",
        "and",
        "group",
        "groups",
        "patients",
        "subjects",
        "participants",
        "children",
        "adults",
    }
)


def _tokens(text: str) -> frozenset[str]:
    """Through `labels`, not a local character class, and the accents are the reason.

    This was `re.findall(r"[a-z0-9]+", text.lower())`, which splits a word on any letter
    outside the class rather than folding it: `naive` came back as `na` and `ve`, and
    `Etude` lost its first letter to become `tude`. `folding.fold` decomposes and drops the
    combining mark instead, which is the failure its own docstring was written about.

    32 of the corpus's 18,824 level and name strings were affected, and they are the ones
    the literature turns on: `same_level` scored `Fagerstrom Test for Nicotine Dependence`
    against its accented spelling as False, and likewise `Montgomery-Asberg Depression
    Rating Scale` and `drug-naive controls`. One instrument, two spellings, two levels.
    """
    return labels.tokens(text or "")


def _words(text: str) -> frozenset[str]:
    """The content words, or every word when the phrase is nothing but stopwords."""
    return labels.content(text or "", stop=_STOP)


def _singular(words: frozenset[str]) -> frozenset[str]:
    """`controls` as `control`: a level is named in the singular and a comparison in the
    plural ("Combined PTSD and major depression groups < Controls", 21418787)."""
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
                     for w in words)


def same_level(a: str, b: str) -> bool:
    """Do these two strings name the same level?

    Word-set containment, never a graded ratio. The failure this prevents is real: a
    0.85 similarity threshold matches `men` to `women` and `synchronous` to
    `asynchronous`, and a cell given the wrong level is a wrong direction.

    Compared on content words when both sides have them, so `ASD` reaches `ASD group`,
    and on every word when one side is built only from stopwords, so `patients` does not
    silently match everything.
    """

    left_all, right_all = _singular(_tokens(a)), _singular(_tokens(b))
    if not left_all or not right_all:
        return False
    left, right = left_all - _STOP, right_all - _STOP
    if not left or not right:
        left, right = left_all, right_all
    return left <= right or right <= left


def _comparison(text: str) -> tuple[re.Match[str], int] | None:
    """The first match that reads as a comparison of two named things, and its sign."""
    for index, (pattern, sign) in enumerate(_COMPARISONS):
        match = pattern.search(text or "")
        if not match:
            continue
        left, right = match.group("a").strip(" .,:;"), match.group("b").strip(" .,:;")
        # A threshold is not a comparison: "FTD-MND compared with FTD at P <0.001" signed
        # both of 10526199's levels negative.
        if re.search(r"(?i)\b[pq]$", left) or not re.search(r"[A-Za-z]", right):
            continue
        # Nor is a worded quantity: "patients with more than 1 year of heavy alcohol use"
        # (20487539). Only the worded forms -- `7d > 28d` compares two levels.
        if index >= 2 and not re.match(r"[A-Za-z]", right):
            continue
        if _words(left) and _words(right):
            return match, sign
    return None


def polarity(text: str) -> tuple[str, str, int] | None:
    """(left side, right side, +1 or -1) for a contrast named as a comparison."""
    found = _comparison(text)
    if found is None:
        return None
    match, sign = found
    return match.group("a").strip(" .,:;"), match.group("b").strip(" .,:;"), sign


def reverse_comparison(text: str) -> str | None:
    """`text` with its comparison's `<`/`>` swapped, and nothing else; None when the
    comparison is worded ("greater than") or there is none."""
    found = _comparison(text)
    if found is None:
        return None
    match, _sign = found
    between = text[match.end("a"):match.start("b")]
    if not re.fullmatch(r"\s*[<>]\s*", between):
        return None
    op = match.end("a") + between.index(between.strip())
    return text[:op] + {"<": ">", ">": "<"}[text[op]] + text[op + 1:]


def direction_of(level: str, contrast: str) -> str | None:
    """`positive`, `negative`, or None when the contrast does not name this level.

    None is the common answer and must stay distinguishable from a direction: a cell the
    name does not mention is one the model still has to be asked about.
    """

    read = polarity(contrast)
    if read is None:
        return None
    left, right, sign = read
    on_left, on_right = same_level(level, left), same_level(level, right)
    if on_left == on_right:
        # Named on both sides or neither -- no answer, rather than a coin flip.
        return None
    if on_left:
        return "positive" if sign > 0 else "negative"
    return "negative" if sign > 0 else "positive"


#: Directions that survive a reversal unchanged. `undirected` has no sign to flip;
#: `held` marks a level the contrast holds constant, which is true from either side.
_FIXED = frozenset({"undirected", "held", "absent"})
_OPPOSITE = {"positive": "negative", "negative": "positive"}


def reverse(direction: str) -> str:
    return _OPPOSITE.get(direction, direction) if direction not in _FIXED else direction


#: Statistics with no sign to reverse. A p-value is positive in either reading of a
#: contrast, and flipping it would make a number the paper never printed.
_UNSIGNED_STATISTICS = frozenset({"p", "p-value", "pvalue", "cluster_size", "voxels", "k"})


def mirror_analysis(described: dict, withheld: dict, parse_key: str = "") -> dict:
    """Rebuild the half of a sign-split contrast the paper never describes.

    `sign_split.split_opposite_signs` partitions a mixed-sign table and hands the
    extraction pass only the positive half, because that is the half the paper's prose is
    about: "FESZ > NC" prints positive statistics for the effects it names. The negative
    rows are the same contrast read the other way, and asking a model to name and define
    a contrast with no prose behind it produces invention rather than extraction.

    The reversed analysis is arithmetic, not extraction: the described half's cells
    with their directions flipped, addressing the withheld half's own row group.
    """

    mirrored = json.loads(json.dumps(described))
    mirrored["local_id"] = f"{described.get('local_id', 'analysis')}-reversed"
    mirrored["mirror_of"] = described.get("local_id")

    # The withheld entry's name, not the described half's. Copying the name wholesale left
    # an analysis called "FESZ > NC" whose cells say NC > FESZ -- a name contradicting its
    # own content, colliding with the real "FESZ > NC" on the same table. Every one of the
    # 36 mirrors in the schizophrenia corpus carried a colliding name, and it is what made
    # the direction bench score a correct extraction as a sign flip: two candidates share
    # a name and the matcher took the reversed one.
    #
    # The parse's label -- "<described name> (reversed)" -- is used rather than an inverted
    # operator because a described name is usually not a contrast expression at all: 46 of
    # 50 are labels like "GM Spatial Map" or "Seed: Right anterior cingulate cortex", which
    # have no operator to invert.
    reversed_name = (withheld or {}).get("name")
    if reversed_name:
        mirrored["name"] = values.wrap(
            str(reversed_name), source="generated", evidence="not_found"
        )

    # The reversed half's coordinates are reached the way every other analysis reaches
    # its own -- by the parse key -- and not by carrying flipped rows inline. The schema
    # stores no coordinates, so an inline `points` list is an attribute no class declares:
    # the validator reported it on 21 of 299 records, all of them this function's output.
    # The key must be the WITHHELD entry's, because that is the row group holding the
    # rows this half is about.
    mirrored.pop("points", None)
    mirrored.pop("coordinates", None)
    if parse_key:
        # A wrapper and not a bare string: the slot is `model_extracted`, so it projects
        # into the extraction schema as an ExtractedString, `generated` because the key is
        # the parse's and no sentence of the paper warrants a reversal it never describes.
        mirrored["source_table_analysis"] = values.wrap(
            parse_key, source="generated", evidence="not_found"
        )
    else:
        mirrored.pop("source_table_analysis", None)

    effect = mirrored.get("effect") or {}
    for cell in effect.get("cells") or []:
        node = cell.get("direction")
        if isinstance(node, dict) and isinstance(node.get("value"), str):
            flipped = reverse(node["value"])
            if flipped != node["value"]:
                node["value"] = flipped
                # Only a flipped direction is generated. A `held` level is held from
                # either side of the contrast and keeps the warrant it was read from.
                node["value_source"] = "generated"
                # It loses the span for the same reason as the mirrored `name` and
                # `source_table_analysis` never had one: no sentence of the paper warrants a
                # reversal the paper never describes. Keeping the described half's quote
                # here shipped a *verified* span -- `_walk` resolves it to real offsets
                # afterwards -- supporting the opposite claim to the one the cell now makes,
                # on the values a reviewer actually reads. That is a false citation.
                node["evidence"] = {"status": "not_found"}
        elif isinstance(node, str):
            cell["direction"] = reverse(node)
    return mirrored
