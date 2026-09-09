"""Coordinates a paper reports in prose rather than in a table.

A result stated only in a sentence -- "activation peaked at x = 9, y = -12, z = -6" -- has no
table row for `tables` to parse, so without this it reaches the extractor as prose and leaves
the record with an analysis and no foci. `prose_parse_entries` finds those triples and
`prose_coordinate_block` writes them into the prompt as candidates the model confirms or
drops, never as facts: a regex over prose over-generates, and a block the model is told to
trust would import every false positive into the record.

Every offset the record carries still addresses the original text. `builder.py` is handed the
untransformed file, so nothing here moves an `EvidenceSpan.start_char`.

This module used to hold a set of text-transformation strategies as well -- dropping
sections, reordering them, selecting sentences, and a family of derived digest blocks.
`docs/text-preprocessing-experiments.md` measured them against a fixed pipeline, found the
run-to-run spread swamped the differences, and none was ever wired in. They are in the
history.

Standard library only, and deliberately: the measured wins here have to survive being adopted
by a repo whose dependency list is three packages.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# --------------------------------------------------------------- section segmentation

#: `text.py` writes the corpus text with its headings as `## `/`### ` markdown and
#: inlines each coordinate table under a `Table N — caption` line, so segmentation here is
#: a line scan and not a layout model.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)

#: Zone per heading, in two tiers. A STRONG pattern is a heading that names its own zone
#: ("Results", "Participants") and overrides whatever it is nested under; a WEAK one names
#: a kind of work that happens in more than one zone ("Univariate analysis" is Methods in
#: one paper and Results in another) and is used only when the parent heading said nothing.
#: Without the tiers "Voxel-based morphometry analyses" -- a Results subsection -- lands in
#: Methods on the word "analyses", and dropping the Introduction takes the results with it.
_STRONG_ZONES: list[tuple[str, re.Pattern[str]]] = [
    (
        "back",
        re.compile(
            r"\b(reference|bibliograph|acknowledg|conflict|competing interest|funding|"
            r"author contribution|data availability|supplement|appendix|abbreviation|"
            r"disclosure|declaration|footnote|highlight)",
            re.I,
        ),
    ),
    ("tables", re.compile(r"^tables?\b", re.I)),
    # A structured abstract is a run of `## Background:` / `## Methods:` / `## Results:`
    # headings, and taken at face value its Conclusion is a Discussion and its Results a
    # Results. The trailing colon is what distinguishes the label from the section.
    ("front", re.compile(r"^\s*abstract\b|:\s*$", re.I)),
    ("results", re.compile(r"\b(results?|findings?)\b", re.I)),
    (
        "discussion",
        re.compile(
            r"\b(discussion|conclusions?|limitations?|implications?|interpretation)\b", re.I
        ),
    ),
    ("intro", re.compile(r"\b(introduction|background|rationale)\b", re.I)),
    (
        "methods",
        re.compile(
            r"\b(methods?|materials?|procedures?|participants?|subjects?|sample|cohort|"
            r"acquisitions?|apparatus|preprocess\w*|pre-process\w*|ethics?|recruit\w*)\b",
            re.I,
        ),
    ),
]

_WEAK_ZONES: list[tuple[str, re.Pattern[str]]] = [
    (
        "methods",
        re.compile(
            r"\b(analys[ei]s|modell?ing|statistic\w*|design|tasks?|stimul\w*|measures?|"
            r"mask\w*|roi|region of interest|segmentation|normali[sz]ation|pipeline|"
            r"experiment\w*|questionnaire|assessment|drug|dose|administration|scann?\w*|"
            r"imaging|connectivity|quantification|parcellation|registration)\b",
            re.I,
        ),
    ),
]


@dataclass
class Section:
    """One heading and the text under it, up to the next heading of any level."""

    level: int
    heading: str
    body: str
    zone: str

    @property
    def text(self) -> str:
        # Level 0 is the front matter, whose heading is this module's own label for it.
        # Emitting it would put a sentence in the prompt the paper does not contain.
        if not self.level:
            return self.body
        return f"{'#' * self.level} {self.heading}\n{self.body}"


def classify(heading: str) -> tuple[str, str]:
    """(zone, "strong" | "weak" | "none") for one heading."""

    for zone, pattern in _STRONG_ZONES:
        if pattern.search(heading):
            return zone, "strong"
    for zone, pattern in _WEAK_ZONES:
        if pattern.search(heading):
            return zone, "weak"
    return "other", "none"


def split_sections(text: str) -> list[Section]:
    """Front matter first, then one Section per heading.

    A subsection inherits its parent's zone when its own heading says nothing -- "Study
    sample" under "Materials and Methods" is Methods -- but overrides it when it does.
    That is what keeps a Results subsection called "Voxel-based morphometry analyses"
    out of Methods, which a flat keyword match on "analyses" gets wrong.
    """

    marks = list(_HEADING.finditer(text))
    sections: list[Section] = []
    front = text[: marks[0].start()] if marks else text
    if front.strip():
        # Title, keywords and abstract, which carry no heading in the pubget text. The
        # abstract states the design and every headline result in two hundred words and
        # is the densest part of the paper; it is never a candidate for dropping.
        sections.append(
            Section(0, "Front matter (title, keywords, abstract)", front.strip(), "front")
        )

    stack: list[tuple[int, str]] = []
    for index, mark in enumerate(marks):
        level, heading = len(mark.group(1)), mark.group(2)
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        body = text[mark.end() : end].strip("\n")
        while stack and stack[-1][0] >= level:
            stack.pop()
        zone, strength = classify(heading)
        parent = stack[-1][1] if stack else "other"
        if strength != "strong" and parent != "other":
            zone = parent
        elif strength == "none":
            zone = parent
        stack.append((level, zone))
        sections.append(Section(level, heading, body, zone))
    return sections


# ------------------------------------------------------------------ sentence splitting

#: Tokens whose trailing period does not end a sentence. Mostly citation and unit
#: furniture; `et al.` and `Fig.` alone account for most bad splits in this corpus. A
#: single capital is in the list because an initial is the other common case.
_NON_TERMINAL = frozenset("""
et al e.g i.e cf vs etc approx ca resp viz fig figs tab tabs eq ref refs no nos dr prof
mr mrs ms st inc ltd co univ dept min sec ms mm cm ml mg kg vol ed eds pp al s.d s.e
i.v p.o a.m p.m
""".split()) | frozenset(chr(c) for c in range(ord("a"), ord("z") + 1))

#: A period, question or exclamation mark followed by space and something that could
#: start a sentence. Python's `re` will not take a variable-width lookbehind, so the
#: "unless the word before it is an abbreviation" half is a token check on the match and
#: not part of the pattern.
_BOUNDARY = re.compile(r"[.!?][\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.?$")


def ends_mid_sentence(text: str) -> bool:
    """Whether a run of text ends on a period that does not end a sentence.

    Public, and the only copy: `evidence/retrieval.py` cuts its units on the same
    punctuation and needs the same answer. It had no guard at all, and 9.6% of the units it
    returned over the corpus ended at `et al.` or `e.g.` -- a second abbreviation list would
    have drifted from this one, which is the reason this is a function and not a duplicate.
    """

    word = _LAST_WORD.search(text.rstrip(".!?"))
    return bool(word and word.group(1).lower().rstrip(".") in _NON_TERMINAL)


def _split_sentences(line: str) -> list[str]:
    pieces, start = [], 0
    for boundary in _BOUNDARY.finditer(line):
        head = line[start : boundary.start() + 1]
        if ends_mid_sentence(head):
            continue
        pieces.append(head)
        start = boundary.end()
    pieces.append(line[start:])
    return pieces


def paragraphs(text: str) -> list[str]:
    """Prose paragraphs, with headings and table rows left out.

    Consecutive prose lines are one paragraph. The corpus text keeps a paragraph on a
    single line, but a hard-wrapped one would otherwise be split at every line end, and
    a claim whose subject and verdict land in different "sentences" loses both. A pipe
    row ends a paragraph rather than joining it, or a coordinate gets quoted as part of
    the Methods claim above it.
    """

    found, current = [], []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "|")):
            if current:
                found.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        found.append(" ".join(current))
    return found


def sentences_of(paragraph: str) -> list[str]:
    # A decimal point is not a sentence end. Protecting the digits is cheaper and more
    # reliable than adding a case for every numeric shape in a Methods section
    # (p = 0.05, 4.6 mm, 1.5 T, r > 0.7).
    guarded = re.sub(r"(\d)\.(\d)", lambda m: f"{m[1]}\x00{m[2]}", paragraph)
    out = []
    for piece in _split_sentences(guarded):
        piece = piece.replace("\x00", ".").strip()
        if len(piece) > 2:
            out.append(re.sub(r"\s+", " ", piece))
    return out


def sentences(text: str) -> list[str]:
    """Every prose sentence, in document order."""

    return [s for paragraph in paragraphs(text) for s in sentences_of(paragraph)]


# --------------------------------------------------------------- abbreviation glossary


# ------------------------------------------------------------------- statistic digest

#: Signed values in this corpus use U+2212 as often as the hyphen, and a coordinate
#: parsed with the wrong sign is a coordinate in the other hemisphere.
_MINUS = r"[-‐‑‒–—−]"
_NUM = rf"{_MINUS}?\d+(?:\.\d+)?"

#: The APA shapes `statcheck` recognises (t, F, r, chi-square, Z, with exact or inexact
#: p), plus the neuroimaging-specific ones it has no reason to: a coordinate triple, a
#: cluster extent, a corrected threshold. The last three are what say an analysis
#: happened at all.
#: Three comma-separated numbers. Guarded by `_COORDINATE_CUE` below, because pubget
#: renders a citation list as "( 14 , 15 , 34 , 35 )", which is this shape and is not a
#: location.
_BARE_TRIPLE = re.compile(rf"(?<![\w.]){_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}(?![\w.])")

_STATISTIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("t", re.compile(rf"\bt\s*\(\s*\d+(?:\.\d+)?\s*\)\s*[=<>]\s*{_NUM}")),
    ("F", re.compile(rf"\bF\s*\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)\s*[=<>]\s*{_NUM}")),
    ("r", re.compile(rf"\br\s*(?:\(\s*\d+\s*\))?\s*[=<>]\s*{_MINUS}?\.?\d+(?:\.\d+)?")),
    ("chi2", re.compile(rf"(?:χ|chi)\s*2?\s*\([^)]*\)\s*[=<>]\s*{_NUM}", re.I)),
    ("Z", re.compile(rf"\b[Zz]\s*[=<>]\s*{_NUM}")),
    ("beta", re.compile(rf"(?:β|beta)\s*[=<>]\s*{_NUM}", re.I)),
    ("d", re.compile(rf"\b(?:Cohen's\s*)?d\s*[=<>]\s*{_NUM}")),
    ("eta2", re.compile(rf"(?:η|eta)\s*p?\s*2?\s*[=<>]\s*{_NUM}", re.I)),
    (
        "p",
        re.compile(
            r"\bp\s*(?:-?\s*value)?\s*[=<>≤≥]\s*\.?\d+(?:\.\d+)?" r"(?:\s*[eE]\s*-\s*\d+)?",
            re.I,
        ),
    ),
    ("CI", re.compile(rf"\d+\s*%\s*CI[^.;]{{0,40}}{_NUM}")),
    # Guarded by _COORDINATE_CUE below. pubget renders a citation list as "( 14 , 15 ,
    # 34 , 35 )", which is three comma-separated numbers and is not a location.
    ("coordinate", _BARE_TRIPLE),
    (
        "cluster",
        re.compile(
            r"\b(?:k\s*[=>]\s*\d+|\d+\s*(?:contiguous\s*)?voxels?|"
            r"cluster[- ](?:size|extent)[^.;]{0,30}\d+)",
            re.I,
        ),
    ),
    (
        "correction",
        re.compile(
            r"\b(?:FWE|FDR|Bonferroni|TFCE|"
            r"family[- ]wise|false discovery|small volume correction|"
            r"uncorrected|whole[- ]brain corrected)\b",
            re.I,
        ),
    ),
]


#: What has to be in the sentence before a comma-separated number triple is read as a
#: location. Prose that gives a coordinate says so; a reference list does not.
_COORDINATE_CUE = re.compile(
    r"\b(?:MNI|Talairach|ICBM|coordinates?\b|co-ordinates?\b|\bx\s*,\s*y\s*,\s*z\b|"
    r"peak|maxim(?:um|a)|centre of mass|center of mass|voxel|cluster|"
    r"located at|centred? (?:at|on)|centered? (?:at|on))",
    re.I,
)


#: `x = 9 y = -12 z = -6`, and the notations the corpus actually writes it in: with or
#: without `=`, with `:`, with thin spaces, upper or lower case. Measured over 610 papers,
#: 151 Results-prose matches and no false positive in an audited sample -- naming the axes
#: is itself the cue, so this form needs no `_COORDINATE_CUE` guard. The bare-comma form
#: cannot match it: `x = -42, y = 14, z = 8` has ", y = " between the numbers.
_AXIS_TRIPLE = re.compile(
    rf"\bx\s*[=:]?\s*({_NUM})[\s,;]+y\s*[=:]?\s*({_NUM})[\s,;]+z\s*[=:]?\s*({_NUM})",
    re.I,
)

#: A smoothing kernel is three numbers in millimetres and is not a place.
#: `FWHM(mm)=15.7, 15.7, 13.7` was matched as a location in the wider corpus.
_KERNEL_LABEL = re.compile(
    r"\b(?:FWHM|kernel|smooth(?:ed|ing)?|voxel siz\w+|resolution)\b[^.;]{0,24}$", re.I
)

#: How near the cue has to sit. "coordinated neural activity ... (104, 105, 106, 107)"
#: put a cue and a citation list in one sentence, and the whole-sentence test read them
#: as related. Measured over 600 studies outside the cue_reactivity corpus.
_CUE_WINDOW = 80

#: A bare triple straight after a Brodmann label is a list of areas, not a location.
#: `(BA 6, 8, 9, 10)` and `(BA 29, 30, 31)` are the commonest false positive in the corpus
#: -- but `BA = 46; -42, 17, 25` is a real coordinate, so the guard has to look at what
#: sits immediately before the numbers rather than anywhere in the sentence.
_AREA_LABEL = re.compile(r"\b(?:BA|Brodmann(?:\s+area)?s?|areas?)\s*$", re.I)


#: "[- 18, 15, 12]" is one coordinate, not a positive 18. Typesetting puts a space after
#: the minus and `_NUM` does not allow one, so the sign is silently dropped and the point
#: lands in the other hemisphere. `table_parse` normalises the same way for the same reason.
_LOOSE_SIGN = re.compile(rf"({_MINUS}|\+)\s+(?=[\d.])")


def _tighten(sentence: str) -> str:
    return _LOOSE_SIGN.sub(r"\1", sentence)


def coordinates_in(sentence: str) -> list[tuple[float, float, float]]:
    """Every stereotactic triple the sentence states, in either notation.

    Returns the values and not a boolean because an anchor needs the numbers: a prose
    coordinate is only interesting when no parsed table carries it, and that is a
    comparison against the table's points.

    Three rejections, each one a class measured in the corpus rather than imagined:
    a triple of probabilities or effect sizes (`p < 0.01, 0.001, 0.05`), a Brodmann area
    list (`BA 6, 8, 9`), and anything outside the bounding box a human brain occupies in
    either standard space.
    """

    sentence = _tighten(sentence)
    found: list[tuple[float, float, float]] = []
    for match in _AXIS_TRIPLE.finditer(sentence):
        trip = _triple(match.groups())
        if trip:
            found.append(trip)
    for match in _BARE_TRIPLE.finditer(sentence):
        before = sentence[: match.start()]
        if _AREA_LABEL.search(before) or _KERNEL_LABEL.search(before):
            continue
        near = sentence[max(0, match.start() - _CUE_WINDOW) : match.end() + 40]
        if not _COORDINATE_CUE.search(near):
            continue
        trip = _triple(re.findall(_NUM, match.group(0)))
        if trip and not _runs_on(trip):
            found.append(trip)
    return found


def _runs_on(values: tuple[float, float, float]) -> bool:
    """Three consecutive integers: a citation list, a session index, a Brodmann run.

    `[ 34 , 35 , 36 ]`, `proposed39,40,41`, `(104, 105, 106, 107)`, `NF session 1,2,3`.
    A real location can be three consecutive integers, and the corpus holds none -- the
    ratio of citation lists to (10, 11, 12) is not close.
    """
    return (
        all(float(v).is_integer() for v in values)
        and values[1] - values[0] == 1
        and values[2] - values[1] == 1
    )


def _triple(parts) -> tuple[float, float, float] | None:
    try:
        values = tuple(float(str(p).translate(_COORD_MINUS)) for p in parts)
    except (TypeError, ValueError):
        return None
    if len(values) != 3:
        return None
    if all(abs(v) < 1 for v in values):
        return None  # p-values and effect sizes, not a location
    if any(abs(v) > 120 for v in values):
        return None  # outside either standard space
    return values


#: The unicode dashes a typeset paper uses for a minus sign, which `float` will not read.
_COORD_MINUS = {ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"}


# ---------------------------------------------------------------- contrast candidates


# ------------------------------------------------------------------- method parameters


# ----------------------------------------------------------------------- cohort digest


# --------------------------------------------------------------------- region gazetteer


# ------------------------------------------------------------- BM25 sentence retrieval


#: BM25's saturation and length-normalisation terms, at the values the literature
#: gives them. Named rather than inlined because that is how they are cited.
_K1, _B = 1.5, 0.75


# ------------------------------------------------------------------- digest rendering

_CAUTION = (
    "Derived from the paper by regular expression, not read. It over-generates: some\n"
    "entries are not what they look like and some are duplicates of one fact. Treat it as\n"
    "a list of places to look, confirm every entry against the paper text below, and drop\n"
    "whatever the paper does not support. It adds nothing the paper does not contain, so\n"
    "it can never be the source for a value -- the paper is."
)


def _block(title: str, body: str) -> str:
    if not body.strip():
        return ""
    return f"\n## {title}\n\n{_CAUTION}\n\n{body.rstrip()}\n"


PROSE_COORD_TITLE = "Possible analyses reported only in prose"

PROSE_COORD_NOTE = (
    "Each sentence below states a coordinate that NO parsed result table carries. A cue\n"
    "sweep found them, not a parse, so every one is a PROPOSAL and not a finding. Decide\n"
    "each against the sentence, and emit nothing for the ones it does not support.\n"
    "\n"
    "Some are a result this paper reports in the text and in no table -- the case the\n"
    "table parse cannot reach, and the reason this list exists. Others are not results at\n"
    "all: a seed or sphere centre, an ROI taken from an atlas, or a peak quoted from\n"
    "another study to compare against. A coordinate this paper did not find is not this\n"
    "paper's analysis. Emit one only where the sentence says THIS study tested something\n"
    "and this is where it found it.\n"
    "\n"
    "`[in a table]` means a parsed table already reports that voxel. It does NOT mean the\n"
    "sentence is a duplicate: a table lists one contrast's peaks, and a sentence naming\n"
    "the same voxel for a DIFFERENT comparison is a second analysis, not a repeat of the\n"
    "first. Check which contrast the sentence names before deciding."
)


def prose_coordinates(
    text: str, known: Iterable[tuple[float, float, float]] = ()
) -> list[tuple[str, list[tuple[tuple[float, float, float], bool]]]]:
    """(sentence, [(coordinate, is it already in a parsed table)]) for locating sentences.

    A coordinate a table already carries is MARKED and not dropped. Dropping it discarded
    the case this exists for: 18823721 reports "right STN activation ... when contrasting
    heroin stimuli to neutral stimuli (x = 9 y = -12 z = -6)", and that voxel is in the
    Heroin>BL table under a different contrast. What the sentence adds is the comparison,
    not the number, and filtering on the number threw the comparison away.
    """
    seen = {tuple(round(float(v)) for v in k) for k in known}
    out = []
    for section in split_sections(text):
        if section.zone in ("back", "intro"):
            continue
        for sentence in sentences(section.text):
            found = [(c, tuple(round(v) for v in c) in seen) for c in coordinates_in(sentence)]
            if found:
                out.append((" ".join(sentence.split()), found))
    return out


#: How the parse names the statistics it reads off a table, so a prose point and a table
#: point describe their values the same way.
_STAT_KIND = {
    "Z": "z-statistic",
    "t": "t-statistic",
    "F": "f-statistic",
    "beta": "beta",
    "r": "correlation",
}

#: A space named in the sentence. `Analysis.coordinate_space` is authoritative over this,
#: but a point that knows its space is what lets the query engine compare it with others.
_SPACE = re.compile(r"\b(MNI(?:152)?|Talairach|ICBM)\b", re.I)


def prose_points(sentence: str) -> list[dict]:
    """Parse-shaped points for one sentence: coordinates, their space, their statistics.

    The statistic is paired to the coordinate it follows, which is how the sentences write
    it -- "x = 22, y = -3, z = -15, Z = 3.85; x = -16, y = -3 z = -19, Z = 4.01" is two
    points with one value each, not two points and two loose numbers.

    Statistic matches falling INSIDE a coordinate are dropped: `z = -6` is the third axis
    and also matches the Z-statistic pattern, and taking it would give every point a
    z-statistic equal to its own z coordinate.
    """
    sentence = _tighten(sentence)
    spans = _coordinate_spans(sentence)
    if not spans:
        return []
    space = _SPACE.search(sentence)
    stats = []
    for name, pattern in _STATISTIC_PATTERNS:
        kind = _STAT_KIND.get(name)
        if not kind:
            continue
        for match in pattern.finditer(sentence):
            if any(start <= match.start() < end for _c, start, end in spans):
                continue
            numbers = re.findall(_NUM, match.group(0).translate(_COORD_MINUS))
            if numbers:
                stats.append((match.start(), kind, float(numbers[-1])))

    points = []
    for index, (coord, _start, end) in enumerate(spans):
        following = spans[index + 1][1] if index + 1 < len(spans) else len(sentence)
        values = [
            {"value": value, "kind": kind} for at, kind, value in stats if end <= at < following
        ]
        points.append(
            {
                "coordinates": list(coord),
                "space": space.group(0).upper() if space else None,
                "values": values,
            }
        )
    return points


def _coordinate_spans(sentence: str) -> list[tuple[tuple[float, float, float], int, int]]:
    """Every accepted coordinate with where it sits, so statistics can be paired to it."""
    out = []
    for match in _AXIS_TRIPLE.finditer(sentence):
        trip = _triple(match.groups())
        if trip:
            out.append((trip, match.start(), match.end()))
    for match in _BARE_TRIPLE.finditer(sentence):
        before = sentence[: match.start()]
        if _AREA_LABEL.search(before) or _KERNEL_LABEL.search(before):
            continue
        near = sentence[max(0, match.start() - _CUE_WINDOW) : match.end() + 40]
        if not _COORDINATE_CUE.search(near):
            continue
        trip = _triple(re.findall(_NUM, match.group(0)))
        if trip and not _runs_on(trip):
            out.append((trip, match.start(), match.end()))
    return sorted(out, key=lambda row: row[1])


def prose_parse_entries(text: str, known: Iterable[tuple[float, float, float]] = ()) -> list[dict]:
    """Prose coordinate sentences, shaped as parse entries so they share one address space.

    The schema stores no coordinates -- `Analysis.source_table_analysis` is the only route
    from an analysis to its foci, and it addresses the parse. So a coordinate found in prose
    has to become a parse entry or it has nowhere to be stored: adding a coordinate field to
    the schema for prose alone would give the same fact two homes.

    `table_id` is "prose", which makes the keys `prose#1`, `prose#2` under the existing
    `<table_id>#<ordinal>` format, distinct from any real table's.
    """
    entries = []
    for sentence, found in prose_coordinates(text, known):
        entries.append(
            {
                "name": "",  # named by the extraction pass, from the sentence
                "description": sentence,
                "table_id": "prose",
                "table_number": None,
                "table_caption": sentence[:300],
                "table_footer": "",
                "from_prose": True,
                "points": _mark(prose_points(sentence), found),
            }
        )
    return entries


def _mark(points: list[dict], found) -> list[dict]:
    """Carry the "a table reports this voxel too" flag onto the parse-shaped points."""
    seen = {tuple(round(v) for v in coord): hit for coord, hit in found}
    for point in points:
        key = tuple(round(float(v)) for v in point["coordinates"])
        point["also_in_table"] = seen.get(key, False)
    return points


def prose_coordinate_block(text: str, known: Iterable[tuple[float, float, float]] = ()) -> str:
    rows = prose_coordinates(text, known)
    if not rows:
        return ""
    lines = []
    for sentence, found in rows[:30]:
        shown = "; ".join(
            "(" + ", ".join(f"{v:g}" for v in coord) + ")" + (" [in a table]" if hit else "")
            for coord, hit in found
        )
        lines.append(f"  {shown}\n      {sentence[:400]}")
    return _block(PROSE_COORD_TITLE, PROSE_COORD_NOTE + "\n\n" + "\n".join(lines))


# ------------------------------------------------------------------------- strategies


# ------------------------------------------------------------------------------- main
