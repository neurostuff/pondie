"""What `review/preprocess.py` must not get wrong.

Two classes of check, and the second is the one that matters. The first is that each
transform does what it says -- zones classified, sections dropped, sentences split. The
second is the invariant every arm of the preprocessing experiment rests on: a *text*
strategy adds nothing, a *digest* strategy removes nothing, and neither ever changes the
file `builder.py` resolves offsets against. If a transform could invent a sentence,
a value read out of the reduced prompt would have no warrant in the paper, and every
number in docs/text-preprocessing-experiments.md would be measuring something else.

The gold-derived expectations are all from `xevP8UDRAVh9`, the one human-verified record:
its two `Region`s are `frontal lobe` and `temporal lobe`, from a Methods sentence naming
both with one head noun, and its two text-only VBM analyses come from a single Results
sentence that reports a null result.
"""

from __future__ import annotations

import pytest

from pondie.extraction.prompt import preprocess

#: corpus file being present.
PAPER = """Title of the paper

An abstract sentence about gray matter (GM) volume in 14 patients.

## Introduction

Earlier work showed decreased GM in the prefrontal cortex ( 14 ).

## Materials and Methods

### Study sample

Fourteen (eight male, six female; mean age 40.7 +/- 6.8 years) non-left-handed patients
were recruited. Each patient was scanned twice.

### Image acquisition

Scanning used a 3T MRI scanner (Magnetom Verio, Siemens), with a repetition time of 2000
ms and an echo time of 3.4 ms. Images were smoothed with an 8 mm FWHM kernel using SPM8.
We used an explicit mask of the frontal and temporal lobe by WFU PickAtlas.

## Results

### Voxel-based morphometry analyses

Comparison of the heroin and placebo conditions found no significant difference in either
direction.

### Correlation analyses

There was a significant positive correlation between perfusion and GM volume (Table 1).

Table 1 - Correlation between gray matter and perfusion.

| Area             | MNI coordinates | Pearson r |
|------------------|-----------------|-----------|
| Precentral gyrus | 60, 16, 40      | 0.91      |

## Discussion

We found that perfusion correlated positively with GM ( 41 ).

## Conflict of Interest Statement

The authors declare no conflict.
"""


@pytest.fixture(scope="module")
def zones() -> dict[str, str]:
    return {section.heading: section.zone for section in preprocess.split_sections(PAPER)}


# ------------------------------------------------------------------------------- zones


def test_every_imrad_zone_is_found(zones):
    assert zones["Introduction"] == "intro"
    assert zones["Materials and Methods"] == "methods"
    assert zones["Results"] == "results"
    assert zones["Discussion"] == "discussion"
    assert zones["Conflict of Interest Statement"] == "back"


def test_front_matter_becomes_its_own_section():
    first = preprocess.split_sections(PAPER)[0]
    assert first.zone == "front" and "abstract sentence" in first.body


def test_a_weak_heading_inherits_its_parent_rather_than_its_own_keyword(zones):
    """ "Voxel-based morphometry analyses" is a Results subsection.

    A flat keyword match puts it in Methods on the word "analyses", and `sections` then
    drops the only place two of gold's six analyses are reported.
    """

    assert zones["Voxel-based morphometry analyses"] == "results"
    assert zones["Correlation analyses"] == "results"
    assert zones["Image acquisition"] == "methods"


def test_a_structured_abstract_label_is_front_matter_and_not_its_own_zone():
    labelled = "Title\n\n## Background:\n\nx\n\n## Results:\n\ny\n\n## Introduction\n\nz\n"
    zones = {s.heading: s.zone for s in preprocess.split_sections(labelled)}
    assert zones["Background:"] == "front" and zones["Results:"] == "front"
    assert zones["Introduction"] == "intro"


# -------------------------------------------------------------------------- invariants
#
# Each invariant is parametrized over the strategies it applies to, not over all of them
# with a `skip` inside. A strategy that transforms nothing has no text invariant to break,
# and reporting that as a skipped test says "this did not run" where the truth is "there
# was nothing here to run".


# ----------------------------------------------------------------------- the extractors


# ------------------------------------------------- the digest's slot names must be real


# ---------------------------------------------------------------------------
# The audit in docs/regex-audit.md, findings 1, 3 and 7.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk,mid",
    [
        ("Additionally, Li et al.", True),
        ("as reported previously (e.g.", True),
        ("opaque idioms (vs.", True),
        ("shown in Fig.", True),
        ("The scan lasted 12 min.", True),
        ("a whole sentence that ends here.", False),
        ("threshold was p < 0.001.", False),
        ("", False),
    ],
)
def test_a_period_after_an_abbreviation_is_not_a_sentence_end(chunk, mid):
    """The one abbreviation list in the repo, exported because `evidence/retrieval.py`
    cuts units on the same punctuation and had no guard at all."""

    assert preprocess.ends_mid_sentence(chunk) is mid


def test_the_sentence_splitter_still_uses_the_shared_guard():
    sentences = preprocess.sentences_of(
        "Activation was reported by Li et al. The cluster survived correction."
    )
    assert sentences == ["Activation was reported by Li et al. The cluster survived correction."]


# ------------------------------------------------------------------ prose coordinates

#: (sentence, does it state a location). The positives are the notations the corpus
#: actually writes, the negatives are the three things that share the shape. Every one is
#: copied from a cue_reactivity paper rather than invented, because the false positives
#: worth guarding against are the ones that occur.
COORDINATE_CASES = [
    # 18823721: the contrast that cost a gold inclusion. Axis-labelled, no commas between
    # the pairs, so the bare three-comma pattern cannot see it.
    (
        "right STN activation when contrasting heroin stimuli to neutral stimuli "
        "(x = 9 y = \u221212 z = \u22126; Z = 3.31)",
        True,
    ),
    # 32541652 writes thin spaces around the equals signs.
    ("the OFC (x \u2009=\u2009\u221235, y \u2009=\u200932, z \u2009=\u2009\u22126)", True),
    ("center of mass at ( X \u221244, Y \u221220, Z 32)", True),  # 16123763: no equals
    ("right NAcc ([x:16, y:8, z:\u221210]; Z = 3.35)", True),  # 31113931: colons
    ("MNI coordinates of the maximum voxel = [\u2212 18, 15, 12], z = 3.66", True),
    # A Brodmann label carrying its own number, then a real coordinate. The area guard
    # must look at what precedes the numbers, not for "BA" anywhere in the sentence.
    ("DLPFC (BA = 46; \u221242, 32, 17) peak voxel", True),
    ("as reported previously ( 14 , 15 , 34 )", False),  # pubget citation list
    ("These included the frontal (BA 6, 8, 9, 10), anterior cingulate", False),
    ("the anterior cingulate (BA 29, 30, 31), temporal (BA 20, 21, 22)", False),
    ("significantly higher levels ( P < 0.01, 0.001, 0.05, respectively)", False),
    ("effect sizes: d = 0.42, 0.37, 0.53, respectively", False),
    ("clusters at 300, 400, 500 mm", False),  # outside either space
]


@pytest.mark.parametrize("sentence,is_location", COORDINATE_CASES)
def test_a_prose_coordinate_is_told_from_the_numbers_that_look_like_one(sentence, is_location):
    """The table parse cannot see a result reported only in prose, and 18823721 lost a
    gold inclusion to one. Reading them back means telling a location from a citation
    list, a Brodmann enumeration and a row of p-values, all of which are three numbers."""
    assert bool(preprocess.coordinates_in(sentence)) is is_location


def test_the_axis_form_needs_no_cue_word():
    """Naming the axes is the cue. Requiring MNI/peak/voxel as well would drop the
    18823721 sentence, which says only "contrasting heroin stimuli to neutral stimuli"."""
    assert preprocess.coordinates_in("activation at x = 9 y = \u221212 z = \u22126")


def test_a_bare_triple_still_needs_one():
    """Without the guard a citation list is a coordinate, which is why the guard exists."""
    assert not preprocess.coordinates_in("as shown previously ( 14 , 15 , 34 )")
    assert preprocess.coordinates_in("the peak voxel ( 14 , 15 , 34 )")


#: Found by running the extractor over ~800 studies from the 39,273-study ns-pond corpus,
#: across the ace, pubget and elsevier renderings, scanning whole documents rather than
#: Results alone. cue_reactivity showed none of these: a corpus of one topic, read only in
#: its Results sections, is not where a three-number pattern goes wrong.
WIDE_CORPUS_CASES = [
    # ACE flattens superscript citation markers into the word before them, so the digits
    # are glued to a letter and a `(?<![\d.])` lookbehind lets them through.
    ("nonlocal algorithms that operate on a single voxel were proposed39,40,41,42,43.", False),
    ("deterministic tractography16,23,24, a method that fits a tensor at each voxel", False),
    # "coordinated" satisfied a cue that meant to say "coordinate".
    ("a modulatory effect on coordinated neural activity (104, 105, 106, 107).", False),
    (
        "registered to anatomical images (FLIRT, registration [ 34 , 35 , 36 ]), smoothed voxel",
        False,
    ),
    ("increased connectivity (FWHM(mm)=15.7, 15.7, 13.7, volume=48619 voxels)", False),
    ("the 3rd NF run minus the 1st NF run of NF session 1,2,3; cluster corrected", False),
    # Must still be found: the cue sits well before the numbers in real reporting.
    ("Peak activation at left SMG −56, −50, 26, right AG 48, −62, 26", True),
    ("the local maximum nearest the group FFA maximum (42, -51, -20)", True),
    ("Talairach coordinates: 0, 40, 0) and presented in neurological convention", True),
]


@pytest.mark.parametrize("sentence,is_location", WIDE_CORPUS_CASES)
def test_the_wider_corpus_false_positives_stay_rejected(sentence, is_location):
    """Every negative here was matched as a coordinate before its guard existed.

    Auditing 30 ACE matches found 7 wrong, which is the rate a prose anchor would have
    carried into the record. The guards took ACE from 282 matches over 300 studies to 222,
    and the audited error rate from roughly a quarter to under a tenth.
    """
    assert bool(preprocess.coordinates_in(sentence)) is is_location


def test_three_consecutive_integers_are_a_list_not_a_place():
    """`[ 34 , 35 , 36 ]`, `proposed39,40,41`, `session 1,2,3`. A location can be three
    consecutive integers and the corpus holds none; citation lists are everywhere."""
    assert not preprocess.coordinates_in("peak voxel references [ 34 , 35 , 36 ]")
    assert preprocess.coordinates_in("peak voxel at [ 34 , 35 , 40 ]")


def test_a_coordinate_a_table_carries_is_marked_not_dropped():
    """18823721 is why. It reports "right STN activation ... when contrasting heroin
    stimuli to neutral stimuli (x = 9 y = -12 z = -6)", and that voxel is also in the
    Heroin>BL table under a different contrast. Dropping the sentence because the number
    was already known discarded the comparison, which is the only thing it added -- and
    that comparison is the paper's sole qualifying cue>control contrast."""
    sentence = (
        "In opioid-dependent subjects, right STN activation was also observed when "
        "contrasting heroin stimuli to neutral stimuli ( x = 9 y = −12 z = −6)."
    )
    rows = preprocess.prose_coordinates(sentence, known=[(9, -12, -6)])
    assert rows, "the sentence was dropped because a table already held its coordinate"
    ((_sentence, found),) = rows
    assert found == [((9.0, -12.0, -6.0), True)], "the overlap must be marked, not silent"


def test_a_coordinate_no_table_carries_is_unmarked():
    rows = preprocess.prose_coordinates(
        "The peak was at x = 9 y = −12 z = −6.", known=[(40, 40, 40)]
    )
    ((_sentence, found),) = rows
    assert found == [((9.0, -12.0, -6.0), False)]


def test_prose_entries_share_the_parse_address_space_without_shifting_it():
    """A prose coordinate has to become a parse entry or it has nowhere to be stored.

    `Analysis.source_table_analysis` is the only route from an analysis to its foci and it
    addresses the parse; the schema stores no coordinates itself. So prose entries take
    keys in the same `<table_id>#<ordinal>` space -- and must not renumber the table
    entries, because a key that exists but is wrong attaches an analysis to another
    contrast's coordinates, which is worse than one that is missing.
    """
    from pondie.formats import parse_keys

    parsed = [{"table_id": "tbl1"}, {"table_id": "tbl1"}, {"table_id": "tbl2"}]
    entries = preprocess.prose_parse_entries(
        "The peak was at x = 9 y = −12 z = −6 for heroin versus neutral."
    )
    assert entries and entries[0]["table_id"] == "prose"
    assert entries[0]["points"][0]["coordinates"] == [9.0, -12.0, -6.0]

    before = parse_keys.parse_keys(parsed)
    after = parse_keys.parse_keys([*parsed, *entries])
    assert after[: len(before)] == before, "appending prose renumbered the table entries"
    assert after[len(before) :] == ["prose#1"]


def test_a_statistic_is_paired_to_the_coordinate_it_follows():
    """A prose point should carry what a table point carries. The sentences write the
    value after the location, so pairing is positional -- and the third axis is the trap:
    `z = -6` matches the Z-statistic pattern too, and taking it would give every point a
    z-statistic equal to its own z coordinate."""
    points = preprocess.prose_points(
        "both bilateral amygdala ( x = 22, y = −3, z = −15, Z = 3.85; "
        "x = −16, y = −3 z = −19, Z = 4.01)"
    )
    assert [p["coordinates"] for p in points] == [[22.0, -3.0, -15.0], [-16.0, -3.0, -19.0]]
    assert [p["values"] for p in points] == [
        [{"value": 3.85, "kind": "z-statistic"}],
        [{"value": 4.01, "kind": "z-statistic"}],
    ]


def test_a_space_between_the_minus_and_the_digits_keeps_the_sign():
    """ "[- 18, 15, 12]" is one coordinate, not a positive 18. Typesetting puts a space
    after the minus; a coordinate parsed with the wrong sign is in the other hemisphere,
    and `table_parse` normalises the same way for the same reason."""
    points = preprocess.prose_points("MNI coordinates of the peak voxel = [− 18, 15, 12]")
    assert points[0]["coordinates"] == [-18.0, 15.0, 12.0]
    assert points[0]["space"] == "MNI"


def test_a_prose_point_without_a_stated_space_leaves_it_open():
    """`Analysis.coordinate_space` is authoritative; guessing here would compete with it."""
    points = preprocess.prose_points("the peak was at x = 9 y = −12 z = −6")
    assert points[0]["space"] is None
