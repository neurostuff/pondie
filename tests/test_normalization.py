"""One test per shape, on the cases that were wrong before they were rules."""

from pathlib import Path

import pytest

from pondie.normalization import (
    coordinate_space,
    handedness_distribution,
    medication_status,
    modality,
    multiple_comparison_method,
    prespecification,
)
from pondie.vocabularies import folding


def test_a_space_naming_both_is_unknown_rather_than_a_guess():
    assert coordinate_space.normalize("MNI152").value == "MNI"
    ambiguous = coordinate_space.normalize("MNI/TAL")
    assert ambiguous.value == "UNKNOWN" and "MNI and TAL" in ambiguous.reason


def test_other_and_unknown_are_not_the_same_claim():
    assert coordinate_space.normalize("fsaverage").value == "OTHER", "a third space"
    assert coordinate_space.normalize("").value == "UNKNOWN", "no information"


def test_negation_decides_medication_status():
    for text in (
        "not medicated",
        "no longer receiving medication",
        "unmedicated",
        "free of psychotropic medication",
    ):
        assert medication_status.normalize(text).value == "FREE", text
    assert medication_status.normalize("on stable antipsychotic medication").value == "MEDICATED"


def test_a_negation_in_a_later_clause_does_not_invert_the_cohort():
    text = "Most patients were taking antidepressant medication; no medication changes"
    assert medication_status.normalize(text).value == "MEDICATED"


def test_non_left_handed_is_not_left_handed():
    assert handedness_distribution.normalize("non-left-handed").value == "RIGHT"
    assert handedness_distribution.normalize("left-handed").value == "LEFT"


def test_cluster_level_names_a_unit_not_an_error_family():
    assert multiple_comparison_method.normalize("cluster correction").value == "OTHER"
    assert multiple_comparison_method.normalize("family-wise error (FWE)").value == "FWE"
    assert multiple_comparison_method.normalize("uncorrected").value == "UNCORRECTED"


def test_a_missing_parser_is_an_error_not_an_unreported_field(monkeypatch):
    """Without a parse the field read UNKNOWN, which is what a silent paper reads too.

    A broken environment was therefore indistinguishable from a corpus that stopped reporting
    medication. It raises now, naming the package and the install.
    """

    from pondie import _deps
    from pondie.normalization import _negation

    _negation._parser.cache_clear()
    monkeypatch.setattr(
        _deps.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name))
    )
    with pytest.raises(_deps.MissingDependency, match="spacy"):
        medication_status.normalize("patients were medicated")
    _negation._parser.cache_clear()


# ---------------------------------------------------------------------------
# The audit in docs/regex-audit.md, finding 6.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("naïve", "naive"),
        ("Étude", "etude"),
        ("Möbitz II", "mobitzii"),
        ("Müllerian", "mullerian"),
        ("gray matter", "graymatter"),
        ("Alzheimer's disease", "alzheimersdisease"),
    ],
)
def test_squash_folds_an_accent_rather_than_deleting_it(value, expected):
    """`squash` said it was "`fold` with the spaces removed" and skipped `fold`'s NFKD step,
    so an accented letter fell out of `[a-z0-9]` and was dropped: `naïve` squashed to
    `nave`. 245 of 160,565 MONDO and Cognitive Atlas surface forms are affected."""

    assert folding.squash(value) == expected


def test_squash_is_exactly_fold_without_the_spaces():
    for value in ["naïve", "Étude", "first-episode schizophrenia", "ADHD"]:
        assert folding.squash(value) == folding.fold(value).replace(" ", "")


def test_one_name_spelled_two_ways_keys_as_one():
    """The caller is exact-key grouping, so a deleted accent turned one name into two
    categories -- the failure the grouping exists to prevent."""

    from pondie.normalization import task as module

    def one(name):
        return module.Task(
            study="s", name=name, description="d", instructions="",
            design_type="", response_modality="", performance_measures="",
            conditions=(), stimulus_content=(),
        )

    assert module.name_key(one("drug naïve patients")) == module.name_key(
        one("drug naive patients")
    )


# ------------------------------------------------- the stimulus facet, stated not guessed


def _task(**over):
    from pondie.normalization.task import Task

    base = dict(
        study="s",
        name="cue reactivity task",
        description="d" * 80,
        instructions="",
        design_type="block",
        response_modality="button",
        performance_measures="",
        conditions=("cue", "neutral"),
        stimulus_content=(),
    )
    return Task(**{**base, **over})


def test_a_task_from_a_record_without_the_slot_has_no_stimulus():
    """Every record in the corpus predates `Condition.stimulus_content`, so absence is the
    normal case and nothing may require the field."""
    assert _task().stimulus == "none"


def test_the_stated_stimulus_is_pooled_over_the_conditions():
    assert _task(stimulus_content=("alcohol cues", "neutral images")).stimulus == (
        "alcohol cues. neutral images"
    )


def test_the_stimulus_stays_out_of_the_paradigm_channels():
    """`apparatus` measures how the task ran and `prose` what it was. The facet split is
    only worth having if the stimulus separates tasks on neither: alcohol and food cue
    reactivity differ here and must still cluster together."""
    task = _task(stimulus_content=("alcohol cues",))
    assert "alcohol" not in task.apparatus
    assert "alcohol" not in task.prose


def test_the_facet_is_the_stated_one_or_nothing():
    """A 58-term lexicon used to infer this from words in a name. It is gone: a guess is the
    wrong shape for a query axis, because it is confidently wrong on a name whose paper said
    otherwise and silent on a paper that said so plainly in its Methods."""
    from pondie.normalization import task as module
    assert module.stimulus_of(_task(stimulus_content=("cigarette cues",))) == "cigarette cues"
    # The name says alcohol and no slot is filled. Nothing is inferred from it.
    assert module.stimulus_of(_task(name="alcohol cue reactivity task")) == "(unspecified)"


# ------------------------------------------- one module, and it seeds before it clusters


def test_only_one_module_normalizes_tasks():
    """There were two: a seeded one reachable only from a script, and an unseeded one that
    `pondie normalize task` ran because it was the one exposing `normalize`. They
    disagreed -- the unseeded route merged `novelty oddball task`, `Go/No-go tasks` and
    `sustained attention task` into `stop signal task`, which the Atlas names apart.

    Asserted as absence, because "both exist and one is preferred" is the state this is
    meant to prevent and a test that only checks the survivor would pass in it.
    """

    import importlib.util
    from pathlib import Path

    from pondie import normalization
    from pondie.normalization import task

    package = Path(normalization.__file__).parent
    others = sorted(
        p.name for p in package.glob("*.py")
        if p.stem not in {"task", "atlas"} and "task" in p.stem
    )
    assert others == [], f"a second task module is back: {others}"
    assert importlib.util.find_spec("pondie.normalization._clustering") is None

    # the whole workflow is reachable from the one module, so no script is needed
    assert "task" in normalization.fields()
    for step in ("Seeds", "categorise", "paradigm_distances", "normalize", "report", "main"):
        assert callable(getattr(task, step, None)), f"{step} is not on the module"


def test_the_seed_match_expands_the_papers_own_abbreviations():
    """`normalise` has always been able to expand; `Seeds.match` passed it nothing.

    So a task the paper named only by its short form reached a list of expanded Atlas
    labels as an acronym and matched nothing. Over the 100-paper defect set `SVF test`
    is the case: the paper defines it as semantic verbal fluency, and the Atlas carries
    `verbal fluency task`.
    """

    from pondie.normalization import task
    from pondie.vocabularies.abbreviations import Abbreviations

    scoped = Abbreviations().for_paper(
        "We ran the semantic verbal fluency (SVF) test in the scanner.", "p1"
    )
    assert task.normalise("SVF test") == "SVF test", "no store, no expansion"
    assert "verbal fluency" in task.normalise("SVF test", scoped).lower()


def test_a_store_a_caller_holds_cannot_reach_past_its_own_paper():
    """`FA` is fractional anisotropy in one paper and flip angle in another.

    There was a module-level `_STORE` here holding the whole corpus file, and `normalise`
    looked up `(short, paper)` in it. Lookups were scoped so nothing leaked, but a global
    abbreviation list is the shape this repository has ruled out -- and it never read the
    paper, so a definition present only in the paper's own Methods resolved to nothing.
    `paper_stores` returns stores that each ARE one paper's, so reaching past one is not
    possible rather than merely not done.
    """

    from pondie.vocabularies.abbreviations import Abbreviations

    corpus = Abbreviations()
    corpus.learn("fractional anisotropy (FA) was computed", paper="p1")
    corpus.learn("the flip angle (FA) was 90 degrees", paper="p2")

    one = corpus.for_paper("fractional anisotropy (FA) was computed", "p1")
    assert one.paper == "p1"
    assert "fractional anisotropy" in (one.expand("FA") or "").lower()
    # p2's definition is not in p1's store at all
    assert one.expand("FA", "p2") is None


def test_no_paper_text_means_no_expansion_rather_than_a_guess():
    """A study with no text on disk gets no store at all, and `normalise` with no store
    leaves the name alone. A missing corpus must not silently fall back to whatever some
    other paper meant."""

    from pondie.normalization import task

    assert task.paper_stores({"nosuchstudy"}, None) == {}
    assert task.paper_stores({"nosuchstudy"}, Path("/nonexistent")) == {}
    assert task.normalise("SVF test", None) == "SVF test"


def test_rescue_attaches_a_singleton_and_reports_what_it_moved():
    """Off by default (`--rescue 0`), so it had no test and an unread `tasks` parameter.

    Average linkage votes down a task adjacent to one member of a large cluster, which is
    what this exists to undo -- and it now returns what it moved instead of printing it,
    because `categorise` is a library call.
    """
    import numpy as np

    from pondie.normalization import task

    #    0,1 are a category; 2 is a singleton sitting close to 0
    labels = {0: 7, 1: 7, 2: 9}
    sizes = {7: 2, 9: 1}
    d = np.array([[0.0, 0.1, 0.2],
                  [0.1, 0.0, 0.9],
                  [0.2, 0.9, 0.0]], dtype="float32")

    out, moved = task.rescue(dict(labels), sizes, d, threshold=0.70)
    assert moved == {2: 7}, "0.2 apart is 0.8 similar, over the bar"
    assert out[2] == 7

    out, moved = task.rescue(dict(labels), sizes, d, threshold=0.90)
    assert moved == {} and out[2] == 9, "under the bar it stays a singleton"


def test_a_task_no_longer_carries_a_slot_nothing_reads():
    """`Task.stimuli` was read off every record and used only by a channel that excluded
    it. The dataclass is the reader's contract, so an unread field is a claim that the
    clustering uses something it does not."""

    from pondie.normalization import task

    assert not hasattr(task.Task, "setting")
    assert "stimuli" not in task.Task.__dataclass_fields__
    assert "stimulus_content" in task.Task.__dataclass_fields__


# -- the two slots that were closed enums ----------------------------------
#
# `Acquisition.modality` and `Analysis.prespecification` were required closed enums, so one
# unrecognised word discarded the whole entity: `create` answered "Acquisition would be
# missing modality" and everything else the proposal carried went with it. Both are open
# vocabularies now and these map the wording back on.


@pytest.mark.parametrize(
    "wording,expected",
    [
        ("functional MRI", "fMRI"),
        ("functional magnetic resonance imaging", "fMRI"),
        ("structural MRI", "sMRI"),
        ("voxel-based morphometry", "sMRI"),
        ("diffusion tensor imaging", "dMRI"),
        ("near-infrared spectroscopy", "fNIRS"),
        ("positron emission tomography", "PET"),
    ],
)
def test_a_modality_written_out_reaches_its_vocabulary_value(wording, expected):
    assert modality.normalize(wording).value == expected


def test_a_qualified_mri_is_not_also_the_bare_one():
    """ "functional MRI" contains "MRI", so without the lookbehind both rules match and
    `classify` reads two distinct answers as an ambiguity -- making the commonest wording in
    the corpus the one value this cannot resolve. The bare value stays reachable."""
    assert modality.normalize("functional MRI").value == "fMRI"
    assert modality.normalize("structural MRI").value == "sMRI"
    assert modality.normalize("MRI").value == "MRI"
    assert modality.normalize("magnetic resonance imaging").value == "MRI"


def test_an_acronym_does_not_leak_into_the_bare_mri_rule():
    """`\bmri\b` must not reach the MRI inside "fMRI": one token, no boundary. If it did,
    every acronym would match two rules and answer UNKNOWN."""
    for acronym, expected in (("fMRI", "fMRI"), ("sMRI", "sMRI"), ("dMRI", "dMRI")):
        assert modality.normalize(acronym).value == expected


def test_a_modality_no_rule_covers_is_reported_rather_than_guessed():
    """Arterial spin labelling is a real gap and the vocabulary has `other`, which asserts
    the paper placed itself outside the named modalities. It did not say that, so this is
    UNKNOWN and appears in the residual -- which is how the rule gets written."""
    decision = modality.normalize("arterial spin labelling")
    assert decision.value == "UNKNOWN"
    assert decision.reason == "unmatched"


@pytest.mark.parametrize(
    "wording,expected",
    [
        ("post-hoc", "exploratory"),
        ("post hoc", "exploratory"),
        ("data-driven", "exploratory"),
        ("confirmatory", "planned"),
        ("planned comparisons", "planned"),
        ("hypothesis-driven", "planned"),
        ("registered report", "preregistered"),
        ("pre-registered on OSF", "preregistered"),
    ],
)
def test_a_prespecification_synonym_reaches_its_value(wording, expected):
    assert prespecification.normalize(wording).value == expected


def test_a_negated_preregistration_is_exploratory_not_ambiguous():
    """ "not preregistered" contains "preregistered", so the two compete and the negation has
    to win -- the case `_lexicon.Rule.decisive` exists for, met here as it is by medication."""
    decision = prespecification.normalize("analyses were not preregistered")
    assert decision.value == "exploratory"
    assert decision.reason == "decisive"


def test_prespecification_has_no_third_answer():
    """The field asks when the contrast was decided, so there is no value outside the two --
    wording that does not say is UNKNOWN, and OTHER would assert a third kind."""
    assert "OTHER" not in prespecification.VALUES
    assert prespecification.normalize("we decided later").value == "UNKNOWN"


def test_a_separator_class_admits_the_underscore_the_enum_uses():
    """`whole_brain` is what the slot holds 212 times in 1,817 records, and a class of
    space-or-hyphen matched none of them -- the field answered half of what it saw."""
    from pondie.normalization import correction_scope

    for text in ("whole_brain", "whole brain", "whole-brain"):
        assert correction_scope.normalize(text).value == "WHOLE_BRAIN", text
    assert correction_scope.normalize("region_of_interest").value == "RESTRICTED"
    assert correction_scope.normalize("searchlight").value == "OTHER", "a geometry, not a volume"


def test_a_space_name_is_matched_without_a_trailing_boundary():
    """ "MNI152", "ICBM152" and "fsaverage6" are each one token."""
    for text, expected in (("MNI152", "MNI"), ("ICBM152", "MNI"), ("fsaverage6", "OTHER")):
        assert coordinate_space.normalize(text).value == expected, text


def test_a_template_a_study_built_for_itself_is_a_third_space():
    """OTHER refuses the transform; UNKNOWN lets a caller default, so the two must not swap."""
    for text in (
        "customized template",
        "in-house DARTEL template",
        "SUIT template space",
        "dementia-specific SPM FDG-PET template",
        "SPM5 template",
    ):
        assert coordinate_space.normalize(text).value == "OTHER", text
    for text in ("template image space", "reference atlas"):
        assert coordinate_space.normalize(text).value == "UNKNOWN", text


def test_a_space_spelled_out_reaches_the_same_answer_as_its_acronym():
    assert coordinate_space.normalize("International Consortium for Brain Mapping").value == "MNI"
    assert coordinate_space.normalize("Colin27 Brain").value == "MNI", "the MNI single subject"
    assert coordinate_space.normalize("Talaraich").value == "TAL", "a transposition in the corpus"


# ---------------------------------------------- the derived siblings, and the one seam


def _group(**over):
    base = {
        "age_mean": {"value": 24.0, "extraction_status": "extracted"},
        "age_unit": {"value": "yrs", "extraction_status": "extracted"},
        "sex_distribution": [
            {"category": {"value": "Females", "extraction_status": "extracted"}}
        ],
        "handedness_distribution": [
            {"category": {"value": "right handed", "extraction_status": "extracted"}}
        ],
    }
    return {**base, **over}


def test_a_derived_sibling_keeps_the_sources_own_wording():
    """The point of a sibling rather than an overwrite: `category` is the study's words and
    the schema says so, and a query needs the folded value. Both, or the field loses one."""

    from pondie.normalization import sex_distribution

    record = {"groups": [_group()]}
    sex_distribution.apply(record)
    entry = record["groups"][0]["sex_distribution"][0]

    assert entry["category"]["value"] == "Females", "the paper's wording survives"
    assert entry["category_normalized"]["value"] == "FEMALE"
    assert entry["category_normalized"]["value_source"] == "generated"


def test_the_age_unit_of_a_group_with_no_age_is_left_unset():
    """UNKNOWN says a normalizer read a wording and could not place it. A group with no age
    has no wording to read, and writing UNKNOWN there claims the paper gave a unit."""

    from pondie.normalization import age_unit

    record = {"groups": [{"name": {"value": "controls", "extraction_status": "extracted"}}]}
    tally = age_unit.apply(record)

    assert "age_unit_normalized" not in record["groups"][0]
    assert tally["skipped"] == 1 and tally["set"] == 0


def test_gestational_weeks_beats_weeks():
    """`gestational weeks` matches both rules, and `classify` reads two distinct matches as
    an ambiguity -- so the commonest way to write the unit answered UNKNOWN until the
    specific claim was made decisive. The same case `modality` has with `functional MRI`."""

    from pondie.normalization import age_unit

    assert age_unit.normalize("gestational weeks").value == "gestational_weeks"
    assert age_unit.normalize("postmenstrual age").value == "gestational_weeks"
    assert age_unit.normalize("weeks").value == "weeks"


def test_apply_derived_runs_every_module_that_exposes_apply():
    """The list and the package cannot drift: `derived_fields()` is derived from the
    modules, `DERIVED` is the order they run in, and the two must name the same set.

    `DERIVED` named `group_role` after that module was removed, which would have raised on
    the first record. Nothing caught it because nothing compared the two.
    """

    from pondie import normalization

    assert set(normalization.DERIVED) == set(normalization.derived_fields())


def test_apply_derived_fills_every_derived_slot_on_one_record():
    from pondie import normalization

    record = {"groups": [_group()]}
    tallies = normalization.apply_derived(record)
    group = record["groups"][0]

    assert set(tallies) == set(normalization.DERIVED)
    assert group["age_unit_normalized"]["value"] == "years"
    assert group["sex_distribution"][0]["category_normalized"]["value"] == "FEMALE"
    assert group["handedness_distribution"][0]["category_normalized"]["value"] == "RIGHT"


def test_every_derived_slot_is_declared_on_the_extraction_schema():
    """`is_healthy` was written onto 4,133 groups the schema did not declare, and the
    validator reported every one. A derived slot needs its `add_slot` deviation or it
    repeats that."""

    from pondie import schema
    from pondie.schema import reader

    extraction = reader.load(schema.EXTRACTION)
    assert extraction.attributes("Group").get("age_unit_normalized") is not None
    assert extraction.attributes("CategoryDistribution").get("category_normalized") is not None


def test_a_group_whose_age_was_not_reported_has_no_unit_either():
    """The sibling test above, for the shape the corpus actually holds: a slot the paper
    did not report arrives as a wrapper with no `value`, which reads as the NOT_REPORTED
    sentinel, and a sentinel is not None. 711 of the 3951 groups this filled across the
    1817-record corpus had no age at all and were being told their absent age was in an
    unreadable unit."""

    from pondie.normalization import age_unit

    not_reported = {"extraction_status": "not_reported"}
    record = {"groups": [{"age_mean": not_reported, "age_median": not_reported}]}
    tally = age_unit.apply(record)

    assert "age_unit_normalized" not in record["groups"][0]
    assert tally["skipped"] == 1 and tally["set"] == 0


@pytest.mark.parametrize(
    "wording,expected",
    [
        # every one of these is a value the 1817-record corpus holds and no rule reached
        ("y", "years"),
        ("R", "RIGHT"),
        ("L", "LEFT"),
        ("male patients", "MALE"),
        ("female patients", "FEMALE"),
        ("planned", "planned"),
        ("hypothesis-led", "planned"),
        ("random Gaussian fields", "FWE"),
        ("3dClustSim", "FWE"),
        ("clusterwise correction", "OTHER"),
    ],
)
def test_a_wording_the_corpus_holds_reaches_its_value(wording, expected):
    from pondie.normalization import age_unit, sex_distribution

    for module in (age_unit, handedness_distribution, sex_distribution,
                   prespecification, multiple_comparison_method):
        decision = module.normalize(wording)
        if decision.value == expected:
            return
    raise AssertionError(f"{wording!r} reached no rule for {expected}")


def test_a_category_beside_a_second_one_is_still_not_an_answer():
    """The trailing role noun is an allowance, not a relaxation of the anchors: `men and
    women` names both and must stay UNKNOWN."""

    from pondie.normalization import sex_distribution

    assert sex_distribution.normalize("men and women").value == "UNKNOWN"
    assert sex_distribution.normalize("male patients").value == "MALE"


def test_one_tool_named_twice_is_not_two_answers():
    """AFNI's AlphaSim IS a Monte Carlo simulation, so a paper naming both matched the
    PERMUTATION rule and the FWE rule at once and `classify` read one tool as an ambiguity.
    40 values in the corpus write it this way."""

    for wording in ("AlphaSim Monte Carlo simulation", "Monte Carlo simulations (AlphaSim)",
                    "Monte-Carlo simulations using 3dClustSim"):
        assert multiple_comparison_method.normalize(wording).value == "FWE"


def test_a_family_wise_threshold_derived_by_resampling_is_the_resampling():
    """Rule order could not deliver this: `classify` consults order only when one of the
    matches is OTHER, so two answers were an ambiguity however they were ordered."""

    for wording in ("family-wise error correction using permutations",
                    "FWE-corrected after using threshold-free cluster enhancement",
                    "family-wise error rate corrected permutation test"):
        assert multiple_comparison_method.normalize(wording).value == "PERMUTATION"
    # two error rates named is still two answers, and neither is the specific claim
    assert multiple_comparison_method.normalize("FWE and FDR").value == "UNKNOWN"
