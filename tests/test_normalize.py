"""What the normalization and cross-corpus query must not get wrong.

A wrong mapping is worse than a missing one. A missing mapping is visible -- the row says
no match and someone looks. A wrong one is queried across a corpus and believed, and the
paper's own wording that would have exposed it is sitting in a field nobody re-reads. Therefore,
most of these tests are about refusing, not about matching.
"""

from __future__ import annotations

import pytest

from pondie import paths  # noqa: E402
from pondie.normalization import contrasts as q  # noqa: E402
from pondie.vocabularies import onvoc as nz  # noqa: E402


@pytest.fixture(scope="module")
def onvoc():
    """The real ONVOC, which is fetched rather than committed.

    `data/vocab/` is documented as "fetched, none in git", so on a checkout that has not
    fetched it these tests have no vocabulary to run against. They are not written against a
    stand-in because several of them assert something about ONVOC's actual content -- that
    exactly one label has the initials MDD, that `Wechsler Abbreviated Scale of Intelligence`
    sits under `Tests` -- and a synthetic vocabulary would turn those into assertions about
    the fixture.

    Skipping states that; erroring on a missing file did not.
    """
    path = paths.VOCAB / "onvoc.json"
    if not path.exists():
        pytest.skip(f"{path} not fetched; see docs/pipeline-architecture.md for data/vocab")
    return nz.load_onvoc()


# --- surface forms ----------------------------------------------------------


def _wrapped(value):
    """A realistic `ExtractedValue`.

    `extraction_status` is what makes a mapping a wrapper -- `values.read` keys on it, and
    the builder's `repair_wrappers` exists to guarantee it. These fixtures used a bare
    `{"value": ...}`, a shape that appears in **zero** of the 128 wrappers in a shipped
    record, so they were exercising an unwrapper more permissive than the one production
    uses and would have passed against a reader that was wrong about real data.
    """
    return {"extraction_status": "extracted", "value": value}


def test_a_parenthetical_acronym_is_its_own_candidate():
    got = nz.surface_forms("Autism Diagnostic Observation Schedule (ADOS)")
    assert "ADOS" in got
    assert "Autism Diagnostic Observation Schedule" in got


def test_laterality_is_stripped_but_only_after_the_whole_phrase_is_tried():
    got = nz.surface_forms("left anterior insula")
    assert got[0] == "left anterior insula"
    assert any("insula" == v.strip().lower() for v in got)


def test_a_phrase_of_only_qualifiers_keeps_its_words():
    # Stripping every content word would leave nothing to look up, and an empty query
    # matches whatever is shortest.
    assert nz.surface_forms("left right") == ["left right"]


# --- acronyms ---------------------------------------------------------------


def test_an_apostrophe_does_not_manufacture_an_initial():
    # Folding `Alzheimer's Disease` leaves a stray `s`, which turned a two-word name
    # into the three-letter ASD -- a different disorder entirely.
    assert nz.acronym("Alzheimer's Disease") == ""


def test_domain_nouns_carry_a_letter():
    # `disorder` and `scale` are exactly the words a clinical acronym is built from.
    assert nz.acronym("Autism Spectrum Disorder") == "asd"


def test_a_two_word_label_has_no_acronym():
    assert nz.acronym("Drug Use") == ""


def test_an_uncorroborated_acronym_is_refused(onvoc):
    # ONVOC has exactly one label whose initials are MDD and it is Mood Dysregulation
    # Disorder, while a paper writing MDD means Major Depressive Disorder.
    record = {"local_id": "S1", "groups": [{"name": _wrapped("MDD")}]}
    mapped = nz.normalize(record, {"ONVOC": onvoc})
    assert [m.matched for m in mapped] == [False]


def test_an_acronym_the_record_spells_out_is_accepted(onvoc):
    record = {
        "local_id": "S1",
        "groups": [
            {
                "name": _wrapped("ASD"),
                "description": _wrapped("children with autism spectrum disorder"),
            }
        ],
    }
    mapped = [m for m in nz.normalize(record, {"ONVOC": onvoc}) if m.path == "groups.name"]
    assert mapped[0].matched and mapped[0].method == "acronym"


# --- branch routing ---------------------------------------------------------


def test_a_test_is_not_matched_to_a_psychological_concept(onvoc):
    # `Wechsler Abbreviated Scale of Intelligence` contains the word `Intelligence`, and
    # an unscoped lookup returns that concept confidently and wrongly.
    record = {
        "local_id": "S1",
        "assessments": [
            {"name": _wrapped("Wechsler Abbreviated Scale of Intelligence (WASI-IV)")}
        ],
    }
    mapped = nz.normalize(record, {"ONVOC": onvoc})
    assert all(m.concept is None or m.concept.branch == "Tests" for m in mapped)


def test_a_group_may_draw_from_disorders_or_populations(onvoc):
    scoped = onvoc.scoped(("disorders", "population"))
    assert len(scoped) < len(onvoc)
    concept, _method, _others = scoped.match("patients with major depressive disorder")
    assert concept is not None and "Depress" in concept.label


def test_an_agent_is_looked_up_only_among_drugs(onvoc):
    scoped = onvoc.scoped(("drugs",))
    assert all(c.branch in nz.BRANCHES["drugs"] for c in scoped.concepts)


# --- morphology -------------------------------------------------------------


def test_a_stem_bridges_depression_to_depressive_disorder(onvoc):
    concept, _method, _others = onvoc.scoped(("disorders",)).match("depression")
    assert concept is not None and "Depress" in concept.label


def test_an_ambiguous_stem_is_not_guessed():
    a = nz.Concept("1", "Alpha Thing", "V")
    b = nz.Concept("2", "Alpha Things", "V")
    vocabulary = nz.Vocabulary("V", [a, b])
    # Both stem alike, so the stem cannot decide and must not.
    assert vocabulary.by_stem == {}


# --- the treatment/control query --------------------------------------------


def _trial(levels, kinds=("pharmacological", "placebo")):
    return {
        "local_id": "S1",
        "design": {
            "arms": [
                {
                    "local_id": "a1",
                    "name": _wrapped("escitalopram"),
                    "arm_kind": _wrapped(kinds[0]),
                    "agent": _wrapped("escitalopram"),
                },
                {
                    "local_id": "a2",
                    "name": _wrapped("placebo"),
                    "arm_kind": _wrapped(kinds[1]),
                    "agent": _wrapped("saline"),
                },
            ]
        },
        "analyses": [
            {
                "local_id": "an1",
                "name": _wrapped("drug > placebo"),
                "effect": {
                    "cells": [
                        {"level": _wrapped(levels[0]), "direction": _wrapped("positive")},
                        {"level": _wrapped(levels[1]), "direction": _wrapped("negative")},
                    ]
                },
            }
        ],
    }


def test_an_intervention_against_a_comparator_is_found():
    found = list(q.treatment_contrasts(_trial(("escitalopram", "placebo"))))
    assert len(found) == 1
    assert found[0].intervention.name == "escitalopram"
    assert found[0].comparator.kind == "placebo"
    # The direction reported is the intervention's, which is the only reading that
    # survives pooling -- each paper names its contrast whichever way round it likes.
    assert found[0].direction == "positive"


def test_a_trial_with_no_comparator_arm_yields_nothing():
    assert (
        list(
            q.treatment_contrasts(
                _trial(("escitalopram", "placebo"), kinds=("pharmacological", "pharmacological"))
            )
        )
        == []
    )


def test_a_group_contrast_is_not_a_treatment_contrast():
    record = _trial(("patients", "healthy controls"))
    assert list(q.treatment_contrasts(record)) == []


def test_a_level_naming_two_arms_places_neither():
    record = _trial(("escitalopram", "placebo"))
    # Give both arms the same name so the level is ambiguous.
    record["design"]["arms"][1]["name"]["value"] = "escitalopram"
    assert list(q.treatment_contrasts(record)) == []


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("pharmacological", "intervention"),
        ("stimulation", "intervention"),
        ("active_comparator", "intervention"),
        ("placebo", "comparator"),
        ("sham", "comparator"),
        ("usual_care", "comparator"),
        ("no_intervention", "comparator"),
        ("", None),
    ],
)
def test_every_arm_kind_has_a_side(kind, expected):
    assert q.role(kind) == expected


def test_levels_are_matched_by_words_not_similarity():
    assert not q.same("men", "women")
    assert q.same("REAL", "the REAL group")


# --- abbreviations ----------------------------------------------------------

from pondie.vocabularies import abbreviations as ab  # noqa: E402


def test_a_definition_in_brackets_is_mined():
    got = ab.mine("We used the Autism Diagnostic Observation Schedule (ADOS) throughout.")
    assert got.get("ADOS") == "Autism Diagnostic Observation Schedule"


def test_a_manufacturer_string_is_not_an_abbreviation():
    # A detector looking for `long form (SF)` also finds `(Philips Medical Systems, Best,
    # The Netherlands)`, whose letters happen to fit.
    assert not ab.Abbreviations.plausible("Systems, Best, The Netherlands")
    assert ab.Abbreviations.plausible("dorsolateral prefrontal cortex")


def test_the_canonical_expansion_is_the_one_seen_most():
    """`canonical` still reports the corpus consensus -- for auditing the store, not for
    expanding a paper's text. `expand` no longer consults it; see
    `test_an_abbreviation_resolves_only_against_the_paper_that_defined_it`."""
    store = ab.Abbreviations()
    for _ in range(3):
        store.learn("the echo-planar imaging (EPI) sequence", "a")
    store.learn("the echoplanar imaging (EPI) sequence", "b")
    assert store.canonical("EPI") == "echo-planar imaging"
    assert store.expand("EPI", "") is None


def test_spelling_variants_are_not_reported_as_disagreements():
    store = ab.Abbreviations()
    store.learn("the Brodmann Area (BA) map")
    store.learn("the Brodmann Areas (BA) map")
    assert store.disagreements() == []


def test_a_genuine_conflict_is_reported():
    store = ab.Abbreviations()
    store.learn("the fractional anisotropy (FA) map")
    store.learn("a flip angle (FA) of ninety degrees")
    assert [s for s, _v in store.disagreements()] == ["fa"]


def test_a_papers_own_definition_is_the_only_one_that_counts():
    """`FA` is fractional anisotropy in a diffusion paper and flip angle in an acquisition
    section, and the corpus does not get a vote in either. It used to: `for_paper` layered
    the paper over a copy of the whole store, so a short form the paper never defined still
    resolved to whatever was commonest elsewhere."""
    corpus = ab.Abbreviations()
    for i in range(5):
        corpus.learn("the fractional anisotropy (FA) map", f"dti{i}")
    assert corpus.canonical("FA") == "fractional anisotropy"

    paper = corpus.for_paper("images used a flip angle (FA) of 90 degrees", "acq")
    assert paper.expand("FA", "") == "flip angle"
    assert paper.entries.keys() == {"fa"}, "nothing this paper did not define is reachable"
    assert corpus.canonical("FA") == "fractional anisotropy", "the corpus store is untouched"


def test_a_hand_curated_entry_does_not_resolve_for_a_paper_that_never_defined_it():
    """Curated entries are gone from the store and must not come back.

    `SSRI` obviously expands to selective serotonin reuptake inhibitor -- and a paper that
    writes `a superficial siderosis of the retina interface (SSRI)` has said otherwise about
    itself. A hand entry keyed to no article overrules every article, which is the failure
    mode curation is supposed to prevent."""
    store = ab.Abbreviations()
    store.add("SSRI", "selective serotonin reuptake inhibitor", "curated")
    store.learn("a superficial siderosis of the retina interface (SSRI)", "paper1")
    assert store.expand("SSRI", "") is None
    assert store.expand("SSRI", "paper1") == "siderosis of the retina interface"


def test_expansion_reaches_a_vocabulary_the_acronym_cannot(onvoc):
    store = ab.Abbreviations().for_paper(
        "the dorsolateral prefrontal cortex (dlPFC) was seeded", "S1")
    record = {"local_id": "S1", "regions": [{"name": _wrapped("left dlPFC parcel")}]}
    without = nz.normalize(record, {"ONVOC": onvoc})
    with_store = nz.normalize(record, {"ONVOC": onvoc}, store)
    assert [m.expansions for m in with_store] == [("dorsolateral prefrontal cortex",)]
    # ONVOC has no dorsolateral entry, so this one still does not match -- but the
    # expansion is now recorded, which is what makes it a usable term proposal.
    assert not without[0].matched


# --- new-term candidates ----------------------------------------------------


def test_unmatched_values_become_counted_candidates():
    rows = [
        nz.Mapping("A", "assessments.name", "Beck Depression Inventory", None),
        nz.Mapping("B", "assessments.name", "beck depression inventory (BDI)", None),
        nz.Mapping("C", "assessments.name", "Something Else", None),
    ]
    got = nz.candidates(rows, minimum=2)
    assert len(got) == 1
    assert got[0].support == 2
    # The longest surface form is kept: it is the most informative proposal.
    assert got[0].text == "beck depression inventory (BDI)"
    assert got[0].branch_group == "tests"


def test_a_matched_value_is_not_a_candidate(onvoc):
    concept = onvoc.concepts[0]
    rows = [nz.Mapping("A", "groups.name", "x", concept, "exact")]
    assert nz.candidates(rows) == []


# -- matching an initialism against the words it stands for ----------------


def test_an_initialism_expands_to_all_of_its_words() -> None:
    """Schwartz & Hearst returns the shortest valid suffix, which for "African American
    (AA)" is "American" -- leading `a` at a word start, second `a` inside the same word.
    Expanding `AA` to "American" is worse than not expanding it, and it is what had
    grounding score `African American` at 0.016 against a sentence saying only `AA`."""
    from pondie.vocabularies import abbreviations

    mined = abbreviations.mine_builtin(
        "prevalence rates for certain segments of the population, e.g. African American "
        "(AA) men, Native Americans, and those of low income."
    )
    assert mined.get("AA") == "African American"


def test_schwartz_hearst_still_owns_the_shapes_it_had_right() -> None:
    """The initialism reading is tried first and only for an all-capital short form, so a
    hyphenated or multi-word expansion keeps the answer it already had."""
    from pondie.vocabularies import abbreviations

    caps = abbreviations.mine_builtin(
        "assessed with the clinician-administered PTSD scale (CAPS) at baseline."
    )
    assert "clinician-administered PTSD scale" in (caps.get("CAPS") or "")

    fwhm = abbreviations.mine_builtin("smoothed at full-width-at-half-maximum (FWHM) 6 mm.")
    assert fwhm.get("FWHM") == "full-width-at-half-maximum"


def test_an_initialism_will_not_take_a_number_or_a_bracket_for_a_word() -> None:
    """The window is raw text, so the words before a short form can be punctuation or a
    table cell. An expansion built from those is noise dressed as a definition."""
    from pondie.vocabularies import abbreviations

    assert abbreviations._initialism("AB", "| 12 | 3.4 (0.2)") is None
    assert (
        abbreviations._initialism("aa", "african american") is None
    ), "lower-case short forms are not initialisms"


# --- the route table against the schema it addresses ------------------------


def test_every_route_names_a_field_the_schema_has():
    """A route whose field does not exist maps nothing and says nothing about it.

    `ROUTES` carried `groups.diagnosis` from the first version of this table until
    2026-09. Neither the extraction schema nor the storage schema has ever had that
    field -- the slot is `medical_condition` -- so the route fired on zero values while
    the table read as though diagnoses were covered, and the corpus's 3,898
    `medical_condition` mentions went unnormalized. Nothing failed, because a route that
    matches no field and a field whose values all miss look identical from the outside.

    Walked against the STORAGE schema from `Study`, which is the record root, so a route
    reaches a single nested object (`design.arms`) the same way it reaches an entity list.
    """
    from pondie import schema
    from pondie.schema import reader

    loaded = reader.load(schema.STORAGE)

    missing = []
    for path, _vocabulary, _groups in nz.ROUTES:
        class_name = "Study"
        for step in path.split("."):
            attributes = loaded.attributes(class_name)
            if step not in attributes:
                missing.append(f"{path}: {class_name} has no attribute {step!r}")
                break
            ranges = loaded.value_ranges(attributes[step])
            class_name = next((r for r in ranges if r in loaded), class_name)

    assert not missing, "routes addressing fields no schema has: " + "; ".join(missing)


def test_a_negated_condition_is_not_a_diagnosis(onvoc):
    """The gate that the ONVOC route did not have.

    Ungated, every one of these returned a disorder: a disease vocabulary asked about
    "absence of major depressive disorder" retrieves depression, confidently, at the
    `contains` layer. 13% of the corpus's `medical_condition` values are negation-shaped,
    and they are overwhelmingly the control cohorts -- so the error is not scattered, it
    lands on exactly the groups a patients-versus-controls query has to tell apart.
    """
    record = {
        "local_id": _wrapped("study"),
        "groups": [
            {"medical_condition": _wrapped(text)}
            for text in (
                "no neurological or psychiatric disorder",
                "absence of major depressive disorder",
                "No clinically significant cognitive impairment No dementia",
            )
        ],
    }
    mapped = nz.normalize(record, {"ONVOC": onvoc})
    assert [m.sentinel for m in mapped] == ["NO_CONDITION"] * 3
    assert not any(m.matched for m in mapped)


def test_a_comorbidity_list_maps_each_head(onvoc):
    """One row per condition, because one row per value loses the others.

    `medical_condition` is multivalued in the schema and routinely arrives as one string
    anyway. Mapped as a single value, the ladder returns whichever comorbidity it reaches
    first and the rest of the cohort's diagnoses are not in the output at all.
    """
    record = {
        "local_id": _wrapped("study"),
        "groups": [{"medical_condition": _wrapped("schizophrenia or bipolar disorder")}],
    }
    mapped = nz.normalize(record, {"ONVOC": onvoc})
    assert {m.head for m in mapped} == {"schizophrenia", "bipolar disorder"}
    assert {m.concept.label for m in mapped if m.matched} == {"Schizophrenia", "Bipolar Disorder"}


def test_the_primary_condition_wins_over_the_longer_comorbidity(onvoc):
    """Earliest, then longest -- not longest outright.

    Longest outright read `behavioural-variant frontotemporal dementia amyotrophic lateral
    sclerosis` as an ALS cohort, because ALS has the longer name. The primary condition is
    the one named first.
    """
    scoped = onvoc.scoped(("disorders",))
    concept, method, _ = scoped.match(
        "behavioural-variant frontotemporal dementia amyotrophic lateral sclerosis"
    )
    assert (concept.label, method) == ("Dementia", "contains")


def test_a_generalized_mapping_says_so(onvoc):
    """`bvFTD -> Dementia` and `Schizophrenia patients -> Schizophrenia` are not the
    same claim, and the second is the one that loses nothing.

    Without the flag a query pooling `Dementia` gets the FTD cohorts and the Alzheimer's
    cohorts with no way to tell that one of them arrived by dropping its subtype.
    """
    record = {
        "local_id": _wrapped("study"),
        "groups": [
            {"medical_condition": _wrapped("behavioural variant frontotemporal dementia")},
            {"medical_condition": _wrapped("Schizophrenia patients")},
        ],
    }
    rolled = {m.head: m.rollup for m in nz.normalize(record, {"ONVOC": onvoc})}
    assert rolled == {
        "behavioural variant frontotemporal dementia": True,
        # triage lifted the study-role noun off before the lookup, so the head the mapping
        # is about is the diagnosis alone and nothing was generalized to reach the concept.
        "Schizophrenia": False,
    }


def test_a_recurring_finer_term_becomes_a_candidate_even_though_it_mapped(onvoc):
    """A gap in grain is a proposal too, once enough STUDIES name the finer term.

    ONVOC has `Dementia` and no subtype below it, so every FTD cohort in the corpus maps
    and every one of them loses its variant. Counted per study rather than per mention,
    because a paper naming its diagnosis once per group is one piece of evidence.
    """
    mapped = [
        nz.normalize(
            {
                "local_id": _wrapped(f"study{n}"),
                "groups": [
                    {"medical_condition": _wrapped("behavioural variant frontotemporal dementia")}
                ],
            },
            {"ONVOC": onvoc},
        )
        for n in range(4)
    ]
    rows = [m for batch in mapped for m in batch]

    assert nz.candidates(rows, minimum=1) == []  # every row matched: no coverage gap
    proposed = nz.candidates(rows, minimum=10**9, grain=3)
    assert [(c.text, c.support, c.rolled_up_to) for c in proposed] == [
        ("behavioural variant frontotemporal dementia", 4, "Dementia")
    ]
    assert nz.candidates(rows, minimum=10**9, grain=5) == []  # under threshold


def test_a_closed_spelling_still_corroborates_an_acronym(onvoc):
    """Hyphenation is not evidence, and treating it as evidence refused correct expansions.

    ONVOC writes `Post-Traumatic Stress Disorder`, whose content words are
    {post, traumatic, stress}. The literature more often writes `posttraumatic` -- one
    token, not a superset of those -- so every paper spelling it closed had its `PTSD`
    refused. Measured at 10 studies before the squashed form was checked too.

    The MDD guard has to survive the widening, which is the reason this asserts both: the
    point of corroboration is that ONVOC's only MDD is Mood Dysregulation Disorder and a
    paper writing MDD means Major Depressive Disorder.
    """
    ptsd = next(c for c in onvoc.concepts if c.label == "Post-Traumatic Stress Disorder")
    mdd = next(c for c in onvoc.concepts if c.label == "Mood Dysregulation Disorder")

    assert nz.corroborated(ptsd, "patients with posttraumatic stress disorder (PTSD)")
    assert nz.corroborated(ptsd, "patients with post-traumatic stress disorder")
    assert not nz.corroborated(ptsd, "a study of trait anxiety in healthy adults")
    assert not nz.corroborated(mdd, "patients with major depressive disorder (MDD)")


# --- constraints that have to survive being applied to a subset ---------------


def test_an_abbreviation_is_not_a_capitalised_word():
    """`expansions_in` ran on "has an uppercase letter", which every Title Case word has.

    Right for running prose and wrong for every Title Case field in the record. Measured on
    the real store: `Positive and Negative Syndrome Scale` had `Positive` expanded to
    "positive valence for favorable", `Trail Making Test` had `Test` expanded to
    "multiple choice-vocabulary-intelligence test", and `Left Inferior Frontal Gyrus` had
    `Frontal` expanded -- on `assessments.name`, `groups.name` and `regions.name`, which are
    Title Case almost by definition.

    The mixed-case short forms the field actually writes have to survive the fix, which is
    why this is not simply `token.isupper()`.
    """
    from pondie.vocabularies.abbreviations import is_short_form

    for short in ("PANSS", "BDI", "AUDIT", "MID", "WM", "fMRI", "dlPFC", "mPFC", "CANTAB"):
        assert is_short_form(short), short
    for word in ("Positive", "Negative", "Control", "Test", "Frontal", "Use", "Task", "Beck"):
        assert not is_short_form(word), word


def test_identical_names_stay_grouped_when_only_some_are_seeded():
    """The constraint is a GROUP now, so the star it used to be cannot come back.

    `name_links` emitted a star of pairs anchored on the first member, and a star does not
    survive losing its hub. Star and clique agree whenever every member is clustered --
    which is why it held for as long as one caller existed. The surviving route seeds some
    tasks from the Atlas and clusters only the remainder, so the hub is dropped routinely:
    49 tasks sharing one folded name came out in seven categories that way. `same_name`
    returns the members, and `paradigm_distances` zeroes every pair within them, so the
    property is structural rather than something the emitter has to remember.
    """
    from pondie.normalization import task as module

    def one(name):
        return module.Task(
            study="s", name=name, description="d", instructions="",
            design_type="", response_modality="", performance_measures="",
            conditions=(), stimulus_content=(),
        )

    tasks = [one(n) for n in
             ["cue reactivity task", "Cue-Reactivity Task", "cue reactivity task ", "n-back"]]
    groups = module.same_name(tasks)

    together = next(g for g in groups.values() if len(g) > 1)
    assert set(together) == {0, 1, 2}, "one name spelled three ways is one group"
    assert all(3 not in g or g == [3] for g in groups.values()), (
        "a name nothing matches groups only with itself"
    )
    # dropping any member leaves the rest in one group -- what the star could not do
    for dropped in (0, 1, 2):
        rest = [i for i in together if i != dropped]
        assert len(rest) == 2 and set(rest) <= {0, 1, 2}

def test_an_abbreviation_resolves_only_against_the_paper_that_defined_it():
    """A corpus-wide expansion is a different paper's fact applied to this one.

    `for_paper` used to start from a copy of the whole store and layer the paper's own
    definitions on top, so a short form the paper never defined still resolved -- to
    whichever expansion was commonest elsewhere. The store carries the evidence of how badly
    that goes: `PET` resolved to "P.F. Liddle, R.S.J. FrackowiakComparing functional", mined
    from somebody's reference list.
    """
    from pondie.vocabularies.abbreviations import Abbreviations

    store = Abbreviations(
        {
            "ad": {"expansion": "axial diffusivity", "source": "mined",
                   "papers": ["dti"], "count": 1, "by_paper": {"dti": "axial diffusivity"}},
            "pet": {"expansion": "P.F. Liddle, R.S.J. Frackowiak", "source": "mined",
                    "papers": ["junk"], "count": 1,
                    "by_paper": {"junk": "P.F. Liddle, R.S.J. Frackowiak"}},
        }
    )
    assert store.expand("AD", "dti") == "axial diffusivity"
    assert store.expand("AD", "dementia") is None, "another paper's definition is not evidence"
    assert store.expand("AD", "") is None, "a corpus-wide store has no answer without a paper"

    scoped = store.for_paper("Patients had Alzheimer's disease (AD).", "dementia")
    assert scoped.expand("AD", "") == "Alzheimer's disease", "the paper's own text wins"
    assert scoped.expand("PET", "") is None, "nothing this paper did not define is reachable"
    assert "pet" not in scoped.entries


def test_an_expansion_cannot_be_asked_for_without_a_paper():
    """The failure mode was silence: an unscoped store expands nothing for anybody, so a
    caller that forgot the paper lost the whole layer without a word."""
    corpus_store = ab.Abbreviations({"ad": {"expansion": "axial diffusivity",
                                            "source": "mined", "papers": ["dti"],
                                            "count": 1, "by_paper": {"dti": "axial diffusivity"}}})
    with pytest.raises(ValueError, match="needs a paper"):
        list(ab.expansions_in("AD was measured", corpus_store))
    assert list(ab.expansions_in("AD was measured", corpus_store, "dti")) == [
        ("AD", "axial diffusivity")
    ]


def test_a_scoped_store_refuses_another_papers_question():
    """Scoping is a fact about the store, so asking it about a different paper is a bug."""
    scoped = ab.Abbreviations().for_paper("Alzheimer's disease (AD) patients", "dementia")
    assert scoped.paper == "dementia"
    with pytest.raises(ValueError, match="scoped to"):
        list(ab.expansions_in("AD", scoped, "dti"))


def test_scoping_a_store_requires_naming_the_paper():
    """`for_paper(text)` with the study id in scope and not passed is how it went wrong:
    the store's own per-paper rows were never layered."""
    with pytest.raises(ValueError, match="needs the paper"):
        ab.Abbreviations().for_paper("some text", "")


def test_a_one_argument_expand_answers_rather_than_raising():
    """`repair.guard._words` calls `expand(token)`, and `paper` being a required argument
    made every store that reached it raise `TypeError`."""
    scoped = ab.Abbreviations().for_paper("the Autism Diagnostic Observation Schedule (ADOS)", "p1")
    assert scoped.expand("ADOS") == "Autism Diagnostic Observation Schedule"
    assert ab.Abbreviations().expand("ADOS") is None
