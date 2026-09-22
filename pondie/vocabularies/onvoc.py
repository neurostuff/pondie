"""Map a record's own wording onto shared vocabularies, without changing the record.

The storage schema deliberately binds no subject-matter vocabulary: values are the
source's own words, and `neuroimaging-study-storage.yaml` says mapping them onto ONVOC or
the Cognitive Atlas "is a later stage that reads the free text and its evidence
sentences". This is that stage.

It never edits a record. A mapping is an assertion *about* a record -- "this paper's
`escitalopram` is ONVOC's Escitalopram" -- and writing it into the field would destroy the
thing that makes the mapping checkable: the paper's own wording beside it. The code therefore
mappings are emitted as their own rows, carrying the method that produced them and the
text they were produced from.

Two vocabularies, chosen because they cover different halves of the record:

  ONVOC             drugs, disorders, brain regions, tests, population groups -- the
                    nouns a clinical trial's arms, groups and assessments are made of
  Cognitive Atlas   tasks, cognitive concepts and disorders -- what a paradigm *is*,
                    which ONVOC does not attempt

Matching is layered from certain to merely plausible, and every mapping records which
layer produced it. A token-overlap match and an exact label match are not the same claim,
and collapsing them into one confidence number would hide that.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from pondie import paths
from pondie.formats import values
from pondie.vocabularies.abbreviations import expansions_in
from pondie.vocabularies.folding import fold, squash
from pondie.vocabularies import phrases
from pondie.vocabularies.labels import acronym, content, stem, stems, tokens

VOCAB_DIR = paths.VOCAB

@dataclass(frozen=True)
class Concept:
    """One vocabulary entry, with every string it can be recognised by."""

    id: str
    label: str
    vocabulary: str
    synonyms: tuple[str, ...] = ()
    branch: str = ""
    definition: str = ""
    #: The top concept this one hangs under -- one of ONVOC's nine. `branch` is the
    #: immediate parent and says which list a term came from; `facet` says what KIND of
    #: thing it is, which is what a query filters on and what a wrong mapping shows up
    #: as. `Escitalopram` is branch `Antidepressants`, facet `Drugs and Medications`.
    facet: str = ""

    def surfaces(self) -> tuple[str, ...]:
        return (self.label,) + self.synonyms


#: `Autism Diagnostic Observation Schedule (ADOS)` is two surface forms, and the one a
#: cell level or a table header uses is usually the one in the brackets.
_PARENTHETICAL = re.compile(r"\s*[\(\[]([^)\]]{2,60})[\)\]]")

#: Qualifiers that modify a concept without changing which concept it is. Stripped only
#: when the remainder still has a content word -- `left` alone is not a region.
_QUALIFIERS = re.compile(
    r"\b(left|right|bilateral|ipsilateral|contralateral|anterior|posterior|dorsal|"
    r"ventral|superior|inferior|medial|lateral|rostral|caudal|parcel|roi|seed|mask|"
    r"region|cluster|network|active|sham|total|mean|score|sub|scale)\b",
    re.I,
)


def surface_forms(text: str, abbreviations: Any = None, paper: str = "") -> list[str]:
    """The surface forms worth looking a phrase up under, most specific first.

    Named apart from `folding.variants` deliberately: that one is orthography -- plurals,
    hyphenation, case. This expands abbreviations against the paper's own definitions,
    generates acronyms and strips qualifiers, which is vocabulary matching. Both were
    called `variants`, in one package, doing different work.

    A field's value is rarely a bare vocabulary label. It carries an acronym in
    brackets, a laterality, a version number, a dose. Each of those is stripped into its
    own candidate rather than all at once, so the most specific form still wins.

    When an abbreviation store is given, every short form in the phrase is also offered
    expanded. This is the layer that lets ONVOC do its job: the vocabulary spells
    everything out and papers do not, and `dlPFC` reaches `dorsolateral prefrontal
    cortex` only because THIS paper defined it in brackets. `paper` says which one; a
    store `for_paper` already scoped carries its own.
    """

    text = str(text or "").strip()
    if not text:
        return []
    seen, out = set(), []

    def offer(candidate: str) -> None:
        candidate = candidate.strip(" .,;:-")
        key = fold(candidate)
        if key and key not in seen:
            seen.add(key)
            out.append(candidate)

    offer(text)
    for inner in _PARENTHETICAL.findall(text):
        offer(inner)  # the acronym
    offer(_PARENTHETICAL.sub("", text))  # the phrase without it
    stripped = _QUALIFIERS.sub(" ", _PARENTHETICAL.sub("", text))
    if content(stripped):
        offer(stripped)

    if abbreviations is not None:
        for candidate in list(out):
            replaced = candidate
            for short, expansion in expansions_in(candidate, abbreviations, paper):
                replaced = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(short)}(?![A-Za-z0-9])", expansion, replaced
                )
            if replaced != candidate:
                offer(replaced)
                offer(_QUALIFIERS.sub(" ", replaced))
    return out


#: How a mapping was made, most trustworthy first. Kept as an ordered tuple because the
#: layer *is* the confidence -- an exact label match and a token-overlap match are
#: different claims and a single score would flatten them.
METHODS = ("exact", "synonym", "variant", "acronym", "contains", "stem", "overlap")


@functools.lru_cache(maxsize=None)
def _contains(surface: str) -> re.Pattern:
    """The word-boundary test for one vocabulary surface, compiled once.

    Cached because the `contains` layer is quadratic by construction -- every surface in
    the scoped vocabulary against every value that missed the exact index -- and `re`'s
    own cache holds 512 patterns and thrashes at ONVOC's 750. Mapping the corpus's six
    routed fields went from over twenty minutes to under two.
    """
    return re.compile(rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])")


@dataclass(frozen=True)
class Candidate:
    """A value no vocabulary could place, proposed as a term the vocabulary lacks.

    The useful output of a normalization layer is not only what it mapped. ONVOC's Tests
    branch has 53 entries and the corpus asks it about ADOS, MADRS, HAMD and BDI; that is
    a gap in the vocabulary, and the evidence for it is exactly this list. Counted across
    papers because a term used once is a paper's idiosyncrasy and a term used in ten is a
    term.
    """

    text: str
    path: str
    branch_group: str
    papers: tuple[str, ...]
    expansions: tuple[str, ...] = ()
    #: The ONVOC term this rolled up to, when it rolled up to one. A candidate with this
    #: set is not a gap in coverage -- the value mapped -- it is a gap in GRAIN: ONVOC has
    #: `Dementia` and the literature has `behavioural variant frontotemporal dementia` in
    #: 148 studies, and the second is a term ONVOC could carry. Kept apart from an unmapped
    #: candidate because they are different proposals: one adds a concept, one adds a
    #: child under a concept that already exists.
    rolled_up_to: str = ""

    @property
    def support(self) -> int:
        """Distinct STUDIES, not mentions. A paper that names its cohort's diagnosis in
        four groups is one piece of evidence that the term exists, not four."""
        return len(self.papers)

    def render(self) -> str:
        expanded = f"  (= {self.expansions[0]})" if self.expansions else ""
        under = f"  [under {self.rolled_up_to}]" if self.rolled_up_to else ""
        return (
            f"{self.support:3d} study(s)  {self.branch_group:11s} "
            f"{self.text[:58]!r}{expanded}{under}"
        )


@dataclass(frozen=True)
class Mapping:
    """One assertion about one field of one record."""

    study_id: str
    path: str
    text: str
    concept: Concept | None
    method: str = ""
    alternatives: tuple[str, ...] = ()
    #: What the abbreviation layer thought this phrase's short forms stood for. Carried
    #: even when nothing matched, because an unmapped value with a known expansion is a
    #: far better proposal for the vocabulary than the acronym alone.
    expansions: tuple[str, ...] = ()
    #: The part of the value this mapping is about. A `medical_condition` is routinely a
    #: comorbidity list, and one row per head is what makes each of them queryable;
    #: without it a two-diagnosis cohort is silently a one-diagnosis cohort.
    head: str = ""
    #: What the value says in its own words at the finest grain the CORPUS supports,
    #: which is not always what ONVOC can say. ONVOC has `Dementia` and no subtypes, so
    #: `behavioural-variant frontotemporal dementia` maps to Dementia and keeps bvFTD
    #: here. A query that wants the ONVOC grain reads `concept`; one that wants the
    #: literature's own grain reads this; and a corpus term with enough support across
    #: STUDIES becomes a proposal for ONVOC. The two are never merged, because which
    #: grain a meta-analysis needs is the meta-analysis's decision, not this layer's.
    corpus_term: str = ""
    #: True when the concept was reached by generalizing rather than by naming: the
    #: matched surface is a proper part of the value. `Schizophrenia patients` is not a
    #: rollup, `first-episode schizophrenia` is. A query that cannot tell them apart
    #: treats a first-episode cohort and a chronic one as the same population.
    rollup: bool = False
    #: Set when triage found the value states an ABSENCE rather than a presence. Carried
    #: rather than dropped, because "this cohort was screened for psychiatric illness" is
    #: a fact a query wants and is not the same as the field being empty.
    sentinel: str = ""
    #: Course and state lifted off the head before lookup -- `first-episode`, `chronic`,
    #: `remitted`, `unmedicated`. These are the second way a mapping generalizes: `rollup`
    #: covers what the vocabulary could not say, and this covers what the layer chose not
    #: to ask it. Both have to be visible, because pooling first-episode with chronic
    #: cohorts is a decision a meta-analysis makes and not one a normalizer should make
    #: for it silently.
    qualifiers: tuple[str, ...] = ()
    #: What the value states the ABSENCE of, when it states both. Kept off `head` so it is
    #: never looked up, and out of `sentinel` so a value that denies one thing while
    #: naming another still maps.
    denied: tuple[str, ...] = ()
    #: How the negation was read -- `parse`, `cue`, or `assertion`.
    scope: str = ""

    @property
    def matched(self) -> bool:
        return self.concept is not None

    def render(self) -> str:
        if self.sentinel:
            return f"{self.path}: {self.text!r} -> ({self.sentinel})"
        if not self.concept:
            return f"{self.path}: {self.text!r} -> (no match)"
        extra = f"  ~{len(self.alternatives)} other" if self.alternatives else ""
        grain = f"  <- {self.corpus_term!r}" if self.rollup else ""
        return (
            f"{self.path}: {self.text!r} -> {self.concept.label!r} "
            f"[{self.concept.vocabulary}/{self.method}]{grain}{extra}"
        )


class Vocabulary:
    """A term list, indexed every way the matcher looks things up."""

    def __init__(self, name: str, concepts: list[Concept]):
        self.name = name
        self.concepts = concepts
        self.by_surface: dict[str, list[Concept]] = {}
        for concept in concepts:
            for surface in concept.surfaces():
                key = fold(surface)
                if key:
                    self.by_surface.setdefault(key, []).append(concept)
        # Longest first: `Selective Serotonin Reuptake Inhibitor` must be tried before
        # the shorter labels nested inside it.
        self._ordered = sorted(self.by_surface, key=len, reverse=True)
        self._scopes: dict[tuple[str, ...], "Vocabulary"] = {}
        # A second index on stems, so `depression` reaches `Depressive Disorder`. Only
        # unambiguous stem sets are kept: if two concepts stem alike, the stem cannot
        # decide between them and the match would be a coin flip.
        by_stem: dict[frozenset[str], list[Concept]] = {}
        for concept in concepts:
            for surface in concept.surfaces():
                key = stems(surface)
                if key:
                    by_stem.setdefault(key, []).append(concept)
        self.by_stem = {k: v[0] for k, v in by_stem.items() if len({c.id for c in v}) == 1}
        # Acronyms built from the labels themselves. ONVOC spells `Autism Spectrum
        # Disorder` out and every paper writes `ASD`; the vocabulary carries no synonym
        # for it and nothing else can bridge three letters to three words. Ambiguous
        # acronyms are dropped rather than guessed -- two concepts sharing initials is
        # exactly when an acronym stops identifying anything.
        by_acronym: dict[str, list[Concept]] = {}
        for concept in concepts:
            for surface in concept.surfaces():
                letters = acronym(surface)
                if letters:
                    by_acronym.setdefault(letters, []).append(concept)
        self.by_acronym = {
            k: v[0]
            for k, v in by_acronym.items()
            if len({c.id for c in v}) == 1 and k not in self.by_surface
        }

    def __len__(self) -> int:
        return len(self.concepts)

    def scoped(self, groups: tuple[str, ...]) -> "Vocabulary":
        """The sub-vocabulary a field is allowed to draw from. Cached per group set."""
        key = tuple(sorted(groups))
        if key not in self._scopes:
            allowed = {branch for group in groups for branch in BRANCHES.get(group, ())}
            self._scopes[key] = Vocabulary(
                f"{self.name}/{'+'.join(key)}",
                [c for c in self.concepts if c.branch in allowed],
            )
        return self._scopes[key]

    def match(
        self, text: str, abbreviations: Any = None, paper: str = ""
    ) -> tuple[Concept | None, str, list[Concept]]:
        """The best concept for this phrase, the method used, and the runners-up.

        Each surface form of the phrase is tried in full before the next is considered,
        so an exact hit on the acronym beats a containment hit on the whole phrase.
        """
        for index, candidate in enumerate(surface_forms(text, abbreviations, paper)):
            concept, method, others = self._match_one(candidate)
            if concept:
                return concept, (method if index == 0 else "variant"), others
        return None, "", []

    def _match_one(self, text: str) -> tuple[Concept | None, str, list[Concept]]:
        key = fold(text)
        if not key:
            return None, "", []

        hits = self.by_surface.get(key)
        if hits:
            method = "exact" if fold(hits[0].label) == key else "synonym"
            return hits[0], method, hits[1:]

        # A label appearing whole inside the phrase: "paroxetine 20 mg daily" contains
        # "paroxetine". Guarded on a word boundary so "sham" does not match "shampoo".
        #
        # EARLIEST, then longest -- not longest outright, which is what this did. A
        # `medical_condition` is routinely a comorbidity list crammed into one string
        # (29% of the corpus's values run to five words or more), and the longest
        # vocabulary label inside such a list is whichever comorbidity happens to have
        # the longest name. Measured: `behavioural-variant frontotemporal dementia
        # amyotrophic lateral sclerosis` returned Amyotrophic Lateral Sclerosis and
        # `treatment-resistant depression bipolar disorder` returned Bipolar Disorder.
        # English puts the head of a noun phrase last but puts the PRIMARY condition
        # first, and taking the earliest match returns Dementia and Depressive Disorder.
        # Longest still breaks ties, so `Selective Serotonin Reuptake Inhibitor` beats
        # the shorter labels nested inside it at the same position.
        contained = [
            (found.start(), -len(surface), surface)
            for surface in self._ordered
            if len(surface) >= 4 and (found := _contains(surface).search(key))
        ]
        if contained:
            contained.sort()
            best = self.by_surface[contained[0][2]][0]
            return best, "contains", [self.by_surface[s][0] for _a, _b, s in contained[1:4]]

        # An acronym the vocabulary spells out. Only for short all-caps-ish input: a
        # lowercase word that happens to have the right letters is not an acronym.
        if 2 <= len(key.replace(" ", "")) <= 6 and " " not in key:
            expanded = self.by_acronym.get(key.replace(" ", ""))
            if expanded is not None:
                return expanded, "acronym", []

        # Morphology: `depression` and `Depressive Disorder` share no substring but do
        # share a stem set once the weak words are gone.
        stemmed = self.by_stem.get(stems(text))
        if stemmed is not None:
            return stemmed, "stem", []

        # Last resort: the phrase's content words are exactly a concept's, in some order.
        wanted = content(text)
        if len(wanted) >= 2:
            same = [c for c in self.concepts if content(c.label) == wanted]
            if len(same) == 1:
                return same[0], "overlap", []
        return None, "", []


def crosswalk_synonyms(directory: Path | None = None) -> dict[str, set[str]]:
    """Extra surface forms for an ONVOC term, from its own crosswalks.

    ONVOC publishes maps to MeSH, MONDO, DOID and SNOMED. Each row pairs an ONVOC term
    with the other vocabulary's term for the same thing, and the other vocabulary's
    wording is a surface form a paper might well use. This is the cheapest widening
    available: no model, no network, and the pairings are the ontology author's own.
    """

    directory = (directory or VOCAB_DIR) / "onvoc-mappings"
    extra: dict[str, set[str]] = {}
    for name in ("mesh", "mondo", "doid", "snomed"):
        path = directory / f"{name}.tsv"
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        header = lines[0].split("\t")
        try:
            onvoc_at = header.index("vocabulary_id")
            term_at = next(
                i
                for i, column in enumerate(header)
                if column.endswith("_term") and column != "vocabulary_term"
            )
        except (ValueError, StopIteration):
            continue
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) > max(onvoc_at, term_at) and parts[term_at].strip():
                extra.setdefault(parts[onvoc_at], set()).add(parts[term_at].strip())
    return extra


#: One `skos:` statement of the turtle release: a subject IRI, then predicate/object pairs
#: separated by `;`. Parsed by hand rather than with rdflib because the file is 752 flat
#: SKOS individuals with no reasoning to do, and a parser is a dependency the rest of the
#: package does not need.
_TTL_SUBJECT = re.compile(r"(https://w3id\.org/onvoc/ONVOC_\d+)\n")
_TTL_PREDICATE = re.compile(r"\s*(?:<[^>]*>\s+)?(skos:[A-Za-z]+|rdf:type|a)\s+(.*)", re.S)
_TTL_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')
_TTL_REF = re.compile(r"<(https://w3id\.org/onvoc/ONVOC_\d+)>")


def read_turtle(path: Path) -> dict[str, dict]:
    """The release's concepts, keyed by IRI, with both directions of the hierarchy.

    ONVOC publishes `skos:narrower` on the parent and `skos:broader` on the child and does
    not always publish both for a pair, so a reader that trusts one direction loses edges.
    Every edge is read from whichever end states it and the two are unioned.
    """
    body = path.read_text(encoding="utf-8")
    concepts: dict[str, dict] = {}
    for block in re.split(r"\n###\s+", body):
        subject = _TTL_SUBJECT.match(block)
        if not subject:
            continue
        statements: dict[str, list[str]] = {}
        for chunk in re.split(r"\s;\s", block[subject.end() :].strip()):
            matched = _TTL_PREDICATE.match(chunk)
            if matched:
                statements.setdefault(matched.group(1), []).append(matched.group(2))

        def literals(predicate: str, _at=statements) -> list[str]:
            return [s for raw in _at.get(predicate, []) for s in _TTL_LITERAL.findall(raw)]

        def refs(predicate: str, _at=statements) -> list[str]:
            return [s for raw in _at.get(predicate, []) for s in _TTL_REF.findall(raw)]

        concepts[subject.group(1)] = {
            "label": next(iter(literals("skos:prefLabel")), ""),
            "synonyms": literals("skos:altLabel"),
            "definition": " ".join(literals("skos:definition")),
            "narrower": refs("skos:narrower"),
            "parents": set(refs("skos:broader")),
            "top": "skos:topConceptOf" in statements,
        }
    for iri, concept in concepts.items():
        for child in concept["narrower"]:
            if child in concepts:
                concepts[child]["parents"].add(iri)
    return concepts


def load_onvoc(path: Path | None = None) -> Vocabulary:
    """ONVOC as a Vocabulary, from the pinned turtle release.

    The branch is what makes a mapping auditable at a glance: `Escitalopram` under
    `Antidepressants` is a different kind of claim from `Escitalopram` under `Tests`,
    and one of them would be a bug worth seeing.

    The turtle is the source rather than a BioPortal class dump, which is what this read
    before. The two carry the same 752 concepts with identical labels -- checked, not
    assumed -- so nothing is gained on content; what the release adds is that it is one
    versioned file that can sit in the repository's data directory, where a dump is a
    snapshot of an API with no version on it. Neither carries a single `skos:altLabel` or
    definition, which is the fact the rest of this module is built around: **ONVOC gives a
    label and a place in a three-level tree and nothing else**, so every surface form
    beyond the label has to come from the crosswalks, the paper's own abbreviations, or
    morphology.

    The JSON dump is still read when the turtle is absent, so a checkout that fetched the
    old file keeps working.
    """
    path = path or VOCAB_DIR / "onvoc.ttl"
    crosswalked = crosswalk_synonyms()

    def build(iri: str, label: str, synonyms: set[str], branch: str, facet: str, define: str):
        short = iri.rsplit("/", 1)[-1].replace("_", ":")
        return Concept(
            id=iri,
            label=label,
            vocabulary="ONVOC",
            synonyms=tuple(sorted(synonyms | crosswalked.get(short, set()))),
            branch=branch,
            definition=define[:300],
            facet=facet,
        )

    if not path.is_file():
        raw = json.loads((VOCAB_DIR / "onvoc.json").read_text(encoding="utf-8"))
        return Vocabulary(
            "ONVOC",
            [
                build(
                    entry.get("@id", ""),
                    entry["prefLabel"],
                    set(entry.get("synonym") or ()),
                    next((p.get("prefLabel") or "" for p in entry.get("parents") or []), ""),
                    "",
                    " ".join(entry.get("definition") or ()),
                )
                for entry in raw
                if entry.get("prefLabel")
            ],
        )

    concepts = read_turtle(path)

    def facet_of(iri: str, seen: tuple[str, ...] = ()) -> str:
        """The top concept above this one. Cycle-guarded; ONVOC has none, but a release
        that grew one would otherwise hang the loader rather than report itself."""
        entry = concepts[iri]
        if entry["top"] or iri in seen:
            return entry["label"]
        for parent in sorted(entry["parents"]):
            if parent in concepts:
                return facet_of(parent, seen + (iri,))
        return entry["label"]

    return Vocabulary(
        "ONVOC",
        [
            build(
                iri,
                entry["label"],
                set(entry["synonyms"]),
                next(
                    (concepts[p]["label"] for p in sorted(entry["parents"]) if p in concepts), ""
                ),
                facet_of(iri),
                entry["definition"],
            )
            for iri, entry in concepts.items()
            if entry["label"]
        ],
    )


def load_cognitive_atlas(
    kinds: tuple[str, ...] = ("task", "concept", "disorder"), directory: Path | None = None
) -> Vocabulary:
    directory = directory or VOCAB_DIR
    concepts = []
    for kind in kinds:
        path = directory / f"cognitiveatlas-{kind}.json"
        if not path.is_file():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            label = entry.get("name")
            if not label:
                continue
            alias = entry.get("alias") or ""
            concepts.append(
                Concept(
                    id=entry.get("id", ""),
                    label=label,
                    vocabulary="CognitiveAtlas",
                    synonyms=tuple(a.strip() for a in alias.split(",") if a.strip()),
                    branch=kind,
                    definition=(entry.get("definition_text") or "")[:300],
                )
            )
    return Vocabulary("CognitiveAtlas", concepts)


#: ONVOC branches grouped by what they are about. The vocabulary is one namespace and a
#: field is not: `Wechsler Abbreviated Scale of Intelligence` contains the word
#: `Intelligence`, and matching an assessment against the psychological-concept branch
#: returns exactly that, confidently and wrongly. ONVOC's own README makes the point --
#: "Study Focus: Schizophrenia" and "Exclusion Criteria: Schizophrenia" are different
#: claims about the same term -- so which branch a field may draw from is part of the
#: mapping, not a filter applied to it afterwards.
BRANCHES: dict[str, tuple[str, ...]] = {
    "drugs": (
        "Drugs and Medications",
        "Antidepressants",
        "Anti Psychotics",
        "Anxiolytics",
        "Mood Stabilizers",
        "Psychostimulants",
        "Opioids",
        "Psychedelics",
        "Cannabinoids",
        "Anesthetics",
        "Anti Inflammatory",
        "Parkinsons Disease Medication",
        "Migraine Medication",
        "Contraception",
        "Dementia Medication",
        "ADHD Medications Nonstimulants",
    ),
    "disorders": (
        "Psychiatric Disorders",
        "Neurological Disorders",
        "Medical Disorders",
        "Psychiatric Symptoms",
        "Neurological Symptoms",
        "Medical Symptoms",
        "Health",
        # ONVOC files substance terms under `Behaviors`, not `Disorders`: `Alcohol Use`,
        # `Smoking`, `Tobacco Use`, `Substance Dependence`, `Alcohol Abuse`. A route
        # scoped to the disorder branches therefore missed the two heaviest conditions in
        # this corpus -- `alcohol dependence` at 148 mentions and `nicotine dependence` at
        # 135 -- because the vocabulary calls them behaviours. Which branch a term is
        # filed under is the ontology's judgement; which branches a FIELD may draw from is
        # this table's, and `medical_condition` asks about both.
        "Substance Use",
        "Substance Abuse",
    ),
    "population": (
        "Population Groups",
        "Population Characteristics",
        "Age",
        "Species",
        "Family Relations",
    ),
    "tests": ("Tests",),
    "regions": ("Cortical Regions", "Subcortical Regions", "Brain Networks"),
    "concepts": (
        "Psychological Concepts",
        "Decision Making",
        "Executive Function",
        "Attention",
        "Learning",
        "Memory",
        "Perception",
        "Social Cognition",
    ),
}

#: (field, vocabulary, branch groups). A field may draw from more than one group -- a
#: group's name is a diagnosis or a population, and either is a defensible mapping.
ROUTES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("design.arms.agent", "ONVOC", ("drugs",)),
    ("design.arms.name", "ONVOC", ("drugs",)),
    # `groups.diagnosis` until 2026-09: a field name that exists in NEITHER schema, so the
    # highest-value route in the table fired on nothing and the corpus's 3,898
    # `medical_condition` mentions went unnormalized while the table claimed to cover them.
    # A route naming a field no schema has is now a test failure -- see
    # tests/test_normalize.py::test_every_route_names_a_real_field.
    ("groups.medical_condition", "ONVOC", ("disorders",)),
    ("groups.name", "ONVOC", ("disorders", "population")),
    ("assessments.name", "ONVOC", ("tests",)),
    ("regions.name", "ONVOC", ("regions",)),
    ("tasks.name", "CognitiveAtlas", ()),
    ("tasks.conditions.name", "CognitiveAtlas", ()),
    ("measures.source_label", "CognitiveAtlas", ()),
)

#: Routes whose value states the ABSENCE of what the vocabulary would match. A control
#: cohort's `medical_condition` is "no neurological or psychiatric disorder", and a disease
#: vocabulary retrieves a disease from it every time: measured over the 2,115-record corpus,
#: 22% of values state an absence and the ungated ladder mapped `absence of major
#: depressive disorder` to Depressive Disorder.
#:
#: `groups.name` is deliberately NOT gated, though it was at first and it looks like it
#: should be. The two fields ask different questions. `medical_condition` asks *what
#: condition*, so "none" is an answer to it. `name` asks *who was this group*, so "healthy
#: controls" is not an absence -- it is a study role, and one ONVOC has no term for.
#: Gating it suppressed the single best-supported proposal in the corpus: 640 mentions of
#: `healthy controls` / `healthy volunteers` / `healthy participants` across **592
#: studies**, classified as absences and therefore never proposed.
#:
#: And the gate bought nothing there. Of the 653 `groups.name` mentions it caught, 631 match
#: nothing either way -- so the gate decided only whether they were PROPOSED -- and all 22
#: that would have matched ungated matched CORRECTLY, to the population branch: `healthy
#: elderly` -> Elderly, `healthy weight children` -> Weight. Not one disorder false positive.
#:
#: `regions.name` and `assessments.name` are not gated because a region and a scale are
#: never reported as absent.
NEGATED: frozenset[str] = frozenset({"groups.medical_condition"})


def iter_targets(record: dict) -> Iterator[tuple[str, str, dict]]:
    """(routed path, text, the entity the value belongs to) for every routed field.

    The owner is yielded because a gate needs its siblings: where a `medical_condition`
    says nothing either way, the group's own `is_healthy` is the only thing that knows.
    """
    for route, _vocab, _branches in ROUTES:
        head, _, leaf = route.rpartition(".")
        for owner in _walk_to(record, head.split(".") if head else []):
            text = values.read(owner.get(leaf)) if isinstance(owner, dict) else None
            if isinstance(text, list):
                for item in text:
                    if isinstance(item, str) and item.strip():
                        yield route, item, owner
            elif isinstance(text, str) and text.strip():
                yield route, text, owner


def _walk_to(node: Any, steps: list[str]) -> Iterator[dict]:
    if not steps:
        if isinstance(node, dict):
            yield node
        return
    head, rest = steps[0], steps[1:]
    if isinstance(node, dict):
        child = node.get(head)
        if isinstance(child, list):
            for item in child:
                yield from _walk_to(item, rest)
        elif child is not None:
            yield from _walk_to(child, rest)


def candidates(
    mappings: Iterable["Mapping"], minimum: int = 1, grain: int | None = None
) -> list[Candidate]:
    """Terms the vocabulary should carry and does not, as proposals, with their evidence.

    Two kinds, and the distinction is the reason `rolled_up_to` exists:

      a gap in COVERAGE   nothing matched. `nicotine dependence` is in 135 mentions and
                          ONVOC has no term for it at any grain.
      a gap in GRAIN      something matched by generalizing, and the finer term recurs.
                          `behavioural variant frontotemporal dementia` maps to `Dementia`
                          and is named by enough studies to be a term in its own right.

    The second kind is off unless `grain` is given, because it is a threshold question
    rather than a fact: one study's `remitted anorexia nervosa` is that study's wording
    and forty studies' `bvFTD` is the literature's. Support is counted in distinct
    STUDIES for both -- a paper naming its diagnosis once per group is one piece of
    evidence that the term exists, not four -- so the threshold means what it says.

    Grouped on the head where triage produced one and on the value otherwise, with the
    parenthetical removed, so `Beck Depression Inventory` and `beck depression inventory
    (BDI)` count as one proposal. Grouping on the whole folded string looks equivalent and
    is not: the bracketed acronym is part of it and the two forms never meet.
    """

    routes = {path: groups for path, _vocab, groups in ROUTES}
    grouped: dict[tuple[str, str, str], dict] = {}
    for mapping in mappings:
        if mapping.sentinel:
            continue
        if mapping.matched and not (grain is not None and mapping.rollup):
            continue
        text = mapping.head or mapping.text
        under = mapping.concept.label if mapping.matched and mapping.concept else ""
        key = (fold(_PARENTHETICAL.sub("", text)) or fold(text), mapping.path, under)
        slot = grouped.setdefault(
            key,
            {
                "text": text,
                "path": mapping.path,
                "papers": set(),
                "expansions": set(),
                "under": under,
            },
        )
        slot["papers"].add(mapping.study_id)
        slot["expansions"].update(mapping.expansions)
        # Keep the longest surface form seen; it is the most informative proposal.
        if len(text) > len(slot["text"]):
            slot["text"] = text

    out = [
        Candidate(
            text=slot["text"],
            path=slot["path"],
            branch_group="+".join(routes.get(slot["path"], ())) or "-",
            papers=tuple(sorted(slot["papers"])),
            expansions=tuple(sorted(slot["expansions"])),
            rolled_up_to=slot["under"],
        )
        for slot in grouped.values()
    ]
    return sorted(
        [
            c
            for c in out
            if c.support >= (grain if c.rolled_up_to and grain is not None else minimum)
        ],
        key=lambda c: (-c.support, c.path, c.text.lower()),
    )


def _all_text(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _all_text(value, out)
    elif isinstance(node, list):
        for value in node:
            _all_text(value, out)
    elif isinstance(node, str) and len(node) > 2:
        out.append(node)


def corroborated(concept: Concept, record_text: str) -> bool:
    """Does the record itself spell out what this acronym was expanded to?

    An acronym unambiguous inside a vocabulary can still be the wrong referent outside
    it: ONVOC contains exactly one label whose initials are MDD, and it is Mood
    Dysregulation Disorder, while every paper writing MDD means Major Depressive
    Disorder. The vocabulary cannot tell those apart and the record can -- a paper that
    means the expansion almost always writes it somewhere.

    Hyphenation is not evidence of anything, and treating it as evidence made this refuse
    a correct expansion. ONVOC writes `Post-Traumatic Stress Disorder`, whose content words
    are {post, traumatic, stress}; the literature more often writes `posttraumatic`, one
    token, which is not a superset of those. So every paper spelling it closed had its
    `PTSD` refused -- measured at 10 studies, and PTSD is the third most common acronym in
    the corpus. The squashed form is checked as well, which is what `folding.squash` is
    for: "containment tests where spacing is noise".
    """

    wanted = content(concept.label)
    if not wanted:
        return False
    if wanted <= content(record_text):
        return True
    return squash(concept.label) in squash(record_text)


def normalize(
    record: dict, vocabularies: dict[str, Vocabulary], abbreviations: Any = None
) -> list[Mapping]:
    """Every mapping this record supports, matched and unmatched alike.

    Unmatched rows are kept on purpose. The useful question about a normalization layer
    is what it *cannot* place, and a list of only its successes cannot answer it.

    Three passes, and the order is what makes the result trustworthy:

      triage   on the routes that can carry an absence, decide whether the value names a
               condition at all, and split a comorbidity list into one head per row
      match    each head through the ladder, scoped to the branches the route allows
      grade    record what the match cost -- whether it was reached by generalizing, and
               what the value said at the corpus's own grain

    A row per head rather than per value: a cohort described as `alcohol abuse or
    dependence` is one value and two claims, and one row for it makes the second claim
    unqueryable.
    """
    study_id = values.read(record.get("local_id")) or ""
    routes = {path: (vocab, groups) for path, vocab, groups in ROUTES}
    strings: list[str] = []
    _all_text(record, strings)
    record_text = " ".join(strings)

    found: list[Mapping] = []
    for path, text, owner in iter_targets(record):
        vocabulary_name, groups = routes[path]
        vocabulary = vocabularies.get(vocabulary_name)
        if vocabulary is None:
            continue
        scoped = vocabulary.scoped(groups) if groups else vocabulary
        expanded: tuple[str, ...] = ()
        if abbreviations is not None:
            expanded = tuple(e for _short, e in expansions_in(text, abbreviations, study_id))

        parts = phrases.triage(text) if path in NEGATED else None
        if parts is not None and parts.sentinel:
            # The only gap `is_healthy` fills. It can add an absence, never remove one.
            sentinel = parts.sentinel
            if sentinel == phrases.NOT_READ and phrases.absent(text, values.read(
                owner.get("is_healthy")
            )):
                sentinel = phrases.NO_CONDITION
            found.append(
                Mapping(
                    str(study_id), path, text, None, expansions=expanded,
                    sentinel=sentinel, denied=parts.denied, scope=parts.scope,
                )
            )
            continue

        for head in parts.heads if parts is not None else (text,):
            concept, method, others = scoped.match(head, abbreviations, study_id)
            # An expansion the record never spells out is a guess, and a wrong mapping is
            # worse than a missing one -- it is the kind that gets queried across a corpus
            # and believed.
            if concept and method == "acronym" and not corroborated(concept, record_text):
                concept, method, others = None, "", []
            found.append(
                Mapping(
                    str(study_id),
                    path,
                    text,
                    concept,
                    method,
                    tuple(c.label for c in others),
                    expanded,
                    head=head,
                    corpus_term=head,
                    rollup=_generalized(concept, head),
                    qualifiers=parts.qualifiers if parts is not None else (),
                    denied=parts.denied if parts is not None else (),
                    scope=parts.scope if parts is not None else "",
                )
            )
    return found


def _generalized(concept: Concept | None, head: str) -> bool:
    """Was this concept reached by dropping part of what the value said?

    Compared on content words rather than on the raw string, so `Schizophrenia patients`
    -> Schizophrenia is not a rollup -- `patients` carries no identity and `labels.content`
    already says so -- while `behavioural-variant frontotemporal dementia` -> Dementia is.
    The distinction is the whole point of carrying the flag: the first mapping loses
    nothing and the second loses the subtype, and a query pooling first-episode with
    chronic cohorts needs to know which one it is reading.

    The concept's own acronym is not residue. `Mini-Mental State Examination (MMSE)` leaves
    `mmse` behind once the label's words are removed, and reading that as a finer grain put
    a term at the top of the grain-gap list that is the matched term spelled twice.
    """
    if not concept:
        return False
    residue = content(head) - content(concept.label)
    return bool(residue - {fold(acronym(concept.label))})
