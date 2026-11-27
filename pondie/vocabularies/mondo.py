"""MONDO: diseases, their surface forms, the `is_a` hierarchy, and the crosswalks out.

32,109 live classes and 102,615 exact surface forms, with a UMLS CUI on 67% and a SNOMED
concept id on 28% -- so linking to MONDO does not give up SNOMED, it gives up only the
SNOMED concepts no disease maps to. Why MONDO rather than SNOMED itself, and what the
ONVOC crosswalk can and cannot be trusted for: docs/condition-normalization.md.

Three things this does that a plain string index does not: roll a rare subtype up to the
nearest ancestor the CORPUS uses often enough to be worth querying; bridge to ONVOC by
identifier and then by ancestor; and return what could not be placed, with its support.
"""

from __future__ import annotations

import csv
import functools
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pondie import paths
from pondie.vocabularies.folding import fold

MONDO = paths.VOCAB / "mondo.json"
CROSSWALKS = paths.VOCAB / "onvoc-mappings"
#: Reviewed tables that ship with the code, unlike the releases above: surface forms MONDO
#: lacks or gets wrong (`mondo-aliases.tsv`), and MONDO -> ONVOC ties ONVOC's own crosswalk
#: lacks (`mondo-onvoc.tsv`). Each row names its reason.
CURATED = Path(__file__).with_name("curated")

#: A susceptibility entry (`hemorrhage, intracerebral, susceptibility to`) or a numbered
#: subtype or locus (`autism spectrum disorder 1`, `panic disorder 2`). 5,133 of 32,109
#: labels. A paper that writes no number means the disorder, so retrieval does not offer these
#: to a head without one, and a form a numbered node shares with an unnumbered one belongs to
#: the unnumbered (`CADASIL` -> the disease, not its type 1).
_LOCUS = re.compile(r"susceptibility|\s\d+[A-Z]?$")
#: A susceptibility entry's own forms are not indexed for an exact match, unless they say
#: susceptibility: `ICH` named intracerebral haemorrhage in every paper that wrote it, and the
#: only MONDO node with that form is the susceptibility entry.
_SUSCEPTIBILITY = re.compile(r"susceptib", re.I)


@dataclass
class Vocabulary:
    """Labels, every surface form that reaches one, the crosswalks, and `is_a` edges."""

    labels: list[str] = field(default_factory=list)
    ids: dict[int, str] = field(default_factory=dict)
    umls: dict[int, str] = field(default_factory=dict)
    #: SNOMED CT concept id, where MONDO publishes one.
    sctid: dict[int, str] = field(default_factory=dict)
    surface: dict[str, int] = field(default_factory=dict)
    parents: dict[int, list[int]] = field(default_factory=dict)
    #: Every exact surface form, with `form_node[i]` naming the node the i-th belongs to.
    #: Retrieval runs over this and not over `labels`: a label is one of the names a
    #: disease goes by and the corpus writes the others.
    forms: list[str] = field(default_factory=list)
    form_node: list[int] = field(default_factory=list)
    #: Nodes `_LOCUS` names, kept out of reach of a head that carries no number.
    loci: frozenset[int] = frozenset()
    #: Folded surface form -> node, from `curated/mondo-aliases.tsv`. Consulted before
    #: `surface`, so a row here also overrides a MONDO synonym that names the wrong node.
    aliases: dict[str, int] = field(default_factory=dict)
    #: Folded surface form -> the MONDO node a `condition_extensions` term falls under.
    extensions: dict[str, int] = field(default_factory=dict)

    def exact(self, text: object) -> int | None:
        return self.surface.get(fold(text))

    def ancestors(self, node: int, limit: int = 64) -> list[int]:
        """Nearest first, breadth-first. `limit` guards a cycle, not a depth preference."""
        seen, frontier, out = {node}, [node], []
        while frontier and len(out) < limit:
            nxt = []
            for n in frontier:
                for p in self.parents.get(n, ()):
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
                        nxt.append(p)
            frontier = nxt
        return out

    def rollup(self, node: int, support: dict[int, int], minimum: int) -> int:
        """The nearest ancestor the corpus uses at least `minimum` times, else the node."""
        if support.get(node, 0) >= minimum:
            return node
        return next((a for a in self.ancestors(node) if support.get(a, 0) >= minimum), node)

    def curie(self, node: int) -> str:
        return f"MONDO:{self.ids[node]}"


def load_mondo(path: Path = MONDO) -> Vocabulary:
    """MONDO as a Vocabulary. Raises with what to run when the file is not there."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Fetch the release with `python -m pondie.vocabularies.fetch "
            f"mondo` (about 100 MB, CC-BY, from purl.obolibrary.org/obo/mondo.json)."
        )
    graph = json.loads(path.read_text())["graphs"][0]
    vocab = Vocabulary()
    index_of: dict[str, int] = {}
    loci: set[int] = set()
    for node in graph["nodes"]:
        meta = node.get("meta") or {}
        if (
            node.get("type") != "CLASS"
            or not node.get("lbl")
            or meta.get("deprecated")
            or "MONDO_" not in node["id"]
        ):
            continue
        i = len(vocab.labels)
        index_of[node["id"]] = i
        vocab.labels.append(node["lbl"])
        vocab.ids[i] = node["id"].rsplit("_", 1)[-1]
        locus = bool(_LOCUS.search(node["lbl"]))
        susceptibility = bool(_SUSCEPTIBILITY.search(node["lbl"]))
        if locus:
            loci.add(i)

        def offer(
            form: str, node_index: int = i, locus: bool = locus, risk: bool = susceptibility
        ) -> None:
            """Index one surface form, and keep it where retrieval can see it."""
            key = fold(form)
            if not key:
                return
            vocab.forms.append(form)
            vocab.form_node.append(node_index)
            if risk and not _SUSCEPTIBILITY.search(key):
                return
            held = vocab.surface.get(key)
            if held is None or (held in loci and not locus):
                vocab.surface[key] = node_index

        offer(node["lbl"])
        for syn in meta.get("synonyms") or []:
            # Exact only: `hasRelatedSynonym` holds `schizophrenia 12` and
            # `hasBroadSynonym` holds the parent's name.
            if syn.get("pred") == "hasExactSynonym" and syn.get("val"):
                offer(syn["val"])
        for xref in meta.get("xrefs") or []:
            value = str(xref.get("val", ""))
            if value.startswith("UMLS:"):
                vocab.umls.setdefault(i, value.split(":", 1)[1])
            elif value.startswith("SCTID:"):
                vocab.sctid.setdefault(i, value.split(":", 1)[1])
    parents = defaultdict(list)
    for edge in graph.get("edges") or []:
        if edge.get("pred") == "is_a" and edge["sub"] in index_of and edge["obj"] in index_of:
            parents[index_of[edge["sub"]]].append(index_of[edge["obj"]])
    vocab.parents = dict(parents)
    vocab.loci = frozenset(loci)
    node_of = {f"MONDO:{mondo_id}": i for i, mondo_id in vocab.ids.items()}
    for row in _curated("mondo-aliases.tsv"):
        node = node_of.get(row["mondo"])
        # A row whose node this release does not have is skipped rather than raised: the
        # table outlives a release, and `test_mondo_bridge` names any row that went stale.
        if node is not None and fold(row["surface"]):
            vocab.aliases[fold(row["surface"])] = node
    for form, term in extension_forms().items():
        node = node_of.get(condition_extensions()[term].mondo_parent)
        if node is not None:
            vocab.extensions[form] = node
    return vocab


#: The `Normalization.vocabulary` of a `condition_extensions` term: study_schema's manifest name.
EXTENSION = "condition_extension"


@dataclass(frozen=True)
class ExtensionTerm:
    """A condition MONDO has no node for, and the MONDO node it falls under."""

    id: str
    label: str
    mondo_parent: str
    forms: tuple[str, ...]


@functools.lru_cache(maxsize=1)
def condition_extensions() -> dict[str, ExtensionTerm]:
    """`curated/condition-extension.tsv` by id. Needs no MONDO release."""
    return {
        row["id"]: ExtensionTerm(
            row["id"], row["label"], row["mondo_parent"],
            tuple(f.strip() for f in row["forms"].split(";") if f.strip()),
        )
        for row in _curated("condition-extension.tsv")
    }


@functools.lru_cache(maxsize=1)
def extension_forms() -> dict[str, str]:
    """Folded form -> the `condition_extensions` id it names: each term's label and forms."""
    return {
        fold(form): term.id
        for term in condition_extensions().values()
        for form in (term.label, *term.forms)
        if fold(form)
    }


def _curated(name: str) -> list[dict[str, str]]:
    """The rows of one reviewed table under `CURATED`, `#` lines skipped."""
    path = CURATED / name
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    return [
        {k: (v or "").strip() for k, v in row.items()}
        for row in csv.DictReader(lines, delimiter="\t")
    ]


#: Where a crosswalk row came from: ONVOC's published file, or `curated/mondo-onvoc.tsv`.
PUBLISHED, CURATED_ROW = "published", "curated"


def onvoc_crosswalk(directory: Path = CROSSWALKS) -> dict[str, tuple[str, str, str]]:
    """MONDO CURIE -> (ONVOC id, ONVOC label, source): ONVOC's own crosswalk, then ours.

    Keyed this way because the lookup runs this way: a value is matched against MONDO, and
    the question afterwards is which ONVOC term that node belongs under. ONVOC's file is 90
    rows over 69 ONVOC ids, reaching 66 of the 205 disorder-branch concepts; the curated rows
    (`CURATED_ROW`) add ties it lacks, and `medical_condition.bridge` walks them last.
    """
    claims: dict[str, set[tuple[str, str]]] = defaultdict(set)
    path = directory / "mondo.tsv"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                curie = (row.get("mapped_term_curie") or "").strip()
                onvoc_id = (row.get("vocabulary_id") or "").strip()
                if curie.startswith("MONDO:") and onvoc_id:
                    claims[curie].add((onvoc_id, (row.get("vocabulary_term") or "").strip()))
    # A node claimed by two ONVOC terms cannot decide between them, so it decides nothing:
    # `MONDO:0005148` is listed under both Type 1 and Type 2 Diabetes Mellitus.
    out = {curie: (*next(iter(tie)), PUBLISHED) for curie, tie in claims.items() if len(tie) == 1}
    if not out:
        return out
    # The reviewed rows add ties ONVOC's file lacks. Applied only beside a published crosswalk,
    # so a host without the ONVOC release does not bridge on these alone.
    for row in _curated("mondo-onvoc.tsv"):
        out[row["mondo"]] = (row["onvoc_id"], row["onvoc_label"], CURATED_ROW)
    return out
