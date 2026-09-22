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
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pondie import paths
from pondie.vocabularies.folding import fold

MONDO = paths.VOCAB / "mondo.json"
CROSSWALKS = paths.VOCAB / "onvoc-mappings"


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

        def offer(form: str, node_index: int = i) -> None:
            """Index one surface form, and keep it where retrieval can see it."""
            key = fold(form)
            if not key:
                return
            vocab.surface.setdefault(key, node_index)
            vocab.forms.append(form)
            vocab.form_node.append(node_index)

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
    return vocab


def onvoc_crosswalk(directory: Path = CROSSWALKS) -> dict[str, tuple[str, str]]:
    """MONDO CURIE -> (ONVOC id, ONVOC label), from the crosswalk ONVOC itself publishes.

    Keyed this way because the lookup runs this way: a value is matched against MONDO, and
    the question afterwards is which ONVOC term that node belongs under. 90 rows over 69
    ONVOC ids, reaching 66 of the 205 disorder-branch concepts -- so a hit is evidence, not
    proof, and the ancestor and name layers in `medical_condition.bridge` cover the rest.
    """
    path = directory / "mondo.tsv"
    if not path.is_file():
        return {}
    claims: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            curie = (row.get("mapped_term_curie") or "").strip()
            onvoc_id = (row.get("vocabulary_id") or "").strip()
            if curie.startswith("MONDO:") and onvoc_id:
                claims[curie].add((onvoc_id, (row.get("vocabulary_term") or "").strip()))
    # A node claimed by two ONVOC terms cannot decide between them, so it decides nothing:
    # `MONDO:0005148` is listed under both Type 1 and Type 2 Diabetes Mellitus.
    return {curie: next(iter(tie)) for curie, tie in claims.items() if len(tie) == 1}
