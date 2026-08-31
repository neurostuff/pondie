"""An external vocabulary, its surface forms, and its hierarchy.

The link shape: a field whose answers already exist somewhere, so the work is reaching the
right entry rather than inventing one. MONDO is the target for conditions -- 32,102 classes,
90,374 surface forms, 21,658 carrying a UMLS CUI, plus the `is_a` edges the long tail needs.

Two things this module does that a plain string index does not:

  Rollup, stopped by corpus support.  A rare subtype is mapped to itself and then walked up
  to the nearest ancestor the CORPUS ITSELF uses often enough to be worth querying. Stopping
  at a fixed ontology depth would produce a target no query asks for; stopping at observed
  support makes the rollup target queryable by construction.

  A residual that is evidence.  What could not be placed is returned with its support, so a
  term used by ten papers and absent from the vocabulary is visible as a gap rather than
  silently dropped.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ._folding import fold

MONDO = Path("data/vocab/mondo.json")


@dataclass
class Vocabulary:
    """Labels, every surface form that reaches one, UMLS crosswalk, and `is_a` edges."""

    labels: list[str] = field(default_factory=list)
    ids: dict[int, str] = field(default_factory=dict)
    umls: dict[int, str] = field(default_factory=dict)
    surface: dict[str, int] = field(default_factory=dict)
    parents: dict[int, list[int]] = field(default_factory=dict)

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
                        seen.add(p); out.append(p); nxt.append(p)
            frontier = nxt
        return out

    def rollup(self, node: int, support: dict[int, int], minimum: int) -> int:
        """The nearest ancestor the corpus uses at least `minimum` times, else the node."""
        if support.get(node, 0) >= minimum:
            return node
        return next((a for a in self.ancestors(node) if support.get(a, 0) >= minimum), node)


def load_mondo(path: Path = MONDO) -> Vocabulary:
    graph = json.loads(path.read_text())["graphs"][0]
    vocab = Vocabulary()
    index_of: dict[str, int] = {}
    for node in graph["nodes"]:
        meta = node.get("meta") or {}
        if (node.get("type") != "CLASS" or not node.get("lbl") or meta.get("deprecated")
                or "MONDO_" not in node["id"]):
            continue
        i = len(vocab.labels)
        index_of[node["id"]] = i
        vocab.labels.append(node["lbl"])
        vocab.ids[i] = node["id"].rsplit("_", 1)[-1]
        vocab.surface.setdefault(fold(node["lbl"]), i)
        for syn in (meta.get("synonyms") or []):
            if syn.get("pred") == "hasExactSynonym" and syn.get("val"):
                vocab.surface.setdefault(fold(syn["val"]), i)
        cui = [x["val"] for x in (meta.get("xrefs") or [])
               if str(x.get("val", "")).startswith("UMLS:")]
        if cui:
            vocab.umls[i] = cui[0].split(":", 1)[1]
    parents = defaultdict(list)
    for edge in (graph.get("edges") or []):
        if edge.get("pred") == "is_a" and edge["sub"] in index_of and edge["obj"] in index_of:
            parents[index_of[edge["sub"]]].append(index_of[edge["obj"]])
    vocab.parents = dict(parents)
    return vocab
