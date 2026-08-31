#!/usr/bin/env python3
"""Score a candidate extraction against a gold record.

Four questions, in the order they depend on each other:

1. **Objects** -- did the extractor create the entities the gold record has? Names differ
   between extractors, so entities are matched by optimal bipartite assignment over a
   similarity score rather than by `local_id`. Precision, recall and F1 fall out of the
   matching.
2. **Relationships** -- with the entity map in hand, every cross-reference becomes a
   comparable triple. False positives and false negatives are both listed, not just counted.
3. **Fields** -- for each matched pair, per-field agreement. Numbers are coerced and
   compared with a tolerance; free text is scored by fuzzy overlap and, with `--semantic`,
   by embedding cosine; enums are exact.
4. **Direction** -- the highest-weighted question. A `Cell` is only credited when its
   `term` resolves to the nominally same ModelTerm in the gold record, so a right sign on
   the wrong term earns nothing.

Everything after step 1 is conditional on the matching, so unmatched entities are always
reported alongside the field scores; a high field accuracy over two matched entities out
of ten is not a good extraction.

    python compare_extractions.py data/records/xevP8UDRAVh9.extraction.json \\
        --gold data/gold/xevP8UDRAVh9.extraction.json
    python compare_extractions.py data/records/*.json --gold-dir data/gold --json out.json

Why these metrics and not others: docs/extraction-comparison-metrics.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dataclass_field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


from .. import _schema  # noqa: F401 -- schema_utils lives in the submodule
from schema_utils import (  # noqa: E402
    EXTRACTED_VALUE,
    LOCAL_ID,
    attribute_ranges,
    attributes_for,
    load_imported_classes,
    resolves_to,
    subclasses_of,
)

ROOT = Path(__file__).resolve().parents[2]
#: The schema is a submodule of this repository, not a sibling directory.
SCHEMA = _schema.ROOT / "neuroimaging-study-extraction.yaml"
CACHE = ROOT / ".cache" / "compare_embeddings.json"

#: Provenance, not extraction: comparing these would score the pipeline, not the reading.
SKIP_SLOTS = {"evidence", "extraction_metadata", "value_source"}

#: Slots that carry an entity's identity, weighted up so a name is not drowned by forty
#: sparse attributes when the assignment problem is scored.
IDENTIFYING = {
    "name", "definition", "description", "source_label", "caption", "table_number",
    "model", "manufacturer", "level", "model_type", "estimator", "atlas", "category",
    "specific_metric", "pulse_sequence_type", "acquisition_type", "type", "family",
}

#: Excluded from the score that matches two inline objects, because the metric being
#: computed is agreement on exactly this field. Matching cells on their direction would
#: manufacture the agreement the direction metrics are supposed to measure.
ALIGN_EXCLUDE = {"Cell": {"direction"}}

#: An Analysis's prose says which way its contrast went ("Positive correlation"), so an
#: alignment that reads it has seen the answer. The structure-only pass drops these and
#: matches on what the analysis was *of* -- its measure, groups, model, cell terms and
#: levels -- giving a leak-free second reading of the headline number.
DIRECTION_LEAKING = {"Analysis": {"name", "definition", "interpretations"}}

DIRECTIONAL = {"positive", "negative"}
FLIP = {"positive": "negative", "negative": "positive"}

#: Direction outranks everything else because it is the fact a synthesis cannot recover
#: from anywhere else in the record: a wrong sign inverts the finding.
COMPOSITE_WEIGHTS = {
    "direction": 0.45,
    "entities": 0.20,
    "relationships": 0.20,
    "fields": 0.15,
}

MATCH_THRESHOLD = 0.45
CELL_THRESHOLD = 0.35
NUMERIC_RTOL = 0.01
NUMERIC_ATOL = 1e-9
TEXT_MATCH = 0.85


# ---------------------------------------------------------------------------
# text and number similarity
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")


#: Comparison operators become words before punctuation is stripped, because in a contrast
#: name they are not punctuation -- they are the whole of what it says. `A > B` and `A < B`
#: are the two directions of one comparison, and the schema makes them separate Analyses for
#: that reason. Deleted with the rest of the punctuation they normalize to the same string,
#: `fuzzy` scores them 1.0, and the assignment in §1 pairs a contrast with its own mirror --
#: at which point every cell beneath it reads as a sign flip and the direction metric reports
#: the failure it just manufactured. Spelled-out forms are folded to the same tokens so
#: `A > B` and `A greater than B` still meet.
_COMPARATORS = (
    ("\u2265", " gte "), ("\u2264", " lte "), ("\u2260", " ne "),
    (">=", " gte "), ("<=", " lte "), (">", " gt "), ("<", " lt "),
)
_COMPARATOR_WORDS = (
    (r"\bgreater than or equal to\b", " gte "), (r"\bless than or equal to\b", " lte "),
    (r"\bgreater than\b", " gt "), (r"\bless than\b", " lt "),
    (r"\bversus\b", " vs "),
)


def normalize(text: str) -> str:
    raw = str(text)
    for symbol, word in _COMPARATORS:
        raw = raw.replace(symbol, word)
    folded = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode().lower()
    for pattern, word in _COMPARATOR_WORDS:
        folded = re.sub(pattern, word, folded)
    return _WS.sub(" ", _PUNCT.sub(" ", folded)).strip()


def fuzzy(a: str, b: str) -> float:
    """Character-level, token-order-free and containment agreement, whichever is kindest.

    Containment carries an abbreviation against its expansion -- "SCID-II" inside
    "Structured Clinical Interview for DSM-IV Axis II Disorders (SCID-II)" -- which
    neither an edit ratio nor a symmetric token overlap will find. It is discounted so a
    one-word substring cannot claim a perfect match.
    """

    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 1.0 if na == nb else 0.0
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    shared = len(ta & tb)
    dice = 2 * shared / (len(ta) + len(tb)) if ta and tb else 0.0
    containment = shared / min(len(ta), len(tb)) if ta and tb else 0.0
    return max(SequenceMatcher(None, na, nb).ratio(), dice, 0.9 * containment)


class Semantics:
    """Embedding cosine, when asked for and reachable; fuzzy overlap otherwise.

    Texts are embedded in one prepass rather than pair by pair: the assignment problems
    below are quadratic in entity count, and a per-comparison call would issue thousands
    of requests for a few hundred distinct strings.
    """

    def __init__(self, enabled: bool, model: str = "text-embedding-3-small",
                 base_url: str | None = None) -> None:
        self.enabled = enabled
        self.model = model
        self.base_url = base_url
        self.vectors: dict[str, list[float]] = {}
        self._cache: dict[str, list[float]] = {}
        if enabled and CACHE.is_file():
            try:
                cached = json.loads(CACHE.read_text(encoding="utf-8"))
                self._cache = cached.get(model, {})
            except (OSError, ValueError):
                self._cache = {}

    def prepare(self, texts: Iterable[str]) -> None:
        if not self.enabled:
            return
        wanted = sorted({t[:2000] for t in texts if t and t.strip()})
        missing = [t for t in wanted if t not in self._cache]
        if missing:
            # Any failure here -- no key, no network, a gateway that does not route the
            # embeddings endpoint -- degrades to fuzzy rather than losing the whole run.
            # The other metrics do not depend on embeddings and should still be produced.
            try:
                client = _openai_client(self.base_url)
                for start in range(0, len(missing), 128):
                    batch = missing[start:start + 128]
                    response = client.embeddings.create(model=self.model, input=batch)
                    for text, item in zip(batch, response.data):
                        self._cache[text] = list(item.embedding)
            except Exception as exc:  # noqa: BLE001
                print(f"note: semantic similarity unavailable ({type(exc).__name__}: {exc}); "
                      "scoring strings by fuzzy overlap only.", file=sys.stderr)
                self.enabled = False
                return
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if CACHE.is_file():
                try:
                    existing = json.loads(CACHE.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    existing = {}
            existing[self.model] = self._cache
            CACHE.write_text(json.dumps(existing), encoding="utf-8")
        self.vectors = {t: self._cache[t] for t in wanted if t in self._cache}

    def similarity(self, a: str, b: str) -> float:
        surface = fuzzy(a, b)
        if not self.enabled:
            return surface
        va, vb = self.vectors.get(a[:2000]), self.vectors.get(b[:2000])
        if not va or not vb:
            return surface
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(y * y for y in vb))
        if not na or not nb:
            return surface
        cosine = dot / (na * nb)
        # Cosine over a modern embedding space floors around 0.3 for unrelated text;
        # rescaling keeps a graded score comparable with the fuzzy one it is maxed against.
        return max(surface, max(0.0, (cosine - 0.3) / 0.7))


def _openai_client(base_url: str | None = None):
    """The gateway `OPENAI_API_GATEWAY` names, unless one was given for embeddings alone.

    A Portkey-style gateway routes chat completions on a virtual key and rejects
    `/embeddings` without a provider header, so `--embedding-base-url` exists to send this
    one endpoint straight at the provider.
    """

    if not os.environ.get("OPENAI_API_KEY"):
        env = ROOT / ".env"
        if env.is_file():
            for raw in env.read_text(encoding="utf-8").splitlines():
                line = raw.strip().removeprefix("export ").strip()
                if line and not line.startswith("#") and "=" in line:
                    name, _, value = line.partition("=")
                    os.environ.setdefault(name.strip(), value.strip().strip("'\""))
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=base_url or os.environ.get("OPENAI_EMBEDDING_BASE_URL")
        or os.environ.get("OPENAI_API_GATEWAY"),
    )


def as_number(value: Any) -> float | None:
    """Coerce to float so `2`, `2.0` and `"2"` are one value rather than three."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value.replace(",", ""))
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


def numeric_agreement(a: float, b: float) -> tuple[bool, float]:
    """Tolerance verdict and a graded closeness, from relative distance.

    Relative rather than absolute because the fields span twelve orders of magnitude --
    an alpha level of 0.05 and a permutation count of 5000 cannot share an absolute
    tolerance, and one that suits either is meaningless for the other.
    """

    scale = max(abs(a), abs(b), NUMERIC_ATOL)
    error = abs(a - b) / scale
    return error <= NUMERIC_RTOL, max(0.0, 1.0 - min(error, 1.0))


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------

def hungarian(score: list[list[float]]) -> dict[int, int]:
    """Maximum-weight bipartite matching, rows to columns, by shortest augmenting path.

    Greedy best-first matching is not enough here: one strong pair can consume the only
    partner a second pair had, and the resulting entity map then mis-scores every field
    and relationship hanging off it. O(n^3) is irrelevant at these sizes.
    """

    if not score or not score[0]:
        return {}
    transposed = len(score) > len(score[0])
    cost = [[-v for v in row] for row in score]
    if transposed:
        cost = [list(row) for row in zip(*cost)]
    n, m = len(cost), len(cost[0])
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = {p[j] - 1: j - 1 for j in range(1, m + 1) if p[j] > 0}
    return {c: r for r, c in pairs.items()} if transposed else pairs


def match(rows: Sequence[Any], cols: Sequence[Any], scorer, threshold: float
          ) -> list[tuple[int, int, float]]:
    """Assign rows to columns, dropping pairs the score does not support."""

    if not rows or not cols:
        return []
    grid = [[scorer(r, c) for c in cols] for r in rows]
    return [
        (i, j, grid[i][j])
        for i, j in sorted(hungarian(grid).items())
        if grid[i][j] >= threshold
    ]


# ---------------------------------------------------------------------------
# schema-driven flattening
# ---------------------------------------------------------------------------

@dataclass
class Field:
    path: str
    kind: str
    status: str
    value: Any


@dataclass
class Entity:
    #: The declared range of the slot that held it -- Acquisition, not MRI. Entities are
    #: matched within a family so that a candidate typing an acquisition MRI where gold
    #: says PET is scored as one wrong field rather than as two unmatched objects.
    etype: str
    local_id: str
    fields: dict[str, Field] = dataclass_field(default_factory=dict)
    edges: set[tuple[str, str]] = dataclass_field(default_factory=set)
    #: path -> (class name, objects). Held unflattened because their paths are only
    #: stable once the two records' members have been matched to each other.
    inline: dict[str, tuple[str, list[dict]]] = dataclass_field(default_factory=dict)
    #: The entity this one is declared inside, when it is nested rather than top-level:
    #: a ModelTerm's ModelEstimation. Part of its identity -- a term of one model is not
    #: a term of another however alike they read.
    parent: str | None = None
    #: Every reference this entity makes, including those held by its inline objects, with
    #: list indices stripped. Complete from `flatten` onwards, unlike `edges`, so the
    #: matcher can use references that live on a Cell or an AnalysisGroup.
    ref_edges: set[tuple[str, str]] = dataclass_field(default_factory=set)


@dataclass
class Record:
    label: str
    entities: dict[str, Entity] = dataclass_field(default_factory=dict)
    by_type: dict[str, list[Entity]] = dataclass_field(default_factory=lambda: defaultdict(list))
    #: target -> {(source local_id, path)}. Which entities point at this one, and through
    #: which slot. For an entity that exists to be referenced -- a Measure, a Region, a
    #: continuous ModelTerm with no levels and so no outgoing edges at all -- this is most
    #: of what identifies it, and reading only outgoing edges leaves it identified by name.
    incoming: dict[str, set[tuple[str, str]]] = dataclass_field(
        default_factory=lambda: defaultdict(set))


class Schema:
    def __init__(self, path: Path = SCHEMA) -> None:
        self.classes = load_imported_classes(path)
        self.enums = load_imported_classes(path, key="enums")
        self._designators: dict[str, str | None] = {}

    def is_class(self, name: str | None) -> bool:
        return bool(name) and name in self.classes

    def is_extracted(self, name: str | None) -> bool:
        return bool(name) and resolves_to(self.classes, name, EXTRACTED_VALUE)

    def value_kind(self, wrapper: str) -> str:
        """The comparison kind behind an ExtractedValue subclass."""

        attrs = attributes_for(self.classes, wrapper)
        value = attrs.get("value") or {}
        ranges = attribute_ranges(value)
        base = ranges[0] if ranges else "string"
        multi = bool(value.get("multivalued"))
        if base in self.enums:
            kind = f"enum:{base}"
        elif base in ("integer", "float", "double", "decimal"):
            kind = "number"
        elif base == "boolean":
            kind = "boolean"
        else:
            kind = "string"
        return f"{kind}[]" if multi else kind

    def concrete(self, declared: str, node: Mapping) -> str:
        """Resolve an abstract range through its type designator, e.g. AnalysisDetails."""

        if declared not in self._designators:
            found = None
            for name in {declared} | subclasses_of(self.classes, declared):
                for slot, attr in attributes_for(self.classes, name).items():
                    if attr.get("designates_type") is True:
                        found = slot
            self._designators[declared] = found
        slot = self._designators[declared]
        named = node.get(slot) if slot else None
        return named if isinstance(named, str) and named in self.classes else declared


def flatten(record: Mapping, schema: Schema, label: str) -> Record:
    out = Record(label=label)

    def entity(etype: str, node: Mapping, parent: Entity | None = None) -> Entity:
        ent = Entity(etype=etype, local_id=str(node.get(LOCAL_ID)),
                     parent=parent.local_id if parent else None)
        out.entities[ent.local_id] = ent
        out.by_type[etype].append(ent)
        return ent

    def visit(node: Mapping, class_name: str, owner: Entity, prefix: str) -> None:
        class_name = schema.concrete(class_name, node)
        attrs = attributes_for(schema.classes, class_name)
        for slot, raw in node.items():
            if slot in SKIP_SLOTS or raw is None:
                continue
            attr = attrs.get(slot)
            if attr is None:
                continue
            ranges = [r for r in attribute_ranges(attr) if r]
            target = next((r for r in ranges if schema.is_class(r)), None)
            path = f"{prefix}{slot}"

            if target and schema.is_extracted(target):
                if isinstance(raw, Mapping):
                    owner.fields[path] = Field(
                        path=path,
                        kind=schema.value_kind(target),
                        status=str(raw.get("extraction_status", "extracted")),
                        value=raw.get("value"),
                    )
                continue

            if target and attr.get("inlined") is False:
                for ref in (raw if isinstance(raw, list) else [raw]):
                    if isinstance(ref, str):
                        owner.edges.add((path, ref))
                continue

            if target:
                members = raw if isinstance(raw, list) else [raw]
                members = [m for m in members if isinstance(m, Mapping)]
                if not members:
                    continue
                if any(m.get(LOCAL_ID) for m in members):
                    for member in members:
                        child = entity(target, member, parent=owner)
                        visit(member, target, child, "")
                elif isinstance(raw, list):
                    owner.inline[path] = (target, members)
                else:
                    visit(members[0], target, owner, f"{path}.")
                continue

            # A bare scalar the schema left unwrapped: local_id, a type designator, an
            # acquisition_type. Real content, so compared, but never a missingness fact.
            if slot != LOCAL_ID and isinstance(raw, (str, int, float, bool)):
                kind = "number" if isinstance(raw, (int, float)) and not isinstance(raw, bool) \
                    else "boolean" if isinstance(raw, bool) else "string"
                owner.fields[path] = Field(path=path, kind=kind, status="extracted", value=raw)

    root = entity("Study", record if record.get(LOCAL_ID) else {**record, LOCAL_ID: label})
    visit(record, "Study", root, "")

    # References held by inline objects -- Cell.term, AnalysisGroup.group, FactorLevel.arms
    # -- belong to the owning entity for matching purposes. Gathered here so `ref_edges` is
    # complete before anything reads it, rather than only after the field pass has run.
    for ent in out.entities.values():
        ent.ref_edges = set(ent.edges)
        for path, (cls, members) in ent.inline.items():
            for member in members:
                _, inline_edges = flatten_inline(member, cls, schema, "")
                ent.ref_edges |= {(f"{path}[].{p}", t) for p, t in inline_edges}
    for ent in out.entities.values():
        for path, target in ent.ref_edges:
            out.incoming[target].add((ent.local_id, path))
    return out


def flatten_inline(node: Mapping, class_name: str, schema: Schema, prefix: str
                   ) -> tuple[dict[str, Field], set[tuple[str, str]]]:
    """Fields and edges of one member of an inline list, addressed under `prefix`."""

    holder = Entity(etype=class_name, local_id="")

    def visit(obj: Mapping, cls: str, pre: str) -> None:
        cls = schema.concrete(cls, obj)
        attrs = attributes_for(schema.classes, cls)
        for slot, raw in obj.items():
            if slot in SKIP_SLOTS or raw is None:
                continue
            attr = attrs.get(slot)
            if attr is None:
                continue
            ranges = [r for r in attribute_ranges(attr) if r]
            target = next((r for r in ranges if schema.is_class(r)), None)
            path = f"{pre}{slot}"
            if target and schema.is_extracted(target):
                if isinstance(raw, Mapping):
                    holder.fields[path] = Field(
                        path=path, kind=schema.value_kind(target),
                        status=str(raw.get("extraction_status", "extracted")),
                        value=raw.get("value"),
                    )
            elif target and attr.get("inlined") is False:
                for ref in (raw if isinstance(raw, list) else [raw]):
                    if isinstance(ref, str):
                        holder.edges.add((path, ref))
            elif target:
                members = [m for m in (raw if isinstance(raw, list) else [raw])
                           if isinstance(m, Mapping)]
                if isinstance(raw, list):
                    holder.inline[path] = (target, members)
                elif members:
                    visit(members[0], target, f"{path}.")
            elif isinstance(raw, (str, int, float, bool)):
                kind = "number" if isinstance(raw, (int, float)) and not isinstance(raw, bool) \
                    else "boolean" if isinstance(raw, bool) else "string"
                holder.fields[path] = Field(path=path, kind=kind, status="extracted", value=raw)

    visit(node, class_name, "")
    return ({f"{prefix}{k}": Field(f"{prefix}{k}", v.kind, v.status, v.value)
             for k, v in holder.fields.items()},
            {(f"{prefix}{p}", t) for p, t in holder.edges})


# ---------------------------------------------------------------------------
# value comparison
# ---------------------------------------------------------------------------

@dataclass
class ValueVerdict:
    match: bool
    score: float
    numeric_error: float | None = None


def compare_values(kind: str, gold: Any, cand: Any, sem: Semantics) -> ValueVerdict:
    if kind.endswith("[]"):
        return compare_lists(kind[:-2], gold, cand, sem)
    if kind == "number":
        g, c = as_number(gold), as_number(cand)
        if g is None or c is None:
            return ValueVerdict(match=gold == cand, score=1.0 if gold == cand else 0.0)
        ok, score = numeric_agreement(g, c)
        return ValueVerdict(match=ok, score=score, numeric_error=abs(g - c))
    if kind == "boolean":
        return ValueVerdict(match=bool(gold) == bool(cand),
                            score=1.0 if bool(gold) == bool(cand) else 0.0)
    if kind.startswith("enum:"):
        exact = normalize(str(gold)) == normalize(str(cand))
        # Half credit for near-misses keeps the graded score informative on the open
        # vocabularies (variation_level, assessment_type) without ever calling them right.
        return ValueVerdict(match=exact, score=1.0 if exact else 0.5 * fuzzy(str(gold), str(cand)))
    score = sem.similarity(str(gold), str(cand))
    return ValueVerdict(match=score >= TEXT_MATCH, score=score)


def compare_lists(kind: str, gold: Any, cand: Any, sem: Semantics) -> ValueVerdict:
    """Set agreement, order-free, by best assignment between the two lists.

    Positional comparison would be wrong for a list of inclusion criteria and right for a
    voxel size, so the elements are matched rather than zipped; a permuted voxel size is a
    real error, but it is one `acquisition_voxel_size_mm` states in its own description
    rather than one this scorer should invent a rule for.
    """

    g = list(gold) if isinstance(gold, list) else [gold]
    c = list(cand) if isinstance(cand, list) else [cand]
    if not g and not c:
        return ValueVerdict(match=True, score=1.0)
    if not g or not c:
        return ValueVerdict(match=False, score=0.0)
    pairs = match(g, c, lambda a, b: compare_values(kind, a, b, sem).score, 0.0)
    scores = [s for _, _, s in pairs]
    hits = sum(1 for i, j, _ in pairs if compare_values(kind, g[i], c[j], sem).match)
    precision = hits / len(c)
    recall = hits / len(g)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    graded = sum(scores) / max(len(g), len(c))
    return ValueVerdict(match=hits == len(g) == len(c), score=max(f1, graded))


# ---------------------------------------------------------------------------
# entity alignment
# ---------------------------------------------------------------------------

def _field_key(node: Field | None) -> str:
    """A field's value reduced to something comparable for equality."""

    if node is None or node.status != "extracted":
        return f"<{node.status if node else 'absent'}>"
    return json.dumps(node.value, sort_keys=True, default=str)


def discriminative_weights(record: Record) -> dict[tuple[str, str], float]:
    """How well each field separates the instances of its own type, in [0, 1].

    A field every Acquisition agrees on says nothing about which Acquisition you are
    looking at, however prominent it is; one that differs across all of them says almost
    everything. Weighting by the fraction of instance pairs a field actually separates is
    what stops thirty agreeing boilerplate slots from outvoting the one that matters --
    and it is measured on this record rather than assumed, so it adapts to the paper.
    """

    weights: dict[tuple[str, str], float] = {}
    for etype, entities in record.by_type.items():
        if len(entities) < 2:
            continue
        for path in {p for e in entities for p in e.fields}:
            keys = [_field_key(e.fields.get(path)) for e in entities]
            pairs = separated = 0
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    pairs += 1
                    separated += keys[i] != keys[j]
            weights[(etype, path)] = separated / pairs if pairs else 0.0
    return weights


class Aligner:
    """Collective entity resolution over attributes and neighbourhood, to a fixed point.

    An entity's identity is where it sits in the record as much as what it says. A Measure
    is the quantity these three analyses measured; a ModelTerm is the term those cells are
    contrasts on, declared by that model. Four kinds of evidence are combined, and each is
    dropped when an entity has none of it rather than counted as disagreement:

    - **attributes**, weighted by how well each one separates instances of its type
    - **outgoing** references, translated through the alignment so far
    - **incoming** references -- who points here, through which slot. The only structural
      evidence a referenced-but-referencing-nothing entity has, and a continuous ModelTerm
      is exactly that: no levels, so no outgoing edges, so previously nothing but its name.
    - **containment** -- a term of an aligned model, versus a term of some other one

    The relational three need an alignment to be read through, and produce a better one, so
    the passes repeat until the map stops moving.
    """

    PASSES = 4
    WEIGHTS = {"attributes": 0.45, "outgoing": 0.20, "incoming": 0.25, "parent": 0.10}

    def __init__(self, gold: Record, cand: Record, schema: Schema, sem: Semantics,
                 exclude: Mapping[str, set[str]] | None = None) -> None:
        self.gold, self.cand, self.schema, self.sem = gold, cand, schema, sem
        self.exclude = exclude or {}
        self.weights = discriminative_weights(gold)
        self.map: dict[str, str] = {}
        self.scores: dict[str, float] = {}
        self.parts: dict[str, dict[str, float]] = {}
        self.run()
        #: gold local_id -> candidate local_id, for walking the gold record in its own order.
        self.inverse = {g: c for c, g in self.map.items()}

    def field_score(self, etype: str, a: Entity, b: Entity) -> float:
        blocked = self.exclude.get(etype, set())
        total = weighted = 0.0
        for path in set(a.fields) | set(b.fields):
            if path.split(".")[0] in blocked:
                continue
            fa, fb = a.fields.get(path), b.fields.get(path)
            base = 3.0 if path.split(".")[-1] in IDENTIFYING else 1.0
            # Unknown discriminative power (a type with one instance) sits at the midpoint,
            # so a lone Group is still matched on its attributes rather than on nothing.
            power = self.weights.get((etype, path), 0.5)
            weight = base * (0.5 + power)
            if fa is None or fb is None:
                total += weight * 0.5
            elif fa.status != "extracted" or fb.status != "extracted":
                total += weight * (1.0 if fa.status == fb.status else 0.0)
            else:
                total += weight * compare_values(fa.kind, fa.value, fb.value, self.sem).score
            weighted += weight
        return total / weighted if weighted else 0.5

    def _dice(self, gold_set: set, cand_set: set) -> float | None:
        if not gold_set and not cand_set:
            return None
        shared = len(gold_set & cand_set)
        return 2 * shared / (len(gold_set) + len(cand_set))

    def outgoing_score(self, a: Entity, b: Entity) -> float | None:
        return self._dice(a.ref_edges, {(p, self.map.get(t, t)) for p, t in b.ref_edges})

    def incoming_score(self, a: Entity, b: Entity) -> float | None:
        return self._dice(
            self.gold.incoming.get(a.local_id, set()),
            {(self.map.get(s, s), p) for s, p in self.cand.incoming.get(b.local_id, set())})

    def parent_score(self, a: Entity, b: Entity) -> float | None:
        if a.parent is None and b.parent is None:
            return None
        if a.parent is None or b.parent is None:
            return 0.0
        return 1.0 if self.map.get(b.parent) == a.parent else 0.0

    def pair_score(self, etype: str, a: Entity, b: Entity) -> float:
        attributes = self.field_score(etype, a, b)
        # `local_id` is the extractor's own naming convention, not a fact about the paper.
        # It agrees far too often between runs of the same prompt to be trusted, and not at
        # all between different ones, so it only breaks ties.
        attributes = 0.9 * attributes + 0.1 * fuzzy(a.local_id, b.local_id)
        parts = {"attributes": attributes}
        if self.map:
            for name, value in (("outgoing", self.outgoing_score(a, b)),
                                ("incoming", self.incoming_score(a, b)),
                                ("parent", self.parent_score(a, b))):
                if value is not None:
                    parts[name] = value
        total = sum(self.WEIGHTS[k] for k in parts)
        return sum(self.WEIGHTS[k] * v for k, v in parts.items()) / total

    def explain(self, etype: str, a: Entity, b: Entity) -> dict[str, float]:
        """The components behind one pair's score, for the structure report."""

        parts = {"attributes": round(self.field_score(etype, a, b), 3)}
        for name, value in (("outgoing", self.outgoing_score(a, b)),
                            ("incoming", self.incoming_score(a, b)),
                            ("parent", self.parent_score(a, b))):
            if value is not None:
                parts[name] = round(value, 3)
        return parts

    def run(self) -> None:
        for _ in range(self.PASSES):
            new: dict[str, str] = {}
            scores: dict[str, float] = {}
            for etype in set(self.gold.by_type) | set(self.cand.by_type):
                g = self.gold.by_type.get(etype, [])
                c = self.cand.by_type.get(etype, [])
                for i, j, s in match(g, c, lambda x, y, t=etype: self.pair_score(t, x, y),
                                     MATCH_THRESHOLD):
                    new[c[j].local_id] = g[i].local_id
                    scores[g[i].local_id] = s
            if new == self.map:
                break
            self.map, self.scores = new, scores
        self.parts = {
            g: self.explain(self.gold.entities[g].etype,
                            self.gold.entities[g], self.cand.entities[c])
            for c, g in self.map.items()
        }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": true_positive, "fp": false_positive, "fn": false_negative}


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float:
    """Agreement above what the two label distributions would produce by chance.

    Raw accuracy flatters a skewed vocabulary: a record whose cells are 80% `positive`
    gets 0.8 from an extractor that answers `positive` every time, and kappa near zero.
    """

    if not pairs:
        return float("nan")
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    ga, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(ga[k] * cb[k] for k in set(ga) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def per_class_prf(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    labels = sorted(set(a for a, _ in pairs) | set(b for _, b in pairs))
    out = {}
    for label in labels:
        tp = sum(1 for a, b in pairs if a == label and b == label)
        fp = sum(1 for a, b in pairs if a != label and b == label)
        fn = sum(1 for a, b in pairs if a == label and b != label)
        out[label] = prf(tp, fp, fn)
    macro = sum(v["f1"] for v in out.values()) / len(out) if out else 0.0
    return {"per_class": out, "macro_f1": macro}


def bootstrap(values: Sequence[float], draws: int = 2000, seed: int = 0
              ) -> tuple[float, float] | None:
    """Percentile interval over the unit of analysis, so a headline carries its noise.

    A single record supplies few cells, and a direction accuracy of 0.86 over fourteen of
    them is not distinguishable from 0.7. Resampling is within-record, so the interval
    covers sampling of cells, not of papers.
    """

    if len(values) < 3:
        return None
    import random

    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def direction_of(fields: Mapping[str, Field], path: str) -> str:
    node = fields.get(path)
    if node is None:
        return "absent"
    return node.value if node.status == "extracted" and node.value else node.status


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def reachable(record: Record, schema: Schema, roots: Iterable[str]) -> set[str]:
    """Entity ids reachable from `roots` by following references, inline edges included."""

    edges: dict[str, set[str]] = defaultdict(set)
    for entity in record.entities.values():
        edges[entity.local_id] |= {t for _, t in entity.edges}
        for path, (cls, members) in entity.inline.items():
            for member in members:
                _, inline_edges = flatten_inline(member, cls, schema, "")
                edges[entity.local_id] |= {t for _, t in inline_edges}

    seen: set[str] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(edges.get(current, ()))
    return seen


def scope_to_tables(doc: Mapping, schema: Schema) -> dict:
    """Keep only analyses a publication table reported, and what they reach.

    An analysis with no `tables` contributes no coordinates, and a table-anchored pipeline
    cannot enumerate one -- scoring against it measures a scoping decision rather than an
    extraction. Dropping the analysis alone would not be enough: the entities only it
    referenced would stay in gold as recall misses no in-scope analysis could ever demand.
    An entity reachable from a kept analysis survives even if a dropped one also used it,
    and one reachable from no analysis at all is a study-level fact that is never in
    question here.
    """

    scoped = json.loads(json.dumps(doc))
    analyses = scoped.get("analyses") or []
    kept = [a for a in analyses if a.get("tables")]
    dropped = [a for a in analyses if not a.get("tables")]
    if not dropped:
        return scoped

    record = flatten(scoped, schema, "scope")
    keep_ids = reachable(record, schema, [a["local_id"] for a in kept])
    drop_ids = reachable(record, schema, [a["local_id"] for a in dropped]) - keep_ids

    scoped["analyses"] = kept
    for slot, value in scoped.items():
        if isinstance(value, list) and all(isinstance(v, Mapping) for v in value):
            scoped[slot] = [v for v in value if v.get(LOCAL_ID) not in drop_ids]
    return scoped


def compare(gold_doc: Mapping, cand_doc: Mapping, schema: Schema, sem: Semantics,
            label: str, scope: str = "all") -> dict[str, Any]:
    if scope == "tables":
        gold_doc = scope_to_tables(gold_doc, schema)
        cand_doc = scope_to_tables(cand_doc, schema)

    gold = flatten(gold_doc, schema, "gold")
    cand = flatten(cand_doc, schema, "candidate")

    sem.prepare(_all_text(gold) + _all_text(cand))

    aligner = Aligner(gold, cand, schema, sem)
    blind = Aligner(gold, cand, schema, sem, exclude=DIRECTION_LEAKING)

    result: dict[str, Any] = {"record": label}
    result["entities"] = entity_metrics(gold, cand, aligner)
    pairs = [(gold.entities[g], cand.entities[c]) for c, g in aligner.map.items()]
    inline_alignments = {}
    # Order matters: `field_metrics` aligns the inline lists and folds their edges into the
    # owning entity, so a reference held by an AnalysisGroup or a FactorLevel does not exist
    # as an edge until it has run. Scoring relationships first would silently lose them.
    result["fields"] = field_metrics(pairs, schema, sem, inline_alignments)
    result["relationships"] = relationship_metrics(gold, cand, aligner, inline_alignments, schema)
    result["structure"] = structure_metrics(gold, cand, aligner)
    result["direction"] = {
        "primary": direction_metrics(gold, cand, aligner, schema, sem),
        "structure_only": direction_metrics(gold, cand, blind, schema, sem),
    }
    result["composite"] = composite(result)
    return result


def _all_text(record: Record) -> list[str]:
    texts: list[str] = []
    for ent in record.entities.values():
        for f in ent.fields.values():
            if f.status == "extracted" and f.kind.startswith("string"):
                texts.extend(v for v in (f.value if isinstance(f.value, list) else [f.value])
                             if isinstance(v, str))
        for _, members in ent.inline.values():
            for member in members:
                for value in member.values():
                    if isinstance(value, Mapping) and isinstance(value.get("value"), str):
                        texts.append(value["value"])
    return texts


def entity_metrics(gold: Record, cand: Record, aligner: Aligner) -> dict[str, Any]:
    per_type: dict[str, Any] = {}
    tp = fp = fn = 0
    for etype in sorted(set(gold.by_type) | set(cand.by_type)):
        g = gold.by_type.get(etype, [])
        c = cand.by_type.get(etype, [])
        matched = [e for e in c if e.local_id in aligner.map]
        hit = len(matched)
        stats = prf(hit, len(c) - hit, len(g) - hit)
        stats["gold_n"] = len(g)
        stats["cand_n"] = len(c)
        stats["mean_match_score"] = (
            sum(aligner.scores.get(aligner.map[e.local_id], 0.0) for e in matched) / hit
            if hit else 0.0
        )
        stats["missed"] = sorted(e.local_id for e in g
                                 if e.local_id not in aligner.inverse)
        stats["spurious"] = sorted(e.local_id for e in c if e.local_id not in aligner.map)
        per_type[etype] = stats
        tp, fp, fn = tp + hit, fp + len(c) - hit, fn + len(g) - hit
    return {"micro": prf(tp, fp, fn), "per_type": per_type,
            "macro_f1": sum(v["f1"] for v in per_type.values()) / len(per_type) if per_type else 0.0}


def field_metrics(pairs: Sequence[tuple[Entity, Entity]], schema: Schema, sem: Semantics,
                  inline_out: dict) -> dict[str, Any]:
    per_type: dict[str, dict[str, Any]] = {}
    per_entity: list[dict[str, Any]] = []

    for g_ent, c_ent in pairs:
        fields_g, fields_c = dict(g_ent.fields), dict(c_ent.fields)
        edges_g, edges_c = set(g_ent.edges), set(c_ent.edges)
        for path, (cls, members) in g_ent.inline.items():
            cand_cls, cand_members = c_ent.inline.get(path, (cls, []))
            aligned = match(members, cand_members,
                            lambda a, b, k=cls: inline_similarity(a, b, k, schema, sem),
                            CELL_THRESHOLD)
            inline_out[(g_ent.local_id, path)] = (members, cand_members, aligned, cls)
            for i, j, _ in aligned:
                gf, ge = flatten_inline(members[i], cls, schema, f"{path}[{i}].")
                cf, ce = flatten_inline(cand_members[j], cand_cls, schema, f"{path}[{i}].")
                fields_g.update(gf)
                fields_c.update(cf)
                edges_g |= ge
                edges_c |= ce
            for i, member in enumerate(members):
                if not any(i == a for a, _, _ in aligned):
                    gf, ge = flatten_inline(member, cls, schema, f"{path}[{i}].")
                    fields_g.update(gf)
                    edges_g |= ge
        g_ent.edges, c_ent.edges = edges_g, edges_c

        bucket = per_type.setdefault(g_ent.etype, _empty_field_bucket())
        row = _empty_field_bucket()
        for path in sorted(set(fields_g) | set(fields_c)):
            gf, cf = fields_g.get(path), fields_c.get(path)
            g_has = gf is not None and gf.status == "extracted"
            c_has = cf is not None and cf.status == "extracted"
            for target in (bucket, row):
                target["total"] += 1
            if g_has and c_has:
                verdict = compare_values(gf.kind, gf.value, cf.value, sem)
                for target in (bucket, row):
                    target["both"] += 1
                    target["score"] += verdict.score
                    target["correct"] += int(verdict.match)
                    if verdict.numeric_error is not None:
                        target["numeric"].append((as_number(gf.value), as_number(cf.value)))
                if not verdict.match:
                    row["wrong"].append({"path": path, "gold": gf.value, "cand": cf.value,
                                         "score": round(verdict.score, 3)})
            elif g_has and not c_has:
                for target in (bucket, row):
                    target["missed"] += 1
                row["missing"].append(path)
            elif c_has and not g_has:
                for target in (bucket, row):
                    target["spurious"] += 1
                row["hallucinated"].append(path)
            else:
                for target in (bucket, row):
                    target["absent_agree"] += 1

        per_entity.append({
            "type": g_ent.etype,
            "gold_id": g_ent.local_id,
            "cand_id": c_ent.local_id,
            **_summarize_field_bucket(row),
            "wrong": row["wrong"][:20],
            "not_extracted": row["missing"][:20],
            "over_extracted": row["hallucinated"][:20],
        })

    return {
        "per_type": {k: _summarize_field_bucket(v) for k, v in sorted(per_type.items())},
        "per_entity": per_entity,
        "overall": _summarize_field_bucket(_merge_buckets(per_type.values())),
    }


def _empty_field_bucket() -> dict[str, Any]:
    return {"total": 0, "both": 0, "score": 0.0, "correct": 0, "missed": 0, "spurious": 0,
            "absent_agree": 0, "numeric": [], "wrong": [], "missing": [], "hallucinated": []}


def _merge_buckets(buckets: Iterable[dict[str, Any]]) -> dict[str, Any]:
    out = _empty_field_bucket()
    for b in buckets:
        for key, value in b.items():
            if isinstance(value, list):
                out[key].extend(value)
            else:
                out[key] += value
    return out


def _summarize_field_bucket(b: dict[str, Any]) -> dict[str, Any]:
    both = b["both"]
    numeric = [(g, c) for g, c in b["numeric"] if g is not None and c is not None]
    errors = [abs(g - c) for g, c in numeric]
    relative = [abs(g - c) / max(abs(g), abs(c), 1e-12) for g, c in numeric]
    summary = {
        "fields_compared": b["total"],
        "both_extracted": both,
        "value_accuracy": b["correct"] / both if both else float("nan"),
        "value_score": b["score"] / both if both else float("nan"),
        # Presence is its own extraction decision: `not_reported` is an assertion that the
        # paper is silent, and getting it wrong is a different defect from a wrong value.
        "presence": prf(both, b["spurious"], b["missed"]),
        "agree_not_reported": b["absent_agree"],
    }
    if numeric:
        summary["numeric"] = {
            "n": len(numeric),
            "mae": sum(errors) / len(errors),
            "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "mape": sum(relative) / len(relative),
            "bias": sum(c - g for g, c in numeric) / len(numeric),
            "within_tolerance": sum(1 for r in relative if r <= NUMERIC_RTOL) / len(relative),
        }
    return summary


def inline_similarity(a: Mapping, b: Mapping, class_name: str, schema: Schema,
                      sem: Semantics) -> float:
    blocked = ALIGN_EXCLUDE.get(class_name, set())
    fa, ea = flatten_inline(a, class_name, schema, "")
    fb, eb = flatten_inline(b, class_name, schema, "")
    total = weighted = 0.0
    for path in set(fa) | set(fb):
        if path.split(".")[0] in blocked:
            continue
        x, y = fa.get(path), fb.get(path)
        weight = 3.0 if path.split(".")[-1] in IDENTIFYING else 1.0
        weighted += weight
        if x is None or y is None:
            # Half credit, as `Aligner.field_score` gives it. Scoring a one-sided field as
            # a total mismatch would stop a Cell whose term is right but whose `level` the
            # candidate omitted from aligning at all -- turning a reportable field error
            # into a silent non-match, which is the one outcome the direction metrics must
            # not produce.
            total += weight * 0.5
            continue
        if x.status != "extracted" or y.status != "extracted":
            total += weight * (1.0 if x.status == y.status else 0.0)
        else:
            total += weight * compare_values(x.kind, x.value, y.value, sem).score
    fields = total / weighted if weighted else 0.5
    if not ea and not eb:
        return fields
    shared = len({p for p, _ in ea} & {p for p, _ in eb})
    return 0.7 * fields + 0.3 * (shared / max(len(ea), len(eb), 1))


def structure_metrics(gold: Record, cand: Record, aligner: Aligner) -> dict[str, Any]:
    """Per entity, does its neighbourhood look like its counterpart's?

    Relationship F1 is one number over the whole graph, which says a structure is wrong
    without saying where. This localises it: for each matched entity, which of its
    neighbours are shared, missing and extra, in both directions. An entity whose
    attributes match perfectly but whose neighbourhood does not is the signature of a
    plausible object wired into the wrong place -- the failure a flat edge count averages
    away.
    """

    rows: list[dict[str, Any]] = []
    for cand_id, gold_id in aligner.map.items():
        g, c = gold.entities[gold_id], cand.entities[cand_id]
        out_gold = g.ref_edges
        out_cand = {(p, aligner.map.get(t, f"?{t}")) for p, t in c.ref_edges}
        in_gold = gold.incoming.get(gold_id, set())
        in_cand = {(aligner.map.get(s, f"?{s}"), p)
                   for s, p in cand.incoming.get(cand_id, set())}
        neighbours = prf(len(out_gold & out_cand) + len(in_gold & in_cand),
                         len(out_cand - out_gold) + len(in_cand - in_gold),
                         len(out_gold - out_cand) + len(in_gold - in_cand))
        rows.append({
            "type": g.etype,
            "gold_id": gold_id,
            "cand_id": cand_id,
            "match_score": round(aligner.scores.get(gold_id, 0.0), 3),
            "evidence": aligner.parts.get(gold_id, {}),
            "neighbourhood": neighbours,
            "missing": sorted(f"{p} -> {t}" for p, t in out_gold - out_cand)
            + sorted(f"{s} -{p}-> here" for s, p in in_gold - in_cand),
            "extra": sorted(f"{p} -> {t}" for p, t in out_cand - out_gold)
            + sorted(f"{s} -{p}-> here" for s, p in in_cand - in_gold),
        })

    # An entity that neither refers to anything nor is referred to has no neighbourhood to
    # agree about -- a Timepoint no term's level reaches, the Study root itself. Scoring it
    # zero would report a structure defect where there is no structure to get wrong.
    scored = [r for r in rows if r["neighbourhood"]["tp"] + r["neighbourhood"]["fp"]
              + r["neighbourhood"]["fn"]]
    unconnected = len(rows) - len(scored)
    return {
        "per_entity": sorted(scored, key=lambda r: r["neighbourhood"]["f1"]),
        "unconnected": unconnected,
        "mean_neighbourhood_f1": (
            sum(r["neighbourhood"]["f1"] for r in scored) / len(scored)
            if scored else float("nan")),
        # An entity the attributes like and the graph does not: right object, wrong place.
        "misplaced": [r for r in scored
                      if r["evidence"].get("attributes", 0) >= 0.75
                      and r["neighbourhood"]["f1"] < 0.5],
    }


def relationship_metrics(gold: Record, cand: Record, aligner: Aligner,
                         inline_alignments: dict, schema: Schema) -> dict[str, Any]:
    """Triples over the entity map: an edge touching an unmatched entity can only be wrong.

    Candidate edges are translated into gold identifiers first. An endpoint that never
    matched anything keeps a `?` marker instead and so can never coincide with a gold edge,
    which is the right verdict: it names an entity the gold record does not have. Both
    endpoints are treated alike -- an edge out of a hallucinated entity is as much a false
    positive as an edge into one -- so that relationship recall and precision describe the
    same graph difference from either side.
    """

    gold_edges: set[tuple[str, str, str]] = set()
    cand_edges: set[tuple[str, str, str]] = set()
    for ent in gold.entities.values():
        for path, target in ent.edges:
            gold_edges.add((ent.local_id, path, target))
    for ent in cand.entities.values():
        source = aligner.map.get(ent.local_id, f"?{ent.local_id}")
        for path, target in ent.edges:
            cand_edges.add((source, path, aligner.map.get(target, f"?{target}")))

    tp = gold_edges & cand_edges
    per_slot: dict[str, Any] = {}
    for slot in sorted({re.sub(r"\[\d+\]", "[]", p) for _, p, _ in gold_edges | cand_edges}):
        g = {e for e in gold_edges if re.sub(r"\[\d+\]", "[]", e[1]) == slot}
        c = {e for e in cand_edges if re.sub(r"\[\d+\]", "[]", e[1]) == slot}
        per_slot[slot] = prf(len(g & c), len(c - g), len(g - c))

    return {
        "micro": prf(len(tp), len(cand_edges - gold_edges), len(gold_edges - cand_edges)),
        "per_slot": per_slot,
        "false_negatives": sorted(gold_edges - cand_edges),
        "false_positives": sorted(cand_edges - gold_edges),
        "unmatched_endpoint_edges": sum(1 for s, _, t in cand_edges
                                        if s.startswith("?") or t.startswith("?")),
    }


def direction_metrics(gold: Record, cand: Record, aligner: Aligner, schema: Schema,
                      sem: Semantics) -> dict[str, Any]:
    """The headline: does the candidate say which way each contrast went, on the right term?

    A cell pair is only scored when its `term` reference resolves to the same ModelTerm on
    both sides. The right sign attached to the wrong term is not a right answer -- it names
    a different comparison -- so those pairs are counted as term-ungrounded and excluded
    from the accuracy rather than credited.
    """

    labels: list[tuple[str, str]] = []
    grounded_hits: list[float] = []
    aligned_labels: list[tuple[str, str]] = []
    contrast: Counter[str] = Counter()
    detail: list[dict[str, Any]] = []
    cells_gold = cells_cand = cells_aligned = cells_grounded = 0

    gold_analyses = gold.by_type.get("Analysis", [])
    cand_by_id = {e.local_id: e for e in cand.by_type.get("Analysis", [])}

    for g_ent in gold_analyses:
        g_cells = g_ent.inline.get("effect.cells", ("Cell", []))[1]
        cells_gold += len(g_cells)
        c_id = aligner.inverse.get(g_ent.local_id)
        if c_id is None:
            contrast["analysis_missed"] += 1
            detail.append({"gold_analysis": g_ent.local_id, "verdict": "analysis_missed"})
            continue
        c_ent = cand_by_id[c_id]
        c_cells = c_ent.inline.get("effect.cells", ("Cell", []))[1]
        cells_cand += len(c_cells)
        aligned = match(g_cells, c_cells,
                        lambda a, b: inline_similarity(a, b, "Cell", schema, sem),
                        CELL_THRESHOLD)
        cells_aligned += len(aligned)

        pairs_here: list[tuple[str, str]] = []
        grounded_here = 0
        for i, j, _ in aligned:
            g_term = _cell_term(g_cells[i])
            c_term = _cell_term(c_cells[j])
            g_dir = _cell_direction(g_cells[i])
            c_dir = _cell_direction(c_cells[j])
            same_term = bool(c_term) and aligner.map.get(c_term) == g_term
            aligned_labels.append((g_dir, c_dir))
            if same_term:
                cells_grounded += 1
                grounded_here += 1
                labels.append((g_dir, c_dir))
                grounded_hits.append(1.0 if g_dir == c_dir else 0.0)
                pairs_here.append((g_dir, c_dir))

        verdict = _contrast_verdict(g_cells, c_cells, aligned, grounded_here, pairs_here)
        contrast[verdict] += 1
        detail.append({
            "gold_analysis": g_ent.local_id,
            "cand_analysis": c_id,
            "verdict": verdict,
            "cells": [
                {"gold": _cell_repr(g_cells[i]), "cand": _cell_repr(c_cells[j]),
                 "term_grounded": (_cell_term(c_cells[j]) is not None
                                   and aligner.map.get(_cell_term(c_cells[j]))
                                   == _cell_term(g_cells[i])),
                 "direction_match": _cell_direction(g_cells[i]) == _cell_direction(c_cells[j])}
                for i, j, _ in aligned
            ],
        })

    # Accuracy alone is gameable: an extractor that emits one easy cell per analysis and
    # drops the rest scores 100%. Recall is over every gold cell, so a dropped cell is a
    # miss; precision is over every candidate cell, so an invented one is a false positive.
    correct = sum(grounded_hits)
    direction_prf = prf(int(correct), cells_cand - int(correct), cells_gold - int(correct))

    signed = [(g, c) for g, c in labels if g in DIRECTIONAL]
    flips = sum(1 for g, c in signed if c == FLIP[g])
    lost = sum(1 for g, c in signed if c not in DIRECTIONAL)
    invented = sum(1 for g, c in labels if g not in DIRECTIONAL and c in DIRECTIONAL)
    unsigned_gold = len(labels) - len(signed)
    interval = bootstrap(grounded_hits)

    scored_contrasts = sum(contrast[k] for k in contrast)
    return {
        "cells": {"gold": cells_gold, "candidate": cells_cand, "aligned": cells_aligned,
                  "term_grounded": cells_grounded,
                  "grounding_rate": cells_grounded / cells_aligned if cells_aligned else float("nan"),
                  "cell_recall": cells_grounded / cells_gold if cells_gold else float("nan")},
        "accuracy_term_grounded": sum(grounded_hits) / len(grounded_hits) if grounded_hits else float("nan"),
        "accuracy_ci95": interval,
        "cell_prf": direction_prf,
        "accuracy_aligned_any_term": (
            sum(1 for g, c in aligned_labels if g == c) / len(aligned_labels)
            if aligned_labels else float("nan")
        ),
        "signed_accuracy": (len(signed) - flips - lost) / len(signed) if signed else float("nan"),
        "sign_flip_rate": flips / len(signed) if signed else float("nan"),
        "sign_loss_rate": lost / len(signed) if signed else float("nan"),
        "sign_invention_rate": invented / unsigned_gold if unsigned_gold else float("nan"),
        "kappa": cohen_kappa(labels),
        **per_class_prf(labels),
        "confusion": _confusion(labels),
        "contrast": {"counts": dict(contrast),
                     "exact_rate": contrast["exact"] / scored_contrasts if scored_contrasts else float("nan"),
                     "reversed_rate": contrast["reversed"] / scored_contrasts if scored_contrasts else float("nan")},
        "detail": detail,
    }


def _cell_term(cell: Mapping) -> str | None:
    """A cell's term as a local_id, whatever shape the extractor put it in.

    `Cell.term` is a bare identifier -- a reference has no `not_reported` form -- but a
    model that wraps it anyway produces `{"value": "term_x"}`, and an unguarded lookup on
    that raises rather than scoring the run. A malformed reference is a defect the record
    metrics should report, not one the scorer should die on.
    """

    term = cell.get("term")
    if isinstance(term, str):
        return term
    if isinstance(term, Mapping) and isinstance(term.get("value"), str):
        return term["value"]
    return None


def _cell_direction(cell: Mapping) -> str:
    node = cell.get("direction")
    if not isinstance(node, Mapping):
        return "absent"
    if node.get("extraction_status") == "extracted" and node.get("value"):
        return str(node["value"])
    return str(node.get("extraction_status", "absent"))


def _cell_repr(cell: Mapping) -> dict[str, Any]:
    level = cell.get("level")
    return {"term": _cell_term(cell),
            "level": level.get("value") if isinstance(level, Mapping) else None,
            "direction": _cell_direction(cell)}


def _contrast_verdict(g_cells: Sequence[Mapping], c_cells: Sequence[Mapping],
                      aligned: Sequence[tuple[int, int, float]], grounded: int,
                      pairs: Sequence[tuple[str, str]]) -> str:
    """One label per analysis, because a contrast is only right as a whole.

    `reversed` is split out from `wrong` on purpose. Every sign flipped is one diagnosable
    mistake -- the extractor read the comparison backwards -- and it behaves differently
    downstream from a contrast that is merely partly wrong.
    """

    if grounded != len(g_cells) or len(c_cells) != len(g_cells) or len(aligned) != len(g_cells):
        return "structure_mismatch"
    if all(g == c for g, c in pairs):
        return "exact"
    signed = [(g, c) for g, c in pairs if g in DIRECTIONAL]
    if signed and all(c == FLIP[g] for g, c in signed) and \
            all(g == c for g, c in pairs if g not in DIRECTIONAL):
        return "reversed"
    return "wrong_direction"


def _confusion(pairs: Sequence[tuple[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g, c in pairs:
        out[g][c] += 1
    return {g: dict(c) for g, c in out.items()}


def composite(result: Mapping[str, Any]) -> dict[str, Any]:
    """One number, weighted so direction dominates. Reported with its parts, never alone."""

    parts = {
        "direction": result["direction"]["primary"]["cell_prf"]["f1"],
        "entities": result["entities"]["micro"]["f1"],
        "relationships": result["relationships"]["micro"]["f1"],
        "fields": result["fields"]["overall"]["value_accuracy"],
    }
    usable = {k: v for k, v in parts.items() if isinstance(v, float) and not math.isnan(v)}
    total = sum(COMPOSITE_WEIGHTS[k] for k in usable)
    score = sum(COMPOSITE_WEIGHTS[k] * v for k, v in usable.items()) / total if total else float("nan")
    return {"score": score, "parts": parts, "weights": COMPOSITE_WEIGHTS}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def pct(value: Any) -> str:
    if not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
        return "  --  "
    return f"{value * 100:5.1f}%"


def render(result: Mapping[str, Any], verbose: bool) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"=== {result['record']} ===")

    comp = result["composite"]
    add("")
    add(f"composite {pct(comp['score'])}   " + "  ".join(
        f"{k} {pct(v)}(w={COMPOSITE_WEIGHTS[k]})" for k, v in comp["parts"].items()))

    primary = result["direction"]["primary"]
    blind = result["direction"]["structure_only"]
    add("")
    add("-- DIRECTION (highest weight) ---------------------------------------")
    cells = primary["cells"]
    add(f"cells   gold {cells['gold']}  candidate {cells['candidate']}  "
        f"aligned {cells['aligned']}  term-grounded {cells['term_grounded']} "
        f"({pct(cells['grounding_rate'])} of aligned, {pct(cells['cell_recall'])} of gold)")
    cell_prf = primary["cell_prf"]
    add(f"signed-cell recovery  P {pct(cell_prf['precision'])}  R {pct(cell_prf['recall'])}  "
        f"F1 {pct(cell_prf['f1'])}   (a correct direction on the correct term, "
        f"over every candidate / every gold cell)")
    ci = primary["accuracy_ci95"]
    band = f"  [95% CI {pct(ci[0])} {pct(ci[1])}]" if ci else ""
    add(f"direction accuracy (term-grounded) {pct(primary['accuracy_term_grounded'])}{band}")
    add(f"  alignment blind to analysis prose  {pct(blind['accuracy_term_grounded'])}"
        "   (a bracket, not a correction -- see docs/extraction-comparison-metrics.md)")
    add(f"  ignoring term identity             {pct(primary['accuracy_aligned_any_term'])}"
        "   (upper bound: credits a right sign on a wrong term)")
    add(f"signed cells: accuracy {pct(primary['signed_accuracy'])}   "
        f"sign flips {pct(primary['sign_flip_rate'])}   "
        f"sign lost {pct(primary['sign_loss_rate'])}   "
        f"sign invented {pct(primary['sign_invention_rate'])}")
    kappa = primary["kappa"]
    add(f"macro-F1 {pct(primary['macro_f1'])}   Cohen kappa "
        f"{'  --  ' if math.isnan(kappa) else f'{kappa:6.3f}'}")
    if primary["per_class"]:
        add("  per direction     P       R      F1    n")
        for label, stats in sorted(primary["per_class"].items()):
            add(f"    {label:<14} {pct(stats['precision'])} {pct(stats['recall'])} "
                f"{pct(stats['f1'])}  {stats['tp'] + stats['fn']:3d}")
    counts = primary["contrast"]["counts"]
    add("  whole contrasts: " + ("  ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"))
    for row in primary["detail"]:
        if row["verdict"] != "exact":
            add(f"    ! {row['gold_analysis']} -> {row.get('cand_analysis', '(none)')}: "
                f"{row['verdict']}")
            for cell in row.get("cells", []):
                if not cell["direction_match"] or not cell["term_grounded"]:
                    add(f"        gold {cell['gold']}")
                    add(f"        cand {cell['cand']}"
                        f"{'' if cell['term_grounded'] else '   [different term]'}")

    ents = result["entities"]
    add("")
    add("-- OBJECTS ----------------------------------------------------------")
    add(f"micro  P {pct(ents['micro']['precision'])}  R {pct(ents['micro']['recall'])}  "
        f"F1 {pct(ents['micro']['f1'])}   macro-F1 {pct(ents['macro_f1'])}")
    add(f"  {'type':<20} {'gold':>4} {'cand':>4} {'P':>6} {'R':>6} {'F1':>6}  match")
    for etype, stats in sorted(ents["per_type"].items()):
        add(f"  {etype:<20} {stats['gold_n']:>4} {stats['cand_n']:>4} "
            f"{pct(stats['precision'])} {pct(stats['recall'])} {pct(stats['f1'])}  "
            f"{pct(stats['mean_match_score'])}")
        if stats["missed"]:
            add(f"      missed:   {', '.join(stats['missed'])}")
        if stats["spurious"]:
            add(f"      spurious: {', '.join(stats['spurious'])}")

    rel = result["relationships"]
    add("")
    add("-- RELATIONSHIPS ----------------------------------------------------")
    add(f"micro  P {pct(rel['micro']['precision'])}  R {pct(rel['micro']['recall'])}  "
        f"F1 {pct(rel['micro']['f1'])}   "
        f"FP {rel['micro']['fp']}  FN {rel['micro']['fn']}  "
        f"({rel['unmatched_endpoint_edges']} candidate edges touch an unmatched entity)")
    for slot, stats in sorted(rel["per_slot"].items()):
        add(f"  {slot:<34} P {pct(stats['precision'])} R {pct(stats['recall'])} "
            f"F1 {pct(stats['f1'])}  fp {stats['fp']:2d} fn {stats['fn']:2d}")
    if verbose:
        for kind in ("false_negatives", "false_positives"):
            for edge in rel[kind][:40]:
                add(f"  {'FN' if kind == 'false_negatives' else 'FP'} "
                    f"{edge[0]} --{edge[1]}--> {edge[2]}")

    structure = result["structure"]
    add("")
    add("-- STRUCTURE (is each matched object wired where its counterpart is?) --")
    add(f"mean neighbourhood F1 over {len(structure['per_entity'])} connected matched "
        f"entities {pct(structure['mean_neighbourhood_f1'])}"
        + (f"  ({structure['unconnected']} matched entities have no neighbourhood)"
           if structure["unconnected"] else ""))
    if structure["misplaced"]:
        add("  right object, wrong place -- attributes agree, neighbourhood does not:")
        for row in structure["misplaced"]:
            add(f"    {row['type']}/{row['gold_id']} <- {row['cand_id']}  "
                f"attrs {pct(row['evidence'].get('attributes'))} "
                f"neighbourhood F1 {pct(row['neighbourhood']['f1'])}")
    worst = [r for r in structure["per_entity"] if r["neighbourhood"]["f1"] < 1.0]
    for row in worst[: 40 if verbose else 6]:
        add(f"  {row['type']}/{row['gold_id']} <- {row['cand_id']}: "
            f"F1 {pct(row['neighbourhood']['f1'])}  "
            f"match {pct(row['match_score'])} from "
            + ", ".join(f"{k} {v}" for k, v in row["evidence"].items()))
        for link in row["missing"][: 6 if verbose else 3]:
            add(f"      missing  {link}")
        for link in row["extra"][: 6 if verbose else 3]:
            add(f"      extra    {link}")
    if len(worst) > (40 if verbose else 6):
        add(f"  ... {len(worst) - (40 if verbose else 6)} more with imperfect neighbourhoods")

    fields = result["fields"]
    add("")
    add("-- FIELDS -----------------------------------------------------------")
    overall = fields["overall"]
    add(f"overall  value accuracy {pct(overall['value_accuracy'])}  "
        f"graded {pct(overall['value_score'])}  "
        f"presence F1 {pct(overall['presence']['f1'])}  "
        f"(compared {overall['fields_compared']}, both extracted {overall['both_extracted']})")
    if "numeric" in overall:
        num = overall["numeric"]
        add(f"numeric  n {num['n']}  within {NUMERIC_RTOL:.0%} {pct(num['within_tolerance'])}  "
            f"MAE {num['mae']:.4g}  RMSE {num['rmse']:.4g}  MAPE {num['mape']:.3f}  "
            f"bias {num['bias']:+.4g}")
    add(f"  {'type':<20} {'acc':>6} {'graded':>7} {'presP':>6} {'presR':>6} {'both':>5} "
        f"{'miss':>5} {'extra':>5}")
    for etype, stats in fields["per_type"].items():
        add(f"  {etype:<20} {pct(stats['value_accuracy'])} {pct(stats['value_score'])} "
            f"{pct(stats['presence']['precision'])} {pct(stats['presence']['recall'])} "
            f"{stats['both_extracted']:>5} {stats['presence']['fn']:>5} "
            f"{stats['presence']['fp']:>5}")
    if verbose:
        add("")
        for row in sorted(fields["per_entity"], key=lambda r: r["value_accuracy"]):
            add(f"  {row['type']}/{row['gold_id']} <- {row['cand_id']}: "
                f"acc {pct(row['value_accuracy'])} graded {pct(row['value_score'])}")
            for wrong in row["wrong"]:
                add(f"      {wrong['path']}: gold={_short(wrong['gold'])} "
                    f"cand={_short(wrong['cand'])} ({wrong['score']})")
            if row["not_extracted"]:
                add(f"      not extracted: {', '.join(row['not_extracted'])}")
            if row["over_extracted"]:
                add(f"      over extracted: {', '.join(row['over_extracted'])}")
    return "\n".join(lines)


def _short(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# aggregation across records
# ---------------------------------------------------------------------------

def aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro over records and micro over units, because they answer different questions.

    Macro says how the extractor does on a typical paper; micro says how much of the
    corpus's evidence it got right, and a paper with forty analyses moves only the second.
    """

    def macro(getter) -> float:
        values = [getter(r) for r in results]
        values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        return sum(values) / len(values) if values else float("nan")

    def pooled(section: str) -> dict[str, float]:
        tp = sum(r[section]["micro"]["tp"] for r in results)
        fp = sum(r[section]["micro"]["fp"] for r in results)
        fn = sum(r[section]["micro"]["fn"] for r in results)
        return prf(tp, fp, fn)

    labels: list[tuple[str, str]] = []
    hits: list[float] = []
    for r in results:
        for row in r["direction"]["primary"]["detail"]:
            for cell in row.get("cells", []):
                if cell["term_grounded"]:
                    labels.append((cell["gold"]["direction"], cell["cand"]["direction"]))
                    hits.append(float(cell["direction_match"]))
    cells = [r["direction"]["primary"]["cell_prf"] for r in results]

    return {
        "records": len(results),
        "entities_micro": pooled("entities"),
        "relationships_micro": pooled("relationships"),
        "direction_micro_prf": prf(sum(c["tp"] for c in cells), sum(c["fp"] for c in cells),
                                   sum(c["fn"] for c in cells)),
        "direction_micro_accuracy": sum(hits) / len(hits) if hits else float("nan"),
        "direction_micro_ci95": bootstrap(hits),
        "direction_micro_kappa": cohen_kappa(labels),
        "direction_macro_accuracy": macro(
            lambda r: r["direction"]["primary"]["accuracy_term_grounded"]),
        "field_macro_accuracy": macro(lambda r: r["fields"]["overall"]["value_accuracy"]),
        "composite_macro": macro(lambda r: r["composite"]["score"]),
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate", nargs="+", type=Path,
                        help="candidate extraction JSON, one or more")
    parser.add_argument("--gold", type=Path, help="gold record for a single candidate")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "data" / "gold",
                        help="directory holding <local_id>.extraction.json gold records")
    parser.add_argument("--semantic", action="store_true",
                        help="add embedding cosine to the string similarity (needs OPENAI_API_KEY)")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-base-url",
                        help="send /embeddings here instead of OPENAI_API_GATEWAY")
    parser.add_argument("--json", type=Path, help="write the full metric tree here")
    parser.add_argument("--scope", default="all", choices=["all", "tables"],
                        help="'tables' keeps only analyses a publication table reported, "
                             "on both sides, plus what they reach")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list every wrong field and every mismatched relationship")
    args = parser.parse_args(argv)

    schema = Schema()
    sem = Semantics(args.semantic, args.embedding_model, args.embedding_base_url)
    results = []

    for path in args.candidate:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if args.gold and len(args.candidate) == 1:
            gold_path = args.gold
        else:
            gold_path = args.gold_dir / f"{candidate.get('local_id', path.stem)}.extraction.json"
        if not gold_path.is_file():
            print(f"no gold record for {path.name} (looked for {gold_path})", file=sys.stderr)
            continue
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        result = compare(gold, candidate, schema, sem, label=str(path), scope=args.scope)
        results.append(result)
        print(render(result, args.verbose))
        print()

    if not results:
        return 1

    if len(results) > 1:
        summary = aggregate(results)
        print("=== across %d records ===" % summary["records"])
        print(f"entities      micro-F1 {pct(summary['entities_micro']['f1'])}")
        print(f"relationships micro-F1 {pct(summary['relationships_micro']['f1'])}")
        ci = summary["direction_micro_ci95"]
        band = f"  [95% CI {pct(ci[0])} {pct(ci[1])}]" if ci else ""
        print(f"direction     micro-F1 {pct(summary['direction_micro_prf']['f1'])}  "
              f"accuracy micro {pct(summary['direction_micro_accuracy'])}{band}  "
              f"macro {pct(summary['direction_macro_accuracy'])}  "
              f"kappa {summary['direction_micro_kappa']:.3f}")
        print(f"fields        macro accuracy {pct(summary['field_macro_accuracy'])}")
        print(f"composite     macro {pct(summary['composite_macro'])}")
    else:
        summary = None

    if args.json:
        args.json.write_text(
            json.dumps({"results": results, "summary": summary}, indent=2, default=str),
            encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
