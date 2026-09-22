"""`Group.medical_condition` -> MONDO, and through MONDO to UMLS, SNOMED and ONVOC.

Triage first (`vocabularies.phrases`), then link each head, then derive the ONVOC term from
the MONDO node rather than matching twice. A row carries both grains and the relation
between them: MONDO is the grain the literature writes in, ONVOC the grain a query filters
on, and which one a meta-analysis needs is not this layer's decision.

Rationale, measurements, and why MONDO rather than SNOMED: docs/condition-normalization.md.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pondie.normalization._negation import available as parser_available
from pondie.normalization._records import DEFAULT, iter_records, strings_at, value_of
from pondie.vocabularies.abbreviations import Abbreviations, expansions_in
from pondie.vocabularies.folding import variants
from pondie.vocabularies.mondo import Vocabulary, load_mondo, onvoc_crosswalk
from pondie.vocabularies.onvoc import load_onvoc
from pondie.vocabularies.phrases import (  # noqa: F401 -- callers name these from here
    HEALTHY,
    NO_CONDITION,
    NOT_READ,
    QUALIFIER,
    SPLIT,
    TRAILING,
    Triaged as Mapped,
    absent,
    triage,
)

#: Taken above `ACCEPT`, rejected below `REVIEW`, queued for review between them. Two cuts
#: because one cannot separate the distributions. 0.95 gives 93.5% precision on the sweep
#: `calibrate` prints, and the cut sits there rather than higher because on real records the
#: 0.96-0.97 band is en-GB spellings and inserted prepositions -- correct matches, not near
#: misses. See docs/condition-normalization.md.
ACCEPT = 0.95
REVIEW = 0.85


@dataclass(frozen=True)
class Link:
    """One head of one `medical_condition` value of one group, and where it landed.

    A row per (study, group, head); aggregation is a view over these.
    """

    study: str
    group: str
    text: str
    head: str = ""
    sentinel: str = ""
    denied: tuple[str, ...] = ()
    scope: str = ""
    qualifiers: tuple[str, ...] = ()
    expansions: tuple[str, ...] = ()
    #: The matched node at the FINEST grain that matched, never the rolled-up one. The
    #: rollup is a view.
    mondo: str = ""
    label: str = ""
    umls: str = ""
    sctid: str = ""
    method: str = ""
    score: float = 0.0
    #: Nearest-first ancestors as `label (CURIE)`: the path the ONVOC bridge walked.
    ancestors: tuple[str, ...] = ()
    onvoc: str = ""
    onvoc_id: str = ""
    #: How ONVOC was reached: `crosswalk`, `ancestor`, `name`, `ancestor-name`, or "" --
    #: and "" is what makes the row a proposal.
    onvoc_via: str = ""
    candidates: tuple[str, ...] = ()

    @property
    def matched(self) -> bool:
        return bool(self.mondo)

    @property
    def proposes(self) -> bool:
        """Matched in MONDO and unreachable in ONVOC: a term ONVOC could carry."""
        return bool(self.mondo) and not self.onvoc


def _lexical(head: str, vocab: Vocabulary) -> int | None:
    """The node an exact or orthographic-variant lookup reaches, most faithful first."""
    for form in variants(head):
        node = vocab.surface.get(form)
        if node is not None:
            return node
    return None


@dataclass
class Index:
    """MONDO's surface forms, encoded once, with the node each form belongs to."""

    vocab: Vocabulary
    matrix: object = None
    node_of: tuple[int, ...] = ()

    def build(self) -> Index:
        """Encode every surface form. Cached on disk by model and content."""
        from pondie.normalization._embedding import for_phrases

        self.matrix = for_phrases(self.vocab.forms)
        self.node_of = tuple(self.vocab.form_node)
        return self

    def nearest(self, queries: list[str], top: int = 5, chunk: int = 256):
        """(node, score, alternative nodes) for each query, over every SURFACE FORM.

        Scored against all 102,615 forms rather than the 32,109 labels, and resolved to
        the node behind the best one. Chunked because the similarity matrix is
        queries x 102,615: a whole corpus at once is gigabytes for no gain.
        """
        import numpy as np

        from pondie.normalization._embedding import for_phrases

        encoded = for_phrases(queries, cache=False)
        out = []
        for start in range(0, len(queries), chunk):
            sim = encoded[start : start + chunk] @ self.matrix.T
            # `argpartition` over the candidate window, then sort only that window: a full
            # argsort of 102,615 columns per query is the expensive part and nothing reads
            # past the first few.
            window = min(top * 8, sim.shape[1] - 1)
            front = np.argpartition(-sim, window, axis=1)[:, : window + 1]
            for row, candidates in zip(sim, front):
                order = candidates[np.argsort(-row[candidates])]
                seen: list[tuple[int, float]] = []
                taken: set[int] = set()
                for form_index in order:
                    node = self.node_of[int(form_index)]
                    if node not in taken:
                        taken.add(node)
                        seen.append((node, float(row[form_index])))
                    if len(seen) >= top:
                        break
                out.append((seen[0][0], seen[0][1], [n for n, _ in seen[1:]]))
        return out


#: Ladder layers a MONDO label may reach an ONVOC term by name on -- the tight end of
#: `onvoc.METHODS`, since the input is an ontology label rather than a paper's prose.
BY_NAME = frozenset({"exact", "synonym", "stem"})


def bridge(
    node: int,
    vocab: Vocabulary,
    crosswalk: dict[str, tuple[str, str]],
    onvoc: object = None,
) -> tuple[str, str, str, tuple[str, ...]]:
    """(ONVOC id, ONVOC label, how it was reached, the ancestors walked to get there).

    Four layers: a direct `crosswalk` tie, then the nearest crosswalked `ancestor`, then
    ONVOC's own label by `name`, then an ancestor's label. Empty means ONVOC cannot name
    this at any grain, which makes the row a proposal rather than a failure.
    """
    ancestors = vocab.ancestors(node)
    path = tuple(f"{vocab.labels[a]} ({vocab.curie(a)})" for a in ancestors[:12])
    direct = crosswalk.get(vocab.curie(node))
    if direct:
        return direct[0], direct[1], "crosswalk", path
    for ancestor in ancestors:
        tie = crosswalk.get(vocab.curie(ancestor))
        if tie:
            return tie[0], tie[1], "ancestor", path
    if onvoc is not None:
        concept, method, _others = onvoc.match(vocab.labels[node])
        if concept is not None and method in BY_NAME:
            return concept.id, concept.label, "name", path
        for ancestor in ancestors[:6]:
            concept, method, _others = onvoc.match(vocab.labels[ancestor])
            if concept is not None and method in BY_NAME:
                return concept.id, concept.label, "ancestor-name", path
    return "", "", "", path


def link(
    heads: list[str],
    vocab: Vocabulary,
    index: Index | None = None,
    accept: float = ACCEPT,
    review: float = REVIEW,
) -> dict[str, tuple[int | None, str, float, list[int]]]:
    """head -> (node, method, score, alternatives). Lexical first, then retrieval."""
    out: dict[str, tuple[int | None, str, float, list[int]]] = {}
    rest = []
    for head in heads:
        node = _lexical(head, vocab)
        if node is not None:
            out[head] = (node, "lexical", 1.0, [])
        else:
            rest.append(head)
    if not rest:
        return out
    index = index or Index(vocab).build()
    for head, (node, score, others) in zip(rest, index.nearest(rest)):
        if score >= accept:
            out[head] = (node, "retrieval", score, others)
        elif score >= review:
            out[head] = (None, "review", score, [node, *others])
        else:
            out[head] = (None, "rejected", score, [node])
    return out


@functools.lru_cache(maxsize=1)
def _store() -> Abbreviations:
    """The corpus store, read once. 1.4 MB of JSON, and the caller is a per-study loop."""
    return Abbreviations.load()


def _abbreviations(study: str, texts: Path | None) -> Abbreviations | None:
    """This paper's own expansions, or None. Never another paper's, and never a guess."""
    if texts is None:
        return None
    from pondie import paths

    try:
        body = paths.best_text(study, texts).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return None
    return _store().for_paper(body, study)


def _expanded(head: str, store: Abbreviations | None, study: str) -> tuple[str, tuple[str, ...]]:
    """`head` with this paper's short forms spelled out, and what was expanded."""
    if store is None:
        return head, ()
    expanded, seen = head, []
    for short, expansion in expansions_in(head, store, study):
        expanded = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(short)}(?![A-Za-z0-9])", expansion, expanded
        )
        seen.append(expansion)
    return expanded, tuple(seen)


@dataclass
class Result:
    """Every row, plus what could not be placed and what that says about the vocabulary."""

    links: list[Link] = field(default_factory=list)
    triage: Counter = field(default_factory=Counter)
    cohorts: Counter = field(default_factory=Counter)
    parser: bool = False
    mondo_version: str = ""

    def review(self) -> list[Link]:
        return sorted(
            (link_ for link_ in self.links if link_.method == "review"),
            key=lambda link_: -link_.score,
        )

    def rejected(self) -> list[Link]:
        return [link_ for link_ in self.links if link_.method == "rejected"]

    def proposals(self, minimum: int = 5) -> list[dict]:
        """MONDO nodes ONVOC cannot name at any grain, by how many STUDIES name them."""
        grouped: dict[str, dict] = {}
        for link_ in self.links:
            if not link_.proposes:
                continue
            slot = grouped.setdefault(
                link_.mondo,
                {
                    "mondo": link_.mondo,
                    "label": link_.label,
                    "umls": link_.umls,
                    "sctid": link_.sctid,
                    "studies": set(),
                    "forms": set(),
                },
            )
            slot["studies"].add(link_.study)
            slot["forms"].add(link_.head)
        out = [
            {**slot, "studies": len(slot["studies"]), "forms": sorted(slot["forms"])[:8]}
            for slot in grouped.values()
        ]
        return sorted(
            [p for p in out if p["studies"] >= minimum], key=lambda p: -p["studies"]
        )

    def grain_gaps(self, minimum: int = 5) -> list[dict]:
        """Where MONDO is finer than the ONVOC term it bridged to, and the finer term recurs.

        A row here IS agreement: the MONDO node is a descendant of the node ONVOC names,
        so the two are one concept at two grains.
        """
        grouped: dict[tuple[str, str], dict] = {}
        for link_ in self.links:
            if link_.onvoc_via not in {"ancestor", "ancestor-name"}:
                continue
            key = (link_.mondo, link_.onvoc)
            slot = grouped.setdefault(
                key,
                {
                    "mondo": link_.mondo,
                    "label": link_.label,
                    "under": link_.onvoc,
                    "steps": len(link_.ancestors),
                    "studies": set(),
                    "forms": set(),
                },
            )
            slot["studies"].add(link_.study)
            slot["forms"].add(link_.head)
        out = [
            {**slot, "studies": len(slot["studies"]), "forms": sorted(slot["forms"])[:6]}
            for slot in grouped.values()
        ]
        return sorted([g for g in out if g["studies"] >= minimum], key=lambda g: -g["studies"])


def normalize(
    patterns: tuple[str, ...] = DEFAULT,
    texts: Path | None = None,
    accept: float = ACCEPT,
    review: float = REVIEW,
) -> Result:
    """Every condition in the corpus, linked to MONDO and bridged to ONVOC.

    `texts` is the corpus directory. Given, each paper's own abbreviations are mined from
    its own text; omitted, no expansion happens and the run says so.
    """
    from pondie.vocabularies import fetch

    vocab = load_mondo()
    crosswalk = onvoc_crosswalk()
    # Disorder branches only, as `ROUTES` scopes this field, so the bridge's name layer
    # cannot reach a drug or a brain region.
    onvoc = load_onvoc().scoped(("disorders",))
    result = Result(parser=parser_available(), mondo_version=fetch.version("mondo"))

    staged: list[tuple[Link, str]] = []
    for study, body in iter_records(patterns):
        store = _abbreviations(study, texts)
        for group in body.get("groups") or []:
            if not isinstance(group, dict):
                continue
            flag = value_of(group.get("is_healthy"))
            name = str(value_of(group.get("local_id")) or "")
            values = strings_at(group, "medical_condition")
            if not values:
                result.cohorts["no value"] += 1
                if flag is True:
                    result.cohorts["absent by flag"] += 1
                continue
            for item in values:
                triaged = triage(item)
                result.triage[triaged.kind] += 1
                sentinel, by_flag = triaged.sentinel, False
                # The only gap `is_healthy` fills: a value that says nothing either way.
                # It can add an absence, never remove one.
                if sentinel == NOT_READ and absent(item, flag):
                    sentinel, by_flag = NO_CONDITION, True
                if sentinel:
                    result.cohorts[
                        "unread"
                        if sentinel == NOT_READ
                        else ("absent by flag" if by_flag else "absent by string")
                    ] += 1
                    result.links.append(
                        Link(
                            study=study, group=name, text=item, sentinel=sentinel,
                            denied=triaged.denied,
                            scope="flag" if by_flag else triaged.scope,
                        )
                    )
                    continue
                result.cohorts["condition"] += 1
                for head in triaged.heads:
                    expanded, seen = _expanded(head, store, study)
                    staged.append(
                        (
                            Link(
                                study=study, group=name, text=item, head=head,
                                denied=triaged.denied, scope=triaged.scope,
                                qualifiers=triaged.qualifiers, expansions=seen,
                            ),
                            expanded,
                        )
                    )

    if staged:
        index = Index(vocab).build()
        placed = link(
            sorted({query for _row, query in staged}), vocab, index, accept, review
        )
        for row, query in staged:
            node, method, score, others = placed.get(query, (None, "", 0.0, []))
            if node is None:
                result.links.append(
                    Link(
                        **{
                            **asdict(row),
                            "method": method,
                            "score": round(score, 4),
                            "candidates": tuple(vocab.labels[n] for n in others[:5]),
                        }
                    )
                )
                continue
            onvoc_id, onvoc_label, via, path = bridge(node, vocab, crosswalk, onvoc)
            result.links.append(
                Link(
                    **{
                        **asdict(row),
                        "mondo": vocab.curie(node),
                        "label": vocab.labels[node],
                        "umls": vocab.umls.get(node, ""),
                        "sctid": vocab.sctid.get(node, ""),
                        "method": method,
                        "score": round(score, 4),
                        "ancestors": path,
                        "onvoc": onvoc_label,
                        "onvoc_id": onvoc_id,
                        "onvoc_via": via,
                        "candidates": tuple(vocab.labels[n] for n in others[:5]),
                    }
                )
            )
    return result


def calibrate(sample: int = 1500, seed: int = 0) -> dict:
    """The two score distributions the thresholds are read off, measured not assumed.

    Hold one exact form of a multi-form MONDO node out of the index, retrieve it against
    everything else, and score the top-1 by whether it resolved to the held-out node.
    """
    import random

    import numpy as np

    from pondie.normalization._embedding import for_phrases

    vocab = load_mondo()
    by_node: dict[int, list[int]] = {}
    for position, node in enumerate(vocab.form_node):
        by_node.setdefault(node, []).append(position)
    rich = [n for n, positions in by_node.items() if len(positions) >= 2]
    random.Random(seed).shuffle(rich)
    chosen = rich[:sample]
    held = {by_node[n][-1]: n for n in chosen}

    keep = [i for i in range(len(vocab.forms)) if i not in held]
    matrix = for_phrases(vocab.forms)[keep]
    node_of = [vocab.form_node[i] for i in keep]
    queries = [vocab.forms[i] for i in held]
    truth = [held[i] for i in held]

    encoded = for_phrases(queries, cache=False)
    tops, hits = [], []
    for start in range(0, len(queries), 256):
        scores = encoded[start : start + 256] @ matrix.T
        best = scores.argmax(1)
        tops.append(scores.max(1))
        hits += [node_of[int(b)] == t for b, t in zip(best, truth[start : start + 256])]
    top = np.concatenate(tops)
    correct = np.array(hits)

    def spread(values) -> dict:
        if not len(values):
            return {}
        return {
            "n": len(values),
            "p10": round(float(np.percentile(values, 10)), 3),
            "median": round(float(np.median(values)), 3),
            "p90": round(float(np.percentile(values, 90)), 3),
        }

    right, wrong = top[correct], top[~correct]

    def at(cut: float) -> dict:
        taken = top >= cut
        kept, admitted = int((taken & correct).sum()), int((taken & ~correct).sum())
        return {
            "cut": cut,
            "accepted": kept + admitted,
            "right": kept,
            "wrong": admitted,
            "precision": round(kept / max(kept + admitted, 1), 3),
            "of correct": round(kept / max(int(correct.sum()), 1), 3),
        }

    return {
        "recall@1": round(float(correct.mean()), 3),
        "correct": spread(right),
        "wrong": spread(wrong),
        # The sweep, not a verdict on the cuts in force: where to put them is a
        # precision-against-queue-length trade and the table is what makes it visible.
        "sweep": [at(c) for c in (0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98)],
        "in force": {"accept": ACCEPT, "review": REVIEW},
    }


def report(patterns: tuple[str, ...] = DEFAULT, texts: Path | None = None) -> str:
    out = normalize(patterns, texts)
    matched = [link_ for link_ in out.links if link_.matched]
    bridged = [link_ for link_ in matched if link_.onvoc]
    direct = sum(1 for m in bridged if m.onvoc_via == "crosswalk")
    by_ancestor = sum(1 for m in bridged if m.onvoc_via == "ancestor")
    release = out.mondo_version.rsplit("/", 2)[-2] if out.mondo_version else "?"
    lines = [
        f"MONDO {release}; parser {'on' if out.parser else 'OFF'}",
        (
            f"{len(out.links)} rows, {len(matched)} matched "
            f"({sum(1 for m in matched if m.sctid)} carry a SNOMED id, "
            f"{sum(1 for m in matched if m.umls)} a UMLS CUI)"
        ),
        (
            f"bridged to ONVOC: {len(bridged)} ({direct} direct, {by_ancestor} by "
            f"ancestor); {len(matched) - len(bridged)} ONVOC cannot name"
        ),
        f"triage: {dict(out.triage)}",
        f"cohorts: {dict(out.cohorts)}",
        "",
        f"{'studies':>7s}  {'MONDO':14s} {'SNOMED':11s} label -> ONVOC",
    ]
    for gap in out.grain_gaps()[:10]:
        lines.append(
            f"{gap['studies']:7d}  {gap['mondo']:14s} {'':11s} "
            f"{gap['label'][:34]:34s} -> {gap['under']}"
        )
    lines.append("")
    lines.append("ONVOC cannot name, by studies:")
    for proposal in out.proposals()[:10]:
        lines.append(
            f"{proposal['studies']:7d}  {proposal['mondo']:14s} "
            f"{proposal['sctid'] or '-':11s} {proposal['label'][:44]}"
        )
    lines.append(f"review queue {len(out.review())}, rejected {len(out.rejected())}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="*", default=list(DEFAULT))
    parser.add_argument("--texts", type=Path, help="the corpus, for paper abbreviations")
    parser.add_argument("--out", type=Path, help="write every row as JSON")
    parser.add_argument("--calibrate", action="store_true", help="measure the distributions")
    args = parser.parse_args()
    if args.calibrate:
        print(json.dumps(calibrate(), indent=1))
        return 0
    if args.out:
        result = normalize(tuple(args.records), args.texts)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "mondo_version": result.mondo_version,
                    "parser": result.parser,
                    "triage": dict(result.triage),
                    "cohorts": dict(result.cohorts),
                    "links": [asdict(link_) for link_ in result.links],
                    "proposals": result.proposals(),
                    "grain_gaps": result.grain_gaps(),
                },
                indent=1,
                default=list,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"{len(result.links)} rows -> {args.out}")
        return 0
    print(report(tuple(args.records), args.texts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
