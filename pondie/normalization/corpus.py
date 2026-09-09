"""Map a corpus of records onto shared vocabularies, and report what could not be mapped.

Emits two artefacts, and the second is the point as much as the first:

    mappings    one row per routed field value, matched or not, carrying the method and
                the paper's own wording next to the concept it was mapped to
    candidates  the unmapped values, grouped and counted, as proposals for terms the
                vocabularies lack

    python -m pondie.normalization.corpus --records data/runs/v2/records --texts data/corpus \\
        --out data/runs/<run>/normalization.json
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from pondie import paths, pipeline
from pondie.normalization.contrasts import treatment_contrasts  # noqa: E402

# Qualified, so this module does not re-export a callable `normalize` and read as one of
# the eight field modules. It maps a whole corpus; they each map one field.
from pondie.vocabularies import onvoc
from pondie.vocabularies.abbreviations import Abbreviations  # noqa: E402


def _vocabulary_digest(vocabularies: Mapping[str, Any], store: Any) -> str:
    """What the mappings were computed against.

    The vocabularies are fetched files that change under a corpus that does not, which is
    exactly the case a cache keyed on the record alone gets wrong: ONVOC gains a concept and
    every stale mapping is served as current. Sizes rather than contents, because loading
    them already cost the read and hashing 32k labels per run to save seconds of mapping is
    the wrong trade -- a vocabulary that changes without changing size is the gap, and
    `--redo` is the answer to it.
    """
    return pipeline.digest_of(
        {name: len(vocab) for name, vocab in vocabularies.items()}
        | {"abbreviations": len(getattr(store, "entries", ()) or ())}
    )


def _map_corpus(sources, mappings_for, digest: str, args) -> tuple[list, list, int]:
    """Map every record, cached per record when `--cache` says where to keep them.

    The same scheduler `extraction` runs on, and the point of it being a separate module:
    what differs here is only the domain -- one step rather than nine, a record rather than a
    paper, and a cache that is optional because this pass is seconds per record rather than
    minutes.
    """
    sources = list(sources)
    if args.cache is None:
        rows, contrasts = [], []
        for path in sources:
            got, made = mappings_for(path)
            rows += got
            contrasts += made
        return rows, contrasts, len(sources)

    mapped: dict[Path, tuple[list, list]] = {}
    lock = threading.Lock()
    cache: Path = args.cache

    def produces(path: Path) -> Path:
        return cache / f"{path.name.split('.')[0]}.mappings.json"

    def depends_on(path: Path) -> dict:
        return {
            "record": pipeline.digest_of({"body": path.read_text(encoding="utf-8")}),
            "vocabularies": digest,
        }

    def run(path: Path) -> str:
        got, made = mappings_for(path)
        out = produces(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": len(got), "contrasts": len(made)}), encoding="utf-8")
        with lock:
            mapped[path] = (got, made)
        return f"{len(got)} row(s)"

    pipeline.execute(
        sources,
        [pipeline.Step(name="normalize", produces=produces, depends_on=depends_on, run=run)],
        name_of=lambda path: path.name.split(".")[0],
        workers=args.workers,
        redo=args.redo,
        progress=not args.no_progress,
        events=cache / "events.jsonl",
    )

    # A cached record was not mapped this run, so its rows are not in memory. Re-derive them
    # rather than read the sidecar back: a `Mapping` carries a `Concept`, and a second reader
    # that rebuilt one from JSON would be a second answer to what a mapping is. The sidecar
    # is what the stamp is attached to and a count for a reader, not a wire format.
    rows, contrasts = [], []
    for path in sources:
        got, made = mapped.get(path) or mappings_for(path)
        rows += got
        contrasts += made
    return rows, contrasts, len(sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument(
        "--texts",
        type=Path,
        help="corpus root; lets each paper's own abbreviation "
        "definitions override the corpus-wide ones",
    )
    parser.add_argument("--abbreviations", type=Path, default=paths.VOCAB / "abbreviations.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument(
        "--cache",
        type=Path,
        help="per-record mappings, reused while the record and the vocabularies are "
        "unchanged. Without it every run maps the whole corpus again",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    vocabularies = {"ONVOC": onvoc.load_onvoc(), "CognitiveAtlas": onvoc.load_cognitive_atlas()}
    corpus_store = Abbreviations.load(args.abbreviations)

    sources = [
        path
        for root in args.records
        for path in sorted(root.glob("*.extraction.json"))
        if not path.name.endswith(".raw.json")
    ]

    def mappings_for(path: Path) -> tuple[list, list]:
        """One record's rows and contrasts, from cache when the cache is still good."""
        record = json.loads(path.read_text(encoding="utf-8"))
        study = path.name.split(".")[0]
        store = corpus_store
        if args.texts:
            try:
                store = corpus_store.for_paper(
                    paths.best_text(study, args.texts).read_text(encoding="utf-8")
                )
            except (FileNotFoundError, OSError):
                pass
        return onvoc.normalize(record, vocabularies, store), list(treatment_contrasts(record))

    rows, contrasts, papers = _map_corpus(
        sources, mappings_for, _vocabulary_digest(vocabularies, corpus_store), args
    )

    proposals = onvoc.candidates(rows, minimum=args.min_support)
    matched = sum(1 for r in rows if r.matched)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "papers": papers,
                "routed": len(rows),
                "matched": matched,
                "methods": dict(Counter(r.method for r in rows if r.matched)),
                "mappings": [
                    {
                        "study": r.study_id,
                        "path": r.path,
                        "text": r.text,
                        "concept": r.concept.label if r.concept else None,
                        "concept_id": r.concept.id if r.concept else None,
                        "vocabulary": r.concept.vocabulary if r.concept else None,
                        "branch": r.concept.branch if r.concept else None,
                        "method": r.method,
                        "expansions": list(r.expansions),
                    }
                    for r in rows
                ],
                "candidates": [
                    {
                        "text": c.text,
                        "path": c.path,
                        "group": c.branch_group,
                        "papers": list(c.papers),
                        "support": c.support,
                        "expansions": list(c.expansions),
                    }
                    for c in proposals
                ],
                "treatment_contrasts": [
                    {
                        "study": t.study_id,
                        "analysis": t.analysis,
                        "analysis_name": t.analysis_name,
                        "intervention": t.intervention.name,
                        "intervention_kind": t.intervention.kind,
                        "agent": t.intervention.agent,
                        "comparator": t.comparator.name,
                        "comparator_kind": t.comparator.kind,
                        "direction": t.direction,
                        "relation": t.relation,
                        "measure": t.measure,
                        "held": list(t.held),
                        "consistent": t.consistent,
                    }
                    for t in contrasts
                ],
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"{papers} record(s): {matched}/{len(rows)} values mapped "
        f"({matched * 100 // max(len(rows), 1)}%)"
    )
    print(f"{len(proposals)} term proposal(s) at support >= {args.min_support}")
    print(f"{len(contrasts)} treatment contrast(s)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
