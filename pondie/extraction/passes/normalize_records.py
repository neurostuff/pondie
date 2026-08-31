"""Map a corpus of records onto shared vocabularies, and report what could not be mapped.

Emits two artefacts, and the second is the point as much as the first:

    mappings    one row per routed field value, matched or not, carrying the method and
                the paper's own wording next to the concept it was mapped to
    candidates  the unmapped values, grouped and counted, as proposals for terms the
                vocabularies lack

    python normalize_records.py --records data/runs/v2/records --texts data/texts \\
        --out data/eval/normalization.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pondie.extraction.passes.pipeline.abbreviations import Abbreviations  # noqa: E402
from pondie.extraction.passes.pipeline.kinds import Paper  # noqa: E402
from pondie.extraction.passes.pipeline.normalize import (  # noqa: E402
    candidates,
    load_cognitive_atlas,
    load_onvoc,
    normalize,
)
from pondie.extraction.passes.pipeline.query import treatment_contrasts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument(
        "--texts",
        type=Path,
        help="corpus root; lets each paper's own abbreviation "
        "definitions override the corpus-wide ones",
    )
    parser.add_argument(
        "--abbreviations", type=Path, default=Path("data/vocab/abbreviations.json")
    )
    parser.add_argument("--out", type=Path, default=Path("data/eval/normalization.json"))
    parser.add_argument("--min-support", type=int, default=2)
    args = parser.parse_args()

    vocabularies = {"ONVOC": load_onvoc(), "CognitiveAtlas": load_cognitive_atlas()}
    corpus_store = Abbreviations.load(args.abbreviations)

    rows, contrasts = [], []
    papers = 0
    for root in args.records:
        for path in sorted(root.glob("*.extraction.json")):
            if path.name.endswith(".raw.json"):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            study = path.name.split(".")[0]
            store = corpus_store
            if args.texts:
                try:
                    store = corpus_store.for_paper(Paper(study, args.texts).text())
                except (FileNotFoundError, OSError):
                    pass
            rows += normalize(record, vocabularies, store)
            contrasts += list(treatment_contrasts(record))
            papers += 1

    proposals = candidates(rows, minimum=args.min_support)
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
