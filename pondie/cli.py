"""One entry point, three verbs: extract, normalize, select.

Arguments are parsed into the same pydantic contracts a library caller builds by hand, so a
mistyped flag fails the same way a mistyped keyword does and there is one definition of what
a valid run is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contracts import Flavour, Paper, Settings, StageName, Workflow


def _papers(root: Path, ids: Path, flavour: Flavour) -> list[Paper]:
    """Study ids from the tab-separated pmids file the corpus uses: `pmid<TAB>id<TAB>source`.

    A file of bare ids parses to nothing and the run reports success having done nothing, so
    the shape is checked here rather than discovered three stages later.
    """
    papers, malformed = [], 0
    for line in ids.read_text().splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) >= 2 and parts[1]:
            papers.append(Paper(study_id=parts[1], root=root, flavour=flavour))
        elif line.strip():
            malformed += 1
    if malformed and not papers:
        raise SystemExit(f"{ids}: {malformed} line(s) parsed to nothing. Expected "
                         f"'pmid<TAB>study_id<TAB>source'; a bare id per line is not that.")
    return papers


def _extract(args: argparse.Namespace) -> int:
    from .extraction import GatewayCaller, load_env, plan, run

    if args.env:
        load_env(args.env)
    settings = Settings(
        payloads=args.payloads, records=args.records, model=args.model,
        workflow=Workflow(args.workflow),
        stages=tuple(StageName(s) for s in args.stages) if args.stages else tuple(StageName),
        effort=args.effort, retrieve_evidence=not args.no_evidence, redo=args.redo)
    papers = _papers(args.texts, args.pmids, Flavour(args.flavour))
    if args.plan:
        for study, steps in plan(papers, settings).items():
            print(f"  {study}  {' '.join(steps)}")
        return 0
    report = run(papers, settings, GatewayCaller(), workers=args.workers)
    print(report.summary())
    for paper in report.failures:
        print(f"  FAILED {paper.study_id}: {paper.failed.reason}")
    return 1 if report.failures else 0


def _normalize(args: argparse.Namespace) -> int:
    import importlib
    module = importlib.import_module(f".normalization.{args.field}", package="pondie")
    print(module.report(tuple(args.records)) if args.records else module.report())
    return 0


def _select(args: argparse.Namespace) -> int:
    from .query.engine import Selection, select

    kwargs: dict = {"contrast": args.contrast}
    if args.records:
        kwargs["records"] = tuple(args.records)
    if args.measure_type:
        kwargs["measure_type"] = frozenset(args.measure_type)
    if args.include_roi:
        kwargs["spatial_scope"] = frozenset({"whole_brain", "roi"})
    print(select(Selection(**kwargs)).funnel())
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .benchmark.run import run

    result = run(candidate=args.candidate, reference=args.reference, semantic=args.semantic)
    print(result.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pondie", description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)

    ex = sub.add_parser("extract", help="papers -> validated records")
    ex.add_argument("--pmids", type=Path, required=True)
    ex.add_argument("--texts", type=Path, required=True)
    ex.add_argument("--payloads", type=Path, required=True)
    ex.add_argument("--records", type=Path, required=True)
    ex.add_argument("--model", required=True)
    ex.add_argument("--env", type=Path, help="shell-style file of API credentials")
    ex.add_argument("--flavour", default=Flavour.pubget.value,
                    choices=[f.value for f in Flavour])
    ex.add_argument("--workflow", default=Workflow.demand_driven.value,
                    choices=[w.value for w in Workflow])
    ex.add_argument("--stages", nargs="*", choices=[s.value for s in StageName])
    ex.add_argument("--effort", default="low",
                    choices=["minimal", "low", "medium", "high"])
    ex.add_argument("--no-evidence", action="store_true",
                    help="skip the quote pass; 45%% of input tokens, and the record is "
                         "then structurally complete and unreviewable")
    ex.add_argument("--redo", action="store_true")
    ex.add_argument("--workers", type=int, default=1)
    ex.add_argument("--plan", action="store_true", help="say what would run, spend nothing")
    ex.set_defaults(fn=_extract)

    no = sub.add_parser("normalize", help="report one field's normalization")
    no.add_argument("field", help="e.g. coordinate_space, medication_status, task")
    no.add_argument("--records", action="append")
    no.set_defaults(fn=_normalize)

    se = sub.add_parser("select", help="what a meta-analysis should pool, and what it dropped")
    se.add_argument("--records", action="append")
    se.add_argument("--measure-type", action="append")
    se.add_argument("--contrast", default="any",
                    choices=["any", "within_subject", "between_group"])
    se.add_argument("--include-roi", action="store_true",
                    help="pool region-restricted analyses too; they can only report "
                         "coordinates inside the region they searched")
    se.set_defaults(fn=_select)

    from .benchmark.run import CANDIDATE, REFERENCE

    be = sub.add_parser("benchmark", help="score contrast polarity against the reviewer gold")
    be.add_argument("--candidate", type=Path, default=CANDIDATE,
                    help="the extraction being evaluated")
    be.add_argument("--reference", type=Path, default=REFERENCE,
                    help="the records the reviewer was shown; identity only, never scored")
    be.add_argument("--semantic", action="store_true",
                    help="embeddings for term same-ness rather than string comparison")
    be.set_defaults(fn=_benchmark)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
