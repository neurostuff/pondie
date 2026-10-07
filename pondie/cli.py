"""One entry point, four verbs: extract, normalize, select, benchmark.

Arguments are parsed into the same pydantic models a library caller builds by hand, so a
mistyped flag fails the same way a mistyped keyword does and there is one definition of what
a valid run is.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from pondie import paths
from pondie.extraction.models import (
    Flavour,
    Paper,
    Settings,
    StageName,
)

#: `--flavour best` takes each paper on the best render it actually has, via `Paper.best`.
#: Not a `Flavour` member, because it names a choice rather than a render.
BEST = "best"


def _papers(root: Path, ids: Path, flavour: Flavour | str) -> list[Paper]:
    """Papers from a tab-separated pmids file: `pmid<TAB>study_id<TAB>source`.

    `#` lines are skipped. An empty result raises, so a malformed file cannot run and
    report success having extracted nothing.
    """
    papers, malformed = [], 0
    for line in ids.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) >= 2 and parts[1]:
            study = parts[1]
            if flavour == BEST:
                # A paper with no text still enters the run; `driver.run` reports it.
                try:
                    papers.append(Paper.best(study, root))
                except FileNotFoundError:
                    papers.append(Paper(study_id=study, root=root, flavour=Flavour.pubget))
            else:
                papers.append(Paper(study_id=study, root=root, flavour=flavour))
        elif line.strip():
            malformed += 1
    if not papers:
        detail = (
            f"{malformed} line(s) parsed to nothing" if malformed else "no lines parsed to a study"
        )
        raise SystemExit(
            f"{ids}: {detail}. Expected 'pmid<TAB>study_id<TAB>source'; "
            f"a bare id per line is not that."
        )
    return papers


def _extract(args: argparse.Namespace) -> int:
    import logging

    from pondie.extraction import CodexCaller, GatewayCaller, load_env, plan, run, sequence

    # Configured here, not at import: the application decides how a library logs. `--log`
    # sets pondie's level; other libraries (LinkML, the HTTP client) log warnings only.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("pondie").setLevel(getattr(logging, args.log.upper()))

    if args.env:
        load_env(args.env)
    run_dir = paths.run(args.run)
    settings = Settings(
        payloads=run_dir / "payloads",
        records=run_dir / "records",
        model=args.model,
        **({"stages": tuple(StageName(s) for s in args.stages)} if args.stages else {}),
        effort=args.effort,
        **(
            {"stage_effort": dict(_stage_effort(x) for x in args.stage_effort)}
            if args.stage_effort
            else {}
        ),
        service_tier=args.service_tier,
        retrieve_evidence=not args.no_evidence,
        structured_outputs=args.structured_outputs,
        evidence_format=args.evidence_format,
        recheck_results=args.recheck_results,
        redo=args.redo,
    )
    papers = _papers(
        args.corpus,
        args.pmids,
        BEST if args.flavour == BEST else Flavour(args.flavour),
    )
    if args.plan:
        for study, steps in plan(papers, settings).items():
            print(f"  {study}  {' '.join(steps)}")
        return 0
    log = logging.getLogger("pondie")
    log.info(
        "run %s: %d paper(s), stages %s, model %s via %s, tier %s, effort %s, %d worker(s)",
        args.run,
        len(papers),
        " > ".join(s.name.value for s in sequence(settings)),
        settings.model,
        args.backend,
        "not sent" if args.backend == "codex" else settings.service_tier or "provider default",
        ", ".join(
            f"{s.name.value}={settings.effort_for(s.name)}"
            for s in sequence(settings)
            if getattr(s, "asks_a_model", False)
        )
        or "none (no model stage)",
        args.workers,
    )
    report = run(
        papers,
        settings,
        CodexCaller() if args.backend == "codex" else GatewayCaller(),
        workers=args.workers,
        progress=not args.no_progress,
    )
    print(report.summary())
    for paper in report.failures:
        print(f"  FAILED {paper.study_id}: {paper.failed.reason}")
    return 1 if report.failures else 0


#: `pondie normalize derived` runs the fills instead of reporting one field. Not a field
#: name, and it cannot collide with one: `fields()` returns module names and there is no
#: `normalization.derived` module.
DERIVED = "derived"


def _normalizable() -> list[str]:
    """The fields `pondie normalize` can report on, asked of the package.

    This keeps the parser's choices aligned with the package and
    `pondie normalize _onvoc` is refused by the parser rather than importing a private
    module and dying inside it.
    """
    from pondie import normalization

    return [*normalization.fields(), DERIVED]


def _derived_report(records: tuple[str, ...]) -> int:
    """Run every derived-slot fill over the records and total the tallies.

    The reporting half of `apply_derived`. Without it the fills were the one part of the
    package with no way to be measured on real records short of writing a script, and the
    two defects that run found -- a unit written onto 711 groups that stated no age, and a
    disease vocabulary asked to name "Typically developing adolescents" -- were both
    invisible to the per-field reports, which only ever see values that exist.
    """
    from pondie.normalization import DERIVED as MODULES
    from pondie.normalization import apply_derived
    from pondie.normalization._records import DEFAULT, iter_records

    totals: dict[str, Counter] = {name: Counter() for name in MODULES}
    seen = 0
    for _study, body in iter_records(records or DEFAULT):
        seen += 1
        for name, tally in apply_derived(body).items():
            totals[name].update(tally)
    print(f"{seen} records")
    for name in MODULES:
        counted = "  ".join(f"{k}={v}" for k, v in sorted(totals[name].items()))
        print(f"  {name:28} {counted}")
    return 0


def _normalize(args: argparse.Namespace) -> int:
    import importlib

    if args.field == DERIVED:
        return _derived_report(tuple(args.records or ()))
    module = importlib.import_module(f".normalization.{args.field}", package="pondie")
    print(module.report(tuple(args.records)) if args.records else module.report())
    return 0


def _select(args: argparse.Namespace) -> int:
    from pondie.query.engine import Selection, select

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
    from pondie.benchmark import run

    result = run(candidate=args.candidate, reference=args.reference)
    print(result.summary() if args.brief else result.report(limit=args.limit))
    return 0


def _stage_effort(spec: str) -> tuple[StageName, str]:
    stage, _, level = spec.partition("=")
    if level not in ("minimal", "low", "medium", "high"):
        raise SystemExit(f"--stage-effort {spec!r}: expected STAGE=minimal|low|medium|high")
    return StageName(stage), level


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pondie", description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)

    ex = sub.add_parser("extract", help="papers -> validated records")
    ex.add_argument("--pmids", type=Path, required=True)
    ex.add_argument(
        "--run",
        required=True,
        help="names the run. Its payloads, records and usage log go in one "
        f"directory under {paths.RUNS}, so one extraction is one place",
    )
    ex.add_argument(
        "--corpus",
        type=Path,
        default=paths.CORPUS,
        help="the synced papers; an input, never written by a run",
    )
    ex.add_argument("--model", required=True)
    ex.add_argument("--env", type=Path, help="shell-style file of API credentials")
    ex.add_argument(
        "--flavour",
        default=BEST,
        choices=[BEST, *(f.value for f in Flavour)],
        help="which render to extract from; `best` takes the richest each paper has",
    )
    ex.add_argument("--stages", nargs="*", choices=[s.value for s in StageName])
    ex.add_argument(
        "--structured-outputs",
        action="store_true",
        help="decode single, fill and evidence replies under a schema from the extraction "
        "schema (strict Structured Outputs) instead of JSON mode",
    )
    ex.add_argument(
        "--evidence-format",
        default="quotes",
        choices=["quotes", "indexed"],
        help="how single and fill cite: a quote per value, or the numbers of the sentences "
        "that state it",
    )
    ex.add_argument(
        "--recheck-results",
        action="store_true",
        help="ask once more for analyses reported in Results sentences no analysis cites "
        "(indexed evidence only)",
    )
    ex.add_argument(
        "--effort",
        default="low",
        choices=["minimal", "low", "medium", "high"],
        help="reasoning effort for any stage --stage-effort does not name",
    )
    ex.add_argument(
        "--stage-effort",
        nargs="*",
        metavar="STAGE=LEVEL",
        help="per-stage effort, replacing the default map "
        "(single=medium fill=low evidence=low repair=medium)",
    )
    ex.add_argument(
        "--service-tier",
        default="flex",
        choices=["", "flex", "default", "priority"],
        help="the provider's service tier; `flex` is cheaper and slower, '' leaves the "
        "provider's default",
    )
    ex.add_argument(
        "--backend",
        default="gateway",
        choices=["gateway", "codex"],
        help="how calls reach the model: the API gateway, or `codex exec` on the "
        "`codex login` account (same model and effort; no service tier)",
    )
    ex.add_argument(
        "--no-evidence",
        action="store_true",
        help="skip the quote pass; values are kept, without supporting quotes",
    )
    ex.add_argument("--redo", action="store_true")
    ex.add_argument("--workers", type=int, default=1)
    ex.add_argument(
        "--no-progress",
        action="store_true",
        help="no progress bar; it is off anyway when stderr is not a terminal",
    )
    ex.add_argument(
        "--log",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="`info` logs one line per paper; `debug` adds every step and cache hit",
    )
    ex.add_argument("--plan", action="store_true", help="say what would run, spend nothing")
    ex.set_defaults(fn=_extract)

    no = sub.add_parser(
        "normalize",
        help="report one field's normalization, or `derived` to run the slot fills",
    )
    no.add_argument("field", choices=_normalizable())
    no.add_argument("--records", action="append")
    no.set_defaults(fn=_normalize)

    se = sub.add_parser("select", help="what a meta-analysis should pool, and what it dropped")
    se.add_argument("--records", action="append")
    se.add_argument("--measure-type", action="append")
    se.add_argument(
        "--contrast", default="any", choices=["any", "within_subject", "between_group"]
    )
    se.add_argument(
        "--include-roi",
        action="store_true",
        help="pool region-restricted analyses too; they can only report "
        "coordinates inside the region they searched",
    )
    se.set_defaults(fn=_select)

    from pondie.benchmark import CANDIDATE, REFERENCE

    be = sub.add_parser(
        "benchmark", help="per-field precision/recall/F1, and contrast direction accuracy"
    )
    be.add_argument(
        "--candidate", type=Path, default=CANDIDATE, help="the extraction being evaluated"
    )
    be.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE,
        help="the records the reviewer was shown; identity only, never scored",
    )
    be.add_argument("--brief", action="store_true", help="the headline only, no tables")
    be.add_argument(
        "--limit",
        type=int,
        default=0,
        help="show only the N worst fields; 0 shows every one",
    )
    be.set_defaults(fn=_benchmark)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
