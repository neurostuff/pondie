"""Run papers through the stages and report what happened.

Returns a `RunReport` rather than printing one; cost is summed from what each stage
returned. A paper stops at its first failing stage, since every later stage reads what an
earlier one wrote.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable

from pondie import pipeline
from pondie.extraction.llm import Caller
from pondie.extraction.models import (
    Cost,
    Paper,
    PaperOutcome,
    RunReport,
    Settings,
    StageName,
    StageOutcome,
)
from pondie.extraction.stages import sequence


def _why(error: BaseException, depth: int = 3) -> str:
    """The exception and up to `depth` causes down its chain.

    `llm.py` raises `RuntimeError("… attempt(s) failed") from <the gateway's error>`, and
    the cause is what says whether the call was refused, timed out or came back malformed.
    """
    parts, seen = [f"{type(error).__name__}: {error}"], {id(error)}
    cause = error.__cause__ or error.__context__
    while cause is not None and id(cause) not in seen and len(parts) <= depth:
        seen.add(id(cause))
        parts.append(f"{type(cause).__name__}: {str(cause)[:300]}")
        cause = cause.__cause__ or cause.__context__
    return " <- ".join(parts)


def plan(papers: Iterable[Paper], settings: Settings) -> dict[str, list[str]]:
    """What would run and what would be skipped, without spending anything."""
    out: dict[str, list[str]] = {}
    for paper in papers:
        steps = []
        for stage in sequence(settings):
            state = "skip" if getattr(stage, "done", lambda *_: False)(paper, settings) else "run"
            steps.append(f"{state}:{stage.name.value}")
        out[paper.study_id] = steps
    return out


class _StageFailed(RuntimeError):
    """A stage that returned `ok=False`, so the scheduler stops the paper."""


def run(
    papers: Iterable[Paper],
    settings: Settings,
    caller: Caller,
    workers: int = 1,
    progress: bool = True,
) -> RunReport:
    """Schedule the stages over the papers, and report what each one did.

    Scheduling, caching, progress and the event journal are `pondie.pipeline`'s. A stage
    still reports `skipped=True` when it has nothing to do; the scheduler separately skips
    a stage whose output is fresh. Both reach the report.
    """

    papers = list(papers)
    collected: dict[str, list[StageOutcome]] = defaultdict(list)
    #: (study, stage) pairs a stage reported on itself; the rest were cached or raised.
    spoke: set[tuple[str, str]] = set()
    lock = threading.Lock()

    def as_step(stage) -> pipeline.Step[Paper]:
        def do(paper: Paper):
            try:
                outcome = stage.run(paper, settings, caller)
            except Exception as error:
                raise _StageFailed(_why(error)) from error
            with lock:
                collected[paper.study_id].append(outcome)
                spoke.add((paper.study_id, stage.name.value))
            if not outcome.ok:
                raise _StageFailed(outcome.reason or "stage reported a failure")
            return outcome.reason or ""

        return pipeline.Step(
            name=stage.name.value,
            produces=lambda paper: stage.produces(paper, settings),
            depends_on=lambda paper: stage.depends_on(paper, settings),
            run=do,
        )

    def describe(paper: Paper, outcomes: list[pipeline.Outcome]) -> str:
        """One paper's line: stages run and cached, what it cost, and what failed."""
        with lock:
            cost = sum((o.cost for o in collected[paper.study_id]), Cost())
        ran = sum(o.state == "done" for o in outcomes)
        cached = sum(o.state == "cached" for o in outcomes)
        text = (
            f"{ran} run, {cached} cached · {cost.calls} calls, "
            f"{cost.input_tokens / 1e3:.0f}k in ({cost.cached_tokens / 1e3:.0f}k cached), "
            f"{cost.output_tokens / 1e3:.0f}k out"
        )
        failed = next((o for o in outcomes if not o.ok), None)
        if failed is not None:
            text += f" · FAILED at {failed.step}: {failed.detail[:160]}"
        return text

    steps = [as_step(stage) for stage in sequence(settings)]
    ready = [p for p in papers if p.ready()]
    for paper in papers:
        if paper.ready():
            continue
        collected[paper.study_id].append(
            StageOutcome(
                stage=settings.stages[0],
                study_id=paper.study_id,
                reason=f"missing text or stage-1 parse under {paper.root}",
            )
        )

    run_report = pipeline.execute(
        ready,
        steps,
        name_of=lambda paper: paper.study_id,
        workers=workers,
        redo=settings.redo,
        progress=progress,
        events=settings.records.parent / "events.jsonl",
        describe=describe,
    )
    # A stage that was cached or raised returned no `StageOutcome`; add one for each.
    for outcome in run_report.outcomes:
        if (outcome.item, outcome.step) in spoke:
            continue
        if outcome.state == "cached":
            collected[outcome.item].append(
                StageOutcome(
                    stage=StageName(outcome.step),
                    study_id=outcome.item,
                    skipped=True,
                    reason="unchanged since it was last produced",
                )
            )
        elif outcome.state == "failed":
            collected[outcome.item].append(
                StageOutcome(
                    stage=StageName(outcome.step),
                    study_id=outcome.item,
                    reason=outcome.detail,
                )
            )

    report = RunReport(
        papers=tuple(
            PaperOutcome(study_id=paper.study_id, outcomes=tuple(collected[paper.study_id]))
            for paper in papers
        )
    )
    _record_usage(report, settings)
    return report


def _record_usage(report: RunReport, settings: Settings) -> None:
    """Append one row per stage to the run's `usage.jsonl`. Never fatal."""
    rows = [
        {
            "paper": outcome.study_id,
            "stage": stage.stage.value,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "skipped": stage.skipped,
            "trace_ids": [trace for trace, _status in stage.traces if trace],
            "cache_status": sorted({status for _t, status in stage.traces if status}),
            **stage.cost.model_dump(),
        }
        for outcome in report.papers
        for stage in outcome.outcomes
    ]
    if not rows:
        return
    target = settings.payloads.parent / "usage.jsonl"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as error:
        print(f"  usage not recorded ({type(error).__name__}: {error})", flush=True)
