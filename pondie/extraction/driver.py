"""Run papers through the stages and say exactly what happened.

Returns a `RunReport` rather than printing one. A caller that wants text calls `.summary()`;
a test asserts on `.failures`. Cost is summed from what each stage returned, not scraped back
out of logging.

A paper stops at its first failing stage. The stages are ordered by dependency, so continuing
past a failure would run a stage against inputs that do not exist and report a second, less
informative error.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from pondie.extraction.llm import Caller
from pondie.extraction.models import Paper, PaperOutcome, RunReport, Settings, StageOutcome
from pondie.extraction.stages import sequence


def run_paper(paper: Paper, settings: Settings, caller: Caller) -> PaperOutcome:
    outcomes: list[StageOutcome] = []
    if not paper.ready():
        return PaperOutcome(
            study_id=paper.study_id,
            outcomes=(
                StageOutcome(
                    stage=settings.stages[0],
                    study_id=paper.study_id,
                    reason=f"missing text or stage-1 parse under {paper.root}",
                ),
            ),
        )
    for stage in sequence(settings):
        try:
            outcome = stage.run(paper, settings, caller)
        except Exception as error:  # noqa: BLE001 -- one paper's failure is not the run's
            outcome = StageOutcome(
                stage=stage.name,
                study_id=paper.study_id,
                reason=f"{type(error).__name__}: {error}",
            )
        outcomes.append(outcome)
        if not outcome.ok:
            break
    return PaperOutcome(study_id=paper.study_id, outcomes=tuple(outcomes))


def plan(papers: Iterable[Paper], settings: Settings) -> dict[str, list[str]]:
    """What would run and what would be skipped, without spending anything."""
    out: dict[str, list[str]] = {}
    for paper in papers:
        steps = []
        for stage in sequence(settings):
            state = (
                "skip" if getattr(stage, "done", lambda *_: False)(paper, settings) else "run"
            )
            steps.append(f"{state}:{stage.name.value}")
        out[paper.study_id] = steps
    return out


def run(
    papers: Iterable[Paper], settings: Settings, caller: Caller, workers: int = 1
) -> RunReport:
    papers = list(papers)
    if workers <= 1:
        return RunReport(papers=tuple(run_paper(p, settings, caller) for p in papers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return RunReport(
            papers=tuple(pool.map(lambda p: run_paper(p, settings, caller), papers))
        )
