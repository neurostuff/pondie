"""Run work over many items: cached on what it depends on, in parallel, with a record.

Two workflows in this package do the same thing in different words. `extraction` runs nine
stages over each paper; `normalization` runs one pass over each record. Both want the same
five things, and between them had one and a half:

    provenance    what produced this output, from what
    caching       do not pay for an answer already on disk
    invalidation  unless what it was computed from has changed
    progress      a run that takes forty minutes should say so before minute forty
    parallelism   the work is network-bound, so run several at once

The half was `extraction`'s cache, and the half it was missing is the one that matters. A
stage was done when its output file existed. Change the prompt, the model, the effort or the
schema and every stale payload was reused in silence -- the fault that a cache exists to make
impossible, and the reason this module is a rewrite rather than a wrapper.

The two features are one mechanism. A `Stamp` beside each output records what produced it and
the digest of everything it was computed from; caching is asking whether that digest still
matches, and provenance is reading the same file. There is no second bookkeeping to fall out
of step with the first.

    steps = [Step("demands", produces=..., depends_on=..., run=...)]
    report = execute(papers, steps, workers=8)

`Step` is a plain dataclass of four callables rather than a base class to inherit. Extraction's
stages are already classes with their own hierarchy, and normalization's passes are functions;
asking either to change shape to be scheduled is how a scheduler ends up owning a domain it
should only be running.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

log = logging.getLogger("pondie")

Item = TypeVar("Item")

#: Bumped when a change to this module alters what a stamp means. Part of every digest, so a
#: change to the stamping rules invalidates every cache rather than silently reinterpreting
#: stamps written under the old ones.
STAMP_VERSION = "1"


def digest_of(parts: Mapping[str, Any], step: str = "") -> str:
    """A stable digest of everything an output was computed from.

    `sort_keys` and `default=str` so that a dict whose insertion order differs, or which holds
    a Path, still digests the same. The alternative -- hashing a repr -- makes the cache turn
    over on a refactor that changed nothing a caller can see.

    The step's name is part of it, so two steps that happen to write one path cannot read each
    other's stamp. No stage does that today; it costs nothing to make it undecidable rather
    than a thing to remember.
    """
    import hashlib

    body = json.dumps(
        {"_stamp_version": STAMP_VERSION, "_step": step, **dict(parts)},
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Stamp:
    """What produced an output, and from what.

    Written beside the output rather than inside it, because not every output is JSON and an
    output's schema is not this module's to extend. A missing stamp means "not computed by a
    version of this code that stamps", which is treated as stale: it is the safe reading, and
    it is what makes adopting this module a one-time recompute rather than a silent mixture.
    """

    step: str
    digest: str
    parts: Mapping[str, Any]
    produced_at: str
    seconds: float = 0.0

    def as_json(self) -> str:
        return json.dumps(
            {
                "step": self.step,
                "digest": self.digest,
                "parts": dict(self.parts),
                "produced_at": self.produced_at,
                "seconds": round(self.seconds, 3),
            },
            indent=1,
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def path_for(output: Path) -> Path:
        return output.with_name(output.name + ".stamp.json")

    @classmethod
    def read(cls, output: Path) -> "Stamp | None":
        path = cls.path_for(output)
        if not path.is_file():
            return None
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                step=body["step"],
                digest=body["digest"],
                parts=body.get("parts") or {},
                produced_at=body.get("produced_at", ""),
                seconds=body.get("seconds", 0.0),
            )
        except (OSError, ValueError, KeyError):
            # An unreadable stamp is a stale stamp. Raising here would fail a run over
            # bookkeeping, and trusting it would be worse.
            return None

    def write(self, output: Path) -> Path:
        path = self.path_for(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.as_json() + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class Step(Generic[Item]):
    """One unit of work, and everything the scheduler needs to decide whether to do it.

    `depends_on` is the whole of cache correctness. It must name every input whose change
    should produce a different output -- the paper text's hash, the prompt's version, the
    model, the settings that reach the call -- and nothing that changes without changing the
    answer, or the cache never hits. Naming too little is the dangerous direction: that is a
    stale answer served as a fresh one.
    """

    name: str
    #: Where the answer goes. `None` for a step whose effect is not a file, which is then
    #: never cached -- honest, because this module cannot know when such a step is stale.
    produces: Callable[[Item], Path | None]
    depends_on: Callable[[Item], Mapping[str, Any]]
    run: Callable[[Item], Any]
    #: Ceiling on how many items may be inside this step at once, under the run's own
    #: `workers`. For a step that holds a scarce resource where the others only wait.
    max_parallel: int | None = None


@dataclass
class Outcome:
    """What happened to one item at one step."""

    item: str
    step: str
    state: str  # "done" | "cached" | "failed" | "skipped"
    seconds: float = 0.0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.state != "failed"


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.state == "failed"]

    def tally(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.outcomes:
            out[o.state] = out.get(o.state, 0) + 1
        return out

    def summary(self) -> str:
        counts = self.tally()
        items = len({o.item for o in self.outcomes})
        spent = sum(o.seconds for o in self.outcomes)
        parts = ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))
        return f"{items} item(s) · {parts} · {spent:.0f}s of step time"


class Events:
    """One JSON object per outcome, appended as it happens.

    Appended rather than written at the end, which is what `extraction` did: a run killed at
    minute thirty left no record of the thirty minutes. Locked because the pool writes from
    several threads, and a torn line is worse than a missing one -- it makes the whole file
    unparseable rather than the last row absent.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, outcome: Outcome) -> None:
        if self.path is None:
            return
        line = json.dumps(
            {
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "item": outcome.item,
                "step": outcome.step,
                "state": outcome.state,
                "seconds": round(outcome.seconds, 3),
                "detail": outcome.detail,
            },
            ensure_ascii=False,
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _bar(total: int, enabled: bool):
    """A tqdm bar, or a stand-in with the same three methods.

    Disabled off a TTY as well as on request: a progress bar in a log file is one line of
    control characters per update, and this pipeline's long runs are usually redirected.
    """
    if not enabled or not os.isatty(2):

        class _Quiet:
            def update(self, n: int = 1) -> None: ...
            def set_postfix_str(self, s: str) -> None: ...
            def close(self) -> None: ...

        return _Quiet()
    from tqdm import tqdm

    return tqdm(total=total, unit="item", dynamic_ncols=True, leave=True)


def fresh(step: Step[Item], item: Item, *, redo: bool = False) -> Stamp | None:
    """The stamp on a usable cached output, or None when the step has to run.

    Four ways to be stale, and they are not the same thing: asked to redo, no output, no
    stamp, or a stamp whose digest no longer matches what the step now depends on. The last
    is the one that did not exist before -- the others were already decidable from the
    filesystem.
    """
    if redo:
        return None
    output = step.produces(item)
    if output is None or not output.exists():
        return None
    stamp = Stamp.read(output)
    if stamp is None:
        return None
    return stamp if stamp.digest == digest_of(step.depends_on(item), step.name) else None


def _run_step(step: Step[Item], item: Item, name: str, redo: bool) -> Outcome:
    if (stamp := fresh(step, item, redo=redo)) is not None:
        log.debug("%s/%s cached (%s)", name, step.name, stamp.digest[:12])
        return Outcome(item=name, step=step.name, state="cached")

    started = time.monotonic()
    try:
        detail = step.run(item)
    except Exception as error:  # noqa: BLE001 -- one item's failure is not the run's
        elapsed = time.monotonic() - started
        log.warning("%s/%s failed: %s: %s", name, step.name, type(error).__name__, error)
        return Outcome(
            item=name,
            step=step.name,
            state="failed",
            seconds=elapsed,
            detail=f"{type(error).__name__}: {error}",
        )

    elapsed = time.monotonic() - started
    # Stamped only after the work returned. A stamp written first would mark a crashed step
    # as complete, and the next run would trust it.
    output = step.produces(item)
    if output is not None and output.exists():
        Stamp(
            step=step.name,
            digest=digest_of(step.depends_on(item), step.name),
            parts=dict(step.depends_on(item)),
            produced_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            seconds=elapsed,
        ).write(output)
    log.info("%s/%s done in %.1fs", name, step.name, elapsed)
    return Outcome(
        item=name,
        step=step.name,
        state="done",
        seconds=elapsed,
        detail="" if detail is None else str(detail)[:200],
    )


def execute(
    items: Iterable[Item],
    steps: Sequence[Step[Item]],
    *,
    name_of: Callable[[Item], str] = str,
    workers: int = 1,
    redo: bool = False,
    progress: bool = True,
    events: Path | None = None,
    stop_item_on_failure: bool = True,
) -> Report:
    """Run every step over every item: parallel across items, in order within one.

    That is the shape both workflows have. Extraction's stages feed each other, so a paper's
    steps are a sequence and only the papers are concurrent; normalization's single pass over
    a record is the same shape with one step. Nothing here runs two steps of one item at
    once, because a step whose input is the step before it cannot be told apart from one
    that is merely listed after it.
    """

    items = list(items)
    journal = Events(events)
    report = Report()
    guards = {s.name: threading.Semaphore(s.max_parallel) for s in steps if s.max_parallel}
    bar = _bar(len(items), progress)
    lock = threading.Lock()

    def one(item: Item) -> list[Outcome]:
        label = name_of(item)
        out: list[Outcome] = []
        for step in steps:
            guard = guards.get(step.name)
            if guard is not None:
                with guard:
                    outcome = _run_step(step, item, label, redo)
            else:
                outcome = _run_step(step, item, label, redo)
            out.append(outcome)
            journal.record(outcome)
            if not outcome.ok and stop_item_on_failure:
                break
        with lock:
            bar.update(1)
            bar.set_postfix_str(label[:24])
        return out

    try:
        if workers <= 1:
            for item in items:
                report.outcomes += one(item)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for produced in pool.map(one, items):
                    report.outcomes += produced
    finally:
        bar.close()
    log.info("%s", report.summary())
    return report
