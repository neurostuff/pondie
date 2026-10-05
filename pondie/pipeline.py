"""Run steps over many items: cached on what each depends on, in parallel, with a record.

Extraction runs its stages over each paper and normalization runs one pass over each
record. Both need the same things: provenance, caching, invalidation when an input
changes, progress, and parallelism for network-bound work.

Caching and provenance are one mechanism. A `Stamp` beside each output records the digest
of everything it was computed from; an output is reused only while that digest still
matches, and the same file says what produced it.

    steps = [Step("single", produces=..., depends_on=..., run=...)]
    report = execute(papers, steps, workers=8)

`Step` is four callables rather than a base class, so extraction's stage classes and
normalization's functions can both be scheduled without changing shape.
"""

from __future__ import annotations

import contextlib
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

#: Part of every digest, so changing what a stamp means invalidates every cache.
STAMP_VERSION = "1"


def digest_of(parts: Mapping[str, Any], step: str = "") -> str:
    """A stable digest of everything an output was computed from.

    Keys are sorted and values stringified, so insertion order and `Path` objects do not
    change it. The step's name is included, so two steps writing one path cannot share a
    stamp.
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

    Written beside the output, not inside it, since an output need not be JSON. A missing or
    unreadable stamp reads as stale.
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
    def path_for(output: Path, step: str = "") -> Path:
        """`.stamps/<name>.<step>.json` beside the output (`<name>.json` with no step).

        A subdirectory, because `builder.merge_payloads` merges every `*.json` in a payload
        directory and would read a sibling stamp as a payload. Named per step, because two
        steps can write one output (`prose` and `split` both rewrite the stage-1 parse).
        """
        return (
            output.parent
            / ".stamps"
            / (f"{output.name}.{step}.json" if step else output.name + ".json")
        )

    @classmethod
    def read(cls, output: Path, step: str = "") -> "Stamp | None":
        """The stamp `step` left on `output`, or None.

        Falls back to an unqualified stamp written before stamps were named per step, if it
        was that step's.
        """
        for path in (
            (cls.path_for(output, step), cls.path_for(output)) if step else (cls.path_for(output),)
        ):
            if not path.is_file():
                continue
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
                stamp = cls(
                    step=body["step"],
                    digest=body["digest"],
                    parts=body.get("parts") or {},
                    produced_at=body.get("produced_at", ""),
                    seconds=body.get("seconds", 0.0),
                )
            except (OSError, ValueError, KeyError):
                return None
            return stamp if not step or stamp.step == step else None
        return None

    def write(self, output: Path) -> Path:
        path = self.path_for(output, self.step)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.as_json() + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class Step(Generic[Item]):
    """One unit of work, and what the scheduler needs to decide whether to do it.

    `depends_on` decides cache correctness: it must name every input whose change should
    change the output, or a stale answer is served as a fresh one.
    """

    name: str
    #: Where the answer goes. `None` for a step whose effect is not a file; never cached.
    produces: Callable[[Item], Path | None]
    depends_on: Callable[[Item], Mapping[str, Any]]
    run: Callable[[Item], Any]
    #: How many items may be inside this step at once, under the run's `workers`.
    max_parallel: int | None = None


@dataclass
class Outcome:
    """What happened to one item at one step."""

    item: str
    step: str
    state: str  # "done" | "cached" | "failed"
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
    """One JSON line per outcome, appended as it happens, so a killed run keeps its record."""

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


class _Progress:
    """Per-item progress: a tqdm bar on a terminal, and one log line per finished item.

    The log line is what a redirected run (`nohup … > run.log`) has instead of a bar: count,
    elapsed time and an ETA from the mean time per finished item. On a terminal the lines
    are routed above the bar so neither breaks the other.
    """

    def __init__(self, total: int, enabled: bool) -> None:
        self.total, self.done, self.started = total, 0, time.monotonic()
        self._lock = threading.Lock()
        self.bar = None
        if enabled and os.isatty(2):
            from tqdm import tqdm

            self.bar = tqdm(total=total, unit="paper", dynamic_ncols=True, leave=True)

    def redirect(self):
        """Route logging through the bar while it is open."""
        if self.bar is None:
            return contextlib.nullcontext()
        from tqdm.contrib.logging import logging_redirect_tqdm

        return logging_redirect_tqdm()

    def finished(self, label: str, detail: str) -> None:
        with self._lock:
            self.done += 1
            elapsed = time.monotonic() - self.started
            remaining = elapsed / self.done * (self.total - self.done)
            if self.bar is not None:
                self.bar.update(1)
                self.bar.set_postfix_str(label[:24])
            log.info(
                "[%d/%d] %s%s · %s elapsed, ~%s left",
                self.done,
                self.total,
                label,
                f"  {detail}" if detail else "",
                _clock(elapsed),
                _clock(remaining),
            )

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def fresh(step: Step[Item], item: Item, *, redo: bool = False) -> Stamp | None:
    """The stamp on a usable cached output, or None when the step has to run.

    Stale when asked to redo, when the output or its stamp is missing, or when the stamp's
    digest no longer matches what the step depends on.
    """
    if redo:
        return None
    output = step.produces(item)
    if output is None or not output.exists():
        return None
    stamp = Stamp.read(output, step.name)
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
    # Stamped only after the work returned, so a crash never leaves a trusted stamp.
    output = step.produces(item)
    if output is not None and output.exists():
        Stamp(
            step=step.name,
            digest=digest_of(step.depends_on(item), step.name),
            parts=dict(step.depends_on(item)),
            produced_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            seconds=elapsed,
        ).write(output)
    log.debug("%s/%s done in %.1fs", name, step.name, elapsed)
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
    describe: Callable[[Item, list[Outcome]], str] | None = None,
) -> Report:
    """Run every step over every item: parallel across items, in order within one.

    Parameters
    ----------
    progress
        Show a per-item bar when stderr is a terminal. A line per finished item is logged
        at INFO either way.
    describe
        Returns the text that line carries for an item, given its outcomes -- the caller's
        domain summary (cost, what failed). Without it the line states only the item.
    """

    items = list(items)
    journal = Events(events)
    report = Report()
    guards = {s.name: threading.Semaphore(s.max_parallel) for s in steps if s.max_parallel}
    tracker = _Progress(len(items), progress)

    def one(item: Item) -> list[Outcome]:
        label = name_of(item)
        out: list[Outcome] = []
        for step in steps:
            guard = guards.get(step.name)
            with guard if guard is not None else contextlib.nullcontext():
                outcome = _run_step(step, item, label, redo)
            out.append(outcome)
            journal.record(outcome)
            if not outcome.ok and stop_item_on_failure:
                break
        tracker.finished(label, describe(item, out) if describe else "")
        return out

    try:
        with tracker.redirect():
            if workers <= 1:
                for item in items:
                    report.outcomes += one(item)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for produced in pool.map(one, items):
                        report.outcomes += produced
    finally:
        tracker.close()
    log.info("%s", report.summary())
    return report
