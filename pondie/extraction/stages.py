"""The steps, each one an object that says what it needs, what it leaves behind, and how.

A stage is a function of (paper, settings, caller) -> StageOutcome. It is not a subprocess:
the previous shape shelled out to a script and scraped its cost back off its own logging,
which meant a stage could not be called from a test, composed, or summed.

The order is the design. `Demands` runs before `Satisfy` so the analyses declare the terms
they need before any entity exists -- asked to guess an inventory first, the entity pass
modelled a crossover's condition as a continuous covariate, and a cell cannot be righter than
the term it points at.

`Tables` takes no model at all. `table_number`, `caption` and `footer` are literal strings in
the parse manifest, so putting them through a model can only introduce error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from pondie.extraction.llm import Caller
from pondie.extraction.models import Cost, ModelCall, Paper, Settings, StageName, StageOutcome


@runtime_checkable
class Stage(Protocol):
    name: StageName

    def produces(self, paper: Paper, settings: Settings) -> Path: ...

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome: ...


@dataclass(frozen=True)
class _Base:
    """Skip logic shared by every stage: a stage that has already produced is done."""

    name: StageName

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return settings.payloads / paper.study_id / self.name.value / "payload.json"

    def done(self, paper: Paper, settings: Settings) -> bool:
        return self.produces(paper, settings).is_file() and not settings.redo

    def _skip(self, paper: Paper, reason: str = "already produced") -> StageOutcome:
        return StageOutcome(
            stage=self.name, study_id=paper.study_id, skipped=True, reason=reason
        )

    def _write(self, paper: Paper, settings: Settings, payload: dict) -> Path:
        out = self.produces(paper, settings)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        return out


@dataclass(frozen=True)
class Tables(_Base):
    """Copy the parse manifest into Table records. Deterministic, and first."""

    name: StageName = StageName.tables

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        table_map = (
            json.loads(paper.table_map.read_text()) if paper.table_map.is_file() else {}
        )
        tables = [
            {
                "local_id": local,
                "table_number": meta.get("table_number"),
                "caption": meta.get("caption"),
                "footer": meta.get("footer"),
            }
            for local, meta in sorted(table_map.items())
        ]
        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            produced=(self._write(paper, settings, {"tables": tables}),),
        )


@dataclass(frozen=True)
class _ModelPass(_Base):
    """A stage whose work is one model call. Subclasses supply the prompt mode."""

    mode: str = ""

    def context(self, paper: Paper, settings: Settings) -> str:
        return ""

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        from pondie.extraction.passes.extract_record import build_prompt

        prompt, schema_name = build_prompt(
            paper.text.read_text(encoding="utf-8", errors="replace"),
            self.mode,
            settings.retrieve_evidence,
            self.context(paper, settings),
        )
        reply = caller(
            ModelCall(
                model=settings.model,
                prompt=prompt,
                schema_name=schema_name,
                max_output_tokens=settings.max_output_tokens,
                effort=settings.effort,
                attempts=settings.attempts,
            ),
            paper=paper.study_id,
            stage=self.name.value,
        )
        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            cost=reply.cost,
            produced=(self._write(paper, settings, reply.payload),),
        )


@dataclass(frozen=True)
class Demands(_ModelPass):
    """Analyses first: each declares the entities it needs, before any exist."""

    name: StageName = StageName.demands
    mode: str = "demands"

    def context(self, paper: Paper, settings: Settings) -> str:
        return paper.parse.read_text() if paper.parse.is_file() else ""


@dataclass(frozen=True)
class Satisfy(_ModelPass):
    """Build exactly the entities the demands pass asked for, and nothing else."""

    name: StageName = StageName.satisfy
    mode: str = "satisfy"

    def context(self, paper: Paper, settings: Settings) -> str:
        demands = Demands().produces(paper, settings)
        return demands.read_text() if demands.is_file() else ""


@dataclass(frozen=True)
class Evidence(_Base):
    """A supporting quote for every value the earlier passes emitted.

    45% of the pipeline's input tokens. Omitting it leaves a record that is structurally
    complete and unreviewable, which is a different thing from an incomplete one.

    Not a `_ModelPass`: this stage asks about the fields that already exist rather than about
    the paper, so its prompt is assembled from the earlier payloads. The assembly uses the
    same `iter_fields` / `describe` the record builder uses, and the call still goes through
    `Caller` -- one model boundary, so a test substitutes a fake here as anywhere else.
    """

    name: StageName = StageName.evidence
    batch: int = 60

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if not settings.retrieve_evidence:
            return self._skip(paper, "evidence disabled")
        if self.done(paper, settings):
            return self._skip(paper)
        from pondie.extraction.passes.add_evidence import SYSTEM, describe, iter_fields

        fields = [
            (path, field)
            for stage in (StageName.demands, StageName.satisfy)
            for payload in [settings.payloads / paper.study_id / stage.value / "payload.json"]
            if payload.is_file()
            for path, field in iter_fields(json.loads(payload.read_text()))
            if field.get("extraction_status") == "extracted"
        ]
        if not fields:
            return self._skip(paper, "no extracted values to warrant")

        text = paper.text.read_text(encoding="utf-8", errors="replace")
        quotes: dict[str, str] = {}
        cost = Cost()
        for start in range(0, len(fields), self.batch):
            chunk = fields[start : start + self.batch]
            listing = "\n".join(describe(path, field) for path, field in chunk)
            reply = caller(
                ModelCall(
                    model=settings.model,
                    prompt=f"{SYSTEM}\n\n# Paper\n\n{text}\n\n"
                    f"# Facts needing a supporting quote\n\n{listing}\n\n"
                    "Return the JSON object mapping each id to its quote now.",
                    max_output_tokens=settings.max_output_tokens,
                    effort=settings.effort,
                    attempts=settings.attempts,
                ),
                paper=paper.study_id,
                stage=self.name.value,
            )
            quotes.update({k: v for k, v in reply.payload.items() if isinstance(v, str)})
            cost = cost + reply.cost
        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            cost=cost,
            produced=(self._write(paper, settings, {"quotes": quotes}),),
        )


@dataclass(frozen=True)
class Build(_Base):
    """Merge the payloads, repair, resolve quotes to offsets, validate. No model."""

    name: StageName = StageName.build

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return settings.records / f"{paper.study_id}.extraction.json"

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        from pondie.extraction.passes.build_record import merge_payloads
        from pondie.extraction.passes.pipeline.repairs import Context, apply_all

        body = merge_payloads(settings.payloads / paper.study_id)
        stage1 = json.loads(paper.parse.read_text()) if paper.parse.is_file() else {}
        table_map = (
            json.loads(paper.table_map.read_text()) if paper.table_map.is_file() else {}
        )
        log = apply_all(body, Context(classes=_classes(), stage1=stage1, table_map=table_map))
        out = self.produces(paper, settings)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"study": body}, indent=1, ensure_ascii=False) + "\n")
        return StageOutcome(
            stage=self.name, study_id=paper.study_id, produced=(out,), reason="" if log else ""
        )


def _classes():
    import schema_utils

    from pondie.extraction.passes.extract_record import EXTRACTION_SCHEMA

    return schema_utils.load_imported_classes(EXTRACTION_SCHEMA)


#: Named orderings, so a workflow is a name rather than a remembered set of flags.
DEMAND_DRIVEN: tuple[Stage, ...] = (Tables(), Demands(), Satisfy(), Evidence(), Build())


def sequence(settings: Settings) -> tuple[Stage, ...]:
    """The stages this run will attempt, in order, filtered to those it asked for."""
    return tuple(s for s in DEMAND_DRIVEN if s.name in settings.stages)
