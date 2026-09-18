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

import copy
import functools
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from pondie import pipeline, schema
from pondie.extraction.llm import Caller, MalformedReply
from pondie.extraction.models import (
    Cost,
    EvidenceCounts,
    ModelCall,
    Paper,
    Settings,
    StageName,
    StageOutcome,
)
from pondie.extraction.parse import TableParse
from pondie.extraction.sign_split import adopt_withholding, split_opposite_signs
from pondie.extraction.record.ids import table_local_id
from pondie.extraction.prompt import preprocess, render, worked
from pondie.formats import table_parse, text_index, values
from pondie.schema import reader

#: Written into every record's `extraction_metadata`, so a record says which pipeline made
#: it. Bump it when a change would make two records incomparable.
EXTRACTOR_VERSION = "pondie-1"


@runtime_checkable
class Stage(Protocol):
    name: StageName

    def produces(self, paper: Paper, settings: Settings) -> Path: ...

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome: ...

    def depends_on(self, paper: Paper, settings: Settings) -> Mapping[str, Any]:
        """What the stage reads, and subsequently, what invalidates their cache."""
        ...


@dataclass(frozen=True)
class _Base:
    """Logic shared by every stage"""

    name: StageName

    def produces(self, paper: Paper, settings: Settings) -> Path:
        """Each Stage outputs a single file."""
        return settings.payloads / paper.study_id / f"{self.name.value}.json"

    #: The stages whose output this one reads. Their digests go into this stage's, so a
    #: re-run cascades down to the rest of the stages.
    reads: tuple[StageName, ...] = ()

    #: Whether the stage sends the paper to a model. A model pass depends on which model and
    #: at what effort; `tables` copies a manifest and does not care.
    asks_a_model: bool = False

    #: Whether the stage reads the stage-1 parse.
    reads_the_parse: bool = False

    def depends_on(self, paper: Paper, settings: Settings) -> dict[str, Any]:
        """Every input that would change this stage's answer."""

        parts: dict[str, Any] = {
            "text": (
                text_index.text_hash(
                    text_index.normalize(paper.text.read_text(encoding="utf-8", errors="replace"))
                )
                if paper.text.is_file()
                else ""
            ),
            "flavour": paper.flavour.value,
        }
        if self.reads_the_parse:
            parts["parse"] = (
                text_index.text_hash(paper.parse.read_text(encoding="utf-8", errors="replace"))
                if paper.parse.is_file()
                else ""
            )
        if self.asks_a_model:
            parts |= {
                "model": settings.model,
                "effort": settings.effort,
                "service_tier": settings.service_tier,
                "prompt": prompt_digest(),
            }
        for upstream in self.reads:
            output = settings.payloads / paper.study_id / f"{upstream.value}.json"
            stamp = pipeline.Stamp.read(output)
            parts[f"after:{upstream.value}"] = stamp.digest if stamp else ""
        return parts

    def done(self, paper: Paper, settings: Settings) -> bool:
        return pipeline.fresh(self.as_step(settings), paper, redo=settings.redo) is not None

    def as_step(self, settings: Settings, caller: Caller = None) -> "pipeline.Step[Paper]":
        """This stage as something the scheduler can run."""
        return pipeline.Step(
            name=self.name.value,
            produces=lambda paper: self.produces(paper, settings),
            depends_on=lambda paper: self.depends_on(paper, settings),
            run=lambda paper: self.run(paper, settings, caller),
        )

    def _skip(self, paper: Paper, reason: str = "already produced") -> StageOutcome:
        return StageOutcome(stage=self.name, study_id=paper.study_id, skipped=True, reason=reason)

    def _write(self, paper: Paper, settings: Settings, payload: dict) -> Path:
        out = self.produces(paper, settings)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        return out


@functools.lru_cache(maxsize=1)
def prompt_digest() -> str:
    """A digest of everything that decides what the model is asked.
    
    Allows us to track prompt changes that would invalidate the cache.
    """
    sources = [
        Path(render.__file__),
        Path(worked.__file__),
        schema.ROOT / "extraction-readme.md",
        schema.ROOT / "representing-models.md",
        schema.ROOT / "extraction-deviations.yaml",
        *sorted((schema.ROOT / "neuroimaging-study-extraction").glob("*.yaml")),
    ]
    return pipeline.digest_of(
        {
            str(path.name): text_index.text_hash(path.read_text(encoding="utf-8"))
            for path in sources
            if path.is_file()
        }
    )


def _manifest_value(text: str | None) -> dict:
    """A literal copied from a table source wrapped in the evidence class
    from the extraction schema.
    """
    return values.wrap(text or None, source="reported", evidence="not_applicable")


@dataclass(frozen=True)
class Tables(_Base):
    """Fill Table records deterministically. No model, and first.

    `table_number`, `caption` and `footer` are literal strings in the manifest, so retyping
    them through a model can only introduce error. It runs first because the analyses pass
    is told the local_ids and every `Analysis.tables` reference points at one.
    """

    name: StageName = StageName.tables
    reads_the_parse: bool = True

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)

        sources = list(
            table_parse.read_manifest(paper.study_dir, paper.flavour.value).values()
        )
        origin = f"{paper.flavour.value}/tables.jsonl"
        if not sources:
            # On emptiness as well as absence. A manifest with no rows makes the same claim
            # an absent one does, and the fallback is what keeps the prompt's table headings
            # backed by a declared entity either way.
            sources = TableParse.read(paper.parse).source_tables()
            origin = "the stage-1 parse"

        tables, id_map, taken = [], {}, set()
        for index, source in enumerate(sources, start=1):
            local_id = table_local_id(source["table_number"], source["table_label"], taken)
            if not local_id:
                # No printed number and no label: positional, and only here. An id that
                # moves when a table is added upstream is the thing `table_local_id`
                # avoids, so it is the last resort rather than the default.
                local_id = f"tbl{index}"
                while local_id in taken:
                    index += 1
                    local_id = f"tbl{index}"
            taken.add(local_id)
            id_map[str(source["table_id"] or local_id)] = local_id
            tables.append(
                {
                    "local_id": local_id,
                    "table_number": _manifest_value(
                        source["table_label"]
                        or (f"Table {source['table_number']}" if source["table_number"] else None)
                    ),
                    "caption": _manifest_value(source["caption"]),
                    "footer": _manifest_value(source["footer"]),
                }
            )

        paper.table_map.parent.mkdir(parents=True, exist_ok=True)
        paper.table_map.write_text(json.dumps(id_map, indent=1) + "\n", encoding="utf-8")
        note = f"{len(tables)} Table record(s) from {origin} (deterministic)"
        if not tables:
            note = (
                f"no tables.jsonl beside the {paper.flavour.value} text and no table in the "
                f"stage-1 parse; no Table records, so no `Analysis.tables` target exists"
            )
        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            produced=(self._write(paper, settings, {"tables": tables}),),
            notes=(note,),
        )


@dataclass(frozen=True)
class ProseFoci(_Base):
    """Append coordinates the paper states in prose and not in tables.

    Note: this could still miss legitimate analyses that report the same coordinates.
    """

    name: StageName = StageName.prose_foci

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return paper.parse

    def done(self, paper: Paper, settings: Settings) -> bool:
        if settings.redo or not paper.parse.is_file():
            return False
        return bool(TableParse.load(paper.parse).document.get("prose_foci_applied"))

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        if not paper.parse.is_file():
            return self._skip(paper, "no parse to append to")
        parse = TableParse.load(paper.parse)
        before = parse.document.get("analyses") or []
        entries = preprocess.prose_parse_entries(
            paper.text.read_text(encoding="utf-8", errors="replace"),
            parse.coordinates,
        )
        parse.document["analyses"] = [*before, *entries]
        parse.document["prose_foci_applied"] = True
        parse.save()
        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            produced=(paper.parse,),
            notes=(
                f"{len(entries)} prose coordinate sentence(s) appended "
                f"to {len(before)} parsed",
            ),
        )


@dataclass(frozen=True)
class SignSplit(_Base):
    """Split a parse that reports both positive and negative statistics into inverse analyses.

    Runs before anything reads the parse, because it changes what the extraction pass is
    shown. A table holding effects of both signs is two contrasts and only one of them has
    prose in the paper: the positive half keeps the parsed name and is extracted, the
    negative half is marked `withhold` and rebuilt afterwards by the mirror repair.
    """

    name: StageName = StageName.sign_split

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return paper.parse

    def done(self, paper: Paper, settings: Settings) -> bool:
        if settings.redo or not paper.parse.is_file():
            return False
        parse = TableParse.load(paper.parse)
        if not parse.sign_split_applied:
            return False
        adopted = adopt_withholding(copy.deepcopy(parse.document.get("analyses") or []))
        return not adopted.notes

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        parse = TableParse.load(paper.parse)
        before = parse.document.get("analyses") or []
        split = split_opposite_signs(before)
        # A corpus partitioned before the mirror existed holds both halves as ordinary
        # entries. Re-splitting cannot reach them -- each part already holds one sign --
        # so the pair is converted from what the parts themselves record.
        adopted = adopt_withholding(list(split.analyses))
        parse.replace_analyses(list(adopted.analyses))
        parse.save()
        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            produced=(paper.parse,),
            notes=(
                *split.notes,
                f"{len(before)} -> {len(adopted.analyses)} analyses, "
                f"{adopted.withheld} withheld",
            ),
        )


@dataclass(frozen=True)
class _ModelPass(_Base):
    """A stage whose work is one model call. Subclasses supply the prompt mode."""

    mode: str = ""
    #: The deterministic repairs whose inputs this pass's own payload holds, run on it
    #: before it is written.
    repair_stage: tuple[str, ...] = ()

    def context(self, paper: Paper, settings: Settings) -> str:
        return ""

    def declared(self, paper: Paper, settings: Settings) -> Sequence[Mapping[str, Any]]:
        """The entities this pass was asked to produce, for the post-condition to check."""
        return ()

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        prompt = render.build_prompt(
            paper.text.read_text(encoding="utf-8", errors="replace"),
            self.mode,
            settings.retrieve_evidence,
            self.context(paper, settings),
        )
        declared = self.declared(paper, settings)

        # Retry names the fault rather than resampling blindly. The failure is stochastic --
        # the same prompt succeeds on the next draw most of the time -- but a model told what
        # was wrong with its last answer does better than one asked the same question twice.
        
        cost, traces, notes = Cost(), [], []
        payload: dict = {}
        failures: list[str] = []
        parse_failures: list[str] = []
        truncation_notes: list[str] = []
        parsed = False
        for attempt in range(1, settings.attempts + 1):
            user = (
                prompt.user
                if attempt == 1
                else prompt.user
                + render.RETRY_NOTE.format(
                    failures="\n".join(f"- {failure}" for failure in failures)
                )
            )
            try:
                reply = caller(
                    ModelCall(
                        model=settings.model,
                        system=prompt.system,
                        prompt=user,
                        max_output_tokens=settings.max_output_tokens,
                        effort=settings.effort,
                        service_tier=settings.service_tier,
                        attempts=1,
                    ),
                    paper=paper.study_id,
                    stage=self.name.value,
                )
            except MalformedReply as error:
                # A malformed reply is a rejected answer.
                cost = cost + error.cost
                failures = [str(error)]
                parse_failures.append(str(error))
                continue
            cost = cost + reply.cost
            traces.append((reply.trace_id, reply.cache_status))
            # A reply can be cut off and still parse.
            if reply.stop_reason and reply.stop_reason != "stop":
                truncation_notes.append(
                    f"attempt {attempt} finished on {reply.stop_reason!r}, not 'stop'; "
                    f"the payload may be cut short"
                )
            # Hoisting first: an entity list nested under `study` is otherwise shadowed by
            # an empty top-level sibling, and the post-condition would reject a good answer.
            payload, notes = render.normalize(reply.payload, self.mode)
            parsed = True
            failures = render.postcondition_failures(payload, self.mode, declared)
            if not failures:
                break

        # Never parsed is not the same as parsed-but-imperfect.
        if not parsed:
            raise MalformedReply(
                f"{self.name.value} for {paper.study_id}: no valid JSON in "
                f"{settings.attempts} attempt(s): " + "; ".join(parse_failures),
                body="",
                cost=cost,
            )

        outcome_notes = list(notes) + truncation_notes
        if parse_failures:
            outcome_notes += [f"retried after a malformed reply: {f}" for f in parse_failures]
        if failures:
            outcome_notes.append(
                f"post-condition still failing after {settings.attempts} attempt(s): "
                + "; ".join(failures)
            )

        outcome_notes += [f"suspect: {w}" for w in render.design_model_mismatch(payload)]

        outcome_notes += self._repair(paper, payload)

        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            cost=cost,
            traces=tuple(traces),
            notes=tuple(outcome_notes),
            produced=(self._write(paper, settings, payload),),
        )

    def _repair(self, paper: Paper, payload: dict) -> list[str]:
        """The payload-local repairs, run where their inputs are rather than at the merge."""
        if not self.repair_stage:
            return []
        from pondie.extraction.record import fix

        log = fix.apply_all(
            payload,
            fix.Context(
                schema=reader.load(schema.STORAGE),
                stage1=paper.parse if paper.parse.is_file() else None,
                table_map=paper.table_map if paper.table_map.is_file() else None,
            ),
            stage=self.repair_stage,
        )
        return [f"repaired {name}: {len(lines)}" for name, lines in log.entries if lines]


@dataclass(frozen=True)
class Demands(_ModelPass):
    """Analyses from the table/prose declare what entities are needed to describe them."""

    name: StageName = StageName.demands
    #: `context` prints the `table_map` that `Tables` wrote, so a change to the Table
    #: local_ids has to re-ask this pass. Left empty, a resumed run would pair fresh table
    #: ids with a stale analyses payload and every `Analysis.tables` reference would dangle
    #: -- the failure this dependency exists to prevent, not a cache nicety.
    reads: tuple[StageName, ...] = (StageName.tables,)
    asks_a_model: bool = True
    mode: str = "demands"
    repair_stage: tuple[str, ...] = ("shape", "demands")

    def context(self, paper: Paper, settings: Settings) -> str:
        """Reading the Table->Analyses parse."""
        parse = TableParse.read(paper.parse)
        block = ""
        if parse.document:
            table_ids = (
                json.loads(paper.table_map.read_text("utf-8")) if paper.table_map.is_file() else {}
            )
            block = render.stage1_block(
                parse.document,
                table_ids,
                zero_foci_rule=settings.zero_foci_rule,
            )
        # Offered as proposals the pass confirms or drops, and offered whether or not a
        # parse exists: of the 88 cue_reactivity papers stating a Results coordinate no
        # table carries, 49 have no parsed table at all, and returning early on a missing
        # parse would withhold the list from exactly those.
        return block + preprocess.prose_coordinate_block(
            paper.text.read_text(encoding="utf-8", errors="replace"),
            parse.coordinates,
        )


@dataclass(frozen=True)
class Satisfy(_ModelPass):
    """Build the entities the demands pass asked for."""

    name: StageName = StageName.satisfy
    reads: tuple[StageName, ...] = (StageName.demands,)
    asks_a_model: bool = True
    mode: str = "satisfy"
    repair_stage: tuple[str, ...] = ("shape", "satisfy")

    def declared(self, paper: Paper, settings: Settings) -> Sequence[Mapping[str, Any]]:
        demands = Demands().produces(paper, settings)
        if not demands.is_file():
            return ()
        return json.loads(demands.read_text("utf-8")).get("required_entities") or ()

    def context(self, paper: Paper, settings: Settings) -> str:
        """The shopping list the demands pass wrote, as this pass's contract."""
        demands = Demands().produces(paper, settings)
        if not demands.is_file():
            return ""
        return render.requirements_block(json.loads(demands.read_text("utf-8")))


@dataclass(frozen=True)
class Fill(_Base):
    """Fill the slots that are still open."""

    name: StageName = StageName.fill
    reads: tuple[StageName, ...] = (StageName.demands, StageName.satisfy)
    asks_a_model: bool = True

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return settings.payloads / paper.study_id / "fill.json"

    def _targets(self, paper: Paper, settings: Settings) -> list[Path]:
        """The payloads holding entities."""
        directory = settings.payloads / paper.study_id
        # tables are filled deterministically
        skip = {"tables.json", "fill.json"}
        return sorted(p for p in directory.glob("*.json") if p.name not in skip)

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        from pondie.extraction.prompt import fill as slots

        targets = self._targets(paper, settings)
        if not targets:
            return self._skip(paper, "no payloads to fill")

        sch = reader.load(render.EXTRACTION_SCHEMA)
        text = paper.text.read_text(encoding="utf-8", errors="replace")
        cost, traces, notes = Cost(), [], []
        opened = closed = 0

        for target in targets:
            payload = json.loads(target.read_text(encoding="utf-8"))
            first = len(slots.unsettled(payload, sch))
            opened += first
            for round_number in range(1, settings.fill_rounds + 1):
                rows = slots.unsettled(payload, sch)
                if not rows:
                    notes.append(f"{target.name} round {round_number}: nothing open, done")
                    break
                batch = rows[: settings.fill_batch]
                ids = [r["id"] for r in batch]
                try:
                    reply = caller(
                        ModelCall(
                            model=settings.model,
                            system=slots.SYSTEM,
                            prompt=f"# Paper\n\n{text}\n\n{slots.block(batch)}\n"
                            "Return the JSON object now.",
                            max_output_tokens=settings.max_output_tokens,
                            effort=settings.effort,
                            service_tier=settings.service_tier,
                            attempts=settings.attempts,
                        ),
                        paper=paper.study_id,
                        stage=f"{self.name.value}{round_number}",
                    )
                except Exception as error:  # noqa: BLE001 -- a round must not lose the record
                    notes.append(
                        f"{target.name} round {round_number} skipped: " f"{type(error).__name__}"
                    )
                    break
                cost = cost + reply.cost
                traces.append((reply.trace_id, reply.cache_status))
                filled, reasoned, dropped = slots.apply_fill(payload, reply.payload, ids)
                notes.append(
                    f"{target.name} round {round_number}: {len(batch)} asked, "
                    f"{filled} valued, {reasoned} explained, {dropped} discarded"
                )
                if reply.stop_reason and reply.stop_reason != "stop":
                    notes.append(
                        f"{target.name} round {round_number} stopped on " f"{reply.stop_reason}"
                    )
                from pondie.extraction.record import fix as _fix

                _fix.apply_all(
                    payload,
                    _fix.Context(schema=reader.load(schema.STORAGE)),
                    stage=_fix.AFTER_FILL,
                )
                target.write_text(
                    json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                if not (filled or reasoned):
                    # The round settled nothing, so the next one asks the same model the
                    # same question about the same paper.
                    notes.append(
                        f"{target.name} round {round_number}: nothing settled, " f"stopped"
                    )
                    break
            closed += first - len(slots.unsettled(payload, sch))

        self._write(paper, settings, {"opened": opened, "closed": closed})
        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            cost=cost,
            traces=tuple(traces),
            produced=tuple(targets),
            notes=(f"{closed} of {opened} open slot(s) settled", *notes),
        )


@dataclass(frozen=True)
class Evidence(_Base):
    """A supporting quote for every value the earlier passes emitted."""

    name: StageName = StageName.evidence
    reads: tuple[StageName, ...] = (StageName.demands, StageName.satisfy, StageName.fill)
    asks_a_model: bool = True

    batch: int = 200

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return settings.payloads / paper.study_id / "noev"

    def done(self, paper: Paper, settings: Settings) -> bool:
        """Did the evidence get written?

        `noev/` is the pre-evidence backup and is made BEFORE any evidence is, so its
        presence only proves that the stage began.
        """
        return not settings.redo and self._marker(paper, settings).is_file()

    def _marker(self, paper: Paper, settings: Settings) -> Path:
        """Written after the last payload, so it means finished and not started."""
        return self.produces(paper, settings) / ".complete"

    def _payloads(self, paper: Paper, settings: Settings) -> list[Path]:
        """Write the evidence payloads."""
        directory = settings.payloads / paper.study_id
        deterministic = {"aliases.json", f"{StageName.tables.value}.json"}
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.glob("*.json") if p.name not in deterministic)

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if not settings.retrieve_evidence:
            return self._skip(paper, "evidence disabled")
        if self.done(paper, settings):
            return self._skip(paper)
        from pondie.extraction.evidence.quote import (
            SYSTEM,
            apply_evidence,
            describe,
            iter_fields,
            literal_quotes,
        )

        targets = self._payloads(paper, settings)
        if not targets:
            return self._skip(paper, "no payloads to warrant")

        backup = self.produces(paper, settings)
        restoring = backup.is_dir()
        backup.mkdir(parents=True, exist_ok=True)
        self._marker(paper, settings).unlink(missing_ok=True)
        for saved in backup.glob("*.json") if restoring else ():
            shutil.copy(saved, settings.payloads / paper.study_id / saved.name)
        if not restoring:
            for target in targets:
                shutil.copy(target, backup / target.name)

        text = paper.text.read_text(encoding="utf-8", errors="replace")
        cost = Cost()
        traces: list[tuple[str, str]] = []
        truncated: list[str] = []
        settled = 0
        totals = EvidenceCounts()

        for target in targets:
            payload = json.loads(target.read_text(encoding="utf-8"))
            # Try finding a unique quote first.
            # If the value only occurs once in the paper,
            # then that's likely the evidence for that value.
            quotes: dict[str, str] = literal_quotes(payload, text)
            literal = set(quotes)
            wanted = [
                (path, field)
                for path, field in iter_fields(payload)
                if field.get("extraction_status") == "extracted" and path not in quotes
            ]
            settled += len(quotes)
            for begin in range(0, len(wanted), self.batch):
                chunk = wanted[begin : begin + self.batch]
                listing = "\n".join(describe(path, field) for path, field in chunk)
                reply = caller(
                    ModelCall(
                        model=settings.model,
                        system=SYSTEM,
                        prompt=f"# Paper\n\n{text}\n\n"
                        f"# Facts needing a supporting quote\n\n{listing}\n\n"
                        "Return the JSON object mapping each id to its quote now.",
                        max_output_tokens=settings.max_output_tokens,
                        effort=settings.effort,
                        service_tier=settings.service_tier,
                        attempts=settings.attempts,
                    ),
                    paper=paper.study_id,
                    stage=self.name.value,
                )
                returned = {k: v for k, v in reply.payload.items() if isinstance(v, str)}
                if reply.stop_reason and reply.stop_reason != "stop":
                    truncated.append(
                        f"{target.name} batch {begin // self.batch + 1} finished on "
                        f"{reply.stop_reason!r} with {len(returned)}/{len(chunk)} quotes"
                    )
                quotes.update(returned)
                cost = cost + reply.cost
                traces.append((reply.trace_id, reply.cache_status))

            totals = totals + apply_evidence(payload, quotes, literal=frozenset(literal))
            target.write_text(
                json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )

        self._marker(paper, settings).write_text("", encoding="utf-8")
        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            cost=cost,
            traces=tuple(traces),
            produced=tuple(targets),
            notes=(
                f"{totals.filled} warranted, {totals.unsupported} unsupported, "
                f"{totals.not_reported} not_reported",
                f"{settled} field(s) settled by literal match, not asked of the model",
                *(f"truncated: {t}" for t in truncated),
            ),
        )


@dataclass(frozen=True)
class Build(_Base):
    """Merge the payloads, repair, resolve quotes to offsets, validate.
    """

    name: StageName = StageName.build
    reads: tuple[StageName, ...] = (
        StageName.demands,
        StageName.satisfy,
        StageName.fill,
        StageName.evidence,
    )
    asks_a_model: bool = False

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return settings.records / f"{paper.study_id}.extraction.json"

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        from pondie.extraction.record.builder import build

        record, report = build(
            paper.study_id,
            paper.text,
            settings.payloads / paper.study_id,
            extractor_model=settings.model,
            extractor_version=EXTRACTOR_VERSION,
            extraction_date=date.today().isoformat(),
            stage1=paper.parse if paper.parse.is_file() else None,
            table_map=paper.table_map if paper.table_map.is_file() else None,
        )
        out = self.produces(paper, settings)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")

        notes = [f"repairs: {', '.join(report.repair_log.fired()) or 'none fired'}"]
        if report.warrant.unresolved:
            notes.append(
                f"{len(report.warrant.unresolved)} quote(s) did not resolve, leaving "
                f"{report.warrant.unlocated} field(s) unevidenced despite a quote"
            )
        if report.warrant.case_insensitive:
            notes.append(
                f"{report.warrant.case_insensitive} span(s) placed only by ignoring case")
        if report.dangling:
            notes.append(f"{len(report.dangling)} cross-reference(s) need a human")
        notes += self._validate(record, paper, settings)
        return StageOutcome(
            stage=self.name, study_id=paper.study_id, produced=(out,), notes=tuple(notes)
        )

    def _validate(self, record: dict, paper: Paper, settings: Settings) -> list[str]:
        """Check the record against the schema."""
        from pondie.extraction.record import validate
        from pondie.schema import reader

        try:
            validator = validate.Validator(
                reader.load(schema.EXTRACTION),
                paper.text.read_text(encoding="utf-8", errors="replace"),
            )
            validator.check_record(record)
        except Exception as error:  # noqa: BLE001
            return [f"validation could not run ({type(error).__name__}: {error})"]
        notes = []
        if validator.errors:
            notes.append(
                f"{len(validator.errors)} validation error(s): " + "; ".join(validator.errors[:3])
            )
        if validator.warnings:
            notes.append(f"{len(validator.warnings)} validation warning(s)")
        return notes


@dataclass(frozen=True)
class Repair(_Base):
    """Improve a built record, and report anything the attempt broke."""

    name: StageName = StageName.repair
    reads: tuple[StageName, ...] = (StageName.build,)
    asks_a_model: bool = True

    def produces(self, paper: Paper, settings: Settings) -> Path:
        """Beside the payloads, not among them."""
        return settings.payloads.parent / "repairs" / f"{paper.study_id}.json"

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if not (settings.repair or settings.adjudicate):
            return self._skip(paper, "neither repair nor adjudicate was asked for")
        if self.done(paper, settings):
            return self._skip(paper)

        from pondie.extraction import repair as repair_pass
        from pondie.extraction.record.validate import EXTRACTION_SCHEMA
        from pondie.schema import reader

        record_path = settings.records / f"{paper.study_id}.extraction.json"
        if not record_path.is_file():
            return StageOutcome(
                stage=self.name,
                study_id=paper.study_id,
                reason="no record to repair; build did not produce one",
            )
        record = json.loads(record_path.read_text())
        kept = settings.records.parent / "unrepaired" / f"{paper.study_id}.extraction.json"
        if not kept.is_file():
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_text(record_path.read_text(), encoding="utf-8")
        text, _digest, _sections = text_index.load(paper.text)

        proposer = None
        notes: list[str] = []
        if settings.repair:
            if caller is None:
                notes.append("no caller for the proposer; repairing deterministically")
            else:
                from pondie.extraction.repair.propose_with_extractor import ModelProposer

                proposer = ModelProposer(
                    caller,
                    settings.model,
                    study_id=paper.study_id,
                    service_tier=settings.service_tier,
                    effort=settings.effort,
                )
                notes.append(f"proposer: {settings.model}")

        report = repair_pass.run(
            record,
            text,
            reader.load(EXTRACTION_SCHEMA),
            study_id=paper.study_id,
            proposer=proposer,
            caller=caller if settings.adjudicate else None,
            model=settings.model if settings.adjudicate else "",
            service_tier=settings.service_tier,
            iterations=settings.repair_iterations,
        )
        record_path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
        out = self.produces(paper, settings)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "written": report.written,
                    "refused": [{"slot": r.slot, "why": r.why} for r in report.refused],
                    "adjudicated": report.adjudicated,
                    "introduced": report.introduced,
                },
                indent=1,
            )
            + "\n"
        )
        spent = report.cost
        proposed = getattr(proposer, "cost", None)
        if proposed is not None:
            spent = proposed if spent is None else spent + proposed
        spend = {"cost": spent} if spent is not None else {}
        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            traces=report.traces,
            notes=tuple(notes + report.introduced),
            **spend,
        )


DEMAND_DRIVEN: tuple[Stage, ...] = (
    Tables(),
    ProseFoci(),
    SignSplit(),
    Demands(),
    Satisfy(),
    Fill(),
    Evidence(),
    Build(),
    Repair(),
)


def sequence(settings: Settings) -> tuple[Stage, ...]:
    """The stages this run will attempt, in order."""
    return tuple(s for s in DEMAND_DRIVEN if s.name in settings.stages)
