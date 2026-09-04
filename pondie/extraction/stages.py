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
import json
import os
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from pondie import schema
from pondie.extraction.llm import Caller, MalformedReply
from pondie.formats import text_index
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
from pondie.extraction.prompt import render
from pondie.formats import values

#: Written into every record's `extraction_metadata`, so a record says which pipeline made
#: it. Bump it when a change would make two records incomparable.
EXTRACTOR_VERSION = "pondie-1"


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
        """One file per stage, flat, beside its siblings.

        Flat because `builder.merge_payloads` globs `<payload_dir>/*.json` and merges
        what it finds -- a stage writing `<stage>/payload.json` produces a file the builder
        never sees, and the record comes out with metadata and an empty body. It is also
        the layout the evidence pass reads and writes back through, and the one every
        archived run on disk already uses.
        """
        return settings.payloads / paper.study_id / f"{self.name.value}.json"

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


def _manifest_value(text: str | None) -> dict:
    """A literal copied from the table manifest, in the shape the schema declares.

    `not_applicable` because there is no sentence to quote: the value came from the
    manifest, not from the paper's prose. `None` becomes `not_reported` -- a claim that the
    manifest carried no caption -- rather than an empty string, which reads as a blank one.
    """
    return values.wrap(text, source="reported", evidence="not_applicable")


@dataclass(frozen=True)
class Tables(_Base):
    """Copy the table manifest into Table records. No model, and first.

    `table_number`, `caption` and `footer` are literal strings in the manifest, so retyping
    them through a model can only introduce error. It runs first because the analyses pass
    is told the local_ids it assigns, and every `Analysis.tables` reference points at one.

    Omitting this stage is the regression that motivated writing it down: a rewritten
    pipeline dropped it and 155 of 156 records ended up with no tables declared while 1,076
    of 1,084 analyses referenced one. Direction scoring never noticed -- polarity needs the
    parse, not the Table entity -- so the fault stayed invisible until a coordinate query
    asked for the join.

    The manifest is read from the same flavour the text came from. Hardcoding
    `processed/pubget/tables.jsonl` finds nothing for a paper staged from `ace` or
    `elsevier`, and this corpus is mostly those.
    """

    name: StageName = StageName.tables

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        manifest = paper.text.parent / "tables.jsonl"
        if not manifest.is_file():
            return StageOutcome(
                stage=self.name,
                study_id=paper.study_id,
                produced=(self._write(paper, settings, {"tables": []}),),
                notes=(
                    f"no tables.jsonl beside the {paper.flavour.value} text; "
                    f"no Table records to copy",
                ),
            )

        tables, id_map = [], {}
        for index, line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            source = json.loads(line)
            # Keyed on the manifest's own table_id so the identity the staging wrote keeps
            # holding, and positionally only when it has none. `table_number` is not an
            # identifier: one paper in the corpus carries two tables numbered 1.
            local_id = str(source.get("table_id") or f"tbl{index}")
            id_map[str(source.get("table_id") or local_id)] = local_id
            metadata = source.get("metadata") or {}
            label = metadata.get("table_label") or (
                f"Table {source['table_number']}" if source.get("table_number") else None
            )
            tables.append(
                {
                    "local_id": local_id,
                    "table_number": _manifest_value(label),
                    "caption": _manifest_value(source.get("caption")),
                    "footer": _manifest_value(source.get("footer")),
                }
            )

        paper.table_map.parent.mkdir(parents=True, exist_ok=True)
        paper.table_map.write_text(json.dumps(id_map, indent=1) + "\n", encoding="utf-8")
        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            produced=(self._write(paper, settings, {"tables": tables}),),
            notes=(
                f"{len(tables)} Table record(s) copied from "
                f"{paper.flavour.value}/tables.jsonl (deterministic)",
            ),
        )


@dataclass(frozen=True)
class SignSplit(_Base):
    """Partition a parse that reports both signs, and withhold the reversed half.

    Runs before anything reads the parse, because it changes what the extraction pass is
    shown. A table holding effects of both signs is two contrasts and only one of them has
    prose in the paper: the positive half keeps the parsed name and is extracted, the
    negative half is marked `withhold` and rebuilt afterwards by the mirror repair.

    Its artefact is the parse it rewrote, so `done` cannot be the file existing -- it always
    does. `sign_split_applied` alone is not enough either: a corpus partitioned before the
    mirror existed carries that flag and still holds both halves as ordinary entries, which
    is exactly the case the mirror was built for. Both conditions are checked.
    """

    name: StageName = StageName.sign_split

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return paper.parse

    def done(self, paper: Paper, settings: Settings) -> bool:
        if settings.redo or not paper.parse.is_file():
            return False
        from pondie.extraction.corpus.tables import adopt_withholding

        parse = TableParse.load(paper.parse)
        if not parse.sign_split_applied:
            return False
        adopted = adopt_withholding(copy.deepcopy(parse.document.get("analyses") or []))
        return not adopted.notes

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        from pondie.extraction.corpus.tables import (
            adopt_withholding,
            split_opposite_signs,
        )

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
        #
        # Without this the pass accepted whatever came back. A payload of
        # `{"groups": [], "measures": [], ...}` is well formed, legally empty, and builds
        # and validates into a record about no study at all -- 2 runs in 10 of the best
        # configuration measured, and silent: `finish=stop`, nothing truncated, no validator
        # objection. `attempts` was reaching only the gateway's network retry.
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
                        # One network attempt per post-condition attempt: the two retries
                        # answer different questions, and multiplying them spends the
                        # budget twice.
                        attempts=1,
                    ),
                    paper=paper.study_id,
                    stage=self.name.value,
                )
            except MalformedReply as error:
                # A reply that will not parse is a rejected answer, not a broken run. It is
                # the same stochastic fault the post-condition loop already absorbs -- an
                # unbalanced bracket a few thousand characters in -- so it goes round again
                # carrying the note, and only a paper that cannot produce JSON in
                # `settings.attempts` tries fails.
                cost = cost + error.cost
                failures = [str(error)]
                parse_failures.append(str(error))
                continue
            cost = cost + reply.cost
            traces.append((reply.trace_id, reply.cache_status))
            # A reply can be cut off and still parse: the model closes what it has open and
            # the body is valid JSON describing half a paper. Nothing downstream can tell,
            # and the post-condition only objects when a list is empty rather than short.
            # `stop_reason` is the only witness, and it was written and never read.
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

        # Never parsed is not the same as parsed-but-imperfect. Writing `{}` here would
        # build a record about no study at all, which is the exact silent failure the
        # post-condition was added to stop.
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
        # Reported, never retried on: measured at 64% recall with 27 real findings against
        # one false alarm, which is a good signal to show a reviewer and a bad one to
        # re-ask a model about.
        outcome_notes += [f"suspect: {w}" for w in render.design_model_mismatch(payload)]

        return StageOutcome(
            stage=self.name,
            study_id=paper.study_id,
            cost=cost,
            traces=tuple(traces),
            notes=tuple(outcome_notes),
            produced=(self._write(paper, settings, payload),),
        )


@dataclass(frozen=True)
class Demands(_ModelPass):
    """Analyses first: each declares the entities it needs, before any exist."""

    name: StageName = StageName.demands
    mode: str = "demands"

    def context(self, paper: Paper, settings: Settings) -> str:
        """The stage-1 parse, rendered as instructions rather than dumped as JSON.

        This used to be `paper.parse.read_text()`. The file is a machine artefact, and
        handing it over raw drops everything `stage1_block` exists to say: the parse key
        each analysis must carry back in `source_table_analysis` -- the only exact join
        from an analysis to its coordinates -- the table `local_id`s that `Analysis.tables`
        is required to hold, which entries were sign-split and must not be re-merged, and
        the zero-foci rule, which is worth **+16 points** paired with this ordering and
        **-25** on its own. None of that survives being serialised back to JSON.
        """
        if not paper.parse.is_file():
            return ""
        table_ids = (
            json.loads(paper.table_map.read_text("utf-8")) if paper.table_map.is_file() else {}
        )
        return render.stage1_block(
            json.loads(paper.parse.read_text("utf-8")),
            table_ids,
            zero_foci_rule=settings.zero_foci_rule,
        )


@dataclass(frozen=True)
class Satisfy(_ModelPass):
    """Build exactly the entities the demands pass asked for, and nothing else."""

    name: StageName = StageName.satisfy
    mode: str = "satisfy"

    def declared(self, paper: Paper, settings: Settings) -> Sequence[Mapping[str, Any]]:
        demands = Demands().produces(paper, settings)
        if not demands.is_file():
            return ()
        return json.loads(demands.read_text("utf-8")).get("required_entities") or ()

    def context(self, paper: Paper, settings: Settings) -> str:
        """The shopping list the demands pass wrote, as this pass's contract.

        Raw JSON again before: `requirements_block` is what turns the declared entities
        into "emit one of each, with EXACTLY the local_id given", which is the whole point
        of asking the analyses first. Without it the pass is free to invent its own ids and
        every cell that points at one dangles.
        """
        demands = Demands().produces(paper, settings)
        if not demands.is_file():
            return ""
        return render.requirements_block(json.loads(demands.read_text("utf-8")))


@dataclass(frozen=True)
class Evidence(_Base):
    """A supporting quote for every value the earlier passes emitted.

    45% of the pipeline's input tokens. Omitting it leaves a record that is structurally
    complete and unreviewable, which is a different thing from an incomplete one.

    It does not write a payload of its own: `evidence` is a REQUIRED block on every
    `ExtractedValue`, so the blocks go onto the fields in the payloads the earlier stages
    wrote, and those payloads are rewritten in place. `noev/` is a copy taken first, so the
    stage can be re-run without re-running `satisfy`.

    Two locators, unioned. The model reads the whole paper -- handing it a retrieved
    shortlist instead was measured and cost 21 points -- and the retriever contributes a
    second span when it clears its own gate, at no marginal cost because it runs locally.
    The retriever is optional by design: a host without torch does the quote pass and says
    so, rather than taking the stage down.
    """

    name: StageName = StageName.evidence
    batch: int = 60

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return settings.payloads / paper.study_id / "noev"

    def done(self, paper: Paper, settings: Settings) -> bool:
        """Did the evidence get written, or did the stage merely start?

        `noev/` is the pre-evidence backup and is made BEFORE any evidence is, so its
        presence proves only that the stage began. Reading it as "done" cost seventeen
        papers their evidence: the stage died loading its reranker, the backup was already
        on disk, and every resume skipped it and built records with no evidence at all.

        Nor can the payloads answer it. `retrieve_evidence` also selects the prompt rule
        that asks the extraction passes to emit `evidence` inline, so on the default
        settings every extracted field already carries a block in exactly the shape this
        pass writes -- there is nothing in a payload that distinguishes what the model said
        from what this stage did. Any content check is answering a different question.

        The stage records this state in a file it writes last. The file exists only if the loop
        over every payload completed.
        """
        return not settings.redo and self._marker(paper, settings).is_file()

    def _marker(self, paper: Paper, settings: Settings) -> Path:
        """Written after the last payload, so it means finished and not started."""
        return self.produces(paper, settings) / ".complete"

    def _payloads(self, paper: Paper, settings: Settings) -> list[Path]:
        """The payloads a model pass wrote, which are the ones needing warrant.

        `tables.json` is excluded rather than merely skipped: its values are literals copied
        from the table manifest and are born `not_applicable`, so asking a model to quote
        them spends tokens to replace a correct claim with `not_found` -- which
        `values.py` documents as "a defect a reviewer should see". Three manufactured
        defects per table, per paper.
        """
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
        )

        targets = self._payloads(paper, settings)
        if not targets:
            return self._skip(paper, "no payloads to warrant")

        # Taken before anything is written, so a re-run starts from the payloads as the
        # extraction passes left them rather than from ones already carrying evidence.
        backup = self.produces(paper, settings)
        restoring = backup.is_dir()
        backup.mkdir(parents=True, exist_ok=True)
        self._marker(paper, settings).unlink(missing_ok=True)
        for saved in backup.glob("*.json") if restoring else ():
            shutil.copy(saved, settings.payloads / paper.study_id / saved.name)
        if not restoring:
            for target in targets:
                shutil.copy(target, backup / target.name)

        reranker, units = self._retriever(paper, settings)
        text = paper.text.read_text(encoding="utf-8", errors="replace")
        cost = Cost()
        traces: list[tuple[str, str]] = []
        totals = EvidenceCounts()

        for target in targets:
            payload = json.loads(target.read_text(encoding="utf-8"))
            wanted = [
                (path, field)
                for path, field in iter_fields(payload)
                if field.get("extraction_status") == "extracted"
            ]
            quotes: dict[str, str] = {}
            for begin in range(0, len(wanted), self.batch):
                chunk = wanted[begin : begin + self.batch]
                listing = "\n".join(describe(path, field) for path, field in chunk)
                reply = caller(
                    ModelCall(
                        model=settings.model,
                        # The instructions on the `system` half like every other pass. They
                        # were concatenated into the user turn, which left the largest pass
                        # in the pipeline -- 45% of input tokens -- sending nothing on the
                        # one half a prompt cache can hit.
                        system=SYSTEM,
                        prompt=f"# Paper\n\n{text}\n\n"
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
                traces.append((reply.trace_id, reply.cache_status))

            totals = totals + apply_evidence(payload, quotes, reranker, units)
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
                f"{totals.not_reported} not_reported, {totals.unioned} retrieved",
            ),
        )

    def _retriever(self, paper: Paper, settings: Settings):
        """The second locator, or nothing. An enhancement must not take the stage down."""
        if not settings.union:
            return None, ()
        from pondie.extraction.evidence import retrieval

        reranker = retrieval.load_reranker(device=settings.device_for(paper))
        if reranker is None:
            return None, ()
        text = paper.text.read_text(encoding="utf-8", errors="replace")
        return reranker, retrieval.sentence_units(text)


@dataclass(frozen=True)
class Build(_Base):
    """Merge the payloads, repair, resolve quotes to offsets, validate. No model.

    The validation is the part that had gone missing. `Validator` was constructed only by
    its own CLI and by tests, so a run printed `N paper(s), 0 failed` for records carrying
    dangling references, directions outside their vocabulary and spans that do not verify
    against the declared hash.

    Findings are notes, never a `reason`. A record is written either way: a defect is a
    field for a reviewer rather than a paper to discard, and treating one as a failure is
    what made five of sixteen papers read as lost when all sixteen had been built and
    scored.
    """

    name: StageName = StageName.build

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return settings.records / f"{paper.study_id}.extraction.json"

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if self.done(paper, settings):
            return self._skip(paper)
        from pondie.extraction.record.builder import build

        # `build` and not `merge_payloads`: merging assembles the body, and the record is
        # the body plus the three things only the builder can supply -- every quote
        # resolved to an offset into the normalized text, the `source_text_hash` those
        # offsets are relative to, and the `local_id` the schema requires on Study. A
        # record without them is not a partial record, it is an unvalidatable one.
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
        if report.failures:
            notes.append(f"{len(report.failures)} quote(s) did not resolve")
        if report.dangling:
            notes.append(f"{len(report.dangling)} cross-reference(s) need a human")
        notes += self._validate(record, paper, settings)
        return StageOutcome(
            stage=self.name, study_id=paper.study_id, produced=(out,), notes=tuple(notes)
        )

    def _validate(self, record: dict, paper: Paper, settings: Settings) -> list[str]:
        """Check the record against the schema, with the accepted findings suppressed.

        Never fatal, and never a `reason`: this reports, and the record is already written.
        The text is passed so spans are checked against the document they claim to address,
        which is the invariant the whole evidence design rests on.
        """
        from pondie.extraction.record import validate
        from pondie.schema import reader

        try:
            validator = validate.Validator(
                reader.load(schema.EXTRACTION),
                paper.text.read_text(encoding="utf-8", errors="replace"),
            )
            validator.check_record(record)
        except Exception as error:  # noqa: BLE001
            # The docstring above says never fatal and the code has to mean it. The record
            # is already written at this point, so a fault in the checker must not turn a
            # built paper into a failed one -- a validator bug reported as a paper failure
            # is the worst of both: the paper looks lost and the bug looks like the data.
            return [f"validation could not run ({type(error).__name__}: {error})"]
        notes = []
        if validator.errors:
            notes.append(
                f"{len(validator.errors)} validation error(s): "
                + "; ".join(validator.errors[:3])
            )
        if validator.warnings:
            notes.append(f"{len(validator.warnings)} validation warning(s)")
        return notes


#: Named orderings, so a workflow is a name rather than a remembered set of flags.
@dataclass(frozen=True)
class Repair(_Base):
    """Improve a built record, and report anything the attempt broke.

    Runs on the record `build` wrote, not on a payload, so its own artefact is the report:
    `done()` keys on that rather than on the record, which already exists by the time this
    starts.

    Both halves are on by default and they are independent, so a run without a GPU still
    resolves what the paper plainly answers. The local models are an optional dependency, and
    a missing one is a note rather than a failure: a record that could not be improved is the
    record `build` wrote, which is a worse outcome than repairing it and a much better one
    than losing the paper.
    """

    name: StageName = StageName.repair

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return settings.payloads / paper.study_id / "repair.json"

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        if not (settings.repair or settings.adjudicate):
            return self._skip(paper, "neither repair nor adjudicate was asked for")
        if self.done(paper, settings):
            return self._skip(paper)

        from pondie.extraction import repair as repair_pass
        from pondie.schema import reader

        record_path = settings.records / f"{paper.study_id}.extraction.json"
        if not record_path.is_file():
            return StageOutcome(stage=self.name, study_id=paper.study_id,
                                reason="no record to repair; build did not produce one")
        record = json.loads(record_path.read_text())
        # `text_index.load`, not `read_text`: every offset in the record is measured against
        # the normalized text and hashed into `source_text_hash`, so a span this pass writes
        # against the raw file would address a different string. `Paper.text` is a property
        # returning a Path -- calling it raised TypeError on the first line of real work,
        # and the test stub had a `text()` method, so nothing caught it.
        text, _digest, _sections = text_index.load(paper.text)

        proposer = checker = None
        notes: list[str] = []
        if settings.repair:
            # Imported here, not at module scope: the weights are an optional dependency,
            # and on by default only works if absent weights degrade instead of failing.
            try:
                from pondie.extraction.evidence.grounding import MiniCheck
                from pondie.extraction.recall import NuExtract

                # Visibility once, for the process, before either model loads: MiniCheck
                # places itself from `CUDA_VISIBLE_DEVICES` and nothing else, so a per-model
                # device would restrict the process and hide the proposer's card.
                if settings.visible_devices:
                    os.environ["CUDA_VISIBLE_DEVICES"] = settings.visible_devices
                checker = MiniCheck()
                proposer = NuExtract(device=settings.proposer_device)
            except Exception as error:  # noqa: BLE001 -- absence is expected, not exceptional
                notes.append(
                    f"no local repair models ({type(error).__name__}); "
                    f"install pondie[repair] for entity recall and grounding"
                )
        report = repair_pass.run(
            record, text, reader.load(schema.STORAGE), study_id=paper.study_id,
            proposer=proposer, checker=checker,
            caller=caller if settings.adjudicate else None,
            model=settings.model if settings.adjudicate else "",
        )
        record_path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
        self._write(paper, settings, {
            "written": report.written,
            "refused": [{"slot": r.slot, "why": r.why} for r in report.refused],
            "adjudicated": report.adjudicated,
            "introduced": report.introduced,
        })
        # A finding this pass introduced is a defect in the pass, not in the paper, and is
        # the one thing here worth failing on.
        return StageOutcome(stage=self.name, study_id=paper.study_id,
                            notes=tuple(notes + report.introduced))


DEMAND_DRIVEN: tuple[Stage, ...] = (
    Tables(),
    SignSplit(),
    Demands(),
    Satisfy(),
    Evidence(),
    Build(),
    Repair(),
)


def sequence(settings: Settings) -> tuple[Stage, ...]:
    """The stages this run will attempt, in order, filtered to those it asked for."""
    return tuple(s for s in DEMAND_DRIVEN if s.name in settings.stages)
