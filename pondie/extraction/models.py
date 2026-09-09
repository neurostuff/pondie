"""The extraction pipeline's inputs and outputs: what a caller passes in, what stages return.

Every seam in this pipeline used to be a command line and a parsed stdout, which meant a
stage's inputs were checked by whichever script happened to read them and its cost had to be
scraped back out of its own logging. These models are that contract written down, so a stage
declares what it needs and returns what it produced, and a caller that gets it wrong fails at
the boundary instead of three stages later.

What is NOT here: the record itself. Its shape is the LinkML schema in `study-schema`, which
generates the extraction schema, validates records and answers `multivalued` -- restating any
of that in pydantic would be a second source of truth that drifts. `pondie.records` reads it
through the schema. These models sit around the record, not inside it.

`extra="forbid"` throughout, deliberately: a misspelled field in a config is otherwise a
setting that silently does not apply.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pondie import paths

#: Re-exported: a render is a fact about the corpus layout, so `paths` owns it.
Flavour = paths.Flavour


class Strict(BaseModel):
    """Every contract here forbids unknown fields and is immutable once built."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StageName(str, Enum):
    tables = "tables"
    #: `sign_split`, not `split`: on a `str` Enum a member named `split` shadows
    #: `str.split` on the class. The wire value stays "split" -- it is what
    #: `--stages` takes and what a payload filename is named after.
    #: Coordinates stated in prose and in no table, appended to the parse so they share
    #: `source_table_analysis`'s address space. Before `sign_split`, which owns the
    #: document's final shape.
    prose_foci = "prose"
    sign_split = "split"
    demands = "demands"
    satisfy = "satisfy"
    #: Slot-level, and after `satisfy` because it finishes what that pass left open rather
    #: than deciding anything exists. Loops: a slot is settled when it holds a value or a
    #: reason there is none, so "are we done" is a fact about the record and not a guess.
    fill = "fill"
    evidence = "evidence"
    build = "build"
    #: After `build`, on the record rather than on a payload. It costs model calls for the
    #: proposal sweep and the adjudication, so a run that wants neither turns it off.
    repair = "repair"


class Paper(Strict):
    """One study and where its inputs live. The only filesystem knowledge a stage needs."""

    study_id: str = Field(min_length=1)
    root: Path
    flavour: Flavour = Flavour.pubget

    # The layout is `pondie.paths`'s to state. These are the pipeline's names for it, so a
    # stage says `paper.text` rather than assembling five path segments -- and there is one
    # place to change when the corpus moves.
    @property
    def text(self) -> Path:
        return paths.text(self.study_id, self.flavour, self.root)

    @property
    def parse(self) -> Path:
        return paths.stage1(self.study_id, self.root)

    @property
    def table_map(self) -> Path:
        return paths.table_map(self.study_id, self.root)

    def ready(self) -> bool:
        """Stage 1 is an input, not something this pipeline regenerates."""
        return self.text.is_file() and self.parse.is_file()

    @classmethod
    def best(cls, study_id: str, root: Path) -> "Paper":
        """The paper on the best flavour it actually has a text for.

        `Flavour` is declared best-first by how much of a paper's tables survive the render,
        so the first hit is the right one. Probing matters because the ranking is measured
        rather than cosmetic: over the 39,270-study corpus, pubget ships a table manifest
        for 12,390 of its 13,313 papers and elsevier for all 10,595 of its own, while ace
        ships none -- so taking ace when elsevier exists costs that paper its tables, and a
        locator searching a table-free flavour cannot find the sentence a group size came
        from.
        """
        for flavour in Flavour:
            candidate = cls(study_id=study_id, root=root, flavour=flavour)
            if candidate.text.is_file():
                return candidate
        raise FileNotFoundError(f"{study_id}: no text under {root / study_id}")


class Cost(Strict):
    """What a stage spent. Summed rather than tallied, so a run total is one addition."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    #: Written but never read back: every call reports `cache_write_tokens: 36423,
    #: cached_tokens: 0, cache_status: DISABLED`. Five stages send a near-identical prefix
    #: and each pays full input price. Recorded here because that is the only way the
    #: deviation ("one cached prefix", docs/pipeline-architecture.md D2) can be falsified
    #: from a run's own output rather than from a dashboard.
    cache_write_tokens: int = 0
    seconds: float = 0.0
    calls: int = 0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(**{f: getattr(self, f) + getattr(other, f) for f in type(self).model_fields})


class ModelCall(Strict):
    """One request to a language model. The only place a prompt becomes a network call."""

    model: str
    #: The user turn. Everything that varies per paper.
    prompt: str
    #: The instruction half, sent as its own message. Empty for a pass that has none.
    system: str = ""
    max_output_tokens: Annotated[int, Field(gt=0)] = 48_000
    effort: Literal["minimal", "low", "medium", "high"] = "low"
    attempts: Annotated[int, Field(ge=1)] = 3
    #: Ask the provider to constrain the reply to syntactically valid JSON.
    #:
    #: Measured, not assumed. Asked free-form for one paper's `satisfy` pass, the model
    #: returned a body that would not parse in 5 of 6 draws -- an unbalanced bracket some
    #: thousands of characters in, at a different place each time. The same prompt in JSON
    #: mode parsed 6 of 6. It is not a concurrency artefact: sequential draws failed at the
    #: same rate as parallel ones. Retrying cannot fix a 5-in-6 fault, which is why a run
    #: over 89 papers lost 25 of them with all three attempts spent.
    json_object: bool = True
    #: The provider's service tier, passed through the gateway. Empty leaves it unset, which
    #: is the provider's own default.
    #:
    #: `flex` trades latency for price on an offline run. Verified against the gateway rather
    #: than assumed: a reply to a `flex` request echoes `service_tier: "flex"`, a `default`
    #: one echoes `default`, and an invented tier is refused -- so the field reaches the
    #: provider instead of being swallowed by the proxy, which is the failure a parameter
    #: that is merely *accepted* hides.
    #:
    #: Off by default because it is slower and can be refused for capacity, and a stage that
    #: silently took longer would be indistinguishable from a stage that hung.
    service_tier: Literal["", "flex", "default", "priority"] = ""


class ModelReply(Strict):
    payload: dict
    cost: Cost
    stop_reason: str = ""
    #: The gateway's own id for this request. A header, so it is gone the moment the SDK
    #: parses the response -- which is why the caller reads the raw response first. Without
    #: it a run's own accounting cannot be joined to the gateway's.
    trace_id: str = ""
    #: What the gateway said about the cache for this call, verbatim.
    cache_status: str = ""


class Settings(Strict):
    """Everything the stages need that is not the paper."""

    payloads: Path
    records: Path
    model: str
    stages: tuple[StageName, ...] = tuple(StageName)
    effort: Literal["minimal", "low", "medium", "high"] = "low"
    max_output_tokens: Annotated[int, Field(gt=0)] = 48_000
    attempts: Annotated[int, Field(ge=1)] = 3
    retrieve_evidence: bool = True
    zero_foci_rule: bool = True
    #: Passed to every call this run makes. See `ModelCall.service_tier`; off by default.
    service_tier: Literal["", "flex", "default", "priority"] = ""
    #: Ask the model for the entities and links the extraction missed. On by default, and
    #: degrades rather than fails: with no caller the stage says so once and runs the
    #: deterministic half, which needs none.
    repair: bool = True
    #: Sweeps over the record. Two, because the first pass creates the entities the second
    #: can link to: a region proposed in pass one is a candidate an analysis can name in pass
    #: two, and with one pass those links have nothing to point at.
    repair_iterations: Annotated[int, Field(ge=1)] = 2
    #: How many rounds the `fill` stage may take. This loop has a real exit -- a slot is
    #: settled when it holds a value or an `unreported_reason`, so it stops when nothing is
    #: open rather than when nothing seems to be improving. The cap is
    #: for the case that does not converge: `undetermined` is the one answer that leaves a
    #: slot open, so a model that keeps giving it would otherwise be asked forever.
    fill_rounds: Annotated[int, Field(ge=1)] = 3
    #: Slots per call. The paper dominates a call whatever else is in it, so a round asks
    #: about as many open slots as it can rather than splitting them over calls.
    fill_batch: Annotated[int, Field(ge=1)] = 250

    #: Put the contradictions nothing else could settle to the extraction model, once per
    #: record, with the paper. Costs one call for a record that has one, and nothing for a
    #: record that does not -- 8 cases across 42 records measured.
    adjudicate: bool = True
    redo: bool = False

    @model_validator(mode="after")
    def _build_needs_its_inputs(self) -> "Settings":
        if StageName.build in self.stages and StageName.satisfy not in self.stages:
            existing = self.payloads
            if not existing.exists():
                raise ValueError(
                    "build without satisfy needs payloads on disk from an earlier run; "
                    f"{existing} does not exist"
                )
        return self


class StageOutcome(Strict):
    """What one stage did to one paper. `skipped` is a success, not a failure."""

    stage: StageName
    study_id: str
    produced: tuple[Path, ...] = ()
    cost: Cost = Cost()
    skipped: bool = False
    reason: str = ""
    #: What the stage observed but was not stopped by -- repairs that fired, quotes that
    #: did not resolve. A defect a reviewer should see is a note; `reason` is a failure.
    notes: tuple[str, ...] = ()
    #: One entry per model call: the gateway's trace id and what it said about the cache.
    #: Empty for a deterministic stage, which is the honest answer rather than a zero.
    traces: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        return not self.reason or self.skipped


class PaperOutcome(Strict):
    study_id: str
    outcomes: tuple[StageOutcome, ...] = ()

    @property
    def cost(self) -> Cost:
        total = Cost()
        for outcome in self.outcomes:
            total = total + outcome.cost
        return total

    @property
    def failed(self) -> StageOutcome | None:
        return next((o for o in self.outcomes if not o.ok), None)


class RunReport(Strict):
    """The whole run. A report a caller can assert on, rather than text to read."""

    papers: tuple[PaperOutcome, ...] = ()

    @property
    def cost(self) -> Cost:
        total = Cost()
        for paper in self.papers:
            total = total + paper.cost
        return total

    @property
    def failures(self) -> tuple[PaperOutcome, ...]:
        return tuple(p for p in self.papers if p.failed)

    #: A note whose presence means a paper was extracted without its tables. Matched on the
    #: text because that is what a stage has to say with; the alternative is a typed outcome
    #: field that every other stage would carry and none would set.
    MISSING_TABLES: ClassVar[str] = "no tables.jsonl"

    def starved(self) -> tuple[str, ...]:
        """Papers whose table manifest was absent, so no Table record could be copied."""
        return tuple(
            outcome.study_id
            for paper in self.papers
            for outcome in paper.outcomes
            if any(self.MISSING_TABLES in note for note in outcome.notes)
        )

    def summary(self) -> str:
        """The headline, and the one degradation a run can suffer while reporting success.

        A paper with no table manifest still extracts: the stage writes an empty list, says
        so in a note, and every later stage runs. What it produces is a record whose analyses
        reference tables it does not contain, and the stage that copies them exists because a
        rewrite once dropped it and left 155 of 156 records that way. That went unnoticed
        because nothing above the stage said anything. Counted here for the same reason the
        failures are: a summary that reports only what raised is not a summary of the run.
        """
        cost = self.cost
        line = (
            f"{len(self.papers)} paper(s), {len(self.failures)} failed · "
            f"{cost.input_tokens:,} in / {cost.output_tokens:,} out tokens "
            f"over {cost.calls} call(s)"
        )
        if missing := self.starved():
            shown = ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
            line += (
                f"\n  WARNING: {len(missing)} paper(s) had no table manifest and hold no "
                f"Table records ({shown}). Their analyses reference tables the record does "
                f"not contain; polarity coverage falls with them."
            )
        return line


class Prompt(Strict):
    """One rendered ask, in its two halves.

    Named, and not `tuple[str, str]`, because nothing at a call site said which half was
    which -- and one call site got it wrong. `stages` unpacked this as
    `(prompt, schema_name)`, so the `user` half went into a `ModelCall` field the caller
    ignores and **the model was never sent the paper**. A fake `Caller` in the tests records
    what it was asked and does not read it, so nothing caught it. Two named fields cannot be
    swapped by accident.

    `system` is the instructions and the schema; it is identical across papers, which is what
    a prompt cache needs. `user` is the conventions, the worked models, the context and the
    paper itself.
    """

    system: str = Field(min_length=1)
    user: str = Field(min_length=1)


class EvidenceCounts(Strict):
    """What one payload's evidence pass did, per outcome.

    Named because the five are different claims and a caller summing them would be summing
    apples: `unsupported` is a defect a reviewer should see, `not_reported` is a field the
    paper was silent on, and only `recovered` says the retriever earned its place.
    """

    filled: int = 0
    unsupported: int = 0
    not_reported: int = 0
    unioned: int = 0
    recovered: int = 0

    def __add__(self, other: "EvidenceCounts") -> "EvidenceCounts":
        return EvidenceCounts(
            **{f: getattr(self, f) + getattr(other, f) for f in type(self).model_fields}
        )


class SplitResult(Strict):
    """The analyses after a partition, and what the partition did.

    `notes` is not logging: a `FLAG` line records an analysis the rule declined to split and
    why, which is the difference between a parse the rule found nothing to do in and one it
    refused to touch. Discarding it makes those two look the same.
    """

    analyses: tuple[dict, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def withheld(self) -> int:
        """How many halves are kept from the model, to be rebuilt by arithmetic."""
        return sum(1 for entry in self.analyses if entry.get("withhold"))
