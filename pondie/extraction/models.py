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
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    """Every contract here forbids unknown fields and is immutable once built."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Flavour(str, Enum):
    """Which render of a paper the text came from. The pipeline reads one per paper."""

    pubget = "pubget"
    ace = "ace"
    elsevier = "elsevier"
    local = "local"


class StageName(str, Enum):
    tables = "tables"
    demands = "demands"
    satisfy = "satisfy"
    evidence = "evidence"
    build = "build"


class Workflow(str, Enum):
    """Which pass decides the entities exist.

    `demand_driven` lets the analyses declare their terms first. The alternative asks an
    entity pass to guess the inventory, and a cell cannot be righter than the term it points
    at -- measured: asked to guess, that pass modelled a crossover's condition as a
    continuous covariate.
    """

    demand_driven = "demand_driven"
    entity_first = "entity_first"


class Paper(Strict):
    """One study and where its inputs live. The only filesystem knowledge a stage needs."""

    study_id: str = Field(min_length=1)
    root: Path
    flavour: Flavour = Flavour.pubget

    @property
    def text(self) -> Path:
        return self.root / self.study_id / "processed" / self.flavour.value / "text.txt"

    @property
    def parse(self) -> Path:
        return self.root / self.study_id / "stage1" / "analyses.json"

    @property
    def table_map(self) -> Path:
        return self.root / self.study_id / "stage1" / "table-map.json"

    def ready(self) -> bool:
        """Stage 1 is an input, not something this pipeline regenerates."""
        return self.text.is_file() and self.parse.is_file()


class Cost(Strict):
    """What a stage spent. Summed rather than tallied, so a run total is one addition."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    seconds: float = 0.0
    calls: int = 0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(**{f: getattr(self, f) + getattr(other, f)
                       for f in type(self).model_fields})


class ModelCall(Strict):
    """One request to a language model. The only place a prompt becomes a network call."""

    model: str
    prompt: str
    schema_name: str | None = None
    max_output_tokens: Annotated[int, Field(gt=0)] = 48_000
    effort: Literal["minimal", "low", "medium", "high"] = "low"
    attempts: Annotated[int, Field(ge=1)] = 3


class ModelReply(Strict):
    payload: dict
    cost: Cost
    stop_reason: str = ""


class Settings(Strict):
    """Everything the stages need that is not the paper."""

    payloads: Path
    records: Path
    model: str
    workflow: Workflow = Workflow.demand_driven
    stages: tuple[StageName, ...] = tuple(StageName)
    effort: Literal["minimal", "low", "medium", "high"] = "low"
    max_output_tokens: Annotated[int, Field(gt=0)] = 48_000
    attempts: Annotated[int, Field(ge=1)] = 3
    #: Evidence is 45% of input tokens. Dropping it leaves a record whose values have no
    #: supporting span -- structurally complete, and unreviewable.
    retrieve_evidence: bool = True
    zero_foci_rule: bool = True
    redo: bool = False

    @model_validator(mode="after")
    def _workflow_is_implemented(self) -> "Settings":
        """`entity_first` names an ordering this package does not have.

        It is kept in `Workflow` because it names a real alternative that was measured and
        rejected, and a run recorded as `entity_first` should not silently mean the other
        thing. Refusing here is the same rule as `extra="forbid"`: a setting that does not
        apply is an error, not a default.
        """
        if self.workflow is not Workflow.demand_driven:
            raise ValueError(
                f"workflow={self.workflow.value} is not implemented; the stages run "
                f"{Workflow.demand_driven.value} only")
        return self

    @model_validator(mode="after")
    def _build_needs_its_inputs(self) -> "Settings":
        if StageName.build in self.stages and StageName.satisfy not in self.stages:
            existing = self.payloads
            if not existing.exists():
                raise ValueError(
                    "build without satisfy needs payloads on disk from an earlier run; "
                    f"{existing} does not exist")
        return self


class StageOutcome(Strict):
    """What one stage did to one paper. `skipped` is a success, not a failure."""

    stage: StageName
    study_id: str
    produced: tuple[Path, ...] = ()
    cost: Cost = Cost()
    skipped: bool = False
    reason: str = ""

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

    def summary(self) -> str:
        cost = self.cost
        return (f"{len(self.papers)} paper(s), {len(self.failures)} failed · "
                f"{cost.input_tokens:,} in / {cost.output_tokens:,} out tokens "
                f"over {cost.calls} call(s)")
