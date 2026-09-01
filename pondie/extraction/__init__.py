"""Extraction: papers in, validated records out.

    from pondie.extraction.models import Paper, Settings
    from pondie.extraction import GatewayCaller, run

    report = run(papers, settings, GatewayCaller())
    report.summary()

The directory is the journey a paper takes, so where a thing lives says when it happens:

    corpus/     getting the paper onto disk. An INPUT -- a run reads it, never writes it
    prompt/     what the model is asked, and what the paper looks like when it is asked
    evidence/   which characters of the paper warrant each value
    record/     turning the payloads into a record: assemble, repair, check
    tools/      things done to records afterwards; none of them runs inside a pipeline

and the modules beside them are what every stage needs:

    models      the pydantic contracts that cross a boundary
    values      the `ExtractedValue` wrapper: what one is, how to read one, how to make one
    parse       the stage-1 parse document
    llm         the one place a prompt becomes a network call
    stages      the six steps, in order
    driver      sequencing, parallelism and accounting
    usage       per-call token accounting

Every boundary is a named type. Two of the bugs found while writing them down were invisible
without one: a stage unpacking `build_prompt`'s two halves in the wrong order sent the model
instructions about a paper it had never been shown, and five modules had each written their
own `ExtractedValue` unwrapper that disagreed with the others at the edges.
"""

from pondie.extraction.driver import plan, run, run_paper
from pondie.extraction.llm import Caller, GatewayCaller, MalformedReply, load_env
from pondie.extraction.stages import (
    DEMAND_DRIVEN,
    Build,
    Demands,
    Evidence,
    Satisfy,
    SignSplit,
    Stage,
    Tables,
    sequence,
)

__all__ = [
    "Caller",
    "GatewayCaller",
    "MalformedReply",
    "load_env",
    "plan",
    "run",
    "run_paper",
    "sequence",
    "Stage",
    "Tables",
    "SignSplit",
    "Demands",
    "Satisfy",
    "Evidence",
    "Build",
    "DEMAND_DRIVEN",
]
