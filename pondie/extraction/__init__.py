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
    parse       the stage-1 parse document
    llm         the one place a prompt becomes a network call
    stages      the seven steps, in order
    driver      sequencing, parallelism and accounting
    recall      asking a second model for what the first missed
    repair      improving a built record, and reporting what the attempt broke

`pondie.formats.values` holds the `ExtractedValue` wrapper. It sits at the top of the
package rather than here because every consumer of a record needs it -- the query engine,
normalization, the benchmark and the schema reader -- and importing the extraction package
to read a record closed a cycle.

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
    Repair,
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
    "Repair",
    "DEMAND_DRIVEN",
]
