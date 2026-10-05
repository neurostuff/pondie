"""Extraction: papers in, validated records out.

    from pondie.extraction.models import Paper, Settings
    from pondie.extraction import GatewayCaller, run

    report = run(papers, settings, GatewayCaller())
    report.summary()

The directory follows the paper's journey:

    corpus/     getting the paper onto disk; an input a run reads and never writes
    prompt/     what the model is asked, and how the paper is shown to it
    evidence/   which characters of the paper warrant each value
    record/     turning payloads into a record: assemble, deterministic fixes, checks
    repair/     model-proposed improvements to a built record, guarded

and the modules beside them:

    models      the pydantic contracts that cross a boundary
    parse       the stage-1 parse document
    sign_split  splitting a two-signed table into its two contrasts
    llm         the one place a prompt becomes a network call
    pubmed      publication type, language and authorship from PubMed
    stages      the stages, and the two orders they run in
    driver      scheduling, progress and accounting

`pondie.formats.values` holds the `ExtractedValue` wrapper, at the top of the package
because every reader of a record needs it.
"""

from pondie.extraction.driver import plan, run
from pondie.extraction.llm import Caller, GatewayCaller, MalformedReply, load_env
from pondie.extraction.stages import (
    DEMAND_DRIVEN,
    SINGLE_PASS,
    Build,
    Demands,
    Evidence,
    Fill,
    Repair,
    ProseFoci,
    Satisfy,
    SignSplit,
    Single,
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
    "sequence",
    "Stage",
    "Tables",
    "ProseFoci",
    "SignSplit",
    "Single",
    "Demands",
    "Fill",
    "Satisfy",
    "Evidence",
    "Build",
    "Repair",
    "SINGLE_PASS",
    "DEMAND_DRIVEN",
]
