"""Extraction: papers in, validated records out.

    from pondie.extraction.models import Paper, Settings
    from pondie.extraction import GatewayCaller, run

    report = run(papers, settings, GatewayCaller())
    report.summary()
"""
from .driver import plan, run, run_paper
from .llm import Caller, GatewayCaller, load_env
from .stages import DEMAND_DRIVEN, Build, Demands, Evidence, Satisfy, Stage, Tables, sequence

__all__ = ["Caller", "GatewayCaller", "load_env", "plan", "run", "run_paper", "sequence",
           "Stage", "Tables", "Demands", "Satisfy", "Evidence", "Build", "DEMAND_DRIVEN"]
