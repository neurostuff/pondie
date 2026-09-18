"""Repair a built record: propose, guard, and put what is left to a model.

    stage                    the three steps, and the model calls two of them make
    propose                  what to ask for, per class, projected from the schema
    propose_with_extractor   the extraction model answering it
    guard                    write a proposed change into a record, or say why not

Runs after `build`, on a record that already exists, and changes it in place. Everything
here decides something -- which is the whole difference from `record/fix`, where a fix that
could go two ways reports instead. `repair` names this and nothing else in the package.

`propose` and `propose_with_extractor` were `extraction/recall.py` and
`extraction/recall_llm.py`, outside this package, with the stage as their only production
caller. "Recall" is also the benchmark's metric -- a key in `scoring.py`, read seventeen
times there -- so the name said two things, the way `repair` did before the deterministic
half became `fix`.
"""

from pondie.extraction.repair.stage import REPAIRER, Case, Report, contradictions, run

__all__ = ["REPAIRER", "Case", "Report", "contradictions", "run"]
