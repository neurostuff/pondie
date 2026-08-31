"""Post-hoc normalization of extracted record fields, one module per field.

A field's shape decides its method, and three shapes recur. The measurements behind that
claim are in docs/normalization-pipelines.md; what follows is where each lives.

  closed target   a small fixed set of answers, free-text input   `_lexicon`
                  coordinate_space, multiple_comparison_method, correction_scope,
                  medication_status, sex_distribution, handedness_distribution
  link            an external vocabulary exists                   `_vocabulary`, `_embedding`
                  medical_condition
  cluster         no usable target; the corpus is its own          `_clustering`, `_embedding`
                  task

Every module exposes `normalize(...)` returning a value plus the reason it was chosen, and
`report(...)` for the residual. Nothing is bucketed silently: an input no rule matched is
UNKNOWN with `reason="unmatched"` and is reported, so a new surface form surfaces instead of
disappearing into OTHER.

Modules with a leading underscore are shared machinery and are not part of the interface.
"""

from __future__ import annotations

UNKNOWN = "UNKNOWN"
OTHER = "OTHER"

__all__ = ["UNKNOWN", "OTHER"]
