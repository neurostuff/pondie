"""Post-hoc normalization of extracted record fields, one module per field.

A field's shape decides its method, and three shapes recur. The measurements behind that
claim are in docs/normalization-pipelines.md; what follows is where each lives.

  closed target   a small fixed set of answers, free-text input   `_lexicon`
                  coordinate_space, multiple_comparison_method, correction_scope,
                  medication_status, sex_distribution, handedness_distribution
  link            an external vocabulary exists                   `pondie.vocabularies`, `_embedding`
                  medical_condition
  cluster         no usable target; the corpus is its own          `_clustering`, `_embedding`
                  task

Every module exposes `normalize(...)` returning a value plus the reason it was chosen, and
`report(...)` for the residual. Nothing is bucketed silently: an input no rule matched is
UNKNOWN with `reason="unmatched"` and is reported, so a new surface form surfaces instead of
disappearing into OTHER.

Modules with a leading underscore are shared machinery and are not part of the interface.
`fields()` is the list of field modules, and it is derived rather than written down: a field
module is one that exposes `normalize`, which is the contract above. `corpus` is the odd one
-- it maps a whole corpus rather than one field, and it has a CLI -- so it is deliberately
not in that list.

Eight field modules and five mechanisms, and that is now the whole directory. The two
largest files used to be here -- the ONVOC index and the abbreviation store, 1,035 lines
between them -- imported by no field module and reached only from `extraction`, so a reader
following the table above met 40% of the package that the table does not describe. They are
`pondie.vocabularies` now, which is what they were: fetched term lists both packages use.
"""

from __future__ import annotations

UNKNOWN = "UNKNOWN"
OTHER = "OTHER"

__all__ = ["UNKNOWN", "OTHER"]


def fields() -> list[str]:
    """The field modules, by the contract rather than by a hand-kept list.

    A field module is one that exposes `normalize`. Deriving it means the CLI's choices and
    the contract test cannot drift from what the package actually offers -- which they did:
    five of the eight bound `normalize` and not `report`, and `pondie normalize
    coordinate_space` raised on a field its own help text offered as an example.
    """
    import importlib
    import pkgutil

    found = []
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("_"):
            continue
        loaded = importlib.import_module(f"{__name__}.{module.name}")
        if callable(getattr(loaded, "normalize", None)):
            found.append(module.name)
    return sorted(found)
