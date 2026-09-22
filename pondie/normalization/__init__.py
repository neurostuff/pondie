"""Post-hoc normalization of extracted record fields, one module per field.

A field's shape decides its method, and four shapes recur. The measurements behind that
claim are in docs/normalization-pipelines.md; what follows is where each lives.

`task` was two modules doing this one job: a seeded one reachable only from a script, and
an unseeded one that clustered the whole corpus against itself and, by being the one that
exposed `normalize`, was the one `pondie normalize task` actually ran. They disagreed.
It is one module now, and it seeds against the Cognitive Atlas before it clusters.

  closed target   a small fixed set of answers, free-text input   `_lexicon`
                  coordinate_space, multiple_comparison_method, correction_scope,
                  medication_status, sex_distribution, handedness_distribution,
                  modality, prespecification
  link            an external vocabulary exists                   `pondie.vocabularies`, `_embedding`
                  medical_condition
  seed + cluster  a target covers part of the field; the corpus     `atlas`, `_embedding`
                  covers the rest
                  task
  partition       one field holding two kinds of value              rules, in the module
                  population_characteristics

Every module exposes `normalize(...)` returning a value plus the reason it was chosen, and
`report(...)` for the residual. Nothing is bucketed silently: an input no rule matched is
UNKNOWN with `reason="unmatched"` and is reported, so a new surface form surfaces instead of
disappearing into OTHER.

Modules with a leading underscore are shared machinery and are not part of the interface.
`fields()` is the list of field modules, and it is derived rather than written down: a field
module is one that exposes `normalize`, which is the contract above. `corpus` is the odd one
-- it maps a whole corpus rather than one field, and it has a CLI -- so it is deliberately
not in that list.

`modality` and `prespecification` are closed targets whose answers are the *schema's own*
permissible values rather than a downstream set like MNI or RIGHT -- both slots are open
ranges, and these two put the wording back on the vocabulary. For `modality` that is
load-bearing: only a vocabulary value carries an `instantiates`, so an unmapped one leaves
`Acquisition.acquisition_type` underivable.

`is_healthy` is a fifth thing and deliberately outside the list: it fills a slot rather
than normalizing one, from a field it does not touch, so it has no `normalize` to expose.
`population_characteristics` does both -- it classifies a value and it moves the
non-selective ones into `Group.other_characteristics` -- which is why `apply` sits beside
`normalize` in those two modules and nowhere else.

Eleven field modules and six mechanisms. Term lists both packages fetch and share live in
`pondie.vocabularies`, not here.
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
