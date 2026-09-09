"""Which ModelTerms a cell of an Effect is allowed to name.

An analysis is fitted at a stage, and a cell may name a column fitted at a lower one -- a
group contrast of a first-level condition -- so the question is never "which terms does this
model declare" but "which terms does this model reach". `terms_in_scope` walks
`inputs_from` and answers the second, and `builder`, `validate` and `edit` all ask it.

This module also derived `EffectKind` -- the contrast / simple_effect / interaction label
that `representing-models.md` §3 states in six ordered steps. Deliberately never stored,
because the kind follows entirely from the cells and holding it as well would put one fact in
two places; written for "a caller that wants the label at query or index-build time", and no
such caller was ever written. It is in the history. `EffectKind` is still an enum in the
storage schema with no slot pointing at it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def terms_in_scope(
    model_id: Any,
    models: Mapping[str, Mapping[str, Any]],
    seen: set[str] | None = None,
) -> dict[str, Mapping[str, Any]]:
    """local_id -> ModelTerm for a model and every stage it reaches through `inputs_from`.

    A cell may name a column fitted at a lower stage -- a group contrast of a first-level
    condition -- so the chain rather than the one record. Own terms are collected last, so a
    column refitted at this stage shadows the lower one. Cycle-guarded: a record violating the
    acyclicity invariant would otherwise hang a walk whose job is to report on it.

    This is the only copy of this traversal. Four copies once existed: this one, two in
    `builder`, and one in `validate`. Three agreed. The fourth keyed on
    `values.read(local_id)`, so it put a
    term whose `local_id` had arrived wrapped into scope while the validator, reading the
    same record, did not: a cell could be repointed at a term that was then reported as out
    of scope. Strict is the right reading, because `unwrapped` is repair 2 and the two scope
    walks are repairs 10 and 12 -- an id is plain by the time anything asks this. A wrapped
    `local_id` reaching here is a different defect, and `check_field` is where it is named.
    """

    seen = set() if seen is None else seen
    if not isinstance(model_id, str) or model_id in seen:
        return {}
    seen.add(model_id)
    model = models.get(model_id)
    if not isinstance(model, Mapping):
        return {}

    terms: dict[str, Mapping[str, Any]] = {}
    for lower in model.get("inputs_from") or []:
        terms.update(terms_in_scope(lower, models, seen))
    for term in model.get("terms") or []:
        if isinstance(term, Mapping) and isinstance(term.get("local_id"), str):
            terms[term["local_id"]] = term
    return terms
