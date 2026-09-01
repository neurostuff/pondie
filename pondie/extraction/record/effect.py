"""Compute an effect's derived kind from its cells. Nothing here is ever stored.

`representing-models.md` §3 states the derivation in six ordered steps, `extraction-readme.md`
§3 restates it, and `schema-tutorial.md` §9.6 teaches it -- but until this module existed the rule
was specified in three prose locations and executed in none. `EffectKind` is bound by no slot, and
that is deliberate: the kind follows entirely from the cells, so storing it would put one fact in
two places and let the two disagree. This is the rule as code, for a caller that wants the label
at query or index-build time.

The two non-answers are not failures and must not be collapsed into one:

- `UNDETERMINED_VARIATION` -- a cell sits on a continuous term whose `variation_level` is unset or
  is free text, so step 2 cannot choose between a modulation and a regression. `analysis.yaml`
  already says this is "undetermined for that record rather than wrong".
- `NO_LABEL` -- the cell pattern matches no step. `EffectKind`'s own description reads "a pattern
  yielding none is a record whose cells do not describe a test", so this is a defect signal about
  the record rather than a gap in the rule.

Reads both bare storage values and extraction's `ExtractedValue` wrappers, so one code path serves
the schema's own worked examples and a pipeline record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pondie.formats import values

UNDETERMINED_VARIATION = "undetermined:variation_level"
NO_LABEL = "none"

_SIGNED = ("positive", "negative")


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


def derive_effect_kind(cells: Any, terms: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    # NOTHING CALLS THIS. Kept deliberately, and the reason is not "it might be useful":
    # `Effect.kind` comes from the model today, and this is the only written-down statement
    # of the four-axis rubric that decides it -- the axes `study_schema` documents as the
    # thing this schema has and others do not. Deleting it would leave the rubric nowhere
    # in code, and `tests/test_effect_kind_derivation.py` would go with it.
    #
    # It is a deriver waiting for a caller, not a deriver whose caller was removed. Wire it
    # in `record/rules.py` if a measurement ever shows the model's `kind` disagreeing with
    # the cells it emitted.
    """The derived kind, and the step that decided it.

    Steps are §3's, in §3's order, and the order is load-bearing: a moderation is a product
    column and is caught by step 1, so step 2 never mistakes its continuous arm for a plain
    slope.
    """

    parsed: list[tuple[Any, Any, Any, Any]] = []
    for cell in cells or []:
        if not isinstance(cell, Mapping):
            continue
        term_id = cell.get("term")
        term = terms.get(term_id) if isinstance(term_id, str) else None
        parsed.append(
            (term_id, term, values.read(cell.get("level")), values.read(cell.get("direction")))
        )

    if not parsed:
        return NO_LABEL, "no cells"

    # Step 1 -- a cell on a product column. A non-empty `interaction_with` is what makes a
    # column a product.
    for term_id, term, _level, direction in parsed:
        if isinstance(term, Mapping) and (term.get("interaction_with") or []):
            if direction in _SIGNED:
                return "interaction", "step 1: signed cell on a product column"
            if direction == "undirected":
                return "omnibus", "step 1: undirected cell on a product column"
            # §3 writes only those two branches. A withheld sign keeps the shape and loses the
            # direction, which is step 6's principle applied to a product column.
            return "interaction", f"step 1 (extended): product column, direction={direction!r}"

    # Step 2 -- a cell on a continuous term; `variation_level` chooses the branch.
    for term_id, term, _level, _direction in parsed:
        if isinstance(term, Mapping) and values.read(term.get("type")) == "continuous":
            variation = values.read(term.get("variation_level"))
            if variation == "within_subject":
                return "parametric_modulation", "step 2: continuous, within_subject"
            if variation in {"between_subject", "mixed"}:
                return "cross_subject_regression", f"step 2: continuous, {variation}"
            return (
                UNDETERMINED_VARIATION,
                f"step 2: continuous term {term_id!r}, variation_level={variation!r}",
            )

    # Step 3 -- crossed terms. "Crossed" and not "signed": a term signed once has not been
    # compared against itself, which is why a cohort comparison of an activation map is a
    # contrast and not an interaction.
    signed: dict[Any, set[str]] = {}
    for term_id, _term, _level, direction in parsed:
        if direction in _SIGNED:
            signed.setdefault(term_id, set()).add(direction)
    crossed = [term_id for term_id, sides in signed.items() if len(sides) == 2]
    if len(crossed) >= 2:
        return "interaction", f"step 3: {len(crossed)} crossed terms"
    if len(crossed) == 1:
        return "contrast", "step 3: one crossed term"

    # Step 4 -- signed, nothing crossed.
    if signed:
        return "simple_effect", "step 4: signed cell, no crossed term"

    # Step 5 -- every cell undirected.
    if all(direction == "undirected" for *_rest, direction in parsed):
        return "omnibus", "step 5: every cell undirected"

    # Step 6 -- unsigned, but a term compared at two or more levels. The direction is lost and
    # the shape is not, which is what separates this from step 5.
    levels: dict[Any, set[Any]] = {}
    for term_id, _term, level, _direction in parsed:
        if level is not None:
            levels.setdefault(term_id, set()).add(level)
    compared = [term_id for term_id, seen in levels.items() if len(seen) >= 2]
    if len(compared) >= 2:
        return "interaction", f"step 6: {len(compared)} compared terms, direction lost"
    if len(compared) == 1:
        return "contrast", "step 6: one compared term, direction lost"

    return NO_LABEL, "cells describe no test"
