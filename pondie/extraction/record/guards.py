"""What a repair pass may not write, and why each refusal exists.

A repair pass proposes edits to a built record: a value corrected, a reference drawn, an
entity added. Most proposals are improvements. The ones here are not, and every guard below
is a regression that shipped -- each names the paper it was found on, because that is what
makes a future change checkable against the same evidence rather than against an argument.

The shape they share is worth stating: a repair that is *locally* plausible can still be
wrong about the record it is editing. Shortening a value looks like a correction. Coercing
a scope to a permissible value looks like normalisation. Repointing a reference looks like a
fix. What makes each of them damage is something elsewhere in the record -- the span that
warranted the old value, the regions named beside the scope, the terms the analysis's cells
already name -- so a guard is a function of the record and not only of the edit.

Refusals are returned, not raised, and carry their reason. A pass that silently declines is
indistinguishable from one that was never asked, which is how `existence_unsupported`
rejected an unknown number of candidates for the life of a run.

`repairs.py` holds the deterministic fixes applied to a payload on the way in. These are the
other direction: constraints on writes coming from a model afterwards. Same vocabulary --
name, what, ordering -- and a different signature, so they are a sibling family rather than
`Apply` functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from pondie.formats import values
from pondie.schema.reader import Schema


@dataclass(frozen=True)
class Edit:
    """One proposed write, and everything a guard needs to judge it."""

    schema: Schema
    record: Mapping[str, Any]
    #: The entity being edited, as it stands before the write.
    entity: Mapping[str, Any]
    class_name: str
    slot: str
    #: What the pass wants to put there: a scalar for a value slot, a list of local_ids for
    #: a reference slot.
    value: Any

    @property
    def current(self) -> Any:
        return self.entity.get(self.slot)

    @property
    def current_value(self) -> Any:
        return values.read(self.current)


@dataclass(frozen=True)
class Refusal:
    """Why a write did not happen, in terms a reviewer can act on."""

    slot: str
    why: str
    value: Any = None


Check = Callable[[Edit], "Refusal | None"]


@dataclass(frozen=True)
class Guard:
    """One named refusal, and where it sits relative to the others."""

    name: str
    what: str
    check: Check
    #: Empty when the guard may run anywhere. Stated when it may not -- the paired-slot
    #: guards read a sibling slot, so they must run after that slot has been written or the
    #: value they check against is the one from before the edit. Both directions of the
    #: scope pair failed exactly this way: references were written first, so the guard on
    #: the regions side ran while the scope was still unset, and the scope was then set with
    #: nothing left to check it against.
    after: str = ""


# ------------------------------------------------------------------------------- helpers


def _bare(text: Any) -> str:
    """Letters and digits only, for asking whether one string says what another says."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _scope_pair(slot: str) -> str:
    """The slot on the other side of a scope/regions pair, or ''.

    `Analysis.spatial_scope` says what volume was modelled and `Analysis.regions` names what
    it was restricted to; `InferenceSettings.correction_scope` and `correction_regions` say
    the same about the correction. Each pair is meaningless one half at a time.
    """
    return {"spatial_scope": "regions", "regions": "spatial_scope",
            "correction_scope": "correction_regions",
            "correction_regions": "correction_scope"}.get(slot, "")


UNRESTRICTED = frozenset({"whole_brain", "whole brain", "searchlight"})


# -------------------------------------------------------------------------------- guards


def refuses_truncation(edit: Edit) -> Refusal | None:
    """Shortening a value is not correcting it.

    22952599: `definition` went from "compared to traumatized controls." to "compared to
    traumatized", and lost its evidence doing so. The same path had correctly *extended* a
    definition on 23021615, so the direction is what distinguishes them -- an edit has to
    add something.
    """
    old, new = edit.current_value, edit.value
    if not isinstance(old, str) or not isinstance(new, str):
        return None
    if _bare(new) and _bare(new) in _bare(old) and _bare(new) != _bare(old):
        return Refusal(edit.slot, "shortens the value it replaces", new)
    return None


def refuses_shortening_a_list(edit: Edit) -> Refusal | None:
    """A single value does not replace several.

    16701903 acquires two sequences -- MP-RAGE at TE 4.4 ms and FLASH at TE 5 ms -- and
    `echo_time_seconds` held both. A one-value correction dropped the FLASH echo time, which
    is a truncated definition wearing a different shape.

    That record should be two acquisitions rather than one (`rules.check_one_protocol_per_
    acquisition` reports it), so this guard defends an encoding that is itself wrong. It is
    still the right refusal: a repair pass is not where that is decided.
    """
    old = edit.current_value
    if isinstance(old, list) and len(old) > 1 and not isinstance(edit.value, list):
        return Refusal(edit.slot, "replaces several values with one", edit.value)
    return None


def refuses_losing_the_warrant(edit: Edit) -> Refusal | None:
    """An edit keeps its evidence or says more; it does not quietly drop both.

    12853571: `correction_scope` went from "whole volume analyzed and a priori small
    volumes" -- cited, and true, the paper did both -- to the bare enum "whole_brain", which
    drops the small-volume half and has no sentence behind it. Coercion to a permissible
    value is not a correction.

    The old spans are kept when they still contain the new value, which is what lets a
    genuine extension through: on 23021615 the restored full sentence was already the span.
    """
    node = edit.current
    if not isinstance(node, Mapping):
        return None
    if (node.get("evidence") or {}).get("status") != "present":
        return None
    want = _bare(edit.value)
    for group in (node.get("evidence") or {}).get("sets") or []:
        for span in group.get("spans") or []:
            if want and want in _bare(span.get("text", "")):
                return None
    return Refusal(edit.slot, "loses the span that warranted the value it replaces",
                   edit.value)


def refuses_an_unrestricted_scope_beside_regions(edit: Edit) -> Refusal | None:
    """Whole-brain and searchlight restrict to no region, in both directions.

    11950456, an STG volumetric study, acquired `correction_scope: whole_brain` beside
    `correction_regions: [r_stg]`; 19996042 and 16038682 acquired `roi` beside nothing. The
    pair states one procedure, so half of it alone is not a claim.
    """
    other = _scope_pair(edit.slot)
    if not other:
        return None
    if edit.slot.endswith("scope"):
        scope, regions = str(edit.value).strip().lower(), edit.entity.get(other)
        if scope in UNRESTRICTED and regions:
            return Refusal(edit.slot, f"{other} is not empty", edit.value)
        if scope in {"roi", "region of interest"} and not regions:
            return Refusal(edit.slot, f"{other} is empty, so the restriction is unnamed",
                           edit.value)
        return None
    # The regions side reads the *scope* slot, not its own: what makes naming a region
    # wrong is the scope beside it saying the search was not restricted.
    scope = str(values.read(edit.entity.get(other)) or "").strip().lower()
    if scope in UNRESTRICTED and edit.value:
        return Refusal(edit.slot, f"{other} is not restricted to a region", edit.value)
    return None


def refuses_a_self_reference(edit: Edit) -> Refusal | None:
    """Nothing is its own input, its own mirror, or a term it is a product of.

    27082610 and 19942229: `inputs_from` resolved to the model being edited, because the
    label the reply gave matched the entity whose slot it was filling. The three slots whose
    range is their own class all mean a *different* instance of it.
    """
    own = edit.entity.get("local_id")
    named = edit.value if isinstance(edit.value, list) else [edit.value]
    if own and own in named:
        return Refusal(edit.slot, "names the entity it is written on", own)
    return None


def refuses_orphaning_cell_terms(edit: Edit) -> Refusal | None:
    """An analysis keeps reaching the terms its cells name.

    19942229: `a_793_1` was repointed at a model that does not reach `trm_group_r_nr`, so
    two of its cells named a term nothing could resolve. The cells are the analysis's own
    structure and the model reference is a pointer, so the pointer is what gives way.
    """
    if edit.slot != "model_estimation" or not edit.value:
        return None
    target = edit.value[0] if isinstance(edit.value, list) else edit.value
    named = {cell.get("term")
             for cell in ((edit.entity.get("effect") or {}).get("cells") or [])
             if isinstance(cell, Mapping) and isinstance(cell.get("term"), str)}
    if not named:
        return None
    from pondie.extraction.record.effect import terms_in_scope

    models = {m.get("local_id"): m for m in edit.record.get("model_estimations") or []
              if isinstance(m, Mapping) and m.get("local_id")}
    reachable = set(terms_in_scope(target, models))
    missing = named - reachable
    if missing:
        return Refusal(edit.slot,
                       f"would leave cells naming {', '.join(sorted(missing))}, which this "
                       f"model does not reach", target)
    return None


GUARDS: tuple[Guard, ...] = (
    Guard("truncation", "an edit adds something rather than shortening",
          refuses_truncation),
    Guard("list_shortening", "one value does not replace several",
          refuses_shortening_a_list),
    Guard("warrant", "an edit keeps the span that warranted what it replaces",
          refuses_losing_the_warrant),
    Guard("scope_pair", "a scope and the regions beside it agree",
          refuses_an_unrestricted_scope_beside_regions, after="reference_slots"),
    Guard("self_reference", "nothing references itself", refuses_a_self_reference),
    Guard("cell_terms", "an analysis reaches the terms its cells name",
          refuses_orphaning_cell_terms),
)


def refusals(edit: Edit, guards: tuple[Guard, ...] = GUARDS) -> list[Refusal]:
    """Every reason this write should not happen. Empty means write it."""
    return [refusal for guard in guards if (refusal := guard.check(edit)) is not None]
