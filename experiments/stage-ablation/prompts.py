"""Prompt variants for the `single` pass, applied as in-process patches to pondie's renderer.

    python run_arm.py --arm pondie_single --variant no_worked ...

A variant changes what the model is asked and nothing else: the stage, its post-conditions,
the build and the query are pondie's own. Each run is its own process and its own run
directory, so a patch never leaks into another arm and the stage cache never mixes two
prompts. A variant that wins is then written into pondie properly.
"""
from __future__ import annotations

import json
from typing import Callable

from pondie.extraction.prompt import render
from pondie.extraction.record import builder

#: The scaffold key the inventory variant asks for. Kept out of the record by the builder.
INVENTORY = "results_inventory"


def _cut(text: str, start: str, end: str) -> str:
    """Remove the section from heading `start` up to (not including) heading `end`."""
    i, j = text.index(start), text.index(end)
    return text[:i] + text[j:]


def _wrap_build_prompt(edit_system: Callable[[str], str]) -> None:
    original = render.build_prompt

    def build_prompt(text, mode, evidence, context):
        prompt = original(text, mode, evidence, context)
        return type(prompt)(system=edit_system(prompt.system), user=prompt.user)

    render.build_prompt = build_prompt


def no_worked() -> None:
    """H1: the worked models (~8.5k tokens) are not needed for the single pass."""
    _wrap_build_prompt(lambda s: _cut(s, "# Worked models", "# Schema"))


def no_conventions() -> None:
    """H2: the conventions document (~9.8k tokens) is not needed for the single pass."""
    _wrap_build_prompt(lambda s: _cut(s, "# Conventions", "# Worked models"))


INVENTORY_NOTE = f"""
Before anything else, write a top-level `{INVENTORY}` list: every result the paper reports,
one entry per tested comparison or association, wherever it is reported -- the Results text,
a table, a figure caption, or a sentence pointing to a supplement whose contents you cannot
see. A comparison of patients with controls stated in the text with its coordinates "in
Supplementary Table 2" is a result and gets an entry. Each entry:

  {{"result": "<the paper's statement, short>", "where": "text|table|figure|supplement",
    "analysis": "<the local_id of the Analysis you emit for it>"}}

or, when you emit no Analysis for it, `"analysis": null` and a `"why"`. Then write
`analyses` so that every entry with an `analysis` id has that Analysis, then the entities.
The inventory is your working list; it is not part of the record.
"""


def inventory() -> None:
    """H3: listing every reported result first recovers the comparisons the pass skips."""
    render.MODE_NOTE["single"] = INVENTORY_NOTE + render.MODE_NOTE["single"]
    builder._SCAFFOLDING = builder._SCAFFOLDING | {INVENTORY}
    original = render.normalize

    def normalize(payload, mode):
        held = payload.pop(INVENTORY, None)
        payload, notes = original(payload, mode)
        if held is not None:
            payload[INVENTORY] = held
            notes = list(notes) + [f"{INVENTORY}: {len(held) if isinstance(held, list) else '?'} entries"]
        return payload, notes

    render.normalize = normalize


COHORT_NOTE = """
State every cohort's `medical_condition`, the comparison cohorts included: a control cohort
that has none says so in the paper's words ("no PTSD", "healthy controls", "trauma-exposed
without PTSD"), rather than leaving the slot empty. Whether a cohort is a case or a comparison
is decided from this slot.
"""

SCOPE_NOTE = """
`spatial_scope` is about which voxels were modelled. A mask that keeps the analysis inside
the brain or inside grey matter (an explicit grey-matter mask, an absolute threshold, an
AAL-derived grey-matter mask) leaves it `whole_brain`. `roi` is for an analysis restricted to
regions chosen in advance (the hippocampus, the ACC, a sphere around a peak), and a
small-volume correction applied after a whole-brain test is a correction, not a scope.
"""


def cohort_condition() -> None:
    """H4: comparison cohorts state their condition, so a query can tell case from control."""
    render.MODE_NOTE["single"] = render.MODE_NOTE["single"] + COHORT_NOTE


def scope() -> None:
    """H5: a grey-matter mask is not an ROI."""
    render.MODE_NOTE["single"] = render.MODE_NOTE["single"] + SCOPE_NOTE


VARIANTS: dict[str, Callable[[], None]] = {
    "no_worked": no_worked,
    "no_conventions": no_conventions,
    "inventory": inventory,
    "cohort_condition": cohort_condition,
    "scope": scope,
}


def apply(names: list[str]) -> list[str]:
    for name in names:
        VARIANTS[name]()
    return names


def describe() -> str:
    return json.dumps(sorted(VARIANTS))
