"""What a `local_id` is, and who is allowed to choose one.

A local_id is an ADDRESS. The review layer keys an answer on
`paper|value|<Class>|<local_id>|<path>`, so an id that changes between extractions of the
same paper orphans every answer a reviewer gave against it. That is why they are short,
prefixed by class, and built from the shortest thing the *paper* fixes rather than from a
phrase anyone composed.

The table lives here rather than in the prompt that used to state it, because two things now
mint ids -- the extracting model, told the convention in prose, and the repair pass, which
needs it as data. Two copies of a convention is one copy and one drift.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

#: Class -> the prefix its ids carry.
PREFIX: dict[str, str] = {
    "Group": "grp_", "Acquisition": "acq_", "ModelEstimation": "mod_",
    "Task": "tsk_", "Preprocessing": "prp_", "ModelTerm": "trm_",
    "Assessment": "asm_", "Measure": "mea_", "InferenceSettings": "inf_",
    "Region": "reg_", "Arm": "arm_", "Timepoint": "tp_",
    "Device": "dev_", "ExternalDataset": "ext_",
    #: Only for an analysis with no row group; see `DERIVED`.
    "Analysis": "ana_",
}

#: Classes whose ids nobody chooses: they are derived from the table parse, and an id
#: invented for one would not match the row group the parse produced.
#:
#: `Analysis` is not among them. An analysis reported only in prose has no row group to
#: derive from -- 16038682 reports three peaks in a sentence and has no coordinate table at
#: all -- and refusing to name one is refusing to record it. The convention still holds
#: where a parse exists: an analysis built from a row group keeps the id the parse gave it,
#: and only one with no parse behind it is minted here.
DERIVED: frozenset[str] = frozenset({"Table"})


def prefix_table(width: int = 3) -> str:
    """The convention as the extraction prompt prints it, from the one definition."""
    rows = [f"{p:<6} {c:<18}" for c, p in PREFIX.items()]
    lines = ["     " + "".join(rows[i:i + width]).rstrip()
             for i in range(0, len(rows), width)]
    return "\n".join(lines)


def mint(class_name: str, label: str, taken: Mapping[str, Any] | set[str]) -> str | None:
    """An address for a new entity, or None for a class whose ids are not chosen.

    Built from the label because that is the shortest thing the paper fixes, and suffixed
    only where the paper has two of a kind.
    """
    if class_name in DERIVED:
        return None
    stem = "_".join(re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_").split("_")[:3])
    candidate = f"{PREFIX.get(class_name, class_name[:3].lower() + '_')}{stem[:28]}"
    if not stem:
        return None
    if candidate not in taken:
        return candidate
    n = 2
    while f"{candidate}_{n}" in taken:
        n += 1
    return f"{candidate}_{n}"
