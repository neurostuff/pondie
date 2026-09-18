"""What a `local_id` is, who is allowed to choose one, and how to read one back.

A local_id is an ADDRESS. The review layer keys an answer on
`paper|value|<Class>|<local_id>|<path>`, so an id that changes between extractions of the
same paper orphans every answer a reviewer gave against it. That is why they are short,
prefixed by class, and built from the shortest thing the *paper* fixes rather than from a
phrase anyone composed.

The table lives here rather than in the prompt that used to state it, because two things now
mint ids -- the extracting model, told the convention in prose, and the repair pass, which
needs it as data. Two copies of a convention is one copy and one drift.

`from_local_id` is `mint` run backwards, and `label_of` is what falls back to it. They lived
in the repair stage's guard, which reached `PREFIX` through a deferred import to get them --
so the convention had one writer and a reader somewhere else.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from pondie.formats import values

#: Class -> the prefix its ids carry.
PREFIX: dict[str, str] = {
    "Group": "grp_",
    "Acquisition": "acq_",
    "ModelEstimation": "mod_",
    "Task": "tsk_",
    "Preprocessing": "prp_",
    "ModelTerm": "trm_",
    "Assessment": "asm_",
    "Measure": "mea_",
    "InferenceSettings": "inf_",
    #: Minted by `table_local_id` from the printed number, not by `mint`; see `DERIVED`.
    "Table": "tbl",
    "Region": "reg_",
    "Arm": "arm_",
    "Timepoint": "tp_",
    "Device": "dev_",
    "ExternalDataset": "ext_",
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

#: Words a printed table label wraps its number in. Stripped so "Supplementary Table S4"
#: and "Table S4" mint the same id -- the number is the identity, the wording is not.
_LABEL_NOISE = re.compile(r"\b(?:supplementary|supplemental|suppl?|online|table|tbl)\b", re.I)


def table_local_id(number: Any, label: Any, taken: Mapping[str, Any] | set[str]) -> str:
    """An address for a Table, from the number the paper printed on it.

    Not from the manifest's `table_id`. That key is the staging flavour's, and it varies
    with the flavour rather than with the paper: across the corpus the same slot holds bare
    `20`, `tbl1`, `t1`, `T1`, `Tab1`, `pone-0074164-t002` and `pone.0074164.t002`. A model
    shown `pone.0074164.t002` and reading "Table 2" in the prose writes `tab_2`, and 485
    `Analysis.tables` references dangle that way. The printed number is the thing the
    *paper* fixes, which is the rule the rest of this module already follows, and `tbl2` is
    what a model writes unprompted.

    `table_number` is not unique -- one paper in the corpus carries two tables numbered 1 --
    so a collision takes a suffix rather than overwriting. Positional only when the manifest
    carried no number at all, because a positional id moves if a table is added upstream.
    """
    stem = re.sub(r"[^a-z0-9]+", "", str(number or "").lower())
    if not stem:
        stem = re.sub(r"[^a-z0-9]+", "", _LABEL_NOISE.sub("", str(label or "")).lower())
    if not stem:
        return ""
    candidate = f"{PREFIX['Table']}{stem[:24]}"
    if candidate not in taken:
        return candidate
    n = 2
    while f"{candidate}_{n}" in taken:
        n += 1
    return f"{candidate}_{n}"


def prefix_table() -> str:
    """The convention as the extraction prompt prints it, from the one definition.

    `DERIVED` classes are left out. Printing `tbl Table` beside the classes a model does
    mint ids for reads as permission to mint one, and a minted Table id points at nothing:
    the Tables stage has already assigned them and the prompt hands them over by name.
    """
    width = 3
    rows = [f"{p:<6} {c:<18}" for c, p in PREFIX.items() if c not in DERIVED]
    lines = ["     " + "".join(rows[i : i + width]).rstrip() for i in range(0, len(rows), width)]
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


#: Below this a derived label is a fragment, not a name. `mea_fa` would otherwise offer "fa",
#: which appears inside "factor" and "surface" -- and `same_entity` merging on that is the
#: opposite of the duplicate it exists to prevent.
_SHORTEST_DERIVED = 4


def from_local_id(local_id: str) -> str:
    """A readable label out of a minted id: `dev_siemens_trio` -> "siemens trio".

    `Measure`, `Acquisition`, `Device` and `ModelEstimation` declare no `name`, and only
    `Device` has no usable fallback either, so `label_of` below fell through to the raw id for a
    third of all entities -- 336 of 1,032 over eighty papers. Nothing matches
    `dev_siemens_trio` in a paper, so every mechanism that reads a label was working blind
    on those: `resolve` turning a proposed name into an id, `same_entity` deduplicating, and
    the locator's bonus for a sentence that mentions the entity.

    The ids are minted from content, so the content is recoverable. Of the 333 whose label
    was absent from their paper, 57.7% match verbatim once derived and a further 35.7% have
    every token present. Short results are refused rather than guessed at.
    """
    text = local_id.strip()
    for prefix in sorted(PREFIX.values(), key=len, reverse=True):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = re.sub(r"[_\-]+", " ", text).strip()
    return text if len(text) >= _SHORTEST_DERIVED else ""


def label_of(entity: Mapping[str, Any]) -> str:
    """What a source would call this entity: its name, else its definition, else its id.

    The id is read through `from_local_id` rather than returned raw, because a minted id is
    not a string any paper contains. It is a label for matching and never a name to store:
    `acq_fmri` yields "fmri", which is the modality rather than what the paper calls that
    acquisition.
    """
    for slot in ("name", "definition", "model_type", "type"):
        text = values.read(entity.get(slot))
        if isinstance(text, str) and text.strip():
            return text.strip()
    local_id = str(entity.get("local_id") or "")
    return from_local_id(local_id) or local_id
