"""The worked models, composed from the referent records rather than transcribed.

`representing-models.md` §5 shows twelve reported results and the encoding each takes. Every
one of those encodings is a projection of a record already in `prompt/referents/`, so writing
the YAML by hand stored the same facts twice and let the copy drift from the original. Three
of the twelve had: §5.10 expanded `HC`/`PD` into `healthy controls`/`Parkinson's disease
patients`, §5.12 contracted `healthy controls` to `HCs` -- both against §5.1's own rule that
"the levels must be the source's own" -- and §5.6's second block put an fMRI result's cells on
the VBM model's term, a fact from the paper in the wrong place.

So the source of truth is split in two, and neither half can restate the other:

  manifest.yaml   which record, which model, which analysis, which of their parts to show
  prose/*.md      the words -- what the encoding demonstrates, the sentences it came from

and the YAML in between is projected from the record at render time. A renamed level in a
record now propagates into the prompt and the document; a level that exists in neither cannot
be written down at all.

One renderer serves both consumers, because they want the same text: `compose()` builds the
prompt block, and `document()` wraps it in the section heading for the committed markdown.
"""

from __future__ import annotations

import functools
import pathlib
from typing import Any, Mapping, Sequence

import yaml

HERE = pathlib.Path(__file__).parent
DATA = HERE / "worked_models"
REFERENTS = HERE / "referents"

#: What a `ModelTerm` shows, in this order. `functional_form` and `source_definition` are the
#: two slots no worked model has ever shown: the first is a modelling detail no example turns
#: on, the second restates in prose what `name` and `levels` already carry.
TERM_SLOTS = (
    "name",
    "type",
    "variation_level",
    "unit",
    "assessment",
    "region",
    "interaction_with",
    "levels",
)
LEVEL_SLOTS = ("level", "order", "conditions", "groups", "regions", "timepoints", "arms")
MODEL_SLOTS = ("model_family", "stage", "estimator", "inputs_from")
ANALYSIS_SLOTS = ("model_estimation", "effect", "details")

PREAMBLE = (
    "Only the `ModelEstimation` fragment — its\n"
    "`stage` and `terms` — and the `Effect` are shown; a complete record also needs the "
    "Analysis's\nsample, paradigm, acquisition, measure, statistic and details."
)

SECTION = "## 5. Worked models"


def _unwrap(value: Any) -> Any:
    """The value inside an `ExtractedValue`, or None when the wrapper asserts nothing.

    A `not_reported` wrapper can still carry a stale `value`. Reading it would put a figure in
    a worked model that the record explicitly declines to state, which is the failure this
    module exists to make impossible, so the status decides and not the presence of a key.
    """
    if isinstance(value, dict) and "extraction_status" in value:
        return value.get("value") if value.get("extraction_status") == "extracted" else None
    return value


def _empty(value: Any) -> bool:
    return value is None or value == [] or value == "" or value == {}


@functools.lru_cache(maxsize=None)
def record(referent: str) -> Mapping[str, Any]:
    import json

    path = REFERENTS / f"{referent}.extraction.json"
    if not path.exists():
        raise FileNotFoundError(
            f"worked model cites referent {referent!r}, which is not in {REFERENTS}. The "
            "examples are projected from the records, so a missing record is a missing example."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(source: Mapping[str, Any], slots: Sequence[str]) -> dict[str, Any]:
    """`source`'s named slots, unwrapped, in `slots` order, dropping what it does not state."""
    out: dict[str, Any] = {}
    for slot in slots:
        value = _unwrap(source.get(slot))
        if not _empty(value):
            out[slot] = value
    return out


def _level(level: Mapping[str, Any]) -> dict[str, Any]:
    return _pick(level, LEVEL_SLOTS)


def _term(term: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": term.get("local_id")}
    for slot, value in _pick(term, TERM_SLOTS).items():
        out[slot] = [_level(x) for x in value] if slot == "levels" else value
    return out


def _model(rec: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    found = [m for m in rec.get("model_estimations", []) if m.get("local_id") == spec["id"]]
    if not found:
        raise LookupError(
            f"{rec.get('local_id')}: no model_estimation {spec['id']!r}. The manifest names a "
            "model the record does not have."
        )
    me = found[0]
    out: dict[str, Any] = {"id": me.get("local_id")} if spec.get("as_item") else {}
    out.update(_pick(me, [s for s in MODEL_SLOTS if s in spec.get("show", ())]))
    wanted = list(spec.get("terms") or ())
    by_id = {t.get("local_id"): t for t in me.get("terms") or []}
    missing = [t for t in wanted if t not in by_id]
    if missing:
        raise LookupError(
            f"{rec.get('local_id')}/{spec['id']}: no term(s) {missing}. The manifest names "
            "terms the model does not have."
        )
    if wanted:
        out["terms"] = [_term(by_id[t]) for t in wanted]
    return out


def _declared(rec: Mapping[str, Any]) -> dict[str, set[str]]:
    """Every term's declared level labels, by term id."""
    out: dict[str, set[str]] = {}
    for me in rec.get("model_estimations", []):
        for term in me.get("terms") or []:
            out[term.get("local_id")] = {
                _unwrap(level.get("level")) for level in term.get("levels") or []
            } - {None}
    return out


def _effect(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """The cells, and the mediation path when there is one.

    `Effect` has three slots across every referent -- `cells`, `statistic`, `mediation` -- and
    the worked models show two of them. `statistic` is excluded by the section's own preamble,
    which says the statistic is among the parts a complete record needs and this fragment does
    not show.
    """
    effect = analysis.get("effect") or {}
    cells = []
    for cell in effect.get("cells") or []:
        row: dict[str, Any] = {"term": cell.get("term")}
        level = _unwrap(cell.get("level"))
        if not _empty(level):
            row["level"] = level
        row["direction"] = _unwrap(cell.get("direction"))
        cells.append(row)
    out: dict[str, Any] = {"cells": cells}
    mediation = _pick(effect.get("mediation") or {}, ("path", "mediator"))
    if mediation:
        out["mediation"] = mediation
    return out


def _details(analysis: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    detail = analysis.get("details") or {}
    out: dict[str, Any] = {}
    for slot in fields:
        value = _unwrap(detail.get(slot))
        if _empty(value):
            continue
        if isinstance(value, list):
            value = [_pick(x, sorted(x)) if isinstance(x, dict) else x for x in value]
        out[slot] = value
    return out


def _analysis(rec: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    found = [a for a in rec.get("analyses", []) if a.get("local_id") == spec["id"]]
    if not found:
        raise LookupError(
            f"{rec.get('local_id')}: no analysis {spec['id']!r}. The manifest names an "
            "analysis the record does not have."
        )
    analysis = found[0]
    # A cell names a FactorLevel, so a label the term never declares resolves to nothing. The
    # hand-written section used to quietly substitute the declared label -- §5.10 showed
    # `healthy controls` where its record's cells said `HC` -- which is a document covering
    # for a record. Refusing is the version of that a reader can see.
    declared = _declared(rec)
    for cell in (analysis.get("effect") or {}).get("cells") or []:
        level, term = _unwrap(cell.get("level")), cell.get("term")
        if level is not None and declared.get(term) and level not in declared[term]:
            raise ValueError(
                f"{rec.get('local_id')}/{spec['id']}: cell on {term!r} is level {level!r}, "
                f"which that term does not declare; it has {sorted(declared[term])}. Fix the "
                "record -- a worked model cannot show an encoding that does not resolve."
            )
    out: dict[str, Any] = {}
    for slot in ANALYSIS_SLOTS:
        if slot not in spec.get("show", ()):
            continue
        if slot == "effect":
            out["effect"] = _effect(analysis)
        elif slot == "details":
            out["details"] = _details(analysis, spec.get("details_fields") or ())
        else:
            value = _unwrap(analysis.get(slot))
            if not _empty(value):
                out[slot] = value
    return out


def block(referent: str, spec: Mapping[str, Any], multistage: bool = False) -> dict[str, Any]:
    """One fenced encoding: the models it shows, then the analysis it shows.

    A single model is flattened to the top level, because an example about one model should
    not make a reader index into a list of one. Two or more keep the plural key, and the
    analysis moves into its own mapping -- `multistage`, decided over the whole example rather
    than this block, because §5.12 puts its models and its analysis in separate fences and the
    analysis still has to say which stage its `model_estimation` names.
    """
    rec = record(referent)
    models = list(spec.get("models") or ())
    out: dict[str, Any] = {}
    if len(models) == 1:
        out.update(_model(rec, models[0]))
    elif models:
        out["model_estimations"] = [_model(rec, dict(m, as_item=True)) for m in models]
    if spec.get("analysis"):
        rendered = _analysis(rec, spec["analysis"])
        if multistage:
            out["analysis"] = rendered
        else:
            out.update(rendered)
    return out


# --------------------------------------------------------------------------- rendering


def _scalar_row(mapping: Mapping[str, Any]) -> bool:
    return all(
        not isinstance(v, dict)
        and not (isinstance(v, list) and any(isinstance(x, dict) for x in v))
        for v in mapping.values()
    )


#: A row of scalars goes on one line until that line stops being readable, which is the rule
#: the hand-written section followed without stating: `term-motion` inline at 95 characters,
#: the three-slot mediation terms of §5.11 broken out at 101. Same number as `black` uses on
#: the code, for the same reason.
_WIDTH = 95


def _fits(pad: str, row: Mapping[str, Any]) -> bool:
    return len(pad) + 2 + len(_inline(row)) <= _WIDTH


def _inline(mapping: Mapping[str, Any]) -> str:
    parts = []
    for key, value in mapping.items():
        if isinstance(value, list):
            parts.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        else:
            parts.append(f"{key}: {value}")
    return "{" + ", ".join(parts) + "}"


def _render(node: Any, indent: int, notes: dict[str, str]) -> list[str]:
    """The doc's YAML dialect: flow mappings for rows of scalars, block for everything else.

    `notes` is consumed as it is used, so an annotation written once in the manifest lands on
    the first slot that carries it rather than on every term that happens to have that slot.
    """
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            # A blank line before the result separates the model from what it estimated, in
            # the examples that show both.
            if indent == 0 and key in ("effect", "analysis") and lines:
                lines.append("")
            if isinstance(value, list) and value and not any(isinstance(v, dict) for v in value):
                lines.append(f"{pad}{key}: [{', '.join(map(str, value))}]{_note(key, notes)}")
            elif isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_render(value, indent + 1, notes))
            else:
                shown = f"[{', '.join(map(str, value))}]" if isinstance(value, list) else value
                lines.append(f"{pad}{key}: {shown}{_note(key, notes)}")
        return lines
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and _scalar_row(item) and _fits(pad, item):
                lines.append(f"{pad}- {_inline(item)}{_first_note(item, notes)}")
            elif isinstance(item, dict):
                body = _render(item, indent + 1, notes)
                lines.append(f"{pad}- {body[0].strip()}")
                lines.extend(body[1:])
            else:
                lines.append(f"{pad}- {item}")
        return lines
    return [f"{pad}{node}"]


def _note(key: str, notes: dict[str, str]) -> str:
    comment = notes.pop(key, None)
    return f"   # {comment}" if comment else ""


def _first_note(row: Mapping[str, Any], notes: dict[str, str]) -> str:
    for key in row:
        if key in notes:
            return _note(key, notes)
    return ""


# --------------------------------------------------------------------------- assembly


@functools.lru_cache(maxsize=1)
def manifest() -> tuple[dict[str, Any], ...]:
    entries = yaml.safe_load((DATA / "manifest.yaml").read_text(encoding="utf-8"))
    return tuple(entries)


def prose(entry: Mapping[str, Any]) -> str:
    path = DATA / "prose" / f"{entry['id'].replace('.', '-')}-{entry['slug']}.md"
    if not path.exists():
        raise FileNotFoundError(f"worked model {entry['id']} has no prose at {path}")
    return path.read_text(encoding="utf-8").rstrip("\n")


def example(entry: Mapping[str, Any]) -> str:
    """One `### 5.x` subsection: its prose with each `{{block}}` replaced by its encoding."""
    notes = dict(entry.get("annotations") or {})
    multistage = sum(len(spec.get("models") or ()) for spec in entry["blocks"]) > 1
    blocks = [
        "```yaml\n"
        + "\n".join(_render(block(entry["referent"], spec, multistage), 0, notes))
        + "\n```"
        for spec in entry["blocks"]
    ]
    text = prose(entry)
    if text.count("{{block}}") != len(blocks):
        raise ValueError(
            f"worked model {entry['id']}: prose has {text.count('{{block}}')} block markers "
            f"and the manifest has {len(blocks)} blocks."
        )
    for rendered in blocks:
        text = text.replace("{{block}}", rendered, 1)
    return f"### {entry['id']} {entry['title']}\n\n{text}"


def compose() -> str:
    """The worked models as the prompt carries them."""
    return "\n\n".join([PREAMBLE] + [example(e) for e in manifest()])


def document() -> str:
    """The same text as the committed `representing-models.md` §5, heading included."""
    return f"{SECTION}\n\n{compose()}\n"
