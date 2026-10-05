"""Strict JSON schemas for the model stages' replies (Structured Outputs).

Each stage replies in its own shape, so each has its own schema: `single` the record's
entities, `fill` one answer per open slot, `evidence` one quote per field. All are
generated from the extraction schema, so a reply decoded against one cannot hold a key the
schema does not declare, a wrapper where an entity belongs, or a value outside a closed
vocabulary -- the faults `record.fix.shape` otherwise repairs after the fact.

Strict mode requires every property, so an optional slot is written as `null`. The caller
removes those before a reply is used (`llm._without_nulls`), which restores the shape every
other reply has: an absent slot is an absent key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from pondie import schema
from pondie.extraction.prompt import render
from pondie.extraction.prompt.fill import PLAIN, VOCABULARY
from pondie.schema.reader import Schema

_SCALARS = {"string": "string", "integer": "integer", "float": "number", "double": "number",
            "decimal": "number", "boolean": "boolean"}
_NULL = {"type": "null"}


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "properties": dict(properties), "required": list(properties)}


def _nullable(node: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [node, _NULL]}


def _value_type(sch: Schema, ranges: Sequence[str], multivalued: bool) -> dict[str, Any]:
    """A value's type: a closed vocabulary's terms, an open one's string, or a scalar."""
    concrete = [r for r in ranges if r != "Any"]
    enum = next((sch.enums[r] for r in concrete if r in sch.enums), None)
    terms = sorted((enum.permissible_values or {}).keys()) if enum is not None else []
    if enum is not None and len(concrete) == 1:
        node: dict[str, Any] = {"type": "string", "enum": terms}
    elif terms:
        # An open vocabulary is a plain string to the decoder, so its terms are named here,
        # with the fallback stated as firmly: a thin vocabulary otherwise draws the nearest
        # term whether or not it is true.
        node = {"type": "string", "description": f"Use one of: {', '.join(terms)} -- if it "
                "says what the paper says. If none does, write the paper's own words; never "
                "the nearest term when it is not true."}
    else:
        node = {"type": _SCALARS.get(concrete[0] if concrete else "string", "string")}
    return {"type": "array", "items": node} if multivalued else node


#: How a `single` reply carries its evidence: not at all, a quote per field, or the numbers
#: of the sentences that state it (`evidence.cited`).
EVIDENCE_FORMATS = ("none", "quotes", "indexed")


class _Builder:
    def __init__(self, sch: Schema, evidence: str) -> None:
        self.sch, self.evidence, self.defs = sch, evidence, {}
        reasons = sch.enums.get("UnreportedReason")
        self.reasons = sorted((reasons.permissible_values or {}).keys()) if reasons else []

    def ref(self, name: str) -> dict[str, Any]:
        if name not in self.defs:
            self.defs[name] = {}  # reserved first: classes refer to themselves
            self.defs[name] = self.entity(name)
        return {"$ref": f"#/$defs/{name}"}

    def nested(self, name: str) -> dict[str, Any]:
        """A nested slot's target, or -- for a type-designated class -- the choice of it and
        its subclasses, each with its designator fixed (`acquisition_type: "MRI"`)."""
        definition = self.sch.definition(name)
        concrete = sorted(self.sch.subclasses(name))
        if not concrete or not self.sch.type_designator(name):
            return self.ref(name)
        own = [] if definition is not None and definition.abstract else [name]
        return {"anyOf": [self.ref(c) for c in own + concrete]}

    def wrapper(self, name: str) -> dict[str, Any]:
        key = f"{name}__wrapper"
        if key not in self.defs:
            value = (self.sch.attributes(name) or {}).get("value")
            typed = _value_type(self.sch, self.sch.ranges(value) if value else ["string"],
                                bool(value is not None and value.multivalued))
            extracted = {"extraction_status": {"type": "string", "enum": ["extracted"]},
                         "value": typed,
                         "value_source": {"type": "string", "enum": ["reported", "generated"]}}
            if self.evidence == "quotes":
                extracted["evidence"] = _object({
                    "status": {"type": "string", "enum": ["present", "not_found"]},
                    "sets": {"type": "array", "items": _object(
                        {"quotes": {"type": "array", "items": {"type": "string"}}})},
                })
            elif self.evidence == "indexed":
                extracted["evidence"] = {
                    "type": "array", "items": {"type": "integer"},
                    "description": "Numbers of the sentences ([S12]) that state this value.",
                }
            absent = {"extraction_status": {"type": "string", "enum": ["not_reported"]},
                      "unreported_reason": _nullable({"type": "string", "enum": self.reasons})}
            self.defs[key] = {"anyOf": [_object(extracted), _object(absent)]}
        return {"$ref": f"#/$defs/{key}"}

    def slot(self, owner: str, name: str, spec: Any) -> dict[str, Any]:
        kind = self.sch.classify(name, spec)
        ranges = self.sch.ranges(spec) or ["string"]
        if name == self.sch.type_designator(owner):
            return {"type": "string", "enum": [owner]}
        if kind == "evidence":
            return self.wrapper(ranges[0])
        if kind == "nested":
            node = self.nested(ranges[0])
        elif kind in ("reference", "identifier"):
            node = {"type": "string"}
        else:
            node = _value_type(self.sch, ranges, False)
        return {"type": "array", "items": node} if spec.multivalued else node

    def entity(self, name: str, keep: Sequence[str] | None = None) -> dict[str, Any]:
        """One class as an object, every slot required and the optional ones nullable."""
        properties = {}
        for attr, spec in (self.sch.attributes(name) or {}).items():
            if schema.code_fills(name, attr) or (keep is not None and attr not in keep):
                continue
            node = self.slot(name, attr, spec)
            properties[attr] = node if spec.required else _nullable(node)
        return _object(properties)


def single(sch: Schema, evidence: str) -> dict[str, Any]:
    """The `single` reply: entity lists at the top level, the rest of Study under `study`.
    `evidence` is one of `EVIDENCE_FORMATS`."""
    builder = _Builder(sch, evidence)
    _names, keep = render.mode_classes(sch, "single")
    by_container = sch.classes_by_container()
    # `analyses` first: strict decoding writes keys in schema order, and `SINGLE_NOTE` asks
    # for the analyses before every entity list. Made to start elsewhere, replies came back
    # with the lists empty.
    lists = [k for k in render.payload_keys("single") if k in by_container]
    lists.sort(key=lambda key: key != "analyses")
    root = {key: {"type": "array", "items": builder.nested(by_container[key])} for key in lists}
    root["study"] = builder.entity("Study", [k for k in keep if k not in lists])
    # `key`, as the listing instructions spell it and `render.unconsumed_listing` reads it.
    root["omitted"] = {"type": "array", "items": _object(
        {"key": {"type": "string"}, "reason": {"type": "string"}})}
    return {**_object(root), "$defs": builder.defs}


@lru_cache(maxsize=4)
def for_single(evidence: str) -> dict[str, Any]:
    """`single` against the extraction schema, built once per run."""
    from pondie.schema import reader

    return single(reader.load(schema.EXTRACTION), evidence)


def fill(sch: Schema, rows: Sequence[Mapping[str, Any]], cite: bool = False) -> dict[str, Any]:
    """The `fill` reply: every asked id, answered with a value of its type or a reason.
    `cite` adds the numbers of the sentences that state a value (`evidence.cited`)."""
    reasons = {"type": "string", "enum": sorted(VOCABULARY | {PLAIN})}
    answers = {}
    for row in rows:
        typed = _value_type(sch, row.get("ranges") or [row["range"]], bool(row["multivalued"]))
        value = {"value": typed}
        if cite:
            value["evidence"] = {"type": "array", "items": {"type": "integer"}}
        answers[row["id"]] = {"anyOf": [_object(value),
                                        _object({"unreported_reason": reasons})]}
    return _object(answers)


def evidence(ids: Sequence[str]) -> dict[str, Any]:
    """The `evidence` reply: a quote per asked id, or null where the paper states none."""
    return _object({i: _nullable({"type": "string"}) for i in ids})
