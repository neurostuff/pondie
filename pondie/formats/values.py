"""The `ExtractedValue` wrapper: what one is, how to read one, how to make one.

Every value a model reads off a paper is wrapped, because a record is not a set of answers
but a set of *claims with warrant*: the value, where it came from, and the span that supports
it. The wrapper is what carries the second and third.

    {"extraction_status": "extracted",
     "value": 42,
     "value_source": "reported",
     "evidence": {"status": "present", "sets": [{"quotes": ["forty-two patients"]}]}}

A slot with no value carries the same wrapper without one, and says why it is blank:

    {"extraction_status": "not_reported",
     "unreported_reason": "outside_text",
     "evidence": {"status": "not_applicable"}}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    # Type-only, and deliberately: `formats` is the bottom of the package and importing the
    # schema reader for real would put it above `schema` instead of below it.
    from pondie.schema.reader import Schema

#: How a value came to be in the record. `reported` is read off the paper; `generated` is
#: minted by the pipeline, which is what a mirrored contrast's direction and a code-derived
#: field both are.
ValueSource = Literal["reported", "generated"]

#: `not_applicable` means there is no sentence to quote -- the value came from a table
#: manifest or from arithmetic. `not_found` means there should be one and neither locator
#: placed it, which is a defect a reviewer should see rather than a silence.
EvidenceStatus = Literal["present", "not_found", "not_applicable"]

#: `not_reported` is a positive assertion that the paper is silent. It is NOT the same as
#: the field being absent, and conflating the two is the error this vocabulary exists to
#: prevent: absent means nothing was asked.
ExtractionStatus = Literal["extracted", "not_reported"]

#: The same two, as a tuple to test membership against. Named here because this module owns
#: the wrapper contract, and `fix.repair_wrappers` and `evidence.warrant` both ask.
STATUSES: tuple[str, str] = ("extracted", "not_reported")

UnreportedReason = Literal["ambiguous", "outside_text", "cited_elsewhere", "undetermined"]

#: The key that makes a mapping a wrapper. Structural, so it is checked and not inferred.
MARKER = "extraction_status"


class Evidence(BaseModel):
    """Where in the paper a value is warranted, if anywhere."""

    model_config = ConfigDict(extra="allow")

    status: EvidenceStatus
    #: Omitted rather than empty when there is nothing to point at. An empty `sets` is a
    #: key the hand-built wrappers never wrote, and a record is compared key by key.
    sets: list[dict[str, Any]] | None = None


class ExtractedValue(BaseModel):
    """Holds the claim and its evidence."""

    model_config = ConfigDict(extra="allow")

    extraction_status: ExtractionStatus
    value: Any = None
    value_source: ValueSource | None = None
    unreported_reason: UnreportedReason | None = None
    evidence: Evidence

    def as_field(self) -> dict[str, Any]:
        """The dict a record holds. `value` is dropped when nothing was reported."""
        out = self.model_dump(exclude_none=True)
        if self.extraction_status != "extracted":
            out.pop("value", None)
            out.pop("value_source", None)
        else:
            # There is no blank to explain on an extracted value, and the validator rejects
            # one. Dropped rather than refused so a caller reusing a wrapper it just filled
            # cannot silently invalidate the record.
            out.pop("unreported_reason", None)
        return out


def wrap(
    value: Any,
    *,
    source: ValueSource,
    evidence: EvidenceStatus,
    reason: UnreportedReason | None = None,
) -> dict[str, Any]:
    """A wrapper around `value`, or a `not_reported` one when there is no value."""
    if value is None or value == "":
        return ExtractedValue(
            extraction_status="not_reported",
            unreported_reason=reason,
            evidence=Evidence(status="not_applicable"),
        ).as_field()
    return ExtractedValue(
        extraction_status="extracted",
        value=value,
        value_source=source,
        evidence=Evidence(status=evidence),
    ).as_field()


def is_field(node: Any) -> bool:
    """Whether this node is a wrapper, by the key that defines one."""
    return isinstance(node, Mapping) and MARKER in node


def read(node: Any) -> Any:
    """The value a wrapper asserts, or the node itself when it is not a wrapper."""
    return node.get("value") if is_field(node) else node


def iter_fields(node: Any, path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """Every wrapper in a payload, with the dotted path the builder reports it under."""
    if isinstance(node, Mapping):
        if MARKER in node:
            yield path, node  # type: ignore[misc]
            return
        for key, value in node.items():
            yield from iter_fields(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_fields(value, f"{path}[{index}]")


# ---------------------------------------------------------------------------------------
# Schema-aware reading
# ---------------------------------------------------------------------------------------
#
# `read` above answers the structural question and needs no schema. These answer the one
# that needs the slot's declared shape, and they lived in the schema package -- next to
# the code that reads the schema, rather than next to the code that reads a value. Reading
# a wrapper is one subject and this is where it is.


class _NotReported:
    """A slot the paper did not report: a claim, not an absence."""

    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        return ""

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "NOT_REPORTED"


#: The one instance; compare with `is`.
NOT_REPORTED = _NotReported()


def value_of(node: object, multivalued: bool = False) -> object:
    """The value inside an ExtractedValue wrapper."""
    if node is None:
        return [] if multivalued else None
    if isinstance(node, Mapping):
        if "value" not in node:
            return NOT_REPORTED
        value = node["value"]
    else:
        value = node
    if multivalued:
        if isinstance(value, list):
            return value
        return [] if value is None else [value]
    return value


#: What a model writes when it means yes or no. Spelled out because the answer arrives as
#: whatever word the paper used, and `bool("false")` is True.
_TRUE = frozenset({"true", "yes", "y", "1"})
_FALSE = frozenset({"false", "no", "n", "0"})


def cast(sch: "Schema", class_name: str, slot: str, value: Any) -> Any:
    """Intelligent casting for true/false and for other elements"""

    attribute = sch.attributes(class_name).get(slot)
    if attribute is None:
        return None
    if isinstance(value, list):
        # Element-wise, and all or nothing. `str()` of a list is the list's repr, so a
        # multivalued slot given ["a", "b"] took the single string "['a', 'b']" -- one bogus
        # value where two belong, and legal enough that the validator passed it.
        cast_items = [cast(sch, class_name, slot, item) for item in value]
        return None if any(item is None for item in cast_items) else cast_items
    text = str(value).strip()
    ranges = sch.value_ranges(attribute)
    single = ranges[0] if len(ranges) == 1 else None

    if single == "boolean":
        low = text.lower()
        return True if low in _TRUE else False if low in _FALSE else None
    if single == "integer":
        try:
            return int(float(text))
        except ValueError:
            return None
    if single == "float":
        try:
            return float(text)
        except ValueError:
            return None
    permissible = getattr(sch.enums.get(single or ""), "permissible_values", None)
    if permissible and text not in permissible:
        return None
    return text


def shape(sch: "Schema", class_name: str, slot: str, value: Any) -> Any:
    """Ensure list response if field allows multiple values."""
    result = cast(sch, class_name, slot, value)
    if result is None:
        return None
    if sch.is_multivalued(class_name, slot) and not isinstance(result, list):
        return [result]
    return result


__all__ = [
    "Evidence",
    "EvidenceStatus",
    "ExtractedValue",
    "ExtractionStatus",
    "MARKER",
    "ValueSource",
    "is_field",
    "iter_fields",
    "read",
    "wrap",
    "NOT_REPORTED",
    "value_of",
    "cast",
    "shape",
]
