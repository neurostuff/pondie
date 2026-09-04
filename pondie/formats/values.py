"""The `ExtractedValue` wrapper: what one is, how to read one, how to make one.

Every value a model reads off a paper is wrapped, because a record is not a set of answers
but a set of *claims with warrant*: the value, where it came from, and the span that supports
it. The wrapper is what carries the second and third.

    {"extraction_status": "extracted",
     "value": 42,
     "value_source": "reported",
     "evidence": {"status": "present", "sets": [{"quotes": ["forty-two patients"]}]}}

This module exists because five modules had each written their own unwrapper and **they did
not agree**. `derive_fields` returned `None` for any mapping without a `value` key, including
ones that were not wrappers at all; `validate_record` returned `None` for anything that was
not a mapping, so a bare string that had escaped repair read as absent rather than as the
wrong shape; `build_record` and `derive_effect` passed non-wrappers through unchanged. Three
answers to one question, none of them wrong on its own inputs and no two agreeing on the
edges -- which is how a record that is subtly malformed reads as a record that is empty.

`read` is the one answer. For a *schema-aware* read -- one that knows whether a slot is
multivalued, and returns the `NOT_REPORTED` sentinel rather than `None` -- use
`value_of` below, which takes the shape from the schema instead of guessing it.
The two are different jobs: this one is structural and needs no schema loaded.

It sits at the top of the package, beside `text_index` and `table_parse`, because it is the
same kind of thing: what a record is made of. It was under `pondie.extraction`, which made
every consumer of a record -- the query engine, normalization, the benchmark, and the schema
reader itself -- import the extraction package to read one. The schema reader
importing it closed a cycle outright.
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
#:
#: These are the schema's `ValueSource` vocabulary and nothing else. An earlier version of
#: this line invented a third, `derived`, on the reasoning that a computed value is not a
#: generated one -- and `derive_fields --fill` duly stamped it onto eight fields of every
#: record in the corpus, each of which the validator then rejected with
#: `value_source must be one of ['generated', 'reported']`. Stating the wrapper's shape is
#: this module's whole job, so the vocabulary is the schema's, checked against it.
ValueSource = Literal["reported", "generated"]

#: `not_applicable` means there is no sentence to quote -- the value came from a table
#: manifest or from arithmetic. `not_found` means there should be one and neither locator
#: placed it, which is a defect a reviewer should see rather than a silence.
EvidenceStatus = Literal["present", "not_found", "not_applicable"]

#: `not_reported` is a positive assertion that the paper is silent. It is NOT the same as
#: the field being absent, and conflating the two is the error this vocabulary exists to
#: prevent: absent means nothing was asked.
ExtractionStatus = Literal["extracted", "not_reported"]

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
    """One claim: a value, how it got there, and what warrants it.

    Built through here rather than as a dict literal so the four keys cannot drift apart --
    a wrapper missing `evidence` fails validation at the end of a run rather than at the line
    that wrote it, because `evidence` is REQUIRED on every `ExtractedValue` in the schema.
    """

    model_config = ConfigDict(extra="allow")

    extraction_status: ExtractionStatus
    value: Any = None
    value_source: ValueSource | None = None
    evidence: Evidence

    def as_field(self) -> dict[str, Any]:
        """The dict a record holds. `value` is dropped when nothing was reported."""
        out = self.model_dump(exclude_none=True)
        if self.extraction_status != "extracted":
            out.pop("value", None)
            out.pop("value_source", None)
        return out


def wrap(value: Any, *, source: ValueSource, evidence: EvidenceStatus) -> dict[str, Any]:
    """A wrapper around `value`, or a `not_reported` one when there is no value.

    Both keywords are REQUIRED, with no default, and that is the point. `evidence` first
    defaulted to `not_applicable` -- and `extracted` + `not_applicable` is a hard schema
    error ("extracted fields must not have evidence.status not_applicable"), so the default
    was wrong for every caller that passes a value. Inside the pipeline it was masked,
    because `_resolve_field` rewrites that pair to `not_found` on the way past; outside it,
    `tools/` runs on built records that never reach that walk, and `derive_fields --fill`
    wrote eight invalid fields into every record in the corpus.

    A default that only works for the callers who ignore it is worse than no default. There
    are two honest answers and the caller knows which: `not_applicable` when nothing could
    warrant the value because it did not come from prose (a table caption, a mirrored
    contrast the paper never describes), and `not_found` when a sentence should exist and
    no locator placed it.

    `None` and `""` become `not_reported` because that is what they mean coming out of a
    manifest or a deriver: the field was asked for and the source did not carry it. An empty
    *list* is different -- it is an extracted answer of "none" -- so it is not folded in.
    """
    if value is None or value == "":
        return ExtractedValue(
            extraction_status="not_reported", evidence=Evidence(status="not_applicable")
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
    """The value a wrapper asserts, or the node itself when it is not a wrapper.

    Pass-through and not `None` for a non-wrapper, deliberately: a bare scalar where a
    wrapper belongs is a record that escaped the `wrappers` repair, and returning `None`
    reports it as a field the paper did not mention. The two are different defects and only
    one of them is the extractor's fault.

    A `not_reported` wrapper carries no `value` key, so this yields `None` for it -- which is
    what a withheld direction is.
    """
    return node.get("value") if is_field(node) else node


def read_scalar(node: Any) -> Any:
    """`read`, repeated until the answer is not a wrapper. For a slot that should hold one.

    A wrapper whose `value` is itself a wrapper is a real and recurring shape -- the model
    emits it when a field is described as an ExtractedValue and it wraps the answer twice --
    and a caller that unwraps once gets a dict where it expected a string, then breaks on
    the first thing that groups by it. Two call sites wrote the double unwrap inline; this
    is the name for it.

    Bounded, because a cycle here would hang a reader whose job is to report on bad input.
    """
    for _ in range(4):
        unwrapped = read(node)
        if unwrapped is node or not is_field(unwrapped):
            return unwrapped
        node = unwrapped
    return node


def is_reported(node: Any) -> bool:
    """Whether the wrapper claims a value at all, as opposed to asserting silence."""
    return is_field(node) and node.get("extraction_status") == "extracted"


def iter_fields(node: Any, path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """Every wrapper in a payload, with the dotted path the builder reports it under.

    Descends no further once it finds one: a wrapper's `evidence` holds mappings of its own
    and they are not fields of the record.
    """
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
    """A slot the paper did not report: a claim, not an absence.

    Falsy, empty as a string and empty when iterated, so a caller that only wants the words
    is unaffected; identifiable by `is NOT_REPORTED` so a caller that must tell it from an
    absent slot, or from a slot the paper reported as empty, still can. It is not replaced by
    `[]` on a multivalued slot, because "did not say" and "said none" are different claims and
    collapsing them is the error this accessor exists to prevent.
    """

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
    """The value inside an ExtractedValue wrapper, in the shape its slot declares.

    Reading a record by hand gets three cases wrong, and each is a silent wrong answer
    rather than an error:

      * a wrapper with no `value` key is `not_reported` -- a positive claim that the paper
        did not say. Returning the wrapper makes it look like a scalar value.
      * a `multivalued` slot holds a list. A caller that takes the value as a scalar drops
        every entry but reads as if it succeeded.
      * an absent slot and a reported-empty slot are different; only the latter is a claim.

    `multivalued` comes from the schema -- see `slot_value` -- and is not guessed from
    whether this particular record happens to hold a list.
    """
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


def slot_value(sch: "Schema", class_name: str, entity: Mapping[str, Any], slot: str) -> Any:
    """`value_of` with the slot's shape taken from the schema rather than from the caller."""
    attribute = sch.attributes(class_name).get(slot)
    return value_of(entity.get(slot), bool(attribute.multivalued) if attribute else False)


#: What a model writes when it means yes or no. Spelled out because the answer arrives as
#: whatever word the paper used, and `bool("false")` is True.
_TRUE = frozenset({"true", "yes", "y", "1"})
_FALSE = frozenset({"false", "no", "n", "0"})


def cast(sch: "Schema", class_name: str, slot: str, value: Any) -> Any:
    """`value` in the type and vocabulary its slot declares, or `None` if it will not fit.

    Models answer in the paper's words and the schema does not: `Group.is_healthy` was
    given "true" and `Group.acquired_count` "31", each a correct reading of the paper and
    each a type the wrapper rejects -- `ExtractedBoolean.value must be a boolean, got str`.
    A closed vocabulary is the same problem one level up: `prespecification` was given
    "post-hoc", which describes the analysis accurately and is not one of the two values the
    field holds.

    `None` means *do not write this*, and is the point of the function. Coercing an answer
    that will not fit is how a type error becomes a wrong value: `int(float("about 20"))`
    raises, but a `bool()` of anything non-empty does not, and a slot given "mostly" would
    quietly become True.

    A slot written `any_of: [SomeEnum, string]` is deliberately open -- `region_type` takes
    "gray matter" when the source says so -- and has no single `range`, so it is left alone.
    """
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
    # Through the wrapper: `is_healthy` declares `ExtractedBoolean`, so reading `range`
    # directly gave "ExtractedBoolean" and never "boolean", and the branch below -- written
    # for exactly the error in this docstring -- was unreachable.
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
    """`cast`, in the multiplicity the slot declares.

    A cast value still has to arrive in the shape the slot holds: `Task.response_mode` is
    multivalued, and writing the scalar produced "ExtractedResponseModeList.value must be a
    list of ResponseMode or string, got str".
    """
    result = cast(sch, class_name, slot, value)
    if result is None:
        return None
    attribute = sch.attributes(class_name).get(slot)
    if attribute is not None and attribute.multivalued and not isinstance(result, list):
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
    "is_reported",
    "iter_fields",
    "read",
    "read_scalar",
    "wrap",
    "NOT_REPORTED",
    "value_of",
    "slot_value",
    "cast",
    "shape",
]
