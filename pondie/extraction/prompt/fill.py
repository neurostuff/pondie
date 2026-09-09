"""Ask for the slots that are still open, one slot at a time, until none are.

The other two model passes are shaped around *entities*: `demands` decides which analyses
exist, `satisfy` emits the entities they declared. Both render a class schema and ask for
records back, which is the right shape for deciding what exists and the wrong one for
finishing what already does -- a second `satisfy` re-emits whole entities to add one field,
and its answer is judged by whether the entity came back, not by whether the slot did.

This pass names the slots. Each open slot is one line with its path, its type and the
sentence the schema uses to describe it, and the reply is one entry per line: a value, or
a reason there is none. That makes the loop's exit condition decidable, which is the whole
point -- a slot is *settled* when it holds a value or an `unreported_reason`, and
`unsettled` returns exactly the ones that are neither. Iterating until that set is empty
terminates on a fact about the record rather than on a guess about progress.

`undetermined` is the one reason that does not settle a slot. It is the model saying it
could not work the slot out, which is a different claim from the paper being silent, and
it is what a further round is for. The other three reasons are claims about the source, and
a round that re-asked them would be inviting the model to overwrite a considered answer.

Plain silence settles a slot and records no reason at all: `not_reported` already carries
that claim. The prompt still names the case, because a model offered only the unusual
reasons will reach for the nearest one rather than the right one.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, MutableMapping, Sequence

from pondie.formats import values
from pondie.schema.reader import Schema

#: The schema's `UnreportedReason`, restated so a bad answer is refused where it is written
#: rather than at the end of the run.
VOCABULARY = frozenset({"ambiguous", "outside_text", "cited_elsewhere", "undetermined"})

#: The reason that leaves a slot open. Every other value of `UnreportedReason` is a claim
#: about the paper, and re-asking it would invite the model to overwrite its own finding.
OPEN = "undetermined"

#: What the model answers for plain silence, and not a member of `UnreportedReason`. The
#: schema has no token for the ordinary case: `not_reported` already says the attribute was
#: examined and the source carries no value, so a reason repeating it would add a word and no
#: fact. The prompt still needs a way to *say* the ordinary case, or a model with only the
#: four unusual reasons picks the nearest wrong one -- the mistake `Table.purpose` paid for.
#: So it is asked for by name here and written as a bare `not_reported`.
PLAIN = "silent_default"

SYSTEM = """You fill in specific missing fields of a structured record read from a paper.

You are given the paper, and a list of fields that were left blank. Each line is one
field: its id, the type it must take, and what it means. Answer every line.

Rules:
1. Emit ONE JSON object mapping id -> answer. No prose, no markdown fence.
2. An answer is EITHER {"value": <value>} when the paper supports one, OR
   {"unreported_reason": "<reason>"} when it does not. Never both, never neither. For plain
   silence -- the paper simply does not mention it -- the reason is "silent_default", which
   records the field as examined and carrying no value, and adds no claim beyond that.
3. A reason is a complete and correct answer. These fields were already looked at once
   and left blank; most of them are blank because the paper does not carry them. You are
   not expected to find a value for every line and you must not invent one to fill it.
4. The reasons, and they say different things:
   - "silent_default"   the paper does not mention it. The ordinary case, recorded as a
                        bare `not_reported` with no reason attached.
   - "ambiguous"        the paper addresses it but settles on no single value.
   - "outside_text"     it is in a figure, an image-only table, or a supplement.
   - "cited_elsewhere"  the paper gives it by reference to another publication.
   - "undetermined"     you could not work it out. Use this rather than "silent_default"
                        whenever you did not establish that the paper says nothing.
5. Respect the type. A field marked integer takes a bare number, not a sentence; a field
   with a listed vocabulary takes one of the listed terms.
6. A LISTED TERM BEATS A REASON. Where a field lists a vocabulary, read every term before
   reaching for a reason: these vocabularies carry terms for the awkward cases, and one of
   them is usually the answer. A reason is for when the paper does not support any term,
   not for when choosing between them takes thought. Answering "ambiguous" where a listed
   term describes the situation records the field as blank when the paper settled it.
7. Do not answer under an id that was not given to you. Ids you invent are discarded."""


def _label(entity: Mapping[str, Any]) -> str:
    """What to call this entity, so the model knows which one it is answering about.

    A name when it has one. When it does not, whatever it already holds that tells it
    apart from its siblings -- which for a `Cell` is the term and the level, and is the
    difference between a question and a guess. Falling back to the bare `local_id` (a
    `Cell` has none, so this printed "Cell ?") asks the model to say which way
    `cells[0]` went without telling it what `cells[0]` is: measured on pMZeVGA2rQQi, it
    returned the right set of directions for each contrast and assigned them to the
    wrong cells, which is what guessing looks like when the guesser knows the shape.
    """
    for key in ("name", "title", "label", "source_label"):
        got = values.read(entity.get(key))
        if isinstance(got, str) and got.strip():
            return got.strip()[:60]

    marks: list[str] = []
    for key, held in entity.items():
        if key in ("local_id", "id") or len(marks) >= 3:
            continue
        if isinstance(held, str) and held.strip():  # a reference, by local_id
            marks.append(f"{key}={held.strip()[:34]}")
        elif values.is_field(held):
            got = values.read(held)
            if got not in (None, "", []):
                marks.append(f"{key}={str(got)[:34]}")
    if marks:
        return " ".join(marks)
    return str(entity.get("local_id") or "?")


def _entities(
    payload: Mapping[str, Any], sch: Schema
) -> Iterator[tuple[str, str, MutableMapping[str, Any]]]:
    """(path, class name, entity) for every entity in a payload, nested ones included."""

    def walk(node: Any, cls: str, path: str):
        if not isinstance(node, dict):
            return
        yield path, cls, node
        for name, slot, kind in sch.iter_slots(cls):
            if kind != "nested":
                continue
            inner = str(sch.ranges(slot)[0] if sch.ranges(slot) else slot.range or "")
            child = node.get(name)
            # A singular nested object takes no subscript. `Analysis.effect` is one object,
            # not a list of one, and writing it `effect[0]` produced a path `_resolve` could
            # not follow -- it sees the bracket, demands a list, finds a dict and gives up.
            # Every slot under such an object was then listed to the model and had its answer
            # silently discarded, which is most of `Cell`: cells hang off `effect`.
            if not sch.is_multivalued(cls, name):
                if isinstance(child, dict):
                    yield from walk(child, inner, f"{path}.{name}")
                continue
            for index, item in enumerate(child if isinstance(child, list) else [child]):
                if isinstance(item, dict):
                    key = item.get("local_id") or index
                    yield from walk(item, inner, f"{path}.{name}[{key}]")

    containers = {c: k for c, k in sch.containers().items() if "." not in k}
    for cls, key in containers.items():
        for index, entity in enumerate(payload.get(key) or []):
            if isinstance(entity, dict):
                ident = entity.get("local_id") or index
                yield from walk(entity, cls, f"{key}[{ident}]")


def unsettled(payload: Mapping[str, Any], sch: Schema) -> list[dict[str, Any]]:
    """Every slot that holds neither a value nor a reason there is none.

    Two states qualify, and they are different failures. A slot with no wrapper at all was
    never answered -- `thin` counts these and nothing else. A slot whose wrapper says
    `undetermined` was answered with "I could not tell", which is a positive claim and a
    revisable one. Everything else is settled: a value, or one of the four reasons that
    describe the paper rather than the pass.
    """
    out: list[dict[str, Any]] = []
    for path, cls, entity in _entities(payload, sch):
        label = _label(entity)
        for name, slot, kind in sch.iter_slots(cls):
            if kind in ("identifier", "reference", "nested"):
                continue
            held = entity.get(name)
            if values.is_field(held):
                if held.get("extraction_status") != "not_reported":
                    continue
                if held.get("unreported_reason") != OPEN:
                    continue
            elif held not in (None, "", [], {}):
                # An answer in the wrong shape is still an answer. The extraction passes
                # emit some slots as bare scalars -- `Cell.direction` comes back as
                # `"positive"` rather than a wrapper -- and `repairs.wrappers` puts them
                # right at build time, which is after this stage. Reading the bare form as
                # absence made this loop destructive: on pMZeVGA2rQQi it offered all 12
                # cell directions as open and overwrote them with `ambiguous`, six of which
                # the extraction had got right. Shape is the repair pass's business; what
                # is open here is what holds nothing at all.
                continue
            # `value_ranges`, not `ranges`: the slot's range is the wrapper
            # (`ExtractedTablePurpose`), and what the model has to satisfy is the type
            # inside it (`TablePurpose`). Naming the wrapper would ask for the wrong shape.
            inner = sch.value_ranges(slot) or ["string"]
            enum = sch.enums.get(inner[0])
            out.append(
                {
                    "id": f"{path}.{name}",
                    "owner": f"{cls} {label}",
                    "range": inner[0],
                    "multivalued": sch.is_multivalued(cls, name),
                    "description": (slot.description or "").strip(),
                    "vocabulary": sorted((enum.permissible_values or {}).keys()) if enum else [],
                    # What the terms mean, for the vocabularies small enough to gloss. Names
                    # alone are not enough where the awkward cases have their own term:
                    # `Direction` offers `held` for a level the contrast was taken *within*
                    # and `undirected` for a test with no per-level sign, and a model shown
                    # only "one of: held, negative, positive, undirected" cannot know that,
                    # so it answers `ambiguous` and the slot reads blank on a paper that
                    # settled it. Measured on 12 cells of pMZeVGA2rQQi: every one of them.
                    "glossary": (
                        {
                            k: (v.description or "").strip()
                            for k, v in (enum.permissible_values or {}).items()
                        }
                        if enum and len(enum.permissible_values or {}) <= 6
                        else {}
                    ),
                }
            )
    return out


def block(rows: Sequence[Mapping[str, Any]]) -> str:
    cap = 400
    """The open slots as the pass's question. One line each, truncated to `cap` lines."""
    lines = [
        "\n# Fields left blank\n",
        f"{min(len(rows), cap)} field(s). Answer every id with a value or a reason.\n",
    ]
    owner = None
    for row in rows[:cap]:
        if row["owner"] != owner:
            owner = row["owner"]
            lines.append(f"\n## {owner}")
        kind = row["range"] + (" (list)" if row["multivalued"] else "")
        why = row["description"].replace("\n", " ")
        if row.get("vocabulary"):
            why += "  one of: " + ", ".join(row["vocabulary"][:24])
        lines.append(f"- {row['id']}  [{kind}]  {why[:260]}")
        for term, gloss in (row.get("glossary") or {}).items():
            if gloss:
                lines.append(f"      {term}: {gloss.replace(chr(10), ' ')[:150]}")
    return "\n".join(lines) + "\n"


def apply_fill(
    payload: MutableMapping[str, Any],
    answers: Mapping[str, Any],
    open_ids: Sequence[str],
) -> tuple[int, int, int]:
    """Write answers into the slots they name. Returns (values, reasons, discarded).

    Only into slots this round actually asked about. An id the pass did not offer is
    discarded rather than created: a model that answers under a path of its own invention
    is describing a record that does not exist, and writing it would put a field on an
    entity the schema never gave one.
    """
    allowed = set(open_ids)
    filled = reasoned = dropped = 0
    for ident, answer in (answers or {}).items():
        if ident not in allowed or not isinstance(answer, Mapping):
            dropped += 1
            continue
        target, name = _resolve(payload, ident)
        if target is None:
            # `unsettled` offered this path, so `_resolve` must be able to follow it. That
            # the two disagreed is a bug here and not an answer to discard: `effect[0]`
            # went out to the model and came back to nothing, so every `Cell` slot looked
            # unanswerable while the model had in fact answered. A count folded into
            # `dropped` hid it -- an id the model invented reads the same way.
            raise AssertionError(
                f"{ident} was offered as open and cannot be resolved; `unsettled` and "
                f"`_resolve` disagree about how to address it"
            )
        # Belt and braces, because the failure this guards is silent and destroys data
        # rather than merely wasting a call. `unsettled` should never offer a slot that
        # holds anything, but it did -- a bare `"positive"` is not a wrapper and read as
        # empty -- and by the time the payload is written the old value is gone.
        existing = target.get(name)
        if existing not in (None, "", [], {}) and not (
            values.is_field(existing)
            and existing.get("extraction_status") == "not_reported"
            and existing.get("unreported_reason") == OPEN
        ):
            dropped += 1
            continue
        if "value" in answer and answer["value"] not in (None, ""):
            # Built here rather than through `values.wrap`, which requires an evidence
            # status: this pass does not know one. `Evidence` runs after it and puts the
            # block on, the same as it does for anything `satisfy` emitted.
            target[name] = {
                "extraction_status": "extracted",
                "value": answer["value"],
                "value_source": "reported",
            }
            filled += 1
        elif answer.get("unreported_reason"):
            reason = str(answer["unreported_reason"])
            # Checked before anything is written. Writing the status first and rejecting the
            # reason afterwards left the slot settled on an answer that had just been
            # refused, which is worse than either outcome on its own.
            if reason != PLAIN and reason not in VOCABULARY:
                dropped += 1
                continue
            # `PLAIN` is how the prompt says "the ordinary case" and is not a schema value,
            # so it becomes no reason at all -- the bare status it stands for.
            target[name] = values.wrap(
                None,
                source="reported",
                evidence="not_applicable",
                reason=None if reason == PLAIN else reason,
            )
            reasoned += 1
        else:
            dropped += 1
    return filled, reasoned, dropped


def _segments(ident: str) -> list[str]:
    """`ident` split on the dots that separate slots, and not on any others.

    A `local_id` may contain a dot -- `mod_akt1_allele.trm_akt1_allele_count` namespaces a
    term under its model, and 10 of 6,060 ids in the corpus do something like it. Splitting
    the whole string on "." tore that id in half, so `_resolve` looked for a term called
    `mod_akt1_allele` among the terms, found none, and returned nothing for a path
    `unsettled` had just offered. The assertion below then took the paper down: one dotted id
    cost the whole of `rxaz3qhEmJhx`.

    Only dots outside brackets separate. This is the grammar `_entities` writes, stated once
    so the two halves cannot drift again.
    """
    out, depth, start = [], 0, 0
    for index, char in enumerate(ident):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char == "." and depth == 0:
            out.append(ident[start:index])
            start = index + 1
    out.append(ident[start:])
    return out


def _resolve(payload: Any, ident: str) -> tuple[MutableMapping[str, Any] | None, str]:
    """The container holding `ident`'s last segment, and that segment's name."""
    node: Any = payload
    parts = _segments(ident)
    for part in parts[:-1]:
        key, _, rest = part.partition("[")
        node = node.get(key) if isinstance(node, dict) else None
        if rest:
            want = rest.rstrip("]")
            if not isinstance(node, list):
                return None, ""
            found = None
            for index, item in enumerate(node):
                if not isinstance(item, dict):
                    continue
                if str(item.get("local_id")) == want or str(index) == want:
                    found = item
                    break
            node = found
        if node is None:
            return None, ""
    return (node, parts[-1]) if isinstance(node, dict) else (None, "")
