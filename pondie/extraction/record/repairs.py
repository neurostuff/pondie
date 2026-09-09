"""The deterministic transformations applied to a record, in order, with their reasons.

`build_record` performs nine of these and its ordering carries real constraints -- the
direction fill matches a level against a contrast name and so must run after levels are
aligned; the mirror is taken from the corrected record and so must run last. Those
constraints lived in comments beside consecutive statements, which is a fine place to
state them and a bad place to enforce them: nothing stopped a tenth repair being inserted
in the wrong place, and nothing could report which ones fired without parsing a summary
line.

Here the sequence is data. Each repair carries what it does, why it runs where it does,
and what it changed on this record, so `RepairLog.explain()` answers "what did the builder
do to this paper" without a log.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pondie.schema.reader import Schema


@dataclass(frozen=True)
class Context:
    """What a repair may need beyond the record itself."""

    schema: Schema
    stage1: Path | None = None
    table_map: Path | None = None


#: A repair reports what it changed, one line per change, and mutates the record in
#: place. An empty list means it found nothing to do, which is different from not running.
Apply = Callable[[dict, Context], "list[str]"]


@dataclass(frozen=True)
class Repair:
    """One named deterministic fix, and why it sits where it does in the order."""

    name: str
    what: str
    apply: Apply
    #: Empty when the repair may run anywhere. Stated when it may not, because an
    #: ordering constraint that is only a comment is one an edit can silently break.
    after: str = ""
    #: The earliest payload that holds everything this repair reads, so it can run there
    #: rather than waiting for the merge. `demands` and `satisfy` name a single payload;
    #: `merged` means it needs analyses, entities and tables together.
    #:
    #: The point is `fill`, which runs between `satisfy` and the merge and asks a model
    #: about open slots. Measured over 15 papers, 215 of the 1,774 slots it offered were
    #: ones a repair fills later -- 100 of them `source_table_analysis`, a join key read
    #: off the stage-1 parse that a model can only guess at. Running a repair where its
    #: inputs are ready takes those off the ask.
    stage: str = "merged"


@dataclass
class RepairLog:
    """What each repair did to one record."""

    entries: list[tuple[str, list[str]]] = field(default_factory=list)

    def record(self, name: str, changes: list[str]) -> None:
        self.entries.append((name, changes))

    def changes(self, name: str) -> list[str]:
        return [line for entry, lines in self.entries if entry == name for line in lines]

    @property
    def total(self) -> int:
        return sum(len(lines) for _name, lines in self.entries)

    def fired(self) -> list[str]:
        return [name for name, lines in self.entries if lines]

    def explain(self) -> str:
        if not self.total:
            return "no repairs fired"
        lines = []
        for name, changed in self.entries:
            if not changed:
                continue
            lines.append(f"{name} ({len(changed)}):")
            lines += [f"    {line}" for line in changed[:5]]
            if len(changed) > 5:
                lines.append(f"    ... and {len(changed) - 5} more")
        return "\n".join(lines)


def build_sequence() -> tuple[Repair, ...]:
    """The order, with each constraint stated next to the repair it binds.

    Imported lazily so this module can be read, and its ordering checked, without pulling
    in the schema loader.
    """

    from pondie.extraction.record import builder as br

    return (
        Repair(
            "wrappers",
            "put a malformed ExtractedValue back into wrapper shape",
            lambda body, ctx: br.repair_wrappers(body),
            stage="shape",
        ),
        Repair(
            "unwrapped",
            "unwrap a wrapper the model put in a bare-scalar slot",
            lambda body, ctx: br.unwrap_plain_slots(body, ctx.schema),
            after="wrappers",
            stage="merged",
        ),
        Repair(
            "table_effects",
            "mark a table an analysis cites as reporting that analysis's effect",
            lambda body, ctx: br.derive_table_effects(body),
            stage="merged",
        ),
        Repair(
            "denominators",
            "fill a distribution's denominator from its count and percentage",
            lambda body, ctx: br.derive_denominators(body),
            stage="satisfy",
        ),
        Repair(
            "numbers",
            "turn a numeric string into the number its slot declares",
            lambda body, ctx: br.coerce_numeric_values(body, ctx.schema),
            after="unwrapped",
            stage="merged",
        ),
        Repair(
            "stray_tables",
            "move a Table written as a Study attribute into tables[]",
            lambda body, ctx: br.rehome_stray_tables(body, ctx.schema),
            stage="merged",
        ),
        Repair(
            "acquisition_type",
            "fill an acquisition's type from its own modality",
            lambda body, ctx: br.derive_acquisition_types(body),
            stage="satisfy",
        ),
        Repair(
            "coordinate_space",
            "fill the space stage 1 already read off the table",
            lambda body, ctx: br.derive_coordinate_spaces(body, ctx.stage1, ctx.table_map),
            stage="merged",
        ),
        Repair(
            "listified",
            "unwrap a nested slot the model wrote as an object",
            lambda body, ctx: br.listify_nested(body, ctx.schema),
            stage="shape",
        ),
        Repair(
            "listified_scalars",
            "wrap a lone scalar the slot declares multivalued",
            lambda body, ctx: br.listify_scalars(body, ctx.schema),
            after="listified",
            stage="shape",
        ),
        Repair(
            "cell_levels",
            "rewrite a cell's level to the declared level it folds to",
            lambda body, ctx: br.align_cell_levels(body),
            after="listified",
            stage="merged",
        ),
        Repair(
            "scoped_terms",
            "scope two models' identically-named terms by their model",
            lambda body, ctx: br.scope_duplicate_terms(body),
            stage="satisfy",
        ),
        Repair(
            "references",
            "repoint a dangling reference where the choice is forced",
            lambda body, ctx: br.repair_references(body, ctx.schema),
            after="scoped_terms",
            stage="merged",
        ),
        Repair(
            "cell_terms",
            "repoint a cell at the same-named term its model reaches",
            lambda body, ctx: br.repoint_out_of_scope_terms(body),
            after="listified",
            stage="merged",
        ),
        Repair(
            "source_links",
            "verify or fill each analysis's link to its parsed rows",
            lambda body, ctx: br.resolve_source_table_analysis(body, ctx.stage1),
            stage="demands",
        ),
        Repair(
            "derived_ids",
            "rename each analysis to an id the parse determines",
            lambda body, ctx: br.derive_analysis_ids(body),
            after="source_links",
            stage="demands",
        ),
        Repair(
            "directions",
            "fill a cell's direction from the contrast's own name",
            lambda body, ctx: br.fill_directions(body),
            after="cell_levels",
            stage="merged",
        ),
        Repair(
            "mirrored",
            "rebuild the reversed half of every sign-split contrast",
            lambda body, ctx: br.mirror_withheld(body, ctx.stage1),
            after="directions",
            stage="merged",
        ),
    )


#: What runs where. `shape` is the normalisation that can only make a slot more settled --
#: a bare scalar becomes a wrapper, a lone scalar becomes a list -- so running it before
#: `fill` is what stops the loop reading an answer in the wrong shape as no answer.
#:
#: `unwrapped` is the opposite direction and stays at the merge. It strips a wrapper from a
#: slot storage declares plain, and a `not_reported` wrapper carries no `value` to strip, so
#: the slot empties: five settled slots on 84rGLhCbUJTh became open, which is work handed to
#: a model rather than taken from it. `numbers` is constrained after it and follows it there.
#:
#: `shape` runs after every pass that writes, including `fill`; the others run once, at the earliest payload holding their
#: inputs. Names a group rather than open-coding a set at each call site, so adding a
#: repair means choosing a stage and nothing else.
AFTER_DEMANDS = ("shape", "demands")
AFTER_SATISFY = ("shape", "satisfy")
AFTER_FILL = ("shape",)
#: `shape` again, deliberately. It runs beside each pass that writes, which is what lets
#: `fill` see wrappers rather than bare scalars -- but a payload that reached the merge
#: without passing one of those, `tables.json` or anything a resumed run wrote before this
#: split existed, would arrive unnormalised. Idempotent, so the second run costs nothing
#: and the guarantee holds for every payload rather than for the ones with a producer.
AT_MERGE = ("shape", "merged")


def check_order(sequence: tuple[Repair, ...]) -> list[str]:
    """Every declared `after` is satisfied by the sequence as written."""
    seen: set[str] = set()
    problems = []
    for repair in sequence:
        if repair.after and repair.after not in seen:
            problems.append(f"{repair.name} must run after {repair.after}, and does not")
        seen.add(repair.name)
    return problems


def apply_all(
    body: dict,
    ctx: Context,
    sequence: tuple[Repair, ...] | None = None,
    stage: str | None = None,
) -> RepairLog:
    """Run the sequence in order, recording what each one changed.

    `stage` runs only the repairs whose inputs that payload holds. The order within a stage
    is the registry's, and the constraint check runs over the whole sequence rather than the
    slice -- a repair whose `after` sits in another stage is an ordering error even though
    no single call would notice.
    """
    sequence = build_sequence() if sequence is None else sequence
    broken = check_order(sequence)
    if broken:
        raise ValueError("; ".join(broken))
    wanted = ({stage} if isinstance(stage, str) else set(stage)) if stage is not None else None
    log = RepairLog()
    for repair in sequence:
        if wanted is not None and repair.stage not in wanted:
            continue
        log.record(repair.name, list(repair.apply(body, ctx)))
    return log
