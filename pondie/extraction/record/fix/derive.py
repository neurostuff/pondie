"""Filling a slot from something the record or the parse already says.

The second kind of fix. Nothing here reads the paper: every value is computed from another
field, from the stage-1 parse, or from the join between two entities the record declares. That
is what makes them deterministic, and it is why `Table.purpose`, `Cell.direction` and
`CategoryDistribution.denominator` are not asked of a model -- measured, a model shown eight
kinds of non-analysis and no way to say "it is an analysis" answered with the nearest one and
was wrong on 16 of 18 tables.

Each of these is a `Repair` in `fix.build_sequence`, which holds the order and the reason
each one sits where it does.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from pondie.extraction.record import direction
from pondie.extraction.record import ids
from pondie.extraction.record import spans as span_tools
from pondie.extraction.record import walk
from pondie.formats import parse_keys, values
from pondie.schema import reader
from pondie.schema.reader import Schema
from typing import Any
import json
import re


#: (class, slot) pairs whose value is a conclusion about the record's own structure rather
#: than anything a paper can be quoted saying. Deliberately short.
#:
#: `check_value_source_honesty` warns on `value_source: reported` with
#: `evidence.status: not_found`, and over 1,817 records it fires 43,772 times in 98.6% of
#: papers -- so nobody reads it, including on the four genuinely wrong values it was written
#: for (`family = electrophysiology` on a BOLD study, `spatial_scope = roi` nobody stated).
#: Most of the flood is these five slots: `ModelTerm.type` 3,037 of 3,041 (100%),
#: `Effect.kind` 464 of 467 (99%), `Cell.direction` 3,095 of 3,364 (92%), `FactorLevel.order`
#: 3,694 of 5,575 (66%).
#:
#: The test for membership is that `reported` is impossible **in principle**, not merely
#: unevidenced. No paper writes down that a term is continuous, that a level is the second
#: one, or that an effect's kind is a between-subject contrast -- `direction.py` and
#: `derive_effect_kind` and `derive_denominators` compute those. A slot a paper *could* have
#: stated and did not is the case the warning exists for and stays out: `tfce_used` at 57%,
#: `CategoryDistribution.percentage` at 67% and `Statistic.family` at 31% are all things a
#: results section prints, so an unevidenced one is worth a reviewer's eye rather than a
#: relabel.
CONCLUSION_SLOTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ModelTerm", "type"),
        ("Effect", "kind"),
        ("Cell", "direction"),
        ("FactorLevel", "order"),
        ("CategoryDistribution", "denominator"),
    }
)


def derive_acquisition_types(body: dict[str, Any]) -> list[str]:
    """Fill `Acquisition.acquisition_type` from each record's modality.

    The schema says of this slot: "Derived by the mapper from the modality value's
    `instantiates`, never extracted" (`acquisition.yaml`) -- so the builder owes it, and
    an extraction that leaves it out is not wrong, it is being taken at its word. Without
    it the record resolves only to the base `Acquisition`, where every modality-specific
    parameter the model *did* extract is undeclared: one omission, seven errors.

    The mapping is read out of the `Modality` enum rather than written down here, so a
    new permissible value arrives with its own subclass already attached.
    """

    modality_enum = reader.load(EXTRACTION_SCHEMA).enums.get("Modality")
    modality = (modality_enum.permissible_values or {}) if modality_enum else {}
    instantiates = {
        name: str(spec["instantiates"][0]).split(":")[-1]
        for name, spec in modality.items()
        if spec.instantiates
    }

    filled: list[str] = []
    for index, acquisition in enumerate(body.get("acquisitions") or []):
        if not isinstance(acquisition, dict) or acquisition.get("acquisition_type"):
            continue
        value = acquisition.get("modality")
        if isinstance(value, Mapping):
            value = value.get("value")
        target = instantiates.get(value)
        if target:
            acquisition["acquisition_type"] = target
            filled.append(f"acquisitions[{index}]: {value} -> {target}")
    return filled


def derive_table_effects(body: dict[str, Any]) -> list[str]:
    """Mark a table an analysis cites as reporting that analysis's effect.

    `Table.purpose` says what a table's rows are *when they are not the foci of
    a reported effect*, and absence used to mean both "it reports results" and "nothing
    decided". A model shown eight kinds of non-analysis and no way to say "it is an
    analysis" answers with the nearest one: over twelve papers a repair pass marked 18 of 19
    tables, and 16 of those 18 were cited by an analysis, which makes the mark wrong by the
    slot's own definition.

    The join settles it without a model. A table named in `Analysis.tables` reports that
    analysis's result, so it takes `reported_effect` and the eight other kinds cannot apply.
    `generated`, because the record states this and the paper does not.
    """
    cited: set[str] = set()
    for analysis in body.get("analyses") or []:
        if not isinstance(analysis, dict):
            continue
        for target in analysis.get("tables") or []:
            if isinstance(target, str):
                cited.add(target)

    filled: list[str] = []
    for index, table in enumerate(body.get("tables") or []):
        if not isinstance(table, dict):
            continue
        local_id = str(values.read(table.get("local_id")) or "")
        if local_id not in cited:
            continue
        held = values.read(table.get("purpose"))
        if held == "reported_effect":
            continue
        # An analysis cites it, so any other kind contradicts the record rather than
        # describing it -- which is the contradiction `check_table_content` reports.
        table["purpose"] = values.wrap(
            "reported_effect", source="generated", evidence="not_applicable"
        )
        filled.append(f"tables[{index}].purpose" + (f": was {held!r}" if held else ""))
    return filled


def derive_denominators(body: dict[str, Any]) -> list[str]:
    """Fill `CategoryDistribution.denominator` from the count and the percentage.

    The slot records the base the paper divided by, which is often not the group's `n`: a
    handedness figure may cover only those who answered, and a race breakdown may be of the
    analysed sample after exclusions. That is why it exists, and why the count check reads
    it rather than guessing -- but it is populated in none of 200 records, so nothing
    checked anything.

    Where a paper writes "12 male (60%)" the base is stated twice over and needs no model:
    12 / 0.60 is 20. Filled only when the division lands within a tenth of a whole number
    and every entry of the distribution agrees on the same base, because a rounded
    percentage on a small sample is ambiguous -- 1 of 3 prints as 33% and implies 3.03.
    `value_source` is `generated`, since the paper stated a percentage and not this.
    """
    filled: list[str] = []
    for owner in ("groups",):
        for index, group in enumerate(body.get(owner) or []):
            if not isinstance(group, dict):
                continue
            for slot in ("sex_distribution", "race_distribution", "handedness_distribution"):
                entries = [e for e in (group.get(slot) or []) if isinstance(e, dict)]
                bases: list[float] = []
                for entry in entries:
                    if values.read(entry.get("denominator")) is not None:
                        bases = []
                        break
                    count = values.read(entry.get("count"))
                    share = values.read(entry.get("percentage"))
                    if not isinstance(count, (int, float)) or not isinstance(share, (int, float)):
                        continue
                    if not 0 < float(share) <= 100:
                        continue
                    implied = float(count) / (float(share) / 100.0)
                    if abs(implied - round(implied)) > 0.1:
                        continue
                    bases.append(round(implied))
                if not bases or len(set(bases)) != 1 or len(bases) != len(entries):
                    continue
                base = bases[0]
                # The base has to reproduce every percentage the paper printed. This is a
                # stronger condition than the division landing near a whole number, and it
                # is what separates a base the paper implies from one arithmetic invented:
                # 1 of 3 prints as 33% and 2 of 3 as 67%, and only 3 gives back both.
                if base <= 0 or any(
                    round(float(values.read(entry.get("count"))) / base * 100)
                    != round(float(values.read(entry.get("percentage"))))
                    for entry in entries
                ):
                    continue
                for entry in entries:
                    entry["denominator"] = {
                        "extraction_status": "extracted",
                        "value": int(base),
                        "value_source": "generated",
                        "evidence": {"status": "not_applicable"},
                    }
                filled.append(f"{owner}[{index}].{slot}.denominator = {int(base)}")
    return filled


def derive_coordinate_spaces(
    body: dict[str, Any], stage1: Path | None, table_map: Path | None
) -> list[str]:
    """Fill `Analysis.coordinate_space` from the space stage 1 read off the table.

    Stage 1 parses a space for every coordinate it extracts and is right about it: across
    ten papers it returned MNI for all 197 points and disagreed with itself on none. The
    stage-3 prompt already injects that space, but as a hint "to confirm, not values to
    copy" -- and the model declines to confirm, leaving the slot unreported on 50 of 57
    analyses. A fact this deterministic should not be routed through a model that has been
    told to distrust it, any more than `Table.caption` is.

    Only fills what the model left empty, and only when every point behind the analysis
    agrees, so a genuinely mixed-space paper still reaches a human.
    """

    if not (stage1 and stage1.is_file() and table_map and table_map.is_file()):
        return []

    parsed = json.loads(stage1.read_text(encoding="utf-8")).get("analyses") or []
    mapping = json.loads(table_map.read_text(encoding="utf-8"))

    spaces_by_table: dict[str, set[str]] = {}
    for analysis in parsed:
        local = mapping.get(analysis.get("table_id"))
        if not local:
            continue
        seen = {p.get("space") for p in analysis.get("points") or [] if p.get("space")}
        spaces_by_table.setdefault(local, set()).update(seen)

    filled: list[str] = []
    for index, analysis in enumerate(body.get("analyses") or []):
        slot = analysis.get("coordinate_space")
        if isinstance(slot, Mapping) and slot.get("extraction_status") == "extracted":
            continue
        spaces = set()
        for table in analysis.get("tables") or []:
            spaces |= spaces_by_table.get(table, set())
        if len(spaces) != 1:
            continue
        space = spaces.pop()
        # `not_found` and not `not_applicable`, which is the state this used to synthesise.
        # An extracted field claiming `not_applicable` is one of the shape errors
        # `_resolve_field` repairs, so the walk rewrote every one of these to `not_found`
        # anyway -- and counted it in `report.downgraded`, which meant that number was
        # dominated by the builder's own output and could not be thresholded on. There is no
        # quote to find for a value read off the table parse, so `not_found` is simply true.
        analysis["coordinate_space"] = values.wrap(space, source="reported", evidence="not_found")
        filled.append(f"analyses[{index}] -> {space}")
    return filled


def fill_directions(body: dict[str, Any]) -> list[str]:
    """Give a cell its direction from the contrast's own name, where the model gave none.

    Only fills `absent` and only where the name states a comparison the level appears on
    one side of. Measured over 328 reviewed cells: it answers 17% of them at 98%, and in
    every case where it disagreed with the extraction pass the pass had said `absent` --
    it recovered five and lost none. See docs/deterministic-direction.md.

    It never overrides a direction the model committed to. A rule that answers a sixth of
    the cells has no standing to overturn the pass on the rest, and a silent overwrite
    would hide a disagreement worth reading.
    """

    filled: list[str] = []
    for analysis in body.get("analyses") or []:
        if not isinstance(analysis, Mapping):
            continue
        contrast = " . ".join(
            filter(
                None,
                [
                    str(values.read(analysis.get("name")) or ""),
                    str(values.read(analysis.get("definition")) or ""),
                ],
            )
        )
        if not contrast:
            continue
        for cell in (analysis.get("effect") or {}).get("cells") or []:
            if not isinstance(cell, Mapping):
                continue
            node = cell.get("direction")
            current = values.read(node)
            if current not in (None, "", "absent"):
                continue
            level = str(values.read(cell.get("level")) or "")
            derived = direction.direction_of(level, contrast)
            if derived is None:
                continue
            if isinstance(node, dict):
                node["value"] = derived
                node["extraction_status"] = "extracted"
                node["value_source"] = "generated"
            else:
                cell["direction"] = values.wrap(derived, source="generated", evidence="not_found")
            filled.append(
                f"{values.read(analysis.get('local_id'))}: "
                f"{level or '(unnamed level)'} -> {derived}"
            )
    return filled


def mirror_withheld(body: dict[str, Any], stage1: Path | None) -> list[str]:
    """Rebuild the reversed half of every sign-split contrast, from the corrected record.

    `sign_split.split_opposite_signs` hands the extraction pass only the half the paper
    describes and marks the other `withhold`. This runs last, on the assembled record, so
    the mirror is taken from the contrast the model actually settled on -- including
    whatever the wrapper repairs, level alignment and direction fill changed about it.
    Mirroring the raw payload would copy mistakes the builder had already fixed.
    """

    if not (stage1 and stage1.is_file()):
        return []
    parsed = json.loads(stage1.read_text(encoding="utf-8")).get("analyses") or []
    # Keyed alongside the parse so each withheld entry carries the address of its own row
    # group -- the mirrored analysis's only route to the rows it is about.
    keyed = list(zip(parse_keys.parse_keys(parsed), parsed))
    withheld = [(key, a) for key, a in keyed if a.get("withhold") and a.get("mirror_of")]
    if not withheld:
        return []

    analyses = body.setdefault("analyses", [])

    # Joined on the parse key, not on the name. `entry["mirror_of"]` is the *parse's* name
    # for the described half, and the record's name for that analysis is whatever the model
    # wrote -- which the prompt actively tells it to change: "Each part is its own entry,
    # named `<given name> (<level>)`". The most common case, a SPLIT, was therefore the
    # case the join could not survive, and the reversed half of a mixed-sign table was
    # silently never built. A trailing space did it too.
    #
    # `source_links` is repair 13 and this is repair 16, so every analysis that can carry a
    # key already does. A SPLIT leaves several record analyses sharing one key; each is a
    # contrast in its own right and each gets its mirror, which is the right semantics
    # rather than a tie to break.
    # Keyed on `(table_id, name)`, because a name alone is not unique and `stage1_block`
    # says so: "the same analysis name recurs across tables in the same paper (an ROI table
    # and a whole-brain table reporting one contrast), and the table is the only thing that
    # tells those apart." Keyed on the name alone this was last-wins, and the failure was
    # silent in three directions at once -- the earlier table's half was never mirrored and
    # no MISSING note fired because *a* described half had been found; the mirror carried
    # one table's cells against the other's coordinates; and both mirrors were minted with
    # the same `local_id`.
    #
    # The pair is exact rather than a heuristic: `split_opposite_signs` emits the two halves
    # of one entry consecutively, so they always share a `table_id`.
    described_key = {
        (a.get("table_id"), str(a.get("name") or "")): key
        for key, a in keyed
        if not a.get("withhold")
    }
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for analysis in analyses:
        if isinstance(analysis, Mapping):
            key = values.read(analysis.get("source_table_analysis"))
            if isinstance(key, str) and key:
                by_key.setdefault(key, []).append(analysis)

    # The fallback, for the analyses that carry no key at all -- `derive_analysis_ids`
    # measures that at about a quarter of them. Folded, which is the discipline
    # `resolve_source_table_analysis` already uses for names in this module; the raw `==`
    # this replaced lost a match to a trailing space.
    by_folded_name: dict[str, list[Mapping[str, Any]]] = {}
    for analysis in analyses:
        if isinstance(analysis, Mapping):
            name = span_tools.fold_label(str(values.read(analysis.get("name")) or ""))
            if name:
                by_folded_name.setdefault(name, []).append(analysis)

    made: list[str] = []
    for parse_key, entry in withheld:
        target = described_key.get((entry.get("table_id"), entry["mirror_of"]))
        described = by_key.get(target or "", [])
        route = "parse key"
        if not described:
            described = by_folded_name.get(span_tools.fold_label(entry["mirror_of"]), [])
            route = "name"
        if not described:
            made.append(
                f"MISSING {entry['mirror_of']}: no analysis carries the described half's "
                f"parse key ({target or 'unknown'}) and none matches its name, so its "
                f"reversal cannot be built"
            )
            continue
        for half in described:
            mirrored = direction.mirror_analysis(half, entry, parse_key)
            analyses.append(mirrored)
            made.append(
                f"{entry['mirror_of']} -> {mirrored.get('local_id')} "
                f"(rows at {parse_key}, signs negated, joined on {route})"
            )
    return made


def relabel_conclusions(body: dict[str, Any], sch: Schema) -> list[str]:
    """Say `generated` where the record claims a conclusion was `reported`.

    Only where no sentence was found. A conclusion slot that *does* carry evidence keeps
    `reported`, because then the paper did say it -- "the interaction was negative" is a
    direction read off prose, and relabelling it would throw away the one case where the
    label is earned.

    The point is not tidiness. `reported` and `generated` license different downstream
    actions, and while 98.6% of papers carry the warning there is no way to see the record
    that says `spatial_scope = roi` on no evidence -- which is what cost 17133391 its
    inclusion.
    """

    fixed: list[str] = []
    for slot in walk.fields(body, sch):
        if (slot.owner_class, slot.key) not in CONCLUSION_SLOTS:
            continue
        if not values.is_field(slot.value) or slot.value.get("value_source") != "reported":
            continue
        if (slot.value.get("evidence") or {}).get("status") != "not_found":
            continue
        slot.value["value_source"] = "generated"
        fixed.append(f"{slot.path}: reported -> generated")
    return fixed


def resolve_source_table_analysis(body: dict[str, Any], stage1: Path | None) -> list[str]:
    """Verify, or deterministically fill, each analysis's link to its parsed row group.

    `Analysis.source_table_analysis` is the only exact route from an analysis to its
    coordinates: the schema stores none, `Table.coordinate_count` says only how many
    exist, and `tables` cannot disambiguate because a table usually reports several
    contrasts and several analyses usually cite the same table.

    Not left to the model. A key it invents resolves to nothing, and a key it omits
    leaves the join to a later string match -- which is what this slot exists to replace.
    A present key is checked against the parse and dropped if it names no row group,
    and an absent one is filled when exactly one parsed entry under the cited tables
    carries the same name. Where neither holds, the slot stays empty and the analysis is
    honestly unjoinable rather than joined to a guess.
    """

    if not (stage1 and stage1.is_file()):
        return []
    parsed = json.loads(stage1.read_text(encoding="utf-8")).get("analyses") or []
    if not parsed:
        return []

    # The same keying the prompt prints, from the same function, so the two cannot drift.
    keys = dict(zip(parse_keys.parse_keys(parsed), parsed))

    def fold(text: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())

    notes: list[str] = []
    for index, analysis in enumerate(body.get("analyses") or []):
        if not isinstance(analysis, Mapping):
            continue
        node = analysis.get("source_table_analysis")
        claimed = values.read(node)
        path = f"analyses[{index}].source_table_analysis"

        if isinstance(claimed, str) and claimed in keys:
            continue
        if isinstance(claimed, str) and claimed:
            analysis.pop("source_table_analysis", None)
            notes.append(f"{path}: {claimed!r} names no parsed row group -- dropped")

        cited = [t for t in (values.read(analysis.get("tables")) or []) if isinstance(t, str)]
        wanted = fold(values.read(analysis.get("name")))
        same = [
            key
            for key, entry in keys.items()
            if fold(entry.get("name")) == wanted
            and (not cited or str(entry.get("table_id")) in cited)
        ]
        if len(same) == 1 and wanted:
            analysis["source_table_analysis"] = values.wrap(
                same[0], source="generated", evidence="not_applicable"
            )
            notes.append(
                f"{path}: filled {same[0]!r} from the parsed analysis of the " f"same name"
            )
    return notes


def derive_analysis_ids(body: dict[str, Any]) -> list[str]:
    """Rename each analysis to an id the parse determines, not one the model chose.

    A model-chosen `local_id` is unstable. Over the same sixteen papers extracted twice,
    only four produced identical analysis ids: `a_ic25/a_ic30/a_ic35` one run against
    `a_independent_component_spatial_maps` the next, `a_fa_group` against `a_fa`. That
    matters outside the record -- the review layer addresses an analysis as
    `paper|value|Analysis|<local_id>|<path>`, so a re-extraction orphans every answer a
    reviewer gave.

    `source_table_analysis` is already a deterministic paper-scoped key, so the id is
    derived from it: `a_<table id>_<ordinal>`. Safe to do here because nothing in the
    schema references an Analysis by id -- `Study.analyses` is the only slot with that
    range and it inlines them -- so the sole pointer to follow is `mirror_of`.

    An analysis with no key keeps the model's id. That is 25% of them and it is the
    honest outcome: the parse does not determine an id for a row group it cannot identify,
    and inventing a stable-looking one would claim otherwise.

    Idempotent, and collision-safe: a SPLIT emits several analyses against one listing
    entry, so they share a key and are numbered apart in the order they appear.
    """

    renamed: dict[str, str] = {}
    used: set[str] = set()
    notes: list[str] = []

    for analysis in body.get("analyses") or []:
        if isinstance(analysis, Mapping) and isinstance(analysis.get("local_id"), str):
            used.add(analysis["local_id"])

    seen: dict[str, int] = {}
    for analysis in body.get("analyses") or []:
        if not isinstance(analysis, Mapping):
            continue
        key = values.read(analysis.get("source_table_analysis"))
        if not isinstance(key, str) or "#" not in key:
            continue
        table_id, _, ordinal = key.partition("#")
        stem = f"a_{re.sub(r'[^A-Za-z0-9]+', '_', table_id).strip('_')}_{ordinal}"
        seen[stem] = seen.get(stem, 0) + 1
        derived = stem if seen[stem] == 1 else f"{stem}_{seen[stem]}"
        old = analysis.get("local_id")
        if old == derived:
            continue
        if derived in used and derived != old:
            # Another analysis already answers to this. Leave both alone rather than
            # collapse two analyses into one id.
            notes.append(f"{old!r}: derived id {derived!r} is already taken -- left as is")
            continue
        analysis["local_id"] = derived
        used.discard(old)
        used.add(derived)
        if isinstance(old, str):
            renamed[old] = derived
        notes.append(f"{old!r} -> {derived!r} (from {key!r})")

    # `mirror_of` is the only pointer at an analysis anywhere in the record.
    for analysis in body.get("analyses") or []:
        if isinstance(analysis, Mapping) and analysis.get("mirror_of") in renamed:
            analysis["mirror_of"] = renamed[analysis["mirror_of"]]
    return notes
