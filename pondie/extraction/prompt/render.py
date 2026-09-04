"""Run an LLM extractor over one paper and emit a payload for builder.py.

This is the first half of the review pipeline. `builder.py` consumes extractor
payloads and resolves their verbatim quotes to character offsets; this module
produces those payloads.

The prompt is rendered from the schema itself rather than restated in a string, so
the instructions cannot drift from the YAML. What the schema cannot say -- gates,
the multivalued-wrapper convention, the evidence rules, the direction vocabulary --
comes from `extraction-readme.md`, which is sent alongside it, and the shapes whole
encodings take come from `representing-models.md` §5 (see `worked_models`).

Two modes, because a single call puts the analyses behind thirty-odd entity classes
and drops them (`bench/RESULTS.md` on the pipeline_eval branch: 19% of papers
returned no analyses at all):

    entities   pass 1 -- everything the analyses point at, and nothing else
    analyses   pass 2 -- one Analysis per pre-parsed table analysis, linked by
               local_id to what pass 1 emitted

The class split is computed from the schema, not listed here: `Analysis`'s nested
closure is the analyses prompt and the rest of `Study`'s is the entities prompt. The
two are asserted disjoint by `test_extraction_prompt.py`, so a new class cannot land
in neither.

    pondie extract --pmids papers.pmids --run <run> --model <model> \
        --stages demands satisfy
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pondie import paths, schema
from pondie.extraction.models import Prompt

# `builder` for the payload contract (see ENTITY_LISTS); `preprocess` for the
# deterministic text transforms selected by --preprocess.
from pondie.extraction.record import builder, ids
from pondie.formats import parse_keys
from pondie.schema import reader
from pondie.schema.reader import Schema

REPO = paths.REPO
#: The schema is a submodule of this repository, not the parent directory this
#: module used to sit in.
EXTRACTION_SCHEMA = schema.EXTRACTION
README = schema.ROOT / "extraction-readme.md"
MODELS = schema.ROOT / "representing-models.md"

#: The heading of the section of `representing-models.md` the prompt carries, and
#: the sub-heading `test_extraction_prompt.py` asserts survives the slice.
WORKED_MODELS_SECTION = "## 5. Worked models"

#: Payload keys merge_payloads() accepts, taken from the schema through the same
#: function build_record uses. Hardcoding this list is how `conditions` and `terms`
#: survived here for a schema version after Condition moved under Task and Term
#: became ModelTerm under ModelEstimation -- both would have been merged as
#: "unexpected payload key" and dropped.
ENTITY_LISTS = builder._entity_lists()

#: Filled by the builder from the source text, never by the model.
SCAFFOLDING_CLASSES = {"ExtractionMetadata", "PaperSection"}

#: Supplied deterministically from the pubget table manifest by run_extraction.py:
#: table_number, caption and footer are literal source strings, so asking a model
#: to retype them can only introduce error.
DETERMINISTIC_CLASSES = {"Table"}

DEFAULT_MODEL = "@psyc-aid338-ope-333f18/gpt-5.6-luna"


# --------------------------------------------------------------- class selection


def nested_closure(sch: Schema, roots: list[str]) -> set[str]:
    """Every class reachable from `roots` through slots the record owns.

    Ownership is the boundary: a nested slot holds the record and has to be
    described in the same prompt, a reference slot holds only a local_id and its
    target can be described in the other one.
    """

    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen or name not in sch:
            continue
        seen.add(name)
        stack.extend(sch.subclasses(name))
        for _attr, slot, kind in sch.iter_slots(name):
            if kind == "nested":
                stack.extend(sch.ranges(slot))
    return seen


def mode_classes(sch: Schema, mode: str) -> tuple[set[str], list[str]]:
    """(classes to render, Study attributes to keep) for one pass."""

    analysis_side = nested_closure(sch, ["Analysis"])
    study = sch.attributes("Study")

    if mode == "analyses":
        keep = ["analyses"]
        return analysis_side - DETERMINISTIC_CLASSES, keep

    roots: list[str] = []
    keep = []
    for attr, slot in study.items():
        if attr in ("analyses", "tables", "extraction_metadata", "local_id"):
            continue
        keep.append(attr)
        if sch.classify(attr, slot) == "nested":
            roots.extend(sch.ranges(slot))
    entity_side = nested_closure(sch, roots)
    return entity_side - analysis_side - SCAFFOLDING_CLASSES - DETERMINISTIC_CLASSES, keep


# ------------------------------------------------------------------- rendering


def _wrap(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def enum_of(sch: Schema, range_name: str):
    """(permissible values, closed, multivalued) if `range_name` wraps a vocabulary.

    The wrappers are generated one per vocabulary and keep storage's own range, so
    whether a field is closed is readable here rather than guessable: a bare range
    is closed, an `any_of: [<Enum>, string]` keeps the escape hatch. Getting this
    wrong in the prompt is expensive in both directions -- a closed field filled
    with free text is rejected by storage, and an open field forced to the nearest
    permissible value destroys the evidence that the vocabulary is short a value.
    """

    if range_name not in sch:
        return None
    # The induced slot, so `slot_usage` is already applied: that is exactly where a
    # wrapper narrows `value` from the `Any` it inherits down to its own vocabulary.
    value = sch.attributes(range_name).get("value")
    if value is None:
        return None
    named = [r for r in sch.ranges(value) if r in sch.enums]
    if not named:
        return None
    values = list(sch.enums[named[0]].permissible_values or {})
    closed = value.range in sch.enums
    return values, closed, bool(value.multivalued)


def render_schema(sch: Schema, names: set[str], study_keep: list[str]) -> str:
    """One block per class: its description, then one line per attribute.

    Every class is rendered in schema declaration order so the reading order
    matches the YAML, and `Study` comes first because it is the record's shape.
    """

    out: list[str] = []
    order = ["Study"] + [n for n in sch.declaration_order if n in names and n != "Study"]

    for name in order:
        definition = sch.definition(name)
        if definition is None:
            continue
        attributes = sch.attributes(name)
        if name == "Study":
            attributes = {k: v for k, v in attributes.items() if k in study_keep}
        if not attributes:
            continue

        header = name
        if definition.is_a and definition.is_a in names:
            header += f" (is_a: {definition.is_a})"
        out.append(f"\n### {header}")
        if definition.description:
            out.append(_wrap(definition.description))

        for attr, spec in attributes.items():
            kind = sch.classify(attr, spec)
            ranges = sch.ranges(spec) or ["string"]
            bits = [ranges[0]]
            if spec.multivalued:
                bits.append("multivalued")
            if spec.required:
                bits.append("REQUIRED")
            # A slot's shape is the most easily confused
            # thing in this schema, and the model gets it wrong silently: pass 1
            # emitted `terms` as {"extraction_status": ..., "value": [ModelTerm]},
            # wrapping a nested record list as though it were a multivalued scalar.
            # State the shape on the line instead of relying on rule 4.
            if kind == "reference":
                bits.append(
                    f"local_id of {ranges[0]}"
                    + (
                        " — plain list of id strings"
                        if spec.multivalued
                        else " — plain id string"
                    )
                )
            elif kind == "nested":
                bits.append(
                    f"nested {ranges[0]} record"
                    + (
                        "s — a plain JSON LIST of objects, NOT an ExtractedValue wrapper"
                        if spec.multivalued
                        else " — a plain JSON object, NOT an ExtractedValue wrapper"
                    )
                )

            line = f"- `{attr}` ({', '.join(bits)}): {_wrap(spec.description or '')}"
            vocabulary = enum_of(sch, ranges[0])
            if vocabulary:
                values, closed, multivalued = vocabulary
                joined = " | ".join(values)
                if closed:
                    line += (
                        f"\n    value MUST be one of: {joined}"
                        " -- there is no other permitted answer."
                    )
                else:
                    line += (
                        f"\n    value is one of: {joined}"
                        " -- or the paper's own wording when none of them fits."
                    )
                if multivalued:
                    line += " `value` is a LIST of these."
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------- pass-2 context


ZERO_FOCI_RULE = """
A "0 foci" entry is a TESTED EFFECT THAT FOUND NOTHING, and it is emitted like any other.
It is not one of the OMIT cases above. The contrast was run, the paper reports its result,
and the result was that no cluster survived -- "no significant correlation was found" is a
finding about a comparison that happened. Its `Effect.cells` are filled from what was
compared, exactly as for an entry that did report coordinates; what it lacks is coordinates,
not a comparison.

Dropping it destroys the one thing the record exists to distinguish: an effect tested and
null, versus an effect never tested. Two papers reporting a positive result and a null
result of the same contrast must not extract to the same record.
"""


def stage1_block(
    stage1: Mapping[str, Any],
    table_ids: Mapping[str, str],
    detail: bool = False,
    zero_foci_rule: bool = False,
) -> str:
    """The analyses parsed from the result tables, grouped by the table reporting them.

    Grouping is not decoration: the same analysis name recurs across tables in the
    same paper (an ROI table and a whole-brain table reporting one contrast), and
    the table is the only thing that tells those apart.

    Space and statistic type come from the parser as normalized codes. They are
    offered as hints to confirm, not values to copy, because `coordinate_space`
    wants the paper's own wording.
    """

    # A withheld entry is the reversed half of a sign-split contrast. The paper does not
    # describe it, so showing it to the model produces an invented name and definition;
    # `direction.mirror_analysis` rebuilds it from the described half instead.
    #
    # Keys are computed over the FULL parse and the withheld entries dropped afterwards,
    # so hiding one does not renumber its siblings. `parse_keys.parse_keys` explains why
    # a shifted key is worse than a missing one.
    every = stage1.get("analyses") or []
    keyed = list(zip(parse_keys.parse_keys(every), every))
    shown = [(key, a) for key, a in keyed if not a.get("withhold")]
    if not shown:
        return ""
    analyses = [a for _key, a in shown]

    grouped: dict[str, list[tuple[int, dict]]] = {}
    key_by_index: dict[int, str] = {}
    for index, (key, analysis) in enumerate(shown, start=1):
        grouped.setdefault(analysis.get("table_id") or "", []).append((index, analysis))
        key_by_index[index] = key

    lines = [
        "\n## Analyses already parsed from the result tables (stage 1)",
        f"These {len(analyses)} entries are a first pass over the coordinate tables, made",
        "without seeing the tables' rows. Work through them in order and emit one `analyses`",
        "entry for each, keeping the given name verbatim in `name.value` -- unless one of the",
        "two departures below applies. Never invent an entry for an effect no listing names.",
        "",
        "SPLIT one entry into several when the table distinguishes the rows it covers by a",
        "column the entry's name does not mention -- a frequency band, a diffusion parameter,",
        "a session, an occasion. The parse had the contrast name and not the rows, so a column",
        "can carry a factor it never saw. Each part is its own entry, named",
        "`<given name> (<level>)`, and every part keeps the same `tables`. The signal that this",
        "is needed: one entry would otherwise hold effects of opposite sign, forcing a single",
        "unsigned cell where the paper reports a direction for each.",
        "",
        "OMIT an entry when its table reports no tested effect at all: an ROI or component",
        "definition, an atlas listing, coordinates cited from other papers, a stimulus list,",
        "demographics, descriptive means with no test. Such a table has no comparison, so",
        "`Effect.cells` cannot be filled honestly, and inventing a cell to satisfy it is worse",
        "than emitting no analysis. Say what the table is in that Table's",
        "`non_analysis_content` instead, and put the coordinates on the entity they locate --",
        "a Region's `description` -- rather than on a contrast that never produced them.",
        "Omitting is not for an effect that is merely awkward to encode: an effect the paper",
        "tested belongs in `analyses` however hard its shape.",
        "",
        "`source_table_analysis` is REQUIRED on every entry you emit here: copy the",
        "bracketed `[parse key: ...]` of the listing entry you emitted it for, verbatim. It",
        "is the only exact link between an analysis and the coordinate rows it was read",
        "off -- `tables` cannot do it, because a table usually reports several contrasts",
        "and several analyses usually cite the same table. If you SPLIT one listing entry",
        "into several, every part carries the same key.",
        "",
        "`tables` is REQUIRED on every entry you emit here. It is the bracketed",
        "`[table local_id: ...]` of the heading the entry sits under, copied verbatim, and it",
        "is the only link between the record and the rows the result was read off. Rule 4c",
        "does not apply: under one of these headings there is always something to point at.",
        "",
        "The `space` and `statistic` notes are what the results table showed -- confirm them",
        "against the paper's own wording rather than copying the code.\n",
    ]
    if zero_foci_rule and any(not (a.get("points") or []) for a in analyses):
        lines.append(ZERO_FOCI_RULE)
    for table_id, entries in grouped.items():
        first = entries[0][1]
        label = first.get("table_label") or f"Table {first.get('table_number')}"
        caption = _wrap(first.get("table_caption") or "")[:160]
        lines.append(
            f'{label} — "{caption}"   [table local_id: {table_ids.get(table_id, table_id)}]'
        )
        for number, analysis in entries:
            points = analysis.get("points") or []
            spaces = sorted({p.get("space") for p in points if p.get("space")})
            kinds = sorted(
                {
                    v.get("kind")
                    for p in points
                    for v in (p.get("values") or [])
                    if v.get("kind")
                }
            )
            notes = [
                (
                    f"{len(points)} foci"
                    if points or not zero_foci_rule
                    else "0 foci -- tested, no cluster survived; still an analysis"
                )
            ]
            if spaces:
                notes.append("/".join(spaces))
            if kinds:
                notes.append("/".join(kinds))
            lines.append(
                f"  {number}. {analysis.get('name')}   · {' · '.join(notes)}"
                f"   [parse key: {key_by_index[number]}]"
            )
            if analysis.get("description"):
                lines.append(f"       ({_wrap(analysis['description'])[:150]})")
            if detail:
                # The rows the parse was made without. A signed statistic value is the
                # only place some papers state which way a contrast went, and the digest
                # above reduces it to a count.
                for point in points[:40]:
                    coordinates = ", ".join(f"{c:g}" for c in (point.get("coordinates") or []))
                    values = " ".join(
                        f"{v.get('kind', '?')}={v.get('value')}"
                        for v in (point.get("values") or [])
                    )
                    lines.append(f"       ({coordinates})  {values}")
                if len(points) > 40:
                    lines.append(f"       ... {len(points) - 40} further foci")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- prompts

SYSTEM_HEAD = """You extract structured records from neuroimaging papers.

You are given a LinkML schema, the conventions document that governs it, and worked
encodings of twelve reported results. The schema's own `description:` fields are the
extraction instructions -- follow them exactly, including every statement about what must
not be inferred.

Rules that decide whether a record is usable:

1. Emit ONE JSON object and nothing else. No prose, no markdown fence.
2. These keys go at the TOP LEVEL of the object and nowhere else: {lists}.
   Do NOT also nest them inside "study" -- a list in both places is a list emitted twice.
   Everything else the Study class holds -- `description`, `design` -- goes inside a
   "study" object, and `arms`/`timepoints` go inside `study.design`. Do NOT emit
   `extraction_metadata`; the builder adds it from the source text.
3. {value_rule}
4. Three kinds of field look alike and are not. The schema line for each says which it is:
   a. a source-derived value -> an ExtractedValue wrapper (rule 3). When it is
      multivalued, that is ONE wrapper whose `value` is a list -- never a list of wrappers.
   b. a NESTED RECORD ("nested <Class> record" on its schema line) -> a plain JSON object,
      or a plain JSON list of objects when multivalued. It is NOT wrapped, has no
      `extraction_status`, and its own fields follow these same rules.
      `ModelEstimation.terms`, `ModelTerm.levels`, `Task.conditions`, `Effect.cells` and
      `Analysis.groups` are all of this kind.
   c. a CROSS-REFERENCE ("local_id of <Class>") -> a bare string, or a plain list of
      bare strings. Never wrapped. When there is nothing to point at, OMIT the key
      entirely: a reference is not an ExtractedValue, so it has no `not_reported` form,
      and neither `null` nor a wrapper is a valid value for one. Rule 5 does not apply
      to these.
5. A field the paper does not report takes
   {{"extraction_status": "not_reported"{absent_evidence}}}.
   Use it rather than omitting a REQUIRED field, and never invent a value to fill one.
6. `local_id` is a bare string you assign, unique within its class, referenced by other
   records. Every local_id referenced must exist.

   It is an ADDRESS, not a description. The review layer addresses a field as
   `paper|value|<Class>|<local_id>|<path>`, so an id that changes between extractions of
   the same paper orphans every answer a reviewer gave against it. Use the prefix for the
   class and then the shortest thing the PAPER fixes -- an enum value it states, an
   abbreviation it defines -- never a phrase you compose:

{id_prefixes}

   Use `acq_fmri`, not `acquisition_resting_state_bold`; use `asm_madrs`, not
   `assessment_montgomery_asberg_depression_rating_scale`; `grp_schizophrenia` and not
   `group_patients_with_first_episode_schizophrenia`. Where a paper has two of a kind, add
   the shortest thing that separates them -- `acq_fmri_ge`, `grp_sib_past`. Analyses and
   Tables are exceptions: do not choose their ids, they are derived from the table parse.
7. Set `value_source` to "reported" when the value is the paper's own wording or number,
   and "generated" when you had to phrase it (a summary, a label the paper implies but
   never writes). A field whose schema line gives a closed vocabulary is almost always
   "generated": no paper writes "not_applicable".
8. Where a schema line states a closed vocabulary, no other answer is accepted. Where it
   offers the paper's own wording as a fallback, use it only when no listed value fits.
9. Two rules in the conventions document decide more of this record than any other, so
   read them before you start: the self-naming method payload
   (`AnalysisDetails.details_type`, `Acquisition.acquisition_type`) and what
   `Cell.direction` means, including when a level takes no cell at all.
10. A shape the schema alone does not settle is settled by the worked models. A
    comparison is a term with levels and a sign on each side -- never one column named
    after the comparison it was the subject of.
"""

VALUE_RULE_EVIDENCE = """Every source-derived value is an ExtractedValue wrapper:
   {"extraction_status": "extracted", "value": <value>, "value_source": "reported",
    "evidence": {"status": "present", "sets": [{"quotes": ["<verbatim span>"]}]}}
   A quote MUST be copied character-for-character from the paper. It is located in the
   source text by exact match; a paraphrased or reconstructed quote is dropped."""

VALUE_RULE_NO_EVIDENCE = """Every source-derived value is an ExtractedValue wrapper:
   {"extraction_status": "extracted", "value": <value>, "value_source": "reported"}
   DO NOT emit an `evidence` key anywhere. Supporting spans are added by a separate later
   pass. Spend your output on getting the values right and complete, not on quotation."""

DEMANDS_NOTE = """
This pass emits `analyses`, and the SHOPPING LIST of entities those analyses need.

The supporting entities have NOT been extracted yet. You decide what they are, because you
are the one who knows what the contrasts have to be expressed over -- that is the point of
this ordering. Invent a `local_id` for each entity an analysis references, reference it
normally, and declare it in a top-level `required_entities` list. A later pass fills in each
declared entity's own attributes; here you state only what it must be.

    "required_entities": [
      {"local_id": "t_stimulus", "kind": "ModelTerm", "label": "stimulus category",
       "term_type": "categorical", "levels": ["faces", "houses"], "model": "m_first_level",
       "why": "the contrast weights one level against the other"},
      {"local_id": "t_age", "kind": "ModelTerm", "label": "age at scan",
       "term_type": "continuous", "levels": [], "model": "m_group",
       "why": "the second analysis reports the sign of its slope"},
      {"local_id": "m_first_level", "kind": "ModelEstimation", "label": "subject-level GLM"},
      {"local_id": "r_ffa", "kind": "Region", "label": "fusiform face area"}
    ]

That example is a different study from the one you are reading. Take its shape and none of
its content: no label, level or identifier from it belongs in your answer unless this paper
independently says so.

`kind` is a class name: ModelEstimation, ModelTerm, Measure, Region, Group, Acquisition,
Preprocessing, InferenceSettings, Task, Assessment, Device, Arm, Timepoint.

For a ModelTerm, `term_type` and `levels` are REQUIRED and they are the load-bearing part
of this pass. Decide them from what the contrast does, not from what the term is called:

- A term whose LEVELS the analyses compare, or hold one of, is `categorical`, and its
  `levels` are those level labels. A condition, an occasion, an arm, a cohort.
- A term whose SLOPE an analysis reports the sign of is `continuous`, with `levels: []`.
- An analysis that FIXES something while reporting the sign of something else needs BOTH
  kinds: a categorical term for what was fixed, whose level that analysis holds, and the
  term whose sign is reported. Any result of the form "within X, Y went this way" has this
  shape, whatever X and Y are. Declaring only the signed term leaves the held cell with
  nothing to point at; declaring only the fixed one leaves the sign nowhere to sit.

Every `local_id` any analysis references must appear in `required_entities`, and nothing
else should. Do not emit the entities themselves here -- no `groups`, no `model_estimations`.

`required_entities` IS ITS OWN TOP-LEVEL KEY, a sibling of `analyses`. Declarations do not
go in the `analyses` array. Everything in `analyses` is a tested effect with a name, a
definition and an `effect`; a declaration has none of those and is not an analysis. Nor do
Tables get declared -- they already exist, and the stage-1 listing gives their local_ids.
"""

SATISFY_NOTE = """
This pass extracts the STUDY ENTITIES. The analyses were extracted first and have already
declared, in the shopping list below, which entities they reference and what each must be.

Emit one entity per declared entry, under the right top-level list, using EXACTLY the
`local_id` given. A declared id you do not emit is a dangling reference and the record fails
to build; an id you rename is the same failure. Fill each entity's own attributes from the
paper as usual -- the declaration says which entity it is, not what its attributes are.

Where a declaration carries `term_type` and `levels`, honour them. They were decided by the
pass that knows what the contrasts must be expressed over: a term declared `categorical` with
two levels is emitted with `type: categorical` and those two `FactorLevel`s, each linked to
the arm, condition, timepoint or group carrying it. Do not silently re-model it as a
continuous covariate -- the cells that reference it hold one of its levels, and a continuous
term has no level to hold.

`FactorLevel.arms` is not optional when the level IS an arm. A level naming a treatment or
comparator arm fills `arms` exactly as a level naming a cohort fills `groups`: the level
string is the paper's own wording and carries no identity, so the reference is the only
thing that says which arm a cell is about. Measured over 462 factor levels, `groups` was
filled 178 times and `arms` 33 -- in a corpus of randomised trials, where nearly every
contrast is over an arm. A cell whose level is an arm and whose `FactorLevel` has no `arms`
cannot be resolved to a treatment or a comparator by anything downstream.

Emit any further entity the paper describes that no analysis referenced -- the participant
group's demographics, an assessment, the scanner -- as usual. The list is a floor, not a
ceiling.
"""

MODE_NOTE = {
    "entities": """
This pass extracts the STUDY ENTITIES only. Do NOT emit `analyses` or `tables`: a separate
pass extracts the analyses and will refer to the `local_id`s you assign here, so every
entity needs one. Describe the model the authors estimated -- its terms, their levels, and
which conditions, cohorts, occasions, arms or regions those levels name -- even though the
contrasts themselves come later.

Occasions and cohorts are factors in exactly the sense conditions are: a study with no
paradigm still has a categorical term if it measured the same people twice, its levels
being the occasions, which `FactorLevel.timepoints` names. Do not let the absence of a
task decide that there is no factor. Each level's label is the source's own wording.

A Region is an entity in the sense a Group or a Task is, and THIS PASS IS THE ONLY PLACE
ONE CAN BE CREATED. Emit a Region for each place the study delimited: every ROI or mask an
analysis was restricted to, every connectivity seed and target, every atlas parcel used by
name, every component or cluster reused as a node, and every sphere whose centre the paper
gives. Each carries its own `definition_method` -- how *that* region was delimited -- and
its coordinates, radius or atlas belong in its `description`.

A paper that ran any ROI, seed, mask or parcel analysis and emits no `regions` leaves the
analyses pass with nothing to point `Analysis.regions` at. The ROI information is then not
misplaced but lost: there is no slot on Analysis for how a region was defined, so an
analysis restricted to a region it cannot name has no way to say it was restricted at all.
""",
    "analyses": """
This pass emits `analyses` and nothing else. The supporting entities were extracted
separately and are listed below with their local_ids; refer to them, do not re-emit them.

Two jobs, in this order. First settle the SET of analyses. The stage-1 listing below is a
first pass over the coordinate tables made without seeing their rows, so it is the starting
point and not the answer; the rules there say when one of its entries is really two and
when it is none. Then annotate each analysis you kept: its scope, measure, statistic,
effect cells, inference settings, method payload, and its links by local_id -- `tables`
among them, and `regions` where the analysis was restricted to any.
""",
}


def requirements_block(declared: Mapping[str, Any]) -> str:
    """The shopping list the demands pass wrote, as the entity pass's contract."""

    entries = declared.get("required_entities") or []
    if not entries:
        return ""
    lines = [
        "\n## Entities the analyses have already declared they reference (the shopping list)",
        f"{len(entries)} entries. Emit one entity for each, with EXACTLY the local_id given.",
        "",
    ]
    for entry in entries:
        parts = [f"  {entry.get('local_id')}  [{entry.get('kind', '?')}]"]
        if entry.get("label"):
            parts.append(f'"{_wrap(entry["label"])[:110]}"')
        if entry.get("term_type"):
            parts.append(f"type={entry['term_type']}")
        if entry.get("levels"):
            parts.append(f"levels={entry['levels']}")
        if entry.get("model"):
            parts.append(f"declared by model {entry['model']}")
        lines.append("  ".join(parts))
        if entry.get("why"):
            lines.append(f"       ({_wrap(entry['why'])[:150]})")
    return "\n".join(lines) + "\n"


def worked_models() -> str:
    """`representing-models.md` §5 -- the worked encodings -- for the prompt.

    The conventions document states the rules a term and a cell obey; §5 is the only
    place a whole encoding is shown end to end, and the only place shapes no rule
    reaches on its own appear: a factor over occasions in a study with no paradigm
    (§5.6), an ordered factor contrasted at its extremes (§5.7), a model split across
    stages (§5.12).

    Sliced rather than sent whole. §1-§4 restate what the conventions and the rendered
    `description:` fields already say, and §6 asks whether a paper fits the schema at
    all, which this pass does not decide.

    Raises rather than returning "" when the heading moves. The file is committed here,
    so an empty slice is a repo error, and announcing a section the prompt does not
    carry is worse than failing loudly.
    """

    text = MODELS.read_text(encoding="utf-8")
    # Up to the next `## ` heading. `### 5.1` and friends do not match it -- the
    # character after `##` is `#`, not a space -- so the subsections stay in.
    match = re.search(
        rf"^{re.escape(WORKED_MODELS_SECTION)}$.*?(?=^## |\Z)", text, re.MULTILINE | re.DOTALL
    )
    if match is None:
        raise RuntimeError(
            f"{MODELS.name} has no {WORKED_MODELS_SECTION!r} heading: the worked models "
            "cannot be sliced out for the prompt. Renumbering the section means updating "
            "WORKED_MODELS_SECTION with it."
        )
    return match.group(0).rstrip()


#: The demand-driven pair. `demands` renders the analysis side and `satisfy` the entity
#: side, exactly as `analyses` and `entities` do; what differs is the order they run in and
#: that the shopping list, not a guess, decides which entities exist.
MODE_SCHEMA = {"demands": "analyses", "satisfy": "entities"}
MODE_NOTE["demands"] = DEMANDS_NOTE
MODE_NOTE["satisfy"] = SATISFY_NOTE


def build_prompt(text: str, mode: str, evidence: bool, context: str) -> Prompt:
    sch = reader.load(EXTRACTION_SCHEMA)
    names, study_keep = mode_classes(sch, MODE_SCHEMA.get(mode, mode))

    # Only the lists that sit directly on Study are offered as top-level payload keys.
    # `design.arms` and `design.timepoints` are reachable that way too, but naming them
    # here would contradict rule 2, and merge_payloads resolves a top-level `arms` by
    # assigning over `design.arms` -- so a payload carrying both silently loses one.
    analysis_side = MODE_SCHEMA.get(mode, mode) == "analyses"
    payload_keys = [
        k
        for k, v in ENTITY_LISTS.items()
        if "." not in v and v != "tables" and (v == "analyses") == analysis_side
    ]
    if mode == "demands":
        payload_keys.append("required_entities")

    # Ordering here is not a cache optimisation, and an attempt to make it one failed.
    # Every pass sends the same conventions, worked models and paper -- 29,152 of the
    # 36-42k tokens a call carries -- so leading with them and trailing the mode-specific
    # schema gives two passes a 29k shared prefix. Measured on the gateway: a byte-identical
    # prompt caches 100%, and a prompt sharing a 3.6k prefix with a different suffix caches
    # **zero**. The caching is whole-prompt, not incremental over the prefix, so no reordering
    # can help and the schema is kept ahead of the paper, where instructions belong.
    system = (
        SYSTEM_HEAD.format(
            lists=", ".join(sorted(payload_keys)),
            # From `record/ids.py`, so the convention the model is told and the convention
            # the repair pass mints by cannot drift apart.
            id_prefixes=ids.prefix_table(),
            value_rule=VALUE_RULE_EVIDENCE if evidence else VALUE_RULE_NO_EVIDENCE,
            absent_evidence=', "evidence": {"status": "not_applicable"}' if evidence else "",
        )
        + MODE_NOTE[mode]
    )

    user = (
        "# Conventions (extraction-readme.md)\n\n"
        + README.read_text(encoding="utf-8")
        + "\n\n# Worked models (representing-models.md)\n\n"
        + "Twelve reported results and the encoding each takes. Follow the shape of the\n"
        + "one this paper's result is closest to; do not invent a third when its wording\n"
        + "sits between two of them.\n\n"
        + worked_models()
        + "\n\n# Schema\n"
        + render_schema(sch, names, study_keep)
        + context
        + "\n\n# Paper\n\n"
        + text
        + "\n\nEmit the JSON object now."
    )
    return Prompt(system=system, user=user)


def postcondition_failures(
    payload: Mapping[str, Any], mode: str, declared: Sequence[Mapping[str, Any]] = ()
) -> list[str]:
    """What is wrong with this payload that no schema check would catch.

    A pass that returns `{"groups": [], "measures": [], ...}` is well formed, legally empty,
    and builds and validates into a record about no study at all. That failure was 2 runs in
    10 of the best configuration measured, and it is silent -- `finish=stop`, nothing
    truncated, no validator objection. It is also decidable without a model, which is why
    this is a post-condition and not a critic.
    """

    failures: list[str] = []
    if MODE_SCHEMA.get(mode, mode) == "analyses":
        if not payload.get("analyses"):
            failures.append("no analyses were emitted")
        if mode == "demands" and not payload.get("required_entities"):
            failures.append(
                "no required_entities were declared, so the entity pass that "
                "follows has nothing to be held to"
            )
    else:
        if not any(payload.get(key) for key in ENTITY_LISTS):
            failures.append(
                "every entity list is empty: no group, acquisition, measure or "
                "model estimation was emitted for a paper that has them"
            )
        # Tables are copied from the pubget manifest, never extracted, so a declaration
        # naming one asks this pass for something it is forbidden to emit. Demanding it
        # spends the whole retry budget on a fault no retry can clear.
        missing = [
            entry.get("local_id")
            for entry in declared
            if isinstance(entry, Mapping)
            and entry.get("local_id")
            and entry.get("kind") not in DETERMINISTIC_CLASSES
            and not str(entry["local_id"]).startswith("tbl")
            and not _declares(payload, entry["local_id"])
        ]
        if missing:
            failures.append(
                "declared entities absent, leaving dangling references: "
                + ", ".join(sorted(missing)[:8])
            )
    return failures


def design_model_mismatch(payload: Mapping[str, Any]) -> list[str]:
    """The design says a factor was crossed; the model has no factor to cross it with.

    Both halves are in the same payload, so this needs neither a reference record nor a
    model call. It is the signature of the one systematic failure that survives the other
    post-conditions: the pass models a crossover as several unrelated continuous terms, one
    per condition, and every contrast then reduces to a single signed slope with no level to
    hold. Over 82 runs of every configuration measured it flagged 27 bad records against 1
    false alarm, at 64% recall.

    Reported and not retried. The term types are chosen by the `demands` pass, and `satisfy`
    is under instruction to honour them, so retrying `satisfy` asks the wrong pass to undo a
    decision it did not make -- observed spending a whole retry budget without ever clearing
    the fault. Acting on it means re-running `demands`, which is a caller's decision.
    """

    design = (
        payload.get("study", {}).get("design")
        if isinstance(payload.get("study"), Mapping)
        else None
    )
    design = design if isinstance(design, Mapping) else payload.get("design")
    if not isinstance(design, Mapping):
        return []
    arms = [a for a in (design.get("arms") or []) if isinstance(a, Mapping)]
    timepoints = [t for t in (design.get("timepoints") or []) if isinstance(t, Mapping)]
    if len(arms) + len(timepoints) < 2:
        return []
    for model in payload.get("model_estimations") or []:
        if not isinstance(model, Mapping):
            continue
        for term in model.get("terms") or []:
            if not isinstance(term, Mapping):
                continue
            kind = term.get("type")
            if (kind.get("value") if isinstance(kind, Mapping) else kind) == "categorical":
                return []
    return [
        f"the design declares {len(arms)} arm(s) and {len(timepoints)} timepoint(s) but "
        "no model term is categorical, so nothing in the model can express the "
        "comparison the design says was made"
    ]


def _declares(payload: Mapping[str, Any], local_id: str) -> bool:
    """Whether the payload contains an entity with this id, at any depth."""

    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            if node.get("local_id") == local_id:
                return True
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return False


RETRY_NOTE = """

## Your previous answer was rejected, and this is the retry

{failures}

Emit the complete object this time. Everything the instructions above ask for still applies;
what changed is only that an answer with the fault named here is not acceptable."""


def normalize(payload: dict[str, Any], mode: str) -> tuple[dict[str, Any], list[str]]:
    """Move stray Study attributes under `study` so merge_payloads accepts them.

    Reported rather than silently corrected: a key landing here is a prompt problem
    worth seeing, not a quirk of the model to paper over.
    """

    notes: list[str] = []
    payload.pop("extraction_metadata", None)
    study = payload.get("study")
    if not isinstance(study, dict):
        study = {}

    # An entity list nested under `study` survives merge_payloads, but only until a
    # sibling empty list at the top level shadows it. Hoist it and say so: the model
    # emitting both shapes at once is a prompt problem worth seeing.
    for key in list(study):
        if key in ENTITY_LISTS and isinstance(study[key], list):
            hoisted = study.pop(key)
            if hoisted:
                if payload.get(key):
                    notes.append(f"collision: {key!r} emitted both top-level and under study")
                payload[key] = hoisted
                notes.append(f"hoisted {key!r} out of study to the top level")

    for key in list(payload):
        # `required_entities` is a top-level output of the demands pass, not a stray Study
        # attribute; sweeping it under `study` would hide it and the next line drops it.
        if key in ENTITY_LISTS or key in ("study", "required_entities"):
            continue
        study[key] = payload.pop(key)
        notes.append(f"moved top-level {key!r} under study")

    # arms/timepoints are accepted as top-level payload keys by merge_payloads, which
    # writes them to design.arms by assignment -- so a payload carrying both forms
    # loses one. Keep the nested form, which is where the prompt asks for them.
    design = study.get("design")
    if isinstance(design, dict):
        for key in ("arms", "timepoints"):
            if key in payload and design.get(key):
                payload.pop(key)
                notes.append(f"dropped top-level {key!r} in favour of study.design.{key}")

    if mode == "demands":
        # An Analysis without an `effect` is not one -- the schema requires it -- so an
        # entry shaped like that is a declaration the model filed in the wrong list. Moved
        # rather than dropped, and reported, because losing the shopping list leaves the
        # next pass with nothing to be held to and the failure is silent.
        analyses = payload.get("analyses")
        if isinstance(analyses, list):
            declarations = [
                a for a in analyses if isinstance(a, Mapping) and not a.get("effect")
            ]
            if declarations:
                payload["analyses"] = [a for a in analyses if a not in declarations]
                declared = payload.setdefault("required_entities", [])
                known = {d.get("local_id") for d in declared if isinstance(d, Mapping)}
                for entry in declarations:
                    if entry.get("local_id") not in known:
                        declared.append(
                            {
                                "local_id": entry.get("local_id"),
                                "kind": entry.get("kind"),
                                "label": (
                                    (entry.get("name") or {}).get("value")
                                    if isinstance(entry.get("name"), Mapping)
                                    else entry.get("label")
                                ),
                            }
                        )
                notes.append(
                    f"moved {len(declarations)} effect-less 'analyses' entries "
                    "into required_entities"
                )

    analysis_side = MODE_SCHEMA.get(mode, mode) == "analyses"
    if analysis_side:
        # `required_entities` is this pass's second output, not a stray key: the shopping
        # list is what the entity pass is then held to.
        keep = ("analyses", "study") + (("required_entities",) if mode == "demands" else ())
        for key in list(payload):
            if key not in keep:
                payload.pop(key)
                notes.append(f"dropped {key!r}: not this pass's output")
        payload.pop("study", None)
    if study and not analysis_side:
        payload["study"] = study
    return payload, notes
