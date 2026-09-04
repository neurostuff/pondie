"""The thirteen things a record can be that are structurally legal and scientifically wrong.

Separate from `validate.py` because they answer a different question for a different reader.
That module asks "does this conform to the LinkML schema" and knows only the language; these
know what a crossover is, what a product column means, and why an analysis contrasting two
timepoints is not a treatment contrast. A neuroimaging reviewer can check these without
knowing LinkML, and could not while they were 800-odd lines inside a 1,500-line class.

They need nothing from the validator but somewhere to put a finding, which is what `Findings`
is. Measured before the split: no rule mutates the record (16 records, each rule run alone,
re-serialised), none reads validator state, and 187 runs over 12 random orderings per record
produced identical finding sets. `RULES` therefore carries no ordering constraint, unlike
`repairs.build_sequence()` where later repairs read what earlier ones wrote -- copying that
mechanism here would assert a dependency that does not exist.

The *order* of the output does follow `RULES`, and nothing consumes it -- but a consumer that
started to would be depending on something arbitrary.

Each rule recomputes what it needs. `_model_index` is rebuilt 5 times per record and
`terms_in_scope` 20 times, for 8ms across the whole rule half; threading a shared index
through thirteen signatures would buy none of that back and would reintroduce exactly the
shared mutable state the measurements above rule out.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pondie.extraction.record import spans as span_tools
from pondie.extraction.record.effect import terms_in_scope
from pondie.formats import values


class Findings(Protocol):
    """Somewhere to put a finding. `Validator` satisfies it as it stands.

    An error is a record a consumer cannot rely on; a warning is one a reviewer should look
    at. Two methods and no state, so a rule cannot read what another rule wrote.
    """

    def error(self, path: str, message: str) -> None: ...

    def warn(self, path: str, message: str) -> None: ...



# ------------------------------------------------------------------------------------
# Reading a record's prose. A trigger, never a verdict: these route a record to review
# and the rule that calls one decides what to do about it.
# ------------------------------------------------------------------------------------

_CROSSING_WORDS = ("interaction", "moderat", "×")
_BY_CROSSING = re.compile(r"([a-z]+)-by-([a-z]+)")


#: Comparison syntax in a `ModelTerm.name`. A term is the *axis* of a comparison, never
#: the comparison, so a name stating one is a factor written down from its contrast's
#: side -- the shape `check_occasion_factors` looks for. The operator pattern requires a
#: word character on both sides so that a threshold such as "p < .001" is not an axis.
_COMPARISON_WORDS = (
    "versus",
    " vs ",
    " vs. ",
    "greater than",
    "less than",
    "difference between",
    "change in",
    "change from",
    "pre-post",
    "pre/post",
    "prepost",
)
_COMPARISON_OPERATOR = re.compile(r"[a-z0-9)\]]\s*[<>]\s*[a-z0-9(\[]")

#: Derivation language in a `ModelTerm.name` -- a column computed from several of an
#: instrument's measurements rather than being one of them. Multi-word on purpose. A
#: bare "percent" catches "percentage methylation at CpG sites 11-12", which is a
#: measurement and not a difference; a bare "change" catches "pre > post rsFC change",
#: which is a collapsed occasion factor and `check_occasion_factors`' finding rather
#: than this one.
_DERIVED_WORDS = (
    "change in",
    "change from",
    "change over",
    "percent change",
    "percentage change",
    "percent reduction",
    "difference between",
    "difference in",
    "improvement in",
    "delta ",
)

#: Prose claiming a result is a change across occasions, read off an analysis's `name`
#: and `definition`. Deliberately not "baseline": a record whose analyses are all
#: baseline-only is the legitimate reading of a design that scanned twice and reported
#: once, and it is not what this looks for.
_CHANGE_WORDS = (
    "change",
    "longitudinal",
    "over time",
    "follow-up",
    "followup",
    "pre > post",
    "post > pre",
    "pre-post",
    "following treatment",
    "after treatment",
)


def _prose(*fields: Any) -> str:
    read = [values.read(node) for node in fields]
    return " ".join(str(value) for value in read if value is not None).lower()


def names_a_comparison(*fields: Any) -> bool:
    """Does this term's own name state the comparison it was the subject of?"""

    text = _prose(*fields)
    return (
        any(word in text for word in _COMPARISON_WORDS)
        or _COMPARISON_OPERATOR.search(text) is not None
    )


def names_a_change_over_time(*fields: Any) -> bool:
    """Does this prose claim a result is a change from one occasion to another?"""

    text = _prose(*fields)
    return any(word in text for word in _CHANGE_WORDS)


def names_a_derivation(*fields: Any) -> bool:
    """Does this term's name say its values were computed from several measurements?"""

    text = _prose(*fields)
    return any(word in text for word in _DERIVED_WORDS)


def names_a_crossing(*fields: Any) -> bool:
    """Does this prose claim a crossing was tested?

    Read off the analysis's own `name` and `definition` rather than off its cells,
    because the whole point is to compare the two: the cells are what the record
    says, and this is what the paper said.
    """

    text = _prose(*fields)
    if any(word in text for word in _CROSSING_WORDS):
        return True
    return any(left != right for left, right in _BY_CROSSING.findall(text))

@dataclass(frozen=True)
class Rule:
    """One check, and what it is for. Registered by existing, not by being called."""

    name: str
    what: str
    fn: Callable[[Mapping[str, Any], Findings], None]


def _model_index(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    models: dict[str, Mapping[str, Any]] = {}
    for model in record.get("model_estimations") or []:
        if isinstance(model, Mapping) and isinstance(model.get("local_id"), str):
            models[model["local_id"]] = model
    return models

def _cell_signature(model_id: Any, cells: Any) -> tuple:
    """What an effect compared: its model, and its cells as a set.

    Stringified because a malformed record can put a list where a level belongs,
    and an unhashable signature would raise inside a check whose job is to report.
    """

    parts = sorted(
        (
            str(cell.get("term")),
            str(values.read(cell.get("level"))),
            str(values.read(cell.get("direction"))),
        )
        for cell in (cells or [])
        if isinstance(cell, Mapping)
    )
    return (str(model_id), tuple(parts))

def check_crossings(record: Mapping[str, Any], findings: Findings) -> None:
    """Flag interactions the cells do not actually record.

    The defect this catches is a product column that was never declared, so the
    cell that should sit on it had nowhere to go. It is invisible to every check
    above: each cell resolves, each level agrees with its term, and the record is
    structurally perfect while an interaction and a main effect have become the
    same record. `representing-models.md` §5.5 is where the shape is written down
    -- an interaction reported as an unsigned F or chi-square has nowhere to sit
    but a `ModelTerm` with `interaction_with`, because a factor that was crossed
    rather than averaged over carries no sign to cross it with.

    Warnings, not errors. The trigger reads prose, so it routes a record to review
    rather than rejecting it; a paper whose interaction really was reported as a
    directional per-level comparison needs no product column
    (`extraction-readme.md`, "the converse is a reporting habit worth naming")
    and is expected to answer for itself under review.
    """

    models = _model_index(record)
    # signature -> the analyses sharing it. Two analyses of one model with the
    # same cells are the same estimand, so if their prose disagrees about what was
    # tested, at most one of them can be right.
    signatures: dict[tuple, list[tuple[str, bool]]] = {}

    for index, analysis in enumerate(record.get("analyses") or []):
        if not isinstance(analysis, Mapping):
            continue
        path = f"Study.analyses[{index}]"
        model_id = analysis.get("model_estimation")
        terms = terms_in_scope(model_id, models)
        effect = analysis.get("effect")
        cells = (effect.get("cells") if isinstance(effect, Mapping) else None) or []
        claimed = names_a_crossing(analysis.get("name"), analysis.get("definition"))

        signed: dict[str, set[str]] = {}
        products: list[str] = []
        held = False
        for cell in cells:
            if not isinstance(cell, Mapping):
                continue
            term_id = cell.get("term")
            if not isinstance(term_id, str):
                continue
            term = terms.get(term_id)
            if isinstance(term, Mapping) and (term.get("interaction_with") or []):
                products.append(term_id)
            direction = values.read(cell.get("direction"))
            if direction in {"positive", "negative"}:
                signed.setdefault(term_id, set()).add(direction)
            # `held` on a named level is a factor held constant and nothing else: an
            # undirected test is `undirected` and a withheld sign is `not_reported`, so this
            # cannot catch an omnibus F by accident. An analysis reported within one level
            # of the crossing is §5.5's last row -- a legitimate simple effect, whose prose
            # names the interaction it came from and whose cells are not supposed to
            # record it.
            if values.read(cell.get("level")) is not None and direction == "held":
                held = True
        # A term signed once has not been compared against itself.
        crossed = [term_id for term_id, sides in signed.items() if len(sides) == 2]

        if claimed and not products and not held and len(crossed) < 2:
            findings.warn(
                f"{path}.effect.cells",
                "names a crossing the cells do not record: no cell sits on a product "
                f"column and {len(crossed)} term(s) are crossed, so the derived kind is "
                "the one a main effect of the same terms would get. Either the crossing "
                "needs its sides on both factors' levels, or the model is missing the "
                "ModelTerm with interaction_with that carries an unsigned interaction test",
            )

        signatures.setdefault(_cell_signature(model_id, cells), []).append(
            (path, claimed)
        )

    for (model_id, _), members in signatures.items():
        if len(members) < 2:
            continue
        crossing = [path for path, claimed in members if claimed]
        plain = [path for path, claimed in members if not claimed]
        if crossing and plain:
            findings.warn(
                f"{crossing[0]}.effect.cells",
                f"identical to {', '.join(plain)} on {model_id}, which names no crossing: "
                "the same cells over the same model are the same estimand, so an "
                "interaction and a main effect cannot both be what these record",
            )

def check_product_columns(record: Mapping[str, Any], findings: Findings) -> None:
    """Flag product columns that nothing can reach.

    Two ways a declared column fails to do its job. Its components may name a term
    outside its own stage chain -- which `check_local_ids` cannot see, because the
    reference resolves, but to a sibling model's column with the same name. No
    cell anywhere may name it, which is legal (a design-matrix column that only
    ever adjusted something, per §5.5's main-effect rows) but is also what a paper
    looks like when its interaction table was never extracted at all.
    """

    models = _model_index(record)
    celled = {
        cell.get("term")
        for analysis in record.get("analyses") or []
        if isinstance(analysis, Mapping)
        for cell in (
            (analysis.get("effect") or {}).get("cells")
            if isinstance(analysis.get("effect"), Mapping)
            else None
        )
        or []
        if isinstance(cell, Mapping)
    }

    for m_index, model in enumerate(record.get("model_estimations") or []):
        if not isinstance(model, Mapping):
            continue
        model_id = model.get("local_id")
        terms = terms_in_scope(model_id, models)
        for t_index, term in enumerate(model.get("terms") or []):
            if not isinstance(term, Mapping):
                continue
            components = term.get("interaction_with") or []
            if not components:
                continue
            path = f"Study.model_estimations[{m_index}].terms[{t_index}]"
            for c_index, component in enumerate(components):
                if isinstance(component, str) and component not in terms:
                    findings.warn(
                        f"{path}.interaction_with[{c_index}]",
                        f"{component!r} is not a term of {model_id!r} or of a stage it "
                        "reaches through inputs_from, so this column names a component "
                        "it cannot be a product of",
                    )
            if term.get("local_id") not in celled:
                findings.warn(
                    path,
                    f"product column {term.get('local_id')!r} carries no cell in any "
                    "analysis. Legal if it only adjusted the effects that were reported, "
                    "but it is also what a missing interaction analysis looks like",
                )

# -- the two unsigned values --------------------------------------------

def check_unsigned_cells(record: Mapping[str, Any], findings: Findings) -> None:
    """Flag the two shapes `held` cannot have.

    `representing-models.md` §4 cuts the three unsigned values by two questions.
    Was the level on both sides of the comparison at once? That is `held`, and it
    is the whole of what `held` says on a `Cell`. Otherwise, does the test yield a
    per-level sign at all -- no for an F or chi-square, which is `undirected`; yes
    but unprinted, which is `extraction_status: not_reported`.

    Both halves of the first question are checkable:

    * a cell naming **no level** -- on a slope or a product column -- has no level
      to put on both sides, so it can only be an undirected test miscoded;
    * a factor **all** of whose declared levels are celled `held` is an undirected
      test of that factor, since holding a level constant is a claim about one
      level and leaves the others absent.

    The partial case is deliberately not flagged. A contrast taken within two of a
    factor's three levels holds both of them, and reads as two `held` cells with
    the third absent -- so the trigger is *every declared level celled*, not *more
    than one*.

    Warnings, for `check_crossings`' reason: this is what a record extracted under
    the old reading looks like, and it routes to review rather than rejecting.
    """

    models = _model_index(record)

    for index, analysis in enumerate(record.get("analyses") or []):
        if not isinstance(analysis, Mapping):
            continue
        path = f"Study.analyses[{index}].effect.cells"
        terms = terms_in_scope(analysis.get("model_estimation"), models)
        effect = analysis.get("effect")
        cells = (effect.get("cells") if isinstance(effect, Mapping) else None) or []

        # term -> the levels it celled, and which of those were unsigned this way.
        celled: dict[str, list[Any]] = {}
        unsigned: dict[str, list[Any]] = {}
        for cell in cells:
            if not isinstance(cell, Mapping):
                continue
            term_id = cell.get("term")
            if not isinstance(term_id, str):
                continue
            level = values.read(cell.get("level"))
            celled.setdefault(term_id, []).append(level)
            if values.read(cell.get("direction")) != "held":
                continue
            if level is None:
                findings.warn(
                    path,
                    f"cell on {term_id!r} is held and names no level. A slope or a product "
                    "column has no level to sit on both sides of the comparison, which is "
                    "the only thing held says on a cell; an undirected test of such a "
                    "column is undirected (representing-models.md 4)",
                )
                continue
            unsigned.setdefault(term_id, []).append(level)

        for term_id, levels in unsigned.items():
            term = terms.get(term_id)
            declared = [
                values.read(level.get("level"))
                for level in (term.get("levels") if isinstance(term, Mapping) else None)
                or []
                if isinstance(level, Mapping)
            ]
            if len(declared) < 2 or set(declared) - set(celled.get(term_id, [])):
                continue
            if len(levels) < len(declared):
                continue
            findings.warn(
                path,
                f"every declared level of {term_id!r} is celled held, which says the "
                "factor was held on both sides of its own test. An undirected test over a "
                "factor is undirected on each level; held marks one level and leaves the "
                "rest absent (representing-models.md 4)",
            )

# -- occasions, and the factors that should carry them ------------------

def check_occasion_factors(record: Mapping[str, Any], findings: Findings) -> None:
    """Flag a comparison the record collapsed into a single column.

    Invisible to every check above, in the way `check_crossings`' defect is: each
    cell resolves, each level agrees with its term, and the comparison the paper
    reported is gone. Two halves, from the two ends.

    The term half is a column named after the comparison it was the *subject* of --
    `pre > post change`, continuous, no levels. The design matrix distinguished two
    occasions; the paper labelled the difference and the axis went unnamed. One cell
    on a continuous term then derives a regression where a contrast belongs.

    The design half is the same defect from the other end: several occasions
    declared, analyses reporting change over time, and no `FactorLevel.timepoints`
    naming any of them. That slot is the only route to a `Timepoint`, so when it is
    empty the scans are recorded and the comparison between them is not.

    `ModelTerm.type` and `representing-models.md` §5.6 state the shape. Warnings,
    for `check_crossings`' reason -- the trigger reads prose. Left alone: a genuine
    per-participant covariate, which is continuous, named for its subtraction, and
    `between_subject`, an occasion factor varying within a participant by definition.
    """

    for m_index, model in enumerate(record.get("model_estimations") or []):
        if not isinstance(model, Mapping):
            continue
        for t_index, term in enumerate(model.get("terms") or []):
            if not isinstance(term, Mapping):
                continue
            if values.read(term.get("type")) != "continuous" or term.get("levels"):
                continue
            # A product column has no levels either, and is legitimately named
            # for the crossing it is a product of.
            if term.get("interaction_with"):
                continue
            # A column an instrument or a place in the brain supplies is a real
            # measurement whatever the source called it.
            if term.get("assessment") or term.get("region"):
                continue
            if values.read(term.get("variation_level")) == "between_subject":
                continue
            if not names_a_comparison(term.get("name")):
                continue
            findings.warn(
                f"Study.model_estimations[{m_index}].terms[{t_index}].name",
                f"{values.read(term.get('name'))!r} states a comparison while the term is "
                "continuous with no levels, so nothing records which occasions, "
                "cohorts or conditions were on each side. A comparison is a "
                "categorical term with a level per side and the sign on the cells "
                "(representing-models.md 5.6)",
            )

    design = record.get("design")
    timepoints = (design.get("timepoints") if isinstance(design, Mapping) else None) or []
    declared = [timepoint for timepoint in timepoints if isinstance(timepoint, Mapping)]
    if len(declared) < 2:
        return

    referenced = {
        local_id
        for model in record.get("model_estimations") or []
        if isinstance(model, Mapping)
        for term in model.get("terms") or []
        if isinstance(term, Mapping)
        for level in term.get("levels") or []
        if isinstance(level, Mapping)
        for local_id in level.get("timepoints") or []
    }
    if referenced:
        return

    reporting = [
        f"analyses[{index}]"
        for index, analysis in enumerate(record.get("analyses") or [])
        if isinstance(analysis, Mapping)
        and names_a_change_over_time(analysis.get("name"), analysis.get("definition"))
    ]
    if not reporting:
        return

    shown = ", ".join(reporting[:3]) + (" and others" if len(reporting) > 3 else "")
    findings.warn(
        "Study.design.timepoints",
        f"{len(declared)} occasions are declared and no FactorLevel.timepoints "
        f"names any of them, while {len(reporting)} analysis(es) report change over "
        f"time ({shown}). That slot is the only route to a Timepoint, so as "
        "recorded the scans are here and the comparison between them is not",
    )

# -- arms, and the analyses that cannot say which one they are ----------

def check_arm_reachability(record: Mapping[str, Any], findings: Findings) -> None:
    """Flag an analysis that cannot be linked to an arm.

    An `Arm` reaches an analysis two ways: `FactorLevel.arms` on a term
    some cell names, when the arm was *compared*, and `Group.arm` on a cohort the
    analysis ran on, when the cohort was *assigned* to it. An analysis whose name or
    definition says "heroin" while neither route lands anywhere is one whose arm the
    record states in prose and nowhere a query can reach.

    Two things produce that, and the message names both because a reviewer has to
    tell them apart. Either a `Cell.level` matches no `FactorLevel` and the join to
    the arm broke on the string -- `check_cell_terms` reports that from its own end
    -- or the arm was held constant rather than compared, which is the crossover
    case: one arm's data, no column naming it, and no cohort to hang it on because
    every participant is in both. The schema has no slot for the second, deliberately
    (storage-schema-design-notes.md, "An arm held constant"), so this warning is the
    only place it surfaces.

    Warning, not error, for `check_crossings`' reason: the trigger reads prose. An
    analysis that never names an arm is left alone, which is what keeps a baseline
    contrast in a treatment study, or a substudy pooled across arms, silent.
    """

    design = record.get("design")
    arms = (design.get("arms") if isinstance(design, Mapping) else None) or []
    vocabulary: list[tuple[str, re.Pattern]] = []
    for arm in arms:
        if not isinstance(arm, Mapping) or not isinstance(arm.get("local_id"), str):
            continue
        # `name` and `agent` both, because a paper writes either: "THC" is the arm's
        # name in one record and its agent in another. Short strings are dropped --
        # a two-character arm name matches prose that has nothing to do with it.
        words = {str(values.read(arm.get(slot))) for slot in ("name", "agent")}
        for word in words:
            if len(word.strip()) < 3 or word == "None":
                continue
            vocabulary.append(
                (arm["local_id"], re.compile(rf"\b{re.escape(word.lower())}\b"))
            )
    if not vocabulary:
        return

    models = {
        model["local_id"]: model
        for model in record.get("model_estimations") or []
        if isinstance(model, Mapping) and isinstance(model.get("local_id"), str)
    }
    group_arms = {
        group["local_id"]: group.get("arm")
        for group in record.get("groups") or []
        if isinstance(group, Mapping) and isinstance(group.get("local_id"), str)
    }

    for index, analysis in enumerate(record.get("analyses") or []):
        if not isinstance(analysis, Mapping):
            continue
        prose = _prose(analysis.get("name"), analysis.get("definition"))
        named = sorted({arm_id for arm_id, pattern in vocabulary if pattern.search(prose)})
        if not named:
            continue

        terms = terms_in_scope(analysis.get("model_estimation"), models)
        effect = analysis.get("effect")
        cells = (effect.get("cells") if isinstance(effect, Mapping) else None) or []
        reached = set()
        for cell in cells:
            if not isinstance(cell, Mapping):
                continue
            term = terms.get(cell.get("term"))
            if not isinstance(term, Mapping):
                continue
            wanted = values.read(cell.get("level"))
            for level in term.get("levels") or []:
                if not isinstance(level, Mapping):
                    continue
                if wanted is None or values.read(level.get("level")) == wanted:
                    reached.update(level.get("arms") or [])
        for entry in analysis.get("groups") or []:
            if isinstance(entry, Mapping):
                reached.add(group_arms.get(entry.get("group")))
        if reached - {None}:
            continue

        findings.warn(
            f"Study.analyses[{index}]",
            f"prose names arm(s) {', '.join(named)} while no cell's level and no "
            "analysed cohort reaches an Arm, so nothing queryable says which arm "
            "this map is. Either a Cell.level matches no FactorLevel that names the "
            "arm, or the arm was held constant for this analysis, which has no slot",
        )

# -- derived columns and where they came from ---------------------------

def check_derived_columns(record: Mapping[str, Any], findings: Findings) -> None:
    """Flag a derived column whose origin the record does not state.

    A change score or percent change is one number per participant computed from
    several of an instrument's administrations, and two slots make it interpretable:
    `assessment` names the instrument, `source_definition` says what the derivation
    was and over which occasions.

    Neither is optional here. Deriving a column does not break the link to its
    instrument -- `region` says as much by example, an ROI mean and a PPI regressor
    both naming their region. `source_definition` is the *only* place the
    occasions can go, since a column with no levels has no `FactorLevel.timepoints`:
    in a study with several post-intervention occasions it is what separates a change
    to the endpoint from a change to a later follow-up.

    Warnings, for the reason above: the trigger reads a name. The vocabulary is
    narrow deliberately, so a column named for what it measures rather than for how
    it was built is left alone.
    """

    assessments = len(record.get("assessments") or [])

    for m_index, model in enumerate(record.get("model_estimations") or []):
        if not isinstance(model, Mapping):
            continue
        for t_index, term in enumerate(model.get("terms") or []):
            if not isinstance(term, Mapping):
                continue
            if values.read(term.get("type")) != "continuous" or term.get("levels"):
                continue
            if not names_a_derivation(term.get("name")):
                continue

            path = f"Study.model_estimations[{m_index}].terms[{t_index}]"
            name = values.read(term.get("name"))

            if not values.read(term.get("source_definition")):
                findings.warn(
                    f"{path}.source_definition",
                    f"{name!r} is a derived column and its derivation is not "
                    "recorded. Nothing else can say what was subtracted from what, "
                    "or over which occasions: a column with no levels has no "
                    "FactorLevel.timepoints to name them",
                )

            if term.get("assessment") is None and assessments:
                findings.warn(
                    f"{path}.assessment",
                    f"{name!r} reads as derived from an instrument's measurements "
                    f"but names no assessment, while the record declares "
                    f"{assessments}. Deriving a column does not break the link to "
                    "the instrument it came from",
                )

# -- entry point -------------------------------------------------------

def check_cell_terms(record: Mapping[str, Any], findings: Findings) -> None:
    """§3 invariants 2, 3 and 4: a cell names a term of its own stage chain, and a
    level it names is one that term declares.

    The three travel together because they are one join failed at different depths. A
    `Cell.term` that resolves nowhere, or to a term of a model this analysis does not
    reach through `inputs_from`, makes the cell a sign of nothing. A `Cell.level` that
    matches no declared `FactorLevel.level` is worse than absent: the record looks like
    it recorded which condition was compared, and the mapper's string join will not find
    it, so the comparison is unrecoverable from the entity side.

    `check_crossings` deliberately stays silent on an unresolvable term -- it is asking
    a different question and cannot answer it for a broken cell -- which is why this
    reports instead.
    """

    models = _model_index(record)
    for index, analysis in enumerate(record.get("analyses") or []):
        if not isinstance(analysis, Mapping):
            continue
        model_id = analysis.get("model_estimation")
        terms = terms_in_scope(model_id, models)
        effect = analysis.get("effect")
        if not isinstance(effect, Mapping):
            continue
        local = analysis.get("local_id") or f"analyses[{index}]"

        pointers = [
            (f"analyses[{index}].effect.cells[{n}]", cell.get("term"), cell)
            for n, cell in enumerate(effect.get("cells") or [])
            if isinstance(cell, Mapping)
        ]
        mediation = effect.get("mediation")
        if isinstance(mediation, Mapping):
            pointers.append(
                (f"analyses[{index}].effect.mediation", mediation.get("mediator"), None)
            )

        for path, term_id, cell in pointers:
            if not isinstance(term_id, str):
                continue
            term = terms.get(term_id)
            if term is None:
                owner = next(
                    (
                        m.get("local_id")
                        for m in models.values()
                        for t in (m.get("terms") or [])
                        if isinstance(t, Mapping) and t.get("local_id") == term_id
                    ),
                    None,
                )
                if owner is None:
                    findings.error(path, f"term {term_id!r} names no ModelTerm anywhere")
                else:
                    findings.error(
                        path,
                        f"term {term_id!r} belongs to {owner!r}, which "
                        f"{local!r}'s model ({model_id!r}) does not reach "
                        "through inputs_from",
                    )
                continue
            if cell is None:
                continue

            level = values.read(cell.get("level"))
            if not isinstance(level, str):
                continue
            declared = [
                values.read(entry.get("level"))
                for entry in (term.get("levels") or [])
                if isinstance(entry, Mapping)
            ]
            declared = [name for name in declared if isinstance(name, str)]
            if not declared:
                findings.error(
                    f"{path}.level",
                    f"is {level!r} but term {term_id!r} declares "
                    "no levels to match it against",
                )
            elif level not in declared:
                findings.error(
                    f"{path}.level",
                    f"{level!r} matches none of term {term_id!r}'s declared levels "
                    f"({', '.join(repr(name) for name in declared)}); the mapper joins "
                    "these on the string",
                )

def check_model_stages(record: Mapping[str, Any], findings: Findings) -> None:
    """§3 invariants 6 and 7: `inputs_from` is acyclic, and a term name is unique across
    a whole stage chain.

    Both are already *survived* rather than reported. `terms_in_scope` carries a `seen`
    set so the walk terminates on a cycle, and collects own terms last so a same-named
    lower-stage term is shadowed -- which is the right reading when the name is a
    deliberate refit and silently absorbs the violation when it is not. A validator that
    merely does not hang on bad input has not reported it.

    Neither has ever fired on the corpus. They are in because a cycle is a hang as well
    as a falsehood, and because a first-level `motion` shadowing a group-level `motion`
    makes two columns indistinguishable in one term list -- a reader cannot tell a column
    refitted at the stage above from one restated there by mistake.
    """

    models = _model_index(record)

    for model_id, model in models.items():
        path = f"model_estimations[{model_id}]"

        def walk(current: Any, trail: tuple[str, ...]) -> None:
            if not isinstance(current, str):
                return
            if current in trail:
                cycle = " -> ".join(trail[trail.index(current) :] + (current,))
                findings.error(
                    f"{path}.inputs_from",
                    f"inputs_from is cyclic: {cycle}. A model fitted on its own "
                    "output is not a stage order",
                )
                return
            lower = models.get(current)
            if isinstance(lower, Mapping):
                for below in lower.get("inputs_from") or []:
                    walk(below, trail + (current,))

        for start in model.get("inputs_from") or []:
            walk(start, (model_id,))

        # Names across the chain, which is where the collision matters -- `unique_keys`
        # scopes per record and the projection drops it anyway.
        seen: dict[str, str] = {}
        for owner_id, term in _chain_terms(model_id, models):
            name = values.read(term.get("name"))
            if not isinstance(name, str) or not name.strip():
                continue
            folded = span_tools.fold_label(name)
            if folded in seen and seen[folded] != owner_id:
                findings.error(
                    f"{path}.terms",
                    f"term name {name!r} appears on both {seen[folded]!r} and "
                    f"{owner_id!r} in one stage chain, so a reader cannot tell a column "
                    "refitted at the stage above from one restated there by mistake",
                )
            seen[folded] = owner_id

def _chain_terms(model_id: Any,
    models: Mapping[str, Mapping[str, Any]],
    seen: set[str] | None = None,
) -> list[tuple[str, Mapping[str, Any]]]:
    """`(owning model local_id, term)` for a model and every stage below it.

    Unlike `terms_in_scope`, which keys by term local_id and so *loses* the duplicate
    this reports, this keeps every term with the stage that declared it.
    """

    seen = set() if seen is None else seen
    if not isinstance(model_id, str) or model_id in seen:
        return []
    seen.add(model_id)
    model = models.get(model_id)
    if not isinstance(model, Mapping):
        return []
    found: list[tuple[str, Mapping[str, Any]]] = []
    for lower in model.get("inputs_from") or []:
        found += _chain_terms(lower, models, seen)
    found += [
        (model_id, term) for term in model.get("terms") or [] if isinstance(term, Mapping)
    ]
    return found

def check_table_purpose(record: Mapping[str, Any], findings: Findings) -> None:
    """A coordinate table either reports an analysis or says what it does instead.

    `Table.non_analysis_content` is the only field that can say a table's rows are
    locations rather than findings -- ROI definitions, atlas parcels, the peaks of an
    ICA's components. Two things follow, and the second is the one worth having:

    - a table marked as non-analysis that an `Analysis.tables` nonetheless names is a
      contradiction, and one of the two is wrong;
    - a table **no** analysis names and that carries no marking is the missed-analysis
      case. Before this field existed that was indistinguishable from a table
      deliberately left unencoded, and both read as the same silence. `spec.py`'s
      `not_analyses` and `missed_analysis` verdicts have always been separate; this is
      what lets the record tell them apart before a reviewer does.

    A warning rather than an error on the second, because an unencoded table is a
    judgement to review and not a malformed record.
    """

    referenced = {
        name
        for analysis in record.get("analyses") or []
        if isinstance(analysis, Mapping)
        for name in (analysis.get("tables") or [])
        if isinstance(name, str)
    }
    for index, table in enumerate(record.get("tables") or []):
        if not isinstance(table, Mapping):
            continue
        local_id = table.get("local_id")
        path = f"tables[{index}]"
        marked = values.read(table.get("non_analysis_content"))
        if marked and local_id in referenced:
            findings.error(
                f"{path}.non_analysis_content",
                f"says this table reports {marked!r} rather than an effect, but an "
                "analysis names it in `tables`",
            )
        elif not marked and local_id not in referenced:
            findings.warn(
                path,
                "no analysis names this table and non_analysis_content is empty, so "
                "nothing says whether it was deliberately not encoded or missed",
            )

def check_references_resolve(
    record: Mapping[str, Any],
    findings: Findings,
    *,
    owner: str,
    slot: str,
    container: str,
    tail: str,
    multivalued: bool = False,
) -> None:
    """Every id in `owner[].slot` must name an entry of `container`.

    `tail` completes "names {id}, which is not ...", so each caller keeps its own account
    of what the dangling id costs. That wording is the whole value of the check to a
    reviewer, which is why this shares the traversal and not the message.

    A non-string is not a dangling id but a record still carrying the target inline;
    `check_slot` reports that shape from the schema, and reporting it twice helps nobody.
    """

    known = {
        entry.get("local_id")
        for entry in record.get(container) or []
        if isinstance(entry, Mapping)
    }
    for index, item in enumerate(record.get(owner) or []):
        if not isinstance(item, Mapping):
            continue
        found = item.get(slot)
        ids = found if multivalued and isinstance(found, list) else [found]
        for position, local_id in enumerate(ids):
            if not isinstance(local_id, str) or local_id in known:
                continue
            path = (
                f"{owner}[{index}].{slot}[{position}]"
                if multivalued
                else f"{owner}[{index}].{slot}"
            )
            findings.error(path, f"names {local_id!r}, which is not {tail}")

def check_one_protocol_per_acquisition(
    record: Mapping[str, Any], findings: Findings
) -> None:
    """Several echo times and several repetition times is two protocols, not one sequence.

    `echo_time_seconds` is "one value per echo": a multi-echo sequence has several, and one
    repetition time, because TR belongs to the sequence rather than to an echo. Several of
    both is two acquisitions written into one entity, and `pulse_sequence_type` -- singular,
    for one family -- then holds a conjunction to match: "3D MP-RAGE and 3D FLASH".

    Legal LinkML and scientifically wrong, which is what this module reports. It matters
    because `Acquisition` carries one modality, so a record that fuses a functional EPI with
    a structural scan (TE [0.04, 0.03], TR [3, 2.25]) cannot say which analysis used which:
    the analyses point at one id and the two protocols are behind it.

    Seven of 1,022 acquisitions in the neurometabench corpus do this. It is a warning
    because the values are all present and correctly parsed -- what is wrong is that one
    entity is standing for two, and splitting it is a judgement about the paper.
    """

    for index, acquisition in enumerate(record.get("acquisitions") or []):
        if not isinstance(acquisition, Mapping):
            continue
        echoes = values.read(acquisition.get("echo_time_seconds"))
        repetitions = values.read(acquisition.get("repetition_time_seconds"))
        if not (isinstance(echoes, list) and len(echoes) > 1):
            continue
        if not (isinstance(repetitions, list) and len(repetitions) > 1):
            continue
        local = acquisition.get("local_id") or index
        sequence = values.read(acquisition.get("pulse_sequence_type"))
        findings.warn(
            f"acquisitions[{local}]",
            f"{len(echoes)} echo times and {len(repetitions)} repetition times describe "
            f"two protocols, not one sequence: a repetition time belongs to the sequence, "
            f"not to an echo"
            + (f" -- and pulse_sequence_type reads {sequence!r}" if sequence else "")
            + ". Split them into one Acquisition each, so an analysis can name the one it "
            "used",
        )


def check_group_instruments(record: Mapping[str, Any], findings: Findings) -> None:
    """A group's diagnostic instrument must be one of the study's assessments.

    `Group.diagnostic_instrument` is a reference, so the projection wraps no evidence
    around it: the supporting quote and the purpose the source gave for administering the
    instrument both live on the `Assessment` it points at. A dangling id loses both, and
    leaves a diagnosis whose instrument the record names but nothing describes -- which is
    exactly the distinction between an instrument that classified a cohort and one the
    cohort merely underwent (storage-schema-design-notes.md, "An instrument administered
    is not an instrument that classified").
    """

    check_references_resolve(
        record,
        findings,
        owner="groups",
        slot="diagnostic_instrument",
        container="assessments",
        multivalued=True,
        tail="an assessment of this study. Add it to "
        "`assessments` with the purpose the source states for administering it, or "
        "drop the reference if nothing established this group's diagnosis",
    )

def check_analysis_inference_settings(record: Mapping[str, Any], findings: Findings) -> None:
    """An analysis's thresholding scheme must be one the study declares.

    `Analysis.inference_settings` is a reference, so a dangling id leaves the analysis
    with no threshold, no correction and no alpha at all -- the record then reads as a
    result reported without inference, which is a different claim from one whose
    thresholding the paper never stated (that is a declared scheme whose fields are
    `not_reported`). Sharing is the point of the reference, so the id will usually be one
    several analyses name.
    """

    check_references_resolve(
        record,
        findings,
        owner="analyses",
        slot="inference_settings",
        container="inference_settings",
        tail="an inference settings record of this "
        "study. Add it to `inference_settings` with the thresholding the source "
        "states, or point at the existing scheme this analysis shares",
    )

def check_analysis_measures(record: Mapping[str, Any], findings: Findings) -> None:
    """An analysis's measured quantity must be one the study declares.

    `Analysis.measure` is a required reference, so a dangling id leaves a result whose
    measured quantity nothing in the record states -- which is not the same as a paper
    vague about what it measured. That case is a declared `Measure` carrying the
    source's own wording in `source_label` and `not_reported` elsewhere, and it stays
    queryable; a dangling id is not.
    """

    check_references_resolve(
        record,
        findings,
        owner="analyses",
        slot="measure",
        container="measures",
        tail="a measure of this study. Add it to `measures` with the quantity the "
        "source names, or point at the existing measure this analysis shares",
    )

def check_acquisition_devices(record: Mapping[str, Any], findings: Findings) -> None:
    """An acquisition's device must be one the study declares.

    `Acquisition.device` is a reference, and one record is one physical machine. A
    dangling id loses the manufacturer and model a reader filters studies by, and the
    emptiness reads as a paper that never named its scanner rather than a record that
    lost the reference.
    """

    check_references_resolve(
        record,
        findings,
        owner="acquisitions",
        slot="device",
        container="devices",
        tail="a device of this study. Add it to `devices` with the scanner the source "
        "names, or point at the existing device this acquisition shares",
    )


#: The rules, in declaration order. A rule is registered by being here, which is the point:
#: `check_record` used to hand-call each one by name, so adding a rule meant editing
#: two places and forgetting the second silently disabled it.
#:
#: No `after` field, deliberately. `repairs.build_sequence()` has one because repairs mutate
#: the record and later ones read what earlier ones wrote; these mutate nothing and read no
#: shared state, measured. An ordering constraint written here would assert a dependency
#: that does not exist, and the next person would maintain it.
def check_value_source_honesty(record: Mapping[str, Any], findings: Findings) -> None:
    """A value the paper reported has a sentence, or it is not `reported`.

    `value_source: reported` asserts the source said this; `evidence.status: not_found`
    admits no sentence was ever located for it. Both at once is a contradiction the record
    makes about itself, and the schema already offers the honest alternative: `generated`,
    for a value the pipeline reasoned to rather than read.

    It is the shape of every wrong value found by hand on this corpus. 11549754 -- a paper
    whose corpus text is a PMC landing page -- carries `measures.family` =
    `electrophysiology`, reported, not_found, on a BOLD fMRI study whose own identifiers
    read `mod_fmri` and `mea_neural_response`. On 11515754 the same pairing holds
    `spatial_scope` = `roi`, `definition_method` = `functional_localizer` and
    `correction_scope` = `roi`, none of them stated anywhere in the source.

    A warning and not an error: the reading may well be right, and downgrading it to
    `generated` is a judgement this cannot make. What it can do is stop the two kinds of
    value being indistinguishable in the output.
    """
    from pondie.extraction.evidence.grounding import IDENTIFIERS
    from pondie.formats.values import iter_fields

    for path, node in iter_fields(record):
        if not isinstance(node, Mapping):
            continue
        if node.get("extraction_status") != "extracted":
            continue
        if node.get("value_source") != "reported":
            continue
        if (node.get("evidence") or {}).get("status") != "not_found":
            continue
        # Only what could have come from prose. Unfiltered this fired 1,978 times over 200
        # records, half of it on `caption`, `table_number` and `footer` -- literals copied
        # from the table manifest -- and on `source_table_analysis`, which is an address
        # inside the record rather than a claim about the paper.
        #
        # A `REASONED` slot is deliberately kept. `grounding` exempts those from *scoring*
        # because a paper does not write down that a scope was `roi`, and that is exactly
        # why one asserted as `reported` with no sentence is worth seeing: it is a
        # conclusion wearing the label of a quotation. All four wrong values found by hand
        # on this corpus were of that shape -- `family` = electrophysiology,
        # `spatial_scope` = roi, `definition_method` = functional_localizer,
        # `correction_scope` = roi.
        slot = path.rsplit(".", 1)[-1].split("[")[0]
        if path.startswith("tables[") or slot in IDENTIFIERS:
            continue
        findings.warn(
            path,
            "value_source is 'reported' but no supporting sentence was found. Either cite "
            "one, or set value_source to 'generated' to say the pipeline reasoned to it",
        )


#: What a measure may be, given how the data were acquired. Only the pairings a modality
#: makes impossible are listed; anything unlisted is unconstrained.
_MODALITY_FAMILIES: dict[str, frozenset[str]] = {
    "MRI": frozenset({"electrophysiology", "electrophysiological_amplitude"}),
    "fMRI": frozenset({"electrophysiology", "electrophysiological_amplitude"}),
    "PET": frozenset({"electrophysiology", "electrophysiological_amplitude"}),
    "sMRI": frozenset({"electrophysiology", "electrophysiological_amplitude"}),
    "dMRI": frozenset({"electrophysiology", "electrophysiological_amplitude"}),
    "EEG": frozenset({"functional_bold", "bold_response", "structural_morphometry",
                      "perfusion", "diffusion", "molecular_imaging"}),
}


def check_modality_measures(record: Mapping[str, Any], findings: Findings) -> None:
    """A measure the acquisition could not have produced.

    Nothing related these two, so `measures.family` = `electrophysiology` sat beside an
    `acquisitions.modality` of fMRI without complaint -- on a paper that measured BOLD.
    A reviewer reading either field alone sees nothing wrong; the error is only in the pair.
    """
    from pondie.formats import values as value_tools

    modalities = {
        str(value_tools.read(a.get("modality")) or "").strip()
        for a in record.get("acquisitions") or []
        if isinstance(a, Mapping)
    }
    known = [m for m in modalities if m in _MODALITY_FAMILIES]
    if not known:
        return
    # Forbidden by *every* modality present, not by any one of them. A study with both an
    # sMRI and an fMRI acquisition produces both structural and BOLD measures, and a union
    # would reject each for belonging to the other.
    forbidden = set.intersection(*(set(_MODALITY_FAMILIES[m]) for m in known))
    if not forbidden:
        return
    for index, measure in enumerate(record.get("measures") or []):
        if not isinstance(measure, Mapping):
            continue
        for slot in ("family", "type"):
            named = str(value_tools.read(measure.get(slot)) or "").strip()
            if named and named in forbidden:
                findings.warn(
                    f"measures[{index}].{slot}",
                    f"{named!r} is not something a {'/'.join(sorted(known))} "
                    f"acquisition produces. Either the modality or the measure is wrong",
                )


def check_counts_add_up(record: Mapping[str, Any], findings: Findings) -> None:
    """Arithmetic no model can be trusted with, and none is needed for.

    Two invariants the schema states about itself. A `CategoryDistribution` carries its own
    `denominator`, so its counts sum to that or one of them was misread -- checked against
    the record's own declared total rather than a guessed one. And the enrolment counts are
    a funnel by definition: approached, then those who consented, then those enrolled after
    screening, then those whose data were acquired. Each is a subset of the one before, so
    the sequence cannot increase.

    Warnings, not errors. A paper may omit a category, or report a count this pipeline maps
    to the wrong rung of the funnel, and both are worth a reviewer's eye rather than a
    rejected record.
    """
    from pondie.formats import values as value_tools

    def whole(node: Any) -> int | None:
        read = value_tools.read(node)
        if isinstance(read, bool) or not isinstance(read, (int, float)):
            return None
        return int(read) if float(read).is_integer() else None

    #: In the order the schema defines them, each a subset of the one before.
    funnel = ("approached_count", "consented_count", "enrolled_count", "acquired_count")

    for index, group in enumerate(record.get("groups") or []):
        if not isinstance(group, Mapping):
            continue

        for slot in ("sex_distribution", "race_distribution"):
            entries = [e for e in (group.get(slot) or []) if isinstance(e, Mapping)]
            if not entries:
                continue
            counted = [whole(e.get("count")) for e in entries]
            declared = {whole(e.get("denominator")) for e in entries}
            declared.discard(None)
            if any(c is None for c in counted) or len(declared) != 1:
                continue
            total, got = declared.pop(), sum(c for c in counted if c is not None)
            # Only the impossible direction. A paper reporting "16 female (53%)" of 30 has
            # given one category of two, and its counts sum below the base by design --
            # every one of the nine this fired on was that, and none was an error. More
            # participants in the categories than in the base cannot be right either way.
            if got > total:
                findings.warn(
                    f"groups[{index}].{slot}",
                    f"the counts sum to {got}, more than the denominator of {total} the "
                    f"entries give",
                )

        seen = [(slot, whole(group.get(slot))) for slot in funnel]
        stated = [(slot, value) for slot, value in seen if value is not None]
        for (earlier, before), (later, after) in zip(stated, stated[1:]):
            if after > before:
                findings.warn(
                    f"groups[{index}].{later}",
                    f"{later} is {after} but {earlier} is {before}, and each of these is a "
                    f"subset of the one before it",
                )


RULES: tuple[Rule, ...] = (
    Rule("cell_terms", "every cell names a term its analysis's model can reach", check_cell_terms),
    Rule("model_stages", "a stage chain is acyclic and names each column once", check_model_stages),
    Rule("table_purpose", "every table is either encoded or says why not", check_table_purpose),
    Rule("group_instruments", "a group's diagnostic instrument is an assessment of the study", check_group_instruments),
    Rule("analysis_inference_settings", "a thresholding scheme the study declares", check_analysis_inference_settings),
    Rule("analysis_measures", "an analysis measures something the study declares", check_analysis_measures),
    Rule("acquisition_devices", "an acquisition names a device the study declares", check_acquisition_devices),
    Rule("crossings", "an interaction the prose names is encoded as one", check_crossings),
    Rule("product_columns", "a product column's factors are both in the model", check_product_columns),
    Rule("unsigned_cells", "a contrast the name signs has cells that carry the sign", check_unsigned_cells),
    Rule("occasion_factors", "a within-subject occasion is a factor, not two analyses", check_occasion_factors),
    Rule("arm_reachability", "a trial's arms are reachable from its analyses", check_arm_reachability),
    Rule("derived_columns", "a derived column says what it was derived from", check_derived_columns),
    Rule("one_protocol_per_acquisition", "an acquisition describes one sequence, not two", check_one_protocol_per_acquisition),
    Rule("value_source_honesty", "a value said to be reported has a sentence", check_value_source_honesty),
    Rule("modality_measures", "a measure the acquisition could have produced", check_modality_measures),
    Rule("counts_add_up", "a breakdown sums to the group it breaks down", check_counts_add_up),
)


def check_all(record: Mapping[str, Any], findings: Findings) -> None:
    """Every rule, over one record."""
    for rule in RULES:
        rule.fn(record, findings)
