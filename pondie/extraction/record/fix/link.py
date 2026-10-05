"""Making the references resolve.

The third kind of fix, and the one with the most judgement in it, so it is also where the
most is refused. A record addresses its own entities by `local_id`, and a reference that
resolves nowhere costs an analysis its inference settings or a level its cohort -- which
downstream reads as a missing field rather than a naming slip.

The rule throughout is that a fix decides nothing. `align_cell_levels` rewrites a level that
folds to exactly one declared level and `check_cell_terms` reports one that does not;
`repair_references` repoints a transcription slip and refuses a guess. Where two answers are
possible the record keeps its defect and a human is told.

Each of these is a `Repair` in `fix.build_sequence`, which holds the order and the reason
each one sits where it does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from pondie.extraction.record import ids
from pondie.extraction.record.direction import OPPOSITE as _OPPOSITE
from pondie.extraction.record import spans as span_tools
from pondie.extraction.record import walk
from pondie.extraction.record.effect import levels_a_cell_may_name, terms_in_scope
from pondie.formats import values
from pondie.schema.reader import Schema
from pondie.vocabularies import abbreviations
from typing import Any
import json
import re


#: The slots an entity's identity can be read off. `description` is not among them:
#: matching prose against an entity name is a substring operation, and
#: `normalize_open_fields.py` measured what containment does -- it merged `emotion
#: regulation`, the corpus's most frequent task term, into a rarer variant, with 38
#: candidate hosts.
NAMING_SLOTS = ("name", "level", "label")


#: Words a `Cell.level` carries when it is really stating which way the effect went, by
#: polarity. `Cell.direction` is the slot for that, and on a continuous term there is no
#: level for it to be.
_LEVEL_POLARITY: dict[str, str] = {
    **{w: "positive" for w in (
        "positive", "higher", "greater", "more", "increase", "increased", "up", "activation",
        "positive correlation", "positively correlated",
    )},
    **{w: "negative" for w in (
        "negative", "lower", "less", "fewer", "decrease", "decreased", "down", "deactivation",
        "negative correlation", "negatively correlated",
    )},
}


#: The minted part of a local_id, once its class prefix is off. `ids.mint` builds it from
#: the paper's own wording, which is what lets a dangling id be compared to a real name.
_PREFIXES = tuple(prefix for prefix in ids.PREFIX.values() if prefix)


#: Words that name no entity, so sharing one is not agreement.
_EMPTY_WORDS = frozenset(
    {"the", "of", "a", "and", "main", "effect", "condition", "conditions", "group",
     "groups", "task", "tasks", "all", "1", "2", "3"}
)


def _expansion(defined: Mapping[str, str], short: str) -> str:
    """What the paper defines `short` as, allowing its plural: 26347628 defined `HCs`
    and declared the level `HC`."""
    for form in (short, f"{short}s", short.removesuffix("s")):
        if form in defined:
            return defined[form]
    return ""


def align_cell_levels(body: dict[str, Any], text: str = "") -> list[str]:
    """Rewrite a `Cell.level` to the declared `FactorLevel.level` it folds to.

    The join is on the string (extraction-readme.md §3 invariant 3), so `Healthy controls`
    against a declared `healthy controls` is a broken join that no reader would call a
    disagreement. Repaired only where exactly one declared level folds to it: two would make
    the choice a guess, and a guess about which condition was compared is the one thing this
    field must not contain.

    Deliberately narrow. `AD` against a declared `AD group` does *not* fold, and is left for
    `Validator.check_cell_terms` to report -- shortening a level is a claim about the paper,
    not a transcription slip. The one exception is an abbreviation `text` itself defines:
    26347628 wrote `healthy controls (HCs)`, so a cell's `healthy controls` is the
    declared `HC`. Only this paper's definitions are used, never a store's.
    """

    defined = abbreviations.mine(text) if text else {}

    fixed: list[str] = []
    models = {
        model.get("local_id"): model
        for model in body.get("model_estimations") or []
        if isinstance(model, Mapping)
    }

    for index, analysis in enumerate(body.get("analyses") or []):
        if not isinstance(analysis, Mapping):
            continue
        scope = terms_in_scope(analysis.get("model_estimation"), models)
        effect = analysis.get("effect")
        cells = effect.get("cells") if isinstance(effect, Mapping) else None
        for position, cell in enumerate(cells or []):
            if not isinstance(cell, Mapping) or not values.is_field(cell.get("level")):
                continue
            level = cell["level"].get("value")
            term = scope.get(cell.get("term"))
            if not isinstance(level, str) or not isinstance(term, Mapping):
                continue
            declared = [
                values.read(entry.get("level"))
                for entry in (term.get("levels") or [])
                if isinstance(entry, Mapping)
            ]
            declared = [name for name in declared if isinstance(name, str)]
            if level in declared:
                continue
            folded = span_tools.fold_label(level)
            matches = [name for name in declared if span_tools.fold_label(name) == folded]
            if not matches and defined:
                matches = [
                    name
                    for name in declared
                    if span_tools.fold_label(_expansion(defined, name)) == folded
                    or span_tools.fold_label(_expansion(defined, level))
                    == span_tools.fold_label(name)
                ]
            if len(matches) == 1:
                cell["level"]["value"] = matches[0]
                fixed.append(
                    f"analyses[{index}].effect.cells[{position}].level: "
                    f"{level!r} -> {matches[0]!r}"
                )
    return fixed


def complete_condition_levels(body: dict[str, Any]) -> list[str]:
    """Declare a condition factor's level that a cell names and the term left out.

    24695721 cells cocaine, sexual and aversive cues against neutral cues on one term, and
    the term declares only the first three. The record declares `cond_neutral` ("neutral
    cues"), so the missing level is that condition's. Only on a term whose every level
    carries a condition, for a cell level folding to exactly one condition no other level of
    the term holds: "cocaine vs neutral" names no condition and is still reported.
    """
    conditions: dict[str, list[str]] = {}
    for task in body.get("tasks") or []:
        for condition in (task.get("conditions") or []) if isinstance(task, Mapping) else []:
            if isinstance(condition, Mapping) and isinstance(condition.get("local_id"), str):
                folded = span_tools.fold_label(str(values.read(condition.get("name")) or ""))
                if folded:
                    conditions.setdefault(folded, []).append(condition["local_id"])
    models = {
        m.get("local_id"): m for m in body.get("model_estimations") or [] if isinstance(m, Mapping)
    }
    fixed: list[str] = []
    for analysis in body.get("analyses") or []:
        if not isinstance(analysis, Mapping):
            continue
        scope = terms_in_scope(analysis.get("model_estimation"), models)
        for cell in (analysis.get("effect") or {}).get("cells") or []:
            if not isinstance(cell, Mapping):
                continue
            term, level = scope.get(cell.get("term")), values.read(cell.get("level"))
            if not isinstance(term, dict) or not isinstance(level, str):
                continue
            levels = [lv for lv in term.get("levels") or [] if isinstance(lv, Mapping)]
            if not levels or not all(walk.ids_of(lv.get("conditions")) for lv in levels):
                continue
            names = {span_tools.fold_label(str(values.read(lv.get("level")) or "")) for lv in levels}
            folded = span_tools.fold_label(level)
            match = conditions.get(folded, [])
            held = {c for lv in levels for c in walk.ids_of(lv.get("conditions"))}
            if folded in names or len(match) != 1 or match[0] in held:
                continue
            term["levels"].append({
                "level": values.wrap(level, source="generated", evidence="not_found"),
                "conditions": [match[0]],
            })
            fixed.append(f"{term.get('local_id')}: declared level {level!r} ({match[0]}), "
                         f"which a cell names")
    return fixed


def link_entities_by_name(body: dict[str, Any], sch: Schema) -> list[str]:
    """Write a reference where a name settles which entity it means.

    1,713 of 6,860 `FactorLevel`s are on a categorical term and reach no entity at all -- a
    bare string, and `check_cell_terms` says why that costs: "the mapper joins these on the
    string". 715 of those fold to the exact name of an entity the same record already
    declares, `local_id` and all: `level 'smoking cue'` beside `cond_smoking_cue`,
    `level 'predose'` beside `tp_predose`, `level 'Exercise'` beside `arm_exercise`. Both
    halves of the join are in the record and nothing wrote it down.

    THE SCHEMA DECIDES ALL OF IT, which is the only reason this is safe to run over every
    reference slot rather than a list:

      which entities may connect   `attribute.range`, through `Schema.resolves_to` so a
                                   subclass satisfies a supertype -- a slot declaring
                                   `Acquisition` is satisfied by an `MRI`.

      which references a name may   a reference whose range is the owner's OWN kind is a
      settle                        structural relation, not an identity. Exactly three
                                    slots in the schema are -- `Analysis.mirror_of`,
                                    `ModelEstimation.inputs_from`,
                                    `ModelTerm.interaction_with` -- and they mean
                                    *sign-reversed twin of*, *fitted on the output of* and
                                    *crossed with*. A name says what a thing is and nothing
                                    about how two things relate. Matched on a name these
                                    three propose 9,151 self-links, and after excluding
                                    self, `interaction_with` still proposes 708 of which
                                    97% are a term named `group` in one model matching a
                                    term named `group` in another -- links that would
                                    fabricate interactions `check_crossings` then reports.

      what shape to write           `attribute.multivalued`: a bare id string or a bare
                                    list of them, never an `ExtractedValue`. A reference is
                                    an address inside the record, not a claim about the
                                    paper, which is why `rules.IDENTIFIERS` exempts these
                                    slots from the evidence checks.

    Exact fold match to exactly one candidate, and never to the entity itself. A further 966
    links are reachable on a single substring match and are not taken, for `NAMING_SLOTS`'
    reason. A name matching two candidates of the same kind is left alone; matching two
    *kinds* is written to both, which is the parallel-group case -- 112 of 122 are a level
    naming both the arm and the cohort allocated to it, and `Group.arm` exists to say so.
    """

    index: dict[str, list[tuple[str, str]]] = {}

    def collect(node: Any, class_name: str) -> None:
        if not isinstance(node, dict):
            return
        class_name = sch.designated_type(node, class_name)
        attributes = sch.attributes(class_name)
        if not attributes:
            return
        local_id = node.get("local_id")
        if isinstance(local_id, str) and local_id:
            for slot in NAMING_SLOTS:
                name = _fold_name(values.read(node.get(slot)))
                if name:
                    index.setdefault(name, []).append((class_name, local_id))
        for key, attribute in attributes.items():
            if key not in node or sch.classify(key, attribute) != "nested":
                continue
            if not isinstance(attribute.range, str):
                continue
            child = node[key]
            for item in child if isinstance(child, list) else [child]:
                collect(item, attribute.range)

    def is_relation(owner: str, target: str) -> bool:
        return owner == target or sch.resolves_to(owner, target) or sch.resolves_to(target, owner)

    fixed: list[str] = []

    def visit(node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict) or values.is_field(node):
            return
        class_name = sch.designated_type(node, class_name)
        mine = node.get("local_id") if isinstance(node.get("local_id"), str) else None
        names = [n for n in (_fold_name(values.read(node.get(s))) for s in NAMING_SLOTS) if n]
        for key, attribute in sch.attributes(class_name).items():
            kind = sch.classify(key, attribute)
            if kind == "nested" and isinstance(attribute.range, str) and key in node:
                child = node[key]
                for order, item in enumerate(child if isinstance(child, list) else [child]):
                    suffix = f"[{order}]" if isinstance(child, list) else ""
                    visit(item, attribute.range, f"{path}.{key}{suffix}")
                continue
            if kind != "reference" or not isinstance(attribute.range, str):
                continue
            target = attribute.range
            if not names or is_relation(class_name, target):
                continue
            current = node.get(key)
            if [x for x in (current if isinstance(current, list) else [current]) if x]:
                continue
            found = {
                local_id
                for name in names
                for owner, local_id in index.get(name, ())
                if local_id != mine and (owner == target or sch.resolves_to(owner, target))
            }
            if len(found) != 1:
                continue
            settled = found.pop()
            node[key] = [settled] if attribute.multivalued else settled
            fixed.append(f"{path}.{key}: {settled!r} from its name")

    collect(body, "Study")
    visit(body, "Study", "Study")
    return fixed


def _fold_name(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _restates(one: Any, other: Any) -> bool:
    """Whether either name says no more than the other: `BMI` against term `BMI`."""
    left, right = _fold_name(one), _fold_name(other)
    return bool(left and right) and (left in right or right in left)


#: Words that name no particular group, and the spellings of "not" a group name uses.
_GROUP_FILLER = frozenset({"group", "groups", "subjects", "participants", "sample", "the"})
_NEGATIONS = {"non": "non", "no": "non", "without": "non", "not": "non"}


def _restates_the_only_group(
    level: str, analysis: Mapping[str, Any], groups: Mapping[str, str]
) -> bool:
    """Whether `level` names the one group the analysis ran in: `PTSD group` beside
    `recent onset PTSD`. A negation on one side only (`PTSD` beside `non-PTSD`) is not."""
    named = [
        entry.get("group") if isinstance(entry, Mapping) else entry
        for entry in analysis.get("groups") or []
    ]
    if len(named) != 1 or named[0] not in groups:
        return False

    def words(text: str) -> set[str]:
        found = re.findall(r"[a-z0-9]+", text.lower())
        return {_NEGATIONS.get(w, w) for w in found} - _GROUP_FILLER

    said, group = words(level), words(groups[named[0]])
    return bool(said) and said <= group and ("non" in said) == ("non" in group)


def drop_redundant_cell_levels(body: dict[str, Any]) -> list[str]:
    """Drop a `Cell.level` that a continuous term cannot have and that says nothing new.

    A continuous term declares no levels, so `check_cell_terms` errors on any cell naming
    one: 1,204 errors over 527 papers, the largest error class in the corpus, and 1,185 of
    the 1,205 are on a term typed continuous. Four shapes carry no information and are
    removed here.

      restates the term  547 (46%). `BMI` on term `BMI`, `age` on `age`, `pack-years` on
                         `pack-years`. A regressor's cell has no level, and naming it after
                         the term says nothing a reader did not already have. So is its
                         type: `continuous` on a continuous term (21592738, seven cells).

      duplicates the     185 (16%). `positive` where `direction` already says positive.
      direction          `Cell.direction` is where the sign lives and it is already right.

      restates the       `PTSD group` on a CAPS correlation run within the PTSD group only.
      only group         `Analysis.groups` already says whose scores these are.

      on an unsigned     `PTSD` on a `group x BAI` F-test. The test has no side for a level
      product column     to name.

    One shape is left alone: 424 cells name a genuinely categorical level on a term
    whose `type` is wrong. Fixing that means flipping the type *and* synthesising the levels
    the term should have declared, which is a claim about the model rather than a tidy-up, so
    `check_cell_terms` keeps reporting it.

    So is the case where the two disagree. 26 cells read `direction='negative'` with
    `level='positive'`, and dropping the level there would resolve a contradiction about the
    sign of an effect by picking one side silently -- and the sign is what decides whether a
    coordinate enters an increase map or a decrease map. `check_cell_level_polarity` reports
    those instead.
    """

    models = {
        model["local_id"]: model
        for model in (body.get("model_estimations") or [])
        if isinstance(model, Mapping) and isinstance(model.get("local_id"), str)
    }
    groups = {
        group["local_id"]: str(values.read(group.get("name")) or "")
        for group in (body.get("groups") or [])
        if isinstance(group, Mapping) and isinstance(group.get("local_id"), str)
    }
    fixed: list[str] = []
    for index, analysis in enumerate(body.get("analyses") or []):
        if not isinstance(analysis, dict):
            continue
        terms = terms_in_scope(analysis.get("model_estimation"), models)
        effect = analysis.get("effect")
        if not isinstance(effect, dict):
            continue
        for position, cell in enumerate(effect.get("cells") or []):
            if not isinstance(cell, dict):
                continue
            term_id = cell.get("term")
            level = values.read(cell.get("level"))
            if not isinstance(term_id, str) or not isinstance(level, str) or not level.strip():
                continue
            term = terms.get(term_id)
            if term is None or values.read(term.get("type")) != "continuous":
                continue
            declared = [
                name
                for name in (
                    values.read(entry.get("level"))
                    for entry in (term.get("levels") or [])
                    if isinstance(entry, Mapping)
                )
                if isinstance(name, str)
            ]
            if declared:
                continue
            path = f"analyses[{index}].effect.cells[{position}].level"
            if level in levels_a_cell_may_name(term, terms, cell.get("direction")):
                continue  # a signed product cell's component level: whose slope it is
            if _restates(level, values.read(term.get("name"))):
                cell.pop("level", None)
                fixed.append(f"{path}: {level!r} restated term {term_id!r} -- dropped")
                continue
            if _fold_name(level) == _fold_name(values.read(term.get("type"))):
                cell.pop("level", None)
                fixed.append(f"{path}: {level!r} restated the term's type -- dropped")
                continue
            if _restates_the_only_group(level, analysis, groups):
                cell.pop("level", None)
                fixed.append(f"{path}: {level!r} restated the analysis's only group -- dropped")
                continue
            if term.get("interaction_with") and values.read(cell.get("direction")) not in (
                "positive",
                "negative",
            ):
                cell.pop("level", None)
                fixed.append(f"{path}: {level!r} on an unsigned product column -- dropped")
                continue
            polarity = _LEVEL_POLARITY.get(str(level).strip().lower())
            if polarity is None:
                continue
            # A sign word that another term in this model declares as a LEVEL is a level.
            # 29935441 puts `level: 'negative'` on the product column
            # `PCL-by-emotion-by-task`, and `negative` is one of `trm_emotion`'s three
            # declared levels alongside `positive` and `neutral` -- an emotion, not a sign.
            # Read as a direction it became `direction: negative` and turned a factor level
            # into a polarity, which `check_crossings` then reported as a signed cell on a
            # product column. The level belongs to a term the cell does not name, which is
            # `check_cell_terms`' finding and not something to resolve here.
            if any(
                _fold_name(level) == _fold_name(values.read(entry.get("level")))
                for other in terms.values()
                for entry in (other.get("levels") or [])
                if isinstance(entry, Mapping)
            ):
                continue
            held_raw = values.read(cell.get("direction"))
            held = _LEVEL_POLARITY.get(str(held_raw or "").strip().lower())
            if held == polarity:
                cell.pop("level", None)
                fixed.append(f"{path}: {level!r} duplicated direction -- dropped")
            elif held is None and (held_raw is None or not str(held_raw).strip()):
                # Only where `direction` is genuinely absent. `undirected` is an answer, not
                # a gap, and overwriting it would replace a stated fact with an inference.
                cell["direction"] = values.wrap(
                    polarity, source="generated", evidence="not_found"
                )
                cell.pop("level", None)
                fixed.append(f"{path}: {level!r} moved to direction")

    return fixed


def infer_missing_models(body: dict[str, Any]) -> list[str]:
    """Point an analysis with no `model_estimation` at the one model declaring its terms.

    The analysis's cells name terms, and a term lives in one model, so where exactly one
    model declares every term the cells name, that is the model the analysis estimated
    (17892884's `ana_t2_hippocampus_group`, whose cells name `mod_repeated_hippocampus`'s
    group term). Two candidates, or none, leave it reported.
    """
    declared_by = _declared_by(_models(body))
    fixed: list[str] = []
    for index, analysis in enumerate(body.get("analyses") or []):
        if not isinstance(analysis, dict) or analysis.get("model_estimation"):
            continue
        named = {
            cell.get("term")
            for cell in (analysis.get("effect") or {}).get("cells") or []
            if isinstance(cell, Mapping)
        }
        if not named or not all(isinstance(t, str) for t in named):
            continue
        models = set.intersection(*(declared_by.get(t, set()) for t in named))
        if len(models) == 1:
            analysis["model_estimation"] = models.pop()
            fixed.append(
                f"analyses[{index}].model_estimation: {analysis['model_estimation']!r}, "
                "the one model declaring every term its cells name"
            )
    return fixed


def fill_empty_models(body: dict[str, Any], sch: Schema) -> list[str]:
    """Give a model with no terms the terms its analyses' cells borrow from one other model.

    An extractor declaring one design several times writes it out once: 30343133 tests
    group x time at four seeds, declares the terms under the vmPFC seed's model, leaves the
    other three models empty, and points their analyses' cells at the vmPFC terms. Each
    cell then names a term its own model does not reach. The cells already say which terms
    these analyses tested, so the terms are copied into the empty model, scoped to it as
    `<model>.<term>` (`scope_duplicate_terms`' form), and the cells repointed.

    What is copied is the terms these analyses cell, the donor's terms no analysis cells
    (the shared covariates), and what those are products of. A term another analysis tests is
    not: 17923164 declared the IES total, intrusion and avoidance scores under one model and
    celled one per analysis, so the intrusion model takes the intrusion score and the
    covariates, not the scores tested beside it.

    The donor is a model declaring every borrowed term. 21338692's correlation model borrowed
    `age of first use`, declared by two models, and `years of use`, declared by one of them:
    that one is the donor. Several donors are fine when they would give the same design --
    16701903 declared one `group` factor in two models -- and none, or donors that disagree,
    leave the model for the report.
    """
    models = _models(body)
    declared_by = _declared_by(models)
    celled_by: dict[str, set[str]] = {}
    for analysis in body.get("analyses") or []:
        if isinstance(analysis, Mapping):
            for term_id in _celled([analysis]) if isinstance(analysis, dict) else ():
                celled_by.setdefault(term_id, set()).add(str(analysis.get("model_estimation")))
    fixed: list[str] = []
    for model_id, model in models.items():
        if model.get("terms"):
            continue
        analyses = _analyses_on(body, model_id)
        named = _celled(analyses)
        if not named or set(terms_in_scope(model_id, models)) & named:
            continue
        donor = _agreed({
            d: _borrowed_terms(models[d], named, celled_by)
            for d in set.intersection(*(declared_by.get(t, set()) for t in named))
        })
        if donor is None:
            continue
        scoped = _copy_terms(model_id, model, donor,
                             _borrowed_terms(models[donor], named, celled_by), analyses, sch)
        fixed.append(
            f"model_estimations[{model_id!r}]: had no terms; copied {len(scoped)} from "
            f"{donor!r}, which its {len(analyses)} analysis(es)' cells named"
        )
    return fixed


def complete_partial_models(body: dict[str, Any], sch: Schema) -> list[str]:
    """Give a model the terms its analyses' cells name and it lacks, from the model that
    declares them.

    `fill_empty_models` for a model that declares some terms but not all its analyses test.
    17825801 fits one diagnosis x combat-exposure design four ways, unadjusted and adjusted
    for age, combat severity or others. `mod_age` declares diagnosis and age, its analysis
    "Diagnosis x Combat Exposure Interaction -- Age" cells `trm_exposure`, and only
    `mod_unadjusted` declares it. The cell is the evidence the model had the term.

    Only the missing terms that are celled, and what they are products of, are copied: the
    donor's covariates are its own adjustment, not this model's. Copies are scoped
    `<model>.<term>` and the cells repointed. Several donors are fine when they agree on the
    copied terms; donors that disagree leave the model for the report.
    """
    models = _models(body)
    declared_by = _declared_by(models)
    fixed: list[str] = []
    for model_id, model in models.items():
        if not model.get("terms"):
            continue  # `fill_empty_models`' case
        analyses = _analyses_on(body, model_id)
        scope = set(terms_in_scope(model_id, models))
        missing = _celled(analyses) - scope
        if not missing or not all(declared_by.get(t) for t in missing):
            continue
        candidates = {
            d: _missing_terms(models[d], missing, scope)
            for d in set.intersection(*(declared_by[t] for t in missing)) - {model_id}
        }
        donor = _agreed(candidates)
        if donor is None:
            continue
        scoped = _copy_terms(model_id, model, donor, candidates[donor], analyses, sch)
        fixed.append(
            f"model_estimations[{model_id!r}]: lacked {sorted(missing)} its analyses cell; "
            f"copied {len(scoped)} term(s) from {donor!r}"
        )
    return fixed


def _missing_terms(
    donor: Mapping[str, Any], missing: set[str], scope: set[str]
) -> list[Mapping[str, Any]]:
    """The donor's terms a partial model lacks and cells, and what those are products of,
    in the donor's order."""
    by_id = {
        t["local_id"]: t
        for t in donor.get("terms") or []
        if isinstance(t, Mapping) and isinstance(t.get("local_id"), str)
    }
    wanted = set(missing)
    for term_id in list(missing):
        wanted |= {
            c for c in by_id[term_id].get("interaction_with") or []
            if isinstance(c, str) and c in by_id and c not in scope
        }
    return [term for term_id, term in by_id.items() if term_id in wanted]


def _models(body: dict[str, Any]) -> dict[str, dict]:
    return {
        m["local_id"]: m
        for m in body.get("model_estimations") or []
        if isinstance(m, dict) and isinstance(m.get("local_id"), str)
    }


def _declared_by(models: Mapping[str, Mapping[str, Any]]) -> dict[str, set[str]]:
    """term local_id -> the models declaring it."""
    declared_by: dict[str, set[str]] = {}
    for model_id, model in models.items():
        for term in model.get("terms") or []:
            if isinstance(term, Mapping) and isinstance(term.get("local_id"), str):
                declared_by.setdefault(term["local_id"], set()).add(model_id)
    return declared_by


def _analyses_on(body: dict[str, Any], model_id: str) -> list[dict]:
    return [
        a
        for a in body.get("analyses") or []
        if isinstance(a, dict) and a.get("model_estimation") == model_id
    ]


def _celled(analyses: list[dict]) -> set[str]:
    """The term ids these analyses' cells name."""
    return {
        cell["term"]
        for a in analyses
        for cell in (a.get("effect") or {}).get("cells") or []
        if isinstance(cell, Mapping) and isinstance(cell.get("term"), str)
    }


def _agreed(candidates: Mapping[str, list[Mapping[str, Any]]]) -> str | None:
    """The donor to copy from, when every candidate would give the same terms (`_design`);
    None for no candidate, or candidates that disagree."""
    if len({tuple(map(_design, terms)) for terms in candidates.values()}) != 1:
        return None
    return min(candidates)


def _copy_terms(
    model_id: str, model: dict, donor: str, terms: list[Mapping[str, Any]],
    analyses: list[dict], sch: Schema,
) -> dict[str, str]:
    """Copy `terms` from `donor` into `model`, scoped `<model>.<term>` -- a donor's own
    scope prefix dropped, so a copy of a copy is not scoped twice -- and repoint the
    analyses' cells. Returns the renames."""
    scoped = {
        t["local_id"]: f"{model_id}.{t['local_id'].removeprefix(f'{donor}.')}" for t in terms
    }
    model.setdefault("terms", [])
    for term in json.loads(json.dumps(terms)):
        term["local_id"] = scoped[term["local_id"]]
        walk.repoint(term, sch, scoped, root="ModelTerm", target="ModelTerm")
        model["terms"].append(term)
    for analysis in analyses:
        walk.repoint(analysis, sch, scoped, root="Analysis", target="ModelTerm")
    return scoped


def _borrowed_terms(
    donor: Mapping[str, Any], named: set[Any], celled_by: Mapping[str, set[str]]
) -> list[Mapping[str, Any]]:
    """The donor's terms an empty model takes: those its analyses cell, the ones no analysis
    cells, and the components of either."""
    by_id = {t["local_id"]: t for t in donor.get("terms") or [] if isinstance(t, Mapping)}
    wanted = {t for t in by_id if t in named or not celled_by.get(t)}
    for term_id in list(wanted):
        components = by_id[term_id].get("interaction_with") or []
        wanted |= {c for c in components if isinstance(c, str) and c in by_id}
    return [term for term_id, term in by_id.items() if term_id in wanted]


def _design(term: Mapping[str, Any]) -> str:
    """What a term says about the model, without its evidence."""
    levels = [
        values.read(level.get("level")) if isinstance(level, Mapping) else level
        for level in term.get("levels") or []
    ]
    return json.dumps(
        [term.get("local_id"), values.read(term.get("name")), values.read(term.get("type")),
         levels, term.get("interaction_with") or []],
        sort_keys=True,
        default=str,
    )


def cross_products_of_factors(body: dict[str, Any]) -> list[str]:
    """Rewrite a signed cell on a product of two factors as the factors' crossed cells.

    Two factors cross in their own cells (`representing-models.md` §5.10), so a product
    column of two factors decides nothing and its cell may name no level. 30343133 wrote its
    group x time interactions that way anyway: `{term: group-by-time, level: PTSD,
    direction: positive}`, "PTSD's change over time was positive relative to TD's". That is
    `PTSD +, TD -, follow-up +, baseline -`.

    Only where the reading is certain: both factors have two levels, the cell names a level
    of one, the other's levels carry distinct `order` (later is positive), and no other cell
    of the analysis names either factor. Without an order nothing says which level of the
    other factor is the positive side (26535944's sex x group), and the cell stays reported.

    A second shape is the factor's own cells filed on the product: 28287194 wrote
    `Patients -, Healthy controls +` on group-by-time beside `T2: held` -- "patients below
    controls at T2". Signed cells naming both levels of one factor, opposite in sign, are that
    factor's crossing; they move to its term.
    """

    models = {
        model["local_id"]: model
        for model in (body.get("model_estimations") or [])
        if isinstance(model, Mapping) and isinstance(model.get("local_id"), str)
    }
    fixed: list[str] = []
    for index, analysis in enumerate(body.get("analyses") or []):
        effect = analysis.get("effect") if isinstance(analysis, dict) else None
        if not isinstance(effect, dict) or not isinstance(effect.get("cells"), list):
            continue
        terms = terms_in_scope(analysis.get("model_estimation"), models)
        fixed += [f"analyses[{index}].{note}" for note in _repoint_to_factor(effect["cells"], terms)]
        for position, cell in enumerate(effect["cells"]):
            crossed = _crossed_cells(cell, terms, effect["cells"])
            if crossed is None:
                continue
            effect["cells"][position : position + 1] = crossed
            fixed.append(
                f"analyses[{index}].effect.cells[{position}]: product of factors "
                f"{cell['term']!r} rewritten as crossed cells"
            )
            break  # the list changed under `position`; one product per analysis
    return fixed


def _repoint_to_factor(cells: list[Any], terms: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Move to the factor the cells on a product of factors that name both its levels."""
    notes: list[str] = []
    on_product: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        term = terms.get(cell.get("term")) if isinstance(cell, dict) else None
        if term and term.get("interaction_with") and values.read(cell.get("direction")) in _OPPOSITE:
            on_product.setdefault(cell["term"], []).append(cell)
    for product_id, signed in on_product.items():
        components = [terms.get(c) for c in terms[product_id]["interaction_with"]]
        if not components or not all(isinstance(c, Mapping) and _levels_of(c) for c in components):
            continue  # a product with a continuous term is a moderation, not a crossing
        named = {values.read(c.get("level")) for c in signed}
        owners = [c for c in components if named <= {name for name, _ in _levels_of(c)}]
        signs = {values.read(c.get("direction")) for c in signed}
        if len(owners) != 1 or len(named) < 2 or signs != set(_OPPOSITE):
            continue
        factor = owners[0]["local_id"]
        if any(isinstance(c, Mapping) and c.get("term") == factor for c in cells):
            continue
        for cell in signed:
            cell["term"] = factor
        notes.append(f"cells on product {product_id!r} name both levels of {factor!r}: moved to it")
    return notes


def _crossed_cells(
    cell: Any, terms: Mapping[str, Mapping[str, Any]], cells: list[Any]
) -> list[dict[str, Any]] | None:
    """The four cells `cross_products_of_factors` writes for `cell`, or None to leave it."""
    if not isinstance(cell, Mapping):
        return None
    term = terms.get(cell.get("term"))
    sign = values.read(cell.get("direction"))
    named = values.read(cell.get("level"))
    components = [terms.get(c) for c in (term or {}).get("interaction_with") or []]
    if sign not in _OPPOSITE or not isinstance(named, str) or len(components) != 2:
        return None
    levels = [_levels_of(c) if isinstance(c, Mapping) else [] for c in components]
    if any(len(pair) != 2 for pair in levels):
        return None
    sides = [i for i in (0, 1) if named in [name for name, _ in levels[i]]]
    if len(sides) != 1:
        return None
    side, other = sides[0], 1 - sides[0]
    orders = [order for _, order in levels[other]]
    if not all(isinstance(o, int) for o in orders) or orders[0] == orders[1]:
        return None
    factors = {components[side]["local_id"], components[other]["local_id"]}
    if any(isinstance(c, Mapping) and c.get("term") in factors for c in cells):
        return None

    def made(term_id: str, level: str, direction: str) -> dict[str, Any]:
        return {
            "term": term_id,
            "level": values.wrap(level, source="generated", evidence="not_found"),
            "direction": values.wrap(direction, source="generated", evidence="not_found"),
        }

    unnamed = next(name for name, _ in levels[side] if name != named)
    earlier, later = sorted(levels[other], key=lambda pair: pair[1])
    kept = {"term": components[side]["local_id"], "level": cell["level"],
            "direction": cell["direction"]}
    if cell.get("label") is not None:
        kept["label"] = cell["label"]
    return [
        kept,
        made(components[side]["local_id"], unnamed, _OPPOSITE[sign]),
        made(components[other]["local_id"], later[0], "positive"),
        made(components[other]["local_id"], earlier[0], "negative"),
    ]




def _levels_of(term: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """A factor's declared levels, each with its `order` (None where unordered)."""
    return [
        (values.read(entry.get("level")), values.read(entry.get("order")))
        for entry in term.get("levels") or []
        if isinstance(entry, Mapping) and isinstance(values.read(entry.get("level")), str)
    ]


def scope_duplicate_terms(body: dict[str, Any], sch: Schema) -> list[str]:
    """Make two models' identically-named terms distinguishable, by their model.

    A term's id is only meaningful inside the model that declares it, but the record's id
    namespace is flat, so two estimations that both control for `term_age` collide and
    every reference naming it becomes ambiguous. `check_local_ids` reports that and is
    right to -- nothing downstream can tell which one a cell meant.

    Nothing is being picked. A cell sits in an analysis, the analysis names its model
    estimation, and a reference from that analysis can only have meant that model's term.
    The scope is information the record already carries; this writes it into the id as
    `<model>.<term>`.

    All or nothing per id. A rename that leaves one reference pointing at the old name
    turns an ambiguous reference into a dangling one, which is worse: ambiguity is
    reported against a term that exists, and a dangling reference is a slot that resolves
    to nothing. Every reference is therefore repointed first. An id with a reference that
    cannot be scoped -- one reached from no analysis, or from an analysis whose model does
    not declare it -- is reverted and left for the report.
    """

    declared_by = _declared_by(_models(body))

    counts: dict[str, int] = {}

    def count(node: Any) -> None:
        if isinstance(node, Mapping):
            if values.is_field(node):
                return
            name = node.get("local_id")
            if isinstance(name, str):
                counts[name] = counts.get(name, 0) + 1
            for value in node.values():
                count(value)
        elif isinstance(node, list):
            for value in node:
                count(value)

    count(body)
    # Only ids whose every declaration is a term under a model. An id shared between a
    # term and a group is a conflict rather than a scope, and renaming it would hide that.
    collisions = {
        name
        for name, total in counts.items()
        if total > 1 and len(declared_by.get(name, ())) == total
    }
    if not collisions:
        return []

    scoped: list[str] = []
    for name in sorted(collisions):
        owners = declared_by[name]
        # Read from `body` every time: a revert below replaces its contents, and renaming
        # terms in the models list read before it renamed copies no longer in the record.
        models = [m for m in body.get("model_estimations") or [] if isinstance(m, Mapping)]
        before = walk.dangling(body, sch)
        # Use each analysis's model to determine which copy it could mean.
        reachable = {
            analysis_index: values.read(analysis.get("model_estimation"))
            for analysis_index, analysis in enumerate(body.get("analyses") or [])
            if isinstance(analysis, Mapping)
        }
        snapshot = json.loads(json.dumps(body))

        for model in models:
            model_id = model.get("local_id")
            if model_id not in owners:
                continue
            mapping = {name: f"{model_id}.{name}"}
            for term in model.get("terms") or []:
                if isinstance(term, dict):
                    if term.get("local_id") == name:
                        term["local_id"] = mapping[name]
                    walk.repoint(term, sch, mapping, root="ModelTerm", target="ModelTerm")
            for index, analysis in enumerate(body.get("analyses") or []):
                if reachable.get(index) == model_id and isinstance(analysis, dict):
                    walk.repoint(analysis, sch, mapping, root="Analysis", target="ModelTerm")

        # A reference still naming the bare id is one this could not scope, and any other
        # new dangling reference is a rename gone wrong. Put the record back either way.
        after = walk.dangling(body, sch)
        if after[name] or after - before:
            body.clear()
            body.update(snapshot)
            continue
        scoped.append(f"{name!r} declared by {len(owners)} models -> scoped per model")
    return scoped


def repoint_out_of_scope_terms(body: dict[str, Any]) -> list[str]:
    """Repoint a cell at the same-named term its analysis's model can actually reach.

    A cell must name a term in its analysis's model scope -- that model's own terms plus
    those of the models it reaches through `inputs_from`. When it names a term from
    somewhere else the cell is a sign of nothing, and the validator says so.

    Repaired only where the choice is forced: exactly one term in scope carries the same
    name as the one the cell named. That is the same rule `align_cell_levels` follows for
    levels, and it covers 24 of the 105 out-of-scope references measured over 30 records.
    The other 81 are left reported -- 73 have no same-named term in scope at all, which
    means something larger is wrong than a mistyped identifier.

    A second, narrower case: the cell names an id nothing declares, and exactly one term in
    scope declares that id under a model prefix -- `trm_modality` against
    `mod_mass_univariate.trm_modality`. That is not a mistyped identifier but a disagreement
    between two passes. `demands` declares a term `trm_modality` and writes cells naming it;
    `satisfy` re-declares it once per model estimation that needs it, prefixing each with the
    model to keep them apart, and the cells written by the earlier pass are left pointing at
    an id no longer present. 21 of 183 cells over the fifteen benchmark papers, against 0 of
    186 in the reviewer's reference.

    It is worth repairing because the loss is silent and large. Nothing points at the
    surviving term, so it has no incoming edges; the benchmark's aligner reads that as
    structural disagreement and scores the pair 0.421 against a 0.45 threshold even though
    the names agree and the parent model matches. The term goes unaligned, every cell on it
    is dropped from the polarity score, and one paper loses 19 cells that way.

    Scoped by the analysis, which is what makes it unambiguous: `trm_modality` may be
    declared under two models, but only one of those is in this analysis's scope. Measured
    over the same fifteen papers, all 21 resolve to exactly one term and none to several.
    """

    models = {
        str(values.read(m.get("local_id"))): m
        for m in body.get("model_estimations") or []
        if isinstance(m, Mapping)
    }

    everywhere = {
        str(values.read(t.get("local_id"))): t
        for m in models.values()
        for t in (m.get("terms") or [])
        if isinstance(t, Mapping) and values.read(t.get("local_id"))
    }

    def name_of(term: Any) -> str:
        return str(values.read((term or {}).get("name")) or "").strip().lower()

    fixed: list[str] = []
    for index, analysis in enumerate(body.get("analyses") or []):
        if not isinstance(analysis, Mapping):
            continue
        in_scope = terms_in_scope(str(values.read(analysis.get("model_estimation")) or ""), models)
        for position, cell in enumerate((analysis.get("effect") or {}).get("cells") or []):
            if not isinstance(cell, Mapping):
                continue
            named = values.read(cell.get("term"))
            if not isinstance(named, str) or named in in_scope:
                continue
            if named not in everywhere:
                # Declared nowhere: the id may be the unprefixed half of a term `satisfy`
                # re-declared under its model.
                under = [k for k in in_scope if k.endswith(f".{named}")]
                if len(under) == 1:
                    cell["term"] = under[0]
                    fixed.append(
                        f"analyses[{index}].effect.cells[{position}].term: "
                        f"{named!r} -> {under[0]!r} (declared under its model)"
                    )
                continue
            wanted = name_of(everywhere.get(named))
            if not wanted:
                continue
            same = [k for k, term in in_scope.items() if name_of(term) == wanted]
            if len(same) == 1:
                cell["term"] = same[0]
                fixed.append(
                    f"analyses[{index}].effect.cells[{position}].term: "
                    f"{named!r} -> {same[0]!r} (same name, in scope)"
                )
    return fixed


def _minted_part(local_id: str) -> str:
    for prefix in _PREFIXES:
        if local_id.startswith(prefix):
            return local_id[len(prefix):]
    return local_id


def _words(text: Any) -> frozenset[str]:
    found = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split()
    return frozenset(word for word in found if word not in _EMPTY_WORDS)


def _same_id(one: str, other: str) -> bool:
    squash = lambda name: re.sub(r"[^a-z0-9]+", "", name.lower())  # noqa: E731
    return squash(one) == squash(other)


def names_agree(dangling: str, target: str, target_name: Any) -> bool:
    """Does the name a dangling id carries refer to what the target is called?

    A local_id is minted from the paper's wording, so `trm_three_way_interaction` says the
    model meant a term it called "three way interaction". Measured over the 120 references
    `the_only_candidate` proposes: 70% share a word, 8% more are an initialism of the
    target's name -- `asm_scid` against "Structured Clinical Interview for DSM-V" -- and the
    18% that share nothing are the wrong repoints, `reg_caudate` and `reg_insula` and
    `reg_thalamus` each onto one "Gain versus nongain reward-processing regions".
    """
    asked = _minted_part(dangling)
    offered = _words(_minted_part(target)) | _words(target_name)

    if _words(asked) & offered:
        return True
    name = str(target_name or "")
    return bool(name and abbreviations.matches(asked.replace("_", ""), name))


@dataclass(frozen=True, eq=False)
class _Reference:
    """One id in one slot, and where to write its replacement."""

    slot: walk.Slot
    position: int
    id: str

    def repoint_to(self, target: str) -> str:
        if isinstance(self.slot.value, list):
            self.slot.value[self.position] = target
            return f"{self.slot.path}[{self.position}]: {self.id!r} -> {target!r}"
        self.slot.owner[self.slot.key] = target
        return f"{self.slot.path}: {self.id!r} -> {target!r}"


def _dangling(body: dict[str, Any], sch: Schema):
    for slot, position, local_id in walk.dangling_references(body, sch):
        yield _Reference(slot=slot, position=position, id=local_id)


def _transcription_slip(reference: _Reference, declared: Mapping[str, str]) -> str | None:
    """The one declared id this differs from only in case and punctuation."""
    same = [name for name in declared if _same_id(name, reference.id) and name != reference.id]
    return same[0] if len(same) == 1 else None


def _the_only_candidate(
    reference: _Reference, declared: Mapping[str, str], sch: Schema
) -> str | None:
    """The one declared entity of the kind this slot holds, where there is only one."""
    wanted = reference.slot.range
    if not isinstance(wanted, str):
        return None
    of_that_kind = [
        name
        for name, class_name in declared.items()
        if class_name == wanted or sch.resolves_to(class_name, wanted)
    ]
    return of_that_kind[0] if len(of_that_kind) == 1 else None


def _without_collapses(proposed: dict[_Reference, str]) -> dict[_Reference, str]:
    """Drop any target that two differently-named references both want.

    An interaction term shares a word with the main effect inside it, so
    `trm_smoking_opportunity_cue` and `trm_quitting_motivation_cue` both pass the name test
    against a term called "cue". A record declaring one term whose cells name three is a
    record missing two, and repointing all three at the survivor invents an effect.
    """
    wanted_by = {}
    for reference, target in proposed.items():
        wanted_by.setdefault(target, set()).add(reference.id)
    return {
        reference: target
        for reference, target in proposed.items()
        if len(wanted_by[target]) == 1
    }


def repair_references(body: dict[str, Any], sch: Schema) -> list[str]:
    """Repoint a cross-reference that names a local_id nothing declares, where forced.

    A dangling reference is the commonest reason a build reports a defect, and it is always
    the same shape: the model wrote `inf_baseline` for an entity it declared as
    `inference_wholebrain_dti`. The record is written either way -- the exit code says the
    record has a fault, not that the paper was skipped -- but a reference resolving nowhere
    costs an analysis its inference settings, and downstream that reads as a missing field
    rather than a naming slip.

    A transcription slip is repaired outright. Anything else is repaired only when all three
    hold, and each condition was earned by a wrong repoint the previous version made:

      the only candidate    exactly one declared entity of the slot's kind exists. Alone
                            this repointed `asm_mini` to `asm_ftnd`, a psychiatric interview
                            onto a nicotine-dependence scale.
      the names agree       and it is called something the dangling id names. This is what
                            makes the rest safe, and it is deterministic because a local_id
                            is minted from the paper's own wording.
      no collapse           and no differently-named reference wants the same target.

    `declared` is read schema-guided. Sweeping the Study-level lists, as this did, misses
    10,867 ids over 1,817 records -- every ModelTerm under `model_estimations[].terms`, every
    Condition under `tasks[].conditions` -- so it repaired neither of the two slots that
    dangle most, while `validate.index_ids` descended and disagreed with it all along.
    """

    declared = walk.declared_ids(body, sch)
    names = {
        entity.local_id: values.read(entity.node.get("name"))
        for entity in walk.entities(body, sch)
        if entity.local_id
    }

    proposed: dict[_Reference, str] = {}
    slips: dict[_Reference, str] = {}
    for reference in _dangling(body, sch):
        slip = _transcription_slip(reference, declared)
        if slip:
            slips[reference] = slip
            continue
        candidate = _the_only_candidate(reference, declared, sch)
        if candidate and candidate != reference.slot.owner.get("local_id"):
            if names_agree(reference.id, candidate, names.get(candidate)):
                proposed[reference] = candidate

    settled = {**slips, **_without_collapses(proposed)}
    return [reference.repoint_to(target) for reference, target in settled.items()]


def _table_number(text: Any) -> tuple[bool, int] | None:
    """(supplementary, number) of a table id or label.

    The number after the table token (`table2`, `tbltableS2`, `pone-0042560-t002`, `tbl1_2`,
    whose `_2` is a collision suffix), else the first number standing alone (`Table 2`).
    Parenthetical text is not the number (`Table 2 (n = 30)`), and "supplementary" or an
    `S` before the digits marks a supplementary table.
    """
    if not isinstance(text, str):
        return None
    folded = re.sub(r"\([^)]*\)", " ", text.lower())
    found = re.search(r"(?:table|tbl|\bt)(s?)0*(\d+)", folded) or re.search(
        r"(?:^|[^a-z0-9])(s?)0*(\d+)", folded
    )
    if found is None:
        return None
    return ("supplement" in folded or bool(found.group(1)), int(found.group(2)))


def settle_table_references(
    body: dict[str, Any], sch: Schema, table_map: Path | None
) -> list[str]:
    """Repoint or drop a reference to a table no Table entity declares.

    Tables come from the parse, never from a model pass, so no retry can declare the table a
    dangling reference names. 26952803's text keeps its tables' captions but not their rows,
    the parse found none, and `single` wrote `tables: ["table2"]` twice before a third
    attempt left it out. A reference is repointed when exactly one declared table carries
    its number, and dropped otherwise.

    Only once the `tables` stage has run (it always writes `table_map`): before that, the
    table may yet be declared.
    """
    if table_map is None or not table_map.is_file():
        return []
    tables = {
        t["local_id"]: t
        for t in body.get("tables") or []
        if isinstance(t, Mapping) and isinstance(t.get("local_id"), str)
    }
    by_number: dict[tuple[bool, int] | None, list[str]] = {}
    for local_id, table in tables.items():
        by_number.setdefault(_table_number(values.read(table.get("table_number"))), []).append(
            local_id
        )
    targets = {"Table", *sch.subclasses("Table")}
    fixed: list[str] = []
    for slot in walk.references(body, sch):
        if not targets & set(sch.ranges(slot.attribute)):
            continue
        named = walk.ids_of(slot.value)
        if all(i in tables for i in named):
            continue
        kept: list[str] = []
        for i in named:
            number = _table_number(i)
            match = [i] if i in tables else by_number.get(number, []) if number else []
            if len(match) == 1:
                if match[0] != i:
                    fixed.append(f"{slot.path}: {i!r} -> {match[0]!r}, the only table so numbered")
                if match[0] not in kept:
                    kept.append(match[0])
            else:
                why = (
                    "more than one declared table is so numbered"
                    if len(match) > 1
                    else "no declared table is so numbered"
                    if tables
                    else "the paper has no declared table"
                )
                fixed.append(f"{slot.path}: {i!r} dropped -- {why}")
        if kept:
            slot.owner[slot.key] = kept if isinstance(slot.value, list) else kept[0]
        else:
            del slot.owner[slot.key]
    return fixed


def check_local_ids(body: dict[str, Any], sch: Schema) -> list[str]:
    """Verify that every cross-reference resolves to a declared local_id.

    The schema identifies cross-reference slots; they are not hardcoded. A slot is a
    reference when its range is a native string and it is
    not local_id. That distinction matters because sibling slots differ --
    PredictorSource.group is a local_id string while PredictorSource.other is an
    ExtractedString, and Predictor.source is a nested object, not a reference.
    """

    # Counted: a reference resolves to a *set* membership, so two
    # entities sharing a local_id both "resolve" and nothing downstream can tell which
    # one a Cell.term meant. Sibling model estimations that share covariate names --
    # term_age, term_sex -- produce this without anything looking wrong.
    times: dict[str, int] = {}

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            if values.is_field(node):
                return
            if isinstance(node.get("local_id"), str):
                times[node["local_id"]] = times.get(node["local_id"], 0) + 1
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(body)
    problems: list[str] = [
        f"local_id {name!r} is declared {count} times; every reference to it is ambiguous"
        for name, count in sorted(times.items())
        if count > 1
    ]

    # The walk is `walk`'s: from Study, through the type designator, so a reference under
    # a non-list slot or on a payload subclass is checked too.
    problems += [
        f"{slot.path} -> unknown local_id {name!r}"
        for slot, _position, name in walk.dangling_references(body, sch)
    ]
    return problems
