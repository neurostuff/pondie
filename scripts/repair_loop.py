"""An analysis-anchored repair loop over a pondie record.

Three findings shape this. Prose does not fix extraction: a schema-description change moved
nothing beyond the 82% run-to-run floor. Pondie is demand-driven, so it cannot recover an
entity no analysis thought to ask for -- 450 `roi` analyses across 161 papers name no region.
And its own `evidence.status` says only that a sentence was *located*, not that the sentence
supports the value.

So the loop asks two different questions, and repairs with a different model:

    1. is this FACT supported?      claim vs the span the record already cites
    2. does this ENTITY EXIST?      an existence claim vs Methods+Results
    3. repair and re-search         one NuExtract call per entity class

**Analyses are the anchor.** The goal is not to describe the paper; it is to describe
everything tied to an analysis. An entity is in scope only if it is reachable from an
analysis through the schema's own reference slots, and an entity that is not reachable is
reported as an orphan rather than repaired. That is also why `analyses` is swept every
iteration even when nothing about it looks suspect: an analysis nobody extracted has no
field to look suspicious.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path("/data/james/pondie-vs-fulltext")

#: Reference slots that carry the analysis graph, per storage class. Reachability from an
#: analysis is what "pertinent" means here; anything else is an orphan.
DOWNSTREAM = {
    "analyses": [("groups", "groups"), ("measure", "measures"), ("tasks", "tasks"),
                 ("acquisitions", "acquisitions"), ("model_estimation", "model_estimations"),
                 ("inference_settings", "inference_settings"), ("regions", "regions"),
                 ("tables", "tables"), ("assessments", "assessments")],
    "acquisitions": [("device", "devices")],
    "model_estimations": [("preprocessing", "preprocessings"), ("terms", "model_terms")],
    "tasks": [("acquisitions", "acquisitions")],
}

#: Which of pondie's section labels an existence check may read. Methods and results are
#: where a paper describes its own design; `tables` because the coordinates an analysis
#: exists by are printed there. `intro` and `discussion` are excluded on purpose -- they
#: describe *other* studies' findings, so including them lets an entity from a different
#: paper look supported, which is the failure an existence check exists to catch.
PREMISE_LABELS = ("methods", "results", "tables", "abstract")

#: record key -> {slot: (target class, multivalued)}, filled from the schema at startup.
REF_SLOTS: dict[str, dict[str, tuple[str, bool]]] = {}

#: `cls[:-1]` turns "analyses" into "analyse" and "inference_settings" into
#: "inference_setting". Both read as errors, to a reader and to a checker.
SINGULAR = {
    "analyses": "analysis", "groups": "participant group", "tasks": "task",
    "acquisitions": "acquisition", "devices": "scanner", "measures": "measure",
    "regions": "brain region", "preprocessings": "preprocessing procedure",
    "model_estimations": "statistical model", "inference_settings": "statistical threshold",
    "tables": "table", "assessments": "assessment", "model_terms": "model term",
}

#: What a nested container is, said in words. Without this a claim about
#: `effect.cells[0].level` reads "the level is PTSD" with no indication of which cell, and
#: 13% of claims were unanchored that way.
CONTAINER = {
    "cells": "contrast cell", "terms": "model term", "levels": "factor level",
    "groups": "analysis group", "conditions": "task condition", "arms": "trial arm",
    "timepoints": "timepoint", "sex_distribution": "sex breakdown entry",
    "race_distribution": "race breakdown entry", "steps": "preprocessing step",
    "effect": "reported effect", "statistic": "test statistic",
    "details": "method detail", "design": "study design", "mediation": "mediation path",
}


def enum_glosses(schema) -> dict[str, str]:
    """token -> the same token as words.

    A paper never prints `whole_brain`, so a claim carrying the raw token asks a checker to
    match a string that cannot appear in the text. Words only: appending the schema's
    definition -- "whole brain (inference across the whole acquired volume or surface)" --
    adds a clause the paper never states, and measured 2-6 points of prose precision.
    """
    out: dict[str, str] = {}
    for enum in schema.enums.values():
        for token in (getattr(enum, "permissible_values", None) or {}):
            out.setdefault(str(token), str(token).replace("_", " "))
    return out


def stage1_analyses(corpus: Path, pmid: str) -> list[dict]:
    """The coordinate-table parse: the only place reported foci are enumerated."""
    path = corpus / pmid / "stage1" / "analyses.orig.json"
    if not path.is_file():
        path = corpus / pmid / "stage1" / "analyses.json"
    if not path.is_file():
        return []
    doc = json.loads(path.read_text())
    rows = []
    # No layer carries a row-group identifier. `stage1/analyses.json`, ns-pond's
    # `analyses.jsonl` and autonima's `coordinate_parsing_results.json` all key on
    # `name` + `table_id`, and one table_id covers many row groups -- table 788 of PMID
    # 17825801 holds fourteen. Binding on table_id therefore gave every added analysis the
    # same foci. Position in the parse is the only stable handle, so it becomes the id.
    for index, entry in enumerate(doc.get("analyses") or []):
        # Row groups with no coordinates are kept. A table section reading "n.s." is a
        # reported analysis that found nothing, not an absent one, and dropping those made
        # 142 of the corpus's 1,822 row groups unbindable -- so a real null analysis could
        # never be admitted, however plainly the paper reported it. The necessary condition
        # is a row group to point at, not a coordinate.
        entry = dict(entry)
        entry["_rg_id"] = f"{entry.get('table_id') or 'tbl'}#{index}"
        entry["_index"] = index
        rows.append(entry)
    return rows


def claimed_table_ids(record: dict) -> set[str]:
    """Table ids the record's analyses already point at."""
    out = set()
    for a in entities(record, "analyses"):
        for slot in ("tables", "source_table_analysis"):
            raw = a.get(slot)
            for ref in (raw if isinstance(raw, list) else [raw]):
                got = text_of(ref) if isinstance(ref, dict) else ref
                if isinstance(got, str) and got.strip():
                    out.add(got.strip())
        prov = a.get("_provenance") or {}
        if prov.get("bound_row_group"):
            out.add(str(prov["bound_row_group"]))
    return out


def foci_block(parsed: list[dict], claimed: set[str]) -> str:
    """The parsed coordinate tables, written into the prompt.

    Without this the model proposes analyses blind and names the table it thinks it saw in
    free text, which leaves the binding to a fuzzy match after the fact. Given the actual
    table ids, their captions and their foci, choosing a table becomes a choice among real
    options -- and an analysis that maps to none of them is one the model itself declines
    to bind, rather than one this code rejects afterwards.
    """
    if not parsed:
        return ("## Coordinate tables parsed from this paper\n\n"
                "NONE. No coordinate table was parsed for this paper, so no new analysis "
                "can be tied to reported foci. Return an empty list.\n\n")
    lines = ["## Coordinate row groups parsed from this paper", ""]
    for entry in parsed:
        rg = entry["_rg_id"]
        pts = entry.get("points") or []
        coords = "; ".join(",".join(str(int(float(c))) for c in (p.get("coordinates") or [])[:3])
                           for p in pts[:5])
        already = "already referenced" if rg in claimed else "not yet referenced"
        lines.append(f"- row_group `{rg}` ({already}) — {entry.get('table_number') or ''} "
                     f"{str(entry.get('table_caption') or '')[:120]}")
        lines.append(f"    reported as: {str(entry.get('name') or '(unnamed)')[:110]}")
        lines.append(f"    {len(pts)} foci: {coords}")
    lines += [
        "",
        "Each analysis you list MUST set `source_row_group` to one of the row_group ids "
        "above. Those ids are the only place this paper's coordinates are enumerated, and "
        "an analysis with no coordinates is not an analysis. Match on what the row group is "
        "'reported as'.",
        "",
        "More than one analysis may cite the SAME table_id, and a table marked 'already "
        "referenced' is still available: one set of rows can report an omnibus F-test and "
        "the directed post-hoc contrast that follows it, and those are two analyses. List "
        "the additional one whenever the paper describes it.",
        "",
    ]
    return "\n".join(lines)


def slot_summary(entity: dict, cls: str, limit: int = 8) -> str:
    """An entity's current slot values, so the model can correct them rather than guess."""
    parts = []
    for key, node in entity.items():
        if key in ("local_id", "_provenance", "name"):
            continue
        got = text_of(node)
        if got:
            parts.append(f"{key.replace('_', ' ')}={got[:70]}")
        if len(parts) >= limit:
            break
    return "; ".join(parts)


def entity_block(record: dict, cls: str) -> str:
    """The entities already held, with their ids and current values -- an editable set.

    Listing only labels made the sweep append-only: the model could propose something new
    but had no handle on what was already there, so nothing could be corrected and nothing
    could be linked to. Passing `local_id` and the current slot values makes the same call
    do three jobs -- fix a wrong value, add a missing entity, and reference an existing one
    by an id that is guaranteed to resolve.
    """
    have = entities(record, cls)
    if not have:
        return (f"## {cls.replace('_', ' ')} already extracted\n\nNONE.\n\n")
    lines = [f"## {cls.replace('_', ' ')} already extracted", ""]
    for e in have:
        lines.append(f"- local_id `{e.get('local_id')}` — {label_of(e, cls)}")
        summary = slot_summary(e, cls)
        if summary:
            lines.append(f"    current values: {summary}")
    lines += [
        "",
        "Return the COMPLETE list. For one of these, reuse its `local_id` and correct any "
        "value the paper contradicts; leave a value out if the paper does not state it. For "
        "one the paper describes that is missing above, set `local_id` to `NEW`.",
        "",
    ]
    return "\n".join(lines)


def reference_block(record: dict, cls: str) -> str:
    """The entities *this* class may point at, one list per reference slot it declares.

    Read from `REF_SLOTS`, which the schema fills, rather than from a fixed list of five
    classes. The fixed list showed the same five however the sweep was aimed, so sweeping
    `groups` -- whose `diagnostic_instrument` targets an Assessment -- offered no assessments
    at all, and the model had to name the CAPS in prose; `resolve_refs` then minted a second
    copy of an instrument the record already held. Four of Analysis's own eight reference
    slots (acquisitions, model_estimation, tables, assessments) were never offered either.
    """
    lines = []
    for slot, (target_class, _multi) in sorted((REF_SLOTS.get(cls) or {}).items()):
        key = CLASS_TO_KEY.get(target_class)
        have = entities(record, key) if key else []
        if not have:
            continue
        # No truncation. The cap was 25 per class, and a paper with more regions than that
        # lost the tail silently -- the candidates most likely to be the unlinked ones.
        listed = "; ".join(label_of(e, key) for e in have if label_of(e, key))
        if listed:
            lines.append(f"- `{slot}` may name any of these {target_class}: {listed}")
    if not lines:
        return ""
    noun = SINGULAR.get(cls, cls)
    article = "an" if noun[:1].lower() in "aeiou" else "a"
    return (f"## Entities {article} {noun} can reference\n\n" + "\n".join(lines) +
            f"\n\nName any of these that a given {noun} used, under the slot that lists "
            "it. Name one the paper describes even if it is not listed; it will be "
            "created.\n\n")


#: Measured on this corpus, loaded bf16: the grounding checker (flan-t5-large) takes about
#: 1.9 GB and the proposer (NuExtract3, a 4B model) about 6.2 GB. Together they exceed an
#: 8 GB card, which is why the default puts them on different ones.
CHECKER_GB, PROPOSER_GB = 2.5, 7.0


def device_plan(shard: int, shards: int) -> tuple[str, str, int]:
    """Which cards this shard may see, and where each model goes on them.

    Derived from what is present rather than from a host name, so the answer is right on a
    four-card box and on a one-card laptop. Two rules, both from measurement:

      * the two models do not share a card unless one card can hold both, because
        6.2 + 1.9 GB does not fit in 8 and `device_map="auto"` responds by spilling to CPU
        rather than by failing;
      * a shard is pinned to its own cards with CUDA_VISIBLE_DEVICES, because two processes
        each opening a context on all four cards left ~1 GB of stranded allocation per
        process per card and drove the smaller card into repeated OOM retries.
    """
    import torch

    count = torch.cuda.device_count()
    if not count:
        return "", "cpu", -1
    free = min(torch.cuda.get_device_properties(i).total_memory for i in range(count))
    together = free / 1024**3 >= CHECKER_GB + PROPOSER_GB
    per_shard = 1 if together else 2
    if shards * per_shard > count:
        raise SystemExit(
            f"{shards} shards need {shards * per_shard} of {count} cards "
            f"({'both models fit one card' if together else 'one card each'}); "
            f"use --shards {max(1, count // per_shard)}")
    mine = list(range(shard * per_shard, shard * per_shard + per_shard))
    visible = ",".join(str(i) for i in mine)
    # local indices: CUDA_VISIBLE_DEVICES renumbers from 0
    return visible, "cuda:0", 0 if together else 1


def default_shards() -> int:
    import torch

    count = torch.cuda.device_count()
    if not count:
        return 1
    free = min(torch.cuda.get_device_properties(i).total_memory for i in range(count))
    return max(1, count // (1 if free / 1024**3 >= CHECKER_GB + PROPOSER_GB else 2))


def split_evenly(pmids: list[str], corpus: Path, shards: int, shard: int) -> list[str]:
    """This shard's papers, balanced by how much work each is rather than by count.

    Cost tracks the source length: the grounding pass scores every claim against the
    methods and results, and the proposer's premise is that same slice. Splitting the list
    in half by position left one shard six papers behind the other on this corpus, so the
    long tail is dealt round-robin over shards ordered by load.
    """
    def cost(pmid: str) -> int:
        path = corpus / pmid / "processed/local/text.tables.txt"
        try:
            return path.stat().st_size
        except OSError:
            return 0

    buckets: list[list[str]] = [[] for _ in range(shards)]
    loads = [0] * shards
    for pmid in sorted(pmids, key=cost, reverse=True):
        i = loads.index(min(loads))
        buckets[i].append(pmid)
        loads[i] += cost(pmid)
    return buckets[shard]


def assert_on_gpu(holder, what: str, allow_cpu: bool) -> None:
    """Fail if any weight sits on the CPU, unless the caller asked for that.

    `device_map="auto"` places what fits and spills the rest, silently: it put 9.3 GB of
    NuExtract on the CPU rather than splitting it over two cards, and MiniCheck loads the
    same way. Nothing in either library says so at the time, and the only symptom is a run
    that never finishes -- which reads as a slow GPU, not a model on the wrong device.
    """
    import torch

    model = getattr(holder, "model", holder)
    for _ in range(2):                       # MiniCheck wraps its model one level deep
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "parameters"):
            model = inner
    devices = {str(p.device) for p in getattr(model, "parameters", lambda: [])()}
    on_cpu = {d for d in devices if d.startswith("cpu")}
    print(f"  {what}: weights on {sorted(devices) or ['unknown']}", flush=True)
    if on_cpu and not allow_cpu:
        raise SystemExit(
            f"{what} has weights on {sorted(on_cpu)}; pass --allow-cpu to run there anyway, "
            f"or free a card. torch.cuda.is_available()={torch.cuda.is_available()}, "
            f"devices={torch.cuda.device_count()}")


def paper_abbreviations(text: str):
    """The paper's own abbreviation table, or None if scispacy is unavailable.

    pondie mines these with Schwartz & Hearst -- almost every abbreviation is defined on
    first use, "the Clinician-Administered PTSD Scale (CAPS)" -- and scopes them to the
    paper, which is the only scope in which they are true. A hand-kept list of "generic"
    acronyms was standing in for this, and a hand-kept list is a claim about the field that
    nobody checks: it had to guess that PTSD is uninformative and CAPS is not.
    """
    try:
        from pondie.vocabularies.abbreviations import Abbreviations
        return Abbreviations.load().for_paper(text)
    except Exception as exc:
        print(f"  warn: no abbreviation table ({exc})", flush=True)
        return None


def expand_label(label: str, abbrevs, rounds: int = 3) -> set[str]:
    """The words a label denotes, with the paper's abbreviations spelled out.

    Repeated, because an expansion can itself contain an abbreviation: "CAPS" expands to
    "clinician-administered PTSD scale", which still says PTSD. One pass left the two sides
    of the same instrument written differently -- `...ptsd scale total score` against
    `...posttraumatic stress disorder scale` -- and they failed to match.

    A set of words rather than a string, because expansion duplicates text: writing out
    "clinician-administered PTSD scale (CAPS)" yields the expansion twice, so substring
    containment stops working while subset still does.
    """
    text = label or ""
    for _ in range(rounds):
        out, grew = [], False
        for token in re.findall(r"[A-Za-z0-9-]+", text):
            full = abbrevs.expand(token) if abbrevs is not None and token.isupper() else None
            out.append(full or token)
            grew |= bool(full)
        text = " ".join(out)
        if not grew:
            break
    return set(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def same_entity(a: str, b: str, abbrevs) -> bool:
    """Do two labels name one thing, once the paper's abbreviations are spelled out?

    One name contains the other's words: "CAPS total score" says everything
    "clinician-administered PTSD scale (CAPS)" says and adds which score. "PTSD checklist"
    and "PTSD symptom scale" share their expansion and still differ on the rest, so they stay
    two instruments -- which is the discrimination a stoplist of "generic" acronyms was
    standing in for, and got right only by being told PTSD in advance.
    """
    x, y = expand_label(a, abbrevs), expand_label(b, abbrevs)
    small, large = (x, y) if len(x) <= len(y) else (y, x)
    # Two words at least: "scale" alone is a subset of half the instruments in any paper.
    return len(small) >= 2 and small <= large


#: The address prefixes the extraction prompt gives the model, so a repaired record does not
#: mix two id vocabularies. A local_id is an address -- the review layer keys answers on
#: `paper|value|<Class>|<local_id>|<path>` -- and an entity minted as `nu_regions_0` is not
#: addressable the way every other region beside it is. The `nu_` prefix also fell outside
#: the prune's own exclusion list, making minted entities more deletable than their peers.
ID_PREFIX = {"groups": "grp_", "acquisitions": "acq_", "model_estimations": "mod_",
             "regions": "reg_", "measures": "mea_", "tasks": "tsk_", "devices": "dev_",
             "preprocessings": "prep_", "inference_settings": "inf_", "analyses": "ana_",
             "assessments": "asm_", "tables": "tbl_"}


def mint_id(record: dict, cls: str, label: str, offset: int) -> str:
    """An address for a new entity, in the same form the extractor was told to use."""
    stem = re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_")
    stem = "_".join(stem.split("_")[:3])[:28] or f"n{len(entities(record, cls)) + offset}"
    prefix = ID_PREFIX.get(cls, f"{cls[:3]}_")
    candidate = f"{prefix}{stem}"
    taken = {e.get("local_id") for e in entities(record, cls)}
    if candidate not in taken:
        return candidate
    n = 2
    while f"{candidate}_{n}" in taken:
        n += 1
    return f"{candidate}_{n}"


#: Carried over from the instructions the extracting model is given (`prompt/render.py`),
#: because this pass writes into the same record and was being held to none of them. Only the
#: rules that bear on what this pass can do: addressing, references, and not inventing.
SHARED_RULES = """
`local_id` is an ADDRESS, not a description. To CORRECT an entity already listed above, copy
its `local_id` exactly; the reply is then an edit of that entity rather than a new one. Give a
`local_id` only for an entity you are correcting -- leave it out for one you are adding, and
it will be assigned. Never invent an id for an entity that is not listed.

A cross-reference names another entity. Give the name exactly as listed above, or omit the
field entirely when there is nothing to point at: a reference has no "not reported" form, so
an empty string or a guess is worse than an absent field.

Do not invent a value to fill a field. If the paper does not state it, leave the field out.
A field you fill must be something the paper says, in the paper's own terms.

An analysis restricted to a region names that region; one run over the whole brain names
none. The volume searched and the volume corrected are different claims: a whole-brain model
whose significance was corrected inside a small volume is whole-brain in scope and restricted
in correction, and gray or white matter masking is not a restriction to a region at all.
"""


def sweep_order(classes: list[str]) -> list[str]:
    """Sweep a class after everything it points at, so its links have targets to find.

    `analyses` came first because it is the central object, and that is exactly wrong for
    linking: on 16508348 NuExtract named `hippocampus`, `right hippocampus`, `bilateral
    hippocampal` and `bilateral gray matter parahippocampal` for four ROI analyses, and every
    one was refused because no Region existed yet -- the regions sweep ran last, after the
    references it would have satisfied had already been thrown away.

    The order comes from `REF_SLOTS`, so it follows the schema rather than a list here.
    """
    targets = {key: {CLASS_TO_KEY.get(cls) for cls, _multi in (slots or {}).values()}
               for key, slots in REF_SLOTS.items()}
    out, seen = [], set()

    def visit(key: str, path: frozenset) -> None:
        if key in seen or key in path:      # a cycle means neither can go first; order by
            return                          # the caller's list and let the next iteration fix
        for target in sorted(x for x in (targets.get(key) or set()) if x in classes):
            visit(target, path | {key})
        if key not in seen:
            seen.add(key)
            out.append(key)

    for key in classes:
        visit(key, frozenset())
    return out


def model_terms(model_id: str, record: dict, seen: frozenset = frozenset()) -> set[str]:
    """Term ids a model declares, following `inputs_from` as the stage chain does."""
    if not isinstance(model_id, str) or model_id in seen:
        return set()
    model = next((m for m in record.get("model_estimations") or []
                  if isinstance(m, dict) and m.get("local_id") == model_id), None)
    if model is None:
        return set()
    out = {t.get("local_id") for t in (model.get("terms") or []) if isinstance(t, dict)}
    for lower in model.get("inputs_from") or []:
        out |= model_terms(lower, record, seen | {model_id})
    return out


def orphans_cell_terms(analysis: dict, model_id: str, record: dict) -> bool:
    """Would pointing this analysis at `model_id` leave its cells naming unreachable terms?"""
    named = {c.get("term") for c in ((analysis.get("effect") or {}).get("cells") or [])
             if isinstance(c, dict) and isinstance(c.get("term"), str)}
    if not named:
        return False
    return bool(named - model_terms(model_id, record))


def referenced(record: dict) -> set[str]:
    """Every local_id anything points at, anywhere in the record.

    The whole record, not the top level of each entity list: half the reference slots sit on
    nested objects. `Analysis.groups` holds AnalysisGroup, and it is `AnalysisGroup.group`
    that names the Group -- so a walk over top-level slots alone sees no reference to any
    group at all, and the terminal pass deleted all three of one paper's participant groups
    while three analyses were pointing at them. `Cell.term`, `FactorLevel.regions` and
    `Mediation.mediator` are nested the same way.

    Slot names are collected across every class, so a name that is a reference anywhere
    protects its target everywhere. That over-collects when two classes share a slot name,
    which is the safe direction for a guard whose job is to stop a deletion.
    """
    names = {slot for slots in REF_SLOTS.values() for slot in slots} | REF_SLOT_NAMES
    out: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in names:
                    for one in (value if isinstance(value, list) else [value]):
                        if isinstance(one, str):
                            out.add(one)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(record)
    return out


#: Reference slots on classes that are only ever nested, so they never appear in `REF_SLOTS`
#: -- which is keyed by the top-level record lists. Read from the schema at start-up.
REF_SLOT_NAMES: set[str] = set()


def bind_foci(candidate: dict, unclaimed: list[dict]) -> dict | None:
    """Attach a candidate analysis to reported foci, or refuse it.

    An analysis with no foci is not an analysis this pipeline can use: it cannot enter a
    coordinate meta-analysis, and nothing downstream can check it. So a candidate is
    admitted only when it can be bound to a parsed coordinate table that no existing
    analysis has claimed. Matching is on the table the candidate says it was reported in,
    then on name overlap, then -- only if exactly one table is left -- positionally.
    """
    if not unclaimed:
        return None
    chosen = str(candidate.get("source_row_group") or "").strip()
    if chosen:
        for entry in unclaimed:
            if entry["_rg_id"] == chosen:
                return entry
        return None          # named a row group this paper does not have
    said = re.sub(r"[^a-z0-9]+", " ", str(candidate.get("reported_in") or "").lower())
    want = {t for t in re.findall(r"table\s*(\w+)", said)}
    for entry in unclaimed:
        tid = str(entry.get("table_id") or "")
        num = str(entry.get("table_number") or "")
        if tid and (tid.lower() in want or num.lower() in want):
            return entry
    name = re.sub(r"[^a-z0-9]+", " ", str(candidate.get("name") or "").lower()).split()
    best, score = None, 0
    for entry in unclaimed:
        words = set(re.sub(r"[^a-z0-9]+", " ",
                           str(entry.get("name") or "").lower()).split())
        overlap = len(words & set(name))
        if overlap > score:
            best, score = entry, overlap
    if best is not None and score >= 2:
        return best
    # No positional fallback. Handing a candidate the last unclaimed table because it is the
    # last one is a guess, not the direct mapping the necessary condition asks for: on
    # 22952599 it bound "ROI analysis of hippocampus and amygdala" to the whole-brain table,
    # giving it premotor and parietal peaks, for an analysis the paper reports as null.
    return None


#: A normalized number reads as unsupported against prose that states it differently --
#: "echo time seconds is 0.004" against "TE = 4 ms". Measured on one paper: prose claims mean
#: 0.571, numeric claims 0.114. Scoring them together buries the signal, so they are counted
#: apart and only prose claims drive repair.
NUMERIC = re.compile(r"^[-+0-9.,;:\s]+$")


def is_numeric(value: str) -> bool:
    return bool(NUMERIC.match(value or ""))


#: Factors between the unit the schema stores and the ones papers print. Small on purpose:
#: every entry is a convention someone actually writes, not a search for any factor that
#: happens to fit.
SCALINGS = (1, 1e3, 1e-3, 1e2, 1e-2, 60, 1 / 60)

#: Counts under about thirty are commonly spelled out -- "Twenty-one individuals" -- and a
#: group size is exactly the field where that happens.
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}


def numbers_in(text: str) -> set[float]:
    """Every number the text states, digits or words."""
    found = {float(m.group(0)) for m in re.finditer(r"-?\d+(?:\.\d+)?", text or "")}
    lowered = (text or "").lower()
    for match in re.finditer(r"\b([a-z]+)(?:[- ]([a-z]+))?\b", lowered):
        head, tail = match.group(1), match.group(2)
        if head not in WORD_NUMBERS:
            continue
        total = WORD_NUMBERS[head]
        if tail in WORD_NUMBERS and total >= 20 and WORD_NUMBERS[tail] < 10:
            total += WORD_NUMBERS[tail]       # "twenty-one"
        found.add(float(total))
    return found


def ground_numeric(value: str, span: str) -> tuple[bool, float | None]:
    """Does every number in the value appear in the span at one consistent scaling?

    Grounding a number is arithmetic, not entailment. `repetition_time_seconds = 0.0097`
    against "repetition time=9.7 ms" is the same quantity in the unit the schema stores and
    the unit the paper prints; an NLI model scored that pair 0.11 and the repair pass then
    "corrected" the value to itself. Measured over five papers: 70 of 74 numeric claims
    settle this way, so 95% of them need no model at all -- and the four that do not are
    worth reading, because two were values *derived* from other numbers rather than reported.
    """
    wanted = numbers_in(value)
    present = numbers_in(span)
    if not wanted or not present:
        return False, None
    for factor in SCALINGS:
        if all(any(abs(w * factor - p) <= max(1e-9, abs(p) * 1e-6) for p in present)
               for w in wanted):
            return True, factor
    return False, None


def leaf(node):
    if isinstance(node, dict) and "extraction_status" in node:
        return node.get("value")
    return node


def text_of(node) -> str:
    value = leaf(node)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(text_of(v) for v in value if text_of(v))
    if isinstance(value, dict):
        return "; ".join(f"{k} {text_of(v)}" for k, v in value.items() if text_of(v))
    return re.sub(r"\s+", " ", str(value)).strip()


#: How to describe an entity that has no `name`. Falling back to `local_id` produces
#: "The paper describes acquiring this imaging data: acq_mri", which no fact-checker can
#: support -- and the terminal pass then prunes a perfectly real acquisition. The label has
#: to be built from slots the paper actually stated.
DESCRIPTORS = {
    "acquisitions": ("modality", "acquisition_type", "magnetic_field_strength_tesla",
                     "pulse_sequence_type"),
    "inference_settings": ("multiple_comparison_method", "inference_level",
                           "height_threshold_value", "correction_scope"),
    "measures": ("family", "type", "specific_metric", "source_label"),
    "model_estimations": ("model_family", "model_type", "estimator", "stage"),
    "devices": ("manufacturer", "model"),
    "preprocessings": ("description", "software"),
    "tables": ("table_number", "caption"),
    "groups": ("description", "medical_condition", "species"),
    "regions": ("description", "atlas", "definition_method"),
    "tasks": ("description", "design_type"),
    "analyses": ("definition", "spatial_scope", "measure"),
}

#: What `text_of` yields for a slot the model left as a JSON null, which then reached the
#: claim as the literal word "None".
NOT_A_LABEL = {"", "none", "null", "n/a", "unknown", "(unlabelled)"}


def label_of(entity: dict, cls: str | None = None) -> str:
    for key in ("name", "definition", "description"):
        got = text_of(entity.get(key))
        if got and got.strip().lower() not in NOT_A_LABEL:
            return got[:140]
    parts = [text_of(entity.get(k)) for k in DESCRIPTORS.get(cls or "", ())]
    composed = ", ".join(p for p in parts if p and p.strip().lower() not in NOT_A_LABEL)
    return composed[:140] if composed else (entity.get("local_id") or "(unlabelled)")


def entities(record: dict, key: str) -> list[dict]:
    return [e for e in (record.get(key) or []) if isinstance(e, dict)]


def reachable(record: dict) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(in-scope, orphan) as (class_key, local_id) pairs, walking out from the analyses."""
    index = {(k, e.get("local_id")): e
             for k in record if isinstance(record.get(k), list)
             for e in entities(record, k) if e.get("local_id")}
    seen: set[tuple[str, str]] = set()
    queue = [("analyses", e.get("local_id")) for e in entities(record, "analyses")]
    while queue:
        node = queue.pop()
        if node in seen or node not in index:
            continue
        seen.add(node)
        for slot, target in DOWNSTREAM.get(node[0], []):
            raw = index[node].get(slot)
            refs = raw if isinstance(raw, list) else [raw]
            for ref in refs:
                if isinstance(ref, dict):          # AnalysisGroup wraps its Group reference
                    ref = ref.get("group") or ref.get("local_id")
                if isinstance(ref, str):
                    queue.append((target, ref))
    return seen, set(index) - seen


def sections(text: str, labels: tuple[str, ...] = PREMISE_LABELS) -> str:
    """The methods/results-bearing spans of a paper, concatenated.

    Uses pondie's own `evidence.retrieval.sectionize` rather than a second classifier. It is
    better than the one this file had: it strips section numbering so "2.3. Statistical
    analysis" matches, it lets an unnamed subsection inherit its parent so a Results
    subsection is not read as Methods, and its pattern list is corpus-validated -- the
    comment on `_HEADING` records 1,710 papers once misreading their own Results.

    Disjoint sections fall out for free. `sectionize` returns spans, not one range, so a
    structured abstract's Methods and the body's Methods arrive as two spans and both are
    kept; there is no "first match wins" to get wrong.

    Falls back to the whole paper when the selection is implausibly small, which is what a
    render with no headings produces.
    """
    from pondie.extraction.evidence.retrieval import sectionize

    spans = [(start, end) for start, end, label in sectionize(text) if label in labels]
    joined = "\n\n".join(text[start:end] for start, end in spans)
    if len(joined) < 2000 or len(joined) < 0.10 * len(text):
        return text
    return joined


# --------------------------------------------------------------------------- claims

def gloss(value: str, glosses: dict[str, str]) -> str:
    """Render a value, expanding an enum token into words plus what the schema says it means.

    A paper never prints `whole_brain`, so a claim carrying the raw token asks the checker to
    match a string that cannot appear in the text. 193 of the schema's 210 permissible values
    have a description; this uses them.
    """
    out = []
    for part in (p.strip() for p in value.split(";")):
        if not part:
            continue
        out.append(glosses.get(part, part.replace("_", " ")
                               if re.fullmatch(r"[a-z][a-z0-9_]*", part) else part))
    return "; ".join(out)


def fact_claims(record: dict, scope: set, glosses: dict[str, str] | None = None) -> list[dict]:
    """One claim per extracted leaf on an in-scope entity, with the span it cites.

    The subject names the entity *and* the nested object the leaf sits in. Without the
    second part, `effect.cells[0].level` reads "the level is PTSD" with nothing saying which
    cell -- 13% of claims were unanchored that way, and an unanchored claim is one no
    checker can fairly judge.
    """
    glosses = glosses or {}
    rows: list[dict] = []

    def walk(node, path, subject, trail, cls_id):
        if isinstance(node, dict):
            if "extraction_status" in node:
                if node.get("extraction_status") != "extracted":
                    return
                raw = text_of(node)
                if not raw:
                    return
                field = path.rsplit(".", 1)[-1].split("[")[0].replace("_", " ")
                where = f" {', '.join(trail)}," if trail else ""
                quotes = [sp.get("text", "")
                          for s in ((node.get("evidence") or {}).get("sets") or [])
                          for sp in (s.get("spans") or [])]
                rows.append({"kind": "fact", "entity": cls_id, "path": path, "field": field,
                             "claim": f"{subject}{where} the {field} is "
                                      f"{gloss(raw, glosses)}.",
                             "value": raw, "premise": " ".join(quotes)[:1500],
                             "evidence_status": (node.get("evidence") or {}).get("status")})
                return
            for k, v in node.items():
                if k in ("extraction_metadata", "local_id"):
                    continue
                deeper = trail
                if isinstance(v, dict) and "extraction_status" not in v and k in CONTAINER:
                    deeper = trail + [f"in the {CONTAINER[k]}"]
                walk(v, f"{path}.{k}" if path else k, subject, deeper, cls_id)
        elif isinstance(node, list):
            noun = CONTAINER.get(path.rsplit(".", 1)[-1].split("[")[0])
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]", subject,
                     trail + [f"in {noun} {i + 1}"] if noun else trail, cls_id)

    for cls, local in sorted(scope):
        for e in entities(record, cls):
            if e.get("local_id") != local:
                continue
            walk({k: v for k, v in e.items() if k != "name"}, "",
                 f'For the {SINGULAR.get(cls, cls.rstrip("s"))} "{label_of(e, cls)}",',
                 [], (cls, local))
    return rows


EXISTENCE = {
    "analyses": "The paper reports a statistical analysis: {label}.",
    "groups": "The paper describes a group of participants: {label}.",
    "acquisitions": "The paper describes acquiring this imaging data: {label}.",
    "tasks": "Participants performed this task: {label}.",
    "regions": "The paper analyses this brain region: {label}.",
    "measures": "The paper measures: {label}.",
    "model_estimations": "The paper fits this statistical model: {label}.",
    "inference_settings": "The paper applies this statistical threshold: {label}.",
    "devices": "The paper used this scanner: {label}.",
    "preprocessings": "The paper preprocessed the data as follows: {label}.",
    "tables": "The paper contains this table: {label}.",
}


def describe(entity: dict, limit: int = 5) -> str:
    """A few of the entity's own field values, to say which thing is meant.

    A label alone is a thin thing to ask a checker about: "The paper fits this statistical
    model: group VBM t-tests" was scored unsupported for 22952599, whose methods say
    "t-tests with statistical parametric mapping (SPM5)" and "Total brain volume was treated
    as a confounding variable". The phrase was the extractor's, not the paper's, and judging
    the entity by it judged the wrong thing.
    """
    parts = []
    for slot, node in entity.items():
        if slot in ("local_id", "name", "_provenance") or len(parts) >= limit:
            continue
        value = text_of(node) if isinstance(node, dict) else None
        if value and not is_numeric(value):
            parts.append(f"{humanish(slot)} {value}")
    return "; ".join(parts)


def humanish(slot: str) -> str:
    return slot.replace("_", " ")


def existence_claims(record: dict, scope: set, premise: str) -> list[dict]:
    rows = []
    for cls, local in sorted(scope):
        tpl = EXISTENCE.get(cls)
        if not tpl:
            continue
        for e in entities(record, cls):
            if e.get("local_id") != local:
                continue
            claim = tpl.format(label=label_of(e, cls))
            about = describe(e)
            if about:
                claim = f"{claim} It is described as: {about}."
            rows.append({"kind": "existence", "entity": (cls, local), "path": f"{cls}/{local}",
                         "field": cls, "claim": claim,
                         "value": label_of(e, cls), "premise": premise,
                         "evidence_status": None})
    return rows


# --------------------------------------------------------------------------- main

#: Storage class -> the top-level record list its instances live in. Classes absent here
#: are only ever nested (Arm under `design.arms`, Condition under `tasks[].conditions`,
#: ModelTerm under `model_estimations[].terms`); those can be *referenced* because the id
#: index walks the whole record, but a missing one is reported rather than created, since
#: there is no unambiguous place to put it.
CLASS_TO_KEY = {
    "Analysis": "analyses", "Group": "groups", "Task": "tasks", "Region": "regions",
    "Measure": "measures", "Acquisition": "acquisitions", "Device": "devices",
    "Preprocessing": "preprocessings", "ModelEstimation": "model_estimations",
    "InferenceSettings": "inference_settings", "Table": "tables",
    "Assessment": "assessments",
}

#: The container key an instance sits under -> the class it is. Covers nested containers as
#: well as top-level lists, because a reference may legitimately target something nested --
#: `Cell.term` points at a ModelTerm under `model_estimations[].terms`.
KEY_TO_CLASS = {key: cls for cls, key in CLASS_TO_KEY.items()} | {
    "conditions": "Condition", "terms": "ModelTerm", "arms": "Arm",
    "timepoints": "Timepoint", "cells": "Cell", "levels": "FactorLevel",
    "seed_regions": "Region", "target_regions": "Region", "defines_regions": "Region",
}


def study_keys(schema) -> dict[str, str]:
    """Every top-level key of a record, mapped to the class it holds."""
    study = schema.classes.get("Study")
    out: dict[str, str] = {}
    for name, slot in (getattr(study, "attributes", None) or {}).items():
        rng = getattr(slot, "range", None)
        if rng in schema.classes:
            out[name] = rng
    return out


def unfillable(schema, class_name: str) -> set[str]:
    """Required slots of a class that a bare stub cannot supply.

    `resolve_refs` mints an entity when the model names one that does not exist. For a Region
    that costs `definition_method`, and for a ModelEstimation `model_family`, `stage` and
    `model_type` -- 123 required-attribute errors across 15 papers, because the reply that
    named the entity said nothing else about it. A reference to an entity the paper never
    describes is the dangling reference the validator exists to catch, so it is refused
    rather than satisfied with an invalid object.
    """
    cls = schema.classes.get(class_name)
    if cls is None:
        return set()
    # Identifier slots are excluded: `id` is `required` on almost every class but is
    # satisfied by `local_id`, and the validator never reports it missing. Counting it would
    # refuse every mint on every class, which is a different policy than the one intended.
    return {name for name, slot in (getattr(cls, "attributes", None) or {}).items()
            if getattr(slot, "required", False)
            and not getattr(slot, "identifier", False)
            and name not in ("local_id", "name")}


def declared_slots(schema, class_name: str) -> set[str]:
    """Every attribute the class actually declares.

    NuExtract is asked about an entity with a template built from one class, but nothing
    stopped an answer being written back onto a different one. On 23021615 it put
    `correction_scope` on all three analyses -- a slot that belongs to `InferenceSettings`,
    which those analyses already reference and which already carried the paper's own answer.
    The record then held a schema violation asserting `not_stated` next to a referenced
    entity saying `voxel level`.
    """
    cls = schema.classes.get(class_name)
    return set((getattr(cls, "attributes", None) or {})) if cls is not None else set()


def reference_slots(schema, class_name: str) -> dict[str, tuple[str, bool]]:
    """slot -> (target class, multivalued) for every foreign key on this class.

    The schema draws the line itself: a slot whose range is a class and whose `inlined` is
    False is a reference held by local_id, while `inlined: true` or unset is a nested
    object. Reading it from the schema rather than naming slots here means a new reference
    -- models to analyses, groups to assessments, tasks to acquisitions -- is linkable the
    day it is added, with no change to this file.
    """
    cls = schema.classes.get(class_name)
    if cls is None:
        return {}
    out: dict[str, tuple[str, bool]] = {}
    for name, slot in (getattr(cls, "attributes", None) or {}).items():
        if name in SKIP_REF_SLOTS:
            continue
        rng = getattr(slot, "range", None)
        if rng in schema.classes and getattr(slot, "inlined", None) is False:
            out[name] = (rng, bool(getattr(slot, "multivalued", False)))
    return out


#: `mirror_of` is minted by the sign-split stage, and `defines_regions` says where a
#: functional ROI came *from*; neither is something to ask a reader to name.
SKIP_REF_SLOTS = {"mirror_of", "defines_regions"}

#: key -> the slots that class declares, filled from the schema at start-up. Empty means the
#: schema did not load, and an empty set must not silently refuse every edit -- hence the
#: `or {field}` fallback at the one place it is read.
DECLARED: dict[str, set[str]] = {}

#: key -> required slots a minted stub could not fill, so minting is refused for that class.
REQUIRED: dict[str, set[str]] = {}


def id_index(record: dict) -> dict[str, tuple[str, str]]:
    """Every local_id in the record -> (normalized label, the class key it sits under).

    Walks the whole record, not just the top-level lists, so a nested Condition or ModelTerm
    can still be the target of a reference.
    """
    index: dict[str, tuple[str, str]] = {}

    def walk(node, key):
        if isinstance(node, dict):
            local = node.get("local_id")
            if isinstance(local, str):
                index[local] = (norm(label_of(node, key)), key)
            for k, v in node.items():
                walk(v, k if isinstance(v, list) else key)
        elif isinstance(node, list):
            for item in node:
                walk(item, key)

    walk(record, "")
    return index


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


#: What the evidence-repair pass asks for. `supporting_sentences` is a list because
#: grounding a value often takes two sentences and pondie almost always cites one -- the
#: scanner in one sentence, the sequence in the next.
EVIDENCE_TEMPLATE = {
    "fields": [{
        # A short tag, not the dotted path. Offered the path, the model answered with the
        # leaf ("modality" for "acquisitions[0].modality"), and with 49 contested fields many
        # leaves repeat -- three `description`s, two `name`s -- so the answer could not be
        # mapped back. A tag is unambiguous and short enough to echo exactly.
        "field_id": "string",
        "supporting_sentences": ["verbatim-string"],
        "value_is_wrong": "boolean",
        "corrected_value": "string",
    }]
}


def contested_block(rows: list[dict], scores: list[float],
                    threshold: float) -> tuple[str, list[dict]]:
    """The failing facts, written out for the model.

    Previously a failing fact contributed one thing to the repair: the *class* it belonged
    to. Its path, its value and the sentence that failed to support it were discarded, so
    nothing could be said about the field itself. This hands all three over.
    """
    failing = [r for r, p in zip(rows, scores) if p < threshold]
    if not failing:
        return "", []
    failing = failing[:40]
    for index, row in enumerate(failing, 1):
        row.setdefault("_tag", f"f{index}")
    lines = ["## Extracted values whose cited sentence does not support them", ""]
    for row in failing:
        lines.append(f"- [{row['_tag']}] {row['field']} of "
                     f"{row['path'].split('[')[0]} = {row['value'][:80]}")
        lines.append(f"    cited: {(row['premise'] or '(nothing cited)')[:170]}")
    lines += [
        "",
        "For each `field_id` above, quote the sentence or sentences from the paper that DO "
        "support the value -- copied character for character, and more than one where it "
        "takes more than one. If the paper states something different, set `value_is_wrong` "
        "and give `corrected_value`. If the paper does not state it, return no sentences.",
        "",
    ]
    return "\n".join(lines), failing


def bare(text: str) -> str:
    """Letters and digits only, lowercased -- for asking whether one string says what
    another says, without punctuation or spacing deciding the answer."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def carried_evidence(node, new: str) -> dict:
    """The evidence to attach to an edited value.

    Most edits here extend a value rather than replace it -- NuExtract restores the tail of a
    sentence the extractor truncated. Discarding the old evidence unconditionally, which is
    what this used to do, then marked the *better* value unsupported while its warrant was
    sitting on the field already. Observed on 23021615: `definition` went from a clause
    ending "showed reduced gray matter" to the paper's full sentence naming sgACC, caudate
    and hypothalamus, and lost its citation for being more nearly right.

    So the old spans are kept when they still contain the new value. They are not kept
    otherwise: a span that does not say the new thing is not a warrant for it.
    """
    evidence = (node or {}).get("evidence") or {}
    want = bare(new)
    for group in evidence.get("sets") or []:
        for span in group.get("spans") or []:
            if want and want in bare(span.get("text", "")):
                return evidence
    return {"status": "not_found"}


def introduced(before: dict, after: dict) -> list[str]:
    """Schema violations the repair added, as `count  message` lines.

    The loop ran for weeks putting 665 validation errors into 15 records -- an undeclared
    `correction_scope`, minted Regions with no `definition_method`, its own `_provenance` --
    and nothing noticed, because nothing validated what it wrote. Every paper is now checked
    against its own input, so a repair that damages the record says so on the line it happens.
    """
    from collections import Counter as _C
    from pondie.extraction.record.validate import Validator, EXTRACTION_SCHEMA
    from pondie.schema import reader as _rd

    schema = _rd.load(EXTRACTION_SCHEMA)

    def issues(record):
        v = Validator(schema, None)
        v.check_record(record)
        return _C(re.sub(r"\[\d+\]", "[]", m) for m in v.errors + v.warnings)

    delta = issues(after) - issues(before)
    return [f"{n}  {msg}" for msg, n in delta.most_common()]


def strip_provenance(node, path: str = "") -> dict:
    """Remove every `_provenance` key from the record, returning them keyed by path."""
    out: dict[str, object] = {}
    if isinstance(node, dict):
        if "_provenance" in node:
            out[path or "<root>"] = node.pop("_provenance")
        for key, value in node.items():
            out.update(strip_provenance(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            out.update(strip_provenance(value, f"{path}[{index}]"))
    return out


#: key -> {slot: declared range}, filled from the schema at start-up. Needed because the
#: model answers in strings and the schema does not: writing "true" into `Group.is_healthy`
#: produced `ExtractedBoolean.value must be a boolean, got str` on 28416565.
RANGES: dict[str, dict[str, str]] = {}

#: key -> {slot: whether it is multivalued}. A cast value still has to arrive in the shape
#: the slot declares: `Task.response_mode` takes a list, and writing the scalar produced
#: "ExtractedResponseModeList.value must be a list of ResponseMode or string, got str".
MULTI: dict[str, dict[str, bool]] = {}

#: enum name -> its permissible values, for slots whose range is a closed enum. A slot
#: written `any_of: [SomeEnum, string]` has no `range` and is deliberately open, so it is
#: absent here and free text stays legal for it.
ENUMS: dict[str, set[str]] = {}

TRUE = {"true", "yes", "y", "1"}
FALSE = {"false", "no", "n", "0"}


def typed(value: str, cls: str, slot: str):
    """The value cast to what its slot declares, or None when it will not cast.

    None means "do not write this": a boolean slot given "mostly" and an integer slot given
    "about 20" are answers the schema cannot hold, and coercing them by force is how a type
    error becomes a wrong number.
    """
    rng = (RANGES.get(cls) or {}).get(slot)
    text = str(value).strip()
    if rng == "boolean":
        low = text.lower()
        return True if low in TRUE else False if low in FALSE else None
    if rng == "integer":
        try:
            return int(float(text))
        except ValueError:
            return None
    if rng == "float":
        try:
            return float(text)
        except ValueError:
            return None
    # A closed enum takes only what it declares. The model answers in the paper's words, and
    # those are often right about the world and wrong for the slot: `prespecification` was
    # given "post-hoc" where the vocabulary is exploratory/preregistered, and `direction` --
    # the sign of an effect -- was given "read direction (head-foot direction)", which is a
    # phase-encoding axis. Neither is a value the field can hold, so neither is written.
    allowed = ENUMS.get(rng) if rng else None
    if allowed is not None and text not in allowed:
        return None
    return text


def like(old, new):
    """`new` in the type `old` already had, or `old` unchanged if it will not convert."""
    if isinstance(old, bool):
        low = str(new).strip().lower()
        return True if low in TRUE else False if low in FALSE else old
    for kind in (int, float):
        if isinstance(old, kind) and not isinstance(old, bool):
            try:
                return kind(float(str(new).strip()))
            except ValueError:
                return old
    return new


def would_shorten(node, correction) -> bool:
    """Whether replacing this field's value with `correction` drops values it already holds."""
    current = (node or {}).get("value")
    return isinstance(current, list) and len(current) > 1 and not isinstance(correction, list)


def shaped(value, cls: str, slot: str):
    """The cast value in the shape its slot declares, or None if it will not cast."""
    cast = typed(value, cls, slot)
    if cast is None:
        return None
    if (MULTI.get(cls) or {}).get(slot) and not isinstance(cast, list):
        return [cast]
    return cast


def wrap(value, text: str) -> dict:
    """An `ExtractedValue` for a value the model reported, cited where the paper says it."""
    quote = str(value)
    return {"extraction_status": "extracted", "value": value, "value_source": "reported",
            "evidence": write_evidence(text, [quote]) or {"status": "not_found"}}


def write_evidence(text: str, sentences: list) -> dict | None:
    """Resolve quotes to offsets and build an EvidenceSet, or None if none resolve.

    Offsets come from `pondie.extraction.record.spans`, which matches exactly then falls
    back to a length-preserving character fold and sets `text` to the substring it found
    rather than to what the model produced. `verify` then asserts the schema invariant, so
    a span that survives is one the document actually contains.
    """
    from pondie.extraction.record import spans as span_tools

    resolved = []
    for sentence in sentences or []:
        quote = re.sub(r"\s+", " ", str(sentence)).strip()
        if len(quote) < 20:
            continue
        try:
            # `ResolvedSpan.as_record()` already emits the schema shape, with `text` set to
            # the substring found rather than to what the model produced -- so a
            # whitespace-tolerant match can never yield a span whose text disagrees.
            span = span_tools.resolve(text, quote).as_record()
            span_tools.verify(text, span)
        except Exception:
            continue
        resolved.append(span)
    # Labelled so a reader can tell a repaired span from one the extraction model quoted
    # or the cross-encoder retrieved.
    return ({"status": "present",
             "sets": [{"source": "repair_pass", "spans": resolved}]} if resolved else None)


def node_at(record: dict, entity: tuple[str, str], path: str):
    """The wrapper a fact claim came from, or None.

    Paths from `fact_claims` are relative to the entity, because the walk starts at the
    entity's own dict -- `modality`, not `acquisitions[0].modality`. Resolving them from the
    record root returned None for every field, so no evidence proposal ever survived and the
    pass reported "nothing to repair" while the model was answering correctly.
    """
    cls, local = entity
    node = next((e for e in (record.get(cls) or [])
                 if isinstance(e, dict) and e.get("local_id") == local), None)
    if node is None:
        return None
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            index = int(part[1:-1])
            if not isinstance(node, list) or index >= len(node):
                return None
            node = node[index]
        else:
            if not isinstance(node, dict):
                return None
            node = node.get(part)
        if node is None:
            return None
    return node if isinstance(node, dict) and "extraction_status" in node else None


#: The abbreviation table of the paper currently being repaired. Module-level because
#: `resolve_refs` is reached from several call sites that do not carry the text.
ABBREV = None


def resolve_refs(names: list, cls: str, record: dict,
                 target_class: str | None = None) -> tuple[list[str], list[str]]:
    """Names the model gave -> local_ids, restricted to the class the slot declares.

    Matching is on the label because a name is what the paper prints and a local_id is not.
    It is *also* on the class: without that check, `ModelEstimation.inputs_from` -- which
    the schema says targets another ModelEstimation -- resolved to `cond_begin_smoking`, a
    Condition nested under a Task, purely because the name matched. A type-violating link is
    worse than a missing one, so a candidate of the wrong class is refused and reported.
    """
    out: list[str] = []
    refused: list[str] = []
    index = id_index(record)

    def right_class(lid: str, key: str) -> bool:
        if target_class is None:
            return True
        return KEY_TO_CLASS.get(key) == target_class

    for raw in names or []:
        want = norm(raw)
        if not want:
            continue
        # an id the model echoed back, or a name that matches something anywhere in the
        # record -- nested targets included, provided the class is the declared one
        if raw in index:
            (out if right_class(raw, index[raw][1]) else refused).append(raw)
            continue
        found = next((lid for lid, (lab, key) in index.items()
                      if lab == want and right_class(lid, key)), None)
        if found:
            out.append(found)
            continue
        if any(lab == want for lab, _k in index.values()):
            refused.append(str(raw))     # exists, but is the wrong class for this slot
            continue
        if cls not in set(CLASS_TO_KEY.values()):
            refused.append(str(raw))   # nested-only class: referenceable, not creatable
            continue
        hit = next((e for e in entities(record, cls)
                    if norm(label_of(e, cls)) == want), None)
        # The same instrument under a second surface form is the same instrument. Exact
        # normalized equality alone let "clinician-administered PTSD scale (CAPS)" mint a
        # second copy of `asm_caps` ("CAPS total score"), and analyses then linked to the
        # copy. Acronyms are checked here as well as in the sweep because this is the path
        # that actually created it.
        if hit is None:
            hit = next((e for e in entities(record, cls)
                        if same_entity(label_of(e, cls), str(raw), ABBREV)), None)
        if hit is None:
            if "name" not in (DECLARED.get(cls) or {"name"}) or REQUIRED.get(cls):
                refused.append(str(raw))
                continue
            hit = {"local_id": mint_id(record, cls, str(raw), 0),
                   "name": {"extraction_status": "extracted", "value": str(raw),
                            "value_source": "reported",
                            "evidence": {"status": "not_found"}},
                   "_provenance": {"source": "nuextract3", "created_for": "reference"}}
            record.setdefault(cls, []).append(hit)
        out.append(hit["local_id"])
    return out, refused


def apply_edit(target: dict, item: dict, cls: str, record: dict, iteration: int) -> list[str]:
    """Correct an existing entity in place, and write the references the model named.

    Only slots the model actually returned are touched, and the old value is kept in
    `_provenance.edits` so an edit is reversible and auditable. References are written as
    local_ids, which is what makes `Analysis.regions` stop being empty.
    """
    changed: list[str] = []
    prov = target.setdefault("_provenance", {})
    edits = prov.setdefault("edits", [])

    for slot, (target_class, multi) in (REF_SLOTS.get(cls) or {}).items():
        named = item.get(slot)
        if not named:
            continue
        key = CLASS_TO_KEY.get(target_class, "")
        refs, refused = resolve_refs(named if isinstance(named, list) else [named],
                                     key, record, target_class)
        if refused:
            prov.setdefault("refused_refs", []).append(
                {"slot": slot, "expected": target_class, "names": refused})
        # Nothing points at itself. On 27082610 `inputs_from` resolved to the very model
        # being edited, three times over -- "a model fitted on its own output" -- because the
        # label the reply gave matched the entity whose slot it was filling.
        # Repointing an analysis at a different model orphans the terms its cells name. On
        # 19942229 `a_793_1` was moved to `mod_ns_s` while its two cells still named
        # `trm_group_r_nr`, a term of `mod_r_nr` -- "which a_793_1's model does not reach".
        # The cells are the analysis's own structure and the model reference is a pointer, so
        # the pointer is what gives way.
        if slot == "model_estimation" and refs and orphans_cell_terms(target, refs[0], record):
            prov.setdefault("refused_refs", []).append(
                {"slot": slot, "expected": target_class, "names": refs,
                 "why": "would orphan the terms this analysis's cells name"})
            continue
        own = target.get("local_id")
        if own and own in refs:
            prov.setdefault("refused_refs", []).append(
                {"slot": slot, "expected": target_class, "names": [own],
                 "why": "self-reference"})
            refs = [r for r in refs if r != own]
        if not refs:
            continue
        # "Whole-brain and searchlight analyses restrict inference to no region" is a rule the
        # schema states and the loop broke 30 times in 15 papers, by linking regions onto
        # analyses whose own spatial_scope says the search covered the brain.
        # Same contradiction on the correction slots: `correction_regions` names what
        # `correction_scope: roi` restricts to, so naming regions beside a whole-brain scope
        # asserts two incompatible things about one procedure. Seen on 11950456, an STG
        # volumetric study written as whole_brain with correction_regions [r_stg].
        scope_slot = "correction_scope" if slot == "correction_regions" else "spatial_scope"
        if slot in ("regions", "correction_regions") and text_of(
                target.get(scope_slot)) in ("whole_brain", "whole brain", "searchlight"):
            prov.setdefault("refused_refs", []).append(
                {"slot": slot, "expected": target_class, "names": refs,
                 "why": f"{scope_slot} is not roi"})
            continue
        if multi:
            # Union, not replacement. On 12853571 the model returned four assessments for
            # `an_caps_correlation` and the write dropped `asm_caps` -- the CAPS total score,
            # which is the one thing that correlation is of. A model naming more entities is
            # not a model saying the existing link was wrong.
            value = list(target.get(slot) or [])
            for r in refs:                      # `refs` itself repeats: four preprocessing
                if r not in value:              # names all resolving to `prp_vbm` wrote it
                    value.append(r)             # four times into one slot
        else:
            value = refs[0]
        if value != target.get(slot):
            edits.append({"slot": slot, "was": target.get(slot), "now": value,
                          "iteration": iteration})
            target[slot] = value
            changed.append(slot)

    for field in ("spatial_scope", "correction_scope", "definition"):
        new = item.get(field)
        if not new or not isinstance(new, str):
            continue
        if field not in (DECLARED.get(cls) or {field}):
            prov.setdefault("refused_slots", []).append({"slot": field, "on": cls})
            continue
        node = target.get(field)
        old = text_of(node) if node is not None else None
        if old and old.strip().lower() == new.strip().lower():
            continue
        # Truncation is not correction. The same path that restored the tail of a sentence on
        # 23021615 cut "compared to traumatized controls." down to "compared to traumatized"
        # on 22952599, and dropped the evidence with it. An edit has to add something.
        if old and bare(new) and bare(new) in bare(old):
            prov.setdefault("refused_edits", []).append(
                {"slot": field, "was": old, "now": new, "why": "truncation"})
            continue
        # The paired-slot rule, in the other direction. `apply_edit` writes reference slots
        # first and scalars second, so on 11950456 `correction_regions` was linked while
        # `correction_scope` was still None -- passing the guard on that side -- and the scope
        # was then set to whole_brain, recreating exactly the contradiction the guard exists
        # to prevent. The regions carry a warrant and the scope does not, so the scope loses.
        pair = {"correction_scope": "correction_regions", "spatial_scope": "regions"}
        if field in pair and new.strip().lower() in ("whole_brain", "whole brain",
                                                     "searchlight") \
                and target.get(pair[field]):
            prov.setdefault("refused_edits", []).append(
                {"slot": field, "was": old, "now": new,
                 "why": f"{pair[field]} is not empty"})
            continue
        # And the same pair the other way. `roi` beside an empty region list asserts that
        # inference was restricted while naming no restriction -- the ambiguity the rule
        # exists to separate from a genuine whole-brain search. Reference slots are written
        # before scalars, so an empty list here is final for this pass, not merely not-yet.
        # Seen on 19996042 and 16038682.
        if field in pair and new.strip().lower() in ("roi", "region of interest") \
                and not target.get(pair[field]):
            prov.setdefault("refused_edits", []).append(
                {"slot": field, "was": old, "now": new,
                 "why": f"{pair[field]} is empty, so the restriction is unnamed"})
            continue
        # An edit has to keep its warrant or say more. `correction_scope` on 12853571 went
        # from "whole volume analyzed and a priori small volumes" -- cited, and true, the
        # paper did both -- to the bare enum "whole_brain", which drops the small-volume half
        # and has no sentence behind it. Coercion to a permissible value is not a correction.
        carried = carried_evidence(node, new)
        if (node or {}).get("evidence", {}).get("status") == "present" \
                and carried.get("status") != "present":
            prov.setdefault("refused_edits", []).append(
                {"slot": field, "was": old, "now": new, "why": "would lose the warrant"})
            continue
        edits.append({"slot": field, "was": old, "now": new, "iteration": iteration})
        target[field] = {"extraction_status": "extracted", "value": new,
                         "value_source": "reported", "evidence": carried}
        changed.append(field)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmids", nargs="+", required=True)
    ap.add_argument("--records", type=Path,
                    default=EXP / "pondie-data/runs/pondie-907/records")
    ap.add_argument("--corpus", type=Path, default=EXP / "corpus")
    ap.add_argument("--out", type=Path, default=EXP / "reports" / "repair")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="support below this marks a fact or entity suspect")
    ap.add_argument("--shards", type=int, default=None,
                    help="how many processes share this corpus; default is as many as the "
                         "cards allow, given that the two models need a card each")
    ap.add_argument("--shard", type=int, default=0, help="which shard this process is")
    ap.add_argument("--checker-device", default=None,
                    help="overrides the plan; set before the checker loads, because it "
                         "reads the visible devices rather than taking a device argument")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit a model to sit on the CPU; by default that is an error, "
                         "because `device_map='auto'` spills there silently and a 4B model "
                         "decoding a 20k-token prompt on CPU does not finish")
    ap.add_argument("--nuextract-device", type=int, default=None,
                    help="overrides the plan")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-premise-chars", type=int, default=45000)
    ap.add_argument("--prune", action="store_true", default=True,
                    help="terminal iteration removes entities the paper does not support")
    ap.add_argument("--no-prune", dest="prune", action="store_false")
    ap.add_argument("--prune-floor", type=float, default=0.5,
                    help="skip pruning entirely when this fraction of entities fails to "
                         "score as supported: the checker, not the record, is the outlier")
    ap.add_argument("--no-evidence-repair", action="store_true")
    ap.add_argument("--no-repair", action="store_true",
                    help="measure only; skip the NuExtract pass")
    ap.add_argument("--resume", action="store_true",
                    help="skip pmids that already have a .repaired.json in --out")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    # Placement first, and before either model is imported: MiniCheck takes no device
    # argument -- it loads with `device_map="auto"` and reads the visible devices, which is
    # why its own logs say to set the CUDA device before constructing it. `--checker-device`
    # was accepted and ignored for an entire run.
    shards = args.shards if args.shards is not None else default_shards()
    visible, checker_device, proposer_device = device_plan(args.shard, shards)
    if visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    if args.checker_device:
        checker_device = args.checker_device
    if args.nuextract_device is not None:
        proposer_device = args.nuextract_device
    args.checker_device, args.nuextract_device = checker_device, proposer_device
    if shards > 1:
        args.pmids = split_evenly(args.pmids, args.corpus, shards, args.shard)
    print(f"  shard {args.shard + 1}/{shards}: {len(args.pmids)} papers, "
          f"cards {visible or 'none'}, checker on {checker_device}, "
          f"proposer on cuda:{proposer_device}", flush=True)
    from minicheck.minicheck import MiniCheck
    # batch_size 16 against a Methods+Results premise exhausts an 8 GB card; MiniCheck
    # chunks the document per claim, so the batch is what has to shrink.
    checker = MiniCheck(model_name="flan-t5-large", enable_prefix_caching=False,
                        batch_size=args.batch_size,
                        cache_dir="/home/james/.cache/huggingface")
    assert_on_gpu(checker, "grounding checker", args.allow_cpu)

    def check(docs: list[str], claims: list[str]) -> list[float]:
        """Score in slices, freeing between them, so one long premise cannot OOM the run."""
        out: list[float] = []
        for i in range(0, len(claims), args.batch_size):
            _, probs, _, _ = checker.score(docs=docs[i:i + args.batch_size],
                                           claims=claims[i:i + args.batch_size])
            out.extend(probs)
            torch.cuda.empty_cache()
        return out

    nu = None
    if not args.no_repair:
        from nuextract_recall import (ANALYSIS_INSTRUCTION, ANALYSIS_TEMPLATE, CLASSES,
                                      template_for)
        from transformers import (AutoModelForImageTextToText, AutoProcessor,
                                  BitsAndBytesConfig)
        from pondie import schema as psch
        from pondie.schema import reader
        sch = reader.load(psch.STORAGE)
        proc = AutoProcessor.from_pretrained("numind/NuExtract3", trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            "numind/NuExtract3", trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            device_map={"": args.nuextract_device}).eval()
        assert_on_gpu(model, "entity proposer", args.allow_cpu)

        def nu_call(text: str, template: dict, instruction: str) -> list:
            # A 4B model in 4-bit leaves roughly 4.9 GB for the KV cache, and a 96k-character
            # paper overruns it inside `torch_chunk_gated_delta_rule`. Capping the premise
            # and catching the OOM keeps one long paper from killing a whole shard.
            # Halve and retry rather than drop. A skipped call loses the whole sweep for
            # that class: 20673548 lost all ten of its NuExtract calls and got no entity pass
            # at all, because `sections` found no methods or results and fell back to the
            # whole 51,789-character paper. A truncated premise loses the tail; a dropped one
            # loses everything, and 14% of papers have slices over 30,000 characters, so a
            # fixed cap low enough to always fit would truncate them all unnecessarily.
            limit, ids, inp = args.max_premise_chars, None, None
            while ids is None:
                body = text[:limit]
                msgs = [{"role": "user", "content": [{"type": "text",
                                                      "text": instruction + body}]}]
                inp = proc.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                    return_tensors="pt", template=json.dumps(template, indent=2),
                    enable_thinking=False).to(model.device)
                try:
                    with torch.inference_mode():
                        ids = model.generate(**inp, max_new_tokens=2048, do_sample=False)
                except torch.OutOfMemoryError:
                    del inp
                    torch.cuda.empty_cache()
                    if limit <= 6000:
                        print(f"      nuextract OOM at {limit} chars, the floor; skipped",
                              flush=True)
                        return []
                    limit //= 2
                    print(f"      nuextract OOM; retrying at {limit} chars", flush=True)
            raw = proc.batch_decode(ids[:, inp.input_ids.shape[1]:],
                                    skip_special_tokens=True)[0].strip()
            torch.cuda.empty_cache()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return []
            key = next(iter(parsed)) if isinstance(parsed, dict) else None
            got = parsed.get(key) if key else None
            return [g for g in (got or []) if isinstance(g, dict)]
        nu = (nu_call, sch, ANALYSIS_TEMPLATE, ANALYSIS_INSTRUCTION, CLASSES, template_for)

    def score(rows: list[dict], fallback: str) -> list[float]:
        if not rows:
            return []
        docs = [r["premise"] or fallback for r in rows]
        return check(docs, [r["claim"] for r in rows])

    global REF_SLOTS, DECLARED, REQUIRED, RANGES, ENUMS, REF_SLOT_NAMES, MULTI
    REF_SLOTS, DECLARED, REQUIRED, RANGES, ENUMS, MULTI = {}, {}, {}, {}, {}, {}
    REF_SLOT_NAMES = set()
    glosses: dict[str, str] = {}
    try:
        from pondie import schema as _ps
        from pondie.schema import reader as _rd
        _sch = _rd.load(_ps.STORAGE)
        glosses = enum_glosses(_sch)
        from nuextract_recall import CLASSES as _CLS
        REF_SLOTS = {key: reference_slots(_sch, cname) for key, cname in _CLS.items()}
        # Every key on Study, not only the eight NuExtract sweeps: `tables`,
        # `preprocessings` and `devices` are minted by the reference resolver, and building
        # this from the sweep list alone let them through the fallback and wrote a `name`
        # onto three classes that do not declare one.
        DECLARED = {key: declared_slots(_sch, cname)
                    for key, cname in study_keys(_sch).items()}
        DECLARED.update({key: declared_slots(_sch, cname) for key, cname in _CLS.items()})
        REQUIRED = {key: unfillable(_sch, cname)
                    for key, cname in {**study_keys(_sch), **_CLS}.items()}
        REF_SLOT_NAMES = {name for cname in _sch.classes
                          for name, _slot, kind in _sch.iter_slots(cname)
                          if kind == "reference"}
        ENUMS = {name: set(getattr(e, "permissible_values", {}) or {})
                 for name, e in _sch.enums.items()}
        MULTI = {key: {n: bool(getattr(sl, "multivalued", False))
                       for n, sl in (getattr(_sch.classes.get(cname), "attributes", None)
                                     or {}).items()}
                 for key, cname in {**study_keys(_sch), **_CLS}.items()}
        RANGES = {key: {n: getattr(sl, "range", None)
                        for n, sl in (getattr(_sch.classes.get(cname), "attributes", None)
                                      or {}).items()}
                  for key, cname in {**study_keys(_sch), **_CLS}.items()}
        print(f"  minting refused for: "
              f"{sorted(k for k, v in REQUIRED.items() if v)}")
        print(f"enum glosses: {len(glosses)} | reference slots: "
              f"{ {k: list(v) for k, v in REF_SLOTS.items() if v} }")
    except Exception as exc:
        print(f"  warn: no enum glosses ({exc})")

    report = []
    for pmid in args.pmids:
        # A 26-paper shard runs ~1.7 h on one card; without this, a crash at paper 24 costs
        # the whole shard. Off by default so a code change re-runs everything, which is what
        # you want when the thing that changed is what the loop writes.
        if args.resume and (args.out / f"{pmid}.repaired.json").is_file():
            print(f"  [{pmid}] already repaired, skipped", flush=True)
            continue
        text = (args.corpus / pmid / "processed/local/text.tables.txt").read_text(errors="replace")
        method_results = sections(text)
        global ABBREV
        ABBREV = paper_abbreviations(text)
        record = json.loads((args.records / f"{pmid}.extraction.json").read_text())
        parsed_foci = stage1_analyses(args.corpus, pmid)
        history = []
        rejected, edited, linked = Counter(), Counter(), Counter()
        evidence_stats: Counter = Counter()

        for it in range(args.iterations + 1):
            scope, orphans = reachable(record)
            facts = fact_claims(record, scope, glosses)
            exists = existence_claims(record, scope, method_results)
            fs = score(facts, text)
            es = score(exists, method_results)
            prose = [(r, p) for r, p in zip(facts, fs) if not is_numeric(r["value"])]
            # Numerics never reach the checker: they are settled by scaling against their own
            # cited span, and one that does not settle is reported as possibly-derived
            # rather than as unsupported.
            numeric = []
            for row in facts:
                if not is_numeric(row["value"]):
                    continue
                ok, factor = ground_numeric(row["value"], row["premise"])
                numeric.append((row, ok, factor))
            grounded_n = sum(1 for _r, ok, _f in numeric if ok)
            supported_f = sum(1 for p in fs if p >= args.threshold) 
            supported_p = sum(1 for _, p in prose if p >= args.threshold)
            supported_e = sum(1 for p in es if p >= args.threshold)
            snap = {
                "iteration": it, "pmid": pmid,
                "n_analyses": len(entities(record, "analyses")),
                "in_scope_entities": len(scope), "orphan_entities": len(orphans),
                "facts": len(facts), "facts_supported": supported_f,
                "fact_precision": round(supported_f / len(facts), 4) if facts else 0.0,
                "prose_facts": len(prose), "prose_supported": supported_p,
                "prose_precision": round(supported_p / len(prose), 4) if prose else 0.0,
                "numeric_facts": len(numeric),
                "numeric_grounded": grounded_n,
                "fields_missing_evidence": sum(
                    1 for r in facts if r["evidence_status"] == "not_found"),
                "numeric_ungrounded": len(numeric) - grounded_n,
                "parsed_coord_tables": len(parsed_foci),
                "analyses_with_foci": sum(
                    1 for a in entities(record, "analyses")
                    if (a.get("_provenance") or {}).get("n_foci")
                    or a.get("tables") or a.get("source_table_analysis")),
                "rejected_no_foci": rejected["analysis_without_foci"],
                "evidence_replaced": sum(v for k, v in evidence_stats.items()
                                         if k.startswith("replaced")),
                "evidence_rejected": evidence_stats["rejected: no better"],
                "pruned": 0,
                "entities_checked": len(exists), "entities_supported": supported_e,
                "entity_precision": round(supported_e / len(exists), 4) if exists else 0.0,
            }
            history.append(snap)
            print(f"  [{pmid} iter {it}] analyses={snap['n_analyses']} "
                  f"scope={snap['in_scope_entities']} orphan={snap['orphan_entities']} | "
                  f"prose {supported_p}/{len(prose)} ({snap['prose_precision']:.2f}) | "
                  f"numeric {grounded_n}/{len(numeric)} grounded | "
                  f"entities {supported_e}/{len(exists)} ({snap['entity_precision']:.2f})",
                  flush=True)

            if it == args.iterations:
                # The terminal pass adds nothing and removes what the paper does not
                # support. Pruning last matters: an entity can only be judged once no
                # further sweep is going to justify it, and an analysis pruned earlier
                # would drop everything downstream of it from scope.
                judgeable = {r["entity"] for r in exists
                             if r["value"] and not r["value"].startswith(("acq_", "inf_",
                                 "mea_", "mod_", "grp_", "reg_", "tsk_", "dev_", "prep_"))}
                # A reference is evidence the entity matters, and removing its target
                # leaves a dangling pointer the validator then reports. On 22952599 the
                # terminal pass deleted `mod_vbm_group` -- t-tests in SPM5 with total brain
                # volume as a confound, stated verbatim in the paper -- because MiniCheck did
                # not entail its label "group VBM t-tests", and took its ModelTerms with it.
                # A checker that cannot validate most of a well-formed record is not
                # measuring existence on this document, and deleting on its say-so is
                # unjustified. 16038682 scored 5 of 13 entities supported and the pass then
                # removed both of its analyses -- "VBM Analyses" and "Volumetric Analyses",
                # named in the paper's own abstract -- leaving the record with none at all.
                rate = supported_e / len(exists) if exists else 1.0
                if rate < args.prune_floor:
                    print(f"      prune skipped: only {supported_e}/{len(exists)} entities "
                          f"scored supported ({rate:.2f}), below {args.prune_floor}", flush=True)
                    break
                pointed_at = referenced(record)
                doomed = {r["entity"] for r, p in zip(exists, es)
                          if p < args.threshold and r["entity"] in judgeable
                          and r["entity"][1] not in pointed_at}
                if args.prune and doomed:
                    for cls, local in doomed:
                        record[cls] = [e for e in entities(record, cls)
                                       if e.get("local_id") != local]
                    snap["pruned"] = len(doomed)
                    print(f"      pruned {len(doomed)} unsupported: "
                          f"{sorted(f'{c}/{l}' for c, l in doomed)[:6]}", flush=True)
                break
            if nu is None:
                break

            # which classes need a sweep: analyses always (a missing analysis has no field
            # to look suspect), plus any class holding a suspect entity or suspect fact
            # --- evidence repair: hand the failing facts back with their paths, ask for
            # the sentences that do support them, and keep a replacement only if it scores
            # better than what it replaces. Without the re-score the pass could make
            # evidence worse while reporting that it repaired it.
            # Contest the failing prose claims, plus the numerics arithmetic could not
            # ground. A grounded numeric is settled and must not be sent for repair: doing
            # so is what had the model "correcting" 0.0097 to 0.0097 because the paper
            # printed 9.7 ms.
            # Three things get contested. A prose claim its own cited span does not support;
            # a numeric arithmetic could not ground; and -- whatever it scored -- any field
            # the record marks `not_found`, because that status says no sentence was ever
            # located for it. Those were previously scored against the whole paper as a
            # fallback and so were only contested when that failed, which left 96 fields
            # across five papers with missing evidence that nothing asked about.
            missing = [r for r, _ in prose if r["evidence_status"] == "not_found"
                       and not r["premise"]]
            missing += [r for r, ok, _f in numeric
                        if r["evidence_status"] == "not_found" and not r["premise"] and ok]
            failing_rows = [r for r, p in zip([x for x, _ in prose], [p for _, p in prose])
                            if p < args.threshold and r not in missing]
            failing_rows += [r for r, ok, _f in numeric if not ok and r not in missing]
            block, contested = contested_block(
                failing_rows + missing,
                [0.0] * (len(failing_rows) + len(missing)),
                args.threshold)
            if contested and not args.no_evidence_repair:
                # Batched. Asked about all 49 contested fields at once the reply ran past
                # `max_new_tokens` and came back as truncated JSON, which parses to nothing
                # and reports as "no proposals" -- a silent failure that looked like the
                # model declining to answer. Twelve at a time keeps both the enum and the
                # reply short.
                got = []
                for start in range(0, len(contested), 12):
                    chunk = contested[start:start + 12]
                    sub, _ = contested_block(chunk, [0.0] * len(chunk), 1.0)
                    tpl = json.loads(json.dumps(EVIDENCE_TEMPLATE))
                    tpl["fields"] = [{**tpl["fields"][0],
                                      "field_id": [r["_tag"] for r in chunk]}]
                    part = nu_call(method_results, tpl,
                                   "Find the sentences in this paper that support each "
                                   "extracted value listed below.\n\n" + sub)
                    if not part:
                        evidence_stats["no reply"] += 1
                    got.extend(part)
                by_tag = {r["_tag"]: r for r in contested}
                proposals = []
                for item in got:
                    row = by_tag.get(str(item.get("field_id") or "").strip())
                    node = node_at(record, row["entity"], row["path"]) if row else None
                    if node is None:
                        continue
                    fresh = write_evidence(text, item.get("supporting_sentences"))
                    if fresh is None:
                        continue
                    premise = " ".join(sp["text"] for sp in fresh["sets"][0]["spans"])
                    proposals.append((row, node, fresh, premise, item))
                if proposals:
                    after = check([p for *_x, p, _i in proposals],
                                  [r["claim"] for r, *_ in proposals])
                    # The bar a replacement has to clear. A field with no span has no
                    # evidence to be better than, so its floor is 0 -- scoring it against
                    # the whole paper, as the fallback did, would set a bar the cited
                    # sentence never had to meet.
                    before = {r["path"]: (0.0 if not r["premise"] else p)
                              for r, p in zip([x for x, _ in prose],
                                              [p for _, p in prose])}
                    for (row, node, fresh, _pr, item), new in zip(proposals, after):
                        old = before.get(row["path"], 0.0)
                        if new <= old:
                            evidence_stats["rejected: no better"] += 1
                            continue
                        node["evidence"] = fresh
                        n_spans = len(fresh["sets"][0]["spans"])
                        evidence_stats[f"replaced ({n_spans} span"
                                       f"{'s' if n_spans > 1 else ''})"] += 1
                        evidence_stats["_gain"] += new - old
                        if item.get("value_is_wrong") and item.get("corrected_value"):
                            # A list stays a list. Writing the scalar straight through turned
                            # `echo_time_seconds` [0.0044, 0.005] into '0.0044' -- a lost
                            # value and a type the schema does not allow -- and flattened
                            # `medical_condition` the same way.
                            old_value = node.get("value")
                            # Shortening a list is not correcting it. 16701903 acquires two
                            # sequences -- "3D MP-RAGE: echo time [TE]=4.4 ms" and "FLASH
                            # sequence: TE = 5 ms" -- and `echo_time_seconds` held both.
                            # Casting the reply into `[value]` fixed the type error the first
                            # pass made and dropped the FLASH echo time doing it, which is
                            # the same failure as a truncated definition wearing a different
                            # shape.
                            if isinstance(old_value, list) and len(old_value) > 1:
                                evidence_stats["value correction would shorten a list"] += 1
                                continue
                            if isinstance(old_value, list):
                                # And typed like the list it replaces. Wrapping the string
                                # straight into `echo_time_seconds` produced
                                # `ExtractedNumberList.value must be a float, got str` on
                                # 16701903 -- a fix for one type error that made another.
                                new_value = item["corrected_value"]
                                if old_value and isinstance(old_value[0], (int, float)):
                                    try:
                                        new_value = float(str(new_value).strip())
                                    except ValueError:
                                        evidence_stats["value correction not numeric"] += 1
                                        continue
                                node["value"] = [new_value]
                            else:
                                # A correction does not change the type. `analyses[].groups[].n`
                                # took the string "38" on 20673548 -- the cast on the creation
                                # path never reached here, because this writes into a node
                                # resolved by path and knows no class or slot. The old value's
                                # own type is enough: a corrected count is still a count.
                                node["value"] = like(old_value, item["corrected_value"])
                            evidence_stats["value corrected"] += 1
                print(f"      evidence: {dict(evidence_stats) or '{}'}", flush=True)

            suspect_e = {r["entity"][0] for r, p in zip(exists, es) if p < args.threshold}
            suspect_f = {r["entity"][0] for r, p in zip(prose, [p for _, p in prose])
                         if False} | {r["entity"][0] for r, p in prose if p < args.threshold}
            # An ROI analysis with no regions is the defect this loop exists to fix, and
            # the region has to be *swept* to be fixable: `resolve_refs` will not mint one
            # from a bare name because `definition_method` is required and a name does not
            # supply it, so "sgACC (BA25)" named in `analyses.regions` was refused outright.
            # The sweep asks for the whole entity, `definition_method` included.
            if any("roi" in text_of(a.get("spatial_scope")).lower() and not a.get("regions")
                   for a in entities(record, "analyses")):
                suspect_e = suspect_e | {"regions"}
            todo = sweep_order(["analyses"] + sorted((suspect_e | suspect_f) - {"analyses"}))
            nu_call, sch, A_TPL, A_INS, CLS, tpl_for = nu
            added = Counter()
            for cls in todo:
                if cls not in CLS and cls != "analyses":
                    continue
                template = A_TPL if cls == "analyses" else tpl_for(sch, CLS[cls], cls)
                template = json.loads(json.dumps(template))     # don't mutate the shared one
                fields = template[cls][0]
                fields.setdefault("local_id", "string")
                # Every foreign key the schema declares for this class becomes a slot the
                # model can fill by naming the target. Generic on purpose: models to
                # analyses, groups to assessments, tasks to acquisitions all arrive here
                # without this file naming any of them.
                for slot, (_target, multi) in (REF_SLOTS.get(cls) or {}).items():
                    fields[slot] = ["verbatim-string"] if multi else "verbatim-string"
                refs = ", ".join((REF_SLOTS.get(cls) or {}))
                link_note = (f"\n\nWhere this {SINGULAR.get(cls, cls)} uses another entity, "
                             f"name it in the matching field ({refs}). Use the exact name as "
                             f"listed above where one is listed.\n" if refs else "")
                instruction = A_INS if cls == "analyses" else (
                    f"List every {cls[:-1].replace('_', ' ')} in this paper that is used by, "
                    f"or reported for, one of its statistical analyses. Ignore anything not "
                    f"tied to an analysis.\n\n")
                instruction = instruction + link_note + SHARED_RULES
                claimed = claimed_table_ids(record)
                context = entity_block(record, cls)
                if cls == "analyses":
                    # The candidate block is appended once, below, for every class now that
                    # it is per-class; adding it here too sent analyses two copies.
                    context = foci_block(parsed_foci, claimed) + context
                    template = dict(template)
                    ids = [e["_rg_id"] for e in parsed_foci]
                    if ids:
                        template["analyses"] = [{**template["analyses"][0],
                                                 "source_row_group": ids}]
                found = nu_call(method_results, template,
                                instruction + context + reference_block(record, cls))
                have = {re.sub(r"[^a-z0-9]+", " ", label_of(e, cls).lower()).strip()
                        for e in entities(record, cls)}

                # every parsed table stays on offer: an omnibus and its post-hoc contrast
                # are two analyses over one set of rows, so binding is not exclusive
                unclaimed = list(parsed_foci)
                by_id = {e.get("local_id"): e for e in entities(record, cls)}
                for item in found:
                    lab = str(item.get("name") or item.get("definition") or "").strip()
                    norm = re.sub(r"[^a-z0-9]+", " ", lab.lower()).strip()
                    # an edit: the model reused an id we gave it
                    target = by_id.get(str(item.get("local_id") or "").strip())
                    if target is not None:
                        changed = apply_edit(target, item, cls, record, it)
                        edited[cls] += bool(changed)
                        linked[cls] += sum(1 for c in changed
                                           if c in (REF_SLOTS.get(cls) or {}))
                        continue
                    if not norm or norm in have:
                        continue
                    # "clinician-administered PTSD scale (CAPS)" and "CAPS total score" are
                    # one instrument spelled two ways; normalized-equality alone minted a
                    # second copy and then linked analyses to the copy.
                    if any(same_entity(label_of(e, cls), lab, ABBREV)
                           for e in entities(record, cls)):
                        rejected["duplicate_of_existing_entity"] += 1
                        continue
                    bound = None
                    if cls == "analyses":
                        # foci are a necessary condition: an analysis nothing reports
                        # coordinates for cannot be admitted, however well it reads
                        bound = bind_foci(item, unclaimed)
                        if bound is None:
                            rejected["analysis_without_foci"] += 1
                            continue
                    # an entity only joins the record if the paper is judged to describe it
                    claim = EXISTENCE.get(cls, "The paper describes: {label}.").format(label=lab)
                    probs = check([method_results], [claim])
                    if probs[0] < args.threshold:
                        # Counted, not silent. This gate rejected an unknown number of
                        # candidates for the whole life of the loop, because a bare
                        # `continue` leaves no trace and `rejected={}` then reads as "nothing
                        # was turned away".
                        rejected["existence_unsupported"] += 1
                        continue
                    # Eight of the fourteen record classes do not declare `name` -- a
                    # Measure is identified by its `type`, a ModelEstimation by its
                    # `model_type`. Writing a `name` onto them produced 66 undeclared-
                    # attribute errors across 15 papers. The label still does its work in the
                    # existence check and the dedupe above; it just does not become a slot.
                    allowed = DECLARED.get(cls) or set()
                    new = {"local_id": mint_id(record, cls, lab, added[cls])}
                    if "name" in allowed:
                        new["name"] = wrap(lab, text)
                    new["_provenance"] = {
                        "source": "nuextract3", "iteration": it,
                        "existence_support": round(float(probs[0]), 4),
                        "bound_row_group": (bound or {}).get("_rg_id"),
                        "bound_table_id": (bound or {}).get("table_id"),
                        "reported_as": (bound or {}).get("name"),
                        "n_foci": len((bound or {}).get("points") or []),
                        "foci": [q.get("coordinates")
                                 for q in ((bound or {}).get("points") or [])],
                        "fields": {k: v for k, v in item.items() if k != "name"}}
                    # The model answered for the other slots too; filing those answers under
                    # `_provenance` and leaving the entity bare is what left 75 Regions
                    # without `definition_method` and 16 ModelEstimations without
                    # `model_family`/`stage`/`model_type` -- required slots, every one of them
                    # already sitting in the reply.
                    for slot, value in item.items():
                        if slot in ("name", "local_id") or slot in (REF_SLOTS.get(cls) or {}):
                            continue
                        if slot not in allowed:
                            continue
                        if isinstance(value, (str, int, float, bool)) and str(value).strip():
                            cast = shaped(value, cls, slot)
                            if cast is None:
                                rejected[f"{slot}: not castable to its range"] += 1
                                continue
                            new[slot] = wrap(cast, text)
                    # An entity carrying nothing but an id says nothing the record did not
                    # already know, and still costs a required-attribute error apiece.
                    if not [k for k in new if k not in ("local_id", "_provenance")]:
                        rejected["no_declared_slot_to_fill"] += 1
                        continue
                    record.setdefault(cls, []).append(new)
                    added[cls] += 1
            print(f"      repaired: swept {todo}, added {dict(added) or '{}'}"
                  f"  edited={dict(edited) or '{}'} linked={dict(linked) or '{}'}"
                  f"  rejected={dict(rejected) or '{}'}"
                  f"  coord_tables={len(parsed_foci)}", flush=True)
            if not added:
                break

        report.extend(history)
        # `_provenance` is this loop's audit trail, not part of the schema: leaving it inline
        # was 300 of the 665 validation errors the repair introduced. It moves to a sidecar,
        # so the record validates and the audit survives.
        # Strip first: `_provenance` never reaches the file, so counting it as damage
        # reports a violation that does not exist in what anyone downstream will read.
        audit = strip_provenance(record)
        try:
            damage = introduced(json.loads(
                (args.records / f"{pmid}.extraction.json").read_text()), record)
        except Exception as exc:            # a validator failure must not lose the record
            damage = [f"validation skipped: {exc}"]
        if damage:
            print(f"      INTRODUCED {sum(int(d.split()[0]) for d in damage if d[0].isdigit())}"
                  f" schema violations:", flush=True)
            for line in damage[:6]:
                print(f"        {line[:150]}", flush=True)
        (args.out / f"{pmid}.repaired.json").write_text(json.dumps(record, indent=1))
        (args.out / f"{pmid}.provenance.json").write_text(json.dumps(audit, indent=1))

    (args.out / "history.json").write_text(json.dumps(report, indent=1))
    print("\n### before -> after")
    by = defaultdict(list)
    for row in report:
        by[row["pmid"]].append(row)
    for pmid, rows in by.items():
        a, b = rows[0], rows[-1]
        print(f"  {pmid}: analyses {a['n_analyses']}->{b['n_analyses']} | "
              f"prose_supported {a['prose_supported']}->{b['prose_supported']} "
              f"(recall proxy) | prose_precision {a['prose_precision']:.3f}->"
              f"{b['prose_precision']:.3f} | entity_precision "
              f"{a['entity_precision']:.3f}->{b['entity_precision']:.3f} | "
              f"pruned {b['pruned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
