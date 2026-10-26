"""Scoring for the Vista phase: agreement with Astra. One call per paper writes the whole compact record
(tasks.py); this scores that record once per paper.

Unit for the effect scorers: one coordinate parse entry (its key). The predicted analyses of a key are the analyses
whose `source_table_analysis` is the key, with their model chain and tables; the gold is the same built from the
record (tasks.call2_target). 88% of keys have one gold analysis; for the others the predicted and gold analyses of
the key are paired one to one by name similarity (Hungarian, no threshold) and the rest count as misses / false
positives. Entries with points and no Astra analysis have the gold "no analyses": predicted cells there are false
positives. Analyses on keys that are not in the listing count as false positives too.

Prediction file (JSON): {"record": {sid: compact record | null}}; null (or absent) is a failed call. Created entities
are matched to Astra's per class (Hungarian on names, ent/score.py's similarities, thresholds 0.45) and created ids
are translated through that match; an id the match leaves out is `X:<id>`, wrong under "strict" and mapped to its most
similar Astra entity (same threshold) under "linked".

Effect scorers (ported from e2e2/score_e2e2.py; ids are made up by the call itself, so everything is aligned by names
and level strings within the analysis's catalog = the terms of its model chain):
  strict   the predicted model must pair with Astra's (ent m_sim >= 0.40); then terms (fuzzy name >= 0.7, Hungarian)
           and levels (fuzzy >= 0.8) of the two chains are paired; a cell whose term or level is unpaired is wrong
  local    the catalog alignment with no model gate (terms >= 0.7, levels >= 0.8)
  linked   (H24, primary) when the predicted model is equivalent to Astra's (m_sim with the paper's abbreviations >=
           0.40 against Astra's model or a near-duplicate twin of it): terms aligned with abbreviation-aware
           similarity (+ one remaining term of the same type and variation level on each side), levels by string,
           abbreviations or the names of the entities they link to, or equal translated link sets; otherwise strict.
Cells compare as (term, level, direction); also term F1, role F1 (term, direction) and the exact rate.
Model slot: id equal / paired (m_sim >= 0.40) / equivalent (same family, type and term names) / linked, per gold
analysis. Entity slots: micro F1 over (analysis, id) for groups, measure, inference_settings, tasks, acquisitions,
assessments, regions, seed_regions, target_regions, defines_regions. Compared entities: {(entity, direction)} of
the groups, conditions, timepoints and arms the effect cells' levels link to, strict and linked translation.
Also per gold analysis: kind accuracy, outcome, spatial_scope, statistic family, details_type.
Entities: per class (groups, tasks, conditions, assessments, regions, devices, acquisitions, preprocessings,
inference_settings, measures, arms, timepoints, external_datasets) micro P / R / F1 over papers.
Validity: guided_valid (the record validates against its guided schema, parse keys enumerated), valid_record (it
restores and validates against the extraction schema), reference_resolution (share of reference values that name a
local_id of the record), listing_keys_share (share of analyses whose source_table_analysis is a listed key).

    python score.py run PREDS.json --split val.txt --out score.json     # + --gold-preds to score the gold itself
    python score.py compare A.json B.json                               # paired paper bootstrap of B - A
    python score.py selftest --split val.txt --limit 40                 # gold vs gold, and vs perturbed gold
"""
import argparse
import collections
import copy
import json
import os
import random
import re
import sys
import unicodedata
import zlib
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assemble as A  # noqa: E402  (reference_stats)
import compact as K  # noqa: E402
import tasks as T  # noqa: E402

# ----------------------------------------------------------------------------------------------- text similarity (ent/common.py)

_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")
_COMPARATORS = (("≥", " gte "), ("≤", " lte "), ("≠", " ne "), (">=", " gte "), ("<=", " lte "),
                (">", " gt "), ("<", " lt "))
_COMPARATOR_WORDS = ((r"\bgreater than or equal to\b", " gte "), (r"\bless than or equal to\b", " lte "),
                     (r"\bgreater than\b", " gt "), (r"\bless than\b", " lt "), (r"\bversus\b", " vs "))


def normalize(t):
    raw = str(t)
    for symbol, word in _COMPARATORS:
        raw = raw.replace(symbol, word)
    dec = unicodedata.normalize("NFKD", raw)
    folded = "".join(c for c in dec if not unicodedata.combining(c)).lower()
    for pattern, word in _COMPARATOR_WORDS:
        folded = re.sub(pattern, word, folded)
    return _WS.sub(" ", _PUNCT.sub(" ", folded)).strip()


def fuzzy(a, b):
    """max(edit ratio, token Dice, 0.9 * token containment) over normalized strings"""
    if a is None or b is None:
        return 0.0
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 1.0 if na == nb else 0.0
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    shared = len(ta & tb)
    dice = 2 * shared / (len(ta) + len(tb)) if ta and tb else 0.0
    containment = shared / min(len(ta), len(tb)) if ta and tb else 0.0
    return max(SequenceMatcher(None, na, nb).ratio(), dice, 0.9 * containment)


def num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else None


def num_eq(a, b, rtol=0.01):
    a, b = num(a), num(b)
    if a is None or b is None:
        return None
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-9) + 1e-9


def norm_level(x):
    return None if x is None else " ".join(str(x).lower().split())


def assign(sim, thresh):
    if sim.size == 0:
        return []
    r, c = linear_sum_assignment(-sim)
    return [(int(i), int(j), float(sim[i, j])) for i, j in zip(r, c) if sim[i, j] >= thresh]


def hung(a, b, sim, th, tie=None):
    """one-to-one pairs (i, j) of lists a, b with sim >= th; tie(i, j) adds a tiny bonus to break ties between equal
    similarities (equal ids), never lifting a pair over the threshold"""
    if not a or not b:
        return []
    m = np.array([[sim(x, y) for y in b] for x in a])
    ok = m >= th
    if tie is not None:
        m = m + np.array([[1e-4 if tie(i, j) else 0.0 for j in range(len(b))] for i in range(len(a))]) * ok
    return [(i, j) for i, j, _ in assign(np.where(ok, m, m - 10), th)]


def setf1(p, g, sim=None, thresh=0.7):
    p, g = [x for x in p if x], [x for x in g if x]
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    m = np.array([[(sim or fuzzy)(a, b) for b in g] for a in p])
    tp = len(assign(m, thresh))
    return 2 * tp / (len(p) + len(g))


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


# ----------------------------------------------------------------------------------------------- paper access

class Ctx:
    """paths and caches; one per process"""
    paths = T.Paths()
    empty_keys, max_analyses = "include", T.MAX_ANALYSES
    ix = None
    _text, _gold = {}, {}

    @classmethod
    def index(cls):
        if cls.ix is None:
            cls.ix = T.index(str(cls.paths.schema))
        return cls.ix


def text_of(sid):
    if sid not in Ctx._text:
        try:
            Ctx._text[sid] = T.load_text(sid, Ctx.paths)
        except FileNotFoundError:
            Ctx._text[sid] = ""
    return Ctx._text[sid]


class Gold:
    """what the scorer needs of one paper's gold: the compact record, its entities, the gold call-2 object per key"""

    def __init__(self, sid):
        self.sid = sid
        self.c = T.load_compact(sid, Ctx.paths)
        self.by = T.analyses_by_key(self.c)
        # the entries that get an example: with points, at most MAX_ANALYSES analyses; an entry with no Astra analysis
        # has the gold {"analyses": []}, so cells predicted there are false positives
        parse = T.load_parse(sid, Ctx.paths, captions=False)
        self.parse_keys = list(parse)
        keep, self.dropped = T.select_keys(self.c, parse, Ctx.empty_keys, Ctx.max_analyses)
        self.call2 = {k: (T.call2_target(self.c, an) if an else {"analyses": []}) for k, _, an in keep}
        self.ents = T.entity_lists(self.c)
        self.ents["external_datasets"] = self.c.get("external_datasets") or []
        self.cand = T.candidates(self.c)
        self.models = {m["local_id"]: m for m in self.c.get("model_estimations") or []}


def gold_of(sid):
    if sid not in Ctx._gold:
        Ctx._gold[sid] = Gold(sid)
    return Ctx._gold[sid]


# ----------------------------------------------------------------------------------------------- abbreviations (e2e2)

_SF = re.compile(r"\(([A-Za-z][A-Za-z0-9\-]{1,11})\)")


def short_form(t):
    return t[:-1] if t.endswith("s") and sum(c.isupper() for c in t[:-1]) >= 2 else t


def is_short(t):
    return sum(c.isupper() for c in t) >= 2


@lru_cache(None)
def abbrevs(sid):
    """{short form: long form} defined in this paper's own text as "long form (SF)"; article-scoped"""
    out = {}
    T_ = text_of(sid)
    for m in _SF.finditer(T_):
        sf = short_form(m.group(1))
        if not is_short(sf):
            continue
        letters = [c.lower() for c in sf if c.isalpha()]
        words = re.findall(r"[A-Za-z][A-Za-z\-']*", T_[max(0, m.start() - 160):m.start()])
        for k in range(1, min(len(words), len(letters) + 3) + 1):
            ws = words[-k:]
            if ws[0][0].lower() != letters[0]:
                continue
            ini, i = "".join(w[0].lower() for w in ws), 0
            for c in ini:
                if i < len(letters) and c == letters[i]:
                    i += 1
            if i == len(letters):
                out.setdefault(sf, " ".join(ws).lower())
                break
    return out


def expand(sid, s):
    if not s:
        return s or ""
    ab = abbrevs(sid)
    return re.sub(r"[A-Za-z][A-Za-z0-9\-]*", lambda m: ab.get(short_form(m.group(0)), m.group(0)), str(s))


def shorts(s):
    return {short_form(t) for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", s or "") if is_short(short_form(t))}


def term_sim(sid, a, b):
    if a is None or b is None:
        return 0.0
    v = max(fuzzy(a, b), fuzzy(expand(sid, a), expand(sid, b)))
    return max(v, 0.75) if shorts(a) & shorts(b) else v


# ----------------------------------------------------------------------------------------------- entity similarity (ent/score.py, ent/aux.py)

THRESH = 0.45
ENT_CLASSES = ("groups", "tasks", "conditions", "assessments", "regions", "devices", "acquisitions", "preprocessings",
               "inference_settings", "measures", "arms", "timepoints", "external_datasets")
#: class -> (fields compared on matched pairs, key fields)
SPEC = {
    "measures": (("family", "type", "specific_metric"), ("source_label",)),
    "acquisitions": (("modality", "pulse_sequence_type", "magnetic_field_strength_tesla", "repetition_time_seconds",
                      "echo_time_seconds", "acquisition_voxel_size_mm"), ("modality", "pulse_sequence_type")),
    "inference_settings": (("inference_level", "multiple_comparison_method", "correction_scope", "height_threshold_type",
                            "height_threshold_value", "alpha_level", "cluster_extent_threshold", "tfce_used",
                            "permutation_count"), ("inference_level", "multiple_comparison_method",
                                                   "height_threshold_value")),
    "regions": (("region_type", "definition_method", "atlas"), ("name",)),
    "tasks": (("design_type",), ("name",)),
    "conditions": (("condition_kind",), ("name",)),
    "assessments": (("assessment_type",), ("name",)),
    "timepoints": (("relation_to_intervention",), ("name",)),
    "arms": (("arm_kind", "agent"), ("name",)),
    "devices": (("manufacturer", "model"), ("manufacturer", "model")),
    "preprocessings": (("smoothing_fwhm_mm",), ("software", "steps")),
    "external_datasets": ((), ("name",)),
}


def g_n(g):
    for k in ("acquired_count", "enrolled_count"):
        if g.get(k) is not None:
            return num(g[k])
    return None


def g_sim(p, g):
    np_, ng = p.get("name") or "", g.get("name") or ""
    t = max(fuzzy(np_, ng), 0.8 * fuzzy(f"{np_} {p.get('description') or ''}", f"{ng} {g.get('description') or ''}"))
    a, b = g_n(p), g_n(g)
    n = 0.5 if a is None or b is None else float(a == b)
    hp, hg = p.get("is_healthy"), g.get("is_healthy")
    h = 0.5 if hp is None or hg is None else float(bool(hp) == bool(hg))
    return 0.55 * t + 0.30 * n + 0.15 * h


def _flat(v):
    return ", ".join(map(str, v)) if isinstance(v, list) else str(v)


def key_str(cls, e):
    return " ".join(_flat(e[k]) for k in SPEC[cls][1] if e.get(k) not in (None, "", []))


def aux_eq(a, b):
    if isinstance(a, list) or isinstance(b, list):
        a, b = (a if isinstance(a, list) else [a]), (b if isinstance(b, list) else [b])
        return len(a) == len(b) and all(num_eq(x, y) if num(x) is not None and num(y) is not None else fuzzy(str(x), str(y)) >= .8
                                        for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if num(a) is not None and num(b) is not None and not isinstance(a, str):
        return bool(num_eq(a, b))
    return fuzzy(str(a), str(b)) >= 0.8


def ent_sim(cls):
    if cls == "groups":
        return g_sim
    fields = SPEC[cls][0]

    def f(p, g):
        if cls == "acquisitions" and p.get("modality") and g.get("modality") and p["modality"] != g["modality"]:
            return 0.0
        k = fuzzy(key_str(cls, p), key_str(cls, g))
        ag = [aux_eq(p[fl], g[fl]) for fl in fields if p.get(fl) is not None and g.get(fl) is not None]
        return 0.6 * k + 0.4 * (sum(ag) / len(ag) if ag else 0.5)
    return f


def match_entities(pred_ents, gold_ents):
    """per class: ({pred id: gold id}, tp, n_pred, n_gold) by Hungarian at THRESH"""
    out = {}
    for cls in ENT_CLASSES:
        P, G = pred_ents.get(cls) or [], gold_ents.get(cls) or []
        fn = ent_sim(cls)
        pairs = assign(np.array([[fn(p, g) for g in G] for p in P]), THRESH) if P and G else []
        idm = {P[i]["local_id"]: G[j]["local_id"] for i, j, _ in pairs if "local_id" in P[i] and "local_id" in G[j]}
        out[cls] = (idm, len(pairs), len(P), len(G))
    return out


class Trans:
    """created -> Astra entity id translation; identity when the entities are Astra's own (oracle)"""

    def __init__(self, gold, pred_ents=None):
        self.oracle = pred_ents is None
        self.gold, self.pred_ents = gold, pred_ents or {}
        self.match = {} if self.oracle else match_entities(pred_ents, gold.ents)
        self._near = {}
        self._gidx = {c: {e["local_id"]: e for e in gold.ents.get(c) or [] if "local_id" in e} for c in ENT_CLASSES}
        self._pidx = {c: {e["local_id"]: e for e in (pred_ents or {}).get(c) or [] if "local_id" in e} for c in ENT_CLASSES}

    def strict(self, cls, x):
        if x is None or self.oracle:
            return x
        return self.match[cls][0].get(x, "X:" + str(x)) if cls in self.match else "X:" + str(x)

    def linked(self, cls, x):
        """the pair, else the most similar Astra entity of the class at THRESH, else X:"""
        if x is None or self.oracle:
            return x
        got = self.match.get(cls, ({},))[0].get(x)
        if got is not None:
            return got
        key = (cls, x)
        if key not in self._near:
            ce, best = self._pidx.get(cls, {}).get(x), None
            if ce is not None:
                fn = ent_sim(cls)
                for gid, g in self._gidx[cls].items():
                    v = fn(ce, g)
                    if v >= THRESH and (best is None or v > best[0]):
                        best = (v, gid)
            self._near[key] = best[1] if best else None
        return self._near[key] or "X:" + str(x)

    def names(self, cls, level, created):
        """names of the entities a level links to"""
        ids = set(level.get(cls) or [])
        idx = self._pidx if created and not self.oracle else self._gidx
        return [e.get("name") for i, e in idx.get(cls, {}).items() if i in ids and e.get("name")]


LINKED = ("groups", "conditions", "timepoints", "arms")


# ----------------------------------------------------------------------------------------------- analysis rows

def _closure(chain, mid):
    order, todo = [], [mid]
    while todo:
        m = todo.pop(0)
        if m in order or m not in chain:
            continue
        order.append(m)
        todo += chain[m].get("inputs_from") or []
    return order


def rows_of(obj):
    """one row per analysis of a call-2 object: its cells, model and catalog (the terms of its model chain)"""
    out = []
    if not isinstance(obj, dict):
        return out
    chain = {m.get("local_id"): m for m in obj.get("model_chain") or [] if isinstance(m, dict)}
    for a in obj.get("analyses") or []:
        if not isinstance(a, dict):
            continue
        mid = a.get("model_estimation")
        terms = []
        for m in _closure(chain, mid):
            for t in chain[m].get("terms") or []:
                terms.append({"term": t.get("local_id"), "name": t.get("name") or "", "type": t.get("type"),
                              "variation_level": t.get("variation_level"), "model": m,
                              "levels": [lv for lv in t.get("levels") or [] if isinstance(lv, dict)]})
        eff = a.get("effect") or {}
        out.append({"a": a, "id": a.get("local_id"), "name": a.get("name") or "", "model": mid,
                    "model_obj": chain.get(mid), "terms": terms,
                    "cells": [{"term": c.get("term"), "level": c.get("level"), "direction": c.get("direction")}
                              for c in eff.get("cells") or [] if isinstance(c, dict)],
                    "kind": eff.get("kind")})
    return out


def lvl_str(lv):
    return None if lv.get("level") is None else str(lv["level"])


def cellset(cells, with_level=True, with_dir=True):
    out = set()
    for c in cells or []:
        k = (c.get("term"),)
        if with_level:
            k += (norm_level(c.get("level")),)
        if with_dir:
            k += (c.get("direction"),)
        out.add(k)
    return out


def link_set(level, tr):
    return frozenset((c, tr(c, x) if tr else x) for c in LINKED for x in (level.get(c) or []))


# ----------------------------------------------------------------------------------------------- model similarity

def m_sim(p, g, sim=None):
    fp, fg = p.get("model_family"), g.get("model_family")
    fam = 0.5 if not fp or not fg else fuzzy(fp, fg)
    typ = fuzzy(p.get("model_type") or "", g.get("model_type") or "")
    tf = setf1([t.get("name") for t in p.get("terms") or []], [t.get("name") for t in g.get("terms") or []], sim=sim)
    return 0.2 * fam + 0.3 * typ + 0.5 * tf


M_THRESH = 0.40


def model_key(m):
    return (normalize(m.get("model_family") or ""), normalize(m.get("model_type") or ""),
            tuple(sorted(normalize(t.get("name") or "") for t in m.get("terms") or [])))


def twin_key(m):
    return (normalize(m.get("model_family") or ""),
            tuple(sorted((normalize(t.get("name") or ""),
                          tuple(sorted(normalize(str(lv.get("level"))) for lv in t.get("levels") or [] if lv.get("level") is not None)))
                         for t in m.get("terms") or [])))


def twins(gold, gm_id):
    g = gold.models.get(gm_id)
    if g is None:
        return []
    k = twin_key(g)
    return [m for m in gold.models.values() if m["local_id"] == gm_id or twin_key(m) == k]


# ----------------------------------------------------------------------------------------------- effect translations

def strict_maps(sid, P, G):
    """term / level maps of the strict column: P's chain onto G's, only when the models pair"""
    tm, lm = {}, {}
    if not (P["model_obj"] and G["model_obj"] and m_sim(P["model_obj"], G["model_obj"]) >= M_THRESH):
        return tm, lm
    pt, gt = P["terms"], G["terms"]
    for i, j in hung([t["name"] for t in pt], [t["name"] for t in gt], fuzzy, 0.7,
                     tie=lambda i, j: pt[i]["term"] == gt[j]["term"]):
        tm[pt[i]["term"]] = gt[j]["term"]
        pls = [x for x in pt[i]["levels"] if x.get("level") is not None]
        gls = [x for x in gt[j]["levels"] if x.get("level") is not None]
        for a, b in hung([str(x["level"]) for x in pls], [str(x["level"]) for x in gls], fuzzy, 0.8,
                         tie=lambda a, b: str(pls[a]["level"]) == str(gls[b]["level"])):
            lm[(pt[i]["term"], norm_level(pls[a]["level"]))] = str(gls[b]["level"])
    return tm, lm


def translate(cells, tm, lm, P):
    """cells of P through the term map tm and level map lm. A level that is not one of the term's catalog levels
    (free text on a term with no levels, e.g. an interaction term) cannot be aligned through the catalog and stays
    as written, so that Astra's own cells score 1.0 against themselves."""
    catalog = {(t["term"], norm_level(lv.get("level"))) for t in P["terms"] for lv in t["levels"] if lv.get("level") is not None}
    out = []
    for c in cells:
        t, lv = c.get("term"), c.get("level")
        key = (t, norm_level(lv))
        out.append({"term": tm.get(t, f"X:{t}"),
                    "level": None if lv is None else (lm[key] if key in lm else lv if key not in catalog else f"X:{lv}"),
                    "direction": c.get("direction")})
    return out


def strict_cells(sid, P, G):
    return translate(P["cells"], *strict_maps(sid, P, G), P)


def local_cells(P, G):
    pt, gt = P["terms"], G["terms"]
    tm, lm = {}, {}
    for i, j in hung([t["name"] for t in pt], [t["name"] for t in gt], fuzzy, 0.7,
                     tie=lambda i, j: pt[i]["term"] == gt[j]["term"]):
        tm[pt[i]["term"]] = gt[j]["term"]
        pls = [x for x in pt[i]["levels"] if x.get("level") is not None]
        gls = [x for x in gt[j]["levels"] if x.get("level") is not None]
        for a, b in hung([str(x["level"]) for x in pls], [str(x["level"]) for x in gls], fuzzy, 0.8,
                         tie=lambda a, b: str(pls[a]["level"]) == str(gls[b]["level"])):
            lm[(pt[i]["term"], norm_level(pls[a]["level"]))] = str(gls[b]["level"])
    return translate(P["cells"], tm, lm, P)


def model_equiv(sid, gold, P, G, trans):
    """the predicted model is Astra's model or a near-duplicate twin of it, at the matcher's threshold"""
    if not P["model_obj"] or not G["model"]:
        return False
    sim = lambda a, b: term_sim(sid, a, b)  # noqa: E731
    cands = twins(gold, G["model"]) or ([G["model_obj"]] if G["model_obj"] else [])
    return any(m_sim(P["model_obj"], g, sim) >= M_THRESH for g in cands)


def linked_cells(sid, P, G, trans):
    pt, gt = P["terms"], G["terms"]
    tm, lm = {}, {}
    pairs = hung([t["name"] for t in pt], [t["name"] for t in gt], lambda a, b: term_sim(sid, a, b), 0.7,
                 tie=lambda i, j: pt[i]["term"] == gt[j]["term"])
    di, dj = {i for i, _ in pairs}, {j for _, j in pairs}
    ri = [i for i in range(len(pt)) if i not in di]
    rj = [j for j in range(len(gt)) if j not in dj]
    for sig in {(pt[i]["type"], pt[i]["variation_level"]) for i in ri}:
        a_ = [i for i in ri if (pt[i]["type"], pt[i]["variation_level"]) == sig]
        b_ = [j for j in rj if (gt[j]["type"], gt[j]["variation_level"]) == sig]
        if len(a_) == 1 and len(b_) == 1:
            pairs.append((a_[0], b_[0]))
    for i, j in pairs:
        tm[pt[i]["term"]] = gt[j]["term"]
        pls = [x for x in pt[i]["levels"] if x.get("level") is not None]
        gls = [x for x in gt[j]["levels"] if x.get("level") is not None]

        def lsim(p, g):
            ps, gs = str(p["level"]), str(g["level"])
            pn = [ps, expand(sid, ps)] + [n for c in LINKED for n in trans.names(c, p, True)]
            gn = [gs, expand(sid, gs)] + [n for c in LINKED for n in trans.names(c, g, False)]
            v = max(fuzzy(x, y) for x in pn for y in gn)
            pl_, gl_ = link_set(p, trans.linked), link_set(g, None)
            return max(v, 1.0 if pl_ and pl_ == gl_ else 0.0)
        for a_, b_ in hung(pls, gls, lsim, 0.8, tie=lambda a, b: str(pls[a]["level"]) == str(gls[b]["level"])):
            lm[(pt[i]["term"], norm_level(pls[a_]["level"]))] = str(gls[b_]["level"])
    return translate(P["cells"], tm, lm, P)


# ----------------------------------------------------------------------------------------------- compared entities and slots

def level_index(terms):
    out = {}
    for t in terms:
        for lv in t["levels"]:
            if lv.get("level") is not None:
                out.setdefault((t["term"], str(lv["level"])), lv)
    return out


def compared(row, tr):
    """{class: {(entity, direction)}} over the row's cells whose level links to the entity"""
    idx = level_index(row["terms"])
    out = {c: set() for c in LINKED}
    for c in row["cells"]:
        lv = idx.get((c.get("term"), None if c.get("level") is None else str(c["level"])))
        if lv is None:
            continue
        for k in LINKED:
            for d in lv.get(k) or []:
                out[k].add((tr(k, d) if tr else d, c.get("direction")))
    return out


SLOTS = (("groups", "groups"), ("measure", "measures"), ("inference_settings", "inference_settings"),
         ("tasks", "tasks"), ("acquisitions", "acquisitions"), ("assessments", "assessments"),
         ("regions", "regions"), ("seed_regions", "regions"), ("target_regions", "regions"),
         ("defines_regions", "regions"))


def slot_ids(a, slot):
    if slot == "groups":
        return {g.get("group") for g in a.get("groups") or [] if isinstance(g, dict) and g.get("group")}
    d = a.get("details") or {}
    v = d.get(slot) if slot in ("seed_regions", "target_regions") else a.get(slot)
    if v is None:
        return set()
    return {x for x in (v if isinstance(v, list) else [v]) if isinstance(x, str)}


FIELDS = (("kind", lambda a: (a.get("effect") or {}).get("kind")), ("outcome", lambda a: a.get("outcome")),
          ("spatial_scope", lambda a: a.get("spatial_scope")),
          ("statistic_family", lambda a: ((a.get("effect") or {}).get("statistic") or {}).get("family")),
          ("details_type", lambda a: (a.get("details") or {}).get("details_type")))


def pair_analyses(P, G):
    """[(pred row | None, gold row | None)]: a single pair when each side has one, else name-similarity Hungarian"""
    if len(P) == 1 and len(G) == 1:
        return [(P[0], G[0])]
    pairs = []
    used_p, used_g = set(), set()
    if P and G:
        m = np.array([[fuzzy(p["name"], g["name"]) + (1e-4 if p["id"] == g["id"] else 0) for g in G] for p in P])
        for i, j, _ in assign(m, -1):
            pairs.append((P[i], G[j]))
            used_p.add(i)
            used_g.add(j)
    pairs += [(P[i], None) for i in range(len(P)) if i not in used_p]
    pairs += [(None, G[j]) for j in range(len(G)) if j not in used_g]
    return pairs


def score_key(sid, gold, key, pobj, trans):
    """count vectors for one parse entry"""
    N = collections.defaultdict(lambda: [0, 0, 0, 0])

    def add(name, *v):
        for i, x in enumerate(v):
            N[name][i] += x

    gobj = gold.call2.get(key)
    G, P = rows_of(gobj), rows_of(pobj)
    N["n"][0] = len(G)
    N["n_pred"][0] = len(P)
    for p, g in pair_analyses(P, G):
        gc = cellset(g["cells"]) if g else set()
        gslots = {s: (slot_ids(g["a"], s) if g else set()) for s, _ in SLOTS}
        if p is None:  # missed analysis
            for col in ("strict", "local", "linked"):
                add(f"cells_{col}", 0, 0, len(gc), 0)
                for nm, wl, wd in (("term", False, False), ("role", False, True)):
                    add(f"{nm}_{col}", 0, 0, len(cellset(g["cells"], wl, wd)), 0)
            for s, _ in SLOTS:
                for tr in ("strict", "linked"):
                    add(f"slot_{s}_{tr}", 0, 0, len(gslots[s]), 0)
            add("model", 0, 0, 0, 0)
            for nm, fnc in FIELDS:
                add(f"field_{nm}", 0, int(fnc(g["a"]) is not None), 0, 0)
            for c in LINKED:
                for tr in ("strict", "linked"):
                    gcmp = compared(g, None)
                    add(f"cmp_ent_{tr}", 0, 0, len({x for x, _ in gcmp[c]}), 0)
                    add(f"cmp_dir_{tr}", 0, 0, len(gcmp[c]), 0)
            continue
        if g is None:  # an analysis Astra has no counterpart of: false positives
            for col in ("strict", "local", "linked"):
                add(f"cells_{col}", 0, len(cellset(p["cells"])), 0, 0)
                for nm, wl, wd in (("term", False, False), ("role", False, True)):
                    add(f"{nm}_{col}", 0, len(cellset(p["cells"], wl, wd)), 0, 0)
            for s, _ in SLOTS:
                for tr in ("strict", "linked"):
                    add(f"slot_{s}_{tr}", 0, len(slot_ids(p["a"], s)), 0, 0)
            pc = compared(p, None)
            for c in LINKED:
                for tr in ("strict", "linked"):
                    add(f"cmp_ent_{tr}", 0, len({x for x, _ in pc[c]}), 0, 0)
                    add(f"cmp_dir_{tr}", 0, len(pc[c]), 0, 0)
            continue
        # effect columns
        eq = model_equiv(sid, gold, p, g, trans)
        trs = {"strict": strict_cells(sid, p, g), "local": local_cells(p, g),
               "linked": linked_cells(sid, p, g, trans) if eq else strict_cells(sid, p, g)}
        for col, cells in trs.items():
            for nm, wl, wd in (("cells", True, True), ("term", False, False), ("role", False, True)):
                G_, P_ = cellset(g["cells"], wl, wd), cellset(cells, wl, wd)
                add(f"{nm}_{col}" if nm != "cells" else f"cells_{col}", len(G_ & P_), len(P_ - G_), len(G_ - P_),
                    int(G_ == P_) if nm == "cells" else 0)
        # model slot
        add("model", int(p["model"] == g["model"] and p["model"] is not None),
            int(bool(p["model_obj"] and g["model_obj"] and m_sim(p["model_obj"], g["model_obj"]) >= M_THRESH)),
            int(bool(p["model_obj"] and g["model_obj"] and model_key(p["model_obj"]) == model_key(g["model_obj"]))),
            int(eq))
        # entity slots
        for s, cls in SLOTS:
            gs = gslots[s]
            for tr, fn in (("strict", trans.strict), ("linked", trans.linked)):
                ps = {fn(cls, x) for x in slot_ids(p["a"], s)}
                add(f"slot_{s}_{tr}", len(gs & ps), len(ps - gs), len(gs - ps), 0)
        # compared entities
        gcmp = compared(g, None)
        for tr, fn in (("strict", trans.strict), ("linked", trans.linked)):
            pcmp = compared(p, fn)
            for c in LINKED:
                ge, pe = {x for x, _ in gcmp[c]}, {x for x, _ in pcmp[c]}
                add(f"cmp_ent_{tr}", len(ge & pe), len(pe - ge), len(ge - pe), 0)
                add(f"cmp_dir_{tr}", len(gcmp[c] & pcmp[c]), len(pcmp[c] - gcmp[c]), len(gcmp[c] - pcmp[c]), 0)
        # scalar fields
        for nm, fnc in FIELDS:
            gv = fnc(g["a"])
            add(f"field_{nm}", int(gv is not None and gv == fnc(p["a"])), int(gv is not None), 0, 0)
    return dict(N)


VALIDATORS = {}


def score_record(sid, rec):
    """-> {"units": {sid|key: counts}, "ent": {cls: [tp, n_pred, n_gold]}, "guided": [ok, 1], "valid_record": 0|1,
    "refs": [resolved, total], "keys": [in listing, total]} for one paper's predicted compact record (None: failed)"""
    from jsonschema import Draft202012Validator
    gold = gold_of(sid)
    ix = Ctx.index()
    slots = K.ref_names(ix)
    ok = isinstance(rec, dict)
    r = rec if ok else {}
    pred_ents = T.entity_lists(r)
    pred_ents["external_datasets"] = [e for e in r.get("external_datasets") or [] if isinstance(e, dict)]
    trans = Trans(gold, pred_ents)
    res = {"units": {}, "ent": {cls: [tp, npred, ngold] for cls, (_, tp, npred, ngold) in trans.match.items()}}
    by = T.analyses_by_key(r)
    # an analysis with no coordinate-set key (the guided schema allows null) is its own unit: no gold, so every
    # cell it predicts is a false positive
    for key in sorted(set(gold.call2) | set(by), key=lambda k: (k is None, str(k))):
        pobj = T.call2_target(r, by[key]) if key in by else None
        res["units"][f"{sid}|{key}"] = score_key(sid, gold, key, pobj, trans)
    for k in [k for k in VALIDATORS if k != "record"]:
        del VALIDATORS[k]
    gv = VALIDATORS.setdefault("guided", Draft202012Validator(T.record_shape(ix, "schema", gold.parse_keys)))
    res["guided"] = [int(ok and gv.is_valid(T.record_target(r))), 1]
    res["valid_record"], res["refs"] = 0, [0, 0]
    if ok:
        res["refs"] = A.reference_stats(r, slots)
        try:
            v = VALIDATORS.setdefault("record", Draft202012Validator(ix.schema))
            res["valid_record"] = int(v.is_valid(K.from_compact(r, ix, local_id=sid)))
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {e}"[:200]
    ks = [a.get("source_table_analysis") for a in r.get("analyses") or [] if isinstance(a, dict)]
    res["keys"] = [sum(k in set(gold.parse_keys) for k in ks), len(ks)]
    return res


# ----------------------------------------------------------------------------------------------- aggregation, bootstrap

COLS = ("strict", "local", "linked")


def metrics(U):
    """metrics from summed count vectors U: {name: [a, b, c, d]}"""
    g = lambda k: U.get(k, [0, 0, 0, 0])  # noqa: E731
    n = g("n")[0]
    m = {}
    for col in COLS:
        c = g(f"cells_{col}")
        m[f"cell_f1_{col}"] = f1(*c[:3])
        m[f"exact_{col}"] = c[3] / n if n else 0.0
        m[f"term_f1_{col}"] = f1(*g(f"term_{col}")[:3])
        m[f"role_f1_{col}"] = f1(*g(f"role_{col}")[:3])
    mo = g("model")
    for i, nm in enumerate(("model_id_exact", "model_paired", "model_equivalent", "model_linked")):
        m[nm] = mo[i] / n if n else 0.0
    for tr in ("strict", "linked"):
        m[f"compared_ent_f1_{tr}"] = f1(*g(f"cmp_ent_{tr}")[:3])
        m[f"compared_dir_f1_{tr}"] = f1(*g(f"cmp_dir_{tr}")[:3])
        tot = [0, 0, 0]
        for s, _ in SLOTS:
            v = g(f"slot_{s}_{tr}")
            m[f"slot_{s}_f1_{tr}"] = f1(*v[:3])
            tot = [tot[i] + v[i] for i in range(3)]
        m[f"slots_f1_{tr}"] = f1(*tot)
    for nm, _ in FIELDS:
        v = g(f"field_{nm}")
        m[f"acc_{nm}"] = v[0] / v[1] if v[1] else 0.0
    m["kind_acc"] = m["acc_kind"]
    m["n_gold_analyses"] = n
    m["n_pred_analyses"] = g("n_pred")[0]
    return m


def sum_units(units):
    U = collections.defaultdict(lambda: [0, 0, 0, 0])
    for u in units:
        for k, v in u.items():
            for i in range(4):
                U[k][i] += v[i]
    return U


MISC = ("valid_ok", "valid_n", "refs_ok", "refs_n", "g_ok", "g_n", "k_ok", "k_n")


def pack(papers, sids, names=None):
    """per-paper arrays: counts (P x names x 4), entity counts (P x classes x 3), misc (P x 8)"""
    names = names or sorted({k for s in sids for u in papers[s]["units"].values() for k in u})
    ix = {k: i for i, k in enumerate(names)}
    U = np.zeros((len(sids), len(names), 4))
    E = np.zeros((len(sids), len(ENT_CLASSES), 3))
    M = np.zeros((len(sids), len(MISC)))
    for p, s in enumerate(sids):
        for u in papers[s]["units"].values():
            for k, v in u.items():
                if k in ix:
                    U[p, ix[k]] += v
        for c, v in papers[s]["ent"].items():
            E[p, ENT_CLASSES.index(c)] = v
        m = papers[s]["misc"]
        M[p] = [m["valid_record"], 1, *m["refs"], *m["guided"], *m["keys"]]
    return names, U, E, M


def metrics_from(names, U, E, M):
    """metrics from arrays summed over papers (U: names x 4, E: classes x 3, M: misc)"""
    m = metrics({k: U[i] for i, k in enumerate(names)})
    if E.sum() > 0:
        for i, cls in enumerate(ENT_CLASSES):
            tp, np_, ng = E[i]
            m[f"ent_f1_{cls}"] = 2 * tp / (np_ + ng) if np_ + ng else 0.0
            m[f"ent_n_gold_{cls}"] = ng
        tot = E.sum(0)
        m["ent_f1_all"] = 2 * tot[0] / (tot[1] + tot[2]) if tot[1] + tot[2] else 0.0
    d = dict(zip(MISC, M))
    m["valid_record"] = d["valid_ok"] / d["valid_n"] if d["valid_n"] else None
    m["reference_resolution"] = d["refs_ok"] / d["refs_n"] if d["refs_n"] else None
    m["guided_valid"] = d["g_ok"] / d["g_n"] if d["g_n"] else None
    m["listing_keys_share"] = d["k_ok"] / d["k_n"] if d["k_n"] else None
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in m.items()}


def bootstrap(papers, nb=2000, seed=0):
    """{metric: [point, lo, hi]}: paper-level percentile bootstrap (resample papers with replacement)"""
    sids = sorted(papers)
    names, U, E, M = pack(papers, sids)
    pt = metrics_from(names, U.sum(0), E.sum(0), M.sum(0))
    rng = np.random.default_rng(seed)
    samples = collections.defaultdict(list)
    for d in rng.integers(0, len(sids), size=(nb, len(sids))):
        mm = metrics_from(names, U[d].sum(0), E[d].sum(0), M[d].sum(0))
        for k, v in mm.items():
            if v is not None:
                samples[k].append(v)
    return {k: [v, float(np.percentile(samples[k], 2.5)), float(np.percentile(samples[k], 97.5))]
            for k, v in pt.items() if v is not None and k in samples}


def compare_boot(pa, pb, nb=2000, seed=0):
    """paired paper bootstrap of metric(B) - metric(A) over the papers both have"""
    sids = sorted(set(pa) & set(pb))
    names = sorted({k for P in (pa, pb) for s in sids for u in P[s]["units"].values() for k in u})
    A_, B_ = pack(pa, sids, names), pack(pb, sids, names)
    ma, mb = metrics_from(names, A_[1].sum(0), A_[2].sum(0), A_[3].sum(0)), metrics_from(names, B_[1].sum(0), B_[2].sum(0), B_[3].sum(0))
    keys = [k for k in ma if ma[k] is not None and mb.get(k) is not None and not k.startswith(("n_", "ent_n_"))]
    rng = np.random.default_rng(seed)
    ds = collections.defaultdict(list)
    for d in rng.integers(0, len(sids), size=(nb, len(sids))):
        a = metrics_from(names, A_[1][d].sum(0), A_[2][d].sum(0), A_[3][d].sum(0))
        b = metrics_from(names, B_[1][d].sum(0), B_[2][d].sum(0), B_[3][d].sum(0))
        for k in keys:
            if a.get(k) is not None and b.get(k) is not None:
                ds[k].append(b[k] - a[k])
    return {"papers": len(sids), "metrics": {k: {"A": ma[k], "B": mb[k], "diff": mb[k] - ma[k],
                                                  "ci95": [float(np.percentile(ds[k], 2.5)), float(np.percentile(ds[k], 97.5))]}
                                              for k in keys if ds[k]}}


# ----------------------------------------------------------------------------------------------- running

def read_split(files, limit=0):
    sids = []
    for sp in files:
        f = Path(sp)
        f = f if f.exists() else Ctx.paths.splits / sp
        if f.name == "test.txt":
            raise SystemExit("test split: not scored here")
        sids += f.read_text().split()
    return sids[:limit] if limit else sids


def _paper_worker(args):
    sid, rec, paths = args
    Ctx.paths = T.Paths(**paths)
    Ctx.ix = None
    try:
        return sid, score_record(sid, rec), None
    except Exception as e:  # noqa: BLE001
        return sid, None, f"{type(e).__name__}: {e}"


def run(preds, sids, paths, workers=16, nb=2000):
    """preds: {"record": {sid: compact record | null}}"""
    from concurrent.futures import ProcessPoolExecutor
    recs = preds.get("record") or {}
    jobs = [(s, recs.get(s), paths) for s in sids]
    papers, errs = {}, []
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    for sid, r, err in (ex.map(_paper_worker, jobs, chunksize=4) if ex else map(_paper_worker, jobs)):
        if err:
            errs.append((sid, err))
            continue
        papers[sid] = {"units": r["units"], "ent": r["ent"],
                       "misc": {"valid_record": r["valid_record"], "refs": r["refs"], "guided": r["guided"], "keys": r["keys"]}}
    if ex:
        ex.shutdown()
    return {"papers_scored": len(papers), "errors": errs[:10], "n_errors": len(errs),
            "metrics": bootstrap(papers, nb) if papers else {}, "per_paper": papers}


# ----------------------------------------------------------------------------------------------- self test

def gold_preds(sids, paths):
    Ctx.paths = T.Paths(**paths)
    return {"record": {s: copy.deepcopy(T.record_target(gold_of(s).c)) for s in sids}}


def perturb(preds, kind, seed=0):
    """gold-derived perturbed records"""
    rng = random.Random(seed)
    P = copy.deepcopy(preds)
    slots = K.ref_names(Ctx.index())
    for sid, rec in list(P["record"].items()):
        if kind == "drop_half_cells":
            for a in rec.get("analyses") or []:
                cells = (a.get("effect") or {}).get("cells") or []
                if cells:
                    n_drop = len(cells) // 2 + (len(cells) % 2 and rng.random() < 0.5)
                    drop = set(rng.sample(range(len(cells)), n_drop))
                    a["effect"]["cells"] = [c for i, c in enumerate(cells) if i not in drop]
        elif kind in ("rename_term_ids", "rename_term_names"):
            mapping = {}
            for m in rec.get("model_estimations") or []:
                for t in m.get("terms") or []:
                    if kind == "rename_term_ids":
                        mapping[t["local_id"]] = "tt_" + str(zlib.crc32(t["local_id"].encode()) % 10**6)
                    else:
                        t["name"] = "zq " + "".join(rng.choice("abcdefghij") for _ in range(8))
            if mapping:
                P["record"][sid] = K.rename_ids(rec, slots, mapping)
        elif kind == "swap_group":
            ids = [g["local_id"] for g in rec.get("groups") or []]
            if len(ids) >= 2:
                sw = {ids[0]: ids[1], ids[1]: ids[0]}
                for a in rec.get("analyses") or []:
                    for g in a.get("groups") or []:
                        g["group"] = sw.get(g["group"], g["group"])
                for m in rec.get("model_estimations") or []:
                    for t in m.get("terms") or []:
                        for lv in t.get("levels") or []:
                            if lv.get("groups"):
                                lv["groups"] = [sw.get(x, x) for x in lv["groups"]]
        elif kind == "reverse_order":
            rec["analyses"] = list(reversed(rec.get("analyses") or []))
        elif kind == "drop_half_regions":
            regs = rec.get("regions") or []
            rec["regions"] = regs[: len(regs) // 2]
        elif kind == "entity_ids_renamed":  # created entities with other ids, translated through the entity match
            mapping = {}
            for e in K_entities(rec):
                mapping[e["local_id"]] = "n_" + e["local_id"]
            P["record"][sid] = K.rename_ids(rec, slots, mapping)
    return P


def K_entities(rec):
    el = T.entity_lists(rec)
    return [e for k, v in el.items() for e in v if isinstance(e.get("local_id"), str)]


def selftest(sids, paths, workers, nb):
    Ctx.paths = T.Paths(**paths)
    gp = gold_preds(sids, paths)
    show = ["cell_f1_strict", "cell_f1_local", "cell_f1_linked", "exact_linked", "term_f1_linked", "role_f1_linked",
            "model_linked", "model_equivalent", "model_paired", "slots_f1_strict", "slot_groups_f1_strict",
            "compared_ent_f1_strict", "compared_ent_f1_linked", "kind_acc", "ent_f1_all", "ent_f1_regions",
            "valid_record", "reference_resolution", "guided_valid", "listing_keys_share"]
    runs = [("gold record", gp)]
    for kind in ("drop_half_cells", "rename_term_ids", "rename_term_names", "swap_group", "reverse_order",
                 "drop_half_regions", "entity_ids_renamed"):
        runs.append((f"perturbed: {kind}", perturb(gp, kind)))
    runs.append(("no output (every record null)", {"record": {s: None for s in sids}}))
    out = {}
    for name, p in runs:
        r = run(p, sids, paths, workers, nb=nb)
        out[name] = {k: r["metrics"].get(k) for k in show if k in r["metrics"]}
        out[name].update({"papers": r["papers_scored"], "errors": r["n_errors"]})
        print(f"\n== {name}: {r['papers_scored']} papers, {r['n_errors']} errors {r['errors'][:2]}")
        for k in show:
            if k in r["metrics"]:
                v = r["metrics"][k]
                print(f"  {k:28s} {v[0]:.3f}  [{v[1]:.3f}, {v[2]:.3f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "compare", "selftest"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--split", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--gold-preds", action="store_true", help="score the gold itself (a check of the harness)")
    for k, v in T.DEFAULTS.items():
        ap.add_argument(f"--{k}", default=str(v))
    a = ap.parse_args()
    paths = {k: getattr(a, k) for k in T.DEFAULTS}
    Ctx.paths = T.Paths(**paths)
    if a.cmd == "selftest":
        out = selftest(read_split(a.split, a.limit), paths, a.workers, a.boot)
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=1))
        return
    if a.cmd == "compare":
        pa, pb = (json.loads(Path(f).read_text())["per_paper"] for f in a.files[:2])
        res = compare_boot(pa, pb, a.boot)
        print(json.dumps(res, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        return
    sids = read_split(a.split, a.limit)
    preds = gold_preds(sids, paths) if a.gold_preds else json.loads(Path(a.files[0]).read_text())
    r = run(preds, sids, paths, a.workers, nb=a.boot)
    for k, v in sorted(r["metrics"].items()):
        print(f"{k:32s} {v[0]:.4f}  [{v[1]:.4f}, {v[2]:.4f}]" if v[1] is not None else f"{k:32s} {v[0]}")
    if a.out:
        Path(a.out).write_text(json.dumps(r))


if __name__ == "__main__":
    main()
