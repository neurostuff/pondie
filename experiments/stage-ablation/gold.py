"""The benchmark's side: criteria, included sets, and the gold analyses with their foci.

Read from the neurometabench checkout on beast. Nothing here calls a model.
"""
from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

BENCH = Path("/data/james/pondie-vs-fulltext/repos/neurometabench/data")

#: meta pmid -> nimads project directory
PROJECTS = {"36100907": "vbm_of_ptsd", "35664889": "dementia",
            "36115222": "vbm_of_substance_use", "34400176": "cue_reactivity",
            "32078973": "decision_making", "29944961": "problem_solving",
            "36436737": "social"}

#: Corrections to the benchmark, each argued in JOURNAL.md. wrong pmid -> right pmid.
REMAP = {
    # "Zhang et al., 2018" was fuzzy-matched by author+year to an fMRI emotion-perception
    # paper. The study the coordinates come from (n=35, focus 48,24,-29, "PTSD < TEC") is
    # 30555358, which prints that focus in its Table 2.
    "36100907": {"29740753": "30555358"},
}


@lru_cache
def meta(meta_pmid: str) -> dict:
    for row in csv.DictReader(open(BENCH / "meta_datasets.csv", encoding="utf-8")):
        if row["pmid"] == meta_pmid:
            return row
    raise KeyError(meta_pmid)


@lru_cache
def included(meta_pmid: str) -> frozenset[str]:
    remap = REMAP.get(meta_pmid, {})
    return frozenset(remap.get(r["study_pmid"], r["study_pmid"])
                     for r in csv.DictReader(open(BENCH / "included_studies.csv"))
                     if r["meta_pmid"] == meta_pmid)


#: Where the published included set contradicts the meta-analysis's own stated criteria,
#: read from the papers. JOURNAL.md argues each. pmid -> (in?, why); in=None is unscored.
ADJUDICATED = {
    "36100907": {
        "21118656": (False, "a priori ROI: VBM restricted to the parcellation's ROIs"),
        # Ambiguous, so unscored: its pooled foci are AAL-mask ROI peaks (hippocampus, ACC),
        # but the paper also reports whole-brain "nonhypothesized" reductions in prose.
        "19794316": (None, "ROI foci pooled from a paper that also reports a whole-brain result"),
        # Hunan fire, Nov 2003, the same 12 PTSD (8F/4M) vs 12 non-PTSD in all three,
        # each reporting non-PTSD vs PTSD. "overlapping samples to previous studies"
        # keeps the first report.
        "16371250": (True, "first report of the Hunan-fire sample (Psychiatry Res, Jan 2006)"),
        "16838824": (False, "same Hunan-fire sample as 16371250, reported later"),
        "19538748": (False, "same Hunan-fire sample as 16371250, reported later"),
    },
    "36115222": {
        # "assessing GM volume differences": this measures cortical thickness only.
        "20875635": (False, "cortical thickness, not grey-matter volume"),
        # Its VBM is cerebellum-only (SUIT); a whole-brain DARTEL analysis is said to be in
        # the supplement, but the pooled foci are the cerebellar ones.
        "29065207": (None, "cerebellum-restricted VBM, whole-brain version in the supplement"),
    },
}


def labels(meta_pmid: str, mode: str = "benchmark") -> frozenset[str]:
    """The included set: as published (`benchmark`) or with ADJUDICATED applied."""
    inc = set(included(meta_pmid))
    if mode == "adjudicated":
        for pmid, (keep, _why) in ADJUDICATED.get(meta_pmid, {}).items():
            (inc.add if keep else inc.discard)(pmid)
    return frozenset(inc)


def unscored(meta_pmid: str, mode: str) -> frozenset[str]:
    if mode != "adjudicated":
        return frozenset()
    return frozenset(p for p, (keep, _) in ADJUDICATED.get(meta_pmid, {}).items() if keep is None)


@lru_cache
def gold_studyset(meta_pmid: str) -> dict[str, list[dict]]:
    """study pmid -> [{name, points: [(x,y,z)], space, n}] from the merged NiMADS studyset."""
    path = BENCH / "nimads" / PROJECTS[meta_pmid] / "merged" / "nimads_studyset.json"
    out: dict[str, list[dict]] = {}
    for study in json.loads(path.read_text())["studies"]:
        for a in study.get("analyses") or []:
            pts = [tuple(p["coordinates"]) for p in a.get("points") or []]
            spaces = {p.get("space") for p in a.get("points") or []}
            sid = REMAP.get(meta_pmid, {}).get(str(study["id"]), str(study["id"]))
            out.setdefault(sid, []).append({
                "name": a.get("name"), "points": pts, "space": "/".join(sorted(filter(None, spaces))),
                "n": (a.get("metadata") or {}).get("sample_sizes")})
    return out


def near(a, b, tol=1.01) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


_NUM = r"[-−–]?\s?\d+(?:\.\d+)?"


def in_text(point, text: str) -> bool:
    """Whether the three numbers of a focus appear in order, close together, in the text."""
    def pat(v):
        v = float(v)
        whole = str(int(abs(v))) if v == int(v) else f"{abs(v):g}"
        sign = r"[-−–]\s?" if v < 0 else r"(?<![-−–\d])\+?"
        return sign + re.escape(whole) + r"(?:\.0+)?(?!\d)"
    rx = re.compile(r"\s*[,;|/\s]\s*(?:\|\s*)*".join(pat(v) for v in point))
    return bool(rx.search(text.replace(" ", " ")))
