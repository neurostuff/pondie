"""Overlapping samples, the one criterion that is about two papers at once. Deterministic.

"overlapping samples to previous studies" is excluded by the PTSD meta-analysis, and two of
its screened-out papers reuse a cohort without citing the earlier report (Nardo 2013 /
Nardo 2010; the Hunan-fire papers). No model is asked: selection is a query over records,
and the records carry what decides it -- each cohort's size, sex counts and PTSD status.

A later paper is excluded as overlapping an earlier one when

  1. they share >= 2 PubMed authors (the same group could have scanned the same people),
  2. the earlier paper is itself selected (it is a "previous study" the pool keeps), and
  3. EVERY cohort of the later paper fits inside a cohort of the earlier paper with the same
     PTSD status: its size is <= that cohort's, and each sex count it reports is <= the
     earlier cohort's count for that sex. A cohort with no size cannot be shown to fit.

Condition 3 is what keeps a paper that adds a new cohort: gold 23155380 re-reports
21498053's 10 vs 10 coal-mine survivors but adds 20 unexposed controls, and 20 does not fit
inside 10. Shared authorship alone would have excluded it; the two share six.
"""
from __future__ import annotations

import itertools
import re

import queries

MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}


def date_key(meta: dict, pmid: str) -> tuple:
    """PubMed's pubdate ("2006 Jan 30", "2018", "2010 May") as a sortable key."""
    parts = (meta.get("pubdate") or "").split()
    y = int(parts[0]) if parts and parts[0].isdigit() else 9999
    m = MONTHS.get(parts[1][:3].lower(), 0) if len(parts) > 1 else 0
    return (y, m, int(pmid))


def _sex(label: str) -> str | None:
    if re.search(r"fem|wom[ae]n|girl", label, re.I):
        return "f"
    if re.search(r"male|\bm[ae]n\b|boy", label, re.I):
        return "m"
    return None


def cohorts(record: dict) -> list[dict]:
    """{status, size, sex counts} for every cohort of a record."""
    ix = queries.Index(record)
    out = []
    for gid, g in ix.groups.items():
        sexes = {}
        for d in g.get("sex_distribution") or []:
            if isinstance(d, dict):
                key = _sex(" ".join(queries.strs(d.get("category"))))
                count = queries.num(d.get("count"))
                if key and count is not None:
                    sexes[key] = count
        size = queries.num(g.get("acquired_count"))
        if size is None:
            size = queries.num(g.get("enrolled_count"))
        if size is None and sexes:
            size = sum(sexes.values())
        out.append({"id": gid, "status": ix.is_ptsd_group(gid), "size": size, "sex": sexes})
    return out


def fits(later: dict, earlier: dict) -> bool:
    if later["status"] is None or later["status"] != earlier["status"]:
        return False
    if later["size"] is None or earlier["size"] is None or later["size"] > earlier["size"]:
        return False
    return all(earlier["sex"].get(k, float("inf")) >= v for k, v in later["sex"].items())


def subset(later: dict, earlier: dict) -> bool:
    """Every cohort of `later` fits inside some cohort of `earlier`."""
    mine, theirs = cohorts(later), cohorts(earlier)
    return bool(mine) and all(any(fits(c, e) for e in theirs) for c in mine)


def excluded(records: dict[str, dict], selected: set[str], meta: dict) -> dict[str, str]:
    """pmid -> the earlier selected pmid whose sample it re-reports."""
    out: dict[str, str] = {}
    for a, b in itertools.combinations(sorted(records), 2):
        shared = set(meta.get(a, {}).get("authors") or []) & set(meta.get(b, {}).get("authors") or [])
        if len(shared) < 2:
            continue
        if date_key(meta[a], a) > date_key(meta[b], b):
            a, b = b, a  # a is the earlier
        if a in selected and b in selected and subset(records[b], records[a]):
            out[b] = a
    return out
