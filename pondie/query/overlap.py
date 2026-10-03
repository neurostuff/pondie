"""Overlapping samples: the one inclusion criterion that is about two papers at once.

"overlapping samples to previous studies" excludes a paper that re-reports people an
earlier included paper already reported. A record cannot answer it alone. The paper may
never cite the earlier report -- Nardo 2013 re-reports Nardo 2010's train drivers without a
word -- and when it does, `Group.sample_source_reference` is a citation string, not a paper.
And `previously_reported` cannot veto by itself: a twin study truthfully says all four of its
cohorts were reported before, in a hippocampal-tracing paper no VBM meta-analysis would pool.

So this compares records, deterministically. A later paper re-reports an earlier one when

  1. they share at least `min_shared_authors` PubMed authors (3), so the same group could
     have scanned the same people. Two was too few: a coal-mine-flood paper and a Hunan-fire
     paper share two authors, and when the coal record gave no sex counts its 10 survivors
     "fit" inside the fire paper's 12. Every true re-report in the PTSD pool shares 3-5.
  2. the earlier paper is itself selected -- it is the "previous study" the pool keeps, and
  3. EVERY cohort of the later paper fits inside a cohort of the earlier one with the same
     status: its size is no larger, it reports no sex the earlier cohort does not, and each
     sex count is no larger. A cohort with no size cannot be shown to fit.

Condition 3 is what keeps a paper that adds a cohort. Two coal-mine-flood papers share six
authors and the same 10 vs 10 survivors, and the later one adds 20 unexposed controls; 20
does not fit inside 10, so it stays. Shared authorship alone would have dropped it.

The sex condition is one-sided on purpose. A later paper may report less than the earlier
one -- a re-report of 8F/4M cohorts that gives no sex split still fits -- but not more: 10 men
from a coal-mine flood must not "fit" inside 12 fire survivors of unreported sex.

Measured on the VBM-of-PTSD pool (experiments/stage-ablation/JOURNAL.md, E5): it makes the
same two exclusions a model judging the same pairs made, and none of the pairs it keeps.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pondie.formats.values import value_of

#: A cohort's status: which side of the comparison it is on. Two cohorts can only be the
#: same people if they are on the same side.
Status = Callable[[Mapping], object]

MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}


def healthy_status(group: Mapping) -> object:
    """The default status: pondie's derived `is_healthy`, None where nobody could tell."""
    from pondie.normalization.is_healthy import derive

    return derive(dict(group))


def date_key(published: str, pmid: str) -> tuple[int, int, int]:
    """PubMed's `pubdate` ("2006 Jan 30", "2018", "2010 May") as a sortable key."""
    parts = (published or "").split()
    year = int(parts[0]) if parts and parts[0].isdigit() else 9999
    month = MONTHS.get(parts[1][:3].lower(), 0) if len(parts) > 1 else 0
    return (year, month, int(pmid) if pmid.isdigit() else 0)


def _number(node) -> float | None:
    value = value_of(node)
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _sex(label: str) -> str | None:
    if re.search(r"fem|wom[ae]n|girl", label, re.I):
        return "f"
    if re.search(r"male|\bm[ae]n\b|boy", label, re.I):
        return "m"
    return None


@dataclass(frozen=True)
class Cohort:
    status: object
    size: float | None
    sex: Mapping[str, float]
    name: str = ""

    def fits_inside(self, earlier: "Cohort") -> bool:
        if self.status is None or self.status != earlier.status:
            return False
        if self.size is None or earlier.size is None or self.size > earlier.size:
            return False
        if not set(self.sex) <= set(earlier.sex):
            return False
        return all(earlier.sex[k] >= v for k, v in self.sex.items())


def cohorts(record: Mapping, status: Status = healthy_status) -> list[Cohort]:
    out = []
    for group in record.get("groups") or []:
        if not isinstance(group, Mapping):
            continue
        sex = {}
        for row in group.get("sex_distribution") or []:
            if isinstance(row, Mapping):
                key = _sex(str(value_of(row.get("category")) or ""))
                count = _number(row.get("count"))
                if key and count is not None:
                    sex[key] = count
        size = _number(group.get("acquired_count"))
        if size is None:
            size = _number(group.get("enrolled_count"))
        if size is None and sex:
            size = sum(sex.values())
        out.append(Cohort(status(group), size, sex, str(value_of(group.get("name")) or "")))
    return _without_aggregates(out)


def _without_aggregates(cohorts: list[Cohort]) -> list[Cohort]:
    """Drop a "whole sample" row: named as one, and as large as the others together.

    Records carry one ("all subjects", 43; "whole sample", 32), and it is not a cohort but
    the union of the rest. Kept, it cannot fit inside any single earlier cohort, so a
    re-report stopped being one as soon as `repair` gave the row a medical condition. The
    name is required as well as the sum: a later paper's 20 new controls beside 10 + 10
    re-reported survivors add up the same way and are a real cohort.
    """
    sized = [c for c in cohorts if c.size is not None]
    total = sum(c.size for c in sized)
    return [c for c in cohorts
            if not (len(sized) >= 3 and c.size is not None and c.size * 2 == total
                    and _AGGREGATE.search(c.name))]


_AGGREGATE = re.compile(r"\b(all|whole|total|combined|entire|full)\b", re.I)


def re_reports(later: Mapping, earlier: Mapping, status: Status = healthy_status) -> bool:
    """Every cohort of `later` fits inside some cohort of `earlier`."""
    mine, theirs = cohorts(later, status), cohorts(earlier, status)
    return bool(mine) and all(any(c.fits_inside(e) for e in theirs) for c in mine)


def overlapping(
    records: Mapping[str, Mapping],
    selected: set[str],
    authorship: Mapping[str, Mapping],
    status: Status = healthy_status,
    min_shared_authors: int = 3,
) -> dict[str, str]:
    """pmid -> the earlier selected pmid whose participants it re-reports.

    `authorship` is `pondie.extraction.pubmed.authorship`'s output. A pair either side of
    which PubMed does not know is left alone: no authors is no evidence of a shared group.
    """
    out: dict[str, str] = {}
    for a, b in itertools.combinations(sorted(selected), 2):
        meta_a, meta_b = authorship.get(a), authorship.get(b)
        if not meta_a or not meta_b or a not in records or b not in records:
            continue
        shared = set(meta_a.get("authors") or []) & set(meta_b.get("authors") or [])
        if len(shared) < min_shared_authors:
            continue
        if date_key(meta_a.get("pubdate", ""), a) > date_key(meta_b.get("pubdate", ""), b):
            a, b = b, a
        if re_reports(records[b], records[a], status):
            out[b] = a
    return out
