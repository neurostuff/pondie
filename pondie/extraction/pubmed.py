"""Publication types from PubMed, for the criteria that exclude by them.

`Study.study_type` is `deterministic` and was `None` on all 1,817 committed records with no
writer anywhere in the package, which makes a whole class of inclusion criterion
unexpressible. Six of the benchmark's sixteen meta-analyses state one -- "editorial letters,
case-reports, systematic reviews, meta-analyses, and methodological studies" excluded (sleep
deprivation), "systematic reviews or meta-analyses" (social) -- and the enum the schema
declares is exactly PubMed's own vocabulary.

`esummary` rather than `efetch`: it returns `pubtype` as a list of the same strings the
schema quotes, so nothing has to be parsed out of MEDLINE XML or mapped. The values are
written verbatim, which is what the slot's description asks for.

Not a repair's own business. A repair takes a record and returns changes; this needs the
network and batches 200 ids per request, so the caller fetches and `fill` applies. That is
the shape `stage1` and `table_map` already have on `fix.Context`.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping

ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

#: E-utilities takes up to 200 ids per request and asks for at most 3 requests a second
#: without a key, 10 with one. Batching is why this is not a per-record lookup: 1,817 papers
#: is 10 requests rather than 1,817.
BATCH = 200

#: Types that mean the article is not a report of original data. Not used here -- a query
#: applies it -- but named so the exclusion has one definition rather than one per caller.
NOT_ORIGINAL_RESEARCH = frozenset(
    {"Review", "Systematic Review", "Meta-Analysis", "Editorial", "Letter", "Comment",
     "Case Reports", "Published Erratum", "Retraction of Publication", "Retracted Publication"}
)


def _credentials() -> dict[str, str]:
    """Whatever of tool/email/api_key the environment offers.

    NCBI asks for tool and email and rate-limits harder without a key. All three are
    optional so the module works on a host that has none, at the slower limit.
    """
    pairs = {
        "api_key": os.environ.get("PUBMED_API_KEY", ""),
        "tool": os.environ.get("PUBMED_TOOL", "pondie"),
        "email": os.environ.get("EMAIL", ""),
    }
    return {k: v for k, v in pairs.items() if v}


def publication_types(
    pmids: Iterable[str], *, batch: int = BATCH, pause: float = 0.15, retries: int = 4
) -> dict[str, list[str]]:
    """pmid -> its PubMed publication types, verbatim.

    A pmid PubMed does not know is absent from the result rather than present and empty:
    "we asked and it has none" and "we could not ask" are different claims, and only the
    first should ever reach the record.
    """
    wanted = [str(p).strip() for p in pmids if str(p).strip().isdigit()]
    found: dict[str, list[str]] = {}
    credentials = _credentials()
    for start in range(0, len(wanted), batch):
        chunk = wanted[start : start + batch]
        query = urllib.parse.urlencode(
            {"db": "pubmed", "retmode": "json", "id": ",".join(chunk), **credentials}
        )
        delay = 1.0
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(f"{ESUMMARY}?{query}", timeout=60) as response:
                    payload = json.loads(response.read()).get("result") or {}
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt == retries - 1:
                    payload = {}
                else:
                    time.sleep(delay)
                    delay *= 2
        for uid in payload.get("uids") or []:
            entry = payload.get(uid) or {}
            if entry.get("error"):
                continue
            types = [str(t) for t in (entry.get("pubtype") or []) if str(t).strip()]
            found[str(uid)] = types
        time.sleep(pause)
    return found


def fill(record: dict, types: Mapping[str, list[str]]) -> list[str]:
    """Write `Study.study_type` onto one record. Returns what changed.

    Keyed on the record's own `local_id`, which for this corpus is the pmid. A record whose
    id is not a pmid, or a pmid the lookup did not reach, is left alone -- writing an empty
    list would assert that PubMed assigns the article no type, which is never true: every
    article is at least `Journal Article`.
    """
    local_id = record.get("local_id")
    if not isinstance(local_id, str) or not local_id.isdigit():
        return []
    assigned = types.get(local_id)
    if not assigned:
        return []
    if record.get("study_type") == assigned:
        return []
    record["study_type"] = list(assigned)
    return [f"Study.study_type: {', '.join(assigned)}"]
