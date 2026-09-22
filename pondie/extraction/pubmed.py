"""What PubMed states about an article, for the criteria that exclude by it.

`Study.study_type` is `deterministic` and was `None` on all 1,817 committed records with no
writer anywhere in the package, which makes a whole class of inclusion criterion
unexpressible. Six of the benchmark's sixteen meta-analyses state one -- "editorial letters,
case-reports, systematic reviews, meta-analyses, and methodological studies" excluded (sleep
deprivation), "systematic reviews or meta-analyses" (social) -- and the enum the schema
declares is exactly PubMed's own vocabulary.

`Study.language` rides along. "English language" is an inclusion criterion in at least six
of the sixteen and was unexpressible for want of a slot; `esummary` returns it in the same
response as the publication types, so the second field costs no second request. On the
1,804 records measured it excludes one paper -- a criterion that is stated should be
expressible even where it is nearly inert.

`esummary` rather than `efetch`: it returns `pubtype` and `lang` as lists of the same
strings the schema quotes, so nothing has to be parsed out of MEDLINE XML or mapped. The
values are written verbatim, which is what the slots' descriptions ask for.

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


#: The `esummary` fields this module reads, and the record slot each fills.
FIELDS = {"pubtype": "study_type", "lang": "language"}


def summaries(
    pmids: Iterable[str], *, batch: int = BATCH, pause: float = 0.15, retries: int = 4
) -> dict[str, dict[str, list[str]]]:
    """pmid -> {slot: values}, for every slot `FIELDS` names.

    A pmid PubMed does not know is absent from the result rather than present and empty:
    "we asked and it has none" and "we could not ask" are different claims, and only the
    first should ever reach the record.

    One request per 200 ids for both fields rather than one per field: `esummary` returns
    the whole record and the caller pays for the round trip, not for the columns.
    """
    wanted = [str(p).strip() for p in pmids if str(p).strip().isdigit()]
    found: dict[str, dict[str, list[str]]] = {}
    credentials = _credentials()
    for start in range(0, len(wanted), batch):
        chunk = wanted[start : start + batch]
        query = urllib.parse.urlencode(
            {"db": "pubmed", "retmode": "json", "id": ",".join(chunk), **credentials}
        ).encode()
        delay = 1.0
        for attempt in range(retries):
            try:
                # POST, not a query string. 200 ids is a 2kB URL and NCBI answers a GET of
                # that length with a 500 often enough to lose a whole batch; the endpoint
                # takes the same parameters as a form body.
                with urllib.request.urlopen(ESUMMARY, data=query, timeout=60) as response:
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
            found[str(uid)] = {
                slot: [str(v) for v in (entry.get(field) or []) if str(v).strip()]
                for field, slot in FIELDS.items()
            }
        time.sleep(pause)
    return found


def publication_types(pmids: Iterable[str], **kwargs) -> dict[str, list[str]]:
    """pmid -> its PubMed publication types, verbatim. `summaries` for one field."""
    return {pmid: found["study_type"] for pmid, found in summaries(pmids, **kwargs).items()}


def fill(record: dict, found: Mapping[str, Mapping[str, list[str]] | list[str]]) -> list[str]:
    """Write what PubMed said onto one record. Returns what changed.

    Keyed on the record's own `local_id`, which for this corpus is the pmid. A record whose
    id is not a pmid, or a pmid the lookup did not reach, is left alone -- writing an empty
    list would assert that PubMed says nothing about the article, which is never true:
    every article is at least a `Journal Article` in at least one language.

    Takes either shape `summaries` and `publication_types` return, because the second was
    the whole module once and callers still hold its output.
    """
    local_id = record.get("local_id")
    if not isinstance(local_id, str) or not local_id.isdigit():
        return []
    answer = found.get(local_id)
    if not answer:
        return []
    if isinstance(answer, list):                      # `publication_types`' shape
        answer = {"study_type": answer}

    changed = []
    for slot, values in answer.items():
        if not values or record.get(slot) == list(values):
            continue
        record[slot] = list(values)
        changed.append(f"Study.{slot}: {', '.join(values)}")
    return changed
