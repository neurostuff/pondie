"""Who wrote an article and when, from PubMed's `esummary`, for the overlap query.

Language and publication types are not read here: they come from the parsed paper's
bibliography, which ingestion fills, and a record carries neither.

`esummary` takes 200 ids a request, so the caller fetches in batches and the query applies
the result (`pondie.query.overlap`).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator

ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

#: E-utilities takes up to 200 ids per request and asks for at most 3 requests a second
#: without a key, 10 with one. Batching is why this is not a per-record lookup: 1,817 papers
#: is 10 requests rather than 1,817.
BATCH = 200

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


def _entries(
    pmids: Iterable[str], *, batch: int = BATCH, pause: float = 0.15, retries: int = 4
) -> Iterator[tuple[str, dict]]:
    """(pmid, esummary entry) for every pmid PubMed knows, 200 to a request."""
    wanted = [str(p).strip() for p in pmids if str(p).strip().isdigit()]
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
            if not entry.get("error"):
                yield str(uid), entry
        time.sleep(pause)


def authorship(pmids: Iterable[str], **kwargs) -> dict[str, dict]:
    """pmid -> {"authors": [...], "pubdate": "2006 Jan 30"}, verbatim from `esummary`.

    They are what a query
    about two papers at once needs -- whether the same group could have reported the same
    participants twice, and which report came first. See `pondie.query.overlap`.
    """
    return {
        uid: {
            "authors": [a.get("name") for a in entry.get("authors") or [] if a.get("name")],
            "pubdate": str(entry.get("pubdate") or ""),
        }
        for uid, entry in _entries(pmids, **kwargs)
    }
