"""What the paper says it will report, against what the record holds.

Every other check asks whether what is in the record is right. None asks whether it is all of
it, and a missing entity leaves no trace to check: 26424424 defines ten regions of interest
and the record holds six, and nothing anywhere said so. The four absent ones -- both
hippocampi, the right amygdala, the left orbitofrontal cortex -- are exactly those with no
significant finding, so the extractor kept the regions that appear in the results and
dropped the ones that appear only in the definition.

Asking the proposer does not recover them. Given the defining sentence in its premise and
the six it already has in its instruction, it returns those six and nothing else -- it
confirms rather than extends. So this reads the enumeration itself, which is deterministic
and needs no model: a paper that lists its ROIs writes them in one sentence.

Reported, never created. Minting a region from a name in a sentence is the invention this
package spends most of its guards preventing; a finding tells a reviewer where to look.
"""
from __future__ import annotations

import re
from typing import Any, Iterator, Mapping, Sequence

#: How a paper introduces the set it will report on. Deliberately narrow: a false enumeration
#: produces a false gap, and a reviewer who is sent to look at nothing stops looking.
OPENERS = re.compile(
    r"(?:the\s+)?(?:regions?|ROIs?|regions?\s+of\s+interest|volumes?\s+of\s+interest)"
    r"\s*(?:included|were|comprised|consisted\s+of|examined\s+were|analy[sz]ed\s+were)\s*:?\s*",
    re.IGNORECASE)

#: The tail of an enumeration, so a sentence that runs on does not swallow the next clause.
STOP = re.compile(r"[.;]|\s+(?:Each|These|All|This|We|The\s+ROIs?)\s", re.IGNORECASE)

#: How a paper says it *built* its regions. An enumeration is a definition only if one of
#: these follows it: without the test, "regions were activated in cocaine users" opens a
#: list and every clause of the results paragraph becomes a missing region -- seventeen
#: false findings on 11058476 alone, which is worse than saying nothing.
DEFINING = re.compile(
    r"\b(?:sphere|spheres|mask|masks|atlas|marsbar|wfu|aal|a\s+priori|anatomically|"
    r"defined|drawn|created|constructed|centered|coordinates|template|toolbox)\b",
    re.IGNORECASE)

#: What a results sentence does, which a definition does not.
REPORTING = re.compile(
    r"\b(?:activat\w+|observ\w+|show\w+|found|identifi\w+|revealed|greater|reduced|"
    r"increas\w+|decreas\w+|correlat\w+|significant\w*)\b", re.IGNORECASE)

#: A definition enumerates noun phrases. Anything with a finite verb in it is a sentence
#: about the regions rather than a list of them: "the regions of interest were compared by
#: using two-sample t tests" opened a list whose items were clauses of the paragraph that
#: followed, and three of them reached a reviewer as missing regions.
CLAUSAL = re.compile(
    r"\b(?:were|was|are|is|by|using|compared|tests?|maps?|values?|analys[ei]s|"
    r"superimposed|specific)\b", re.IGNORECASE)

#: A parenthesised short form, which is how a paper names an ROI it will use again.
SHORT_FORM = re.compile(r"\(([A-Za-z][A-Za-z0-9\-]{1,7})\)")

#: Both sides, stated once.
BILATERAL = re.compile(r"\b(?:bi-?lateral(?:ly)?|both\s+(?:hemispheres|sides))\b", re.IGNORECASE)


def enumerations(text: str) -> Iterator[tuple[str, str]]:
    """(the phrase that opened the list, the list itself) for each enumeration found."""
    for opener in OPENERS.finditer(text):
        tail = text[opener.end():opener.end() + 400]
        stop = STOP.search(tail)
        listing = tail[:stop.start()] if stop else tail
        if not (listing.count(",") >= 1 or " and " in listing):
            continue
        # A definition says how the regions were made, and a result says what they did.
        following = text[opener.end():opener.end() + 400]
        if REPORTING.search(listing) or CLAUSAL.search(listing):
            continue
        if not DEFINING.search(following):
            continue
        yield opener.group(0).strip(), listing.strip()


def named(listing: str) -> list[tuple[str, bool]]:
    """(name, is bilateral) for each item of an enumeration.

    The short form is preferred when the paper gives one, because that is what the record's
    labels tend to carry -- "anterior piriform cortex (aPC)" is stored as "left aPC".
    """
    items: list[tuple[str, bool]] = []
    # Split outside parentheses only: "prefrontal (medial, dorsolateral, orbitofrontal)" is
    # one item, and splitting inside it yields "medial" and "orbitofrontal" as regions.
    depth, current, pieces = 0, [], []
    for char in listing:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            pieces.append("".join(current)); current = []
        else:
            current.append(char)
    pieces.append("".join(current))
    for piece in [p for whole in pieces for p in re.split(r"\band\b", whole)]:
        piece = piece.strip()
        if not piece or len(piece) < 3:
            continue
        both = bool(BILATERAL.search(piece))
        short = SHORT_FORM.search(piece)
        name = short.group(1) if short else SHORT_FORM.sub("", piece).strip()
        # "The remaining three regions of interest were located in the right superior
        # frontal gyrus" names a region; the preposition in front of it is not part of the
        # name, and a finding that says 'located in the right superior frontal gyrus' reads
        # as a parsing failure even when the gap it reports is real.
        name = re.sub(r"^(?:located\s+|situated\s+)?(?:in|within|at|of)\s+", "", name,
                      flags=re.IGNORECASE)
        name = re.sub(r"^(?:bi-?lateral(?:ly)?|the|both)\s+", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^(?:the)\s+", "", name, flags=re.IGNORECASE)
        if 2 <= len(name) <= 60:
            items.append((name.strip(" ()"), both))
    return items


def missing_regions(record: Mapping[str, Any], text: str) -> list[str]:
    """Regions the paper enumerates that the record does not hold, as readable findings."""
    from pondie.extraction.record.edit import label_of

    held = " | ".join((label_of(r) or "").lower()
                      for r in record.get("regions") or [] if isinstance(r, Mapping))
    findings: list[str] = []
    for opener, listing in enumerations(text):
        # A bilateral run states it once and means it for the whole list.
        run_is_bilateral = bool(BILATERAL.search(listing))
        for name, both in named(listing):
            folded = name.lower()
            if folded in held:
                if not (both or run_is_bilateral):
                    continue
                sides = [s for s in ("left", "right") if f"{s} {folded}" not in held]
                if sides:
                    findings.append(
                        f"the paper lists {name!r} bilaterally after {opener!r}, and the "
                        f"record has no {' or '.join(sides)} one")
                continue
            findings.append(
                f"the paper lists {name!r} after {opener!r}, and the record holds no such "
                f"region")
    return findings
