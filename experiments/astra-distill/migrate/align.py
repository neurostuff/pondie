"""Re-anchor an old record's evidence span text in the new formatted text.

Passes, in order (the first that places the span wins):
  exact      the span's text occurs verbatim; nearest occurrence to the old relative position
  canonical  both texts canonicalized (NFKC, quotes/dashes/minus/spaces unified, markdown markers and
             whitespace runs dropped); the match is mapped back to new-text offsets
  fuzzy      best partial alignment of the canonical span in the canonical text (rapidfuzz), score >= FUZZY_MIN,
             first in a window around the old relative position, then the whole text
Anything else is left unresolved for manual alignment (`manual.py`).
"""
import re
import unicodedata

from rapidfuzz import fuzz

FUZZY_MIN = 90.0
WINDOW = 4000

_TRANS = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    "×": "x",
})
_DROP = set("#*_|­​‌‍﻿")


def canonical(s):
    """Canonical form of `s` and, for each canonical char, the index of the original char it came from."""
    out, idx = [], []
    prev_space = True
    for i, ch in enumerate(s):
        for c in unicodedata.normalize("NFKC", ch).translate(_TRANS):
            if c in _DROP:
                continue
            if c.isspace():
                if prev_space:
                    continue
                c, prev_space = " ", True
            else:
                prev_space = False
            out.append(c.lower())
            idx.append(i)
    while out and out[-1] == " ":
        out.pop(), idx.pop()
    return "".join(out), idx


def char_similarity(span, region):
    """Indel similarity of the two texts with case, punctuation forms and all whitespace canonicalized away."""
    from rapidfuzz.distance import Indel
    a = canonical(span)[0].replace(" ", "")
    b = canonical(region)[0].replace(" ", "")
    return Indel.normalized_similarity(a, b)


def coverage(span, region):
    """Share of the span's tokens found in order in the region (1.0 = the span is a subsequence of it)."""
    toks, rt = span.split(), region.split()
    j = hit = 0
    for t in toks:
        k = j
        while k < len(rt) and rt[k] != t:
            k += 1
        if k < len(rt):
            hit += 1
            j = k + 1
    return hit / max(1, len(toks))


GAP_COV = 0.95
GAP_STRETCH = 2.0


GAP_A = 80      # a gap where the old text has a run of 2+ spaces: dropped markup the new text restores
GAP_B = 12      # a gap between any two words (second tier)


def gap_pattern(text, every_space):
    """The old span's canonical segments (split at its gaps) and a regex over canonical text matching them in order
    with bounded wildcards between. Returns (regex, segments) or (None, None)."""
    raw = unicodedata.normalize("NFKC", text).translate(_TRANS)
    raw = raw.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    segs = re.split(r"\s{2,}", raw.strip()) if not every_space else raw.split()
    segs = [canonical(s)[0] for s in segs]
    segs = [s for s in segs if s]
    if len(segs) < 2:
        return None, None
    gap = f".{{0,{GAP_B if every_space else GAP_A}}}?"
    parts = [re.escape(s).replace("\\ ", " ?") for s in segs]
    return re.compile((" ?" + gap + " ?").join(parts)), segs


class Aligner:
    def __init__(self, new_text, old_len):
        self.new = new_text
        self.old_len = max(1, old_len)
        self.can, self.idx = canonical(new_text)
        self.dsp = "".join(c for c in self.can if c != " ")
        self.didx = [i for c, i in zip(self.can, self.idx) if c != " "]

    def _expected(self, old_start):
        return int((old_start or 0) / self.old_len * len(self.new))

    def _nearest(self, hay, needle, expect):
        best = None
        i = hay.find(needle)
        while i != -1:
            if best is None or abs(i - expect) < abs(best - expect):
                best = i
            i = hay.find(needle, i + 1)
        return best

    def _in_order(self, segs, gap, cexpect):
        """Canonical (start, end) where `segs` occur in order, each starting at most `gap` chars after the previous
        one ends; linear in the number of first-segment occurrences. Nearest to `cexpect` wins."""
        first = segs[0].strip()
        if len(first) < 3 and len(segs) > 1:
            return None
        best, i = None, self.can.find(first)
        while i != -1:
            pos, ok = i + len(first), True
            for seg in segs[1:]:
                seg = seg.strip()
                j = self.can.find(seg, pos)
                if j == -1 or j - pos > gap:
                    ok = False
                    break
                pos = j + len(seg)
            if ok and (best is None or abs(i - cexpect) < abs(best[0] - cexpect)):
                best = (i, pos)
            i = self.can.find(first, i + 1)
        return best

    def _gapped(self, ct, cexpect):
        """The span with words inserted in the new text (restored sub/superscripts, labels): anchor its first and
        last words, accept when every span token appears in order between them and the region is not stretched."""
        toks = ct.split()
        if len(toks) < 4:
            return None
        for k in (5, 4, 3, 2):
            if 2 * k > len(toks):
                continue
            head, tail = " ".join(toks[:k]), " ".join(toks[-k:])
            best = None
            i = self.can.find(head)
            while i != -1:
                j = self.can.find(tail, i + len(head))
                if j != -1:
                    end = j + len(tail)
                    if end - i <= GAP_STRETCH * len(ct) + 40 and coverage(ct, self.can[i:end]) >= GAP_COV:
                        if best is None or abs(i - cexpect) < abs(best[0] - cexpect):
                            best = (i, end)
                i = self.can.find(head, i + 1)
            if best is not None:
                return best
        return None

    def _back(self, cs, ce):
        """Canonical [cs, ce) -> new-text [start, end)."""
        return self.idx[cs], self.idx[ce - 1] + 1

    def place(self, text, old_start=None):
        """(start, end, method, score) or None."""
        if not text:
            return None
        expect = self._expected(old_start)
        i = self._nearest(self.new, text, expect)
        if i is not None:
            return i, i + len(text), "exact", 100.0
        ct, _ = canonical(text)
        if not ct:
            return None
        cexpect = int(expect / max(1, len(self.new)) * len(self.can))
        i = self._nearest(self.can, ct, cexpect)
        if i is not None:
            return (*self._back(i, i + len(ct)), "canonical", 100.0)
        dt = ct.replace(" ", "")
        dexpect = int(cexpect / max(1, len(self.can)) * len(self.dsp))
        i = self._nearest(self.dsp, dt, dexpect)
        if i is not None:
            return self.didx[i], self.didx[i + len(dt) - 1] + 1, "despaced", 100.0
        for every in (False, True):
            pat, segs = gap_pattern(text, every)
            if pat is None:
                continue
            g = self._in_order(segs, GAP_B if every else GAP_A, cexpect)
            if g is not None:
                return (*self._back(*g), "wildcard_b" if every else "wildcard", 100.0)
        g = self._gapped(ct, cexpect)
        if g is not None:
            return (*self._back(*g), "gapped", 100.0 * coverage(ct, self.can[g[0]:g[1]]))
        lo = max(0, cexpect - WINDOW)
        for a, b in ((lo, cexpect + WINDOW + len(ct)), (0, len(self.can))):
            hay = self.can[a:b]
            if len(hay) < len(ct) * 0.5:
                continue
            r = fuzz.partial_ratio_alignment(ct, hay, score_cutoff=FUZZY_MIN)
            if r is not None and r.dest_end > r.dest_start:
                return (*self._back(a + r.dest_start, a + r.dest_end), "fuzzy", r.score)
        return None
