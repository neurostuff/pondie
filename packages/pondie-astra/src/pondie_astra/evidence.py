"""Evidence by unit index, as ev1-full was trained on and run with: the paper text cut into units
tagged [S1] [S2] ..., and the compact record (no evidence) in; the record's evidence as unit
numbers out. Moved from experiments/astra-distill/vista/evidence.py unchanged; `prompt_of` and
`merge` are ns_ev1_bodies.py's request and ns_ev1_merge.py's merge (pondie#26).

Units (`units`): a line holding a tab (a table row) or of at most 200 characters (a heading, a
caption line) is one unit; longer lines are cut into sentences.
"""
import bisect
import collections
import json
import re

from . import compact as K
from . import prompt as P

ABBR = r"(?<!\bet al)(?<!\be\.g)(?<!\bi\.e)(?<!\bFig)(?<!\bFigs)(?<!\bvs)(?<!\bcf)(?<!\bapprox)(?<!\bNo)(?<!\bEq)(?<!\bRef)"
CUT = re.compile(ABBR + r"[.!?](?=\s+[A-Z0-9(\[])")
SHORT = 200
INSTR = ("Below is a neuroimaging paper cut into numbered units ([S1], [S2], ...: sentences, table rows, captions) and "
         "the record extracted from it. For every value in the record that the paper supports, give the numbers of the "
         "units that state it. Return the record's structure with each supported value replaced by its unit numbers "
         "(a list of integers); keep local_id, keep one entry per list item ({} when nothing in it is supported), and "
         "leave out values the paper does not state.")
TEMPLATE = '{"<record field>": ["integer"], "<record list>": [{"local_id": "string", "<field>": ["integer"]}]}'


def units(text):
    """[(start, end)] of every unit, in text order"""
    out, pos = [], 0
    for line in text.split("\n"):
        s, e = pos, pos + len(line)
        pos = e + 1
        if not line.strip():
            continue
        if "\t" in line or len(line) <= SHORT:
            out.append((s, e))
            continue
        a = s
        for m in CUT.finditer(line):
            out.append((a, s + m.end()))
            a = s + m.end()
        if a < e:
            out.append((a, e))
    return [(a + len(text[a:b]) - len(text[a:b].lstrip()), b) for a, b in out if text[a:b].strip()]


def tagged(text, U):
    return "\n".join(f"[S{i + 1}] {text[a:b]}" for i, (a, b) in enumerate(U))


class Mapper:
    def __init__(self, U):
        self.U, self.starts = U, [a for a, _ in U]

    def __call__(self, ev):
        """unit numbers (1-based) an evidence object's spans touch"""
        idx = set()
        for st in (ev or {}).get("sets") or []:
            for sp in st.get("spans") or []:
                a, b = sp["start_char"], sp["end_char"]
                i = max(0, bisect.bisect_right(self.starts, a) - 1)
                while i < len(self.U) and self.U[i][0] < b:
                    if self.U[i][1] > a:
                        idx.add(i + 1)
                    i += 1
        return sorted(idx)


_M = K._MISSING


def _ev(x, excluded, mp, item_excluded=frozenset()):
    """(compact value, evidence shape) of x, in step with compact._compact; `item_excluded`: fields left out of each
    item of a top-level list (compact.to_compact's EXCLUDED)"""
    if K.is_wrapped(x):
        c = K._unwrap(x)
        if c is _M:
            return _M, _M
        idx = mp(x.get("evidence"))
        return c, (idx or _M)
    if isinstance(x, dict):
        cout, eout = {}, {}
        for k, v in x.items():
            if k in K.DROP_KEYS or k in excluded:
                continue
            c, e = _ev(v, frozenset(), mp)
            c = K._clean(c)
            if c is not _M:
                cout[k] = c
                if e is not _M:
                    eout[k] = e
        if eout and isinstance(cout.get("local_id"), str):
            eout = {"local_id": cout["local_id"], **eout}
        return cout, (eout or _M)
    if isinstance(x, list):
        cout, eout = [], []
        for v in x:
            c, e = _ev(v, item_excluded, mp)
            c = K._clean(c)
            if c is not _M:
                cout.append(c)
                eout.append(e if e is not _M else {})
        while eout and eout[-1] == {}:
            eout.pop()
        return cout, (eout or _M)
    return x, _M


def ev_target(record, mp):
    """(compact record, evidence shape); the compact record equals compact.to_compact(record)"""
    cout, eout = {}, {}
    for k, v in record.items():
        if k in K.DROP_KEYS:
            continue
        ex = K.EXCLUDED[K._COLLECTION_DEF[k]] if k in K._COLLECTION_DEF and isinstance(v, list) else frozenset()
        c, e = _ev(v, frozenset(), mp, ex)
        c = K._clean(c)
        if c is not _M and c != []:
            cout[k] = c
            if e is not _M:
                eout[k] = e
    return cout, eout


def pairs(ev, path=""):
    """(slot path, unit) for every unit an evidence shape cites; list items by position"""
    if isinstance(ev, dict):
        for k, v in ev.items():
            if k != "local_id":
                yield from pairs(v, f"{path}.{k}")
    elif isinstance(ev, list):
        if ev and all(isinstance(i, int) for i in ev):
            for i in ev:
                yield path, i
        else:
            for n, v in enumerate(ev):
                yield from pairs(v, f"{path}[{n}]")


def prompt_of(record, text):
    """(the ev1 request's prompt, its units) for a record and the text it was extracted from"""
    U = units(text)
    comp = K.to_compact(record)
    return (f"{INSTR}\n\nText:\n{tagged(text, U)}\n\nRecord:\n"
            f"{json.dumps(comp, ensure_ascii=False, separators=(',', ':'))}"), U


def n_input_tokens(prompt):
    """The request's size as the ns-ev1 shards were cut by"""
    return P.ntok(prompt) + P.ntok(TEMPLATE) + 40


def merge(record, text, prediction):
    """Put ev1's answer into the record, in place: each cited unit becomes a span of the value's
    evidence, `{"status": "present", "sets": [{"source": "retriever", "spans": [...]}]}`. Values cited
    nothing keep their evidence. Citations of a unit past the text's end or of a path the record
    does not have are dropped and counted. -> counts, with `roundtrip` "ok" when `ev_target` on the
    merged record gives back exactly the kept citations."""
    U = units(text)
    slots = []                                   # slot n -> that value's evidence object (in record)
    _, ev = ev_target(record, lambda e: slots.append(e) or [len(slots)])
    where = {path: slots[n - 1] for path, n in pairs(ev)}
    cited = collections.defaultdict(set)
    counts = collections.Counter()
    for path, u in pairs(prediction):
        if path not in where:
            counts["dropped_no_slot"] += 1
        elif not 1 <= u <= len(U):
            counts["dropped_out_of_range"] += 1
        else:
            cited[path].add(u)
    for path, us in cited.items():
        spans = [{"text": text[a:b], "start_char": a, "end_char": b} for a, b in (U[u - 1] for u in sorted(us))]
        e = where[path]
        e.clear()
        e.update(status="present", sets=[{"source": "retriever", "spans": spans}])
    counts.update(values=len(where), cited_values=len(cited), spans=sum(map(len, cited.values())))
    _, back = ev_target(record, Mapper(U))
    got = set(pairs(back))
    counts["roundtrip"] = "ok" if got == {(p, u) for p, us in cited.items() for u in us} else "MISMATCH"
    return dict(counts)
