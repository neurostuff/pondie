"""The old stage-1 parse written as a current CoordinateParse, keyed for one-to-one re-keying of a record.

The records were extracted against the old parse's grouping (`stage1/analyses.json`). The grouping is as good as any
other parse's, so the training input keeps it: each old entry becomes one `ParsedAnalysis`, keyed by study_schema's
own `keys.analysis_key` over its name, space and points. Then every record coordinate set maps to exactly one key.

`build(sid, entries, record, text, identifiers)` returns:
  parse    the CoordinateParse dict (schema `PARSE_SCHEMA_VERSION`)
  keymap   {old record key (`t2#1`, `prose#3` or `text#3`): new key}
  report   what happened to each record set and each old entry
A record set whose old key is not an entry (a set the extractor added from prose) becomes an extra `text` analysis built
from the coordinates its description gives. One with no coordinates anywhere gets no key, so its analyses are not
coordinate-attached.
"""
from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace

from study_schema.keys import analysis_key

PARSE_SCHEMA_VERSION = "0.9.0"
PRODUCER = {"name": "pondie", "version": "astra-distill-migrate-1", "stage": "old-parse",
            "model": None, "prompt_version": None}
SPACES = {"MNI", "TAL", "OTHER"}
KINDS = {"t": "t", "z": "z", "f": "f", "chi2": "chi_square", "chi_square": "chi_square", "x2": "chi_square",
         "d": "d", "g": "g", "r": "r", "beta": "beta", "b": "beta", "p": "p"}
ANCHOR = {"roi": "roi", "seed": "seed", "target": "stimulation_target"}
MEASURES = {"voxels", "mm3", "cm3", "other"}
TRIPLE = re.compile(r"(-?\d+(?:\.\d+)?)\s*[,;/ ]\s*(-?\d+(?:\.\d+)?)\s*[,;/ ]\s*(-?\d+(?:\.\d+)?)")


MAX_MM = 200.0


def plausible(c):
    """A brain coordinate in mm: within ±200 and not a fraction below 0.01 (a p value or a proportion)."""
    return all(abs(v) <= MAX_MM and (v == 0 or abs(v) >= 0.01) for v in c)


def positional_keys(entries):
    """As pondie.formats.parse_keys.positional_keys numbered them for the extraction prompt."""
    ordinals, keys = {}, []
    for e in entries:
        tid = str((e or {}).get("table_id") or "")
        ordinals[tid] = ordinals.get(tid, 0) + 1
        keys.append(f"{'text' if tid == 'prose' else tid}#{ordinals[tid]}")
    return keys


def canon(k):
    return "text#" + k[len("prose#"):] if isinstance(k, str) and k.startswith("prose#") else k


def _space(s):
    s = (s or "").upper()
    return s if s in SPACES else ("OTHER" if s else None)


def _value(v):
    raw = str(v.get("kind") or "")
    kind = KINDS.get(raw.strip().lower().replace("-", "").replace(" ", ""))
    out = {"kind": kind or "other", "value": float(v["value"])}
    if kind is None or raw != kind:
        out["kind_as_printed"] = raw or None
    return out


def _point(p, space):
    c = p.get("coordinates") or []
    if len(c) != 3 or not all(isinstance(x, (int, float)) for x in c):
        return None
    out = {"coordinates": [float(x) for x in c]}
    ps = _space(p.get("space"))
    if ps and ps != space:
        out["space"] = ps
    vals = [_value(v) for v in p.get("values") or [] if isinstance(v.get("value"), (int, float))]
    if vals:
        out["values"] = vals
    if isinstance(p.get("cluster_size"), (int, float)) and p["cluster_size"] >= 0:
        out["cluster_size"] = float(p["cluster_size"])
        if p.get("cluster_measure") in MEASURES:
            out["cluster_measure"] = p["cluster_measure"]
    return out


def _key(name, space, points):
    ns = [SimpleNamespace(coordinates=p["coordinates"], space=p.get("space"),
                          values=[SimpleNamespace(kind=v["kind"], value=v["value"]) for v in p.get("values") or []])
          for p in points]
    return analysis_key(name, space, ns)


def _role(record_role, stage1_role):
    r = record_role or stage1_role
    if r in ANCHOR:
        return "anchor", ANCHOR[r]
    if r in ("result", "anchor", "localization", "reference", "other"):
        return r, None
    if r == "display":                         # dropped from the vocabulary: a figure of the study's own peaks
        return "result", None
    return (r or "result"), None


def desc_points(cs):
    d = cs.get("description")
    t = (d.get("value") if isinstance(d, dict) else d) or ""
    t = t.replace("−", "-").replace("–", "-")
    pts = []
    for m in TRIPLE.finditer(t):
        c = [float(x) for x in m.groups()]
        if not plausible(c):
            c = split_brodmann(c, t[max(0, m.start() - 6):m.start()]) or c
        pts.append(c)
    return [{"coordinates": c} for c in pts if plausible(c)]


def split_brodmann(c, before):
    """A Brodmann area fused onto x, as in "BA=1344,-11,18" (BA 13, x = 44): when "BA" directly precedes the
    triple and x's leading 1-2 digits are an area (1-52) leaving a plausible x, return the coordinate."""
    if (not re.search(r"BA\s*=?\s*$", before, re.I) or not (1000 <= abs(c[0]) < 53000)
            or any(v != int(v) for v in c)):       # Brodmann tables print whole-millimetre coordinates
        return None
    digits = str(int(abs(c[0])))
    for k in (2, 1):
        area, rest = int(digits[:k]), digits[k:]
        if 1 <= area <= 52 and rest and not rest.startswith("0"):
            x = float(rest) * (1 if c[0] > 0 else -1)
            cand = [x, c[1], c[2]]
            if plausible(cand) and abs(x) <= 100:
                return cand
    return None


def resolver(record):
    """record key -> the old-parse positional key it names, or None for a set the extractor made itself.

    Old records spell an old-parse text entry `prose#N`; newer ones spell it `text#N`. A record that uses `prose#`
    keys also holds `text#N` sets of the extractor's own, which name no old-parse entry, so `text#` keys there are
    never resolved to one (a record holding `prose#3` and `text#3` holds two different sets)."""
    keys = {c["local_id"] for c in record.get("coordinate_sets") or []}
    for a in record.get("analyses") or []:
        v = a.get("source_table_analysis")
        if isinstance(v, dict) and isinstance(v.get("value"), str):
            keys.add(v["value"])
    old_spelling = any(k.startswith("prose#") for k in keys)

    def resolve(k):
        if k.startswith("prose#"):
            return "text#" + k[len("prose#"):]
        if k.startswith("text#") and old_spelling:
            return None
        return k
    return resolve


def _val(x):
    return x.get("value") if isinstance(x, dict) else x


def build(sid, entries, record, text, identifiers):
    """See the module docstring. `keymap` is keyed by the record's own keys (sets' local_ids and analyses'
    `source_table_analysis`); `roles` gives the role each record set should carry, matching its parse entry."""
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    resolve = resolver(record)
    rsets = {c["local_id"]: c for c in record.get("coordinate_sets") or []}
    with_cells = set()
    for a in record.get("analyses") or []:
        k = _val(a.get("source_table_analysis"))
        if k and ((a.get("effect") or {}).get("cells")):
            with_cells.add(k)
    s1 = dict(zip(positional_keys(entries), entries))
    set_of = {}                                     # positional key -> the record set naming it
    for k in rsets:
        p = resolve(k)
        if p in s1:
            set_of.setdefault(p, k)
    analyses, s1map, keymap, roles = [], {}, {}, {}
    report = {"entries": 0, "duplicate_keys": [], "sets": {}, "role_changes": []}
    seen = {}

    def role_for(rec_keys, stage1_role):
        rk = next((k for k in rec_keys if k in rsets), None)
        cs = rsets.get(rk) or {}
        role, anchor = _role(_val(cs.get("role")), stage1_role)
        if role == "anchor" and _val(cs.get("anchor_kind")):
            anchor = _val(cs.get("anchor_kind"))
        # coordinates reported with the statistics of an effect are a result (study_schema CoordinateRole: result)
        if role != "result" and any(k in with_cells for k in rec_keys):
            report["role_changes"].append([rk, _val(cs.get("role")), "result", "effect cells"])
            role, anchor = "result", None
        return role, anchor

    def add(rec_keys, name, printed, space, points, origin, table_id, description, stage1_role):
        role, anchor = role_for(rec_keys, stage1_role)
        key = _key(name, space, points)
        if key in seen:
            report["duplicate_keys"].append([seen[key], rec_keys])
            return key
        a = {"key": key, "origin": origin, "table_id": table_id if origin == "table" else None, "name": name,
             "name_is_printed": printed, "description": description, "coordinate_space": space,
             "role": role, "points": points}
        if anchor:
            a["anchor_kind"] = anchor
        analyses.append(a)
        seen[key] = rec_keys
        for k in rec_keys:
            roles[k] = role
        return key

    ana_keys = {_val(a.get("source_table_analysis")) for a in record.get("analyses") or []} - {None}
    report["implausible_entries"] = []
    for pos, e in s1.items():
        report["entries"] += 1
        space = _space(e.get("coordinate_space"))
        pts = [p for p in (_point(p, space) for p in e.get("points") or []) if p]
        if any(not plausible(pt["coordinates"]) for pt in pts):
            # a table's statistics read as coordinates: a misparse, not an analysis (REVIEW-2 B)
            report["implausible_entries"].append([pos, e.get("name"), [pt["coordinates"] for pt in pts][:5]])
            continue
        prose = (e.get("table_id") or "prose") == "prose"
        rec_keys = [k for k in list(rsets) + sorted(ana_keys) if resolve(k) == pos]
        key = add(rec_keys, e.get("name") or e.get("table_caption") or pos, bool(e.get("name")), space, pts,
                  "text" if prose else "table", None if prose else e.get("table_id"), e.get("description"),
                  e.get("role"))
        s1map[pos] = key
        for k in rec_keys:
            keymap[k] = key
    for k, cs in rsets.items():
        if k in keymap:
            report["sets"][k] = "entry"
            continue
        pts = desc_points(cs)
        if not pts:
            report["sets"][k] = "no_points"
            continue
        dtext = _val(cs.get("description")) or k
        keymap[k] = add([k], dtext[:200], False, None, pts, "text", None, dtext, None)
        report["sets"][k] = "from_description"
    parse = {
        "header": {"artifact_kind": "coordinate_parse", "schema_version": PARSE_SCHEMA_VERSION, "article_id": sid,
                   "identifiers": identifiers, "producer": PRODUCER,
                   "inputs": [{"artifact_kind": "parsed_paper", "fingerprint": text_sha, "producer_version": None}],
                   "created_at": None, "neurostore_base_study_id": sid},
        "text_sha256": text_sha, "revision_of": None, "analyses": analyses,
    }
    parse["parse_id"] = hashlib.sha256(json.dumps(parse, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return parse, keymap, report, roles


def identifiers_from(path):
    """ns-pond's identifiers.json as the schema's ArticleIdentifiers."""
    d = json.load(open(path)) if path and path.exists() else {}
    o = d.get("other_ids") or {}

    def many(k):
        v = o.get(k)
        return [v] if isinstance(v, str) else (v or None)

    ids = {"pmid": d.get("pmid"), "pmcid": d.get("pmcid"), "doi": d.get("doi"),
           "alternative_pmids": None, "alternative_pmcids": None, "alternative_dois": None,
           "preprint_dois": many("preprint_doi"), "preprint_pmids": many("preprint_pmid"),
           "preprint_pmcids": many("preprint_pmcid")}
    pre = any(ids[k] for k in ("preprint_dois", "preprint_pmids", "preprint_pmcids"))
    ids["publication_status"] = "preprint" if pre and not any(ids[k] for k in ("doi", "pmid", "pmcid")) \
        else "published"
    return ids
