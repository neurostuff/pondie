"""Migrate one old Astra extraction record (study_schema 74da550) to study_schema main against the new text and parse.

    python migrate.py RUN_DIR --records DIR --old DIR --new DIR [--sids FILE] [--manual FILE] [--workers N]

Inputs per study id:
  --records DIR/<sid>.extraction.json             old record
  --old     DIR/<sid>/stage1/analyses.json        old stage-1 parse (points behind the old positional keys)
            DIR/<sid>/processed/pubget/text.txt   old text (its hash is the record's source_text_hash)
  --new     DIR/<sid>/processed/pubget/text.txt   new formatted text
            DIR/<sid>/parse/coordinate_parse.json new coordinate parse (optional: without it keys wait)
  --manual  JSONL of manual decisions (manual.py) applied on top: span placements and coordinate-set keys

Outputs under RUN_DIR:
  records/<sid>.extraction.json  the migrated record, written only when nothing is left unresolved and it validates
  sidecar/<sid>.json             every change: span placements (old and new offsets, pass, similarity), coordinate-set
                                 matches, role conflicts, dropped outcomes, filled result_reporting
  queue.jsonl                    what needs a manual decision: spans the passes did not place, or placed below
                                 SIM_MIN; coordinate sets without an exact or subset match; empty records'
                                 result_reporting
  flags.jsonl                    role conflicts (old role kept, the parse's role named) and other notices
  summary.json                   counts
Old values are kept wherever the brief says so: the old role, the old coordinate set and its description.
"""
import argparse
import collections
import copy
import hashlib
import json
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from align import Aligner, char_similarity  # noqa: E402
import coords as CO  # noqa: E402
import oldparse as OP  # noqa: E402

SIM_MIN = 0.90
ABSTRACT_MAX = 6000
TEXT_SAME_MIN = 0.80   # share of the old text's 8-word shingles found in the new text (textdiff.py)
SCHEMA_COMMIT = "8fe87a1"
REPAIRED_BY = f"astra-distill-migrate-1 (study_schema {SCHEMA_COMMIT}; after pondie-repair-1)"
TRIPLE = re.compile(r"(-?\d+(?:\.\d+)?)\s*[,;/ ]\s*(-?\d+(?:\.\d+)?)\s*[,;/ ]\s*(-?\d+(?:\.\d+)?)")


def V(x):
    return x.get("value") if isinstance(x, dict) else None


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sections(text):
    """paper_sections from the new text: line 0 is the title (level 0); `#`*n headings are level n-1."""
    heads, pos = [], 0
    for ln in text.splitlines(keepends=True):
        s = ln.strip()
        if not heads and s:
            heads.append((0, s, pos))
        else:
            m = re.match(r"(#{1,6})\s+(.*\S)\s*$", ln)
            if m:
                heads.append((len(m.group(1)) - 1, m.group(2), pos))
        pos += len(ln)
    out = []
    for i, (lvl, title, start) in enumerate(heads):
        end = len(text)
        for lvl2, _, s2 in heads[i + 1:]:
            if lvl2 <= lvl:
                end = s2
                break
        parent = next((t for l2, t, _ in reversed(heads[:i]) if l2 < lvl), None)
        out.append({"ordinal": i, "title": title, "level": lvl, "parent_section": parent,
                    "start_char": start, "end_char": end})
    return out


def iter_spans(x, path=""):
    if isinstance(x, dict):
        ev = x.get("evidence")
        if isinstance(ev, dict):
            for si, s in enumerate(ev.get("sets") or []):
                for pi, sp in enumerate(s.get("spans") or []):
                    yield f"{path}/evidence/sets/{si}/spans/{pi}", sp
        for k, v in x.items():
            if k != "evidence":
                yield from iter_spans(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from iter_spans(v, f"{path}/{i}")


def load_manual(path):
    M = collections.defaultdict(dict)
    if path:
        for ln in open(path):
            d = json.loads(ln)
            M[d["sid"]][(d["kind"], d["item"])] = d
    return M


def desc_points(cs):
    """Coordinates written in an extractor-added set's description (`14, 16, −2`)."""
    t = (V(cs.get("description")) or "").replace("−", "-").replace("–", "-")
    return [tuple(float(v) for v in m.groups()) for m in TRIPLE.finditer(t)]


def candidates(op, new, old_table_id):
    """Every new analysis sharing a point with `op`, best first: exact, then subset, then partial; same origin
    (old prose <-> new text, old table <-> new table, same table id) breaks ties."""
    out = []
    for a in new:
        np_ = CO.pts(a)
        hit = CO._shared(op, np_)
        if not hit:
            continue
        rel = "exact" if hit == len(op) == len(np_) else "subset" if hit == len(op) else \
            "contains" if hit == len(np_) else "partial"
        old_prose = (old_table_id or "prose") == "prose"
        same_origin = (a.get("origin") == "text") == old_prose
        same_table = bool(old_table_id) and a.get("table_id") == old_table_id
        rank = ({"exact": 0, "subset": 1, "contains": 2, "partial": 3}[rel], not same_table, not same_origin,
                -hit)
        out.append({"key": a["key"], "rel": rel, "shared": hit, "new_n": len(np_), "role": a.get("role"),
                    "origin": a.get("origin"), "table_id": a.get("table_id"), "rank": rank})
    out.sort(key=lambda c: c["rank"])
    for c in out:
        c.pop("rank")
    return out


def assign(old_sets, new):
    """old_sets: [(old key, points, old table id)]. One new key per old set, each new key used once.
    exact/subset candidates are taken in order (exact first); a set whose candidates are all taken is a
    collision, one with no exact or subset candidate a non-match. Returns {old key: result}."""
    res, taken = {}, {}
    cand = {k: candidates(op, new, tid) if op else [] for k, op, tid in old_sets}
    order = sorted(old_sets, key=lambda x: (0 if cand[x[0]] and cand[x[0]][0]["rel"] == "exact" else 1,
                                           x[0]))
    for k, op, tid in order:
        cs = cand[k]
        if not op:
            res[k] = {"status": "no_old_points", "candidates": []}
            continue
        ok = [c for c in cs if c["rel"] in ("exact", "subset")]
        free = [c for c in ok if c["key"] not in taken]
        if free:
            c = free[0]
            taken[c["key"]] = k
            res[k] = {"status": c["rel"], "new_keys": [c["key"]], "new_roles": [c["role"]], "old_n": len(op),
                      "new_n": c["new_n"], "candidates": cs[:5]}
        elif ok:
            res[k] = {"status": "collision", "with": [taken[c["key"]] for c in ok], "old_n": len(op),
                      "candidates": cs[:5]}
        else:
            contains = [c for c in cs if c["rel"] == "contains"]
            if contains and sum(c["shared"] for c in contains) >= len(op):
                st = "split"
            else:
                st = "partial" if cs else "none"
            res[k] = {"status": st, "old_n": len(op),
                      "candidates": [c for c in cs if c["rel"] == "contains"] + [c for c in cs if c["rel"] != "contains"][:5]}
    return res


def abstract_only(text):
    """The new text holds only a title and abstract: no section heading and under ABSTRACT_MAX characters."""
    return len(text) < ABSTRACT_MAX and not re.search(r"(?m)^#{1,6} ", text)


NAME_AGREE = 0.5


def OP_norm(name):
    from study_schema.keys import normalize_name
    return normalize_name(name or "")


def _name_overlap(a, b):
    """Share of the shorter name's words (3+ letters) that the other name holds."""
    wa, wb = (set(re.findall(r"[a-z0-9]{3,}", x.casefold())) for x in (a, b))
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


def coords_from_old_parse(sid, rec, sets, O, N, new_text, side, out, cfg):
    """Re-key every coordinate set to the old parse written as a current CoordinateParse (oldparse.py), one to one.
    Writes RUN/parse/<sid>/coordinate_parse.json.
    - A set with no coordinates anywhere is dropped, and its analyses lose their coordinate link.
    - An analysis naming an old-parse entry the record has no set for gets a set built from the entry.
    - A set's role is set to its parse entry's: `display` and roles of sets whose analyses carry effect cells become
      `result` (study_schema: coordinates reported with an effect's statistics are a result)."""
    st1_p = O / sid / "stage1/analyses.json"
    entries = (json.load(open(st1_p)).get("analyses") or []) if st1_p.exists() else []
    ids = OP.identifiers_from(N / sid / "identifiers.json")
    parse, keymap, report, roles = OP.build(sid, entries, rec, new_text, ids)
    side["old_parse"] = {k: report[k] for k in ("entries", "duplicate_keys", "sets", "role_changes",
                                                 "implausible_entries")}
    out["counts"]["parse_implausible_entries"] += len(report["implausible_entries"])
    side["old_parse"]["parse_id"] = parse["parse_id"]
    pdir = pathlib.Path(cfg["run"]) / "parse" / sid
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / "coordinate_parse.json", "w") as f:
        json.dump(parse, f, ensure_ascii=False)
    out["counts"]["parse_entries"] += report["entries"]
    out["counts"]["parse_duplicate_keys"] += len(report["duplicate_keys"])
    out["counts"]["set_role_changed"] += len(report["role_changes"])
    first, keep = {}, []
    for cs in sets:
        k = cs["local_id"]
        out["counts"][f"set_{report['sets'].get(k)}"] += 1
        nk = keymap.get(k)
        if nk is None:
            side.setdefault("dropped_sets", []).append({"old_key": k, "reason": "no coordinates", "removed_set": cs})
            continue
        if nk in first:                     # identical name, space and points: one analysis in the parse
            side.setdefault("merged_sets", []).append({"old_key": k, "into": first[nk], "removed_set": cs})
            out["counts"]["set_merged_identical"] += 1
            continue
        first[nk] = k
        if roles.get(k) and V(cs.get("role")) != roles[k]:
            side.setdefault("role_set", []).append({"old_key": k, "from": V(cs.get("role")), "to": roles[k]})
            # the old role's evidence argued for the old role: the new one is derived, with no span of its own
            cs["role"] = {"extraction_status": "extracted", "value": roles[k], "value_source": "generated",
                          "evidence": {"status": "not_found"}}
            if roles[k] != "anchor":
                cs.pop("anchor_kind", None)
        cs["local_id"] = nk
        keep.append(cs)
    have = {cs["local_id"] for cs in keep}
    for a in rec.get("analyses") or []:
        sta = a.get("source_table_analysis")
        v = V(sta) if isinstance(sta, dict) else None
        if v is None:
            continue
        nk = keymap.get(v)
        if nk is None:
            a["source_table_analysis"] = {"extraction_status": "not_reported", "unreported_reason": "undetermined",
                                          "evidence": {"status": "not_applicable"}}
            side.setdefault("detached_analyses", []).append({"analysis": a.get("local_id"), "old_key": v})
            out["counts"]["analysis_detached"] += 1
            continue
        if nk not in have:                  # the record named an old-parse entry without writing its set
            entry = next(e for e in parse["analyses"] if e["key"] == nk)
            # Astra sometimes names the wrong entry here (2mETSYuFa3mu: a PANSS analysis on the HAMD entry), and with
            # no set of its own nothing else confirms it: create the set only when the names agree
            twin = next((t for t in rec.get("analyses") or [] if t.get("local_id") == a.get("mirror_of")), None)
            twin_entry = None
            if twin is not None:
                tk = keymap.get(V(twin.get("source_table_analysis")) or "")
                twin_entry = next((e for e in parse["analyses"] if e["key"] == tk), None)
            # a sign-split mirror reads its twin's entry: one naming another entry is mis-keyed
            mirror_off = twin is not None and (twin_entry is None or
                                               OP_norm(twin_entry["name"]) != OP_norm(entry["name"]))
            if mirror_off or _name_overlap(V(a.get("name")) or "", entry["name"]) < NAME_AGREE:
                a["source_table_analysis"] = {"extraction_status": "not_reported",
                                              "unreported_reason": "undetermined",
                                              "evidence": {"status": "not_applicable"}}
                side.setdefault("detached_analyses", []).append(
                    {"analysis": a.get("local_id"), "old_key": v, "reason": "names an entry without a set, and "
                     "its name disagrees with the entry's", "entry_name": entry["name"]})
                out["counts"]["analysis_detached_name_disagrees"] += 1
                continue
        sta["value"] = nk
        if nk not in have:
            entry = next(e for e in parse["analyses"] if e["key"] == nk)
            keep.append({"local_id": nk, "role": {"extraction_status": "extracted", "value": entry["role"],
                                                  "value_source": "generated", "evidence": {"status": "not_found"}},
                         "analysis": a.get("local_id")})
            have.add(nk)
            side.setdefault("created_sets", []).append({"key": nk, "old_key": v, "analysis": a.get("local_id")})
            out["counts"]["set_created_for_analysis"] += 1
    rec["coordinate_sets"] = keep
    return keymap


def migrate_one(args):
    sid, cfg = args
    R, O, N = (pathlib.Path(cfg[k]) for k in ("records", "old", "new"))
    manual = cfg["manual"].get(sid, {})
    out = {"sid": sid, "queue": [], "flags": [], "counts": collections.Counter()}
    rec_p = R / f"{sid}.extraction.json"
    new_txt_p = N / sid / "processed/pubget/text.txt"
    if not new_txt_p.exists():
        out["status"] = "no_new_text"
        return out
    new_head = new_txt_p.read_text()
    if abstract_only(new_head):
        out["status"] = "excluded_abstract_only"
        out["flags"].append({"sid": sid, "kind": "abstract_only", "new_len": len(new_head)})
        return out
    same = cfg.get("textdiff", {}).get(sid)
    if same is not None and same < TEXT_SAME_MIN:
        # the new text is a different document (another version of the paper, or a partial fetch): the record's
        # evidence was read from text the training input no longer contains, so it is not migrated span by span
        out["status"] = "excluded_text_changed"
        out["flags"].append({"sid": sid, "kind": "text_changed", "old_shingles_in_new": round(same, 3)})
        return out
    old = json.load(open(rec_p))
    rec = copy.deepcopy(old)
    new_text = new_txt_p.read_text()
    old_txt_p = O / sid / "processed/pubget/text.txt"
    old_text = old_txt_p.read_text() if old_txt_p.exists() else ""
    side = {"sid": sid, "schema_commit": SCHEMA_COMMIT, "old_text_hash": V(None) or old.get(
        "extraction_metadata", {}).get("source_text_hash"), "new_text_hash": sha(new_text), "spans": [],
            "coordinate_sets": {}, "role_conflicts": [], "dropped_outcomes": [], "result_reporting": None}
    if old_text and sha(old_text) != side["old_text_hash"]:
        out["flags"].append({"sid": sid, "kind": "old_text_hash_mismatch"})

    # 1. evidence spans
    al = Aligner(new_text, len(old_text) or 1)
    unalignable = []
    extra = []   # (span, new span) to insert after it: one old quote now spread over separate passages
    cache = {}
    for path, sp in iter_spans(rec):
        t = sp.get("text") or ""
        key = (t, sp.get("start_char"))
        if key not in cache:
            cache[key] = al.place(t, sp.get("start_char"))
        r = cache[key]
        md = manual.get(("span", path))
        entry = {"path": path, "old": {"start_char": sp.get("start_char"), "end_char": sp.get("end_char"),
                                       "text": t}}
        if md and md.get("decision") == "absent":
            unalignable.append(path)
            continue
        if md and md.get("decision") == "place":
            rng = md.get("ranges") or [[md["start_char"], md["end_char"]]]
            s, e = rng[0]
            for s2, e2 in rng[1:]:
                extra.append((sp, {"text": new_text[s2:e2], "start_char": s2, "end_char": e2}))
            entry.update(method="manual", similarity=round(char_similarity(t, new_text[s:e]), 3),
                         reviewer=md.get("reviewer"), note=md.get("note"))
        elif r is not None and (r[2] in ("exact", "canonical", "despaced")
                                or char_similarity(t, new_text[r[0]:r[1]]) >= SIM_MIN):
            s, e = r[0], r[1]
            entry.update(method=r[2], similarity=round(char_similarity(t, new_text[s:e]), 3))
        else:
            cand = None if r is None else {"start_char": r[0], "end_char": r[1], "method": r[2],
                                           "similarity": round(char_similarity(t, new_text[r[0]:r[1]]), 3)}
            out["queue"].append({"sid": sid, "kind": "span", "item": path, "text": t,
                                 "old_start_char": sp.get("start_char"), "candidate": cand})
            out["counts"]["span_queued"] += 1
            continue
        sp["start_char"], sp["end_char"], sp["text"] = s, e, new_text[s:e]
        entry["new"] = {"start_char": s, "end_char": e}
        side["spans"].append(entry)
        out["counts"][f"span_{entry['method']}"] += 1

    if extra:
        def _insert(x):
            if isinstance(x, dict):
                ev = x.get("evidence")
                if isinstance(ev, dict):
                    for st in ev.get("sets") or []:
                        sp_ = st.get("spans") or []
                        for orig, add in extra:
                            for i, cur in enumerate(sp_):
                                if cur is orig:
                                    sp_.insert(i + 1, add)
                                    break
                for k, v in x.items():
                    if k != "evidence":
                        _insert(v)
            elif isinstance(x, list):
                for v in x:
                    _insert(v)
        _insert(rec)
        out["counts"]["span_manual_extra_ranges"] += len(extra)
    if unalignable:
        # a reviewer found the evidence absent from the new text: the paper is kept apart, in neither training
        # nor testing (user, 2026-10-23)
        out["status"] = "excluded_unalignable"
        out["flags"].append({"sid": sid, "kind": "unalignable_spans", "n": len(unalignable)})
        out["sidecar"] = side
        return out

    # 2. coordinate sets
    sets = rec.get("coordinate_sets") or []
    if cfg.get("parse_mode") == "old":
        keymap = coords_from_old_parse(sid, rec, sets, O, N, new_text, side, out, cfg)
        sets = rec.get("coordinate_sets") or []
    else:
        cp_p = N / sid / "parse/coordinate_parse.json"
        st1_p = O / sid / "stage1/analyses.json"
        sets = rec.get("coordinate_sets") or []
        ana_keys = {V(a.get("source_table_analysis") or {}) for a in rec.get("analyses") or []} - {None}
        keymap = {}
        if (sets or ana_keys) and not cp_p.exists():
            out["queue"].append({"sid": sid, "kind": "awaiting_parse", "item": "coordinate_sets",
                                 "n_sets": len(sets)})
            out["counts"]["sets_awaiting_parse"] += len(sets)
        elif sets or ana_keys:
            new = json.load(open(cp_p)).get("analyses") or []
            st1 = (json.load(open(st1_p)).get("analyses") or []) if st1_p.exists() else []
            st1_pts = dict(zip(CO.positional_keys(st1), (CO.pts(e) for e in st1)))
            by_key = {a["key"]: a for a in new}
            old_sets = []
            for cs in sets:
                ok_ = CO.canon_key(cs["local_id"])
                op, src = st1_pts.get(ok_), "old_parse"
                if op is None:
                    op, src = desc_points(cs), "description"
                tid = next((e.get("table_id") for k_, e in zip(CO.positional_keys(st1), st1) if k_ == ok_), None)
                if tid is None and src == "description":
                    tid = "prose"
                old_sets.append((cs["local_id"], op, tid, src))
            res = assign([(k, op, tid) for k, op, tid, _ in old_sets], new)
            role_old = {c["local_id"]: V(c.get("role")) for c in sets}
            merges, splits, nomatch = {}, {}, {}
            for k, op, tid, src in old_sets:
                m = res[k]
                m["points_from"] = src
                md = manual.get(("coordinate_set", k))
                if md and md.get("decision") == "map":
                    m.update(status="manual", new_keys=[md["new_key"]], reviewer=md.get("reviewer"),
                             note=md.get("note"))
                cs = next(c for c in sets if c["local_id"] == k)
                side["coordinate_sets"][k] = m
                out["counts"][f"set_{m['status']}"] += 1
                if md and md.get("decision") == "no_match":
                    nomatch[k] = md
                    m["status"] = "no_match_reviewed"
                    out["counts"]["set_no_match_reviewed"] += 1
                elif m["status"] in ("exact", "subset", "manual"):
                    keymap[k] = m["new_keys"][0]
                elif m["status"] == "split" and manual.get(("coordinate_set", k), {}).get("decision") != "no_match":
                    pieces = [c["key"] for c in m.get("candidates") or [] if c["rel"] == "contains"]
                    used = set(keymap.values()) | {w for v in splits.values() for w in v}
                    if pieces and not used & set(pieces):
                        splits[k] = pieces
                        m["status"] = "split_copied"
                        out["counts"]["set_split_copied"] += 1
                    else:
                        out["queue"].append({"sid": sid, "kind": "coordinate_set", "item": k, "match": m,
                                             "description": V(cs.get("description")), "old_role": V(cs.get("role")),
                                             "old_table_id": tid, "why": "split piece already used"})
                elif m["status"] == "collision" and all(role_old.get(w) == V(cs.get("role")) for w in m["with"]):
                    merges[k] = m["with"][0]
                    m["status"] = "merged"
                    out["counts"]["set_merged"] += 1
                else:
                    out["queue"].append({"sid": sid, "kind": "coordinate_set", "item": k, "match": m,
                                         "description": V(cs.get("description")), "old_role": V(cs.get("role")),
                                         "old_table_id": tid})
            # split: one coordinate set and one copy of each analysis per piece, so no old point is lost
            for k, pieces in splits.items():
                src = next(c for c in sets if c["local_id"] == k)
                anas = [a for a in rec.get("analyses") or [] if V(a.get("source_table_analysis") or {}) == k]
                new_sets, copies = [], []
                for n, pk in enumerate(pieces, 1):
                    ns = copy.deepcopy(src)
                    ns["local_id"] = pk
                    for a in anas:
                        ca = copy.deepcopy(a)
                        ca["local_id"] = f"{a.get('local_id')}__part{n}"
                        ca["source_table_analysis"]["value"] = pk
                        copies.append(ca)
                        if V(ns.get("analysis") if isinstance(ns.get("analysis"), dict) else None) is None and \
                                ns.get("analysis") == a.get("local_id"):
                            ns["analysis"] = ca["local_id"]
                    new_sets.append(ns)
                side["coordinate_sets"][k]["split_into"] = {"new_keys": pieces,
                                                            "analysis_copies": [c["local_id"] for c in copies],
                                                            "replaced_analyses": [a.get("local_id") for a in anas]}
                rec["analyses"] = [a for a in rec.get("analyses") or [] if a not in anas] + copies
                sets.extend(new_sets)
                for pk in pieces:
                    keymap[pk] = pk
                for a in anas:
                    for other in rec.get("analyses") or []:
                        if other.get("mirror_of") == a.get("local_id"):
                            out["queue"].append({"sid": sid, "kind": "analysis_key", "item": other.get("local_id"),
                                                 "old_key": k, "why": "mirror_of an analysis that was split"})
            sets = [c for c in sets if c["local_id"] not in splits]
            # reviewed: the new parse has no counterpart. The set is dropped and its analyses lose their coordinate link,
            # and the record is excluded from coordinate-linked training (its input cannot support the label)
            for k, md in nomatch.items():
                gone = next(c for c in sets if c["local_id"] == k)
                anas = [a.get("local_id") for a in rec.get("analyses") or []
                        if V(a.get("source_table_analysis") or {}) == k]
                for a in rec.get("analyses") or []:
                    if a.get("local_id") in anas:
                        a["source_table_analysis"] = {"extraction_status": "not_reported",
                                                      "unreported_reason": "undetermined",
                                                      "evidence": {"status": "not_applicable"}}
                side.setdefault("training_exclusions", []).append(
                    {"old_key": k, "reason": "no counterpart in the new coordinate parse", "note": md.get("note"),
                     "analyses": anas, "removed_set": copy.deepcopy(gone)})
            sets = [c for c in sets if c["local_id"] not in nomatch]
            for k, twin in merges.items():
                keymap[k] = keymap[twin]
                gone = next(c for c in sets if c["local_id"] == k)
                side["coordinate_sets"][k]["merged_into"] = {"old_key": twin, "new_key": keymap[twin],
                                                             "removed_set": copy.deepcopy(gone)}
            rec["coordinate_sets"] = sets = [c for c in sets if c["local_id"] not in merges]
            # analyses naming an old parse entry the record has no coordinate set for: map by that entry's points;
            # no set is created, so a key another set holds may be shared
            for k in sorted(ana_keys - set(keymap) - {k for k, *_ in old_sets}):
                op = st1_pts.get(CO.canon_key(k))
                md = manual.get(("analysis_key", k))
                if md and md.get("decision") == "map":
                    keymap[k] = md["new_key"]
                    side["coordinate_sets"][k] = {"status": "manual", "analysis_only": True, "new_keys": [md["new_key"]]}
                    continue
                cs_ = candidates(op, new, next((e.get("table_id") for k_, e in zip(CO.positional_keys(st1), st1)
                                                if k_ == CO.canon_key(k)), None)) if op else []
                ok = [c for c in cs_ if c["rel"] in ("exact", "subset")]
                side["coordinate_sets"][k] = {"status": ok[0]["rel"] if ok else ("none" if op else "no_old_points"),
                                              "analysis_only": True, "candidates": cs_[:5],
                                              "new_keys": [ok[0]["key"]] if ok else []}
                out["counts"]["analysis_key_" + side["coordinate_sets"][k]["status"]] += 1
                if ok:
                    keymap[k] = ok[0]["key"]
            for cs in sets:
                nk = keymap.get(cs["local_id"])
                if nk is None:
                    continue
                old_role = V(cs.get("role"))
                new_role = by_key[nk].get("role")
                if old_role and new_role and old_role != new_role:
                    c = {"sid": sid, "kind": "role_conflict", "old_key": cs["local_id"], "new_key": nk,
                         "old_role": old_role, "parse_role": new_role,
                         "parse_role_confidence": by_key[nk].get("role_confidence"),
                         "parse_role_source": by_key[nk].get("role_source")}
                    out["flags"].append(c)
                    side["role_conflicts"].append(c)
                if old_role == "display":
                    out["flags"].append({"sid": sid, "kind": "role_not_in_vocabulary", "old_key": cs["local_id"],
                                         "new_key": nk, "old_role": old_role, "parse_role": new_role})
                cs["local_id"] = nk
            for a in rec.get("analyses") or []:
                sta = a.get("source_table_analysis")
                if isinstance(sta, dict) and V(sta) is not None:
                    if V(sta) in keymap:
                        sta["value"] = keymap[V(sta)]
                    elif any(q.get("item") == V(sta) for q in out["queue"]):
                        pass  # waits with its set
                    else:
                        out["queue"].append({"sid": sid, "kind": "analysis_key", "item": a.get("local_id"),
                                             "old_key": V(sta)})


    # 3. outcome only on result sets (old roles)
    role_of = {cs["local_id"]: V(cs.get("role")) for cs in sets}
    for a in rec.get("analyses") or []:
        k = V(a.get("source_table_analysis") or {})
        role = role_of.get(k)
        if role and role != "result" and V(a.get("outcome") or {}) is not None:
            side["dropped_outcomes"].append({"analysis": a.get("local_id"), "set_role": role,
                                             "outcome": copy.deepcopy(a["outcome"])})
            a["outcome"] = {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}}
            out["counts"]["outcome_dropped"] += 1

    # 3b. stimulus content shared by every condition is the task's (study_schema: Task.stimulus_content holds what
    # the stimuli were about across all conditions; Condition.stimulus_content is left unset when it does not tell
    # the conditions apart). A single condition's content cannot tell anything apart, so it moves too.
    for t in rec.get("tasks") or []:
        conds = t.get("conditions") or []
        vals = [tuple(sorted(V(c.get("stimulus_content")) or [])) for c in conds]
        if not conds or not vals[0] or len(set(vals)) != 1:
            continue
        if V(t.get("stimulus_content")):
            continue
        sets = [st for c in conds for st in ((c.get("stimulus_content") or {}).get("evidence") or {}).get("sets") or []]
        first = conds[0]["stimulus_content"]
        t["stimulus_content"] = {"extraction_status": "extracted", "value": list(V(first)),
                                 "value_source": first.get("value_source") or "reported",
                                 "evidence": {"status": "present", "sets": sets} if sets else {"status": "not_found"}}
        for c in conds:
            c["stimulus_content"] = {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}}
        side.setdefault("stimulus_content_moved", []).append({"task": t.get("local_id"), "value": list(V(first)),
                                                               "conditions": len(conds)})
        out["counts"]["task_stimulus_content_moved"] += 1

    # 4. result_reporting
    an = rec.get("analyses") or []
    md = manual.get(("result_reporting", "study"))
    if md:
        rec["result_reporting"] = md["value"]
        side["result_reporting"] = "manual"
    elif an:
        first = next((a for a in an if (a.get("name") or {}).get("evidence", {}).get("sets")), None)
        ev = copy.deepcopy(first["name"]["evidence"]) if first else {"status": "not_found"}
        rec["result_reporting"] = {"extraction_status": "extracted", "value": "analyses_reported",
                                   "value_source": "generated", "evidence": ev}
        side["result_reporting"] = "derived_from_analyses"
    else:
        out["queue"].append({"sid": sid, "kind": "result_reporting", "item": "study"})

    # 5. metadata
    meta = rec.setdefault("extraction_metadata", {})
    meta["source_text_hash"] = sha(new_text)
    meta["paper_sections"] = sections(new_text)
    meta["repaired_by"] = REPAIRED_BY

    out["record"] = rec
    out["sidecar"] = side
    out["status"] = "queued" if out["queue"] else "done"
    return out


_VAL = None


def run_one(args):
    """Migrate, validate and write one study in the worker; return only its status and counts.
    Writes RUN/sidecar/<sid>.json (with its queue items and flags), and RUN/records/<sid>.extraction.json when done.
    The parent never holds a record: results that arrive out of order used to pile up there (OOM, 2026-10-23)."""
    global _VAL
    sid, cfg = args
    run = pathlib.Path(cfg["run"])
    if _VAL is None:
        from jsonschema import Draft202012Validator
        _VAL = Draft202012Validator(json.load(open(cfg["schema"])))
    try:
        o = migrate_one((sid, cfg))
    except Exception:
        import traceback
        o = {"sid": sid, "status": "error", "queue": [], "flags": [], "counts": {},
             "sidecar": {"sid": sid, "error": traceback.format_exc()}}
    st = o["status"]
    side = o.get("sidecar") or {"sid": sid}
    if "record" in o:
        errs = list(_VAL.iter_errors(o["record"]))
        if errs:
            st = "invalid"
            side["validation_errors"] = [f"/{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}"
                                         for e in errs[:20]]
        if st == "done":
            with open(run / "records" / f"{sid}.extraction.json", "w") as f:
                json.dump(o["record"], f, ensure_ascii=False)
    side["status"], side["queue"], side["flags"] = st, o["queue"], o["flags"]
    side["counts"] = dict(o["counts"])
    tmp = run / "sidecar" / f".{sid}.json.tmp"
    with open(tmp, "w") as f:
        json.dump(side, f)
    tmp.rename(run / "sidecar" / f"{sid}.json")
    return st


def collect(run):
    """queue.jsonl, flags.jsonl and summary.json from the sidecars, one file at a time."""
    C, S = collections.Counter(), collections.Counter()
    with open(run / "queue.jsonl", "w") as fq, open(run / "flags.jsonl", "w") as ff:
        for p in sorted((run / "sidecar").glob("*.json")):
            side = json.load(open(p))
            S[side.get("status")] += 1
            C.update(side.get("counts") or {})
            for q in side.get("queue") or []:
                fq.write(json.dumps(q) + "\n")
                C[f"queue_{q['kind']}"] += 1
            for f in side.get("flags") or []:
                ff.write(json.dumps(f) + "\n")
                C[f"flag_{f['kind']}"] += 1
    summary = {"records": dict(S), "counts": dict(sorted(C.items()))}
    json.dump(summary, open(run / "summary.json", "w"), indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--records", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--sids")
    ap.add_argument("--manual")
    ap.add_argument("--schema", required=True, help="extraction-record.schema.json from study_schema main")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--redo", action="store_true", help="re-run studies that already have a sidecar")
    ap.add_argument("--parse", choices=("old", "new"), default="old",
                    help="old: write the old stage-1 parse as a CoordinateParse and key records to it")
    ap.add_argument("--textdiff", help="textdiff.jsonl: studies whose new text is a different document")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    (run / "records").mkdir(parents=True, exist_ok=True)
    (run / "sidecar").mkdir(exist_ok=True)
    sids = ([s.strip() for s in open(a.sids) if s.strip()] if a.sids else
            sorted(p.name[: -len(".extraction.json")] for p in pathlib.Path(a.records).glob("*.extraction.json")))
    if not a.redo:
        sids = [s for s in sids if not (run / "sidecar" / f"{s}.json").exists()]
    td = {}
    if a.textdiff:
        for ln in open(a.textdiff):
            d = json.loads(ln)
            td[d["sid"]] = d["old_in_new"]
    cfg = {"records": a.records, "old": a.old, "new": a.new, "manual": load_manual(a.manual), "textdiff": td, "parse_mode": a.parse,
           "run": str(run), "schema": a.schema}
    print(f"{len(sids)} studies to run", flush=True)
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for st in ex.map(run_one, ((s, cfg) for s in sids), chunksize=4):
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(sids)}", flush=True)
    print(json.dumps(collect(run), indent=1))


if __name__ == "__main__":
    main()
