"""Table caption and footer of each parse entry, from the old stage-1 parse.

`stage1/analyses.json` has `table_caption` and `table_footer` on each entry. The entries become parse entries through
migrate/code/oldparse.py (`positional_keys`, `build`); a stage-1 entry's key is `keymap[positional key]`. Needs
`migrate/code/oldparse.py` and `migrate/study_schema/src` on the path (set below). Truncation as the earlier work:
caption 300 characters, footer 700.
"""
import json
import sys
from pathlib import Path

MIG = Path("/data/james/agents/astra-distill/migrate")
CORPUS = MIG / "in/old/pubget-bundle-20261007/corpus"
CAPTION_CHARS, FOOTER_CHARS = 300, 700
_OP = None


def _oldparse():
    global _OP
    if _OP is None:
        for p in (str(MIG / "study_schema/src"), str(MIG / "code")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import oldparse
        _OP = oldparse
    return _OP


def captions(sid, corpus=CORPUS):
    """{parse key: {"caption", "footer"}} for the entries of one paper that have either (table entries)"""
    f = Path(corpus) / sid / "stage1/analyses.json"
    if not f.exists():
        return {}
    entries = json.load(open(f)).get("analyses") or []
    OP = _oldparse()
    pos = OP.positional_keys(entries)
    fake = {"coordinate_sets": [{"local_id": k} for k in pos], "analyses": []}
    _, keymap, _, _ = OP.build(sid, entries, fake, "", {})
    out = {}
    for k, e in zip(pos, entries):
        key = keymap.get(k)
        cap, foot = (e.get("table_caption") or "").strip(), (e.get("table_footer") or "").strip()
        if key and (cap or foot) and key not in out:
            out[key] = {"caption": cap[:CAPTION_CHARS], "footer": foot[:FOOTER_CHARS]}
    return out
