"""Normalise the Cognitive Atlas task list into seed categories.

The target half of `normalization.task`: it seeds against what this builds, then clusters
whatever the seeds do not name. Hand-curated where no rule works; the curated lists are
short and named below. The four error classes the rules exist to avoid -- negation, junk
parents, wrong parent, eponyms -- are worked through in docs/cognitive-atlas-seeds.md.

    python -m pondie.normalization.atlas
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from pondie import paths

#: The Atlas dump this normalises. A default rather than a required argument: every caller
#: wants the same file, and `build()` stays overridable for a test fixture.
ATLAS = paths.VOCAB / "cognitiveatlas-task.json"

GENERIC_TOK = {"fmri","mri","task","tasks","paradigm","paradigms","test","tests","scan",
 "scanning","imaging","functional","version","modified","study","protocol","related",
 "based","trials","trial","block","blocks","event","design","procedure","experiment",
 "session","runs","run","combining","using","the","a","an","of","and","with","for"}

#: Auto-generated Atlas entries: a generic head with the apparatus bolted on. Never a seed.
#: Tested against the ORIGINAL label, before `normalise_label` removes the scanner word --
#: after that the pattern would no longer be there to recognise.
JUNK_PARENT = re.compile(r"f?mri task paradigm|f?mri tasks$|f?mri paradigm$", re.I)

#: Dropped by name. `motor fMRI task paradigm` is a generic head that attracted
#: `Motor Screening Task`, `motor sequencing task` and the Beery-Buktenica visual-motor
#: test, none of which are one paradigm.
DROP_BY_NAME = {"motor fMRI task paradigm"}

#: The scanner is not part of a task's name. `Go-NoGo fMRI paradigm` and `social bargaining
#: fMRI task` name the same paradigms as `Go-NoGo paradigm` and `social bargaining task`;
#: carrying `fMRI` into a category label just repeats what every record in this corpus is.
SCANNER = re.compile(r"\b(f?mri|functional magnetic resonance imaging)\b", re.I)


def normalise_label(name: str) -> str:
    """The label a category should carry: the Atlas's name without the scanner word."""
    out = re.sub(r"\s{2,}", " ", SCANNER.sub(" ", name)).strip(" -,:")
    return out or name

#: A variant that NEGATES its parent is a different task.
NEGATED = re.compile(r"^(non|anti|un|in|no)[- ]", re.I)

#: Eponyms and place names: the extra token names an instrument, not a variant.
EPONYM = {"iowa","cambridge","penn","penns","benton","hayling","warrington","california",
 "american","wisconsin","wechsler","wais","wasi","eriksen","simon","posner","stockings",
 "toolbox","nih","salthouse","babcock","boston","rey","osterrieth","hopkins","corsi",
 "beery","buktenica","uznadze","stockings","catoon","comprehensive","early","wcst","ravlt"}

#: Paradigms I judge distinct from the parent the containment rule proposed, after reading
#: every collapse. Curated because there is no rule: `Space Fortress with Oddball` uses an
#: oddball but is the Space Fortress paradigm, and nothing in the string says so.
KEEP_SEPARATE = {"Space Fortress with Oddball", "Biological Motion Perception (Passive Viewing) Paradigm",
 "Continuous Tapping Task", "finger tapping task", "Test of Early Language Development",
 "dual sensitization", "self ordered pointing task", "Comprehensive Test of Phonological Processing",
 "CAToon (cognitive and affective Theory of Mind Cartoon Task)", "Motor Screening Task",
 "Manipulation of predictability and acceptability", "dual-task weather prediction"}

#: Whole labels too generic to seed anything.
GENERIC_WHOLE = {"maze","gating","drawing","vigilance","whistling","faces","recall test",
 "encoding task","semantic task","naming tasks","orientation test","reading","writing",
 "counting","tapping task","imitation","observation","planning","learning","video games",
 "decision making","judgment","discrimination","ataxia","time wall","cups task","shift task",
 "drawing from memory task","reading covert","reading overt","counting calculation",
 "eating drinking","dimensions task","categorization task","reaction time","motion processing"}
INSTRUMENT = re.compile(r"(scale|inventory|questionnaire|battery|system|index|profile|"
                        r"schedule|interview|checklist|survey|form)s?$", re.I)
COMPOUND = re.compile(r"\b(combining|combined with|followed by)\b", re.I)


def fold(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def core(s): return tuple(w for w in fold(s).split() if w not in GENERIC_TOK)
def sq(c): return "".join(c)
def build(path: Path = ATLAS):
    raw = json.load(open(path))
    ca = sorted({e["name"].strip() for e in raw if e.get("name")})
    # The Atlas carries an `alias` field on 259 of its 857 entries and it is where the
    # synonyms live: `temporal discounting task` aliases `delay discounting task`,
    # `balloon analogue risk task` aliases `BART`, `pavlovian conditioning task` aliases
    # `classical conditioning task`. Nothing was reading it.
    aliases = collections.defaultdict(set)
    for e in raw:
        name = (e.get("name") or "").strip()
        for a in (e.get("alias") or "").split(","):
            a = a.strip()
            if name and a and fold(a) != fold(name):
                aliases[name].add(a)
    kept_raw = [n for n in ca if not (
        n in DROP_BY_NAME or fold(n) in GENERIC_WHOLE or INSTRUMENT.search(n)
        or COMPOUND.search(n) or JUNK_PARENT.search(n)
        or len(fold(n).replace(" ", "")) <= 6 or len(core(n)) > 6)]
    # Filter on the original, then normalise. `renamed` keeps the Atlas's own spelling so a
    # mapping stays traceable to the term it came from.
    renamed = {n: normalise_label(n) for n in kept_raw if normalise_label(n) != n}
    cand = [normalise_label(n) for n in kept_raw]

    bysq = collections.defaultdict(list)
    for n in cand:
        if core(n): bysq[sq(core(n))].append(n)
    canon = {s: min(g, key=lambda x: (len(x), x)) for s, g in bysq.items()}
    equal = {n: canon[sq(core(n))] for g in bysq.values() for n in g
             if n != canon[sq(core(n))]}

    parents = list(canon.values())
    collapse, kept = {}, []
    for b in parents:
        if b in KEEP_SEPARATE:
            continue
        cb = core(b)
        # EARLIEST in the child, then longest. Shortest-first order sent `Motor Selective
        # Stop Signal Task` to a motor label; longest-only left `Stop signal task with dot
        # motion discrimination` on `dot motion task`, because both candidate cores are two
        # tokens. The paradigm is named first and the qualifier follows it, so position
        # breaks the tie -- the same rule the ONVOC `contains` layer needed.
        best, rank = None, None
        for a in parents:
            if a is b:
                continue
            ca_ = core(a)
            if len(ca_) >= len(cb):
                continue
            at = next((k for k in range(len(cb) - len(ca_) + 1)
                       if cb[k:k + len(ca_)] == ca_), None)
            if at is None:
                if not (len(sq(ca_)) >= 6 and sq(ca_) in sq(cb)
                        and any(w.startswith(ca_[0]) for w in cb)):
                    continue
                at = len(cb)          # a squashed hit has no token position; rank it last
            here = (at, -len(ca_))
            if rank is None or here < rank:
                best, rank = a, here
        if best is None:
            continue
        extra = set(cb) - set(core(best))
        if extra & EPONYM or NEGATED.match(b):
            kept.append((b, best))
        else:
            collapse[b] = best

    seeds = [n for n in parents if n not in collapse]
    # aliases follow the label through renaming and collapsing, onto the surviving seed
    resolved = collections.defaultdict(set)
    for original, alts in aliases.items():
        label = renamed.get(original, original)
        label = equal.get(label, label)
        seen = set()
        while label in collapse and label not in seen:
            seen.add(label); label = collapse[label]
        if label in seeds:
            resolved[label] |= alts
    return {"all": ca, "seeds": sorted(seeds), "equal": equal, "renamed": renamed,
            "collapse": collapse, "kept": kept,
            "aliases": {k: sorted(v) for k, v in resolved.items()}}


def report(path: Path = ATLAS) -> str:
    """What the rules above did to the Atlas list, and where eight probes landed.

    The probes are regression cases, each one a failure the rules were changed to fix:
    `Go-NoGo fMRI paradigm` carried the scanner word, `Motor Selective Stop Signal Task`
    went to a motor label under shortest-first re-parenting, `Stop signal task with dot
    motion discrimination` went to `dot motion task` under longest-only.
    """

    out = build(path)
    lines = [
        f"aliases carried onto {len(out['aliases'])} seeds",
        f"{len(out['all'])} labels -> {len(out['seeds'])} seeds "
        f"| method-word dupes {len(out['equal'])} | variants {len(out['collapse'])} "
        f"| kept separate {len(out['kept'])} "
        f"| scanner word removed from {len(out['renamed'])}",
    ]
    lines += [f"   {a[:48]:50s} -> {b}" for a, b in sorted(out["renamed"].items())]
    for probe in ("Go-No-Go Zoo Task", "Go-NoGo fMRI paradigm", "letter n-back task",
                  "Motor Selective Stop Signal Task", "non-spatial cuing paradigm",
                  "Stop signal task with dot motion discrimination", "Iowa Gambling Task",
                  "Space Fortress with Oddball"):
        label = out["renamed"].get(probe, probe)
        target = out["collapse"].get(out["equal"].get(label, label))
        if target is None and label not in out["seeds"]:
            target = "(dropped)"
        lines.append(f"   {probe[:46]:48s} -> {target or '(kept as its own seed)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
