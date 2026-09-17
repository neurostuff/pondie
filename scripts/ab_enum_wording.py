"""A/B two versions of an enum's wording, on the text the records were built from.

Written for `AssignmentStructure` and reusable for any enum whose answers a description is
meant to steer. Three description rewrites of `population_characteristics` moved its numbers
by two and three entries out of thirty -- below the noise of a 10-paper re-extraction -- and
`is_healthy` needed five test cases to establish that a description could not carry a
distinction at all. This is the cheaper instrument for that question.

WHAT IT IS NOT. Not a re-extraction. The corpus of paper texts is not always on the host, so
the input is the design evidence each record already quotes, which is the text the original
answer was read off. Same input, same model, same temperature; the only difference between
the two arms is the enum block. That isolates the wording -- which is what changes -- and
affords a sample large enough to see past run-to-run variation.

The model here is whatever the local key reaches, not the pipeline's, so a result says the
wording does or does not carry the distinction. It does not predict production behaviour.

    # tune, then hold out: the same seeded shuffle, second disjoint slice
    python scripts/ab_enum_wording.py 60 0 ab_tuning.json
    python scripts/ab_enum_wording.py 60 1 ab_heldout.json

`enum_old.txt` and `enum_new.txt` are read from the script's directory; render them from the
two schema versions with `yaml.safe_load(...)["enums"][<name>]`. `case()` is the per-field
half and is the only thing to rewrite for another slot.
"""
import json, os, random, sys, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/jdkent/projects/StudyIESchema/pondie")
from pondie.formats import values
from pondie.normalization._records import iter_records

S = Path(__file__).parent
R = "/home/jdkent/projects/autonima-results/experiments/record_arms/records"
MODEL = "gpt-4.1"
KEY = os.environ["OPENAI_API_KEY"]

def spans(node):
    ev = (node or {}).get("evidence") or {}
    return [sp.get("text", "") for st in (ev.get("sets") or []) for sp in (st.get("spans") or [])]

def case(body):
    d = body.get("design") or {}
    groups = [str(values.read(g.get("name"))) for g in (body.get("groups") or []) if isinstance(g, dict)]
    quotes = []
    for slot in ("description", "allocation", "assignment_structure"):
        quotes += spans(d.get(slot))
    text = "\n".join(f"- {q}" for q in dict.fromkeys(q for q in quotes if q))
    return {
        "design_description": str(values.read(d.get("description")) or ""),
        "allocation": str(values.read(d.get("allocation")) or "unset"),
        "cohorts": groups,
        "arms_declared": [str(values.read(a.get("name"))) for a in (d.get("arms") or []) if isinstance(a, dict)],
        "held": str(values.read(d.get("assignment_structure")) or "unset"),
        "quotes": text,
    }

PROMPT = """You are filling one field of a neuroimaging study record: `assignment_structure`.

{enum}

Here is what the record says about this study's design:

design description: {design_description}
allocation: {allocation}
cohorts declared: {cohorts}
arms declared: {arms_declared}

sentences quoted from the paper:
{quotes}

Answer with JSON only: {{"assignment_structure": "<one permissible value, or the source's own wording>"}}"""

def ask(enum, c):
    body = json.dumps({
        "model": MODEL, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": PROMPT.format(enum=enum, **c)}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    # Backoff, because a rate limit that lands on one arm and not the other is not a
    # measurement of the wording. The two arms are also interleaved below for the same reason.
    delay = 2.0
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
                return str(out.get("assignment_structure", "?"))
        except Exception as exc:
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 45)
    return f"ERROR {type(last).__name__}"

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    # A held-out draw: the same seeded shuffle, skipping the papers the wording was tuned
    # against. Disjoint by construction rather than by a fresh seed, which could overlap.
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out_name = sys.argv[3] if len(sys.argv) > 3 else "ab_results.json"
    strata = {}
    for study, body in iter_records((R + "/*/*.extraction.json",)):
        c = case(body)
        if not c["quotes"] and not c["design_description"]:
            continue
        if c["held"] == "parallel" and not c["arms_declared"] and len(c["cohorts"]) >= 2:
            k = "parallel, no arms (the target)"
        elif c["held"] == "parallel" and c["arms_declared"]:
            k = "parallel, arms declared"
        elif c["held"] == "single_group":
            k = "single_group"
        elif c["held"] in ("crossover", "within_subject"):
            k = c["held"]
        else:
            continue
        strata.setdefault(k, []).append((study, c))
    rng = random.Random(0)
    picked = []
    per = {"parallel, no arms (the target)": n // 2, "parallel, arms declared": n // 5,
           "single_group": n // 5, "crossover": n // 10, "within_subject": n // 10}
    for k, rows in strata.items():
        rng.shuffle(rows)
        take = per.get(k, 5)
        picked += [(k, s, c) for s, c in rows[offset * take : offset * take + take]]
    print(f"{len(picked)} papers: " + ", ".join(f"{k}={sum(1 for a,_,_ in picked if a==k)}"
                                               for k in per), flush=True)
    old, new = (S / "enum_old.txt").read_text(), (S / "enum_new.txt").read_text()
    jobs = [(i, arm, enum) for i, _ in enumerate(picked)
            for arm, enum in (("old", old), ("new", new))]
    with ThreadPoolExecutor(4) as pool:
        done = list(pool.map(lambda j: (j[0], j[1], ask(j[2], picked[j[0]][2])), jobs))
    a = [next(v for i, arm, v in done if i == k and arm == "old") for k in range(len(picked))]
    b = [next(v for i, arm, v in done if i == k and arm == "new") for k in range(len(picked))]
    rows = [{"stratum": k, "pmid": s, "held": c["held"], "alloc": c["allocation"],
             "cohorts": len(c["cohorts"]), "arms": len(c["arms_declared"]),
             "old": x, "new": y} for (k, s, c), x, y in zip(picked, a, b)]
    (S / out_name).write_text(json.dumps(rows, indent=1))
    for k in per:
        sub = [r for r in rows if r["stratum"] == k]
        if not sub: continue
        print(f"\n{k}  (n={len(sub)})")
        print("   old:", dict(Counter(r["old"] for r in sub).most_common()))
        print("   new:", dict(Counter(r["new"] for r in sub).most_common()))

main()
