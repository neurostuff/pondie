"""Experiment 2: does a retrieved shortlist preserve the LLM's evidence accuracy?

The union design puts the retriever in front of the quote pass to cut cost -- hand the
model twelve candidate sentences instead of the whole paper. That is the entire cost
argument and it was untested. If the shortlist costs the model the slots it uniquely
wins, the saving is not a saving.

Both arms run fresh against the same model so the comparison is controlled: the evidence
already in the records was produced by a different run and cannot serve as a baseline.

    full       the paper, and a list of values needing a quote -- what add_evidence.py does
    shortlist  per value, its twelve best-ranked sentences, and a request for one id

Scored the same way as everything else: `correct` means the answer overlaps a span a
reviewer marked, and it is a floor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

from .build_evidence_gold import locate  # noqa: E402


FULL_SYSTEM = """You locate supporting quotes in a scientific paper.

You are given a paper and a list of facts already extracted from it, each with an id and
the value that was recorded. For each id, return the single shortest span of the paper
that supports that value.

Rules:
1. Emit ONE JSON object mapping id -> quote. No prose, no markdown fence.
2. A quote MUST be copied character-for-character from the paper text given to you.
3. Prefer one sentence. Never return a whole paragraph when a clause will do.
4. If the paper does not state the fact anywhere, OMIT that id entirely.
5. Some values are classifications the paper never words that way. Quote the sentence the
   classification was read from, not a sentence containing the term."""

SHORT_SYSTEM = """You choose which candidate sentence supports a recorded fact.

For each id you are given the recorded value and a numbered list of candidate sentences
drawn from the paper. Return the number of the one sentence that best supports the value.

Rules:
1. Emit ONE JSON object mapping id -> candidate number. No prose, no markdown fence.
2. If no candidate supports the value, OMIT that id entirely. Do not guess.
3. Some values are classifications the paper never words that way. Choose the sentence the
   classification would be read from, not one containing the term."""


def load_key(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def overlaps(a, b, c, d) -> bool:
    return a < d and c < b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=ROOT / "data/eval/evidence-eval-rows.json")
    parser.add_argument("--jobs", type=Path, default=ROOT / "data/eval/evidence-jobs.json")
    parser.add_argument("--texts", type=Path, default=ROOT / "data/texts")
    parser.add_argument("--out", type=Path, default=ROOT / "data/eval/shortlist-arms.json")
    parser.add_argument("--model", default="@psyc-aid338-ope-333f18/gpt-5.6-luna")
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--key-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()

    load_key(args.key_file)
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_API_GATEWAY"))

    rows = {r["key"]: r for r in json.loads(args.rows.read_text(encoding="utf-8"))}
    jobs = [j for j in json.loads(args.jobs.read_text(encoding="utf-8")) if j["key"] in rows]

    by_paper: dict[str, list] = {}
    for job in jobs:
        by_paper.setdefault(job["paper"], []).append(job)

    results, usage = [], {"full": [0, 0], "shortlist": [0, 0]}
    for paper, group in sorted(by_paper.items()):
        found = sorted(args.texts.glob(f"{paper}/processed/*/text.txt"))
        if not found:
            continue
        text = found[0].read_text(encoding="utf-8")

        def ask(system: str, user: str, arm: str) -> dict:
            response = client.chat.completions.create(
                model=args.model, reasoning_effort="low",
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            usage[arm][0] += response.usage.prompt_tokens
            usage[arm][1] += response.usage.completion_tokens
            body = (response.choices[0].message.content or "{}").strip()
            if body.startswith("```"):
                body = body.split("\n", 1)[1].rsplit("```", 1)[0]
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {}

        answers: dict[str, dict] = {}
        for start in range(0, len(group), args.batch):
            batch = group[start:start + args.batch]

            listing = "\n".join(f"{i}. {j['field']} = {j['value'][:200]}"
                                for i, j in enumerate(batch))
            quotes = ask(FULL_SYSTEM,
                         f"# Paper\n\n{text}\n\n# Facts needing a supporting quote\n\n"
                         f"{listing}\n\nReturn the JSON object now.", "full")

            blocks = []
            for i, job in enumerate(batch):
                candidates = rows[job["key"]]["top12_texts"]
                lines = "\n".join(f"   [{k}] {c}" for k, c in enumerate(candidates))
                blocks.append(f"{i}. {job['field']} = {job['value'][:200]}\n{lines}")
            picks = ask(SHORT_SYSTEM,
                        "# Facts and their candidate sentences\n\n" + "\n\n".join(blocks)
                        + "\n\nReturn the JSON object now.", "shortlist")

            for i, job in enumerate(batch):
                answers[job["key"]] = {"quote": quotes.get(str(i)), "pick": picks.get(str(i))}

        for job in group:
            answer = answers.get(job["key"], {})
            positives = [tuple(s) for s in job["positive"]]
            verdict = {}

            quote = answer.get("quote")
            # Elastic, not `text.find`: the pipeline's own resolver is exact, but scoring
            # accuracy on an exact match confuses "quoted a different sentence" with
            # "quoted the right sentence through a line break".
            at = locate(text, quote) if isinstance(quote, str) else None
            verdict["full"] = ("no pick" if not quote else
                               "unlocatable" if at is None else
                               "correct" if any(overlaps(at[0], at[1], c, d)
                                                for c, d in positives) else "unknown")

            pick = answer.get("pick")
            row = rows[job["key"]]
            if not isinstance(pick, int) or not 0 <= pick < len(row["top12_spans"]):
                verdict["shortlist"] = "no pick"
            else:
                # The offset, never a string search. Candidates include sentence-ified
                # table rows, which are synthesised and appear nowhere in the paper, so
                # searching for their text scores every table pick as unlocatable.
                begin, end = row["top12_spans"][pick]
                verdict["shortlist"] = ("correct" if any(overlaps(begin, end, c, d)
                                                         for c, d in positives) else "unknown")
            results.append({"key": job["key"], "field": job["field"], **verdict,
                            "retriever_top1": row["top1"]})
        print(f"  {paper}: {len(group)} slots  "
              f"full={usage['full']} shortlist={usage['shortlist']}", flush=True)

    args.out.write_text(json.dumps({"results": results, "usage": usage}, indent=1) + "\n",
                        encoding="utf-8")
    n = len(results)
    print(f"\n{n} slots\n")
    print(f'{"arm":12s} {"correct":>9s} {"unknown":>9s} {"no pick":>9s} {"unlocatable":>13s}'
          f' {"prompt tok":>12s} {"completion":>11s}')
    for arm in ("full", "shortlist"):
        counts = {k: sum(1 for r in results if r[arm] == k)
                  for k in ("correct", "unknown", "no pick", "unlocatable")}
        print(f"{arm:12s} {counts['correct']*100/n:8.1f}% {counts['unknown']*100/n:8.1f}% "
              f"{counts['no pick']*100/n:8.1f}% {counts['unlocatable']*100/n:12.1f}% "
              f"{usage[arm][0]:12d} {usage[arm][1]:11d}")
    both = sum(1 for r in results if "correct" in (r["full"], r["shortlist"]))
    print(f"\nunion of the two arms: {both*100/n:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
