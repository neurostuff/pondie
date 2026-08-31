"""Ask the model to repair a record's model graph, where the choice is not forced.

`build_record` already repairs the forced cases: a cell naming a term that exactly one
in-scope term shares a name with is repointed deterministically. What is left is 53
references across 30 records where the analysis's model has *no terms at all* and one
other model holds everything its cells need. Whether that means the analysis names the
wrong model, or means its model should reach the other through `inputs_from`, is a claim
about how the authors estimated their models -- and only the methods text says.

The design follows the one rule that made LLM repair defensible elsewhere in this
pipeline: **a closed answer space with an external check.** The model may return one of
four repairs and nothing else; the repair is applied and the reachability rule re-run; and
the sentence offered as justification must resolve in the paper. A repair that does not
satisfy all three is discarded, not trusted.

What is deliberately NOT sent: the whole record, and the whole paper. Sending the record
invites rewriting fields that were right -- the failure mode measured when an
unconditional prompt instruction cost 7.4 points of direction accuracy. Sending the whole
paper buries the four sentences that decide the question.

    python repair_model_graph.py --records data/runs/... --texts ... --effort low
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from pondie.extraction.passes import evidence_retrieval  # noqa: E402

SYSTEM = """You repair the model graph of a structured record extracted from a neuroimaging paper.

A record describes each model the authors estimated as a ModelEstimation with a local_id,
a list of ModelTerms (the columns of its design), and `inputs_from`: the local_ids of the
models whose outputs it takes as input. An Analysis names one model in
`model_estimation`, and each cell of its effect names one ModelTerm. The rule that has
been broken: a cell may only name a term the analysis's model can reach -- its own terms,
or the terms of models it reaches through `inputs_from`.

You are given the broken analysis, the tables it is anchored on, the whole model graph,
and the paper's methods text. The table caption often states the contrast outright, and it
is the strongest evidence available for which model an analysis belongs to.
Choose exactly ONE repair:

  A. "wrong_model"   -- the analysis names the wrong model; it should name <model_id>.
  B. "missing_edge"  -- the analysis's model should reach the other through inputs_from.
  C. "missing_terms" -- the analysis's model has its own terms which were not extracted;
                        they should be added to it, not borrowed from elsewhere.
  D. "cannot_tell"   -- the methods text does not say which of these is true.

Rules:
1. Emit ONE JSON object: {"repair": "...", "target": "<model local_id or null>",
   "quote": "<the sentence from the methods text that decides it>", "why": "<one line>"}
2. The quote MUST be copied character-for-character from the methods text given to you.
   It is checked by exact match and a paraphrase is discarded with the repair.
3. Choose D whenever the text does not settle it. D is a useful answer. A confident
   wrong graph is worse than an unrepaired one, because nothing downstream can tell.
4. Do not propose renaming or deleting anything else."""


def value(node):
    return node.get("value") if isinstance(node, dict) and "value" in node else node


_field_value = value


def scope(models: dict, model_id: str, seen: set | None = None) -> dict:
    seen = seen or set()
    if model_id in seen or model_id not in models:
        return {}
    seen.add(model_id)
    found = {}
    for lower in models[model_id].get("inputs_from") or []:
        found.update(
            scope(models, lower if isinstance(lower, str) else str(value(lower)), seen)
        )
    for term in models[model_id].get("terms") or []:
        if isinstance(term, dict) and value(term.get("local_id")):
            found[str(value(term["local_id"]))] = term
    return found


def graph_block(record: dict) -> str:
    """Every model, its terms and its edges. The whole graph, because the question is
    which of several models an analysis belongs to."""
    lines = []
    for model in record.get("model_estimations") or []:
        mid = str(value(model.get("local_id")))
        edges = [
            e if isinstance(e, str) else str(value(e))
            for e in (model.get("inputs_from") or [])
        ]
        lines.append(
            f"- model {mid!r}  family={value(model.get('model_family'))!r} "
            f"inputs_from={edges}"
        )
        for term in model.get("terms") or []:
            levels = [str(value(l.get("level"))) for l in (term.get("levels") or [])]
            lines.append(
                f"    term {str(value(term.get('local_id')))!r} "
                f"name={str(value(term.get('name')))!r} "
                f"type={value(term.get('type'))!r} levels={levels}"
            )
        if not (model.get("terms") or []):
            lines.append("    (no terms extracted for this model)")
        used = [
            str(value(a.get("name")))[:50]
            for a in record.get("analyses") or []
            if str(value(a.get("model_estimation"))) == mid
        ]
        lines.append(f"    analyses naming this model: {used[:4]}")
    return "\n".join(lines)


#: Plain-text section headings, for corpora that carry no markdown. The `ace` flavour has
#: none at all -- `sectionize` returns a single `unknown` span for it -- so a methods
#: selection built only on markdown silently sends the introduction instead. Matched as a
#: short line that is a heading and nothing else.
_PLAIN_METHOD = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*)?(MATERIALS?[ \t]+AND[ \t]+METHODS?|METHODS?|METHOD[ \t]+AND[ \t]+"
    r"MATERIALS?|Materials?[ \t]+and[ \t]+[Mm]ethods?|Methods?)[ \t]*:?[ \t]*$",
    re.M,
)
_PLAIN_AFTER = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*)?(RESULTS?|DISCUSSION|Results?|Discussion)[ \t]*:?[ \t]*$", re.M
)

#: Vocabulary that marks a paragraph as describing how the models were estimated. Used
#: only when no heading of either kind can be found, which is the case for roughly half
#: this corpus.
_MODEL_TALK = re.compile(
    r"\b(first[- ]level|second[- ]level|1st[- ]level|2nd[- ]level|random[- ]effects?|"
    r"fixed[- ]effects?|design matrix|regressor|contrast images?|brought forward|entered "
    r"into|flexible factorial|full factorial|ANOVA|ANCOVA|general linear model|\bGLM\b|"
    r"mixed[- ]effects?|regression|covariate|group[- ]level|subject[- ]level|SPM|FSL|"
    r"FEAT|AFNI|statistical (?:analys\w+|threshold|parametric))\b",
    re.I,
)

#: Generous, because the whole point is not to truncate the paragraphs that decide the
#: question. 40k chars is ~10k tokens, which is $0.002 of input at luna prices.
METHODS_LIMIT = 40_000


def tables_block(record: dict, stage1: dict, analysis: dict) -> str:
    """Every table in the paper, with what reads off it and who points at it.

    Every output of this pipeline hangs off a table: an Analysis exists because a
    coordinate table reports it, and `Analysis.tables` is the only link between the
    record and the rows its coordinates came from. The caption is frequently where the
    contrast is stated outright -- "Between-group differences in brain activations" --
    which is the strongest evidence for which model an analysis belongs to.

    All of them rather than only the ones this analysis names, for two reasons. The
    reference is often dangling -- `t0035: NOT declared` is common in this corpus -- so
    restricting to the named tables can leave nothing at all. And the question is
    comparative: which model an analysis belongs to depends on what the other tables
    report and which models their analyses name, so hiding them removes the comparison.

    Both sources are shown because they differ: the record's `tables` entries are what
    the model emitted, and the stage-1 parse is what was read off the page, including
    statistic kinds a caption never mentions.
    """

    named = _field_value(analysis.get("tables")) or []
    named = [t for t in (named if isinstance(named, list) else [named]) if isinstance(t, str)]

    #: Which analyses point at each table, and under which model. The comparative signal.
    users: dict[str, list[str]] = {}
    for other in record.get("analyses") or []:
        if not isinstance(other, dict):
            continue
        refs = _field_value(other.get("tables")) or []
        for table_id in (refs if isinstance(refs, list) else [refs]):
            if isinstance(table_id, str):
                users.setdefault(table_id, []).append(
                    f"{str(value(other.get('name')))[:40]!r} "
                    f"(model {str(value(other.get('model_estimation')))!r})"
                )

    declared = {
        str(value(t.get("local_id"))): t
        for t in record.get("tables") or []
        if isinstance(t, dict)
    }
    parsed: dict[str, list[dict]] = {}
    for entry in stage1.get("analyses") or []:
        parsed.setdefault(str(entry.get("table_id")), []).append(entry)

    lines = [f"The analysis being repaired names tables: {named or '(none)'}"]
    if named and not any(t in declared for t in named):
        lines.append(
            "NOTE: none of those are declared in the record's `tables`, so the "
            "reference is dangling. Every table the paper has is listed below."
        )
    lines.append("")

    for table_id in sorted(set(declared) | set(parsed) | set(named)):
        mark = "  <-- named by the analysis being repaired" if table_id in named else ""
        entry = declared.get(table_id)
        number = value(entry.get("table_number")) if entry else None
        lines.append(
            f"- table {table_id!r} number={number!r}"
            f"{'' if entry else '  [not declared in `tables`]'}{mark}"
        )
        for slot in ("caption", "footer"):
            said = str(value((entry or {}).get(slot)) or "").strip()
            if said:
                lines.append(f"    {slot}: {said[:300]!r}")
        for read in parsed.get(table_id, []):
            kinds = sorted(
                {
                    v.get("kind")
                    for pt in (read.get("points") or [])
                    for v in (pt.get("values") or [])
                    if v.get("kind")
                }
            )
            lines.append(
                f"    stage 1 read {str(read.get('name'))[:56]!r} -- "
                f"{len(read.get('points') or [])} foci, statistics={kinds}"
            )
            caption = str(read.get("table_caption") or "").strip()
            if caption and not entry:
                lines.append(f"      caption from the parse: {caption[:300]!r}")
        if users.get(table_id):
            lines.append(f"    analyses pointing here: {users[table_id][:5]}")
    return "\n".join(lines)


def methods_text(text: str, limit: int = METHODS_LIMIT) -> tuple[str, str]:
    """(the methods passages, how they were found).

    Three routes, tried in order, because the corpus is not uniformly structured:
    markdown headings, plain-text headings, then paragraphs that talk about model
    estimation. Returning *how* it was found matters -- a pass reasoning from the
    introduction because no heading was detected looks identical to one reading the
    methods, and that is exactly what happened before this was measured.
    """

    spans = evidence_retrieval.sectionize(text)
    kept = [text[start:end] for start, end, label in spans if label == "methods"]
    if kept:
        return "\n\n".join(kept)[:limit], "markdown headings"

    start = _PLAIN_METHOD.search(text)
    if start:
        after = _PLAIN_AFTER.search(text, start.end())
        block = text[start.start() : after.start() if after else len(text)]
        if len(block) > 500:
            return block[:limit], "plain-text heading"

    # Sentence windows, not paragraphs. The `ace` flavour is one 62,000-character
    # paragraph with five blank lines in the whole document, so a paragraph filter
    # returns the paper. A window rather than a bare sentence because the deciding
    # sentences refer backwards -- "these contrasts were brought forward" needs to know
    # which contrasts.
    units = evidence_retrieval.sentence_units(text)
    if units:
        wanted: set[int] = set()
        for index, unit in enumerate(units):
            if _MODEL_TALK.search(unit.rendered):
                wanted |= {i for i in range(index - 2, index + 3) if 0 <= i < len(units)}
        if wanted:
            picked, previous = [], None
            for index in sorted(wanted):
                if previous is not None and index != previous + 1:
                    picked.append("[...]")
                picked.append(units[index].text.strip())
                previous = index
            return "\n".join(picked)[:limit], f"{len(wanted)} sentence window(s)"
    return text[:limit], "NO METHODS FOUND -- whole text, truncated"


def broken(record: dict):
    """(analysis index, analysis, the terms it names out of scope, the models owning them)."""
    models = {
        str(value(m.get("local_id"))): m
        for m in record.get("model_estimations") or []
        if isinstance(m, dict)
    }
    owner = {
        str(value(t.get("local_id"))): mid
        for mid, m in models.items()
        for t in (m.get("terms") or [])
        if isinstance(t, dict)
    }
    for index, analysis in enumerate(record.get("analyses") or []):
        if not isinstance(analysis, dict):
            continue
        mid = str(value(analysis.get("model_estimation")) or "")
        in_scope = scope(models, mid)
        bad = [
            str(value(c.get("term")))
            for c in (analysis.get("effect") or {}).get("cells") or []
            if isinstance(value(c.get("term")), str)
            and str(value(c.get("term"))) not in in_scope
        ]
        if bad:
            yield index, analysis, sorted(set(bad)), {owner.get(t) for t in bad}, models


def ask(
    client,
    model: str,
    effort: str,
    record: dict,
    index: int,
    analysis: dict,
    bad: list[str],
    text: str,
    stage1: dict,
) -> dict:
    methods, how = methods_text(text)
    user = (
        f"# The broken analysis\n\n"
        f"analyses[{index}]  name={str(value(analysis.get('name')))!r}\n"
        f"definition={str(value(analysis.get('definition')) or '')[:300]!r}\n"
        f"model_estimation={str(value(analysis.get('model_estimation')))!r}\n"
        f"cells name these terms, which that model cannot reach: {bad}\n\n"
        f"# The tables this analysis is anchored on\n\n"
        f"{tables_block(record, stage1, analysis)}\n\n"
        f"# The model graph\n\n{graph_block(record)}\n\n"
        f"# The paper's methods (selected by {how})\n\n{methods}\n\n"
        f"Return the JSON object now."
    )
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
    }
    if effort:
        kwargs["reasoning_effort"] = effort
    response = client.chat.completions.create(**kwargs)
    body = (response.choices[0].message.content or "{}").strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {"repair": "unparseable"}
    parsed["_tokens"] = (response.usage.prompt_tokens, response.usage.completion_tokens)
    parsed["_methods"] = (how, len(methods))
    return parsed


def apply_and_check(record: dict, index: int, answer: dict, text: str) -> tuple[bool, str]:
    """Apply the proposed repair to a copy and re-run the reachability rule.

    Three checks, and a repair failing any is discarded: the option must be one of the
    four, the quote must resolve in the paper, and the graph must actually become valid.
    """
    repair = answer.get("repair")
    if repair not in ("wrong_model", "missing_edge", "missing_terms", "cannot_tell"):
        return False, f"not one of the four options: {repair!r}"
    if repair == "cannot_tell":
        return True, "declined"
    quote = answer.get("quote") or ""
    if len(quote) < 15 or " ".join(quote.split()) not in " ".join(text.split()):
        return False, "the quote does not resolve in the paper"
    # The quote must also be from what was actually sent. A quote found elsewhere in the
    # paper means the model answered from something it was not shown, which is not a
    # justification it can be held to.
    methods, _how = methods_text(text)
    if " ".join(quote.split()) not in " ".join(methods.split()):
        return False, "the quote resolves in the paper but not in the methods sent"
    if repair == "missing_terms":
        return True, "proposes terms, not a graph edit -- nothing to verify structurally"

    trial = json.loads(json.dumps(record))
    target = answer.get("target")
    if not isinstance(target, str):
        return False, "no target model named"
    analysis = (trial.get("analyses") or [])[index]
    if repair == "wrong_model":
        analysis["model_estimation"] = target
    else:
        mid = str(value(analysis.get("model_estimation")))
        for model in trial.get("model_estimations") or []:
            if str(value(model.get("local_id"))) == mid:
                model.setdefault("inputs_from", []).append(target)
    still = [b for b in broken(trial) if b[0] == index]
    return (not still), (
        "graph is valid after the repair"
        if not still
        else "the repair does not satisfy reachability"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--model", default="@psyc-aid338-ope-333f18/gpt-5.6-luna")
    parser.add_argument("--key-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    for line in args.key_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, val = line.partition("=")
            os.environ.setdefault(k.strip(), val.strip().strip("'\""))
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_API_GATEWAY")
    )

    rows, spent = [], [0, 0]
    for path in sorted(args.records.glob("*.extraction.json")):
        if len(rows) >= args.limit:
            break
        study = path.name.split(".")[0]
        record = json.loads(path.read_text(encoding="utf-8"))
        found = list(broken(record))
        if not found:
            continue
        flavour = next(
            (
                f
                for f in ("pubget", "ace", "elsevier")
                if (args.texts / study / "processed" / f / "text.txt").is_file()
            ),
            None,
        )
        if not flavour:
            continue
        text = (args.texts / study / "processed" / flavour / "text.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        index, analysis, bad, owners, _models = found[0]
        stage1_path = args.texts / study / "stage1" / "analyses.json"
        stage1 = (
            json.loads(stage1_path.read_text(encoding="utf-8"))
            if stage1_path.is_file()
            else {}
        )
        answer = ask(
            client, args.model, args.effort, record, index, analysis, bad, text, stage1
        )
        spent[0] += answer["_tokens"][0]
        spent[1] += answer["_tokens"][1]
        ok, note = apply_and_check(record, index, answer, text)
        rows.append(
            {
                "study": study,
                "analysis_index": index,
                "analysis": str(value(analysis.get("name")))[:60],
                "model": str(value(analysis.get("model_estimation"))),
                "bad_terms": bad,
                "owned_by": sorted(x for x in owners if x),
                "repair": answer.get("repair"),
                "target": answer.get("target"),
                "quote": (answer.get("quote") or "")[:200],
                "why": (answer.get("why") or "")[:200],
                "accepted": ok,
                "check": note,
                "methods_route": answer.get("_methods", ("?", 0))[0],
                "methods_chars": answer.get("_methods", ("?", 0))[1],
            }
        )
        print(
            f"  {study}: {answer.get('repair')} -> {answer.get('target')}  [{note}]",
            flush=True,
        )

    args.out.write_text(
        json.dumps({"effort": args.effort, "tokens": spent, "rows": rows}, indent=1) + "\n",
        encoding="utf-8",
    )
    accepted = sum(r["accepted"] for r in rows)
    declined = sum(r["repair"] == "cannot_tell" for r in rows)
    print(
        f"\n{len(rows)} cases at effort={args.effort!r}: {accepted} passed the checks "
        f"({declined} of them by declining), {len(rows) - accepted} discarded"
    )
    print(f"tokens {spent[0]:,} in / {spent[1]:,} out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
