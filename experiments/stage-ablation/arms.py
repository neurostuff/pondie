"""The arms of the ablation, each a named recipe over pondie's own pieces.

Forward stepwise, like building a regression model: start from the one-prompt extraction of
the whole schema, then add pondie's stages one at a time and keep what buys something.

    mono         one call, whole extraction schema, no parse. Scored on the merged payload
                 with NO repairs (`raw`) and through pondie's `build` (`built`).
    mono_parse   one call, whole schema, plus the deterministic stages (tables, prose,
                 split) and the stage-1 listing the demands pass is shown.
    ds           pondie's demands -> satisfy split, then build. No fill, no evidence.
    ds_fill      + fill
    ds_fill_ev   + evidence
    full         + repair (proposer and adjudicator through the gateway)

Every arm gets a private copy of the corpus, because `prose` and `split` rewrite the stage-1
parse in place and one arm must not see another's rewrite.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from pondie import schema
from pondie.extraction.llm import Caller, MalformedReply
from pondie.extraction.models import Cost, ModelCall, Paper, Settings, StageName, StageOutcome
from pondie.extraction.prompt import render
from pondie.extraction.record import ids
from pondie.extraction.stages import Demands, _Base
from pondie.schema import reader

MODEL = "@psyc-aid338-ope-333f18/gpt-6-luna"

MONO_NOTE = """
This is a SINGLE PASS that emits the WHOLE record at once: every analysis the paper reports,
and every entity those analyses reference -- groups, tasks, acquisitions, model estimations
and their terms, measures, inference settings, regions, assessments, design (with its arms
and timepoints) -- together in one JSON object. No other pass runs before or after this one,
so nothing you leave out will be filled later and every local_id you reference must be an
entity you emit here.

Emit an Analysis for every tested effect the paper reports results for, including a contrast
that found nothing. {ids_note}
"""

IDS_NO_PARSE = (
    "There is NO pre-parsed table listing in this pass, so you assign Analysis and Table "
    "local_ids yourself: `ana_` plus a short stable suffix for an analysis, `tbl` plus the "
    "printed table number for a table (`tbl2`). Emit a Table record for every table an "
    "analysis cites, with its printed number, caption and footer."
)
IDS_WITH_PARSE = (
    "The stage-1 listing below enumerates the coordinate tables and prose coordinate "
    "sentences already parsed out of this paper, and the Table local_ids that already exist. "
    "Account for every listing entry exactly as the listing's own instructions say, and do "
    "not emit Table records -- they already exist."
)


def mono_prompt(text: str, context: str, with_parse: bool):
    """The demands/satisfy system prompt, rendered over BOTH sides of the schema."""
    sch = reader.load(render.EXTRACTION_SCHEMA)
    analysis_side, _ = render.mode_classes(sch, "analyses")
    entity_side, keep = render.mode_classes(sch, "entities")
    names = analysis_side | entity_side
    keep = keep + ["analyses"]
    lists = [k for k, v in schema.entity_lists().items() if "." not in v and v != "tables"]
    if not with_parse:
        names = names | {"Table"}
        keep = keep + ["tables"]
        lists.append("tables")
    system = (
        render.SYSTEM_HEAD.format(
            lists=", ".join(sorted(lists)),
            id_prefixes=ids.prefix_table(),
            value_rule=render.VALUE_RULE_NO_EVIDENCE,
        )
        + "\n\n# Conventions (extraction-readme.md)\n\n"
        + render.conventions()
        + "\n\n# Worked models (representing-models.md)\n\n"
        + "Twelve reported results and the encoding each takes. Follow the shape of the\n"
        + "one this paper's result is closest to; do not invent a third when its wording\n"
        + "sits between two of them.\n\n"
        + render.worked_models()
        + "\n\n# Schema\n"
        + render.render_schema(sch, names, keep)
    )
    note = MONO_NOTE.format(ids_note=IDS_WITH_PARSE if with_parse else IDS_NO_PARSE)
    user = note + context + "\n\n# Paper\n\n" + text + "\n\nEmit the JSON object now."
    return system, user


@dataclass(frozen=True)
class Mono(_Base):
    """One model call for the whole record. Writes `mono.json` beside pondie's payloads."""

    name: StageName = StageName.demands  # only for the outcome's stage label
    with_parse: bool = False
    #: Hold the reply to pondie's deterministic checks and retry, naming what failed.
    check: bool = False

    def failures(self, payload: dict, paper: Paper, settings: Settings) -> list[str]:
        if not self.check:
            return []
        from pondie.extraction.record.fix.link import check_local_ids

        out = []
        if not [a for a in payload.get("analyses") or [] if isinstance(a, dict)]:
            out.append("no analyses were emitted. Every tested effect the paper reports is "
                       "an Analysis, and a record without one is unusable.")
        body = {k: v for k, v in payload.items() if k != "study"} | dict(payload.get("study") or {})
        tables = settings.payloads / paper.study_id / "tables.json"
        if tables.is_file():
            body["tables"] = (body.get("tables") or []) + json.loads(tables.read_text()).get("tables", [])
        dangling = check_local_ids(body, reader.load(render.EXTRACTION_SCHEMA))
        if dangling:
            out.append(f"{len(dangling)} cross-reference problem(s); every local_id you "
                       "reference must be an entity you emit in this same object: "
                       + "; ".join(dangling[:12]))
        if self.with_parse:
            listing = Demands().listing(paper, settings)
            missing = render.unconsumed_listing(payload, listing)
            if missing:
                out.append("listing entries neither emitted as an analysis "
                           "(`source_table_analysis`) nor declined in `omitted`: "
                           + "; ".join(missing[:12]))
        return out

    def produces(self, paper: Paper, settings: Settings) -> Path:
        return settings.payloads / paper.study_id / "mono.json"

    def run(self, paper: Paper, settings: Settings, caller: Caller) -> StageOutcome:
        out = self.produces(paper, settings)
        best, best_failures, failures, first = None, None, [], 0
        if out.is_file() and not settings.redo:
            # A seeded reply is attempt 1: re-ask only if it fails the checks.
            best = json.loads(out.read_text())
            best_failures = failures = self.failures(best, paper, settings)
            if not failures:
                return self._skip(paper)
            first = 1
        text = paper.text.read_text(encoding="utf-8", errors="replace")
        context = Demands().context(paper, settings) if self.with_parse else ""
        system, user = mono_prompt(text, context, self.with_parse)
        cost, notes = Cost(), [f"seeded reply rejected: {len(failures)} failure(s)"] if first else []
        for attempt in range(first, settings.attempts):
            ask = user if not failures else user + render.RETRY_NOTE.format(
                failures="\n".join(f"- {f}" for f in failures))
            try:
                reply = caller(
                    ModelCall(model=settings.model, system=system, prompt=ask,
                              max_output_tokens=settings.max_output_tokens,
                              effort=settings.effort, service_tier=settings.service_tier,
                              attempts=1),
                    paper=paper.study_id, stage="mono")
            except MalformedReply as error:
                cost = cost + error.cost
                notes.append(str(error)[:200])
                continue
            cost = cost + reply.cost
            payload, more = render.normalize(reply.payload, "satisfy")
            notes += more
            if reply.stop_reason and reply.stop_reason != "stop":
                notes.append(f"stop_reason={reply.stop_reason}")
            failures = self.failures(payload, paper, settings)
            if best is None or len(failures) < len(best_failures):
                best, best_failures = payload, failures
            if not failures:
                break
            notes.append(f"attempt {attempt + 1} rejected: " + " | ".join(f[:150] for f in failures))
        if best is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(best, indent=1, ensure_ascii=False) + "\n")
            return StageOutcome(stage=self.name, study_id=paper.study_id, cost=cost,
                                produced=(out,), notes=tuple(notes))
        return StageOutcome(stage=self.name, study_id=paper.study_id, cost=cost,
                            reason="no parseable reply: " + "; ".join(notes))


#: arm -> (pondie stages before the custom pass, custom pass or None, pondie stages after)
ARMS: dict[str, tuple[tuple[StageName, ...], object, tuple[StageName, ...]]] = {
    "mono": ((), Mono(with_parse=False), (StageName.build,)),
    "mono_parse": ((StageName.tables, StageName.prose_foci, StageName.sign_split),
                   Mono(with_parse=True), (StageName.build,)),
    "mono_check": ((), Mono(with_parse=False, check=True), (StageName.build,)),
    "mono_parse_check": ((StageName.tables, StageName.prose_foci, StageName.sign_split),
                         Mono(with_parse=True, check=True), (StageName.build,)),
    # Forward steps on top of the best arm, seeded from its payloads with --seed-from.
    "mpc_fill": ((StageName.tables, StageName.prose_foci, StageName.sign_split),
                 Mono(with_parse=True, check=True), (StageName.fill, StageName.build)),
    "mpc_ev": ((StageName.tables, StageName.prose_foci, StageName.sign_split),
               Mono(with_parse=True, check=True), (StageName.evidence, StageName.build)),
    # pondie's own single-pass stage (pondie/extraction/stages.py `Single`).
    "pondie_single": ((StageName.tables, StageName.prose_foci, StageName.sign_split,
                       StageName.single), None, (StageName.build,)),
    "ds": ((StageName.tables, StageName.prose_foci, StageName.sign_split, StageName.demands,
            StageName.satisfy), None, (StageName.build,)),
    "ds_fill": ((StageName.tables, StageName.prose_foci, StageName.sign_split,
                 StageName.demands, StageName.satisfy, StageName.fill), None,
                (StageName.build,)),
    "ds_fill_ev": ((StageName.tables, StageName.prose_foci, StageName.sign_split,
                    StageName.demands, StageName.satisfy, StageName.fill,
                    StageName.evidence), None, (StageName.build,)),
    "full": ((StageName.tables, StageName.prose_foci, StageName.sign_split, StageName.demands,
              StageName.satisfy, StageName.fill, StageName.evidence), None,
             (StageName.build, StageName.repair)),
}


def private_corpus(base: Path, run_dir: Path, pmids: list[str]) -> Path:
    """A copy of each paper's corpus entry, its parse reset to the immutable original."""
    corpus = run_dir / "corpus"
    for pmid in pmids:
        dest = corpus / pmid
        if dest.is_dir():
            continue
        shutil.copytree(base / pmid, dest)
        shutil.copy(dest / "stage1/analyses.orig.json", dest / "stage1/analyses.json")
    return corpus
