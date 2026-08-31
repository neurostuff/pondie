"""Choose a review set that exercises the schema rather than one that agrees with it.

`bench-baseline.pmids` is three clean papers, picked so that "a failure here is an
extraction defect, not a schema gap". This picks for the opposite property: papers
between them carrying interventions, longitudinal designs, medications, factorial models,
resting-state connectivity, non-BOLD measures and multivariate analyses, because those are
the parts of the schema nothing has ever been extracted into.

Three stages, cheapest first:

    index    every ns-pond study with a complete pubget artifact set   (build_index.sh)
    screen   keyword score per axis, over title + abstract + keywords  (free)
    adjudge  one structured call per shortlisted paper                 (costs money)

The keyword stage is deliberately crude and deliberately generous: it exists to get a few
hundred plausible papers in front of the model, not to decide anything. AXES is the part
worth arguing with -- each entry names the schema surface it is a proxy for, so a term
list can be checked against what it is supposed to be finding.

    python select_papers.py screen  --out data/shortlist.jsonl
    python select_papers.py adjudge --out data/scored.jsonl
    python select_papers.py pick    --out data/ns-validate-10.pmids
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "@psyc-aid338-ope-333f18/gpt-5.6-luna"
DEFAULT_EFFORT = "low"

#: `code: (what it is, schema surface, terms)`. The terms are matched case-insensitively
#: as whole words against title + abstract + keywords; a paper's score on an axis is the
#: number of *distinct* terms that hit, so one word repeated twenty times counts once.
AXES: dict[str, tuple[str, str, list[str]]] = {
    "clinical": (
        "patients vs controls, with a named diagnosis",
        "Group.is_healthy / medical_condition / diagnostic_system / diagnostic_instrument",
        [
            "patients",
            "patient",
            "diagnosis",
            "diagnostic",
            "disorder",
            "disease",
            "syndrome",
            "dsm",
            "icd",
            "schizophrenia",
            "psychosis",
            "depression",
            "depressive",
            "mdd",
            "bipolar",
            "adhd",
            "autism",
            "asd",
            "ptsd",
            "obsessive",
            "ocd",
            "anxiety",
            "addiction",
            "dependence",
            "epilepsy",
            "parkinson",
            "alzheimer",
            "dementia",
            "mci",
            "stroke",
            "aphasia",
            "traumatic brain injury",
            "multiple sclerosis",
            "healthy controls",
            "comorbid",
            "symptom severity",
            "clinical",
        ],
    ),
    "medication": (
        "drug status of the cohort is stated",
        "Group.medications / medication_status",
        [
            "medication",
            "medicated",
            "unmedicated",
            "drug-naive",
            "drug naive",
            "antipsychotic",
            "antidepressant",
            "ssri",
            "lithium",
            "methylphenidate",
            "benzodiazepine",
            "stimulant",
            "pharmacological",
            "pharmacotherapy",
            "dose",
            "dosage",
            "administration",
            "washout",
            "wash-out",
            "placebo",
            "agonist",
            "antagonist",
            "infusion",
            "ketamine",
            "levodopa",
            "oxytocin",
        ],
    ),
    "intervention": (
        "something was done to the participants",
        "StudyDesign.allocation / Arm.arm_kind / Blinding",
        [
            "randomized",
            "randomised",
            "randomly assigned",
            "rct",
            "clinical trial",
            "intervention",
            "treatment",
            "training",
            "tdcs",
            "tms",
            "rtms",
            "tacs",
            "stimulation",
            "neurofeedback",
            "therapy",
            "psychotherapy",
            "cbt",
            "sham",
            "double-blind",
            "single-blind",
            "placebo-controlled",
            "crossover",
            "cross-over",
            "parallel-group",
            "waitlist",
        ],
    ),
    "longitudinal": (
        "the same people scanned more than once",
        "Timepoint.relation_to_intervention / order / time_from_intervention",
        [
            "longitudinal",
            "follow-up",
            "followup",
            "baseline",
            "pre-treatment",
            "post-treatment",
            "pre-intervention",
            "post-intervention",
            "prospective",
            "repeated measures",
            "time point",
            "timepoint",
            "before and after",
            "weeks later",
            "months later",
            "retest",
            "test-retest",
            "session 1",
            "two sessions",
            "sessions",
        ],
    ),
    "factorial": (
        "a crossed design with an interaction term",
        "ModelTerm.interaction_with / FactorLevel / Cell.direction",
        [
            "factorial",
            "2x2",
            "2 x 2",
            "2 × 2",
            "two-way",
            "three-way",
            "interaction effect",
            "interaction between",
            "main effect",
            "main effects",
            "anova",
            "ancova",
            "repeated-measures anova",
            "mixed-effects model",
            "full factorial",
        ],
    ),
    "resting": (
        "resting state and/or a connectivity method",
        "Task.design_type / ConnectivityDetails / ConnectivityMethod",
        [
            "resting-state",
            "resting state",
            "rest-fmri",
            "functional connectivity",
            "seed-based",
            "seed based",
            "seed region",
            "default mode",
            "dmn",
            "alff",
            "falff",
            "reho",
            "regional homogeneity",
            "graph theory",
            "psychophysiological interaction",
            "ppi",
            "gppi",
            "dynamic causal",
            "dcm",
            "granger",
            "effective connectivity",
            "coherence",
            "connectivity matrix",
            "network",
        ],
    ),
    "multimodal": (
        "more than one acquisition on the same people",
        "two or more Acquisition subclasses (MRI / EEG / FNIRS / PET)",
        [
            "multimodal",
            "multi-modal",
            "simultaneous eeg",
            "eeg-fmri",
            "eeg/fmri",
            "meg",
            "fnirs",
            "nirs",
            "pet/mri",
            "pet-mri",
            "concurrent",
            "structural and functional",
            "combined",
            "multi-echo",
            "in the same participants",
            "same subjects underwent",
        ],
    ),
    "nonbold": (
        "the measure is not a BOLD contrast",
        "MeasureFamily / MeasureType",
        [
            "voxel-based morphometry",
            "vbm",
            "gray matter volume",
            "grey matter volume",
            "gray matter density",
            "cortical thickness",
            "surface area",
            "morphometry",
            "diffusion tensor",
            "dti",
            "fractional anisotropy",
            "mean diffusivity",
            "tractography",
            "white matter integrity",
            "tbss",
            "arterial spin labeling",
            "arterial spin labelling",
            "asl",
            "perfusion",
            "cerebral blood flow",
            "pet",
            "tracer",
            "binding potential",
            "receptor availability",
            "radioligand",
            "fdg",
            "amyloid",
            "tau",
        ],
    ),
    "multivariate": (
        "not a mass-univariate GLM",
        "AnalysisType multivariate_decoding / representational_similarity / "
        "latent_decomposition / conjunction",
        [
            "mvpa",
            "multivariate pattern",
            "multi-voxel pattern",
            "decoding",
            "classifier",
            "classification accuracy",
            "support vector",
            "searchlight",
            "representational similarity",
            "rsa",
            "independent component analysis",
            "ica",
            "partial least squares",
            "pls",
            "canonical correlation",
            "dictionary learning",
            "conjunction analysis",
            "conjunction",
        ],
    ),
    "roi": (
        "an ROI whose provenance can be judged for independence",
        "SpatialScope / Region.definition_method",
        [
            "region of interest",
            "regions of interest",
            "roi",
            "rois",
            "anatomical mask",
            "atlas",
            "aal",
            "harvard-oxford",
            "freesurfer",
            "functional localizer",
            "localiser",
            "small volume correction",
            "svc",
            "a priori",
            "predefined region",
            "parcellation",
            "parcel",
        ],
    ),
    "continuous": (
        "an individual-difference measure entered as a regressor",
        "ModelTerm.type=continuous / ModelTerm.assessment / FunctionalForm",
        [
            "individual differences",
            "correlated with",
            "correlation between",
            "regression analysis",
            "covariate",
            "covariates",
            "predictor",
            "questionnaire",
            "self-report",
            "scale score",
            "trait",
            "severity score",
            "iq",
            "age-related",
            "associated with",
            "parametric modulation",
            "parametric modulator",
        ],
    ),
    "inference": (
        "the thresholding is elaborate enough to be worth recording",
        "InferenceSettings",
        [
            "tfce",
            "threshold-free",
            "permutation",
            "nonparametric",
            "non-parametric",
            "randomise",
            "randomize",
            "family-wise",
            "fwe",
            "false discovery",
            "fdr",
            "cluster-level",
            "cluster-forming",
            "cluster extent",
            "monte carlo",
            "3dclustsim",
            "bonferroni",
        ],
    ),
    "mediation": (
        "a mediation or path model",
        "Mediation.path / EffectPath",
        [
            "mediation",
            "mediator",
            "mediates",
            "mediated",
            "indirect effect",
            "path analysis",
            "structural equation",
            "sem",
            "moderated mediation",
            "bootstrapped indirect",
        ],
    ),
    "external": (
        "the data came from a public cohort",
        "Study.external_datasets",
        [
            "human connectome project",
            "hcp",
            "abcd study",
            "abcd",
            "uk biobank",
            "biobank",
            "adni",
            "openneuro",
            "openfmri",
            "imagen",
            "enigma",
            "publicly available dataset",
            "public dataset",
            "open dataset",
            "data sharing",
            "1000 functional connectomes",
            "nki",
            "camcan",
            "cam-can",
            "ukb",
        ],
    ),
    "species": (
        "not humans",
        "Group.species",
        [
            "macaque",
            "macaques",
            "monkey",
            "monkeys",
            "marmoset",
            "baboon",
            "rhesus",
            "rat",
            "rats",
            "mouse",
            "mice",
            "rodent",
            "non-human primate",
            "nonhuman primate",
        ],
    ),
}

#: Tractability. The upper bound is well under `extract_record --max-chars 200000`;
#: the lower one drops letters and corrections that carry a coordinate table.
MIN_TEXT, MAX_TEXT = 20_000, 120_000
MIN_ANALYSES, MIN_POINTS = 2, 5


def haystack(row: dict) -> str:
    keywords = row.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    return " ".join([row.get("title") or "", row.get("abstract") or "", *keywords]).lower()


def score_row(row: dict) -> dict[str, int]:
    """Distinct matched terms per axis. Whole-word, so 'roi' does not fire inside 'heroic'."""

    text = haystack(row)
    scores = {}
    for code, (_, _, terms) in AXES.items():
        hit = sum(
            1 for term in terms if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
        )
        if hit:
            scores[code] = hit
    return scores


def tractable(row: dict) -> bool:
    return (
        MIN_TEXT <= row.get("text_len", 0) <= MAX_TEXT
        and row.get("n_analyses", 0) >= MIN_ANALYSES
        and row.get("n_points", 0) >= MIN_POINTS
        and len(row.get("abstract") or "") > 200
    )


def command_screen(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in args.candidates.open(encoding="utf-8")]
    pool = [row for row in rows if tractable(row)]
    print(f"{len(rows)} eligible -> {len(pool)} tractable")

    for row in pool:
        row["scores"] = score_row(row)

    # Top-N per axis rather than top-N overall: a paper that is the best resting-state
    # candidate in the corpus should not be crowded out by fourteen better clinical ones.
    shortlist: dict[str, dict] = {}
    for code in AXES:
        ranked = sorted(
            (r for r in pool if r["scores"].get(code)),
            key=lambda r: (-r["scores"][code], -r["n_analyses"]),
        )
        for row in ranked[: args.per_axis]:
            shortlist.setdefault(row["study"], row)
        print(
            f"  {code:13s} {sum(1 for r in pool if r['scores'].get(code)):>5} match  "
            f"top={ranked[0]['scores'][code] if ranked else 0}"
        )

    with args.out.open("w", encoding="utf-8") as fh:
        for row in shortlist.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nshortlist: {len(shortlist)} unique papers -> {args.out}")
    return 0


ADJUDGE_SYSTEM = """\
You judge whether a neuroimaging paper would exercise particular parts of a study
extraction schema. You are given a title, keywords and abstract, and a list of axes.

For each axis, answer whether the paper clearly exercises it. Judge only what the text
supports: an abstract that does not mention medication does not establish that the cohort
was unmedicated, and "not stated" is a no, not a maybe. Be strict -- the point of this set
is that each axis is genuinely represented, so a generous yes is worse than a no.

Also give the paper an overall `richness` 0-3: how much structure an extractor would have
to represent (0 = one group, one task, one contrast; 3 = several cohorts, several
conditions or timepoints, several analyses of different kinds).
"""


def adjudge_schema() -> dict:
    axis_properties = {
        code: {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "why": {"type": "string", "description": "at most 15 words"},
            },
            "required": ["present", "why"],
            "additionalProperties": False,
        }
        for code in AXES
    }
    return {
        "type": "object",
        "properties": {
            "axes": {
                "type": "object",
                "properties": axis_properties,
                "required": list(AXES),
                "additionalProperties": False,
            },
            "richness": {"type": "integer"},
            "one_line": {"type": "string", "description": "what the study did, <= 20 words"},
        },
        "required": ["axes", "richness", "one_line"],
        "additionalProperties": False,
    }


def command_adjudge(args: argparse.Namespace) -> int:
    import openai  # noqa: PLC0415

    load_key_file(args.key_file)
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_API_GATEWAY")
    )

    done: set[str] = set()
    if args.out.is_file() and not args.redo:
        done = {json.loads(l)["study"] for l in args.out.open(encoding="utf-8")}
        print(f"resuming: {len(done)} already scored")

    rows = [json.loads(line) for line in args.shortlist.open(encoding="utf-8")]
    todo = [r for r in rows if r["study"] not in done]
    axis_list = "\n".join(
        f"- {code}: {what} (schema: {surface})" for code, (what, surface, _) in AXES.items()
    )

    with args.out.open("a", encoding="utf-8") as fh:
        for n, row in enumerate(todo, 1):
            keywords = row.get("keywords") or []
            user = (
                f"# Axes\n\n{axis_list}\n\n# Paper\n\n"
                f"Title: {row['title']}\n"
                f"Journal: {row.get('journal')} ({row.get('year')})\n"
                f"Keywords: {', '.join(keywords) if keywords else 'none listed'}\n\n"
                f"Abstract:\n{row['abstract']}\n"
            )
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    reasoning_effort=args.effort,
                    messages=[
                        {"role": "system", "content": ADJUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "axis_judgement",
                            "strict": True,
                            "schema": adjudge_schema(),
                        },
                    },
                )
                verdict = json.loads(response.choices[0].message.content)
            except Exception as exc:
                print(
                    f"  {row['study']}: FAILED {type(exc).__name__}: {exc}"[:200],
                    file=sys.stderr,
                )
                continue
            fh.write(
                json.dumps(
                    {
                        "study": row["study"],
                        "pmid": row["pmid"],
                        "title": row["title"],
                        **verdict,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            hits = [c for c, v in verdict["axes"].items() if v["present"]]
            print(
                f"  [{n}/{len(todo)}] {row['study']} r={verdict['richness']} "
                f"{'+'.join(hits) or 'none'}"
            )
    return 0


def command_pick(args: argparse.Namespace) -> int:
    """Greedy set cover over the axes, then fill the remainder with the richest papers.

    Coverage first and quality second, deliberately: a tenth good factorial paper is
    worth less to this set than the only mediation one, because the question it answers
    ("can the schema hold a mediation model?") is otherwise unanswered.
    """

    index = {
        row["study"]: row
        for row in (json.loads(l) for l in args.shortlist.open(encoding="utf-8"))
    }
    scored = [json.loads(l) for l in args.scored.open(encoding="utf-8")]
    for row in scored:
        row["hits"] = {c for c, v in row["axes"].items() if v["present"]}
        row["meta"] = index.get(row["study"], {})
    # A paper carrying one weak axis is not a review-set paper.
    scored = [r for r in scored if len(r["hits"]) >= 2 and r["meta"]]

    chosen: list[dict] = []
    covered: set[str] = set()
    pool = list(scored)

    def quality(row: dict) -> tuple:
        return (row["richness"], len(row["hits"]), row["meta"].get("n_analyses", 0))

    while pool and len(chosen) < args.count:
        gain = max(len(r["hits"] - covered) for r in pool)
        if gain == 0:
            break
        best = max((r for r in pool if len(r["hits"] - covered) == gain), key=quality)
        chosen.append(best)
        covered |= best["hits"]
        pool.remove(best)

    # Every axis reachable is now covered; spend what is left on papers that will make
    # the extractor work hardest.
    for row in sorted(pool, key=quality, reverse=True):
        if len(chosen) >= args.count:
            break
        chosen.append(row)

    missing = set(AXES) - covered
    print(f"picked {len(chosen)}; axes covered {len(covered)}/{len(AXES)}")
    if missing:
        print(f"  NOT covered: {', '.join(sorted(missing))}")

    with args.out.open("w", encoding="utf-8") as fh:
        fh.write("# pmid\tneurostore_id\taxes\n")
        fh.write(
            "# Chosen by select_papers.py to span the schema; see deploy/DEPLOYMENT.md.\n"
        )
        for row in chosen:
            axes = "+".join(sorted(row["hits"]))
            fh.write(f"{row['pmid']}\t{row['study']}\t{axes}\n")
            print(f"  {row['study']}  r={row['richness']}  {axes}")
            print(f"      {row['title'][:96]}")

    print(f"\n-> {args.out}")
    return 0


def load_key_file(path: Path) -> None:
    for raw in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ[name.strip()] = value.strip().strip("'\"")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser("screen", help="keyword-score the index, write a shortlist")
    screen.add_argument("--candidates", type=Path, default=REPO / "data/candidates.jsonl")
    screen.add_argument("--per-axis", type=int, default=15)
    screen.add_argument("--out", type=Path, default=REPO / "data/shortlist.jsonl")
    screen.set_defaults(func=command_screen)

    adjudge = sub.add_parser("adjudge", help="one model call per shortlisted paper")
    adjudge.add_argument("--shortlist", type=Path, default=REPO / "data/shortlist.jsonl")
    adjudge.add_argument("--out", type=Path, default=REPO / "data/scored.jsonl")
    adjudge.add_argument("--key-file", type=Path, default=REPO / ".env")
    adjudge.add_argument("--model", default=DEFAULT_MODEL)
    adjudge.add_argument("--effort", default=DEFAULT_EFFORT)
    adjudge.add_argument("--redo", action="store_true")
    adjudge.set_defaults(func=command_adjudge)

    pick = sub.add_parser("pick", help="greedy axis cover, write the pmids file")
    pick.add_argument("--shortlist", type=Path, default=REPO / "data/shortlist.jsonl")
    pick.add_argument("--scored", type=Path, default=REPO / "data/scored.jsonl")
    pick.add_argument("--count", type=int, default=10)
    pick.add_argument("--out", type=Path, default=REPO / "data/ns-validate-10.pmids")
    pick.set_defaults(func=command_pick)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
