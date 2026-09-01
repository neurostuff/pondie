"""`Arm.name` -> whether an allocated arm is the intervention or the comparator.

A coordinate query for "treatment > non-treatment" has to know which side of a trial's
contrast is which, and nothing else in the pipeline answers that. `Group.arm` says a cohort
was allocated somewhere and `FactorLevel.arms` carries that arm into an analysis, but the
arm's own name is free text: `placebo`, `sham stimulation`, `normal saline placebo`,
`no intervention`, `0.5 mg/kg ketamine`, `LDLPFC rTMS`.

A rule and not an encoder, for the reason `_lexicon` gives: a wrong answer here is acted on
rather than displayed. Calling an active arm the comparator does not weaken a meta-analysis,
it inverts it -- the foci land in the map for the opposite contrast, and nothing downstream
can tell.

CONTROL is asserted, never assumed. An arm no rule matches is UNKNOWN, and a query that
cannot tell which side an analysis sits on drops it rather than guessing a side. That is why
there is no "everything not obviously a placebo is a treatment" fallback: the corpus contains
`no intervention`, `waiting list` and `treatment as usual`, all comparators, and none of them
contains the word placebo.
"""

from __future__ import annotations

from pondie.normalization import UNKNOWN
from pondie.normalization._lexicon import Rule, classify

ACTIVE, CONTROL = "ACTIVE", "CONTROL"
VALUES = (ACTIVE, CONTROL, UNKNOWN)

RULES = (
    # Decisive, and first: `active tPEMF` and `sham tPEMF` differ by one word, and
    # `normal saline placebo` names its own comparator. A control marker anywhere in the
    # name settles it, because no trial calls its intervention arm "sham".
    Rule.of(
        CONTROL,
        r"\bplacebo\b|\bsham\b|\bvehicle\b|\bsaline\b|\bcontrol(?:\s+arm|\s+group|\s+condition)?\b"
        r"|\bno[\s-]intervention\b|\bno[\s-]treatment\b|\bwait(?:ing)?[\s-]?list\b"
        r"|\btreatment[\s-]as[\s-]usual\b|\bTAU\b|\buntreated\b|\bunmedicated\b"
        r"|\bnon[\s-]?active\b|\binactive\b|\bdummy\b",
        decisive=True,
    ),
    # `active` is the trial's own word for the intervention arm and is only reached when no
    # control marker fired, so `active tPEMF` is ACTIVE and `sham` is not.
    Rule.of(ACTIVE, r"\bactive\b|\bverum\b|\breal\b|\bgenuine\b"),
    # Modality. Named because a drug or a stimulation protocol in the arm's name is what an
    # intervention arm looks like when it is not labelled "active".
    Rule.of(
        ACTIVE,
        r"\brTMS\b|\bTMS\b|\btDCS\b|\bECT\b|\btPEMF\b|\bPEMF\b|\bDBS\b|\bVNS\b"
        r"|\bstimulation\b|\bpsychotherapy\b|\bCBT\b|\bMBCT\b|\btherapy\b"
        r"|\bmg/kg\b|\bmg\b\s*$|\bdose\b|\binfusion\b",
    ),
    # Compounds seen in this corpus. A named drug is an intervention; the list is explicit
    # rather than a "looks like a drug name" pattern, because guessing from morphology is
    # how `saline` becomes a treatment.
    Rule.of(
        ACTIVE,
        r"\bketamine\b|\bcitalopram\b|\bescitalopram\b|\bfluoxetine\b|\bduloxetine\b"
        r"|\breboxetine\b|\bsertraline\b|\bparoxetine\b|\bvenlafaxine\b|\bbupropion\b"
        r"|\bmirtazapine\b|\bamitriptyline\b|\bnortriptyline\b|\blithium\b|\bEPO\b"
        r"|\berythropoietin\b|\bpsilocybin\b|\bSSRI\b|\bSNRI\b|\bantidepressant\b"
        r"|\bscopolamine\b|\briluzole\b|\blamotrigine\b|\bquetiapine\b|\bBLT\b"
        r"|\bbright[\s-]light\b|\blight[\s-]therapy\b",
    ),
)


def role(name: object) -> str:
    """ACTIVE, CONTROL, or UNKNOWN. UNKNOWN means the query must drop the analysis."""
    return classify(name, RULES).value
