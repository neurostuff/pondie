"""What a field value is *about*, before any vocabulary is consulted.

A layer above `labels`, and the one that was missing. `folding` claims orthography and
`labels` claims which tokens carry identity; both assume the value names one thing and
names it positively. A `Group.medical_condition` frequently does neither:

    "no neurological or psychiatric disorder"     an ABSENCE. A disease vocabulary
                                                  retrieves a disease from this every time.
    "alcohol abuse or dependence"                 TWO heads, and one row each is what makes
                                                  a comorbid cohort queryable.
    "first-episode schizophrenia"                 one head plus a course qualifier, which
                                                  is not a different disease.

Measured over the 2,115-record corpus on beast, **22% of `medical_condition` values state
an absence** -- 452 as `no`/`none`/`without` and 414 as a bare `healthy` -- and 4% split into
more than one head under the conservative splitter below. Matching those without triage is
where the ONVOC route's errors came from: `absence of major depressive disorder` returned
Depressive Disorder, and `No clinically significant cognitive impairment No dementia`
returned Depressive Disorder from a cohort that had neither.

Triage first, then look up each head. Both the MONDO route in
`normalization.medical_condition` and the ONVOC route in `vocabularies.onvoc` read it from
here, so the two cannot drift -- which they had, the MONDO one having the gate and the
ONVOC one not.

Negation is read from the sentence's syntax (`normalization._negation`); the regexes here
are the fallback for a host with no parser. Why, and what the anchored version got wrong:
docs/condition-normalization.md.

The rules are domain-general over clinical writing rather than over English, which is why
they are here and not in `folding`: `first-episode` and `treatment-resistant` are course
and state in psychiatry, and `or` separates comorbidities where `and` does not -- "attention
deficit and hyperactivity disorder" is one disorder.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from study_schema.models.storage import RiskBasis

from pondie.normalization import _negation as negation
from pondie.vocabularies.folding import fold

# `unimpaired`, `intact`, `good` and `normal weight` are here because the corpus put them
# in `medical_condition` and every one of them was being looked up as a disease: see
# docs/normalization-rationale.md, "is_healthy".
_WELL = (
    r"healthy|health|normal[\s-]weight|normal|unaffected|unimpaired|non-?impaired|intact|"
    r"asymptomatic|non-?psychotic|"
    r"good|excellent|non-?clinical|non-?patient|non-?smok(?:er|ers|ing)|"
    r"typically\s+develop(?:ing|ed)|typical|neuro-?typical|comparison|none|nil"
)
_ROLE = (
    r"controls?|volunteers?|participants?|subjects?|adults?|individuals?|groups?|child|"
    r"children|adolescents?|teens?|teenagers?|youths?|infants?|toddlers?|neonates?|boys?|girls?|"
    r"students?|twins?|siblings?|aged|cognition|"
    r"men|women|males?|females?|elderly|older|younger|young|aging|ageing|"
    r"persons?|people|population|cohort|sample|donors?|reported|noted|documented|"
    r"identified|found|present|"
    # `Normal hearing` and `normal-hearing listeners` are control groups in audiology papers.
    r"hearing|vision|sight|eyesight|listeners?|readers?|speakers?"
)
# Adjectives sit here beside their adverbs: the pattern only uses this list as a modifier
# slot, and "good physical health" needs `physical` exactly where `physically` is allowed.
_ADVERB = (
    r"physically|mentally|medically|generally|otherwise|neurologically|psychiatrically|"
    r"cognitively|clinically|overall|physical|mental|medical|general|neurological|"
    r"psychiatric|cognitive|somatic"
)
#: A value made of nothing but wellness words, study-role nouns and punctuation, anchored
#: at both ends. A word in neither list -- `smokers`, `obese` -- ends the match, so a bare
#: assertion of health is an absence and `healthy` in front of a condition is not.
HEALTHY = re.compile(
    rf"^\W*(?:(?:{_ADVERB})\W+)?(?:{_WELL})(?:\W+(?:{_WELL}|{_ROLE}|{_ADVERB}))*\W*$",
    re.I,
)

#: A non-answer rather than an answer, kept apart from `HEALTHY` because the schema keeps
#: them apart: an unread cohort is not a well one. `extraction_status` carries this when
#: the extractor sets it; this catches it arriving as a string in the value.
UNREAD = re.compile(
    r"^\W*(?:not[_ ]?(?:reported|applicable|stated|specified|available|assessed)|"
    r"unknown|unspecified|undetermined|n/?a)\W*$",
    re.I,
)

#: Course and state, not a different disease. Lifted off the head and kept beside it:
#: `first-episode schizophrenia` and `chronic schizophrenia` are one vocabulary term, and a
#: query that cannot tell them apart is a different problem from one that cannot find them.
QUALIFIER = re.compile(
    r"\b(first[- ]episode|(?:childhood|adolescent|adult|juvenile|youth)[- ]onset|"
    r"chronic|acute|early[- ]onset|late[- ]onset|"
    r"remitted|in remission|treatment[- ]resistant|refractory|"
    r"drug[- ]na[iï]ve|medicated|unmedicated|recent[- ]onset|stable|"
    # `mild` names a severity everywhere but in mild (neuro)cognitive impairment, a condition
    r"current|lifetime|past|subclinical|mild(?!\s+(?:neuro-?)?cognitive\b)|moderate|severe|recurrent|"
    r"childhood|adolescent|adult|p(?:a)?ediatric|comorbid|probable|possible|"
    r"suspected|sporadic|familial|presymptomatic|"
    r"intractable|(?:drug|medication|pharmaco)[- ]?resistant|newly[- ]diagnosed|de novo|"
    r"combat[- ]related|history of|prior|previous|concurrent)\b",
    re.I,
)

#: A prevalence or count, which says how many in the group have the condition and is not part
#: of its name: "panic disorder in 2", "cognitive impairment in 18/44 patients", "PTSD: 9
#: (23.7%)", "(n = 12)". Removed before the split, because `18/44` would otherwise split on
#: its slash, and before retrieval, which accepted "panic disorder in 2" as *panic disorder 2*,
#: an OMIM locus. The item keeps the count, so a comorbid role can still read it.
_NUMBER = (
    r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|a few|few|some|"
    r"several|a subset|a minority|a small number)"
)
_UNITS = r"(?:patients?|participants?|subjects?|cases?|individuals?|children|adults|people)"
_OF = rf"(?:\s+(?:of|out of)\s+(?:the\s+)?(?:\d+\s+)?(?:{_UNITS}|sample|cohort|group|total|\d+))?"
PREVALENCE = re.compile(
    r"\s*\(\s*(?:n\s*=\s*\d+|\d+(?:\.\d+)?\s*%|\d+\s*/\s*\d+|\d+\s+" + _UNITS + r")[^)]*\)"
    r"|[,;:]?\s*\bn\s*=\s*\d+\b"
    r"|[,;:]\s*\d+\s*(?:\(\s*\d+(?:\.\d+)?\s*%\s*\))?\s*$"
    rf"|[,;:]?\s+in\s+(?:\d+(?:\.\d+)?\s*%|\d+\s*/\s*\d+|{_NUMBER}){_OF}"
    rf"(?:\s+{_UNITS})?(?:\s+(?:overall|in total))?(?![\w%])"
    # and a clause about those participants: "in 43 participants meeting PCL-R total >= 30"
    r"(?:\s+(?:meeting|who|whom|with|according|based|defined|scoring|during|receiving|"
    r"having|treated|diagnosed)\b.*$)?"
    rf"|[,;:]?\s+\d+(?:\.\d+)?\s*%{_OF}",
    re.I,
)

#: A slash inside one name, where `SPLIT` would otherwise make two heads of it. The commonest
#: in the corpus: "attention-deficit/hyperactivity disorder" (515 items) split into
#: "attention-deficit", which retrieval placed at *ADHD, inattentive type*. Listed rather than
#: inferred: "panic disorder/agoraphobia" and "overweight/obesity" are two conditions written
#: the same way.
IN_NAME = re.compile(
    r"\b(attention[- ]?deficit|non-?fluent|logopenic|avoidant|tremor|depersonali[sz]ation|"
    r"vegetative state|hyperactive|binge[- ]eating)"
    r"\s*/\s*"
    r"(?=hyperactivity|agrammatic|phonological|restrictive|ataxia|dereali[sz]ation|"
    r"unresponsive|impulsive|purging)",
    re.I,
)
#: What stands in for an `IN_NAME` slash until the split is done. `fold` reads either as a space.
_KEPT = "\u2215"

#: "MCI due to Alzheimer's disease" states two conditions the group has: the impairment and the
#: disease causing it (James, 2026-11-27). Written as two parts before the split.
_DUE_TO = re.compile(
    r"\b((?:amnestic |amnesic )?(?:mild cognitive impairment|MCI))\s+due to\s+", re.I
)

#: "alcohol abuse/dependence" and "cannabis abuse or dependence" name the substance once for
#: both conditions. Split as written, the second head is a bare "dependence", which MONDO
#: places at *substance dependence*; the substance is written onto both sides first.
_SUBSTANCE_PAIR = re.compile(
    r"\b([a-z]+)\s+(abuse|dependence|dependency|misuse)(\s*/\s*|\s+or\s+)"
    r"(abuse|dependence|dependency|misuse)\b",
    re.I,
)

#: Split on `or`, `/` and `;` only -- never on `and` or a comma. The slot is multivalued, so
#: several conditions crammed into one string is an extraction defect, and the splitter is
#: deliberately conservative about calling one: splitting on `and` breaks "attention deficit
#: and hyperactivity disorder", and on a comma it breaks "dementia, Alzheimer's type". 4% of
#: the corpus's values split here, where a loose test finds a separator in 13% -- the other 9%
#: are prose that should not have been put in the slot, and leaving them whole keeps them
#: visible as an extraction defect instead of shredding them into fragments.
SPLIT = re.compile(r"\s+and\s*/\s*or\s+|\s+or\s+|\s*/\s*|\s*;\s*")

#: Study-role nouns at the end of a value. `Schizophrenia patients` is a cohort description
#: whose condition is Schizophrenia; the head noun is about the study, not the disease.
TRAILING = re.compile(
    r"\s*\b(in some participants|in a subset|patients?|subjects?|"
    r"participants?|individuals?|group|cohort|status)\b\s*$",
    re.I,
)

# --- Risk ----------------------------------------------------------------------------------
# A group at risk of a condition does not have it: "familial high risk for schizophrenia",
# "premanifest Huntington's disease", "clinical high risk for psychosis". The condition is
# still linked, as any head is, and carries study_schema's `ConditionRole.at_risk` with one
# `RiskBasis` per kind of evidence the clause states. Decisions and the corpus survey:
# docs/condition-normalization.md, "Risk".

FAMILIAL, GENETIC = RiskBasis.familial.value, RiskBasis.genetic.value
CLINICAL, BIOMARKER = RiskBasis.clinical.value, RiskBasis.biomarker.value
CONDITION = RiskBasis.condition.value

#: The evidence words for each basis. Coded by the evidence, not the wording: "high risk
#: of schizophrenia for familial reasons" is `familial`.
_BASIS_WORDS = tuple(
    (RiskBasis(basis).value, re.compile(words, re.I))
    for basis, words in (
        ("familial", r"\bfamil(?:y|ial)\b|\bfirst[\s-]degree\b|\brelatives?\b|\bsiblings?\b|"
                     r"\boffspring\b|\bparent(?:s|al)?\b|\bmaternal\b|\bpaternal\b|\bco-?twins?\b|"
                     r"\bchildren of\b"),
        ("genetic", r"\bgen(?:e|es|etic|etically)\b|\bcarriers?\b|\bmutations?\b|\balleles?\b|"
                    r"\bpolygenic\b|\bAPOE|ε4|\bexpansion\b"),
        ("clinical", r"\bclinical(?:ly)?\b|\bultra[\s-]*high\b|\battenuated\b|\bprodrom|"
                     r"\bbasic symptoms\b|\bsub-?threshold\b|\bmental state\b"),
        ("biomarker", r"\bamyloid|\btau\b|\bbiomarkers?\b"),
        ("exposure", r"\bexpos(?:ed|ure)\b|\bmaltreat|\btrauma|\badversity\b|\bpreterm\b|"
                     r"\bprenatal\b|\bperinatal\b|\bcombat\b"),
        ("behavioral", r"\bdrinking\b|\bgambling\b|\bsubstance use\b|\bsmoking\b|\boffending\b"),
        ("trait", r"\btemperament|\bpersonality\b|\bneuroticism\b|\bbehavio(?:u)?ral inhibition\b|"
                  r"\bschizotyp|\bcognitive vulnerability\b"),
    )
)

_RISK_ADJ = (
    r"(?:familial|family|genetic(?:ally)?|clinical(?:ly)?|ultra|ultrahigh|high|elevated|"
    r"increased|enhanced|heightened|very|early|late|initial|cognitive|temperamental)"
)
#: "at risk for", "familial high risk of", "ultra-high-risk state of", "vulnerability to".
#: Not "risk factors": a group with vascular risk factors has them.
_RISK = re.compile(
    rf"\b(?:{_RISK_ADJ}[\s-]+)*(?:at[\s-]+)?(?:{_RISK_ADJ}[\s-]+)*"
    r"(?:risk|likelihood|vulnerability)(?:[\s-]+(?:state|stage))?"
    r"\s+(?:for|of|to)\s+(?:develop(?:ing)?\s+)?(?:an?\s+)?",
    re.I,
)
#: A low-risk comparison group, which is the absence of the risk.
_LOW_RISK = re.compile(
    r"\b(?:at\s+)?low(?:er)?[\s-]+(?:(?:familial|genetic)[\s-]+)?risk\b"
    r"(?:\s+(?:for|of)\s+(?:developing\s+)?(?P<object>[^;]*))?",
    re.I,
)
#: The psychosis-risk states, which name their condition by convention.
_PSYCHOSIS_RISK = re.compile(
    r"\b(?:at[\s-]+risk[\s-]+mental[\s-]+states?|"
    r"attenuated[\s-]+(?:psychosis|psychotic|positive)[\s-]+(?:symptoms?|syndrome)"
    r"(?:[\s-]+(?:state|syndrome))?(?:[\s-]+psychosis[\s-]+risk[\s-]+syndrome)?|"
    r"psychosis[\s-]+risk(?:[\s-]+(?:syndromes?|states?))?|basic[\s-]+symptoms|"
    r"(?:initial[\s-]+)?prodromal[\s-]+(?:syndromes?|states?|symptoms|psychosis)|prodromes?|"
    r"(?:ultra[\s-]*high|clinical(?:ly)?[\s-]+high)[\s-]+risk(?:[\s-]+(?:state|stage))?"
    r"(?=\s*(?:$|[;,(]|long|at\b)))",
    re.I,
)
#: SIPS's genetic risk and deterioration syndrome: a psychotic first-degree relative (or
#: schizotypal personality) with a fall in functioning. Familial and clinical, whatever
#: its name says.
_GRD = re.compile(
    r"\bgenetic\s+risk\s+(?:and|&)\s+deterioration(?:\s+(?:prodromal\s+)?syndrome)?", re.I
)
#: "unaffected first-degree relatives of schizophrenia patients", "offspring of familial AD".
_RELATIVES = re.compile(
    r"\b(?:(?:unaffected|non-?affected|non-?psychotic|healthy|undiagnosed|unmedicated|"
    r"non-?dependent|at[\s-]+risk|biological|first[\s-]+degree|second[\s-]+degree|"
    r"asymptomatic)[\s-]+)*"
    r"(?:relatives?|siblings?|sibs|offspring|children|daughters?|sons?|co-?twins?|"
    r"family\s+members?)\s+of\s+"
    r"(?:(?:the\s+)?(?:patients?|participants?|individuals?|persons?|people|someone|subjects?|"
    r"children|adults|probands?|parents?|mothers?|fathers?)\s+"
    r"(?:with|who\s+have|diagnosed\s+with)\s+)?"
    r"(?:an?\s+)?",
    re.I,
)
#: "family history of AD", "first-degree family history of late-onset AD".
_FAMILY_HISTORY = re.compile(
    r"\b(?:(?:a|positive|first[\s-]+degree)\s+)*(?:family|familial|parental|maternal|paternal)"
    r"\s+(?:psychiatric\s+)?history\s+of\s+(?:an?\s+)?",
    re.I,
)
#: A disease before its onset. Prodromal is early signs, so a clinical risk (James,
#: 2026-11-27); preclinical AD is defined by its biomarkers; a premanifest or presymptomatic
#: carrier is known by the gene.
_STAGE = re.compile(
    r"\b(?P<stage>prodromal|pre-?clinical|pre-?manifest|pre-?symptomatic|non-?manifesting)"
    r"[\s-]+(?!to\b)",
    re.I,
)
_STAGE_BASIS = {"prodromal": CLINICAL, "preclinical": BIOMARKER}
#: Huntington disease is known before onset only by the gene, whatever the stage is called.
_GENE_DEFINED = re.compile(r"huntington", re.I)
#: Preclinical Alzheimer disease is the biomarker-positive stage (NIA-AA); elsewhere
#: "preclinical" says only that it is below a diagnosis.
_AMYLOID_DEFINED = re.compile(r"alzheimer|\bAD\b", re.I)
#: A carrier of a named gene. A risk only when the gene is all it names, or the carrier is
#: said to be unaffected: "Definite Parkinson's disease in four homozygous carriers" is a
#: diagnosis.
_CARRIER = re.compile(
    r"^(?P<pre>(?:(?:asymptomatic|presymptomatic|pre-?symptomatic|non-?manifesting|"
    r"unaffected|healthy|cognitively\s+(?:normal|unimpaired|intact)|heterozygous|"
    r"homozygous|one|two|three|four|five|six|seven|eight|nine|ten|\d+)[\s;,-]+)*)"
    r"(?P<what>.+?)(?:[\s-]+(?:gene|mutation|expansion|repeat|allele|variant))*"
    r"[\s-]+(?:carriers?|carriage)(?:\s+(?:status|state))?$",
    re.I,
)
#: "high obstructive sleep apnea risk": the condition before the word. Tried last.
_RISK_AFTER = re.compile(
    r"^(?:(?:high|elevated|increased|heightened)[\s-]+)(?P<object>.+?)[\s-]+risk$", re.I
)
#: Relatives with no condition of their own named: part of a familial risk, not a head.
_KIN = re.compile(
    r"^(?:(?:unaffected|non-?affected|non-?psychotic|healthy|undiagnosed|unmedicated|"
    r"non-?dependent|biological|first[\s-]+degree|second[\s-]+degree|asymptomatic)[\s-]+)*"
    r"(?:relatives?|siblings?|sibs|offspring|co-?twins?|family\s+members?)$",
    re.I,
)
_NONCARRIER = re.compile(r"\bnon-?(?![-\s]?manifest)(?:\S+\s+)?carriers?\b", re.I)
#: What follows the condition and is not part of it.
_OBJECT_END = re.compile(
    r"\s+(?:risk|at\s+(?:the\s+)?(?:baseline|intake|scanning|initial|revisit|year|follow)|"
    r"with\s+(?:a\s+)?(?:genetic|attenuated|reliably|no)\b|defined\s+by|for\s+familial|"
    r"based\s+on|due\s+to\s+(?:a\s+)?famil\w*|in\s+(?:one|a|\d+)\s+stud|long[\s-]term)\b.*$"
    r"|\s*\(.*$",
    re.I,
)
_CARRIER_TAIL = re.compile(
    r"(?:[\s-]+(?:gene|mutation|expansion|repeat|allele|pathogenic|variant|risk|carriers?|"
    r"carriage|status|state))+$",
    re.I,
)
#: The words a risk clause's leftover can be without naming anything.
_CONNECTIVE = re.compile(
    r"^(?:\W|\b(?:with|and|or|a|an|the|who|are|were|at|having|has|have|least|most|"
    r"one|two|three|any|some|\d+)\b)+|"
    r"(?:\W|\b(?:with|and|or|a|an|the|who|are|were|at|having|has|have|least|most|"
    r"one|two|three|any|some|\d+)\b)+$",
    re.I,
)
#: An earlier disjunct that says some of the group have the condition the risk names.
_HAS = re.compile(r"\b(?:current|personal|lifetime|own|with|have|having)\s*$", re.I)
_ONLY_ADJ = re.compile(rf"^(?:{_RISK_ADJ}|and|or|/|\W)+$", re.I)


def _gene_only(text: str) -> bool:
    """Whether a carrier's text names genes and no condition: `APOE ε4`, `G2019S LRRK2`."""
    words = re.findall(
        r"[^\W\d_]+",
        re.sub(r"orf|apolipoprotein|parkin|progranulin|presenilin|hexanucleotide|"
               r"trinucleotide", "", text, flags=re.I),
    )
    return bool(text.strip()) and not any(
        re.search(r"[a-z]{3}", w) and w.lower() not in {"and", "or", "the"} for w in words
    )


def _bases(text: str, *implied: str) -> tuple[str, ...]:
    """The `RiskBasis` values a clause's words give, with those its form implies, in enum order."""
    found = set(implied) | {basis for basis, words in _BASIS_WORDS if words.search(text)}
    return tuple(b.value for b in RiskBasis if b.value in found)


@dataclass
class _Risks:
    """What a value's risk reading left: text to triage as usual, and the risks it lifted."""

    regular: list[str] = field(default_factory=list)
    #: (the condition's text, its bases)
    named: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    #: (a gene or ""; its bases) for a risk whose condition the value does not name
    unnamed: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)


def _object(text: str, known: Callable[[str], bool] | None) -> str:
    """The condition a risk phrase is about, without what follows it."""
    text = re.sub(r"^.*\bdiscordant\s+for\s+", "", text.strip(" ,;:."), flags=re.I)
    if known is not None and known(text):
        return text
    text = _CARRIER_TAIL.sub("", _OBJECT_END.sub("", text)).strip(" ,;:.")
    return re.sub(r"\s+(?:probands?|cases|index\s+patients)$", "", text, flags=re.I)


_FORMS = (
    ("grd", _GRD), ("psychosis", _PSYCHOSIS_RISK), ("relatives", _RELATIVES),
    ("history", _FAMILY_HISTORY), ("stage", _STAGE), ("carrier", _CARRIER), ("risk", _RISK),
    ("after", _RISK_AFTER),
)


def _match(clause: str):
    """The leftmost risk form in a clause, the earlier-listed on a tie: (rule, match), or None."""
    found = [(m.start(), i, rule, m) for i, (rule, pattern) in enumerate(_FORMS)
             if (m := pattern.search(clause)) is not None]
    found = [f for f in found if f[2] != "after"] or found
    return min(found, key=lambda f: f[:2])[2:] if found else None


def _read_risks(clause: str, known: Callable[[str], bool] | None, out: _Risks) -> None:
    """Lift the risk out of one `;`-clause, if it states one, and file what is left."""
    low = _LOW_RISK.search(clause)
    if low is not None:
        if low.group("object"):
            out.denied.append(low.group("object").strip())
        clause = (clause[: low.start()] + clause[low.end():]).strip()
        if clause:
            out.regular.append(clause)
        return
    if _NONCARRIER.search(clause):
        out.denied.append(clause)
        return
    found = _match(clause)
    if found is None:
        out.regular.append(clause)
        return
    rule, m = found
    before, after = clause[: m.start()], clause[m.end():]
    # The prefix is this risk's own words back to the last separator; an earlier disjunct
    # is a statement of its own ("APOE4 carrier status or family history of AD") unless it
    # is only a kind of risk ("genetic or clinical high risk").
    cut = list(SPLIT.finditer(before))
    own, also_has = before, False
    if cut and not _ONLY_ADJ.match(before[: cut[-1].start()]):
        own, earlier = before[cut[-1].end():], before[: cut[-1].start()]
        # "current or family history of X", "participants with or at risk for X": some
        # of the group have X.
        also_has = bool(_HAS.search(earlier))
        earlier = _HAS.sub("", earlier)
        if _CONNECTIVE.sub("", earlier).strip():
            _read_risks(earlier, known, out)
    rest = _CONNECTIVE.sub("", own).strip()
    bare = re.sub(r"\b(?:and|or|with)\b", " ", rest)
    kin = bool(_KIN.match(TRAILING.sub("", bare).strip()))
    # "manifest and premanifest Huntington disease": some of the group have it.
    if re.fullmatch(r"(?:manifest|symptomatic|affected)", rest, re.I):
        rest, also_has = "", True
    present = bool(rest) and not (_ONLY_ADJ.match(rest) or HEALTHY.match(bare) or kin)
    span = own + clause[m.start():]

    if rule in ("grd", "psychosis"):
        # The condition is psychosis unless the clause names another; what follows is a
        # disjunct of its own, or a trailer ("at baseline", "(ARMS)").
        target = re.match(r"\s+(?:for|of)\s+(?:developing\s+)?(?:an?\s+)?(.+)$", after, re.I)
        sep = SPLIT.match(after) if target is None else None
        if sep is not None and after[sep.end():].strip():
            _read_risks(after[sep.end():], known, out)
        obj = _object(target.group(1), known) if target else "psychosis"
        implied = (FAMILIAL, CLINICAL) if rule == "grd" else (CLINICAL,)
        span = own if rule == "grd" else own + m.group(0)
    elif rule == "carrier":
        what = m.group("what")
        if not (m.group("pre") or _gene_only(what)):
            out.regular.append(clause)
            return
        obj = "" if _gene_only(what) else _object(what, known)
        implied = (GENETIC,)
        if not obj:
            out.unnamed.append((what.strip(), _bases(span, GENETIC)))
            return
    else:
        raw = m.group("object") if rule == "after" else after
        obj = _object(raw, known)
        implied = ()
        # "unaffected familial high-risk individuals": a kind of risk, and no condition.
        if rule == "after" and _ONLY_ADJ.match(
            re.sub(rf"\b(?:{_WELL})\b", "", obj, flags=re.I) or "-"
        ):
            obj = ""
        if rule == "stage":
            stage = re.sub(r"[\s-]", "", m.group("stage").lower())
            # "preclinical depressive symptoms" is below a diagnosis, not before one;
            # "presymptomatic or early symptomatic HD" is a mixed group.
            if not obj or re.search(r"\bsymptoms?$|^(?:or|and)\b", obj, re.I):
                out.regular.append(clause)
                return
            carriers = re.match(r"carriers?\s+of\s+(?:the\s+)?(.+)$", obj, re.I)
            if carriers:
                out.unnamed.append((carriers.group(1), _bases(span, GENETIC)))
                return
            gene_defined = bool(_GENE_DEFINED.search(obj))
            implied = (_STAGE_BASIS.get(stage, GENETIC),) + ((GENETIC,) if gene_defined else ())
            if gene_defined or (stage == "preclinical" and not _AMYLOID_DEFINED.search(obj)):
                implied = tuple(b for b in implied if b != BIOMARKER)
            # A carrier's gene, not a disease: "presymptomatic GRN mutation carrier".
            if _CARRIER_TAIL.search(_OBJECT_END.sub("", raw.strip())) and _gene_only(obj):
                out.unnamed.append((obj, _bases(span.replace(obj, " "), *implied)))
                return
        elif rule in ("relatives", "history"):
            if re.match(r"non-?\w", obj, re.I):
                out.regular.append(clause)
                return
            implied = (FAMILIAL,)
    if kin:
        implied = (*implied, FAMILIAL)
    # The condition's own words are not evidence ("familial risk for substance use disorder"
    # is not behavioral), except that a familial form of it is.
    bases = _bases(span.replace(obj, " ", 1) if obj else span, *implied,
                   *((FAMILIAL,) if re.search(r"\bfamilial\b", obj, re.I) else ()))
    # "insomnia with a high risk of depression": the condition the group has is the
    # evidence, when the clause names no other.
    if rule == "risk" and present and not bases:
        bases = (CONDITION,)
    if not obj:
        out.unnamed.append(("", bases))
        return
    if present:
        out.regular.append(rest)
    if also_has:
        out.regular.append(obj)
    out.named.append((obj, bases))


def risks(
    text: str, known: Callable[[str], bool] | None = None, negated: bool = False
) -> _Risks:
    """Read the risk out of each `;`-clause of a value.

    `negated`: pass a clause that carries a negation cue through untouched, for the
    negation reading to take first ("no family history of dyslexia" is an absence). The
    cue `unaffected` is not one here: in "unaffected siblings of schizophrenia patients" it
    says the siblings do not have what they are at risk of.
    """
    out = _Risks()
    for clause in (c.strip() for c in text.split(";")):
        if not clause:
            continue
        if negated and any(
            c.group(0).lower() != "unaffected" for c in negation._CUE.finditer(clause)
        ):
            out.regular.append(clause)
            continue
        _read_risks(clause, known, out)
    return out


#: What triage concluded a value was, when it was not a condition to look up.
NO_CONDITION = "NO_CONDITION"

#: A value whose only content is a risk of a condition it does not name ("APOE ε4 carrier").
AT_RISK = "AT_RISK"

#: What triage concluded when the value is a non-answer.
NOT_READ = "NOT_READ"


@dataclass(frozen=True)
class Triaged:
    """A value decomposed into what to look up, what was lifted off it, and what it denies."""

    heads: tuple[str, ...]
    qualifiers: tuple[str, ...]
    kind: str
    #: Heads the value states the ABSENCE of. Never looked up, and carried rather than
    #: dropped: "screened and found none" is not "never asked".
    denied: tuple[str, ...] = ()
    #: How the negation was read -- `parse`, `cue`, `assertion`, or "". The method is the
    #: confidence, the same way `Mapping.method` is.
    scope: str = ""
    #: Parallel to `heads`: None for a condition the group has, and for one it is only at
    #: risk of (`ConditionRole.at_risk`), the `RiskBasis` values its clause states, which may
    #: be none.
    risk: tuple[tuple[str, ...] | None, ...] = ()
    #: The bases of a risk whose condition the value does not name ("APOE ε4 carrier"),
    #: or None when there is no such risk.
    unnamed: tuple[str, ...] | None = None

    def bases(self, index: int) -> tuple[str, ...] | None:
        """The risk bases of `heads[index]`, or None if the group has that condition."""
        return self.risk[index] if index < len(self.risk) else None

    @property
    def present(self) -> tuple[str, ...]:
        """The heads the group has, not those it is only at risk of."""
        return tuple(h for i, h in enumerate(self.heads) if self.bases(i) is None)

    @property
    def sentinel(self) -> str:
        """The answer when there is nothing to look up, and "" when there is."""
        return self.kind if self.kind in (NO_CONDITION, NOT_READ, "empty") else ""


def _heads_of(text: str) -> tuple[list[str], list[str]]:
    """Split a positive span into heads, lifting the course-and-state qualifiers off."""
    heads: list[str] = []
    quals: list[str] = []
    for part in (p.strip() for p in SPLIT.split(text) if p and p.strip()):
        quals += [q.lower() for q in QUALIFIER.findall(part)]
        head = TRAILING.sub("", QUALIFIER.sub("", part)).strip(" -,")
        head = re.sub(r"\s{2,}", " ", re.sub(r"\(\s*\)", "", head)).strip().replace(_KEPT, "/")
        if head:
            heads.append(head)
    return heads, quals


def triage(raw: object, known: Callable[[str], bool] | None = None) -> Triaged:
    """Head terms, the qualifiers lifted off them, and what the value turned out to be.

    Four layers in order: `UNREAD`, `HEALTHY`, a negation scope over the WHOLE value, then
    a cue window over each part of what survives. Parsing before the split is what lets a
    cue scope over a coordination -- see docs/condition-normalization.md.

    What the value is *about* is all this decides. Whether it belongs in the slot is the
    schema's question: `healthy smokers` in `medical_condition` is a slot defect, and the
    condition in it is recovered rather than swallowed.

    `TRAILING` runs twice on purpose -- once on the whole value and once per head -- because
    a compound puts a role noun on the last head only.

    `known`, given, says whether a string is a name the caller's vocabulary holds. A value
    that is one whole name is then one head, before negation or splitting could break it:
    "migraine without aura" reads "without" as a negation cue, and "chronic fatigue
    syndrome" loses `chronic` as a course qualifier.
    """
    raw_text = str(raw or "").strip()
    if not raw_text:
        return Triaged((), (), "empty")
    if UNREAD.match(raw_text):
        return Triaged((), (), NOT_READ, scope="assertion")
    text = TRAILING.sub("", PREVALENCE.sub("", raw_text).strip(" ,;:")).strip()
    if not text:
        return Triaged((), (), "empty")
    if HEALTHY.match(text):
        return Triaged((), (), NO_CONDITION, scope="assertion")
    if known is not None and known(text) and _match(text) is None:
        return Triaged((text,), (), "single")
    text = _SUBSTANCE_PAIR.sub(r"\1 \2\3\1 \4", _DUE_TO.sub(r"\1; ", text))
    text = IN_NAME.sub(lambda m: m.group(1) + _KEPT, text)
    if HEALTHY.match(text):
        return Triaged((), (), NO_CONDITION, scope="assertion")

    # Risks first in the clauses no negation cue touches: the parse reads `unaffected` as
    # one, and "unaffected siblings of schizophrenia patients" is a risk, not an absence.
    early = risks(text, known, negated=True)
    if early.named or early.unnamed or early.denied:
        text = "; ".join(early.regular)

    scoped = negation.scope(text) if text else ("", "")
    if scoped is not None:
        asserted, denied_text, how = scoped[0], scoped[1], "parse"
    else:
        # The cue window runs over the WHOLE value before the split, not only per part.
        # `SPLIT` breaks on ` or `, and without a parse nothing else knows that a cue
        # scopes over the coordination it introduces: "no neurological or psychiatric
        # disorder" would hand `psychiatric disorder` to the vocabulary as a diagnosis.
        asserted = negation.cue_forward_scope(text)
        denied_text = text[len(asserted) :] if asserted != text else ""
        how = "cue"

    # The risk reading, on what the negation left: "no family history of dyslexia" is an
    # absence before it is a risk. What it lifts is linked below as any head is.
    read = risks(asserted, known)
    read.named[:0], read.unnamed[:0], read.denied[:0] = early.named, early.unnamed, early.denied
    heads, quals, denied = [], [], list(read.denied)
    for part in (p.strip() for p in SPLIT.split("; ".join(read.regular)) if p and p.strip()):
        kept = negation.cue_forward_scope(part)
        if fold(kept) != fold(part):
            denied.append(part)
            how = how if kept else ("cue" if scoped is None else how)
        part_heads, part_quals = _heads_of(kept)
        heads += part_heads
        quals += part_quals
    if denied_text.strip():
        denied.append(denied_text.strip())

    at_risk: list[str] = []
    risk: list[tuple[str, ...]] = []
    for text_, bases in read.named:
        object_heads, object_quals = _heads_of(text_)
        quals += object_quals
        for head in object_heads:
            # One head per condition, with the union of its bases.
            same = [i for i, h in enumerate(at_risk) if fold(h) == fold(head)]
            if same:
                merged = set(risk[same[0]]) | set(bases)
                risk[same[0]] = tuple(b.value for b in RiskBasis if b.value in merged)
                continue
            at_risk.append(head)
            risk.append(bases)
    unnamed: tuple[str, ...] | None = None
    if read.unnamed:
        quals += [gene for gene, _ in read.unnamed if gene]
        unnamed = tuple(
            b.value for b in RiskBasis if any(b.value in bases for _, bases in read.unnamed)
        )

    # The wellness test again, on what the negation left standing: "healthy adults without
    # neurologic disorders" survives as "healthy adults", which is still an absence.
    # And once per head, which the joined test cannot do: a compound states one thing per
    # part, so "Healthy; obesity" is a cohort with obesity and `Healthy` is not a second
    # condition to look up. The joined test comes first because a head can be a bare role
    # noun -- "healthy; adults" joins to a wellness phrase and neither part is one.
    stated = bool(heads)
    if heads and HEALTHY.match(" ".join(heads)):
        heads = []
    heads = [h for h in heads if not HEALTHY.match(h)]
    quals_out = tuple(sorted(set(quals)))

    if not heads and not at_risk:
        if unnamed is not None:
            return Triaged((), quals_out, AT_RISK, tuple(denied), how if denied else "",
                           unnamed=unnamed)
        if stated:
            return Triaged((), quals_out, NO_CONDITION, tuple(denied), how)
        kind = NO_CONDITION if denied else "empty"
        return Triaged((), quals_out, kind, tuple(denied), how if denied else "")
    every = (*heads, *at_risk)
    return Triaged(
        every,
        quals_out,
        "compound" if len(every) > 1 else "single",
        tuple(denied),
        how if denied else "",
        risk=(None,) * len(heads) + tuple(risk),
        unnamed=unnamed,
    )


def absent(text: object, is_healthy: object = None) -> bool:
    """Does this cohort record an absence? The string decides; `is_healthy` fills gaps.

    The flag can add an absence and never remove one: it is derived from this field, so a
    flag that could overrule its own source would be a loop. Measurements in
    docs/condition-normalization.md.
    """
    triaged = triage(text)
    if triaged.kind == NO_CONDITION:
        return True
    if triaged.heads:
        return False
    return is_healthy is True
