# The omnibus interaction is extracted; its simple effects are not

Found by auditing three placebo-controlled trials whose records carried a
treatment-versus-comparator contrast with no direction. Two of the three failed the same
way, and the failure is not a wrong field. It is a **missing analysis**.

## What happens

A placebo-controlled imaging trial is usually analysed as a `treatment x time` or
`group x treatment` interaction, and the paper then reports post-hoc simple effects that
say which way it went. The extraction captures the interaction and stops.

`39101053` -- bright light therapy against placebo. The record holds two analyses, both
omnibus interactions, every cell `undirected`. The paper says:

> Post-hoc analyses indicated that **BLT group had lower sFC than placebo group in
> post-treatment**.

That sentence is a `placebo > treatment` contrast. The record does not contain it.

`32929215` -- ketamine against placebo, in treatment-resistant depression and healthy
volunteers. One analysis, `Group x treatment`, every cell `undirected`. The paper says:

> functional connectivity between VS-left dlPFC, DC-right vlPFC, DCP-pgACC and VRP-OFC
> was **increased in TRD participants but decreased in HVs** post-ketamine

A crossover interaction, whose correct encoding is *two* analyses with opposite
directions. The record has neither.

The third, `35493813`, is not a failure and is the useful control: its between-group
analysis is `direction: not_reported`, and the paper's only statement is "there was also a
significant difference ... between the experimental group and the control group". The sign
appears nowhere in the text -- only in a figure's bar panel. `not_reported` is correct, and
the extractor got it right.

## `undirected` on the interaction is not the bug

An F-test for an interaction has no direction, and encoding it as `undirected` is what the
schema asks for. Changing that would be wrong. What is missing is the *additional*
analyses the paper reports alongside it.

## Why it matters more than three papers

Placebo-controlled imaging trials commonly use interaction designs. This
failure systematically removes exactly the directed treatment-versus-comparator contrasts
a coordinate-based meta-analysis needs, while leaving a record that looks complete: the
arms are there, the analysis is there, the cells are there, and every direction is a
defensible `undirected`. Nothing downstream can tell that a post-hoc result was dropped.

`docs/pipeline-variant-results.md` reports 26 of 89 depression papers yielding an
arm-versus-arm contrast. An unknown share of the remainder is this.

## The fix

The analyses pass is shown one stage-1 entry per coordinate table and told to emit one
`analyses` entry for each, with two named departures -- SPLIT when a table's rows are
distinguished by a column the name does not mention, and OMIT when a table reports no
tested effect. Neither covers this case, because the *table* is fine: it is the
interaction's table, and the post-hoc effects have no table of their own.

The instructions need a third departure that mirrors SPLIT:

**ADD an entry for a post-hoc or simple effect the text reports for an interaction whose
table you were given.** An omnibus interaction has no direction, and a paper that reports
one almost always also reports which way it went, in prose: "post-hoc analyses indicated
that X was lower than Y", "increased in A but decreased in B". Each such statement is its
own analysis, contrasting the two levels the sentence names, with the direction the
sentence gives. It carries the interaction's `tables` reference, because that is where its
coordinates are.

Two guards belong in the same instruction, because the failure mode of "emit more
analyses" is invention:

- The added entry must quote the sentence that states the direction. A simple effect with
  no sentence behind it is exactly the thing that must not be added.
- It must not be added when the paper reports only that a difference was significant.
  `35493813` is that case, and an added entry there would replace an honest
  `not_reported` with a guess.

## A separate stage-1 bug, found in passing

`32929215`'s parse produced 6 coordinate points with **no statistic values at all**
(`kinds={}`). Even a correct downstream pass could not have recovered a direction from
that table, and `parse_tables.split_opposite_signs` cannot act on rows that carry no
sign. Worth its own look: how often does the parse extract coordinates but drop the
statistic column?


## Verification: all 27 read against their papers

Every record whose arms are present but whose cells link to no arm, from
`data/eval/meta/excluded-with-arms.json`, read against the paper text.

| class | n | what it means |
|---|---|---|
| **A** | 7 | the between-arm analysis the paper reports is missing from the record |
| **B** | 2 | interaction captured; its simple effects missing or its directions wrong |
| **C** | 5 | within-arm pre/post, but no arm cell -- the change cannot be attributed |
| **D** | 2 | the contrast is in the record; my query fails to link the level to the arm |
| **E** | 1 | no analyses extracted at all |
| **F** | 1 | an arm is mis-typed |
| **G** | 9 | correctly excluded -- the paper reports no between-arm imaging contrast |

**18 of 27 are extraction problems; 9 are correct exclusions.**
Class A alone -- the failure this document is about -- accounts for 7, and B for another 2.

### A -- the between-arm analysis the paper reports is missing from the record (7)

- **24682502** -- modafinil vs placebo. Record has only parametric reward-magnitude contrasts (+¥20/+¥100/+¥500). Paper: 'The modafinil condition showed significantly higher BOLD signal change at the highest gain (+¥500) cue compared to the placebo condition.'
- **35903009** -- fluoxetine vs placebo. Record has only 'Main effect of task across groups'. Paper: 'increased visuo-cerebellar activity ... when compared to depressed adolescents on placebo.'
- **17625917** -- reboxetine vs placebo. Record has only task/valence contrasts. Paper: 'Reboxetine reduced activation during successful retrieval in a fronto-parietal network compared to placebo.'
- **19585106** -- citalopram vs placebo. Record has only 'Main effect of fear vs. happy (placebo)'. Paper: 'Citalopram was associated with increased amygdala activation to happy faces relative to placebo control.'
- **38580858** -- active tPEMF vs sham. Record has only task contrasts. Paper: 'Participants in the active treatment group showed a stronger decrease in activation post-treatment compared to sham during reward-outcome processing in the left inferior frontal gyrus.'
- **39111747** -- verum tFUS vs sham. Record has the within-arm pre/post but not the between-arm one. Paper: 'higher FC correlation between the right superior sgACC and several other brain regions in the verum group compared with the sham group.'
- **38040678** -- ketamine vs placebo. Record has ONE analysis, 'Group', TRD/HV, no directions. Paper: 'ketamine differentially modulated rsFC to the right insula and anterior ventromedial prefrontal cortex, compared to placebo, in TRD vs HV' and 'in the TRD group alone, sgACC rsFC was most substantially modulated by ketamine vs placebo'. Severe under-extraction.

### B -- interaction captured; its simple effects missing or its directions wrong (2)

- **20421135** -- BATD vs no intervention. One Group x Time analysis, every direction None rather than undirected. Paper reports pre-to-post connectivity increases by arm.
- **38681626** -- electro- vs sham acupuncture. Directions asserted ON the omnibus interaction's group cells (EAS positive, SAS negative), but the paper's post hoc says 'In both groups, ALFF was increased in the left MTG and the left CPL compared to the pre-treatment group.' The encoding claims a group ordering the post hoc contradicts.

### C -- within-arm pre/post, but no arm cell -- the change cannot be attributed (5)

- **33038791** -- bifrontal ECT. Four Before/After ECT analyses with no arm cell, so the change cannot be attributed to an arm.
- **19087899** -- TSD + paroxetine / TSD + placebo / paroxetine only. Nineteen pre/post analyses, no arm cell. The paper's between-arm comparisons are clinical (HDS-13), not imaging.
- **30353262** -- ketamine vs saline. One 24h-post vs baseline analysis, no arm cell.
- **32440602** -- EFMT vs CT. Two pre/post analyses, no arm cell.
- **31437695** -- ketamine 0.5 / 0.2 mg/kg / placebo. Arms ARE held correctly per analysis, but every pre/post direction is None. Not a link failure -- a missing direction.

### D -- the contrast is in the record; my query fails to link the level to the arm (2)

- **29740358** -- craving behavioural intervention vs non-intervention. Cells are 'CBI+' and 'CBI-' with directions; arms are 'craving behavioral intervention' and 'non-intervention'. The link needs the abbreviation CBI, which the query does not expand.
- **39402015** -- ketamine / lamotrigine-ketamine / placebo-placebo. The record HAS the directed contrast ('placebo-ketamine group' negative vs 'placebo-placebo group' positive), matching the paper's 'contrast: PK < PP'. My query drops it because word containment lets the shorter arm name 'placebo-placebo' also match the level 'placebo-ketamine group' -- the level then names two arms and is discarded.

### E -- no analyses extracted at all (1)

- **39536019** -- home-based tDCS vs sham. ZERO analyses extracted. The paper reports EEG lagged-coherence results, not fMRI coordinates, so OMIT may be right -- but a record with no analyses and a paper with results needs checking, not assuming.

### F -- an arm is mis-typed (1)

- **17601497** -- fluoxetine vs 'healthy control' typed as a no_intervention ARM. Healthy controls are a comparison group, not a trial arm; the mis-typing makes the study look placebo-controlled when its imaging contrasts are patient-vs-control.

### G -- correctly excluded -- the paper reports no between-arm imaging contrast (9)

- **27880789** -- RFCBT vs assessment only. Within-arm pre/post with the arm held. No between-arm imaging contrast in the paper.
- **26454185** -- behavioural activation vs no-treatment. Imaging analyses are subthreshold-depression vs control group contrasts.
- **27133029** -- lanicemine / ketamine / placebo. The one analysis is ketamine vs lanicemine -- two intervention arms. Paper: 'There was no antidepressant effect of either drug when compared to saline.'
- **30685701** -- buprenorphine vs placebo. Imaging results are MADRS correlations. Paper: 'There was no significant group (placebo vs ...)'.
- **21969917** -- SYN115 / placebo / levodopa. Drug is a nuisance factor in the design; the reported CBF results are valence/arousal components.
- **32726408** -- ketamine doses vs saline. The imaging analyses are baseline TRD-vs-healthy-control comparisons.
- **19423079** -- typhoid vaccine vs placebo. The between-arm result quoted is serum IL-6, not imaging.
- **38288056** -- dual / single / sham stimulation. Four within-arm pre/post analyses with 'dual group' held -- correctly encoded, simply not a between-arm contrast.
- **32111579** -- active vs sham tDCS. Six task contrasts, some split by group. The between-arm sentences found are background citations, not this study's results.
