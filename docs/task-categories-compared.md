# MiniLM vs SapBERT: the same method, one encoder swapped

Both runs hold everything constant except which encoder the two **entity** channels
(`name`, `conditions`) use. The two full listings and the per-task table are gone: they were outputs of
`scripts/cluster_tasks.py`, a pipeline stage since folded into
`pondie/normalization/task.py`, and keeping stale outputs of deleted code is worse than
keeping none. The counts and the diff below are the finding and they stand. Re-run
`python -m pondie.normalization.task --encoder sapbert` to regenerate the comparison against
the current pipeline — the numbers will differ, because the pipeline has changed since.

| | MiniLM | SapBERT |
|---|---|---|
| categories | 173 | 132 |
| tasks in a category | 1468 | 1546 |
| in no category | 204 | 126 |
| largest category | 226 | 272 |

**ARI 0.731.** SapBERT merges harder: 41 fewer categories, and it pulls 84 tasks out of
"no category" that MiniLM left alone (6 go the other way).

Which is right is not settled here and cannot be from these numbers — there is no gold. What
follows is the structural diff so the disagreements can be read.

## Where SapBERT merges what MiniLM kept apart

36 SapBERT categories draw from more than one MiniLM category. The largest:

| SapBERT category | tasks | merged from these MiniLM categories |
|---|---|---|
| **emotion regulation task** | 300 | emotion regulation task (220); cognitive reappraisal of sad images (20); emotional film viewing (7); cognitive reappraisal (6); emotion induction task |
| **resting-state protocol** | 183 | resting-state protocol (170); resting-state PET (7); repeat letters during PET uptake (3); Resting-state fixation (2); + 1 that MiniLM left uncategori |
| **visual food cues** | 112 | visual food cues (95); food cue-reactivity task (8); food picture viewing (4); food craving reappraisal task (4); object-sensitive functional localize |
| **cue exposure task** | 91 | smoking cue exposure (31); cue-induced craving task (21); cue-reactivity paradigm (11); cannabis cue-exposure task (8); cue exposure task (7); smoking |
| **cue reactivity task** | 75 | smoking cue exposure (31); cue reactivity task (28); cue-reactivity paradigm (7); cue-induced craving task (6); food cue-reactivity task (1); block-re |
| **facial expression processing** | 73 | facial expression processing (24); emotional faces task (17); implicit facial expression task (13); affect labeling (10); dynamic faces task (2); targ |
| **emotional Stroop task** | 64 | implicit emotion processing task (17); distraction task (9); emotional go/no-go task (8); emotional Stroop task (7); approach-avoidance task (3); hybr |
| **alcohol cue reactivity task** | 57 | alcohol cue reactivity task (36); Taste Cue Paradigm (12); implicit approach-avoidance task (3); cue reactivity task (2); cue-reactivity paradigm (2); |
| **cue-reactivity task** | 38 | smoking cue-reactivity task (19); attentional bias line counting task (6); smoking cue exposure (3); block-related cue-exposure (3); working memory re |
| **monetary incentive delay task** | 35 | monetary incentive delay task (24); monetary reward paradigm (5); reward and personal reference task (3); affective Posner task (2); + 1 that MiniLM l |
| **drug Stroop fMRI task** | 21 | drug Stroop fMRI task (16); color-word Stroop task (3); computerized smoking emotional Stroop task (2) |
| **food craving reappraisal task** | 18 | food craving task (9); food craving reappraisal task (9) |
| **go/no-go task** | 15 | go/no-go task (10); alcohol Go/NoGo task (5) |
| **chocolate stimulus task** | 12 | milkshake paradigm (9); chocolate stimulus task (3) |
| **Beliefs** | 12 | Beliefs (8); voluntary emotion regulation during negative autobiographical memories (3); positive mood induction (1) |
| **personalized guided-imagery task** | 11 | personalized guided-imagery task (10); alcohol cue reactivity task (1) |
| **Go-NoGo response inhibition** | 10 | Go-NoGo response inhibition (6); Virtual-environment cue exposure treatment (3); + 1 that MiniLM left uncategorised |
| **rewarded guessing task** | 10 | rewarded guessing task (5); monetary incentive delay task (2); fear conditioning and extinction paradigm (1); + 2 that MiniLM left uncategorised |
| **modified Sternberg working memory paradigm** | 7 | modified Sternberg working memory paradigm (3); encoding and immediate JOLs task (3); working memory recall task (1) |
| **Faces** | 7 | Faces (5); Social Evaluation Task (2) |
| **three film viewing paradigm** | 6 | videotapes designed to elicit happy feelings, sad feelings, or the desire to use cocaine (2); cue-induced craving task (2); personalized guided-imager |
| **picture viewing** | 6 | picture viewing (3); chocolate and neutral pictures (2); + 1 that MiniLM left uncategorised |
| **music emotion processing passive-listening p** | 6 | music emotion processing passive-listening paradigm (3); positive mood induction (1); + 2 that MiniLM left uncategorised |
| **emotion regulation with aversive pictures** | 6 | emotion regulation with aversive pictures (4); neurofeedback task (2) |
| **Cue-induced MRI scanning procedure** | 5 | Cue-induced MRI scanning procedure (2); cue exposure fMRI task (2); cue-reactivity paradigm (1) |
| **anticipatory anxiety paradigm** | 5 | anticipatory anxiety paradigm (2); aversive and neutral anticipation (1); aversive classical conditioning with cognitive emotion regulation (1); + 1 t |
| **Wheel of Fortune task** | 5 | Wheel of Fortune task (2); RT probe task (1); + 2 that MiniLM left uncategorised |
| **Pavlovian-instrumental transfer task** | 5 | Pavlovian-instrumental transfer task (3); Temporal Difference Error/Juice Paradigm (2) |
| **object-sensitive functional localizer** | 5 | object-sensitive functional localizer (3); functional localizer (2) |
| **watching movies** | 4 | watching movies (2); facial expression processing (1); + 1 that MiniLM left uncategorised |

## Where MiniLM merges what SapBERT kept apart

37 MiniLM categories are split by SapBERT. The largest:

| MiniLM category | tasks | split by SapBERT into |
|---|---|---|
| **emotion regulation task** | 224 | emotion regulation task (220); cue exposure task (1); worry modulation task (1); emotional Stroop task (1); + 1 left uncategorised |
| **visual food cues** | 96 | visual food cues (95); Food portion size cue fMRI task (1) |
| **smoking cue exposure** | 67 | cue reactivity task (31); cue exposure task (31); cue-reactivity task (3); emotion regulation task (1); + 1 left uncategorised |
| **alcohol cue reactivity task** | 40 | alcohol cue reactivity task (36); cue exposure task (2); personalized guided-imagery task (1); cue-reactivity task (1) |
| **cue reactivity task** | 31 | cue reactivity task (28); alcohol cue reactivity task (2); emotion regulation task (1) |
| **cue-induced craving task** | 29 | cue exposure task (21); cue reactivity task (6); three film viewing paradigm (2) |
| **monetary incentive delay task** | 27 | monetary incentive delay task (24); rewarded guessing task (2); + 1 left uncategorised |
| **facial expression processing** | 26 | facial expression processing (24); social contact, attachment, and social regulation of emotion (1); watching movies (1) |
| **cue-reactivity paradigm** | 23 | cue exposure task (11); cue reactivity task (7); alcohol cue reactivity task (2); cue-reactivity task (1); Cue-induced MRI scanning procedure (1); + 1 |
| **smoking cue-reactivity task** | 22 | cue-reactivity task (19); cue exposure task (3) |
| **implicit emotion processing task** | 22 | emotional Stroop task (17); implicit emotion processing task (3); affective two-back working memory task (1); alternating emotion-identification/digit |
| **drug Stroop fMRI task** | 18 | drug Stroop fMRI task (16); scrambled sentences task (1); emotional Stroop task (1) |
| **emotional Stroop task** | 15 | emotional conflict task (8); emotional Stroop task (7) |
| **distraction task** | 13 | emotional Stroop task (9); emotion regulation task (4) |
| **food craving reappraisal task** | 13 | food craving reappraisal task (9); visual food cues (4) |
| **personalized guided-imagery task** | 11 | personalized guided-imagery task (10); three film viewing paradigm (1) |
| **cannabis cue-exposure task** | 11 | cue exposure task (8); Methamphetamine Cues Task (2); cue-reactivity task (1) |
| **go/no-go task** | 11 | go/no-go task (10); go/no-go inhibitory task (1) |
| **food cue-reactivity task** | 9 | visual food cues (8); cue reactivity task (1) |
| **emotional film viewing** | 8 | emotion regulation task (7); male–female interaction film scenes (1) |

## Reading it

The cue-reactivity fragmentation that [task-categories.md](task-categories.md)
flags as its clearest fault is largely what SapBERT repairs — it is the biggest single merge
below. That is a point in SapBERT's favour and it is not proof: the same pressure that fixes
fragmentation is the pressure that produced the 421-task blob under the old pipeline, and
SapBERT's largest category is already bigger than MiniLM's.

The honest position: these are two defensible answers, 27% apart, and the 304 labelled pairs
of task-clustering-method.md §5.1 decide between them. Until then, prefer MiniLM if a wrong
merge costs more than a wrong split — which for a meta-analysis it usually does, since a
split can be repaired by pooling two categories and a merge cannot be undone without
re-reading the papers.

> The per-task table is keyed on (study, task name), which collapses 52 rows where one paper
> was extracted in more than one run. Both copies carry an identical name, so rule-1 must-link
> puts them in the same category by construction and nothing is lost by collapsing them.

