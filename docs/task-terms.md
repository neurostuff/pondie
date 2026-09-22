# The task terms this corpus actually uses

> The script that produced this was removed when the workflow was consolidated into
> `pondie/normalization/task.py`. The measurements below stand; the code does not exist.

Measured over the 2,115 records on `beast-proxy`:
**1,696 task mentions, 1,120 distinct names.** The scheme and the evidence for it are in
[normalization-layer.md](normalization-layer.md#the-scheme-for-tasks-and-conditions); this
file is the list itself, so a term can be argued with.

A task name is a **paradigm** (the response logic) plus a **content** (what the stimuli were
about), and sometimes names a **process**. Each facet takes the longest span of the name its
vocabulary can claim at `exact`, `synonym` or `stem` -- never `contains`, which is the layer
that produces the layer's wrong answers. The support column counts **distinct studies**.

Nothing here is normalized *into* a record. These are assertions about records, and the
paradigm column is what a meta-analysis query should be written against instead of the name.


## Coverage, and the honest part

| facet | terms | mentions filled |
|---|---|---|
| paradigm | 73 | 493 / 1696 (29%) |
| process | 95 | 333 / 1696 (19%) |
| content | 3 | 87 / 1696 (5%) |
| **unnamed** | 647 residues | 1164 / 1696 |

73 paradigm terms against 1,120 name forms, and 52% of them are shared between two or more
studies where only 15% of the raw names are. That is the result. But **a vocabulary reaches
under a third of mentions**, and the unnamed table at the bottom is the rest -- headed by
`resting state` and `cue reactivity`, which no vocabulary in this stack contains.

Of the 493 paradigm mentions, 411 came from an `exact` label match, 47 from a `synonym` and
**35 from `stem`**, which is where the errors are. Four are visibly wrong and are marked (!)
below: `art emotion test` is not the Angling Risk Task and `Free and Cued Selective Reminding
Test` is not the Wason card selection task. Ten mentions in 1,696 -- small, and the reason the
stem layer is worth keeping separate rather than folding into a score.


## Paradigms -- 73 terms

From the Cognitive Atlas task list. ONVOC has no paradigm terms at all.

| studies | term | names it folded |
|---|---|---|
| 147 | `Emotion Regulation Task` | emotion regulation task; emotion regulation; Emotion Regulation Task; +42 more |
| 73 | `reappraisal task` | cognitive reappraisal task; reappraisal task; emotion reappraisal task; +30 more |
| 34 | `go/no-go task` | go/no-go task; emotional go/no-go task; Go-NoGo task; +27 more |
| 24 | `monetary incentive delay task` | monetary incentive delay task; Monetary Incentive Delay task; Monetary Incentive Delay Task; +9 more |
| 22 | `Stroop task` | drug Stroop fMRI task; emotional Stroop task; cocaine-word Stroop task; +14 more |
| 14 | `n-back task` | N-back working memory task; n-back task; emotional faces n-back task; +8 more |
| 13 | `mental imagery task` | personalized guided-imagery task; script-guided mental imagery; guided imagery and recall of stress and neutral situations; +9 more |
| 10 | `Emotion Recognition Task` | emotion recognition task; smoking cues emotion recognition task (SCERT); emotion recognition; +5 more |
| 9 | `stop signal task` | stop signal task; Stop-Signal Alcohol-Cue Task; modified stop-signal task; +3 more |
| 6 | `face matching task` | emotional face-matching task; extended Hariri face matching task; emotional face matching task; +2 more |
| 5 | `Naming tasks` | Confrontation naming; environmental sound naming; Category-naming fluency; +2 more |
| 5 | `oddball task` | visual oddball task; two-choice oddball task; forced-choice visual oddball task; +2 more |
| 5 | `Shift Task` | Shifted-Attention Emotion Appraisal Task; Shifted-Attention Emotion Appraisal Task (SEAT); shifted-attention emotion appraisal task; +1 more |
| 5 | `temporal discounting task` | delay discounting task; episodic intertemporal choice task; delay discounting; +1 more |
| 4 | `color-word stroop task` | color-word Stroop task; Chinese version color-word Stroop task; classical color-word Stroop task |
| 4 | `cyberball task` | Cyberball social rejection task; Cyberball; Cyberball social rejection paradigm |
| 4 | `Iowa Gambling Task` | Iowa Gambling Task; modified Iowa Gambling Task |
| 3 | `antisaccade/prosaccade task` | rewarded antisaccade task; fMRI reward cue antisaccade (AS) task; Antisaccade task |
| 3 | `Continuous Performance Task` | continuous performance task; continuous performance task with emotional and neutral distractors; Continuous performance task |
| 3 | `Emotion Identification Task` | emotion identification; alternating emotion-identification/digit-sorting task; emotion identification task |
| 3 | `encoding task` | encoding and immediate JOLs task; face encoding; Encoding of Affective Pictures; +2 more |
| 3 | `Eriksen flanker task` | flanker task; Flanker task; affective variant of the flanker task |
| 3 | `pavlovian conditioning task` | higher-order Pavlovian conditioning paradigm; aversive classical conditioning with cognitive emotion regulation; aversive Pavlovian conditioning |
| 3 | `recall test` | working memory recall task; probed recall task; Familiar melody title recall |
| 3 | `verbal working memory task` | parametric verbal working memory task; Sentence comprehension and verbal working memory; verbal working memory n-back task |
| 2 | `Angling Risk Task` **(!)** | art emotion test; passive art-viewing sponsorship task |
| 2 | `autobiographical memory task` | negative autobiographical memory strategy task; positive autobiographical memory neurofeedback |
| 2 | `backward masking` | fMRI backward-masked cue task; backward-masked cannabis cue task |
| 2 | `categorization task` | Experiment 3: familiarity categorization and famous face-name matching; categorization task |
| 2 | `emotional regulation task` | cognitive-linguistic regulation of emotional reactivity; emotional regulation task |
| 2 | `face n-back task` | emotional face n-back (EFNBACK); emotional face N-back (EFNBACK) task |
| 2 | `letter n-back task` | letter n-back task; letter n-back task with olfactory emotion induction |
| 2 | `orientation test` **(!)** | behavioral orientation affordance task; line orientation and picture categorization task |
| 2 | `rest eyes closed` | eyes-closed resting state; eyes-closed resting-state scan |
| 2 | `rest eyes open` | eyes-open resting state |
| 2 | `verbal fluency task` | semantic verbal fluency task; phonemic verbal fluency task; verbal fluency testing |
| 2 | `video games` | Internet video game cue; violent and non-violent Carmageddon video game |
| 2 | `word-picture matching task` | word-picture matching task; Word-picture matching task |
| 1 | `acupuncture task` | heroin-visual stimulation and acupuncture |
| 1 | `associative memory encoding task` | associative memory encoding |
| 1 | `attention switching task` | switching task |
| 1 | `balloon analogue risk task` | Balloon analogue risk task (BART) |
| 1 | `Cambridge Face Memory Test` | Cambridge Face Memory Test |
| 1 | `Cambridge Gambling Task` | Cambridge Gambling Task |
| 1 | `Dimensions task` | Dimensional Card Sorting Task |
| 1 | `directed forgetting task` | directed forgetting task |
| 1 | `electric stimulation` **(!)** | painful electrical stimulation with religious and non-religious images |
| 1 | `finger tapping task` | finger-tapping task |
| 1 | `fixation task` | Resting-state fixation |
| 1 | `gambling task` | gambling task |
| 1 | `gender discrimination task` | gender discrimination task |
| 1 | `Information Sampling Task` | Information Sampling Task (IST) |
| 1 | `meditation task` | mindfulness meditation |
| 1 | `Moral Dilemma Task` | moral dilemmas |
| 1 | `movie watching task` | watching movies |
| 1 | `pitch/monitor discrimination` | Pitch discrimination |
| 1 | `Probabilistic classification task` | Weather Prediction Task |
| 1 | `probabilistic reversal learning task` | Probabilistic reversal learning task |
| 1 | `prospective memory task` | Modified Cambridge Prospective Memory test |
| 1 | `recognition memory test` | recognition memory task |
| 1 | `same-different task` | same/different perceptual discrimination task |
| 1 | `semantic classification task` | auditory semantic classification task |
| 1 | `semantic task` | semantic congruity task; auditory semantic control task |
| 1 | `set-shifting task` | Intra-Extra Dimensional set shifting task |
| 1 | `source memory test` | self-reference source memory task |
| 1 | `spatial working memory task` | spatial working memory |
| 1 | `Sternberg delayed recognition task` | modified Sternberg working memory paradigm; Sternberg WM task |
| 1 | `target detection task` | target detection task |
| 1 | `visual pursuit/tracking` | threshold-tracking transcranial magnetic stimulation |
| 1 | `visual search task` | visual search task |
| 1 | `visually guided saccade task` | Visually guided saccade task |
| 1 | `Wason card selection task` **(!)** | Free and Cued Selective Reminding Test |
| 1 | `Wisconsin card sorting test` | Wisconsin Card-Sorting Test (WCST) |

## Processes -- 95 terms

From ONVOC `Psychological Concepts` and the Cognitive Atlas concept list. This is the branch [normalizing-with-onvoc.md](normalizing-with-onvoc.md) concluded ONVOC did not have.

| studies | term | names it folded |
|---|---|---|
| 59 | `Emotion` | emotion reappraisal task; emotion reappraisal; emotion processing task; +47 more |
| 18 | `reward processing` | monetary reward paradigm; reward and personal reference task; reward conditioning paradigm with cognitive reappraisal; +15 more |
| 13 | `Working Memory` | modified Sternberg working memory paradigm; N-back working memory task; working memory recall task; +10 more |
| 10 | `Food cue reactivity` | food cue-reactivity task; Cocaine and food cue-reactivity task; food-cue reactivity paradigm; +6 more |
| 9 | `Attention` | Shifted-Attention Emotion Appraisal Task; food picture attention task; Interoceptive Attention task; +5 more |
| 9 | `fear` | fear conditioning and extinction paradigm; emotional fear processing task; fear conditioning; +6 more |
| 8 | `anticipation` | cue-elicited anticipation and receipt of palatable food; Alcohol Prediction Error (APE) Task; anticipation of emotional stimuli; +5 more |
| 7 | `distraction` | distraction task; rejecter and neutral photograph viewing with countback distraction; food distraction; +3 more |
| 6 | `Perception` | food perception task; picture perception task; picture-perception task; +3 more |
| 6 | `stress` | Montreal Imaging Stress Task; neutral-relaxing, alcohol, and stress cue exposure; stress, alcohol cue, and neutral visual cue task; +3 more |
| 5 | `attentional bias` | attentional bias line counting task; attentional bias paradigm; attentional bias line-counting task; +2 more |
| 5 | `audition` | auditory food cues; auditory semantic classification task; Auditory perceptual tasks; +2 more |
| 5 | `discrimination` | same/different perceptual discrimination task; food/non-food discrimination task; sad facial affect discrimination task; +2 more |
| 5 | `Empathy` | empathy attribution task; Story-based Empathy Task; empathy for pain task; +2 more |
| 5 | `interference` | Emotional Faces Interference Task; emotional-interference task; emotional interference conflict task; +1 more |
| 5 | `Memory` | Cambridge Car Memory Test; Memory and Temporal Experience Questionnaire; Emotional Faces Memory Task; +2 more |
| 5 | `mood` | mood induction; positive mood induction; sad mood elaboration; +1 more |
| 5 | `response inhibition` | Go-NoGo response inhibition; go no-go response inhibition task; response inhibition; +2 more |
| 5 | `visual masking` | backward-masking task; backward-masking paradigm; backward masking paradigm; +2 more |
| 4 | `feedback processing` | feedback-based learning task; probabilistic feedback expectancy task; feedback task; +1 more |
| 4 | `inhibition` | attempted inhibition of cue-induced craving; food inhibition go/no-go task; cognitive–emotional inhibition task; +1 more |
| 4 | `judgment` | script event order judgment; moral judgment task; emotional judgment task; +1 more |
| 4 | `recognition` | Recognition task; incidental recognition task; incidental emotional recognition task; +1 more |
| 3 | `anxiety` | anticipatory anxiety paradigm; autobiographical social anxiety scripts; anticipatory anxiety |
| 3 | `appetite` | Food image viewing and hunger rating task; appetite-provoking fMRI task; cognitive appetite control |
| 3 | `decision` | food decision task; real smoking decision task; humour decision task |
| 3 | `Decision Making` | decision-making under risk; emotion regulation and risky decision-making task; Decision-Making Task |
| 3 | `desire` | videotapes designed to elicit happy feelings, sad feelings, or the desire to use cocaine; desire for palatable food regulation paradigm; volitional re |
| 3 | `expectancy` | tonal expectancy task; emotional expectation paradigm; emotional expectation task |
| 3 | `facial expression` | facial expression processing; implicit facial expression task |
| 3 | `internalizing` | worry modulation task; worry induction and worry reappraisal; worry induction and reappraisal task |
| 3 | `irritability` | adapted MRI version of the Anger Articulated Thoughts during Simulated Situations (ATSS) paradigm; anger-provoking movie viewing; anger-infused Ultima |
| 3 | `negative emotion` | cognitive reappraisal of negative emotion; Voluntary regulation of negative emotion; negative emotion regulation |
| 3 | `pain` | pain control task; pain regulation task; placebo analgesia pain task |
| 2 | `cognitive control` | cognitive control strategy food-picture task; cognitive control |
| 2 | `coordination` | behavioral coordination task; social coordination task |
| 2 | `eating` | food/non-food preference task; food preference decision-making task |
| 2 | `Episodic Memory` | Episodic memory testing |
| 2 | `extinction` | conditioning and extinction task; extinction learning task |
| 2 | `face perception` | Face-Perception task; emotional face perception task |
| 2 | `face recognition` | Face recognition; face recognition |
| 2 | `frustration` | frustration-induction Go-NoGo task; Frustration Emotion Task for Children (FETCH) |
| 2 | `induction` | alcohol cue-induction paradigm; alcohol and beverage cue-induction paradigm; smoking and neutral image cue-induction task; +1 more |
| 2 | `priming` | Priming emotion and alcohol Stroop Match-to-Sample task; automatic emotion regulation priming paradigm |
| 2 | `Social Cognition` | TASIT social cognition assessment; social cognition tasks |
| 2 | `social inference` | Social Inference — Minimal subtest of The Awareness of Social Inference Test; Emotion Evaluation subtest of The Awareness of Social Inference Test |
| 2 | `thought` | mind wandering thought-sampling task; thought suppression |
| 2 | `valence` | valence decision task; valence bias task |
| 1 | `activation` | main activation task |
| 1 | `addiction` | addiction-Stroop color match-to-sample task |
| 1 | `affect recognition` | Implicit Sad Facial Affect Recognition Task |
| 1 | `agreeableness` | two-player cooperative games |
| 1 | `attachment` | social contact, attachment, and social regulation of emotion |
| 1 | `auditory scene` | auditory scene control task |
| 1 | `belief` | Negative self-belief emotion regulation task |
| 1 | `categorization` | line orientation and picture categorization task |
| 1 | `concept` | social concept discrimination task |
| 1 | `Coping` | Cue exposure/coping task |
| 1 | `cueing` | spatial cueing paradigm |
| 1 | `Declarative Memory` | declarative memory task |
| 1 | `detection` | Experiment 1: face shape detection |
| 1 | `emotion perception` | classic face/emotion perception task |
| 1 | `emotion regulation` | aversive classical conditioning with cognitive emotion regulation |
| 1 | `emotional expression` | intentional emotional expression task |
| 1 | `emotional memory` | emotional memory task |
| 1 | `error detection` | Familiar melody pitch error detection |
| 1 | `Executive Function` | executive function tests |
| 1 | `familiarity` | Experiment 3: familiarity categorization and famous face-name matching |
| 1 | `feature comparison` | feature-based comparison task |
| 1 | `gaze` | gaze-directed cognitive reappraisal |
| 1 | `Learning` | encoding and immediate judgments of learning task |
| 1 | `listening` | music listening reward responsiveness paradigm |
| 1 | `maintenance` | emotional maintenance task |
| 1 | `melody` | Unfamiliar melody discrimination; Familiar melody title recall |
| 1 | `Metacognition` | Perceptual metacognition task |
| 1 | `Mindfulness` | mindfulness meditation |
| 1 | `monitoring` | visual monitoring task |
| 1 | `Moral Reasoning` | moral reasoning task |
| 1 | `movement` | Smooth pursuit eye movement task |
| 1 | `narrative` | Frog, Where Are You narrative |
| 1 | `perceptual similarity` | perceptual similarity control task |
| 1 | `punishment processing` | Go/No-Go task with visual negative feedback |
| 1 | `reading` | Reading the Mind in the Eyes Test |
| 1 | `recall` | guided imagery and recall of stress and neutral situations |
| 1 | `reinstatement` | reinstatement test |
| 1 | `repetition priming` | repetition priming paradigm |
| 1 | `risk` | Balloon Analog Risk Task |
| 1 | `Risk Taking` | sequential risk-taking task |
| 1 | `sadness` | cognitive reappraisal of sadness |
| 1 | `schema` | baby schema task |
| 1 | `sentence comprehension` | Sentence comprehension and verbal working memory |
| 1 | `sleep` | sleep–wake cycle; sleep-wake recording; overnight sleep-wake EEG-fMRI |
| 1 | `strategy` | negative autobiographical memory strategy task |
| 1 | `trust` | Trust game memory task |
| 1 | `worldview` | perspective-taking task |

## Contents -- 3 terms

From ONVOC's drug and `Behaviors` branches -- and this is the whole list, which is the point. `food` is in 124 papers' task names, `alcohol` in 68; ONVOC has neither.

| studies | term | names it folded |
|---|---|---|
| 61 | `Smoking` | smoking cue exposure; smoking cue-reactivity task; smoking-related or neutral images; +54 more |
| 11 | `Cannabis` | cannabis cue-exposure task; cannabis cue-exposure fMRI task; cannabis cue–reactivity task; +7 more |
| 9 | `Heroin` | visual heroin-related stimuli; heroin-related visual stimuli; event-related heroin-related and neutral cue task; +6 more |

## Unnamed -- 647 residues, no vocabulary claims a paradigm

The corpus-clustering job, and the proposal list for the Cognitive Atlas. The head of
this table is the head of the whole corpus: `resting state` in 144 studies and `cue
reactivity` in 50 are two of the three most common paradigms in the literature being
extracted, and neither the Cognitive Atlas nor ONVOC has a term for either.

Shown to support 2 or more studies; the tail is 1-study residues and is in the TSV.

| studies | residue |
|---|---|
| 144 | `resting state` |
| 50 | `cue reactivity` |
| 26 | `cue exposure` |
| 18 | `alcohol cue reactivity` |
| 15 | `visual food cues` |
| 11 | `emotional` |
| 10 | `food cue` |
| 9 | `emotional faces` |
| 8 | `processing` |
| 7 | `cue` |
| 7 | `emotional conflict` |
| 7 | `resting state magnetic resonance` |
| 6 | `food cue viewing` |
| 5 | `affect labeling` |
| 5 | `backward` |
| 5 | `food cues` |
| 5 | `picture viewing` |
| 5 | `resting state connectivity` |
| 5 | `visual food cue` |
| 4 | `alcohol taste cue` |
| 4 | `conditioning` |
| 4 | `food` |
| 4 | `induction` |
| 4 | `neurofeedback` |
| 4 | `ultimatum game` |
| 4 | `visual cue` |
| 4 | `visual stimuli` |
| 3 | `control` |
| 3 | `cookie theft picture description` |
| 3 | `cue viewing` |
| 3 | `drug word` |
| 3 | `emotional face assessment` |
| 3 | `emotional processing` |
| 3 | `emotional reactivity` |
| 3 | `event cue reactivity` |
| 3 | `faces` |
| 3 | `food motivation` |
| 3 | `free resting state` |
| 3 | `implicit processing` |
| 3 | `incentive delay` |
| 3 | `localizer` |
| 3 | `monetary` |
| 3 | `or neutral images` |
| 3 | `passive viewing` |
| 3 | `picture` |
| 3 | `regulation` |
| 3 | `resting pet` |
| 3 | `visual alcohol cue reactivity` |
| 3 | `visual cue reactivity` |
| 2 | `active regulation` |
| 2 | `advertisement viewing` |
| 2 | `affective picture` |
| 2 | `alcohol` |
| 2 | `alcohol associated control stimuli` |
| 2 | `alcohol cue` |
| 2 | `alcohol pictures` |
| 2 | `alcohol taste cues` |
| 2 | `anticipatory` |
| 2 | `appraisal` |
| 2 | `approach avoidance` |
| 2 | `beer incentive delay` |
| 2 | `beliefs` |
| 2 | `cigarette cue reactivity` |
| 2 | `cigarette cues` |
| 2 | `cocaine cue` |
| 2 | `cocaine cue exposure` |
| 2 | `cocaine cue reactivity` |
| 2 | `cognitive re appraisal` |
| 2 | `conditioning extinction` |
| 2 | `craving regulation` |
| 2 | `criticism` |
| 2 | `cue elicited craving` |
| 2 | `cue induced alcohol craving` |
| 2 | `cue induced cocaine craving` |
| 2 | `cue induced craving` |
| 2 | `drug cue reactivity` |
| 2 | `dynamic faces` |
| 2 | `emotional face processing` |
| 2 | `emotional stimuli` |
| 2 | `evaluation` |
| 2 | `event cue exposure` |
| 2 | `facial processing` |
| 2 | `facial viewing` |
| 2 | `fearful face` |
| 2 | `food craving` |
| 2 | `food craving regulation` |
| 2 | `food images` |
| 2 | `food nonfood picture` |
| 2 | `food picture viewing` |
| 2 | `free` |
| 2 | `generation regulation` |
| 2 | `kidvid` |
| 2 | `learning` |
| 2 | `line counting` |
| 2 | `methamphetamine cues` |
| 2 | `montreal` |
| 2 | `neutral images` |
| 2 | `pavlovian instrumental transfer` |
| 2 | `perceptual` |
| 2 | `probabilistic` |
| 2 | `regulation craving` |
| 2 | `resting fdg pet` |
| 2 | `resting state pet` |
| 2 | `rewarded guessing` |
| 2 | `rt probe` |
| 2 | `social` |
| 2 | `social evaluation` |
| 2 | `taste cue` |
| 2 | `taylor aggression` |
| 2 | `temporal difference error juice` |
| 2 | `testing` |
| 2 | `transfer` |
| 2 | `virtual supermarket` |
| 2 | `visual cue exposure` |
| 2 | `visual food cue viewing` |
| 2 | `wheel fortune` |

---

The script that produced these tables was removed when the workflow was consolidated, and the
per-task facet file with it. The successor is `data/task-facets/task-categories.tsv` from
`python -m pondie.normalization.task`, which carries a paradigm and a stimulus for every task —
see [task-categories.md](task-categories.md). The term counts above are the finding and are
not regenerated by it: the paradigm column there comes from the normalised Cognitive Atlas
seed list ([cognitive-atlas-seeds.md](cognitive-atlas-seeds.md)) rather than from the
three-facet decomposition this file measured.
