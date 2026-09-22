# Task categories and their members

Every one of the 1672 tasks the clusterer sees, the category it landed in, and the
stimulus it used. Produced by

```
python -m pondie.normalization.task --cut 0.6 --encoder minilm --out data/task-facets
```

**184 categories holding 1530 tasks; 142 tasks in no category.**
61 categories are named by the Cognitive Atlas (473 tasks matched against
639 normalised seeds — see [cognitive-atlas-seeds.md](cognitive-atlas-seeds.md));
the other 123 were clustered and are named by their most frequent member.

`data/task-facets/task-categories.tsv` is the same content one row per task.

## How to read it

Each category carries a **stimulus breakdown**, because a task name is a paradigm crossed
with a stimulus and the two are separate query axes. One paradigm, several substances, is the
normal case and the reason the stimulus is a column rather than part of the category:

| category | stimuli |
|---|---|

| resting state | (unspecified) (192), stress (1) |
| Emotion Regulation Task | emotion (151), pain (1), social (1) |
| reappraisal task | emotion (61), food (6), (unspecified) (6), faces (2), money (1) |
| alcohol cue reactivity task | alcohol (34), tobacco (16), cocaine (5), opioid (4), food (2) |
| cue-reactivity task | tobacco (24), food (11), alcohol (8), cannabis (6), cocaine (1) |
| cue reactivity task | alcohol (19), tobacco (10), (unspecified) (6), emotion (4), cocaine (2) |

**Filter on the member's stimulus, not the category name.** A clustered category is named
after one member, so `alcohol cue reactivity task` holds tobacco and cocaine members too. The
paradigm is right; the borrowed name is not a claim about content.

**Known problem: fragmentation.** Cue reactivity is split several ways and faces two ways,
because the second (`family`) tier that would group them is not fitted. Nothing here
over-merges the way the previous pipeline did — `go/no-go`, `n-back`, `Stroop` and `monetary
incentive delay` are separate categories — but the error now runs the other way.

**Unvalidated.** Better on inspection than what it replaced, and there is still no gold; see
task-clustering-method.md §5.

## Index

| # | source | studies | tasks | names | category |
|---|---|---|---|---|---|
| 1 | clustered | 174 | 193 | 45 | [resting state](#1-resting-state) |
| 2 | atlas | 148 | 153 | 47 | [Emotion Regulation Task](#2-emotion-regulation-task) |
| 3 | atlas | 74 | 78 | 34 | [reappraisal task](#3-reappraisal-task) |
| 4 | clustered | 59 | 64 | 46 | [alcohol cue reactivity task](#4-alcohol-cue-reactivity-task) |
| 5 | clustered | 52 | 53 | 42 | [cue-reactivity task](#5-cue-reactivity-task) |
| 6 | clustered | 44 | 49 | 39 | [cue reactivity task](#6-cue-reactivity-task) |
| 7 | clustered | 36 | 44 | 31 | [Taste Cue Paradigm](#7-taste-cue-paradigm) |
| 8 | clustered | 36 | 42 | 38 | [modified Sternberg working memory paradigm](#8-modified-sternberg-working-memory-paradigm) |
| 9 | clustered | 35 | 42 | 21 | [visual food cues](#9-visual-food-cues) |
| 10 | atlas | 34 | 35 | 30 | [go/no-go task](#10-gono-go-task) |
| 11 | atlas | 30 | 31 | 17 | [monetary incentive delay task](#11-monetary-incentive-delay-task) |
| 12 | clustered | 30 | 33 | 31 | [cue-exposure task](#12-cue-exposure-task) |
| 13 | atlas | 26 | 29 | 20 | [Stroop task](#13-stroop-task) |
| 14 | clustered | 24 | 25 | 20 | [emotional faces task](#14-emotional-faces-task) |
| 15 | clustered | 22 | 24 | 19 | [facial expression processing](#15-facial-expression-processing) |
| 16 | atlas | 17 | 17 | 14 | [n-back task](#16-n-back-task) |
| 17 | atlas | 16 | 17 | 16 | [Facial Emotion Paradigm](#17-facial-emotion-paradigm) |
| 18 | clustered | 16 | 17 | 15 | [food images](#18-food-images) |
| 19 | clustered | 16 | 16 | 15 | [event-related cue-reactivity task](#19-event-related-cue-reactivity-task) |
| 20 | clustered | 15 | 18 | 15 | [emotion generation and regulation task](#20-emotion-generation-and-regulation-task) |
| 21 | clustered | 13 | 13 | 11 | [RT probe task](#21-rt-probe-task) |
| 22 | clustered | 12 | 15 | 12 | [food craving task](#22-food-craving-task) |
| 23 | clustered | 12 | 12 | 6 | [food cue task](#23-food-cue-task) |
| 24 | clustered | 12 | 12 | 11 | [Social Evaluation Task](#24-social-evaluation-task) |
| 25 | clustered | 11 | 11 | 10 | [personalized guided-imagery task](#25-personalized-guided-imagery-task) |
| 26 | clustered | 11 | 11 | 9 | [affect labeling](#26-affect-labeling) |
| 27 | clustered | 10 | 10 | 9 | [Pavlovian-instrumental transfer task](#27-pavlovian-instrumental-transfer-task) |
| 28 | atlas | 9 | 9 | 7 | [Emotion Recognition Task](#28-emotion-recognition-task) |
| 29 | clustered | 9 | 9 | 9 | [food visualization](#29-food-visualization) |
| 30 | atlas | 8 | 8 | 6 | [stop signal task](#30-stop-signal-task) |
| 31 | clustered | 8 | 11 | 10 | [viewing photographs of high and low-calorie foods](#31-viewing-photographs-of-high-and-low-calorie-foods) |
| 32 | clustered | 8 | 8 | 8 | [drug word fMRI task](#32-drug-word-fmri-task) |
| 33 | clustered | 8 | 11 | 7 | [emotional conflict task](#33-emotional-conflict-task) |
| 34 | clustered | 8 | 8 | 7 | [distraction task](#34-distraction-task) |
| 35 | atlas | 7 | 7 | 6 | [face matching task](#35-face-matching-task) |
| 36 | clustered | 7 | 8 | 8 | [viewing images of food and nonfood objects](#36-viewing-images-of-food-and-nonfood-objects) |
| 37 | clustered | 7 | 7 | 6 | [smoking cue exposure](#37-smoking-cue-exposure) |
| 38 | clustered | 7 | 7 | 7 | [food cue visual stimulation](#38-food-cue-visual-stimulation) |
| 39 | clustered | 7 | 7 | 6 | [approach-avoidance task](#39-approach-avoidance-task) |
| 40 | clustered | 7 | 7 | 6 | [cognitive re-appraisal task](#40-cognitive-re-appraisal-task) |
| 41 | clustered | 7 | 7 | 5 | [Ultimatum Game](#41-ultimatum-game) |
| 42 | clustered | 6 | 7 | 5 | [food picture viewing](#42-food-picture-viewing) |
| 43 | clustered | 6 | 6 | 6 | [appetite-provoking fMRI task](#43-appetite-provoking-fmri-task) |
| 44 | atlas | 5 | 5 | 5 | [backward masking](#44-backward-masking) |
| 45 | atlas | 5 | 5 | 4 | [passive viewing](#45-passive-viewing) |
| 46 | atlas | 5 | 5 | 5 | [oddball task](#46-oddball-task) |
| 47 | clustered | 5 | 5 | 5 | [picture cue viewing task](#47-picture-cue-viewing-task) |
| 48 | atlas | 4 | 4 | 4 | [film viewing](#48-film-viewing) |
| 49 | atlas | 4 | 7 | 3 | [cyberball task](#49-cyberball-task) |
| 50 | atlas | 4 | 4 | 2 | [Iowa Gambling Task](#50-iowa-gambling-task) |
| 51 | clustered | 4 | 4 | 3 | [cue-viewing task](#51-cue-viewing-task) |
| 52 | clustered | 4 | 4 | 4 | [smoking and nonsmoking cue stimuli](#52-smoking-and-nonsmoking-cue-stimuli) |
| 53 | clustered | 4 | 4 | 4 | [reward and personal reference task](#53-reward-and-personal-reference-task) |
| 54 | clustered | 4 | 4 | 4 | [food perception task](#54-food-perception-task) |
| 55 | clustered | 4 | 5 | 4 | [cognitive control strategy food-picture task](#55-cognitive-control-strategy-food-picture-task) |
| 56 | clustered | 4 | 4 | 1 | [food cue viewing](#56-food-cue-viewing) |
| 57 | clustered | 4 | 4 | 4 | [food and nonfood picture paradigm](#57-food-and-nonfood-picture-paradigm) |
| 58 | clustered | 4 | 4 | 3 | [food motivation paradigm](#58-food-motivation-paradigm) |
| 59 | clustered | 4 | 4 | 2 | [emotional faces](#59-emotional-faces) |
| 60 | clustered | 4 | 4 | 4 | [food choice task](#60-food-choice-task) |
| 61 | clustered | 4 | 4 | 3 | [emotion processing](#61-emotion-processing) |
| 62 | clustered | 4 | 4 | 3 | [Wheel of Fortune task](#62-wheel-of-fortune-task) |
| 63 | clustered | 4 | 4 | 3 | [mood induction](#63-mood-induction) |
| 64 | clustered | 4 | 4 | 4 | [partner and opposite-sex stranger facial expressions](#64-partner-and-opposite-sex-stranger-facial-expressions) |
| 65 | clustered | 4 | 4 | 4 | [rtfMRI-nf training](#65-rtfmri-nf-training) |
| 66 | atlas | 3 | 3 | 3 | [verbal working memory task](#66-verbal-working-memory-task) |
| 67 | atlas | 3 | 3 | 3 | [Continuous Performance Task](#67-continuous-performance-task) |
| 68 | clustered | 3 | 3 | 3 | [smoking and control videos](#68-smoking-and-control-videos) |
| 69 | clustered | 3 | 3 | 3 | [heroin-related visual stimuli](#69-heroin-related-visual-stimuli) |
| 70 | clustered | 3 | 3 | 3 | [attentional bias line counting task](#70-attentional-bias-line-counting-task) |
| 71 | clustered | 3 | 4 | 4 | [visual cue paradigm](#71-visual-cue-paradigm) |
| 72 | clustered | 3 | 3 | 3 | [response inhibition](#72-response-inhibition) |
| 73 | clustered | 3 | 3 | 3 | [rewarded antisaccade task](#73-rewarded-antisaccade-task) |
| 74 | clustered | 3 | 3 | 3 | [taste task](#74-taste-task) |
| 75 | clustered | 3 | 3 | 3 | [rtfMRI neurofeedback task](#75-rtfmri-neurofeedback-task) |
| 76 | clustered | 3 | 4 | 4 | [Experiment 1: face shape detection](#76-experiment-1-face-shape-detection) |
| 77 | clustered | 3 | 3 | 2 | [Cookie Theft picture description](#77-cookie-theft-picture-description) |
| 78 | clustered | 3 | 6 | 5 | [encoding and immediate JOLs task](#78-encoding-and-immediate-jols-task) |
| 79 | clustered | 3 | 3 | 3 | [anticipatory anxiety paradigm](#79-anticipatory-anxiety-paradigm) |
| 80 | clustered | 3 | 3 | 3 | [picture stimuli and thermal stimuli](#80-picture-stimuli-and-thermal-stimuli) |
| 81 | clustered | 3 | 3 | 3 | [aversive and neutral anticipation](#81-aversive-and-neutral-anticipation) |
| 82 | clustered | 3 | 3 | 2 | [Taylor Aggression Paradigm](#82-taylor-aggression-paradigm) |
| 83 | clustered | 3 | 3 | 3 | [risky monetary choices](#83-risky-monetary-choices) |
| 84 | clustered | 3 | 3 | 3 | [affective Posner task](#84-affective-posner-task) |
| 85 | clustered | 3 | 3 | 2 | [Shifted-Attention Emotion Appraisal Task](#85-shifted-attention-emotion-appraisal-task) |
| 86 | clustered | 3 | 3 | 3 | [Amygdala neurofeedback regulation task](#86-amygdala-neurofeedback-regulation-task) |
| 87 | clustered | 3 | 3 | 2 | [emotional face assessment task](#87-emotional-face-assessment-task) |
| 88 | atlas | 2 | 2 | 2 | [mental imagery task](#88-mental-imagery-task) |
| 89 | atlas | 2 | 2 | 2 | [pavlovian conditioning task](#89-pavlovian-conditioning-task) |
| 90 | atlas | 2 | 3 | 3 | [verbal fluency task](#90-verbal-fluency-task) |
| 91 | atlas | 2 | 2 | 2 | [letter naming task](#91-letter-naming-task) |
| 92 | atlas | 2 | 2 | 2 | [word-picture matching task](#92-word-picture-matching-task) |
| 93 | atlas | 2 | 2 | 2 | [autobiographical memory task](#93-autobiographical-memory-task) |
| 94 | clustered | 2 | 2 | 2 | [alcohol pictures task](#94-alcohol-pictures-task) |
| 95 | clustered | 2 | 2 | 2 | [water-related, drug-related, and neutral cues](#95-water-related-drug-related-and-neutral-cues) |
| 96 | clustered | 2 | 2 | 2 | [affective stimuli](#96-affective-stimuli) |
| 97 | clustered | 2 | 2 | 2 | [food picture attention task](#97-food-picture-attention-task) |
| 98 | clustered | 2 | 2 | 2 | [Montreal Imaging Stress Task](#98-montreal-imaging-stress-task) |
| 99 | clustered | 2 | 2 | 2 | [watching movies](#99-watching-movies) |
| 100 | clustered | 2 | 2 | 2 | [attentional bias paradigm](#100-attentional-bias-paradigm) |
| 101 | clustered | 2 | 2 | 2 | [Temporal Difference Error/Juice Paradigm](#101-temporal-difference-errorjuice-paradigm) |
| 102 | clustered | 2 | 2 | 2 | [individualized script imagery](#102-individualized-script-imagery) |
| 103 | clustered | 2 | 2 | 2 | [picture perception task](#103-picture-perception-task) |
| 104 | clustered | 2 | 2 | 2 | [cigarette cue-reactivity and self-expansion task](#104-cigarette-cue-reactivity-and-self-expansion-task) |
| 105 | clustered | 2 | 2 | 2 | [resting-state protocol](#105-resting-state-protocol) |
| 106 | clustered | 2 | 2 | 2 | [neurofeedback](#106-neurofeedback) |
| 107 | clustered | 2 | 3 | 2 | [visual food cue viewing](#107-visual-food-cue-viewing) |
| 108 | clustered | 2 | 2 | 2 | [Smoking Pleasantness task](#108-smoking-pleasantness-task) |
| 109 | clustered | 2 | 2 | 2 | [flavor paradigm](#109-flavor-paradigm) |
| 110 | clustered | 2 | 2 | 2 | [food/non-food discrimination task](#110-foodnon-food-discrimination-task) |
| 111 | clustered | 2 | 2 | 2 | [infant face images](#111-infant-face-images) |
| 112 | clustered | 2 | 2 | 2 | [Probabilistic Reward Task](#112-probabilistic-reward-task) |
| 113 | clustered | 2 | 2 | 1 | [advertisement viewing](#113-advertisement-viewing) |
| 114 | clustered | 2 | 2 | 2 | [delay discounting task](#114-delay-discounting-task) |
| 115 | clustered | 2 | 2 | 2 | [Cue reactivity task](#115-cue-reactivity-task) |
| 116 | clustered | 2 | 2 | 2 | [social concept discrimination task](#116-social-concept-discrimination-task) |
| 117 | clustered | 2 | 2 | 2 | [Social Inference — Minimal subtest of The Awareness ](#117-social-inference--minimal-subtest-of-the-awareness-of-social-inference-test) |
| 118 | clustered | 2 | 4 | 4 | [Pitch discrimination](#118-pitch-discrimination) |
| 119 | clustered | 2 | 2 | 2 | [saccade tasks](#119-saccade-tasks) |
| 120 | clustered | 2 | 2 | 2 | [moral reasoning task](#120-moral-reasoning-task) |
| 121 | clustered | 2 | 2 | 1 | [Episodic memory testing](#121-episodic-memory-testing) |
| 122 | clustered | 2 | 2 | 2 | [empathy attribution task](#122-empathy-attribution-task) |
| 123 | clustered | 2 | 2 | 2 | [emotional memory task](#123-emotional-memory-task) |
| 124 | clustered | 2 | 2 | 2 | [Emotion Evaluation subtest of The Awareness of Socia](#124-emotion-evaluation-subtest-of-the-awareness-of-social-inference-test) |
| 125 | clustered | 2 | 2 | 2 | [virtual supermarket task](#125-virtual-supermarket-task) |
| 126 | clustered | 2 | 2 | 2 | [pupillometry experiment](#126-pupillometry-experiment) |
| 127 | clustered | 2 | 2 | 2 | [research neuropsychological battery](#127-research-neuropsychological-battery) |
| 128 | clustered | 2 | 2 | 2 | [classification of FTD, AD and NC](#128-classification-of-ftd-ad-and-nc) |
| 129 | clustered | 2 | 3 | 3 | [repeat letters during PET uptake](#129-repeat-letters-during-pet-uptake) |
| 130 | clustered | 2 | 2 | 2 | [Implicit Sad Facial Affect Recognition Task](#130-implicit-sad-facial-affect-recognition-task) |
| 131 | clustered | 2 | 2 | 2 | [recognition](#131-recognition) |
| 132 | clustered | 2 | 2 | 1 | [dynamic faces task](#132-dynamic-faces-task) |
| 133 | clustered | 2 | 2 | 2 | [sad mood elaboration](#133-sad-mood-elaboration) |
| 134 | clustered | 2 | 2 | 2 | [Emotional Anticipation Task](#134-emotional-anticipation-task) |
| 135 | clustered | 2 | 2 | 1 | [emotion induction task](#135-emotion-induction-task) |
| 136 | clustered | 2 | 2 | 2 | [adapted MRI version of the Anger Articulated Thought](#136-adapted-mri-version-of-the-anger-articulated-thoughts-during-simulated-situations-atss-paradigm) |
| 137 | clustered | 2 | 2 | 2 | [KidVid](#137-kidvid) |
| 138 | clustered | 2 | 2 | 2 | [Carmageddon virtual violence gameplay](#138-carmageddon-virtual-violence-gameplay) |
| 139 | atlas | 1 | 1 | 1 | [spatial working memory task](#139-spatial-working-memory-task) |
| 140 | atlas | 1 | 1 | 1 | [fixation task](#140-fixation-task) |
| 141 | atlas | 1 | 1 | 1 | [acupuncture task](#141-acupuncture-task) |
| 142 | atlas | 1 | 1 | 1 | [same-different task](#142-same-different-task) |
| 143 | atlas | 1 | 1 | 1 | [object classification](#143-object-classification) |
| 144 | atlas | 1 | 1 | 1 | [Spatial cuing paradigm](#144-spatial-cuing-paradigm) |
| 145 | atlas | 1 | 1 | 1 | [visually guided saccade task](#145-visually-guided-saccade-task) |
| 146 | atlas | 1 | 1 | 1 | [Moral Dilemma Task](#146-moral-dilemma-task) |
| 147 | atlas | 1 | 1 | 1 | [visual search task](#147-visual-search-task) |
| 148 | atlas | 1 | 1 | 1 | [finger tapping task](#148-finger-tapping-task) |
| 149 | atlas | 1 | 1 | 1 | [passive listening](#149-passive-listening) |
| 150 | atlas | 1 | 1 | 1 | [Cambridge Face Memory Test](#150-cambridge-face-memory-test) |
| 151 | atlas | 1 | 1 | 1 | [prospective memory task](#151-prospective-memory-task) |
| 152 | atlas | 1 | 1 | 1 | [semantic classification task](#152-semantic-classification-task) |
| 153 | atlas | 1 | 1 | 1 | [source memory test](#153-source-memory-test) |
| 154 | atlas | 1 | 1 | 1 | [word identification](#154-word-identification) |
| 155 | atlas | 1 | 1 | 1 | [Memory encoding task](#155-memory-encoding-task) |
| 156 | atlas | 1 | 1 | 1 | [target detection task](#156-target-detection-task) |
| 157 | atlas | 1 | 1 | 1 | [gambling task](#157-gambling-task) |
| 158 | atlas | 1 | 1 | 1 | [gender discrimination task](#158-gender-discrimination-task) |
| 159 | atlas | 1 | 1 | 1 | [face working memory task](#159-face-working-memory-task) |
| 160 | atlas | 1 | 1 | 1 | [alternating runs paradigm](#160-alternating-runs-paradigm) |
| 161 | atlas | 1 | 1 | 1 | [social judgment task](#161-social-judgment-task) |
| 162 | atlas | 1 | 1 | 1 | [meditation task](#162-meditation-task) |
| 163 | atlas | 1 | 1 | 1 | [directed forgetting task](#163-directed-forgetting-task) |
| 164 | atlas | 1 | 1 | 1 | [Emotion Identification Task](#164-emotion-identification-task) |
| 165 | atlas | 1 | 1 | 1 | [recognition memory test](#165-recognition-memory-test) |
| 166 | atlas | 1 | 1 | 1 | [emotional regulation task](#166-emotional-regulation-task) |
| 167 | atlas | 1 | 4 | 1 | [rubber hand illusion](#167-rubber-hand-illusion) |
| 168 | atlas | 1 | 1 | 1 | [Rapid Visual Information Processing](#168-rapid-visual-information-processing) |
| 169 | atlas | 1 | 1 | 1 | [social decision-making task](#169-social-decision-making-task) |
| 170 | atlas | 1 | 1 | 1 | [reversal learning task](#170-reversal-learning-task) |
| 171 | atlas | 1 | 1 | 1 | [Wisconsin card sorting test](#171-wisconsin-card-sorting-test) |
| 172 | atlas | 1 | 1 | 1 | [Information Sampling Task](#172-information-sampling-task) |
| 173 | atlas | 1 | 1 | 1 | [balloon analogue risk task](#173-balloon-analogue-risk-task) |
| 174 | atlas | 1 | 1 | 1 | [set-shifting task](#174-set-shifting-task) |
| 175 | atlas | 1 | 1 | 1 | [Cambridge Gambling Task](#175-cambridge-gambling-task) |
| 176 | clustered | 1 | 2 | 2 | [Ekman 60](#176-ekman-60) |
| 177 | clustered | 1 | 2 | 2 | [happy film](#177-happy-film) |
| 178 | clustered | 1 | 5 | 5 | [Experiment 1: static body emotion matching](#178-experiment-1-static-body-emotion-matching) |
| 179 | clustered | 1 | 4 | 4 | [semantic congruity task](#179-semantic-congruity-task) |
| 180 | clustered | 1 | 2 | 2 | [Letter-guided fluency](#180-letter-guided-fluency) |
| 181 | clustered | 1 | 2 | 2 | [categorization task](#181-categorization-task) |
| 182 | clustered | 1 | 2 | 2 | [Scanner anti-smoking messages task](#182-scanner-anti-smoking-messages-task) |
| 183 | clustered | 1 | 4 | 2 | [EBA functional localizer](#183-eba-functional-localizer) |
| 184 | clustered | 1 | 2 | 2 | [aversive and nonaversive image viewing with repetiti](#184-aversive-and-nonaversive-image-viewing-with-repetition) |

---

## Categories

### 1. resting state

clustered · 174 studies · 193 tasks · 45 distinct names

Stimuli: (unspecified) (192), stress (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `6hHwvTQrWwAc` | eyes-closed resting state | (unspecified) | eyes closed resting state |
| `28457805` | eyes-open resting state | (unspecified) |  |
| `4qzjhCnhEGDg` | eyes-open resting state | (unspecified) | resting state |
| `QQCjAAT6SwwQ` | overnight sleep-wake EEG-fMRI | (unspecified) | wakefulness; NREM stage N1; NREM stage N2; NREM stage N3 |
| `14527604` | quiet wakefulness PET | (unspecified) |  |
| `30317048` | Resting baseline | (unspecified) |  |
| `5jE9RSgbcgB8` | resting FDG PET | (unspecified) |  |
| `26721391` | resting FDG-PET protocol | (unspecified) |  |
| `15212830` | resting PET | (unspecified) |  |
| `21958648` | resting PET | (unspecified) |  |
| `3yXBYR2DjdYs` | resting PET scan | (unspecified) | placebo; citalopram |
| `24622915` | resting scan | (unspecified) |  |
| `3AUFqfW6zT4k` | resting SPECT | (unspecified) | resting |
| `21458537` | resting state | (unspecified) |  |
| `22028765` | resting state | (unspecified) |  |
| `23164495` | resting state | (unspecified) |  |
| `23719145` | resting state | (unspecified) |  |
| `24040338` | resting state | (unspecified) |  |
| `24286968` | resting state | (unspecified) |  |
| `25063233` | resting state | (unspecified) |  |
| `25188321` | resting state | (unspecified) |  |
| `25282597` | resting state | (unspecified) |  |
| `25385625` | resting state | (unspecified) |  |
| `26441584` | resting state | (unspecified) |  |
| `26836150` | resting state | (unspecified) |  |
| `26912642` | resting state | (unspecified) |  |
| `27436017` | resting state | (unspecified) |  |
| `27662284` | resting state | (unspecified) |  |
| `27757078` | resting state | (unspecified) | resting state |
| `27935229` | resting state | (unspecified) | resting state |
| `28544994` | resting state | (unspecified) |  |
| `28715907` | resting state | (unspecified) |  |
| `29175464` | resting state | (unspecified) |  |
| `29582514` | resting state | (unspecified) |  |
| `30930736` | resting state | (unspecified) |  |
| `31824254` | resting state | (unspecified) |  |
| `32447010` | resting state | (unspecified) |  |
| `33329355` | resting state | (unspecified) |  |
| `347jHLHiWNjT` | resting state | (unspecified) |  |
| `3RCUCdmrF2gT` | resting state | (unspecified) |  |
| `6XqZHwsAYE7Z` | resting state | (unspecified) | resting state |
| `MTa5xEDtoV9C` | resting state | (unspecified) |  |
| `XF2JVSeyxPvD` | resting state | (unspecified) |  |
| `hfxYxogTC9hi` | resting state | (unspecified) |  |
| `koSSAGizw863` | resting state | (unspecified) |  |
| `kzMj26hGWacQ` | resting state | (unspecified) |  |
| `kzMj26hGWacQ` | resting state | (unspecified) |  |
| `kzMj26hGWacQ` | resting state | (unspecified) |  |
| `pMZeVGA2rQQi` | resting state | (unspecified) |  |
| `pMZeVGA2rQQi` | resting state | (unspecified) |  |
| `wVLtTpXm9fTq` | resting state | (unspecified) |  |
| `28616383` | resting state functional MRI | (unspecified) |  |
| `27654848` | resting state MRI protocol | (unspecified) |  |
| `22998631` | resting state scan | (unspecified) |  |
| `26899786` | resting state task | (unspecified) |  |
| `22174510` | resting state with eyes closed | (unspecified) |  |
| `20621162` | resting-state | (unspecified) |  |
| `23092697` | resting-state | (unspecified) |  |
| `23959214` | resting-state | (unspecified) |  |
| `27870505` | resting-state | (unspecified) |  |
| `28474365` | resting-state | (unspecified) |  |
| `29368421` | resting-state | (unspecified) |  |
| `30306886` | resting-state | (unspecified) |  |
| `30343133` | Resting-state | (unspecified) |  |
| `30414987` | resting-state | (unspecified) |  |
| `3puRtcCeeZKa` | resting-state | (unspecified) |  |
| `4W9dVYMHTWWT` | resting-state | (unspecified) |  |
| `6oTrCJA43Jcd` | resting-state | (unspecified) |  |
| `7HPLh5nJzmP5` | resting-state | (unspecified) | rest |
| `7HPLh5nJzmP5` | resting-state | (unspecified) |  |
| `7HPLh5nJzmP5` | resting-state | (unspecified) | resting-state |
| `7HPLh5nJzmP5` | resting-state | (unspecified) | resting-state |
| `7sJNz4Y8cj2j` | resting-state | (unspecified) |  |
| `JzsUUQbDr2bm` | resting-state | (unspecified) |  |
| `RYUouEzgqEYw` | resting-state | (unspecified) |  |
| `eaEGQiVtDp9e` | resting-state | (unspecified) |  |
| `wEPit6Ugzc9M` | resting-state | (unspecified) |  |
| `wXYv2rfyxdCh` | resting-state | (unspecified) |  |
| `yjdWZMngQXsi` | resting-state | (unspecified) |  |
| `27095057` | Resting-state assessment | (unspecified) |  |
| `rnQuXvACXaM2` | resting-state EEG | (unspecified) | eyes closed; eyes open |
| `23422198` | resting-state fMRI | (unspecified) |  |
| `25990865` | resting-state fMRI | (unspecified) |  |
| `26846195` | resting-state fMRI | (unspecified) |  |
| `27177299` | resting-state fMRI | (unspecified) |  |
| `28059782` | resting-state fMRI | (unspecified) | fixation cross |
| `28119587` | resting-state fMRI | (unspecified) |  |
| `28954876` | resting-state fMRI | (unspecified) |  |
| `29808100` | Resting-state fMRI | (unspecified) |  |
| `30456877` | resting-state fMRI | (unspecified) |  |
| `3VPhNbonqo3D` | resting-state fMRI | (unspecified) |  |
| `42tiPgNNNNhA` | resting-state fMRI | (unspecified) |  |
| `4K9cZoMXFiGY` | resting-state fMRI | (unspecified) |  |
| `6YQJbrTNuPkN` | resting-state fMRI | (unspecified) | resting state |
| `6oTrCJA43Jcd` | resting-state fMRI | (unspecified) |  |
| `6oTrCJA43Jcd` | resting-state fMRI | (unspecified) |  |
| `7HPLh5nJzmP5` | resting-state fMRI | (unspecified) | rest/fixation |
| `GcTmoGJtEscX` | resting-state fMRI | (unspecified) |  |
| `JzsUUQbDr2bm` | resting-state fMRI | (unspecified) | rest |
| `R3G58GFFS5XH` | resting-state fMRI | (unspecified) |  |
| `SLAhLaM6XEwm` | resting-state fMRI | (unspecified) |  |
| `TbGgJuGRDT7C` | resting-state fMRI | (unspecified) |  |
| `aVGe9BmFTMDR` | resting-state fMRI | (unspecified) |  |
| `aVGe9BmFTMDR` | resting-state fMRI | (unspecified) |  |
| `aVGe9BmFTMDR` | resting-state fMRI | (unspecified) | rest |
| `eaEGQiVtDp9e` | resting-state fMRI | (unspecified) |  |
| `eaEGQiVtDp9e` | resting-state fMRI | (unspecified) |  |
| `eaEGQiVtDp9e` | Resting-state fMRI | (unspecified) | resting state |
| `iPQpqvqMLWHi` | resting-state fMRI | (unspecified) |  |
| `kzMj26hGWacQ` | Resting-state fMRI | (unspecified) | Rest |
| `23455594` | resting-state functional connectivity | (unspecified) |  |
| `2Qx5cSsFtZFQ` | resting-state functional connectivity | (unspecified) | resting state |
| `TgcHKMRfrVog` | Resting-state functional connectivity | (unspecified) | Rest |
| `iDRdeSjCPuTK` | resting-state functional connectivity | (unspecified) |  |
| `29562053` | resting-state functional connectivity paradigm | (unspecified) |  |
| `4FurrLyxufH5` | resting-state functional connectome | (unspecified) |  |
| `4NG5rsXzKBfW` | resting-state functional magnetic resonance imaging | (unspecified) |  |
| `6BjpB8RumM3b` | resting-state functional magnetic resonance imaging | (unspecified) |  |
| `6dneAxZtouR6` | resting-state functional magnetic resonance imaging | (unspecified) |  |
| `7BTsVWgRFv88` | resting-state functional magnetic resonance imaging | (unspecified) |  |
| `DZF4iSJtxGRX` | resting-state functional magnetic resonance imaging | (unspecified) |  |
| `PbGfHLAUGUiZ` | resting-state functional magnetic resonance imaging | (unspecified) |  |
| `f5fMiAncVFa4` | resting-state functional magnetic resonance imaging | (unspecified) |  |
| `5Tk6KepsPVNd` | resting-state functional magnetic resonance imaging (rs-fMRI) | (unspecified) |  |
| `30859283` | resting-state functional MRI | (unspecified) |  |
| `S6SPYuLHVuMa` | resting-state functional MRI | (unspecified) |  |
| `kRtKGvVsa9zE` | resting-state functional MRI | (unspecified) |  |
| `2sS2RUb2zgC6` | resting-state functional MRI protocol | (unspecified) |  |
| `5uFUVWdbgU7m` | resting-state functional MRI protocol | (unspecified) |  |
| `UQzNa9ueMEKc` | resting-state functional neuroimaging | (unspecified) |  |
| `26204262` | resting-state paradigm | (unspecified) |  |
| `10526199` | resting-state PET | (unspecified) |  |
| `6zFJ4zjj7Fvf` | resting-state PET | (unspecified) |  |
| `21849646` | resting-state protocol | (unspecified) |  |
| `23455594` | resting-state protocol | (unspecified) |  |
| `25727574` | resting-state protocol | (unspecified) |  |
| `26094186` | resting-state protocol | (unspecified) |  |
| `26888621` | resting-state protocol | (unspecified) |  |
| `29065207` | resting-state protocol | (unspecified) |  |
| `29924294` | resting-state protocol | (unspecified) |  |
| `30046142` | resting-state protocol | (unspecified) |  |
| `32358091` | resting-state protocol | (unspecified) |  |
| `35Dj9JD4kP5t` | resting-state protocol | (unspecified) |  |
| `377FPJ83q26Y` | resting-state protocol | (unspecified) |  |
| `3HahWSVHQz2k` | resting-state protocol | (unspecified) |  |
| `3SiC2gfpTnuV` | resting-state protocol | (unspecified) |  |
| `3kErGccwvKBZ` | resting-state protocol | (unspecified) |  |
| `3qqAEQKxK7hD` | resting-state protocol | (unspecified) |  |
| `4Rnxiv5VE9Xf` | resting-state protocol | (unspecified) |  |
| `4jWsjfqNBdY4` | resting-state protocol | (unspecified) |  |
| `4unqSttnM3yk` | resting-state protocol | (unspecified) |  |
| `5pXAS2MXKepB` | resting-state protocol | (unspecified) |  |
| `5x76V4RZL9kQ` | resting-state protocol | (unspecified) |  |
| `6BBRzUYTbbaY` | resting-state protocol | (unspecified) |  |
| `6EfwdW3t94do` | resting-state protocol | (unspecified) |  |
| `77RxhyrVaLqR` | resting-state protocol | (unspecified) |  |
| `7U5PzK2hGiPh` | resting-state protocol | (unspecified) | rest |
| `7V3UUqDvdM3U` | resting-state protocol | (unspecified) |  |
| `EEhRux9gjjgx` | resting-state protocol | (unspecified) |  |
| `PudCrR4yYaZ4` | resting-state protocol | (unspecified) |  |
| `WGdzLXGFvFxP` | resting-state protocol | (unspecified) |  |
| `XxxrqukPD88i` | resting-state protocol | (unspecified) |  |
| `cKFEupE5R6Vy` | resting-state protocol | (unspecified) |  |
| `eeoD6J7gcU9H` | resting-state protocol | (unspecified) |  |
| `g5sdQaXMkBt7` | resting-state protocol | (unspecified) |  |
| `hJh6Da38bbvc` | resting-state protocol | (unspecified) | resting state |
| `rh3hdtJfUe45` | resting-state protocol | (unspecified) |  |
| `tZPnCvCSyToS` | resting-state protocol | (unspecified) | rest |
| `uYWEcKipN8vP` | resting-state protocol | (unspecified) |  |
| `v7EV4tRutHwW` | resting-state protocol | (unspecified) |  |
| `22781311` | resting-state scan | (unspecified) |  |
| `26364127` | resting-state scan | (unspecified) |  |
| `26594631` | Resting-state scan | (unspecified) |  |
| `26809248` | resting-state scan | (unspecified) |  |
| `29622050` | resting-state scan | (unspecified) |  |
| `29909301` | resting-state scan | (unspecified) |  |
| `29939345` | Resting-state scan | (unspecified) |  |
| `30120426` | resting-state scan | (unspecified) |  |
| `31706906` | resting-state scan | (unspecified) |  |
| `32900658` | resting-state scan | (unspecified) |  |
| `25365801` | resting-state scanning | (unspecified) |  |
| `26955839` | resting-state scanning | (unspecified) |  |
| `27427215` | resting-state task | (unspecified) |  |
| `27510495` | resting-state task | (unspecified) |  |
| `30181137` | resting-state task | (unspecified) |  |
| `QQCjAAT6SwwQ` | sleep–wake cycle | (unspecified) | Wake; N1; N2; N3 |
| `QQCjAAT6SwwQ` | sleep–wake cycle | (unspecified) | Wake; N1; N2; N3 |
| `25906795` | stress induction and post-stress resting-state protocol | stress | incision; sham |
| `28337409` | task-free fMRI | (unspecified) |  |
| `25273996` | task-free functional MRI | (unspecified) |  |
| `20410145` | task-free resting-state fMRI | (unspecified) |  |
| `27071080` | task-free resting-state protocol | (unspecified) | resting |
| `30836325` | task-free resting-state protocol | (unspecified) |  |

### 2. Emotion Regulation Task

named by the Cognitive Atlas · 148 studies · 153 tasks · 47 distinct names

Stimuli: emotion (151), pain (1), social (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27444935` | attentional deployment emotion regulation task | emotion | unpleasant no focus; unpleasant arousing focus; unpleasant non-arousing focus |
| `30834098` | automatic emotion regulation priming paradigm | emotion | implicit inhibition condition; implicit neutral condition; explicit inhibition condition; explicit neutral condition |
| `29990584` | Automatic Emotion Regulation Task | emotion | negative; neutral; positive |
| `29990584` | Automatic Emotion Regulation Task | emotion | negative; neutral; positive |
| `29990584` | Automatic Emotion Regulation Task | emotion | positive pictures; neutral pictures; negative pictures |
| `29990584` | Automatic Emotion Regulation Task | emotion | negative; neutral; positive |
| `26306990` | aversive classical conditioning with cognitive emotion regulation | emotion | Distance; Attend |
| `17133391` | bidirectional emotion regulation through reappraisal strategies | emotion | neutral-view; negative-view; negative-increase; negative-decrease |
| `30008118` | cognitive emotion regulation | emotion | CR; AD; baseline |
| `28560818` | cognitive emotion regulation reappraisal paradigm | emotion | spontaneous_negative; distance; neutral |
| `26299299` | cognitive emotion regulation task | emotion | view neutral; view sad; reappraise sad |
| `19589502` | deliberate emotion regulation | emotion |  |
| `27665000` | disgust image viewing and emotion regulation | emotion | Passive Viewing; Placebo; Reappraisal |
| `29430790` | effortful emotion regulation | emotion | reappraising; attending-negative; attending-neutral |
| `17493834` | emotion regulation | emotion | increase; attend; decrease |
| `19429172` | emotion regulation | emotion | regulation of negative emotion; regulation of positive emotion; viewing of negative emotion; viewing of positive emotion (+1) |
| `19945471` | emotion regulation | emotion |  |
| `21212840` | emotion regulation | emotion | view; reappraise; suppress; neutral |
| `21345342` | emotion regulation | emotion | BASE; NAT; REAP; ESUP |
| `22613776` | emotion regulation | emotion | view; reappraisal; distraction |
| `23123362` | emotion regulation | emotion | placebo condition; control condition; reappraisal condition |
| `23644585` | emotion regulation | emotion | endogenous inhibition; exogenous inhibition; endogenous feel; exogenous feel |
| `24682003` | emotion regulation | emotion | regulated aversive stimulation; unregulated aversive stimulation; fixation following regulated aversive stimulation; fixation following unregulated aversive stimulation |
| `24782800` | emotion regulation | emotion | passive viewing; selective attention; reappraisal; emotion regulation |
| `24808872` | emotion regulation | emotion | view neutral; view negative; permit negative |
| `24936178` | emotion regulation | emotion | intrapersonal emotion regulation; interpersonal emotion regulation; watch |
| `25461265` | emotion regulation | emotion | view; regulate |
| `27709512` | emotion regulation | emotion | up-regulation; down-regulation; distraction; look aversive (+1) |
| `28960669` | emotion regulation | emotion | distraction; reappraisal |
| `30527356` | emotion regulation and reactivity paradigm | emotion | reappraise; change image; maintain; just look |
| `31680150` | emotion regulation and risky decision-making task | emotion | Decrease; Look-Negative; Look-Neutral |
| `28148724` | emotion regulation choice task | emotion |  |
| `29529407` | emotion regulation fMRI task | emotion | look negative; look neutral; decrease |
| `18786365` | emotion regulation paradigm | emotion | CS+; CS−; Attend; Regulate |
| `23223803` | emotion regulation paradigm | emotion |  |
| `24173657` | emotion regulation paradigm | emotion |  |
| `25771686` | emotion regulation paradigm | emotion |  |
| `25798822` | emotion regulation paradigm | emotion | implementation intention; goal intention; attend |
| `30946860` | emotion regulation paradigm | emotion | view neutral; view negative; distract; reappraise |
| `16249098` | emotion regulation picture-viewing task | emotion | Watch Moral; Watch Non-moral; Decrease Moral; Decrease Non-Moral (+1) |
| `16624961` | emotion regulation task | emotion | increase; attend; decrease |
| `17488204` | emotion regulation task | emotion | increase; decrease; not alter |
| `18985136` | emotion regulation task | emotion | Maintain; Reappraise; Baseline |
| `19398537` | emotion regulation task | emotion | spider; neutral; aversive |
| `20673804` | emotion regulation task | emotion | Reduce; Maintain |
| `21041200` | emotion regulation task | emotion | view; reappraisal; distraction; emotional (+1) |
| `21686071` | emotion regulation task | emotion | look-neutral; look-gross; decrease-gross; increase-gross |
| `21861676` | emotion regulation task | emotion |  |
| `22228751` | emotion regulation task | emotion | look negative; look neutral; decrease negative |
| `22401827` | emotion regulation task | emotion | Reduce; Maintain |
| `22634856` | emotion regulation task | emotion | enhance; suppress; maintain |
| `22832855` | emotion regulation task | emotion | increase; decrease; maintain |
| `22998631` | Emotion Regulation task | emotion | Sad Reappraise; Sad View; Sad Distract |
| `23111120` | emotion regulation task | emotion | Baseline; Maintain; Reappraise |
| `23216889` | emotion regulation task | emotion | negative photographs; neutral photographs; regulate negative; look negative |
| `23482626` | emotion regulation task | emotion | attend positive; attend negative; reduce negative; enhance positive |
| `23500898` | emotion regulation task | emotion | Decrease; Increase; Maintain |
| `23887812` | emotion regulation task | emotion | positive–decrease; positive–view; negative–decrease; negative–view (+1) |
| `23948633` | emotion regulation task | emotion | view negative; view neutral; view positive; reappraisal negative (+1) |
| `24145409` | Emotion Regulation Task | emotion | Look; Maintain; Reappraisal; fixation |
| `24246489` | Emotion Regulation Task | emotion | Reappraise; Maintain; Neutral Look |
| `24270731` | emotion regulation task | emotion | maintain; reappraise; pixel-wise scrambled baseline |
| `24397574` | emotion regulation task | emotion | increase; attend |
| `24677490` | Emotion Regulation Task | emotion | Reappraise; Maintain; Look |
| `24690369` | emotion regulation task | emotion | PermA; PermN |
| `24760016` | emotion regulation task | emotion | Attend Neutral; Attend Negative; Reappraise; Suppress |
| `24941136` | emotion regulation task | emotion | attend negative; reappraise; suppress; attend neutral (+1) |
| `24993897` | emotion regulation task | emotion | Regulate Negative; Permit Negative; Permit Neutral |
| `25156399` | emotion regulation task | emotion | images; Reappraisal; Maintain; fixation baseline |
| `25380765` | emotion regulation task | emotion | reappraisal; distraction; fixation cross |
| `25439326` | emotion regulation task | emotion | reactivity/negative; regulation/negative; reactivity/neutral; regulation/neutral |
| `25603413` | emotion regulation task | emotion | reappraisal; view; distraction |
| `25617820` | emotion regulation task | emotion | accept; reappraise; view |
| `25997918` | emotion regulation task | emotion |  |
| `26111649` | emotion regulation task | emotion | Look; Maintain; Reappraise |
| `26210693` | emotion regulation task | emotion | Think Objectively; Watch |
| `26299297` | emotion regulation task | emotion | look negative; decrease; look positive; increase |
| `26341903` | emotion regulation task | emotion | Regulate Negative; Attend Negative |
| `26529426` | emotion regulation task | emotion | Maintain; Reappraisal; fixation baseline |
| `26537018` | Emotion regulation task | emotion |  |
| `26647971` | Emotion Regulation Task | emotion | Look; Maintain; Reappraise |
| `26692636` | emotion regulation task | emotion |  |
| `26896742` | emotion regulation task | emotion | Think Objectively; Watch |
| `27013102` | emotion regulation task | emotion | WatchNeu; WatchNeg; RegulateNeg |
| `27217106` | emotion regulation task | emotion | Look; Reappraise; Label |
| `27336036` | emotion regulation task | emotion | View Neutral; Fixation; View Negative; Reappraise (+2) |
| `27973443` | Emotion Regulation Task | emotion | Look; Maintain; Reappraise |
| `28126372` | emotion regulation task | emotion | Reappraise; Look Negative; Look Neutral |
| `28197859` | emotion regulation task | emotion | view-neutral; view-angry; increase; decrease |
| `28273918` | emotion regulation task | emotion | watch-negative; watch-neutral; reappraise-negative; suppress-negative |
| `28372994` | emotion regulation task | emotion | View Negative; View Neutral; Reappraise Negative; Attend Negative |
| `28402571` | emotion regulation task | emotion | Accept; Worry; Suppress |
| `28462086` | Emotion Regulation Task | emotion | Reappraise; Maintain; Look; fixation cross |
| `28501740` | Emotion Regulation Task | emotion | Reappraise; Look-Negative; Look-Neutral |
| `28946039` | emotion regulation task | emotion | passive observation; affect labeling |
| `29061386` | Emotion Regulation Task | emotion | ReappNeg; LookNeg; LookNeut; fixation |
| `29128142` | emotion regulation task | emotion | feel; analyze; active baseline |
| `29154365` | emotion regulation task | emotion | Neutral; Look; Decrease |
| `29321971` | emotion regulation task | emotion | negative watch; neutral watch; negative distance; positive watch (+1) |
| `29362440` | emotion regulation task | emotion | positive watch; positive distance; neutral watch |
| `29428771` | Emotion Regulation Task | emotion | Reappraise; Look-Negative; Look-Neutral |
| `29743808` | emotion regulation task | emotion | look negative; look neutral; decrease negative |
| `29753591` | emotion regulation task | emotion | neutral attend; fear-related attend; fear-related regulate; OCD-related attend (+1) |
| `29872413` | emotion regulation task | emotion | Negative; Neutral |
| `29931375` | emotion regulation task | emotion | Maintain; Reappraise |
| `30062613` | emotion regulation task | emotion | observe; label |
| `30261360` | emotion regulation task | emotion | attend neutral; attend negative; reappraisal; suppression (+2) |
| `30287300` | emotion regulation task | emotion | regulate; view |
| `30341276` | emotion regulation task | emotion | Maintain; Reappraisal |
| `30355375` | Emotion Regulation Task | emotion | ReappNeg; LookNeg; LookNeut |
| `30408261` | Emotion Regulation Task | emotion | Reappraise; Look-Negative; Look-Neutral; fixation |
| `30414987` | emotion regulation task | emotion | view; internal; external |
| `30455624` | Emotion regulation task | emotion |  |
| `30773147` | emotion regulation task | emotion | attend-neutral; attend-negative; reappraise-negative |
| `31146576` | emotion regulation task | emotion | negative images; neutral images; positive images |
| `31414234` | emotion regulation task | emotion | view; increase; decrease |
| `31431608` | emotion regulation task | emotion | Observe; Maintain; Regulate |
| `31679906` | emotion regulation task | emotion | attend; regulate |
| `31680164` | emotion regulation task | emotion | positive images; neutral images; up-regulation; watch |
| `31892465` | Emotion Regulation task | emotion | REAPPRAISE-Negative; LOOK-Negative |
| `31892465` | Emotion Regulation task | emotion | REAPPRAISE-Negative; LOOK-Negative; LOOK-Neutral |
| `32435746` | emotion regulation task | emotion | negative permit stimulation; negative detach stimulation; negative permit relaxation; negative detach relaxation |
| `35EHzQ5XwmDB` | emotion regulation task | emotion | Maintain; Reduce |
| `5sgAMpW9KGHJ` | emotion regulation task | emotion | negative decrease; negative look; positive decrease; positive look (+1) |
| `6Qh7BaQ43Gu8` | emotion regulation task | emotion | Maintain; Reappraise; baseline |
| `27559724` | Emotion Regulation Task (ERT) | emotion | Look; Maintain; Reappraise |
| `26083379` | emotion regulation tasks | emotion | Look-neutral; Look-negative; Suppress-negative; Observe-negative |
| `22217336` | emotion regulation viewing task | emotion | NAT; REAP; ESUP; BASE |
| `23696200` | emotion regulation while viewing suffering and neutral images | emotion | suffering; neutral |
| `25278851` | emotion regulation with aversive pictures | emotion | REGULATE; VIEW; NEUTRAL |
| `25209373` | emotion regulation with film clips | emotion | reappraisal; passive viewing; selective attention |
| `22956675` | emotion-regulation paradigm | emotion | Suppression; Reappraisal; Appraise |
| `31025560` | emotion-regulation reappraisal task | emotion | neutral view; negative view; reappraise |
| `29145910` | emotion-regulation task | emotion | negative, close; negative, far; neutral, close; neutral, far |
| `31311717` | emotion-regulation task | emotion | look neutral; look negative; decrease negative |
| `22592057` | explicit emotion regulation | emotion | View; Up-regulate; Down-regulate |
| `25631055` | explicit emotion regulation task | emotion | Increase; Decrease; Look-Sports; Look-Neutral |
| `30173058` | explicit emotion regulation task | emotion |  |
| `28419607` | fMRI Emotion Regulation Task | emotion | Reappraise; Maintain; Look |
| `22345383` | instructed emotion regulation task | emotion | negative look; neutral look; downregulate; upregulate |
| `24349161` | negative emotion regulation | emotion | negative-diminish; negative-enhance; negative-maintain; neutral-maintain |
| `24865373` | negative emotional reactivity and emotion regulation task | emotion | Amygdala-baseline; Amygdala-reactivity; Amygdala-regulation |
| `22586252` | Negative self-belief emotion regulation task | emotion |  |
| `26027740` | OCD-specific emotion regulation task | emotion | fear; neutral; OCD-related |
| `21861676` | pain regulation task | pain |  |
| `21867991` | positive emotion regulation task | emotion | positive suppress; positive maintain |
| `28373890` | Positive Emotion Regulation Task | emotion | view positive; reappraise positive; neutral view |
| `29787789` | positive emotion regulation task | emotion | Increase; Maintain; Decrease |
| `30245249` | Positive emotion regulation task | emotion |  |
| `21896496` | regulation task | emotion | decrease; attend; stigma; IAPS |
| `31379539` | rtfMRI-NF emotion regulation | emotion | view; regulation; baseline |
| `23450458` | social emotion regulation task | social | IMITATION (IMT); expressive suppression (eSUP); Gender decision (GND) |
| `6fuBs3Spk2Dv` | voluntary emotion regulation during negative autobiographical memories | emotion | feel; analyze |

### 3. reappraisal task

named by the Cognitive Atlas · 74 studies · 78 tasks · 34 distinct names

Stimuli: emotion (61), food (6), (unspecified) (6), faces (2), money (1), tobacco (1), social (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `17699669` | affective reappraisal task | emotion | decrease; attend |
| `25533974` | Chatroom reappraisal task | faces | reappraisal; attend |
| `21907809` | cognitive reappraisal | (unspecified) | Up; Down; Maintain |
| `25198094` | cognitive reappraisal | emotion | look-neutral; look-gross; decrease-gross; increase-gross |
| `25337107` | cognitive reappraisal | emotion | control; reappraisal; unpleasant; neutral |
| `25618212` | cognitive reappraisal | emotion | Pre-Instruction; Enhance Positive; Enhance Negative |
| `26106309` | cognitive reappraisal | (unspecified) | regulate; watch |
| `15488398` | cognitive reappraisal of negative emotion | emotion | increase; decrease; look |
| `23945981` | cognitive reappraisal of negative self-beliefs | emotion | neutral; react NSB; reappraise NSB |
| `23796796` | cognitive reappraisal of sad images | emotion | reappraise-sad; attend-sad; attend-neutral |
| `3myrCWvfLtQa` | cognitive reappraisal of sad images | emotion | sad images; neutral images; reappraise-sad; attend-sad |
| `24646887` | cognitive reappraisal of sadness | emotion | view neutral; view sad; reappraise sad |
| `21195392` | cognitive reappraisal paradigm | emotion | negative; neutral; decrease; maintain (+1) |
| `20385663` | cognitive reappraisal task | emotion | Experience; Reappraise |
| `23712090` | cognitive reappraisal task | emotion | Observe; Maintain; Suppress |
| `23825408` | cognitive reappraisal task | emotion | Positive-Experience; Negative-Experience; Positive-Regulate; Negative-Regulate (+1) |
| `24493837` | cognitive reappraisal task | emotion | decrease negative; view negative |
| `25433095` | cognitive reappraisal task | faces | Look-Neut; Maintain-Neg; Reappraise-Neg |
| `26435254` | cognitive reappraisal task | emotion | reduce; maintain; rest |
| `26596970` | cognitive reappraisal task | emotion | look-neutral; look-negative; reappraise |
| `26746624` | cognitive reappraisal task | emotion | View Neutral; View Sad; Reappraise |
| `27003840` | cognitive reappraisal task | emotion | Observe; Maintain; Regulate |
| `27217113` | cognitive reappraisal task | emotion | Negative; Neutral |
| `27524285` | cognitive reappraisal task | emotion | reappraise negative images; view negative images; view neutral images |
| `28402574` | cognitive reappraisal task | emotion | look-neutral; look-negative; reappraise; fixation |
| `28598734` | cognitive reappraisal task | emotion | decrease-negative; look-negative; look-neutral |
| `29689500` | cognitive reappraisal task | emotion | Decrease Negative; Look Negative; Increase Negative; Look Neutral |
| `30142170` | cognitive reappraisal task | emotion |  |
| `31032025` | cognitive reappraisal task | emotion |  |
| `31464495` | cognitive reappraisal task | emotion | Far; Look-Negative; Look-Neutral |
| `31769305` | cognitive reappraisal task | emotion | attend-positive; attend-negative; attend-neutral; reinterpret-positive (+2) |
| `25485181` | craving reappraisal task | tobacco | ReappraiseSmoking; LookSmoking |
| `31254647` | creative cognitive reappraisal task | emotion | creative reappraisal; ordinary reappraisal; objective description; negative picture presentation (+1) |
| `19957268` | delayed cognitive reappraisal paradigm | emotion | negative pictures; neutral pictures; decrease; maintain (+1) |
| `21922013` | detachment reappraisal paradigm | (unspecified) | NT/NR; NT/R; T/NR; T/R |
| `19400679` | Distraction and reappraisal emotion regulation task | emotion |  |
| `18817740` | emotion reappraisal | emotion | LookNeu; LookNeg; ReappNeg |
| `24603024` | emotion reappraisal | emotion | reappraise/low/negative; look/low/negative; reappraise/high/negative; look/high/negative |
| `27998996` | emotion reappraisal | emotion | Increase; Decrease; Look-Negative; Look-Neutral |
| `26231911` | emotion reappraisal and long-term reexposure | emotion | single-look neutral; single-look negative; single-reappraise negative; repeated-look negative (+2) |
| `20188516` | emotion reappraisal task | emotion | Negative; Reappraise; Neutral |
| `23144849` | emotion reappraisal task | emotion | Reappraise; Attend; negative; neutral |
| `25485181` | emotion reappraisal task | emotion | ReappraiseDistressing; LookDistressing |
| `26809287` | emotion reappraisal task | emotion | aversive-maintain; aversive-reappraise; neutral-maintain; neutral-reappraise |
| `31057439` | emotion reappraisal task | emotion | attend; reappraise |
| `31133889` | emotion reappraisal task | emotion | negative; positive; neutral; alcohol |
| `24715880` | fMRI affect labeling and reappraisal task | emotion | Observe; Label; Reappraise; Shape Match |
| `31394491` | fMRI reappraisal emotion regulation task | emotion |  |
| `23567923` | food cognitive reappraisal task | food | imagine eating; costs of eating; benefits of not eating; suppress craving |
| `24392892` | food craving reappraisal task | food | Look Neutral; Look Craved; Look Not Craved; Regulate Craved (+1) |
| `25536500` | food craving reappraisal task | food | Look Neutral; Look Craved; Look Not Craved; Regulate Craved (+1) |
| `25536500` | food craving reappraisal task | food | Look Neutral; Look Craved; Look Not Craved; Regulate Craved (+1) |
| `23567923` | food reappraisal task | food | imagine eating; costs of eating; benefits of not eating; suppress craving |
| `24392892` | food-craving cognitive reappraisal task | food |  |
| `20674090` | gaze-directed cognitive reappraisal | emotion | increase; view; decrease |
| `25451388` | image-based reappraisal task | emotion | Reappraise Cue; Look Cue; Look Negative; Look Neutral (+1) |
| `28192177` | implicit cognitive reappraisal task | (unspecified) | NNEG-DESC; NEG-DESC |
| `20226799` | reappraisal by distancing | emotion | distance; look; negative; neutral |
| `29653096` | reappraisal of negative pictures | emotion | negative reappraise; negative maintain |
| `19486944` | reappraisal task | emotion |  |
| `20147457` | reappraisal task | emotion | Neutral; Negative; Reappraise |
| `22468617` | Reappraisal Task | emotion | Look; Decrease; Neutral |
| `23202664` | reappraisal task | emotion | Look Neu; Look Neg; Reapp Neg |
| `23570916` | reappraisal task | emotion | negative/ real; negative/ look; negative/ photo |
| `24267410` | Reappraisal Task | emotion | Look Negative; Look Neutral; Regulate Negative |
| `25999363` | reappraisal task | emotion | Look Neutral; Look Negative; Reappraise Negative |
| `27341851` | reappraisal task | emotion | Negative; Neutral; Reactivity; Reappraisal |
| `27379614` | reappraisal task | emotion | immerse/aversive; immerse/neutral; distance/aversive |
| `27510495` | reappraisal task | emotion | Increase; Decrease; Observe-angry |
| `27626229` | reappraisal task | emotion | positive reappraisal; natural response; minimizing reappraisal |
| `28105532` | reappraisal task | social | IAPS1; IAPS2; CONTROL1; CONTROL2 |
| `28715907` | reappraisal task | emotion |  |
| `28715908` | Reappraisal Task | emotion |  |
| `29318489` | reappraisal task | emotion | LookNeu; LookNeg; Decrease |
| `22928000` | reward conditioning paradigm with cognitive reappraisal | money | Attend CS+; Regulate CS+; Attend CS−; Regulate CS− |
| `24804219` | self-related reappraisal | emotion | attend/positive; detach/positive; immerse/positive |
| `28809854` | worry induction and reappraisal task | (unspecified) | rest; worry induction; worry reappraisal |
| `24996397` | worry induction and worry reappraisal | (unspecified) | worry induction; worry reappraisal |

### 4. alcohol cue reactivity task

clustered · 59 studies · 64 tasks · 46 distinct names

Stimuli: alcohol (34), tobacco (16), cocaine (5), opioid (4), food (2), drug (1), cannabis (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `11296095` | alcohol and beverage cue-induction paradigm | alcohol | alcohol; beverage; visual; rest |
| `18391135` | alcohol cue presentation | alcohol | alcohol; beverage |
| `32636394` | alcohol cue reactivity | alcohol | alcohol; neutral |
| `14667419` | alcohol cue reactivity task | alcohol | alcohol words; neutral words |
| `21316465` | alcohol cue reactivity task | alcohol | alcohol; neutral beverage; blurred images; fixation |
| `26418276` | alcohol cue reactivity task | alcohol | alcohol; soft drink; oddball |
| `28409564` | alcohol cue reactivity task | alcohol | ALC; BEV; blurred images; fixation cross |
| `20571434` | alcohol cue-induced fMRI paradigm | alcohol | alcohol cues; beverage cues |
| `11296095` | alcohol cue-induction paradigm | alcohol | Alcohol; Beverage; Visual; Rest |
| `11296095` | alcohol cue-induction paradigm | alcohol | Alcohol; Beverage; Visual; Rest |
| `25697860` | alcohol cue-reactivity paradigm | alcohol | alcohol stimuli; neutral stimuli |
| `25937240` | alcohol cue-reactivity task | alcohol | alcohol; neutral |
| `26289945` | alcohol cue-reactivity task | alcohol |  |
| `29362512` | alcohol cue-reactivity task | alcohol | ALC; BEV; blurred images; fixation cross |
| `31362183` | alcohol cue-reactivity task | alcohol | alcohol; neutral; rest |
| `11296095` | alcohol-induction paradigm | alcohol | Alcohol; Beverage; Visual; Rest |
| `23571420` | alcohol-related visual and olfactory cue reactivity | alcohol | alcohol odor; control odor; alcohol pictures; control pictures |
| `28898485` | cigarette smoking cue fMRI paradigm | tobacco | cigarette smoking; neutral control; cross-hair rest |
| `23497788` | cocaine cue-reactivity fMRI paradigm | cocaine | cocaine; neutral |
| `23683790` | cocaine-cue reactivity paradigm | cocaine | cocaine images; neutral objects; visual control images; cross-hair |
| `22890475` | cue reactivity | tobacco | smoking paraphernalia; control objects |
| `24337077` | cue reactivity | tobacco | smoking-related stimuli; neutral stimuli |
| `31730369` | cue reactivity | cocaine | cocaine videos; food videos |
| `16133128` | cue reactivity paradigm | tobacco | smoking-related stimuli; neutral stimuli; fixation cross |
| `21185518` | cue reactivity paradigm | tobacco | smoking cues; neutral cues |
| `30096639` | cue reactivity paradigm | drug | Drug movies; Neutral movies |
| `30217552` | Cue reactivity paradigm | alcohol |  |
| `28373890` | Cue Reactivity Task | tobacco | control images; smoking-related images |
| `30711509` | cue-induced alcohol craving task | alcohol | Alcohol cues; Neutral cues |
| `31202048` | cue-induced alcohol craving task | alcohol | alcohol; neutral |
| `30295396` | cue-induced cocaine craving task | cocaine | cocaine picture; neutral picture |
| `31420667` | cue-induced cocaine craving task | cocaine | cocaine picture; neutral picture |
| `28881072` | cue-induced craving and craving regulation task | tobacco | regulation condition; craving condition; neutral condition |
| `23422198` | cue-induced craving task | opioid | heroin-related; neutral |
| `31420667` | cue-induced food craving task | food | food picture; neutral picture |
| `31376437` | cue-reactivity | (unspecified) | spicy; non-spicy; water |
| `20670348` | cue-reactivity fMRI task | alcohol | alcohol cues; neutral cues |
| `18266213` | cue-reactivity paradigm | opioid |  |
| `31204249` | cue-reactivity paradigm | cannabis | cannabis; neutral |
| `32546139` | cue-reactivity paradigm | alcohol | alcohol beverage; fixation |
| `31908107` | cue-reactivity task | alcohol | alcohol cues; neutral cues |
| `30268001` | fMRI alcohol cue-reactivity task | alcohol | alcohol-related; neutral |
| `30748046` | fMRI alcohol cue-reactivity task | alcohol | alcohol pictures; neutral pictures |
| `21292243` | fMRI cue reactivity | alcohol | alcohol; neutral |
| `25526597` | fMRI cue reactivity | alcohol | alcohol; soft drink |
| `21790907` | fMRI cue-reactivity experiment | alcohol | alcohol; neutral |
| `28877410` | fMRI cue-reactivity procedure | tobacco |  |
| `24314346` | fMRI cue-reactivity task | alcohol | alcohol; neutral |
| `23359677` | fMRI smoking cue reactivity task | tobacco | smoking video; neutral video |
| `30388597` | food-cue-reactivity task | food | HC; LC |
| `25781230` | heroin cues-induced craving procedure | opioid | drug cue; neutral |
| `26204262` | neurofeedback paradigm | alcohol | alcohol-related pictures; neutral pictures |
| `23683344` | nicotine cue-induced craving paradigm | tobacco | smoke; neutral; rest |
| `26505139` | smoking cue exposure and neurofeedback task | tobacco | smoking-related pictures; non–smoking related pictures; rest |
| `22458676` | smoking cue neurofeedback | tobacco | smoking related; neutral; REST |
| `21790899` | smoking cue reactivity | tobacco | Smoke; Neutral; Crave; Resist |
| `16133128` | smoking cue reactivity paradigm | tobacco | smoking-related stimuli; neutral stimuli; fixation cross |
| `26475784` | smoking cue-reactivity fMRI task | tobacco | CIG-I; NEU-I; PG-O; CIG-O |
| `24659022` | smoking cue-reactivity paradigm | tobacco | smoking cues; neutral cues |
| `30788529` | visual alcohol cue reactivity task | alcohol |  |
| `24400099` | visual alcohol cue-reactivity task | alcohol | alcohol; neutral |
| `29984874` | visual alcohol cue-reactivity task | alcohol | alcohol; neutral |
| `17307790` | visual cue exposure paradigm | alcohol | alcoholic beverage pictures; non-alcoholic beverage pictures |
| `18823721` | visual cue-exposure paradigm | opioid | heroin; pleasant; neutral; low-level baseline |

### 5. cue-reactivity task

clustered · 52 studies · 53 tasks · 42 distinct names

Stimuli: tobacco (24), food (11), alcohol (8), cannabis (6), cocaine (1), sexual (1), drug (1), social (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23078363` | Alcohol Cue Reactivity task | alcohol | alcohol images; non-alcohol beverage images; degraded stimuli |
| `28801730` | Alcohol Cue Reactivity Task | alcohol | alcoholic beverages; non-alcoholic beverage; active control conditions; fixation |
| `25796007` | Alcohol pictures cue reactivity task | alcohol | alcohol; non-alcohol |
| `25875013` | Alcohol-Emotion-Picture fMRI task paradigm | alcohol | emotional faces; alcoholic beverages; non-alcoholic beverages |
| `27005897` | appetitive smoking-related images and neutral pictures | food | Craving; Neutral |
| `26295336` | block-related cue-exposure | alcohol | neutral |
| `26727534` | block-related cue-exposure paradigm | alcohol | alcohol; neutral |
| `32761688` | cannabis-cigarette fMRI cue reactivity task | cannabis | cannabis; cigarette; neutral; animal |
| `23382517` | chocolate and neutral pictures | food | Chocolate; neutral |
| `17573781` | cue exposure task | tobacco | smoking; control |
| `20840335` | Cue Reactivity Task | tobacco | neutral pictures; gambling pictures; smoking-related pictures; low-level baseline pictures |
| `22591950` | cue reactivity task | cocaine | cocaine; neutral |
| `24789842` | cue reactivity task | food | food; people; nature scenes |
| `26645206` | cue-reactivity functional magnetic resonance imaging task | cannabis | cannabis; neutral |
| `22514316` | cue-reactivity paradigm | alcohol | animals; food; sexual scenes; people drinking alcohol (+2) |
| `25567427` | cue-reactivity task | tobacco | smoking; neutral; target; fixation |
| `28351544` | cue-reactivity task | cannabis | cannabis; neutral; positive; baseline |
| `28398588` | cue-reactivity task | tobacco | smoking; neutral |
| `29152692` | cue-reactivity task | drug | erotic; drug-related; aversive; neutral |
| `29880873` | cue-reactivity task | tobacco | smoking; neutral; animal |
| `32179061` | cue-reactivity task | tobacco | smoking; neutral |
| `17987060` | Event-Related Cue Task | tobacco | smoking cues; control cues; targets |
| `26295336` | event-related cue-exposure | alcohol | alcohol; neutral |
| `27168331` | fMRI cannabis cue exposure task | cannabis | cannabis; fruit; pencil |
| `28958900` | fMRI food-cue reactivity | food | Food; Object |
| `28964904` | fMRI food-cue reactivity task | food | food; objects |
| `27058281` | food cue reactivity task | food | food images; nature scenes |
| `28158874` | food cue-reactivity task | food | Food; Non-Food Control |
| `30466438` | food cue-reactivity task | food | food images; non-food images |
| `29421336` | food-cue reactivity | food | food images; object images |
| `29985099` | food-cue reactivity task | food | food images; control images |
| `23188041` | MJ cue reactivity task | tobacco | MJ cues; Non-MJ cues |
| `19907419` | smoking and control cue task | tobacco | smoking; control; fixation |
| `21764527` | smoking and neutral images | tobacco | smoking images; neutral images |
| `30806013` | smoking cue reactivity task | tobacco | CUE; neutral; target stimuli |
| `28711813` | smoking cue still-image task | tobacco | SC; non-SC |
| `31039580` | smoking cue-reactivity task | tobacco | smoking; neutral; fixation |
| `31706906` | smoking cue-reactivity task | tobacco | smoking images; neutral images; target images |
| `32145666` | smoking cue-reactivity task | tobacco | smoking; neutral; target |
| `32900658` | smoking cue-reactivity task | tobacco | smoking; nonsmoking; target |
| `29106683` | Smoking Cues Task | tobacco | Smoking; Neutral |
| `17611740` | smoking, control, and target cues | tobacco | smoke; control; target |
| `19015835` | smoking-related and neutral image task | tobacco | smoking-related images; neutral images |
| `20172508` | smoking-related and neutral images | tobacco | smoking-related images; neutral images |
| `16023086` | smoking-related and neutral pictorial cues | tobacco | smoking-related; neutral |
| `19968401` | smoking-related or neutral images | tobacco | Smoking-Related; Neutral |
| `20703221` | smoking-related or neutral images | tobacco | smoking-related images; neutral images |
| `30278517` | smoking-related visual cue task | tobacco | S; SD + NP; SD + PP; SD |
| `31498505` | social-alcohol cue-exposure task | social | social alcohol; social soda; non-social alcohol; non-social soda (+1) |
| `25013940` | video cue reactivity | sexual | explicit sexual; erotic; non-sexual exciting; money (+1) |
| `25528694` | visual chocolate and neutral stimuli | food | chocolate pictures; neutral pictures; question blocks |
| `30243035` | visual fMRI cannabis cue reactivity task | cannabis | cannabis; non-cannabis |
| `31132681` | visual fMRI cannabis cue-reactivity task | cannabis | cannabis images; non-cannabis images |

### 6. cue reactivity task

clustered · 44 studies · 49 tasks · 39 distinct names

Stimuli: alcohol (19), tobacco (10), (unspecified) (6), emotion (4), cocaine (2), food (2), drug (2), opioid (1), stimulant (1), money (1), social (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22926600` | alcohol cue reactivity paradigm | alcohol | alcohol; neutral; abstract |
| `23032071` | alcohol cue reactivity task | alcohol |  |
| `23389755` | alcohol cue reactivity task | alcohol | ALC; BEV; REST |
| `24647921` | alcohol cue reactivity task | alcohol | ALC; BEV; fixation cross |
| `23325372` | alcohol cue stimulation | alcohol | alcohol drinking scene; mosaic control scene |
| `12381499` | alcohol-associated and abstract picture presentation | alcohol | alcohol-associated stimuli; abstract pictures |
| `15127179` | alcohol-associated and control stimuli | alcohol | alcohol-associated; affectively neutral; abstract; fixation |
| `16899037` | alcohol-associated and control stimuli | alcohol | alcohol-associated; abstract; affectively neutral; fixation |
| `22868938` | alcohol-associated and neutral stimuli | alcohol | CUE alc; IAPS neutral; fixation condition |
| `23921439` | alcohol-related and neutral video sequences | alcohol | alcohol-related; neutral |
| `16237382` | cocaine cue audiovisual presentation | cocaine | neutral; cocaine-related |
| `16133128` | cue reactivity paradigm | tobacco | smoking-related stimuli; neutral stimuli |
| `25035299` | cue reactivity paradigm | tobacco | tobacco; money; neutral; rest |
| `25094019` | cue reactivity task | alcohol |  |
| `25094019` | cue reactivity task | alcohol |  |
| `25094019` | cue reactivity task | alcohol |  |
| `25094019` | cue reactivity task | alcohol |  |
| `27306727` | cue reactivity task | alcohol | alcohol images; pleasant images |
| `28521241` | cue reactivity task | (unspecified) | craving-inducing cue; aversion-inducing cue |
| `24939441` | cue reactivity tasks | alcohol | Craving-inducing cues; Aversion-inducing cues |
| `29343732` | cue-reactivity paradigm | alcohol | alcohol condition; beverage condition; control condition; fix condition |
| `25593047` | cue-reactivity task | alcohol | alcohol-associated stimuli; affectively neutral stimuli; neutral abstract stimuli; fixation |
| `31995187` | cue-reactivity task | stimulant | methamphetamine; sexual; neutral |
| `29372058` | Directed Rumination Task | (unspecified) | provocation-focus; self-focused; neutral-focus |
| `30194288` | drug-cue reactivity task | drug | alcohol cues; cigarette cues; cocaine cues; neutral cues |
| `29776789` | drug/alcohol cue reactivity task | drug | drug/alcohol; neutral; blur; rest |
| `23827769` | fearful and neutral face viewing | emotion | fearful; neutral; rest |
| `24241476` | feedback task | emotion | regulate; view; rest |
| `29410011` | fMRI cue task | cocaine |  |
| `30909426` | food cue viewing | food |  |
| `24637623` | heroin-related and neutral picture cue exposure | opioid | heroin-related stimuli; neutral stimuli |
| `26769333` | marketing-exposure task | alcohol | alcohol marketing clips; cannabis-related clips; neutral clips |
| `31647945` | mental simulation task | tobacco | smoking cue; neutral cue; visual fixation |
| `23153997` | methamphetamine-cue, happy, sad, and neutral stimuli | emotion | methamphetamine-cue; sad; happy; neutral (+1) |
| `26649946` | mother-child video stimuli | (unspecified) | separation; play |
| `6rjSnrz4ioTM` | music listening reward responsiveness paradigm | money | preferred music; neutral music; silence |
| `bManksWzpSBt` | preferred classical music | (unspecified) | classical music; noise |
| `29372058` | Provocation Task | (unspecified) | feedback; pre-feedback baseline |
| `24894701` | smoking cue exposure | tobacco | smoking cue; neutral cue |
| `16133128` | smoking cues and neutral stimuli | tobacco | smoking-related stimuli; neutral stimuli |
| `22234380` | smoking or neutral images | tobacco | smoking; neutral; fixation |
| `23146252` | smoking-related and emotional images | tobacco | Smoking; Negative; Positive; Neutral |
| `20604987` | smoking-related and neutral photographic cues | tobacco | smoking cues; neutral cues; rest |
| `20688176` | smoking-related and neutral picture cue paradigm | tobacco | smoking; neutral |
| `22542509` | smoking-related and neutral visual cues | tobacco | smoking-related cues; neutral cues; fixation |
| `23473935` | social contact, attachment, and social regulation of emotion | social | threat; safety; rest |
| `27542906` | visual food and nonfood cue task | food | food; nonfood; fixation cross |
| `28284776` | visual stimulation | (unspecified) | neutral; erotic |
| `30065778` | visual stimuli paradigm | emotion | neutral state; negative state; reappraisal state |

### 7. Taste Cue Paradigm

clustered · 36 studies · 44 tasks · 31 distinct names

Stimuli: tobacco (16), food (8), alcohol (6), cocaine (4), cannabis (4), stimulant (2), drug (2), emotion (1), money (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24411804` | alcohol taste cues task | alcohol | Alcohol cue; water cue |
| `23876228` | alcohol taste-cue paradigm | alcohol | Alcohol cue; Water cue |
| `24880692` | alcohol taste-cue paradigm | alcohol | Alcohol Cue; Control Taste |
| `26125586` | alcohol taste-cue paradigm | alcohol | alcohol; water |
| `32432821` | cannabis cue exposure task | cannabis | neutral cue ON; neutral cue OFF; natural reward cue ON; natural reward cue OFF (+2) |
| `24838032` | cannabis cue-exposure task | cannabis | cannabis cue ON; neutral cue ON |
| `32113057` | cannabis cue-exposure task | cannabis | cannabis cue; appetitive cue; neutral cue |
| `17217932` | cigarette cue exposure | tobacco | cigarette cue crave; cigarette cue resist; neutral cue |
| `17217932` | cigarette cue exposure | tobacco | cigarette cue crave; cigarette cue resist; neutral cue |
| `26679479` | cigarette cue reactivity task | tobacco | smoking cues; neutral cues |
| `24880692` | cigarette cues task | tobacco | Cigarette Cue; Control Cue |
| `24949564` | cigarette cues task | tobacco | Cigarette cues; Neutral cues |
| `21199957` | cigarette-related and neutral cue videos | tobacco | neutral; crave-allow; crave-resist |
| `16123763` | cocaine cue exposure | cocaine | cocaine cue; baseline |
| `16085533` | cue exposure | tobacco | cigarette; neutral |
| `15920499` | Cue Exposure Task | tobacco | smoking cues; control cues |
| `22483100` | cue exposure task | tobacco | cigarette cue; control cue |
| `26038158` | cue exposure task | cocaine | cocaine cues; neutral cues |
| `26975550` | cue exposure task | tobacco | notepad; electrical tape; cigarette |
| `21223301` | cue exposure with client statements and alcohol cues | alcohol | CT; CCT |
| `25142207` | cue video stimulation task | cocaine | cocaine cues; food cues; neutral cues; control |
| `28940758` | drug cue or neutral cue video | drug | drug cue; neutral cue |
| `29130147` | drug-cue exposure task | drug | drug-related pictures; neutral pictures; null fixation stimuli |
| `22580204` | food cue exposure | food | WATCH Food; WATCH Neutral; DECREASE Food; INCREASE Food |
| `29486871` | gustatory alcohol cue reactivity task | alcohol | alcohol cue; juice cue |
| `20010552` | marijuana cue | cannabis | marijuana cue; control cue |
| `27312405` | Methamphetamine Cues Task | stimulant | methamphetamine cue; control cue |
| `31014470` | Methamphetamine Cues Task | stimulant | Methamphetamine Cue; Control Cue |
| `22960252` | personally relevant high-calorie taste cue exposure task | food | high-calorie taste cues; water |
| `30646331` | reward cue task | money | drug cues; food cues; neutral cues |
| `26303184` | smoking cue and control videos | tobacco | smoking cue videos; control videos; fixation cross |
| `26303184` | smoking cue and control videos | tobacco | smoking cue; control videos; fixation cross |
| `28653791` | Smoking Cue Exposure Task | tobacco | smoking cue; neutral cue |
| `23421569` | smoking cue paradigm | tobacco | smoking cues; neutral cues; rest |
| `18540916` | Taste Cue Paradigm | food |  |
| `18540916` | Taste Cue Paradigm | food | alcohol taste; litchi juice |
| `18540916` | Taste Cue Paradigm | food | alcohol taste; litchi juice |
| `18540916` | Taste Cue Paradigm | food | alcohol taste; control taste |
| `20028366` | taste-cue paradigm | food | alcohol taste cue; appetitive control taste cue; rest |
| `26119472` | video craving task | cocaine | cocaine; gambling; sad |
| `17217932` | videotaped cigarette and neutral cue exposure | tobacco | cigarette cue crave; cigarette cue resist; neutral cue |
| `17217932` | videotaped cigarette and neutral cue exposure | tobacco | cigarette cue crave; cigarette cue resist; neutral cue |
| `11136638` | videotapes designed to elicit happy feelings, sad feelings, or the desire to use cocaine | emotion | cocaine-cue tapes; happy tapes; sad tapes |
| `28698012` | visual food and neutral cue paradigm | food | food cues; neutral cues |

### 8. modified Sternberg working memory paradigm

clustered · 36 studies · 42 tasks · 38 distinct names

Stimuli: emotion (23), (unspecified) (6), letters (4), tobacco (2), cocaine (1), cannabis (1), money (1), alcohol (1), faces (1), social (1), words (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `4UoCgF3UJSXq` | affect-laden pictures | emotion | negative; positive; neutral; fixation crosses |
| `23035965` | affective two-back working memory task | emotion | angry faces; neutral faces |
| `o8WEzV9Uy5GB` | attentional bias dot probe task | emotion | angry; happy; neutral; congruent (+1) |
| `22445780` | Cued Emotional Conflict Task | emotion | actual; opposite |
| `19538748` | declarative memory task | words | encoding; retrieval; fixation |
| `24261848` | delay-match-to-sample task | tobacco | smoking; neutral; encoding; maintenance (+1) |
| `26619965` | delayed WM task under negative emotional distraction | emotion | negative emotional; neutral; scrambled |
| `28262423` | dynamic affect labeling task | emotion | View; Blur; Label; Fixation |
| `29569801` | emotional attention task | emotion | attending emotional stimuli; ignoring emotional stimuli; negative; positive |
| `6W8nYjEeZEcL` | emotional categorisation task | emotion | Emotional categorisation |
| `28462086` | Emotional Faces Interference Task | emotion | Threat distractors Low load; Threat distractors High load |
| `29428771` | Emotional Faces Interference Task | emotion | Threat Low; Threat High |
| `rxaz3qhEmJhx` | emotional fear processing task | emotion | neutral faces; mildly fearful faces; intensely fearful faces |
| `rxaz3qhEmJhx` | emotional fear processing task | emotion | neutral faces; mildly fearful faces; intensely fearful faces |
| `rxaz3qhEmJhx` | Emotional fear-processing task | emotion | Intensely fearful faces; Mildly fearful faces; Neutral faces; Fixation cross |
| `27318594` | emotional interference conflict task | emotion | AF; IF |
| `27998997` | emotional interference word task | emotion | pleasant words; neutral words; unpleasant words |
| `23590840` | emotional working memory paradigm | emotion | fearful; happy; neutral |
| `24179757` | emotional-interference task | emotion | attend fearful faces; attend neutral faces; ignore fearful faces; ignore neutral faces |
| `25502775` | face encoding | faces | angry; neutral |
| `28224080` | feature-based comparison task | social | disorder-related; neutral |
| `19652138` | flanker task | (unspecified) | congruent; incongruent |
| `19747928` | Flanker task | (unspecified) | congruent; incongruent |
| `32285159` | hybrid imaging task | alcohol | alcohol; neutral; scramble; congruent (+1) |
| `jK4E25LJYX8L` | incidental emotional recognition task | emotion | positive/hit; positive/correct rejection; negative/hit; negative/correct rejection |
| `6W8nYjEeZEcL` | incidental recognition task | (unspecified) | Incidental memory |
| `30885230` | magnitude comparison task | (unspecified) | symbolic; non-symbolic |
| `24754423` | MID task | (unspecified) | gain; loss; neutral |
| `YwwKWoEFwY3G` | modified Sternberg working memory paradigm | letters | encoding; maintenance; retrieval |
| `YwwKWoEFwY3G` | modified Sternberg working memory paradigm | letters | encoding; maintenance; retrieval |
| `YwwKWoEFwY3G` | modified Sternberg working memory paradigm | letters | encoding; maintenance; retrieval |
| `27012714` | probed recall task | tobacco | control; neutral cue; smoking cue |
| `20729527` | repetition priming paradigm | (unspecified) | studied; non-studied |
| `30143454` | Reward Incentive Delay With Shock Task | money | alcohol; food; neutral |
| `20667170` | sad facial affect discrimination task | emotion | sad facial expression; neutral facial expression; fixation cross |
| `23348009` | Shifted-Attention Emotion Appraisal Task (SEAT) | emotion | Gender; Inside/Outside; Like/Dislike |
| `21183158` | spatial cueing paradigm | emotion | happy; neutral; fearful |
| `YwwKWoEFwY3G` | Sternberg WM task | letters | encoding; maintenance; retrieval |
| `22957019` | Stimulus Response Compatibility task | cannabis | approach block; avoid block; baseline block |
| `22592057` | top-down attention control | emotion | View; Congruent; Incongruent |
| `16930447` | working memory and emotional-picture paradigm | emotion | negative 0-back anticipation; neutral 0-back anticipation; negative 2-back anticipation; neutral 2-back anticipation |
| `19135471` | working memory recall task | cocaine | low load/cocaine; high load/cocaine; low load/neutral; high load/neutral |

### 9. visual food cues

clustered · 35 studies · 42 tasks · 21 distinct names

Stimuli: food (37), tobacco (3), emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23261871` | auditory food cues | food | auditory high-ED; auditory low-ED; auditory non-food |
| `31234864` | emotional visual stimuli task | emotion | unpleasant; neutral; pleasant |
| `18413289` | food and car image task | food | high-calorie foods; low-calorie foods; cars; high-calorie sweet foods (+1) |
| `22406414` | food cue paradigm | food | high-calorie foods; low-calorie foods; neutral non-foods |
| `22921709` | food cue paradigm | food | high-calorie food cues; low-calorie food cues |
| `21169809` | food cues | food | high-ED; low-ED; neutral nonfoods |
| `28017966` | food cues | food |  |
| `28715141` | food cues | food | highly desirable food cues; less desirable food cues; non-food images |
| `30019382` | food cues | food | large-portion High-ED; small-portion High-ED; large-portion Low-ED; small-portion Low-ED |
| `19365394` | food photographs | food | non-fattening food; object |
| `22576622` | food-related visual cues | food | high-calorie foods; low-calorie foods; brick walls |
| `18568078` | Passive food and non-food cue viewing | food |  |
| `28929362` | pictures of high- and low-calorie foods and blurred control pictures | food | high-calorie; low-calorie; blurred control pictures |
| `24529072` | smoking and neutral cues | tobacco | smoke; neutral |
| `23455593` | smoking-related visual cues | tobacco | smoking-related; neutral |
| `12042183` | visual cue task | tobacco | smoking-related images; neutral images; target images |
| `20096712` | visual food and non-food image presentation | food | hedonic foods; non-food objects; high hedonic value foods; neutral hedonic value foods |
| `26283736` | visual food cue task | food | high-calorie food items; low-calorie food items; nonfood items |
| `32541652` | visual food cue task | food |  |
| `17921372` | visual food cues | food | H; U; O |
| `18568078` | visual food cues | food | food; non-food |
| `18568078` | visual food cues | food | food; non-food |
| `19636426` | visual food cues | food | foods of high hedonic value; neutral nonfood objects; foods of neutral hedonic or utilitarian value |
| `22155218` | visual food cues | food | hedonic foods; nonfood objects; fixation cross |
| `22647305` | visual food cues | food | fattening food; non-fattening food; non-food objects |
| `23261871` | visual food cues | food | visual high-ED; visual low-ED; visual non-food |
| `23313402` | visual food cues | food | hedonic foods; neutral nonfood objects |
| `24583185` | visual food cues | food | HC; LC; C |
| `26739033` | visual food cues | food | fattening; nonfattening |
| `28655900` | visual food cues | food | high-calorie foods; low-calorie foods; fixation |
| `28911135` | visual food cues | food | high-calorie sweet foods; high-calorie savory foods; nonfood objects |
| `29239138` | visual food cues | food | hedonic foods; neutral foods; non-food objects |
| `30445718` | visual food cues | food | high-calorie food images; low-calorie food images; neutral images |
| `31174338` | visual food cues | food | high-calorie foods; low-calorie foods; fixation cross |
| `31701698` | visual food cues | food | high-calorie food; low-calorie food; nonfood; baseline block |
| `20920491` | visual food pictures | food | HC; LC; control |
| `28929362` | visual food stimuli | food | high-calorie; low-calorie; blurred control pictures |
| `28929362` | visual food stimuli task | food | high-calorie; low-calorie; blurred control pictures |
| `28929362` | visual food stimuli task | food | high-calorie foods; low-calorie foods; blurred control pictures |
| `18568078` | visual food versus non-food cues | food | food; non-food |
| `29025878` | Visual food-cue task | food |  |
| `24465445` | visual stimuli | emotion | PEN; BOD |

### 10. go/no-go task

named by the Cognitive Atlas · 34 studies · 35 tasks · 30 distinct names

Stimuli: emotion (9), letters (6), (unspecified) (5), food (5), alcohol (3), social (2), words (2), sexual (1), faces (1), money (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `31069895` | Affect-congruent Go/NoGo task | food |  |
| `28161772` | alcohol go/no-go task | alcohol | alcoholic beverage no-go; non-alcoholic beverage go; geometric no-go; geometric go |
| `25172182` | alcohol Go/NoGo task | alcohol | Go trials; NoGo trials; false alarms |
| `31506678` | control go/no-go task | letters |  |
| `23046830` | emotion induction and go/no-go task | emotion | anger; joy; neutral |
| `24145410` | emotional face go/no-go task | emotion | go condition; no-go condition |
| `18452757` | emotional go-nogo task | emotion | fearful expressions; happy expressions; calm expressions |
| `20419567` | emotional go/no-go task | emotion | Winning; Losing; Recovery |
| `20971474` | emotional go/no-go task | emotion | Winning; Losing; Recovery |
| `23046115` | emotional go/no-go task | emotion | angry; calm; happy |
| `15780849` | emotional go/nogo task | emotion | target trials; nontarget trials; negative emotional context; positive and negative emotional context (+2) |
| `26483645` | emotional Go/NoGo task | emotion | neutral Go; neutral NoGo; aversive Go; aversive NoGo |
| `30649528` | food inhibition go/no-go task | food |  |
| `27575974` | food-related go/nogo task | food | HGo; LGo; HNogo; LNogo |
| `29944963` | food-specific go/no-go task | food | dessert; vegetable |
| `25228353` | food-specific go/nogo tasks | food | Go; Nogo; High-calorie; Low-calorie |
| `26784537` | frustration-induction Go-NoGo task | (unspecified) | recovery; winning |
| `23683790` | go no-go response inhibition task | letters | go no-go blocks; fixation blocks |
| `30082140` | Go-No-Go task | words |  |
| `16340649` | Go-NoGo response inhibition | (unspecified) |  |
| `26001387` | Go-NoGo task | letters | Go; NoGo |
| `SdvwpgEnZm4u` | Go-NoGo task | (unspecified) |  |
| `21440645` | go/no-go inhibitory task | letters |  |
| `24918068` | Go/No-go monetary reward paradigm | money | Go; No-go |
| `23306064` | go/no-go task | faces | go; no-go |
| `23395930` | go/no-go task | alcohol | respond alcohol; respond neutral |
| `24789842` | go/no-go task | social | go; no-go |
| `24961260` | Go/No-go task | words | Go; No-go |
| `27575491` | Go/No-Go task | (unspecified) |  |
| `23020994` | Go/No-Go task with visual negative feedback | emotion |  |
| `mQHuh2mRPfdY` | Go/NoGo task | (unspecified) | Go; NoGo |
| `32359232` | modified go/no-go task | sexual |  |
| `30145507` | Parametric Go/No-Go | letters | Targets; Rejections; Commissions |
| `7EEMMXSEGr7p` | Parametric Go/No-go task | letters | hits; rejections; commissions |
| `31506678` | social go/no-go task | social | socially appetitive; socially aversive |

### 11. monetary incentive delay task

named by the Cognitive Atlas · 30 studies · 31 tasks · 17 distinct names

Stimuli: money (27), alcohol (3), social (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30444488` | adapted monetary incentive delay task | money | erotic pictures; monetary rewards |
| `26209857` | Alcohol-Food Incentive Delay task | alcohol | alcohol; food; neutral |
| `30423017` | Beer Incentive Delay task | alcohol | beer; water |
| `31066137` | beer incentive delay task | alcohol | beer; water |
| `27056455` | food and monetary incentive delay task | money |  |
| `24837478` | incentive delay task | money | monetary; erotic; control |
| `27845255` | Incentive delay task | money |  |
| `28409565` | incentive delay task | money |  |
| `21459835` | mixed monetary incentive delay/memory task | money | permit; regulate |
| `18672069` | modified monetary incentive delay task | money | reward outcomes; loss outcomes; Again notifications |
| `17291784` | Monetary incentive delay (MID) task | money |  |
| `22281932` | Monetary Incentive Delay (MID) task | money | $1, $10 reward cues; $0 no incentive cues; WIN; HIT (+1) |
| `24793365` | monetary incentive delay (MID) task | money | win; loss; neutral comparison event |
| `18851716` | monetary incentive delay task | money | reward; non-reward |
| `19442745` | monetary incentive delay task | money | permit; distance |
| `19560123` | monetary incentive delay task | money | gain anticipation; loss anticipation; neutral |
| `21704307` | Monetary Incentive Delay Task | money |  |
| `21926423` | monetary incentive delay task | money | anticipation of gain; anticipation of loss; neutral |
| `21955931` | monetary incentive delay task | money | anticipation large reward; baseline |
| `29059451` | Monetary Incentive Delay Task | money |  |
| `29951769` | monetary incentive delay task | money | gain anticipation; loss anticipation |
| `31342097` | monetary incentive delay task | money | gain anticipation; loss anticipation |
| `31931509` | monetary incentive delay task | money | reward anticipation; reward outcome |
| `32150321` | Monetary Incentive Delay task | money | premature money incentive; correct money incentive; premature drug incentive; correct drug incentive |
| `4NDaGF2J8dwU` | monetary incentive delay task | money | +¥0; +¥20; +¥100; +¥500 (+5) |
| `6EBJdaTdYBPp` | monetary incentive delay task | money | rewarding; nonrewarding; low-level fixation |
| `8D4im3bFixCF` | monetary incentive delay task | money | gain; loss; neutral |
| `VhsCe36Z2K42` | Monetary Incentive Delay task | money | anticipation reward; neutral; anticipation loss; consumption reward (+1) |
| `VhsCe36Z2K42` | Monetary Incentive Delay task | money | anticipation reward; neutral |
| `25742873` | revised monetary incentive delay task | money | gain; loss; neutral; rest (null) |
| `23128082` | social incentive delay task | social |  |

### 12. cue-exposure task

clustered · 30 studies · 33 tasks · 31 distinct names

Stimuli: emotion (9), food (8), tobacco (4), alcohol (3), (unspecified) (3), cocaine (2), drug (2), sexual (1), cannabis (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21106812` | active regulation | emotion | negative; neutral; regulation; no regulation |
| `21949675` | active regulation | emotion | negative; neutral; regulation; no regulation |
| `23836764` | Advertisement fMRI paradigm | food | all Coke ads; Coke product ads; Coke logo ads; non-food ads |
| `18606956` | alcohol versus control taste-cue paradigm | alcohol | alcohol taste cue; control taste cue; rest |
| `29464814` | alcohol versus juice taste task | alcohol | alcohol cue; juice cue; REST |
| `25409596` | alcoholic and neutral beverages | alcohol | alcoholic beverages; neutral beverages |
| `26237321` | backward-masked cannabis cue task | cannabis | cannabis cues; neutral cues |
| `26094857` | chocolate milk fMRI paradigm | food | chocolate milk receipt; tasteless solution receipt; anticipation of chocolate milk receipt; anticipation of tasteless solution receipt |
| `23836764` | Coke intake fMRI paradigm | (unspecified) | Coke intake; tasteless solution intake |
| `26478134` | cue videos | tobacco | e-cigarette; neutral |
| `27489006` | cue-elicited anticipation and receipt of palatable food | food | milkshake cue; tasteless-solution cue; milkshake receipt; tasteless-solution receipt |
| `23708507` | cue-exposure task | tobacco | Smoking; Food; fixation cross |
| `28231626` | cue-exposure task | food | smoke cues; food cues |
| `24720731` | drug cue-related task | drug | heroin-related; neutral |
| `20630713` | emotion expectation and picture presentation task | emotion | positive; negative; neutral |
| `30802854` | emotion processing and regulation tasks | emotion | NEU; NEG; NER |
| `31884222` | emotional picture processing task | emotion | NegExp; PosExp; NeutrExp; NegExpPic (+2) |
| `w9jwwHkfYFQG` | emotional stimuli | emotion | positive; negative; neutral; unknown |
| `22359374` | empathizing | (unspecified) |  |
| `27654662` | fast event-related fMRI task | cocaine | cocaine; sexual; aversive; neutral |
| `24695721` | fMRI backward-masked cue task | cocaine | cocaine; neutral; sexual; aversive |
| `29947607` | fMRI cue-reactivity task | drug | drug; sexual; aversive; neutral |
| `25447334` | food picture and milkshake fMRI paradigms | food | high-fat/high-sugar food; glasses of water; milkshake cue; tasteless solution cue |
| `29899546` | food/nonfood picture task | food | food; nonfood |
| `22359374` | imitation | (unspecified) |  |
| `23836764` | Milkshake intake fMRI paradigm | food |  |
| `28486715` | picture viewing paradigm | tobacco |  |
| `22097928` | picture-viewing task | emotion | erotic pictures; romantic pictures; mutilation pictures; sad pictures (+2) |
| `24376278` | Picture-viewing task | emotion |  |
| `4VAqiDPTPcJL` | positive social stimuli picture-viewing paradigm | emotion | social interaction; no social interaction; faces; no faces (+3) |
| `26145276` | romantic versus neutral cues | sexual | romantic cues; neutral cues; Fasted; Fed |
| `28960762` | smoking cue exposure task | tobacco | smoking cues; neutral cues |
| `28235080` | taste and visual food-cue paradigm | food | milkshake picture; water picture; milkshake taste; tasteless taste |

### 13. Stroop task

named by the Cognitive Atlas · 26 studies · 29 tasks · 20 distinct names

Stimuli: emotion (11), words (6), drug (4), cocaine (3), tobacco (1), food (1), opioid (1), alcohol (1), social (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `31715379` | addiction-Stroop color match-to-sample task | alcohol | alcohol; cannabis; color |
| `26640766` | affective Stroop paradigm | emotion | congruent; incongruent; view |
| `29935441` | affective stroop task | emotion | negative; positive; neutral; congruent (+2) |
| `28960762` | Chinese version color-word Stroop task | words |  |
| `26164485` | classical color-word Stroop task | words | congruent; incongruent |
| `23303067` | cocaine Stroop task | cocaine | cocaine words; neutral words; letter strings |
| `23809860` | cocaine-word Stroop task | cocaine | cocaine words; neutral words |
| `29108734` | cocaine-word Stroop task | cocaine | CW; NW |
| `24258223` | color-word Stroop task | words | error; correct |
| `30937347` | color-word Stroop task | words | congruent; incongruent |
| `20703221` | computerized smoking emotional Stroop task | tobacco |  |
| `17197102` | drug Stroop fMRI task | drug | drug; neutral |
| `17197102` | drug Stroop fMRI task | drug | drug; neutral |
| `17197102` | drug Stroop fMRI task | drug | drug; neutral |
| `17197102` | drug Stroop fMRI task | drug | drug; neutral |
| `21098213` | emotion-word Stroop task | emotion | pleasant words; neutral words; unpleasant words |
| `26996601` | Emotional conflict Stroop task | emotion |  |
| `31414234` | emotional face-word Stroop task | emotion | congruent; incongruent |
| `20172508` | emotional Stroop task | emotion |  |
| `28983519` | emotional Stroop task | emotion | incongruent; congruent |
| `29489856` | emotional Stroop task | emotion | threat-related scene; neutral scene; threat-related face; neutral face |
| `JMyuJvbhjJeF` | emotional Stroop task | emotion | incongruent; congruent |
| `31439409` | opioid-word Stroop task | opioid | OW; NW |
| `27815773` | Priming emotion and alcohol Stroop Match-to-Sample task | emotion | alcohol; positive emotion; negative emotion; congruent |
| `27457415` | social rejection-themed emotional Stroop task | social | Rejection; Neutral; Incongruent colour words |
| `26430731` | Stroop Match-to-Sample Task | emotion | emotional interference; without emotional interference |
| `24646809` | Stroop task | words | congruent; incongruent; cannabis-related words; neutral words |
| `27845255` | Stroop task | food |  |
| `26244883` | word-face emotional Stroop task | words | LR; HR |

### 14. emotional faces task

clustered · 24 studies · 25 tasks · 20 distinct names

Stimuli: emotion (20), faces (3), money (1), food (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27349456` | Emotion Processing Task | emotion | positive; neutral; negative |
| `30488228` | emotion processing task | emotion | face matching; shape matching |
| `31156078` | emotion processing task | emotion | emotion; shape |
| `24064198` | emotional face processing task | emotion | emotional faces; geometric shapes |
| `28412558` | emotional faces processing paradigm | emotion | Fixation; Neutral; Angry; Fearful (+1) |
| `22056201` | emotional faces task | emotion | emotion labeling; emotion matching; gender labeling; gender matching (+1) |
| `24019460` | emotional faces task | emotion | fear run; happy run |
| `qvmuEkgxgpBj` | emotional faces task | emotion | affective condition; neutral condition |
| `28992272` | emotional processing task | emotion | positive adjectives; negative adjectives |
| `qicgkAQBpirv` | emotional processing task | emotion | emotion; gender |
| `5hpdMbxy4acB` | emotional task | emotion | positive pictures; neutral pictures; negative pictures |
| `7LcT9Tieaogu` | emotional words task | emotion | negative; neutral; positive |
| `21799066` | faces and shapes affective reactivity task | faces | faces; shapes |
| `23469861` | faces–shapes task | faces | faces; shapes |
| `29997532` | fearful face task | emotion | block 1; block 2; block 3 |
| `22115148` | fearful-face task | emotion | fearful; neutral |
| `28451921` | food decision task | food | food; activities; pixels |
| `30172004` | Gender identification task | emotion | fearful; happy; sad; neutral (+1) |
| `22115148` | happy-face task | emotion | happy; neutral |
| `25151338` | implicit emotion processing task | emotion | emotion; shapes |
| `29339309` | implicit emotion processing task | emotion | fearful faces; happy faces; fixation cross |
| `24973815` | implicit emotion-processing task | emotion |  |
| `23031250` | implicit facial expression task | faces | angry; disgusted; fearful; happy (+2) |
| `8C2kFBVv54HZ` | matching of fearful and happy facial expressions | emotion | fear faces; happy faces; sensorimotor control task |
| `26809268` | monetary reward task | money | 0¢; 50¢ |

### 15. facial expression processing

clustered · 22 studies · 24 tasks · 19 distinct names

Stimuli: emotion (10), faces (8), money (2), words (2), (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24737710` | affective variant of the flanker task | emotion | sad faces; neutral faces |
| `29622050` | appraisal task | emotion |  |
| `3CkFS4xCjeHS` | classic face/emotion perception task | faces | fearful; neutral; happy |
| `22910460` | cognitive–emotional inhibition task | emotion | emotional trials; geometrical trials |
| `28715907` | emotional reactivity task | emotion |  |
| `28715908` | Emotional Reactivity Task | emotion | fearful faces; neutral faces |
| `PoiCaeLDyEjZ` | emotional reactivity task | emotion | emotional faces; shapes |
| `3jx9yRZ5ZZzs` | emotional self-referential task | emotion | self; general |
| `18544183` | facial expression processing | faces | negative emotional faces; baseline |
| `87nvVEvLSV5a` | facial expression processing | faces | fearful; happy; neutral |
| `87nvVEvLSV5a` | facial expression processing | faces | fearful; happy; neutral |
| `30979647` | fearful versus calm faces | emotion | fearful faces; calm faces |
| `7pCAyjTQEwTn` | future-thinking task | (unspecified) | distant future; near future; distant past; near past |
| `z9dTsmCi9EDs` | implicit facial-affect processing task | faces | happy; sad; angry; neutral |
| `8MN9yzzr3c3e` | masked fearful and happy facial expressions | emotion | masked fearful facial expressions; masked happy facial expressions; fixation |
| `18930182` | perceptual face processing task | faces | faces; shapes |
| `24342923` | rewarded guessing task | money | smoking reward; monetary reward; neutral |
| `26660448` | rewarded guessing task | money | monetary reward; smoking reward; nothing |
| `7E7F3tdRYjPH` | self-referential processing task | (unspecified) | self; general; control |
| `4Ghm5oksA2Lu` | self-referential word task | words | positive and negative words; neutral words |
| `R3nxPHDjQzbS` | self-reflection task | words | self-condition; word-condition |
| `yuiTej75RcY7` | valence decision task | emotion | neutral; positive; negative |
| `VhsCe36Z2K42` | Wall-of-Faces task | faces | affect; gender; ambiguous affect; ambiguous gender (+2) |
| `VhsCe36Z2K42` | Wall-of-Faces task | faces | affect; gender; ambiguous affect; ambiguous gender (+2) |

### 16. n-back task

named by the Cognitive Atlas · 17 studies · 17 tasks · 14 distinct names

Stimuli: (unspecified) (6), letters (6), emotion (5)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24622915` | 15-minute visual N-back WM task | (unspecified) | 0-back; 1-back; 2-back; 3-back |
| `27625988` | Curb Your Addiction (C-Ya) modified N-back working-memory task | letters |  |
| `22099606` | emotional face n-back (EFNBACK) | emotion | 2-back neutral-no face distracter |
| `29524918` | emotional face N-back (EFNBACK) task | emotion | 0-back; 2-back; neutral distractors; happy distractors (+1) |
| `26793128` | emotional faces n-back task | emotion | 0-back; 2-back; happy faces; fearful faces (+2) |
| `26930284` | emotional faces n-back task | emotion | happy; fearful; neutral |
| `24468022` | emotional n-back task | emotion |  |
| `32599553` | letter n-back task | letters | Explicit; Neutral |
| `20642398` | letter n-back task with olfactory emotion induction | letters | 0-back; 2-back; negative odour; neutral odour |
| `30082140` | N-back task | letters | 0-back; 1-back; 2-back |
| `5dsgzeZNpCKB` | n-back task | (unspecified) | 2B; 0B |
| `TP6ep7PbLD3y` | n-back task | (unspecified) | 0-back; 2-back |
| `24147643` | n-back working memory paradigm | (unspecified) |  |
| `24894701` | n-back working memory task | letters | find X; 1-back; 2-back; 3-back |
| `5jUzCnXdVoC7` | N-back working memory task | (unspecified) | N-back; 0-back |
| `hW9ivaUNrqhF` | N-back working memory task | letters | 0-Back; 2-Back |
| `30991248` | Working-memory n-back task | (unspecified) |  |

### 17. Facial Emotion Paradigm

named by the Cognitive Atlas · 16 studies · 17 tasks · 16 distinct names

Stimuli: faces (9), money (3), food (2), emotion (2), alcohol (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `17291784` | Alcohol paradigm | alcohol |  |
| `31321407` | dynamic facial emotion viewing | faces | facial emotion; fixation cross; scrambled video |
| `23221317` | emotion paradigm | emotion | alcohol-related stimuli; non-alcoholic beverage-related stimuli; negative emotional stimuli; positive emotional stimuli |
| `rncsZBCjRsut` | emotional faces paradigm | emotion | neutral faces; fearful faces; angry faces; fixation cross |
| `3VDaMUXTf9PH` | event-related facial emotion-processing task | faces | sad; fearful; angry; happy (+1) |
| `21557888` | Faces paradigm | faces | angry; fearful; sad; happy (+2) |
| `25422962` | facial emotion processing task | faces |  |
| `23966929` | Facial emotion recognition | faces |  |
| `31105553` | Facial Emotion Recognition Test | faces |  |
| `29172052` | facial emotion viewing | faces | Explicit; Subliminal |
| `23800487` | facial emotion viewing task | faces | neutral; happy; sad; fixation |
| `3VDaMUXTf9PH` | facial emotion-processing task | faces | sad; fearful; angry; happy (+1) |
| `24238299` | food paradigm | food | food; non-food; LLB |
| `21464344` | milkshake paradigm | food | milkshake cue; tasteless solution cue; milkshake receipt; tasteless solution receipt |
| `22098260` | monetary reward paradigm | money | 45¢; 1¢; 0¢ |
| `22775285` | monetary reward paradigm | money | 0¢; 1¢; 25¢; 50¢ |
| `26096546` | reward paradigm | money | win cue; neutral condition |

### 18. food images

clustered · 16 studies · 17 tasks · 15 distinct names

Stimuli: emotion (11), food (5), alcohol (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26073417` | affective images task | emotion | negative; positive |
| `25757651` | affective picture evaluation | emotion | pleasant; unpleasant; neutral |
| `30239920` | affective picture task | emotion | negative; neutral |
| `4joscm6chBBL` | affective picture task | emotion | positive; neutral; negative |
| `19766164` | affective pictures | emotion | Disgust; Fear; Neutral |
| `17588776` | emotion expectation task | emotion | negative; unknown; positive; neutral |
| `85gRg3uqodAi` | emotion picture task | emotion | negative; neutral |
| `26804333` | emotional anticipation and perception task | emotion | positive; negative; neutral; unknown |
| `24902936` | emotional expectation task | emotion | positive; negative; neutral; unknown valence |
| `18507736` | emotional images with alcohol and non-alcohol beverage cues | alcohol | negative non-alcohol; positive non-alcohol; negative alcohol; positive alcohol |
| `28959198` | Encoding of Affective Pictures | emotion | positive valence; negative valence; high arousal; low arousal |
| `18460331` | food and scenery picture task | food | food; scenery |
| `23566308` | food images | food | food images; neutral images |
| `23867619` | food images | food | low-calorie food images; high-calorie food images; non-food items; rest period |
| `26331843` | food pictures | food | high-calorie food; low-calorie food; neutral non-food |
| `85gRg3uqodAi` | perception of emotional pictures | emotion | negative; neutral; positive |
| `20168310` | visual food-image paradigm | food | neutral; low-calorie; high-calorie |

### 19. event-related cue-reactivity task

clustered · 16 studies · 16 tasks · 15 distinct names

Stimuli: opioid (9), drug (2), cannabis (2), tobacco (2), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `17010993` | anticipation of emotional stimuli | emotion | negative expectation; neutral expectation; positive expectation; negative presentation (+2) |
| `22747521` | audiovisual heroin and neutral cues | opioid | heroin; neutral |
| `22459915` | cue-elicited craving task | tobacco | smoking-related cues; neutral cues; rest |
| `31972520` | Cue-elicited craving task | drug | DRUG; FOOD |
| `30734203` | cue-induced craving task | opioid | heroin-related cues; neutral cues |
| `23667541` | cue-reactivity task | opioid | heroin-related cues; neutral cues |
| `22028765` | drug-related cue-reactivity task | drug | heroin-related stimuli; neutral stimuli |
| `22759909` | event-related cue-reactivity paradigm | opioid | heroin-related cues; neutral cues |
| `22264344` | event-related cue-reactivity task | cannabis | cannabis images; control images; target images |
| `25214465` | event-related cue-reactivity task | opioid | heroin-related cues; neutral cues |
| `21219260` | event-related heroin-related and neutral cue task | opioid | heroin-related stimuli; neutral stimuli |
| `30928885` | heroin cue-reactivity task | opioid | heroin-related cues; neutral cues |
| `25157798` | Heroin-related cue task | opioid |  |
| `32862560` | multimodal cannabis cue reactivity paradigm | cannabis | cannabis odor; cannabis picture; cannabis odor + picture; flower odor (+3) |
| `30467911` | smoking-related cue task | tobacco | smoking pictures; neutral pictures |
| `18056224` | visual heroin-related stimuli | opioid | heroin-related stimuli; neutral stimuli |

### 20. emotion generation and regulation task

clustered · 15 studies · 18 tasks · 15 distinct names

Stimuli: emotion (10), social (8)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19717138` | autobiographical social anxiety scripts | social | neutral; react NSB; reappraise NSB |
| `30656602` | autobiographical social situation task | social | neutral; react; reappraisal; acceptance |
| `23448192` | Beliefs | social |  |
| `24517388` | Beliefs | social | React; Reappraise |
| `19188539` | cognitive-linguistic regulation of emotional reactivity | emotion | Look Harsh Face; Regulate Harsh Face; Look Violent Scene; Regulate Violent Scene (+1) |
| `23448192` | Criticism | social |  |
| `24517388` | Criticism | social | React; Reappraise |
| `21296865` | emotion generation and regulation task | emotion | top-down look; bottom-up look; top-down reappraise; bottom-up reappraise |
| `24430617` | emotion generation and regulation task | emotion | reappraise; look; top-down; bottom-up |
| `30321093` | emotion reactivity and regulation task | emotion | look negative; look neutral; decrease |
| `19555702` | emotion regulatory paradigm | emotion | regulate negative; view negative; regulate positive; view positive |
| `29252164` | emotion sensitivity and regulation paradigm | emotion | negative-look; neutral-look; negative-safe; neutral-safe |
| `24493847` | emotion task | emotion | look/negative; look/neutral; reappraise/negative |
| `31119296` | film-based emotion reactivity and regulation task | emotion | watch neutral; watch positive; watch negative; regulate negative |
| `20141305` | Regulation of Negative Self-Beliefs Task | emotion | react negative self-belief; asterisk counting; breath-focused attention; distraction-focused attention |
| `27095057` | Self-Reg task | emotion | Self-Reg; Self-Look |
| `21998031` | social distance regulation task | social | victim–offender scenes; neutral scenes; disengage; view (+1) |
| `27095057` | Social-Reg task | social | Social-Reg; Social-Look |

### 21. RT probe task

clustered · 13 studies · 13 tasks · 11 distinct names

Stimuli: (unspecified) (8), money (3), food (1), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22505321` | appetitive conditioning | food | CS+; CS− |
| `26606725` | conditioning and extinction task | money | CS+Sex; CS+Money; CS− |
| `21254801` | decision-making under risk | (unspecified) | CS+; CS− |
| `kur8omYaWZqa` | differential fear conditioning task | (unspecified) | CS+; CS− |
| `18492729` | fear conditioning | (unspecified) | CS+; CS- |
| `28235692` | fear conditioning and extinction paradigm | (unspecified) | CS+E; CS-; lCS+; lCS- |
| `31303261` | fear conditioning and extinction paradigm | emotion | CS+E; CS− |
| `18587392` | monetary reward-conditioning procedure | money | CS+; CS− |
| `31113931` | reinstatement test | (unspecified) | CS+; CS− |
| `22928000` | reward conditioning paradigm | money | Attend CS+; Attend CS−; Regulate CS+; Regulate CS− |
| `21993878` | RT probe task | (unspecified) | CS+; CS−; irrelevant CS |
| `30066136` | RT probe task | (unspecified) | CS+; CS− |
| `26149610` | yoked Pavlovian fear conditioning | (unspecified) | CS+UCS; UCS alone |

### 22. food craving task

clustered · 12 studies · 15 tasks · 12 distinct names

Stimuli: food (9), alcohol (3), tobacco (2), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25939814` | alcohol craving task | alcohol | alcohol; juice |
| `23221317` | craving paradigm | emotion | alcohol-related stimuli; non-alcoholic beverage-related stimuli; negative emotional stimuli; positive emotional stimuli |
| `29462475` | craving regulation task | food |  |
| `29472643` | craving-regulation task | tobacco | smoke close; smoke far; non-smoke close; non-smoke far |
| `29472643` | craving-regulation task | tobacco | smoke; non-smoke; close; far |
| `28894291` | desire for palatable food regulation paradigm | food | Enhance; Regulate |
| `27381253` | food craving regulation task | food | LATER; NOW |
| `27381253` | food craving task | food | LATER; NOW |
| `31746092` | food craving task | food | food; nonfood |
| `26883294` | food-craving regulation task | food | ADMIT; REGULATE |
| `25193941` | regulation of craving | food | Close; Far |
| `31892465` | Regulation of Craving task | alcohol | NOW; LATER |
| `31892465` | Regulation of Craving task | alcohol | NOW; LATER |
| `20679212` | ROC task | food | NOW; LATER |
| `21712804` | volitional regulation of the desire for food | food | REGULATE_TASTY; ADMIT_TASTY; ADMIT_NONTASTY; REGULATE_NONTASTY |

### 23. food cue task

clustered · 12 studies · 12 tasks · 6 distinct names

Stimuli: food (7), tobacco (2), cocaine (2), alcohol (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27015258` | alcohol cue task | alcohol | prayer; passive; news |
| `24951856` | cocaine cue task | cocaine | cocaine cues; positive images; neutral images; XY task |
| `32853997` | cocaine cue task | cocaine | cocaine picture; neutral picture |
| `30646204` | food cue paradigm | food | food cues; nonfood cues |
| `22566584` | food cue task | food | food stimuli; neutral stimuli; fixation cross |
| `25229205` | food cue task | food | food stimuli; neutral stimuli |
| `28551112` | food cue task | food | food cues; non-food cues |
| `29352524` | food cue task | food | food cues; non-food cues |
| `30019454` | food cue task | food | food; non-food |
| `25941364` | food-cue task | food |  |
| `20688176` | smoking cue task | tobacco | smoking-related; neutral |
| `27522872` | smoking cue task | tobacco | smoking; neutral |

### 24. Social Evaluation Task

clustered · 12 studies · 12 tasks · 11 distinct names

Stimuli: emotion (8), social (2), (unspecified) (1), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21246665` | emotion appraisal | emotion | self; label; control |
| `25120498` | emotion evaluation | emotion |  |
| `6jVZ7TCUCWHF` | emotion evaluation task | emotion | negative; neutral; positive |
| `26011391` | emotion-rating task | emotion | negative photo with happy facial prime; neutral photo with happy facial prime; negative photo with neutral facial prime; neutral photo with neutral facial prime (+2) |
| `24493835` | emotional maintenance task | emotion | maintain negative; non-maintain negative; maintain positive; non-maintain positive (+1) |
| `29432767` | emotional rating task | emotion |  |
| `30631474` | empathic accuracy task | emotion | OTHER; SELF; GAZE |
| `26294367` | empathy tasks | faces | ER; EPT; AR |
| `25062841` | male–female interaction film scenes | (unspecified) | menacing; prosocial; neutral |
| `25193002` | Social Evaluation Task | social | react praise; react criticism; reappraise criticism; asterisk-counting |
| `30321093` | Social Evaluation Task | social |  |
| `25698699` | socio-affective video task | emotion | Compassion; Reappraisal; Watch-Negative; Watch-Neutral |

### 25. personalized guided-imagery task

clustered · 11 studies · 11 tasks · 10 distinct names

Stimuli: stress (6), food (4), alcohol (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23069840` | guided imagery | food | favorite-food cue; stress cue; neutral-relaxing condition |
| `16163517` | guided imagery and recall of stress and neutral situations | stress | stress imagery; neutral imagery |
| `28888152` | individualized guided imagery with appetitive, stressful and neutral-relaxing cue scripts | food | favorite-food; stress; neutral-relaxing |
| `27501356` | individualized imagery trials | stress | stress; alcohol-cue; neutral-relaxing |
| `26537217` | neutral-relaxing, alcohol, and stress cue exposure | alcohol | neutral-relaxing; alcohol; stress |
| `25567424` | personalized guided-imagery fMRI procedure | stress | stress; neutral/relaxing; favorite-food-craving |
| `25444233` | personalized guided-imagery task | food | favorite-food cue; stress; neutral-relaxing |
| `26627911` | personalized guided-imagery task | stress | stress; favorite-food; neutral-relaxing |
| `24903650` | personalized neutral/relaxing, stressful, and favorite-food cues | food | favorite-food; stress; neutral–relaxing |
| `23636842` | script-driven imagery trials | stress | stress; alcohol cue; neutral-relaxing |
| `32854534` | stress, alcohol cue, and neutral visual cue task | stress | stress; alcohol cue; neutral |

### 26. affect labeling

clustered · 11 studies · 11 tasks · 9 distinct names

Stimuli: emotion (8), faces (3)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22468617` | Affect Label Task | faces | Match; Label; Shapes |
| `24179799` | affect labeling | emotion | affect labeling; gender labeling |
| `28992270` | affect labeling | emotion | affect label; gender label |
| `23774393` | affect labeling paradigm | faces | match facial affect; label facial affect; match forms |
| `28129555` | affect labeling task | emotion | affect labeling; gender labeling |
| `30173058` | affect labeling task | emotion | affect labeling; gender labeling |
| `21041607` | Affect Matching/Labeling Task | emotion | affect match; affect label; shape match |
| `29162186` | emotion label task | emotion | emotion label; emotion match; shape match; rest |
| `16460697` | emotional face labeling and observation | emotion | observe only; emotion label; gender label |
| `24813437` | fMRI affect labeling and reactivity task | emotion | affect label; gender label; affect match; shape match |
| `19656642` | masked facial affect paradigm | faces | masked anger; masked happy; neutral condition |

### 27. Pavlovian-instrumental transfer task

clustered · 10 studies · 10 tasks · 9 distinct names

Stimuli: alcohol (4), money (3), food (1), words (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30556913` | Alcohol Cues Task | alcohol | Alcohol Taste; Water Taste |
| `28633363` | Alcohol Prediction Error (APE) Task | alcohol | alcohol taste; water taste |
| `32406553` | Alcohol Taste Cues Task | alcohol | Alcohol; Water |
| `17714197` | chocolate stimulus task | food | choc; chocpic; chocd; chocw (+1) |
| `20044075` | instrumental motivation task | money | money; cigarettes |
| `28444823` | Pavlovian-instrumental transfer task | words | non-cued trials; cued trials |
| `29313106` | Pavlovian-instrumental transfer task | alcohol | alcohol; water |
| `32188908` | Pavlovian-to-Instrumental Transfer and outcome-devaluation task | (unspecified) | specific PIT; CS+; CS−; ITI |
| `27188979` | reward and aversion task | money | pleasant cue; unpleasant taste |
| `6bQRZaSyC4HV` | reward and aversive stimulus task | money | Chocolate in Mouth; Sight of Chocolate; Chocolate in Mouth with Sight of Chocolate; Strawberry in Mouth (+2) |

### 28. Emotion Recognition Task

named by the Cognitive Atlas · 9 studies · 9 tasks · 7 distinct names

Stimuli: emotion (5), faces (2), tobacco (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21385617` | emotion recognition | emotion |  |
| `27814960` | emotion recognition and voluntary attentional regulation task | emotion | SRT; ERT |
| `29928652` | emotion recognition task | emotion |  |
| `31683488` | emotion recognition task | emotion |  |
| `WugqME6uHuMD` | emotion recognition task | emotion | emotion recognition; rest period |
| `MBPUwjmmCaL7` | faces emotion recognition test task | faces | fearful faces; happy faces; rest |
| `25123156` | Facial emotion recognition task | faces |  |
| `25564288` | Recognition task | (unspecified) |  |
| `19650816` | smoking cues emotion recognition task (SCERT) | tobacco | SCs; NCs |

### 29. food visualization

clustered · 9 studies · 9 tasks · 9 distinct names

Stimuli: food (7), tobacco (1), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30041007` | food and non-food cue reward task | food | HFHS; LFLS; NF |
| `28958004` | food and nonfood image task | food | food; nonfood |
| `21926468` | food image viewing task | food | high-calorie food; low-calorie food; non-food |
| `21593494` | food visualization | food | food; nonfood |
| `28719580` | food/non-food preference task | food | food; non-food |
| `28148724` | negative image viewing task | emotion |  |
| `22114078` | smoking and neutral image cue-induction task | tobacco | LookSmoking; LookNeutral; MindfulSmoking |
| `25139883` | visual food (picture) task | food | HCF; LCF; NF |
| `27569684` | visual food cue task | food | HC; LC; NF; HC and LC food pictures |

### 30. stop signal task

named by the Cognitive Atlas · 8 studies · 8 tasks · 6 distinct names

Stimuli: (unspecified) (6), alcohol (1), sexual (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `32359231` | modified stop-signal task | sexual | neutral; pornographic |
| `24090712` | stop signal task | (unspecified) |  |
| `25485181` | stop signal task | (unspecified) | StopSuccess; GoSuccess |
| `27217106` | Stop Signal Task | (unspecified) |  |
| `32179061` | stop signal task | (unspecified) | green; yellow; orange; red |
| `24988265` | Stop-Signal Alcohol-Cue Task | alcohol | CR; GO; Alcohol; Control |
| `21690575` | Stop-Signal task | (unspecified) |  |
| `26078198` | stop-signal task | (unspecified) | go trials; stop trials |

### 31. viewing photographs of high and low-calorie foods

clustered · 8 studies · 11 tasks · 10 distinct names

Stimuli: food (4), alcohol (3), (unspecified) (3), opioid (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `16340649` | alcohol word presentation | alcohol |  |
| `11515754` | alcohol-associated visual stimuli | alcohol |  |
| `32546139` | Dot Probe Task with Alcohol stimuli | alcohol |  |
| `19674446` | FDG-PET diagnostic imaging | (unspecified) |  |
| `24956915` | motor and psychological-inducing movie cues | opioid | motor activities; heroin puffing movie; movie with explicit sexual content |
| `26497657` | SVM-RFE classification | (unspecified) |  |
| `16565998` | viewing photographs of high and low-calorie foods | food | high-calorie foods; low-calorie foods |
| `16565998` | viewing photographs of high and low-calorie foods | food | high-calorie foods; low-calorie foods |
| `16565998` | Viewing photographs of high- and low-calorie foods | food |  |
| `16565998` | viewing photographs of high- and low-calorie foods | food | high-calorie foods; low-calorie foods |
| `19025463` | Virtual-environment cue exposure treatment | (unspecified) |  |

### 32. drug word fMRI task

clustered · 8 studies · 8 tasks · 8 distinct names

Stimuli: drug (3), emotion (2), words (2), social (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19420266` | drug word fMRI task | drug | drug words; neutral words; fixation cross |
| `20823246` | drug-word fMRI task | drug | drug words; neutral words; fixation baseline |
| `26434802` | ecStroop | words | grief; neutral |
| `24623788` | emotion-arousal word task | emotion | negative words; neutral words; positive words; rest blocks |
| `23761898` | fMRI drug word task | drug | drug word; neutral word; fixation |
| `22016480` | math and word task | words | math; word |
| `21079747` | scrambled sentences task | emotion | emotional; neutral; resting baseline |
| `30513084` | Social fMRI task | social | faces; bodies; scrambled images |

### 33. emotional conflict task

clustered · 8 studies · 11 tasks · 7 distinct names

Stimuli: emotion (7), money (2), faces (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `20123913` | emotional conflict task | emotion |  |
| `25413183` | emotional conflict task | emotion | congruent; incongruent |
| `28715907` | emotional conflict task | emotion |  |
| `28715908` | Emotional Conflict Task | emotion | Congruent Fear; Congruent Happy |
| `28913886` | Emotional conflict task | emotion |  |
| `4As5VxmrRpSX` | Emotional Conflict Task | emotion | All Trials; Congruent; Incongruent |
| `24119861` | face-word emotion conflict | faces | congruent; incongruent |
| `28715908` | Gender Conflict Task | faces |  |
| `21612768` | incentive conflict task | money | A+ or B+; C– or D–; AB–; CD– |
| `21612768` | incentive conflict task | money | AB–; CD– |
| `28913886` | Nonemotional conflict task | emotion |  |

### 34. distraction task

clustered · 8 studies · 8 tasks · 7 distinct names

Stimuli: emotion (5), food (2), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30554861` | distraction | emotion | distraction; passive viewing of negative pictures; passive viewing of neutral pictures |
| `27091455` | distraction task | emotion | MEMORIZE-NEGATIVE; MEMORIZE-NEUTRAL; VIEW-NEGATIVE; VIEW-NEUTRAL |
| `29622050` | distraction task | emotion | OCD-relevant pictures; aversive pictures; neutral pictures |
| `25038629` | emotion processing picture task | emotion | unpleasant pictures; neutral pictures; pleasant pictures; fixation cross |
| `27479051` | food and non-food picture viewing | food | food images; non-food images |
| `31931900` | food distraction | food | distraction food; distraction non-food |
| `27551094` | perspective-taking task | emotion | negative images; neutral images |
| `31394197` | SEM and neutral stimulus task | (unspecified) | SEM picture; neutral picture; SEM video; neutral video |

### 35. face matching task

named by the Cognitive Atlas · 7 studies · 7 tasks · 6 distinct names

Stimuli: faces (4), emotion (3)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `VFpYrpiDJLyS` | emotional face matching task | emotion | Angry; fearful; happy; sad (+1) |
| `28589968` | emotional face-matching task | emotion | emotional face condition; control condition |
| `orpHcxqvpXhd` | emotional face-matching task | emotion | emotional faces; geometrical shapes |
| `29959970` | extended Hariri face matching task | faces | erotic scenes; fearful scenes; neutral control condition |
| `27579051` | Face Matching Task | faces | control; neutral; fear |
| `24853295` | face-matching task | faces | face-matching; shape-matching |
| `23768841` | facial emotion matching task | faces | face; shape |

### 36. viewing images of food and nonfood objects

clustered · 7 studies · 8 tasks · 8 distinct names

Stimuli: (unspecified) (4), food (2), emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30287300` | amygdala localizer | emotion |  |
| `26899786` | fMRI localizer task | (unspecified) |  |
| `24241476` | functional localizer | emotion |  |
| `22453299` | functional localizer scan | (unspecified) | objects/artifacts; scrambled images |
| `29906489` | localizer run | food | healthy foods; unhealthy foods; non-food objects |
| `28429068` | object-sensitive functional localizer | (unspecified) |  |
| `22453299` | object/artifact and scrambled-image localizer | (unspecified) | objects/artifacts; scrambled images |
| `16687507` | viewing images of food and nonfood objects | food | appetizing foods; nonfood objects |

### 37. smoking cue exposure

clustered · 7 studies · 7 tasks · 6 distinct names

Stimuli: tobacco (6), cannabis (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24035535` | cannabis cue-exposure fMRI task | cannabis | cannabis cues; Gaussian baseline; nature cues; food cues |
| `18704100` | cue exposure | tobacco | SCs; non-SCs |
| `21859165` | cue exposure fMRI task | tobacco |  |
| `26179147` | cue-exposure fMRI task | tobacco | PSEs; PNEs |
| `23061530` | Smoking cue exposure | tobacco |  |
| `25762748` | smoking cue exposure | tobacco | SCs; non-SCs |
| `30901743` | smoking cue exposure | tobacco | SC; nonSCs |

### 38. food cue visual stimulation

clustered · 7 studies · 7 tasks · 7 distinct names

Stimuli: emotion (3), food (1), cannabis (1), drug (1), letters (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19818382` | attention, memory, and emotion control tasks | emotion | forced-left; non-forced; source memory; item memory (+2) |
| `24962329` | cannabis cue–reactivity task | cannabis | neutral pictures; cannabis pictures |
| `22858151` | emotion down-regulation task | emotion | Decrease Negative; Observe Negative |
| `23307469` | food cue visual stimulation | food | high caloric food pictures; low caloric food pictures; non-food pictures; fixation cross |
| `22579718` | instructed fear-conditioning task | letters | Early Threat Focus; Late Threat Focus; Early Alternative Focus; Late Alternative Focus |
| `19478067` | rewarded drug cue-reactivity task | drug | drug words at 50¢; drug words at 0¢; neutral words at 50¢; neutral words at 0¢ |
| `24818080` | symptom provocation | emotion | OC; AV; NE |

### 39. approach-avoidance task

clustered · 7 studies · 7 tasks · 6 distinct names

Stimuli: alcohol (2), faces (2), money (1), food (1), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19047074` | AA and GE face-response tasks | faces | AA; GE; congruent; incongruent |
| `31251695` | approach-avoidance task | emotion | congruent; incongruent |
| `31492567` | approach-avoidance task | faces | affect-congruent; affect-incongruent |
| `25639749` | functional MRI AAT | alcohol | alcohol pull; alcohol push; soft drink pull; soft drink push |
| `24060832` | implicit approach-avoidance task | alcohol | alcohol; soft drink; approach; avoid |
| `29438844` | instrumental approach-avoidance task | food | high-calorie; low-calorie; neutral |
| `27649775` | probabilistic feedback expectancy task | money | gain; loss; bivalent; fixation |

### 40. cognitive re-appraisal task

clustered · 7 studies · 7 tasks · 6 distinct names

Stimuli: emotion (4), (unspecified) (2), social (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23270876` | attentional deployment | (unspecified) | unpleasant no focus; unpleasant arousing focus; unpleasant non-arousing focus; neutral no focus (+1) |
| `22978709` | cognitive re-appraisal task | emotion | Maintain; Observe; Suppress |
| `28060454` | cognitive re-appraisal task | emotion | observe; experience; regulate |
| `30124815` | Dynamic Interpersonal Criticism fMRI Task | (unspecified) | Watch Criticism; Reappraise Criticism; asterisk-counting task |
| `31989171` | mindful acceptance emotion and pain regulation task | emotion | negative images; neutral images; painfully hot temperatures; warm temperatures |
| `31547972` | NSB task | social | react; reappraise; accept |
| `15691521` | Voluntary regulation of negative emotion | emotion |  |

### 41. Ultimatum Game

clustered · 7 studies · 7 tasks · 5 distinct names

Stimuli: money (5), (unspecified) (1), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30343211` | anger-infused Ultimatum Game | money | fair; medium; unfair |
| `24027512` | Dictator Game | money | fair offers down-regulate; midfair offers down-regulate; unfair offers down-regulate; fair offers look (+5) |
| `26166623` | modified Ultimatum Game | money | fair; unfair |
| `28052779` | modified Ultimatum Game | money | fair offers; unfair offers |
| `29309499` | UG task | faces | unfair; fair |
| `22368088` | Ultimatum Game | money | Down; Look; Up |
| `25720857` | Ultimatum Game | (unspecified) | unfair proposal; fair proposal; unfair accepted; fair accepted (+1) |

### 42. food picture viewing

clustered · 6 studies · 7 tasks · 5 distinct names

Stimuli: emotion (4), food (2), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `17488322` | affective picture viewing | emotion | alcohol pictures; positive pictures; negative pictures; neutral pictures (+1) |
| `19806158` | food picture viewing | food | food; pleasant; neutral |
| `26796027` | food picture viewing | food | high caloric available; high caloric non-available; low caloric available; low caloric non-available |
| `29931375` | free viewing faces | faces | Surprise; Neutral |
| `21045002` | picture viewing | emotion | alcohol; concern; positive; negative (+1) |
| `24361634` | picture viewing | emotion | aversive smoking-related; aversive IAPS; appetitive smoking-related; neutral |
| `19806158` | pleasant picture viewing | emotion | pleasant |

### 43. appetite-provoking fMRI task

clustered · 6 studies · 6 tasks · 6 distinct names

Stimuli: food (5), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22209236` | appetite-provoking fMRI task | food | food images; scrambled pictures |
| `22974271` | food and control pictures | food | food pictures; control pictures |
| `22461323` | food pictures and low-level control pictures | food | food pictures; control pictures |
| `25060944` | food-cue reactivity paradigm | food | food; low-level control |
| `23778853` | food-cue task | food | food; control |
| `25564288` | graphic labels fMRI task | (unspecified) | High ER labels; Low ER labels; control images |

### 44. backward masking

named by the Cognitive Atlas · 5 studies · 5 tasks · 5 distinct names

Stimuli: cocaine (1), cannabis (1), sexual (1), faces (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24275010` | backward masking paradigm | sexual | Sex; Fix; Neu; Emo |
| `26996601` | Backward masking task | (unspecified) |  |
| `GcyiPEos9qZR` | backward masking task | faces | sad faces; happy faces; neutral faces; baseline fixation cross |
| `24186078` | backward-masking paradigm | cannabis | cannabis cues; sexual cues; aversive cues; neutral cues |
| `18231593` | backward-masking task | cocaine | cocaine; sexual; aversive; neutral (+1) |

### 45. passive viewing

named by the Cognitive Atlas · 5 studies · 5 tasks · 4 distinct names

Stimuli: emotion (4), food (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27033686` | Emotional picture perception with attention-to-breath and passive viewing | emotion |  |
| `21106812` | passive viewing | emotion | negative; neutral |
| `21949675` | passive viewing | emotion | negative; neutral |
| `29622050` | passive viewing task | emotion | OCD-relevant pictures; aversive pictures; neutral pictures |
| `22542330` | passive viewing task with appetizing and neutral images | food | appetizing (AC); neutral (NC) |

### 46. oddball task

named by the Cognitive Atlas · 5 studies · 5 tasks · 5 distinct names

Stimuli: emotion (2), alcohol (1), food (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `15734342` | auditory oddball task | (unspecified) |  |
| `22448769` | emotional oddball | emotion | Negative; Neutral |
| `18843096` | forced-choice visual oddball task | emotion | target; aversive |
| `27524674` | two-choice oddball task | food | High; Low; Neutral |
| `23131612` | visual oddball task | alcohol | alcohol distractors; non-alcohol distractors |

### 47. picture cue viewing task

clustered · 5 studies · 5 tasks · 5 distinct names

Stimuli: alcohol (2), cocaine (2), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21995620` | alcohol cue exposure paradigm | alcohol | alcohol taste; juice taste |
| `26900792` | cocaine cue and neutral cue sessions | cocaine | neutral cues; cocaine cues |
| `22458561` | cocaine cue-exposure paradigm | cocaine | cocaine cues; neutral objects; visual control images; rest |
| `26727534` | event-related cue-exposure paradigm | alcohol |  |
| `20729530` | picture cue viewing task | emotion | alcohol-related picture cues; polydrug-related picture cues; marijuana-related picture cues; positive emotional picture cues (+2) |

### 48. film viewing

named by the Cognitive Atlas · 4 studies · 4 tasks · 4 distinct names

Stimuli: emotion (3), cocaine (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30488224` | emotional film viewing | emotion | positive; neutral; negative |
| `17888411` | film-viewing task | emotion | Watch-Neutral; Watch-Negative; Reappraise-Negative; Suppress-Negative |
| `21531382` | sad and neutral film viewing | emotion | sad film; neutral film |
| `11058476` | three film viewing paradigm | cocaine | cocaine film; nature film; sex film |

### 49. cyberball task

named by the Cognitive Atlas · 4 studies · 7 tasks · 3 distinct names

Stimuli: social (5), (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24033579` | Cyberball | (unspecified) | observed inclusion; observed exclusion |
| `28946039` | Cyberball | (unspecified) |  |
| `29868921` | Cyberball social rejection paradigm | social | acceptance; rejection |
| `25094019` | Cyberball social rejection task | social | rejection; acceptance |
| `25094019` | Cyberball social rejection task | social | rejection; acceptance |
| `25094019` | Cyberball social rejection task | social | rejection; acceptance |
| `25094019` | Cyberball social rejection task | social | rejection; acceptance |

### 50. Iowa Gambling Task

named by the Cognitive Atlas · 4 studies · 4 tasks · 2 distinct names

Stimuli: money (3), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23182846` | Iowa Gambling Task | money | advantageous decks (C + D); disadvantageous decks (A + B) |
| `24179781` | Iowa Gambling Task | money |  |
| `24676464` | Iowa Gambling Task | money |  |
| `18801475` | modified Iowa Gambling Task | (unspecified) |  |

### 51. cue-viewing task

clustered · 4 studies · 4 tasks · 3 distinct names

Stimuli: tobacco (2), opioid (1), food (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `16759342` | Cue viewing task | tobacco |  |
| `19107465` | cue-viewing task | tobacco | smoking cues; control cues |
| `22329835` | cue-viewing task | opioid | heroin-related images; neutral images |
| `23929709` | Food cue viewing task | food |  |

### 52. smoking and nonsmoking cue stimuli

clustered · 4 studies · 4 tasks · 4 distinct names

Stimuli: tobacco (3), cocaine (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `9433350` | cocaine-related and neutral cue stimuli | cocaine | cocaine-related cues; neutral cues |
| `17375140` | smoking and nonsmoking cue stimuli | tobacco | smoking cue; nonsmoking cue |
| `31069895` | smoking cue and nonsmoking cue videos | tobacco | smoking cues; nonsmoking cues |
| `28711813` | smoking cue video task | tobacco | SC; non-SC |

### 53. reward and personal reference task

clustered · 4 studies · 4 tasks · 4 distinct names

Stimuli: money (1), alcohol (1), tobacco (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24373127` | ecological decisions-to-drink task | alcohol | alcohol; food; item |
| `28290265` | Optimistic Bias (OB) Task | (unspecified) | Low impact; High impact |
| `32828721` | real smoking decision task | tobacco | cognitive stress; emotional stress; no stress |
| `18711709` | reward and personal reference task | money | win; lose; high personal reference; low personal reference |

### 54. food perception task

clustered · 4 studies · 4 tasks · 4 distinct names

Stimuli: emotion (2), food (1), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26784537` | emotional face perception task | emotion |  |
| `4vQUWokc86xd` | emotional perception task | emotion | faces; houses; baseline |
| `25797589` | Face-Perception task | faces |  |
| `19260039` | food perception task | food | high-calorie foods; control images |

### 55. cognitive control strategy food-picture task

clustered · 4 studies · 5 tasks · 4 distinct names

Stimuli: food (3), tobacco (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22230946` | cognitive control strategy food-picture task | food | cognitive reappraisal; suppression; up-regulation; passive viewing |
| `22230946` | cognitive control strategy food-picture task | food | suppression; cognitive reappraisal; up-regulation; passive viewing |
| `22381514` | food and control picture task | food | appetizing food; neutral stimuli |
| `20090671` | smoking and control picture task | tobacco | TAKING OUT; BEGIN; LAST PUFF; END |
| `20331560` | tobacco and control advertisement task | tobacco | tobacco advertisement; control advertisement |

### 56. food cue viewing

clustered · 4 studies · 4 tasks · 1 distinct names

Stimuli: food (4)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21945867` | food cue viewing | food | fattening food; non-fattening food; object |
| `22776461` | food cue viewing | food | food images; non-food images |
| `23954410` | food cue viewing | food | HiCal; LoCal; Control |
| `25062455` | food cue viewing | food | appetizing foods; disgusting foods; bland foods; non-food objects |

### 57. food and nonfood picture paradigm

clustered · 4 studies · 4 tasks · 4 distinct names

Stimuli: food (4)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23408738` | food and non-food image viewing | food | high-calorie food; non-food items |
| `25279500` | food and nonfood image viewing | food | food images; nonfood images |
| `30590423` | food and nonfood images | food | high-calorie food images; low-calorie food images; nonfood images |
| `23364016` | food and nonfood picture paradigm | food | food; nonfood |

### 58. food motivation paradigm

clustered · 4 studies · 4 tasks · 3 distinct names

Stimuli: food (4)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24862390` | food cue functional magnetic resonance imaging paradigm | food | high-calorie foods; low-calorie foods; objects; fixation |
| `25533729` | food motivation fMRI paradigm | food | food; non-food |
| `24183133` | food motivation paradigm | food | food; blurry baseline; non-food (animal) |
| `31740723` | food motivation paradigm | food | high-calorie foods; low-calorie food items; objects; fixation stimuli |

### 59. emotional faces

clustered · 4 studies · 4 tasks · 2 distinct names

Stimuli: emotion (2), faces (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25409596` | emotional faces | emotion | emotional faces; non-emotional control cross-hair |
| `27109623` | emotional faces | emotion | fearful faces; non-emotional control cross-hair |
| `23448192` | Faces | faces |  |
| `24517388` | Faces | faces | React; Reappraise |

### 60. food choice task

clustered · 4 studies · 4 tasks · 4 distinct names

Stimuli: food (4)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24167442` | episodic intertemporal choice task | food | ecological condition; control condition |
| `27428267` | food choice task | food | healthy diet; tasty diet; no diet |
| `30388113` | food evaluation paradigm | food | NO CALORIES; CALORIES |
| `30165099` | food preference decision-making task | food | appetizing; plain; baseline |

### 61. emotion processing

clustered · 4 studies · 4 tasks · 3 distinct names

Stimuli: emotion (3), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27510944` | emotion processing | emotion | emotional stimuli; neutral stimuli |
| `30291441` | emotion processing | emotion | permit negative; permit neutral; detach negative; detach neutral |
| `7Kzym5AWTBKt` | emotional face processing | emotion | fearful faces; happy faces; fixation baseline |
| `21172863` | face processing | faces |  |

### 62. Wheel of Fortune task

clustered · 4 studies · 4 tasks · 3 distinct names

Stimuli: (unspecified) (3), money (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `29075567` | Giving Game | (unspecified) | prosocial giving; selfless giving; catch trials 1; catch trials 2 |
| `26786150` | lottery task | (unspecified) | win; loss; boost |
| `22998631` | Wheel of Fortune task | money | selection; anticipation; win; non-win |
| `4z9qeVwZU2nH` | Wheel of Fortune task | (unspecified) | 10% chance of winning $7 and 90% chance of winning $1; 30% chance of winning $2 and 70% chance of winning $1; two 50% chances of winning $2; control condition |

### 63. mood induction

clustered · 4 studies · 4 tasks · 3 distinct names

Stimuli: emotion (3), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `17467008` | mood induction | faces | happiness; sadness; gender discrimination; resting baseline with fixation cross |
| `23576810` | mood induction | emotion | happy; fixation cross |
| `28100219` | negative mood induction task | emotion | sad mood recall; neutral mood recall |
| `2Qx5cSsFtZFQ` | positive mood induction | emotion | positive mood induction |

### 64. partner and opposite-sex stranger facial expressions

clustered · 4 studies · 4 tasks · 4 distinct names

Stimuli: faces (2), (unspecified) (1), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24278126` | handholding threat paradigm | (unspecified) | alone; stranger; partner |
| `25280904` | mother/stranger task | emotion | mother; stranger |
| `30952600` | Parent/Stranger fMRI Task | faces | parent; stranger |
| `20004365` | partner and opposite-sex stranger facial expressions | faces | partner negative; partner positive; partner neutral; stranger negative (+2) |

### 65. rtfMRI-nf training

clustered · 4 studies · 4 tasks · 4 distinct names

Stimuli: emotion (3), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26958462` | Happy Memories task | emotion | Happy Memories; Count; Rest |
| `27534862` | rtfMRI-nf experiment | emotion | happy; sad; rest |
| `24223175` | rtfMRI-nf training | emotion | Rest; Happy Memories; Count |
| `30087646` | rtfMRI-NFB | (unspecified) | tenderness; anguish; neutral |

### 66. verbal working memory task

named by the Cognitive Atlas · 3 studies · 3 tasks · 3 distinct names

Stimuli: (unspecified) (2), letters (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `12821731` | parametric verbal working memory task | letters | fixation; target = X; 1-back; 2-back |
| `19727332` | Sentence comprehension and verbal working memory | (unspecified) |  |
| `JBFdQdKQQX37` | verbal working memory n-back task | (unspecified) | 0-back; 1-back; 2-back; 3-back |

### 67. Continuous Performance Task

named by the Cognitive Atlas · 3 studies · 3 tasks · 3 distinct names

Stimuli: letters (2), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `18940597` | Continuous performance task | letters |  |
| `21590316` | continuous performance task | letters | decrease; natural; increase |
| `25359589` | continuous performance task with emotional and neutral distractors | emotion | squares; circles; emotionally neutral pictures; negatively valenced pictures |

### 68. smoking and control videos

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: tobacco (3)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19632211` | smoking and control cues | tobacco |  |
| `16598192` | smoking and control videos | tobacco | smoking; control |
| `22342802` | smoking and non-smoking cue videos | tobacco | smoking; control |

### 69. heroin-related visual stimuli

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: emotion (2), opioid (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `20678551` | heroin-related visual stimuli | opioid | heroin-related; neutral |
| `23887820` | self-critical processing | emotion | self-critical; neutral; negative non-self-referential; rest |
| `16087352` | thought suppression | emotion | Don't Think; Think; Don't Think Relationship; Think Relationship (+3) |

### 70. attentional bias line counting task

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: tobacco (2), sexual (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `20932921` | attentional bias line counting task | tobacco | LCSP; LCNP; PNSP; PNNP |
| `22850734` | attentional bias line-counting task | tobacco | line-counting smoke picture; line-counting neutral picture; picture-naming smoke picture; picture-naming neutral picture |
| `29729394` | line orientation and picture categorization task | sexual | sexual; neutral |

### 71. visual cue paradigm

clustered · 3 studies · 4 tasks · 4 distinct names

Stimuli: sexual (1), alcohol (1), tobacco (1), food (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25208199` | Alcohol-taste-cue paradigm | alcohol |  |
| `23378222` | visual cue paradigm | sexual | sexual visual cues; neutral visual cues |
| `29529147` | visual food cue paradigm | food | fattening food; nonfattening food; objects |
| `25208199` | Visual smoking-cue paradigm | tobacco |  |

### 72. response inhibition

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: emotion (2), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `31557545` | Emotional interference and motor response inhibition | emotion | Con Go; Incon Go; Stop |
| `24337077` | response inhibition | (unspecified) | correctly inhibited targets; control targets (go stimuli) |
| `24951856` | response inhibition task | emotion | HITS; STOPS; ERRORS |

### 73. rewarded antisaccade task

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: money (2), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `16763044` | Antisaccade task | (unspecified) |  |
| `26026506` | fMRI reward cue antisaccade (AS) task | money |  |
| `24914005` | rewarded antisaccade task | money | reward; neutral |

### 74. taste task

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: food (1), pain (1), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30225341` | control task | faces |  |
| `17079679` | pain control task | pain | self-controlled condition; other-controlled condition; computer-controlled condition |
| `31326440` | taste task | food |  |

### 75. rtfMRI neurofeedback task

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: (unspecified) (2), pain (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26475487` | neurofeedback task | (unspecified) | up-regulation; down-regulation |
| `26899786` | neurofeedback training task | pain | regulation; baseline |
| `32597838` | rtfMRI neurofeedback task | (unspecified) |  |

### 76. Experiment 1: face shape detection

clustered · 3 studies · 4 tasks · 4 distinct names

Stimuli: faces (2), emotion (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27298765` | Experiment 1: face shape detection | faces |  |
| `27298765` | Experiment 2: unfamiliar face identity matching | faces |  |
| `15880108` | perceptual processing of fearful and threatening facial expressions | emotion |  |
| `23516294` | placebo training task | (unspecified) |  |

### 77. Cookie Theft picture description

clustered · 3 studies · 3 tasks · 2 distinct names

Stimuli: (unspecified) (3)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27301638` | Cookie Theft picture description | (unspecified) |  |
| `28724588` | Cookie Theft picture description | (unspecified) |  |
| `30783611` | Cookie Theft picture description task | (unspecified) |  |

### 78. encoding and immediate JOLs task

clustered · 3 studies · 6 tasks · 5 distinct names

Stimuli: words (4), (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `SULKxviGFurw` | encoding and immediate JOLs task | words | JOLhighMlow; JOLlowMhigh |
| `SULKxviGFurw` | encoding and immediate JOLs task | words | JOLhighMlow; JOLlowMhigh |
| `SULKxviGFurw` | encoding and immediate judgment of learning task | words | JOLhighMlow; JOLlowMhigh |
| `SULKxviGFurw` | encoding and immediate judgments of learning task | words | overestimated bias (JOL high, recognition failed); underestimated bias (JOL low, recognition correct) |
| `26948669` | Perceptual metacognition task | (unspecified) |  |
| `31575976` | visual monitoring task | (unspecified) | perceptual judgment; confidence; wagering |

### 79. anticipatory anxiety paradigm

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: (unspecified) (2), pain (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25048028` | anticipatory anxiety | (unspecified) | regulation condition; anxiety condition; control condition |
| `16859413` | anticipatory anxiety paradigm | pain | No self-distraction; Self-distraction |
| `21394853` | worry modulation task | (unspecified) | worry induction; worry suppression; resting state |

### 80. picture stimuli and thermal stimuli

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: emotion (2), pain (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22428013` | Picture and temperature stimulation task | emotion |  |
| `20537612` | picture stimuli and thermal stimuli | emotion |  |
| `28338955` | placebo analgesia pain task | pain | placebo; control |

### 81. aversive and neutral anticipation

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: emotion (2), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24753211` | anticipation of viewing fear faces | faces | fear anticipation; neutral anticipation |
| `20832448` | aversive and neutral anticipation | emotion | aversive anticipation; neutral anticipation |
| `28197108` | fear processing with suppression priming | emotion | threat; safe |

### 82. Taylor Aggression Paradigm

clustered · 3 studies · 3 tasks · 2 distinct names

Stimuli: (unspecified) (3)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21041607` | Competitive Reaction Time Task | (unspecified) |  |
| `30343211` | Taylor Aggression Paradigm | (unspecified) |  |
| `30459667` | Taylor Aggression Paradigm | (unspecified) | non-provocative condition; provocative condition |

### 83. risky monetary choices

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: money (1), (unspecified) (1), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `29487519` | investment choice task | emotion | positive emotion; negative emotion; neutral emotion |
| `22275168` | risky monetary choices | money | Attend; Regulate |
| `24382784` | sequential investment task | (unspecified) | Regulate; Attend |

### 84. affective Posner task

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: emotion (2), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23732841` | affective Posner task | emotion | positive feedback; negative feedback |
| `26175679` | feedback processing task | faces | informative feedback; confirmatory feedback |
| `25550068` | simulated peer interaction chatroom paradigm | emotion | positive feedback from high-value peers; positive feedback from low-value peers; negative feedback from high-value peers; negative feedback from low-value peers |

### 85. Shifted-Attention Emotion Appraisal Task

clustered · 3 studies · 3 tasks · 2 distinct names

Stimuli: emotion (3)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25939653` | shifted-attention emotion appraisal task | emotion | Male/Female; Faces Only; Fearful; Neutral (+3) |
| `27973443` | Shifted-Attention Emotion Appraisal Task | emotion |  |
| `28032303` | Shifted-Attention Emotion Appraisal Task | emotion | male/female; face-only; indoor/outdoor; like/dislike |

### 86. Amygdala neurofeedback regulation task

clustered · 3 studies · 3 tasks · 3 distinct names

Stimuli: emotion (3)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `29602255` | amygdala neurofeedback | emotion | up; down; view |
| `26481674` | Amygdala neurofeedback regulation task | emotion |  |
| `26833918` | real-time fMRI neurofeedback | emotion | regulate; view; neutral |

### 87. emotional face assessment task

clustered · 3 studies · 3 tasks · 2 distinct names

Stimuli: emotion (3)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26908926` | emotional face assessment task | emotion | emotional; neutral |
| `27973443` | Emotional Face Assessment Task | emotion |  |
| `31060042` | emotional face assessment task | emotion |  |

### 88. mental imagery task

named by the Cognitive Atlas · 2 studies · 2 tasks · 2 distinct names

Stimuli: cocaine (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24367315` | script-driven mental imagery | (unspecified) | pre-hurtful condition; hurtful condition; forgiving response; unforgiving response |
| `14754771` | script-guided mental imagery | cocaine | cocaine use imagery; drug-neutral imagery |

### 89. pavlovian conditioning task

named by the Cognitive Atlas · 2 studies · 2 tasks · 2 distinct names

Stimuli: food (1), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `28040541` | aversive Pavlovian conditioning | emotion | CER; NoCER |
| `26490862` | higher-order Pavlovian conditioning paradigm | food | appetitive; aversive |

### 90. verbal fluency task

named by the Cognitive Atlas · 2 studies · 3 tasks · 3 distinct names

Stimuli: (unspecified) (2), letters (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19363702` | phonemic verbal fluency task | letters |  |
| `19363702` | semantic verbal fluency task | (unspecified) |  |
| `19687454` | verbal fluency testing | (unspecified) |  |

### 91. letter naming task

named by the Cognitive Atlas · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (1), food (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `31092880` | naming task | (unspecified) |  |
| `31617330` | Naming task | food |  |

### 92. word-picture matching task

named by the Cognitive Atlas · 2 studies · 2 tasks · 2 distinct names

Stimuli: words (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `31092880` | word-picture matching task | words |  |
| `31617330` | Word-picture matching task | words |  |

### 93. autobiographical memory task

named by the Cognitive Atlas · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19058792` | negative autobiographical memory strategy task | emotion | feel; accept; analyze; spatial perception task |
| `31711031` | positive autobiographical memory neurofeedback | emotion | rest; happy; count |

### 94. alcohol pictures task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: alcohol (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `12860777` | alcohol pictures task | alcohol | alcohol pictures; neutral beverage pictures |
| `24304235` | Alcohol Pictures Task | alcohol |  |

### 95. water-related, drug-related, and neutral cues

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: drug (1), tobacco (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19369561` | smoking-related and control images | tobacco | smoke; control |
| `16406379` | water-related, drug-related, and neutral cues | drug | Thirst; Drug; Neutral |

### 96. affective stimuli

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (1), tobacco (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `18276852` | affective stimuli | emotion | negative affective images; positive IAPS images |
| `23155368` | smoking stimuli | tobacco | BEGIN-smoking-stimuli; BEGIN-control-stimuli; END-smoking-stimuli; END-control-stimuli |

### 97. food picture attention task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: food (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26978737` | food palatability evaluation task | food | palatable foods; neutral foods; neutral items |
| `19028527` | food picture attention task | food | high-caloric foods; low-caloric foods; neutral objects |

### 98. Montreal Imaging Stress Task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: stress (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19632211` | Montreal Imaging Stress Task | stress |  |
| `25376429` | Montreal Imaging Stress Test | stress | rest; control; stress |

### 99. watching movies

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: tobacco (1), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `5wLcBc9qz8n7` | emotional narratives | emotion | pleasant; unpleasant; neutral |
| `21248113` | watching movies | tobacco | smoking scenes; nonsmoking scenes |

### 100. attentional bias paradigm

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: tobacco (1), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21440645` | attentional bias paradigm | tobacco |  |
| `29931375` | valence bias task | emotion |  |

### 101. Temporal Difference Error/Juice Paradigm

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (1), money (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24569319` | temporal difference error juice paradigm | money | CS; UCS; PTDE; NTDE |
| `22032832` | Temporal Difference Error/Juice Paradigm | (unspecified) | CS; UCS; NTDE; PTDE |

### 102. individualized script imagery

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: stress (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22294257` | individualized script imagery | stress | stress; drug/alcohol cue; neutral-relaxing |
| `19522883` | script-driven imagery | (unspecified) | neutral; trigger; reactions; SIB (+1) |

### 103. picture perception task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: tobacco (1), alcohol (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22476609` | picture perception task | tobacco |  |
| `25421512` | picture-perception task | alcohol | alcohol; control |

### 104. cigarette cue-reactivity and self-expansion task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: tobacco (1), gaming (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22860092` | cigarette cue-reactivity and self-expansion task | tobacco | partner+cig; partner+pen; acquaintance+cig; acquaintance+pen |
| `23245948` | cue-induced reactivity paradigm | gaming | gaming cue; smoking cue; neutral stimulation |

### 105. resting-state protocol

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24944870` | resting-state protocol | (unspecified) | abstinent; satiated |
| `25425031` | resting-state speech anticipation paradigm | (unspecified) | baseline; speech anticipation; recovery |

### 106. neurofeedback

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: alcohol (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26096546` | neurofeedback | alcohol | alcohol cues; rest |
| `32546139` | Neurofeedback | (unspecified) |  |

### 107. visual food cue viewing

clustered · 2 studies · 3 tasks · 2 distinct names

Stimuli: food (3)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26867073` | visual food cue viewing | food |  |
| `29499312` | visual food cue viewing | food | Tea high-caloric; Tea low-caloric; Water high-caloric; Water low-caloric |
| `29499312` | visual food cues with bitter aftertaste and water | food | Tea; Water; high-caloric; low-caloric |

### 108. Smoking Pleasantness task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: tobacco (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `28401670` | e-cigarette and neutral advertising images | tobacco | e-cigarette advertising images; neutral advertising images |
| `27427215` | Smoking Pleasantness task | tobacco |  |

### 109. flavor paradigm

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: alcohol (1), food (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30093174` | flavor conditioning paradigm | food | sweet; unsweetened; nicotine; sweet+nicotine |
| `27459715` | flavor paradigm | alcohol | beer; Gatorade; water |

### 110. food/non-food discrimination task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: food (1), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26180190` | emotional face discrimination task | emotion | threatening; not threatening; null trial events |
| `27524657` | food/non-food discrimination task | food | low-caloric food; high-caloric food; non-food |

### 111. infant face images

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: faces (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `28746733` | infant face images | faces | own happy (OH); unknown happy (UH); own sad (OS); unknown sad (US) |
| `31874448` | infant faces and cries | faces | own-happy; own-sad; unknown-happy; unknown-sad (+2) |

### 112. Probabilistic Reward Task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: money (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `29059451` | Probabilistic Reward Task | money |  |
| `32145666` | probabilistic reward task | money |  |

### 113. advertisement viewing

clustered · 2 studies · 2 tasks · 1 distinct names

Stimuli: alcohol (1), food (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `29227227` | advertisement viewing | alcohol | ALCOHOL; CONTROL |
| `29626776` | advertisement viewing | food | sweet/fruit; tobacco; control |

### 114. delay discounting task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (1), money (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `20096794` | delay discounting | money |  |
| `30590514` | delay discounting task | (unspecified) | costly option; default option |

### 115. Cue reactivity task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (1), alcohol (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30991248` | Cue reactivity task | (unspecified) |  |
| `32068323` | cue-reactivity and cue-devaluation task | alcohol |  |

### 116. social concept discrimination task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: social (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `20728544` | Moral Sentiment Task | (unspecified) |  |
| `19153155` | social concept discrimination task | social |  |

### 117. Social Inference — Minimal subtest of The Awareness of Social Inference Test

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: social (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19501175` | Social Inference — Minimal subtest of The Awareness of Social Inference Test | social | Sincere; Simple Sarcasm |
| `26236629` | TASIT social cognition assessment | social |  |

### 118. Pitch discrimination

clustered · 2 studies · 4 tasks · 4 distinct names

Stimuli: (unspecified) (4)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21617528` | Familiar melody pitch error detection | (unspecified) |  |
| `21617528` | Pitch discrimination | (unspecified) |  |
| `29186630` | pitch pattern processing task | (unspecified) |  |
| `21617528` | Unfamiliar melody discrimination | (unspecified) |  |

### 119. saccade tasks

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `33613262` | Oculomotor evaluation | (unspecified) |  |
| `22491196` | saccade tasks | (unspecified) |  |

### 120. moral reasoning task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21432689` | hypothetical romantic partner attractiveness evaluation | (unspecified) | attractive; unattractive; reject; accept (+2) |
| `23576128` | moral reasoning task | (unspecified) | non-moral; moral-impersonal; moral-personal |

### 121. Episodic memory testing

clustered · 2 studies · 2 tasks · 1 distinct names

Stimuli: (unspecified) (1), words (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23670951` | Episodic memory testing | (unspecified) |  |
| `24179835` | Episodic memory testing | words |  |

### 122. empathy attribution task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24589435` | empathy attribution task | emotion | intention attribution; emotion attribution; causal inferences |
| `26594631` | Story-based Empathy Task | (unspecified) |  |

### 123. emotional memory task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `oE9rxptjbz9f` | Emotional Faces Memory Task | emotion | 0-back; 1-back; 2-back |
| `25009480` | emotional memory task | emotion | emotional story; neutral story |

### 124. Emotion Evaluation subtest of The Awareness of Social Inference Test

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (1), social (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25890642` | Emotion Evaluation subtest of The Awareness of Social Inference Test | emotion |  |
| `26513651` | social cognition tasks | social |  |

### 125. virtual supermarket task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25913063` | virtual supermarket task | (unspecified) |  |
| `28697554` | Virtual supermarket task | (unspecified) |  |

### 126. pupillometry experiment

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26889604` | pupillometry experiment | (unspecified) | M+; M− |
| `29186630` | tonal expectancy task | (unspecified) | finished; unfinished |

### 127. research neuropsychological battery

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21691942` | CANTAB cognitive assessment battery | (unspecified) |  |
| `29088311` | research neuropsychological battery | (unspecified) |  |

### 128. classification of FTD, AD and NC

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `33551735` | classification of FTD, AD and NC | (unspecified) |  |
| `JzsUUQbDr2bm` | structural MRI | (unspecified) |  |

### 129. repeat letters during PET uptake

clustered · 2 studies · 3 tasks · 3 distinct names

Stimuli: (unspecified) (2), letters (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `7HPLh5nJzmP5` | intermittent theta-burst stimulation | (unspecified) |  |
| `7HPLh5nJzmP5` | iTBS treatment with fNIRS recording | (unspecified) |  |
| `7DguqkHqbZeG` | repeat letters during PET uptake | letters |  |

### 130. Implicit Sad Facial Affect Recognition Task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (1), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `825vgXr4yySy` | facial affect processing | faces | sad facial expression; crosshair fixation |
| `56zrUuEZ2o6N` | Implicit Sad Facial Affect Recognition Task | emotion | low; medium; high; baseline trials |

### 131. recognition

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (1), faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25502775` | face recognition | faces |  |
| `19945471` | recognition | emotion |  |

### 132. dynamic faces task

clustered · 2 studies · 2 tasks · 1 distinct names

Stimuli: faces (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22163223` | dynamic faces task | faces | happy faces; angry faces; fearful faces; sad faces (+1) |
| `30193355` | dynamic faces task | faces |  |

### 133. sad mood elaboration

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30564105` | emotion-evoking film-clip task | emotion | emotion-evoking (EE); emotion-neutral (EN) |
| `22483075` | sad mood elaboration | emotion | fixation baseline; sad mood elaboration |

### 134. Emotional Anticipation Task

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26806862` | Emotional Anticipation Task | emotion | uncertain event; certain reward; certain threat; baseline |
| `28119507` | sensory shift paradigm | emotion | negative painful; negative baseline; neutral painful; neutral baseline |

### 135. emotion induction task

clustered · 2 studies · 2 tasks · 1 distinct names

Stimuli: emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27838143` | emotion induction task | emotion | disgust; anger; sad; happy (+1) |
| `29741803` | emotion induction task | emotion | emotional clips; neutral clips |

### 136. adapted MRI version of the Anger Articulated Thoughts during Simulated Situations (ATSS) paradigm

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `28620226` | adapted MRI version of the Anger Articulated Thoughts during Simulated Situations (ATSS) paradigm | emotion | anger-engagement; neutral-engagement; happy-engagement; anger-distraction (+2) |
| `29681803` | anger-provoking movie viewing | (unspecified) | high anger; low anger |

### 137. KidVid

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `28737296` | KidVid | emotion | negative; positive; neutral |
| `30537564` | KidVid fMRI task | emotion | positive; negative; neutral |

### 138. Carmageddon virtual violence gameplay

clustered · 2 studies · 2 tasks · 2 distinct names

Stimuli: (unspecified) (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `29948188` | Carmageddon virtual violence gameplay | (unspecified) | violent action; attempted violent action; non-violent action; non-intended action |
| `30853880` | violent and non-violent Carmageddon video game | (unspecified) | Violence; Non-Violence |

### 139. spatial working memory task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `16340649` | spatial working memory | (unspecified) |  |

### 140. fixation task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21466926` | Resting-state fixation | (unspecified) |  |

### 141. acupuncture task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: opioid (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21717780` | heroin-visual stimulation and acupuncture | opioid | heroin-visual stimulation; acupuncture |

### 142. same-different task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: food (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23028988` | same/different perceptual discrimination task | food | high-calorie foods; low-calorie foods; furniture |

### 143. object classification

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `28429068` | object classification task | (unspecified) | cigarettes; pencils |

### 144. Spatial cuing paradigm

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: tobacco (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30844426` | spatial cuing task | tobacco | neutral trials; congruent trials; incongruent trials |

### 145. visually guided saccade task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `16763044` | Visually guided saccade task | (unspecified) |  |

### 146. Moral Dilemma Task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `19499384` | moral dilemmas | emotion | reasoned moral dilemmas; emotional moral dilemmas |

### 147. visual search task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: letters (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21215762` | visual search task | letters |  |

### 148. finger tapping task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25447066` | finger-tapping task | (unspecified) |  |

### 149. passive listening

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25773639` | music emotion processing passive-listening paradigm | emotion | MCM; MFC; MCD; VC (+1) |

### 150. Cambridge Face Memory Test

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25797589` | Cambridge Face Memory Test | faces |  |

### 151. prospective memory task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26682697` | Modified Cambridge Prospective Memory test | (unspecified) |  |

### 152. semantic classification task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26889604` | auditory semantic classification task | (unspecified) |  |

### 153. source memory test

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27771044` | self-reference source memory task | (unspecified) | self-reference; other-reference; perceptual |

### 154. word identification

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `29348485` | emotion identification | emotion |  |

### 155. Memory encoding task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `4iMgZjYRo6KR` | associative memory encoding | faces | attempted encoding; fixation-events |

### 156. target detection task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `8Nkn4bHtgHmz` | target detection task | emotion | sad; neutral |

### 157. gambling task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `PoiCaeLDyEjZ` | gambling task | (unspecified) | win; loss; neutral |

### 158. gender discrimination task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: faces (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `35Eixun6BBdo` | gender discrimination task | faces | fear faces; happy faces; baseline fixation cross |

### 159. face working memory task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: letters (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `12764211` | working memory task | letters | no-load control; static 3-item memory load; spatial 2-back |

### 160. alternating runs paradigm

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21447417` | alternating emotion-identification/digit-sorting task | emotion |  |

### 161. social judgment task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21782901` | emotional judgment task | emotion | neutral; unpleasant; rest |

### 162. meditation task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: words (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23790741` | mindfulness meditation | words | meditation; control task |

### 163. directed forgetting task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26639452` | directed forgetting task | emotion | Remember-Negative (R-NG); Remember-Neutral (R-NE); Forget-Negative (F-NG); Forget-Neutral (F-NE) |

### 164. Emotion Identification Task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27622993` | emotion identification task | emotion | unpleasant images; neutral images |

### 165. recognition memory test

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `29432767` | recognition memory task | emotion |  |

### 166. emotional regulation task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `31060042` | emotional regulation task | emotion |  |

### 167. rubber hand illusion

named by the Cognitive Atlas · 1 studies · 4 tasks · 1 distinct names

Stimuli: (unspecified) (4)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `ngDTY5BgJUuX` | rubber hand illusion | (unspecified) | visuotactile synchronous; visuotactile asynchronous; visuomotor synchronous; visuomotor asynchronous |
| `ngDTY5BgJUuX` | rubber hand illusion | (unspecified) | visuotactile synchronous; visuotactile asynchronous; visuomotor synchronous; visuomotor asynchronous |
| `ngDTY5BgJUuX` | rubber hand illusion | (unspecified) | visuotactile synchronous; visuotactile asynchronous; visuomotor synchronous; visuomotor asynchronous |
| `ngDTY5BgJUuX` | rubber hand illusion | (unspecified) | visuotactile synchronous; visuotactile asynchronous; visuomotor synchronous; visuomotor asynchronous |

### 168. Rapid Visual Information Processing

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `21690575` | Rapid Visual Information Processing Task | (unspecified) |  |

### 169. social decision-making task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23153869` | Decision-Making Task | (unspecified) |  |

### 170. reversal learning task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `24738841` | Probabilistic reversal learning task | emotion |  |

### 171. Wisconsin card sorting test

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25637267` | Wisconsin Card-Sorting Test (WCST) | (unspecified) |  |

### 172. Information Sampling Task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25678093` | Information Sampling Task (IST) | (unspecified) |  |

### 173. balloon analogue risk task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `27575491` | Balloon analogue risk task (BART) | (unspecified) |  |

### 174. set-shifting task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `28918267` | Intra-Extra Dimensional set shifting task | (unspecified) |  |

### 175. Cambridge Gambling Task

named by the Cognitive Atlas · 1 studies · 1 tasks · 1 distinct names

Stimuli: (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `28918267` | Cambridge Gambling Task | (unspecified) |  |

### 176. Ekman 60

clustered · 1 studies · 2 tasks · 2 distinct names

Stimuli: emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `23805313` | Ekman 60 | emotion |  |
| `23805313` | Ekman Caricatures | emotion |  |

### 177. happy film

clustered · 1 studies · 2 tasks · 2 distinct names

Stimuli: emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `25461707` | happy film | emotion | happy film; sad film |
| `25461707` | sad film | emotion |  |

### 178. Experiment 1: static body emotion matching

clustered · 1 studies · 5 tasks · 5 distinct names

Stimuli: faces (3), emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26162615` | Experiment 1: static body emotion matching | emotion |  |
| `26162615` | Experiment 2: dynamic body emotion matching | emotion |  |
| `26162615` | Experiment 3: static face emotion matching | faces |  |
| `26162615` | Experiment 4: dynamic face emotion matching | faces |  |
| `26162615` | Experiment 5 (control experiment): face identity matching | faces |  |

### 179. semantic congruity task

clustered · 1 studies · 4 tasks · 4 distinct names

Stimuli: (unspecified) (3), emotion (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `28811257` | auditory semantic control task | (unspecified) |  |
| `28811257` | emotional congruity task | emotion |  |
| `28811257` | perceptual similarity control task | (unspecified) |  |
| `28811257` | semantic congruity task | (unspecified) |  |

### 180. Letter-guided fluency

clustered · 1 studies · 2 tasks · 2 distinct names

Stimuli: letters (1), (unspecified) (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `29542053` | Category-naming fluency | (unspecified) |  |
| `29542053` | Letter-guided fluency | letters |  |

### 181. categorization task

clustered · 1 studies · 2 tasks · 2 distinct names

Stimuli: food (1), words (1)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `31092880` | categorization task | food |  |
| `31092880` | sensory/functional matching task | words |  |

### 182. Scanner anti-smoking messages task

clustered · 1 studies · 2 tasks · 2 distinct names

Stimuli: tobacco (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `30617213` | Postscanner anti-smoking messages rating task | tobacco |  |
| `30617213` | Scanner anti-smoking messages task | tobacco |  |

### 183. EBA functional localizer

clustered · 1 studies · 4 tasks · 2 distinct names

Stimuli: (unspecified) (4)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `ngDTY5BgJUuX` | EBA functional localizer | (unspecified) |  |
| `ngDTY5BgJUuX` | EBA functional localizer | (unspecified) | body parts; chairs |
| `ngDTY5BgJUuX` | EBA localizer scan | (unspecified) | body parts; chairs |
| `ngDTY5BgJUuX` | EBA localizer scan | (unspecified) | body parts; chairs |

### 184. aversive and nonaversive image viewing with repetition

clustered · 1 studies · 2 tasks · 2 distinct names

Stimuli: emotion (2)

| study | task name | stimulus | conditions |
|---|---|---|---|
| `22453299` | aversive and nonaversive image viewing with repetition | emotion |  |
| `22453299` | main activation task | emotion |  |

---

## In no category — 142 tasks

Nothing else in the corpus is within the cut. Two kinds are mixed here and want different
handling: paradigms genuinely run by one paper, and tasks that belong to a category above and
missed the cut. The singleton rescue pass is deliberately not run.

| study | task name | stimulus | conditions |
|---|---|---|---|
| `26810632` | Ad libitum breakfast test meal and dessert experiment | (unspecified) |  |
| `25280904` | affect-related regulation task | emotion |  |
| `26748236` | art emotion test | emotion |  |
| `22981242` | attempted inhibition of cue-induced craving | cocaine |  |
| `27630558` | Auditory perceptual tasks | (unspecified) |  |
| `28811257` | auditory scene control task | (unspecified) |  |
| `25396740` | Autobiographical Interview | (unspecified) | Free Recall; General Probe; Specific Probe |
| `29146290` | baby schema task | faces | Low; Unmanipulated; High |
| `21697710` | Balloon Analog Risk Task | money |  |
| `27217113` | baseline emotional reactivity localizer | emotion |  |
| `25678093` | Beads task | (unspecified) |  |
| `22009019` | behavioral coordination task | (unspecified) | survey; coordination |
| `19369561` | behavioral orientation affordance task | tobacco |  |
| `32678839` | beverage-related images | alcohol | juice; drinking juice; sake; drinking sake |
| `24238299` | body image paradigm | (unspecified) | body image; non-body; LLB |
| `25797589` | Cambridge Car Memory Test | (unspecified) |  |
| `25887154` | cigarette warning labels task | tobacco | graphic labels; non-graphic labels |
| `21466926` | Cocaine and food cue-reactivity task | cocaine |  |
| `16340649` | cocaine self-administration | cocaine |  |
| `25658479` | cognitive appetite control | food | inhibition; passive viewing; imaginary eating |
| `28755988` | cognitive challenge tasks | words | Stroop Word Color; Dual 1-Back |
| `31730369` | cognitive control | words | incongruent trials; congruent trials |
| `18330460` | cognitive-behavioural therapy for PTSD | (unspecified) |  |
| `28715908` | concurrent TMS-fMRI | (unspecified) |  |
| `14697007` | Confrontation naming | (unspecified) |  |
| `29963008` | conversational laughter | (unspecified) |  |
| `29058352` | creativity tasks | (unspecified) | RGT; AUT |
| `22288977` | Cue exposure/coping task | tobacco |  |
| `14679386` | Cue-induced MRI scanning procedure | alcohol |  |
| `22835330` | D-KEFS Design Fluency Test | (unspecified) |  |
| `27575491` | Delay discounting task (DDT) | money |  |
| `xevP8UDRAVh9` | drug-associated cerebral perfusion | drug | heroin; placebo |
| `27389802` | dynamic facial expressions | faces | fearful; chewing |
| `18093623` | D–KEFS Tower task | (unspecified) |  |
| `28698012` | ecological momentary assessment | (unspecified) |  |
| `23146247` | emotion processing task | emotion | exp ng; exp nt; perc ng; perc nt |
| `31733523` | Emotion Self-Other Morph Neurofeedback (ESOM_NF) | emotion |  |
| `23563850` | emotional expectation paradigm | emotion |  |
| `30193355` | emotional face 2-back task | emotion | fearful faces; happy faces; neutral faces; no faces |
| `Qb4NaXyjtAnF` | emotional state rating | emotion | before levodopa infusion while taking oral SYN115; during levodopa infusion while taking oral SYN115; before levodopa infusion while taking placebo pills; during levodopa infusion while taking placebo pills |
| `WbrbHFVtSBAK` | empathy for pain task | pain | effective; ineffective |
| `20089342` | environmental sound naming | (unspecified) |  |
| `21520350` | episodic FOK task | words |  |
| `23516294` | ER task | emotion |  |
| `23516294` | eWM training task | faces |  |
| `22184615` | executive function tests | (unspecified) |  |
| `27298765` | Experiment 3: familiarity categorization and famous face-name matching | faces |  |
| `29906489` | experimental runs | food | food; unhealthy foods; healthy foods; objects |
| `26707083` | Experimental task | (unspecified) |  |
| `28872745` | extinction learning task | drug |  |
| `23966929` | Face recognition | faces |  |
| `21617528` | Familiar melody title recall | (unspecified) |  |
| `31079279` | faux pas task | (unspecified) | faux pas stories |
| `28483719` | fear conditioning task | (unspecified) |  |
| `30964611` | Fearful and neutral face movie viewing | emotion |  |
| `25339705` | feedback-based learning task | money |  |
| `16340649` | finger sequencing | (unspecified) |  |
| `23138765` | flavour identification task | words | flavour identification trials |
| `17986612` | Food image viewing and hunger rating task | food |  |
| `29408590` | Food portion size cue fMRI task | food |  |
| `23924756` | food video paradigm | food | food video; neutral video |
| `27845255` | Food-choice satiety task | food |  |
| `19707568` | Free and Cued Selective Reminding Test | words |  |
| `17909155` | free-feeding study | food | variety condition; nonvariety condition |
| `22180700` | Frog, Where Are You narrative | (unspecified) |  |
| `29226482` | Frustration Emotion Task for Children (FETCH) | emotion | win trials; frustration trials |
| `29025878` | Gustatory food-cue task | food |  |
| `30884367` | Happé-Frith animation task | (unspecified) | random; goal directed; ToM |
| `30317048` | Helping task | (unspecified) |  |
| `23803881` | high- and low-GI test meals | (unspecified) | high-GI meal; low-GI meal |
| `25973788` | humour decision task | (unspecified) |  |
| `31046591` | IAPS passive picture viewing | emotion |  |
| `25797589` | Identity-Matching task | faces |  |
| `25123156` | IED: CANTAB | (unspecified) |  |
| `30225341` | impression-formation task | faces |  |
| `28373956` | intentional emotional expression task | emotion | verbal command condition; picture imitation condition |
| `21220070` | Internet video game cue | gaming | Internet video game stimuli; neutral stimuli; resting |
| `27427215` | Interoceptive Attention task | words | interoception; exteroception |
| `pHxUfNVDVMBp` | intrinsic functional connectivity | (unspecified) |  |
| `26475487` | localiser task | emotion |  |
| `27252632` | Memory and Temporal Experience Questionnaire | (unspecified) |  |
| `30718430` | mind wandering thought-sampling task | (unspecified) |  |
| `4eEpNcRPsxPi` | modified BART | (unspecified) | loss outcomes |
| `29463908` | modified forced choice preference task | stimulant | methamphetamine-paired cues; placebo-paired cues |
| `25678093` | Monetary Choice Questionnaire | money |  |
| `26432341` | Moral behavior assessment | (unspecified) |  |
| `25047907` | moral judgment task | (unspecified) | no harm; accidental harm; attempted harm; successfully attempted harm |
| `30565822` | MSIT-IAPS task | emotion | Interference; Non-Interference; Negative Interference; Negative Non-Interference |
| `23107380` | music mentalising task | words | mentalising; non-mentalising |
| `26025509` | NEPSY-II Inhibition subtest | (unspecified) |  |
| `22952324` | Neuropsychological testing and self-appraisal | (unspecified) |  |
| `10103096` | neutral and drug-salient videos | drug | drug video; neutral video |
| `24145410` | novelty-induced hypophagia task | (unspecified) |  |
| `26424424` | odor sampling | tobacco |  |
| `20698837` | olfactory stimulation | food | AO; ApCO |
| `11431229` | olfactory stimulation with ethanol odor and neutral room air | (unspecified) | ethanol; neutral room air |
| `30991248` | Online working-memory training | (unspecified) |  |
| `18774224` | painful electrical stimulation with religious and non-religious images | (unspecified) | religious condition; non-religious condition |
| `21646526` | passive art-viewing sponsorship task | (unspecified) |  |
| `23582296` | Past–future task | words |  |
| `7ysELhgb8aYp` | phMRI infusion protocol | (unspecified) | infusion |
| `17137758` | pictorial memory fMRI-paradigm | (unspecified) | AL; SC; RE; RS |
| `29058369` | picture stimulus task | alcohol |  |
| `21930136` | Quantifier comprehension task | (unspecified) |  |
| `23966929` | Reading the Mind in the Eyes Test | (unspecified) |  |
| `20445032` | rejecter and neutral photograph viewing with countback distraction | (unspecified) | rejecter; neutral |
| `18940597` | Relative preference task | faces |  |
| `25365800` | Reward Prediction Task | money |  |
| `29053832` | Reward task | money |  |
| `31993653` | sad film clip | emotion | pre-film baseline; sad film clip |
| `31887311` | scene construction task | (unspecified) |  |
| `24290886` | sch-evaluation and cc-evaluation | (unspecified) | sch-evaluation; cc-evaluation; rest |
| `20071459` | script event order judgment | (unspecified) | Within-Hierarchy; Different-Hierarchy |
| `28388995` | script-driven imagery task | (unspecified) |  |
| `23565102` | sentence-picture verification task | (unspecified) |  |
| `22517323` | sequential risk-taking task | (unspecified) | optimal; nonoptimal |
| `32553642` | single-trial naturalistic appetitive conditioning | food | CS+; CS- |
| `QQCjAAT6SwwQ` | sleep-wake recording | (unspecified) |  |
| `23708507` | smoking abstinence | tobacco |  |
| `21199958` | smoking cue exposure | tobacco |  |
| `16763044` | Smooth pursuit eye movement task | (unspecified) |  |
| `25619850` | social coordination task | social | common ground; colorblind; privileged ground |
| `29462404` | stressor and stress recovery | stress | stressor; stress recovery |
| `22006991` | Subliminal and supraliminal sexual and neutral picture prime liking task | sexual |  |
| `25637267` | switching task | (unspecified) |  |
| `28662418` | TASIT-S | emotion |  |
| `17532012` | TEMPau task | (unspecified) |  |
| `23748383` | three-choice visual discrimination problems | (unspecified) | acquisition; retention; reversal |
| `26932257` | threshold-tracking transcranial magnetic stimulation | (unspecified) |  |
| `32546139` | Transfer | (unspecified) |  |
| `30287300` | transfer task | emotion |  |
| `28811259` | Trust game memory task | faces |  |
| `26733906` | two-armed bandit task | (unspecified) | bandit choice |
| `25637267` | two-back test | (unspecified) |  |
| `24727905` | two-player cooperative games | tobacco | SECig; SEnoCig; NSECig; NSEnoCig |
| `25013940` | video-rating task | (unspecified) |  |
| `19129383` | viewing appetizing and bland foods | (unspecified) | appetizing; bland; null events |
| `28374078` | visual assessment | (unspecified) |  |
| `23061530` | Visual dot-probe attentional bias task | tobacco |  |
| `32713944` | visual food images | food |  |
| `24179759` | Weather Prediction Task | (unspecified) |  |
| `23142417` | working memory | (unspecified) |  |
