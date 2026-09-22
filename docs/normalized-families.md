# Normalized task families and medical conditions

Over the 100 re-extracted defect papers. Produced by `pondie.normalization.task.normalize` and `pondie.normalization.medical_condition.normalize` -- the same calls `pondie normalize <field>` makes, so these lists and that report agree.

## Tasks

**90 extracted task names -> 36 identities -> 31 families.** (2 singleton(s) rescued into a family.)

Two grains, and the difference is the point. An **identity** is the same task under different wording -- `n-back` and `N-back working memory task`. A **family** is tasks a meta-analysis might pool -- several inhibition tasks, say -- and is deliberately looser. A family holding one identity means nothing else in this corpus was judged close enough.

> **Read these as a draft, not an answer.** 21 of the 36 identities hold more than one
> name, and at least 9 of those 21 have absorbed a name that is plainly a different task:
> `resting-state protocol` took in `executive functions`; `stop signal task` took in
> `involuntary memory task`, `novelty oddball task` and `sustained attention task`;
> `spatial working memory task` took in `Conners' Continuous Performance Test` and
> `Learning`; `Reading the Mind in the Eyes test` took in `Attention Network Test`;
> `Emotional arousal` took in `Language`.
>
> This is the cluster shape working as designed on a corpus too small for it. `normalize`
> trains its pair classifier by distant supervision on connected components of three or
> more, and over 90 tasks there are barely any -- so the model it fits is weak and the
> distance threshold does the deciding. The remedy is more corpus or a curated target,
> not a threshold tweak: see [task-condition-normalization.md](task-condition-normalization.md),
> which argues against a flat task vocabulary, and [task-schema-proposal.md](task-schema-proposal.md).
>
> The conditions below do not have this problem, because `medical_condition` is the link
> shape: each head is looked up in MONDO independently, so a wrong answer is one row
> wrong rather than a cluster wrong.


### 1. resting-state protocol

19 tasks · 19 studies · 1 identity

- **resting-state protocol** — `resting-state protocol` ×5, `resting-state` ×4, `resting state` ×3, `resting-state functional connectivity paradigm`, `awake and relaxed resting-state protocol`, `rest condition`, `resting-state fMRI bladder filling paradigm`, `resting-state night sleep`, `executive functions`, `resting-state fMRI`

### 2. stop signal task

9 tasks · 9 studies · 2 identities

- **stop signal task** — `stop signal task`, `facial-emotion go/no-go task`, `involuntary memory task`, `novelty oddball task`, `Go/No-go tasks`, `sustained attention task`, `fMRI stop task`
- **auditory discrimination task** — `auditory discrimination task`, `Auditory`

### 3. spatial working memory task

8 tasks · 7 studies · 2 identities

- **spatial working memory task** — `spatial working memory task`, `Conners' Continuous Performance Test`, `Working memory`, `Learning`, `N-back working memory task`, `digit order and letter order tasks`
- **Pediatric Affective Color Matching Task** — `Pediatric Affective Color Matching Task`, `lexical tone discrimination and font size judgment`

### 4. lexical decision task

7 tasks · 6 studies · 2 identities

- **lexical decision task** — `lexical decision task`, `Semantic List Learning Task`, `passive semantic language task`, `word generation task`
- **olfactory stimulation** — `olfactory stimulation`, `PET sensory stimulation`, `fMRI sensory stimulation`

### 5. picture matching task

7 tasks · 6 studies · 3 identities

- **picture matching task** — `picture matching task`
- **Reading the Mind in the Eyes test** — `Reading the Mind in the Eyes test`, `gaze discrimination task`, `Attention Network Test`
- **illusory correlation experiment** — `illusory correlation experiment`, `speaking experiment`, `control experiment`

### 6. causal attribution task

3 tasks · 3 studies · 1 identity

- **causal attribution task** — `causal attribution task`, `word judgment task`, `content judgement task`

### 7. Sensory-motor

3 tasks · 3 studies · 1 identity

- **Sensory-motor** — `Sensory-motor`, `motor task`, `sensorimotor tasks`

### 8. Alternative Uses Task

2 tasks · 2 studies · 1 identity

- **Alternative Uses Task** — `Alternative Uses Task`, `tactile object recognition`

### 9. beads in the bottle task

2 tasks · 2 studies · 1 identity

- **beads in the bottle task** — `beads in the bottle task`, `fluid reasoning task`

### 10. dynamic localizer

2 tasks · 1 study · 1 identity

- **dynamic localizer** — `dynamic localizer`, `static face localizer`

### 11. Emotional arousal

2 tasks · 1 study · 1 identity

- **Emotional arousal** — `Emotional arousal`, `Language`

### 12. fear conditioning, retrieval extinction and test

2 tasks · 2 studies · 1 identity

- **fear conditioning, retrieval extinction and test** — `fear conditioning, retrieval extinction and test`, `Approach-Avoidance task`

### 13. painful and nonpainful pinprick stimulation

2 tasks · 2 studies · 1 identity

- **painful and nonpainful pinprick stimulation** — `painful and nonpainful pinprick stimulation`, `noxious thermal stimuli`

### 14. prosodic phrase comprehension

2 tasks · 1 study · 1 identity

- **prosodic phrase comprehension** — `prosodic phrase comprehension`, `ERP phrase comprehension`

### 15. self-processing paradigms

2 tasks · 1 study · 1 identity

- **self-processing paradigms** — `self-processing paradigms`, `familiarity paradigms`

### 16. virtual social interaction paradigm

2 tasks · 2 studies · 1 identity

- **virtual social interaction paradigm** — `virtual social interaction paradigm`, `video-clip social interaction task`

### 17. visual motion paradigm

2 tasks · 2 studies · 1 identity

- **visual motion paradigm** — `visual motion paradigm`, `Visual`

### Families of one — 14 tasks nothing else matched

One study each, and no other task in the corpus was judged close enough to pool with. On a corpus this size that is the expected outcome for a task only one paper ran, not evidence the task is unusual.

- `action observation`
- `Cyberball social exclusion`
- `delayed match-to-sample`
- `modified mental clock task`
- `Monetary Choice Questionnaire`
- `MTrP compression experiment`
- `Neuropsychological testing`
- `script-driven imagery paradigm`
- `Stroop`
- `SVF test`
- `swallowing`
- `TAP attention battery`
- `TASIT`
- `voice emotion recognition task`

## Medical conditions

**208 condition heads** off the extracted `Group.medical_condition` values. A head is one condition: `triage` splits a value that names several and drops one that names an absence, so the counts below are conditions rather than strings.

### Linked to MONDO — 59 heads on 39 distinct terms

| MONDO term | studies | ONVOC (how) | the wording it absorbed |
|---|---:|---|---|
| **schizoaffective disorder** <br><sub>MONDO:0005487</sub> | 7 | Psychotic Disorder (ancestor) | `schizoaffective disorder` ×4; `Schizoaffective disorder` ×3 |
| **schizophrenia** <br><sub>MONDO:0005090</sub> | 7 | Schizophrenia (crosswalk) | `schizophrenia` ×4; `Schizophrenia` ×3 |
| **major depressive disorder** <br><sub>MONDO:0002009</sub> | 4 | Depressive Disorder (ancestor) | `major depressive disorder` ×2; `Major depression`; `major depression` |
| **Alzheimer disease** <br><sub>MONDO:0004975</sub> | 3 | Alzheimer's Disease (crosswalk) | `Alzheimer's disease` ×2; `AD` |
| **bipolar I disorder** <br><sub>MONDO:0001866</sub> | 2 | Bipolar Disorder (crosswalk) | `bipolar disorder type I`; `bipolar I disorder` |
| **progressive supranuclear palsy** <br><sub>MONDO:0019037</sub> | 2 | — *not in ONVOC* | `progressive supranuclear palsy` ×2 |
| **attention deficit hyperactivity disorder, inattentive type** <br><sub>MONDO:0005302</sub> | 1 | Attention-Deficit Hyperactivity Disorder (ancestor) | `Attention-Deficit` |
| **autism spectrum disorder** <br><sub>MONDO:0005258</sub> | 1 | Autism Spectrum Disorder (crosswalk) | `autism spectrum disorders` |
| **bipolar disorder** <br><sub>MONDO:0004985</sub> | 1 | Bipolar Disorder (crosswalk) | `bipolar disorder` |
| **bipolar II disorder** <br><sub>MONDO:0000693</sub> | 1 | Bipolar Disorder (ancestor) | `bipolar disorder type II` |
| **brain neoplasm** <br><sub>MONDO:0021211</sub> | 1 | Brain Tumor (crosswalk) | `brain tumor` |
| **cocaine dependence** <br><sub>MONDO:0005186</sub> | 1 | Substance Dependence (ancestor) | `cocaine dependence` |
| **corticobasal degeneration disorder** <br><sub>MONDO:0022308</sub> | 1 | — *not in ONVOC* | `corticobasal degeneration` |
| **delusional disorder** <br><sub>MONDO:0004359</sub> | 1 | Psychotic Disorder (ancestor) | `delusional disorder` |
| **dysthymic disorder** <br><sub>MONDO:0001442</sub> | 1 | Mood Disorder (ancestor-name) | `dysthymia` |
| **epilepsy** <br><sub>MONDO:0005027</sub> | 1 | Epilepsy (crosswalk) | `epilepsy` |
| **frontotemporal dementia** <br><sub>MONDO:0017276</sub> | 1 | Dementia (ancestor) | `frontotemporal dementia` |
| **hemorrhagic stroke** <br><sub>MONDO:1060199</sub> | 1 | Stroke (ancestor) | `hemorrhagic stroke` |
| **herpes simplex encephalitis** <br><sub>MONDO:0012521</sub> | 1 | — *not in ONVOC* | `herpes simplex encephalitis` |
| **insomnia** <br><sub>MONDO:0013600</sub> | 1 | Sleep Disturbances (ancestor) | `insomnia` |
| **ischemic stroke** <br><sub>MONDO:1060198</sub> | 1 | Stroke (ancestor) | `ischemic stroke` |
| **Lewy body dementia** <br><sub>MONDO:0007488</sub> | 1 | Dementia (ancestor) | `DLB` |
| **metabolic dysfunction-associated steatotic liver disease** <br><sub>MONDO:0013209</sub> | 1 | Liver Disease (ancestor) | `non-alcoholic fatty liver disease` |
| **migraine disorder** <br><sub>MONDO:0005277</sub> | 1 | Migraine (crosswalk) | `migraine` |
| **mood disorder** <br><sub>MONDO:0005371</sub> | 1 | Mood Disorder (name) | `mood disorder` |
| **obsessive-compulsive disorder** <br><sub>MONDO:0008114</sub> | 1 | Obsessive-Compulsive Disorder (crosswalk) | `obsessive compulsive disorder` |
| **panic disorder** <br><sub>MONDO:0005383</sub> | 1 | Panic Disorder (crosswalk) | `panic disorder` |
| **paranoid schizophrenia** <br><sub>MONDO:0001484</sub> | 1 | Schizophrenia (ancestor) | `Paranoid schizophrenia` |
| **progressive non-fluent aphasia** <br><sub>MONDO:0015059</sub> | 1 | Dementia (ancestor) | `progressive nonfluent aphasia` |
| **schizoid personality disorder** <br><sub>MONDO:0001161</sub> | 1 | Personality Disorders (ancestor) | `Schizoid Personality Disorder` |
| **schizophreniform disorder** <br><sub>MONDO:0001265</sub> | 1 | Psychotic Disorder (ancestor) | `Schizophreniform disorder` |
| **schizotypal personality disorder** <br><sub>MONDO:0001087</sub> | 1 | Personality Disorders (ancestor) | `Schizotypal Personality Disorder` |
| **semantic dementia** <br><sub>MONDO:0010857</sub> | 1 | Dementia (ancestor) | `semantic dementia` |
| **somatization disorder** <br><sub>MONDO:0001830</sub> | 1 | — *not in ONVOC* | `somatization disorder` |
| **specific phobia** <br><sub>MONDO:0012000</sub> | 1 | Phobia (ancestor) | `specific phobia` |
| **spinocerebellar ataxia type 2** <br><sub>MONDO:0008458</sub> | 1 | Motor Neuron Disease (ancestor) | `spinocerebellar ataxia type 2` |
| **systemic lupus erythematosus** <br><sub>MONDO:0007915</sub> | 1 | — *not in ONVOC* | `systemic lupus erythematosus` |
| **temporal lobe epilepsy** <br><sub>MONDO:0005115</sub> | 1 | Epilepsy (ancestor) | `epilepsy of the temporal lobe` ×2 |
| **traumatic brain injury** <br><sub>MONDO:0858950</sub> | 1 | Traumatic Brain Injury (ancestor) | `traumatic brain injury` |

### Not linked — 36 for review, 36 rejected

`review` is a near miss the threshold would not accept on its own; `rejected` is nothing in MONDO came close. Nothing is bucketed silently -- both are reported so a missing term forces a decision rather than disappearing.

**For review**

- `ADHD` ×5 (score 0.91) — nearest: attention deficit-hyperactivity disorder, susceptibility to, 1
- `cognitive impairment` ×3 (score 0.88) — nearest: cognitive disorder
- `unipolar major depression` ×2 (score 0.95) — nearest: major depressive disorder
- `posttraumatic stress disorder` ×2 (score 0.92) — nearest: post-traumatic stress disorder
- `subjective cognitive impairment` ×2 (score 0.92) — nearest: subjective cognitive decline
- `temporal lobe epilepsy with hippocampal sclerosis` ×2 (score 0.91) — nearest: mesial temporal lobe epilepsy with hippocampal sclerosis
- `aquaporin-4 antibody-positive neuromyelitis optica spectrum disorder` (score 0.95) — nearest: neuromyelitis optica spectrum disorder with anti-AQP4 antibodies
- `Hyperactivity Disorder (ADHD)` (score 0.94) — nearest: attention deficit-hyperactivity disorder
- `AD dementia` (score 0.94) — nearest: Alzheimer disease
- `clinical depression` (score 0.94) — nearest: depressive disorder
- `posttraumatic stress disorder (PTSD)` (score 0.94) — nearest: post-traumatic stress disorder
- `developmental dyslexia` (score 0.93) — nearest: dyslexia
- `bipolar disorder, types I and II` (score 0.92) — nearest: bipolar II disorder
- `Alcohol Use Disorder` (score 0.90) — nearest: alcohol-related disorders
- `amnestic cognitive impairment` (score 0.90) — nearest: amnestic disorder
- `schizophrenia spectrum disorder` (score 0.90) — nearest: schizophrenia
- `nicotine abuse` (score 0.89) — nearest: nicotine dependence
- `frontotemporal lobar degeneration` (score 0.89) — nearest: frontotemporal dementia
- `trauma exposure` (score 0.88) — nearest: injury
- `Trauma exposure` (score 0.88) — nearest: injury
- `reading difficulty` (score 0.88) — nearest: reading disorder
- `cocaine use disorder` (score 0.87) — nearest: cocaine abuse
- `asymptomatic Moyamoya disease` (score 0.87) — nearest: Moyamoya disease
- `developmental dyscalculia` (score 0.86) — nearest: dyscalculia
- `Restrictive anorexia nervosa` (score 0.86) — nearest: anorexia nervosa
- `inattentive subtype` (score 0.85) — nearest: attention deficit hyperactivity disorder, inattentive type

**Rejected**

- `autosomal dominant Alzheimer’s disease mutation carrier` ×2 (score 0.77) — nearest: Alzheimer disease 2
- `normally-hearing` ×2 (score 0.65) — nearest: hearing loss disorder
- `cortical sensorimotor stroke` ×2 (score 0.69) — nearest: stroke disorder
- `tobacco smoking` ×2 (score 0.76) — nearest: tobacco addiction, susceptibility to
- `low back pain` ×2 (score 0.70) — nearest: pelvis syndrome
- `early dementia` (score 0.83) — nearest: Alzheimer disease
- `other dementias` (score 0.84) — nearest: dementia
- `At risk for bipolar disorder` (score 0.70) — nearest: bipolar disorder
- `Healthy and normosmic` (score 0.65) — nearest: normokalemic periodic paralysis
- `occipital cerebral hemorrhage from rupture of an arteriovenous malformation` (score 0.63) — nearest: arteriovenous malformations of the brain
- `right posterior cerebral artery infarction` (score 0.81) — nearest: posterior cerebral artery infarction
- `right middle cerebral artery infarction` (score 0.78) — nearest: middle cerebral artery infarction
- `developmental phonological dyslexia` (score 0.84) — nearest: dyslexia
- `history of drug abuse and dependence` (score 0.72) — nearest: drug dependence
- `history of alcohol abuse and dependence` (score 0.80) — nearest: alcohol dependence
- `genetic risk of bipolar disorder` (score 0.66) — nearest: bipolar disorder
- `psychosis induced through psychoactive substances` (score 0.81) — nearest: substance-induced psychosis
- `acquired brain injury` (score 0.83) — nearest: traumatic brain injury
- `right hemispheric focus` (score 0.58) — nearest: focal epilepsy
- `left hemispheric focus` (score 0.59) — nearest: restricted to specific location
- `very low birth weight` (score 0.74) — nearest: fetal growth restriction
- `preterm birth` (score 0.66) — nearest: neonate prematurity/dysmaturity, non-human animal
- `non-affective psychosis` (score 0.78) — nearest: psychotic disorder
- `Congenital blindness` (score 0.81) — nearest: blindness (disorder)
- `Good sleep quality and regular sleep habits` (score 0.67) — nearest: circadian rhythm sleep disorder, irregular sleep wake type
- `congenital deafness` (score 0.78) — nearest: sensorineural hearing loss disorder
- `Typically developing adolescents` (score 0.64) — nearest: developmental disability
- `subthreshold depression` (score 0.79) — nearest: neurotic depression
- `combined hyperactive` (score 0.75) — nearest: combined hyperactive dysfunction syndrome of the cranial nerves
- `impulsive and inattentive subtype` (score 0.79) — nearest: attention deficit hyperactivity disorder, inattentive type
- `vestibular migraine` (score 0.77) — nearest: vestibular neuronitis

### Values that state an absence

Counted, not linked: a cohort described as free of a condition is an answer, and `is_healthy` is derived from it. See [condition-normalization.md](condition-normalization.md).

- `single`: 127
- `NO_CONDITION`: 77
- `compound`: 2

