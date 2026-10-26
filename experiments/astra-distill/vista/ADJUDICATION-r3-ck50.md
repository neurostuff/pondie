# Adjudication: QLoRA Qwen3.8-27B (r3-27b-ck50) vs Astra, validation split (read-only, no GPU, test split untouched)

Sample: 25 val papers, seed 11, from papers with linked cell F1 < 0.6 (13 drawn of 45 eligible) and >= 0.6 (12 drawn of 40 eligible); papers where model and Astra agree exactly were skipped. Method and dump script reused from h34-10k (adj/dump.py, per-paper dumps adj/d_<sid>.txt, adj/sample.json, quote checker adj/vq.py). 46 disagreements judged against the pubget text.txt; offsets are char offsets in that file and were checked with text[off:off+len(q)]==q (all True; quotes contain the file's own whitespace/tabs).

Low-F1: caTme4KSzp34 obVXFvscgkTW fzfEymXD8DuP zrWff5sJKCrb kFTdpnRj5N8T ouZjktvZh2wW LCuhWT87Vsa5 HETbPtxfHWPq wRM5PyEyTezN gYhqRTaVZGZn tDJu8SHfMKQh 6B6zGj6nHrUD wubMcWUYibqb
High-F1: RwmhBN3dP82k BSGoTkY8SLJN 6teXKZNVzJ62 jHSNMakGotVy 3bFL4tESYK8c bJrWYwKNV7xM f5Zu2kabHbnj DQixkiizyFQN 32WaBSTPCZYm d38NZN3N36E6 PsrZY8awrRRn ovMNArvAW7ok

## Counts (type x judgment)

| type | astra_right | model_right | both_acceptable | both_wrong | unclear | total |
|---|---|---|---|---|---|---|
| (a) effect cell / direction / level | 7 | 0 | 4 | 0 | 1 | 12 |
| (b) model or term | 4 | 1 | 11 | 0 | 0 | 16 |
| (c) analysis on another key or missing | 2 | 3 | 4 | 0 | 1 | 10 |
| (d) group / region / n / name | 0 | 1 | 7 | 0 | 0 | 8 |
| all | 13 | 5 | 26 | 0 | 2 | 46 |

Model share right among decided (astra_right + model_right = 18): 5/18 = 28%. Model not wrong (model_right + both_acceptable) among judged excluding unclear: 31/44 = 70%.
(wubMcWUYibqb has two (a) items, so 46 items for 25 papers.)

## Tables
No judgment depended on a shifted pubget table. Tables read: jHSNMakGotVy Table 1 (header Group difference / Nr of voxels / X Y Z / Significant correlation; values consistent), f5Zu2kabHbnj Table 3 (Contrast / Region / x y z / P / K / Z, consistent), tDJu8SHfMKQh Table 3 (Seed point / Brain region / X Y Z / T / Cluster size, consistent), PsrZY8awrRRn Table 2 (Z value column consistent).

## Patterns
Model wrong (astra_right, 13):
- Imports a signal Astra drops or the text withholds: positive direction for correlations the paper reports as null (LCuhWT87Vsa5), a covariate (age) added to a plain one-sample test (3bFL4tESYK8c), invented extra analysis from a dipole-location coordinate (d38NZN3N36E6).
- Wrong direction or comparison for a table-defined contrast: direction flip AD-vs-elderly (PsrZY8awrRRn), cluster direction contradicting "p > c" (jHSNMakGotVy), group x time interaction instead of a within-high-group time effect (wRM5PyEyTezN), omnibus interaction given directions (wubMcWUYibqb), correlation direction/term (wubMcWUYibqb).
- Drops a restricted-condition "held" cell (RwmhBN3dP82k Visit 2) or a separately reported inference (SVC peak, DQixkiizyFQN).
- Model level: first-level vs second-level (gYhqRTaVZGZn), ancova vs flexible factorial/anova (wRM5PyEyTezN), simplified mixed-effects vs integrated group model (caTme4KSzp34).
Astra wrong (model_right, 5): analyses Astra omits though the paper reports them (Cox regressions in BSGoTkY8SLJN, gPPI cerebellum cluster in 6teXKZNVzJ62), an Astra analysis with no basis in the paper's table (f5Zu2kabHbnj "Words > Partial"), an analysis name contradicting Astra's own cells (BSGoTkY8SLJN "PD-Compl > PD-no-Compl (DMN)" with PD-Compl negative), and the regressors actually entered in a step-wise regression (32WaBSTPCZYm).
Both acceptable (26): mostly naming/detail; the scorer penalises all of these (strict term names, model gate, per-key analysis counts).

## model_right items (5)

1. 6teXKZNVzJ62, (c), key analysis#050b8851f19450c3 (right cerebellum peak 4,-72,-8). Astra: no analysis. Model: males > females connectivity, sex cells males +, females -. Quote @826: "Functional connectivity between Exner's area and the right cerebellum was greater in males than in females."
2. BSGoTkY8SLJN, (c), keys analysis#8ce7f8c837d03ec3 (left cuneus) and #ef1c872f7eb1fa36 (MFG). Astra: none. Model: multivariate Cox regression, cuneus negative, MFG positive. Quote @22800: "Cox regression models indicated that functional connectivity within the MFG"
3. f5Zu2kabHbnj, (c), key analysis#fec8b7f1968a0ad4 (table name "Initial > Final"). Astra has two analyses: "Initial > Final" and "Words > Partial" (words +, initial -, final -); model only the first. Table 3 has "Partial > Words" and "Initial > Final", no "Words > Partial". Quote @32284: "Contrast\tPartial > Words\tPartial > Words\tPartial > Words\tPartial > Words\tInitial > Final" (header checked against values).
4. BSGoTkY8SLJN, (d), analysis name on key analysis#76031db2553a5c5f. Astra "PD-Compl > PD-no-Compl (DMN)" with PD-Compl negative, PD-no-Compl positive; model "PD-Compl < PD-no-Compl (DMN)", same cells. Quote @1359: "a decreased connectivity in the left cuneus within the default-mode network were detected in PD-Compl compared with PD-no-Compl"
5. 32WaBSTPCZYm, (b), analysis#0516014560ea0e7b. Astra terms: SCR performance plus amygdala time series and eight preprocessing nuisance terms (no UCS expectancy term). Model terms: UCS expectancy performance, SCR performance. Quote @16409: "We ran a step-wise multiple regression using the AFNI command 3dRegAna with UCS expectancy and SCR performance as regressors" (Astra's nuisance terms are preprocessing, @13911; judgment is partial).

## both_acceptable items (26)

Format: sid | type | Astra vs model | kind | quote @offset.

1. obVXFvscgkTW | b | term "image category" vs "food cue preceding taste" (cells identical) | naming | "Significant differences between taste perception following high-calorie and low-calorie food viewing" @17611
2. fzfEymXD8DuP | a | level "double-cue" (red-green+green-red pooled, +) vs two cells red-green +, green-red + | contrast term vs separate cells | "we selected voxels within each ROI that showed a significant effect for the double-cue condition" @12399
3. fzfEymXD8DuP | b | model "single subject analysis within regions of interest" vs "general linear model for event-related designs" | naming | "The data were analyzed using the general linear model for event-related designs" @7197
4. zrWff5sJKCrb | b | term "music-taste condition" vs "music-taste interaction condition" (cells identical; model also uses trm_ ids) | naming | "greater activation during concurrent auditory–gustatory stimulation compared to taste alone" @22000
5. kFTdpnRj5N8T | a | Condition 2 t-test: Astra stimulus SNF +, ONF - cells; model holds conditions "SNF vs. ONF" | contrast term vs separate cells | "in Condition 2 (SNF vs. ONF)" @22907
6. kFTdpnRj5N8T | d | levels "Condition 1/2/3" vs "SNF vs. BCP/ONF/ODF" | naming | "Condition 1: SNF vs. BCP, Condition 2: SNF vs. ONF, and Condition 3: SNF vs. ODF" @18604
7. kFTdpnRj5N8T | b | Astra adds a "stimulus" term to the ANCOVA terms | naming/detail | "a two-way analysis of covariance (ANCOVA) was used to examine the main effect of diagnosis (FEMDD and HC) and conditions respectively" @18716
8. ouZjktvZh2wW | c | one "Interaction" analysis (key analysis#1dd8b7a536407d36) vs per-network analyses (LFPN, LGN) | one analysis vs several | "and in the LGN, in the bilateral intracalcarine cortex" @15480
9. LCuhWT87Vsa5 | b | family None/"Correlation" vs glm/"linear regression" | naming | "linear regressions with the following variables as predictors were performed with the contrasts" @23936
10. LCuhWT87Vsa5 | d | term "TSMQmean" vs "TSMQ" | naming | "a general index for trait sexual motivation (TSMQ" @20419
11. HETbPtxfHWPq | b | Analysis III (analysis#d942ee4f328144e5): Astra "ROI group comparison" with age/gender/TIV/BMI/wear time/employment covariates; model "mass univariate GLM" with the group term only. The text does not list covariates for this comparison | detail | "The total intracranial volume (TIV) was computed for each participant and added as a covariate to the analyses." @16489
12. HETbPtxfHWPq | d | group names "high levels of sedentary time ..." vs "high levels of SB ..." | naming | "(1) individuals with high levels of SB and low levels of MVPA and (2) individuals with high levels of SB and high levels of MVPA" @19208
13. gYhqRTaVZGZn | a | "aversive emotions > baseline (geometric figures)": Astra faces + only (baseline implicit); model faces +, geometric figures - | naming/detail (implicit baseline as a cell) | "the effect of face presentation relative to the implicit baseline of the geometric figures" @8160
14. tDJu8SHfMKQh | c | seed-wise FC analyses sit on the table keys (#65e9..., #0fdd...) in Astra, on the text-origin twins (#c201..., #05dac...) in the model; the same findings appear in text and Table 3 | other (analysis attached to the text twin of a table key) | "The adolescents with BPD showed increased functional connectivity compared with the HCs, involving the following regions" @24739
15. tDJu8SHfMKQh | c | cortical thickness (#ab841fe86f023fbf): Astra one analysis; model two (DMN, limbic-cortical) | one analysis vs several | "There was altered cortical thickness involving the DMN regions in the adolescents with BPD compared with the HCs" @19814
16. 6B6zGj6nHrUD | b | Astra terms Spatial relevance + drift; model adds a "session" term | naming/detail | "a constant and a linear drift term were included separately for each session" @22342
17. 6B6zGj6nHrUD | d | term "Spatial relevance" vs "spatial location" (cell levels identical in meaning) | naming | "survived the weighted contrast between two bilateral relevant spatial locations" @22596
18. wubMcWUYibqb | b | term "condition" vs "incorrect responses" | naming/detail | "Modeled conditions included novel‐rewarded, novel‐not‐rewarded, familiar‐rewarded, familiar‐not‐rewarded and incorrect responses." @18899 (U+2010 hyphens in the file)
19. RwmhBN3dP82k | d | term "task condition" vs "MSIT condition" | naming | "Multi-Source Interference Task" @27304 (acronym expansion; both names fit)
20. ovMNArvAW7ok | d | group "HCC"/"LCC" vs "high/low contamination concern" | naming | "HCC, high contamination concern; LCC, low contamination concern." @23976
21. 3bFL4tESYK8c | a | "load x age interaction" (analysis#651d197bec8672fd): Astra one term "load x age" undirected; model cells age and load undirected | other (interaction term vs separate terms) | "significant “load” × “age” interaction effect" @26624
22. bJrWYwKNV7xM | c | key analysis#28b211e04afe3f5f (lAMG-lSMG, text) duplicates #43f2553da3316bf3 (same finding); Astra analysis on the former, model only on the latter | other (duplicate coordinate-set key) | "Analyses of covariance were then performed in the two groups (TDC and ASD) with covariates of age and gender." @14649
23. bJrWYwKNV7xM | b | Astra ANCOVA terms include seed signal and six motion/WM/CSF/global terms; model has group, age, gender | model chain vs single | "the six head motion parameters, the WM signal, the CSF signal and the global mean signal [ 44 ] were regressed out as covariates." @14155
24. BSGoTkY8SLJN | b | family mixed_effects vs glm for the RFX two-sample t test | naming | "a multi-subject random effects (RFX) analysis was carried out" @13580
25. jHSNMakGotVy | b | Astra one merged term "Motion, nuisance signals and filtering confounders"; model splits it into motion, spikes, CSF, WM, global | naming/detail | "signal oscillations at a frequency of 0.1–0.25 Hz (sine-cosine pairs) were added as confounders" @38186
26. PsrZY8awrRRn | d | term "head motion parameters" vs "working memory task head motion parameters" | naming | "Head motion parameters were included as regressors in the first level analyses." @12176

Equivalence kinds: naming or detail 18 (items 1,3,4,6,7,9,10,11,12,13,16,17,18,19,20,24,25,26); contrast term vs separate cells 2 (2,5); one analysis vs several 2 (8,15); model chain vs single 1 (23); other 3 (14, 21, 22); group vs condition 0; direction flip 0.

## astra_right (13), representative examples

- PsrZY8awrRRn, (a), key analysis#029d2dda2659ffbc "AD baseline versus elderly (masking for deactivation in elderly controls)" (working memory table). Astra: AD +, elderly -. Model: elderly +, AD baseline - (reversed; same title read correctly in the semantic task key). Quote @29447: "AD baseline versus elderly (masking for deactivation in elderly controls)\t\t\t\t\t\t\nThalamus". (The same title also occurs at @23000 in the semantic table, so the quote is made unique by the next row.)
- jHSNMakGotVy, (a), key analysis#b950b443485fa99f "p > c (negative)", point -27 23 4 r=-0.67. Astra group direction None; model patients -, controls +. Table 1 row @10939: "Left mPFC – Cl. 10 (left anterior insula)\tp > c\t550\t−27\t23\t4\tVisuospatial working memory (r = −0.67; p = 0.02)" (thin spaces in the file): a patients > controls cluster, the negative sign is the correlation. Model's direction is wrong; Astra's None is uninformative, not wrong.
- LCuhWT87Vsa5, (a), correlation analyses (e.g. TSMQ, key analysis#0058f1b47bcb9213). Astra direction None; model positive. Quote @1421: "However, we found no relationship between these activations and indicators of problematic pornography use"; the positive correlation was only a hypothesis (@10883).
- wRM5PyEyTezN, (a), key analysis#a04be2f94203c39c. Astra: time (baseline +, follow-up -) within high BAGCH (group held). Model: group x time interaction with high BAGCH -, low +. Quote @24403: "VBM analysis showed ipsilesional areas with marked reduction in GM volume from the sub-acute to chronic phase of stroke in the subset of patients with high BAGCH but not for patients with low BAGCH".
Others (quotes in the report): 3bFL4tESYK8c (b) model adds age to a one-sample test, text @22213 "To examine age effects on load-related activation (load x age interaction), age was included as covariate of interest into the group analysis of contrast 4."; gYhqRTaVZGZn (b) second-level not first-level, @8286; caTme4KSzp34 (b) integrated group model, @7338; d38NZN3N36E6 (c) extra analysis from a dipole location, @27574; DQixkiizyFQN (c) SVC peak, @17288; wRM5PyEyTezN (b) flexible factorial, @18181; RwmhBN3dP82k (a) Visit 2 held, @26243; wubMcWUYibqb (a) interaction omnibus @28039 and correlation for familiar images @32351.

## unclear (2)
- ouZjktvZh2wW (a), analysis#1dd8b7a536407d36 interaction: PFO cells Astra PFO+ +, PFO- -; model PFO+ -, PFO- +. Text @15304 "A significant negative interaction effect between MO and PFO was observed in the LFPN"; a sign-product reading favours Astra, but the paper does not fix per-factor signs.
- wubMcWUYibqb (c): Astra has two "small volume correction" analyses (keys analysis#0357412b3044c4e0, #3840529552e7762f) that the model omits; the text has no SVC statement (all contrasts "P = 0.001 (uncorrected)"), but tables may be missing from the pubget text.

## Caveats
n=25 papers, one seed; items are not independent (several per paper, 46 total, roughly one per disagreement type, wubMcWUYibqb has two (a) items). Strata (about half low-F1) over-weight bad papers, so shares are not population rates. The "both_acceptable" calls on naming/detail are judgments about equivalence, and several lean on the paper not specifying a detail (HETbPtxfHWPq covariates, 32WaBSTPCZYm terms). Three tabular judgments rest on pubget table text (jHSN, f5Zu, tDJu). Papers were read by targeted search, not in full. No bootstrap done (n too small); no GPU used.
