# Adjudication: final QLoRA Qwen3.8-27B (r3-27b-final, step 776) vs Astra, validation split (read-only, no GPU, test split untouched)

Sample: the same 25 val papers as the ck50 adjudication (13 Low-F1, 12 High-F1 by the ck50 model's linked F1; strata kept for paper-by-paper comparison). Four papers were skipped because the final model and Astra now agree exactly on cells (linked cells fp = fn = 0): 6teXKZNVzJ62 (5/0/0), obVXFvscgkTW (8/0/0), HETbPtxfHWPq (4/0/0), PsrZY8awrRRn (60/0/0). 21 papers remain, 35 effect-cell items judged (max 3 per paper; twin analyses with the same issue are one item). Scripts: adj/dump.py (ck50 dump script pointed at the final run), adj/d_<sid>.txt (per-paper dumps), adj/vq2.py (quote checker, whitespace-tolerant), adj/quotes_in2.json / quotes_out2.json. All offsets below are `text.find(quote)` in the pubget text.txt, and `text[off:off+len(q)]==q` was True for every quote (the quotes keep the file's own spacing, thin spaces and tabs).

Final-model linked F1 per paper (tp/fp/fn): caTme4KSzp34 0/6/6, fzfEymXD8DuP 2/4/2, zrWff5sJKCrb 2/10/10, kFTdpnRj5N8T 14/2/4, LCuhWT87Vsa5 4/0/8, ouZjktvZh2wW 2/1/4, wRM5PyEyTezN 2/2/1, gYhqRTaVZGZn 0/6/7, tDJu8SHfMKQh 2/16/12, 6B6zGj6nHrUD 2/4/4, wubMcWUYibqb 0/7/15, ovMNArvAW7ok 1/1/2, 3bFL4tESYK8c 5/0/1, bJrWYwKNV7xM 8/2/0, BSGoTkY8SLJN 24/2/0, d38NZN3N36E6 2/2/0, DQixkiizyFQN 8/0/2, 32WaBSTPCZYm 4/2/2, RwmhBN3dP82k 14/0/3, f5Zu2kabHbnj 27/0/3, jHSNMakGotVy 20/4/2.

## Counts (error shape x judgment)

| shape | astra_right | model_right | both_acceptable | both_wrong | unclear | total |
|---|---|---|---|---|---|---|
| missing cell (gold has it, model not) | 11 | 0 | 4 | 0 | 0 | 15 |
| extra cell (model has it, gold not) | 1 | 3 | 5 | 0 | 0 | 9 |
| direction | 2 | 0 | 0 | 0 | 1 | 3 |
| level | 0 | 0 | 4 | 0 | 0 | 4 |
| term | 1 | 0 | 3 | 0 | 0 | 4 |
| all | 15 | 3 | 16 | 0 | 1 | 35 |

Model share right among decided (astra_right + model_right = 18): 3/18 = 17%. Model "not wrong" (model_right + both_acceptable) among judged excluding unclear (34): 19/34 = 56%. ck50 comparison (all types, 46 items): 5/18 = 28% and 31/44 = 70%; ck50 type (a) alone: 0/7 and 4/11. Different item sets (the final dumps have more whole-analysis omissions and fewer wrong-direction cells), so the shift is descriptive, not a tested difference.

Equivalence kinds among the 16 both_acceptable are listed after the items.

## Tables
No judgment rested on a shifted table. Checked against values: f5Zu2kabHbnj Table 3 (Contrast row "Words > Partial x4, Final > Initial x4"; Region / x y z / P(FWE) / K / Z; row "R angular gyrus 45 -52 32 0.004 9 4.87" consistent); jHSNMakGotVy Table 1 (Group difference / Nr of voxels / X Y Z / Significant correlation; rows consistent); BSGoTkY8SLJN Table 2 (Univariate p / Multivariate p / Coefficient / Hazard ratio; MFG 0.615 / 1.849, cuneus -0.278 / 0.757, consistent); tDJu8SHfMKQh Table 3 and zrWff5sJKCrb Tables 1 and 3 (header Z last column, consistent).

## Judged items (35)

Format: id | sid | analysis key | Astra vs model | judgment, kind, shape | quote @offset (verified).

A | caTme4KSzp34 | #2649fa60beeb4427 Infant faces, #817c619234dcd02f Adult faces | Astra state: active (linked to the infant/adult face conditions) +, passive (linked to the passive condition) -, face condition held. Model: state active +, passive -, no links on the levels, face condition held | both_acceptable, naming or detail (entity links on the state levels); shape level | "Power changes between the active and passive states were calculated for the beta (10–30 Hz) and gamma (30–60 Hz) frequency bands." @24582

B | fzfEymXD8DuP | #4c0b9c5faf2270e7 Anterior/Lateral IPS, #61e1be1ed9c52351 Posterior IPS | Astra: double-cue (red-green + green-red pooled) +, fixation -. Model: red-green +, green-red +, fixation - | both_acceptable, contrast term vs separate cells; shape level | "we selected voxels within each ROI that showed a significant effect for the double-cue condition" @12399

C | zrWff5sJKCrb | #097c1fefcfa42e0a MSw > baseline (also #1b43d16320e50f89, #368d9a50cb44eb5b, #74fe08234970f52d, #d4cb2f85786ff3e3) | Astra: term "music-only condition", level MSw, positive. Model: the cell's term is the id "trm_msw", which is not a term of its model chain (chain terms are "music-taste interaction condition", "taste-only condition", ...); same for trm_mso, trm_tso, trm_tsw | astra_right (term). Finding equivalent, but the model's cell points at an undeclared term, a structural defect | "Brain areas activated during sweet music and sweet taste conditions relative to baseline." @12268

D | zrWff5sJKCrb | #b245d7f96cda19d4 MSwTSw > TSw (also #cd87e6b6c82d5d01 MSoTSo > TSo) | Astra: term "sweet across-task input condition", MSwTSw +, TSw -. Model: term "sweet music–taste versus sweet taste condition", same levels and directions | both_acceptable, naming; shape term | "greater activation during concurrent auditory–gustatory stimulation compared to taste alone" @22000

F | kFTdpnRj5N8T | #cbeb302f16e2a665 t-test FEMDD vs HC in Condition 2 | Astra: FEMDD -, HC +, plus stimulus SNF +, ONF -. Model: FEMDD -, HC +, conditions "Condition 2" held | both_acceptable, held condition vs separate cells; shape missing cell | "The simple effect of the   t-  test between FEMDD vs. HC in Condition 2 (SNF vs. ONF) revealed that the group difference of brain activation located in the right inferior frontal gyrus/Insula" @22850

G | kFTdpnRj5N8T | #ff85029c13002554 Condition 1 | as F, SNF vs BCP | both_acceptable, held condition vs separate cells; shape missing cell | "The simple effect of the   t-  test between FEMDD vs. HC in Condition 1 (SNF vs. BCP) showed that the group difference of brain activation was located in the left occipital gyrus/fusiform gyrus" @22489

H | LCuhWT87Vsa5 | #0058f1b47bcb9213 correlation with TSMQ (anticipation and delivery) | Astra: analysis with cell [TSMQmean, direction None]. Model: no analysis | astra_right; shape missing cell (null result, but the paper reports it) | "Neither questionnaires measuring PPU (HBI and s-IAT sex ), nor the amount of time spend on porn, nor trait sexual motivation (TSMQ) were significantly correlated with brain activities of the reward-related brain areas" @32964

I | LCuhWT87Vsa5 | #2da146364185a65d s-IATsex (both phases) | Astra analysis, model none (same quote as H) | astra_right; missing cell

J | LCuhWT87Vsa5 | #cbed78119ba775f0 TimePU (both phases); HBI #6ef27c667f2ba806 is the same case | Astra analysis, model none (same quote as H; TimePU is in the "amount of time spend on porn" clause) | astra_right; missing cell

K | ouZjktvZh2wW | #1dd8b7a536407d36 Interaction | Astra: MO+ -, MO- +, PFO+ +, PFO- - (four cells). Model: one cell [MO × PFO, level "MO+", negative] | both_acceptable, interaction term vs separate terms; shape level. The text only gives "a significant negative interaction effect" and does not fix per-factor signs, so Astra's four signs go beyond the text and the model's one cell is the closer reading; its level "MO+" is arbitrary | "A significant negative interaction effect between MO and PFO was observed in the LFPN" @15304

L | wRM5PyEyTezN | #a04be2f94203c39c VBM GM reduction in the high-BAGCH group | Astra: time baseline +, follow-up -, group high BAGCH held. Model: time TP1 +, TP2 - (match), plus group high BAGCH +, low BAGCH - | astra_right; shape extra cell (the contrast is within the high group; "not for low BAGCH" means no effect there, not the opposite sign) | "VBM analysis showed ipsilesional areas with marked reduction in GM volume from the sub-acute to chronic phase of stroke in the subset of patients with high BAGCH but not for patients with low BAGCH" @24403

M | gYhqRTaVZGZn | #8981acd94a59dc01 aversive emotions > baseline | Astra: faces +. Model: faces +, geometric figures - | both_acceptable, implicit baseline as a cell; shape extra cell | "the effect of face presentation relative to the implicit baseline of the geometric figures" @8160

N | gYhqRTaVZGZn | #cf8e5fb618f6add8 habituation test (null) | Astra: faces +, first measurement +, second -. Model: no analysis | astra_right; shape missing cell. The paper reports this as its "main contrast of interest" and it was null; Astra's signs encode the tested contrast, not a finding, so Astra is only partly right | "Against the hypothesis that a habituation may occur and result in a decrease of the activation of the amygdala, this effect was not significant." @11263

O | gYhqRTaVZGZn | #efa4582766cf3ec8 Decreases from first to second measurement | Astra: faces +, first +, second -. Model: adds geometric figures - | both_acceptable, implicit baseline as a cell; shape extra cell | quote as M @8160

P | tDJu8SHfMKQh | Astra on #65e9f46929cd72e7 (4 increased-FC seeds: left/right lateral occipital, left/right inferior temporal); model on #c201e2261219b724 | same cells (BPD +, HCs -), different parse key | both_acceptable, other (analysis attached to the text twin of a table key); shape missing cell (and extra on the twin) | "The adolescents with BPD showed increased functional connectivity compared with the HCs, involving the following regions" @24739

Q | tDJu8SHfMKQh | Astra on #0fdd7203a30e32ca (left parahippocampal, right precuneus decreased FC); model on #05dacf6f06a79e5a | same cells (BPD -, HCs +) | both_acceptable, other (twin key); shape missing cell | "The adolescents with BPD showed decreased functional connectivity compared with the HCs, involving the following regions" @25956

R | tDJu8SHfMKQh | #ab841fe86f023fbf increased cortical thickness | Astra: only "decreased cortical thickness" (BPD -, HCs +). Model also has "increased cortical thickness" (BPD +, HCs -) | model_right (Astra omits a reported finding); shape extra cell | "(2) increased cortical thickness in the bilateral parahippocampal gyrus" @20213

S | 6B6zGj6nHrUD | #431078c153b1d4cd Relevant > irrelevant, #ddbf276c09d1ce01 Irrelevant > relevant | Astra: term "Spatial relevance", levels relevant/irrelevant with opposite signs. Model: term "stimulated spatial location", same levels and signs (the scorer's term-name threshold fails) | both_acceptable, naming; shape term | "survived the weighted contrast between two bilateral relevant spatial locations versus one bilateral irrelevant spatial location" @22596

T | wubMcWUYibqb | #0357412b3044c4e0 novelty x reward interaction | Astra: undirected omnibus cells for novelty and reward. Model: novel +, familiar -, rewarding +, not rewarding - | astra_right; shape direction. The text calls it an F-contrast, and a crossover interaction has no sign per factor | "Secondly, we assessed the interaction (F‐contrast) between novelty and reward." @27603

V | wubMcWUYibqb | #0357412b3044c4e0 and #3840529552e7762f, the two "small volume correction" analyses | Astra: separate SVC analyses (interaction undirected; main effect of novelty novel +, familiar -). Model: none | astra_right; shape missing cell. This resolves the ck50 "unclear": the text does state the SVC | "we applied small volume correction to both activation patterns using the right anterior hippocampus as volume" @29824

W | wubMcWUYibqb | #6822a853b2d3d78d SN/VTA voxel correlation | Astra: "improved recollection rate for familiar images" negative plus familiar rewarded +/not rewarded -, under an analysis named "for novel images". Model: "improved recollection rate" positive | unclear; shape direction. The voxel showed a significant positive correlation for novel images (Delta corrected hit rate), and r = -0.07, P = 0.811 (null) for familiar images; Astra's analysis name says novel but its term says familiar, so neither side is clearly right | "A statistically more sensitive post hoc analysis of the SN/VTA voxel [4, −18, −16] that showed a significant correlation for novel images" @32213

X | ovMNArvAW7ok | #768e575a6f2c523d independent t-test under GS4 | Astra: HCC +, LCC -, stimulus type GS4 held. Model: high +, low - (no held cell) | astra_right; shape missing cell | "exhibited significantly higher engagement of the right putamen and left anterior insula (FWE-corrected,   p   < 0.05) under the GS4 condition" @29947 (text has U+00A0 in "p < 0.05")

Y | ovMNArvAW7ok | same key | group levels "HCC"/"LCC" vs "high/low contamination concern" | both_acceptable, naming; shape level | "HCC, high contamination concern; LCC, low contamination concern." @23976

Z | 3bFL4tESYK8c | #651d197bec8672fd load x age interaction | Astra: [load × age, undirected]. Model: no analysis | astra_right; shape missing cell. It is a behavioural repeated-measures ANOVA (errors) and a DLPFC signal ANOVA, so whether it belongs in a brain-effects record is arguable | "as well as a significant “load” × “age” interaction effect" @26611

AA | bJrWYwKNV7xM | #2ed4dc3028675e80 (model only), twin of #a9bcc2979a59e5de | Astra: one analysis (ASD -, TDC +) on #a9bc. Model: the same on #a9bc and again on #2ed4 | both_acceptable, other (duplicate coordinate-set key); shape extra cell | "showed significant negative FC to the left amygdala (lAMG) in children with ASD compared to TDC when controlled for age and gender" @22381

AB | BSGoTkY8SLJN | #8ce7f8c837d03ec3 Cox, left cuneus | Astra: no analysis. Model: DMN left cuneus, negative (Table 2 coefficient -0.278) | model_right; shape extra cell | "functional connectivity within the MFG (  x =   39;   y =   46;   z =   10) and the left cuneus (  x =   − 3;   y =   − 70;   z =   19) were independent predictors of treatment-related motor complications" @22837

AC | BSGoTkY8SLJN | #ef1c872f7eb1fa36 Cox, right MFG | Astra: none. Model: FPN right MFG, positive (coefficient 0.615) | model_right; shape extra cell | same quote @22837

AD | d38NZN3N36E6 | #d4d42a7a5ca67eb2 face processing task | Astra: one analysis (active +, control -). Model: two analyses, SPMIC and SickKids, same cells | both_acceptable, other (one analysis vs several); shape extra cell | "The peak in the pseudo-t-statistical image is found at the border of the left temporal and occipital lobes for both sites" @38080

AE | DQixkiizyFQN | #751450f907461752 Normal > Deaf (SVC) | Astra: separate SVC analysis (Normal +, Deaf -). Model: none | astra_right; shape missing cell (separately reported inference of the same contrast) | "but survived following a small volume correction (SVC; 5 mm sphere at peak voxel, FWE-corrected   p   = 0.001)" @17284

AF | 32WaBSTPCZYm | #9c649698db841cc0 post > pre conditioning | Astra: term "scan occasion", post +, pre -. Model: term "Resting-state scan", same levels and signs | both_acceptable, naming; shape term | "A   t  -test directly comparing the pre-conditioning resting-state correlations values to the post-conditioning resting-state correlation values was also calculated." @15333

AG | RwmhBN3dP82k | #0263e9ffc65cabaa and #0fec7297101a6ad2 Visit 2 | Astra: Interference +, Control -, plus visit occasion "Visit 2" held. Model: Interference +, Control - only | astra_right; shape missing cell (held) | "single-sample analyses revealed no significant activation at Visit 1, yet at Visit 2 patients exhibited focal activation within the midcingulate cortex" @26243

AH | RwmhBN3dP82k | #eb01709bdb0ebcc4 Visit 1 | same, "Visit 1" held | astra_right; shape missing cell | same quote @26243

AI | f5Zu2kabHbnj | #efbb750b6f14a597 Words > Partial | Astra: whole word +, initial -, final -. Model: no analysis | astra_right; shape missing cell | "The contrast Words > Partial showed a significant activation in the right angular gyrus" @35920 (Table 3 also has a "Words > Partial" block @32720)

AJ | jHSNMakGotVy | #b950b443485fa99f p > c (negative) | Astra: patients None, controls None. Model: patients -, controls + | astra_right; shape direction. Table 1 row says p > c (patients > controls) and the "negative" is the r = -0.67 correlation; the model's flip is wrong, Astra's None is uninformative | "Left mPFC – Cl. 10 (left anterior insula)\tp > c\t550\t−27\t23\t4\tVisuospatial working memory (r = −0.67; p = 0.02)" @10939

AK | jHSNMakGotVy | #190fd7f001fc8b37, model-only analysis "c > p (Right mPFC – Cl. 2)" | Astra has "c > p (Right mPFC)" without this cluster; model adds the cluster (patients -, controls +) | both_acceptable, one analysis vs several (the table row is a real c > p cluster); shape extra cell | "Right mPFC – Cl. 2 (right temporal lobe)\tc > p\t711\t39\t−7\t−26\tVoice onset times (r = 0.58; p = 0.05)" @11103

Kinds among both_acceptable (16): naming or detail 5 (A, D, S, Y, AF); contrast term vs separate cells / held condition 3 (B, F, G); implicit baseline as a cell 2 (M, O); interaction term vs separate terms 1 (K); one analysis vs several 2 (AD, AK); other 3 (P, Q twin key; AA duplicate key). Group vs condition, direction convention: 0.

Unjudged: wubMcWUYibqb #3840529552e7762f main effect of novelty (Astra novel +, familiar -; model undirected): the text names it a main effect of the 2x2 ANOVA and gives no direction; over the 3-per-paper cap, so not counted (would be unclear).

## Patterns
Model wrong (astra_right, 15):
- Drops whole analyses Astra records: null correlations (LCuhWT87Vsa5 H, I, J), the null habituation interaction (gYhqRTaVZGZn N), a second inference of the same contrast (SVC: DQixkiizyFQN AE, wubMcWUYibqb V), a reported contrast (Words > Partial, f5Zu2kabHbnj AI), a behavioural ANOVA interaction (3bFL4tESYK8c Z). 11 of the 15 astra_right items are missing cells.
- Drops a "held" restriction cell (RwmhBN3dP82k Visit 1 and Visit 2, ovMNArvAW7ok GS4); the analysis name still carries the restriction.
- Adds signs Astra withholds or the text does not give: directions on an F-contrast interaction (wubMcWUYibqb T), a second group with the opposite sign in a within-group contrast (wRM5PyEyTezN L), a flipped direction on a table row (jHSNMakGotVy AJ).
- Structural defect: cells pointing at term ids that are not in the model chain (zrWff5sJKCrb C; five simple-effect analyses in that paper). The finding is right; the cell is not valid against the record.
Astra wrong (model_right, 3): Astra omits analyses the paper reports (Cox regressions with signs confirmed by Table 2 in BSGoTkY8SLJN, "increased cortical thickness" in tDJu8SHfMKQh).
Both acceptable (16): implicit baseline written as a cell, held condition vs separate cells, term renames that miss the scorer's name threshold (6B6zGj6nHrUD, 32WaBSTPCZYm, zrWff5sJKCrb), the same finding on a twin parse key (tDJu8SHfMKQh, bJrWYwKNV7xM), one analysis vs several. All of these cost linked F1.

## Representative astra_right examples
1. f5Zu2kabHbnj AI: Astra's "Words > Partial" (whole word +, initial -, final -) is absent from the model. The paper states it at @35920 and Table 3 has a "Words > Partial" block at @32720. (This also corrects the ck50 report, which said Astra's "Words > Partial" had no basis in the table; it does.)
2. LCuhWT87Vsa5 H: the model no longer asserts positive correlations (ck50's error) but now omits the TSMQ, s-IATsex, HBI and TimePU analyses altogether; the paper tested them all (null), @32964.
3. jHSNMakGotVy AJ: the model gives patients - / controls + for a "p > c" cluster; Table 1 @10939 says p > c, the negative sign belongs to the correlation.
4. wRM5PyEyTezN L: the model gives the high-BAGCH group + and the low-BAGCH group - in a contrast that is within the high group only; the text @24403 says "but not for patients with low BAGCH".

## ck50 follow-up: the 12 type (a) items
| ck50 item | now |
|---|---|
| PsrZY8awrRRn AD baseline vs elderly (direction flip) | now matches Astra (paper agrees exactly on cells, 60/0/0) |
| fzfEymXD8DuP double-cue vs red-green/green-red | still disagrees the same way (item B) |
| kFTdpnRj5N8T Condition 2 held vs separate cells | still disagrees the same way (item F; also Condition 1, item G) |
| gYhqRTaVZGZn faces + only vs faces + / geometric figures - | still disagrees the same way (items M, O) |
| jHSNMakGotVy p > c (negative) direction | still disagrees the same way (item AJ: patients -, controls +) |
| RwmhBN3dP82k Visit 2 held cell | still disagrees the same way (items AG, AH; Visit 1 also missing) |
| wubMcWUYibqb omnibus interaction given directions | still disagrees the same way (item T; the reward-factor signs flipped relative to ck50, still directed) |
| wubMcWUYibqb correlation direction/term | still disagrees the same way (item W: Astra negative, model positive; I now read it as unclear, ck50 read astra_right) |
| wRM5PyEyTezN group x time instead of within-high-group time | disagrees differently: time cells now match Astra; the group cells are still extra and their signs flipped (high +, low -) (item L) |
| LCuhWT87Vsa5 positive direction for null correlations | disagrees differently: the analyses are now omitted (items H, I, J) |
| 3bFL4tESYK8c load x age interaction (undirected cell vs separate terms) | disagrees differently: the analysis is now omitted (item Z) |
| ouZjktvZh2wW interaction PFO cells (unclear) | disagrees differently: model now writes one MO x PFO cell, negative, matching the text; Astra's four signs are unsupported (item K, both_acceptable) |
Summary: 1 matches, 7 same, 4 differently.

## Caveats
n = 25 papers (21 judged), one model and one seed. 35 items, not independent (up to 3 per paper, twins merged); shares have wide uncertainty (3/18, 19/34) and no bootstrap was run. Strata come from the ck50 model's F1 (13 low), so the sample over-weights papers that were hard for that model, not for the final one. Papers were read by targeted search, not in full; "both_acceptable" is a judgment about equivalence and four items (C, K, N, Z) have an arguable side noted above. Choosing which 3 disagreements per paper were judged favours distinct analyses, so one-off omission types are over-represented relative to repeated patterns (twin analyses were merged). Three judgments revise ck50 (f5Zu Words > Partial exists; wubM SVC is in the text; wubM correlation now unclear). No GPU, no test papers, nothing outside adj/ and this file was modified.
