# Adjudication: fine-tuned NuExtract3 (10k) vs Astra, validation split (read-only, no GPU)

Sample: 25 val papers, seed 7, drawn from papers with linked cell F1 < 0.6 (13 of 147 eligible) and >= 0.6 (12 of 117 eligible); papers where model and Astra agree exactly were skipped. Code and per-paper dumps: adj/dump.py, adj/d_<sid>.txt, adj/sample.json. 45 disagreements judged against the paper text (pubget text.txt; offsets are char offsets in that file). Test split not touched.

Sample sids: low F1: UsTYeUMRgLny 7EYr88f3gNWL e3e7fZNLRaba 4NcBxbvtnTs6 4z4c4xkQhH8z w99sCK9imDR5 5axNgmQAMB3f aiaFguLWJcco 4UQKV4GwCv8p pZ7LbCGoTAsZ EJ9Y27PQCRDB 3k3tbteEwyRR 5WpF756kadRz; high F1: SvmuAPdQNF4m SF3knMDwKcrp 4ZKzYhgL3ChX 7tAVtBvQGd7U 4ne4iCoXonVe ds2mKHx39PGb SJVpTpZ6p2g2 4Z3RncoCr6CV v3KFFN4pmyeV etj29FQTJ7si 5MXEcfwbQfPT 7LTv9SJ2fC3p.

## Counts

| type | astra_right | model_right | both_acceptable | both_wrong | unclear | total |
|---|---|---|---|---|---|---|
| (a) effect cell / direction / level | 9 | 4 | 8 | 0 | 0 | 21 |
| (b) model or term | 5 | 0 | 5 | 0 | 0 | 10 |
| (c) analysis on another key or missing | 5 | 0 | 2 | 0 | 2 | 9 |
| (d) group / region / n / name | 4 | 0 | 1 | 0 | 0 | 5 |
| all | 23 | 4 | 16 | 0 | 2 | 45 |

Model right among decided (astra_right + model_right = 27): 4/27 = 15%. If both_acceptable is counted as "model not wrong", the model is not wrong in 20/43 = 47% of the judged items (excluding unclear). Everything is n=25 papers, one seed; the paper-level stratification (half low-F1) over-weights bad papers.

## Patterns

Model wrong (Astra right):
- Reads the wrong comparison for table-defining contrasts: aiaFguLWJcco (ALE task-type contrasts written as BD/C group cells, undirected), 3k3tbteEwyRR (adds after>before tracking cells to a switching-condition contrast), 7LTv9SJ2fC3p (Group x TST interaction written as plain group effect), v3KFFN4pmyeV (N0D vs CN written as N0D vs N10D).
- Drops restricted-condition context cells ("held" cells: postcue delay, fixation baseline, own-hand feedback, scan occasion) and Astra's linking of group/time/arm entities (controls scanned only at T0; individual cases vs one patient group; pooled stimulus conditions across tasks).
- Collapses analyses: misses post-hoc / SVC / voxel-level-vs-cluster-level / within-group correlation analyses that Astra lists; also over-splits in a few places (extra "Late assessment >" analysis).
- Model type: picks the first-level GLM or the decoder where the table was produced by the group-level test (paired t-test, one-sample t-test, full factorial).
Astra wrong (model right):
- Direction errors in correlation tables (e3e7fZNLRaba age of onset, three ROIs negative where every table value is positive), wrong comparison linked (5axNgmQAMB3f), missing the second modality of a conjunction (7tAVtBvQGd7U), direction None where the analysis name says "<" (SJVpTpZ6p2g2).
Both: naming of terms/levels/models differs in a way the paper does not decide (see Equivalent representations).
The low F1 of many low-F1 papers is mostly representation mismatch, not wrong content: UsTYeUMRgLny, w99sCK9imDR5, EJ9Y27PQCRDB, 5WpF756kadRz, 4z4c4xkQhH8z score 0 linked F1 with essentially identical effect cells.

## Representative examples

1. UsTYeUMRgLny, key analysis#0ed01b..., (b) model. Astra glm / paired-sample t-test; model glm / first-level GLM; cells identical (sort +, maintain -). Text @24247: "We extracted first-level GLM beta values and performed a paired-sample t-test using the AFNI program 3dttest++." -> astra_right (model type); term name "task condition" vs "Sternberg condition" both_acceptable (naming). Scorer gave linked F1 0.
2. e3e7fZNLRaba, key analysis#3988..., (a) age-of-onset direction. Astra: L_OR, R_OR, aIFOF negative; model: positive for all six. Table 2 @23736: age-of-onset columns read 0.86/0.56, 0.67/0.27, 0.71/0.031, 0.37/0.07, 0.86/0.86, 0.93/0.83 (all positive; the text says "not significantly correlated" @12405). -> model_right (table is garbled, but no value is negative).
3. aiaFguLWJcco, key analysis#137dc..., (a) "Emotional > cognitive". Astra task type: Emotional +, Cognitive -; model Group: BD undirected, C undirected. Figure legend @54241: "Emotional > cognitive clusters are shown in yellow; cognitive > emotional clusters are shown in blue." -> astra_right. Same paper "Cognitive tasks (C > BD)" table row @31044 -> directions needed, model gave undirected: astra_right.
4. pZ7LbCGoTAsZ, key analysis#23465..., (b) model. Astra anova/full factorial; model glm/first-level GLM. @25806: "submitted to second level full factorial model for group analyses." -> astra_right. (a) feedback cells: Astra own hand + / other hand -, model "held"; @31941: "the group interaction (group [HC > SSD] X condition [passive> active] X feedback [own hand > other hand])" -> astra_right. All cells otherwise identical, linked F1 0 from the model gate.
5. 5axNgmQAMB3f, key analysis#8d14..., (a) Table 1. Astra: motor imagery + / rest - (music and visual NF levels); model: feedback active + / sham -. @29309: "Table 1 provides a detailed breakdown of the clusters that were significantly active only in the active NF condition." -> model_right (Astra's cells come from the music-vs-visual comparison, a different table; the model's sham "negative" is approximate).
6. 7tAVtBvQGd7U, key analysis#5875..., (a) conjunction. Astra only smoking group (nonsmokers +, smokers -); model also acute drug condition (baseline +, drug -). @8756: "we tested the structural-functional hypothesis that chronic smoking-related gray matter alterations overlap those same regions showing acute nicotinic drug-induced functional effects" -> model_right (conjunction involves both modalities; kind=interaction vs contrast is a convention).
7. SF3knMDwKcrp, key analysis#230a..., (d) timepoint of controls. Astra controls t:tp_t0; model controls t:tp_followup. @9512: "24 patients with sporadic ALS and the same number of age- and gender-matched healthy controls underwent MRI (T0). In addition, DTI was performed in 15 patients with ALS after six months" -> astra_right. Also (c) SVC analysis (Astra has it, model not): @9129 "a small volume correction (SVC) was undertaken" -> astra_right.
8. 5WpF756kadRz, all keys, (b)/(a) term structure. Astra one term "item type" with levels "visual artificial", "auditory words", ...; model two terms "visual stimulus type" and "auditory stimulus type" with levels "artificial", "words". Text @16695: "the parameter estimates reflecting signal change for each item type vs. rest" -> both_acceptable (one term vs two terms per modality); linked F1 0 on 15 identical-meaning cells.
Others with ids: v3KFFN4pmyeV analysis#bec1... ("N0D vs CN" header @18432; model has N10D) astra_right; 7LTv9SJ2fC3p analysis#3765... (@24512 "A significant Group x TST interaction for BR > PG (see Table 3 top)") astra_right; 3k3tbteEwyRR analysis#8412..., @15902 "three contrast analyses were conducted: target-switching condition vs. distractor-switching condition, ..." (after/before not part of the contrast) astra_right; 4z4c4xkQhH8z @29960 "A one-sample t-test was used to reveal significant decoding of semantic concepts across languages" astra_right on the model (whole-brain), @33306 permutation/bootstrap for the ROI keys astra_right; 4ne4iCoXonVe @18632 "cluster-level FWE correction with a p-value of .05 ... also used a less stringent threshold with an uncorrected p-value of .001" (Astra lists both thresholds as analyses, model one) astra_right; 4NcBxbvtnTs6 @19826 "We focused on the postcue delay" (Astra's held postcue-delay cell) astra_right; 4UQKV4GwCv8p @24649 "(HER + LER) vs. fixation" (Astra's fixation cell) astra_right; 4Z3RncoCr6CV table header @21544 "Correlation of cluster activity with BDNF rank" (within-group correlation analyses missing in model) astra_right; 4UQKV4GwCv8p post-hoc t-test analyses and ds2mKHx39PGb duplicate "HC > BPD" analyses: unclear.

## Equivalent representations (judged both_acceptable: 16 items)

Does the scorer penalise? In all 16 cases, yes: they lower linked F1 (strict term-name matching with the model-similarity gate, strict direction, extra analyses count as false positives). No case of flipped direction (A>B vs B<A) or mirror_of occurred in the sample.

| kind | n | example | scorer |
|---|---|---|---|
| naming or level of detail (term/level/model-family/timepoint names, None vs "undirected", level link ids) | 10 | UsTYeUMRgLny: "task condition" vs "Sternberg condition"; w99sCK9imDR5 family None vs glm; 4ne4iCoXonVe direction None vs undirected on threshold analyses; etj29FQTJ7si level cond_navigation vs cond_navmath_navigation | penalised; when the model-equivalence gate fails all cells of the analysis score 0 (UsTYeUMRgLny, w99sCK9imDR5, pZ7LbCGoTAsZ) |
| one analysis with several cells vs several analyses (per-ROI splits) | 2 | e3e7fZNLRaba "Blindness duration (negative)": Astra 1 analysis, model 5 per-ROI analyses, Table 2 has per-ROI rows (@23736); 4z4c4xkQhH8z ROI keys 1 vs 5 | penalised as 4-5 false positives per key |
| single model vs a chain through inputs_from | 1 | 7EYr88f3gNWL: Astra random effects over components; model Group ICA (GIFT @31921 then SPM random effects @32279) | penalised (model gate and term mismatch) |
| factor written as a group vs as a condition | 1 | aiaFguLWJcco simple effects: Astra term "experiment set" level "Emotional tasks"; model term Group BD/C undirected (the paper's ALE sets BD vs controls; the set identity is in the analysis name) | penalised (cells never match) |
| one contrast term vs separate per-condition cells / terms | 2 | 5WpF756kadRz one "item type" term vs visual+auditory stimulus-type terms (@16695); 5MXEcfwbQfPT "self-regulation" vs "baseline" vs separate "SMA up/PHC down" and "SMA down/PHC up" levels, both positive vs baseline (@24554 lists self-regulation, up and down vs baseline as separate contrasts) | penalised |
| direction flipped (A>B vs B<A) | 0 | | would be penalised if present |
| mirror_of vs separate analysis | 0 | | |

Rough F1 cost, over the 25-paper sample (pooled linked cells): observed tp/fp/fn = 187/191/246 (F1 0.46). Moving the cells that differ only by an equivalent representation from fp/fn to tp (UsTYeUMRgLny 4, EJ9Y27PQCRDB 1, w99sCK9imDR5 4, 5WpF756kadRz 15, 4z4c4xkQhH8z 7, pZ7LbCGoTAsZ about 35 by estimate, 4NcBxbvtnTs6 about 11 by estimate, SvmuAPdQNF4m 1, 4ne4iCoXonVe 4, etj29FQTJ7si 1; plus 10 false positives from per-ROI splits and the 5MXE split) gives 270/98/163, F1 0.67: +0.21 on this sample. Per stratum: low-F1 papers 0.10 -> 0.49, high-F1 papers 0.82 -> 0.85. Reweighting the strata to the 147/117 eligible papers gives about 0.43 -> 0.66 on non-exact papers; the 217 exact or unscored val papers dilute that, so a whole-val gain on the order of +0.1 to +0.15 is a guess, not a measurement. The cell counts for pZ7LbCGoTAsZ and 4NcBxbvtnTs6 are my estimates, not recounts. Equivalence handling in the scorer (a model gate that falls back to strict when names differ) is where most of the cost sits: it is the single biggest mover, concentrated in a few papers.

## Gold that looks like a migration artifact (suspected, not proven)

- SJVpTpZ6p2g2 analysis#1335e...: "Case #5 < controls" has direction None on both cells, while the duplicate "Case #5 < controls" analysis#66c49... has negative/positive; the paper's table has two identical "Case #5 < controls" headers (@34412, @34773). Likely a mapping gap or an Astra omission.
- 4ne4iCoXonVe analysis#5ed26... and #defa8...: gold directions None for the threshold analyses (migrated from an undirected cell?).
- 4UQKV4GwCv8p: several gold cells with direction None (non-significant peaks) and post-hoc t-test analyses attached to table keys with "(left thalamus)" in their names: looks like column-to-analysis expansion rather than Astra's own choice.
- Gold model_family None while a model name is present (w99sCK9imDR5 "Second-level contrasts", 4NcBxbvtnTs6 "MDD group localizer activation maps", 4UQKV4GwCv8p baseline): family unmapped in migration.
- 4z4c4xkQhH8z: gold terms such as "word-reading semantic concept" and "word-reading six motion parameters" duplicate nuisance terms per chain: reads like a model chain merge.
- ds2mKHx39PGb analysis#1d23d...: two identical "Healthy controls > borderline personality disorder" analyses on one key, could be real (narrow vs extended task selection) or a duplicate.

## Caveats

n=25 papers, 45 judged items (not independent: several per paper, and sets such as aiaFguLWJcco's 17 analyses share one cause), up to 6 per paper was not binding because analyses repeat; I judged one item per kind of disagreement per paper. Reading garbled pubget tables (e3e7fZNLRaba) is uncertain. One seed, stratified toward bad papers, so the share of "Astra right" is not a rate over the val population.
