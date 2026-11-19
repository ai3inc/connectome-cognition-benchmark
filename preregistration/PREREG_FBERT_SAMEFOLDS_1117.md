# Preregistration — F-BERT on identical subjects and folds (2026-11-17)

Written **before any HCP F-BERT embedding or any HCP number from the new
timeseries exists.** Handoff §8 item 1: the published Table 3 numbers
(F-BERT-8M 0.502, BrainLM 0.023, Brain-JEPA −0.003, …) are on the authors'
n = 955 cohort and their own folds; no same-folds comparison exists.

## What is compared, and why this design

The only released deep model with weights we can run is F-BERT (HuggingFace
`Marraffini-Giovanni/fbert-8m`, `fbert-1m`; sha256 `0fbaa3d0…` and `7d482022…`).
Its code repository (`github.com/GioMarraffini/fc-power`) returned 404 today,
so the forward pass is rebuilt from the checkpoint (`code/fbert/fbert_model.py`)
and the unfixed switches are chosen by reproducing the checkpoint's own recorded
training loss on the public pretraining corpus (`code/fbert/validate_variants.py`,
`results/FBERT_VARIANTS_1117.json`: chosen variant 0.229 vs recorded 0.2115 for 8M,
0.331 vs 0.305 for 1M; next variant ≥ 0.07 further for both).

F-BERT reads timeseries, not the benchmark's published FC matrices. So the fair
comparison is on **one set of timeseries that every arm reads**:

- HCP-YA, the benchmark's 753 subjects (`protocol/MMP_HCP_753_subs.txt`).
- Four resting runs, `rfMRI_<run>_Atlas_MSMAll_hp2000_clean.dtseries.nii`,
  averaged within the 450 labels of Schaefer-400 (7 networks) + Tian-S3
  (`code/fbert/extract_hcp_ts.py`). Raw parcel means are stored.
- Preprocessing as the F-BERT model card states for its input: band-pass
  0.01–0.08 Hz (Butterworth order 5, zero phase, TR 0.72 s), z-score per region,
  per run; runs concatenated → 4,800 timepoints, read as one sequence.
- Region order for F-BERT's input: decided by a label-free check before any
  label is touched (match of group-mean FC to the corpus windows and
  embedding–teacher CKA on HCP), recorded in its own result file.

Target `varimax_cog` from `ooi_comp.csv`; the benchmark's 60 × 10 family-aware
splits; all 60 seeds reported (the paper used 20 × 10).

## Arms

| arm | representation (all from the same timeseries) | probe |
|---|---|---|
| **A8** | F-BERT-8M frozen embedding, full scan | best of P1 / P2 below |
| A1 | F-BERT-1M frozen embedding, full scan | best of P1 / P2 |
| B | plain Pearson FC | correlation-kernel KRR |
| C | FC^0.35 (F-BERT's own teacher, `V D^0.35 V^T`) | correlation-kernel KRR |
| **D0** | ours without the kernel filter: shrink g → partial correlation → α = 0.10 → mix 5 % full; **g chosen inside each training fold** | correlation-kernel KRR |
| D1 | D0 + kernel spectral filter (τ 0.5, keep 150; benchmark protocol, label-free, uses all subjects' features) | same |

Probes. P1 = the paper's probe: features z-scored with training-fold statistics,
linear kernel, ridge. P2 = the correlation kernel with HKH centring that the FC
arms use. λ for every arm chosen by inner 4-fold CV on one shared grid,
`logspace(-6, 2, 17)` on trace-normalised kernels. F-BERT gets the better of P1
and P2 **per seed-averaged r** — a choice that favours F-BERT and makes any
margin conservative.

D0 is the primary "ours" because it is strictly per-subject like F-BERT (no
subject's representation depends on another's). D1 uses the benchmark's
transductive kernel filter and is reported separately.

## Primary contrast and decision rule

**D0 − A8**, paired per seed over 60 seeds. "Beats F-BERT on identical folds"
is claimed only if the difference is positive in ≥ 57 / 60 seeds (95 %).
Otherwise the result is reported as found, whichever direction.

Secondary, same rule: D0 − A1, D1 − A8, B − A8, C − A8.

## Short scan (secondary, preregistered)

F-BERT's model card says the encoder beats kernel ridge on the raw connectome
on short scans. Repeat A8, A1, B, C, D0 on the **first 2.4 min of REST1_LR
(200 timepoints)**, where FC (450 regions, 200 timepoints) is singular and the
conditioning claim makes its sharpest prediction: D0 − B grows and D0 − A8 stays
positive. Same decision rule.

## Sanity bounds — failure means the pipeline, not the method, is in question

- A8 full scan expected near the paper's 0.502 (n = 955); at n = 753 a value in
  0.42–0.52 is consistent. **Below 0.35 → the rebuild or preprocessing is
  suspect; no claim against F-BERT is made from it.**
- C full scan expected near the paper's 0.642; 0.55–0.66 consistent.

## Not done here

BrainLM, Brain-JEPA, BrainMass, BrainHarmonix, Brain-Semantoks: different
parcellations (A424, Sch-100, 17-network Sch-400, Sch-457) and separate weights;
F-BERT is the strongest of them in the paper's own table (0.502 vs ≤ 0.441), so
it is the one comparison that matters first.

---

## Addendum A — six modalities (written 2026-11-17 after the resting-state result, before any task number)

The resting-state result (`FBERT_SAMEFOLDS_RESULT_1117.md`) cannot be set
against the project headline (0.7344), which is a six-modality ensemble. So the
same comparison is repeated with all six modalities for **every** arm.

- Task runs: `tfMRI_<T>_{LR,RL}_Atlas_MSMAll.dtseries.nii` for LANGUAGE, SOCIAL,
  GAMBLING, MOTOR, WM; same 450-region averaging; each run detrended,
  band-passed 0.01–0.08 Hz, z-scored; LR and RL concatenated per task.
  F-BERT was pretrained on resting **and task** fMRI, so task input is in its
  distribution. Resting state as in the main comparison.
- Per modality: plain FC (B), FC^0.35 (C), ours D0 / D1 (per-g), F-BERT-8M and
  -1M embeddings.
- Ensembles, the same construction for every arm: weighted mean of the six
  per-modality kernels. **Primary weighting: equal.** Secondary: language ×2
  (the canonical weighting, tuned on these subjects with the benchmark's
  features — applied to every arm alike).
- F-BERT ensemble gets the better (seed-averaged) of: mean of the six
  correlation kernels, or one linear probe on the six embeddings concatenated
  and z-scored in-fold.
- Ours: one g shared by the six modalities, chosen inside each training fold
  with λ (7 candidates).
- **Primary contrast: D0-6 − A8-6 (equal weights)**, 60 seeds, claim only if
  positive in ≥ 57 / 60. Secondary: D0-6 − A1-6, D1-6 − A8-6, D0-6 − C-6, and
  the same under language ×2.
- Sanity: our six-modality D1 on our own timeseries is expected below the
  canonical 0.7344 (different extraction, MSMAll without ICA-FIX on task runs,
  450 regions); a value in 0.62–0.74 is consistent. F-BERT-6 has no published
  reference.

---

## Addendum B — post-hoc development, declared before it runs (2026-11-17 evening)

**Status: development after a failed primary, not a confirmation.** Addendum A's
primary (D0-6 − F-BERT-8M = −0.0074, 9/60) stays on record as failed. This
addendum fixes, before running, the *only* candidates that will be tried, and makes
the choice among them part of the nested CV, so the reported number is the honest
performance of "pick among these inside the training fold".

Diagnosis: ours wins every modality alone but gains less from combining
(+0.115 rest→six vs F-BERT +0.18), consistent with our six conditioned
connectomes sharing a large subject-specific component. Remedy from the
literature: general functional connectivity (Elliott et al. 2019, NeuroImage) —
one FC from all of a subject's timeseries, rest + tasks (~7,800 timepoints), which
is better conditioned and more reliable.

Candidate kernels (all per-subject representations, no labels, no cross-subject
filter; λ grid and inner splits as before):

| family | kernels |
|---|---|
| K1 | D0-6, shared g ∈ {0.90, 0.95, 0.97, 0.98, 0.99} |
| K2 | GFC-D0: time-weighted mean of the six Pearson FCs (= FC of the concatenated z-scored runs), then the D0 chain, g ∈ {0.85, 0.90, 0.95, 0.97, 0.98} |
| K3 | ½ K1(g) + ½ K2(g), g ∈ {0.90, 0.95, 0.97, 0.98} |
| K4 | multi-g D0-6: per modality the mean of D0 at g ∈ {0.85, 0.90, 0.95, 0.98} |
| K5 | ½ K1(g) + ½ FC^0.35-6, g ∈ {0.95, 0.97, 0.98} |

**Primary B: "PLUS" = in-fold selection over all 18 kernels, minus F-BERT-8M
six-modality concatenated probe (its best arm, re-run in the same job).**
Claim only if positive in ≥ 57 / 60.

**Fairness gate before any claim:** if PLUS wins through K2/K3 (general FC), an
F-BERT arm reading the same concatenated rest + task sequence is run, and the
claim requires PLUS to beat that arm too (≥ 57 / 60). Every family's own nested
value is reported, winners and losers. No further candidates are added after
this runs; if it fails, the failure is reported.

---

## Addendum C — third development round, declared before it runs (2026-11-17 evening)

**Status: development, not confirmation.** Primaries of A and B failed and stay
on record. This is the last declared round for this comparison today.

Label-free diagnostic that motivates it (`results/FBERT_REDUNDANCY_1117.json`,
no outcome used): mean pairwise kernel alignment (CKA) between modalities is
**0.68 for our six conditioned connectomes vs 0.34 for F-BERT-8M's six
embeddings**. Our modalities share a large subject-specific component, which the
kernel mean counts six times; B's general FC (shared part only) fell to 0.630.

Remedy, per subject and label-free: with x_m the conditioned partial-correlation
edges of modality m and s = mean_m x_m, use d_m = x_m − β·s, then the usual
modality kernel 0.95·K(d_m) + 0.05·K(full FC_m) and the six-modality mean.

| arm | candidates (all chosen in-fold with λ) |
|---|---|
| **DECOR** (ours) | g ∈ {0.95, 0.97, 0.98} × β ∈ {0, 0.25, 0.5, 0.75, 1.0} (β = 0 is the old ensemble) |
| **A8DECOR** (F-BERT, same tool) | e_m − β·ē on the six F-BERT-8M embeddings, concatenated, z-scored in-fold, β ∈ {0, 0.25, 0.5, 0.75, 1.0} (β = 0 is A8cat) |

**Primary C: DECOR − max(A8cat, A8DECOR)**, i.e. against whichever F-BERT arm is
better (seed-averaged). Claim only if DECOR beats **each** of A8cat and A8DECOR in
≥ 57 / 60 seeds. The kernel-alignment of the selected β is reported as mechanism.

---

## Addendum D — confirmation on subjects outside the 753 (written before any held-out number)

Agreed with the main session (session): this session owns the out-of-753
held-out test; the main session owns the Ooi benchmark line and the paper.

**Data.** All HCP-YA subjects in `ooi_comp.csv` with `varimax_cog` that are not in
the 753 (269 listed), with all six modalities extracted exactly like the 753
(`run_extra.sh`: rest 4 runs + 5 tasks LR/RL, 450 regions, same cleaning,
same F-BERT order). Held-out = those with complete six modalities.
No family IDs exist; relatives may cross the split. Paired differences are
inflated equally on both arms, so the *difference* is the tested quantity.
Sensitivity: the main session's similarity control (keep held-out subjects whose
max rest-FC correlation with any training subject, after removing the training
mean edge profile, is below the 90th percentile of within-training maxima).

**Fit.** Train on all 753 once; every hyperparameter (g, λ, F-BERT probe choice)
by 10-fold CV inside the 753; predict the held-out subjects once.

**Arms.**
- Inductive (strict — no held-out subject in any kernel spectrum or centring):
  A8cat (F-BERT-8M concat, train z-stats), D0 (ours, shared g ∈ {0.90, 0.95,
  0.97, 0.98}), H0 = ½ D0 + ½ F-BERT kernel (secondary).
- Batch-transductive (the benchmark's protocol: kernels over train + held-out,
  **no labels**, canonical filter τ 0.5 / keep 150 unchanged): D1 (ours + filter),
  A8F (F-BERT per-modality correlation kernels, filtered, averaged), H1 = ½ D1 +
  ½ A8F (secondary). F-BERT's comparator is whichever of A8cat / A8F the inner CV
  on the 753 prefers; the larger of the two held-out values is also reported.

**Primary D:** D1 − F-BERT (batch-transductive protocol, where development on the
753 found +0.017, 60/60). Claim "confirmed on new subjects" only if the paired
bootstrap (10,000 resamples of held-out subjects) gives P(diff > 0) ≥ 0.95 **and**
D1 also beats the larger F-BERT held-out value.
**Secondary:** D0 − A8cat (inductive; development found −0.007), H0/H1 − F-BERT,
and every contrast on the leak-controlled subset. All reported whatever they show.

---

## Addendum E — second look at the 263 held-out subjects, for the D1 + F-BERT hybrid (written before it runs)

**Status: a second, declared look at the held-out set** (Addendum D used it once).
Motivated by development on the 753 (`results/FBERT_DEV_D1_HYBRID_1117.json`):
D1 mixed with F-BERT-8M's concatenated kernel, chosen in-fold, 0.7396 vs the
canonical 0.7344 — but found after several development rounds on the same folds.

- Arm **HYB** (benchmark protocol, like D1): (1 − ρ)·D1(g, weighting) + ρ·F-BERT
  concatenated z-scored linear kernel (train-only z statistics). D1 kernels over
  train + held-out with the canonical filter, no labels. Candidates g ∈ {0.85,
  0.90, 0.95} × weighting {equal, language ×2} × ρ ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}
  (ρ widened because 0.3 was the development grid's edge), chosen with λ by 10-fold
  CV inside the 753. Train once, predict the 263 once.
- **Primary E:** HYB − F-BERT-8M best (Addendum D comparator), claim if P(diff > 0)
  ≥ 0.95. **Secondary:** HYB − D1 (does adding F-BERT help on new subjects?), HYB
  vs the canonical chain on the main session's 233-subject intersection when its
  file exists. All reported whatever they show.

---

## Addendum F — joint ensemble on the held-out subjects (written before any joint held-out number)

Development (`results/FBERT_JOINT_DEV_1117.json`): the mean of the canonical
chain's prediction (benchmark's 400-region features, main session) and our D1's
prediction (450-region re-extraction) reached 0.7483 vs canonical 0.7357 on the
753 (+0.013, 60/60), with no tuning in the combination.

- **PRED50** on held-out = ½ (canonical held-out prediction from the main session's
  `results/CANON_HOLDOUT_233_1117.json`, benchmark protocol) + ½ (our D1 held-out
  prediction from Addendum D), on their intersection of subjects. No refitting.
- **Primary F:** PRED50 − canonical, paired bootstrap over the intersection,
  claim if P(diff > 0) ≥ 0.95. Secondary: PRED50 − D1, PRED50 − F-BERT, and the
  canonical strictly-inductive version if the file holds one.

### Addendum F — amendment (before any per-subject joint computation; NOT blind)

The main session reports that its 233-subject held-out canonical predictions come
from **its own Schaefer-400 extraction** (CV 0.62 on 728), not from the benchmark's
published features (CV 0.72), because the published features do not exist for
subjects outside the 753. So the developed combination (published-feature canonical
+ our D1) cannot be tested on held-out subjects at all, and "PRED50 − canonical"
would compare against a weaker component — uninformative. Amended:
- **Primary F (amended): PRED50 − our D1** on the 233 intersection — does a second,
  independent extraction add information over ours on new subjects? Claim only if
  P(diff > 0) ≥ 0.95.
- Disclosure: before amending, the aggregate held-out r of both components was known
  (main session's canonical-on-own-extraction ≈ 0.674 first run; our D1 0.786).
  Per-subject predictions of the main session had not been seen.
- The 753 joint result (0.7483) stays a CV result; its honest absolute needs the
  canonical component nested (requested from the main session).

---

## Addendum G — multi-atlas ensemble of our own extractions (written before any 232-region number)

Motivation: two extractions of the same scans were complementary on the 753
(joint 0.7481 vs 0.7348 / 0.7291), but the published-feature half cannot reach new
subjects, and the main session's own-extraction half gave +0.0095 (P = 0.85) on the
233. A second extraction that **we** control for all 1,022 subjects (Schaefer-200 +
Tian S2, 232 regions; `extract_atlas_fc.py`, same cleaning) tests the idea where it
can be confirmed. **This is the fourth look at the 263 held-out subjects** (D, E, F
before it); stated so the family of tests is visible.

- D1-232: the D1 chain (shrink g → partial → α 0.10 → canonical filter τ 0.5 /
  keep 150 → 5 % full mix → six-modality mean) on the 232-region FC; g ∈ {0.85,
  0.90, 0.95} × weighting {equal, language ×2} chosen in-fold / by inner CV.
- **MA = mean of the D1-450 and D1-232 predictions** (no tuning).
- On the 753 (60 × 10 folds): MA vs D1-450 reported (development).
- **Primary G (held-out, train on the 753 once):** MA − D1-450 on the 263, paired
  bootstrap, claim if P(diff > 0) ≥ 0.95. Secondary: MA vs F-BERT; three-way mean
  with the main session's own-400 prediction on the 233 intersection vs D1-450.

---

## Addendum H — F-BERT on a second cohort, AOMIC-ID1000 (written before any AOMIC F-BERT number)

Handoff §8 item 2 applied to the deep-model comparison. The F-BERT paper reports
AOMIC cognition 0.364 (8M) and FC^α* 0.430 under its own protocol; the project's
AOMIC numbers (0.4071 with GSR) are on 400 surface regions without subcortex, so
F-BERT could not read them.

- Data: OpenNeuro ds003097 fMRIPrep `space-MNI152NLin2009cAsym_desc-preproc_bold`
  (movie watching, TR 2.2 s), every participant with `IST_intelligence_total`.
  Parcellated with Tian's `Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_3T_MNI152NLin2009cAsym_2mm`
  (450 regions; each atlas voxel mapped to the BOLD voxel containing it, brain-mask
  voxels preferred), raw parcel means stored. Region order for F-BERT fixed
  label-free as on HCP (cortex 7-network first, subcortex last), checked against the
  atlas geometry.
- Cleaning as the F-BERT card (detrend, Butterworth 0.01–0.08 Hz order 5 zero-phase
  at TR 2.2 s, z-score). Two variants for every arm: no GSR, and GSR (regress the
  450-region mean signal, re-z-score) — the project found GSR required on AOMIC.
- Splits: the project's AOMIC protocol, 20 repeats of 10-fold with
  `RandomState(7000 + s)` over participants sorted by ID. λ by inner 4-fold mean-r
  (`eval_samefolds.krr_select`), grid logspace(−6, 2, 17), for every arm.
- Arms (variant {noGSR, GSR} chosen in-fold together with the arm's other knobs):
  F-BERT-8M and -1M (correlation kernel or z-scored linear, in-fold), plain FC,
  FC^0.35, ours D0 (g ∈ {0.80, 0.85, 0.90, 0.95}, no filter), ours D1 (+ filter).
- **Primary H: D0 − F-BERT-8M** (strictly per-subject, the comparison our method
  lost by 0.007 on HCP's six-modality ensemble; here a single scan, where on HCP we
  won +0.060). Claim if positive in ≥ 19 / 20 repeats. Secondary: D1 − F-BERT-8M,
  D0 − F-BERT-1M, D0 − FC^0.35, D0 − plain FC.

---

## Addendum I — Brain-JEPA on identical folds (written before any Brain-JEPA number)

Second released foundation model that reads the same 450 regions (Dong et al.,
NeurIPS 2024). Authors' code (`src/helper.init_model`, ViT-B, gradient positioning
`data/gradient_mapping_450.csv`, add_w = mapping, crop 450 × 160, patch 16) and their
pretrained `jepa-ep300.pth.tar` target encoder; the build loads with 0 missing / 0
unexpected keys (86.8 M parameters). Region order = Tian subcortex then Schaefer
cortex = our stored atlas order.
- Input per subject: the first 160 timepoints (115 s) of each available resting run
  (≤ 4 windows), linear detrend and z per region, then the authors' per-sample global
  z (`use_normalization`); the UKB robust-scaling medians/IQRs used in pretraining are
  not released — stated as a deviation. Embedding = mean over the 4,500 tokens, then
  over windows (more scan than the F-BERT paper's single 1.9-min window).
- Probe and folds exactly as the main comparison (753, 60 × 10, best of P1/P2).
- **Primary I: D0 (rest) − Brain-JEPA**, claim if positive in ≥ 57 / 60.
  Secondary: F-BERT-8M − Brain-JEPA. Sanity: the F-BERT paper reports Brain-JEPA
  −0.003 frozen on HCP-YA (n = 955).

---

## Addendum J — third cohort, not in F-BERT's corpus: ABIDE-I (written before any ABIDE number)

AOMIC turned out to be in F-BERT's pretraining corpus; ABIDE is not (no ABIDE dataset
in the released corpus). The F-BERT paper reports ABIDE-I autism AUC 0.671 (8M) and
0.778 (FC^α*) under site-grouped CV.

- Data: ABIDE Preprocessed (CPAC, `filt_noglobal/func_preproc`, MNI152 FSL space),
  all subjects with a file and a diagnosis; Tian's
  `Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_2mm`
  (450 regions) mapped to the BOLD grid as on AOMIC (func_mask preferred). Cleaning
  as the F-BERT card (detrend, 0.01–0.08 Hz, z), per-site TR from the phenotype
  table / NIfTI header; variants noGSR / GSR chosen in-fold per arm.
- Targets: autism diagnosis (DX_GROUP; metric AUC of the KRR score on ±1 labels)
  and FIQ (r, where available).
- Folds: site-grouped — 20 repeats of 10 folds with whole sites assigned to folds by
  `RandomState(9000 + s)`. λ by inner 4-fold mean metric. Same probe for every arm.
- Arms: F-BERT-8M / 1M (best of P1/P2), plain FC, FC^0.35, ours D0, ours D1.
- **Primary J: D0 − F-BERT-8M in autism AUC**, claim if positive in ≥ 19 / 20
  repeats. Secondary: the same for FIQ r; D1 − F-BERT-8M; D0 − FC^0.35.

---

## Addendum K — one per-subject family, chosen in-fold (written before any union number)

In every cohort D0 sat just below FC^0.35 (HCP rest +0.0001, AOMIC −0.005, ABIDE
−0.007 AUC / −0.003 FIQ) while both beat F-BERT. Both are closed-form, per-subject
transforms of the same FC. **U** = in-fold choice (with λ) among the D0 kernels
(all g of that cohort's grid) **and** the FC^0.35 kernel, on the same cleaning variants
as before. Same folds, probes and grids as Addenda (main), H and J.
- **Primary K:** U − F-BERT-8M in each of HCP rest (60 seeds), AOMIC (20), ABIDE autism
  AUC (20), ABIDE FIQ (20); claim per cohort at ≥ 95 % of seeds/repeats.
- Secondary: U − max(D0, FC^0.35) per cohort (does choosing in-fold cost or gain?).

---

## Addendum L — fourth cohort, not in F-BERT's corpus: ADHD-200 (written before any ADHD-200 number)

- Data: fcp-indi ADHD200 `Outputs/fmriprep` session-1 resting runs
  (`space-MNI152NLin2009cAsym_desc-preproc_bold` + brain mask), all runs of session 1,
  each cleaned on its own (detrend, 0.01–0.08 Hz at the header TR, z; GSR variant
  regresses the 450-region mean per run) and concatenated. Atlas and mapping as AOMIC.
  Phenotype: each site's `RawDataBIDS/<site>/participants.tsv` (`dx`, `full4_iq`),
  matched by numeric subject ID.
- Targets: ADHD (any subtype) vs typically developing — AUC of the KRR score on ±1;
  `full4_iq` — r.
- Folds: site-grouped, 20 repeats of 10 folds, sites to folds by `RandomState(9100 + s)`.
- Arms, probes and in-fold choices exactly as Addendum J.
- **Primary L: D0 − F-BERT-8M in ADHD AUC**, claim if ≥ 19 / 20. Secondary: IQ r;
  D0 − FC^0.35.

### Addendum L — amendment before extraction (no ADHD-200 number exists yet)

The public fMRIPrep outputs cover 461 subjects; with a diagnosis, 435 subjects in 6
sites (NYU 256, NeuroIMAGE 54, Peking_1 53, OHSU 38, Peking_2 26, Peking_3 8). Sites
without fMRIPrep output, without diagnosis (Brown) or with controls only drop out (the
latter would confound site with diagnosis anyway). With 6 sites the 20-repeat
site-to-fold design collapses to one partition (leave-one-site-out), so: **folds =
leave-one-site-out; primary L decided by a paired bootstrap over subjects of the
out-of-fold AUC difference D0 − F-BERT-8M (10,000 draws), claim if P(diff > 0) ≥ 0.95.**
Everything else as written.

---

## Addendum M — what F-BERT sees in ADHD (diagnostic, written before computing)

**Status: diagnostic follow-up of a post-hoc observation** (within NYU, F-BERT ADHD AUC
0.622 vs D0 0.550). No claim of superiority will be made from it; it decides how the
paper describes F-BERT's one win.
- Motion proxy: the ADHD-200 fMRIPrep outputs carry no confounds file, so motion is
  estimated from the stored raw parcel means — DVARS (RMS over the 450 regions of the
  frame-to-frame change of percent-signal-scaled series), subject mean over runs.
- Closed-form temporal representations from the same cleaned runs: lag-1
  autocorrelation per region (450); lag-1 cross-correlation matrix (450 × 450);
  sliding-window FC variability (window 30 frames, step 5; SD per edge).
- Within NYU, random 10-fold × 20 (as the post-hoc run), AUC:
  M1 DVARS alone; M2 F-BERT-8M and D0 with DVARS regressed out of the score in-fold;
  M3 each temporal representation alone and mixed ½/½ with D0.
- Reading rule, fixed now: if DVARS alone reaches ≥ 0.60 or regressing it removes
  ≥ half of F-BERT's margin over D0, the paper says F-BERT's ADHD edge is
  **motion-confounded**; if a closed-form temporal representation reaches F-BERT, it says
  the edge is **temporal structure a closed-form feature also captures**; otherwise it
  stays "temporal or motion signal the static connectome lacks".

---

## Addendum N — the SOTA system: ours + F-BERT, everything chosen in-fold (written before running)

**Status: designed after seeing where we lose** (ADHD-200 diagnosis and IQ, the HCP
six-modality per-subject ensemble, ADHD within one site). Disclosed as such. The goal is
the best system on every cell, not a new claim about the connectome alone.

- Candidate kernels per cohort (all label-free; same cleaning variants, folds, λ grid
  and inner 4-fold mean-r selection as each cohort's original addendum):
  ours = every D0 kernel (variant × g) and FC^0.35 (and D1 where the original had it);
  F-BERT-8M = its correlation kernel (and its in-fold z-scored linear kernel);
  hybrids = (1 − ρ)·ours + ρ·F-BERT for every ours kernel, ρ ∈ {0.25, 0.5, 0.75}
  (correlation-kernel F-BERT; on HCP six-modality also the in-fold concatenated
  z-linear F-BERT, as in development).
- **SOTA = in-fold choice over all candidates** (ours, F-BERT, hybrids).
- Cells: HCP rest (60 × 10), HCP six-modality (60 × 10), AOMIC (20 × 10), ABIDE autism
  AUC and FIQ (20, site-grouped), ADHD-200 diagnosis and IQ (leave-one-site-out,
  bootstrap), ADHD within NYU (20 × 10).
- **Primary N, per cell:** SOTA − F-BERT-8M (best probe) and SOTA − best of our
  closed-form arms; "SOTA" is claimed for a cell only if it beats **both** (≥ 95 % of
  seeds/repeats, or bootstrap P ≥ 0.95 for leave-one-site-out). Cells where it does not
  are reported as such.

---

## Addendum O — the two ADHD cells F-BERT still holds (written before running)

Addendum N left ADHD within-NYU diagnosis to F-BERT-8M (0.622) and ADHD cross-site IQ to
F-BERT-1M (0.195). Last declared attempt on these two cells: the SOTA candidate set
gains F-BERT-1M — its correlation kernels and in-fold z-linear kernels (both variants),
and hybrids (1 − ρ)·ours + ρ·F-BERT-1M and (1 − ρ)·F-BERT-8M + ρ·F-BERT-1M (ρ ∈ {.25, .5,
.75}), all chosen in-fold. Same folds as N.
- **Primary O:** in each of the two cells, SOTA_O must beat the best of F-BERT-8M, F-BERT-1M
  and ours (NYU: ≥ 19/20 repeats; LOSO: bootstrap P ≥ 0.95). If not, F-BERT keeps the cell
  and the result is written that way. No further attempts on these cells after this.

---

## Addendum P — our own encoder: pretrain on F-BERT's public corpus with our teacher (written before training)

Why: F-BERT keeps ADHD within-site diagnosis (temporal signal static kernels lack), and on
single-scan cohorts the best arm is the F-BERT paper's closed-form FC^0.35. An encoder that
learns temporal structure like F-BERT, but is distilled from a better teacher, is the one
legitimate route to the top of every cell.

- Corpus: `Marraffini-Giovanni/fbert-corpus` windows (106,397 × 450 × 80) **minus every
  AOMIC dataset (ds003097, ds002785, ds002790)** so no evaluation subject is seen in
  pretraining (stricter than F-BERT). 5 % of the remaining recordings held out for
  label-free checkpoint selection (validation CKA).
- Teachers per recording (whole recording = its windows concatenated):
  **T-D0**: our conditioned connectome (shrink g = 0.9 → partial → α 0.10), strict upper
  triangle, centred, unit norm, concatenated with √0.05 × the FC vector (Gram = 0.95 K_D0 +
  0.05 K_FC, our kernel); **T-C**: FC^0.35 (F-BERT's own teacher — reproduction control);
  **T-mix**: ½ T-D0 + ½ T-C (Gram average).
- Student: F-BERT-8M architecture (rebuilt and validated in Addendum 0), from scratch,
  1 − CKA loss, batch 1,024 windows from distinct recordings, AdamW 8.5e-4 / 1e-4, warm-up
  100, cosine, bf16, clip 1, checkpoints every 2,000 steps to 30,000; the checkpoint with
  the best held-out validation CKA is frozen (no evaluation-cohort label is used).
- Evaluation: frozen embeddings, every cell of Addenda (main), A, D, H, J, L, M with the
  same folds and probes. Arms: C-BERT (T-D0), F-BERT-repro (T-C), C-BERT-mix (T-mix),
  each alone and inside the Addendum-N SOTA system.
- **Primary P:** C-BERT-best (the variant with the best held-out validation CKA among T-D0
  and T-mix — chosen label-free) − F-BERT-8M (released) on every cell; and the SOTA system
  with C-BERT in its pool − the best previous arm on every cell (ADHD NYU diagnosis 0.622,
  ADHD cross-site IQ 0.197, AOMIC 0.417, ABIDE 0.787 / 0.257, NYU IQ 0.351, HCP cells).
  Same decision rules as each cell's original addendum.

### Addendum P — amendment before any evaluation (training in progress, no embedding computed)

Validation CKA is measured against each model's own teacher, so it cannot rank T-D0 against
T-mix. **Primary C-BERT = the T-D0 model**; between the 8M and 38M T-D0 runs the one with the
higher held-out validation CKA (same teacher, comparable) is primary. T-mix and the T-C
F-BERT reproduction are secondary; the reproduction must land near the released F-BERT on
HCP rest (0.525) for the training pipeline to be trusted — if it falls below 0.45, every
C-BERT number is flagged as possibly pipeline-limited.

---

## Addendum Q — encoder ensembles and a better-trained 38M (written before running)

**Status: further development after P; the ADHD NYU diagnosis cell has now been attacked
in N, O and P — this is the fourth attempt there, stated so the multiplicity is visible.**

- **Q1 (now, existing 8M embeddings):** the SOTA pool gains encoder-pair hybrids
  (1 − ρ)·A + ρ·B for every pair among {released F-BERT-8M, C-BERT, C-BERT-mix, F-BERT-repro}
  (correlation kernels, same cleaning variant, ρ ∈ {.25, .5, .75}), plus an **ENC** arm =
  in-fold choice over encoder-only candidates (the best "deep" system we can build).
  Cells: AOMIC, ABIDE (autism, FIQ), ADHD (dx, IQ; leave-one-site-out), ADHD NYU (dx, IQ).
- **Q2 (when it finishes):** 38M retrained at lr 3e-4; it replaces the 8M as C-BERT only if
  its held-out validation CKA on the same T-D0 teacher is higher; then the same cells.
- Decision rules unchanged (≥ 19/20, or bootstrap P ≥ 0.95 for leave-one-site-out); a cell
  is "ours" only if SOTA_Q beats every previous arm including released F-BERT-8M and 1M.

---

## Addendum R — drop the motor task: cross-extraction replication (written before running)

The main session found, on its own 400-region extraction, that every motor-including subset
is below the same subset without motor (`results/TASK_SUBSET_MOTOR_DROPS_1117.json`, its file).
That rule was set on **their** data, so applying it unchanged to our 450-region extraction is
a fair replication test, not a choice read off our outcomes.
- On the 753, 60 × 10: D1 (filtered) and D0 (per-subject) ensembles over the six modalities vs
  the same ensembles **without motor**, equal and language-×2 weights, g chosen in-fold.
- **Primary R:** (no-motor) − (all six) for D1, ≥ 57/60 seeds → "replicated". Secondary: D0,
  and the SOTA system (Addendum N) rebuilt without motor.

- **Q3 (written before any Q2 downstream number is seen):** an ~85M encoder (dim 768, 12 layers,
  12 heads, ff 3072), T-D0 teacher only, lr 2e-4, 30k steps, same corpus and recipe. It replaces
  the 38M as C-BERT only if its held-out validation CKA on T-D0 is higher (38M: 0.7841); then the
  same cells and rules as Q2. No further size or recipe search after Q3 on this corpus.

---

## Addendum S — drop-motor on subjects outside the 753 (written before running)

The drop-motor rule was found and checked only on the 753 (Addendum R; main session's
2×2). Subjects outside the 753 (263 with all six modalities, Addendum D) never informed it,
so this is a confirmatory test. Same pipeline as Addendum D (train on all 753, every
hyperparameter by inner CV inside the 753, predict the held-out once; equal weights).
- **Primary S:** D0 (inductive, unfiltered — where R replicated) no-motor − six, paired
  bootstrap over held-out subjects (10,000 draws): "confirmed" if P(diff > 0) ≥ 0.95.
- Secondary: D1 (batch-transductive, filtered) no-motor − six; F-BERT best no-motor − six;
  D0/D1 no-motor vs F-BERT best; the leak-controlled subset for each.
- Whatever comes out, the main session's canonical headline should report six-modality and
  no-motor numbers side by side with this test's result, since the drop was chosen on the 753.

---

## Addendum T — the unused HCP tasks (RELATIONAL, EMOTION): cross-extraction replication (written before extraction)

The main session found that its chains never used two of HCP's seven tasks, and that adding
RELATIONAL helped in-fold (+0.013 to +0.016, 30/30) and pointed the same way on its 233 new
subjects (+0.0099, P = 0.94); EMOTION hurt (`results/SEVENTH_TASK_RELATIONAL_1117.json`, its
file). That was found on its 400-region extraction, so testing it unchanged on our 450-region
extraction (not yet extracted for these tasks) is a replication.
- Extraction: tfMRI_RELATIONAL and tfMRI_EMOTION, LR+RL, same atlas, cleaning and F-BERT
  embedding as the five tasks (Addendum A), for the 753 and the held-out subjects.
- **Primary T:** 753, 60 × 10, D1 (filtered; g and language weighting chosen in-fold):
  rs + 5 tasks + RELATIONAL vs rs + 5 tasks, ≥ 57/60 seeds → "replicated in-fold".
- **Co-primary T-holdout:** train on the 753, predict the held-out once (Addendum D pipeline):
  D1 with vs without RELATIONAL, paired bootstrap P(diff > 0) ≥ 0.95 → "confirmed on new subjects".
- Secondary: D0 the same; EMOTION added (expected to hurt); F-BERT-8M given RELATIONAL too
  (fairness), and D1+RELATIONAL vs F-BERT+RELATIONAL on new subjects.
