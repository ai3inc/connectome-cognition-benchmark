# PRE-REGISTRATION (DRAFT): does the conditioned connectome beat the benchmark's kernel in cohorts never used in development?

**Status: DRAFT, not yet registered** (written 2026-11-19). The registration takes effect only when this file is
renamed to `PREREG_EXTERNAL_COHORT.md` with status REGISTERED, the analysis code is frozen by SHA-256 in an addendum,
and both are committed. That has to happen before any target value of either cohort is read.

## Why

The paper's held-out test used 263 HCP subjects, of whom 242 had been in an early cross-validated
development cohort with their target values. Its main arm also computed the kernel filter with the test subjects
present. So the paper cannot say that the pipeline generalises to unseen subjects. This test is the missing step. It uses
cohorts that played no part in development, and its primary comparison is inductive: no test subject's
features or labels touch training.

## Data never used, and what was read while designing

- A search of the project (code, results, documents) found no analysis of AOMIC-PIOP1 (OpenNeuro ds002785),
  AOMIC-PIOP2 (ds002790), the Aging Adult Brain Connectome (AABC, which includes HCP-Aging) or HCP-Development.
  AABC was considered on 2026-11-14 and not requested (`archive/superseded_2026-11-19/AABC_APPLICATION_PREP.md`).
  The only HCP-Aging material read then was a public demographics list (age, sex, visit; 727 rows), with no
  cognition and no imaging. PIOP1 and PIOP2 appear only in a list of datasets
  excluded from our encoder pre-training (`code/fbert/cbert_train.py`). AOMIC-ID1000 was used in the paper,
  so it is excluded here.
- While designing (2026-11-19), the PIOP1 and PIOP2 `participants.tsv` files were downloaded to read their column
  names and row counts. Only the header line and the line count were displayed: PIOP1 has 216 rows and PIOP2 226;
  both have `raven_score`, `age` and `sex`. No value was displayed, and the files were deleted at once. One
  subject's scan list and the scan metadata (repetition time) were also read. Nothing else from either cohort was
  opened.

## Cohorts

**Stage 1, public, no application: AOMIC-PIOP1 + PIOP2** (university students, Amsterdam, Philips 3T).
- Released fMRIPrep derivatives.
- Modalities common to both datasets: **resting state** (PIOP1 multiband, repetition time 0.75 s; PIOP2 sequential, 2 s) and
  **working memory** (2 s in both).
- Target: `raven_score`.
- Expected n ≈ 442 before exclusions.

**Stage 2, needs the owner to accept the AABC data use terms: AABC Release 2** (HCP-Aging plus AABC; ConnectomeDB on BALSA,
January 2026).
- 1,396 adults aged 36–90+, 1,390 with MRI. **First visits (V1) only**; later visits are not used, so that no person is
  in both training and test.
- HCP minimal preprocessing plus ICA-FIX, the same pipeline family as the benchmark's data, and long resting scans. Our
  extraction code reads these files.
- Target: the NIH Toolbox Cognition Total Composite (uncorrected) from the in-person V1 visit.
- Covariates: age (the finest released), age², sex, site and mean framewise displacement, regressed in-fold.
- The terms require an academic, non-profit or government e-mail address.
- Fallback if AABC cannot be used: HCP-Development (652) and HCP-Aging (725) of the Lifespan 2.0 release through the NIMH
  Data Archive, analysed per cohort with the same protocol.
- HCP-Young Adult Retest is not usable: its subjects are HCP-YA subjects.

Stage 2 is the decisive test if the Stage 1 power check (below) is low. The protocol is the same for both stages.

## Features

- **Parcellation:** Schaefer-400 (7 networks), the benchmark's cortical parcellation.
  - PIOP: MNI152NLin2009cAsym volumetric fMRIPrep outputs with the 2 mm volumetric atlas.
  - AABC (and the fallback): the CIFTI dense time series with the same parcellation.
- **Denoising, identical for every arm:**
  - Nuisance regressors: 24 motion parameters, white-matter and CSF signals, and the global signal.
  - Band-pass 0.009–0.08 Hz.
  - No scrubbing; first frames dropped as flagged by fMRIPrep.
- FC: the Pearson correlation of parcel time series, one matrix per scan type (AABC: runs concatenated per
  scan type after per-run z-scoring).
- **Exclusions, fixed now:**
  - mean framewise displacement > 0.5 mm, or more than 20% of frames above 0.5 mm, in any scan used;
  - a missing target, age or sex;
  - fewer than 150 frames in a scan;
  - for PIOP, missing either scan type.
- **Sensitivity analysis (reported, never used for the decision):** the same analysis without global-signal regression.

## Arms (exactly these)

All arms use the same FC, the same KRR, the same ridge grid (`logspace(-12, 1, 14)` on trace-normalised kernels) and
the same inner 4-fold selection on the residualised training target. They differ only in the kernel.

| arm | kernel | uses test features? |
|---|---|---|
| **CBIG** | the benchmark's correlation kernel: the Pearson correlation between two subjects' edge vectors | no (pairwise) |
| **D0-I** | conditioned connectome (shrinkage g → partial correlation → negative eigenvalues set to zero → eigenvalue power 0.10 → 5% full-correlation kernel mix), correlation kernel centred and scaled with training-fold statistics only | no |
| **PLAIN-I** | the same correlation kernel on the raw FC edges, training-only centring | no |
| **D1-I** | D0-I plus the kernel spectral filter fitted on the training block (τ = 0.5 SD, keep 150), extended to test subjects by Nyström | no |
| **D1-T** | D0 with the filter computed over all subjects of the cohort, as in the benchmark analysis | yes (unlabelled) |

- Shrinkage g is chosen by inner cross-validation from {0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99}. An edge choice is reported.
- Modalities: the arm's per-scan-type kernels are averaged with equal weights. PIOP: rest and working memory. AABC:
  resting state plus every task that all participants of the cohort have, listed in the code-freeze addendum before
  unblinding.

## Protocol and metric

- **Outer loop:** 10-fold cross-validation repeated 20 times with fixed seeds. Folds are stratified by dataset (PIOP1/PIOP2)
  or acquisition site (AABC), and relatives or households are kept together where identifiers are released.
- **Metric (the benchmark's):**
  - Inside each training fold, the target is regressed on age, sex, mean framewise displacement and, for PIOP, a
    dataset indicator; for AABC, also age² and site. The coefficients are applied to the test fold.
  - Accuracy is the Pearson r between prediction and residualised target in each test fold, averaged over the 200 folds.
- **Test:** the Nadeau–Bengio corrected resampled t over all 200 paired per-fold differences,
  t = mean(d) / sqrt((1/200 + n_test/n_train) · var(d)), with df 199 and n_test/n_train from the recorded fold sizes.

## Primary contrast and decision rule (one per cohort)

**D0-I − CBIG.**
- "Confirmed in cohort X" only if the mean difference is > 0 and the corrected t ≥ 1.972 (two-sided p ≤ 0.05).
- Otherwise the result is reported as found, whichever direction.
- Stage 1 decides PIOP (pooled). Stage 2 decides AABC (or, in the fallback, HCP-D and HCP-A separately). The paper may say "the inductive
  conditioned kernel beats the benchmark's kernel in cohorts never used in development" only if it is confirmed in
  every cohort analysed, with every cohort's result shown.

## Secondary contrasts (reported, no claim on their own)

1. D0-I − PLAIN-I (conditioning alone).
2. D1-I − D0-I (inductive filter).
3. D1-T − D0-I (transductive filter), labelled transductive.
4. D1-T − CBIG.
5. Each primary contrast per modality (rest alone, working memory alone).

Same test; no multiplicity correction; labelled secondary.

## Step 0: power check on the development data, before registration

The protocol above is fixed first. Then the primary contrast is run on HCP-YA subsamples, using data already
used, so nothing is learned about the new cohorts:
- 50 random subsamples of the 753 at the expected usable size (n = 400 for PIOP); AABC's expected size (about 1,300)
  exceeds the 753, so its power is extrapolated from the full 753 with the corrected t scaled by the square root of the size ratio;
- the resting scan truncated to the length of the target cohort's scan (PIOP1 and PIOP2 lengths from their released
  confound files, read before registration: they are not labels);
- covariates age band and sex.

The share of subsamples with t ≥ 1.972 estimates the power for an HCP-sized effect, and the 10th percentile of the
difference gives a minimum detectable effect.
- **Rule fixed now:** if the estimated power for PIOP is below 0.5, Stage 1 is relabelled **pilot** in the registered
  version (reported, no claim), and Stage 2 alone can confirm.
- The power check cannot change any setting above. If it reveals a bug, the fix is recorded and the check is re-run, and that is disclosed.

## Order of execution, label blindness

1. Commit this draft. Write the analysis code (`code/external/`) against FC files only.
2. Run step 0 and record the result in an addendum.
3. Build features for the new cohort. Imaging is downloaded, and the phenotype file is not opened.
4. Dry run with a permuted target. A script reads the target only through a seeded permutation, and the run must
   finish with finite outputs and mean r near zero. This checks the code without information.
5. Freeze the code: SHA-256 of every script and of the feature files in an addendum, then rename to REGISTERED and commit.
6. Unblind once: one run, a log kept, and every arm reported. A crash may be re-run without code changes. A code change
   after unblinding is a deviation, disclosed with both results.

## Not done here

- No foundation-model arm: PIOP1 and PIOP2 are in F-BERT's pre-training corpus, which would favour it.
- No settings tuned on the new cohorts.
- No second look at AOMIC-ID1000, ABIDE-I or ADHD-200.
- The six HCP-YA tasks do not exist in these cohorts, so this tests the conditioned kernel, not the six-modality
  ensemble that produced the benchmark number.

## Owner decisions needed

1. Stage 2 data access: whether the AABC data use terms can be accepted with an eligible academic, non-profit or
   government e-mail address (AI3 is a company; this was the blocker on 2026-11-14). If not, use a collaborator at an
   eligible institution, or the NIMH Data Archive fallback.
2. Whether to start Stage 1 (PIOP) now, after step 0, or to wait for Stage 2.

---

## STATUS (added 2026-11-19): not pursued

The owner decided not to run this test. It was never registered. Nothing from AABC was downloaded. From PIOP1 and
PIOP2, only the header-and-row-count check described above was done, and those files were deleted. The paper's claims
stay as published: the benchmark margin is transductive, and the held-out subjects had been used in development.

---

## RESUMED (2026-11-19, owner decision) — protocol fixed for AABC Release 2; supersedes the cohort sections above

The owner reopened the test, and AABC Release 2 is now the cohort. PIOP1/PIOP2 is not used: AABC is larger, it uses the HCP
pipeline, and it became downloadable. Everything in the sections above still holds unless this addendum changes it.

**Data (no phenotype file has been downloaded).**
- Input: `AABC_Release2_rfMRI_REST_FullCorrelationConnectivity.zip` from ConnectomeDB on BALSA, md5
  ac1e100ee1d70959fa1621c5277cdc18, equal to the server's md5 file.
  - It holds one CSV per parcel, 379 parcels: the 360 HCP_MMP v1.0 cortical areas and 19 FreeSurfer subcortical structures.
  - It is the full correlation of the concatenated resting-state runs, after the provider's removal of structured
    artefacts.
- `code/external/aabc_fc.py` keeps first visits (V1) and writes `aabc_rest_fc_v1.npz` (SHA-256 first 16 hex digits
  998aea06ac58cddd): 1,390 V1 visits, all complete, 71,631 edges.
- Download route: the data use terms were accepted in the owner's BALSA account. The browser's own download request
  supplied a token, and IBM's public Aspera transfer SDK (aspera-cli, `ascli config ascp install`) used it on a compute pod.
  The md5 was verified and the token deleted.
- **Parcellation change:** the parcellation is the provider's HCP_MMP plus subcortex, not Schaefer-400. Nothing about the
  pipeline was tuned on this atlas.
- No head-motion variable is released with the phenotypes, so mean framewise displacement is dropped from the covariates.
  The provider's denoising removed structured motion artefacts.

**Target and covariates (variable names from the data dictionary; no values read).**
- Target: `tlbx_ctc_uncorrected_standard_score` (NIH Toolbox Cognition Total Composite, uncorrected) at the V1 visit.
- Covariates, regressed in-fold: `age_open` (years, truncated at 90), its square, `sex`, and `site` (four recruitment sites).
- Families: `pedid`. Folds keep a pedigree together and are balanced within site.
- Exclusions: a V1 visit without a phenotype row, an incomplete FC matrix, or a missing target, age, sex or site.

**Incidental exposure, disclosed.**
- While choosing files, the owner's screenshots of the BALSA subject table showed values of two cognition
  factor-score columns (a crystallized-IQ factor and one other IQ factor) for about ten visit rows.
- These are not the target and were not used. No other target or covariate value has been seen.

**Analysis code.** `code/external/ext_eval.py`, frozen by SHA-256 in the registration commit. It implements the arms,
grids, folds, covariates, metric and test of this document:
- ridge grid `logspace(-12, 1, 14)`;
- g from {0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99}, chosen in the inner 4-fold loop;
- 10 folds × 20 repeats;
- the corrected t with the test-to-training ratio from the realised fold sizes.

**Decision rule (unchanged).** Primary contrast D0_I − CBIG. "Confirmed" only if the mean per-fold difference is > 0 and the
corrected t is at or above the two-sided 5% critical value with df = 199 (1.972). Otherwise the result is reported as found.
The secondary contrasts are reported, with no claim of their own.

**Power (step 0, replaced by an extrapolation).**
- In the paper, resting-state conditioning without the filter beat the benchmark's kernel at n = 753 with corrected t = 4.15
  (kernel centred over all subjects, benchmark features).
- Scaled to n = 1,390 by the square root of the size ratio, an effect of that size gives t ≈ 5.64, and half that effect
  gives t ≈ 2.82.
- The test is therefore well powered for effects down to about half the HCP size. AABC is the deciding cohort; there is no
  pilot stage.

**Dry run.** Before unblinding, `ext_eval.py --synthetic` runs the full analysis with a synthetic target, covariates and
families, and must finish with mean r near zero. Its output is kept as `results/EXT_AABC_DRYRUN_SYNTHETIC_1119.json`.

---

## REGISTERED (2026-11-19 11:38 KST)

This file was renamed to `PREREG_EXTERNAL_COHORT.md` and committed before any phenotype file of AABC was downloaded.

**Frozen code (SHA-256).**
- `code/external/ext_eval.py`: c382130234e515a69853e75621e5f6aea50757de0298f629b8fa676e53fcf5b8
- `code/external/aabc_fc.py`: 9636be8c33eccc0012fcd50052f6bb068d33152e56e04c211bed3a4ead145d1e
- Input FC file `aabc_rest_fc_v1.npz` (first 16 hex digits): 998aea06ac58cddd

**Dry run (synthetic target, covariates and families; no phenotype read).**
- The full pipeline finished: kernels 73 s, total 746 s, n = 1,390, test-to-training ratio 0.1111.
- For one synthetic target (seed 0), mean per-fold r was 0.03–0.07 across arms.
- With eight further synthetic targets, the per-target means scattered between −0.075 and +0.083, and their average was
  −0.001 to +0.010 per arm. This is chance alignment of a single random target, not leakage.
- Results: `results/EXT_AABC_DRYRUN_SYNTHETIC_1119.json`, `results/EXT_AABC_NULLCHECK_1119.json`.
- The null-check mode (environment variable `DRY_NULL_SEEDS`) is inert in the unblinded run.

**Execution.**
1. The owner downloads `AABC_Release2_Non-imaging_Data-XL.csv`.
2. It is copied to the pod, and its SHA-256 is recorded in the result file.
3. `ext_eval.py aabc_rest_fc_v1.npz <csv> EXT_AABC_1119.json` runs once.
4. Every arm and the verdict are reported as found.

### Format step before the run (2026-11-19 11:40 KST, before any value was read)

The phenotype file arrived (`AABC_Release2_Non-imaging_Data-XL.csv`, SHA-256 first 16 hex digits 3cee2868572fd8e6, 2,563
columns, 6,466 event rows). Its header cells read "variable - description", whereas the frozen code looks columns up by
variable name. To leave the frozen code unchanged, a copy was written whose header keeps only the text before the first
" - ". Every data row is unchanged, and the resulting names are unique. The copy is
`AABC_Release2_Non-imaging_varnames.csv` (9b6e867f9685db2d), and the run reads it. No value of any row was displayed.

---

## OUTCOME (2026-11-19 11:53 KST): CONFIRMED

The registered run was executed once with the frozen code (SHA-256 c382130…) on 1,290 first visits; 100 were excluded for
a missing target or covariate. The sample has 1,271 pedigrees and four sites.

| arm | mean per-fold r |
|---|---|
| CBIG (benchmark kernel) | 0.5257 |
| PLAIN_I | 0.5334 |
| **D0_I (conditioned, inductive)** | **0.5861** |
| D1_I (inductive filter) | 0.5770 |
| D1_T (transductive filter) | 0.5884 |

- **Primary, D0_I − CBIG:** +0.0604, corrected t = 5.47 (df 199, p = 1.3 × 10⁻⁷), positive in 20 of 20 repeats.
  **CONFIRMED.**
- **Secondary:**
  - D0_I − PLAIN_I: +0.0527, t = 5.53.
  - D1_I − D0_I: −0.0091, t = −1.21.
  - D1_T − D0_I: +0.0024, t = 0.32.
  - D1_T − CBIG: +0.0628, t = 4.89.
- **Reading:** the conditioning of each subject's connectome carries an inductive margin over the benchmark's kernel in a
  cohort never used in development. The kernel filter, inductive or transductive, adds nothing measurable here.

Result file: `results/EXT_AABC_1119.json` (per-fold scores, fold sizes, chosen settings; no subject identifiers).
