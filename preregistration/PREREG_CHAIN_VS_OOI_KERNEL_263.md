# PREREGISTRATION — does the chain beat the benchmark's own kernel on new subjects?

**Committed before the run.** No result file exists for this test at commit time.
Script: `code/fbert/eval_holdout_chain.py`. Output: `results/CHAIN_VS_OOI_KERNEL_263_1119.json`.

## Why

The paper's headline (0.7215 vs 0.6798 for the benchmark's own kernel, both scored the
benchmark's way) is measured on the same 753 subjects the chain was developed on. Nothing
in the record tests the chain against the benchmark's kernel on subjects that were never
used for development. The 263 HCP subjects outside the 753 (prereg Addendum D,
`results/FBERT_HOLDOUT_1117.json`) are the only such set with all six modalities extracted
by the same pipeline as the training subjects.

The withdrawn holdouts (`CLIP_HOLDOUT_VALID_1119`, `PREREG_CHAIN_HOLDOUT.md`) built every
modality from one resting-state array. This test does not use that code: it reuses the
Addendum D loader and kernels unchanged and only adds baseline arms.

## Data and protocol (identical to Addendum D)

- Features: our 450-region extraction (Schaefer-400 + Tian S3), six modalities, Pearson FC
  edges. Training rows: the 753 benchmark subjects with all six modalities and the target.
  Test rows: the 263 subjects outside the 753 with all six modalities and the target.
- Target: `varimax_cog`, raw (Addendum D did not residualise; kept for the sanity gate).
- Train once on all training rows; (kernel, λ) chosen by 10-fold CV inside the training
  rows (`inner_select`, seed 0); predict the test rows once.
- λ grid `logspace(-6, 2, 17)` for every arm (trace-normalised kernels reach their ridgeless
  limit at 1e-6; the CBIG kernel has unit diagonal, so its floor is lower still).
- Six modalities averaged with equal weights in every arm (as in Addendum D).

## Arms — exactly five

| arm | kernel | protocol |
|---|---|---|
| D1 | chain: shrink (g in {0.90, 0.95, 0.97, 0.98}, chosen in CV) → partial correlation → α 0.10 → correlation kernel → spectral filter (τ 0.5, keep 150) → 5 % full mix | batch-transductive (kernel over train + test features, no labels), as in Addendum D |
| D0 | same chain without the spectral filter, train-only centring | strictly inductive, as in Addendum D |
| CBIG6 | the benchmark's kernel: Pearson correlation between two subjects' edge vectors (CBIG `corr` kernel) on full FC | inductive (no across-subject step) |
| PLAIN_T | correlation kernel on full FC, centred over train + test, no filter | batch-transductive, no filter |
| PLAIN_I | correlation kernel on full FC, train-only centring | strictly inductive |

## Sanity gate (read before any contrast)

D1 and D0 must reproduce Addendum D: held-out r 0.7863 and 0.7696 within ±0.0005. If
either fails, the run is void, no contrast is read, and the result file records only the
gate.

## Primary contrast and decision rule

Pooled r on the test rows; paired subject bootstrap of r(arm) − r(CBIG6), 10,000 draws
(seed 1).

> **CONFIRMED** only if the 95 % CI lower bound of **D1 − CBIG6** is above 0.
> Every other outcome is **NOT CONFIRMED**.

## Secondary contrasts (reported whatever they show; no decision rule)

D0 − CBIG6 (inductive chain vs the benchmark's kernel) · D1 − PLAIN_T · D0 − PLAIN_I ·
D1 − D0 (the filter on new subjects) · all of them on the similarity-filtered subset too.

## Run policy

One run. No change of arms, grid, seeds or subjects after the result is seen.

## Known limits, stated now

- Relatives can cross the split; connectome similarity cannot detect them (Addendum D
  family check). The leak-controlled subset is reported, not used for the decision.
- The 263 are a different population (wider target range); every arm's r rises.
- This is our extraction, not the benchmark's published features, so it tests the chain
  against the benchmark's *kernel*, not against the benchmark's published numbers.

---

## OUTCOME (added after the single run, 2026-11-19)

Sanity gate passed: D1 0.7863 and D0 0.7696 reproduce Addendum D exactly.
**Primary D1 − CBIG6 = +0.064, 95 % CI [0.030, 0.099] → CONFIRMED.**
Secondary: D0 − CBIG6 +0.047 [0.021, 0.076]; D1 − PLAIN_T +0.079 [0.043, 0.116];
D0 − PLAIN_I +0.060 [0.036, 0.085]; **D1 − D0 (the filter) +0.017 [−0.008, 0.041] — not
established on new subjects.** Held-out r: D1 0.7863, D0 0.7696, CBIG6 0.7222,
PLAIN_T 0.7077, PLAIN_I 0.7101. `results/CHAIN_VS_OOI_KERNEL_263_1119.json`,
log `results/CHAIN_VS_OOI_KERNEL_263_1119.log.txt`.
