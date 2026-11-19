# PREREGISTRATION — clip on the 233-subject holdout

**Written and committed BEFORE the result file is read.**
Committed: 2026-11-19 (see the commit timestamp; it precedes any read of
`CLIP_HOLDOUT_VALID.json`).

This is the one claim the project has never established: whether an in-fold
gain survives on subjects no fold ever saw. Every previous attempt failed or
was unresolvable. This registers the test in advance so the rule cannot be
chosen after seeing the answer.

## Question

Does clipping each subject's edge vector at ±1.5 robust sd (the `clip1.5`
transform, +0.0081 in-fold on the covariate-matched condition) improve
prediction on held-out subjects?

## Arms — exactly two, no others

1. **canonical**: the certified chain (g 0.92, partial correlation, α 0.10,
   kernel spectral filter τ 0.5 / keep 150, 5 % full-kernel mix, six
   modalities with language ×2), λ by inner 4-fold.
2. **canonical + clip1.5**: identical, with each subject's edge vector
   clipped at ±1.5 robust sd before the kernel is built.

**The clip value 1.5 is fixed here and no other value (2.5, 1.0, rank, sqrt)
will be tried.** Those other variants were already used to *explain* the
mechanism; using them here as alternatives would be a second selection step.

## Data and protocol

- **Features**: the 999-subject block in `HOLDOUT_RANK_K.npz`, one extraction
  pipeline throughout (this is what makes the test valid — the earlier attempt
  failed because train and test came from different pipelines).
- **Train**: all benchmark subjects present in the block.
- **Test**: every block subject NOT in the 753 that has a finite
  `varimax_cog`.
- **Covariates**: age band (4 levels) + sex regressed out, fitted on the
  training set and applied to the test set.
- **Metric**: pooled r (concatenate all test predictions), as stated.
- **Uncertainty**: subject-level bootstrap, 3000 resamples. If predictions are
  bagged over training subsets, the bootstrap is over the BAGGED predictions,
  per subject — not over training repetitions (the P = 1.000 lesson).

## Decision rule — fixed in advance

> **CONFIRMED** only if the 95 % bootstrap CI lower bound of
> (canonical + clip1.5) − (canonical) is **greater than 0**.
>
> **Every other outcome is NOT CONFIRMED**, including a positive point
> estimate with a CI that includes 0.

The rule is one-sided in the direction the in-fold result predicts. It is
stated before the numbers are seen, so it applies as written.

## Run policy

**One run.** No re-running with different seeds, splits or clip values in
search of a positive interval. If the first run is NOT CONFIRMED, that is the
answer and the finding stays "in-fold gain, not established out of sample."

## What this test can and cannot settle

- It settles whether the transform helps on **new subjects from the same
  extraction pipeline**.
- It does **not** settle whether it helps on a new cohort, a new scanner or a
  different parcellation.
- A NOT CONFIRMED result does not retract the in-fold gain; it caps its
  standing at "in-fold, unconfirmed out of sample" — the same standing the
  ledger gives every other lever in this project.

## Correction carried in

The NB t reported for clip's in-fold runs was inflated 17.9× by a helper bug
(fixed 2026-11-19, commit `0f84617`). Correct in-fold p values are 0.12–0.15,
i.e. **not significant**. This test is therefore against a point estimate the
in-fold analysis already calls unconfirmed — which is exactly why the holdout
decides it.

---

## OUTCOME (added 2026-11-19, after the run)

**The execution did not follow this protocol, so the test is void.**

`CLIP_HOLDOUT_VALID_1119` differs from the registration in six ways: all six
modality kernels were built from the same resting-state features (the loop
never used `m`), clip was applied before the α re-weight rather than after,
covariates were not regressed, training subsets were 60 % rather than all
benchmark subjects, the decision metric was absolute error rather than the
pooled-r difference, and 246 subjects were evaluated instead of 233.

Its verdict is therefore **INVALID EXECUTION**, not "not confirmed". The
reading "clip reverses out of sample" is withdrawn.

**Not rerun.** At ~246 test subjects the bootstrap half-width is about 0.018
(RELATIONAL holdout: +0.011 with CI [−0.006, +0.029]), so an effect of +0.008
would read "not confirmed" even if real. A rerun would add a number and no
information. clip's standing is "in-fold gain, out of sample untested".
