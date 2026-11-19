# PREREGISTRATION — does the conditioning chain beat plain full FC on NEW subjects?

**Committed before the result file exists** (2026-11-19; the pod has no
`CHAIN_HOLDOUT.json` at the time of this commit).

This tests the project's *central* claim out of sample, not a lever bolted on
top of it. The in-fold result is +0.0496 over plain full FC (60/60); nothing
in the ledger has ever tested whether the chain's advantage survives on
subjects the fit never saw.

## Question

On held-out subjects, does the certified conditioning chain (shrink → partial
correlation → α re-weight → kernel spectral filter → 5 % full mix → six-modality
ensemble) predict cognition better than a plain full-correlation kernel?

## Arms — exactly three, no others

1. **plain_full**: HKH-centred Pearson kernel on the raw correlation edges,
   six modalities, language ×2, λ in-fold. This is the "before" arm of the
   additive chain.
2. **chain**: the certified chain, identical otherwise.
3. **chain_no_filter**: the chain without the kernel spectral filter. Included
   because the filter is the largest step in-fold; if the out-of-sample gap
   comes from the filter this arm localises it.

No other settings, no scanned variants.

## Data and protocol

- `HOLDOUT_CLIP_K.npz` — 974 subjects, **one extraction pipeline throughout**
  (this is what makes the test valid).
- Train: the 728 subjects also in the benchmark 753.
- Test: the 246 subjects in the block that are NOT in the 753.
- **Covariates**: age band (4) + sex, fitted on the training set only.
- **Metric**: pooled r over the test subjects.
- **Uncertainty**: subject-level bootstrap, 3000 resamples, over the mean
  prediction across training subsets (bag first, then bootstrap — the
  P = 1.000 lesson).
- Training subsets: 30 random 60 % draws of the 728, predictions averaged.

## Decision rule — fixed in advance

> **CONFIRMED** only if the 95 % bootstrap CI **lower bound** of
> (chain − plain_full) is **greater than 0**.
>
> **Every other outcome is NOT CONFIRMED.**

## Run policy

**One run.** No re-running with different splits or seeds in search of a
positive interval.

## What the three outcomes would mean

- **CONFIRMED**: the conditioning chain's advantage is not a fold artefact —
  the strongest possible statement this project could make, and the first
  out-of-sample confirmation of its central claim.
- **NOT CONFIRMED, positive point estimate**: the in-fold gap shrinks out of
  sample (the pattern of every other lever here); report the size.
- **NOT CONFIRMED, negative**: the chain's in-fold advantage reverses out of
  sample, which would mean the 753 comparison is measuring fold structure as
  much as biology. That would be the most important negative in the project.

---

## STATUS (added 2026-11-19): registered, NOT executed

Superseded before it ran. Its plain-full arm would have been built the same way
as the invalid clip holdout — `E = bpp` inside `for m in MODS`, i.e. one
resting-state feature set for all six modalities — so it would have repeated
the same defect. Not run; the central claim (chain vs plain full, +0.0496
in-fold) remains tested only on the 753.

Headline unchanged: 0.7215 vs their 0.6779.
