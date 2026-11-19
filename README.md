# connectome-cognition-benchmark

Code, per-fold results and pre-registrations for

> Pyo C. *Conditioned connectomes and a transductive subject-kernel filter on a connectome-to-cognition benchmark: what holds under the benchmark's own scoring and on held-out subjects.* Preprint in preparation (2026).

This repository re-scores the public HCP connectome-to-cognition benchmark of Ooi et al. (2022,
NeuroImage 263:119636, doi:10.1016/j.neuroimage.2022.119636) the way its authors score it, evaluates a kernel
ridge pipeline that conditions each subject's connectome and filters the spectrum of the between-subject kernel,
and reports which parts of the gain survive a factorial ablation, an inductive version of the filter and a
pre-registered test on HCP subjects outside the benchmark.

## Main results (all numbers from `paper/ledger/numbers_index.json`)

| Result | Value |
|---|---|
| Our pipeline, fixed settings (mean per-fold r, age band and sex regressed in-fold) | 0.7215 |
| Same, four settings re-chosen inside each fold | 0.7225 |
| Best model in the benchmark paper | 0.6779 |
| Benchmark's kernel, re-run here under identical scoring | 0.6798 |
| Ours − re-run benchmark kernel | 0.0417, Nadeau–Bengio corrected t = 3.23 |
| Same comparison without the kernel filter | 0.0150, t = 1.33 (not significant) |
| Kernel filter computed from training subjects only (inductive) | 0.6886 (0.0087 vs the benchmark kernel, t = 0.72) |
| Held-out subjects outside the benchmark (263), pre-registered: filtered pipeline (kernel and filter computed with the held-out subjects present) − benchmark kernel | 0.064 (95% interval 0.030 to 0.099) |
| Held-out subjects: inductive unfiltered variant − benchmark kernel | 0.047 (0.021 to 0.076) |
| Held-out subjects: filtered minus inductive unfiltered | 0.017 (−0.008 to 0.041), not resolved |
| F-BERT-8M, frozen, identical subjects, inputs and splits: resting state, unfiltered variant vs F-BERT (mean per-fold r, raw target) | 0.5913 vs 0.5235, corrected t = 3.88 |
| Same, six modalities (F-BERT embeddings concatenated): unfiltered / filtered variant vs F-BERT | 0.7075 / 0.7193 vs 0.7026, t = 0.52 / 1.40 |

**Read with these limits.** The benchmark margin is transductive: the kernel filter uses all subjects' unlabelled
features, helps only after conditioning, and gives no significant gain when computed from training subjects alone.
The held-out test was pre-registered and run once, but the score of its main arm was already known, that arm
used the held-out subjects' unlabelled features, most of them (242 of 263) were in an early cross-validated
development cohort with their target values, and the held-out pipeline differs from the benchmark pipeline (our own
extraction, equal modality weights, raw target, pooled r). Relatives can cross the split. A public four-level age
band replaced the benchmark's exact age. All tests on the benchmark subjects are exploratory, because the pipeline
was developed on them.

## Layout

| Path | What |
|---|---|
| `code/canonical.py` | the pipeline (per-subject conditioning, correlation kernel, kernel spectral filter) |
| `code/fbert/metric_check.py` | the benchmark's scoring (per-fold r, in-fold covariate regression) and the corrected t |
| `code/fbert/nested_matched.py` | settings re-chosen inside each fold |
| `code/fbert/review_arms.py` | factorial ablation, inductive filter, single-implementation decomposition |
| `code/fbert/eval_holdout.py`, `eval_holdout_chain.py` | the held-out tests |
| `code/fbert/eval_samefolds.py`, `eval_sixmod.py`, `aomic_eval.py`, `abide_eval.py`, `adhd_eval.py`, ... | foundation-model comparisons; `fold_scores.py` and `fold_nb.py` add exact fold-level tests |
| `code/fbert/fbert_model.py` and the `extract_*` / `features_*` scripts | our rebuild of the F-BERT forward pass and our own extraction |
| `code/clip_cov_gate.py`, `code/rank_lam.py`, ... | changes to the chain tested after the headline was fixed (Table S1); Table S3 names the script behind each result file |
| `preregistration/` | the pre-registrations, with their addenda and outcome notes, including one voided and one never executed |
| `results/` | the result files the paper reads (per-fold and per-split scores; subject-level fields removed) |
| `paper/` | `build_numbers.py` rebuilds `paper/ledger/numbers_index.json` and the figures from `results/`; `claims.py` checks every verbal comparison |
| `MANIFEST.tsv` | size and SHA-256 of every released file |

## Data you need (not included)

- HCP Young Adult imaging and behaviour: ConnectomeDB, under the HCP data use terms.
- The benchmark's features, subject list, splits, per-split results and behavioural component scores: the authors'
  public repository (ThomasYeoLab/Ooi2022_MMP_HCP).
- Sex and age band for the covariate step: the HCP open-access subject table (BALSA, HCP Young Adult, export of the
  selected subjects). It is not redistributed here.
- AOMIC-ID1000, ABIDE-I and ADHD-200 from their public releases; F-BERT and Brain-JEPA weights from their public releases.

The original runs used compute pods with data under `/workspace`; paths in the scripts reflect that. Internal
storage and machine names were replaced by placeholders.

## Reproducing the paper's numbers

`python3 paper/build_numbers.py --figs` re-reads every result file, recomputes the corrected t values from the stored
per-fold scores (and checks them against the values the files recorded), and writes the numbers index and the
figures. The index records the fingerprint of each file it reads. When re-running the analyses, which use
`multiprocessing`, set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` in the shell first; without it each worker
starts its own BLAS threads and the runs slow down by two orders of magnitude. Because subject-level fields were removed before
release, the fingerprints of some released files differ from those in the paper's Table S3; `MANIFEST.tsv` lists
the released ones. The values are unchanged.

## Citation

Archived at Zenodo: https://doi.org/10.5281/zenodo.24355566 (this release). See `CITATION.cff`. Please also cite Ooi et al. (2022) for the benchmark.

## Licence

Code: MIT (`LICENSE`). Result files, numbers index and figures: CC BY 4.0 (`LICENSE-results.md`).
