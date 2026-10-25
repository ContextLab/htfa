# htfa overnight report (2026-10-24 22:50 → 2026-10-25 05:40 EDT)

The machine ran at load average 60-190 on 12 cores for most of the night (other sessions'
jobs), so timings below are upper bounds; idle-machine benchmarks are being re-run.

## Decisions waiting for you

1. **HTFA template update (inherited from BrainIAK).** Each global iteration turns the
   template posterior into the next prior and revisits the same subjects, so the posterior
   covariance is `(S0^-1 + i·n·Sigma^-1)^-1`: identical for every factor, independent of the
   data, and shrinking with the iteration count. On 12 real subjects (docs: Validation →
   "HTFA template update (audit)"): template-center posterior SD 2.64 → 0.71 mm over 15
   iterations while subjects stay 12-20 mm from the template; with the covariance held at its
   one-pass value, template movement levels off at 0.9-1.4 mm/iteration (vs 0.12 mm) and
   subject spread at 14.2-14.5 mm (vs 12.1 mm). So part of HTFA's apparent convergence and of
   the pull of subjects toward the template comes from the shrinking prior. The MAP update
   itself is computed correctly (matches an independent precision-form posterior to 4e-14).
   Not changed ("fix bugs only"). Options: keep BrainIAK's behaviour; hold the template
   covariance at its one-pass value; or use the across-subject dispersion as the local prior.
2. **Release:** PyPI publication (Task 13) and Read the Docs (needs your RTD account).
   Nothing published.

## What is on main (all pushed; CI green on ec4b211: 15 jobs, 3 OS × Python 3.10-3.14)

- **Faster solver (Task 11b).** New default `nlss_method='bounded-lm'`: bounded
  Levenberg-Marquardt on the exact K(D+1)-square normal matrix (true Hessian when positive
  definite, Gauss-Newton otherwise); scipy trf/dogbox remain selectable. Same optimum as scipy
  trf in 96/96 cases (relative cost difference -1.1e-6 to +1.5e-7). Per outer iteration, K=100,
  loaded machine:

  | shape | before (trf+LSMR) | after (bounded-lm) |
  |-|-|-|
  | 3000x300 subsample | 22.9 s | 0.67-1.07 s |
  | 10000x500 subsample | 586 s | 2.6 s |
  | 2000x100 data, 300x30 caps | 21.8 s | 0.68 s |
  | full 70k x 1137, linear loss | 2271 s | 11.2 s |
  | full 70k x 1137, soft_l1 | stopped after 85 min | 66 s |

  Review fixes: memory of the chunked assembly bounded (K=200, V=70k: 4.77 → 1.30 GB peak);
  basin-jump regression locked in a test; MPS float32 matches CPU at 20000x300, K=100 (centers
  within 3.5e-4, weights r = 0.9999999999), so `backend='auto'` uses Metal above 5e8 V·T·K.
- **Real-data usability (Task 11c, from the directed-forgetting pilot):** subsampled fits stop
  on the full-data objective (the scaled change has a subsampling noise floor: 0.56 at
  3000x300, 0.31 at 10000x300 on real data); verbose output flushed; `n_threads` respects
  existing BLAS limits; HTFA exposes `free_fits_`, `template_history_`, `template_max_iter`,
  `fit(starts=, init_template=)`; 2D images keep black pixels by default. Remaining items
  (dependency floors + minimum-versions CI job, KMeans start cost) in progress.
- **Documentation (Task 12):** Sphinx site (furo, ContextLab palette, Nunito Sans): index,
  installation, inputs, optional dependencies, performance, validation, changelog, API
  reference (every public name), 9 executed gallery examples, tutorials 01-06; README and
  CONTRIBUTING rewritten. `sphinx-build -W` clean; checked visually in light and dark.
- **BrainNet-style network plots (Task 14, requested through the directed-forgetting session):**
  `htfa.plot_network`, `htfa.animate_network`, `htfa.plot_timecourse`, `htfa.brainnet`
  (BrainNet .node/.edge/.nv I/O). Renders the 2014 frames within 2.02-2.65/255 mean pixel
  difference. BrainNet Viewer is GPL, so its meshes are downloaded from NITRC on first use
  (SHA-256 verified), not bundled. Code adapted from directed-forgetting-network-dynamics
  (MIT, attributed).

## Evidence of correctness

- **Synthetic recovery:** 3D 20³, K=8: largest center error 0.0164 grid units, width error
  0.31%; inference reduces the start's error 64-fold (centers) and 30-fold (widths).
- **BrainIAK 0.12 cross-check (CI):** residual agrees to 7.6e-8, factors to 2.5e-14; TFA center
  error 0.016-0.020 (ours) vs 0.086-4.89 (BrainIAK); HTFA template 0.05-0.06 vs 3.8-4.8.
- **Real fMRI** (nilearn development_fmri, 12 adults watching "Partly Cloudy", K=50; figures in
  docs/_static/validation/):
  - Leak-free held-out reconstruction (train/test halves with a gap; test weights from one
    voxel set, R² scored on another): inference beats its own frozen start in **12/12 subjects
    from both starts** (mean R² ncorr 0.034 → 0.041, kmeans 0.033 → 0.038; random centers 0.027).
    Matched-K PCA is higher (0.047), as expected for unconstrained components.
  - Timepoint decoding (462 group splits, 6-TR windows, chance 0.036): frozen start 0.570, HTFA
    template factors 0.549, Harvard-Oxford atlas 0.484, HTFA local weights 0.341. Inference does
    not improve decoding over the frozen start. The decoding procedure followed my summary of
    Manning et al. (2018), not the paper itself; worth checking against the paper.
- **Diagnoses, no bugs found:** (a) the pilot's "non-convergence" is the subsampling noise floor
  (full-data objective decreases every iteration; synthetic recovery keeps improving);
  (b) a single raw image collapses all factors onto the upper width bound because BrainIAK's
  ridge penalty `var(X)` is in raw units (BrainIAK does the same); `standardize=True` fixes it
  (r 0.43 → 0.93). Single samples are still not z-scored by default (your decision); a
  UserWarning now flags raw-scale single-sample data.

## Not done

- Tutorial 09 (Mitchell et al. 2008 word decoding with an ungated embedding model),
  notebooks 07 (parameter recovery) and 08 (GPU/performance), BrainIAK on real data,
  split-half template reliability.
- Task 13 (release workflow), final whole-branch review.
- Headings are lowercased site-wide, following hypertools' CSS; this also lowercases proper
  nouns ("nifti", "brainiak"). Easy to change if you prefer.
