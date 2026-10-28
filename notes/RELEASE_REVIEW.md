# v0.1.0 review packet — NOT approved for publication

Jeremy explicitly requires careful review before PyPI publication. No PyPI or
TestPyPI upload has occurred. The candidate workflow only builds/tests artifacts.
Draft PR: https://github.com/ContextLab/htfa/pull/184.

## Changes to review

- Standalone HTFA numerical and fitting behavior, including unchanged default RBF
  results checked against the prior implementation and BrainIAK.
- RBF, Laplace and Mexican-hat kernels; fitted component mixtures and invariant
  component relabeling; derivative/recovery/solver checks.
- Nested group templates and permutation tests (independent subjects, not crossed
  or repeated-measures designs).
- Component lobes and repeated edge values in static/animated BrainNet and Nilearn
  plots, signed lobes, native affine handling and optional centroid summaries.
- `htfa.stats`: nominal conditional reconstruction R²/AIC/AICc/BIC. These are NOT
  marginal hierarchical evidence, effective degrees of freedom, or calibrated
  Bayes factors for correlated fMRI residuals/regularized singular mixtures.
- Tutorials 12–15: kernels, nested groups, statistics, real-fMRI comparisons.
- Refreshed assets for tutorials 09–11 and 13, with immutable filenames and hashes.
- Tutorial 09 evaluation is transductive: the unsupervised template and voxel
  standardization use full runs; only the supervised mappings exclude held-out
  words/participants. Its cached null must match exact fit and embedding hashes.

The PCA-limit exploration is local-only and is not part of this branch.

## Review findings already addressed

1. Mixture relabeling clipped reference logits and changed represented fields.
   Matching now only permutes; stable unbounded logits preserve mixture weights.
2. Plot wrappers lost non-default kernel/component information. Both backends
   now receive full fitted component geometry.
3. Stale grouping attributes survived a subsequent ungrouped fit; cleared on refit.
4. Repeated IDs and invalid permutation counts/levels are rejected.
5. Tutorial 09 used a Unix-only load-average API; now portable to Windows.
6. Cached decoding nulls checked names/seeds but not recorded input hashes;
   mismatched fit/embedding hashes now fail closed.
7. Statistics discarded complex parts and could report extreme nonzero residuals
   as perfect fits; inputs/scales now validated, log variance computed stably.
8. Public stats module was missing from API-index coverage; added.

## Evidence and remaining gates

Detailed logs and exact generation provenance are in
`2026-10-28-release-review.md`, `2026-10-28-codex-integration.md` and
`2026-10-28-stats-tutorials.md`. Do not treat older passes as final-head evidence.

- Fresh pre-hardening wheel suite: 999 passed, 17 skipped, 28 deselected.
- Fresh BrainIAK cross-check: 27 passed, 1042 deselected.
- Installed-wheel notebooks 01–06 and 12–15 executed successfully.
- Source-distribution install and package/statistics smoke: 15 passed.
- Final hardened wheel suite, final-head cross-platform CI, updated notebooks
  09–11 and matched decoding nulls are still being completed.
- The author's review/fixes above do not substitute for an independent review.
  Independent review and Jeremy's explicit publication approval remain gates.
- TestPyPI round-trip and public release metadata/docs-stable checks require the
  appropriate later approval; neither has been performed.

## Statistical review questions (not confirmed defects)

- `group_permutation_test` aligns refits to the observed global template. Check
  whether per-factor uncorrected inference requires a label-independent reference
  for the intended interpretation; max-statistic correction is invariant to a
  permutation of factor order. Also inspect provenance when users supply custom
  stage-0 starts rather than letting the estimator compute them.
- Confirm the conditional nominal parameter-count convention is the desired
  public contract for `htfa.stats`; it intentionally omits hierarchy hyperparameters
  and does not claim marginal evidence or effective degrees of freedom.
