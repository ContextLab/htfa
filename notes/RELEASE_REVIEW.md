# v0.1.0 review packet — publication not approved

Jeremy requires careful review and explicit approval before PyPI publication.
No PyPI or TestPyPI upload has occurred. The candidate workflows build and test
artifacts; the publication workflow remains an inactive example.

The integration is merged in [PR #184](https://github.com/ContextLab/htfa/pull/184).
The documentation and public-API audit is in
[PR #185](https://github.com/ContextLab/htfa/pull/185).

## Scope for review

- Standalone TFA/HTFA numerical behavior, with independent mathematical
  references and BrainIAK cross-checks.
- RBF, Laplace and Mexican-hat factors, fitted component mixtures, invariant
  component relabeling, and component geometry in BrainNet and Nilearn plots.
- Nested templates and label-independent group permutation hypotheses for
  independent subjects; crossed and repeated-measures designs are unsupported.
- `htfa.stats`: nominal conditional reconstruction R²/AIC/AICc/BIC. These are not
  marginal hierarchical evidence, effective degrees of freedom, or calibrated
  Bayes factors for correlated fMRI residuals and regularized singular mixtures.
- Standalone, present-tense documentation and tutorials, including PyPI
  installation instructions and a PyPI badge as a preview of the public release.
- Tutorial 09's current HyperTools API and transductive evaluation boundaries.
- Quiet-host benchmarks with input/source hashes, matched workloads, CPU
  interference checks, convergence records and inspectable raw measurements.

The PCA-limit exploration remains local-only. No Zenodo DOI is inserted or
invented. The Zenodo badge is the final publication step, after Jeremy's review
and the approved release is live, so its DOI resolves to that release.

## Audit findings and verification

Fresh subagents review the public API, documentation and tutorials, with a
second critical pass across the edited material. Their findings include:

- Group permutation refits require a common label-independent factor reference.
  The implementation uses unpenalized stage-0 centers, normalizes their factor
  order, and carries reference metadata in the result. An explicit
  nontransitive-matching counterexample and an end-to-end permutation-orbit
  regression check the issue and its resolution. Caller-supplied starts must
  also be label-independent; arrays alone cannot establish that provenance.
  See [the permutation audit](2026-10-29-permutation-reference-audit.md).
- Public spatial inputs receive explicit validation; invalid values fail with
  actionable errors rather than propagating to numerical or plotting failures.
- Tutorial explanations distinguish variable projection from alternation,
  enumerate stopping rules, separate test-time spatial reconstruction from
  forecasting, and distinguish component lobes from centroid summaries.
- Plotting explanations reflect fixed edge opacity and signed component
  profiles. Connectome colorbar labels retain endpoints and zero without
  colliding with nearby threshold labels; actual renders test label bounds.
- Benchmark inputs are voxel-standardized before timing. CPU counters are
  primed, source and data are hashed, and sustained unrelated jobs invalidate
  a row. Earlier busy-host timing claims are absent from public documentation.

These are fresh agent reviews and regression checks, not a substitute for
Jeremy's scientific and publication review.

## Current evidence

Numerical source: `6e534cc` (the reviewed mixture scaling and norm-overflow fixes).
The local library differs only in reviewed docstrings and the verified Tutorial
13 asset registry entry; executable AST comparisons check that boundary.

- Installed-wheel suite: **1,067 passed, 17 skipped, 28 deselected**;
  `/private/tmp/htfa-audit-review-tests.log`.
- All **13 notebooks** execute successfully against that installed candidate;
  `/private/tmp/htfa-audit-review-tutorials.log`. Executed outputs are copied
  back after checking executable cell equality, preserving reviewed prose.
- Tutorial 13's 100 permutations are recomputed with the fixed reference. All
  checkpoints, seeded labels, statistics, reference metadata, and fit checksum
  are verified. The registered null has SHA-256
  `5bf2ae0b61f729f10be2820ba84cdd5f00448f5b51910024b66d7cca821b30eb`;
  a fresh download passes checksum validation. No center, width, or connectivity
  effects survive the within-statistic family-wise correction in this example.
- BrainIAK checks pass on the numerical source in run `40811513557`.
- Network suite: **27 passed, 1 failed, 2 skipped**; the failure was an NITRC
  mesh archive read timeout. The exact failed test passes on retry (**1 passed**).
  Logs: `/private/tmp/htfa-audit-network.log` and
  `/private/tmp/htfa-audit-network-retry.log`.
- Current source-distribution installation and smoke/statistics checks:
  **15 passed**; `/private/tmp/htfa-audit-review-sdist.log`.
- Lint, formatting, mypy, wheel/sdist metadata, and strict Sphinx build pass.
  Gallery examples are forcibly regenerated; their figures are reviewed.
- Fresh contextual reviewers read all 13 notebooks, public guide/reference
  pages, gallery examples, and API docstrings. They introduce experimental
  examples, explain relevant data and assumptions, distinguish descriptive
  plots from tested statistics, and scope conclusions to the selected data.
  An independent follow-up review resolves the remaining unsupported noise
  claim and label-ordered schematic example.
- The numerical-overflow regressions cover actual vanishing-component problems,
  objective consistency, and both strict-overflow and silent-infinity norm
  behavior. All **341 ordinary-RBF fingerprint arrays remain identical**.
- Final-source isolated benchmark run `40811532627` succeeds: all 24 fits
  pass CPU-interference checks and stop on threshold without warnings. An
  independent reviewer verifies 1,070 monitoring samples, source/settings/data
  hashes, paired ordering, CSV values, and every table statistic. Public
  artifacts are in `docs/_static/benchmarks/`. Final-head CI is pending.

## Remaining gates

1. Complete final-head platform and installed-wheel/tutorial CI after the
   documentation, output, and asset updates are committed.
2. Jeremy reviews the public content, scientific interpretation, and release
   draft. No publication approval is inferred from successful tests.
3. Obtain explicit approval for TestPyPI/PyPI and GitHub release publication.
   Perform approved release checks and, last, verify the release-specific
   Zenodo DOI and add its badge.

Historical evidence and asset provenance remain in
`2026-10-28-release-review.md`, `2026-10-28-codex-integration.md`,
`2026-10-28-stats-tutorials.md`, and `tutorial-assets-2026-10-28.json`.
