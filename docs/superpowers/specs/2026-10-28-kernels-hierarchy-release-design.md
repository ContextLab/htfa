# Kernel shapes, multi-level hierarchy, and the PyPI release: design

Date: 2026-10-28. Base: main at 02b84bc (fast/exact modes, threaded subject fits).

## User decisions (verbatim, session 0d44797c)

- 2026-10-24: "Start versioning at v0.1.0"
- 2026-10-27: "other features i'd like to add (along with tutorials to showcase them):
  other (non-RBF) kernel shapes: laplace, mixture of RBFs, mexican hat; multi-level hierarchy:
  enable fitting nested structure (e.g., subjects --> conditions --> global model). let's also
  make a plan to get this on pypi (once all issues/bugs are addressed and all functions have been
  thoroughly tested and red-teamed)"
- 2026-10-28 (on the kernel proposal): "those all seem good-- for the mixture of RBFs version, the
  number of RBFs per mixture should be an additional parameter"; "re: tutorial, we can generate
  synthetic data and see how each fits, and then compare variance explained with a given number of
  factors, for each kernel shape"
- 2026-10-28 (hierarchy): "each condition is a different group of people"; "yes, this seems like
  the right design"; example: children vs adults with a higher-order template for people in
  general; also disease vs healthy, binned expression of a condition; "yes, leave pre/post for
  later".

## Part 1: kernel shapes

Intent: let a factor's spatial profile be something other than a Gaussian, so that sharper cores
with heavy tails, multi-scale blobs, and centre-surround structure can be represented, and let
users compare shapes on equal terms.

Later user decision (2026-10-28, relayed through the Task 16 agent's thread): "for the mixture
approach: we could have *every* kernel shape support a 'mixture' mode, where n_components=1 by
default (no mixture), but setting n_components > 1 enables each factor to comprise multiple
components, each with their own parameters. note that for each component in a mixture we also need
to fit the relative weights of the mixture components-- and then the full mixture gets fit over
all via the weights inference. so we'll need machinery for that." This supersedes the separate
'mixture' kernel below.

Behaviour
- Options on TFA, HTFA and htfa.fit: `kernel` = 'rbf' (default; current behaviour bit for bit
  in both modes), 'laplace', 'mexican_hat'; and `n_components` (default 1 = no mixture) for every
  kernel. With n_components > 1 each factor is a mixture of n_components components of that shape,
  each with its own centre and shape parameters, plus per-factor mixing weights over the
  components fitted in the nonlinear solve (constrained so they are identifiable against the
  factor's time-series weights W, e.g. softmax/simplex). W is then fitted for the whole mixture
  factor as now (ridge / varpro). The Jacobian/Hessian machinery covers component parameters and
  mixing weights.
- Every factor keeps a centre; the shape adds parameters per factor:
  - rbf: width w (unchanged). F = exp(-d^2 / w).
  - laplace: scale; F = exp(-sqrt(d^2 + eps^2) / lambda) (controller's call: smoothed at d = 0 by a
    small eps relative to the voxel size so derivatives exist; document eps and its effect).
  - mexican_hat: difference of Gaussians or Ricker with a positive centre and a negative surround;
    parameters chosen so the shape is identifiable (builder's call, justified).
- All of it works in fast and exact modes and in the varpro solver; derivatives (gradient,
  Gauss-Newton, Hessian) exact and tested by finite differences.
- HTFA: the template carries every shape parameter, with priors analogous to the width prior;
  the MAP update extends to the extra parameters; the 'brainiak' path for 'rbf' is unchanged.
- Display: effective nodes (in-brain F-weighted centroid and weighted SD) work for every shape;
  for signed shapes (mexican_hat) use |F| or the positive lobe for size, and say which.
- Synthetic generator: make_tfa_data / make_htfa_data accept `kernel` (and its parameters) so data
  can be generated from each shape.

Tutorial (user decision): generate synthetic data from each shape, fit every kernel to every
dataset, compare variance explained at a fixed number of factors (generator x fitted-kernel grid),
plus a sweep over K. Fits fast enough for Colab.

Acceptance
- rbf results bitwise unchanged (both modes); all existing tests pass unchanged.
- Recovery tests per shape (centres/shape parameters vs truth), equivalence-style kernel tests over
  generated scenarios, finite-difference derivative tests, n_jobs independence, BrainIAK checks
  still pass.

## Part 2: multi-level hierarchy (nested groups)

Intent: fit group-level templates (e.g. children, adults) under a higher-order template (people in
general), so groups can be compared on the same factors.

Behaviour
- HTFA gains `groups` (one label per subject; optionally a list of label sequences for deeper
  nesting, e.g. site -> group). Each subject's prior comes from its group template; each group
  template's prior comes from its parent; the top is the global template. Same shrinkage machinery
  as the current subject -> template step, applied per level.
- No labels: results identical to current HTFA bit for bit.
- Outputs: global template, per-group templates with uncertainty, per-subject fits; group factors
  matched to the global template so factor k is the same factor everywhere.
- Group comparison: a permutation test (shuffle group labels, refit) giving per-factor p-values for
  differences in centre, width (and connectivity if feasible). Default modest permutations; reuse
  saved fits in tutorials.
- Nested designs only. Crossed designs (pre vs post on the same people) are out of scope for now
  (user decision).

Tutorial: synthetic groups with known shifts (and a no-difference control for false positives),
then development_fmri children vs adults with a "people in general" template. Saved fits published
like the other tutorials (htfa.datasets.fetch_tutorial_fit).

Acceptance
- Recovery on synthetic data with true group differences and with none (false-positive rate near
  nominal); bitwise equality with current HTFA when no groups; n_jobs independence; both modes.

## Part 3: PyPI release v0.1.0

Gate (user: after all issues are addressed and every function is tested and red-teamed)
1. All open work merged: kernels, hierarchy, regenerated tutorial fits.
2. Red team: fresh reviewers attack every public function (odd/edge inputs, wrong types, extreme
   sizes, install paths, docs examples); every finding gets a test, then a fix.
3. Clean room: build sdist and wheel; install the wheel into empty environments on Linux, macOS and
   Windows for each supported Python; run the test suite and every tutorial against the installed
   package.
Mechanics
- GitHub Actions release workflow with PyPI trusted publishing; TestPyPI first, then install and
  test from TestPyPI, then tag v0.1.0 and publish to PyPI; changelog, README badges, Read the Docs
  'stable' follow the tag.
- The user approves each outward step (TestPyPI, PyPI publish, GitHub release notes).

## Left to the builder
Parameterisations (identifiability) of laplace/mixture/mexican_hat, priors for the extra shape
parameters, the group-permutation statistic, API names consistent with the existing code.
