Validation
==========

Validation combines known-truth synthetic data, independent mathematical
references, BrainIAK comparisons, and real multi-subject fMRI examples. Each
addresses a different question: numerical agreement is not evidence of
scientific validity, and good reconstruction does not establish unique or
biologically interpretable factors.

Synthetic recovery
------------------

:func:`htfa.make_tfa_data` and :func:`htfa.make_htfa_data` provide spatial
parameters and time courses with known ground truth. The recovery tests fit
these data with ``standardize=False`` and compare centers, widths, weights,
and reconstructions after matching factor labels. Tests cover two- and
three-dimensional coordinates, single-subject and hierarchical fits, noise,
priors, subsampling, and multiple initializations.

A nonlinear fit can reach a local minimum. Recovery checks therefore distinguish
agreement with the generating parameters from agreement with a fit initialized
at those parameters. Overlapping mixture components may not be separately
identifiable even when their summed spatial field reconstructs accurately.
The kernel tutorial shows fitted fields and held-out reconstruction rather
than relying only on parameter distances.

Independent numerical checks
----------------------------

The kernel tests compare analytic derivatives with central finite differences.
The symbolic reference in ``tests/reference_shapes.py`` defines each profile
independently, differentiates it with SymPy, and assembles dense objectives,
Jacobians, and Hessians. Comparisons cover all three kernels, one to three
components, two and three spatial dimensions, robust and linear losses,
priors, and both ridge and ordinary least-squares temporal weights.

Generated scenarios include irregular coordinates, near-duplicate factors,
large and small widths, unequal mixture weights, and poorly conditioned
systems. Tolerances account for conditioning; cases with vanishing factors or
truncated eigenvalues require separate checks because ordinary relative-error
comparisons are undefined or discontinuous there. Recovery tests and
finite-difference checks complement these comparisons.

The test suite also checks cache invalidation when spatial parameters or priors
change, component-relabeling invariance, dense versus matrix-free calculations,
and equivalence of serial and parallel subject fitting. The implementation
references in ``tests/reference_kernels_3298278.py`` and
``tests/reference_shapes.py`` are test oracles, not runtime dependencies.

BrainIAK reference
------------------

``tests/test_brainiak_crosscheck.py`` runs BrainIAK 0.12 and HTFA on matching
synthetic inputs. It compares RBF factors, residuals, single-subject recovery,
and hierarchical recovery. The cross-check uses matched model settings and
checks both solver modes where applicable. BrainIAK is a validation dependency;
the package does not import it during ordinary use.

Reproducibility
---------------

``random_state`` controls stochastic initialization and subsampling. With the
same explicit integer seed, repeated fits within a process with identical
inputs, subject order, and settings are deterministic,
and subject-level parallelism preserves the fitted results. BLAS implementations
and hardware can differ in reduction order across processes or machines.
Small numerical differences can lead to different local minima, so bitwise
identity across platforms is not a scientific acceptance criterion.

Performance measurements use a separate quiet-host protocol described in
:doc:`performance`. Tutorial runtimes and cached asset generation times are
not used as comparative benchmarks.

Real fMRI evaluation
--------------------

Several tutorials use Nilearn's ``development_fmri`` dataset as an illustrative
real-data workload: children and adults watch Pixar's *Partly Cloudy* during
fMRI acquisition, producing 168 brain volumes sampled two seconds apart
(the repetition time, or TR). The common film provides aligned stimulus time
for inter-subject connectivity examples; neither movie viewing nor matched
time-series lengths are requirements of HTFA fitting itself. The loader resamples
the MNI152 brain mask, excludes low-intensity and nearly constant voxels,
and regresses the supplied confounds and a linear trend as described by
:mod:`htfa.datasets`. The estimators' ``standardize='auto'`` setting z-scores
each voxel; comparisons must use the same scaling in every condition.

:doc:`tutorials/15_real_fmri_kernel_comparison` compares the three kernel shapes
with one and two components per factor on the same three adult participants,
masking, preprocessing, factor count, and fitting budget. It reports R²,
nominal parameter counts, information criteria, and reconstructed frames.
These are in-sample comparisons on one dataset; they do not establish a
universally preferred kernel or predict performance on unseen subjects.

:doc:`tutorials/14_fit_statistics` explains the scoring conventions. AIC, AICc,
and BIC use a conditional Gaussian residual likelihood and a nominal parameter
count. Correlated voxel-time measurements, regularization, and singular mixture
models limit their interpretation. The scores are not marginal hierarchical
evidence or calibrated Bayes factors. A larger mixture can also reach a worse
local optimum, so optimization outcomes need not improve monotonically with
component count.

Held-out reconstruction
~~~~~~~~~~~~~~~~~~~~~~~

``scripts/validation_heldout.py`` provides a complementary evaluation protocol
for the movie-viewing dataset. Its illustrative split separates a training
segment (samples 0–79) from a test segment (88–167), with a gap between them. These indices describe this 168-volume example, not
a required split for other datasets. Confound regression is performed separately
on each segment; training means and standard deviations scale both segments.

The protocol compares fitted TFA factors with their frozen initializations,
random spatial centers, and matched-count PCA maps. On the test segment,
weights are estimated from one set of voxels in a spatial-block checkerboard,
and R² is scored on the other set; the sets then swap. This separates spatial
factor fitting from evaluation and avoids fitting temporal weights to the
same voxel observations being scored. Ridge and ordinary least-squares
weight estimates are separate comparisons because their scaling effects differ.

An unconstrained PCA map and a radial spatial factor have different parameter
counts. Equal factor counts alone do not make them equally flexible models.

Template updates and group inference
------------------------------------

``template_update='brainiak'`` updates the template posterior and uses it as
the prior for the next global iteration. With fixed subject covariance
``Sigma`` and initial template covariance ``S0``, the covariance after ``i``
iterations over ``n`` subjects is
``(S0^-1 + i n Sigma^-1)^-1``. Repeated passes over the same subjects therefore
shrink the covariance even without additional observations. The reported
covariance should not be read as uncertainty from independent new samples
at every iteration.

For an ungrouped fit, ``'fixed'`` holds the subject prior covariance fixed
and uses the mean of matched local estimates as the template. With groups,
the templates are coupled through the hierarchy's tree equations. ``'empirical'`` estimates dispersion
from matched unpenalized subject fits. The tests check these update equations,
nested group trees, matching, and independence from ``n_jobs``.

Group permutation tests align all refits to a fixed stage-0 reference, shuffle
labels, and refit. The reference subject order and any supplied starts must be
fixed independently of the tested labels and effects. Their interpretation depends
on exchangeable, independent subjects and the specified nesting restrictions;
repeated observations of one participant are not independent subjects.
Synthetic null and shifted-group checks are in ``tests/test_groups.py`` and
``scripts/validation_groups.py``. A max-statistic correction addresses multiple
factors within a chosen statistic; it does not automatically correct across
all statistics or exploratory analyses.

As an example of the grouped API, :doc:`tutorials/13_group_hierarchy` compares
ten children and ten adults from the movie-viewing dataset using 30 factors and
100 permutations. Its cached example shows no center,
width, or connectivity differences surviving correction across factors.
This does not prove equivalence between age groups. The sample is small,
permutation p-values have resolution 1/101, and motion and registration can
affect the fitted factors.

Word decoding
-------------

:doc:`tutorials/09_word_decoding_mitchell2008` illustrates how factor weights
can serve as features for a prediction task. In the Mitchell et al. (2008)
experiment, nine participants view 60 concrete nouns with line drawings, each
presented six times, while fMRI responses are recorded. The example maps
between the responses' factor weights and word embeddings (numerical vector
representations of the nouns). This is one downstream application of HTFA,
not a requirement of its spatial model. The supervised mappings hold out target
words and
participants, but the unsupervised spatial template and voxel standardization
use full runs. This is a transductive evaluation, not a fully inductive test
of preprocessing learned without the held-out observations. Cached permutation
nulls are checked against the exact fitted representation and embeddings.
Exploratory metrics require their own multiplicity considerations.

Plots, input handling, and public API
-------------------------------------

Tests exercise NIfTI affine handling, array and image inputs, reconstruction,
connectivity, persistence, dependency loading, invalid inputs, and plotting
wrappers. Component plots check spatial geometry, factor membership, signed
fields, repeated edge values, static images, and animations. Small renderer
rounding differences are tolerated only alongside exact geometry checks.

The installed-package validation runs outside the source checkout, and the
notebook runner checks that each kernel imports the installed package.
Documentation builds treat warnings as errors. Executed notebook figures and
animations also require visual inspection: successful execution alone cannot
establish that a plot communicates its intended result.
