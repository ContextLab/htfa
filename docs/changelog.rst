Changelog
=========

0.1.0 (unreleased)
------------------

First version: a from-scratch implementation of TFA and HTFA after the algorithm
in BrainIAK. It does not import BrainIAK and has no C++ or MPI code.

Added:

- Component matching preserves the fitted mixture exactly. Reference-relative
  logit boxes were not invariant under relabeling: for three components,
  merely swapping their order could change a 99.9% weight to about 50%.
  Logits are now unbounded with stable softmax and finite warm-start conversion;
  regression tests compare the represented fields before and after matching.

- Component-aware network plots: separate spatial lobes for mixture components,
  signed kernel profiles, identical factor edges repeated between component
  pairs, and ``node_mode='summary'`` for one sphere/centroid per factor.
  Static BrainNet plots, animations and nilearn glass-brain plots share the
  component geometry; group templates retain their component parameters.

- :class:`htfa.TFA` and :class:`htfa.HTFA` estimators, and :func:`htfa.fit`.
- A bounded Levenberg-Marquardt solver for the centers and widths
  (``nlss_method='bounded-lm'``), built on the factor structure of
  the Jacobian: 6 to 220 times faster per iteration (NumPy) than scipy's
  ``least_squares`` on the benchmarks, at the same optimum. ``'trf'`` and
  ``'dogbox'`` (``least_squares`` with an analytic, matrix-free Jacobian) remain
  available. NumPy and torch (CUDA, MPS, CPU) engines and a ``backend='auto'``
  choice.
- Variable projection (``nlss_method='varpro'``): the same solver on the
  linear-loss objective with the ridge weights eliminated, whose stationary
  points are the fixed points of the weights/centers alternation. On real fMRI
  it converges in a few outer iterations where the alternation ran hundreds.
- Defaults changed (user decision, 2026-10-26): ``nlss_loss='linear'`` (was
  ``'soft_l1'``, BrainIAK's) and ``nlss_method='auto'``, which is ``'varpro'``
  for the linear loss and ``'bounded-lm'`` for the robust losses. On the
  held-out reconstruction of :doc:`validation` (12 adults, K = 50, all voxels)
  the linear varpro fit scored a mean R² of 0.0424 against 0.0423 for
  ``'soft_l1'``, in 38 s instead of 392 s per subject. ``'soft_l1'`` and the
  two-stage ``nlss_loss='linear+soft_l1'`` (the linear fit, then ``soft_l1``
  from it) remain for heavy-tailed noise, where they recover synthetic centers
  about 10% more closely.
- Cheaper varpro evaluations that compute the same quantities: per-axis
  factors on grid coordinates, the Hessian assembly reusing the cost
  evaluation at the same point, and the projected objective from
  ``||X||**2 - <F^T X, W>``. A cost evaluation is about 3 times and an
  accepted solver step 1.4 times faster on development_fmri (K = 50 and 100);
  whole fits 1.1 to 1.5 times (see :doc:`performance`).
  Checked against the earlier formulas over 492 generated scenarios and 35
  well-conditioned synthetic fits (see :doc:`validation`).
- An ``'ols'`` fit whose width shrank (with a small ``lower_ratio``) until a
  factor was 0 at every voxel raised ``LinAlgError`` in its final weights;
  ``ridge_weights`` now falls back to the minimum-norm weights the varpro
  objective uses (``gram_solve``). Gram eigenvalues below ``GRAM_FLOOR`` are
  dropped, so factors below about 1e-146 everywhere count as 0 instead of
  giving overflowed (NaN) costs, and a solve that starts where every factor
  vanishes raises a ``ValueError`` that says what to change.
- HTFA: a global stopping rule that also stops on a plateau of the template
  change or on a stalled summed local objective (``converged_reason_``), and
  ``template_update='fixed'`` and ``'empirical'`` next to BrainIAK's update
  (the default) (see :doc:`validation`).
- Of several candidate starts that reach the same optimum (costs within a
  relative 1e-9), the first in ``init`` order wins, so that the choice does not
  depend on the thread count.
- Every K-means runs on one OpenMP and one BLAS thread, with the limits set
  in the thread that runs it (the OpenMP thread count and Accelerate's
  threading mode are per thread), so that identical fits get identical starts:
  multithreaded K-means sums in a varying order.
- HTFA's results do not depend on ``n_jobs``: every subject fit runs BLAS on
  one thread, serial or parallel (before, 8 subjects at K=100 with n_jobs=8
  and n_jobs=4 gave different templates), and the subject fits run as threads
  of the calling process instead of worker
  processes, so that all of them use the same BLAS state (on Windows with some
  AMD EPYC CPUs, OpenBLAS rounds differently in about 0.5 to 2.5% of freshly
  started processes).
- ``mode='fast'`` (the default of `htfa.TFA`, `htfa.HTFA` and
  :func:`htfa.fit`) and ``mode='exact'`` (the earlier behavior). Where the
  Hessian is not positive definite on the free variables, the fast solver
  keeps it and raises the damping instead of taking a Gauss-Newton step, and
  it forms the largest curvature products of the linear loss in float32;
  costs and gradients are unchanged. On six development_fmri adults a
  full-run TFA fit took 8.5 s instead of 24.1 s at K = 50 and 21.0 s instead
  of 108.3 s at K = 100. The fast fits reach other local minima of the same
  objective: on twelve adults their objectives were within -4.7e-4 to
  +3.4e-4 (relative) of the exact fits, with mean held-out R² 0.0429 and
  0.0507 against 0.0427 and 0.0506. The fast HTFA template (12 adults,
  K = 50) is a different map: 15.8 mm from the exact one (median center
  distance), against 2.0 to 5.6 mm between exact fits of data changed by
  1e-12; ``mode='exact'`` gives the reference template. On well-conditioned
  synthetic problems the two modes agree to 2e-7. Exact mode is the
  reference for the equivalence tests and the BrainIAK cross-checks (see
  :doc:`performance` and :doc:`validation`).
- In ``mode='fast'``, the k-th failed Cholesky factorization within a step
  multiplies the damping by 2**k and leaves the step-rejection factor ``nu``
  alone. ``mode='exact'`` keeps the earlier rule and gives the same results,
  bit for bit, as before ``mode`` existed.
- A stopping rule for subsampled fits: the fit stops when the lowest objective
  so far (on a fixed evaluation subsample) has fallen by less than a relative
  1e-4 per iteration over the last three iterations, after at least five
  (``converged_reason_ == 'objective'``), and returns the lowest-objective
  iterate, since the scaled change has a noise floor under subsampling (see
  :doc:`performance`).
- Inputs: NIfTI files and folders, nibabel and nilearn images, numpy arrays and 2D
  images.
- Factor-level static, dynamic and inter-subject connectivity, and the
  :class:`htfa.HTFAResults` container with NIfTI export and plots.
- ``plot_reconstruction`` draws all panels on one symmetric color range (the
  99th percentile of the absolute values), with a diverging colormap, signs
  kept and one colorbar; before, each panel was scaled and thresholded on its
  own, so a good reconstruction could look unlike the data. 3D data are drawn
  as orthogonal slices, one panel per row, over the MNI152 template when the
  image spans the MNI152 brain and without a background otherwise; a glass
  brain only on request (``view='glass'``). ``truth=`` adds a reference panel.
- Synthetic data with ground truth for 2D and 3D coordinates.
- Installation of optional dependencies on first use.
- Groups of subjects: ``HTFA.fit(..., groups=..., subject_ids=...)`` and
  ``htfa.fit(..., groups=...)`` fit a template per group (and per nested
  subgroup) under the global template, each with a prior centred on its
  parent's template and a posterior covariance; group factors are in global
  factor order (:mod:`htfa.groups`). It needs ``template_update='fixed'`` or
  ``'empirical'``; crossed designs raise ``ValueError``. Without ``groups``,
  fits are identical bit for bit to those before (150 arrays of 15 fits
  compared). :func:`htfa.group_permutation_test` refits with shuffled labels
  and gives per-factor p-values for center, width and connectivity
  differences, uncorrected or family-wise; on synthetic data with no
  difference, 3.4%, 5.6% and 5.4% of 1000 factor tests had p <= 0.05
  (:doc:`validation`). :class:`htfa.HTFAResults` gains ``group_centers``,
  ``group_widths``, ``group_subjects`` and ``group=`` on its node and network
  plots; :func:`htfa.make_group_data` simulates groups; tutorial 13.

- Kernel shapes and mixtures: ``kernel`` (``'rbf'``, the default;
  ``'laplace'``; ``'mexican_hat'``, the D-dimensional negative Laplacian of a
  Gaussian) and ``n_components`` (default 1) on :class:`htfa.TFA`,
  :class:`htfa.HTFA` and :func:`htfa.fit`. With ``n_components > 1`` each
  factor is a mixture of components of that shape, each with its own center
  and width, and softmax mixing weights fitted in the nonlinear solve. Exact
  derivatives in every solver (``'varpro'``, ``'bounded-lm'``, ``'trf'``,
  ``'dogbox'``) and both modes; HTFA priors and updates for every component
  and the mixing weights; shapes and mixtures in :func:`htfa.make_tfa_data`,
  :func:`htfa.make_htfa_data` and the effective nodes of the plots;
  :func:`htfa.tfa.component_factors`. ``'rbf'`` with one component gives the
  same results as before, bit for bit. Checked against a dense sympy oracle
  over 240 generated scenarios, by finite differences and by recovery from
  synthetic data (see :doc:`validation`); tutorial 12.

BrainIAK behavior that differs from BrainIAK's code (each has a regression test):

1. ``tfa_extension.cpp::factor`` shares the temporaries ``tmp1`` to ``tmp3`` across
   OpenMP threads, a data race. There is no C++ here.
2. ``recon`` accumulates in a ``float``. Costs, gradients and weights are float64
   here, unless float32 is chosen on the GPU (``mode='fast'`` forms only
   curvature products in float32).
3. The data residual is multiplied by sigma instead of divided.
4. ``HTFA._assign_posterior`` computes the Hungarian permutation but writes the
   centers and widths back unpermuted, so they fall out of step with their
   (co)variances. All fields are permuted here.
5. The template subject is chosen with an unseeded ``np.random``. It uses
   ``random_state`` here.
6. ``np.random.choice(n, max_num_voxel, replace=False)`` raises an error when there
   are fewer voxels than the cap. Sample sizes are clipped to what is available.
7. The factor kernel is hardcoded to 3D. It works for any number of coordinate
   dimensions here.
8. ``utils.from_tri_2_sym`` fills only the upper triangle of the (co)variance
   matrix and never mirrors it, so the HTFA center prior uses ``triu(Sigma)^-1``
   instead of ``Sigma^-1``. Full symmetric covariances are used here.
