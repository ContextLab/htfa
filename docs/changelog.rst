Changelog
=========

0.1.0 (unreleased)
------------------

First version: a from-scratch implementation of TFA and HTFA after the algorithm
in BrainIAK. It does not import BrainIAK and has no C++ or MPI code.

Added:

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
- HTFA: a global stopping rule that also stops on a plateau of the template
  change or on a stalled summed local objective (``converged_reason_``), and
  ``template_update='fixed'`` and ``'empirical'`` next to BrainIAK's update
  (the default) (see :doc:`validation`).
- Of several candidate starts that reach the same optimum (costs within a
  relative 1e-9), the first in ``init`` order wins, so that the choice does not
  depend on the thread count.
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
- Synthetic data with ground truth for 2D and 3D coordinates.
- Installation of optional dependencies on first use.

BrainIAK behavior that differs from BrainIAK's code (each has a regression test):

1. ``tfa_extension.cpp::factor`` shares the temporaries ``tmp1`` to ``tmp3`` across
   OpenMP threads, a data race. There is no C++ here.
2. ``recon`` accumulates in a ``float``. Everything is float64, unless float32 is
   chosen on the GPU.
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
