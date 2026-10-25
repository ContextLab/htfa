Changelog
=========

0.1.0 (unreleased)
------------------

First version: a from-scratch implementation of TFA and HTFA after the algorithm
in BrainIAK. It does not import BrainIAK and has no C++ or MPI code.

Added:

- :class:`htfa.TFA` and :class:`htfa.HTFA` estimators, and :func:`htfa.fit`.
- A bounded Levenberg-Marquardt solver for the centers and widths
  (``nlss_method='bounded-lm'``, the default), built on the factor structure of
  the Jacobian: 6 to 220 times faster per iteration (NumPy) than scipy's
  ``least_squares`` on the benchmarks, at the same optimum. ``'trf'`` and
  ``'dogbox'`` (``least_squares`` with an analytic, matrix-free Jacobian) remain
  available. NumPy and torch (CUDA, MPS, CPU) engines and a ``backend='auto'``
  choice.
- A stopping rule for subsampled fits: the fit stops when the full-data
  objective falls by less than a relative 1e-4 in one iteration
  (``converged_reason_ == 'objective'``), since the scaled change has a noise
  floor under subsampling (see :doc:`performance`).
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
