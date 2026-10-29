Release overview
================

0.1.0
-----

htfa implements topographic factor analysis (TFA) and hierarchical topographic
factor analysis (HTFA) in Python. The package fits spatial factors and their
time courses to one subject or several subjects sharing a template. The
estimators do not require BrainIAK or MPI, and htfa has no locally compiled
extension.

Fitting
~~~~~~~

- :class:`htfa.TFA` and :class:`htfa.HTFA` work directly with arrays;
  :func:`htfa.fit` also loads NIfTI files, folders, images and other supported
  inputs. See :doc:`inputs`.
- ``kernel`` selects Gaussian (``'rbf'``), Laplace (``'laplace'``), or signed
  Mexican-hat (``'mexican_hat'``) spatial profiles. ``n_components`` sets the
  number of spatial components per factor. Components have fitted centers,
  widths and softmax mixing weights; the factor has one shared time course.
- The default linear-loss variable-projection solver eliminates the temporal
  weights from the spatial optimization. Robust losses use bounded
  Levenberg–Marquardt updates. ``'trf'`` and ``'dogbox'`` solvers use SciPy's
  ``least_squares`` with analytic, matrix-free Jacobians.
- NumPy and torch backends support CPU, CUDA and Apple Metal. ``backend='auto'``
  chooses a backend using the available devices and problem size.
- ``mode='fast'`` is the default. ``mode='exact'`` is the numerical reference
  for equivalence and BrainIAK checks. Both modes optimize the same objective;
  their different step models can reach different local minima on real data.
  See :doc:`performance` and :doc:`validation`.
- Subsampled fits use a fixed evaluation subset to track their best iterate.
  Convergence attributes distinguish parameter thresholds, objective stopping,
  plateaus and iteration limits.

Hierarchies and comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``groups`` fits one template per subject group, with optional deeper nested
  levels. Each subject's prior is centered on its group template, and each
  group's prior on its parent. Group templates require
  ``template_update='fixed'`` or ``'empirical'``.
- Factors and mixture components are matched across subjects and templates.
  Component relabeling preserves the represented spatial field and its mixing
  weights.
- :func:`htfa.group_permutation_test` refits shuffled group labels to compare
  centers, widths and connectivity. It reports per-factor p-values with an
  optional family-wise correction. Its sampling unit is an independent subject;
  crossed and repeated-measures designs are outside its scope.
- :mod:`htfa.stats` reports reconstruction R², Gaussian plug-in likelihood,
  AIC, AICc and BIC with explicit nominal conditional parameter counts. These
  descriptive criteria are not marginal hierarchical evidence or calibrated
  Bayes factors for correlated fMRI residuals and singular mixtures.

Results and visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

- :class:`htfa.HTFAResults` provides reconstruction, NIfTI export, factor-level
  static and dynamic connectivity, inter-subject connectivity, and plots.
- BrainNet Viewer-style plots, animations and Nilearn glass-brain plots show
  each mixture component as a spatial lobe, including signed profiles. Each
  factor edge is repeated with the same value between its component pairs;
  components within the same factor have no extra edges.
- ``node_mode='summary'`` draws one sphere or centroid per factor. The fitted
  geometry is separate from display snapping and masking.
- Reconstruction panels share a symmetric color scale and a colorbar.
  Three-dimensional data use orthogonal slices by default, with an MNI152
  background when the coordinates span that template; ``view='glass'`` selects
  a glass-brain projection. ``truth=`` supplies a reference panel.

Data and documentation
~~~~~~~~~~~~~~~~~~~~~~

- :func:`htfa.make_tfa_data`, :func:`htfa.make_htfa_data` and
  :func:`htfa.make_group_data` generate synthetic data with known parameters.
- :mod:`htfa.datasets` provides preprocessed development-fMRI data and
  checksum-verified tutorial fits and permutation nulls.
- Optional dependencies install on first use. :func:`htfa.set_autoinstall`
  controls this behavior; :doc:`optional_dependencies` describes manual
  installation and platform requirements.
- The :doc:`tutorials` cover fitting, data inputs, 2D images, connectivity,
  word decoding, dynamic ISFC, plotting, kernels, groups, statistics and a
  real-fMRI kernel comparison. The :doc:`validation` page describes recovery,
  numerical checks and the interpretation of fitted results.
