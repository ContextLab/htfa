Release overview
================

0.2.0
-----

0.2.0 adds options to the BrainNet Viewer-style network plots and animations in
``htfa.brainnet``. Existing calls are unchanged: a call that passes none of the
new arguments takes the 0.1.0 code path and gives the same images.

Network plots and animations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``node_color`` accepts one color per node: a sequence of K color specs, or an
  array of shape (K, 3) or (K, 4). In :func:`htfa.animate_network` it also
  accepts per-frame colors, (T, K) color specs or an array of shape (T, K, 3)
  or (T, K, 4); each frame uses the colors of the nearest window. A flat
  sequence of 3 or 4 numbers is still one color.
- :func:`htfa.animate_network` accepts ``sizes`` of shape (T, K). Radii are
  scaled jointly over all frames and follow ``interpolation`` between windows.
  ``connectivity=None`` draws no edges when ``sizes`` or ``node_color`` has a
  time axis. In :meth:`htfa.HTFAResults.animate_network`, ``connectivity=None``
  still computes ISFC.
- ``views`` draws several views side by side, ``legend`` adds rows of colored
  dots and lines with labels, and ``frame_size`` sets the exact frame size.
  One crop box serves every view and frame; with ``spin`` the views are not
  cropped. Passing any of the three selects the layout path. A call that uses
  none of the new options gives the same output as in 0.1.0.
- :func:`htfa.animate_network` accepts ``compose``, a callable
  ``compose(images, t)`` that builds each frame from the per-view images, and
  ``n_jobs``, which draws frames in worker processes. With ``n_jobs`` other
  than 1, a blank (black) render or a dead worker restarts the pool, up to 5
  times.
- :func:`htfa.plot_network` accepts ``caption``, one line of text drawn as
  :func:`htfa.animate_network` draws a frame's caption: in a band above the
  image, or, with ``views``, ``legend`` or ``frame_size``, on a line under the
  view titles. The image equals the frame of a one-window movie with
  ``captions=[caption]``.
- An empty legend row reserves a blank band of the normal row height:
  ``legend=[[], row]`` puts the row one band lower, and a legend of empty rows
  only is allowed. ``legend=[]`` still raises ``ValueError``.
- ``crf`` and ``faststart`` set the x264 constant rate factor and move the mp4
  index to the start of the file, in :func:`htfa.animate_network` and
  :func:`htfa.brainnet.write_animation`. With ``crf=None`` the encoding is as
  in 0.1.0.
- ``preset`` passes an x264 preset name (``"ultrafast"`` to ``"placebo"``) to
  the encoder, in :func:`htfa.animate_network` and
  :func:`htfa.brainnet.write_animation`. Other names raise ``ValueError``, and
  so does a GIF output, as for ``crf``. With ``preset=None`` the encoding is
  as in 0.1.0.
- The style key ``supersample`` sets the rendering multiple used for
  anti-aliasing (default 2).
- :func:`htfa.content_box` returns the bounding box of the non-background
  content of one or more images.
- On macOS, network plots and animations release their closed render windows.
  In 0.1.0 each render kept its window in memory (5 to 9 MB per render with
  the ch2 mesh), and after about 40 renders in one process macOS printed
  "Context leak detected". This uses VTK's ``vtkCocoaAutoreleasePool`` (VTK
  9.5 and later); with an older VTK, rendering runs as in 0.1.0.
- Edges and per-node colors are stored as per-point instead of per-cell
  colors. With Mesa 26's software renderer (Ubuntu 26.04), 0.1.0 drew no
  edges at all, because actors colored by cell data were not drawn there.

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
  chooses a backend using the available devices and problem size. HTFA can
  distribute subjects across multiple CUDA GPUs.
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
