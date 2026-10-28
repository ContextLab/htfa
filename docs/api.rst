API reference
=============

Fitting
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   htfa.fit
   htfa.TFA
   htfa.HTFA
   htfa.HTFAResults

:func:`htfa.fit` loads its inputs and fits :class:`htfa.HTFA` when there are
several subjects, or :class:`htfa.TFA` when there is one. Its ``data``,
``coords``, ``mask``, ``stack``, ``channels`` and ``spacing`` arguments go to
:func:`htfa.load`; the other keyword arguments go to the estimator, and an
unknown keyword raises ``TypeError``. HTFA-only keywords, such as
``max_global_iter``, fail when there is only one subject. See :doc:`inputs` for
the accepted inputs.

Use the estimators directly when you already have arrays:

.. code-block:: python

   from htfa import HTFA, TFA

   # Xs: list of (V_s, T_s) arrays; Rs: list of (V_s, D) coordinate arrays
   model = HTFA(K=20, max_global_iter=30, random_state=0).fit(Xs, Rs)
   model.global_centers_, model.global_widths_  # template
   model.centers_, model.widths_, model.weights_  # per-subject lists

   model = TFA(K=10, random_state=0).fit(X, R)  # X: (V, T), R: (V, D)
   model.converged_reason_  # 'threshold', 'objective', 'plateau' or 'max_iter'

Factors are in template order for HTFA fits: factor ``k`` of every subject
corresponds to global factor ``k``. Widths use the BrainIAK convention
``w = 2 * sigma**2``.

Kernel shapes and mixtures
~~~~~~~~~~~~~~~~~~~~~~~~~~

``kernel`` sets the profile of every factor as a function of ``z = ||r - c||**2
/ w``: ``'rbf'`` (default) ``exp(-z)``; ``'laplace'`` ``exp(eps - sqrt(z +
eps**2))`` with ``eps = 0.1`` (exponential tails ``exp(-r / sqrt(w))``, rounded
at the center so that it can be differentiated there); ``'mexican_hat'`` ``(1 - 2
z / D) exp(-z)``, the negative Laplacian of a Gaussian in D dimensions (a
positive center, a negative surround, integral 0). ``n_components = M > 1`` makes
each factor a mixture of M components of that shape, each with its own center
and width, with mixing weights (a softmax of M - 1 fitted logits) that are fitted
with them; the factor's time series are fitted for the whole mixture.

.. code-block:: python

   model = TFA(K=10, kernel="laplace", n_components=2).fit(X, R)
   model.component_centers_, model.component_widths_  # (K, 2, D), (K, 2)
   model.mixing_weights_  # (K, 2), rows sum to 1
   model.centers_, model.widths_  # mixing-weighted means of the components

HTFA's template then carries every component and the mixing weights, each with
a prior and update analogous to the widths' (``global_component_centers_``,
``global_component_widths_``, ``global_mixing_weights_``). ``kernel='rbf'`` with
one component is the original model, bit for bit. :func:`htfa.make_tfa_data` and
:func:`htfa.make_htfa_data` generate data from any shape and number of
components. See :doc:`tutorials/12_kernel_shapes` and :doc:`validation`.

Common model parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Meaning
   * - ``K``
     - 50
     - Number of factors
   * - ``standardize``
     - ``'auto'``
     - ``'auto'`` z-scores each voxel over time when ``T > 1``; ``False`` fits raw values
   * - ``random_state``
     - ``None``
     - Seed for the template subject, the initialization and the subsampling
   * - ``max_global_iter`` (HTFA)
     - 10
     - Template updates. The fit may warn ``ConvergenceWarning`` at 10; raise it (for example to 30) to let the template settle
   * - ``max_local_iter`` (HTFA)
     - 10
     - Iterations of each subject's TFA per template update
   * - ``max_iter`` (TFA)
     - 100
     - Iterations of a single-subject fit
   * - ``threshold``
     - 1e-3
     - Convergence threshold on the scaled change in centers and widths
   * - ``voxel_ratio``, ``tr_ratio``
     - 1.0
     - Fraction of voxels and time points drawn (anew each iteration) for the center and width solve; the weights always use all data
   * - ``max_voxels``, ``max_trs``
     - ``None``
     - Absolute caps on those subsample sizes
   * - ``init``
     - ``'auto'``
     - Starting points tried for the template subject: ``'ncorr'``, ``'power'``, ``'kmeans'``, or a list; the start with the lowest final objective is kept
   * - ``n_screen_iter``
     - 0
     - If greater than 0, run each start for this many iterations and continue only the best one
   * - ``kernel``
     - ``'rbf'``
     - Factor shape: ``'rbf'``, ``'laplace'`` or ``'mexican_hat'`` (see above)
   * - ``n_components``
     - 1
     - Components per factor; more than 1 fits each factor as a mixture
   * - ``backend``
     - ``'auto'``
     - ``'numpy'``, ``'cpu'``, ``'torch'``, ``'cuda'``, ``'mps'`` or ``'auto'``; see :doc:`performance`
   * - ``mode``
     - ``'fast'``
     - Step model of the center/width solver: ``'fast'`` keeps the Hessian where it is indefinite, ``'exact'`` (the reference, used by the equivalence tests and BrainIAK cross-checks) takes Gauss-Newton steps there; see :doc:`performance`
   * - ``n_jobs`` (HTFA)
     - ``None``
     - joblib workers for the per-subject fits (NumPy backend)
   * - ``verbose``
     - ``False``
     - Print per-iteration progress

.. rubric:: Examples

.. minigallery:: htfa.fit htfa.TFA htfa.HTFA htfa.HTFAResults

Loading
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   htfa.load
   htfa.Subject
   htfa.io.load
   htfa.io.Subject

.. rubric:: Examples

.. minigallery:: htfa.load htfa.Subject

Connectivity
------------

These functions work on any ``(K, T)`` weight matrices.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   htfa.static_connectivity
   htfa.dynamic_connectivity
   htfa.isfc

.. rubric:: Examples

.. minigallery:: htfa.static_connectivity htfa.dynamic_connectivity htfa.isfc

Network plots and time courses
------------------------------

BrainNet Viewer-style renderings of factor networks (nodes at the factor centers, edges
where ``|connectivity|`` exceeds a threshold) and slide-style time-course plots. They
need the ``brain3d`` extra (pyvista, imageio, imageio-ffmpeg, pillow). The brain
meshes come from BrainNet Viewer 1.43, which is not bundled: the first call downloads
it from NITRC, checks its SHA-256 checksums and caches the meshes.

A fitted center can lie outside the brain (its RBF still reaches in). The built-in
node plots (:func:`htfa.plot_network`, :func:`htfa.animate_network`, and the
:class:`htfa.HTFAResults` methods ``plot_network``, ``animate_network``,
``plot_nodes`` and ``plot_connectivity``) therefore draw each node as the part of its
RBF inside the brain: the RBF-weighted centroid of the in-brain voxels, sized by
``lambda_eff = 2 * sigma_eff**2`` from the weighted spread
(:func:`htfa.brainnet.effective_nodes`). Interior nodes are unchanged; nodes near the
edge move inward and shrink. The fit itself is not changed. ``snap=False`` draws the
fitted centers and widths. The mask (``mask=``) defaults to the subject's voxels (all
subjects' voxels for the template) for an :class:`htfa.HTFAResults`, and to the
MNI152 brain mask (:func:`htfa.brainnet.mni_mask`) for bare MNI centers or a fitted
estimator. ``HTFAResults.display_nodes(subject)`` returns the drawn centers, widths and
per-node details (distance moved, snapped and ambiguous flags, in-brain mass
fraction, ``sigma_eff``, ``lambda_eff / lambda``), also for fits made before this
existed. Fits with another ``kernel`` or several components are drawn from their
own profiles (the positive part for ``'mexican_hat'``), with ``lambda_eff`` the
width of the RBF with the same spread:

.. code-block:: python

   centers, widths, info = res.display_nodes("global")
   (info["distance"] > 5).sum()  # nodes drawn more than 5 mm from their fitted center

.. autosummary::
   :toctree: generated/
   :nosignatures:

   htfa.plot_network
   htfa.animate_network
   htfa.plot_timecourse
   htfa.brainnet.read_node
   htfa.brainnet.write_node
   htfa.brainnet.read_edge
   htfa.brainnet.write_edge
   htfa.brainnet.read_nv
   htfa.brainnet.write_nv
   htfa.brainnet.read_brainnet_config
   htfa.brainnet.config_style
   htfa.brainnet.fetch_mesh
   htfa.brainnet.load_mesh
   htfa.brainnet.node_radius
   htfa.brainnet.effective_nodes
   htfa.brainnet.mni_mask
   htfa.brainnet.write_animation
   htfa.brainnet.image_diff
   htfa.brainnet.annotate_pair
   htfa.brainnet.montage
   htfa.timecourse.global_connectivity
   htfa.timecourse.edge_count
   htfa.timecourse.moving_average

Example data
------------

:func:`htfa.datasets.fetch_development_fmri` downloads nilearn's
``development_fmri`` dataset (children and adults watching a short film) on first
use and returns its runs masked to in-brain voxels and cleaned of a linear trend
and confounds, ready for :func:`htfa.fit`:

.. code-block:: python

   import htfa

   ds = htfa.datasets.fetch_development_fmri(n_subjects=6)  # adults by default
   res = htfa.fit(ds.data, coords=ds.coords, K=30)

.. autosummary::
   :toctree: generated/
   :nosignatures:

   htfa.datasets.fetch_development_fmri
   htfa.datasets.DevelopmentFMRI
   htfa.datasets.development_fmri_files
   htfa.datasets.load_development_fmri_subject
   htfa.datasets.regress_confounds

:func:`htfa.datasets.fetch_tutorial_fit` downloads the saved HTFA fit of a
tutorial (09, 10 or 11) from the ``tutorial-data-v1`` pre-release of the htfa
repository, checks its SHA-256 and returns its arrays; the tutorials use it unless
``REFIT = True``.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   htfa.datasets.fetch_tutorial_fit

Synthetic data
--------------

The synthetic generators return data already on a z-scored scale, so fit them with
``standardize=False``.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   htfa.make_tfa_data
   htfa.make_htfa_data
   htfa.synthetic.match_factors

.. rubric:: Examples

.. minigallery:: htfa.make_tfa_data htfa.make_htfa_data htfa.synthetic.match_factors

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   htfa.set_autoinstall
   htfa.require
   htfa.tfa.standardize_data
   htfa.tfa.component_factors

.. rubric:: Examples

.. minigallery:: htfa.set_autoinstall htfa.tfa.standardize_data
