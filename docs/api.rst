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
   * - ``backend``
     - ``'auto'``
     - ``'numpy'``, ``'cpu'``, ``'torch'``, ``'cuda'``, ``'mps'`` or ``'auto'``; see :doc:`performance`
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
   htfa.brainnet.write_animation
   htfa.brainnet.image_diff
   htfa.brainnet.annotate_pair
   htfa.brainnet.montage
   htfa.timecourse.global_connectivity
   htfa.timecourse.edge_count
   htfa.timecourse.moving_average

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
   htfa.tfa.standardize_data

.. rubric:: Examples

.. minigallery:: htfa.set_autoinstall htfa.tfa.standardize_data
