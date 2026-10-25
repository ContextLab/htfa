Performance
===========

Backends
--------

The centers and widths are found by a bounded Levenberg-Marquardt solve
(``nlss_method='bounded-lm'``, the default) on the explicit ``K (D + 1)``-square
Gauss-Newton system. Assembling that system is dominated by products between the
``(V, T)`` data, the ``(V, K)`` factor images and the ``(K, T)`` weights; with the
``'linear'`` loss the data term is compressed exactly and its products do not
depend on ``T``. An engine computes these products. The ``backend`` argument of
:class:`htfa.TFA`, :class:`htfa.HTFA` and :func:`htfa.fit` selects the engine:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``backend``
     - Engine
   * - ``'numpy'``, ``'cpu'``
     - NumPy on the CPU, in float64
   * - ``'torch'``
     - torch on the first CUDA or MPS device if one exists, otherwise on the CPU
   * - ``'cuda'``, ``'mps'``
     - torch on that device; ``RuntimeError`` if the device does not exist
   * - ``'auto'`` (default)
     - See below

``nlss_method='trf'`` or ``'dogbox'`` run BrainIAK's solver
(``scipy.optimize.least_squares`` with a matrix-free Jacobian) instead; they
minimize the same objective and are much slower (see the benchmarks).

Choosing a backend with ``'auto'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``'auto'`` uses a GPU torch engine only if all of the following hold:

1. torch is already installed (``'auto'`` never installs it);
2. a CUDA or MPS device is available;
3. ``V * T * K`` is at least ``htfa._engines.GPU_MIN_WORK``.

Otherwise it uses NumPy. ``V`` and ``T`` in the rule are the subsampled sizes
that the solve works on, not the full data size (see below). The threshold is
provisional: it was chosen from the 5-iteration timings below, measured while
other jobs loaded the CPU, which favors the GPU.

An explicit ``backend='mps'`` or ``'cuda'`` always uses that device, whatever the
problem size.

Subsampling
-----------

Each iteration solves for the centers and widths on a random subset of voxels and
time points, drawn anew every iteration. The final weights always use all of the
data.

- ``voxel_ratio`` and ``tr_ratio`` (default 1.0, that is, all) set the fraction of
  voxels and of time points drawn.
- ``max_voxels`` and ``max_trs`` (default ``None``) cap the number drawn.
- The sample sizes are clipped to what is available, so a cap larger than the
  data is not an error.

Smaller subsets reduce the cost of each solve and change the result; the
subsampled fits in the test suite recover synthetic centers less exactly than the
full fits. ``random_state`` seeds the subsampling.

Stopping under subsampling
~~~~~~~~~~~~~~~~~~~~~~~~~~

With a fresh subset every iteration, the scaled change between iterations (the
quantity compared with ``threshold``) does not settle: it has a noise floor, the
difference between the optima of two subsets. Measured on real fMRI data
(MNI 2 mm, ``V = 69597``, ``T = 1137``, ``K = 100``; six independent solves per
row, from the same start):

.. list-table::
   :header-rows: 1

   * - ``max_voxels``, ``max_trs``
     - scaled change (floor)
     - center noise (mm)
     - width noise (median / 90th percentile)
   * - 3000, 300
     - 0.56
     - 2.1
     - 5.6% / 23%
   * - 10000, 300
     - 0.31
     - 1.3
     - 4.3% / 15%
   * - 30000, 300
     - 0.31
     - 0.85
     - 2.5% / 14%

The default ``threshold=1e-3`` is therefore out of reach, so a subsampled fit
evaluates the full-data objective after every iteration and stops when it falls
by less than a relative ``1e-4`` in one iteration (``converged_reason_ ==
'objective'``; no ``ConvergenceWarning``). On synthetic data with the same mask,
shape and caps 3000 x 300, the relative decrease fell from 1.5e-2 after the first
iteration to 9.2e-5 at iteration 12, where the median center error was 2.32 mm
(2.16 to 2.32 mm from iteration 9 on, 2.18 mm at iteration 20). The fit also
still stops when the scaled change has not reached a new minimum for five
iterations (``'plateau'``).

Each evaluation of the full-data objective at that shape took 2.0 to 2.2 s with
``nlss_loss='linear'`` and 13.6 to 17.8 s with ``'soft_l1'`` (NumPy, machine
loaded as below), against 0.7 to 1.1 s for one subsampled iteration with caps
3000 x 300 and the linear loss. Larger caps lower the noise floor (and the error
of the result) more than more iterations do.

Other settings that change the cost of a fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``init='auto'`` runs three starts (``'ncorr'``, ``'power'``, ``'kmeans'``) and
  keeps the one with the lowest final objective.
  ``init='power'`` runs one start. ``n_screen_iter > 0`` runs each start for that
  many iterations and continues only the best. Each start runs a K-means++ on
  the coordinates (55 to 83 s per start at ``V = 69597``, ``K = 100``).
- ``n_jobs`` runs the per-subject fits of an HTFA in parallel (NumPy backend).
  ``n_threads`` sets the BLAS/OpenMP threads of each subject fit; by default
  HTFA takes the lowest limit already in effect (``OMP_NUM_THREADS``,
  ``OPENBLAS_NUM_THREADS``, an enclosing ``threadpoolctl.threadpool_limits``),
  or all CPUs, divided among the ``n_jobs`` workers.
- ``nlss_loss='soft_l1'`` (the default) costs ``O(V T K^2)`` per Gauss-Newton
  assembly; ``'linear'`` is independent of ``T`` after an exact compression.
- ``K``, the number of factors, enters every product at least linearly.

Benchmarks
----------

**Machine.** M2 Max (12 CPUs), macOS 26.6.2, Python 3.12.11, NumPy 2.5.4,
torch 2.14.1 (MPS, float32). Other jobs kept the CPU load average between 60 and
190 throughout (on 12 cores), so the absolute times are inflated, NumPy times
more than MPS times.

**Data.** Synthetic data from :func:`htfa.make_tfa_data` with
``standardize=False``, all voxels and samples. The grids are ``(25, 20, 10)``
(small, ``V = 5000``, ``T = 200``, ``K = 20``), ``(40, 25, 20)`` (medium,
``V = 20000``, ``T = 300``, ``K = 100``) and ``(50, 40, 25)`` (large,
``V = 50000``, ``T = 1000``, ``K = 200``).

**Before and after.** "trf" is ``scipy.optimize.least_squares`` with a
matrix-free Jacobian (measured 2026-10-24 at commit ``5912bf0``, now
``nlss_method='trf'``); "bounded-lm" is the default solver (2026-10-25, commits
``915c15b`` to ``9f1c3bf``). Reproduce with ``scripts/benchmark.py`` (fits) and
``scripts/workload_projection.py`` (per-iteration timings at a given shape).

.. list-table::
   :header-rows: 1

   * - case
     - trf
     - bounded-lm
   * - small, full fit, NumPy (``init='auto'``)
     - 402 s
     - 276 s
   * - small, full fit, MPS
     - 793 s
     - 70 s
   * - medium, full fit, NumPy / MPS
     - not run (hours)
     - 1627 s / 880 s
   * - small, 5 iterations, NumPy / MPS (``init='power'``)
     - 36.5 s / 90.3 s
     - 5.7 s / 2.6 s
   * - medium, 5 iterations, NumPy / MPS
     - 1344 s / 1810 s
     - 47.9 s / 31.4 s
   * - large, 5 iterations, NumPy
     - more than 3 h for 1 iteration (abandoned)
     - not measured
   * - fMRI shape ``69597 x 1137``, ``K = 100``, linear loss, caps 3000 x 300,
       2 threads: one iteration
     - 22.9 s
     - 0.67 to 1.07 s
   * - same, caps 10000 x 500: one iteration
     - 586 s
     - 2.63 s
   * - same, all data, linear loss, 12 threads: one iteration
     - 2271 s
     - 11.2 s
   * - same, all data, ``soft_l1``, 12 threads: one iteration
     - more than 85 min (stopped)
     - 65.7 s

The full small and medium fits reached ``max_iter=100`` with both solvers (an
outer-loop property, not the solver's: the objective stays near a saddle for tens
of iterations before reaching the true solution). Both solvers reach the same
optimum: over 96 comparisons (2D and 3D, six seeds, with and without a prior,
linear and ``soft_l1``) the relative cost difference was between -1.1e-6 and
+1.5e-7 (``scripts/compare_solvers.py``).
