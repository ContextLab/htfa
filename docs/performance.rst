Performance
===========

Backends
--------

The centers and widths are found by a bounded Levenberg-Marquardt solve
(``nlss_method='bounded-lm'``) on the explicit ``K (D + 1)``-square
Gauss-Newton system; with the default linear loss the same solver runs on the
variable-projection objective (``'varpro'``, see `Solver and outer loop`_). Assembling that system is dominated by products between the
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
evaluates the objective after every iteration and stops when its lowest value so
far has fallen by less than a relative ``1e-4`` per iteration over the last three
iterations, after at least five iterations (``converged_reason_ ==
'objective'``; no ``ConvergenceWarning``). It then returns the iterate with the
lowest objective, not the last one; ``objective_history_`` records the values.
The running minimum and the window keep a single noisy step up (the objective is
not minimized by any one subsample) from stopping a fit that is still improving.
The fit also still stops when the scaled change has not reached a new minimum for
five iterations (``'plateau'``).

The stopping objective is computed on one fixed random subsample per fit,
``max(2 Vs, 6000)`` voxels (``Vs`` the per-iteration voxel count; all voxels if
fewer) and all time points unless ``max_trs`` caps them, so that its values are
comparable across iterations at a fraction of the full-data cost. ``cost_`` is
still the full-data objective at the returned iterate. Measured on synthetic data
with the real mask (69585 voxels, 1137 time points, ``K = 100``, caps 3000 x 300,
load average about 5):

.. list-table::
   :header-rows: 1

   * - loss, stopping objective
     - stop (iteration)
     - final full-data cost
     - median center error
     - one evaluation
     - per iteration (with evaluation)
   * - ``soft_l1``, 6000 x 300 subsample
     - 12
     - 13368240
     - 2.369 mm
     - 0.05 s
     - 0.34 to 0.51 s (about 13% for the evaluation)
   * - ``soft_l1``, all data
     - 12
     - 13368240
     - 2.369 mm
     - 1.73 s
     - 2.13 to 2.91 s
   * - ``linear``, 6000 x 300 subsample
     - 14
     - 16216090
     - 2.168 mm
     - 0.02 s
     - 0.07 to 0.09 s
   * - ``linear``, all data
     - 15
     - 16215297
     - 2.204 mm
     - 0.42 s
     - 0.45 to 0.48 s

The one-iteration relative decreases on the subsample differed from the
full-data ones by at most 1.2e-4 (4e-5 from iteration 6 on); a 10000 x 300
subsample took 0.08 s and 20000 x 300 0.16 s per evaluation. Larger caps lower
the noise floor (and the error of the result) more than more iterations do.

Other settings that change the cost of a fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``init='auto'`` runs three starts (``'ncorr'``, ``'power'``, ``'kmeans'``) and
  keeps the one with the lowest final objective.
  ``init='power'`` runs one start. ``n_screen_iter > 0`` runs each start for that
  many iterations and continues only the best. Each start runs a K-means++ on
  the coordinates with ``n_init=10`` on one thread, for reproducible sums (7.6
  to 8.0 s per start at ``V = 69585``, ``K = 100`` at a load average of 7; 55 to 83 s
  under a load average of 100 to 190).
- ``n_jobs`` runs the per-subject fits of an HTFA in parallel (NumPy backend).
  ``n_threads`` sets the BLAS/OpenMP threads of each subject fit; by default
  HTFA takes the lowest limit already in effect (``OMP_NUM_THREADS``,
  ``OPENBLAS_NUM_THREADS``, an enclosing ``threadpoolctl.threadpool_limits``),
  or all CPUs, divided among the ``n_jobs`` workers.
- ``nlss_loss='soft_l1'`` costs ``O(V T K^2)`` per Gauss-Newton assembly and
  needs the slow alternation (below); the default ``'linear'`` with
  ``'varpro'`` costs ``O(V T K (D + 1))`` per assembly.
- ``K``, the number of factors, enters every product at least linearly.

Solver and outer loop
---------------------

**Where the time went.** Profile of the earlier default HTFA (``soft_l1``, all
data, ``init='auto'``, BrainIAK's template update) on three adults of
``development_fmri`` (V = 23726 to 26739, T = 168, K = 50), 12 threads, one
subject at a time (``n_jobs=None``), load average 2 to 10, 851 s in all:

.. list-table:: Profile, earlier defaults
   :header-rows: 1

   * - part
     - time
     - notes
   * - stage 0 (unpenalized fits)
     - 644 s (76%)
     - 215 s per subject: 2 starts x 100 outer iterations, all at ``max_iter``
   * - global iteration 1 (from the template and from each own start)
     - 46 s
     - 10 outer iterations per fit
   * - global iterations 2 to 10
     - 13 to 24 s each
     - template change 0.027 after 10 iterations: no stop
   * - inside the fits: center/width solves
     - 824 s (97%)
     - 920 solves, 10.5 cost evaluations each
   * - Hessian assemblies
     - 499 s
     - 3873, 129 ms each
   * - cost evaluations
     - 306 s
     - 5785, 53 ms each (fractional powers; 19 ms after commit ``02b5700``)
   * - ridge weights, factors, starts, objectives
     - 23 s
     -

The full-data fits do not converge because alternating the ridge weights with
the center/width solve is coordinate descent on strongly coupled variables: on a
real subject (K = 50) the ``soft_l1`` objective still fell from 810222 to 810009
between outer iterations 100 and 300, and the linear-loss alternation stopped on
``threshold`` after 190 iterations (50 s) at a cost of 1003458.7.
``nlss_method='varpro'`` minimizes the linear-loss objective with the weights
eliminated (`htfa._kernels.ProjectedProblem`): the same stationary points, and
on that subject a cost of 1002690.4 after 2 outer iterations (23.6 s; K = 100:
65 s). Anderson acceleration of the alternation did not help on real data
(810010 after 300 iterations against 810009 without it). The default is
therefore the linear loss with ``'varpro'`` (``nlss_method='auto'``); held-out
reconstruction and synthetic recovery are compared in :doc:`validation`.

A varpro solve is a bounded Levenberg-Marquardt run of 50 to 380 Hessian
assemblies on real data, most of them in long, nearly flat valleys where the
exact Hessian is slightly indefinite and the Gauss-Newton model is used (on
one subject the cost fell by 95% of its total decrease in the first 8
assemblies and by the remaining 5% over the next 120). Updating the variable
scaling at every iteration as scipy's ``x_scale='jac'`` does halved the
assemblies (62 instead of 129, 140 instead of 379 at K = 100) at costs within
1.5e-4 (relative) of the current solver, sometimes lower and sometimes higher;
it is not adopted.

Fitting a real dataset: where the time goes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measured 2026-10-26 on the M2 Max below while other jobs kept the load average
at the value given (12 cores), so these are upper bounds.

.. list-table:: One subject's stage-0 fit, development_fmri (V = 25079, T = 168)
   :header-rows: 1

   * - settings
     - K = 50
     - K = 100
     - load average
   * - earlier defaults (``soft_l1``, ``init='auto'``), 12 threads
     - 174 s (2 starts x 100 outer iterations, ``max_iter``)
     - not measured
     - 2 to 8
   * - defaults (linear, varpro, ``init='auto'``), 12 threads
     - 83 s (2 outer iterations)
     - 293 s (4)
     - 95 to 115
   * - same, 2 threads
     - 83 s
     - 271 s
     - 95 to 115
   * - ``init='ncorr'`` (one start), 12 threads
     - 47 s
     - 79 s
     - 95 to 115
   * - ``init='ncorr'``, 2 threads
     - 47 s
     - 79 s
     - 95 to 115

One fit gains nothing from 12 threads over 2 at this size, so ``n_jobs`` workers
with 2 threads each (``n_jobs=6, n_threads=2`` on 12 cores) give about six times
the throughput of the default one-subject-at-a-time HTFA.

HTFA rounds (each round fits every subject once, in parallel):

.. list-table:: HTFA wall time per round, development_fmri adults, K = 50
   :header-rows: 1

   * - settings
     - stage 0
     - global iteration 1 (from the template / from own starts)
     - later global iterations
     - total
   * - earlier defaults, 3 adults, sequential, 12 threads (load 2 to 10)
     - 644 s
     - 33 s / 13 s
     - 15 to 24 s each, 9 of them (no stop)
     - 851 s
   * - defaults, 33 adults, ``n_jobs=6, n_threads=2`` (load 90 to 160)
     - 1328 s
     - 1013 s / stopped
     - stopped
     - more than 40 min
   * - linear varpro, ``init='ncorr'``, ``template_update='fixed'``, 12 adults,
       ``n_jobs=6, n_threads=2`` (load 9 to 12)
     - 71 s
     - 117 s / 110 s
     - 2 to 29 s each, 9 of them (``max_global_iter``)
     - 383 s

So with the defaults the stage-0 fits and the first global iteration's two
fits per subject dominate. A real dataset of 33 subjects at K = 50 does not fit
in 10 minutes with the defaults on this machine; with ``init='ncorr'``, one of
the faster template updates and ``n_jobs`` workers the 12-subject run took 6.4
minutes.

At the 2014 DFFR shape (synthetic, ``scripts/workload_projection.py``: V = 69597,
T = 1137, K = 100, 'ncorr', 12 threads, load 100 to 150), one full-data varpro
fit took 27 s for its start and 2 outer iterations of 243 s; with
``max_voxels=3000, max_trs=300`` it took 22.5 s for its start and 6 iterations
of 6.1 s. The pilot run on the real data of that shape (23 subjects, the
linear-loss alternation, full data, 30 global iterations, 6 workers of 2
threads) took 2.78 h. With full data one varpro fit is about 8 minutes per
subject, so a 10-minute fit at that size needs subsampling.

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
