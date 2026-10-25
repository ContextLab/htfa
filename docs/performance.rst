Performance
===========

Backends
--------

The center and width of each factor are found by a least-squares solve whose
cost is dominated by three matrix products between the ``(V, T)`` data, the
``(V, K)`` factor images and the ``(K, T)`` weights. An engine computes these
products. The ``backend`` argument of :class:`htfa.TFA`, :class:`htfa.HTFA` and
:func:`htfa.fit` selects the engine:

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

Choosing a backend with ``'auto'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``'auto'`` uses a GPU torch engine only if all of the following hold:

1. torch is already installed (``'auto'`` never installs it);
2. a CUDA or MPS device is available;
3. ``V * T * K`` is at least ``htfa._engines.GPU_MIN_WORK``, currently
   ``1e10``.

Otherwise it uses NumPy. ``V`` and ``T`` in the rule are the subsampled sizes
that the solve works on, not the full data size (see below). The ``1e10``
threshold is provisional: on the machine described below, no fit that finished
ran faster on MPS than on NumPy, and the engine's matrix products alone ran 1.3 to
1.5 times faster on MPS at ``V * T * K = 1e10``.

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

Other settings that change the cost of a fit:

- ``init='auto'`` runs three starts (``'ncorr'``, ``'power'``, ``'kmeans'``) and
  keeps the one with the lowest final objective.
  ``init='power'`` runs one start. ``n_screen_iter > 0`` runs each start for that
  many iterations and continues only the best.
- ``n_jobs`` runs the per-subject fits of an HTFA in parallel (NumPy backend).
- ``K``, the number of factors, enters every product linearly.

Benchmarks
----------

.. note::

   The tables below were measured on **2026-10-24** with the solver as of
   commit ``5912bf0`` ("Add benchmarks, GPU accuracy test, tuned auto threshold").
   A faster solver is in development, and these numbers will be replaced when it
   is measured.

**Machine.** M2 Max (12 CPUs), macOS 26.6.2, Python 3.12.11, NumPy 2.5.4,
torch 2.14.1 (MPS, float32). Other jobs kept about 9 to 11 of the 12 cores busy
throughout, so the absolute times are inflated.

**Data.** Synthetic data from :func:`htfa.make_tfa_data` with
``standardize=False``, all voxels and samples. The grids are ``(25, 20, 10)``
(small), ``(40, 25, 20)`` (medium) and ``(50, 40, 25)`` (large).

**How to read the tables.** ``V``, ``T`` and ``K`` are the number of voxels, time
points and factors, and ``V*T*K`` is the quantity compared with ``GPU_MIN_WORK``.
"fit (s)" is wall-clock seconds for one TFA fit. "speedup vs numpy" is the NumPy
time divided by that row's time, so a value below 1 means the backend was slower
than NumPy. "max center err" is the largest distance between an estimated and the
true factor center, in grid units, after matching with
:func:`htfa.synthetic.match_factors`. ``n/a`` means the quantity was not
measured, and a row that gives a reason in place of a time did not finish.
Reproduce the tables with ``scripts/benchmark.py``.

Full fits
~~~~~~~~~

``scripts/benchmark.py --sizes small --backends numpy,mps,brainiak``; TFA defaults
(``init='auto'``, ``max_iter=100``).

.. list-table::
   :header-rows: 1

   * - size
     - V
     - T
     - K
     - V*T*K
     - backend
     - fit (s)
     - speedup vs numpy
     - max center err
     - notes
   * - small
     - 5000
     - 200
     - 20
     - 2.00e+07
     - numpy
     - 402.0
     - 1.00
     - 0.0148
     - init ncorr, 100 iterations
   * - small
     - 5000
     - 200
     - 20
     - 2.00e+07
     - mps
     - 792.5
     - 0.51
     - 0.0149
     - torch-mps, init ncorr, 100 iterations
   * - small
     - 5000
     - 200
     - 20
     - 2.00e+07
     - brainiak
     - n/a
     - n/a
     - n/a
     - timed out after 3600 s

The NumPy fit reached ``max_iter``. Its objective stayed near a saddle (objective
13885, center error about 4.6 mm, scaled change about 1.7e-3) from iteration 40 to
60, then left it and reached the true solution (objective 9881, center error
0.015) by iteration 100.

Engine throughput
~~~~~~~~~~~~~~~~~

``scripts/benchmark.py --sizes small,medium,large --backends numpy,mps
--init power --max-iter 5``: a fixed 5 iterations, so the times compare engines
and are not fit times.

.. list-table::
   :header-rows: 1

   * - size
     - V
     - T
     - K
     - V*T*K
     - backend
     - fit (s)
     - speedup vs numpy
   * - small
     - 5000
     - 200
     - 20
     - 2.00e+07
     - numpy
     - 36.5
     - 1.00
   * - small
     - 5000
     - 200
     - 20
     - 2.00e+07
     - mps
     - 90.3
     - 0.40
   * - medium
     - 20000
     - 300
     - 100
     - 6.00e+08
     - numpy
     - 1344.2
     - 1.00
   * - medium
     - 20000
     - 300
     - 100
     - 6.00e+08
     - mps
     - 1810.1
     - 0.74
   * - large
     - 50000
     - 1000
     - 200
     - 1.00e+10
     - numpy
     - more than 10800 for 1 iteration (abandoned)
     - n/a

**Where the time goes.** scipy's ``least_squares`` (LSMR) works on vectors of
length ``V*T`` on the CPU, so on a GPU backend every LSMR step moves ``V*T``
floats between the CPU and the GPU.
