Performance
===========

Backends
--------

The centers and widths are found by a bounded Levenberg-Marquardt solve
(``nlss_method='bounded-lm'``) on the explicit ``K (D + 1)``-square
Gauss-Newton system; with the default linear loss the same solver runs on the
variable-projection objective (``'varpro'``, see `Solver and outer loop`_).
Assembling that system is dominated by products between the
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
  Every subject fit runs BLAS on one thread, serial or parallel, so that the
  results do not depend on ``n_jobs`` (BLAS rounds differently with one thread
  than with several, and joblib gives each worker ``cpu_count // n_jobs``
  threads). On development_fmri subject 0 (Accelerate, 12 cores, load average
  3 to 6, mean of two fits) one BLAS thread made a fit 12% slower at ``K = 50``
  (15.7 s instead of 14.1 s) and 17% slower at ``K = 100`` (28.9 s instead of
  24.7 s); workers that joblib already limits to one thread
  (``n_jobs > cpu_count / 2``) lose nothing. A standalone :class:`htfa.TFA`
  keeps the BLAS threads of the caller, so on macOS it and HTFA's stage-0 fit
  of the same subject can differ at rounding level.
  ``n_threads`` sets the OpenMP and NumPy engine threads of each subject fit; by default
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

Fast and exact modes
~~~~~~~~~~~~~~~~~~~~

`htfa.TFA` and `htfa.HTFA` take ``mode='fast'`` (the default) or
``mode='exact'``. Both modes minimize the same objective, with costs and
gradients in float64 on NumPy and the same stopping rules, and both give
results that do not depend on ``n_jobs``. They differ in two places of
`htfa._kernels.bounded_lm`, the solver of ``'bounded-lm'`` and ``'varpro'``:

- The step where the Hessian is not positive definite on the free variables.
  ``'exact'`` takes the damped Gauss-Newton step there, as earlier versions
  did. ``'fast'`` keeps the Hessian and raises the damping until the damped
  matrix is positive definite (a damped Newton step). At the k-th failed
  factorization within a step the fast mode multiplies the damping by 2**k
  (`htfa._kernels.FAILED_FACTOR_GROWTH`); the exact mode keeps the earlier
  rule.
- The precision of the curvature. ``'exact'`` assembles the Gauss-Newton
  matrix and the Hessian in float64. ``'fast'`` forms their three largest
  products for the linear loss (``G G^T``, ``F G^T`` and ``G X``, with ``G``
  the factor derivatives) in float32 and sums them in float64
  (`htfa._kernels.curvature_operand`). Costs and gradients are not affected.

The equivalence tests against commit ``3298278`` and the BrainIAK cross-checks
run in ``'exact'`` mode (the cross-checks in both).

**Why the step model matters.** On development_fmri the Hessian is indefinite
at nearly every point of a solve. In six logged solves (sub-pixar123 and
sub-pixar124, K = 50 and 100, 'ncorr' and 'kmeans' starts) the exact mode
took the Newton step in 0 to 11 of 85 to 523 accepted steps; the others were
Gauss-Newton steps, which converge linearly on this large-residual problem.
With a prototype of the damped Newton step (float64), the 'kmeans' solve of
sub-pixar123 at K = 100 took 155 steps to a cost of 988678.64 against 466
steps to 990024.95 in exact mode, and its 'ncorr' solve at K = 50 98 steps
(1003673.70) against 258 (1003580.12). The fast mode can therefore end at a
different local minimum, lower or higher; :doc:`validation` gives its effect
on the objective, the held-out reconstruction, the decoding and the HTFA
template.

**The damping after failed factorizations.** The exact mode, as earlier
versions, multiplies the damping by ``nu`` and doubles ``nu`` at each failed
factorization, so ``nu`` stays raised for the rejections of the same step. In
the fast mode ``nu`` counts rejected steps only, and the k-th failure within a
step multiplies the damping by 2**k. Doubling the damping per failure (at most twice the damping that
works) was tried: on a well-conditioned synthetic problem of the test suite
(two-stage loss, 12³ grid, K = 8) the fast fit then reached another local
minimum, 2.9e-3 above the exact fit's objective with centers 1.45 lengths
away; with the growth by 2**k it ends within the test's tolerances of the
exact fit (objective within 1e-8). Exact-mode fits are those of the
code before ``mode`` existed, bit for bit (checked on six development_fmri
adults at K = 50 and 100, including fits with failed factorizations, and by
``tests/test_fast_mode.py``).

**Why float32.** On sub-pixar123 (V = 25079, T = 168, K = 100) ``G G^T``,
``F G^T`` and ``G X`` took 4.4, 2.5 and 2.7 ms in float32 against 10.6, 7.5
and 7.1 ms in float64 (Accelerate), and a Hessian assembly 9% to 17% less
(K = 50 and 100), with relative errors of 5e-6 to 1e-5 in the Gauss-Newton
matrix and the Hessian.

**Timings.** The first six adults (full runs, defaults, 12 threads), the
exact and the fast fit of each subject one after the other, 2026-10-27, at
load averages of 2.5 to 5.7 (every speedup quoted in these docs is from
this run):

.. list-table:: Whole TFA fits, 6 development_fmri adults
   :header-rows: 1

   * - K
     - exact, mean
     - fast, mean
     - ratio of total times (per subject)
     - Hessian assemblies per fit, exact / fast
   * - 50
     - 24.6 s
     - 8.7 s
     - 2.8 (1.7 to 4.4)
     - 241 to 693 / 105 to 186
   * - 100
     - 113.4 s
     - 21.6 s
     - 5.3 (3.2 to 8.4)
     - 695 to 1881 / 195 to 247

HTFA with its defaults on the twelve adults (K = 50, ``n_jobs=6``, 10 global
iterations in both modes; ``scripts/validation_modes.py --htfa``) took 664 s
in exact mode and 244 s in fast mode (2.7 times; load average 4.9 when the
run started, other jobs raised it during the run).

**Options measured and not adopted** (development_fmri, same machine, with
the earlier damping rule):

- Gauss-Newton steps only (skipping the second-order terms, which the exact
  mode computes and then rarely uses). The few Newton steps matter near the
  end: on sub-pixar123 at K = 50 the outer loop ran 30 iterations instead of
  2, and the fit took 19.8 s against 22.8 s for the exact mode, with the
  'ncorr' candidate 3.4e-4 above the exact one's objective. With float32
  curvature products as well it ran 7 iterations in 22.9 s.
- The same Gauss-Newton-only, float32 variant with a looser ``ftol`` (1e-7,
  1e-6): each solve stops earlier and the outer loop runs more iterations (18
  and 54), for 22.8 s and 19.5 s.
- float32 curvature products with the exact step model: 20.4, 20.9 and
  20.8 s against 22.8, 20.8 and 24.9 s (sub-pixar123 to 125, K = 50).
- Limiting the damped Newton step to the length of the Gauss-Newton step: on
  the synthetic fits of the test suite it moved other well-conditioned fits
  to other minima (one 6.2e-3 from the exact objective).
- The approximate Hessians of "Cost of one evaluation" below (from a voxel
  subsample, or reused across steps): 0.3 to 2.0 times the exact solver's
  time on single solves.

Fitting a real dataset: where the time goes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measured 2026-10-26 on the M2 Max below while other jobs kept the load average
at the value given (12 cores), so these are upper bounds. They predate
``mode='fast'``: every row is an exact-mode fit (see "Fast and exact modes"
above for fast-mode times).

.. list-table:: One subject's stage-0 fit, development_fmri (V = 25079, T = 168)
   :header-rows: 1

   * - settings
     - K = 50
     - K = 100
     - load average
   * - earlier defaults (``soft_l1``, ``init='auto'``), 12 threads, one
       ``TFA.fit`` of ``sub-pixar123``
     - 174 s (2 starts x 100 outer iterations, ``max_iter``; 215 s per
       subject in the 3-subject HTFA profile above, load 4 to 10)
     - not measured
     - 2 to 9
   * - ``init='ncorr'``, 12 threads (defaults otherwise)
     - 18.7 s
     - not measured
     - 3 to 7
   * - ``init='ncorr'``, 2 threads
     - 19.2 s
     - not measured
     - 3 to 7
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

At a load average of 3 to 7, one fit took 18.7 s with 12 threads and 19.2 s
with 2, so ``n_jobs`` workers of 2 threads use the cores better than one subject
at a time (``n_jobs=None``, the default): six adults (``init='ncorr'``, K = 50)
took 104.8 s one after the other with 12 threads and 49.0 s as 6 workers of 2
threads, 2.1 times the throughput. The rows at a load average of 95 to 115 are
not comparable with these.

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

Cost of one evaluation
~~~~~~~~~~~~~~~~~~~~~~

A varpro solve alternates trial-point cost evaluations with Hessian
assemblies; an accepted step costs one of each, at the same point. Four
changes (2026-10-26) make these cheaper without changing what is computed:

- On grid coordinates (`htfa._kernels.axis_grid`: at most half as many
  distinct coordinate values, summed over the axes, as voxels), each factor is
  the product over the axes of ``exp(-(r_j - c_j)**2 / w)``, evaluated on the
  distinct values and gathered. Arbitrary coordinates keep the direct form.
- The Hessian assembly reuses the factors, ridge weights and gram inverse of
  the cost evaluation at the same point.
- The projected objective is evaluated as ``||X||**2 - <F^T X, W>``, exact at
  the ridge solution, instead of from the ``V x T`` residual. A problem whose
  objective at its first point is below ``IDENTITY_MIN_FRACTION`` (1e-3) of
  ``||X||**2`` keeps the residual form, since the identity loses about
  ``log10(||X||**2 / objective)`` digits.
- ``factor_derivatives`` subtracts the centers from a contiguous copy of the
  coordinates.

Median of 15 evaluations at nearby points, development_fmri subject 0
(V = 25079, T = 168), 12 threads, load average 1.2 to 2.8; before = commit
``3298278``, after = ``8ba060a``:

.. list-table:: Time per evaluation (ms), before / after
   :header-rows: 1

   * - evaluation
     - K = 50
     - K = 100
   * - cost at a trial point
     - 16.0 / 5.4
     - 31.1 / 9.5
   * - Hessian assembly at a new point
     - 46.6 / 47.0
     - 86.0 / 84.3
   * - accepted step (cost, then Hessian at that point)
     - 65.5 / 46.2
     - 118.7 / 82.9

Whole fits, ``init='ncorr'`` (deterministic; ``'auto'`` adds K-means starts),
12 threads, load average 2.4 to 4.5:

.. list-table:: TFA fit time (s), before / after
   :header-rows: 1

   * - subject
     - K = 50
     - K = 100
   * - 0
     - 18.95 / 13.69
     - 32.84 / 23.82
   * - 1
     - 18.60 / 13.90
     - 101.77 / 67.09 (38 / 23 outer iterations)
   * - 2
     - 7.21 / 5.65
     - 80.08 / 72.66 (586 / 768 Hessian assemblies)

HTFA with the defaults (K = 50, ``init='auto'``, BrainIAK's template update,
which ran all 10 global iterations) on the first six adults, ``n_jobs=6,
n_threads=2``: 495.4 s before, 368.8 s after (load average 2.0 and 6.4 when
each run started; the runs themselves fill the 12 cores). The template centers
agree to 4.2e-5 (relative to the largest), the local weights to 5.6e-4 and the
summed objective to 2e-8; the same HTFA on data changed by a relative 1e-14
moved by 3.7e-3, 0.62 and 2.2e-5.

**Same answers.** The reuse and the contiguous copy give bit-identical
results (fits of 12 synthetic and 6 real cases). The per-axis factors agree
with the direct ones to 8.7e-14 (relative, per entry) and the identity with
the residual form to 3e-15 on real data and 3.2e-13 on unstandardized
synthetic data with noise 0.003. At fixed points (the fitted optima and two
perturbations of them, real subjects 0 to 2), cost, Gauss-Newton matrix,
Hessian and weights agree with the earlier code to 6.3e-13, and gradients at
the perturbed points to 1.4e-11 (relative to the largest entry); at the optima
themselves the gradient is rounding noise around 0 and its relative difference
(up to 1.6e-11 here, 1e-2 on synthetic data) is not meaningful. A sweep over
492 generated scenarios is in :doc:`validation`.

Whole fits are determined only to the precision of the stopping rules, and
move under any rounding change. Changing the data by a relative 1e-14, 1e-13
or 1e-12 moves the earlier code's results by up to 1.0e-8, 2.6e-8 and 4.4e-8
(K = 50) and by up to 2.5e-3, 8.4e-4 and 2.1e-3 (K = 100; one more subject
reached another optimum under the 1e-14 and the 1e-12 change, its objective
8e-5 lower). The new code's
results differ from the earlier code's by up to 1.1e-7 (K = 50), 2.5 times
the largest of these, and 1.5e-3 (K = 100), with objectives within 1.1e-7 (at
K = 100 equal or lower). On the synthetic fits they differ by up to 1.8e-10,
except one high-SNR fit where another start won with a 1.1e-7 lower objective
(the same factors, 7e-8 apart).

**Not adopted.** Three approximate Hessians, with the exact cost and gradient
(so the same stationary points), were tried on single varpro solves from the
``'ncorr'`` start of subjects 0 to 2: the Hessian from a random half or quarter
of the voxels, from float32 products, and reused for up to 2 or 5 accepted
steps while the gain ratio stays above 0.5. Only the float32 Hessian at
K = 50 reached the same point (costs within 1.2e-11, centers within 4.3e-3
mm), and it was no faster (0.95 to 1.03 times the time). All other runs
ended at other points of the long flat valleys these solves end in: final
costs differed from
the exact solver's by -3.5e-4 to +2.3e-4 (relative), in both directions, and
centers by up to tens of mm. Their times ranged from 0.3 to 2.0 times the
exact solver's.

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
