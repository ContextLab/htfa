Performance
===========

A fit's cost depends on the number of voxels, time points, factors, mixture
components, starts, and subjects. Use the measurements below as a description
of the stated workload, not a prediction for an arbitrary dataset.

Backends
--------

``backend='numpy'`` (or ``'cpu'``) uses NumPy on the CPU. ``'torch'`` uses
an available CUDA or MPS device, falling back to torch on the CPU;
``'cuda'`` and ``'mps'`` require that specific device. ``'auto'`` uses a GPU
only when torch is already installed, a GPU is available, and the subsampled
workload ``V * T * K`` reaches ``htfa._engines.GPU_MIN_WORK``. Otherwise it
uses NumPy. This heuristic is not a guarantee that a GPU is faster; compare
explicit backends on the workload of interest.

Solver modes
------------

The default linear loss and ``nlss_method='auto'`` use variable projection
(``'varpro'``): the weights are eliminated analytically, and a bounded
Levenberg–Marquardt solver optimizes the spatial parameters. Robust losses
use alternating weight and spatial-parameter updates. ``'bounded-lm'``,
``'trf'`` and ``'dogbox'`` select explicit spatial solvers.

``mode='fast'`` and ``mode='exact'`` minimize the same objective and use the
same outer stopping rules. On NumPy, costs and gradients use float64 in both
modes. Fast mode forms the largest curvature products in float32 and sums
them in float64; it retains the Hessian with sufficient damping to make a
Newton step possible. Exact mode forms curvature in float64 and uses a
Gauss–Newton step when the Hessian is not positive definite. These choices
can lead to different local minima. A shorter runtime does not establish a
better fit; compare objective values and held-out performance as well.

Subsampling and stopping
------------------------

``voxel_ratio`` and ``tr_ratio`` select fractions of voxels and time points;
``max_voxels`` and ``max_trs`` cap their counts. Each iteration draws a fresh
subset, while the final temporal weights use all data. ``random_state``
controls the draws. Smaller subsets reduce spatial-solve cost but can change
the fitted factors.

Changing subsets introduces variation between successive spatial estimates.
A subsampled fit therefore also monitors the objective on a fixed evaluation
subset and returns the best evaluated iterate. ``objective_history_`` records
that objective, ``cost_`` reports the full-data objective at the returned
iterate, and ``converged_reason_`` identifies the stopping rule. Reaching an
iteration limit is distinct from satisfying a convergence rule.

Starts, factors, and parallel subjects
--------------------------------------

``init='auto'`` evaluates multiple initializations. A specific start such as
``'ncorr'`` limits the fit to that initialization; ``n_screen_iter`` screens
starts before continuing the best. More factors or mixture components increase
the number of spatial parameters and the cost of matrix products.

``n_jobs`` controls concurrent subject fits in HTFA. NumPy subject fits share
one process using threads, and ``n_threads`` controls the engine thread budget
per worker. HTFA pins BLAS reductions consistently so that changing ``n_jobs``
does not change results within a process. Standalone TFA uses the caller's
BLAS thread limits. Avoid assigning every worker all available CPU cores.

Interrupting parallel fitting cancels subject fits that have not started;
already-running subject fits finish before control returns. Memory requirements
include the subjects' input arrays, factor images, and spatial-solver matrices.

Measured CPU performance
------------------------

The reproducible benchmark is ``scripts/benchmark_release.py``. It uses three
adult participants from Nilearn's ``development_fmri`` dataset, an illustrative
fMRI workload in which participants watch Pixar's *Partly Cloudy*. Each run
contains 168 brain volumes sampled two seconds apart. The benchmark uses full
runs, K = 50 and 100, ``init='ncorr'``, voxel-wise temporal z-scoring, and two
BLAS threads. Each mode runs twice per subject, with the mode order reversed on the second repetition. Data
loading, preprocessing, and a warmup fit are outside the timed region.

The benchmark runs one fit at a time, waits for a quiet host, and samples CPU
activity outside its process family during each fit. Baseline CPU use must
remain below 10% for three samples; an external process using more than
25% of one core for three consecutive samples invalidates a timed fit.
No tutorial executions, test suites, or other computational jobs run alongside it. Raw results include
hardware and software versions, input hashes, settings, stopping reasons,
objective values, and external CPU measurements. These details make the scope
of the comparison explicit.

.. include:: _static/benchmarks/summary.rst

To reproduce the table, install the development dependencies and ``psutil``,
then run from the repository root on an otherwise idle machine::

    python scripts/benchmark_release.py benchmark.json

The benchmark refuses to start on a busy host and flags rows with competing
CPU activity. Do not report those rows as uncontended measurements. Absolute
runtime and the relative performance of modes depend on hardware, BLAS,
thread count, initialization, data, and stopping behavior. See :doc:`validation`
for numerical checks and the tutorials for reconstruction comparisons.
