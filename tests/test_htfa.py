"""Tests for the HTFA estimator."""

import os
import time
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from htfa._matching import match_factors
from htfa.htfa import HTFA, _fit_subject, assign_posterior, map_update
from htfa.synthetic import make_htfa_data

# Tolerances: measured values (seed 0) are recorded with ~2-3x headroom.
# Global error is measured against the mean of the true subject centers.
GLOBAL_CENTER_TOL_MM = 0.3  # measured: 0.119 (seed 0, mixed T; 3D 0.050, 2D 0.054)
LOCAL_CENTER_TOL_MM = 0.1  # measured: 0.034 (seed 0, 2D subject 0)
LOCAL_CENTER_TOL_T1_MM = 1.0  # measured: 0.369 (seed 0, the T=1 subject)
WEIGHT_MIN_R = 0.995  # measured: 0.9981 (seed 0, 3D subject 3); 1 - r has 2.5x room
# Measured (seed 0): the mixed-T and different-V cases converge after 11 global
# iterations (scaled change 8.3e-4 and 8.6e-4; 1.02e-3 and 1.04e-3 after 10).
SLOW_MAX_GLOBAL_ITER = 15


def test_map_update_matches_closed_form() -> None:
    # Scalar Gaussian conjugate update: prior N(m0, s0), observations whose mean
    # has variance g (= global_var / n_subj). Posterior precision 1/s0 + 1/g.
    m0, s0, g = 2.0, 4.0, 1.0
    obs = np.array([[1.0], [3.0], [5.0], [-1.0]])  # (n_subj, D=1), mean 2.0
    obs = obs + 1.0  # mean 3.0
    post_mean, post_cov = map_update(
        np.array([m0]), np.array([[s0]]), np.array([[g]]), obs
    )
    expected_var = 1.0 / (1.0 / s0 + 1.0 / g)
    expected_mean = expected_var * (m0 / s0 + 3.0 / g)
    print("posterior", post_mean, post_cov, "expected", expected_mean, expected_var)
    np.testing.assert_allclose(post_mean, [expected_mean], rtol=1e-12)
    np.testing.assert_allclose(post_cov, [[expected_var]], rtol=1e-12)


def test_map_update_multivariate_matches_precision_form() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    S0 = A @ A.T + np.eye(3)
    B = rng.normal(size=(3, 3))
    G = B @ B.T + np.eye(3)
    m0 = rng.normal(size=3)
    obs = rng.normal(size=(5, 3))
    mean, cov = map_update(m0, S0, G, obs)
    P = np.linalg.inv(S0) + np.linalg.inv(G)
    exp_cov = np.linalg.inv(P)
    exp_mean = exp_cov @ (np.linalg.solve(S0, m0) + np.linalg.solve(G, obs.mean(0)))
    np.testing.assert_allclose(mean, exp_mean, rtol=1e-10)
    np.testing.assert_allclose(cov, exp_cov, rtol=1e-10)


def test_global_assignment_permutes_all_fields() -> None:
    rng = np.random.default_rng(0)
    K, D = 5, 3
    centers = rng.uniform(0, 50, size=(K, D))
    widths = rng.uniform(1, 10, size=K)
    cov = np.stack([np.eye(D) * (k + 1) for k in range(K)])
    var = np.arange(1.0, K + 1)
    perm = np.array([3, 0, 4, 1, 2])
    noisy = centers[perm] + rng.normal(scale=0.01, size=(K, D))
    out = assign_posterior(centers, noisy, widths[perm], cov[perm], var[perm])
    print("assigned centers error", np.max(np.abs(out[0] - centers)))
    np.testing.assert_allclose(out[0], centers, atol=0.05)
    np.testing.assert_array_equal(out[1], widths)
    np.testing.assert_array_equal(out[2], cov)
    np.testing.assert_array_equal(out[3], var)


def _check_recovery(model: HTFA, Rs: list, truth: dict, Xs: list) -> None:
    # Target: mean of the true subject centers (truth keeps the global order), the
    # best any estimator can do from these subjects; also report the generator's.
    target = np.mean(truth["centers"], axis=0)
    g_perm = match_factors(target, model.global_centers_)
    est = model.global_centers_[g_perm]
    g_err = np.max(np.linalg.norm(est - target, axis=1))
    gen_err = np.max(np.linalg.norm(est - truth["global_centers"], axis=1))
    print(f"global center max error: {g_err:.4f} (to generator: {gen_err:.4f})")
    assert g_err < GLOBAL_CENTER_TOL_MM
    for s in range(len(Xs)):
        perm = match_factors(truth["centers"][s], model.centers_[s])
        err = np.max(
            np.linalg.norm(model.centers_[s][perm] - truth["centers"][s], axis=1)
        )
        W = model.weights_[s][perm]
        if Xs[s].shape[1] > 1:
            rs = [
                np.corrcoef(W[k], truth["weights"][s][k])[0, 1] for k in range(model.K)
            ]
            print(f"subject {s}: center err {err:.4f}, min weight r {min(rs):.4f}")
            assert min(rs) > WEIGHT_MIN_R
            assert err < LOCAL_CENTER_TOL_MM
        else:
            print(f"subject {s} (T=1): center err {err:.4f}")
            assert err < LOCAL_CENTER_TOL_T1_MM


def test_htfa_recovers_global_template() -> None:
    Xs, Rs, truth = make_htfa_data(
        n_subjects=4, grid_shape=(15, 15, 15), n_samples=80, K=6, random_state=0
    )
    t0 = time.perf_counter()
    model = HTFA(K=6, standardize=False, random_state=0).fit(Xs, Rs)
    print(f"fit time {time.perf_counter() - t0:.2f}s, n_iter {model.n_iter_}")
    assert model.global_centers_.shape == (6, 3)
    assert model.global_widths_.shape == (6,)
    assert model.global_centers_cov_.shape == (6, 3, 3)
    assert model.global_widths_var_.shape == (6,)
    assert len(model.centers_) == 4 and model.weights_[0].shape == (6, 80)
    _check_recovery(model, Rs, truth, Xs)


def test_htfa_2d() -> None:
    Xs, Rs, truth = make_htfa_data(
        n_subjects=4, grid_shape=(30, 30), n_samples=80, K=6, random_state=0
    )
    model = HTFA(K=6, standardize=False, random_state=0).fit(Xs, Rs)
    _check_recovery(model, Rs, truth, Xs)


def _fit_converged(Xs: list, Rs: list) -> HTFA:
    """HTFA fit that must converge within `SLOW_MAX_GLOBAL_ITER` global iterations.

    The template change shrinks by ~0.8x per global iteration in the mixed-T and
    different-V cases (seed 0), so they need 11 iterations, one past the default.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = HTFA(
            K=6,
            max_global_iter=SLOW_MAX_GLOBAL_ITER,
            standardize=False,
            random_state=0,
        ).fit(Xs, Rs)
    print(f"converged after {model.n_iter_} global iterations")
    assert model.converged_
    return model


def test_htfa_mixed_sample_counts() -> None:
    Xs, Rs, truth = make_htfa_data(
        n_subjects=3, grid_shape=(15, 15, 15), n_samples=80, K=6, random_state=0
    )
    Xs[2] = Xs[2][:, :1]
    truth["weights"][2] = truth["weights"][2][:, :1]
    model = _fit_converged(Xs, Rs)
    assert model.weights_[2].shape == (6, 1)
    _check_recovery(model, Rs, truth, Xs)


def test_htfa_parallel_equals_serial() -> None:
    Xs, Rs, _ = make_htfa_data(
        n_subjects=3, grid_shape=(12, 12, 12), n_samples=40, K=4, random_state=1
    )
    a = HTFA(K=4, standardize=False, n_jobs=1, random_state=3).fit(Xs, Rs)
    b = HTFA(K=4, standardize=False, n_jobs=2, random_state=3).fit(Xs, Rs)
    # Serial and threaded fits run BLAS on one thread in one process: bit for bit.
    np.testing.assert_array_equal(a.global_centers_, b.global_centers_)
    np.testing.assert_array_equal(a.global_widths_, b.global_widths_)
    for s in range(3):
        np.testing.assert_array_equal(a.centers_[s], b.centers_[s])
        np.testing.assert_array_equal(a.weights_[s], b.weights_[s])


# Factors of the n_jobs test: K (D + 1) * 8 bytes = 1152 >= 1024, so every
# parameter-sized vector is above NumPy's small-allocation cache (< 1024 bytes)
# and macOS's tiny malloc zone (<= 1008 bytes, 16-byte quantum), whose varying
# 64-byte alignment changes Accelerate's rounding (measured: a 400 x 400 gemv
# gave 8 results at 8 offsets; at K=20, 4 to 6 distinct results in 12 repeated
# fits).
N_JOBS_TEST_K = 36


def _scramble_small_allocations(seed: int) -> list:
    """Live arrays of random small sizes, so that the next small allocations
    land at other addresses (and alignments)."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 130, size=int(rng.integers(50, 400)))
    return [np.empty(int(n)) for n in sizes]


@pytest.mark.parametrize("mode", ["fast", "exact"])
def test_htfa_bitwise_independent_of_n_jobs(mode: str) -> None:
    """Serial and parallel subject fits must agree bit for bit. Measured before
    the fixes: (1) loky worker processes got BLAS thread variables of
    cpu_count // n_jobs, and BLAS rounds differently with one thread than with
    several (12 cores, 7 workers: different subject centers; real data at
    K=100, 8 subjects, n_jobs=8 vs 4: a different template); (2) with every fit
    on one BLAS thread but still in worker processes, Windows CI runners with
    AMD EPYC 9V45/9V74 CPUs failed in 1-7% of runs, because OpenBLAS rounds
    differently in about 0.5-2.5% of freshly started processes there. The fits
    now run as threads of one process.

    The comparison is meaningful only where repeated serial fits agree bit for
    bit whatever the memory layout, so that is checked first, with the small
    allocations scrambled between fits (see N_JOBS_TEST_K). Both modes."""
    n = max(2, (os.cpu_count() or 1) // 2 + 1)
    K = N_JOBS_TEST_K
    Xs, Rs, _ = make_htfa_data(
        n_subjects=n, grid_shape=(20, 20, 20), n_samples=60, K=20, random_state=0
    )
    kw = {
        "K": K, "init": "ncorr", "max_global_iter": 2, "mode": mode, "random_state": 0
    }  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        layouts = set()
        for seed in range(4):
            junk = _scramble_small_allocations(seed)
            fit = _fit_subject(
                HTFA(**kw)._tfa_params(10), Xs[0], Rs[0], None, None, 7, 1
            )
            layouts.add((fit.centers_.tobytes(), fit.widths_.tobytes()))
            del junk
        print(f"serial subject fits under 4 memory layouts: {len(layouts)} distinct")
        assert len(layouts) == 1, "serial subject fits depend on the memory layout"
        serial = HTFA(n_jobs=1, **kw).fit(Xs, Rs)
        parallel = HTFA(n_jobs=n, **kw).fit(Xs, Rs)
    for s in range(n):
        diff = np.abs(serial.centers_[s] - parallel.centers_[s]).max()
        print(f"subject {s}: max |center difference| {diff:.3g}")
    np.testing.assert_array_equal(serial.global_centers_, parallel.global_centers_)
    np.testing.assert_array_equal(serial.global_widths_, parallel.global_widths_)
    for s in range(n):
        np.testing.assert_array_equal(serial.centers_[s], parallel.centers_[s])
        np.testing.assert_array_equal(serial.widths_[s], parallel.widths_[s])
        np.testing.assert_array_equal(serial.weights_[s], parallel.weights_[s])


def test_htfa_different_voxel_counts_per_subject() -> None:
    Xs, Rs, truth = make_htfa_data(
        n_subjects=3, grid_shape=(15, 15, 15), n_samples=80, K=6, random_state=0
    )
    # Drop one edge slab from subject 1 and two from subject 2 (different sides).
    for s, (axis, keep) in {1: (0, lambda r: r > 0), 2: (1, lambda r: r < 13)}.items():
        mask = keep(Rs[s][:, axis])
        Xs[s], Rs[s] = Xs[s][mask], Rs[s][mask]
    assert len({X.shape[0] for X in Xs}) == 3
    model = _fit_converged(Xs, Rs)
    _check_recovery(model, Rs, truth, Xs)


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_htfa_no_local_warning_spam(n_jobs: int) -> None:
    """The local fits' ConvergenceWarnings are suppressed, also when they run on
    worker threads (the filter is set once, around all fits)."""
    Xs, Rs, _ = make_htfa_data(
        n_subjects=3, grid_shape=(10, 10), n_samples=20, K=3, random_state=0
    )
    model = HTFA(
        K=3,
        max_global_iter=2,
        max_local_iter=1,
        threshold=0.0,
        standardize=False,
        n_jobs=n_jobs,
        random_state=0,
    )
    with pytest.warns(ConvergenceWarning) as record:
        model.fit(Xs, Rs)
    messages = [str(w.message) for w in record if w.category is ConvergenceWarning]
    print("ConvergenceWarnings:", messages)
    assert not model.converged_ and model.n_iter_ == 2
    assert len(messages) == 1 and messages[0].startswith("HTFA did not converge")


def test_htfa_worker_exception_reaches_caller() -> None:
    """An error inside one subject's fit on a worker thread reaches the caller
    as itself, with the worker's frames (_fit_subject, TFA.fit) in the
    formatted traceback, and only after the other running fits have finished:
    no worker thread outlives the call, and the process-wide BLAS limits are
    restored. Measured with joblib's threading backend, which returned at once:
    fits still running afterwards left OpenBLAS limited to one thread (a later
    test saw 1 instead of 2 on CI)."""
    import threading
    import traceback

    from threadpoolctl import threadpool_info

    Xs, Rs, _ = make_htfa_data(
        n_subjects=4, grid_shape=(20, 20), n_samples=40, K=3, random_state=0
    )
    Xs[1], Rs[1] = Xs[1][:4], Rs[1][:4]  # 4 voxels: K=5 is too many for TFA
    before = [p["num_threads"] for p in threadpool_info()]
    threads_before = threading.active_count()
    with pytest.raises(
        ValueError, match="must be smaller than the number of voxels"
    ) as err:
        HTFA(K=5, n_jobs=3, random_state=0).fit(Xs, Rs)
    after = [p["num_threads"] for p in threadpool_info()]
    print("threads before/after:", threads_before, threading.active_count())
    print("threadpool sizes before/after:", before, after)
    assert threading.active_count() == threads_before
    assert after == before
    text = "".join(traceback.format_exception(err.value))
    print(text[-1200:])
    assert "in _fit_subject" in text and "in _validate" in text


def test_htfa_validation_errors() -> None:
    Xs, Rs, _ = make_htfa_data(
        n_subjects=2, grid_shape=(10, 10), n_samples=20, K=3, random_state=0
    )
    with pytest.raises(ValueError, match="non-empty lists of equal length"):
        HTFA(K=3).fit([], [])
    with pytest.raises(ValueError, match="non-empty lists of equal length"):
        HTFA(K=3).fit(Xs, Rs[:1])
    R3 = np.column_stack([Rs[1], np.zeros(len(Rs[1]))])
    with pytest.raises(ValueError, match="same dimension D"):
        HTFA(K=3).fit(Xs, [Rs[0], R3])
    with pytest.raises(ValueError, match="max_global_iter must be >= 1"):
        HTFA(K=3, max_global_iter=0).fit(Xs, Rs)


def test_subject_threads_respect_caller_limits():
    """n_threads=None takes the caller's BLAS/OpenMP limit, never all CPUs."""
    import os
    import subprocess
    import sys

    from threadpoolctl import threadpool_limits

    from htfa.htfa import _available_threads

    with threadpool_limits(2):
        assert _available_threads() == 2
        assert HTFA()._subject_threads(1) == 2
        assert HTFA()._subject_threads(4) == 1
        assert HTFA(n_threads=3)._subject_threads(4) == 3
    code = "from htfa.htfa import HTFA; print(HTFA()._subject_threads(1))"
    env = os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    print(out.stdout, out.stderr)
    assert out.stdout.strip() == "1"
    with pytest.raises(ValueError, match="n_threads"):
        HTFA(n_threads=0)._subject_threads(1)


def test_htfa_fit_runs_with_n_threads():
    Xs, Rs, _ = make_htfa_data(2, (12, 12), 10, K=2, random_state=0)
    kw = {"K": 2, "max_global_iter": 1, "max_local_iter": 2, "random_state": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        a = HTFA(n_threads=1, **kw).fit(Xs, Rs)
        b = HTFA(n_threads=2, **kw).fit(Xs, Rs)
    # The engine's chunks do not depend on its thread count: bit for bit.
    np.testing.assert_array_equal(a.global_centers_, b.global_centers_)


def _small_htfa_data():
    return make_htfa_data(3, (10, 10), 20, K=3, random_state=0)


SMALL_HTFA = {
    "K": 3,
    "max_global_iter": 2,
    "max_local_iter": 3,
    "init": "power",
    "standardize": False,
    "random_state": 0,
}


def _fit_quiet(model, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(*args, **kwargs)


def test_htfa_exposes_free_fits_and_template_history():
    Xs, Rs, _ = _small_htfa_data()
    model = _fit_quiet(HTFA(**SMALL_HTFA), Xs, Rs)
    assert len(model.free_fits_) == 3
    for C, w in model.free_fits_:
        assert C.shape == (3, 2) and w.shape == (3,)
    t = model.template_subject_
    history = model.template_history_
    assert len(history) == model.n_iter_ + 1
    np.testing.assert_array_equal(history[0][0], model.free_fits_[t][0])
    np.testing.assert_array_equal(history[0][1], model.free_fits_[t][1])
    np.testing.assert_array_equal(history[-1][0], model.global_centers_)
    np.testing.assert_array_equal(history[-1][1], model.global_widths_)
    assert not np.array_equal(history[0][0], history[1][0])


def test_htfa_template_max_iter_controls_free_fits():
    Xs, Rs, _ = _small_htfa_data()
    zero = _fit_quiet(HTFA(template_max_iter=0, **SMALL_HTFA), Xs, Rs)
    full = _fit_quiet(HTFA(**SMALL_HTFA), Xs, Rs)
    assert HTFA().get_params()["template_max_iter"] == 100
    # max_iter=0 keeps each start; the default iterates away from it.
    diffs = [
        float(np.max(np.abs(a[0] - b[0])))
        for a, b in zip(zero.free_fits_, full.free_fits_, strict=True)
    ]
    print("free-fit center differences, 0 vs 100 iterations:", diffs)
    assert min(diffs) > 0


def test_htfa_starts_replace_free_fits_exactly():
    """Passing a model's own free fits as starts reproduces it bit for bit."""
    Xs, Rs, _ = _small_htfa_data()
    a = _fit_quiet(HTFA(**SMALL_HTFA), Xs, Rs)
    b = _fit_quiet(HTFA(**SMALL_HTFA), Xs, Rs, starts=a.free_fits_)
    np.testing.assert_array_equal(a.global_centers_, b.global_centers_)
    np.testing.assert_array_equal(a.global_widths_, b.global_widths_)
    for s in range(3):
        np.testing.assert_array_equal(a.centers_[s], b.centers_[s])
    with pytest.raises(ValueError, match="starts"):
        HTFA(**SMALL_HTFA).fit(Xs, Rs, starts=a.free_fits_[:2])
    with pytest.raises(ValueError, match="starts"):
        HTFA(**SMALL_HTFA).fit(Xs, Rs, starts=[(np.zeros((2, 2)), np.ones(2))] * 3)


def test_htfa_resume_from_init_template():
    """init_template sets the starting template; with the model's per-subject
    estimates as starts this resumes a fit."""
    Xs, Rs, _ = _small_htfa_data()
    first = _fit_quiet(HTFA(**SMALL_HTFA), Xs, Rs)
    resumed = _fit_quiet(
        HTFA(**SMALL_HTFA),
        Xs,
        Rs,
        starts=list(zip(first.centers_, first.widths_, strict=True)),
        init_template=(first.global_centers_, first.global_widths_),
    )
    np.testing.assert_array_equal(
        resumed.template_history_[0][0], first.global_centers_
    )
    np.testing.assert_array_equal(resumed.template_history_[0][1], first.global_widths_)
    step_first = np.abs(first.template_history_[-1][0] - first.template_history_[-2][0])
    step_resumed = np.abs(
        resumed.template_history_[1][0] - resumed.template_history_[0][0]
    )
    print(
        "last step of the first fit, first step after resuming:",
        step_first.max(),
        step_resumed.max(),
    )
    # The resumed fit continues the trajectory (measured: 0.147660 vs 0.147647).
    np.testing.assert_allclose(step_resumed.max(), step_first.max(), rtol=1e-3)
    with pytest.raises(ValueError, match="init_template"):
        HTFA(**SMALL_HTFA).fit(Xs, Rs, init_template=(np.zeros((2, 2)), np.ones(2)))


def test_subject_threads_without_limits_use_library_defaults():
    """No limit set: the BLAS libraries' own default thread counts (OpenMP's when
    no BLAS library is visible, as with macOS Accelerate), not os.cpu_count."""
    import json
    import os
    import subprocess
    import sys

    code = (
        "import json\n"
        "from threadpoolctl import threadpool_info\n"
        "from htfa.htfa import HTFA\n"
        "pools = threadpool_info()\n"
        "blas = [p['num_threads'] for p in pools if p['user_api'] == 'blas']\n"
        "omp = [p['num_threads'] for p in pools if p['user_api'] == 'openmp']\n"
        "counts = blas or omp\n"
        "one, many = HTFA()._subject_threads(1), HTFA()._subject_threads(10**6)\n"
        "print(json.dumps([counts, one, many]))\n"
    )
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    env = {k: v for k, v in os.environ.items() if k not in names}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    print(out.stdout, out.stderr)
    defaults, one, many = json.loads(out.stdout.splitlines()[-1])
    assert defaults and one == min(defaults) >= 1
    assert many == 1  # more workers than threads: one thread each


def test_available_threads_ignores_serial_openmp_stub():
    """torch on Windows loads libiompstubs5md.dll, a serial OpenMP stub that
    reports 1 thread under any limit (CI: windows-latest, after test_engines);
    the caller's limit must still read as 2 with torch imported first."""
    import json
    import subprocess
    import sys

    pytest.importorskip("torch")
    code = (
        "import json, torch\n"
        "from threadpoolctl import threadpool_info, threadpool_limits\n"
        "from htfa.htfa import _available_threads\n"
        "with threadpool_limits(2):\n"
        "    pools = [(p['user_api'], p['prefix'], p['num_threads'], p['filepath'])\n"
        "             for p in threadpool_info()]\n"
        "    print(json.dumps([pools, _available_threads()]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(out.stdout, out.stderr[-2000:])
    pools, available = json.loads(out.stdout.splitlines()[-1])
    for pool in pools:
        print(pool)
    assert available == 2


# Measured with this test's data (make_htfa_data(4, (15, 15, 15), 80, K=6), seed
# 0, defaults: linear varpro, init 'auto'): 2 global iterations for 'fixed' and
# for 'empirical' (Mahalanobis-trimmed dispersion), 10 for 'brainiak'.
FAST_UPDATE_MAX_ITER = 5


@pytest.mark.parametrize("update", ["fixed", "empirical"])
def test_fixed_prior_updates_recover_truth_in_few_global_iterations(update) -> None:
    Xs, Rs, truth = make_htfa_data(
        n_subjects=4, grid_shape=(15, 15, 15), n_samples=80, K=6, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        t0 = time.perf_counter()
        model = HTFA(
            K=6, template_update=update, standardize=False, random_state=0
        ).fit(Xs, Rs)
        reference = HTFA(K=6, standardize=False, random_state=0).fit(Xs, Rs)
    print(
        f"{update}: {model.n_iter_} global iterations ({model.converged_reason_}); "
        f"'brainiak' {reference.n_iter_}; {time.perf_counter() - t0:.1f} s; "
        f"prior center sd {np.sqrt(np.diag(model.prior_centers_cov_))}"
    )
    _check_recovery(model, Rs, truth, Xs)
    assert model.converged_reason_ in ("threshold", "plateau", "objective")
    assert model.n_iter_ <= FAST_UPDATE_MAX_ITER < reference.n_iter_
    assert len(model.objective_history_) == model.n_iter_
    # The template is the mean of the local estimates (in template order), and
    # the subject-level prior stays fixed.
    np.testing.assert_allclose(
        model.template_history_[-1][0], np.mean(model.centers_, axis=0), atol=1e-12
    )
    np.testing.assert_allclose(
        model.global_centers_cov_[0], model.prior_centers_cov_ / 4, rtol=1e-12
    )


def test_empirical_dispersion_recovers_covariance_and_ignores_mismatches() -> None:
    from htfa.htfa import empirical_dispersion

    rng = np.random.default_rng(1)
    true_cov = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    n, K = 30, 40
    mu = rng.uniform(0, 100, (K, 3))
    centers = mu + rng.multivariate_normal(np.zeros(3), true_cov, (n, K))
    widths = 50.0 + rng.normal(0.0, 3.0, (n, K))
    clean_cov, clean_var = empirical_dispersion(centers, widths, 1e-3 * np.eye(3), 0.0)
    # 3% of the (s, k) pairs are factor mismatches: centers 40 units away.
    bad = rng.random((n, K)) < 0.03
    shifted = centers + bad[:, :, None] * 40.0
    cov, var = empirical_dispersion(shifted, widths, 1e-3 * np.eye(3), 0.0)
    print(f"clean {np.round(clean_cov, 2).tolist()}, {clean_var:.3f}")
    print(f"with mismatches {np.round(cov, 2).tolist()}")
    np.testing.assert_allclose(clean_cov, true_cov, atol=0.25)
    assert clean_var == pytest.approx(9.0, rel=0.1)
    np.testing.assert_allclose(cov, true_cov, atol=0.5)
    floored, floored_var = empirical_dispersion(centers, widths, 100.0 * np.eye(3), 1e3)
    np.testing.assert_allclose(np.linalg.eigvalsh(floored), 100.0, rtol=0.05)
    assert floored_var == 1e3
    with pytest.raises(ValueError, match="at least 2 subjects"):
        empirical_dispersion(centers[:1], widths[:1], np.eye(3), 0.0)


def test_global_loop_stops_on_plateau_or_objective_without_warning() -> None:
    """With threshold=0 the change rule can never fire; the plateau and
    objective rules must still stop the loop well before max_global_iter, with
    no ConvergenceWarning."""
    Xs, Rs, _ = make_htfa_data(
        n_subjects=3, grid_shape=(12, 12), n_samples=40, K=3, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = HTFA(
            K=3,
            threshold=0.0,
            max_global_iter=40,
            init="ncorr",
            standardize=False,
            random_state=0,
        ).fit(Xs, Rs)
    obj = np.array(model.objective_history_)
    print(
        f"{model.n_iter_} iterations ({model.converged_reason_}); relative "
        f"objective changes {np.abs(np.diff(obj)) / obj[:-1]}"
    )
    assert model.converged_ and model.converged_reason_ in ("plateau", "objective")
    assert model.n_iter_ < 40


def test_htfa_rejects_unknown_template_update() -> None:
    Xs, Rs, _ = make_htfa_data(
        n_subjects=2, grid_shape=(10, 10), n_samples=20, K=3, random_state=0
    )
    with pytest.raises(ValueError, match="template_update must be one of"):
        HTFA(K=3, template_update="em").fit(Xs, Rs)


@pytest.mark.parametrize("n_iter", [1, 2, 3])
def test_htfa_two_stage_loss_with_few_local_iterations(n_iter) -> None:
    """template_max_iter and max_local_iter at or below the varpro stage's own
    stop leave the soft_l1 stage no iterations; the fit must still finish."""
    Xs, Rs, _ = make_htfa_data(
        n_subjects=2, grid_shape=(12, 12), n_samples=30, K=3, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = HTFA(
            K=3,
            nlss_loss="linear+soft_l1",
            template_max_iter=n_iter,
            max_local_iter=n_iter,
            max_global_iter=2,
            init="ncorr",
            standardize=False,
            random_state=0,
        ).fit(Xs, Rs)
    print(f"n_iter={n_iter}: {model.n_iter_} global ({model.converged_reason_})")
    assert model.n_iter_ >= 1 and len(model.centers_) == 2
