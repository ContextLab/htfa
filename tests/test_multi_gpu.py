"""CUDA selection and subject scheduling, using real fits and available devices."""

import subprocess
import sys
import textwrap
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from htfa import HTFA
from htfa._engines import select_engine
from htfa.htfa import _cuda_devices, _fit_subject, _fit_subject_batch, _subject_batches
from htfa.synthetic import make_htfa_data


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [True], [1.5], "cuda:0", 1])
def test_invalid_device_selection(indices):
    with pytest.raises(ValueError, match="gpu_devices"):
        _cuda_devices(indices, 2)


def test_device_selection_respects_visible_ordinals():
    assert _cuda_devices(None, 3) == ["cuda:0", "cuda:1", "cuda:2"]
    assert _cuda_devices([2, 0], 3) == ["cuda:2", "cuda:0"]
    with pytest.raises(ValueError, match="unavailable"):
        _cuda_devices([2], 2)


@pytest.mark.parametrize("backend", ["cuda:-1", "cuda:", "cuda:1.0", "cuda:one"])
def test_invalid_indexed_backend(backend):
    with pytest.raises(ValueError, match="backend"):
        select_engine(backend, 10, 10, 2)


def test_unavailable_indexed_device():
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="CUDA device"):
        select_engine(f"cuda:{torch.cuda.device_count()}", 10, 10, 2)


def _data():
    Xs, Rs, _ = make_htfa_data(3, (6, 6), 8, K=2, random_state=10)
    return Xs, Rs


def test_cpu_rejects_gpu_selection():
    Xs, Rs = _data()
    with pytest.raises(ValueError, match="gpu_devices requires"):
        HTFA(K=2, backend="numpy", gpu_devices=[0]).fit(Xs, Rs)


def test_device_queues_preserve_subject_results():
    """Exercise production queue workers with real CPU fits, no GPU emulator."""
    Xs, Rs = _data()
    params = dict(
        K=2, backend="numpy", standardize=False, init="power", max_iter=2, threshold=1e6
    )
    jobs = [
        (params, x, r, None, None, s, 1)
        for s, (x, r) in enumerate(zip(Xs, Rs, strict=True))
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        expected = [_fit_subject(*job) for job in jobs]
        with ThreadPoolExecutor(2) as pool:
            results = [
                pool.submit(_fit_subject_batch, [(i, jobs[i]) for i in batch], Event())
                for batch in _subject_batches(range(3), 2)
            ]
            indexed = [item for future in results for item in future.result()]
    assert sorted(i for i, _ in indexed) == [0, 1, 2]
    for i, result in indexed:
        np.testing.assert_array_equal(result.centers_, expected[i].centers_)
        np.testing.assert_array_equal(result.weights_, expected[i].weights_)


def test_device_queue_error_cancels_pending_subjects():
    cancelled = Event()
    invalid = (
        dict(K=2, backend="numpy"),
        np.ones((2, 2)),
        np.ones((3, 2)),
        None,
        None,
        0,
        1,
    )
    with pytest.raises(ValueError):
        _fit_subject_batch([(0, invalid)], cancelled)
    assert cancelled.is_set()
    assert _fit_subject_batch([(1, invalid)], cancelled) == []


@pytest.mark.gpu
def test_multi_cuda_matches_serial_subject_order():
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")
    Xs, Rs = _data()
    kw = dict(
        K=2,
        backend="cuda",
        gpu_devices=[0, 1],
        standardize=False,
        max_global_iter=1,
        max_local_iter=2,
        template_max_iter=2,
        init="power",
        random_state=0,
        threshold=1e6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        serial = HTFA(n_jobs=1, **kw).fit(Xs, Rs)
        parallel = HTFA(n_jobs=2, **kw).fit(Xs, Rs)
    assert serial.subject_engines_ == ["torch-cuda:0"] * 3
    assert parallel.subject_engines_ == ["torch-cuda:0", "torch-cuda:1", "torch-cuda:0"]
    assert parallel.gpu_devices_ == ["cuda:0", "cuda:1"]
    for actual, expected in zip(parallel.weights_, serial.weights_, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        parallel.global_centers_, serial.global_centers_, rtol=1e-6, atol=1e-7
    )


@pytest.mark.gpu
def test_mps_subjects_remain_serial_device():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    Xs, Rs = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = HTFA(
            K=2,
            backend="mps",
            n_jobs=2,
            standardize=False,
            max_global_iter=1,
            max_local_iter=2,
            template_max_iter=2,
            init="power",
            random_state=0,
            threshold=1e6,
        ).fit(Xs, Rs)
    assert model.gpu_devices_ == []
    assert model.subject_engines_ == ["torch-mps"] * 3
    assert all(np.isfinite(w).all() for w in model.weights_)


def test_subset_keeps_original_device_assignment():
    # Subject 0 is the template subject and skips the second start. Subjects
    # 1 and 2 must retain devices 1 and 0, not shift into devices 0 and 1.
    assert _subject_batches([1, 2], 2) == [[1], [0]]


def test_subject_scheduler_routes_subsets_to_stable_backends():
    """Run the production device scheduler with two real NumPy backend aliases."""
    Xs, Rs = _data()
    model = HTFA(K=2, standardize=False, init="power", threshold=1e6)
    model.gpu_devices_ = ["numpy", "cpu"]
    model._gpu_worker_count_ = 2
    fitted = model._fit_subjects(
        Xs[1:],
        Rs[1:],
        2,
        [None, None],
        [None, None],
        np.array([1, 2]),
        parallel=True,
        subject_indices=[1, 2],
    )
    assert [fit.backend for fit in fitted] == ["cpu", "numpy"]
    for fit, X in zip(fitted, Xs[1:], strict=True):
        assert fit.weights_.shape == (2, X.shape[1])
        assert np.isfinite(fit.cost_)


def test_explicit_device_is_the_initial_backend():
    torch = pytest.importorskip("torch")
    index = torch.cuda.device_count()
    Xs, Rs = _data()
    with pytest.raises(RuntimeError, match=f"cuda:{index}"):
        HTFA(K=2, gpu_devices=[index]).fit(Xs, Rs)


@pytest.mark.gpu
def test_cuda_subset_does_not_initialize_excluded_default_device():
    """A fresh process selecting only GPU 1 must leave GPU 0 untouched."""
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")
    if not hasattr(torch._C, "_cuda_hasPrimaryContext"):
        pytest.skip("this torch build cannot inspect CUDA context creation")
    script = textwrap.dedent("""
        import torch
        from htfa import HTFA
        from htfa.synthetic import make_htfa_data
        assert not torch._C._cuda_hasPrimaryContext(0)
        X, R, _ = make_htfa_data(2, (6, 6), 8, K=2, random_state=10)
        model = HTFA(
            K=2, gpu_devices=[1], standardize=False, init="power",
            max_global_iter=1, max_local_iter=2, template_max_iter=2,
            threshold=1e6, random_state=0,
        ).fit(X, R)
        assert model.subject_engines_ == ["torch-cuda:1"] * 2
        assert torch._C._cuda_hasPrimaryContext(1)
        assert not torch._C._cuda_hasPrimaryContext(0)
    """)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)


def test_refit_clears_device_metadata_and_clone_preserves_selection():
    from sklearn.base import clone

    selected = [1, 0]
    estimator = HTFA(K=2, gpu_devices=selected)
    copied = clone(estimator)
    assert copied.gpu_devices == selected
    assert copied.gpu_devices is not selected
    Xs, Rs = _data()
    # Refit an estimator whose previous device metadata belongs to a different
    # workload; all metadata must describe the new CPU fit.
    estimator.gpu_devices_ = ["cuda:1"]
    estimator.subject_engines_ = ["torch-cuda:1"]
    estimator.set_params(
        gpu_devices=None,
        backend="numpy",
        init="power",
        standardize=False,
        max_global_iter=1,
        max_local_iter=2,
        template_max_iter=2,
        threshold=1e6,
        random_state=0,
    ).fit(Xs, Rs)
    assert estimator.gpu_devices_ == []
    assert estimator.subject_engines_ == ["numpy"] * len(Xs)
