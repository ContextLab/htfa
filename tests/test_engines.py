"""Engines: NumPy vs torch parity and backend selection (real devices, no mocks)."""

import importlib.util

import numpy as np
import pytest

from htfa._engines import GPU_MIN_WORK, NumpyEngine, TorchEngine, select_engine


def _devices() -> list[str]:
    if importlib.util.find_spec("torch") is None:
        return []
    import torch

    found = ["cpu"]
    if torch.cuda.is_available():
        found.append("cuda")
    if torch.backends.mps.is_available():
        found.append("mps")
    return found


def _has_gpu() -> bool:
    return any(d in ("cuda", "mps") for d in _devices())


def test_select_engine_cpu():
    assert select_engine("cpu", 10, 5, 2).name == "numpy"
    assert select_engine("numpy", 10, 5, 2).name == "numpy"


def test_select_engine_rejects_unknown():
    with pytest.raises(ValueError, match="backend"):
        select_engine("tpu", 10, 5, 2)


def test_auto_small_problem_uses_numpy():
    assert select_engine("auto", 100, 10, 5).name == "numpy"


def test_auto_switches_to_gpu_at_gpu_min_work():
    below = select_engine("auto", GPU_MIN_WORK - 1, 1, 1)
    at = select_engine("auto", GPU_MIN_WORK, 1, 1)
    print(f"GPU_MIN_WORK={GPU_MIN_WORK:.3g}: below -> {below.name}, at -> {at.name}")
    assert below.name == "numpy"
    if _has_gpu():
        assert at.name in ("torch-cuda", "torch-mps")
    else:
        assert at.name == "numpy"


def test_unavailable_device_raises():
    if "cuda" in _devices():
        pytest.skip("cuda is available here, so the unavailable case cannot occur")
    with pytest.raises(RuntimeError, match="cuda"):
        select_engine("cuda", 10, 5, 2)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_torch_engine_matches_numpy(device):
    if device not in _devices():
        pytest.skip(f"no {device}")
    rng = np.random.default_rng(0)
    V, T, K = 40, 25, 6
    X = rng.standard_normal((V, T))
    W = rng.standard_normal((K, T))
    F = rng.standard_normal((V, K))
    u = rng.standard_normal(V * T)
    ref = NumpyEngine()
    ref.set(X, W)
    eng = TorchEngine(device)
    eng.set(X, W)
    assert eng.name == f"torch-{device}"
    rtol, atol = (1e-5, 1e-5) if device == "mps" else (1e-10, 1e-10)
    for got, want in [
        (eng.residual(F), ref.residual(F)),
        (eng.times_w(F), ref.times_w(F)),
        (eng.times_wt(u), ref.times_wt(u)),
    ]:
        assert got.dtype == np.float64
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


def test_torch_backend_picks_best_device():
    if not _devices():
        pytest.skip("torch is not installed")
    eng = select_engine("torch", 10, 5, 2)
    best = "cuda" if "cuda" in _devices() else "mps" if "mps" in _devices() else "cpu"
    assert eng.name == f"torch-{best}"


@pytest.mark.parametrize("loss", ["linear", "soft_l1", "huber", "cauchy", "arctan"])
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_torch_engine_gauss_newton_matches_numpy(device, loss):
    """The structured solver's data-term pass on torch matches NumPy (cost,
    (rho' r) W^T and J^T u J): float64 devices to 1e-10, float32 MPS to 1e-4."""
    if device not in _devices():
        pytest.skip(f"no {device}")
    from htfa._engines import loss_terms

    rng = np.random.default_rng(1)
    V, T, K, A = 300, 30, 5, 4
    X = rng.standard_normal((V, T)) * 1.5
    W = rng.standard_normal((K, T))
    F = rng.uniform(size=(V, K))
    G = rng.standard_normal((K, V, A))
    s = 0.7
    ref = NumpyEngine()
    ref.set(X, W)
    eng = TorchEngine(device)
    eng.set(X, W)
    eng.chunk_elements = 50 * T  # several chunks, also on the GPU
    want = ref.gauss_newton(F, G, s, loss)
    got = eng.gauss_newton(F, G, s, loss)
    rtol = 1e-4 if device == "mps" else 1e-10
    for name, g_, w_ in zip(("cost", "M", "H"), got, want, strict=True):
        g_, w_ = np.asarray(g_), np.asarray(w_)
        rel = np.abs(g_ - w_).max() / np.abs(w_).max()
        print(f"{device} {loss} {name}: max relative difference {rel:.3e}")
        assert g_.dtype == np.float64 and g_.shape == w_.shape
        assert rel < rtol
    r = s * (X - F @ W)
    expected_cost = 0.5 * np.sum(loss_terms(loss, r * r)[0])
    assert eng.data_cost(F, s, loss) == pytest.approx(expected_cost, rel=rtol)


@pytest.mark.parametrize("loss", ["linear", "soft_l1"])
def test_numpy_engine_threads_give_identical_results(loss):
    """Chunks on several threads are summed in chunk order: bit-identical."""
    rng = np.random.default_rng(2)
    V, T, K, A = 500, 40, 6, 4
    X = rng.standard_normal((V, T))
    W = rng.standard_normal((K, T))
    F = rng.uniform(size=(V, K))
    G = rng.standard_normal((K, V, A))
    results = []
    for n in (1, 4):
        eng = NumpyEngine(n_threads=n)
        eng.chunk_elements = 30 * K * K  # 17 chunks
        eng.set(X, W)
        results.append(
            eng.gauss_newton(F, G, 0.7, loss) + (eng.data_cost(F, 0.7, loss),)
        )
    print(f"{loss}: chunks {len(eng._chunks(K * K))}")
    for one, four in zip(*results, strict=True):
        np.testing.assert_array_equal(one, four)


def test_thread_limit_sets_numpy_engine_threads():
    import os

    from htfa._engines import default_threads, thread_limit

    assert NumpyEngine().n_threads == (os.cpu_count() or 1) == default_threads()
    with thread_limit(2):
        assert NumpyEngine().n_threads == 2
        with thread_limit(0):
            assert default_threads() == 1
        assert select_engine("numpy", 10, 5, 2).n_threads == 2
    assert NumpyEngine().n_threads == (os.cpu_count() or 1)
    assert NumpyEngine(n_threads=3).n_threads == 3
