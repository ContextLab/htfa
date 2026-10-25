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
