"""GPU (float32 MPS or CUDA) fits match the float64 CPU fit's accuracy."""

import time

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from htfa.synthetic import make_tfa_data, match_factors
from htfa.tfa import TFA

# Spec: the GPU's matched center error may exceed the CPU's by at most 0.05 mm,
# and the matched weights must correlate above 0.999.
CENTER_SLACK_MM = 0.05
WEIGHT_MIN_R = 0.999
# Exercise the default multi-start on both numerical backends.
INIT = "auto"


def _gpu() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.gpu
@pytest.mark.slow
def test_mps_fit_matches_cpu_accuracy() -> None:
    device = _gpu()
    if device is None:
        pytest.skip("needs torch with an MPS or CUDA device")
    X, R, truth = make_tfa_data((30, 30, 30), 200, K=20, random_state=0)
    fits = {}
    for backend in (device, "cpu"):
        t0 = time.perf_counter()
        model = TFA(K=20, standardize=False, init=INIT, backend=backend, random_state=0)
        model.fit(X, R)
        perm = match_factors(truth["centers"], model.centers_)
        err = np.linalg.norm(model.centers_[perm] - truth["centers"], axis=1)
        fits[backend] = (float(err.max()), model.weights_[perm])
        print(
            f"{backend}: {time.perf_counter() - t0:.1f} s, engine {model.engine_}, "
            f"init {model.init_}, {model.n_iter_} iter, max center err "
            f"{err.max():.4f} mm, mean {err.mean():.4f} mm"
        )
    (gpu_err, gpu_W), (cpu_err, cpu_W) = fits[device], fits["cpu"]
    r = np.corrcoef(gpu_W.ravel(), cpu_W.ravel())[0, 1]
    print(f"weights r = {r:.8f}; center err {device} {gpu_err:.4f}, cpu {cpu_err:.4f}")
    assert gpu_err <= cpu_err + CENTER_SLACK_MM
    assert r > WEIGHT_MIN_R


# K=100 exercises robust-loss reductions and voxel chunking on both engines.
# These are numerical-agreement tolerances, not performance assertions.
K100_MAX_CENTER_DIFF = 1e-3
K100_MIN_WEIGHT_R = 0.99999


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_fit_matches_cpu_at_k100() -> None:
    device = _gpu()
    if device is None:
        pytest.skip("needs torch with an MPS or CUDA device")
    X, R, truth = make_tfa_data(
        (20, 20, 15), 200, K=100, min_separation=1.0, random_state=0
    )
    fits = {}
    for backend in (device, "cpu"):
        t0 = time.perf_counter()
        model = TFA(
            K=100, max_iter=10, standardize=False, init="power", backend=backend,
            nlss_loss="soft_l1", nlss_method="bounded-lm", random_state=0,
        )  # fmt: skip
        with pytest.warns(ConvergenceWarning):
            model.fit(X, R)
        perm = match_factors(truth["centers"], model.centers_)
        err = np.linalg.norm(model.centers_[perm] - truth["centers"], axis=1)
        fits[backend] = (model.centers_[perm], model.weights_[perm], err.max())
        print(
            f"{backend}: {time.perf_counter() - t0:.1f} s, engine {model.engine_}, "
            f"max center err {err.max():.4f}, cost {model.cost_:.8g}"
        )
    (C_g, W_g, e_g), (C_c, W_c, e_c) = fits[device], fits["cpu"]
    diff = float(np.abs(C_g - C_c).max())
    r = np.corrcoef(W_g.ravel(), W_c.ravel())[0, 1]
    print(f"max |dC| {device} vs cpu {diff:.3e}; weights r {r:.10f}")
    assert e_g <= e_c + CENTER_SLACK_MM
    assert diff < K100_MAX_CENTER_DIFF
    assert r > K100_MIN_WEIGHT_R
