"""GPU (float32 MPS or CUDA) fits match the float64 CPU fit's accuracy."""

import time

import numpy as np
import pytest

from htfa.synthetic import make_tfa_data, match_factors
from htfa.tfa import TFA

# Spec: the GPU's matched center error may exceed the CPU's by at most 0.05 mm,
# and the matched weights must correlate above 0.999.
CENTER_SLACK_MM = 0.05
WEIGHT_MIN_R = 0.999
# The default multi-start. With the structured solver both fits pick 'power' and
# take 131 s (MPS) and 222 s (CPU) on an M2 Max under a CPU load of ~175 (with
# scipy's trf the MPS fit had not finished after an hour).
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
