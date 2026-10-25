"""
Sliding-window (dynamic) connectivity
=====================================

Simulate four factors on a 24 x 24 grid in which factors 0 and 1 are positively
correlated in the first half of the scan and negatively correlated in the second
half. Fit :class:`htfa.TFA` and compute sliding-window correlations with
:func:`htfa.dynamic_connectivity` (also available as
:meth:`htfa.HTFAResults.dynamic_connectivity`). The printed numbers come from the
code below.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import htfa
from htfa.synthetic import match_factors

K, T, window = 4, 60, 15
_, R, truth = htfa.make_tfa_data((24, 24), n_samples=T, K=K, random_state=5)

rng = np.random.default_rng(1)
W_true = rng.standard_normal((K, T))
sign = np.where(np.arange(T) < T // 2, 1.0, -1.0)
W_true[1] = sign * W_true[0] + 0.3 * rng.standard_normal(T)
d2 = ((R[:, None, :] - truth["centers"][None]) ** 2).sum(axis=2)
F = np.exp(-d2 / truth["widths"][None, :])  # F[v, k] = exp(-||r_v - c_k||^2 / w_k)
X = F @ W_true + 0.1 * rng.standard_normal((R.shape[0], T))

start = time.perf_counter()
res = htfa.fit(X, K=K, coords=R, standardize=False, random_state=0)
elapsed = time.perf_counter() - start

perm = match_factors(truth["centers"], res.centers[0])
W_est = res.weights[0][perm]
dyn_est = htfa.dynamic_connectivity(W_est, window)
dyn_true = htfa.dynamic_connectivity(W_true, window)
print(f"fit time: {elapsed:.1f} s; dynamic connectivity shape {dyn_est.shape}")
print(
    "res.dynamic_connectivity matches:",
    np.allclose(
        res.dynamic_connectivity(0, window),
        htfa.dynamic_connectivity(res.weights[0], window),
    ),
)
print(
    "correlation of the 0-1 edge over windows, true vs. estimated:",
    round(float(np.corrcoef(dyn_true[:, 0, 1], dyn_est[:, 0, 1])[0, 1]), 3),
)

starts = np.arange(dyn_est.shape[0])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(starts, dyn_true[:, 0, 1], label="true")
ax.plot(starts, dyn_est[:, 0, 1], "--", label="estimated")
ax.axvline(
    T // 2 - window / 2, color="gray", lw=0.8, label="sign change (window center)"
)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("window start (sample)")
ax.set_ylabel("correlation of factors 0 and 1")
ax.set_title(f"Sliding-window correlation (window = {window} samples)")
ax.legend()
plt.show()
