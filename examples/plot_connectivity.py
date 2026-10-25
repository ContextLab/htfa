"""
Factor connectivity
===================

Simulate a 24 x 24 grid with six factors whose time courses form two correlated
groups (factors 0-2 share one signal, factors 3-5 another). Fit
:class:`htfa.TFA`, compute the factor-by-factor correlation with
:func:`htfa.static_connectivity` and draw it as a matrix and as edges between
factor centers (:meth:`htfa.HTFAResults.plot_connectivity`). The estimated matrix
is compared with the correlation of the true weights after matching factors with
:func:`htfa.synthetic.match_factors`. The printed numbers come from the code
below.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import htfa
from htfa.synthetic import match_factors

K, T = 6, 60
X0, R, truth = htfa.make_tfa_data((24, 24), n_samples=T, K=K, random_state=2)

# Replace the independent weights by two groups of correlated time courses, and
# rebuild the data with the same radial basis functions
# F[v, k] = exp(-||r_v - c_k||^2 / w_k).
rng = np.random.default_rng(0)
shared = rng.standard_normal((2, T))
groups = (0, 0, 0, 1, 1, 1)
W_true = np.vstack([shared[g] + 0.6 * rng.standard_normal(T) for g in groups])
d2 = ((R[:, None, :] - truth["centers"][None]) ** 2).sum(axis=2)
F = np.exp(-d2 / truth["widths"][None, :])
X = F @ W_true + 0.1 * rng.standard_normal((R.shape[0], T))

start = time.perf_counter()
res = htfa.fit(X, K=K, coords=R, standardize=False, random_state=0)
elapsed = time.perf_counter() - start

perm = match_factors(truth["centers"], res.centers[0])
C_est = htfa.static_connectivity(res.weights[0])[np.ix_(perm, perm)]
C_true = htfa.static_connectivity(W_true)
iu = np.triu_indices(K, k=1)
r = np.corrcoef(C_true[iu], C_est[iu])[0, 1]
print(f"fit time: {elapsed:.1f} s")
print(f"correlation between true and estimated edges: {r:.3f}")
print(
    "res.connectivity(0) equals static_connectivity(weights):",
    np.allclose(res.connectivity(0), htfa.static_connectivity(res.weights[0])),
)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, C, title in ((axes[0], C_true, "True"), (axes[1], C_est, "Estimated")):
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(f"{title} factor correlation")
    ax.set_xlabel("factor")
    ax.set_ylabel("factor")
fig.colorbar(im, ax=axes[:2], label="Pearson r", shrink=0.8)
res.plot_connectivity(subject=0, threshold="50%", ax=axes[2])
axes[2].set_title("Strongest 50% of edges")
axes[2].set_xlabel("column coordinate")
axes[2].set_ylabel("row coordinate")
plt.show()
