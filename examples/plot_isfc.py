"""
Inter-subject functional correlation (ISFC)
===========================================

Simulate three subjects on a 24 x 24 grid with four factors each. Factors 0 and 1
follow a signal shared by all subjects (plus subject-specific noise), and factors
2 and 3 are independent in every subject. Fit :class:`htfa.HTFA`; because the
factors of every subject are in template order, :func:`htfa.isfc` can compare
them across subjects. ISFC (average correlation of one subject's factor with the
other subjects' factors) is compared with within-subject
:func:`htfa.static_connectivity`; ``htfa.isfc(..., window=20)`` gives the
sliding-window version. Template order is arbitrary, so factors are matched to the
simulated ones with :func:`htfa.synthetic.match_factors` before display. The
printed numbers come from the code below.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import htfa
from htfa.synthetic import match_factors

K, T, n_sub = 4, 40, 3
_, Rs, truth = htfa.make_htfa_data(n_sub, (24, 24), n_samples=T, K=K, random_state=4)

rng = np.random.default_rng(2)
shared = rng.standard_normal((2, T))
Xs = []
for s in range(n_sub):
    W = rng.standard_normal((K, T))
    W[:2] = shared + 0.5 * rng.standard_normal((2, T))
    d2 = ((Rs[s][:, None, :] - truth["centers"][s][None]) ** 2).sum(axis=2)
    F = np.exp(-d2 / truth["widths"][s][None, :])
    Xs.append(F @ W + 0.1 * rng.standard_normal((Rs[s].shape[0], T)))

start = time.perf_counter()
res = htfa.fit(
    Xs, K=K, coords=Rs, standardize=False, max_global_iter=15, random_state=0
)
elapsed = time.perf_counter() - start
print(f"fit time: {elapsed:.1f} s")

# Template order is arbitrary; reorder to the simulated factor order for display.
perm = match_factors(truth["global_centers"], res.global_centers)
weights = [w[perm] for w in res.weights]
C_isfc = htfa.isfc(weights)
C_within = htfa.static_connectivity(weights[0])
print(
    "res.isfc() matches htfa.isfc(res.weights):",
    np.allclose(res.isfc(), htfa.isfc(res.weights)),
)
print(
    "ISFC diagonal (factor vs. same factor in other subjects):",
    np.round(np.diag(C_isfc), 2),
)
win = htfa.isfc(weights, window=20)
print("windowed ISFC shape:", win.shape)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, C, title in (
    (axes[0], C_within, "Within subject 0 (static connectivity)"),
    (axes[1], C_isfc, "Across subjects (ISFC)"),
):
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("factor")
    ax.set_ylabel("factor")
fig.colorbar(im, ax=axes, label="correlation", shrink=0.8)
plt.show()
