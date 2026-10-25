"""
The HTFA global template
========================

Simulate three subjects whose factor centers are jittered copies of a shared
template (:func:`htfa.make_htfa_data`), fit :class:`htfa.HTFA` through
:func:`htfa.fit` and draw the estimated template (black) together with each
subject's own centers (colors) using :meth:`htfa.HTFAResults.plot_nodes`.
The printed numbers come from the code below.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import htfa
from htfa.synthetic import match_factors

Xs, Rs, truth = htfa.make_htfa_data(
    n_subjects=3, grid_shape=(24, 24), n_samples=40, K=4, random_state=0
)

start = time.perf_counter()
res = htfa.fit(
    Xs,
    K=4,
    coords=Rs,
    standardize=False,
    max_global_iter=15,
    random_state=0,
)
elapsed = time.perf_counter() - start
print(f"fit time: {elapsed:.1f} s over {len(res.subjects)} subjects")

perm = match_factors(truth["global_centers"], res.global_centers)
err = np.linalg.norm(res.global_centers[perm] - truth["global_centers"], axis=1)
print("template center error per factor (grid units):", np.round(err, 3))
for s in range(len(res.subjects)):
    p = match_factors(truth["centers"][s], res.centers[s])
    e = np.linalg.norm(res.centers[s][p] - truth["centers"][s], axis=1).mean()
    print(f"subject {s}: mean center error {e:.3f}")

fig = res.plot_nodes("global")
ax = fig.axes[0]
ax.set_title("HTFA template (black) and per-subject centers (colors)")
ax.set_xlabel("column coordinate")
ax.set_ylabel("row coordinate")
plt.show()
