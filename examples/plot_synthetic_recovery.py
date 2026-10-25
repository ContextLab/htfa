"""
Recovering known factors
========================

Fit :class:`htfa.TFA` to simulated data with known centers and compare the
estimates with the truth. Estimated factors come in arbitrary order, so
:func:`htfa.synthetic.match_factors` pairs them with the true ones (Hungarian
algorithm on squared distance) before the error is computed. The printed numbers
come from the code below.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import htfa
from htfa.synthetic import match_factors

X, R, truth = htfa.make_tfa_data((30, 30), n_samples=50, K=5, noise=0.1, random_state=3)

start = time.perf_counter()
model = htfa.TFA(K=5, standardize=False, random_state=0).fit(X, R)
elapsed = time.perf_counter() - start

perm = match_factors(truth["centers"], model.centers_)
est_c = model.centers_[perm]
est_w = model.widths_[perm]
err = np.linalg.norm(est_c - truth["centers"], axis=1)
print(f"fit time: {elapsed:.1f} s, stopped because: {model.converged_reason_}")
print("matching permutation:", perm)
print("center error per factor (grid units):", np.round(err, 3))
print(f"mean center error: {err.mean():.3f}")
print("true widths     :", np.round(truth["widths"], 2))
print("estimated widths:", np.round(est_w, 2))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
ax = axes[0]
ax.scatter(truth["centers"][:, 1], truth["centers"][:, 0], s=90, label="true")
ax.scatter(est_c[:, 1], est_c[:, 0], marker="x", s=70, c="r", label="estimated")
for t, e in zip(truth["centers"], est_c, strict=True):
    ax.plot([t[1], e[1]], [t[0], e[0]], "k-", lw=0.8)
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_aspect("equal")
ax.set_xlabel("column coordinate")
ax.set_ylabel("row coordinate")
ax.set_title(f"Centers (mean error {err.mean():.2f})")
ax.legend()

ax = axes[1]
lim = [0, max(truth["widths"].max(), est_w.max()) * 1.1]
ax.plot(lim, lim, "k--", lw=0.8)
ax.scatter(truth["widths"], est_w)
ax.set_xlabel("true width $w = 2\\sigma^2$")
ax.set_ylabel("estimated width")
ax.set_title("Widths")
fig.tight_layout()
plt.show()
