"""
Original data and reconstruction
================================

Fit a single-subject TFA to a simulated 24 x 24 grid with 6 samples (time points)
and compare one sample with the model's reconstruction
(:meth:`htfa.HTFAResults.plot_reconstruction`), with the simulation's
noiseless signal as a third panel (``truth=``) on the same color scale. The
reconstruction is
``factors @ weights`` in the space the model was fitted in (here the raw values,
because ``standardize=False``). The example also prints the fraction of variance
explained, shows what :func:`htfa.tfa.standardize_data` does to the data, and
maps results back to image space with :meth:`htfa.HTFAResults.to_image` and
:meth:`htfa.Subject.to_image`.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import htfa
from htfa.tfa import standardize_data

X, R, truth = htfa.make_tfa_data((24, 24), n_samples=6, K=4, noise=0.2, random_state=1)

# six 24 x 24 images, stacked as the samples of one subject
images = [X[:, t].reshape(24, 24) for t in range(6)]
start = time.perf_counter()
res = htfa.fit(images, K=4, stack=True, standardize=False, random_state=0)
elapsed = time.perf_counter() - start

raw = res.subjects[0].data
Z = standardize_data(raw, False)  # False leaves the data unchanged
print("row std after 'auto' z-scoring:", standardize_data(raw, "auto").std(axis=1)[:3])
recon = res.reconstruct(0)
r2 = 1 - np.sum((Z - recon) ** 2) / np.sum((Z - Z.mean()) ** 2)
print(f"fit time: {elapsed:.1f} s, variance explained: {r2:.3f}")
print("factor images:", res.factors(0).shape, "(V, K)")
print("reconstruction in image space:", res.to_image(0).shape)
print("one sample as an image:", res.subjects[0].to_image(recon[:, 0]).shape)

# the noiseless signal F W of the simulation, shown as a third panel
d2 = ((R[:, None, :] - truth["centers"][None]) ** 2).sum(axis=2)
signal = np.exp(-d2 / truth["widths"]) @ truth["weights"]
r_recon = np.corrcoef(recon.ravel(), signal.ravel())[0, 1]
print(f"correlation of the reconstruction with the noiseless signal: {r_recon:.3f}")

fig = res.plot_reconstruction(subject=0, sample=0, truth=signal)
fig.suptitle(f"Sample 0: data vs. reconstruction (R$^2$ = {r2:.2f} over all samples)")
for ax in fig.axes[:3]:  # the three panels; the last axes is the colorbar
    ax.set_xlabel("column")
    ax.set_ylabel("row")
plt.show()
