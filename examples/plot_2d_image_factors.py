"""
TFA on a real 2D image
======================

TFA treats the pixels of a single 2D image as locations and its intensities as one
sample. The image used here is ``china.jpg``, the photograph bundled with
scikit-learn (:func:`sklearn.datasets.load_sample_image`, no download). It is
converted to grayscale (ITU-R 601 luma) and block-averaged by 12 to keep the fit
fast. Six radial basis factors are fitted and the reconstruction is drawn next to
the image with the factor centers. The printed numbers come from the code below.

A single image is one sample, and ``standardize='auto'`` leaves single-sample data
unchanged. The fit here passes ``standardize=True``, which z-scores the pixels across
the image. Without it, the ridge penalty on the weights (``var(X)``, as in BrainIAK)
is in raw intensity units and the model has no intercept: on this image every factor
then moves to the same location at the upper width bound. The second fit below shows
that for comparison.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image

import htfa

htfa.set_autoinstall(False)

photo = load_sample_image("china.jpg")  # (427, 640, 3) uint8
b = 12
h, w = photo.shape[0] // b, photo.shape[1] // b
small = photo[: h * b, : w * b].reshape(h, b, w, b, 3).mean(axis=(1, 3))
gray = small @ np.array([0.299, 0.587, 0.114])  # ITU-R 601 luma
print(f"image: {photo.shape} -> {gray.shape} after {b} x {b} block averaging")

start = time.perf_counter()
res = htfa.fit(gray, K=6, standardize=True, random_state=0)
elapsed = time.perf_counter() - start

original = res.subjects[0].to_image(res.subjects[0].data[:, 0])
recon = res.to_image(0)[..., 0]
r = np.corrcoef(res.subjects[0].data[:, 0], res.reconstruct(0)[:, 0])[0, 1]
print(f"fit time: {elapsed:.1f} s, stopped because: {res.model.converged_reason_}")
print(f"correlation between image and reconstruction: {r:.3f}")
print("factor widths w = 2*sigma^2 (pixels^2):", np.round(res.widths[0], 1))

# htfa warns about this setting; print the message (the default display would
# also print the path of the htfa source file that issued it)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    raw = htfa.fit(gray, K=6, standardize=False, random_state=0)
for w in caught:
    print(f"{w.category.__name__}: {w.message}")
r_raw = np.corrcoef(raw.subjects[0].data[:, 0], raw.reconstruct(0)[:, 0])[0, 1]
print(f"standardize=False: correlation {r_raw:.3f}, widths", np.round(raw.widths[0], 1))
upper = 2 * 1.8 * float(raw.subjects[0].coords.std(0).max()) ** 2
print("upper width bound:", round(upper, 1))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].imshow(original, cmap="gray")
axes[0].set_title("Grayscale image")
axes[1].imshow(recon, cmap="gray")
axes[1].set_title("Reconstruction from 6 factors")
axes[2].imshow(original, cmap="gray")
c = res.centers[0]
axes[2].scatter(c[:, 1], c[:, 0], c="r", s=40)
axes[2].set_title("Factor centers")
for ax in axes:
    ax.set_xlabel("column (pixel)")
    ax.set_ylabel("row (pixel)")
fig.tight_layout()
plt.show()
