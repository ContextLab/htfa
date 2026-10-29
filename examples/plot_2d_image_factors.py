"""
TFA on a real 2D image
======================

TFA represents image pixels as spatial locations and grayscale intensity as one
sample. This example compares 6, 80 and 320 radial basis factors on ``china.jpg``,
the photograph bundled with scikit-learn (no download). Block averaging by 8
produces a 53 x 80 image. The same image, seed, width bounds and initialization
are used for each fit. More factors can resolve the building's outline and
levels, while fine texture remains harder to reconstruct.

``standardize=True`` z-scores the image across pixels. The smaller width bound
``lower_ratio=0.001`` permits Gaussian standard deviations down to about 0.73
pixels; the default bound would rule out these narrow features. R² uses the
image-wide mean as its baseline and describes fitted pixels, not generalization.
All panels share grayscale limits. The largest fit takes the most time.

See the :doc:`2D image tutorial </tutorials/05_2d_images>` for image loading,
color channels, multiple pages and a wider factor-count comparison.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image
from threadpoolctl import threadpool_limits

import htfa

htfa.set_autoinstall(False)
photo = load_sample_image("china.jpg")
b = 8
h, w = photo.shape[0] // b, photo.shape[1] // b
small = photo[: h * b, : w * b].reshape(h, b, w, b, 3).mean(axis=(1, 3))
gray = small @ np.array([0.299, 0.587, 0.114])
original = (gray - gray.mean()) / gray.std()
print(f"image: {photo.shape} -> {gray.shape} after {b} x {b} block averaging")

reconstructions, scores = {}, {}
for k in (6, 80, 320):
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        result = htfa.fit(
            gray,
            K=k,
            standardize=True,
            init="power",
            random_state=0,
            lower_ratio=0.001,
            threshold=1e-5,
        )
    reconstructions[k] = result.to_image(0)[..., 0]
    scores[k] = 1 - np.sum((original - reconstructions[k]) ** 2) / np.sum(original**2)
    print(
        f"K={k}: {time.perf_counter() - start:.1f} s, "
        f"stop={result.model.converged_reason_}, image-wide R²={scores[k]:.3f}"
    )

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
lim = np.abs(original).max()
panels = [("Grayscale image (z-scored)", original)] + [
    (f"K = {k}; R² = {scores[k]:.3f}", reconstructions[k]) for k in reconstructions
]
for ax, (title, image) in zip(axes, panels, strict=True):
    ax.imshow(image, cmap="gray", vmin=-lim, vmax=lim)
    ax.set(title=title, xlabel="Column (pixel)", ylabel="Row (pixel)")
fig.tight_layout()
plt.show()
