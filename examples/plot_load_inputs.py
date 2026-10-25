"""
Loading different kinds of input
================================

:func:`htfa.load` turns files, folders, nibabel images and numpy arrays into
:class:`htfa.Subject` objects, each holding a ``(V, T)`` data matrix and ``(V, D)``
coordinates. This example writes tiny synthetic NIfTI files (two subjects, 8 x 8 x 8
voxels, 10 volumes) and a PNG image to a temporary directory, loads them in each
supported way and prints what comes back. It then fits the folder with
:func:`htfa.fit`. Automatic installation of optional packages is switched off with
:func:`htfa.set_autoinstall` so that the build never calls pip. The printed numbers
come from the code below.
"""

import tempfile
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image

import htfa

htfa.set_autoinstall(False)

rng = np.random.default_rng(0)
affine = np.diag([3.0, 3.0, 3.0, 1.0])  # 3 mm voxels
vols = [rng.random((8, 8, 8, 10)) + 1 for _ in range(2)]

with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp) / "nifti"
    folder.mkdir()
    paths = []
    for i, v in enumerate(vols):
        p = folder / f"sub-{i + 1}.nii.gz"
        nib.save(nib.Nifti1Image(v, affine), p)
        paths.append(str(p))
    png = Path(tmp) / "image.png"
    pixels = (1 + 254 * rng.random((16, 20))).astype(np.uint8)
    Image.fromarray(pixels).save(png)

    from_folder = htfa.load(str(folder))
    from_list = htfa.load(paths)
    from_nib = htfa.load(nib.load(paths[0]))
    from_array = htfa.load(vols[0])
    from_coords = htfa.load(rng.random((30, 10)), coords=rng.random((30, 2)) * 10)
    from_png = htfa.load(str(png))

    sources = {
        "folder of NIfTI files": from_folder,
        "list of paths": from_list,
        "nibabel image": from_nib,
        "(X, Y, Z, T) array": from_array,
        "(V, T) array + coords": from_coords,
        "PNG image": from_png,
    }
    for name, subjects in sources.items():
        s = subjects[0]
        print(
            f"{name:24s}: {len(subjects)} subject(s), data {s.data.shape}, "
            f"coords {s.coords.shape}, affine: {s.affine is not None}"
        )
    print("first coordinate of the NIfTI subject (mm):", from_folder[0].coords[0])
    print(
        "first coordinate of the array subject (voxel index):", from_array[0].coords[0]
    )
    print(
        "isinstance(subject, htfa.Subject):", isinstance(from_folder[0], htfa.Subject)
    )

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = htfa.fit(
            str(folder), K=2, standardize=False, max_global_iter=10, random_state=0
        )
    elapsed = time.perf_counter() - start
    print(
        "warnings raised by the fit (noise data, so no real structure):",
        sorted({w.category.__name__ for w in caught}),
    )
    print(
        f"htfa.fit on the folder: {elapsed:.1f} s, subjects fitted: {len(res.subjects)}"
    )
    print("reconstruction as NIfTI:", res.to_nifti(0).shape)

    shown = from_png[0].to_image(from_png[0].data[:, 0])
    first_volume = from_folder[0].to_image(from_folder[0].data[:, 0])
    first_volume = np.asarray(first_volume.dataobj)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(shown, cmap="gray")
axes[0].set_title("PNG loaded as a 16 x 20 image")
axes[0].set_xlabel("column")
axes[0].set_ylabel("row")
im = axes[1].imshow(first_volume[:, :, 4], cmap="viridis")
axes[1].set_title("NIfTI subject 1, volume 0, slice z = 4")
axes[1].set_xlabel("x voxel")
axes[1].set_ylabel("y voxel")
fig.colorbar(im, ax=axes[1], label="intensity")
fig.tight_layout()
plt.show()
