"""
Loading different kinds of input
================================

:func:`htfa.load` turns files, folders, nibabel images and numpy arrays into
:class:`htfa.Subject` objects, each holding a ``(V, T)`` data matrix and ``(V, D)``
coordinates. This example builds two small synthetic subjects in a real brain shape:
nilearn's bundled MNI152 template and brain mask (shipped with nilearn, so nothing is
downloaded) are averaged into 8 mm voxels, giving a 24 x 29 x 23 grid with the MNI
affine. Each subject has 10 volumes: the template intensity inside the brain, two
smooth blobs whose amplitudes change over time, and a little noise; voxels outside
the brain are zero. The example writes these as NIfTI files, together with a PNG
image, to a temporary directory, loads them in each supported way and prints what
comes back. It then fits the folder with :func:`htfa.fit`, as a quick check that
loaded data go straight into a fit (10 volumes are far too few for a meaningful
model). Automatic installation of optional packages is switched off with
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
from nilearn import datasets, plotting
from PIL import Image

import htfa

htfa.set_autoinstall(False)

rng = np.random.default_rng(0)

# %%
# A brain-shaped grid from the bundled MNI152 template
# ----------------------------------------------------
# The 2 mm template is averaged over 4 x 4 x 4 blocks of voxels into 8 mm voxels;
# a block counts as brain when most of it lies inside the brain mask.
template = datasets.load_mni152_template(resolution=2)
brain = datasets.load_mni152_brain_mask(resolution=2)
block = 4
crop = tuple(slice(0, n - n % block) for n in template.shape)


def block_mean(img):
    a = np.asarray(img.dataobj, dtype=float)[crop]
    nx, ny, nz = (s // block for s in a.shape)
    return a.reshape(nx, block, ny, block, nz, block).mean(axis=(1, 3, 5))


inside = block_mean(brain) > 0.5
anatomy = block_mean(template)
anatomy /= anatomy[inside].mean()
affine = template.affine @ np.diag([block, block, block, 1.0])
affine[:3, 3] = nib.affines.apply_affine(template.affine, [(block - 1) / 2] * 3)
print("grid:", inside.shape, "brain voxels:", int(inside.sum()))

# %%
# Two subjects with changing activations
# --------------------------------------
ijk = np.indices(inside.shape).reshape(3, -1).T
xyz = nib.affines.apply_affine(affine, ijk).reshape(*inside.shape, 3)
centers = np.array([[-40.0, -20.0, 40.0], [30.0, -70.0, 10.0]])  # mm, MNI
T = 10
vols = []
for subject in range(2):
    v = np.repeat(anatomy[..., None], T, axis=-1)
    for c, phase in zip(centers, (0.0, np.pi / 2), strict=True):
        blob = np.exp(-np.sum((xyz - c) ** 2, axis=-1) / (2 * 12.0**2))
        amplitude = np.cos(2 * np.pi * np.arange(T) / T + phase + subject)
        v += 0.8 * blob[..., None] * amplitude
    v += 0.05 * rng.standard_normal(v.shape)
    v[~inside] = 0.0
    vols.append(v)

# %%
# Loading each kind of input
# --------------------------
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
    print("warnings raised by the fit:", sorted({w.category.__name__ for w in caught}))
    print(
        f"htfa.fit on the folder: {elapsed:.1f} s, subjects fitted: {len(res.subjects)}"
    )
    print("reconstruction as NIfTI:", res.to_nifti(0).shape)

    shown = from_png[0].to_image(from_png[0].data[:, 0])
    first_volume = from_folder[0].to_image(from_folder[0].data[:, 0])

# %%
# The PNG and the first NIfTI volume
# ----------------------------------
# :meth:`htfa.Subject.to_image` puts a column of the data back into image space,
# with NaN outside the loaded voxels; those are set to zero and left blank in the
# plot. The cuts pass through the first blob's centre.
volume = np.nan_to_num(np.asarray(first_volume.dataobj))
fig = plt.figure(figsize=(11, 4))
ax = fig.add_axes((0.06, 0.13, 0.24, 0.72))
ax.imshow(shown, cmap="gray")
ax.set_title("PNG (16 x 20 pixels)")
ax.set_xlabel("column")
ax.set_ylabel("row")
brain_ax = fig.add_axes((0.34, 0.02, 0.65, 0.84))
brain_ax.set_title("NIfTI subject 1, volume 0 (8 mm voxels)")
plotting.plot_img(
    nib.Nifti1Image(volume, affine),
    display_mode="ortho",
    cut_coords=tuple(centers[0]),
    colorbar=True,
    cmap="viridis",
    threshold=1e-3,
    black_bg=False,
    axes=brain_ax,
    figure=fig,
)
plt.show()
