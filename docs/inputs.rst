Inputs
======

:func:`htfa.fit` and :func:`htfa.load` accept the inputs below. :func:`htfa.fit`
loads its input with :func:`htfa.load` and then fits :class:`htfa.HTFA` when there
are several subjects, or :class:`htfa.TFA` when there is one. :func:`htfa.load`
returns a list of :class:`htfa.Subject` objects, each holding a ``(V, T)`` data
matrix and the ``(V, D)`` coordinates of its ``V`` locations.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Input
     - Treated as
   * - Path to a ``.nii`` / ``.nii.gz`` file
     - One subject
   * - Path to a folder
     - One subject per NIfTI or image file (see below)
   * - nibabel image (for example from nilearn)
     - One subject
   * - ``(X, Y, Z, T)``, ``(X, Y, Z)`` or ``(H, W)`` numpy array, no ``coords``
     - Image grid; coordinates are voxel indices times ``spacing``
   * - ``(V, T)`` numpy array plus ``coords=(V, D)``
     - Voxel-by-sample data at the given coordinates
   * - ``.png``, ``.jpg``, ``.tif`` or other image file
     - One subject with one sample (TIFF pages become samples)
   * - List of any of the above
     - One subject per element; ``coords`` may be a matching list

2D images need the ``images`` extra and nilearn masks need the ``nilearn`` extra;
see :doc:`optional_dependencies`.

Folders of NIfTI files
----------------------

.. code-block:: python

   import htfa

   res = htfa.fit("path/to/folder", K=50, random_state=0)

- Each ``.nii`` or ``.nii.gz`` file is one subject. A 4D file is that subject's
  time series. A 3D file is one subject with a single sample.
- Files are taken in natural filename order (``sub-2`` before ``sub-10``). The
  extension check ignores case. Dot files and other files (``.json`` sidecars,
  ``.tsv``) are skipped.
- The folder is not searched recursively.
- Voxel coordinates are world coordinates computed from each file's affine, so
  they are in mm for standard NIfTI files.
- Subjects can have different numbers of voxels and of time points.
  ``ValueError`` is raised only when they differ in coordinate dimensionality, for
  example 2D images mixed with NIfTI.
- ``stack=True`` treats every 3D volume in the folder as one sample of a single
  subject's time series.

BIDS datasets
-------------

A BIDS dataset is a nested folder tree, so pass a list of paths:

.. code-block:: python

   from pathlib import Path

   import htfa

   runs = sorted(Path("bids_root").glob("sub-*/func/*_task-rest_bold.nii.gz"))
   res = htfa.fit([str(p) for p in runs], K=50, random_state=0)

Masks
-----

With the default ``mask=None``, a voxel of a NIfTI image, a 3D/4D array or a
``(V, T)`` array is kept when it is finite at every time point and not zero at
every time point (zero everywhere marks out-of-brain locations). For 2D images
(PNG, JPEG, TIFF, ... files and bare 2D arrays without ``coords``) every finite
pixel is kept, because black pixels are data there; pass ``mask=image > 0`` to
drop them. ``mask=`` chooses voxels explicitly:

- a path to a NIfTI mask file, or a nibabel image (nonzero voxels are kept);
- a fitted nilearn ``NiftiMasker`` (its ``mask_img_`` is used);
- a boolean numpy array with the shape of the image grid;
- ``'epi'`` or ``'background'``: nilearn's automatic masks, computed for each
  subject (NIfTI input only).

If a NaN or infinite value falls inside an explicit mask, :func:`htfa.load` raises
``ValueError``.

Arrays
------

Without ``coords``, a numpy array is an image grid, ``(H, W)``, ``(X, Y, Z)`` or
``(X, Y, Z, T)``. Coordinates are the grid indices of the kept voxels times
``spacing`` (a scalar, or one entry per axis; default 1).

With ``coords``, a 2D array is ``(V, T)`` and ``coords`` is ``(V, D)``. To fit
several subjects, pass lists of arrays and a matching list of coordinate arrays:

.. code-block:: python

   res = htfa.fit(Xs, K=5, coords=Rs, standardize=False)

nibabel and nilearn images
--------------------------

A nibabel image, such as one returned by nilearn, is one subject. Its affine gives
the coordinates, and :meth:`htfa.HTFAResults.to_nifti` returns images on the same
grid.

2D images
---------

A ``.png``, ``.jpg``, ``.tif`` (or ``.bmp``, ``.gif``) file is one subject with one
sample per image (TIFF pages are separate samples). Color images are converted to
ITU-R 601 grayscale with ``channels="gray"`` (default), or ``channels="separate"``
makes each channel a sample. A 2D numpy array of shape ``(H, W)`` is read the same
way. Several images of the same size can be fitted as the samples of one subject
with ``stack=True``.

Standardization
---------------

The estimators z-score each location over samples by default
(``standardize='auto'``, applied when ``T > 1``). Keep this on for fMRI: without it
the fit is dominated by anatomical intensity differences. Synthetic data from
:func:`htfa.make_tfa_data` and :func:`htfa.make_htfa_data` are already on a
z-scored scale; fit them with ``standardize=False``.
:meth:`htfa.HTFAResults.reconstruct` returns values in the standardized space.
