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
     - One subject per NIfTI or image file, recursively; subfolders define groups
   * - nibabel image (for example from nilearn)
     - One subject
   * - ``(X, Y, Z, T)``, ``(X, Y, Z)`` or ``(H, W)`` numpy array, no ``coords``
     - Image grid; coordinates are voxel indices times ``spacing``
   * - ``(V, T)`` numpy array plus ``coords=(V, D)``
     - Voxel-by-sample data at the given coordinates
   * - ``.png``, ``.jpg``, ``.tif`` or other image file
     - One subject with one sample (TIFF pages become samples)
   * - List of any of the above
     - Folders expand recursively; each file, array, or image is one subject.
       ``coords`` may be a matching list.

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
- Subfolders are searched recursively. Hidden directories and directory symlinks
  are skipped; empty subfolders contribute no groups. Each subject's
  ``group_path`` records the folder names relative to the supplied root.
  The root itself is the global template, not an extra group.
- Voxel coordinates are world coordinates computed from each file's affine, so
  they are in mm for standard NIfTI files.
- Subjects can have different numbers of voxels and of time points.
  They must share coordinate dimensionality; for
  example, mixing 2D images with NIfTI raises ``ValueError``.
- ``stack=True`` treats every 3D volume in the folder as one sample of a single
  subject's time series. Stacked volumes must share shape and affine;
  explicit coordinate arrays must also have identical row order. Inputs with
  and without world-coordinate affines cannot be stacked together. Stacking
  combines every selected file, including nested folders, and clears group
  metadata; it does not fit separate conditions.

Folder-derived hierarchies
--------------------------

A nested folder tree supplies group templates automatically. This illustrative
layout groups independent participants by acquisition site, then by condition:

.. code-block:: text

   participants/
       site-a/
           control/
               subject-01.nii.gz
           treatment/
               subject-02.nii
       site-b/
           control/
               subject-03.nii.gz

.. code-block:: python

   subjects = htfa.load("participants")
   print(subjects[0].group_path)  # ('site-a', 'control')
   res = htfa.fit("participants", K=5, random_state=0)
   print(list(res.group_centers))
   # ['site-a', 'site-b', ('site-a', 'control'),
   #  ('site-a', 'treatment'), ('site-b', 'control')]

Nodes are identified by the full folder path, so the two ``control`` folders
are distinct groups. Files directly inside a group folder attach to that group;
files directly inside the supplied root attach to the global template. This
also supports trees with unequal depths without adding artificial groups.
A single loaded subject still fits TFA, while retaining its ``group_path``.

The inferred hierarchy uses ``template_update="fixed"`` by default. Set
``template_update="empirical"`` to estimate hierarchy variances from the data;
``"brainiak"`` does not support group templates and raises an error when groups
are inferred. Pass explicit ``groups=`` to override the folder-derived groups,
or ``groups=False`` to fit the same files with only a global template. Explicit
label sequences retain the estimator's requirement to set a compatible
``template_update``. To construct an arbitrary tree directly, pass a
:class:`htfa.groups.GroupTree` made with ``GroupTree.from_paths`` as ``groups``.

A list of folders includes each supplied folder's name as its first group
level. A list of individual files has no inferred grouping. If different roots
have the same name, explicit groups disambiguate them. Files are not recognized
as independent participants from their names: repeated runs or conditions for
the same people need an appropriate analysis design, not separate independent
subject entries. ``subject_ids=`` can detect repeated identifiers in a grouped
fit.

Group fitting and plotting support unequal depths and repeated leaf names.
Group permutation tests require equally deep paths and distinct labels for
groups under different parents, because their level-wise shuffling treats
subgroups as exchangeability units. To run such a test, select equally deep group paths
and pass explicit distinct labels (for example ``site-a-control`` and
``site-b-control``). The test rejects unsupported trees rather than assigning
an invalid null distribution. Choosing labels does not establish independence
or exchangeability of the underlying observations.

BIDS datasets
-------------

The Brain Imaging Data Structure (BIDS) organizes imaging files in a nested
folder tree. Its structural folders (such as ``func`` and subject directories)
are not automatically experimental groups. Pass a selected list of paths
rather than the dataset root to avoid treating those folders as a hierarchy.
For example,
this pattern selects resting-state runs (``task-rest``); change the pattern
for your task and preprocessed files:

.. code-block:: python

   from pathlib import Path

   import htfa

   runs = sorted(Path("bids_root").glob("sub-*/func/*_task-rest_bold.nii.gz"))
   res = htfa.fit([str(p) for p in runs], K=50, random_state=0)

Each selected file is treated as a separate subject. If a participant has
multiple runs, combine those runs appropriately before an analysis that assumes
independent subjects, such as a group permutation test. For anatomical
interpretation across participants, coordinates must refer to a shared space;
equal coordinate dimensionality alone does not establish anatomical alignment.

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
``ValueError``. An image-based mask must match both the image shape and affine
(with tolerance for header rounding). The loader does not implicitly resample
a mismatched mask; resample it onto the data grid explicitly.

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
:func:`htfa.make_tfa_data` and :func:`htfa.make_htfa_data` use their generating
scale; fit them with ``standardize=False`` when comparing recovered parameters
with the known truth. :meth:`htfa.HTFAResults.reconstruct` returns values in
the space used for fitting: standardized values when standardization is enabled,
and the input scale otherwise. Complex-valued data and coordinates are rejected
rather than converted silently to real values.
