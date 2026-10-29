Quick start
===========

HTFA represents spatial data with a small set of localized factors and their
weights over samples. TFA fits one subject; HTFA fits several subjects with a
shared spatial template. Samples can be time points, trials, or image channels.

Install
-------

Install htfa with the plotting dependencies:

.. code-block:: bash

   pip install "htfa[plot]"

See :doc:`installation` for other optional dependencies and platform details.

Fit a small, reproducible example
---------------------------------

This simulation contains three subjects, three spatial factors, and 40 samples
per subject. It runs without downloading a dataset. Each data matrix has shape
``(locations, samples)``; each coordinate matrix has shape
``(locations, spatial dimensions)``.

.. code-block:: python

   import htfa
   import matplotlib.pyplot as plt
   from htfa.stats import reconstruction_statistics

   data, coords, truth = htfa.make_htfa_data(
       n_subjects=3, grid_shape=(24, 24), n_samples=40,
       K=3, noise=0.1, random_state=0,
   )
   result = htfa.fit(
       data, coords=coords, K=3, standardize=False,
       max_global_iter=30, random_state=0,
   )

``K`` is the number of factors. ``standardize=False`` preserves this simulation's
generating scale. For fMRI, the default standardizes each voxel's time series
before fitting. ``random_state`` controls random initialization; convergence and
reconstruction should still be checked for the chosen factor count.

Inspect the fit
---------------

.. code-block:: python

   print(result.global_centers.shape)  # (3, 2): factors by spatial dimensions
   print(result.weights[0].shape)      # (3, 40): factor weights for subject 0
   print(result.model.converged_reason_)
   print(reconstruction_statistics(result).r2)
   reconstruction = result.reconstruct(0)  # (576, 40)

The reconstruction is the product of the spatial factors and their sample
weights. The reported R² measures in-sample reconstruction relative to each
location's mean over samples; it does not measure prediction on unseen data.
The results object retains the loaded subjects and fitted estimator for further
inspection.

Plot the results
----------------

.. code-block:: python

   result.plot_nodes()                 # shared template and subject centers
   result.plot_connectivity(subject=0) # correlations between factor weights
   result.plot_reconstruction(subject=0, sample=0)
   plt.show()

These synthetic coordinates describe a 2D grid, not brain anatomy. For registered
brain images, :meth:`htfa.HTFAResults.plot_network` draws a BrainNet-style network,
and :meth:`htfa.HTFAResults.animate_network` displays changing connectivity.
The :doc:`gallery <auto_examples/index>` shows examples of both anatomical and
non-anatomical visualization.

Use your own images
-------------------

Pass one 4D NIfTI file per subject, or a folder containing those files. Replace
the example paths and factor count with values appropriate for your data:

.. code-block:: python

   single = htfa.fit("data/subject-01.nii.gz", K=20, random_state=0)
   multiple = htfa.fit("data/participants", K=20, random_state=0)
   reconstructed_image = multiple.to_nifti(subject=0)
   reconstructed_image.to_filename("subject-01-reconstruction.nii.gz")

A single file fits TFA; several files fit HTFA. NIfTI coordinates use the image
affine, so images must share a meaningful spatial coordinate system. Loading
does not register images. A sequence of 3D frames belonging to one subject can
be loaded with ``stack=True``; separate runs of the same person should not be
treated as independent subjects.

Nested folders define group templates by default. For example, the illustrative
layout below groups independent participants by acquisition site and then by
condition. The folder names supply the group names:

.. code-block:: text

   participants/
       site-a/
           control/
               subject-01.nii.gz
               subject-02.nii.gz
           treatment/
               subject-03.nii.gz
       site-b/
           control/
               subject-04.nii.gz

Pass ``groups=False`` to ignore folder-derived groups, or supply explicit
``groups`` to define the design yourself. Fitting supports unequal folder depths
and repeated names under different parents. Group permutation tests require a
hierarchy with equally deep group paths and distinct subgroup labels; see :doc:`inputs` before
interpreting differences between groups.

Subjects need not have the same number of samples. Analyses that compare
matching time points, such as inter-subject functional correlation, do require
aligned samples. See :doc:`inputs` for masks, explicit group labels, arrays,
images, and folder handling.

Choose a spatial model
----------------------

The default is one Gaussian kernel per factor. ``kernel='laplace'`` and
``kernel='mexican_hat'`` choose alternative shapes; ``n_components=2`` gives each
factor two spatial lobes sharing the same sample weights:

.. code-block:: python

   mixture = htfa.fit(
       data, coords=coords, K=3, kernel="rbf", n_components=2,
       standardize=False, max_global_iter=30, random_state=0,
   )

See :doc:`tutorials/12_kernel_shapes` for factor geometry,
:doc:`tutorials/14_fit_statistics` for complexity-adjusted comparisons, and
:doc:`tutorials/15_real_fmri_kernel_comparison` for a worked fMRI comparison.
:doc:`performance` explains CPU and GPU execution options.

Continue with :doc:`tutorials/01_quickstart` for a complete NIfTI walkthrough,
:doc:`tutorials` for worked analyses, or :doc:`api` for the function reference.
