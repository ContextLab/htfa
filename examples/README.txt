Gallery
=======

Tutorials
---------

Each card opens the corresponding tutorial, including its code, figures, data
background, and interpretation. The notebooks use simulated data, ordinary
images, and illustrative fMRI experiments. Each includes an Open in Colab link.

.. container:: tutorial-gallery

   .. container:: tutorial-card

      .. image:: /_static/tutorials/01_quickstart.png
         :target: ../tutorials/01_quickstart.html
         :alt: Quick start with NIfTI files: Fit a shared template, inspect factor weights, and reconstruct synthetic brain images.

      :doc:`Quick start with NIfTI files </tutorials/01_quickstart>`

      Fit a shared template, inspect factor weights, and reconstruct synthetic brain images.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/02_tfa_single_subject.png
         :target: ../tutorials/02_tfa_single_subject.html
         :alt: Single-subject TFA: Recover spatial factors and explore single-subject fitting options.

      :doc:`Single-subject TFA </tutorials/02_tfa_single_subject>`

      Recover spatial factors and explore single-subject fitting options.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/03_htfa_multi_subject.png
         :target: ../tutorials/03_htfa_multi_subject.html
         :alt: Multi-subject HTFA: Compare a shared template with individual subjects and known simulated factors.

      :doc:`Multi-subject HTFA </tutorials/03_htfa_multi_subject>`

      Compare a shared template with individual subjects and known simulated factors.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/04_inputs_nifti_folders_nilearn_arrays.png
         :target: ../tutorials/04_inputs_nifti_folders_nilearn_arrays.html
         :alt: Loading spatial data: Load NIfTI files, nested folders, masks, arrays, and image objects.

      :doc:`Loading spatial data </tutorials/04_inputs_nifti_folders_nilearn_arrays>`

      Load NIfTI files, nested folders, masks, arrays, and image objects.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/05_2d_images.png
         :target: ../tutorials/05_2d_images.html
         :alt: Detailed 2D image reconstruction: Compare factor counts and reconstruct a photograph with localized spatial factors.

      :doc:`Detailed 2D image reconstruction </tutorials/05_2d_images>`

      Compare factor counts and reconstruct a photograph with localized spatial factors.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/06_connectivity_and_dynamics.png
         :target: ../tutorials/06_connectivity_and_dynamics.html
         :alt: Connectivity and dynamics: Visualize static, sliding-window, and inter-subject factor connectivity.

      :doc:`Connectivity and dynamics </tutorials/06_connectivity_and_dynamics>`

      Visualize static, sliding-window, and inter-subject factor connectivity.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/09_word_decoding_mitchell2008.png
         :target: ../tutorials/09_word_decoding_mitchell2008.html
         :alt: Word decoding: Explore fMRI responses to nouns and drawings with an adapted decoding workflow.

      :doc:`Word decoding </tutorials/09_word_decoding_mitchell2008>`

      Explore fMRI responses to nouns and drawings with an adapted decoding workflow.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/10_dynamic_isfc_animation.png
         :target: ../tutorials/10_dynamic_isfc_animation.html
         :alt: Dynamic brain networks: Animate inter-subject factor connectivity during a shared movie.

      :doc:`Dynamic brain networks </tutorials/10_dynamic_isfc_animation>`

      Animate inter-subject factor connectivity during a shared movie.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/11_plotting_and_appearance.png
         :target: ../tutorials/11_plotting_and_appearance.html
         :alt: Plotting and appearance: Customize network geometry, colors, surfaces, and illustrative task annotations.

      :doc:`Plotting and appearance </tutorials/11_plotting_and_appearance>`

      Customize network geometry, colors, surfaces, and illustrative task annotations.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/12_kernel_shapes.png
         :target: ../tutorials/12_kernel_shapes.html
         :alt: Spatial kernels and mixtures: Compare Gaussian, Laplace, and Mexican-hat factors and their component lobes.

      :doc:`Spatial kernels and mixtures </tutorials/12_kernel_shapes>`

      Compare Gaussian, Laplace, and Mexican-hat factors and their component lobes.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/13_group_hierarchy.png
         :target: ../tutorials/13_group_hierarchy.html
         :alt: Nested group templates: Fit child and adult templates and illustrate permutation comparisons.

      :doc:`Nested group templates </tutorials/13_group_hierarchy>`

      Fit child and adult templates and illustrate permutation comparisons.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/14_fit_statistics.png
         :target: ../tutorials/14_fit_statistics.html
         :alt: Reconstruction statistics: Compare R² and complexity penalties for alternative spatial models.

      :doc:`Reconstruction statistics </tutorials/14_fit_statistics>`

      Compare R² and complexity penalties for alternative spatial models.

   .. container:: tutorial-card

      .. image:: /_static/tutorials/15_real_fmri_kernel_comparison.png
         :target: ../tutorials/15_real_fmri_kernel_comparison.html
         :alt: Kernel comparison on fMRI: Compare single-component and mixture models on the same multi-subject dataset.

      :doc:`Kernel comparison on fMRI </tutorials/15_real_fmri_kernel_comparison>`

      Compare single-component and mixture models on the same multi-subject dataset.

API examples
------------

The examples below illustrate individual API operations with small,
self-contained datasets. Most use simulated spatial factors with known
parameters and time courses; the image example uses a photograph. These are
demonstrations of fitting and visualization, with no required experimental
protocol. Each example runs when the documentation is built, and its printed
numbers come from that run.
