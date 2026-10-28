# Task 19: kernel shapes (Part 1 of docs/superpowers/specs/2026-10-28-kernels-hierarchy-release-design.md)

Design as built (after the user's later decision: n_components for every kernel, no 'mixture' kernel):
- `htfa._kernels.Kernel(name in rbf/laplace/mexican_hat, n_components=1)`.
- Factor k = sum_m pi_km phi(||r - c_km||^2 / w_km); pi = softmax(0, a_k2..a_kM)
  (first logit fixed: identifiable against the time-series weights; weights sum to 1, phi(0)=1).
- x = [component centers (K M, D), component widths (K M), logits (K, M-1)];
  M = 1 gives [centers, widths] and the old code paths (`Kernel.plain_rbf`; tfa.py calls the
  kernel functions with the old arguments so tests/test_fit_equivalence.py's oracle swap works).
- laplace: exp(eps - sqrt(z + eps^2)), eps = 0.1 relative to the width (departs from the spec's
  "relative to the voxel size": keeps phi a function of d/w, no stored constant per fit).
- mexican_hat: (1 - 2z/D) exp(-z), the D-dim negative LoG (integral 0). The 2D Ricker form
  (1 - z) e^-z in 3D had the default starts reach the optimum in 3/6 datasets vs 5/6.
- Prior gains logits/logits_var; centers/widths per component. HTFA MIXING_LOGIT_VAR = 4.
- Mixture start: components at the factor's start center, widths 0.25^m times, equal weights.

Evidence
- scripts/rbf_fingerprint.py: 341 arrays bitwise equal to main (main.npz in scratchpad t19).
- tests/test_kernel_shapes.py (FD), tests/test_shape_equivalence.py + reference_shapes.py
  (sympy dense oracle, 240 scenarios, 4135 comparisons), tests/test_shape_fits.py (recovery,
  n_jobs, HTFA, effective nodes, results, htfa.fit). Linux docker (python:3.12-slim,
  scipy-openblas 0.3.34, aarch64): 250 shape + fit-equivalence tests pass.
- -m brainiak (venv-brainiak with PYTHONPATH=worktree): 27 passed.
- Tutorial 12: 20 s on M2 Max.

Status: all commits made (Task 19: ...), not pushed. Expect a rebase with Part 2 (htfa.py).
