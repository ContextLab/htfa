# Task 19: kernel shapes (Part 1 of docs/superpowers/specs/2026-10-28-kernels-hierarchy-release-design.md)

Design as built (after the user's later decision: n_components for every kernel, no 'mixture' kernel):
- `htfa._kernels.Kernel(name in rbf/laplace/mexican_hat, n_components=1)`.
- Factor k = sum_m pi_km phi(||r - c_km||^2 / w_km); pi = softmax(0, a_k2..a_kM).
- x = [component centers (K M, D), component widths (K M), logits (K, M-1)];
  M = 1 gives [centers, widths] and the old code paths (`Kernel.plain_rbf`).
- laplace: exp(eps - sqrt(z + eps^2)), eps = 0.1 (relative to the width; departs from the
  spec's "relative to voxel size": keeps phi a function of d/w, no stored constant).
- mexican_hat: Ricker (1 - z) exp(-z), one parameter (identifiable; no degenerate directions).
- Prior gains logits/logits_var; centers/widths per component.

Progress
- [x] _kernels.py general path + FD tests (tests/test_kernel_shapes.py), commit 5978151
- [x] rbf bitwise: scripts/rbf_fingerprint.py, main fingerprint in scratchpad t19/main.npz
- [ ] tfa.py, htfa.py, synthetic, results/brainnet effective nodes
- [ ] equivalence sweep (sympy dense oracle), recovery, n_jobs, docker
- [ ] tutorial 12, docs, changelog
