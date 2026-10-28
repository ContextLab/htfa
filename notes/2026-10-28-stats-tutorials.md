# Real-data kernel comparison and reconstruction statistics

Integration worktree: /private/tmp/htfa-integration.

User requested a real multi-subject kernel/mixture comparison with R²,
reconstructed frames and complexity-adjusted scores, plus a stats API/tutorial.

Implementation: htfa.stats.gaussian_fit_statistics, parameter_count,
reconstruction_statistics and immutable FitStatistics. Counts subject centers,
widths, independent mixture weights, temporal coefficients, one pooled residual
variance. Explicitly CONDITIONAL on hierarchy templates/dispersion; not marginal
HTFA evidence or effective degrees of freedom. AIC/AICc/BIC are descriptive
Gaussian plug-in scores; residual dependence, regularization and singular
mixtures invalidate usual regular-model evidence interpretation. No invented
WAIC without posterior draws. R² uses per-voxel temporal means and pools RSS/TSS.

Tutorial 14 demonstrates formulas, identical predictions at different parameter
budgets, and scoring a fitted mixture HTFA. Tutorial 15 fits 3 development_fmri
adults, K=15, 168 frames, all three kernels x M=1,2, seeds 0,1. Select lowest
full-mask RSS per candidate; same preprocessing, masks, data throughout. Fit
max_voxels=3000, max_global_iter=25, fixed template priors, n_jobs=3. Reports
start/convergence diagnostics and pooled R²/criteria, compares frames 30/80/130
at the axial slice nearest MNI z=0 with shared observed-data scaling.

Sources: statsmodels eval_measures formulas; Watanabe 2013 JMLR singular-model
BIC discussion, linked in tutorial 14. Comparison is in-sample, not held-out.
No data/model assets uploaded; no branch push or publication.


Completed: both notebooks executed and final figures inspected. Final real-data
winner: two-component RBF, R² 0.057273 vs single RBF 0.054125; lowest AIC/AICc/BIC.
12 starts finish at plateau/threshold within 13 global iterations; not proof of
global optima. First 10-iteration trial was extended, final outputs use 25.
Full suite: 999 passed, 16 skipped, 28 deselected (412.13s), log
/private/tmp/htfa-stats-full-tests.log. One additional unequal-subject pooling
test was added after collection; focused stats suite: 10 passed, log
/private/tmp/htfa-stats-focused-tests.log. Mypy/lint pass. Strict Sphinx passed
before final notebook outputs; final rebuild recorded separately below.
Final strict Sphinx rebuild passed including both executed notebooks:
/private/tmp/htfa-stats-docs.log, HTML /private/tmp/htfa-stats-docs.
All lint/format/type checks passed. Figure artifacts are
 docs/_static/tutorials/15_comparison_4.png (scores) and
 docs/_static/tutorials/15_comparison_6.png (frames).
