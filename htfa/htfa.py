"""Hierarchical Topographic Factor Analysis (HTFA)."""

import os
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from joblib import effective_n_jobs
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

from htfa._engines import (
    one_blas_thread,
    one_blas_thread_here,
    select_engine,
    thread_limit,
)
from htfa._kernels import MIXING_LOGIT_BOUND, Kernel, Prior, max_sigma
from htfa._matching import match_components, match_factors
from htfa.tfa import TFA

Array = NDArray[np.float64]

TEMPLATE_UPDATES = ("brainiak", "fixed", "empirical")
# Global stopping: besides the scaled template change reaching ``threshold``,
# stop when it has not fallen below PLATEAU_RATIO times its lowest value so far
# in PLATEAU_PATIENCE consecutive global iterations ('plateau'; a plain "no new
# minimum" rule never fires on a slow steady decrease such as the real-data one
# below, which reached a new minimum almost every iteration), or when the summed final
# local objectives changed by less than OBJECTIVE_RTOL (relative) in each of the
# last OBJECTIVE_PATIENCE iterations ('objective'). Measured: on real data
# (2014 DFFR, 23 subjects, K=100, 'brainiak') the change went 0.0039, 0.0036,
# 0.0035, 0.0035, 0.0034 (each step < 10% lower: a plateau); on
# make_htfa_data(3, (12, 12), 40, K=3) with 'brainiak' it fell by about 0.85x per
# 3 iterations from iteration 13 on while the summed objective kept rising by
# 5e-4 to 9e-4 per iteration (the prior tightens every iteration), and with
# 'fixed' the summed objective changed by 7.7e-4, then < 1e-5.
PLATEAU_RATIO = 0.9
PLATEAU_PATIENCE = 3
OBJECTIVE_RTOL = 1e-5
OBJECTIVE_PATIENCE = 2
# 'empirical': (s, k) deviations from the across-subject median whose squared
# Mahalanobis length is above this quantile are treated as factor mismatches
# and left out of the dispersion estimate (TRIM_STEPS reweighting passes).
DISPERSION_QUANTILE = 0.9
TRIM_STEPS = 3
# Subject-level prior variance of the mixing logits log(pi_m / pi_1) of a
# factor with several components (the initial value; 'brainiak' shrinks it
# like the width variance): a standard deviation of 2, i.e. a factor of
# about 7.4 in the weight ratio.
MIXING_LOGIT_VAR = 4.0


def map_update(
    prior_mean: ArrayLike,
    prior_cov: ArrayLike,
    global_cov_scaled: ArrayLike,
    observations: ArrayLike,
) -> tuple[Array, Array]:
    """Gaussian MAP update of a template mean from per-subject observations.

    BrainIAK's ``_map_update`` with ``np.linalg.solve`` in place of ``inv``::

        A = prior_cov + global_cov_scaled
        post_mean = prior_cov A^-1 mean(obs) + global_cov_scaled A^-1 prior_mean
        post_cov = prior_cov A^-1 global_cov_scaled

    Parameters
    ----------
    prior_mean : array_like, shape (D,)
        Prior mean of the template parameter.
    prior_cov : array_like, shape (D, D)
        Prior covariance of the template mean.
    global_cov_scaled : array_like, shape (D, D)
        Subject-level covariance divided by the number of subjects.
    observations : array_like, shape (n_subj, D)
        One observation per subject.

    Returns
    -------
    post_mean : ndarray, shape (D,)
    post_cov : ndarray, shape (D, D)

    Examples
    --------
    >>> import numpy as np
    >>> mean, cov = map_update([0.0], [[1.0]], [[1.0]], [[2.0], [2.0]])
    >>> mean, cov
    (array([1.]), array([[0.5]]))
    """
    m0 = np.asarray(prior_mean, dtype=np.float64)
    S0 = np.asarray(prior_cov, dtype=np.float64)
    G = np.asarray(global_cov_scaled, dtype=np.float64)
    obs_mean = np.mean(np.asarray(observations, dtype=np.float64), axis=0)
    A = S0 + G
    post_mean = S0 @ np.linalg.solve(A, obs_mean) + G @ np.linalg.solve(A, m0)
    post_cov = S0 @ np.linalg.solve(A, G)
    return post_mean, 0.5 * (post_cov + post_cov.T)


def assign_posterior(
    prior_centers: Array,
    centers: Array,
    widths: Array,
    centers_cov: Array,
    widths_var: Array,
) -> tuple[Array, Array, Array, Array]:
    """Reorder a global posterior to match the prior's factor order.

    Unlike BrainIAK's ``_assign_posterior``, the permutation is applied to all four
    fields (centers, widths, centers_cov, widths_var) together.
    """
    perm = match_factors(prior_centers, centers)
    return centers[perm], widths[perm], centers_cov[perm], widths_var[perm]


def empirical_dispersion(
    centers: Array, widths: Array, floor_cov: Array, floor_var: float
) -> tuple[Array, float]:
    """Pooled across-subject covariance (D, D) of matched centers and variance of
    matched widths, ``centers`` (n_subj, K, D) and ``widths`` (n_subj, K).

    Deviations from the per-factor median over subjects are pooled over
    factors (with the ``n / (n - 1)`` correction). `TRIM_STEPS` times, the
    deviations whose squared Mahalanobis length under the current estimate is
    above its `DISPERSION_QUANTILE` quantile are left out as likely factor
    mismatches, and the trimmed estimate is divided by its expectation under
    Gaussian deviations, ``F_{D+2}(q) / F_D(q)`` with ``F_m`` the chi-square
    distribution function with m degrees of freedom and ``q`` its
    `DISPERSION_QUANTILE` quantile (D = 1 for the widths). The covariance is
    floored at ``floor_cov`` (its eigenvalues are raised to at least
    ``floor_cov``'s smallest) and the variance at ``floor_var``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> c = rng.normal(0.0, 2.0, (40, 30, 3))
    >>> w = 10.0 + rng.normal(0.0, 0.5, (40, 30))
    >>> cov, var = empirical_dispersion(c, w, 1e-3 * np.eye(3), 1e-6)
    >>> np.round(np.diag(cov), 1), round(var, 2)
    (array([3.9, 4. , 4.2]), 0.26)
    """
    n = centers.shape[0]
    if n < 2:
        raise ValueError("template_update='empirical' needs at least 2 subjects")

    cov = _trimmed_cov(centers)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, float(np.min(np.linalg.eigvalsh(floor_cov))))
    var = float(_trimmed_cov(widths[:, :, None])[0, 0])
    return (vecs * vals) @ vecs.T, max(var, floor_var)


def _trimmed_cov(values: Array) -> Array:
    """Trimmed pooled covariance (m, m) of the deviations of ``values``
    (n_subj, K, m) from their median over subjects (see
    `empirical_dispersion`)."""
    n = values.shape[0]
    dev = (values - np.median(values, axis=0)).reshape(-1, values.shape[-1])
    m = dev.shape[1]
    chi_q = chi2.ppf(DISPERSION_QUANTILE, m)
    consistency = chi2.cdf(chi_q, m + 2) / chi2.cdf(chi_q, m)
    cov = dev.T @ dev / len(dev)
    for _ in range(TRIM_STEPS):
        d2 = np.einsum("ij,ij->i", dev @ np.linalg.pinv(cov), dev)
        keep = dev[d2 <= np.quantile(d2, DISPERSION_QUANTILE)]
        cov = keep.T @ keep / len(keep) * n / (n - 1) / consistency
    return cov


def _fit_params(f: TFA, M: int) -> tuple[Array, ...]:
    """A subject fit as a `TFA` start: (centers, widths), or (component
    centers, component widths, mixing weights) with M > 1 components."""
    if M == 1:
        return f.centers_, f.widths_
    return f.component_centers_, f.component_widths_, f.mixing_weights_


def _as_x(params: Sequence[Array], kernel: Kernel) -> tuple[Array, Array, Array]:
    """Component centers (K M, D), widths (K M,) and logits (K, M - 1) of a
    `_fit_params` tuple."""
    if kernel.n_components == 1:
        C, w = params
        return C, w, np.zeros((len(w), 0))
    C, w, pi = params
    return C.reshape(-1, C.shape[-1]), np.ravel(w), Kernel.logits(pi)


def _from_x(C: Array, w: Array, a: Array, kernel: Kernel) -> tuple[Array, ...]:
    """Inverse of `_as_x`."""
    M = kernel.n_components
    if M == 1:
        return C, w
    K = len(a)
    return C.reshape(K, M, -1), w.reshape(K, M), Kernel.mixing(a)


def _factor_centers(C: Array, w: Array, a: Array, kernel: Kernel) -> Array:
    """Factor centers (K, D) for matching: the centers (one component) or the
    mixing-weighted component centers."""
    if kernel.n_components == 1:
        return C
    K = len(a)
    return np.einsum(
        "km,kmd->kd", Kernel.mixing(a), C.reshape(K, kernel.n_components, -1)
    )


def _match_params(
    C: Array, w: Array, a: Array, params: Sequence[Array], kernel: Kernel
) -> tuple[NDArray[np.intp], NDArray[np.intp] | None, tuple[Array, ...]]:
    """Order of a fit (`_fit_params`) matched to the template (``C``, ``w``,
    ``a``): factors by their centers (`_factor_centers`), then, with several
    components, each factor's components by their centers
    (`htfa._matching.match_components`). Returns the factor permutation, the
    per-factor component permutations (K, M) (None for one component) and
    the reordered parameters."""
    M = kernel.n_components
    if M == 1:
        perm = match_factors(C, params[0])
        return perm, None, (params[0][perm], params[1][perm])
    K = len(a)
    perm = match_factors(
        _factor_centers(C, w, a, kernel),
        _factor_centers(*_as_x(params, kernel), kernel),
    )
    Cc, wc, pi = (np.asarray(p)[perm] for p in params)
    comp = match_components(C.reshape(K, M, -1), Cc)
    rows = np.arange(K)[:, None]
    pi = pi[rows, comp]
    # Reordering changes the reference (first) component of the logits; with
    # M >= 3 they can then leave the solver's box, where TFA would clip its
    # start. Clip them here, so that template and starts stay feasible.
    a_new = Kernel.logits(pi)
    if np.any(np.abs(a_new) > MIXING_LOGIT_BOUND):
        pi = Kernel.mixing(np.clip(a_new, -MIXING_LOGIT_BOUND, MIXING_LOGIT_BOUND))
    return perm, comp, (Cc[rows, comp], wc[rows, comp], pi)


def _match_fit(
    C: Array, w: Array, a: Array, f: TFA, kernel: Kernel
) -> tuple[Array, Array, Array]:
    """Reorder fit ``f`` in place to the template order of `_match_params`;
    returns its component centers, widths and logits (`_as_x`)."""
    M = kernel.n_components
    perm, _, matched = _match_params(C, w, a, _fit_params(f, M), kernel)
    f.weights_ = f.weights_[perm]
    if M == 1:
        f.centers_, f.widths_ = matched
        f.component_centers_ = f.centers_[:, None].copy()
        f.component_widths_ = f.widths_[:, None].copy()
    else:
        f.centers_, f.widths_ = f.centers_[perm], f.widths_[perm]
        f.component_centers_, f.component_widths_, f.mixing_weights_ = matched
    return _as_x(matched, kernel)


def _available_threads() -> int:
    """Threads the caller allows: the lowest thread count of the loaded BLAS
    libraries (a limit from ``OPENBLAS_NUM_THREADS``, ``OMP_NUM_THREADS``, ... or
    an enclosing ``threadpoolctl.threadpool_limits``, else their defaults), or of
    the OpenMP libraries when no BLAS library is visible (macOS Accelerate);
    ``os.cpu_count()`` when neither is.

    BLAS is where the subject fits spend their threads. OpenMP pools are not
    trusted when BLAS is visible: torch on Windows loads the serial stub
    ``libiompstubs5md.dll``, which always reports 1 thread whatever the limit.
    """
    pools = threadpool_info()
    for api in ("blas", "openmp"):
        counts = [int(p["num_threads"]) for p in pools if p["user_api"] == api]
        if counts:
            return min(counts)
    return os.cpu_count() or 1


def _fit_subject(
    params: dict[str, Any],
    X: Array,
    R: Array,
    prior: Prior | None,
    start: tuple[Array, ...] | None,
    seed: int,
    n_threads: int,
) -> TFA:
    """One subject's TFA fit, on the calling thread or a worker thread.

    Sets per-thread state only: Accelerate's threading mode (one thread,
    `htfa._engines.one_blas_thread_here`) and the NumPy engine's
    ``n_threads`` chunk threads (`htfa._engines.thread_limit`). The
    process-wide settings, the threadpoolctl BLAS limit (one thread) and the
    warning filters, are set once by `HTFA._fit_subjects` around all fits,
    since threads entering and leaving them concurrently would race. Inside
    the fit, sklearn's K-means also enters a process-wide one-thread BLAS limit
    of its own; that limit equals the outer one, so its exit restores the
    same value.
    """
    with one_blas_thread_here(), thread_limit(n_threads):
        return TFA(**params, random_state=seed).fit(X, R, prior=prior, start=start)


class HTFA(BaseEstimator):
    """Hierarchical Topographic Factor Analysis.

    Fits per-subject TFA models whose centers and widths are tied to a global
    template through Gaussian priors, alternating local fits with MAP updates of
    the template (Manning et al., 2014; BrainIAK's ``HTFA``).

    Parameters
    ----------
    K : int
        Number of factors.
    max_global_iter : int
        Maximum number of template updates.
    max_local_iter : int
        ``max_iter`` of each subject's TFA per global iteration.
    template_max_iter : int, default=100
        ``max_iter`` of the unpenalized per-subject TFA fits (stage 0) that
        give the initial template and the subjects' own starts.
    threshold : float
        Convergence threshold on the scaled template change
        ``max(max|dC| / length, max|dw| / w)``, as in `TFA`. The global loop
        also stops when that change has not fallen below 0.9 times its lowest
        value so far in 3 consecutive iterations ("plateau": a rate of
        decrease slower than 10% per iteration counts as a plateau, so a slow
        drift of the template stops the loop), or when the summed final
        local objectives (``cost_`` of the subject fits) changed by less than a
        relative 1e-5 in each of the last 2 iterations ("objective").
    template_update : {'brainiak', 'fixed', 'empirical'}, default='brainiak'
        How the template and the subjects' prior are updated after each round
        of local fits. ``'brainiak'``: BrainIAK's Gaussian MAP update; its
        posterior (mean and covariance) is the next round's prior, so the
        subject-level covariance shrinks as ``Sigma / (1 + i n)`` over
        iterations. ``'fixed'``: the subject-level prior covariance stays at its
        initial value ``Sigma = cov(R) K^(-2/3)`` (widths: ``max_sigma(R)``) and
        the template is the mean of the matched local estimates (the minimizer
        of the summed penalized objectives over the template).
        ``'empirical'``: like ``'fixed'``, with ``Sigma`` and the width
        variance estimated once from the across-subject dispersion of the
        matched stage-0 fits (`empirical_dispersion`, floored at 1% of the
        initial values).
    nlss_method, nlss_loss, x_scale, weight_method, upper_ratio, lower_ratio,
    voxel_ratio, tr_ratio, max_voxels, max_trs, standardize, backend, init,
    n_screen_iter, kernel, n_components
        Passed to each subject's `TFA` (``init`` and ``n_screen_iter`` only matter
        for the template fits, the only ones with several candidate starts).
        With ``n_components = M > 1`` the template holds every component's
        center and width, each with the prior and update of a factor's (the
        center covariance and width variance per component), and the mixing
        logits ``log(pi_m / pi_1)`` (K, M - 1), each with an independent
        Gaussian prior of initial variance 4 (`MIXING_LOGIT_VAR`), updated as
        the widths are (``'brainiak'``: the same MAP update; ``'fixed'``: the
        mean; ``'empirical'``: variance from the matched stage-0 fits, floored
        at 1% of 4). Subject fits are matched to the template by their
        mixing-weighted centers, then each factor's components by their
        centers. The scaled change also covers the logits, unscaled.
    mode : {'fast', 'exact'}, default='fast'
        Passed to each subject's `TFA` (see there). On the twelve
        development_fmri adults (K = 50, defaults, 10 global iterations) the
        fast fit took 233 s against 628 s, and its template is a different
        map: its centers are 15.8 mm (median; 42.3 mm at most) from the exact
        template's, while a 1e-12 change of the data moved the exact template
        by 2.0 to 5.6 mm (median; 8.8 to 13.6 at most) in three trials. The
        template subject's stage-0 fit is 16.4 mm (median) from the exact one
        in fast mode and was unchanged by the perturbation in exact mode. The
        summed objective was 9.5e-4 lower than the exact fit's (the perturbed
        exact fits: -6.0e-4 to +1.1e-3), and the held-out R^2 of fits to the
        training segments 0.04078 against 0.04127. Run on to the stopping
        rule (14 to 17 global iterations) the templates were 15.8 mm apart
        (median), a perturbed exact fit 2.3 mm from the exact one
        (docs/validation.rst). Pass ``mode='exact'`` for the reference
        template.
    n_jobs : int, optional
        Number of subject fits run at once (numpy engines only), as threads of
        this process (values as in joblib, e.g. -1 for all CPUs). The results
        do not depend on it: all fits share one process, and so one BLAS
        library state, and every subject fit, serial or parallel, runs BLAS on
        one thread
        (`htfa._engines.one_blas_thread`), since BLAS rounds differently with
        different thread counts. On macOS, a standalone `htfa.TFA` fit and the
        stage-0 fit of the same subject can therefore differ at rounding level.
        Threads cannot be stopped from outside: after an error in one subject's
        fit or a ``KeyboardInterrupt`` (Ctrl-C), the fits not yet started are
        cancelled and those already running finish before control returns.
    n_threads : int, optional
        NumPy engine threads per subject fit (the engine splits each pass into
        fixed voxel chunks, so the count does not change results; BLAS runs on
        one thread, see ``n_jobs``). ``None`` uses the
        lowest thread count the loaded BLAS libraries currently report (the
        OpenMP libraries when no BLAS library is visible to
        ``threadpoolctl.threadpool_info``): a limit set by ``OMP_NUM_THREADS``,
        ``OPENBLAS_NUM_THREADS`` and the like or by an enclosing
        ``threadpoolctl.threadpool_limits``, else the libraries' own defaults
        (usually the number of cores), divided among the ``n_jobs`` workers and
        at least 1 per worker. ``n_jobs`` workers times ``n_threads`` can
        therefore exceed the caller's limit when ``n_threads`` is given, or when
        there are more workers than allowed threads.
    random_state : int, optional
        Seed for the template subject choice and per-subject fits.
    verbose : bool
        Print per-iteration progress.

    Attributes
    ----------
    global_centers_ : ndarray, shape (K, D)
        (With ``n_components > 1``: mixing-weighted means, as in `TFA`.)
    global_widths_ : ndarray, shape (K,)
    global_centers_cov_ : ndarray, shape (K, D, D)
        Posterior covariance of the global centers (``'fixed'`` and
        ``'empirical'``: ``Sigma / n_subjects``, that of a mean of
        ``n_subjects`` draws). Shape (K, M, D, D), per component, with
        ``n_components = M > 1``.
    global_widths_var_ : ndarray, shape (K,)
        Posterior variance of the global widths (likewise); (K, M) with
        M > 1.
    global_component_centers_, global_component_widths_, global_mixing_weights_
        Template components, shapes (K, M, D), (K, M) and (K, M) (one
        component: the template centers and widths, and weights 1).
    global_logits_var_ : ndarray, shape (K, M - 1)
        Posterior variance of the template's mixing logits.
    prior_centers_cov_ : ndarray, shape (D, D)
        Subject-level center covariance of the last local fits' prior
        (``'brainiak'``: that of factor 0).
    prior_widths_var_ : float
        Subject-level width variance of the last local fits' prior.
    centers_, widths_, weights_ : list of ndarray
        Per-subject (K, D), (K,) and (K, T_s) estimates, in template order.
    component_centers_, component_widths_, mixing_weights_ : list of ndarray
        Per-subject components (K, M, D), (K, M) and (K, M), in template
        order (components too).
    free_fits_ : list of tuple of ndarray
        Per-subject (centers (K, D), widths (K,)) of the unpenalized stage-0
        fits, or the ``starts`` passed to `fit`; with M > 1 components
        (component centers (K, M, D), widths (K, M), mixing weights (K, M)).
    template_subject_ : int
        Index of the subject whose free fit is the initial template (when no
        ``init_template`` is passed).
    template_history_ : list of tuple of ndarray
        Global (centers, widths) (with M > 1: component centers, widths and
        mixing weights): the initial template, then the template after each
        global iteration (``n_iter_ + 1`` entries).
    n_iter_ : int
    converged_ : bool
        Whether the global loop stopped before ``max_global_iter``.
    converged_reason_ : {'threshold', 'plateau', 'objective', 'max_iter'}
        Why the global loop stopped (see ``threshold``).
    objective_history_ : list of float
        Summed final local objectives after each global iteration.
    engine_ : str

    Examples
    --------
    >>> from htfa.synthetic import make_htfa_data
    >>> Xs, Rs, _ = make_htfa_data(3, (10, 10), 20, K=3, random_state=0)
    >>> model = HTFA(
    ...     K=3, max_global_iter=2, max_local_iter=3, init="power",
    ...     standardize=False, random_state=0,
    ... )
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")  # 2 global iterations: not converged
    ...     model = model.fit(Xs, Rs)
    >>> model.global_centers_.shape
    (3, 2)
    """

    def __init__(
        self,
        K: int = 50,
        max_global_iter: int = 10,
        max_local_iter: int = 10,
        template_max_iter: int = 100,
        threshold: float = 1e-3,
        template_update: str = "brainiak",
        nlss_method: str = "auto",
        nlss_loss: str = "linear",
        x_scale: Any = "jac",
        weight_method: str = "rr",
        upper_ratio: float = 1.8,
        lower_ratio: float = 0.02,
        voxel_ratio: float = 1.0,
        tr_ratio: float = 1.0,
        max_voxels: int | None = None,
        max_trs: int | None = None,
        standardize: str | bool = "auto",
        init: str | Sequence[str] = "auto",
        n_screen_iter: int = 0,
        backend: str = "auto",
        mode: str = "fast",
        n_jobs: int | None = None,
        n_threads: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        kernel: str = "rbf",
        n_components: int = 1,
    ) -> None:
        self.K = K
        self.max_global_iter = max_global_iter
        self.max_local_iter = max_local_iter
        self.template_max_iter = template_max_iter
        self.threshold = threshold
        self.template_update = template_update
        self.nlss_method = nlss_method
        self.nlss_loss = nlss_loss
        self.x_scale = x_scale
        self.weight_method = weight_method
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio
        self.voxel_ratio = voxel_ratio
        self.tr_ratio = tr_ratio
        self.max_voxels = max_voxels
        self.max_trs = max_trs
        self.standardize = standardize
        self.init = init
        self.n_screen_iter = n_screen_iter
        self.backend = backend
        self.mode = mode
        self.n_jobs = n_jobs
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose
        self.kernel = kernel
        self.n_components = n_components

    def _tfa_params(self, max_iter: int) -> dict[str, Any]:
        names = (
            "K nlss_method nlss_loss x_scale weight_method upper_ratio lower_ratio "
            "voxel_ratio tr_ratio max_voxels max_trs standardize backend mode init "
            "n_screen_iter verbose kernel n_components"
        ).split()
        params = {n: getattr(self, n) for n in names}
        params["max_iter"] = max_iter
        params["threshold"] = self.threshold
        return params

    def _subject_threads(self, n_workers: int) -> int:
        """Threads per subject fit with ``n_workers`` concurrent fits."""
        if self.n_threads is not None:
            if not isinstance(self.n_threads, (int, np.integer)) or self.n_threads < 1:
                raise ValueError(
                    f"n_threads must be a positive integer or None, "
                    f"got {self.n_threads!r}"
                )
            return int(self.n_threads)
        return max(1, _available_threads() // n_workers)

    def _fit_subjects(
        self,
        Xs: list[Array],
        Rs: list[Array],
        max_iter: int,
        prior: Prior | None,
        starts: Sequence[tuple[Array, ...] | None],
        seeds: NDArray[np.int64],
        parallel: bool,
    ) -> list[TFA]:
        """Fit every subject's TFA; seeds are fixed per subject, not per worker.

        Local (penalized) fits (``prior`` given) have their per-fit
        ConvergenceWarnings suppressed in favour of one HTFA-level warning.
        """
        params = self._tfa_params(max_iter)
        n = len(Xs)
        n_workers = min(n, effective_n_jobs(self.n_jobs)) if parallel else 1
        n_threads = self._subject_threads(n_workers)
        jobs = [
            (params, Xs[s], Rs[s], prior, starts[s], int(seeds[s]), n_threads)
            for s in range(n)
        ]
        # Worker threads in this process, not worker processes: every fit then
        # uses the same BLAS state. On Windows with some AMD EPYC CPUs, OpenBLAS
        # rounds differently in about 0.5-2.5% of freshly started processes, so
        # process workers made results depend on n_jobs there.
        with (
            threadpool_limits(n_threads),
            one_blas_thread(),
            warnings.catch_warnings(),
        ):
            if prior is not None:
                warnings.simplefilter("ignore", ConvergenceWarning)
            if n_workers == 1:
                return [_fit_subject(*job) for job in jobs]
            # On an error or interrupt, queued fits are cancelled and the running
            # ones finish before the limits and filters above are restored:
            # a fit still running afterwards would leave its BLAS limits behind.
            pool = ThreadPoolExecutor(n_workers)
            try:
                futures = [pool.submit(_fit_subject, *job) for job in jobs]
                return [future.result() for future in futures]
            finally:
                pool.shutdown(wait=True, cancel_futures=True)

    def fit(
        self,
        X: list,
        coords: list,
        starts: Sequence[tuple[ArrayLike, ArrayLike]] | None = None,
        init_template: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> "HTFA":
        """Fit the global template and per-subject factors.

        Parameters
        ----------
        X : list of array_like, each shape (V_s, T_s)
            Per-subject data.
        coords : list of array_like, each shape (V_s, D)
            Per-subject voxel coordinates (same D for all subjects).
        starts : list of (centers, widths), optional
            One (K, D), (K,) pair per subject, used in place of the unpenalized
            stage-0 fits (which are then skipped), e.g. a previous model's
            ``free_fits_``. With ``n_components = M > 1``: (component centers
            (K, M, D), widths (K, M), mixing weights (K, M)) per subject.
        init_template : tuple of (centers, widths), optional
            Initial global template, shapes (K, D) and (K,) (or the three
            arrays of ``starts`` with M > 1), in place of the template
            subject's stage-0 fit. To resume a fit, pass
            ``starts=list(zip(model.centers_, model.widths_))`` and
            ``init_template=(model.global_centers_, model.global_widths_)``.
            This approximates continuing the fit rather than reproducing it:
            the template (co)variances restart from their initial values and
            the first global iteration's two-start step (from the template and
            from each subject's own start) is run again.

        Returns
        -------
        HTFA
            The fitted estimator.

        Raises
        ------
        ValueError
            On empty or mismatched lists, mixed dimensionalities,
            ``max_global_iter < 1`` or ``starts``/``init_template`` of the wrong
            length or shapes.

        Warns
        -----
        ConvergenceWarning
            Once, when the template has not converged after ``max_global_iter``
            iterations (per-subject local warnings are suppressed).

        Examples
        --------
        >>> import warnings
        >>> from htfa.synthetic import make_htfa_data
        >>> Xs, Rs, _ = make_htfa_data(2, (8, 8), 10, K=2, random_state=1)
        >>> model = HTFA(K=2, max_global_iter=1, max_local_iter=2, init="power",
        ...              random_state=0)
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")  # one global iteration only
        ...     model = model.fit(Xs, Rs)
        >>> len(model.weights_), model.weights_[0].shape
        (2, (2, 10))
        """
        if self.template_update not in TEMPLATE_UPDATES:
            raise ValueError(
                f"template_update must be one of {TEMPLATE_UPDATES}, "
                f"got {self.template_update!r}"
            )
        if self.max_global_iter < 1:
            raise ValueError(
                f"max_global_iter must be >= 1, got {self.max_global_iter}"
            )
        if len(X) != len(coords) or len(X) == 0:
            raise ValueError(
                f"X and coords must be non-empty lists of equal length, "
                f"got {len(X)} and {len(coords)}"
            )
        Xs = [np.asarray(x, dtype=np.float64) for x in X]
        Rs = [np.asarray(r, dtype=np.float64) for r in coords]
        if len({R.shape[1] for R in Rs}) != 1:
            raise ValueError("all subjects' coords must have the same dimension D")
        n_subj, K, D = len(Xs), self.K, Rs[0].shape[1]
        kernel = Kernel(self.kernel, self.n_components)
        M = kernel.n_components

        def checked(pair: Sequence[ArrayLike], name: str) -> tuple[Array, ...]:
            arrays = tuple(np.asarray(a, np.float64) for a in pair)
            if M == 1:
                expected = [(K, D), (K,)]
                what = f"(centers ({K}, {D}), widths ({K},))"
            else:
                expected = [(K, M, D), (K, M), (K, M)]
                what = (
                    f"(centers ({K}, {M}, {D}), widths ({K}, {M}), "
                    f"mixing_weights ({K}, {M}))"
                )
            if [a.shape for a in arrays] != expected:
                raise ValueError(
                    f"{name} must be {what}, got {[a.shape for a in arrays]}"
                )
            return arrays

        if starts is not None and len(starts) != n_subj:
            raise ValueError(f"starts must have one entry per subject ({n_subj})")
        given = None if starts is None else [checked(p, "starts") for p in starts]
        template_start = (
            None if init_template is None else checked(init_template, "init_template")
        )
        rng = np.random.default_rng(self.random_state)
        seed_max = np.iinfo(np.int32).max

        engine = select_engine(
            self.backend,
            max(x.shape[0] for x in Xs),
            max(x.shape[1] for x in Xs),
            K,
        )
        parallel = engine.name == "numpy" and self.n_jobs not in (None, 1)
        none: list[tuple[Array, ...] | None] = [None] * n_subj

        # Unpenalized multi-start TFA per subject; the template is the fit of a
        # random subject among those with the most samples.
        T_max = max(x.shape[1] for x in Xs)
        t = int(rng.choice([s for s in range(n_subj) if Xs[s].shape[1] == T_max]))
        free_seeds = rng.integers(seed_max, size=n_subj)  # drawn either way
        if given is None:
            fitted = self._fit_subjects(
                Xs, Rs, self.template_max_iter, None, none, free_seeds, parallel
            )
            given = [_fit_params(f, M) for f in fitted]
        self.free_fits_, self.template_subject_ = given, t
        R_t = Rs[t]
        centers_cov = np.cov(R_t.T) * K ** (-2.0 / 3.0)
        widths_var = max_sigma(R_t)
        logits_var = MIXING_LOGIT_VAR
        g_cov_scaled = centers_cov / n_subj
        g_var_scaled = widths_var / n_subj
        g_logit_scaled = logits_var / n_subj
        C, w, a = _as_x(given[t] if template_start is None else template_start, kernel)
        C, w, a = C.copy(), w.copy(), a.copy()
        self.template_history_ = [_from_x(C, w, a, kernel)]
        brainiak = self.template_update == "brainiak"
        if self.template_update == "empirical":
            local = [_match_params(C, w, a, g, kernel)[2] for g in given]
            local = [_as_x(p, kernel) for p in local]
            centers_cov, widths_var = empirical_dispersion(
                np.stack([p[0] for p in local]),
                np.stack([p[1] for p in local]),
                0.01 * centers_cov,
                0.01 * widths_var,
            )
            if M > 1:
                spread = _trimmed_cov(np.stack([p[2] for p in local])[..., None])
                logits_var = max(float(spread[0, 0]), 0.01 * MIXING_LOGIT_VAR)
        C_cov = np.tile(centers_cov, (K * M, 1, 1))
        w_var = np.full(K * M, widths_var)
        a_var = np.full((K, M - 1), logits_var)
        length = np.sqrt(max_sigma(R_t) / 2.0)

        reason = "max_iter"
        fits: list[TFA] = []
        objectives: list[float] = []
        best, stale, n_iter = np.inf, 0, 0
        while n_iter < self.max_global_iter:
            n_iter += 1
            prior = (
                Prior(C, w, C_cov, w_var, 1.0)
                if M == 1
                else Prior(C, w, C_cov, w_var, 1.0, a, a_var)
            )
            seeds = rng.integers(seed_max, size=n_subj)
            fit_local = self.max_local_iter
            if n_iter == 1:
                # From the template and from each subject's own solution; keep the
                # lower penalized objective.
                from_template = self._fit_subjects(
                    Xs, Rs, fit_local, prior, none, seeds, parallel
                )
                # The template subject's own start is the template: skip it
                # (unless an init_template replaced it).
                others = [
                    s for s in range(n_subj) if s != t or template_start is not None
                ]
                from_own = self._fit_subjects(
                    [Xs[s] for s in others],
                    [Rs[s] for s in others],
                    fit_local,
                    prior,
                    [given[s] for s in others],
                    seeds[others],
                    parallel,
                )
                fits = list(from_template)
                for s, b in zip(others, from_own, strict=True):
                    if b.cost_ < fits[s].cost_:
                        fits[s] = b
            else:
                warm = [_fit_params(f, M) for f in fits]
                fits = self._fit_subjects(
                    Xs, Rs, fit_local, prior, warm, seeds, parallel
                )
            local_C = np.empty((n_subj, K * M, C.shape[1]))
            local_w = np.empty((n_subj, K * M))
            local_a = np.empty((n_subj, K, M - 1))
            for s, f in enumerate(fits):
                local_C[s], local_w[s], local_a[s] = _match_fit(C, w, a, f, kernel)
            if brainiak:
                new_C, new_C_cov = np.empty_like(C), np.empty_like(C_cov)
                for k in range(K * M):
                    new_C[k], new_C_cov[k] = map_update(
                        C[k], C_cov[k], g_cov_scaled, local_C[:, k]
                    )
                common = 1.0 / (w_var + g_var_scaled)
                new_w = (
                    w_var * common * local_w.mean(axis=0) + common * g_var_scaled * w
                )
                new_w_var = w_var * common * g_var_scaled
                common_a = 1.0 / (a_var + g_logit_scaled)
                new_a = (
                    a_var * common_a * local_a.mean(axis=0)
                    + common_a * g_logit_scaled * a
                )
                new_a_var = a_var * common_a * g_logit_scaled
                if M == 1:
                    new_C, new_w, new_C_cov, new_w_var = assign_posterior(
                        C, new_C, new_w, new_C_cov, new_w_var
                    )
                else:
                    # Factor order of the posterior: by the flattened component
                    # centers (each posterior component stays with its own).
                    perm = match_factors(C.reshape(K, -1), new_C.reshape(K, -1))
                    idx = (perm[:, None] * M + np.arange(M)).ravel()
                    new_C, new_C_cov = new_C[idx], new_C_cov[idx]
                    new_w, new_w_var = new_w[idx], new_w_var[idx]
                    new_a, new_a_var = new_a[perm], new_a_var[perm]
            else:
                # Fixed subject-level prior: the template minimizing the summed
                # penalized objectives is the mean of the local estimates.
                new_C, new_w = local_C.mean(axis=0), local_w.mean(axis=0)
                new_a = local_a.mean(axis=0)
                new_C_cov, new_w_var, new_a_var = C_cov, w_var, a_var
            change = max(
                float(np.max(np.abs(new_C - C))) / length,
                float(np.max(np.abs(new_w - w) / w)),
            )
            if M > 1:
                change = max(change, float(np.max(np.abs(new_a - a))))
            self.prior_centers_cov_, self.prior_widths_var_ = C_cov[0], float(w_var[0])
            C, w, C_cov, w_var = new_C, new_w, new_C_cov, new_w_var
            a, a_var = new_a, new_a_var
            self.template_history_.append(_from_x(C, w, a, kernel))
            objectives.append(float(sum(f.cost_ for f in fits)))
            rel = [
                abs(b - a) / abs(a)
                for a, b in zip(objectives[:-1], objectives[1:], strict=True)
            ]
            if self.verbose:
                print(
                    f"HTFA global iteration {n_iter}: scaled change {change:.6g}"
                    + (f", objective change {rel[-1]:.3g}" if rel else ""),
                    flush=True,
                )
            if change < PLATEAU_RATIO * best:
                best, stale = change, 0
            else:
                best, stale = min(best, change), stale + 1
            if change <= self.threshold:
                reason = "threshold"
            elif stale >= PLATEAU_PATIENCE:
                reason = "plateau"
            elif (
                len(rel) >= OBJECTIVE_PATIENCE
                and max(rel[-OBJECTIVE_PATIENCE:]) < OBJECTIVE_RTOL
            ):
                reason = "objective"
            if reason != "max_iter":
                break
        self.n_iter_ = n_iter
        self.converged_reason_ = reason
        self.converged_ = reason != "max_iter"
        self.objective_history_ = objectives
        if not brainiak:
            C_cov, w_var = C_cov / n_subj, w_var / n_subj
            a_var = a_var / n_subj
        if not self.converged_:
            warnings.warn(
                f"HTFA did not converge in max_global_iter={self.max_global_iter} "
                "iterations; increase max_global_iter or threshold",
                ConvergenceWarning,
                stacklevel=2,
            )
        if M == 1:
            self.global_centers_, self.global_widths_ = C, w
            self.global_centers_cov_, self.global_widths_var_ = C_cov, w_var
        else:
            pi = Kernel.mixing(a)
            Cg, wg = C.reshape(K, M, D), w.reshape(K, M)
            self.global_centers_ = np.einsum("km,kmd->kd", pi, Cg)
            self.global_widths_ = np.sum(pi * wg, axis=1)
            self.global_centers_cov_ = C_cov.reshape(K, M, D, D)
            self.global_widths_var_ = np.reshape(w_var, (K, M))
        self.global_component_centers_ = C.reshape(K, M, D)
        self.global_component_widths_ = w.reshape(K, M)
        self.global_mixing_weights_ = Kernel.mixing(a)
        self.global_logits_var_ = a_var
        self.centers_ = [f.centers_ for f in fits]
        self.widths_ = [f.widths_ for f in fits]
        self.weights_ = [f.weights_ for f in fits]
        self.component_centers_ = [f.component_centers_ for f in fits]
        self.component_widths_ = [f.component_widths_ for f in fits]
        self.mixing_weights_ = [f.mixing_weights_ for f in fits]
        self.engine_ = fits[0].engine_
        return self
