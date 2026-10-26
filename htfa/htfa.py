"""Hierarchical Topographic Factor Analysis (HTFA)."""

import os
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

from htfa._engines import select_engine, thread_limit
from htfa._kernels import Prior, max_sigma
from htfa._matching import match_factors
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

    def trimmed(values: Array) -> Array:
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

    cov = trimmed(centers)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, float(np.min(np.linalg.eigvalsh(floor_cov))))
    var = float(trimmed(widths[:, :, None])[0, 0])
    return (vecs * vals) @ vecs.T, max(var, floor_var)


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
    start: tuple[Array, Array] | None,
    seed: int,
    n_threads: int,
    quiet: bool,
) -> TFA:
    # Filters are set here so that they also apply inside loky workers.
    with (
        threadpool_limits(n_threads),
        thread_limit(n_threads),
        warnings.catch_warnings(),
    ):
        if quiet:
            warnings.simplefilter("ignore", ConvergenceWarning)
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
    n_screen_iter
        Passed to each subject's `TFA` (``init`` and ``n_screen_iter`` only matter
        for the template fits, the only ones with several candidate starts).
    n_jobs : int, optional
        Number of joblib workers for the subject fits (numpy engines only).
    n_threads : int, optional
        BLAS/OpenMP (and NumPy engine) threads per subject fit. ``None`` uses the
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
    global_widths_ : ndarray, shape (K,)
    global_centers_cov_ : ndarray, shape (K, D, D)
        Posterior covariance of the global centers (``'fixed'`` and
        ``'empirical'``: ``Sigma / n_subjects``, that of a mean of
        ``n_subjects`` draws).
    global_widths_var_ : ndarray, shape (K,)
        Posterior variance of the global widths (likewise).
    prior_centers_cov_ : ndarray, shape (D, D)
        Subject-level center covariance of the last local fits' prior
        (``'brainiak'``: that of factor 0).
    prior_widths_var_ : float
        Subject-level width variance of the last local fits' prior.
    centers_, widths_, weights_ : list of ndarray
        Per-subject (K, D), (K,) and (K, T_s) estimates, in template order.
    free_fits_ : list of (ndarray, ndarray)
        Per-subject (centers (K, D), widths (K,)) of the unpenalized stage-0
        fits, or the ``starts`` passed to `fit`.
    template_subject_ : int
        Index of the subject whose free fit is the initial template (when no
        ``init_template`` is passed).
    template_history_ : list of (ndarray, ndarray)
        Global (centers, widths): the initial template, then the template after
        each global iteration (``n_iter_ + 1`` entries).
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
        n_jobs: int | None = None,
        n_threads: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
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
        self.n_jobs = n_jobs
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose

    def _tfa_params(self, max_iter: int) -> dict[str, Any]:
        names = (
            "K nlss_method nlss_loss x_scale weight_method upper_ratio lower_ratio "
            "voxel_ratio tr_ratio max_voxels max_trs standardize backend init "
            "n_screen_iter verbose"
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
        starts: Sequence[tuple[Array, Array] | None],
        seeds: NDArray[np.int64],
        parallel: bool,
    ) -> list[TFA]:
        """Fit every subject's TFA; seeds are fixed per subject, not per worker.

        Local (penalized) fits run with ``quiet=True``: their per-fit
        ConvergenceWarnings are suppressed in favour of one HTFA-level warning.
        """
        params = self._tfa_params(max_iter)
        quiet = prior is not None
        n = len(Xs)
        n_workers = min(n, effective_n_jobs(self.n_jobs)) if parallel else 1
        n_threads = self._subject_threads(n_workers)
        if not parallel:
            return [
                _fit_subject(
                    params,
                    Xs[s],
                    Rs[s],
                    prior,
                    starts[s],
                    int(seeds[s]),
                    n_threads,
                    quiet,
                )
                for s in range(n)
            ]
        fits: list[TFA] = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_fit_subject)(
                params, Xs[s], Rs[s], prior, starts[s], int(seeds[s]), n_threads, quiet
            )
            for s in range(n)
        )
        return fits

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
            ``free_fits_``.
        init_template : tuple of (centers, widths), optional
            Initial global template, shapes (K, D) and (K,), in place of the
            template subject's stage-0 fit. To resume a fit, pass
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

        def checked(
            pair: tuple[ArrayLike, ArrayLike], name: str
        ) -> tuple[Array, Array]:
            C, w = np.asarray(pair[0], np.float64), np.asarray(pair[1], np.float64)
            if C.shape != (K, D) or w.shape != (K,):
                raise ValueError(
                    f"{name} must be (centers ({K}, {D}), widths ({K},)), got "
                    f"{C.shape} and {w.shape}"
                )
            return C, w

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
        none: list[tuple[Array, Array] | None] = [None] * n_subj

        # Unpenalized multi-start TFA per subject; the template is the fit of a
        # random subject among those with the most samples.
        T_max = max(x.shape[1] for x in Xs)
        t = int(rng.choice([s for s in range(n_subj) if Xs[s].shape[1] == T_max]))
        free_seeds = rng.integers(seed_max, size=n_subj)  # drawn either way
        if given is None:
            fitted = self._fit_subjects(
                Xs, Rs, self.template_max_iter, None, none, free_seeds, parallel
            )
            given = [(f.centers_, f.widths_) for f in fitted]
        self.free_fits_, self.template_subject_ = given, t
        R_t = Rs[t]
        centers_cov = np.cov(R_t.T) * K ** (-2.0 / 3.0)
        widths_var = max_sigma(R_t)
        g_cov_scaled = centers_cov / n_subj
        g_var_scaled = widths_var / n_subj
        C, w = given[t] if template_start is None else template_start
        C, w = C.copy(), w.copy()
        self.template_history_ = [(C, w)]
        brainiak = self.template_update == "brainiak"
        if self.template_update == "empirical":
            matched = [match_factors(C, g[0]) for g in given]
            centers_cov, widths_var = empirical_dispersion(
                np.stack([g[0][p] for g, p in zip(given, matched, strict=True)]),
                np.stack([g[1][p] for g, p in zip(given, matched, strict=True)]),
                0.01 * centers_cov,
                0.01 * widths_var,
            )
        C_cov = np.tile(centers_cov, (K, 1, 1))
        w_var = np.full(K, widths_var)
        length = np.sqrt(max_sigma(R_t) / 2.0)

        reason = "max_iter"
        fits: list[TFA] = []
        objectives: list[float] = []
        best, stale, n_iter = np.inf, 0, 0
        while n_iter < self.max_global_iter:
            n_iter += 1
            prior = Prior(C, w, C_cov, w_var, 1.0)
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
                warm = [(f.centers_, f.widths_) for f in fits]
                fits = self._fit_subjects(
                    Xs, Rs, fit_local, prior, warm, seeds, parallel
                )
            local_C = np.empty((n_subj, K, C.shape[1]))
            local_w = np.empty((n_subj, K))
            for s, f in enumerate(fits):
                perm = match_factors(C, f.centers_)
                local_C[s], local_w[s] = f.centers_[perm], f.widths_[perm]
                f.centers_, f.widths_ = local_C[s], local_w[s]
                f.weights_ = f.weights_[perm]
            if brainiak:
                new_C, new_C_cov = np.empty_like(C), np.empty_like(C_cov)
                for k in range(K):
                    new_C[k], new_C_cov[k] = map_update(
                        C[k], C_cov[k], g_cov_scaled, local_C[:, k]
                    )
                common = 1.0 / (w_var + g_var_scaled)
                new_w = (
                    w_var * common * local_w.mean(axis=0) + common * g_var_scaled * w
                )
                new_w_var = w_var * common * g_var_scaled
                new_C, new_w, new_C_cov, new_w_var = assign_posterior(
                    C, new_C, new_w, new_C_cov, new_w_var
                )
            else:
                # Fixed subject-level prior: the template minimizing the summed
                # penalized objectives is the mean of the local estimates.
                new_C, new_w = local_C.mean(axis=0), local_w.mean(axis=0)
                new_C_cov, new_w_var = C_cov, w_var
            change = max(
                float(np.max(np.abs(new_C - C))) / length,
                float(np.max(np.abs(new_w - w) / w)),
            )
            self.prior_centers_cov_, self.prior_widths_var_ = C_cov[0], float(w_var[0])
            C, w, C_cov, w_var = new_C, new_w, new_C_cov, new_w_var
            self.template_history_.append((C, w))
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
        if not self.converged_:
            warnings.warn(
                f"HTFA did not converge in max_global_iter={self.max_global_iter} "
                "iterations; increase max_global_iter or threshold",
                ConvergenceWarning,
                stacklevel=2,
            )
        self.global_centers_, self.global_widths_ = C, w
        self.global_centers_cov_, self.global_widths_var_ = C_cov, w_var
        self.centers_ = [f.centers_ for f in fits]
        self.widths_ = [f.widths_ for f in fits]
        self.weights_ = [f.weights_ for f in fits]
        self.engine_ = fits[0].engine_
        return self
