"""Hierarchical Topographic Factor Analysis (HTFA)."""

import os
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from htfa._engines import select_engine, thread_limit
from htfa._kernels import Prior, max_sigma
from htfa._matching import match_factors
from htfa.tfa import TFA

Array = NDArray[np.float64]

TEMPLATE_MAX_ITER = 100  # max_iter of the unpenalized per-subject TFA fits


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
    threshold : float
        Convergence threshold on the scaled template change
        ``max(max|dC| / length, max|dw| / w)``, as in `TFA`.
    nlss_method, nlss_loss, x_scale, weight_method, upper_ratio, lower_ratio,
    voxel_ratio, tr_ratio, max_voxels, max_trs, standardize, backend, init,
    n_screen_iter
        Passed to each subject's `TFA` (``init`` and ``n_screen_iter`` only matter
        for the template fits, the only ones with several candidate starts).
    n_jobs : int, optional
        Number of joblib workers for the subject fits (numpy engines only).
    random_state : int, optional
        Seed for the template subject choice and per-subject fits.
    verbose : bool
        Print per-iteration progress.

    Attributes
    ----------
    global_centers_ : ndarray, shape (K, D)
    global_widths_ : ndarray, shape (K,)
    global_centers_cov_ : ndarray, shape (K, D, D)
        Posterior covariance of the global centers.
    global_widths_var_ : ndarray, shape (K,)
        Posterior variance of the global widths.
    centers_, widths_, weights_ : list of ndarray
        Per-subject (K, D), (K,) and (K, T_s) estimates, in template order.
    n_iter_ : int
    converged_ : bool
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
        threshold: float = 1e-3,
        nlss_method: str = "bounded-lm",
        nlss_loss: str = "soft_l1",
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
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.K = K
        self.max_global_iter = max_global_iter
        self.max_local_iter = max_local_iter
        self.threshold = threshold
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
        n, n_cpu = len(Xs), os.cpu_count() or 1
        if not parallel:
            return [
                _fit_subject(
                    params, Xs[s], Rs[s], prior, starts[s], int(seeds[s]), n_cpu, quiet
                )
                for s in range(n)
            ]
        n_workers = min(n, effective_n_jobs(self.n_jobs))
        n_threads = max(1, n_cpu // n_workers)
        fits: list[TFA] = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_fit_subject)(
                params, Xs[s], Rs[s], prior, starts[s], int(seeds[s]), n_threads, quiet
            )
            for s in range(n)
        )
        return fits

    def fit(self, X: list, coords: list) -> "HTFA":
        """Fit the global template and per-subject factors.

        Parameters
        ----------
        X : list of array_like, each shape (V_s, T_s)
            Per-subject data.
        coords : list of array_like, each shape (V_s, D)
            Per-subject voxel coordinates (same D for all subjects).

        Returns
        -------
        HTFA
            The fitted estimator.

        Raises
        ------
        ValueError
            On empty or mismatched lists, mixed dimensionalities or
            ``max_global_iter < 1``.

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
        n_subj, K = len(Xs), self.K
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
        free = self._fit_subjects(
            Xs,
            Rs,
            TEMPLATE_MAX_ITER,
            None,
            none,
            rng.integers(seed_max, size=n_subj),
            parallel,
        )
        template, R_t = free[t], Rs[t]
        centers_cov = np.cov(R_t.T) * K ** (-2.0 / 3.0)
        widths_var = max_sigma(R_t)
        g_cov_scaled = centers_cov / n_subj
        g_var_scaled = widths_var / n_subj
        C = template.centers_.copy()
        w = template.widths_.copy()
        C_cov = np.tile(centers_cov, (K, 1, 1))
        w_var = np.full(K, widths_var)
        length = np.sqrt(widths_var / 2.0)

        self.converged_ = False
        fits: list[TFA] = []
        n_iter = 0
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
                # The template subject's own start is the template: skip it.
                others = [s for s in range(n_subj) if s != t]
                from_own = self._fit_subjects(
                    [Xs[s] for s in others],
                    [Rs[s] for s in others],
                    fit_local,
                    prior,
                    [(free[s].centers_, free[s].widths_) for s in others],
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
            new_C, new_C_cov = np.empty_like(C), np.empty_like(C_cov)
            for k in range(K):
                new_C[k], new_C_cov[k] = map_update(
                    C[k], C_cov[k], g_cov_scaled, local_C[:, k]
                )
            common = 1.0 / (w_var + g_var_scaled)
            new_w = w_var * common * local_w.mean(axis=0) + common * g_var_scaled * w
            new_w_var = w_var * common * g_var_scaled
            new_C, new_w, new_C_cov, new_w_var = assign_posterior(
                C, new_C, new_w, new_C_cov, new_w_var
            )
            change = max(
                float(np.max(np.abs(new_C - C))) / length,
                float(np.max(np.abs(new_w - w) / w)),
            )
            C, w, C_cov, w_var = new_C, new_w, new_C_cov, new_w_var
            if self.verbose:
                print(f"HTFA global iteration {n_iter}: scaled change {change:.6g}")
            if change <= self.threshold:
                self.converged_ = True
                break
        self.n_iter_ = n_iter
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
