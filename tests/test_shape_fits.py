"""Whole fits with each kernel shape and with mixtures (``kernel``,
``n_components``): recovery, n_jobs independence, results and display.

Fits are compared on the machine that runs them only: a case counts as
well-conditioned when the default fit ends at least as low as a fit started
from the true parameters (relative `START_RTOL`), i.e. the default starts
found the optimum the data define; recovery is checked on those cases, and a
minimum number of cases must qualify.
"""

import itertools
import warnings

import numpy as np
import pytest

from htfa import HTFAResults, load
from htfa._kernels import KERNELS, Kernel, ridge_weights
from htfa._matching import match_factors
from htfa.brainnet import effective_nodes
from htfa.htfa import HTFA
from htfa.synthetic import make_htfa_data, make_tfa_data
from htfa.tfa import TFA, component_factors

START_RTOL = 1e-8
# One component (make_tfa_data, noise 0.1, T = 100, K = 3; 24x24 and 18x18x18
# grids, seeds 0-5). Measured on the qualifying cases: centers <= 0.0055
# lengths, widths <= 5.3% (laplace; rbf 2.8%, mexican_hat 1.8%). In 3D the
# default starts missed the optimum of one mexican_hat case of six.
CENTER_LENGTHS = 0.02
WIDTH_RTOL = 0.1
MIN_QUALIFYING = 5
# Two well-separated components (`_mixture_case`), measured: component
# centers <= 0.0096 lengths, widths <= 10.1% (laplace), mixing weights
# <= 0.011.
COMPONENT_LENGTHS = 0.03
COMPONENT_WIDTH_RTOL = 0.2
MIXING_ATOL = 0.03


def _length(R):
    return float(np.sqrt(np.max(np.std(R, axis=0)) ** 2))


def _fit(X, R, K, name, M, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TFA(
            K=K, kernel=name, n_components=M, standardize=False, random_state=0
        ).fit(X, R, **kwargs)


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("grid", [(24, 24), (18, 18, 18)], ids=["2D", "3D"])
def test_tfa_recovers_each_kernel(name, grid):
    """make_tfa_data from each kernel, fitted with the same kernel: on every
    well-conditioned case of six, centers within CENTER_LENGTHS lengths and
    widths within WIDTH_RTOL; at least MIN_QUALIFYING qualify."""
    rows = []
    for seed in range(6):
        X, R, truth = make_tfa_data(
            grid, 100, K=3, noise=0.1, random_state=seed, kernel=name
        )
        m = _fit(X, R, 3, name, 1)
        ref = _fit(X, R, 3, name, 1, start=(truth["centers"], truth["widths"]))
        qualifies = m.cost_ <= ref.cost_ * (1 + START_RTOL)
        p = match_factors(truth["centers"], m.centers_)
        c_err = np.max(np.linalg.norm(m.centers_[p] - truth["centers"], axis=1))
        w_err = np.max(np.abs(m.widths_[p] / truth["widths"] - 1))
        rows.append((qualifies, c_err / _length(R), w_err))
        print(f"{name} {grid} seed {seed}: qualifies {qualifies}, center "
              f"{c_err / _length(R):.4f} lengths, width {w_err:.4f}")  # fmt: skip
    good = [r for r in rows if r[0]]
    assert len(good) >= MIN_QUALIFYING
    assert max(r[1] for r in good) < CENTER_LENGTHS
    assert max(r[2] for r in good) < WIDTH_RTOL


def _mixture_case(D, name, seed):
    """Two factors of two components each: the second component 1.5 sigma
    from the first, 0.4 times its width; weights (0.6, 0.4) and (0.35,
    0.65); noise 0.05, T = 150."""
    rng = np.random.default_rng(seed)
    n = 30 if D == 2 else 16
    ax = np.arange(float(n))
    R = np.stack(np.meshgrid(*[ax] * D, indexing="ij"), -1).reshape(-1, D)
    base = np.array([[0.3] * D, [0.7] * D]) * (n - 1) + rng.normal(0, 1, (2, D))
    w0 = rng.uniform(0.012, 0.02, 2) * (n - 1) ** 2
    dirs = rng.normal(size=(2, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    C = np.stack([base, base + 1.5 * np.sqrt(w0 / 2)[:, None] * dirs], 1)
    w = np.stack([w0, 0.4 * w0], 1)
    pi = np.array([[0.6, 0.4], [0.35, 0.65]])
    W = rng.normal(size=(2, 150))
    X = component_factors(R, C, w, pi, Kernel(name, 2)) @ W
    return X + 0.05 * rng.normal(size=X.shape), R, C, w, pi


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("D", [2, 3])
def test_tfa_recovers_mixture_components(name, D):
    """Well-separated two-component mixtures: the default fit reaches the
    optimum of a fit from the truth, and recovers every component's center,
    width and mixing weight (up to the order of a factor's components)."""
    for seed in range(3):
        X, R, C, w, pi = _mixture_case(D, name, seed)
        m = _fit(X, R, 2, name, 2)
        ref = _fit(X, R, 2, name, 2, start=(C, w, pi))
        p = match_factors(np.einsum("km,kmd->kd", pi, C), m.centers_)
        errs = []
        for k, j in enumerate(p):
            errs.append(
                min(
                    (
                        np.max(
                            np.linalg.norm(m.component_centers_[j][o] - C[k], axis=1)
                        ),
                        np.max(np.abs(m.component_widths_[j][o] / w[k] - 1)),
                        np.max(np.abs(m.mixing_weights_[j][o] - pi[k])),
                    )
                    for o in map(list, itertools.permutations(range(2)))
                )
            )
        c_err, w_err, pi_err = np.max(np.array(errs), axis=0)
        print(f"{name} D={D} seed {seed}: cost {m.cost_:.6f} vs truth start "
              f"{ref.cost_:.6f}; components {c_err / _length(R):.4f} lengths, "
              f"widths {w_err:.4f}, mixing {pi_err:.4f}")  # fmt: skip
        assert m.cost_ <= ref.cost_ * (1 + START_RTOL)
        assert c_err / _length(R) < COMPONENT_LENGTHS
        assert w_err < COMPONENT_WIDTH_RTOL
        assert pi_err < MIXING_ATOL
        np.testing.assert_allclose(m.mixing_weights_.sum(axis=1), 1.0, rtol=1e-14)


@pytest.mark.parametrize("name", KERNELS)
def test_mixture_fit_explains_generated_mixtures(name):
    """make_tfa_data with two components (2D, seeds 0-5): the two-component
    fit's R^2 is within 1e-3 of the true parameters' (with ridge weights);
    measured -1.4e-4 to 5.6e-4. Their components need not be identifiable
    (component 2 is 0.75 sigma from component 1)."""
    for seed in range(6):
        X, R, t = make_tfa_data(
            (24, 24), 100, K=3, noise=0.1, random_state=seed, kernel=name,
            n_components=2,
        )  # fmt: skip
        F = component_factors(
            R, t["component_centers"], t["component_widths"], t["mixing_weights"],
            Kernel(name, 2),
        )  # fmt: skip
        r2_true = 1 - np.sum((X - F @ ridge_weights(X, F)) ** 2) / np.sum(X**2)
        m = _fit(X, R, 3, name, 2)
        r2 = 1 - np.sum((X - m.reconstruct()) ** 2) / np.sum(X**2)
        print(f"{name} seed {seed}: R^2 {r2:.5f}, true parameters {r2_true:.5f}")
        assert r2 > r2_true - 1e-3


def test_mixture_start_and_prior_validation():
    X, R, t = make_tfa_data((12, 12), 20, K=2, random_state=0, n_components=2)
    with pytest.raises(ValueError, match=r"start must be \(centers \(2, 2, 2\)"):
        TFA(K=2, n_components=2).fit(X, R, start=(t["centers"], t["widths"]))
    with pytest.raises(ValueError, match="mixing weights must be a positive"):
        TFA(K=2, n_components=2).fit(
            X, R, start=(t["component_centers"], t["component_widths"],
                         np.array([[1.0, 0.0], [0.5, 0.5]]))
        )  # fmt: skip
    with pytest.raises(ValueError, match="kernel must be one of"):
        TFA(K=2, kernel="gauss").fit(X, R)
    with pytest.raises(ValueError, match="n_components must be an integer"):
        TFA(K=2, n_components=0).fit(X, R)
    with pytest.raises(ValueError, match="must be"):
        HTFA(K=2, n_components=2, max_global_iter=1).fit(
            [X, X], [R, R], starts=[(t["centers"], t["widths"])] * 2
        )


def test_one_component_kernels_are_the_plain_kernels():
    """n_components=1 fits the plain kernel: the components are the factors
    and the mixing weights are 1."""
    X, R, _ = make_tfa_data((14, 14), 30, K=2, random_state=3, kernel="laplace")
    m = _fit(X, R, 2, "laplace", 1)
    np.testing.assert_array_equal(m.component_centers_[:, 0], m.centers_)
    np.testing.assert_array_equal(m.component_widths_[:, 0], m.widths_)
    np.testing.assert_array_equal(m.mixing_weights_, 1.0)
    F = component_factors(R, m.component_centers_, m.component_widths_,
                          m.mixing_weights_, Kernel("laplace"))  # fmt: skip
    np.testing.assert_array_equal(m.factors(), F)


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("mode", ["fast", "exact"])
def test_htfa_shapes_do_not_depend_on_n_jobs(name, M, mode):
    """HTFA with each kernel and mixtures: serial and n_jobs=2 fits are bit
    for bit equal (template, subject components, weights)."""
    Xs, Rs, _ = make_htfa_data(3, (14, 14), 30, K=2, noise=0.2, random_state=4,
                               kernel=name, n_components=M)  # fmt: skip
    fits = []
    for n_jobs in (None, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.append(
                HTFA(
                    K=2,
                    kernel=name,
                    n_components=M,
                    mode=mode,
                    n_jobs=n_jobs,
                    max_global_iter=3,
                    max_local_iter=5,
                    random_state=0,
                    standardize=False,
                ).fit(Xs, Rs)  # fmt: skip
            )
    a, b = fits
    for attr in ("global_component_centers_", "global_component_widths_",
                 "global_mixing_weights_", "global_centers_", "global_widths_",
                 "global_centers_cov_", "global_widths_var_",
                 "global_logits_var_"):  # fmt: skip
        np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr), attr)
    for attr in ("component_centers_", "component_widths_", "mixing_weights_",
                 "weights_"):  # fmt: skip
        for s in range(3):
            np.testing.assert_array_equal(getattr(a, attr)[s], getattr(b, attr)[s])
    assert a.objective_history_ == b.objective_history_


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("update", ["brainiak", "fixed", "empirical"])
def test_htfa_template_recovery_per_shape(name, M, update):
    """make_htfa_data (3 subjects, 24x24, K = 3, center jitter 1 voxel): the
    template's centers are within 0.15 lengths of the global truth (the mean
    of three jittered subjects is about 0.6 voxels = 0.08 lengths away;
    measured 0.079 to 0.115), with one component its widths within 6%
    (measured <= 3.3%); mixing weights are rows of positive numbers summing
    to 1, and the template's attributes have the documented shapes."""
    Xs, Rs, t = make_htfa_data(3, (24, 24), 60, K=3, noise=0.1, random_state=2,
                               kernel=name, n_components=M)  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = HTFA(K=3, kernel=name, n_components=M, template_update=update,
                 max_global_iter=5, random_state=0,
                 standardize=False).fit(Xs, Rs)  # fmt: skip
    L = _length(Rs[0])
    p = match_factors(t["global_centers"], h.global_centers_)
    c_err = np.max(np.linalg.norm(h.global_centers_[p] - t["global_centers"], axis=1))
    w_err = np.max(np.abs(h.global_widths_[p] / t["global_widths"] - 1))
    print(f"{name} M={M} {update}: centers {c_err / L:.3f} lengths, widths {w_err:.3f}")
    assert c_err / L < 0.15
    if M == 1:
        assert w_err < 0.06
    assert h.global_component_centers_.shape == (3, M, 2)
    assert h.global_component_widths_.shape == (3, M)
    assert h.global_logits_var_.shape == (3, M - 1)
    assert np.all(h.global_mixing_weights_ > 0)
    np.testing.assert_allclose(h.global_mixing_weights_.sum(1), 1.0, rtol=1e-14)
    expected_cov = (3, 2, 2) if M == 1 else (3, M, 2, 2)
    assert h.global_centers_cov_.shape == expected_cov
    for s in range(3):
        assert h.component_centers_[s].shape == (3, M, 2)
        np.testing.assert_allclose(
            np.einsum("km,kmd->kd", h.mixing_weights_[s], h.component_centers_[s]),
            h.centers_[s], rtol=1e-12,
        )  # fmt: skip


def test_htfa_resume_from_mixture_fit():
    """``starts`` and ``init_template`` take a mixture fit's own arrays."""
    Xs, Rs, _ = make_htfa_data(2, (12, 12), 20, K=2, random_state=5,
                               n_components=2)  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = HTFA(K=2, n_components=2, max_global_iter=2, random_state=0).fit(Xs, Rs)
        again = HTFA(K=2, n_components=2, max_global_iter=1, random_state=0).fit(
            Xs, Rs, starts=h.free_fits_,
            init_template=(h.global_component_centers_, h.global_component_widths_,
                           h.global_mixing_weights_),
        )  # fmt: skip
    assert len(h.free_fits_[0]) == 3
    np.testing.assert_array_equal(again.template_history_[0][0],
                                  h.global_component_centers_)  # fmt: skip


@pytest.mark.parametrize("D", [2, 3])
def test_effective_nodes_of_each_kernel(D):
    """An interior node keeps its center, and its effective width is that of
    the RBF with its spread: w for 'rbf', 6.02 w / 8.01 w for 'laplace' and
    0.282 w / 0.351 w for 'mexican_hat' (D = 2 / 3, from radial integrals),
    to the grid's resolution; the in-mask mass fraction is about 1."""
    n = 61 if D == 2 else 41
    ax = np.arange(n, dtype=float) - n // 2
    grid = np.stack(np.meshgrid(*[ax] * D, indexing="ij"), -1).reshape(-1, D)
    expected = {"rbf": 1.0, "laplace": (6.018, 8.010)[D - 2],
                "mexican_hat": (0.2817, 0.3514)[D - 2]}  # fmt: skip
    for name, ratio in expected.items():
        # Positive lobe of the mexican hat: radius sqrt(D w / 2), 4.9 voxels.
        w = (24.0 if D == 2 else 16.0) if name == "mexican_hat" else 6.0 / (D - 1)
        c, w_eff, info = effective_nodes(np.zeros((1, D)), [w], grid, kernel=name)
        print(f"{name} D={D}: ratio {w_eff[0] / w:.4f} (expected {ratio}), mass "
              f"{info['mass_fraction'][0]:.4f}")  # fmt: skip
        np.testing.assert_allclose(c, 0.0, atol=1e-12)
        assert w_eff[0] / w == pytest.approx(ratio, rel=0.03)
        assert info["mass_fraction"][0] == pytest.approx(1.0, abs=0.03)


def test_effective_nodes_of_mixtures_and_far_mexican_hats():
    """A mixture node's centroid is pulled towards its heavier component; a
    mexican hat whose positive lobe holds no mask voxel is moved to the mask
    like a point, with NaN width; component centers need widths."""
    ax = np.arange(-20.0, 21.0)
    grid = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
    C = np.array([[[-5.0, 0.0], [5.0, 0.0]]])
    for pi, sign in (([[0.8, 0.2]], -1), ([[0.2, 0.8]], 1)):
        c, w_eff, _ = effective_nodes(C, [[8.0, 8.0]], grid, "rbf", pi)
        assert np.sign(c[0, 0]) == sign and abs(c[0, 1]) < 1e-9
        assert w_eff[0] > 8.0  # two separated blobs spread more than one
    c, w_eff, info = effective_nodes([[60.0, 0.0]], [4.0], grid, "mexican_hat")
    assert info["snapped"][0] and np.isnan(w_eff[0])
    np.testing.assert_array_equal(c[0], [20.0, 0.0])
    with pytest.raises(ValueError, match="need their widths"):
        effective_nodes(C, None, grid)


def test_results_and_display_nodes_for_mixture_fits():
    """HTFAResults of TFA and HTFA mixture fits: factors equal the model's,
    and display_nodes uses the component profiles."""
    X, R, _ = make_tfa_data((14, 14), 30, K=2, random_state=6,
                            kernel="mexican_hat", n_components=2)  # fmt: skip
    res = HTFAResults(_fit(X, R, 2, "mexican_hat", 2), load(X, coords=R))
    np.testing.assert_array_equal(res.factors(), res.model.factors())
    centers, widths, info = res.display_nodes()
    m = res.model
    expected = effective_nodes(m.component_centers_, m.component_widths_, R,
                               "mexican_hat", m.mixing_weights_)  # fmt: skip
    np.testing.assert_array_equal(centers, expected[0])
    np.testing.assert_array_equal(widths, expected[1])
    Xs, Rs, _ = make_htfa_data(2, (12, 12), 20, K=2, random_state=7,
                               kernel="laplace", n_components=2)  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = HTFA(K=2, kernel="laplace", n_components=2, max_global_iter=2,
                 random_state=0).fit(Xs, Rs)  # fmt: skip
    res = HTFAResults(h, load(Xs, coords=Rs))
    F = component_factors(Rs[1], h.component_centers_[1], h.component_widths_[1],
                          h.mixing_weights_[1], Kernel("laplace", 2))  # fmt: skip
    np.testing.assert_array_equal(res.factors(1), F)
    g = res.display_nodes("global")
    assert g[0].shape == (2, 2) and np.all(np.isfinite(g[1]))
