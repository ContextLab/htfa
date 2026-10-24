"""Tests for factor-level connectivity."""

import numpy as np
import pytest

from htfa.connectivity import dynamic_connectivity, isfc, static_connectivity


@pytest.fixture
def w():
    return np.random.default_rng(0).standard_normal((4, 60))


def test_static_connectivity_equals_corrcoef(w):
    np.testing.assert_allclose(static_connectivity(w), np.corrcoef(w))


def test_dynamic_connectivity_window(w):
    out = dynamic_connectivity(w, 20)
    assert out.shape == (41, 4, 4)
    np.testing.assert_allclose(out[5], np.corrcoef(w[:, 5:25]))
    np.testing.assert_allclose(dynamic_connectivity(w, 60)[0], static_connectivity(w))


def test_dynamic_connectivity_bad_window(w):
    with pytest.raises(ValueError, match="window"):
        dynamic_connectivity(w, 61)


def test_isfc_identical_subjects_is_static(w):
    # measured: identical subjects give corr = static, up to the 1e-7 Fisher clip
    out = isfc([w, w, w])
    np.testing.assert_allclose(out, static_connectivity(w), atol=1e-6)
    d = isfc([w, w], window=20)
    assert d.shape == (41, 4, 4)
    np.testing.assert_allclose(d[3], np.corrcoef(w[:, 3:23]), atol=1e-6)


def test_isfc_is_symmetric_and_bounded():
    rng = np.random.default_rng(1)
    ws = [rng.standard_normal((3, 40)) for _ in range(4)]
    out = isfc(ws)
    np.testing.assert_allclose(out, out.T)
    assert np.all(np.abs(out) <= 1)


def test_isfc_requires_equal_T(w):
    with pytest.raises(ValueError, match="share K and T"):
        isfc([w, w[:, :50]])
