"""Root pytest configuration: keep doctests from leaking global state."""

import pytest

from htfa import _lazy


@pytest.fixture(autouse=True)
def _restore_autoinstall_override():
    saved = _lazy._override
    yield
    _lazy._override = saved
