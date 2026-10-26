"""The API reference must list every public name."""

import re
from pathlib import Path

import htfa

API_RST = Path(__file__).resolve().parents[1] / "docs" / "api.rst"
EXTRA_NAMES = [
    "htfa.io.load",
    "htfa.io.Subject",
    "htfa.datasets.fetch_development_fmri",
    "htfa.datasets.regress_confounds",
    "htfa.synthetic.match_factors",
    "htfa.tfa.standardize_data",
]


def _listed_names() -> set[str]:
    text = API_RST.read_text()
    return set(re.findall(r"^   (htfa\.[\w.]+)$", text, flags=re.MULTILINE))


def test_api_rst_lists_every_public_name():
    listed = _listed_names()
    missing = [n for n in htfa.__all__ if f"htfa.{n}" not in listed]
    assert not missing, f"docs/api.rst does not list {missing}"


def test_api_rst_lists_extra_names():
    missing = [n for n in EXTRA_NAMES if n not in _listed_names()]
    assert not missing, f"docs/api.rst does not list {missing}"


def test_api_rst_names_resolve():
    for name in _listed_names():
        obj = htfa
        for part in name.split(".")[1:]:
            obj = getattr(obj, part)
        assert obj.__doc__, f"{name} has no docstring"
