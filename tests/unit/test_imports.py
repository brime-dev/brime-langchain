"""Public surface drift guard.

If a new top-level export is added or removed without updating
``langchain_brime.__all__``, this test fails. Keeps the public API
contract honest across releases.
"""

from __future__ import annotations

import langchain_brime

EXPECTED_PUBLIC = {
    "BrimeExtract",
    "BrimeResearch",
    "BrimeRetriever",
    "BrimeSearch",
    "__version__",
}


def test_all_matches_expected_surface() -> None:
    assert set(langchain_brime.__all__) == EXPECTED_PUBLIC


def test_every_public_name_is_importable() -> None:
    for name in langchain_brime.__all__:
        assert hasattr(langchain_brime, name), f"missing public attribute: {name}"


def test_version_is_a_non_empty_string() -> None:
    v = langchain_brime.__version__
    assert isinstance(v, str)
    assert v
