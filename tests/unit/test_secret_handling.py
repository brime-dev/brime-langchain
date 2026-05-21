"""SecretStr no-leak guarantees for tools and retriever.

The API key field is declared as ``SecretStr | None`` with
``Field(exclude=True)``. That guarantees:

- ``repr(tool)`` never leaks the raw key (pydantic prints ``SecretStr('**********')``).
- ``str(tool)``  ditto.
- ``tool.model_dump()`` and ``model_dump_json()`` omit the field entirely
  because of ``exclude=True``.
- The env fallback (``BRIME_API_KEY``) is picked up by the model_validator.
"""

from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from langchain_brime import BrimeExtract, BrimeResearch, BrimeRetriever, BrimeSearch

_SECRET = "sk-brime-supersecret-do-not-leak"


@pytest.mark.parametrize(
    "ctor",
    [BrimeSearch, BrimeExtract, BrimeResearch, BrimeRetriever],
)
def test_secret_never_leaks_in_repr(ctor: type) -> None:
    tool = ctor(api_key=_SECRET)
    assert _SECRET not in repr(tool)
    assert _SECRET not in str(tool)


@pytest.mark.parametrize(
    "ctor",
    [BrimeSearch, BrimeExtract, BrimeResearch, BrimeRetriever],
)
def test_secret_excluded_from_serialisation(ctor: type) -> None:
    tool = ctor(api_key=_SECRET)
    dumped = tool.model_dump()
    assert "api_key" not in dumped
    # Make sure no nested field accidentally surfaces the secret.
    assert _SECRET not in json.dumps(dumped, default=str)


@pytest.mark.parametrize(
    "ctor",
    [BrimeSearch, BrimeExtract, BrimeResearch, BrimeRetriever],
)
def test_secret_is_coerced_to_secretstr(ctor: type) -> None:
    tool = ctor(api_key=_SECRET)
    assert isinstance(tool.api_key, SecretStr)
    assert tool.api_key.get_secret_value() == _SECRET


@pytest.mark.parametrize(
    "ctor",
    [BrimeSearch, BrimeExtract, BrimeResearch, BrimeRetriever],
)
def test_env_fallback_when_api_key_omitted(monkeypatch: pytest.MonkeyPatch, ctor: type) -> None:
    monkeypatch.setenv("BRIME_API_KEY", _SECRET)
    tool = ctor()
    assert isinstance(tool.api_key, SecretStr)
    assert tool.api_key.get_secret_value() == _SECRET


@pytest.mark.parametrize(
    "ctor",
    [BrimeSearch, BrimeExtract, BrimeResearch, BrimeRetriever],
)
def test_missing_api_key_resolves_to_none(monkeypatch: pytest.MonkeyPatch, ctor: type) -> None:
    monkeypatch.delenv("BRIME_API_KEY", raising=False)
    tool = ctor()
    assert tool.api_key is None
