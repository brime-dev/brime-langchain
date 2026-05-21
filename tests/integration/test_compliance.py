"""LangChain official ToolsUnitTests compliance suite.

Reuses the FakeSyncClient/FakeAsyncClient stubs from tests/unit/conftest.py
so the suite runs without a network call.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

# Reuse the unit test stubs (FakeSyncClient/FakeAsyncClient)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from conftest import FakeAsyncClient, FakeSyncClient  # type: ignore[import-not-found]  # noqa: E402

from langchain_brime import BrimeExtract, BrimeResearch, BrimeSearch  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_brime_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langchain_brime._client.Brime", FakeSyncClient)
    monkeypatch.setattr("langchain_brime._client.AsyncBrime", FakeAsyncClient)


class TestBrimeSearchCompliance(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return BrimeSearch

    @property
    def tool_constructor_params(self) -> Dict[str, Any]:
        return {"api_key": "sk-test"}

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"query": "BM25"}


class TestBrimeExtractCompliance(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return BrimeExtract

    @property
    def tool_constructor_params(self) -> Dict[str, Any]:
        return {"api_key": "sk-test"}

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"urls": ["https://example.com"]}


class TestBrimeResearchCompliance(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return BrimeResearch

    @property
    def tool_constructor_params(self) -> Dict[str, Any]:
        return {"api_key": "sk-test"}

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"query": "what is BM25"}


# Suppress lint complaining unused symbols
_ = (Tuple,)
