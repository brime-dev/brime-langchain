from __future__ import annotations

import asyncio

from langchain_brime import BrimeSearch


def test_search_tool_metadata() -> None:
    t = BrimeSearch(api_key="sk-test")
    assert t.name == "brime_search"
    assert "brime" in t.description.lower()
    schema = t.args_schema.model_json_schema()
    assert "query" in schema["properties"]
    assert "depth" in schema["properties"]


def test_search_tool_run() -> None:
    t = BrimeSearch(api_key="sk-test")
    out = t.invoke({"query": "BM25"})
    assert isinstance(out, str)
    assert "answer for BM25" in out
    assert "Sources:" in out
    assert "https://r0.example.com" in out


def test_search_tool_invoke_with_filters() -> None:
    t = BrimeSearch(api_key="sk-test")
    out = t.invoke({"query": "tesla", "depth": "advanced", "max_results": 2, "topic": "finance"})
    assert "answer for tesla" in out


def test_search_tool_async() -> None:
    t = BrimeSearch(api_key="sk-test")
    out = asyncio.run(t.ainvoke({"query": "py"}))
    assert "answer for py" in out
