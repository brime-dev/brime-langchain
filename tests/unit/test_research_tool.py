from __future__ import annotations

import asyncio

from langchain_brime import BrimeResearch


def test_research_tool_metadata() -> None:
    t = BrimeResearch(api_key="sk-test")
    assert t.name == "brime_research"
    schema = t.args_schema.model_json_schema()
    assert {"query", "depth", "max_rounds"} <= set(schema["properties"])


def test_research_basic() -> None:
    t = BrimeResearch(api_key="sk-test")
    out = t.invoke({"query": "what is BM25"})
    assert "basic answer for what is BM25" in out
    assert "Sources:" in out
    assert "https://src.example.com" in out


def test_research_deep_wait() -> None:
    t = BrimeResearch(api_key="sk-test")
    out = t.invoke({"query": "deep query", "depth": "deep", "max_rounds": 3})
    assert "deep answer for deep query" in out
    assert "Sources: 12" in out


def test_research_async_basic() -> None:
    t = BrimeResearch(api_key="sk-test")
    out = asyncio.run(t.ainvoke({"query": "async q"}))
    assert "basic answer for async q" in out
