from __future__ import annotations

import asyncio

from langchain_brime import BrimeExtract


def test_extract_tool_metadata() -> None:
    t = BrimeExtract(api_key="sk-test")
    assert t.name == "brime_extract"
    schema = t.args_schema.model_json_schema()
    assert "urls" in schema["properties"]


def test_extract_tool_success() -> None:
    t = BrimeExtract(api_key="sk-test")
    out = t.invoke({"urls": ["https://example.com"]})
    assert "## https://example.com" in out
    assert "body" in out
    assert "Failed URLs" not in out


def test_extract_tool_mixed_failed() -> None:
    t = BrimeExtract(api_key="sk-test")
    # Fake stub treats last URL as failed when multiple given.
    out = t.invoke({"urls": ["https://a.com", "https://b.com"]})
    assert "## https://a.com" in out
    assert "Failed URLs" in out
    assert "fetch_failed" in out


def test_extract_tool_async() -> None:
    t = BrimeExtract(api_key="sk-test")
    out = asyncio.run(t.ainvoke({"urls": ["https://x.com"]}))
    assert "## https://x.com" in out
