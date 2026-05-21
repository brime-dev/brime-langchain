from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from langchain_brime import BrimeRetriever


def test_retriever_returns_documents() -> None:
    r = BrimeRetriever(api_key="sk-test", k=3)
    docs = r.invoke("BM25")
    assert len(docs) == 3
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].metadata["source"] == "brime"
    assert docs[0].metadata["url"].startswith("https://")
    assert docs[0].metadata["score"] is not None
    # score order preserved: r0 highest
    assert docs[0].metadata["score"] >= docs[1].metadata["score"]


def test_retriever_clamps_k() -> None:
    r = BrimeRetriever(api_key="sk-test", k=100)
    docs = r.invoke("x")
    # Stub caps at 3 results regardless; ensure no crash on large k
    assert len(docs) >= 1


def test_retriever_async() -> None:
    r = BrimeRetriever(api_key="sk-test", k=2)
    docs = asyncio.run(r.ainvoke("py"))
    assert len(docs) == 2
    assert docs[0].page_content == "content 0"
