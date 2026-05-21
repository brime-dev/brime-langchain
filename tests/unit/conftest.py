"""Shared fixtures: monkeypatch Brime/AsyncBrime clients with stubs."""

from __future__ import annotations

from typing import Any

import pytest


class FakeSearchResult:
    def __init__(self, query: str, depth: str, max_results: int) -> None:
        self.query = query
        self.depth = depth
        self.max_results = max_results
        self.answer = f"answer for {query}"
        self.results = [
            type(
                "R",
                (),
                {
                    "title": f"Result {i}",
                    "url": f"https://r{i}.example.com",
                    "content": f"content {i}",
                    "score": 1.0 - i * 0.1,
                    "published_date": None,
                },
            )()
            for i in range(min(max_results, 3))
        ]
        self.request_id = "req_test"
        self.credits_used = 0.5
        self.latency_ms = 50


class FakeExtractResult:
    def __init__(self, urls: list[str]) -> None:
        ok = urls[: max(0, len(urls) - 1)] if len(urls) > 1 else urls
        bad = urls[len(ok) :]
        self.results = [
            type(
                "E",
                (),
                {
                    "url": u,
                    "markdown": f"# {u}\nbody",
                    "method": "worker_static",
                    "content_type": "html",
                    "status": 200,
                    "latency_ms": 100,
                },
            )()
            for u in ok
        ]
        self.failed = [
            type(
                "F",
                (),
                {
                    "url": u,
                    "error": type(
                        "Err",
                        (),
                        {
                            "code": "fetch_failed",
                            "message": "404",
                            "needs_browser": False,
                        },
                    )(),
                },
            )()
            for u in bad
        ]
        self.request_id = "req"
        self.credits_used = 1
        self.latency_ms = 200


class FakeResearchBasic:
    def __init__(self, query: str) -> None:
        self.query = query
        self.answer = f"basic answer for {query}"
        self.sources = [
            type(
                "S",
                (),
                {
                    "title": "Src",
                    "url": "https://src.example.com",
                    "content": "c",
                    "score": 0.9,
                    "published_date": None,
                },
            )()
        ]
        self.steps: list[Any] = []
        self.request_id = "req"
        self.credits_used = 2
        self.latency_ms = 100


class FakeResearchStatus:
    def __init__(self, query: str, status: str = "complete") -> None:
        self.job_id = "job_test"
        self.status = status
        self.current_round = 3
        self.max_rounds = 5
        self.query = query
        self.depth = "deep"
        self.started_at = "2026-05-06T00:00:00Z"
        self.updated_at = "2026-05-06T00:00:30Z"
        self.completed_at = "2026-05-06T00:00:30Z"
        self.answer = f"deep answer for {query}"
        self.sources_count = 12
        self.steps_count = 3
        self.error: Any | None = None
        self.credits_used = 5


class FakeSyncClient:
    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    def search(self, query: str, **kw: Any) -> Any:
        return FakeSearchResult(query, kw.get("depth", "basic"), kw.get("max_results", 5))

    def extract(self, urls: Any, **kw: Any) -> Any:
        url_list = [urls] if isinstance(urls, str) else list(urls)
        return FakeExtractResult(url_list)

    def research(self, query: str, **kw: Any) -> Any:
        if kw.get("depth") == "deep" and kw.get("wait"):
            return FakeResearchStatus(query)
        return FakeResearchBasic(query)

    def close(self) -> None:
        pass


class FakeAsyncClient:
    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    async def search(self, query: str, **kw: Any) -> Any:
        return FakeSearchResult(query, kw.get("depth", "basic"), kw.get("max_results", 5))

    async def extract(self, urls: Any, **kw: Any) -> Any:
        url_list = [urls] if isinstance(urls, str) else list(urls)
        return FakeExtractResult(url_list)

    async def research(self, query: str, **kw: Any) -> Any:
        if kw.get("depth") == "deep" and kw.get("wait"):
            return FakeResearchStatus(query)
        return FakeResearchBasic(query)

    async def aclose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _stub_brime_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langchain_brime._utilities.Brime", FakeSyncClient)
    monkeypatch.setattr("langchain_brime._utilities.AsyncBrime", FakeAsyncClient)
