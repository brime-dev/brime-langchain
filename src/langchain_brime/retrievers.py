"""BrimeRetriever — LangChain BaseRetriever backed by /v1/search."""

from __future__ import annotations

from typing import Any, Literal, cast

from brime.async_client import AsyncBrime
from brime.client import Brime
from brime.models.search import SearchResultItem
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, model_validator

from langchain_brime._utilities import (
    build_async_client,
    build_sync_client,
    resolve_brime_api_key,
)


class BrimeRetriever(BaseRetriever):
    """Retriever that turns a Brime /v1/search call into a list of Documents.

    Each result becomes a ``Document`` with ``page_content`` set to the
    BM25-ranked snippet and ``metadata`` containing url, title, score,
    published_date and ``source="brime"``. ``Document.id`` is set to the
    result URL so LangChain 1.x consumers can deduplicate.
    """

    api_key: SecretStr | None = Field(default=None, exclude=True)
    base_url: str | None = None
    timeout: float = 30.0

    k: int = 5
    depth: Literal["instant", "basic", "advanced"] = "basic"
    topic: Literal["general", "news", "finance"] = "general"
    time_range: Literal["day", "week", "month", "year"] | None = None
    domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    include_answer: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sync_client: Brime | None = PrivateAttr(default=None)
    _async_client: AsyncBrime | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_api_key(cls, values: Any) -> Any:
        if isinstance(values, dict):
            return resolve_brime_api_key(cast("dict[str, Any]", values))
        return values

    def _get_sync(self) -> Brime:
        if self._sync_client is None:
            self._sync_client = build_sync_client(self.api_key, self.base_url, self.timeout)
        return self._sync_client

    def _get_async(self) -> AsyncBrime:
        if self._async_client is None:
            self._async_client = build_async_client(self.api_key, self.base_url, self.timeout)
        return self._async_client

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **_: Any,
    ) -> list[Document]:
        res = self._get_sync().search(
            query,
            depth=self.depth,
            max_results=max(1, min(self.k, 20)),
            topic=self.topic,
            time_range=self.time_range,
            domains=self.domains,
            exclude_domains=self.exclude_domains,
            include_answer=self.include_answer,
        )
        k = max(1, min(self.k, 20))
        return [_to_document(r) for r in res.results[:k]]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **_: Any,
    ) -> list[Document]:
        res = await self._get_async().search(
            query,
            depth=self.depth,
            max_results=max(1, min(self.k, 20)),
            topic=self.topic,
            time_range=self.time_range,
            domains=self.domains,
            exclude_domains=self.exclude_domains,
            include_answer=self.include_answer,
        )
        k = max(1, min(self.k, 20))
        return [_to_document(r) for r in res.results[:k]]


def _to_document(r: SearchResultItem) -> Document:
    return Document(
        id=r.url,
        page_content=r.content,
        metadata={
            "url": r.url,
            "title": r.title,
            "score": r.score,
            "published_date": r.published_date,
            "source": "brime",
        },
    )


__all__ = ["BrimeRetriever"]
