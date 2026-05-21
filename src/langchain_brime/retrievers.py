"""BrimeRetriever — LangChain BaseRetriever backed by /v1/search."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from brime.models.search import SearchResultItem
from langchain_brime._client import _ClientHolder


class BrimeRetriever(BaseRetriever):
    """Retriever that turns a Brime /v1/search call into a list of Documents.

    Each result becomes a `Document` with `page_content` set to the result
    snippet (BM25-ranked content) and `metadata` containing url, title,
    score, published_date, and source="brime".
    """

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 30.0

    k: int = 5
    depth: Literal["instant", "basic", "advanced"] = "basic"
    topic: Literal["general", "news", "finance"] = "general"
    time_range: Optional[Literal["day", "week", "month", "year"]] = None
    domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    include_answer: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _holder: Optional[_ClientHolder] = PrivateAttr(default=None)

    def _client(self) -> _ClientHolder:
        if self._holder is None:
            self._holder = _ClientHolder(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._holder

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        res = self._client().sync.search(
            query,
            depth=self.depth,
            max_results=max(1, min(self.k, 20)),
            topic=self.topic,
            time_range=self.time_range,
            domains=self.domains,
            exclude_domains=self.exclude_domains,
            include_answer=self.include_answer,
        )
        return [_to_document(r) for r in res.results]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        res = await self._client().async_.search(
            query,
            depth=self.depth,
            max_results=max(1, min(self.k, 20)),
            topic=self.topic,
            time_range=self.time_range,
            domains=self.domains,
            exclude_domains=self.exclude_domains,
            include_answer=self.include_answer,
        )
        return [_to_document(r) for r in res.results]


def _to_document(r: SearchResultItem) -> Document:
    return Document(
        page_content=r.content,
        metadata={
            "url": r.url,
            "title": r.title,
            "score": r.score,
            "published_date": r.published_date,
            "source": "brime",
        },
    )
