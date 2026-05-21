"""BrimeSearch — LangChain BaseTool wrapping POST /v1/search."""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from brime.models.search import SearchResponse
from langchain_brime._client import _ClientHolder


class BrimeSearchInput(BaseModel):
    """Schema exposed to the LLM for the brime_search tool."""

    query: str = Field(..., description="Natural-language search query.")
    depth: Literal["instant", "basic", "advanced"] = Field(
        "basic",
        description=(
            "Trade-off knob. 'instant' = SERP snippets, no scrape, no LLM "
            "answer (fast, cheap). 'basic' = SERP + chunk + BM25 + LLM "
            "answer (default). 'advanced' = better ranking + chunk rerank "
            "(slower, costs more)."
        ),
    )
    max_results: int = Field(5, ge=1, le=20, description="Number of results (1-20).")
    topic: Literal["general", "news", "finance"] = Field(
        "general", description="Recency hint. Use 'news' for current events."
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        None, description="Restrict results to the given recency window."
    )


class BrimeSearch(BaseTool):
    """Search the web through Brime.

    Use for: factual lookups, current events, finding URLs to scrape later.
    Use BrimeResearch for multi-hop synthesis with sources.
    """

    name: str = "brime_search"
    description: str = (
        "Search the web with Brime. Returns an LLM-friendly summary plus a "
        "list of source URLs. Pick depth='instant' for cheap snippets, "
        "'basic' for a synthesised answer, 'advanced' for higher quality."
    )
    args_schema: Type[BaseModel] = BrimeSearchInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 30.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _holder: Optional[_ClientHolder] = PrivateAttr(default=None)

    def _client(self) -> _ClientHolder:
        if self._holder is None:
            self._holder = _ClientHolder(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._holder

    def _run(
        self,
        query: str,
        depth: Literal["instant", "basic", "advanced"] = "basic",
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        res = self._client().sync.search(
            query,
            depth=depth,
            max_results=max_results,
            topic=topic,
            time_range=time_range,
        )
        return _format_search(res)

    async def _arun(
        self,
        query: str,
        depth: Literal["instant", "basic", "advanced"] = "basic",
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        res = await self._client().async_.search(
            query,
            depth=depth,
            max_results=max_results,
            topic=topic,
            time_range=time_range,
        )
        return _format_search(res)


def _format_search(res: SearchResponse) -> str:
    lines: List[str] = []
    if res.answer:
        lines.append(res.answer.strip())
        lines.append("")
    lines.append("Sources:")
    for r in res.results:
        score = f" (score={r.score:.2f})" if r.score is not None else ""
        lines.append(f"- [{r.title}]({r.url}){score}")
    return "\n".join(lines)
