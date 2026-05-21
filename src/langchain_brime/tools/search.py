"""BrimeSearch — LangChain BaseTool wrapping POST /v1/search."""

from __future__ import annotations

from typing import Any, Literal, cast

from brime.async_client import AsyncBrime
from brime.client import Brime
from brime.errors import BrimeError
from brime.models.search import SearchResponse
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, model_validator

from langchain_brime._errors import brime_to_tool_exception
from langchain_brime._utilities import (
    build_async_client,
    build_sync_client,
    resolve_brime_api_key,
)


class BrimeSearchInput(BaseModel):
    """Schema exposed to the LLM for the brime_search tool."""

    query: str = Field(
        ...,
        description=(
            "Natural-language search query. Be specific and self-contained — "
            "the LLM you are calling has no other context for this query.\n\n"
            "Good: 'BM25 ranking function in information retrieval'.\n"
            "Bad: 'the thing we were discussing'."
        ),
    )
    depth: Literal["instant", "basic", "advanced"] = Field(
        "basic",
        description=(
            "Trade-off knob between speed and answer quality:\n"
            "- 'instant' — SERP snippets only, no scrape, no LLM answer. "
            "Cheapest and fastest. Use when you only need URLs to feed to "
            "brime_extract afterwards.\n"
            "- 'basic' (default) — SERP + BM25 ranking + LLM-synthesised "
            "answer with citations. Use for typical Q&A.\n"
            "- 'advanced' — basic plus stronger reranking and more thorough "
            "chunk synthesis. Use when the user explicitly asks for higher "
            "quality and is willing to pay more credits."
        ),
    )
    max_results: int = Field(
        5,
        ge=1,
        le=20,
        description=(
            "Number of result items to return (1-20). Default 5 is usually "
            "enough; raise for harder queries that need broader coverage."
        ),
    )
    topic: Literal["general", "news", "finance"] = Field(
        "general",
        description=(
            "Recency hint for the SERP backend. Use 'news' for current "
            "events that broke in the last few days; 'finance' for ticker, "
            "earnings and market data; 'general' for everything else."
        ),
    )
    time_range: Literal["day", "week", "month", "year"] | None = Field(
        None,
        description=(
            "Restrict results to a specific recency window. Use this when "
            "the user asks for 'recent' / 'latest' / 'this week'."
        ),
    )


class BrimeSearch(BaseTool):
    """Search the web through Brime.

    Use this tool for: factual lookups, current events, finding URLs to
    scrape later with brime_extract. For multi-hop synthesis with citations,
    use brime_research instead.
    """

    name: str = "brime_search"
    description: str = (
        "Search the web with Brime. Returns an LLM-friendly synthesised "
        "answer plus a ranked list of source URLs. Pick depth='instant' for "
        "cheap snippets, 'basic' for a synthesised answer (default), "
        "'advanced' for higher quality."
    )
    args_schema: type[BaseModel] = BrimeSearchInput  # type: ignore[assignment,unused-ignore]
    handle_tool_error: bool = True  # type: ignore[assignment,unused-ignore]

    api_key: SecretStr | None = Field(default=None, exclude=True)
    base_url: str | None = None
    timeout: float = 30.0

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

    def _run(
        self,
        query: str,
        depth: Literal["instant", "basic", "advanced"] = "basic",
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        time_range: Literal["day", "week", "month", "year"] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
        **_: Any,
    ) -> str:
        try:
            res = self._get_sync().search(
                query,
                depth=depth,
                max_results=max_results,
                topic=topic,
                time_range=time_range,
            )
        except BrimeError as exc:
            raise brime_to_tool_exception(exc) from exc
        except ToolException:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolException(f"Unexpected Brime error: {exc}") from exc
        return _format_search(res)

    async def _arun(
        self,
        query: str,
        depth: Literal["instant", "basic", "advanced"] = "basic",
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        time_range: Literal["day", "week", "month", "year"] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **_: Any,
    ) -> str:
        try:
            res = await self._get_async().search(
                query,
                depth=depth,
                max_results=max_results,
                topic=topic,
                time_range=time_range,
            )
        except BrimeError as exc:
            raise brime_to_tool_exception(exc) from exc
        except ToolException:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolException(f"Unexpected Brime error: {exc}") from exc
        return _format_search(res)


def _format_search(res: SearchResponse) -> str:
    lines: list[str] = []
    if res.answer:
        lines.append(res.answer.strip())
        lines.append("")
    lines.append("Sources:")
    for r in res.results:
        score = f" (score={r.score:.2f})" if r.score is not None else ""
        lines.append(f"- [{r.title}]({r.url}){score}")
    return "\n".join(lines)


__all__ = ["BrimeSearch", "BrimeSearchInput"]
