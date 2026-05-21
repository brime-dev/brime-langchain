"""BrimeResearch — LangChain BaseTool wrapping POST /v1/research."""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from brime.models.research import (
    ResearchBasicResponse,
    ResearchStatusResponse,
)
from langchain_brime._client import _ClientHolder


class BrimeResearchInput(BaseModel):
    """Schema exposed to the LLM for the brime_research tool."""

    query: str = Field(..., description="Natural-language research question.")
    depth: Literal["basic", "deep"] = Field(
        "basic",
        description=(
            "'basic' = single-shot agent loop (~10-30s, 1-3 rounds). "
            "'deep' = multi-step iterative research (60-600s, up to 8 rounds). "
            "Pick 'deep' only when the user asks for detailed synthesis "
            "across multiple sources."
        ),
    )
    max_rounds: Optional[int] = Field(
        None,
        ge=1,
        le=8,
        description="Tool-call rounds. basic 1-3, deep 1-8.",
    )


class BrimeResearch(BaseTool):
    """Run a research job through Brime's iterative agent.

    For factual look-ups prefer brime_search. Use brime_research when the
    user wants synthesised analysis with citations. depth='deep' is
    higher-quality but slow (minutes) — only use when asked.
    """

    name: str = "brime_research"
    description: str = (
        "Run iterative research with citations. depth='basic' returns a "
        "synthesised answer in ~10-30s. depth='deep' runs multi-round "
        "research in 1-10 minutes. Use 'deep' only when the question "
        "warrants thorough multi-source synthesis."
    )
    args_schema: Type[BaseModel] = BrimeResearchInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    deep_poll_timeout: float = 420.0
    deep_poll_interval: float = 8.0

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
        depth: Literal["basic", "deep"] = "basic",
        max_rounds: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        c = self._client().sync
        if depth == "basic":
            res = c.research(query, depth="basic", max_rounds=max_rounds)
            return _format_basic(res)  # type: ignore[arg-type]
        res = c.research(
            query,
            depth="deep",
            max_rounds=max_rounds,
            wait=True,
            poll_interval=self.deep_poll_interval,
            poll_timeout=self.deep_poll_timeout,
        )
        return _format_deep(res)  # type: ignore[arg-type]

    async def _arun(
        self,
        query: str,
        depth: Literal["basic", "deep"] = "basic",
        max_rounds: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        c = self._client().async_
        if depth == "basic":
            res = await c.research(query, depth="basic", max_rounds=max_rounds)
            return _format_basic(res)  # type: ignore[arg-type]
        res = await c.research(
            query,
            depth="deep",
            max_rounds=max_rounds,
            wait=True,
            poll_interval=self.deep_poll_interval,
            poll_timeout=self.deep_poll_timeout,
        )
        return _format_deep(res)  # type: ignore[arg-type]


def _format_basic(res: ResearchBasicResponse) -> str:
    lines: List[str] = []
    lines.append((res.answer or "(no answer)").strip())
    lines.append("")
    lines.append("Sources:")
    for s in res.sources:
        lines.append(f"- [{s.title}]({s.url})")
    return "\n".join(lines)


def _format_deep(res: ResearchStatusResponse) -> str:
    if res.status != "complete":
        err = res.error.message if res.error else res.status
        return f"Research did not complete (status={res.status}): {err}"
    lines: List[str] = []
    lines.append((res.answer or "(no answer)").strip())
    lines.append("")
    lines.append(
        f"_Rounds: {res.current_round}/{res.max_rounds} · "
        f"Sources: {res.sources_count} · Steps: {res.steps_count}_"
    )
    return "\n".join(lines)


__all__: List[str] = ["BrimeResearch", "BrimeResearchInput"]
