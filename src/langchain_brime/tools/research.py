"""BrimeResearch — LangChain BaseTool wrapping POST /v1/research."""

from __future__ import annotations

from typing import Any, Literal, cast

from brime.async_client import AsyncBrime
from brime.client import Brime
from brime.errors import BrimeError
from brime.models.research import (
    ResearchBasicResponse,
    ResearchStatusResponse,
)
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


class BrimeResearchInput(BaseModel):
    """Schema exposed to the LLM for the brime_research tool."""

    query: str = Field(
        ...,
        description=(
            "Natural-language research question. Be specific — the research "
            "agent expands this into multiple sub-queries on its own, so a "
            "well-scoped prompt yields a tighter synthesis.\n\n"
            "Good: 'Compare BM25 vs dense retrieval for short-document search'.\n"
            "Bad: 'tell me about search'."
        ),
    )
    depth: Literal["basic", "deep"] = Field(
        "basic",
        description=(
            "'basic' — single-shot agent loop (~10-30s, 1-3 rounds). Good "
            "default. Returns synthesised answer + cited sources.\n"
            "'deep' — multi-step iterative research (1-10 min, up to 8 "
            "rounds). Higher quality but slow and costly. Pick this only "
            "when the user explicitly asks for thorough multi-source "
            "synthesis."
        ),
    )
    max_rounds: int | None = Field(
        None,
        ge=1,
        le=8,
        description=(
            "Override the round budget. basic accepts 1-3, deep accepts "
            "1-8. Leave None to use the server-side default for the chosen "
            "depth."
        ),
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
    args_schema: type[BaseModel] = BrimeResearchInput  # type: ignore[assignment,unused-ignore]
    handle_tool_error: bool = True  # type: ignore[assignment,unused-ignore]

    api_key: SecretStr | None = Field(default=None, exclude=True)
    base_url: str | None = None
    timeout: float = 60.0
    deep_poll_timeout: float = 420.0
    deep_poll_interval: float = 8.0

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
        depth: Literal["basic", "deep"] = "basic",
        max_rounds: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
        **_: Any,
    ) -> str:
        c = self._get_sync()
        try:
            if depth == "basic":
                res_basic = c.research(query, depth="basic", max_rounds=max_rounds)
                return _format_basic(res_basic)  # type: ignore[arg-type]
            res_deep = c.research(
                query,
                depth="deep",
                max_rounds=max_rounds,
                wait=True,
                poll_interval=self.deep_poll_interval,
                poll_timeout=self.deep_poll_timeout,
            )
        except BrimeError as exc:
            raise brime_to_tool_exception(exc) from exc
        except ToolException:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolException(f"Unexpected Brime error: {exc}") from exc
        return _format_deep(res_deep)  # type: ignore[arg-type]

    async def _arun(
        self,
        query: str,
        depth: Literal["basic", "deep"] = "basic",
        max_rounds: int | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **_: Any,
    ) -> str:
        c = self._get_async()
        try:
            if depth == "basic":
                res_basic = await c.research(query, depth="basic", max_rounds=max_rounds)
                return _format_basic(res_basic)  # type: ignore[arg-type]
            res_deep = await c.research(
                query,
                depth="deep",
                max_rounds=max_rounds,
                wait=True,
                poll_interval=self.deep_poll_interval,
                poll_timeout=self.deep_poll_timeout,
            )
        except BrimeError as exc:
            raise brime_to_tool_exception(exc) from exc
        except ToolException:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolException(f"Unexpected Brime error: {exc}") from exc
        return _format_deep(res_deep)  # type: ignore[arg-type]


def _format_basic(res: ResearchBasicResponse) -> str:
    lines: list[str] = []
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
    lines: list[str] = []
    lines.append((res.answer or "(no answer)").strip())
    lines.append("")
    lines.append(
        f"_Rounds: {res.current_round}/{res.max_rounds} · "
        f"Sources: {res.sources_count} · Steps: {res.steps_count}_"
    )
    return "\n".join(lines)


__all__: list[str] = ["BrimeResearch", "BrimeResearchInput"]
