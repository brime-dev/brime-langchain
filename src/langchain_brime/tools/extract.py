"""BrimeExtract — LangChain BaseTool wrapping POST /v1/extract."""

from __future__ import annotations

from typing import Any, cast

from brime.async_client import AsyncBrime
from brime.client import Brime
from brime.errors import BrimeError
from brime.models.extract import ExtractResponse
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


class BrimeExtractInput(BaseModel):
    """Schema exposed to the LLM for the brime_extract tool."""

    urls: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "One to ten absolute URLs to fetch and convert to clean markdown.\n\n"
            "Use this AFTER brime_search has produced a ranked list — feed the "
            "URLs you actually want to read in full. Do not pass URLs you only "
            "want a snippet from; brime_search already gives you snippets."
        ),
    )
    include_metadata: bool = Field(
        False,
        description=(
            "When true, include extra per-result metadata (Open Graph, "
            "canonical URL, language). Leave false unless the user asks "
            "for structured metadata — the extra payload wastes tokens."
        ),
    )


class BrimeExtract(BaseTool):
    """Fetch URLs and return clean markdown.

    Use after brime_search when the agent needs the full body of specific
    pages (PDFs, HTML, DOCX, SPA hydration). Browser tier (Chrome) escalates
    automatically when needed.
    """

    name: str = "brime_extract"
    description: str = (
        "Fetch up to 10 URLs and return clean markdown for each. Use this "
        "to read the full body of pages found via brime_search. Failed "
        "URLs are reported with a reason at the bottom of the output."
    )
    args_schema: type[BaseModel] = BrimeExtractInput  # type: ignore[assignment,unused-ignore]
    handle_tool_error: bool = True  # type: ignore[assignment,unused-ignore]

    api_key: SecretStr | None = Field(default=None, exclude=True)
    base_url: str | None = None
    timeout: float = 60.0

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
        urls: list[str],
        include_metadata: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
        **_: Any,
    ) -> str:
        try:
            res = self._get_sync().extract(urls, include_metadata=include_metadata)
        except BrimeError as exc:
            raise brime_to_tool_exception(exc) from exc
        except ToolException:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolException(f"Unexpected Brime error: {exc}") from exc
        return _format_extract(res)

    async def _arun(
        self,
        urls: list[str],
        include_metadata: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **_: Any,
    ) -> str:
        try:
            res = await self._get_async().extract(urls, include_metadata=include_metadata)
        except BrimeError as exc:
            raise brime_to_tool_exception(exc) from exc
        except ToolException:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolException(f"Unexpected Brime error: {exc}") from exc
        return _format_extract(res)


def _format_extract(res: ExtractResponse) -> str:
    parts: list[str] = []
    for r in res.results:
        parts.append(f"## {r.url}")
        parts.append(f"_method={r.method}, content_type={r.content_type}_")
        parts.append("")
        parts.append(r.markdown.strip())
        parts.append("")
    if res.failed:
        parts.append("## Failed URLs")
        for f in res.failed:
            parts.append(f"- {f.url}: {f.error.code} — {f.error.message}")
    return "\n".join(parts).rstrip() or "(no content extracted)"


__all__ = ["BrimeExtract", "BrimeExtractInput"]
