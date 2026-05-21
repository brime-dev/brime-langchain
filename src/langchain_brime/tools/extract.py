"""BrimeExtract — LangChain BaseTool wrapping POST /v1/extract."""

from __future__ import annotations

from typing import Any, List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from brime.models.extract import ExtractResponse
from langchain_brime._client import _ClientHolder


class BrimeExtractInput(BaseModel):
    """Schema exposed to the LLM for the brime_extract tool."""

    urls: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="One or more URLs to fetch and convert to clean markdown (max 10).",
    )
    include_metadata: bool = Field(
        False,
        description="Include extra metadata fields per result (slightly larger payload).",
    )


class BrimeExtract(BaseTool):
    """Fetch URLs and return clean markdown.

    Use after BrimeSearch when the agent needs the full body of specific
    pages (PDFs, HTML, DOCX, SPA hydration). Browser tier (Chrome) escalates
    automatically when needed.
    """

    name: str = "brime_extract"
    description: str = (
        "Fetch up to 10 URLs and return clean markdown for each. Use this "
        "to read the full body of pages found via brime_search. Failed "
        "URLs are reported with a reason at the bottom of the output."
    )
    args_schema: Type[BaseModel] = BrimeExtractInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0

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
        urls: List[str],
        include_metadata: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        res = self._client().sync.extract(urls, include_metadata=include_metadata)
        return _format_extract(res)

    async def _arun(
        self,
        urls: List[str],
        include_metadata: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        res = await self._client().async_.extract(urls, include_metadata=include_metadata)
        return _format_extract(res)


def _format_extract(res: ExtractResponse) -> str:
    parts: List[str] = []
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
