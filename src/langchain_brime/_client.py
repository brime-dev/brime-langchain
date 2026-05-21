"""Cached Brime client factory shared by tools and retriever.

Each tool/retriever instance owns one Brime (sync) and lazily one
AsyncBrime (async). Reuses httpx connection pools across calls.
"""

from __future__ import annotations

from typing import Optional

from brime.async_client import AsyncBrime
from brime.client import Brime


class _ClientHolder:
    """Per-instance container for sync + lazy async Brime clients."""

    __slots__ = ("_api_key", "_base_url", "_timeout", "_sync", "_async")

    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: float,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        # Eagerly construct sync (cheap; resolves api_key/base_url once).
        self._sync: Brime = Brime(api_key=api_key, base_url=base_url, timeout=timeout)
        self._async: Optional[AsyncBrime] = None

    @property
    def sync(self) -> Brime:
        return self._sync

    @property
    def async_(self) -> AsyncBrime:
        if self._async is None:
            self._async = AsyncBrime(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._async

    def close(self) -> None:
        try:
            self._sync.close()
        except Exception:  # pragma: no cover
            pass

    async def aclose(self) -> None:
        if self._async is not None:
            try:
                await self._async.aclose()
            except Exception:  # pragma: no cover
                pass
