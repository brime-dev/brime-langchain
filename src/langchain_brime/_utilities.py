"""Shared helpers for the LangChain tools and retriever.

Each tool/retriever class:
  - Declares `api_key: Optional[SecretStr]` so the API key is hidden in
    ``__repr__`` and serialised dumps.
  - Runs ``model_validator(mode="before")`` which calls
    :func:`resolve_brime_api_key` to fall back to the ``BRIME_API_KEY`` env
    var and coerce raw strings into ``SecretStr``.
  - Builds the underlying ``brime.Brime`` / ``brime.AsyncBrime`` client
    lazily on first access via :func:`build_sync_client` /
    :func:`build_async_client` so importing the package is cheap.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from brime.async_client import AsyncBrime
from brime.client import Brime
from pydantic import SecretStr


def _package_version() -> str:
    try:
        return version("langchain-brime")
    except PackageNotFoundError:  # pragma: no cover
        return "0.0.0+unknown"


CLIENT_SOURCE_HEADER = f"langchain-brime/{_package_version()}"


def resolve_brime_api_key(values: dict[str, Any]) -> dict[str, Any]:
    """``model_validator(mode="before")`` helper.

    - If ``api_key`` is missing or empty, fall back to ``BRIME_API_KEY``
      from the environment.
    - Coerce any plain-string ``api_key`` value to ``SecretStr`` so the
      hidden-repr guarantee holds even when callers pass raw strings.
    - Leave a missing key as ``None``; the underlying ``brime.Brime``
      constructor surfaces a clear error on first call.
    """
    raw: Any = values.get("api_key")
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raw = os.environ.get("BRIME_API_KEY")
        if raw is not None and raw.strip() == "":
            raw = None

    if isinstance(raw, SecretStr):
        values["api_key"] = raw
    elif isinstance(raw, str):
        values["api_key"] = SecretStr(raw)
    else:
        values["api_key"] = None
    return values


def _api_key_str(api_key: SecretStr | None) -> str | None:
    return api_key.get_secret_value() if api_key is not None else None


def build_sync_client(
    api_key: SecretStr | None,
    base_url: str | None,
    timeout: float,
) -> Brime:
    return Brime(
        api_key=_api_key_str(api_key),
        base_url=base_url,
        timeout=timeout,
    )


def build_async_client(
    api_key: SecretStr | None,
    base_url: str | None,
    timeout: float,
) -> AsyncBrime:
    return AsyncBrime(
        api_key=_api_key_str(api_key),
        base_url=base_url,
        timeout=timeout,
    )
