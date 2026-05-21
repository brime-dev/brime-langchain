"""Map Brime SDK exceptions onto agent-actionable ``ToolException`` messages.

When ``handle_tool_error = True`` is set on a ``BaseTool`` subclass, the
framework catches ``ToolException`` and returns its message to the LLM
verbatim. We use that to deliver next-step guidance — "ask the user for a
fresh key", "back off and retry", etc. — instead of leaking a raw traceback.
"""

from __future__ import annotations

from brime.errors import (
    AuthenticationError,
    BrimeError,
    InsufficientCreditsError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    UpstreamError,
)
from brime.errors import (
    ConnectionError as BrimeConnectionError,
)
from brime.errors import (
    TimeoutError as BrimeTimeoutError,
)
from langchain_core.tools import ToolException


def brime_to_tool_exception(exc: BrimeError) -> ToolException:
    """Translate a Brime SDK exception into a ``ToolException`` for agents.

    Each branch produces an actionable, second-person message the agent
    can either surface to the user or react to autonomously.
    """
    request_tail = f" (request_id={exc.request_id})" if exc.request_id else ""
    if isinstance(exc, AuthenticationError):
        return ToolException(
            "Brime API key is invalid or revoked. Ask the user for a fresh "
            "key (starts with `sk-brime-`) and retry." + request_tail
        )
    if isinstance(exc, RateLimitError):
        wait = exc.retry_after
        if wait is not None:
            return ToolException(
                f"Brime rate limit hit. Wait {wait}s and retry the same query." + request_tail
            )
        return ToolException(
            "Brime rate limit hit. Wait a short moment before retrying." + request_tail
        )
    if isinstance(exc, InsufficientCreditsError):
        return ToolException(
            "Brime account is out of credits for this billing period. "
            "Ask the user to top up at https://brime.dev/billing." + request_tail
        )
    if isinstance(exc, InvalidRequestError):
        return ToolException(
            f"Brime rejected the request as invalid ({exc.code}). "
            f"Re-read the tool args_schema and try again. Detail: {exc}" + request_tail
        )
    if isinstance(exc, NotFoundError):
        return ToolException(f"Brime resource not found: {exc}{request_tail}")
    if isinstance(exc, BrimeTimeoutError):
        return ToolException(
            "Brime request timed out. Try again with a narrower scope "
            "(fewer URLs, shorter query, or lower max_results)." + request_tail
        )
    if isinstance(exc, BrimeConnectionError):
        return ToolException(
            "Could not reach Brime over the network. Check connectivity "
            "and retry shortly." + request_tail
        )
    if isinstance(exc, UpstreamError):
        return ToolException(
            "Brime upstream is temporarily degraded. Retry in a few seconds." + request_tail
        )
    return ToolException(f"Brime API error ({exc.code}): {exc}{request_tail}")
