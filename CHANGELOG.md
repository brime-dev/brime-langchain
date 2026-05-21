# Changelog

## 0.2.0 — 2026-05-21

A-tier polish. No breaking API changes for normal use, but `api_key` is now
typed as `SecretStr` — passing a plain `str` still works (it is coerced on
construction) and the field is hidden from `repr`/`model_dump`.

### Added
- `api_key: SecretStr | None` on every tool and the retriever — keys never
  leak via `repr(tool)`, `str(tool)`, or `tool.model_dump()`
- `model_validator(mode="before")` resolves the key chain: explicit arg →
  `BRIME_API_KEY` env → `None`
- `handle_tool_error = True` on every tool — Brime SDK exceptions are mapped
  to actionable `ToolException` messages the LLM can react to
  (auth → "ask the user for a fresh key", rate-limit → "wait Xs", insufficient
  credits → billing link, timeout → "narrow the scope", upstream-degraded →
  "retry shortly")
- `langchain_brime._utilities` module with `resolve_brime_api_key`,
  `build_sync_client`, `build_async_client` (Exa-style canonical helpers)
- `langchain_brime._errors.brime_to_tool_exception` — SDK error → agent
  message mapping
- Rich Tavily-grade `Field(description=...)` on every tool input — when/why
  guidance the LLM reads at tool-selection time
- `Document.id` is now set to the result URL (LangChain 1.x dedup convention)
- New unit tests: `test_imports.py` (public-surface drift guard),
  `test_secret_handling.py` (SecretStr no-leak guarantees — 20 cases)
- GitHub Actions: `release-langchain-brime.yml` (tag-driven OIDC Trusted
  Publishing, no `PYPI_TOKEN` secret) and `ci-langchain-brime.yml`
  (ruff + pyright + mypy + pytest + uv build + twine check matrix)
- `__version__` now resolved via `importlib.metadata` — no hand-maintained
  `_version.py`
- `uv.lock` committed for reproducible CI builds

### Changed
- `langchain-core` floor raised to `>=1.4,<2.0` (modal 2026 floor; matches
  `langchain-tests` 1.x requirements)
- `brime` SDK floor raised to `>=0.2.0` — inherits retry, request_id, and the
  typed error hierarchy
- Internal client construction is now lazy per tool/retriever instance;
  removed the `_client._ClientHolder` shim in favour of direct
  `Brime` / `AsyncBrime` ownership on the class

### Removed
- `langchain_brime._version` (replaced by `importlib.metadata`)
- `langchain_brime._client._ClientHolder` (folded into `_utilities`)

## 0.1.0 — 2026-05-06

Initial release. Beta.

### Added
- `BrimeSearch`, `BrimeExtract`, `BrimeResearch` — `BaseTool` subclasses with sync + async (`_run` / `_arun`)
- `BrimeRetriever` — `BaseRetriever` returning `Document` objects with url/title/score/published_date metadata
- Pydantic v2 `args_schema` for every tool (LLM-friendly JSON Schema)
- Backed by the [`brime`](https://pypi.org/project/brime/) Python SDK — single HTTP client + error hierarchy across the integration
- `BRIME_API_KEY` and `BRIME_BASE_URL` env-var fallbacks (inherited from the `brime` SDK)
- Deep research polling with `poll_timeout` and exponential backoff
- Auto-generated `Idempotency-Key` for `extract` and deep `research` calls
- Passes LangChain `ToolsUnitTests` official compliance suite (3 tools × 6 tests = 18 PASS)
- Live e2e validated: tool wiring via tool_calls, retriever in LCEL chain
