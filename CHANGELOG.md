# Changelog

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
