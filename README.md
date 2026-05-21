# langchain-brime

[![PyPI](https://img.shields.io/pypi/v/langchain-brime.svg)](https://pypi.org/project/langchain-brime/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-brime.svg)](https://pypi.org/project/langchain-brime/)
[![Typed](https://img.shields.io/badge/typed-py.typed-blue.svg)](https://peps.python.org/pep-0561/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Brime API](https://brime.dev) · [Docs](https://docs.brime.dev) · [LangChain guide](https://docs.brime.dev/sdks/langchain) · [Python SDK](https://pypi.org/project/brime/) · [Issues](https://github.com/brime-dev/brime-langchain/issues)

**LangChain meets the live web.** Drop the Brime API into any LangChain agent, chain, or RAG pipeline — as three typed tools and a retriever you can plug in unchanged.

```bash
pip install langchain-brime
```

Python 3.9+. Sync and async. Fully typed (`py.typed`). `SecretStr` API keys. Actionable agent errors via `handle_tool_error`. Passes the official LangChain `ToolsUnitTests` compliance suite.

## What you get

| Class            | Type            | What it does                                                |
|------------------|-----------------|-------------------------------------------------------------|
| `BrimeSearch`    | `BaseTool`      | Fast SERP search with snippets + synthesised answer.        |
| `BrimeExtract`   | `BaseTool`      | Any URL → clean markdown (HTML / PDF / DOCX / SPA).         |
| `BrimeResearch`  | `BaseTool`      | Multi-round agent research with citations.                  |
| `BrimeRetriever` | `BaseRetriever` | Live-web retriever for RAG — returns LangChain `Document`s. |

One API key powers all four. One set of tuned defaults. No knobs to babysit.

## 30 seconds

```python
from langchain.agents import create_agent
from langchain_brime import BrimeSearch, BrimeExtract

agent = create_agent(
    model="anthropic:claude-3-5-haiku-latest",
    tools=[BrimeSearch(), BrimeExtract()],
    system_prompt="You are a research assistant.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is BM25 ranking? Cite sources."}],
})
print(result["messages"][-1].content)
```

## RAG without the RAG plumbing

`BrimeRetriever` returns standard LangChain `Document`s — drop it into any retrieval chain and you're querying the live web with BM25-ranked chunks:

```python
from langchain_brime import BrimeRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

retriever = BrimeRetriever(k=5)
llm = init_chat_model("anthropic:claude-3-5-haiku-latest")

prompt = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations."
)
chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm)

print(chain.invoke("How does BM25 weight term frequency?").content)
```

Each `Document` carries `page_content` (ranked chunk) and rich metadata: `url`, `title`, `score`, `published_date`, `source="brime"`.

## Async, end-to-end

```python
import asyncio
from langchain_brime import BrimeSearch

async def main():
    result = await BrimeSearch().ainvoke({"query": "python async io patterns"})
    print(result)

asyncio.run(main())
```

Every tool has `_arun`. The retriever has `_aget_relevant_documents`. No surprises.

## Authentication

```bash
export BRIME_API_KEY="sk-brime-..."
```

```python
BrimeSearch()                        # uses BRIME_API_KEY
BrimeSearch(api_key="sk-brime-...")  # explicit
BrimeSearch(base_url="https://...")  # staging override (or BRIME_BASE_URL)
```

Get a key at [brime.dev](https://brime.dev) — free tier: 1,000 credits/month, no card.

## Tool reference

### `BrimeSearch`

| Field         | Default     | Notes                                              |
|---------------|-------------|----------------------------------------------------|
| `query`       | (required)  | Natural-language query                             |
| `topic`       | `"general"` | `general` · `news` · `finance` (recency hint)      |
| `max_results` | 5           | 1–20                                               |
| `time_range`  | None        | `day` · `week` · `month` · `year`                  |

Returns a markdown string with a synthesised answer plus a `Sources:` list.

### `BrimeExtract`

| Field              | Default | Notes                                    |
|--------------------|---------|------------------------------------------|
| `urls`             | (req.)  | List of URLs (1–10)                      |
| `include_metadata` | False   | Add extra metadata fields per result     |

Returns markdown with a `## <url>` heading per page, plus a `## Failed URLs` block if any failed.

### `BrimeResearch`

| Field         | Default     | Notes                                       |
|---------------|-------------|---------------------------------------------|
| `query`       | (required)  | Research question                           |
| `topic`       | `"general"` | Recency hint                                |
| `max_results` | 5           | Sources per round, 1–20                     |

Single tuned multi-round flow (~30–90 s) with cited synthesis.

### `BrimeRetriever`

| Field             | Default     | Notes                                                |
|-------------------|-------------|------------------------------------------------------|
| `k`               | 5           | Max documents (1–20)                                 |
| `topic`           | `"general"` | Recency hint                                         |
| `time_range`      | None        | Recency window                                       |
| `domains`         | None        | Domain allowlist                                     |
| `exclude_domains` | None        | Domain denylist                                      |
| `include_answer`  | False       | Skip the LLM answer when used purely for retrieval   |

## Why this instead of writing your own wrapper?

- **One package, four classes.** Search, extract, research, and retriever — all sharing one HTTP client (`brime` SDK).
- **Idempotency, handled.** Auto `Idempotency-Key` for `extract` (Brime API requirement, easy to miss).
- **Async parity throughout.** No partially-async surface area.
- **LangChain-tested.** Passes the official `ToolsUnitTests` compliance suite (args schema serialisable, async parity, no override violations).

## Compatibility

- Python 3.9+
- `langchain-core` ≥ 1.4, < 2.0
- Backed by [`brime`](https://pypi.org/project/brime/) Python SDK ≥ 0.2.0 (retry, request_id, typed error hierarchy)

## Links

- Docs — [docs.brime.dev](https://docs.brime.dev)
- LangChain integration guide — [docs.brime.dev/sdks/langchain](https://docs.brime.dev/sdks/langchain)
- Status — [brime.dev](https://brime.dev)
- Issues — [github.com/brime-dev/brime-langchain/issues](https://github.com/brime-dev/brime-langchain/issues)

## License

MIT © Brime
