# langchain-brime

LangChain integration for the [Brime API](https://brime.dev). Drop-in tools and a retriever for any LangChain agent or chain.

```bash
pip install langchain-brime
```

## What's in the box

| Class            | Type            | Wraps                  |
|------------------|-----------------|------------------------|
| `BrimeSearch`    | `BaseTool`      | `POST /v1/search`      |
| `BrimeExtract`   | `BaseTool`      | `POST /v1/extract`     |
| `BrimeResearch`  | `BaseTool`      | `POST /v1/research`    |
| `BrimeRetriever` | `BaseRetriever` | `POST /v1/search` → `Document` |

All tools and the retriever work in **sync and async** modes. The package is fully typed (`py.typed`) and passes the official `langchain-tests` `ToolsUnitTests` compliance suite.

## Quickstart

### As a tool in a LangChain agent

```python
from langchain.agents import create_agent
from langchain_brime import BrimeSearch, BrimeExtract

agent = create_agent(
    model="anthropic:claude-3-5-haiku-latest",
    tools=[BrimeSearch(), BrimeExtract()],
    system_prompt="You are a research assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is BM25 ranking? Cite sources."}]}
)
print(result["messages"][-1].content)
```

### As a retriever in a RAG chain

```python
from langchain_brime import BrimeRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

retriever = BrimeRetriever(k=5, depth="basic")
llm = init_chat_model("anthropic:claude-3-5-haiku-latest")
prompt = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations."
)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
print(chain.invoke("How does BM25 weight term frequency?").content)
```

### Async

```python
import asyncio
from langchain_brime import BrimeSearch

async def main():
    tool = BrimeSearch()
    result = await tool.ainvoke({"query": "python async io patterns"})
    print(result)

asyncio.run(main())
```

## Authentication

Either pass `api_key=` to the constructor, or set `BRIME_API_KEY` in your environment:

```bash
export BRIME_API_KEY="sk-brime-..."
```

```python
BrimeSearch(api_key="sk-brime-...")     # explicit
BrimeSearch()                            # picks up BRIME_API_KEY
BrimeSearch(base_url="https://...")      # staging override (or BRIME_BASE_URL env)
```

## Tool reference

### `BrimeSearch`

| Field         | Default     | Notes                                                              |
|---------------|-------------|--------------------------------------------------------------------|
| `query`       | (required)  | Natural-language search query                                      |
| `depth`       | `"basic"`   | `instant` (cheap snippets) · `basic` (LLM answer) · `advanced`     |
| `max_results` | 5           | 1–20                                                               |
| `topic`       | `"general"` | `general` · `news` · `finance` (recency hint)                      |
| `time_range`  | None        | `day` · `week` · `month` · `year`                                  |

Returns a markdown string: synthesised answer (when depth ≠ `instant`) plus a `Sources:` list.

### `BrimeExtract`

| Field              | Default | Notes                                              |
|--------------------|---------|----------------------------------------------------|
| `urls`             | (req.)  | List of URLs (1–10)                                |
| `include_metadata` | False   | Include extra metadata fields per result           |

Returns a markdown blob with one `## <url>` heading per fetched page, followed by a `## Failed URLs` section if any fetches failed.

### `BrimeResearch`

| Field         | Default   | Notes                                                                   |
|---------------|-----------|-------------------------------------------------------------------------|
| `query`       | (req.)    | Research question                                                       |
| `depth`       | `"basic"` | `basic` (10–30 s sync) · `deep` (1–10 min iterative)                    |
| `max_rounds`  | None      | basic 1–3, deep 1–8                                                     |

Deep mode auto-blocks until the job reaches a terminal state (`poll_timeout=420 s` by default; configure via the `deep_poll_timeout` constructor kwarg).

### `BrimeRetriever`

| Field             | Default     | Notes                                                |
|-------------------|-------------|------------------------------------------------------|
| `k`               | 5           | Max documents (1–20)                                 |
| `depth`           | `"basic"`   | Use `instant` for fast retrieval-only flows          |
| `topic`           | `"general"` | Recency hint                                         |
| `time_range`      | None        | Recency window                                       |
| `domains`         | None        | Domain allowlist                                     |
| `exclude_domains` | None        | Domain denylist                                      |
| `include_answer`  | False       | LLM answer is wasted when used purely for retrieval  |

Each result becomes a `Document` with:
- `page_content` = ranked content snippet
- `metadata` = `{url, title, score, published_date, source: "brime"}`

## Why use this instead of writing your own wrapper?

- One package, four classes — search, extract, two-mode research, and retriever, all sharing one HTTP client (`brime` SDK).
- Auto `Idempotency-Key` for `extract` and deep `research` (Brime API requirement, easy to miss).
- Deep research polling with exponential backoff and `poll_timeout` baked in.
- Async parity throughout: every tool has `_arun`; the retriever has `_aget_relevant_documents`.
- Passes the official LangChain `ToolsUnitTests` compliance suite (args schema serialisable, async parity, no override violations).

## Compatibility

- Python 3.9+
- `langchain-core>=0.3,<2.0` (works with both 0.3.x and 1.x)
- Backed by [`brime`](https://pypi.org/project/brime/) Python SDK >= 0.1.0

## License

MIT © Brime
