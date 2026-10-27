# Support triage demo

A small AI feature: a customer message goes in, and a structured result comes out
(category, priority, summary, suggested reply), grounded in the shop's policy documents.

```
React UI  →  POST /triage (FastAPI)  →  RAG: find 2 relevant policy sections
                                      →  OpenAI structured output (TriageResult schema)
          ←  { result, sources, mode }
```

## Run it

Backend (port 8000):

```bash
cd backend
cp .env.example .env   # add your OpenAI key; leave it empty for mock mode
uv run uvicorn app.main:app --reload --port 8000
```

Frontend (port 5173):

```bash
cd frontend
npm install
npm run dev
```

Tests (run offline in mock mode):

```bash
cd backend
uv run pytest -q
```

## Where things live

| File | What it does |
|---|---|
| `backend/docs/*.md` | Reference documents, split into one chunk per `##` section |
| `backend/app/schemas.py` | The contract: request, LLM output (`TriageResult`) and response |
| `backend/app/rag.py` | Retrieval: OpenAI embeddings + cosine similarity (word overlap in mock mode) |
| `backend/app/llm.py` | `responses.parse(text_format=TriageResult)`, low temperature, one retry on bad output |
| `backend/app/main.py` | The endpoint, plus OpenAI errors mapped to clear HTTP errors |
| `backend/tests/test_api.py` | Happy path, "not covered", validation and every failure path |
| `frontend/src/App.tsx` | Form, loading state, result card, error message |

## Reliability choices

- **Structured output**: the model must fill the `TriageResult` schema; `Literal` fields limit it to fixed values.
- **No invented citations**: `sources` come from retrieval, never from the model.
- **Grounding**: the prompt allows only the retrieved policy text, with a `policy_found: false` path instead of guessing.
- **Retries**: the SDK retries rate limits, timeouts and 5xx with backoff (`max_retries=2`, 20 s timeout); schema failures retry once.
- **Clear errors**: 429 busy, 504 timeout, 503 unavailable, 502 bad AI output, 422 invalid input.
