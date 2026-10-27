"""FastAPI app: one endpoint that runs retrieval + the LLM and maps failures to clear errors."""

import logging

import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .llm import InvalidLLMOutput, triage
from .rag import Retriever
from .schemas import Source, TriageRequest, TriageResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("triage")

app = FastAPI(title="Support triage demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # the React dev server
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

retriever = Retriever()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "mock" if config.MOCK_MODE else "openai"}


@app.post("/triage", response_model=TriageResponse)
def triage_message(request: TriageRequest) -> TriageResponse:
    try:
        hits = retriever.search(request.message)
        result = triage(request.message, [chunk for chunk, _ in hits])
    # The SDK has already retried these with backoff; now turn each into a clear message.
    except openai.AuthenticationError:
        log.exception("OpenAI rejected the API key")
        raise HTTPException(500, "The AI service is not configured correctly.")
    except openai.RateLimitError as error:
        # OpenAI also uses 429 for an empty account; retrying won't help, so say so.
        if error.code in ("insufficient_quota", "credit_balance_exhausted") or getattr(error, "type", None) == "insufficient_quota":
            log.error("OpenAI account has no credits left")
            raise HTTPException(503, "The AI service is unavailable right now (usage limit reached). Please contact support.")
        log.warning("OpenAI rate limit hit")
        raise HTTPException(429, "The AI service is busy. Please try again in a minute.")
    except openai.APITimeoutError:
        log.warning("OpenAI request timed out")
        raise HTTPException(504, "The AI service took too long to respond. Please try again.")
    except (openai.APIConnectionError, openai.InternalServerError):
        log.warning("OpenAI unavailable")
        raise HTTPException(503, "The AI service is temporarily unavailable. Please try again.")
    except InvalidLLMOutput:
        log.error("Model output failed schema validation twice")
        raise HTTPException(502, "The AI returned an unexpected answer. Please try again.")

    sources = [Source(document=c.document, section=c.section, score=s) for c, s in hits]
    log.info("Triaged as %s/%s from %d sources", result.category, result.priority, len(sources))
    return TriageResponse(result=result, sources=sources, mode="mock" if config.MOCK_MODE else "openai")
