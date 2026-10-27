"""The contract: what the API accepts and what it returns."""

from typing import Literal

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """What the user submits. FastAPI rejects anything else with a 422."""

    message: str = Field(min_length=5, max_length=2000)


class TriageResult(BaseModel):
    """The structured output the LLM must fill in, field by field.

    Literal types limit the model to a fixed set of values, so the
    frontend (or any other system) can rely on them.
    """

    category: Literal["refund", "shipping", "warranty", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str = Field(description="One sentence describing the customer's problem")
    suggested_reply: str = Field(description="A short, polite reply based only on the policy text")
    policy_found: bool = Field(description="False if the policy text does not cover this question")


class Source(BaseModel):
    """A policy section the answer was based on (from retrieval, not the LLM)."""

    document: str
    section: str
    score: float


class TriageResponse(BaseModel):
    """What the API sends back to the UI."""

    result: TriageResult
    sources: list[Source]
    mode: Literal["openai", "mock"]
