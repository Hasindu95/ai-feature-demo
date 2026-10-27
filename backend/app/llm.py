"""The LLM step: message + policy sections in, a validated TriageResult out."""

from pydantic import ValidationError

from . import config
from .rag import Chunk
from .schemas import TriageResult

INSTRUCTIONS = """You triage customer support messages for an online electronics shop.
Use ONLY the policy sections provided. Do not invent policies, prices or timeframes.
If the sections do not answer the customer's question, set policy_found to false,
set category to "other", and write a reply saying a team member will follow up.
Priority: high = money lost, damaged item or lost parcel; medium = a claim or change
request; low = a general question."""


class InvalidLLMOutput(Exception):
    """The model's answer did not match the schema, even after a retry."""


def _prompt(message: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(chunk.text for chunk in chunks)
    return f"Policy sections:\n{context}\n\nCustomer message:\n{message}"


def triage(message: str, chunks: list[Chunk]) -> TriageResult:
    if config.MOCK_MODE:
        return _mock_triage(message, chunks)

    client = config.get_client()
    for _attempt in range(2):  # one retry if the output doesn't fit the schema
        try:
            response = client.responses.parse(
                model=config.CHAT_MODEL,
                instructions=INSTRUCTIONS,
                input=_prompt(message, chunks),
                text_format=TriageResult,  # structured output: forces our JSON schema
                temperature=0.2,  # low = consistent, not creative
            )
            if response.output_parsed is not None:
                return response.output_parsed
        except ValidationError:
            pass
    raise InvalidLLMOutput("The model returned output that did not match the schema.")


def _mock_triage(message: str, chunks: list[Chunk]) -> TriageResult:
    """Offline stand-in for the LLM, so the demo and tests run without an API key."""
    top = chunks[0] if chunks else None
    category = {"Refund policy": "refund", "Shipping policy": "shipping", "Warranty policy": "warranty"}
    urgent = any(w in message.lower() for w in ("damaged", "broken", "lost", "not arrived", "never arrived", "charged"))
    if top is None:
        return TriageResult(
            category="other",
            priority="low",
            summary="The message does not match any policy.",
            suggested_reply="Thanks for reaching out. A team member will follow up shortly.",
            policy_found=False,
        )
    return TriageResult(
        category=category.get(top.document, "other"),
        priority="high" if urgent else "medium",
        summary=f"Customer question related to: {top.section.lower()}.",
        suggested_reply=f"Thanks for contacting us. {top.text.split(': ', 1)[1]}",
        policy_found=True,
    )
