import os

os.environ["MOCK_MODE"] = "1"  # never call the real API from tests

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app import config, llm, main
from app.rag import Retriever

client = TestClient(main.app)
FAKE_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


def test_health_reports_mock_mode():
    assert client.get("/health").json() == {"status": "ok", "mode": "mock"}


def test_refund_message_returns_structured_result_with_sources():
    response = client.post("/triage", json={"message": "My headphones arrived damaged, I want a refund"})
    body = response.json()
    assert response.status_code == 200
    assert body["result"]["category"] == "refund"
    assert body["result"]["priority"] == "high"
    assert body["sources"][0]["document"] == "Refund policy"


def test_unrelated_message_is_marked_not_found():
    body = client.post("/triage", json={"message": "Do you sell pizza?"}).json()
    assert body["result"]["policy_found"] is False
    assert body["sources"] == []


def test_too_short_message_is_rejected():
    assert client.post("/triage", json={"message": "hi"}).status_code == 422


def test_retriever_splits_docs_into_sections():
    sections = {c.section for c in Retriever().chunks}
    assert {"Refund window", "Late or lost parcels", "Battery issues"} <= sections


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (openai.RateLimitError("slow down", response=httpx.Response(429, request=FAKE_REQUEST), body=None), 429),
        (openai.APITimeoutError(request=FAKE_REQUEST), 504),
        (openai.APIConnectionError(request=FAKE_REQUEST), 503),
        (llm.InvalidLLMOutput(), 502),
    ],
)
def test_api_failures_become_clear_errors(monkeypatch, error, status):
    def fail(*_args):
        raise error

    monkeypatch.setattr(main, "triage", fail)
    response = client.post("/triage", json={"message": "Where is my parcel?"})
    assert response.status_code == status
    assert "try again" in response.json()["detail"]


def test_out_of_credits_is_not_reported_as_busy(monkeypatch):
    error = openai.RateLimitError(
        "no credits",
        response=httpx.Response(429, request=FAKE_REQUEST),
        body={"code": "credit_balance_exhausted", "type": "insufficient_quota"},
    )

    def fail(*_args):
        raise error

    monkeypatch.setattr(main, "triage", fail)
    response = client.post("/triage", json={"message": "Where is my parcel?"})
    assert response.status_code == 503
    assert "usage limit" in response.json()["detail"]


def test_bad_model_output_is_retried_once_then_rejected(monkeypatch):
    calls = []

    class FakeResponses:
        def parse(self, **_kwargs):
            calls.append(1)
            return type("Parsed", (), {"output_parsed": None})()  # output that didn't fit the schema

    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(config, "get_client", lambda: type("Client", (), {"responses": FakeResponses()})())
    with pytest.raises(llm.InvalidLLMOutput):
        llm.triage("Where is my parcel?", [])
    assert len(calls) == 2
