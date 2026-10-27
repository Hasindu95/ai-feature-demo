"""Settings, read once from environment variables (or backend/.env)."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# No key (or MOCK_MODE=1) -> run fully offline with fake answers, for demos and tests.
MOCK_MODE = os.getenv("MOCK_MODE") == "1" or not OPENAI_API_KEY


def get_client() -> OpenAI:
    # timeout: give up on a hung request after 20 s.
    # max_retries: the SDK retries rate limits, timeouts and 5xx errors
    # with exponential backoff before raising.
    return OpenAI(api_key=OPENAI_API_KEY, timeout=20.0, max_retries=2)
