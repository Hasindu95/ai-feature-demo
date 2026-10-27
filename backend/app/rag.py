"""Retrieval: find the policy sections most relevant to a message."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


@dataclass
class Chunk:
    document: str  # e.g. "Refund policy"
    section: str  # e.g. "Refund window"
    text: str


def load_chunks() -> list[Chunk]:
    """Split every docs/*.md file into one chunk per '## ' section."""
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        raw = path.read_text()
        title = raw.splitlines()[0].lstrip("# ").strip()
        for part in raw.split("\n## ")[1:]:
            heading, _, body = part.partition("\n")
            # Prefix the document title so a chunk still makes sense on its own.
            chunks.append(Chunk(title, heading.strip(), f"{title} - {heading.strip()}: {body.strip()}"))
    return chunks


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3}


class Retriever:
    def __init__(self) -> None:
        self.chunks = load_chunks()
        self._vectors: np.ndarray | None = None  # embedded lazily, on first real search

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = config.get_client().embeddings.create(model=config.EMBED_MODEL, input=texts)
        vectors = np.array([item.embedding for item in response.data])
        # Normalise so a dot product equals cosine similarity.
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def search(self, query: str, k: int = 2) -> list[tuple[Chunk, float]]:
        if config.MOCK_MODE:
            # Offline stand-in for embeddings: share of the query's words found in each chunk.
            query_words = _words(query) or {""}
            scores = [len(query_words & _words(c.text)) / len(query_words) for c in self.chunks]
        else:
            if self._vectors is None:
                self._vectors = self._embed([c.text for c in self.chunks])
            scores = (self._vectors @ self._embed([query])[0]).tolist()

        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        # Drop sections with nothing in common, so "no match" really means no context.
        return [(chunk, round(score, 3)) for chunk, score in ranked[:k] if score > 0]
