"""
Ollama BGE-M3 embeddings for chunks and queries.
"""
from __future__ import annotations

import ollama

from graph_rag.config import EMBED_MODEL_OLLAMA


def embed_texts(texts: list[str], model: str = EMBED_MODEL_OLLAMA) -> list[list[float]]:
    """Embed texts using Ollama BGE-M3. Returns list of 1024-dim vectors."""
    vectors: list[list[float]] = []
    for t in texts:
        r = ollama.embed(model=model, input=t)
        emb = r.embeddings[0] if r.embeddings else []
        vectors.append(emb)
    return vectors
