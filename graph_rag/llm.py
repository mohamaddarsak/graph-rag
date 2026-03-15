"""
Single LLM call (OpenRouter or Ollama). Shared by RAG answer and step-back query generation.
"""
from __future__ import annotations

import ollama
from openai import OpenAI

from graph_rag.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, RAG_LLM_MODEL


def generate(prompt: str) -> str:
    """Generate text using OpenRouter (if API key set) or Ollama."""
    if OPENROUTER_API_KEY:
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        resp = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        if not resp.choices:
            return ""
        msg = resp.choices[0].message
        content = getattr(msg, "content", None) if msg else None
        return (content or "").strip()
    return (
        ollama.chat(model=RAG_LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        .message.content or ""
    ).strip()
