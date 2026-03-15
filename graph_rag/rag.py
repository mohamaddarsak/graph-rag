"""
RAG: retrieve context via hybrid search (with optional step-back for vector), then generate answer.
"""
from __future__ import annotations

import logging

from graph_rag.llm import generate
from graph_rag.neo4j_io import get_neo4j_driver
from graph_rag.retrieval import retrieve_for_rag
from graph_rag.stepback import generate_step_back_question

log = logging.getLogger(__name__)


def _generate_answer(prompt: str) -> str:
    """Generate answer using OpenRouter (if API key set) or Ollama."""
    return generate(prompt)


def answer(query: str, top_k: int = 5) -> str:
    """RAG: step-back for vector retrieval, original query for fulltext and answer; then generate answer."""
    driver = get_neo4j_driver()
    try:
        step_back = generate_step_back_question(query)
        hits = retrieve_for_rag(driver, query, top_k=top_k, query_for_vector=step_back)
        if not hits:
            return "No relevant context found in the knowledge base."
        context = "\n\n".join(f"[{i+1}] {text}" for i, (text, _) in enumerate(hits))
        prompt = f"""Use only the following context to answer the question. If the context does not contain the answer, say so briefly.

Context:
{context}

Question: {query}

Answer:"""
        return _generate_answer(prompt)
    finally:
        driver.close()
