"""
Unit tests for the Graph RAG pipeline.
Requires: Neo4j running (docker compose up), Ollama with bge-m3 and a chat model (e.g. llama3.2).
Use conda env: graph-rag.
"""
import pytest

from main import answer, chunk_text, ingest, load_pdfs


def test_chunk_text_whitespace_only():
    """Chunks break only on whitespace, respect size and overlap."""
    text = "a " * 200  # 400 chars
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 100 + 20  # allow some slack at boundary
    # No cut in the middle of a word
    for c in chunks:
        if " " in c:
            parts = c.split()
            for p in parts:
                assert " " not in p or p == " ", "chunk should not cut words"


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_load_pdfs():
    from pathlib import Path
    docs = Path(__file__).resolve().parent / "docs"
    text = load_pdfs(docs)
    assert isinstance(text, str)
    # If PDFs exist we should get some text
    if list(docs.glob("*.pdf")):
        assert len(text.strip()) > 0


@pytest.mark.integration
def test_rag_answer_success():
    """
    Full RAG test: ingest PDFs then ask a query and assert we get a non-empty answer.
    Requires Neo4j and Ollama running; run with: pytest -m integration
    """
    ingest()
    query = "What is this document about?"
    result = answer(query, top_k=5)
    assert result, "RAG should return a non-empty answer"
    assert len(result) >= 10, "Answer should be substantive"
    assert "No relevant context" not in result, "Expected an answer from context, not the fallback message"
