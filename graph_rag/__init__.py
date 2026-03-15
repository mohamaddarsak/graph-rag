"""
Graph RAG: docs/ → Docling → HybridChunker → embed (Ollama BGE-M3) → Neo4j.
Query: hybrid search (vector + full-text) + RRF → RAG (OpenRouter or Ollama).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from graph_rag.chunking import create_chunks
from graph_rag.config import DOCS_DIR
from graph_rag.embeddings import embed_texts
from graph_rag.graph import build_graph_data
from graph_rag.neo4j_io import get_neo4j_driver, insert_into_neo4j
from graph_rag.parsing import parse_document
from graph_rag.postprocess import post_process_chunks
from graph_rag.rag import answer
from graph_rag.summary import print_summary

log = logging.getLogger(__name__)

# File extensions Docling can convert (document formats)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".html", ".htm", ".csv", ".xlsx", ".pptx"}


def _discover_sources(docs_path: Path) -> list[Path]:
    """Return sorted list of supported files under docs_path."""
    if not docs_path.is_dir():
        return []
    out = []
    for p in docs_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            out.append(p)
    return sorted(out)


def ingest(docs_path: Path | None = None, save_json_path: Path | None = None) -> None:
    """
    Process all supported files under docs_path (default DOCS_DIR), build graph with
    embeddings, clear existing graph and insert into Neo4j.
    """
    path = docs_path or DOCS_DIR
    sources = _discover_sources(path)
    if not sources:
        raise ValueError(
            f"No supported files found under {path}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    log.info("Found %s file(s) to process under %s", len(sources), path)
    all_nodes: list[dict] = []
    all_rels: list[dict] = []

    for source in sources:
        source_str = str(source)
        doc = parse_document(source_str)
        chunks, chunker = create_chunks(doc)
        preamble_text, chunks = post_process_chunks(chunks)
        nodes, rels = build_graph_data(doc, chunks, chunker, preamble_text)
        all_nodes.extend(nodes)
        all_rels.extend(rels)

    # Add embeddings to Chunk nodes (use contextualized_text for richer context)
    chunk_nodes = [n for n in all_nodes if "Chunk" in n["labels"]]
    texts_to_embed = [
        n["properties"].get("contextualized_text") or n["properties"]["text"]
        for n in chunk_nodes
    ]
    log.info("Embedding %s chunks with Ollama BGE-M3...", len(texts_to_embed))
    embeddings = embed_texts(texts_to_embed)
    for n, emb in zip(chunk_nodes, embeddings):
        n["properties"]["embedding"] = emb

    if save_json_path:
        with open(save_json_path, "w") as f:
            json.dump(
                {"nodes": all_nodes, "relationships": all_rels},
                f,
                indent=2,
                default=str,
            )
        log.info("Graph data saved to %s", save_json_path)

    print_summary(all_nodes, all_rels)

    # Clear existing graph and insert
    driver = get_neo4j_driver()
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            session.run("MATCH (c:Chunk) DETACH DELETE c")
            session.run("MATCH (s:Section) DETACH DELETE s")
            session.run("MATCH (d:Document) DETACH DELETE d")
    finally:
        driver.close()

    insert_into_neo4j(all_nodes, all_rels)


__all__ = ["ingest", "answer", "DOCS_DIR"]
