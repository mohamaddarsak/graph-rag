"""
Configuration and environment loading.
Load .env from project root; expose DOCS_DIR, Neo4j, embedding, RAG, and RRF settings.
"""
from __future__ import annotations

import os
from pathlib import Path

# Project root (parent of graph_rag package)
_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _ROOT / ".env"

if _ENV_FILE.exists():
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Neo4j
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

# Docs directory: all files under here are processed on ingest
DOCS_DIR = _ROOT / "docs"

# Docling + HybridChunker (HuggingFace tokenizer for chunking)
EMBED_MODEL_TOKENIZER = "BAAI/bge-m3"
MAX_TOKENS = 512

# Post-processing
MIN_CHUNK_CHARS = 25
MIN_CHUNK_WORDS = 4
SHORT_OK_LABELS = {"section_header", "title", "caption"}

# Ollama BGE-M3 for embeddings (and retrieval query embedding)
EMBED_MODEL_OLLAMA = "bge-m3"
BGE_M3_DIM = 1024

# Neo4j indexes
VECTOR_INDEX_NAME = "chunk_embeddings"
FULLTEXT_INDEX_NAME = "chunk_fulltext"

# Hybrid search: RRF
RRF_K = 60

# RAG retrieval: pool size for hybrid search when expanding by parent (fill remaining from pool)
RAG_CANDIDATE_POOL_SIZE = 20

# RAG: OpenRouter (if API key set) or Ollama
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "moonshotai/kimi-k2.5")
_DEFAULT_OLLAMA_CHAT = "llama3.2"
RAG_LLM_MODEL = os.environ.get("RAG_LLM_MODEL", _DEFAULT_OLLAMA_CHAT)
if RAG_LLM_MODEL == EMBED_MODEL_OLLAMA:
    RAG_LLM_MODEL = _DEFAULT_OLLAMA_CHAT
