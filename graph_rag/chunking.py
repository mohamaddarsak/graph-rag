"""
Chunk documents with HybridChunker (BGE-M3 tokenizer aligned).
"""
from __future__ import annotations

import logging

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from graph_rag.config import EMBED_MODEL_TOKENIZER, MAX_TOKENS

log = logging.getLogger(__name__)


def create_chunks(doc):
    """Apply HybridChunker with BGE-M3 tokenizer alignment."""
    log.info("Loading tokenizer: %s", EMBED_MODEL_TOKENIZER)
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_TOKENIZER),
        max_tokens=MAX_TOKENS,
    )

    log.info("Chunking with HybridChunker (merge_peers=True)")
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    chunks = list(chunker.chunk(dl_doc=doc))
    log.info("Raw chunks produced: %s", len(chunks))

    return chunks, chunker
