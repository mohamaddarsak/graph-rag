"""
Post-process chunks: merge preamble, filter junk.
"""
from __future__ import annotations

import logging

from graph_rag.config import MIN_CHUNK_CHARS, MIN_CHUNK_WORDS, SHORT_OK_LABELS

log = logging.getLogger(__name__)


def get_chunk_labels(chunk) -> set[str]:
    """Extract label strings from a chunk's doc_items."""
    labels = set()
    for di in chunk.meta.doc_items:
        if hasattr(di.label, "value"):
            labels.add(di.label.value)
        else:
            labels.add(str(di.label))
    return labels


def is_junk_chunk(chunk) -> bool:
    """Return True if chunk is too small to be useful."""
    text = chunk.text.strip()
    labels = get_chunk_labels(chunk)

    if labels & SHORT_OK_LABELS:
        return False

    return len(text) < MIN_CHUNK_CHARS or len(text.split()) < MIN_CHUNK_WORDS


def merge_preamble_chunks(chunks):
    """
    Merge consecutive heading-less short chunks at the start of the document
    into a single preamble text.
    """
    preamble_parts = []
    main_chunks = []
    preamble_ended = False

    for chunk in chunks:
        has_headings = bool(chunk.meta.headings)
        is_short = len(chunk.text.strip()) < 100

        if not preamble_ended and not has_headings and is_short:
            preamble_parts.append(chunk.text.strip())
        else:
            preamble_ended = True
            main_chunks.append(chunk)

    preamble_text = None
    if preamble_parts:
        preamble_text = " | ".join(preamble_parts)
        log.info(
            "Merged %s preamble fragments into 1 block (%s chars)",
            len(preamble_parts),
            len(preamble_text),
        )

    return preamble_text, main_chunks


def post_process_chunks(chunks):
    """Filter junk and merge preamble."""
    before = len(chunks)

    preamble_text, chunks = merge_preamble_chunks(chunks)
    chunks = [c for c in chunks if not is_junk_chunk(c)]

    after = len(chunks)
    log.info("Post-processing: %s → %s chunks (removed %s)", before, after, before - after)

    return preamble_text, chunks
