"""
Parse documents with Docling (PDF and other supported formats).
"""
from __future__ import annotations

import logging

from docling.document_converter import DocumentConverter

log = logging.getLogger(__name__)


def parse_document(source: str):
    """Convert a PDF (local path or URL) to a DoclingDocument."""
    log.info("Parsing document: %s", source)
    converter = DocumentConverter()
    conv_result = converter.convert(source=source)
    doc = conv_result.document

    log.info(
        "Parsed: %s (texts=%s, tables=%s, pictures=%s, groups=%s, pages=%s)",
        doc.name,
        len(doc.texts),
        len(doc.tables),
        len(doc.pictures),
        len(doc.groups),
        len(doc.pages),
    )
    return doc
