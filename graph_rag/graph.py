"""
Build Neo4j-ready graph data (Document, Section, Chunk nodes and relationships).
Chunk and section IDs are scoped by doc_id so multiple documents can be merged.
"""
from __future__ import annotations

import hashlib

from graph_rag.postprocess import get_chunk_labels


def build_graph_data(doc, chunks, chunker, preamble_text: str | None):
    """
    Build Neo4j-ready nodes and relationships.

    Graph schema:
        (:Document) -[:HAS_SECTION]→ (:Section) -[:HAS_SECTION]→ (:Section)
        (:Document) -[:HAS_CHUNK]→ (:Chunk)
        (:Section)  -[:HAS_CHUNK]→ (:Chunk)
        (:Chunk)    -[:NEXT]→ (:Chunk)

    IDs are doc-scoped (doc_id::chunk:N, doc_id::sec:hash) so multiple docs can be merged.
    """
    nodes = []
    rels = []

    doc_dict = doc.export_to_dict()
    origin = doc_dict.get("origin", {})
    doc_id = f"doc:{origin.get('binary_hash', doc.name)}"

    nodes.append({
        "id": doc_id,
        "labels": ["Document"],
        "properties": {
            "name": doc.name,
            "filename": origin.get("filename", ""),
            "mimetype": origin.get("mimetype", ""),
            "hash": str(origin.get("binary_hash", "")),
            "num_pages": len(doc.pages),
        },
    })

    prev_chunk_id = None
    chunk_counter = 0

    if preamble_text:
        chunk_id = f"{doc_id}::chunk:{chunk_counter}"
        nodes.append({
            "id": chunk_id,
            "labels": ["Chunk"],
            "properties": {
                "index": chunk_counter,
                "text": preamble_text,
                "contextualized_text": preamble_text,
                "chunk_type": "preamble",
                "labels_in_chunk": ["preamble"],
                "self_refs": [],
                "headings": [],
                "captions": [],
                "page_numbers": [1],
            },
        })
        rels.append({
            "type": "HAS_CHUNK",
            "start_id": doc_id,
            "end_id": chunk_id,
            "properties": {"order": chunk_counter},
        })
        prev_chunk_id = chunk_id
        chunk_counter += 1

    section_cache: dict[tuple, str] = {}

    def get_or_create_section(headings: list[str]) -> str:
        if not headings:
            return doc_id

        for depth in range(1, len(headings) + 1):
            path = tuple(headings[:depth])
            if path not in section_cache:
                sec_id = f"{doc_id}::sec:{hashlib.md5('|'.join(path).encode()).hexdigest()[:12]}"
                nodes.append({
                    "id": sec_id,
                    "labels": ["Section"],
                    "properties": {
                        "title": headings[depth - 1],
                        "level": depth,
                        "path": " > ".join(path),
                    },
                })
                parent_id = doc_id if depth == 1 else section_cache[tuple(headings[: depth - 1])]
                rels.append({
                    "type": "HAS_SECTION",
                    "start_id": parent_id,
                    "end_id": sec_id,
                    "properties": {},
                })
                section_cache[path] = sec_id

        return section_cache[tuple(headings)]

    for chunk in chunks:
        chunk_id = f"{doc_id}::chunk:{chunk_counter}"

        labels_in_chunk = list(get_chunk_labels(chunk))
        self_refs = [di.self_ref for di in chunk.meta.doc_items]

        page_nos = set()
        for di in chunk.meta.doc_items:
            if hasattr(di, "prov") and di.prov:
                for p in di.prov:
                    if hasattr(p, "page_no"):
                        page_nos.add(p.page_no)

        ctx_text = chunker.contextualize(chunk)

        nodes.append({
            "id": chunk_id,
            "labels": ["Chunk"],
            "properties": {
                "index": chunk_counter,
                "text": chunk.text,
                "contextualized_text": ctx_text,
                "chunk_type": "content",
                "labels_in_chunk": labels_in_chunk,
                "self_refs": self_refs,
                "headings": chunk.meta.headings if chunk.meta.headings else [],
                "captions": chunk.meta.captions if chunk.meta.captions else [],
                "page_numbers": sorted(page_nos) if page_nos else [],
            },
        })

        headings = chunk.meta.headings if chunk.meta.headings else []
        parent_id = get_or_create_section(headings)
        rels.append({
            "type": "HAS_CHUNK",
            "start_id": parent_id,
            "end_id": chunk_id,
            "properties": {"order": chunk_counter},
        })

        if prev_chunk_id is not None:
            rels.append({
                "type": "NEXT",
                "start_id": prev_chunk_id,
                "end_id": chunk_id,
                "properties": {},
            })
        prev_chunk_id = chunk_id
        chunk_counter += 1

    return nodes, rels
