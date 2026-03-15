"""
Hybrid retrieval: vector + full-text search combined with RRF (no reranker).
RAG retrieval can expand by parent: take top result's section, use its siblings, then fill with next-best.
"""
from __future__ import annotations

import logging

from graph_rag.config import (
    FULLTEXT_INDEX_NAME,
    RAG_CANDIDATE_POOL_SIZE,
    RRF_K,
    VECTOR_INDEX_NAME,
)
from graph_rag.embeddings import embed_texts

log = logging.getLogger(__name__)

# Max chars to show per chunk in debug logs
SNIPPET_LEN = 180


def retrieve_hybrid(
    driver,
    query: str,
    top_k: int = 5,
    rrf_k: int = RRF_K,
    *,
    query_for_vector: str | None = None,
    query_for_fulltext: str | None = None,
) -> list[tuple[str, str, float]]:
    """
    Hybrid search: vector + full-text, combined with Reciprocal Rank Fusion (RRF).
    Vector search uses query_for_vector if set, else query; fulltext uses query_for_fulltext if set, else query.
    Returns list of (chunk_id, text, rrf_score) for top_k results.
    """
    embed_query = query_for_vector if query_for_vector is not None else query
    ft_query = query_for_fulltext if query_for_fulltext is not None else query
    vecs = embed_texts([embed_query])
    query_vec = vecs[0]
    with driver.session() as session:
        result = session.run(
            """
            CALL {
                CALL db.index.vector.queryNodes($vector_index, $k, $query_vector)
                YIELD node, score
                WITH node, score ORDER BY score DESC
                WITH collect(node) AS nodes
                UNWIND range(0, size(nodes) - 1) AS rank
                RETURN nodes[rank] AS node, rank AS rank
                UNION
                CALL db.index.fulltext.queryNodes($ft_index, $question) YIELD node, score
                WITH node, score ORDER BY score DESC
                WITH collect(node) AS nodes
                UNWIND range(0, size(nodes) - 1) AS rank
                RETURN nodes[rank] AS node, rank AS rank
            }
            WITH node, sum(1.0 / ($rrf_k + rank)) AS rrf_score
            ORDER BY rrf_score DESC
            LIMIT $k
            RETURN node._id AS chunk_id, node.text AS text, rrf_score
            """,
            vector_index=VECTOR_INDEX_NAME,
            ft_index=FULLTEXT_INDEX_NAME,
            question=ft_query,
            query_vector=query_vec,
            k=top_k,
            rrf_k=rrf_k,
        )
        hits = [(r["chunk_id"], r["text"], r["rrf_score"]) for r in result]
        if log.isEnabledFor(logging.DEBUG) or log.isEnabledFor(logging.INFO):
            log.info("Top candidates from hybrid search (RRF), top_k=%s:", top_k)
            for i, (_cid, text, score) in enumerate(hits, 1):
                snippet = (text[:SNIPPET_LEN] + "…") if len(text) > SNIPPET_LEN else text
                snippet = snippet.replace("\n", " ")
                log.info("  [%s] rrf_score=%.4f | %s", i, score, snippet)
        return hits


def retrieve_for_rag(
    driver,
    query: str,
    top_k: int = 5,
    rrf_k: int = RRF_K,
    *,
    query_for_vector: str | None = None,
) -> list[tuple[str, float]]:
    """
    Retrieve chunks for RAG: take the highest-ranked chunk, get all chunks under the same
    parent (section/document), use up to top_k of those; fill remaining slots with the
    next highest-ranked from hybrid search. Returns list of (text, score).
    If query_for_vector is set (e.g. step-back question), it is used only for vector search;
    query is used for fulltext and for all other logic.
    """
    pool_size = max(RAG_CANDIDATE_POOL_SIZE, top_k)
    hits = retrieve_hybrid(
        driver,
        query,
        top_k=pool_size,
        rrf_k=rrf_k,
        query_for_vector=query_for_vector,
        query_for_fulltext=query,
    )
    if not hits:
        return []

    top_chunk_id, top_text, top_score = hits[0]
    with driver.session() as session:
        parent_row = session.run(
            "MATCH (c:Chunk {_id: $cid})<-[:HAS_CHUNK]-(p) RETURN p._id AS parent_id",
            cid=top_chunk_id,
        ).single()
        if not parent_row:
            log.warning("No parent found for top chunk %s; using hybrid order only.", top_chunk_id)
            return [(text, score) for _cid, text, score in hits[:top_k]]

        parent_id = parent_row["parent_id"]
        siblings = list(
            session.run(
                """
                MATCH (p {_id: $pid})-[r:HAS_CHUNK]->(c:Chunk)
                RETURN c._id AS chunk_id, c.text AS text
                ORDER BY r.order
                """,
                pid=parent_id,
            )
        )
        sibling_list = [(r["chunk_id"], r["text"]) for r in siblings]

    if not sibling_list:
        return [(text, score) for _cid, text, score in hits[:top_k]]

    sibling_ids = {cid for cid, _ in sibling_list}
    # Use up to top_k from siblings (document order)
    from_siblings = sibling_list[:top_k]
    from_sibling_ids = {cid for cid, _ in from_siblings}
    # Scores for siblings: use top_score for the first (the one that was #1), 0 for rest
    final: list[tuple[str, float]] = []
    for i, (cid, text) in enumerate(from_siblings):
        score = top_score if cid == top_chunk_id else 0.0
        final.append((text, score))

    remaining = top_k - len(final)
    if remaining > 0:
        for cid, text, score in hits[1:]:
            if cid not in from_sibling_ids:
                final.append((text, score))
                from_sibling_ids.add(cid)
                remaining -= 1
                if remaining == 0:
                    break

    if log.isEnabledFor(logging.DEBUG) or log.isEnabledFor(logging.INFO):
        log.info(
            "RAG context: %s from same parent (section/doc), %s filled from hybrid; total=%s",
            len(from_siblings),
            len(final) - len(from_siblings),
            len(final),
        )
        for i, (text, score) in enumerate(final, 1):
            snippet = (text[:SNIPPET_LEN] + "…") if len(text) > SNIPPET_LEN else text
            snippet = snippet.replace("\n", " ")
            log.info("  [%s] (score=%.4f) %s", i, score, snippet)

    return final
