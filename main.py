"""
Graph RAG pipeline: PDFs from docs/ → chunk → embed (Ollama BGE-M3) → Neo4j → query & answer.
Retrieval: hybrid search (vector similarity + full-text) combined with Reciprocal Rank Fusion (RRF).
RAG answer: OpenRouter (if OPENROUTER_API_KEY set) or Ollama. Uses conda env: graph-rag.
"""
from __future__ import annotations

import os
from pathlib import Path
import hashlib

# Load .env if present (no extra dependency)
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import ollama
from neo4j import GraphDatabase
from openai import OpenAI
from pypdf import PdfReader

# Config (matches docker-compose.yml)
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
DOCS_DIR = Path(__file__).resolve().parent / "docs"
EMBED_MODEL = "bge-m3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 40
VECTOR_INDEX_NAME = "chunk_embeddings"
FULLTEXT_INDEX_NAME = "chunk_fulltext"
CHUNK_LABEL = "Chunk"
BGE_M3_DIM = 1024
RRF_K = 60  # Reciprocal Rank Fusion damping constant
# RAG answer: OpenRouter (preferred when API key set) or Ollama
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "moonshotai/kimi-k2.5")
_DEFAULT_OLLAMA_CHAT = "llama3.2"
RAG_LLM_MODEL = _DEFAULT_OLLAMA_CHAT if OPENROUTER_API_KEY else os.environ.get("RAG_LLM_MODEL", _DEFAULT_OLLAMA_CHAT)
if RAG_LLM_MODEL == EMBED_MODEL:
    RAG_LLM_MODEL = _DEFAULT_OLLAMA_CHAT


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into chunks of ~chunk_size chars with overlap, breaking only on whitespace."""
    if not text or not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    text = text.replace("\r\n", "\n")
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # move end back to last whitespace in the window [end - overlap, end]
            search_start = max(start, end - overlap)
            last_ws = text.rfind(" ", search_start, end + 1)
            if last_ws != -1:
                end = last_ws + 1
            else:
                # no space in window; break at next space after end
                next_ws = text.find(" ", end)
                if next_ws != -1:
                    end = next_ws + 1
                # else keep end as is
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # next start: step back by overlap so we overlap by ~overlap chars
        start = end - overlap if end - overlap > start else end
        if start >= n:
            break
    return chunks


def load_pdfs(docs_path: Path) -> str:
    """Extract and concatenate text from all PDFs in docs_path."""
    combined: list[str] = []
    if not docs_path.is_dir():
        return ""
    for p in sorted(docs_path.glob("*.pdf")):
        try:
            reader = PdfReader(p)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    combined.append(t)
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF {p}") from e
    return "\n\n".join(combined)


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed texts using Ollama BGE-M3. Returns list of 1024-dim vectors."""
    vectors: list[list[float]] = []
    for t in texts:
        r = ollama.embed(model=model, input=t)
        # single input returns one embedding
        emb = r.embeddings[0] if r.embeddings else []
        vectors.append(emb)
    return vectors


def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def ensure_vector_index(driver) -> None:
    """Create vector index on Chunk.embedding if it does not exist."""
    with driver.session() as session:
        session.run(
            """
            CREATE VECTOR INDEX $name IF NOT EXISTS
            FOR (n:Chunk) ON n.embedding
            OPTIONS { indexConfig: {
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            name=VECTOR_INDEX_NAME,
            dim=BGE_M3_DIM,
        )


def ensure_fulltext_index(driver) -> None:
    """Create full-text index on Chunk.text for hybrid search."""
    with driver.session() as session:
        session.run(
            """
            CREATE FULLTEXT INDEX $name IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text]
            """,
            name=FULLTEXT_INDEX_NAME,
        )


def ingest(docs_path: Path | None = None) -> None:
    """Load PDFs, chunk, embed, and store in Neo4j."""
    path = docs_path or DOCS_DIR
    full_text = load_pdfs(path)
    if not full_text.strip():
        raise ValueError(f"No text extracted from PDFs in {path}")
    chunks = chunk_text(full_text)
    if not chunks:
        raise ValueError("Chunking produced no chunks")
    embeddings = embed_texts(chunks)
    driver = get_neo4j_driver()
    try:
        ensure_vector_index(driver)
        ensure_fulltext_index(driver)
        with driver.session() as session:
            # Clear existing chunks so we can re-ingest idempotently
            session.run("MATCH (c:Chunk) DETACH DELETE c")
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_id = hashlib.sha256(chunk.encode()).hexdigest()
                session.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    ON CREATE SET c.text = $text, c.embedding = $embedding, c.index = $index
                    ON MATCH SET c.embedding = $embedding, c.index = $index
                    """,
                    chunk_id=chunk_id,
                    text=chunk,
                    embedding=emb,
                    index=i,
                )
    finally:
        driver.close()


def retrieve(driver, query: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Embed query and return top_k (text, score) from Neo4j vector index."""
    vecs = embed_texts([query])
    query_vec = vecs[0]
    with driver.session() as session:
        result = session.run(
            """
            CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
            YIELD node, score
            RETURN node.text AS text, score
            """,
            index_name=VECTOR_INDEX_NAME,
            k=top_k,
            query_vector=query_vec,
        )
        return [(r["text"], r["score"]) for r in result]


def retrieve_hybrid(
    driver,
    query: str,
    top_k: int = 5,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Hybrid search: vector + full-text, combined with Reciprocal Rank Fusion (RRF).
    RRF_score = 1/(k + rank_vector) + 1/(k + rank_keyword); missing branch contributes 0.
    """
    vecs = embed_texts([query])
    query_vec = vecs[0]
    with driver.session() as session:
        result = session.run(
            """
            CALL () {
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
            RETURN node.text AS text, rrf_score
            """,
            vector_index=VECTOR_INDEX_NAME,
            ft_index=FULLTEXT_INDEX_NAME,
            question=query,
            query_vector=query_vec,
            k=top_k,
            rrf_k=rrf_k,
        )
        return [(r["text"], r["rrf_score"]) for r in result]


def _generate_answer(prompt: str) -> str:
    """Generate answer using OpenRouter (if API key set) or Ollama."""
    if OPENROUTER_API_KEY:
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        resp = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()
    return (ollama.chat(model=RAG_LLM_MODEL, messages=[{"role": "user", "content": prompt}]).message.content or "").strip()


def answer(query: str, top_k: int = 5) -> str:
    """RAG: retrieve relevant chunks via hybrid search (vector + full-text, RRF) and generate answer."""
    driver = get_neo4j_driver()
    try:
        hits = retrieve_hybrid(driver, query, top_k=top_k)
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


def main() -> None:
    """CLI: ingest then optional query, or just answer a query if already ingested."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
        print("Ingest done.")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "answer" and len(sys.argv) > 2:
        q = " ".join(sys.argv[2:])
        print(answer(q))
        return
    # default: run ingest then example query
    ingest()
    print("Ingest done. Example answer:")
    print(answer("What is the date, public authority and address in the document?"))


if __name__ == "__main__":
    main()
