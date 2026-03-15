# Graph RAG Pipeline

Ingest documents from `docs/` into a **hierarchical graph** (Document → Section → Chunk) in Neo4j, then answer questions with **hybrid retrieval** (vector + full-text + RRF), **step-back prompting**, and **section-aware context** (siblings under the top hit’s parent, then fill from hybrid). Answers are generated with **OpenRouter** or **Ollama**.

---

## What it does

**Ingest**

- Discovers supported files under `docs/` (PDF, DOCX, MD, HTML, CSV, etc.) and parses them with **Docling**.
- Chunks with **HybridChunker** (BGE-M3–aligned tokenizer), post-processes, and builds a graph: **Document** → **Section** (nested) → **Chunk**.
- Embeds each chunk with **Ollama BGE-M3** and stores `text` + `embedding` on Chunk nodes in Neo4j (vector + full-text indexes).

**Answer (RAG)**

- Rewrites the user question into a **step-back question** (broader, for better vector retrieval); the **original question** is used for full-text search and the final answer.
- Runs **hybrid search**: vector (on step-back query) + full-text (on original query), merged with **RRF** (no reranker).
- Takes the **top-ranked chunk**, loads all chunks under the **same parent** (section/document), uses up to **5** of those (in document order), and fills remaining slots with the next-best hybrid results.
- Builds context from those chunks and generates an answer with **OpenRouter** (if `OPENROUTER_API_KEY` is set) or **Ollama**.

---

## Prerequisites

- **Python 3.10+** (e.g. conda env `graph-rag`)
- **Docker** (for Neo4j)
- **Ollama** (for BGE-M3 embeddings; and for answers if not using OpenRouter)

---

## 1. Environment

Activate the project environment and install dependencies:

```bash
conda activate graph-rag
pip install -r requirements.txt
```

Copy env and set Neo4j and (optionally) OpenRouter:

```bash
cp env.example .env
```

Edit `.env`:

- **Neo4j:** `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (defaults work with `docker compose`).
- **OpenRouter (optional):** `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (e.g. `moonshotai/kimi-k2.5`). If unset, answers use **Ollama** (pull a chat model, e.g. `llama3.2`).

---

## 2. Start Neo4j

```bash
docker compose up -d
```

- **Browser UI:** http://localhost:7474 (login: `neo4j` / `password`)
- **Bolt:** `bolt://localhost:7687` (default in `.env`)

---

## 3. Ollama

Ollama is required for **embeddings** (BGE-M3). If you’re not using OpenRouter, it’s also used for **answers** and for **step-back** query rewriting.

1. Start Ollama (app or `ollama serve`).
2. Pull the embedding model:
   ```bash
   ollama pull bge-m3
   ```
3. If not using OpenRouter, pull a chat model for answers (and step-back):
   ```bash
   ollama pull llama3.2
   ```

---

## 4. Run the pipeline

From the project root with the env active.

### Ingest

Process all supported files under `docs/` (or a given path), then load the graph into Neo4j:

```bash
python main.py ingest
```

Optional: custom directory and JSON backup:

```bash
python main.py ingest /path/to/dir --json graph_output.json
```

### Answer

Ask a question (step-back is printed, then retrieval logs, then the answer):

```bash
python main.py answer "Your question here?"
```

Limit context to the top 5 chunks (default):

```bash
python main.py answer "Your question?" -k 5
```

---

## 5. Project layout

| Path | Role |
|------|------|
| `main.py` | CLI: `ingest` / `answer` |
| `graph_rag/` | Package: config, parsing (Docling), chunking (HybridChunker), postprocess, graph build, embeddings (Ollama BGE-M3), neo4j_io, retrieval (hybrid + RRF + parent expansion), stepback, rag, summary |
| `docs/` | Input documents (PDF, DOCX, MD, HTML, etc.) |
| `env.example` | Template for `.env` (Neo4j, OpenRouter) |

---

## Troubleshooting

| Issue | What to do |
|--------|------------|
| `address already in use` (port 11434) | Ollama is already running. |
| `bge-m3` / “does not support generate” | Use BGE-M3 only for embeddings. Use OpenRouter or a chat model (e.g. `ollama pull llama3.2`) for answers and step-back. |
| No OpenRouter key | Set `OPENROUTER_API_KEY` in `.env`, or rely on Ollama for answers and step-back. |
| Neo4j connection refused | Run `docker compose up -d` and wait for Neo4j to be ready. |
| No files / no text | Add at least one supported file (e.g. `.pdf`) under `docs/`. |
| “No supported files found” | Check that `docs/` exists and contains files with supported extensions (e.g. `.pdf`, `.docx`, `.md`). |

---

## Summary

1. `conda activate graph-rag` → `pip install -r requirements.txt`
2. Copy `env.example` to `.env`; set Neo4j (and optionally `OPENROUTER_API_KEY`).
3. `docker compose up -d` (Neo4j).
4. Start Ollama; `ollama pull bge-m3` (and a chat model if not using OpenRouter).
5. `python main.py ingest` then `python main.py answer "Your question?"`.
