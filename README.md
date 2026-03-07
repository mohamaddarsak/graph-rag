# How to Run the Graph RAG Pipeline

This guide covers running the RAG pipeline: PDFs in `docs/` → chunk → embed (Ollama BGE-M3) → Neo4j → query & answer (OpenRouter or Ollama).

---

## Prerequisites

- **Conda** (with the `graph-rag` environment)
- **Docker** (for Neo4j)
- **Ollama** (for embeddings with BGE-M3; optionally for answers if not using OpenRouter)

---

## 1. Environment setup

### Conda

Activate the project environment:

```bash
conda activate graph-rag
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

### Optional: OpenRouter (for RAG answers)

To use **OpenRouter** (e.g. `moonshotai/kimi-k2.5`) instead of Ollama for generating answers:

1. Copy the example env file and add your API key:
   ```bash
   cp env.example .env
   ```
2. Edit `.env` and set:
   - `OPENROUTER_API_KEY=sk-or-v1-your-key`
   - `OPENROUTER_MODEL=moonshotai/kimi-k2.5` (or another OpenRouter model)

If you don’t set `OPENROUTER_API_KEY`, the app will use **Ollama** for answers (you’ll need a chat model, e.g. `ollama pull llama3.2`).

---

## 2. Start Neo4j

From the project root:

```bash
docker compose up -d
```

Neo4j will be available at:

- **Browser UI:** http://localhost:7474 (login: `neo4j` / `password`)
- **Bolt:** `neo4j://localhost:7687`

---

## 3. Ollama (embeddings + optional chat)

Ollama must be running for **embeddings** (BGE-M3). If you’re not using OpenRouter, it’s also used for **answers**.

1. Start Ollama (app or `ollama serve`).
2. Pull the embedding model:
   ```bash
   ollama pull bge-m3
   ```
3. If you’re **not** using OpenRouter, pull a chat model for answers:
   ```bash
   ollama pull llama3.2
   ```

---

## 4. Run the pipeline

From the project root with `graph-rag` conda env active:

### Ingest PDFs only

Load PDFs from `docs/`, chunk, embed, and store in Neo4j:

```bash
python main.py ingest
```

### Answer a question (after ingest)

```bash
python main.py answer "Your question here?"
```

### Ingest + one example question (default)

```bash
python main.py
```

This runs ingest, then asks: *“What is the date, public authority and address in the document?”*

---

## 5. Run tests

**Unit tests only** (no Neo4j/Ollama needed):

```bash
pytest test_rag.py -m "not integration" -v
```

**All tests, including RAG integration** (Neo4j and Ollama must be running; optional OpenRouter for answers):

```bash
pytest test_rag.py -v
```

---

## Troubleshooting

| Issue | What to do |
|--------|------------|
| `address already in use` (port 11434) | Ollama is already running; no need to start it again. |
| `bge-m3 does not support generate` | Don’t use BGE-M3 for chat. Use OpenRouter or pull a chat model (`ollama pull llama3.2`). |
| No OpenRouter key | Set `OPENROUTER_API_KEY` in `.env` or export it; or use Ollama for answers and pull a chat model. |
| Neo4j connection refused | Run `docker compose up -d` and wait until Neo4j is up. |
| No PDFs / no text | Put at least one `.pdf` in the `docs/` folder. |

---

## Summary

1. `conda activate graph-rag` → `pip install -r requirements.txt`
2. (Optional) Copy `env.example` to `.env` and set `OPENROUTER_API_KEY` for OpenRouter answers.
3. `docker compose up -d` (Neo4j).
4. Start Ollama and `ollama pull bge-m3` (and a chat model if not using OpenRouter).
5. `python main.py ingest` then `python main.py answer "Your question?"` (or just `python main.py`).
