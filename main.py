"""
Graph RAG CLI: ingest documents from docs/ (or path), then answer questions.
  python main.py ingest              [--json graph_output.json]
  python main.py ingest /path/to/dir
  python main.py answer "Your question?"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from graph_rag import answer, ingest
from graph_rag.config import DOCS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graph RAG: ingest docs → Neo4j; answer questions with hybrid search + RAG."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Process all files under docs/ (or given path) and load into Neo4j")
    ingest_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        type=Path,
        help=f"Directory to scan (default: {DOCS_DIR})",
    )
    ingest_parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save graph nodes/rels to JSON file",
    )

    answer_parser = sub.add_parser("answer", help="Run RAG: hybrid retrieval + OpenRouter or Ollama")
    answer_parser.add_argument("question", nargs="+", help="Question to answer")
    answer_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")

    args = parser.parse_args()

    if args.command == "ingest":
        docs_path = args.path if args.path is not None else None
        ingest(docs_path=docs_path, save_json_path=args.json)
        print("Ingest done.")

    elif args.command == "answer":
        q = " ".join(args.question)
        result = answer(q, top_k=args.top_k)
        print(result if result else "(No answer text returned from the model.)", flush=True)


if __name__ == "__main__":
    main()
