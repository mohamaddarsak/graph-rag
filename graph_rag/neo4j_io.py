"""
Neo4j driver, indexes (vector + fulltext), and insert.
Embedding property is kept as list of floats (not JSON) for vector index.
"""
from __future__ import annotations

import json
import logging

from neo4j import GraphDatabase

from graph_rag.config import (
    BGE_M3_DIM,
    FULLTEXT_INDEX_NAME,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    VECTOR_INDEX_NAME,
)

log = logging.getLogger(__name__)


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


def insert_into_neo4j(nodes: list[dict], rels: list[dict]) -> None:
    """Insert all nodes and relationships into Neo4j.
    Chunk.embedding is left as list of floats; other list props are JSON strings.
    """
    log.info("Connecting to Neo4j at %s", NEO4J_URI)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    log.info("Connected to Neo4j")

    with driver.session() as session:
        log.info("Creating constraints and indexes...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d._id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s._id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c._id IS UNIQUE")
        session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.index)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (s:Section) ON (s.path)")

        ensure_vector_index(driver)
        ensure_fulltext_index(driver)

        log.info("Inserting %s nodes...", len(nodes))
        for n in nodes:
            label = n["labels"][0]
            props = {**n["properties"], "_id": n["id"]}

            for k, v in list(props.items()):
                if isinstance(v, list):
                    if k == "embedding":
                        continue  # keep as list of floats for vector index
                    props[k] = json.dumps(v)

            session.run(
                f"MERGE (n:{label} {{_id: $id}}) SET n += $props",
                id=n["id"],
                props=props,
            )

        log.info("Inserting %s relationships...", len(rels))
        for r in rels:
            props = r.get("properties", {})
            session.run(
                f"""
                MATCH (a {{_id: $start_id}})
                MATCH (b {{_id: $end_id}})
                MERGE (a)-[rel:{r['type']}]->(b)
                SET rel += $props
                """,
                start_id=r["start_id"],
                end_id=r["end_id"],
                props=props,
            )

    driver.close()
    log.info("Neo4j insertion complete")
