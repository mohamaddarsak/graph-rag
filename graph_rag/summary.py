"""
Print graph summary and document tree.
"""
from __future__ import annotations


def print_summary(nodes: list[dict], rels: list[dict]) -> None:
    """Print a summary of what was built."""
    n_docs = sum(1 for n in nodes if "Document" in n["labels"])
    n_sections = sum(1 for n in nodes if "Section" in n["labels"])
    n_chunks = sum(1 for n in nodes if "Chunk" in n["labels"])
    n_has_section = sum(1 for r in rels if r["type"] == "HAS_SECTION")
    n_has_chunk = sum(1 for r in rels if r["type"] == "HAS_CHUNK")
    n_next = sum(1 for r in rels if r["type"] == "NEXT")

    print("\n" + "=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)
    print(f"  Nodes:  {len(nodes)}")
    print(f"    Documents:  {n_docs}")
    print(f"    Sections:   {n_sections}")
    print(f"    Chunks:     {n_chunks}")
    print(f"  Relationships: {len(rels)}")
    print(f"    HAS_SECTION: {n_has_section}")
    print(f"    HAS_CHUNK:   {n_has_chunk}")
    print(f"    NEXT:        {n_next}")

    print("\n" + "=" * 60)
    print("DOCUMENT TREE")
    print("=" * 60)

    children_map: dict[str, list] = {}
    for r in rels:
        if r["type"] in ("HAS_SECTION", "HAS_CHUNK"):
            children_map.setdefault(r["start_id"], []).append((r["type"], r["end_id"]))

    node_map = {n["id"]: n for n in nodes}

    def _print_tree(node_id: str, depth: int = 0) -> None:
        node = node_map[node_id]
        indent = "  " * depth
        if "Document" in node["labels"]:
            print(f"{indent}[Document] {node['properties']['name']}")
        elif "Section" in node["labels"]:
            print(f"{indent}[Section] {node['properties']['title']}")
        elif "Chunk" in node["labels"]:
            text = node["properties"]["text"][:80]
            ctype = node["properties"].get("chunk_type", "?")
            suffix = "..." if len(node["properties"]["text"]) > 80 else ""
            print(f'{indent}[Chunk:{ctype}] "{text}{suffix}"')
        for _, child_id in children_map.get(node_id, []):
            _print_tree(child_id, depth + 1)

    for n in nodes:
        if "Document" in n["labels"]:
            _print_tree(n["id"])

    print()
