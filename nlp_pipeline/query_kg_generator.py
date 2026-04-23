"""
Query-driven Knowledge Graph (GraphRAG).

Retrieves legal context from Pinecone, extracts legal relationships with Groq,
then saves both clean KG JSON and an interactive HTML visualization.
"""

import hashlib
import json
import os
import time

import networkx as nx
import torch
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

try:
    from pyvis.network import Network
except ModuleNotFoundError:
    Network = None


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "legal-search-agent")
GROQ_MODEL = os.getenv("KG_GROQ_MODEL", "llama-3.1-8b-instant")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
TOP_K = int(os.getenv("KG_TOP_K", "10"))

KG_PROMPT = """You are an expert Legal Knowledge Graph extractor.
Given the legal text below, extract all core entities and factual relationships between them.

OUTPUT FORMAT REQUIREMENTS:
Return strictly valid JSON. Do not add markdown or introductory text.
Return either a JSON array or an object containing a "relationships" array.
Each relationship must have exactly these fields:
[
  {
    "source": "concise entity name",
    "target": "concise entity name",
    "relationship": "concise legal relationship",
    "evidence": "short phrase from the legal text supporting the relationship"
  }
]

Rules:
- Extract only high-confidence legal relationships.
- Keep entity names concise, usually 1-5 words.
- Prefer parties, clauses, obligations, events, remedies, exceptions, courts, statutes, and claims.
- Do not invent facts not supported by the text.
"""


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.lower().strip().encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _entity_type(name: str) -> str:
    lowered = name.lower()
    if any(term in lowered for term in ["party", "supplier", "customer", "company", "employee", "buyer", "seller"]):
        return "party"
    if any(term in lowered for term in ["clause", "agreement", "contract", "section", "provision"]):
        return "legal_instrument"
    if any(term in lowered for term in ["obligation", "duty", "shall", "must", "indemnify", "pay"]):
        return "obligation"
    if any(term in lowered for term in ["claim", "breach", "default", "event", "violation", "termination"]):
        return "event"
    if any(term in lowered for term in ["damages", "losses", "fees", "remedy", "liability", "injunction"]):
        return "remedy"
    if any(term in lowered for term in ["exception", "limitation", "exclusion", "cap"]):
        return "exception"
    return "concept"


def parse_relationships(raw_json: str) -> list:
    parsed = json.loads(raw_json)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get("relationships"), list):
            return parsed["relationships"]
        for value in parsed.values():
            if isinstance(value, list):
                return value
    return []


def build_kg_json(query: str, triples: list, retrieved_chunks: list | None = None) -> dict:
    """Converts extracted triples into an explicit nodes/edges KG schema."""
    nodes_by_label = {}
    edges = []
    seen_edges = set()

    for triple in triples:
        if not isinstance(triple, dict):
            continue

        source = str(triple.get("source", "")).strip()
        target = str(triple.get("target", "")).strip()
        relationship = str(triple.get("relationship", "")).strip()
        evidence = str(triple.get("evidence", "")).strip()

        if not source or not target or not relationship:
            continue

        for label in [source, target]:
            if label not in nodes_by_label:
                nodes_by_label[label] = {
                    "id": _stable_id("ent", label),
                    "label": label,
                    "type": _entity_type(label),
                    "description": "",
                }

        source_id = nodes_by_label[source]["id"]
        target_id = nodes_by_label[target]["id"]
        edge_key = (source_id, relationship.lower(), target_id)
        if edge_key in seen_edges:
            continue

        seen_edges.add(edge_key)
        edges.append({
            "id": _stable_id("rel", "|".join(edge_key)),
            "source": source_id,
            "target": target_id,
            "label": relationship,
            "confidence": "high",
            "evidence": evidence,
        })

    return {
        "metadata": {
            "schema_version": "1.0",
            "graph_type": "legal_knowledge_graph",
            "query": query,
            "generated_at_unix": int(time.time()),
            "retrieval_top_k": TOP_K,
            "retrieved_chunk_count": len(retrieved_chunks or []),
            "node_count": len(nodes_by_label),
            "edge_count": len(edges),
        },
        "nodes": list(nodes_by_label.values()),
        "edges": edges,
        "retrieved_chunks": retrieved_chunks or [],
    }


def render_kg_html(kg_data: dict, output_path: str):
    """Renders the clean KG JSON schema to an interactive PyVis HTML graph."""
    if Network is None:
        from generate_kg_html import render_static_vis_html

        render_static_vis_html(kg_data, output_path)
        return

    graph = nx.DiGraph()

    for node in kg_data.get("nodes", []):
        graph.add_node(
            node["id"],
            label=node.get("label", node["id"]),
            type=node.get("type", "concept"),
            title=node.get("description") or node.get("label", node["id"]),
        )

    for edge in kg_data.get("edges", []):
        graph.add_edge(
            edge["source"],
            edge["target"],
            label=edge.get("label", "related_to"),
            title=edge.get("evidence") or edge.get("label", "related_to"),
        )

    net = Network(height="800px", width="100%", bgcolor="#0f172a", font_color="#f8fafc", directed=True)
    net.force_atlas_2based(
        gravity=-80,
        central_gravity=0.005,
        spring_length=350,
        spring_strength=0.02,
        damping=0.6,
        overlap=1,
    )
    net.from_nx(graph)

    type_colors = {
        "party": ("#38bdf8", "#0369a1"),
        "legal_instrument": ("#a78bfa", "#6d28d9"),
        "obligation": ("#22c55e", "#15803d"),
        "event": ("#f59e0b", "#b45309"),
        "remedy": ("#f43f5e", "#be123c"),
        "exception": ("#f97316", "#c2410c"),
        "concept": ("#94a3b8", "#475569"),
    }

    degrees = dict(graph.degree())
    node_lookup = {node["id"]: node for node in kg_data.get("nodes", [])}
    for node in net.nodes:
        node_data = node_lookup.get(node["id"], {})
        node_type = node_data.get("type", "concept")
        fill, border = type_colors.get(node_type, type_colors["concept"])
        degree = degrees.get(node["id"], 1)
        node["label"] = node_data.get("label", node["id"])
        node["shape"] = "dot"
        node["size"] = 14 + (degree * 4)
        node["color"] = fill
        node["borderColor"] = border
        node["borderWidth"] = 3
        node["title"] = f"{node_data.get('label', node['id'])}<br>Type: {node_type}<br>Connections: {degree}"
        node["font"] = {"size": 16, "color": "#f8fafc", "face": "arial"}

    for edge in net.edges:
        edge["color"] = "#64748b"
        edge["width"] = 2
        edge["smooth"] = {"type": "curvedCW", "roundness": 0.2}
        edge["font"] = {
            "size": 14,
            "color": "#cbd5e1",
            "face": "arial",
            "background": "#1e293b",
            "strokeWidth": 0,
        }

    net.save_graph(output_path)


def extract_relationships(groq: Groq, context_text: str) -> list:
    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": KG_PROMPT},
            {"role": "user", "content": f"TEXT TO ANALYZE:\n{context_text}"},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return parse_relationships(response.choices[0].message.content)


def generate_query_kg(query: str):
    print(f"\nInitializing query-driven graph generation for: '{query}'")

    print("Connecting to vector DB and inference engines...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    groq = Groq(api_key=GROQ_API_KEY)

    print(f"Retrieving top {TOP_K} legal chunks for the query...")
    query_vector = embedder.encode([query], show_progress_bar=False)[0].tolist()
    results = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)

    context_text = ""
    retrieved_chunks = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        text = metadata.get("text", "")
        context_text += text + "\n\n"
        retrieved_chunks.append({
            "id": match.get("id", ""),
            "score": match.get("score", None),
            "task": metadata.get("task", metadata.get("task_group", "")),
            "source": metadata.get("source", ""),
            "text_preview": text[:300],
        })

    print(f"Processing {len(context_text)} characters through {GROQ_MODEL}...")
    triples = extract_relationships(groq, context_text)
    print(f"Found {len(triples)} legal relationships.")

    safe_name = "".join(c if c.isalnum() else "_" for c in query.lower()).strip("_")
    output_filename = f"query_kg_{safe_name}.html"
    json_filename = f"query_kg_{safe_name}.json"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, output_filename)
    json_path = os.path.join(base_dir, json_filename)

    kg_data = build_kg_json(query=query, triples=triples, retrieved_chunks=retrieved_chunks)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(kg_data, file, indent=2, ensure_ascii=True)

    render_kg_html(kg_data, output_path)

    print("Complete. View your query graph here:")
    print(f" --> HTML: {output_path}")
    print(f" --> JSON: {json_path}")


if __name__ == "__main__":
    import sys

    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what is breach of contract"
    generate_query_kg(user_query)
