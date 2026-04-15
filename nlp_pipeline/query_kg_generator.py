"""
Query-Driven Knowledge Graph (GraphRAG)
Takes a specific query, retrieves relevant legal context from Pinecone, 
and generates a targeted Knowledge Graph visualized in HTML.
"""

import os
import json
import time
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch
from groq import Groq
import networkx as nx
from pyvis.network import Network

# ---- CONFIG ----
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX = "legal-search-agent"
# We use the faster, cheaper 8b model for entity extraction
GROQ_MODEL = "llama-3.1-8b-instant" 
TOP_K = 10  # Pull top 10 relevant chunks

KG_PROMPT = """You are an expert Legal Knowledge Graph extractor.
Given the legal text below, extract all core entities and the relationships between them.

OUTPUT FORMAT REQUIREMENTS:
You MUST output strictly in JSON format. Do not add markdown blocks like ```json or any introductory text. 
Return a list of objects with exactly these keys:
[
  {
    "source": "Entity 1 (e.g. The Company)",
    "target": "Entity 2 (e.g. The Employee)",
    "relationship": "action or link (e.g. explicitly indemnifies)"
  }
]
Only extract high-confidence, factual relationships relevant to the context. 
Keep entity names highly concise (1-5 words).
"""

def generate_query_kg(query: str):
    print(f"\n🔍 Initializing Query-Driven Graph Generation for: '{query}'")
    
    # 1. Connect Services
    print("Connecting to Vector DB & Inference Engines...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    groq = Groq(api_key=GROQ_API_KEY)

    # 2. Retrieve Relevant Context
    print(f"Retrieving top {TOP_K} legal chunks for the query...")
    q_vec = embedder.encode([query])[0].tolist()
    results = index.query(vector=q_vec, top_k=TOP_K, include_metadata=True)
    
    context_text = ""
    for r in results['matches']:
        context_text += r['metadata'].get('text', '') + "\n\n"

    # 3. Extract Relationships via LLM
    print(f"🧠 Processing {len(context_text)} characters through Groq Llama-3.1-8B to extract structural nodes...")
    try:
        response = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": KG_PROMPT},
                {"role": "user", "content": f"TEXT TO ANALYZE:\n{context_text}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        raw_json = response.choices[0].message.content
        parsed = json.loads(raw_json)
        
        triples = []
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    triples = v
                    break
        elif isinstance(parsed, list):
            triples = parsed
            
    except Exception as e:
        print(f"Extraction failed: {e}")
        return

    print(f"✅ Found {len(triples)} specific legal relationships!")

    # 4. Build and Save Network
    print("🎨 Rendering interactive HTML visualization...")
    G = nx.DiGraph()
    for t in triples:
        if isinstance(t, dict):
            src = t.get("source", "").strip()
            tgt = t.get("target", "").strip()
            rel = t.get("relationship", "").strip()
            if src and tgt and rel:
                G.add_edge(src, tgt, label=rel)

    net = Network(height="800px", width="100%", bgcolor="#0f172a", font_color="#f8fafc", directed=True)
    
    # Spread out the physics significantly to stop clustering
    net.force_atlas_2based(
        gravity=-80, 
        central_gravity=0.005, 
        spring_length=350, 
        spring_strength=0.02, 
        damping=0.6, 
        overlap=1
    )
    net.from_nx(G)

    # Dynamic Styling based on Node importance
    degrees = dict(G.degree())
    for node in net.nodes:
        degree = degrees.get(node['id'], 1)
        node["shape"] = "dot"
        # Make highly connected nodes larger
        node["size"] = 15 + (degree * 5)
        
        # Color highly connected nodes differently (Cyan vs Amber)
        if degree > 2:
            node["color"] = "#0ea5e9" # Cyan
            node["borderColor"] = "#0369a1"
        else:
            node["color"] = "#f59e0b" # Amber
            node["borderColor"] = "#b45309"
            
        node["borderWidth"] = 3
        node["title"] = f"Entity: {node['id']} (Connections: {degree})"
        node["font"] = {"size": 16, "color": "#f8fafc", "face": "arial"}

    for edge in net.edges:
        edge["color"] = "#64748b"
        edge["width"] = 2
        edge["title"] = edge.get("label", "link")
        # Curve the overlaps, block the background behind text so lines don't cross through text
        edge["smooth"] = {"type": "curvedCW", "roundness": 0.2}
        edge["font"] = {
            "size": 14, 
            "color": "#cbd5e1", 
            "face": "arial", 
            "background": "#1e293b", # Prevents line running through letters
            "strokeWidth": 0
        }

    # Generate snake_case filename from query
    safe_name = "".join(c if c.isalnum() else "_" for c in query.lower())
    output_filename = f"query_kg_{safe_name}.html"
    json_filename = f"query_kg_{safe_name}.json"
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, output_filename)
    json_path = os.path.join(base_dir, json_filename)
    
    # Save the HTML visualization
    net.save_graph(output_path)
    
    # Save the JSON data
    graph_data = nx.node_link_data(G)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=4)

    print(f"🎉 Complete! View your specific query graph here:")
    print(f" --> HTML: {output_path}")
    print(f" --> JSON: {json_path}")

if __name__ == "__main__":
    import sys
    # If run with args, join them as query. Otherwise, default test query.
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what is breach of contract"
    generate_query_kg(q)
