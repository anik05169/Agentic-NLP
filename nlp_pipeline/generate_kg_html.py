"""
Knowledge Graph Visualizer.

Renders the clean legal KG JSON schema into an interactive HTML network. The
visualizer also accepts older NetworkX node-link JSON files for compatibility.
"""

import json
import os
import sys

import networkx as nx

try:
    from pyvis.network import Network
except ModuleNotFoundError:
    Network = None


def load_graph(json_path: str) -> tuple[nx.DiGraph, dict]:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    graph = nx.DiGraph()

    if "nodes" in data and "edges" in data:
        for node in data.get("nodes", []):
            graph.add_node(
                node["id"],
                label=node.get("label", node["id"]),
                type=node.get("type", "concept"),
                title=node.get("description") or node.get("label", node["id"]),
            )

        for edge in data.get("edges", []):
            graph.add_edge(
                edge["source"],
                edge["target"],
                label=edge.get("label", "related_to"),
                title=edge.get("evidence") or edge.get("label", "related_to"),
            )
        return graph, data

    graph = nx.node_link_graph(data, directed=True)
    fallback_data = {
        "metadata": {"schema_version": "legacy_node_link"},
        "nodes": [
            {
                "id": node_id,
                "label": attrs.get("label", node_id),
                "type": attrs.get("type", "concept"),
                "description": attrs.get("title", ""),
            }
            for node_id, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "label": attrs.get("label", "related_to"),
                "evidence": attrs.get("title", ""),
            }
            for source, target, attrs in graph.edges(data=True)
        ],
    }
    return graph, fallback_data


def render_html(graph: nx.DiGraph, kg_data: dict, output_path: str):
    if Network is None:
        render_static_vis_html(kg_data, output_path)
        return

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


def render_static_vis_html(kg_data: dict, output_path: str):
    """Fallback renderer using the repo's local vis-network assets."""
    type_colors = {
        "party": {"background": "#38bdf8", "border": "#0369a1"},
        "legal_instrument": {"background": "#a78bfa", "border": "#6d28d9"},
        "obligation": {"background": "#22c55e", "border": "#15803d"},
        "event": {"background": "#f59e0b", "border": "#b45309"},
        "remedy": {"background": "#f43f5e", "border": "#be123c"},
        "exception": {"background": "#f97316", "border": "#c2410c"},
        "concept": {"background": "#94a3b8", "border": "#475569"},
    }

    nodes = []
    for node in kg_data.get("nodes", []):
        node_type = node.get("type", "concept")
        color = type_colors.get(node_type, type_colors["concept"])
        nodes.append({
            "id": node["id"],
            "label": node.get("label", node["id"]),
            "title": f"{node.get('label', node['id'])}<br>Type: {node_type}<br>{node.get('description', '')}",
            "color": color,
            "shape": "dot",
            "size": 24,
        })

    edges = []
    for edge in kg_data.get("edges", []):
        edges.append({
            "id": edge.get("id"),
            "from": edge["source"],
            "to": edge["target"],
            "label": edge.get("label", "related_to"),
            "title": edge.get("evidence") or edge.get("label", "related_to"),
            "arrows": "to",
            "color": "#64748b",
            "font": {"color": "#cbd5e1", "background": "#1e293b", "size": 14},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
        })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    vis_js = os.path.join(script_dir, "lib", "vis-9.1.2", "vis-network.min.js")
    vis_css = os.path.join(script_dir, "lib", "vis-9.1.2", "vis-network.css")
    rel_js = os.path.relpath(vis_js, os.path.dirname(output_path)).replace("\\", "/")
    rel_css = os.path.relpath(vis_css, os.path.dirname(output_path)).replace("\\", "/")
    metadata = kg_data.get("metadata", {})

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Legal Knowledge Graph</title>
  <link rel="stylesheet" href="{rel_css}">
  <style>
    body {{
      margin: 0;
      background: #0f172a;
      color: #f8fafc;
      font-family: Arial, sans-serif;
    }}
    header {{
      padding: 18px 24px;
      background: #111827;
      border-bottom: 1px solid #334155;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      font-weight: 700;
    }}
    p {{
      margin: 0;
      color: #cbd5e1;
      font-size: 14px;
    }}
    #network {{
      width: 100%;
      height: calc(100vh - 86px);
      min-height: 720px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Legal Knowledge Graph</h1>
    <p>Query: {metadata.get("query", "N/A")} | Nodes: {len(nodes)} | Edges: {len(edges)}</p>
  </header>
  <div id="network"></div>
  <script src="{rel_js}"></script>
  <script>
    const nodes = new vis.DataSet({json.dumps(nodes, ensure_ascii=True)});
    const edges = new vis.DataSet({json.dumps(edges, ensure_ascii=True)});
    const container = document.getElementById("network");
    const data = {{ nodes, edges }};
    const options = {{
      physics: {{
        solver: "forceAtlas2Based",
        forceAtlas2Based: {{
          gravitationalConstant: -80,
          centralGravity: 0.005,
          springLength: 350,
          springConstant: 0.02,
          damping: 0.6,
          avoidOverlap: 1
        }},
        stabilization: {{ iterations: 120 }}
      }},
      interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
      nodes: {{
        borderWidth: 3,
        font: {{ color: "#f8fafc", size: 16, face: "arial" }}
      }},
      edges: {{
        width: 2,
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }}
      }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "legal_kg.json")
    html_output = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base_dir, "interactive_kg.html")

    print("Initializing graph visualizer...")
    print(f"Loading data from {json_path}...")

    try:
        graph, kg_data = load_graph(json_path)
    except FileNotFoundError:
        print(f"Error: KG JSON not found: {json_path}")
        return

    print(f"Imported graph: {len(graph.nodes)} entities, {len(graph.edges)} relationships.")
    render_html(graph, kg_data, html_output)
    print(f"HTML visualization saved to: {html_output}")


if __name__ == "__main__":
    main()
