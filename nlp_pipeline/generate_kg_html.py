"""
Knowledge Graph Visualizer
Converts the generated legal_kg.json into a beautiful, interactive 3D HTML network graph.
"""

import os
import json
import networkx as nx
from pyvis.network import Network

def main():
    print("🎨 Initializing Graph Visualizer...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "legal_kg.json")
    html_output = os.path.join(base_dir, "interactive_kg.html")

    # 1. Load the exported JSON layout
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: legal_kg.json not found! Please run kg_generator.py first.")
        return

    # 2. Rebuild the NetworkX graph from the JSON data
    G = nx.node_link_graph(data)
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    print(f"Imported Graph: {num_nodes} Entities, {num_edges} Relationships.")

    # 3. Create an interactive PyVis physics network
    # We use notebook=True sometimes for jupyter, but here notebook=False works perfectly
    net = Network(height="800px", width="100%", bgcolor="#0F172A", font_color="#E2E8F0", directed=True)
    
    # Configure beautiful UI/physics for the nodes
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
        overlap=0
    )

    # Convert the NetworkX Graph into PyVis format
    net.from_nx(G)

    # Tweak the visual styling of each node and edge for a "Cyber/Legal" aesthetic
    for node in net.nodes:
        node["shape"] = "dot"
        node["size"] = 15
        node["color"] = "#38BDF8"  # Tailwind sky-400
        node["borderWidth"] = 2
        node["borderColor"] = "#0369A1"
        
        # Add a tooltip (title) that appears on hover
        node["title"] = f"Entity: {node['id']}"

    for edge in net.edges:
        edge["color"] = "#475569" # Tailwind slate-600
        edge["width"] = 1
        edge["title"] = edge.get("label", "Related to")
        # Ensure the label renders text onto the arrow
        edge["font"] = {"size": 10, "color": "#94A3B8", "face": "arial"}

    # 4. Generate the HTML file
    print("Render Engine actively building physics constraints...")
    net.save_graph(html_output)
    
    print("\n✅ Success!")
    print(f"You can now double-click to open this file in your browser:\n --> {html_output}")

if __name__ == "__main__":
    main()
