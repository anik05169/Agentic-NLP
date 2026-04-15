"""
Legal Knowledge Graph (KG) Generator
Extracts entities and their relationships from legal text chunks using Groq.
"""

import json
import os
import time
from groq import Groq
import networkx as nx

# ---- CONFIG ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# We use the super fast 8b model for bulk entity extraction, or the 70b versatile one for high accuracy
GROQ_MODEL = "llama-3.1-8b-instant" 

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
Only extract high-confidence, factual relationships. Keep entities concise.
"""

def extract_kg_from_text(client: Groq, text: str) -> list:
    """Uses Groq to extract JSON relationship triples from text."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": KG_PROMPT},
                {"role": "user", "content": f"TEXT TO ANALYZE:\n{text}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"} # Forces JSON output
        )
        
        # Parse the JSON response
        raw_json = response.choices[0].message.content
        # Sometimes the LLM wraps the list in an object dict, handle both forms
        parsed = json.loads(raw_json)
        
        if isinstance(parsed, dict):
            # Find the first list value in the dict
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            return []
        elif isinstance(parsed, list):
            return parsed
        return []
            
    except Exception as e:
        print(f"Extraction error: {e}")
        return []

def main():
    print("🚀 Initializing Legal Knowledge Graph Generator...")
    client = Groq(api_key=GROQ_API_KEY)
    
    # 1. Load a few sample paragraphs from our dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "vectorized_chunks_master.jsonl")
    
    print(f"Reading sample texts from: {data_path}")
    sample_texts = []
    
    # We will just process the first 10 LEDGAR (contract) tasks for the demo
    with open(data_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            rec = json.loads(line)
            if rec.get("task_group") == "lexglue_ledgar":
                sample_texts.append(rec.get("text")[:800]) # Cap text length for speed
                count += 1
            if count >= 10:
                break
                
    # 2. Extract relationships
    all_triples = []
    print(f"\n🧠 Extracting graph relationships via {GROQ_MODEL}...")
    
    for i, text in enumerate(sample_texts):
        print(f"  -> Processing document {i+1}/{len(sample_texts)}...")
        triples = extract_kg_from_text(client, text)
        if triples:
            all_triples.extend(triples)
        time.sleep(1) # Rate limit protection
        
    # 3. Build and Save the Graph
    print(f"\n✅ Extraction Complete! Found {len(all_triples)} relationships.")
    
    G = nx.DiGraph()
    for t in all_triples:
        if not isinstance(t, dict):
            continue
            
        src = t.get("source")
        tgt = t.get("target")
        rel = t.get("relationship")
        
        if src and tgt and rel:
            G.add_edge(str(src).strip(), str(tgt).strip(), label=str(rel).strip())
            
    # Save the graph node/edge data to a JSON file
    output_path = os.path.join(base_dir, "legal_kg.json")
    
    graph_data = nx.node_link_data(G)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=4)
        
    print(f"\n💾 Knowledge Graph saved to: {output_path}")
    print("\n--- SAMPLE EXTRACTED RELATIONSHIPS ---")
    valid_count = 0
    for t in all_triples:
         if isinstance(t, dict) and t.get('source'):
             print(f"[{t.get('source')}] --({t.get('relationship')})--> [{t.get('target')}]")
             valid_count += 1
             if valid_count >= 10: break

if __name__ == "__main__":
    main()
