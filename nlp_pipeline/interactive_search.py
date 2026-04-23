"""
Interactive Legal AI Search (CLI Version)
Combines local ChromaDB retrieval (D: drive) with Groq inference and Memory.
"""

import json
import os
import time
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from groq import Groq

# Internal modules
from memory_manager import MemoryManager
from runtime_config import CHROMA_COLLECTION, CHROMA_DB_PATH, EMBED_MODEL, GROQ_MODEL, KG_GROQ_MODEL

# Load API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- CONFIG ----
COLLECTION_NAME = CHROMA_COLLECTION

QUERY_EXPANSION_PROMPT = """You are a legal query expansion engine.
Rewrite the user's question into a precise legal retrieval query.

RULES:
- Convert casual language into formal legal terminology
- Add relevant legal doctrines, concepts, and remedies that apply
- Keep it under 60 words
- Output ONLY the rewritten query, nothing else

Examples:
User: "I'm 10 days late showing my product to a company"
Expanded: "breach of contract failure to perform contractual obligations within deadline cure period liquidated damages remedies for late delivery performance default"

User: "can I sue my landlord"
Expanded: "tenant legal remedies against landlord breach of lease agreement habitability warranty eviction rights landlord liability negligence claim"
"""

KG_EXTRACTION_PROMPT = """You are a Legal Knowledge Graph extractor.
Extract all core legal entities and their factual relationships from the text.
Return ONLY valid JSON in this shape: {"relationships": [{"source": "A", "target": "B", "relationship": "label"}]}
Focus on parties, obligations, remedies, and contract clauses.
"""

SYSTEM_PROMPT = """You are a professional Legal AI advisor specializing in precise, source-grounded answers.

STRICT RULES:
1. ONLY use the provided Legal Context and Structural Relationships. Do NOT use external knowledge.
2. Every factual statement MUST be tied to a citation [Source X].
3. If the answer is not explicitly supported, say: "The provided sources do not contain sufficient information."
4. Use the 'STRUCTURAL RELATIONSHIPS' to understand the connection between different legal entities.

OUTPUT FORMAT:
- Definition
- Key Details (bullet points)
- Source-backed explanation
- Citations clearly linked

---
LEGAL CONTEXT:
{legal_context}

---
STRUCTURAL RELATIONSHIPS (Knowledge Graph):
{structural_relationships}

---
PAST MEMORIES:
{long_term_memory}

---
RECENT HISTORY:
{short_term_memory}
"""

def expand_query(groq_client, user_query):
    """Rewrites a casual user question into a precise legal retrieval query."""
    try:
        response = groq_client.chat.completions.create(
            model=KG_GROQ_MODEL,
            messages=[
                {"role": "system", "content": QUERY_EXPANSION_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0,
            max_tokens=120
        )
        expanded = response.choices[0].message.content.strip()
        return expanded if expanded else user_query
    except:
        return user_query


def extract_structural_context(groq_client, text):
    """Extracts a mini-KG from retrieved chunks to aid semantic reasoning."""
    try:
        response = groq_client.chat.completions.create(
            model=KG_GROQ_MODEL,
            messages=[
                {"role": "system", "content": KG_EXTRACTION_PROMPT},
                {"role": "user", "content": f"EXTRACT FROM:\n{text[:4000]}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        rels = data.get("relationships", [])
        if not rels:
            return "No clear structural relationships identified."
        return "\n".join([f"- {r.get('source')} --({r.get('relationship')})--> {r.get('target')}" for r in rels[:15]])
    except:
        return "Structural analysis unavailable."

def main():
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found in .env file. Please check your root .env.")
        return

    # 1. Initialize Components
    print("\n[1/4] Loading local embeddings model (MPNet)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    # 2. Local ChromaDB
    print(f"[2/4] Connecting to local ChromaDB at {CHROMA_DB_PATH}...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"ERROR: Could not connect to ChromaDB: {e}")
        return

    # 3. Groq
    print("[3/4] Connecting to Groq Cloud...")
    groq_client = Groq(api_key=GROQ_API_KEY)

    # 4. Memory Manager
    print("[4/4] Initializing Memory Manager (SQLite + ChromaDB)...")
    memory_mgr = MemoryManager(client, embedder, groq_client)

    session_id = f"cli_session_{int(time.time())}"
    user_id = "default_cli_user"

    print("\n" + "="*50)
    print(" READY (GraphRAG Enabled): Type your legal query")
    print(f" Session ID: {session_id}")
    print("="*50)

    while True:
        query = input("\nQuery > ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            break
        if not query:
            continue

        start_time = time.time()

        # A. Query Expansion: rewrite casual question into legal retrieval query
        print("   -> Expanding query for legal precision...")
        expanded_query = expand_query(groq_client, query)
        print(f"   -> Search query: {expanded_query[:100]}...")

        # B. Embed the EXPANDED query (not the original) for better retrieval
        query_vec = embedder.encode([expanded_query])[0].tolist()

        # C. Retrieve Legal Context
        print("   -> Searching legal database...")
        try:
            results = collection.query(query_embeddings=[query_vec], n_results=5)
        except Exception as e:
            print(f"   -> Retrieval Error: {e}")
            continue

        # C. GraphRAG: Extract Structural Relationships
        print("   -> Checking legal relationships (KG)...")
        context_str = ""
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
            task = meta.get("task", "unknown")
            context_str += f"[Source {i}] (Task: {task})\n{doc}\n\n"
        
        structural_rels = extract_structural_context(groq_client, context_str)

        # D. Retrieve Memories
        short_mem = memory_mgr.get_short_term_memory(session_id)
        history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in short_mem])
        long_mem = memory_mgr.get_long_term_memory(user_id, query)

        # E. Format Prompt
        full_prompt = SYSTEM_PROMPT.format(
            legal_context=context_str or "No direct legal matches found.",
            structural_relationships=structural_rels,
            long_term_memory=long_mem or "No relevant past interactions found.",
            short_term_memory=history_str or "No previous history in this session."
        )

        # F. Groq Inference
        print("   -> Generating Final Answer...")
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": query}
                ],
                model=GROQ_MODEL,
                temperature=0.1
            )
            answer = chat_completion.choices[0].message.content
            
            elapsed = time.time() - start_time
            print("\n" + "-"*30)
            print(f"ANSWER (Time: {elapsed:.2f}s):")
            print("-"*30)
            print(answer)
            print("-"*30)

            # G. Save to Memory
            memory_mgr.add_message(session_id, "user", query)
            memory_mgr.add_message(session_id, "assistant", answer)
            
            # Background-ish: update long term (we do it synchronously here for simplicity in CLI)
            memory_mgr.summarize_and_store_long_term(
                user_id, session_id, query, answer, 
                retrieved_docs=[{"text": d} for d in results["documents"][0]]
            )

        except Exception as e:
            print(f"LLM Error: {e}")

if __name__ == "__main__":
    main()
