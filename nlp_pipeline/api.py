"""
FastAPI Server for Legal AI RAG Pipeline
Supports query expansion, GraphRAG entity extraction, short-term SQLite
session memory, and long-term semantic memory for user-specific recall.
"""

import json
import os
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer
import torch
from groq import Groq

# Internal modules
from memory_manager import MemoryManager
from runtime_config import CHROMA_COLLECTION, CHROMA_DB_PATH, EMBED_MODEL, GROQ_MODEL, KG_GROQ_MODEL

# ---- CONFIG ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = CHROMA_COLLECTION
TOP_K_LEGAL = 5
TOP_K_MEMORY = 3

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

# ---- HELPER FUNCTIONS ----

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
    except Exception:
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
    except Exception:
        return "Structural analysis unavailable."


# ---- FASTAPI APP ----

app = FastAPI(title="Legal AI Search API (GraphRAG + Memory)", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
chroma_client = None
legal_collection = None
embedder = None
groq_client = None
memory_mgr = None

# Pydantic schemas
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    session_id: str = "default_session"
    top_k: int = TOP_K_LEGAL

class Citation(BaseModel):
    id: str
    score: float
    task: str
    text: str

class QueryResponse(BaseModel):
    query: str
    expanded_query: str
    answer: str
    citations: List[Citation]
    timings: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    global chroma_client, legal_collection, embedder, groq_client, memory_mgr
    print("Starting up GraphRAG + Memory Legal AI...")
    
    # 1. Local ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    legal_collection = chroma_client.get_collection(COLLECTION_NAME)
    
    # 2. Local Embedder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    
    # 3. Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    # 4. Memory Manager
    memory_mgr = MemoryManager(chroma_client, embedder, groq_client)
    
    print(f"System ready. GraphRAG + Memory active on {device.upper()}.")

@app.get("/")
def read_root():
    return {"status": "Legal AI API v3 is running with GraphRAG and memory enabled.", "chat_ui": "/chat"}

@app.get("/chat")
def serve_chat():
    """Serves the web chat frontend."""
    chat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")
    return FileResponse(chat_path, media_type="text/html")

@app.post("/ask", response_model=QueryResponse)
def ask_legal_question(request: QueryRequest, background_tasks: BackgroundTasks):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    t_start_total = time.time()
    
    # 1. Query Expansion
    t0 = time.time()
    expanded_query = expand_query(groq_client, request.query)
    t_expand = time.time() - t0
    
    # 2. Embed the EXPANDED query for better retrieval
    t1 = time.time()
    query_vec = embedder.encode([expanded_query])[0].tolist()
    t_embed = time.time() - t1
    
    # 3. Retrieve Legal Context
    t2 = time.time()
    try:
        results = legal_collection.query(query_embeddings=[query_vec], n_results=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    t_retrieve = time.time() - t2
    
    # 4. Format context and citations
    context_str = ""
    citations = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ), 1):
        task = meta.get("task", "unknown")
        score = 1.0 - (dist / 2.0) if dist < 2.0 else 0.0
        citations.append(Citation(id=f"doc_{i}", score=round(score, 4), task=task, text=doc[:200] + "..."))
        context_str += f"[Source {i}] (Task: {task})\n{doc}\n\n"
    
    # 5. GraphRAG: Extract Structural Relationships
    t3 = time.time()
    structural_rels = extract_structural_context(groq_client, context_str)
    t_kg = time.time() - t3
    
    # 6. Retrieve Memories
    short_mem = memory_mgr.get_short_term_memory(request.session_id)
    history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in short_mem])
    long_mem = memory_mgr.get_long_term_memory(request.user_id, request.query, top_k=TOP_K_MEMORY)
    
    # 7. Build final prompt and generate answer
    t4 = time.time()
    final_prompt = SYSTEM_PROMPT.format(
        legal_context=context_str or "No direct legal matches found.",
        structural_relationships=structural_rels,
        long_term_memory=long_mem or "No relevant past interactions found.",
        short_term_memory=history_str or "No previous history in this session."
    )
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": request.query}
            ],
            temperature=0.1
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    t_generate = time.time() - t4
    
    answer = response.choices[0].message.content
    t_total = time.time() - t_start_total
    
    # 8. Save to Memory (short-term synchronous, long-term background)
    memory_mgr.add_message(request.session_id, "user", request.query)
    memory_mgr.add_message(request.session_id, "assistant", answer)
    retrieved_memory_docs = [citation.dict() for citation in citations]
    memory_mgr.record_interaction(
        user_id=request.user_id,
        session_id=request.session_id,
        query=request.query,
        answer=answer,
        retrieved_docs=retrieved_memory_docs,
        intermediate_reasoning=(
            f"Expanded query to: '{expanded_query[:100]}'. "
            f"Extracted {structural_rels.count(chr(10)) + 1} structural relationships. "
            "Retrieved legal chunks, combined memory, and produced a cited answer."
        ),
        success_score=1.0 if citations else 0.5,
    )
    
    background_tasks.add_task(
        memory_mgr.summarize_and_store_long_term, 
        request.user_id,
        request.session_id, 
        request.query, 
        answer,
        retrieved_memory_docs,
        f"Query expanded to legal terms. {len(citations)} sources retrieved. GraphRAG applied."
    )
    
    return QueryResponse(
        query=request.query,
        expanded_query=expanded_query,
        answer=answer,
        citations=citations,
        timings={
            "expansion_ms": round(t_expand * 1000, 2),
            "embedding_ms": round(t_embed * 1000, 2),
            "retrieval_ms": round(t_retrieve * 1000, 2),
            "kg_extraction_ms": round(t_kg * 1000, 2),
            "generation_ms": round(t_generate * 1000, 2),
            "total_ms": round(t_total * 1000, 2)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
