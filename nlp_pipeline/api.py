"""
FastAPI Server for Legal AI RAG Pipeline
Exposes the Pinecone + Groq backend as a RESTful JSON API.
"""

import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch
from groq import Groq

# ---- CONFIG ----
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX = "legal-search-agent"
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 5

SYSTEM_PROMPT = """You are a professional Legal AI advisor.
Analyze the provided context chunks and answer the query with precision.

RULES:
1. ONLY use the provided context. If the answer is not there, state that clearly.
2. Cite sources using [Chunk 1], [Chunk 2], etc.
3. Maintain a formal, legal tone.
4. Structure complex answers with bullet points."""

# Initialize FastAPI App
app = FastAPI(title="Legal AI Search API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models/clients
pc_index = None
embedder = None
groq_client = None

# Pydantic schemas for request/response validation
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K

class Citation(BaseModel):
    id: str
    score: float
    task: str
    text: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    timings: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    global pc_index, embedder, groq_client
    print("🚀 Starting up... Loading AI models and connecting to cloud.")
    
    # 1. Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc_index = pc.Index(PINECONE_INDEX)
    
    # 2. Local Embedder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    
    # 3. Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ System Ready!")

@app.get("/")
def read_root():
    return {"status": "Legal AI API is running. Go to /docs to test it."}

@app.post("/ask", response_model=QueryResponse)
def ask_legal_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    t_start_total = time.time()
    
    # 1. Embed Query
    t0 = time.time()
    query_vec = embedder.encode([request.query])[0].tolist()
    t_embed = time.time() - t0
    
    # 2. Retrieve from Pinecone
    t1 = time.time()
    try:
        results = pc_index.query(vector=query_vec, top_k=request.top_k, include_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    t_retrieve = time.time() - t1
    
    # Format Context
    context_str = ""
    citations = []
    
    for i, match in enumerate(results.get('matches', []), 1):
        text = match['metadata'].get('text', '')
        task = match['metadata'].get('task', '')
        score = match['score']
        
        citations.append(Citation(id=match['id'], score=round(score, 4), task=task, text=text[:200] + "..."))
        context_str += f"[Chunk {i}] (Task: {task}, Sim: {score:.4f})\n{text}\n\n"
        
    # 3. Generate Answer
    t2 = time.time()
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUERY: {request.query}"}
            ],
            temperature=0.1
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    t_generate = time.time() - t2
    
    answer = response.choices[0].message.content
    t_total = time.time() - t_start_total
    
    return QueryResponse(
        query=request.query,
        answer=answer,
        citations=citations,
        timings={
            "embedding_ms": round(t_embed * 1000, 2),
            "retrieval_ms": round(t_retrieve * 1000, 2),
            "generation_ms": round(t_generate * 1000, 2),
            "total_ms": round(t_total * 1000, 2)
        }
    )

if __name__ == "__main__":
    import uvicorn
    # To run programmatically
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
