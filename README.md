# Legal AI — RAG Pipeline with GraphRAG and Memory

A production-ready legal Retrieval-Augmented Generation system built on 235k+ unified legal benchmark records. Features query expansion, real-time knowledge graph extraction, dual-layer memory, and strict source-grounded reasoning.

## Features

- **Query Expansion** — Rewrites casual questions into formal legal terms before retrieval
- **GraphRAG** — Extracts entity relationships from retrieved chunks to give the LLM structural understanding
- **Dual Memory** — Short-term session history (SQLite) + long-term semantic recall (ChromaDB)
- **Strict Citation** — Every factual statement is tied to a source; the model refuses to speculate
- **97% Recall@1** — Retrieval accuracy on the unified legal benchmark dataset

## Architecture

```
User Query
    ↓
Query Expansion (Llama-3.1-8B → formal legal terms)
    ↓
Vector Retrieval (ChromaDB → top-k legal chunks)
    ↓
Knowledge Graph (Llama-3.1-8B → entity relationships)
    ↓
Memory Recall (SQLite + ChromaDB → session + long-term context)
    ↓
LLM Reasoning (Llama-3.3-70B → cited legal answer)
    ↓
Memory Save (summarize → store for future sessions)
```

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `all-mpnet-base-v2` (768-dim) |
| Vector Store | ChromaDB (local) / Pinecone (cloud) |
| Heavy LLM | Groq — Llama-3.3-70B-Versatile |
| Light LLM | Groq — Llama-3.1-8B-Instant |
| Memory | SQLite (session) + ChromaDB (long-term) |
| API | FastAPI |
| Visualization | NetworkX + Pyvis |

## Quick Start

```bash
# Activate environment
setconnectenv\Scripts\activate
cd "LegalAI and ImgProcessor\NLP"

# Install dependencies
pip install fastapi uvicorn chromadb sentence-transformers torch groq python-dotenv

# Configure
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

## Usage

### Interactive CLI (Recommended)

```bash
cd nlp_pipeline
python interactive_search.py
```

### REST API

```bash
cd nlp_pipeline
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is indemnification?", "user_id": "demo", "session_id": "s1"}'
```

Response includes: `answer`, `expanded_query`, `citations`, and `timings` (expansion, embedding, retrieval, KG extraction, generation).

### Benchmarks

```bash
python nlp_pipeline/evaluate_rag.py
```

### Knowledge Graph

```bash
python nlp_pipeline/query_kg_generator.py "what is breach of contract"
```

## Environment Variables

| Variable | Required | Default |
|---|---|---|
| `GROQ_API_KEY` | Yes | — |
| `CHROMA_DB_PATH` | No | `chroma_db_fixed/` |
| `CHROMA_COLLECTION` | No | `legal-search-agent` |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` |
| `KG_GROQ_MODEL` | No | `llama-3.1-8b-instant` |
| `EMBED_MODEL` | No | `sentence-transformers/all-mpnet-base-v2` |

## Repository Structure

```
NLP/
├── README.md                    # This file
├── ARCHITECTURE.md              # Full technical reference and code descriptions
├── PROJECT_SUMMARY.md           # Activity log (Phase 1–8)
├── data_processing/             # Dataset preparation pipeline (5 scripts)
└── nlp_pipeline/                # Core runtime engine
    ├── api.py                   # FastAPI server (GraphRAG + Memory)
    ├── interactive_search.py    # CLI query tool (GraphRAG + Memory)
    ├── memory_manager.py        # Dual-memory engine
    ├── runtime_config.py        # Shared configuration
    ├── evaluate_rag.py          # Retrieval benchmarks
    ├── query_kg_generator.py    # KG extraction and graph builder
    ├── generate_kg_html.py      # Interactive graph visualization
    ├── test_memory_api.py       # Memory behavior tests
    └── lib/                     # Vendored JS for HTML graphs
```

See `ARCHITECTURE.md` for detailed descriptions of every file.

## Dataset

Built from unified legal benchmarks:

| Source | Records | Content |
|---|---|---|
| LegalBench | ~162 tasks | Short-form legal reasoning |
| LexGLUE | Multiple | SCOTUS opinions, EU law |
| LEDGAR | ~60k | Contract clause classification |
| **Total** | **~235k** | **Unified instruction-tuning format** |

## License

This repository inherits upstream dataset licenses. Treat the unified data as research and education material. Do a formal license review before commercial use.
