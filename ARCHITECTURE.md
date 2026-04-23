# Architecture and Code Reference

This document is the single technical reference for the Legal AI project. It covers the system design, the data pipeline, the memory architecture, and a description of every code file in the repository.

---

## System Design

```text
User Query
    │
    ▼
Query Expansion (Llama-3.1-8B → formal legal terms)
    │
    ▼
Vector DB (ChromaDB / Pinecone → retrieve top-k legal chunks)
    │
    ▼
Knowledge Graph (Llama-3.1-8B → extract entity relationships)
    │
    ▼
Memory Recall (SQLite short-term + ChromaDB long-term)
    │
    ▼
LLM Reasoning (Llama-3.3-70B → source-grounded legal answer)
    │
    ▼
Memory Save (summarize interaction → store for future recall)
```

### Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Embeddings | `all-mpnet-base-v2` (768-dim) | Semantic encoding for retrieval |
| Local Vector Store | ChromaDB | High-speed local retrieval (~200ms) |
| Cloud Vector Store | Pinecone Serverless | Production-scale cloud retrieval |
| Heavy Inference | Groq Llama-3.3-70B | Final legal answer generation |
| Light Inference | Groq Llama-3.1-8B | Query expansion, KG extraction, memory summarization |
| Short-Term Memory | SQLite | Session messages and interaction audit trails |
| Long-Term Memory | ChromaDB | Summarized past interactions as semantic vectors |
| API | FastAPI | REST interface for external consumers |
| Visualization | NetworkX + Pyvis | Interactive legal knowledge graphs |

---

## Repository Layout

```text
NLP/
├── .env                        # Local API keys (git-ignored)
├── .env.example                # Template for .env setup
├── .gitignore
├── README.md                   # Quick-start guide
├── ARCHITECTURE.md             # This file
├── PROJECT_SUMMARY.md          # Tech overview and project activity log
│
├── data/
│   └── training_master.jsonl   # 235k unified legal records
│
├── data_processing/            # Dataset preparation pipeline
│   ├── config.py
│   ├── eda_report.py
│   ├── clean.py
│   ├── unify_lex_glue.py
│   └── merge_master.py
│
├── nlp_pipeline/               # Core runtime engine
│   ├── runtime_config.py       # Shared configuration
│   ├── api.py                  # FastAPI server
│   ├── interactive_search.py   # CLI query tool (GraphRAG + Memory)
│   ├── memory_manager.py       # Dual-memory engine
│   ├── evaluate_rag.py         # Retrieval benchmarks
│   ├── query_kg_generator.py   # KG extraction and graph builder
│   ├── generate_kg_html.py     # Interactive graph visualization
│   ├── test_memory_api.py      # Memory behavior test suite
│   └── lib/                    # Vendored JS for HTML graphs
│
├── cuad/                       # CUAD contract dataset (external)
└── chroma_db_fixed/            # Local ChromaDB vector index
```

---

## Code File Descriptions

### `nlp_pipeline/` — Core Runtime

| File | Description |
|---|---|
| `runtime_config.py` | Loads `.env` and exposes shared constants: `CHROMA_DB_PATH`, `EMBED_MODEL`, `GROQ_MODEL`, `KG_GROQ_MODEL`, `TRAINING_MASTER_PATH`. All other scripts import from here. |
| `api.py` | FastAPI REST server. Exposes `POST /ask` for legal queries with memory-enhanced RAG. Handles query embedding, ChromaDB retrieval, short/long-term memory injection, Groq answer generation, and background memory saving. |
| `interactive_search.py` | Terminal-based legal query tool. The most feature-complete entry point: includes **Query Expansion** (rewrites casual questions into legal terms), **GraphRAG** (extracts entity relationships from retrieved chunks), **Dual Memory** (session + long-term recall), and strict source-grounded prompting. |
| `memory_manager.py` | Implements the `MemoryManager` class. Manages SQLite tables (`messages`, `interactions`) for short-term session history and audit trails. Manages a ChromaDB `user-memory` collection for long-term semantic recall. Includes Groq-powered interaction summarization. |
| `evaluate_rag.py` | Retrieval benchmarking suite. Samples records from `training_master.jsonl`, queries ChromaDB, and reports `Recall@1/3/5/10`, MRR, embedding latency, and retrieval latency. Saves a JSON report. |
| `query_kg_generator.py` | Query-driven GraphRAG engine. Takes a user query, retrieves context from Pinecone, extracts legal relationships with Groq, builds a structured KG JSON, and renders an interactive HTML visualization. Contains the shared `build_kg_json` function. |
| `generate_kg_html.py` | Knowledge graph visualizer. Reads KG JSON files and renders them as interactive HTML network graphs using Pyvis (with a local vis-network fallback). Supports both the clean schema and legacy NetworkX node-link format. |
| `test_memory_api.py` | Automated test for memory behavior. Sends three sequential queries to the API: a baseline question, a follow-up (tests short-term memory), and a new session for the same user (tests long-term memory). |
| `lib/` | Vendored JavaScript libraries (vis-network 9.1.2, tom-select, bindings) used by `generate_kg_html.py` for offline HTML graph rendering. |

### `data_processing/` — Dataset Pipeline

| File | Description |
|---|---|
| `config.py` | Central configuration for data processing. Defines paths to source datasets, maximum instruction/response lengths, deduplication strategy, and cleaning steps. |
| `eda_report.py` | Exploratory data analysis. Profiles dataset sizes, token/word distributions, label/task distributions, missing fields, and duplicate risks across all source datasets. |
| `clean.py` | Data cleaning pass. Performs whitespace normalization, unicode normalization, empty response filtering, length filtering, and exact text deduplication. |
| `unify_lex_glue.py` | LexGLUE normalizer. Converts LexGLUE-specific formats into the shared schema (`task`, `split`, `instruction`, `response`, `source`). |
| `merge_master.py` | Master merge pipeline. Combines normalized LegalBench, LexGLUE, and auxiliary records into a single `data/training_master.jsonl` file (~235k records). |

### Root Files

| File | Description |
|---|---|
| `README.md` | Quick-start guide: environment setup, indexing, API usage, benchmarking, and KG generation. |
| `ARCHITECTURE.md` | This file. Complete technical reference. |
| `PROJECT_SUMMARY.md` | High-level tech overview and chronological project activity log (Phase 1–8). |
| `.env.example` | Template showing required environment variables (`GROQ_API_KEY`, `CHROMA_DB_PATH`, etc.). |
| `.gitignore` | Excludes datasets, vector databases, generated HTML/JSON artifacts, and environment files from version control. |

---

## Data Pipeline

### Goal

Raw legal datasets arrive in different structures. The pipeline normalizes them into a shared schema:

```json
{
  "task": "dataset_or_task_name",
  "split": "train",
  "instruction": "User-facing legal instruction or question...",
  "response": "Expected answer, label, class, or generated response",
  "source": "legalbench"
}
```

### Source Families

| Source | Purpose | Content |
|---|---|---|
| LegalBench | Short-form legal reasoning | Issue spotting, rule application, clause reasoning |
| LexGLUE | Long-form legal comprehension | SCOTUS opinions, EU law, terms of service |
| LEDGAR | Contract clause classification | Clause text and clause labels |
| Auxiliary | General instruction-following | Conversational and general-purpose examples |

### Pipeline Stages

```text
1. Acquisition       → download/prepare raw source datasets
2. EDA               → python data_processing/eda_report.py
3. Unification       → python data_processing/unify_lex_glue.py
4. Cleaning          → python data_processing/clean.py
5. Merge             → python data_processing/merge_master.py
                        → produces data/training_master.jsonl
6. Vectorization     → embed and index into ChromaDB or Pinecone
```

---

## Memory Architecture

### Overview

| Layer | Storage | Scope | Contents |
|---|---|---|---|
| Short-term messages | SQLite | `session_id` | Recent user and assistant turns |
| Interaction trail | SQLite | `user_id` + `session_id` | Query, answer, retrieved docs, reasoning, success score |
| Long-term semantic | ChromaDB | `user_id` | Summarized past interactions as embedded vectors |

### Request Flow

```text
POST /ask (or CLI query)
  → expand query (8B model → formal legal terms)
  → embed expanded query
  → retrieve legal chunks from ChromaDB
  → extract structural relationships (8B model → mini-KG)
  → retrieve short-term session messages from SQLite
  → retrieve long-term user memories from ChromaDB
  → build final RAG prompt (context + KG + memories + history)
  → generate answer with Groq 70B
  → save user/assistant messages to SQLite
  → save structured interaction trail to SQLite
  → summarize and store long-term memory in background
```

### Runtime Files

| File | Location | Purpose |
|---|---|---|
| `chat_history.db` | `nlp_pipeline/` | SQLite database for messages and interactions (created at runtime) |
| ChromaDB `user-memory` | Configured via `CHROMA_DB_PATH` | Long-term semantic memory vectors |

### Future Production Options

| Option | Best For | Tradeoff |
|---|---|---|
| Pinecone | Durable semantic memory across deployments | Requires cloud API and namespace design |
| Redis | Fast session memory with TTL | Not a semantic store unless vector search is configured |
| Postgres + pgvector | One system for metadata and vectors | More setup, strong production ergonomics |

### Privacy Notes

- Do not store secrets or raw sensitive legal documents in memory summaries
- Treat `user_id` as pseudonymous unless user identity is required
- Add retention controls and a user-facing memory delete endpoint before production deployment
