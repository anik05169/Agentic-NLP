# Project Summary and Activity Log

This document provides a high-level overview of the Legal AI repository, including its tech stack, data foundations, and key milestones.

---

## Technical Architecture

A cloud-native, production-ready Retrieval-Augmented Generation system with GraphRAG and memory-enhanced agent capabilities.

- **Vector Database**: Pinecone Serverless for cloud storage of 330k+ embedded legal chunks.
- **Local Vector Store**: ChromaDB for high-speed local retrieval and benchmarking.
- **Inference Engines**: Groq with Llama-3.3-70B for legal reasoning and Llama-3.1-8B for lightweight extraction/summarization.
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2` for 768-dim semantic processing.
- **API Framework**: FastAPI for asynchronous REST integration.
- **Visualization**: Pyvis and NetworkX for interactive legal knowledge graphs.
- **Memory**: SQLite for short-term session memory and ChromaDB semantic vectors for long-term user memory.

## Dataset Foundations

We use unified legal benchmarks transformed into instruction-ready formats.

- **LegalBench**: 162 tasks for short-form legal logic and classification.
- **LexGLUE**: Heavy-duty benchmarks for long-form comprehension, including Supreme Court and EU law tasks.
- **LEDGAR**: Contract clause classification.
- **Master Format**: A consistent JSONL structure with 235k+ records optimized for fine-tuning, retrieval, and evaluation.

---

## Project Activity Log

### [Phase 1] Initial Acquisition and Setup (March 2026)

- **Dataset Discovery**: Researched and integrated the primary legal benchmarks: LegalBench for short-form reasoning and LexGLUE for long-form comprehension.
- **Automation**: Created bulk acquisition scripts such as `download_legalbench_bulk.py` and `download_lex_glue.py` to reduce manual downloads.
- **Environment**: Established the `setconnectenv` virtual environment and core project directory management.

### [Phase 2] Exploratory Data Analysis and Cleaning (Late March 2026)

- **EDA Implementation**: Built `eda_report.py` to profile millions of tokens across datasets.
- **Optimization**: Identified and removed over 50,000 duplicate text records that would have skewed model training.
- **Categorization**: Developed logic to map raw legal tasks into consistent `instruction` and `response` pairs.

### [Phase 3] Master Dataset Unification (Early April 2026)

- **Schema Alignment**: Developed `unify_lex_glue.py` and related processing scripts to bridge disparate legal data formats.
- **Master Merge**: Created the `merge_master.py` pipeline, generating a single 235,533-record `training_master.jsonl` file.
- **Instruction Tuning Format**: Injected conversational prompts into raw legal data to make it compatible with modern LLM training workflows.

### [Phase 4] Cloud-Native RAG and GraphRAG (April 10, 2026)

- **Vector Migration**: Shifted from memory-limited local storage experiments to Pinecone Serverless.
- **Embedding Precision**: Fixed embedding mismatch issues by standardizing on 768-dim `all-mpnet-base-v2`.
- **API Launch**: Deployed a FastAPI backend (`api.py`) for real-time querying with citations.
- **Knowledge Graphs**: Implemented `query_kg_generator.py` to visualize structural relationships in legal documents.

### [Phase 5] Security and Repo Hardening (April 15, 2026)

- **Security Audit**: Scrubbed Pinecone and Groq API keys from source code.
- **Environment Management**: Introduced `.env` and `.env.example` for secure local development.
- **Cleanup**: Purged one-time query files, temporary agent logs, and sensitive text files from the repository.
- **Deployment**: Consolidated project history and pushed a secured, high-performance version to GitHub.

### [Phase 6] Local ChromaDB and High-Fidelity Benchmarking (April 17, 2026)

- **RAG Evaluation**: Implemented the `evaluate_rag.py` benchmarking suite.
- **Performance Breakthrough**: Achieved 97% retrieval accuracy (`Recall@1`) and MRR of 0.97 across unified legal datasets.
- **Latency Optimization**: Optimized local retrieval to about 274 ms end-to-end using GPU-accelerated embeddings.

### [Phase 7] Documentation and Agent Memory (April 17, 2026)

- **README Refresh**: Rebuilt the README around setup, indexing, API usage, benchmarks, knowledge graphs, and memory testing.
- **Pipeline Documentation**: Added `DATA_PIPELINE.md` to explain acquisition, EDA, cleaning, schema unification, merge, indexing, and validation.
- **Memory Documentation**: Added `MEMORY.md` to define short-term session memory, long-term semantic memory, and future Pinecone/Redis options.
- **Memory Upgrade**: Updated the API memory path with `user_id` for cross-session recall, `session_id` for short-term context, SQLite interaction trails, and ChromaDB long-term memory summaries.

### [Phase 8] GraphRAG, Query Expansion, and Repository Consolidation (April 22, 2026)

- **Repository Cleanup**: Removed redundant one-time scripts (`fuzzy_accuracy.py`, `generate_audit_report.py`, `chunking_benchmark.py`, `tokenize_and_stats.py`, etc.) to streamline the codebase.
- **GraphRAG Integration**: Embedded a real-time Knowledge Graph extraction step into `interactive_search.py`. Retrieved legal chunks are analyzed for entity relationships before the final answer is generated.
- **Query Expansion**: Added an LLM-powered query rewriting step that converts casual user questions into formal legal retrieval queries, dramatically improving vector search precision.
- **Prompt Hardening**: Implemented a strict, source-grounded system prompt to enforce citation discipline and prevent hallucination.
- **Memory CLI**: Upgraded `interactive_search.py` from a basic RAG tool to a full memory-enhanced GraphRAG agent.
- **Documentation Consolidation**: Merged `MEMORY.md`, `DATA_PIPELINE.md`, and `AGENT_CHANGES.md` into a single `ARCHITECTURE.md` with complete code file descriptions.
- **Test Suite**: Re-established `test_memory_api.py` for cross-session memory validation.
