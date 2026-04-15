# Agentic NLP: Dataset Pipeline & Processing

This repository manages the data acquisition, exploratory data analysis (EDA), and data unification pipeline for the Legal AI training process. It transforms disjointed datasets into a strictly organized, single unified format (`JSONL`) ready for underlying LLM instruction-tuning.

## Repository Structure

```text
NLP/
├── data/                       # Processed datasets & vector chunks
│   ├── training_master.jsonl   # 235k cleanly formatted records (LLM text)
│   └── vectorized_chunks_master.jsonl # 330k embedded chunks (Vector Search)
│
├── data_processing/            # Cleaning, Unification, EDA
│   └── ...                     # (clean.py, merge_master.py, eda_report.py)
│
├── nlp_pipeline/               # Core Cloud RAG Architecture
│   ├── vectorize.py            # Embeds chunks via all-mpnet-base-v2
│   ├── index_pinecone_1gb.py   # High-perf async bulk uploader to Pinecone
│   ├── fix_chroma_local.py     # Local ChromaDB build script 
│   ├── rag_fixed.py            # Local RAG interactive terminal
│   ├── api.py                  # Production FastAPI backend
│   └── kg_generator.py         # Zero-Shot Knowledge Graph extraction via Groq
│
└── legalbench/, lex_glue/, auxiliary/ # Raw acquisition scripts
```

---

## The Cloud RAG Framework & Production API

Moving beyond raw strings to fully autonomous legal discovery, the `nlp_pipeline` directory houses our enterprise Retrieval-Augmented Generation (RAG) structure:

1.  **High-Quality Semantic Embedding:**
    We discarded standard masked LLMs in favor of `sentence-transformers/all-mpnet-base-v2`, an incredibly precise 768-dimension sentence embedder capable of correctly discriminating strict legal contexts from generic noise.
2.  **Scalable Vector Retrieval (Pinecone Cloud):**
    Chunks are bulk-uploaded in the background to an AWS Pinecone Serverless index (`index_pinecone_1gb.py`), bypassing local memory limits and delivering massive 2TB-ready scalability.
3.  **FastAPI REST Engine:**
    The pipeline runs asynchronously on a robust `uvicorn` FastAPI server (`api.py`), automatically validating incoming JSON via Pydantic and providing instant swagger docs.
4.  **Deep-Reasoning LLM (Groq Llama-3.3-70B):**
    With context successfully retrieved from Pinecone, the logic is ferried into `llama-3.3-70b-versatile` over the Groq inference engine, outputting precise, lightning-fast answers augmented with verbatim chunk citations. 
5.  **Knowledge Graph Extraction:**
    We also operate `kg_generator.py`, which autonomously crawls legal text and uses `llama-3.1-8b` to output structured JSON representations of legal relationship matrices seamlessly (nodes and edges suitable for NetworkX/Neo4j graphing).

---

## What the Datasets Are About

* **[LegalBench](https://huggingface.co/datasets/nguha/legalbench):** Evaluates an LLM's capacity for short-form legal reasoning. Contains 162 unique legal logic tasks (e.g., classifying a trademark as 'generic', determining if a contract clause permits a certain action, or distinguishing between case facts).
* **[LexGLUE](https://huggingface.co/datasets/lex_glue):** The premier benchmark for long-form legal text comprehension. Contains massive, real-world documents from the US and EU, including Supreme Court opinions, EU regulations, and Consumer Terms of Service. It focuses heavily on complex multi-label classification.
* **[Databricks Dolly (Auxiliary)](https://huggingface.co/datasets/databricks/databricks-dolly-15k):** A high-quality general conversational dataset containing human-generated tasks like brainstorming, QA, and creative writing. This acts as a stabilizer to prevent the AI from losing its basic conversational abilities (catastrophic forgetting) while being fine-tuned aggressively on dense legal jargon.

---

## Data Format Pipeline: Raw vs. Processed

**Initial Raw Text Formats:**
Before this pipeline, the datasets could not be trained together natively because their structures were entirely disparate:
* **LegalBench:** Arrived as 324 scattered files with raw text and hardcoded answers, lacking conversational direction. *(E.g., `{"text": "The mark 'Salt'...", "answer": "generic"}`)*
* **LexGLUE:** Arrived as blobs of text mapping to raw arrays of abstract integer IDs. *(E.g., `{"text": "JUSTICE STEVENS delivered the opinion...", "labels": [1, 5]}`)*
* **Dolly:** Arrived as JSON objects separated by `instruction` and `context` attributes.

**Final Processed Format (`data/training_master.jsonl`):**
To make the AI train successfully, we executed the `data_processing/` pipeline to unflatten the data, inject conversational prompts (e.g., *"Classify the main issue area of the following US Supreme Court opinion:"*), aggressively deduplicate rows (removing >50k exact text duplicates), enforce strict max token bounds, and unify the schemas perfectly. 

Everything was merged into a **single, rigorously shuffled JSONL file** heavily curated to **235,533 train-ready records**. Every single record across all domains now exactly matches this underlying instruction-tuning structure needed for the LLM:
```json
{
  "task": "lexglue_scotus",
  "split": "train",
  "instruction": "Classify the main issue area of the following US Supreme Court opinion:\n\nJUSTICE STEVENS delivered the opinion of the Court...",
  "response": "38",
  "source": "lexglue"
}
```

---

## ⚠️ Dataset Licensing & Usage Disclaimer

When deploying models trained on these datasets, it is critical to adhere to the underlying licensing terms of the original data. This repository unifies the datasets but **inherits their original licenses:**

*   **Databricks Dolly 15k:** Released under the **CC BY-SA 3.0** (Creative Commons Attribution-ShareAlike 3.0) license. Models trained on this data and any modifications must be distributed under the same terms.
*   **LegalBench:** Released primarily under **CC BY 4.0** or **CC BY-NC 4.0** depending on the specific sub-tasks. Some of the 162 tasks contain proprietary formatting. *Disclaimer: Commercial use of certain sub-tasks may be restricted.*
*   **LexGLUE:** A composite benchmark consisting of multiple licenses. Court opinions (SCOTUS, ECtHR) and EU regulations (EUR-LEX) are generally in the **Public Domain**. However, parts of the UNFAIR-ToS and LEDGAR datasets may carry restricted or non-commercial licenses.

> **Legal Disclaimer:** This processing pipeline and the resulting `training_master.jsonl` are provided for research and educational purposes. If you plan to deploy a model trained on this data in a commercial product, you must conduct a formal legal review of all upstream data licenses incorporated in LegalBench and LexGLUE.

---

## Running the Pipeline

If you need to regenerate the unified datasets from scratch:

**1. Install Dependencies**
Ensure your virtual environment is activated:
```powershell
setconnectenv\Scripts\activate
pip install "datasets<3.0.0" pandas
```

**2. Download Datasets**
```powershell
python legalbench/download_legalbench_bulk.py
python legalbench/unify_legalbench.py
python lex_glue/download_lex_glue.py
python auxiliary/download_auxiliary.py
```

**3. Run the Processing Pipeline**
```powershell
# (Optional) Generate the EDA stats
python data_processing/eda_report.py

# Unify, Clean, and Merge
python data_processing/unify_lex_glue.py
python data_processing/clean.py
python data_processing/merge_master.py
```
This systematically creates `data/training_master.jsonl`.
