# Agentic NLP: Dataset Pipeline & Processing

This repository manages the data acquisition, exploratory data analysis (EDA), and data unification pipeline for the Legal AI training process. It transforms disjointed datasets into a strictly organized, single unified format (`JSONL`) ready for underlying LLM instruction-tuning.

## Repository Structure

```text
NLP/
├── data/                       # Final processed datasets
│   └── training_master.jsonl   # 235k cleanly formatted records (The final output)
│
├── data_processing/            # Core cleaning & unifcation pipeline
│   ├── config.py               # Tokens/length limits and paths
│   ├── clean.py                # Removes duplicates, caps lengths, purges empty rows
│   ├── unify_lex_glue.py       # Converts LexGLUE raw numbers to conversational texts
│   ├── merge_master.py         # Shuffles and merges all datasets together
│   ├── tokenize_and_stats.py   # Analyzes token counts against HuggingFace target models
│   ├── eda_report.py           # Generates statistics on lengths, tasks, and class distributions
│   └── eda_results.json        # Output of the EDA script
│
├── legalbench/                 # LegalBench raw download scripts & data
├── lex_glue/                   # LexGLUE raw download scripts & data
└── auxiliary/                  # Dolly 15k auxiliary download scripts & data
```

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
