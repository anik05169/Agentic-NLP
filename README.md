# Agentic NLP: Dataset Acquisition Pipeline

This repository contains the scripts necessary to download and unify the initial datasets required for training our legal AI models, following the "Create initial dataset(s)" mandate. We have prioritized **LegalBench**, **LexGLUE**, and an auxiliary linguistics dataset (**Databricks Dolly**).

## Prerequisites
Ensure your virtual environment is activated and the required Hugging Face libraries are installed:
```powershell
# Activate your environment
setconnectenv\Scripts\activate

# Install required packages
pip install "datasets<3.0.0" pandas
```

---

## 1. LegalBench Dataset
**Goal:** 162 unique short-form legal reasoning tasks (Contracts, Hearsay, Legal Citations, etc.) formatted identically for instruction-tuning.

### Steps to Download and Unify:
1. **Download Raw Data:** `python legalbench/download_legalbench_bulk.py` (saves 324 raw JSONL files).
2. **Unify Schemas:** `python legalbench/unify_legalbench.py` (combines all 162 tasks and injects the ground-truth text templates).
3. **Output File:** `legalbench/data/legalbench_master.jsonl`

### Example Output (`legalbench_master.jsonl`):
```json
{
  "task": "abercrombie",
  "split": "test",
  "instruction": "A mark is generic if it is the common name for the product... \n\nQ: The mark 'Virgin' for wireless communications. What is the type of mark?\nA: arbitrary\n\nQ: The mark 'Tasty' for bread. What is the type of mark?\n",
  "response": "descriptive"
}
```

---

## 2. LexGLUE Dataset
**Goal:** Long-form legal text understanding in both EU and US dialects (EUR-LEX, SCOTUS, ECtHR, Unfair ToS). Essential for training models to maintain context in dense 10+ page documents.

### Steps to Download:
1. **Download Raw Data:** `python lex_glue/download_lex_glue.py`
2. **Output Location:** `lex_glue/lex_glue_data/` (separated by test/train/validation splits per task)

### Example Output (Raw `unfair_tos` Task):
```json
{
  "text": "By using our Services, you agree to be bound by these Terms. The company reserves the right to modify these specific terms at any time with exactly 0 days notice to the user.",
  "labels": [1] 
}
```
*(Note: Label `1` represents an unfair consumer clause).*

---

## 3. Auxiliary Dataset (Databricks Dolly 15k)
**Goal:** General linguistic dataset to prevent the legal fine-tuning from destroying the model's broader conversational capabilities. 

### Steps to Download:
1. **Download & Auto-Unify Data:** `python auxiliary/download_auxiliary.py`
2. **Output File:** `auxiliary/data/auxiliary_master.jsonl`

### Example Output (`auxiliary_master.jsonl`):
```json
{
  "task": "dolly_summarization",
  "split": "train",
  "instruction": "Explain the plot of Cinderella in a single sentence.",
  "response": "A young girl named Cinderella attends a royal ball, meets a handsome prince, and escapes her cruel stepmother by leaving behind a glass slipper."
}
```
