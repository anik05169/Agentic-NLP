"""
Legal AI RAG Evaluation & Benchmarking
Tests the local ChromaDB (mpnet) retrieval quality and latency.
"""

import json
import os
import time
import statistics
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from runtime_config import CHROMA_COLLECTION, CHROMA_DB_PATH, EMBED_MODEL, TRAINING_MASTER_PATH, ensure_parent_dir

# ---- CONFIG ----
COLLECTION_NAME = CHROMA_COLLECTION
EVAL_SAMPLE_SIZE = 100  # Number of queries to test
TOP_K_VALUES = [1, 3, 5, 10]

def main():
    print("=" * 60)
    print("LEGAL AI RAG EVALUATION SUITE")
    print("=" * 60)

    # 1. Setup Services
    print(f"\n[1/4] Connecting to ChromaDB at {CHROMA_DB_PATH}...")
    if not os.path.exists(CHROMA_DB_PATH):
        print("ERROR: Database not found! Complete indexing first.")
        return

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"ERROR: Collection '{COLLECTION_NAME}' not found.")
        return

    total_chunks = collection.count()
    print(f"       Database contains {total_chunks:,} chunks.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[2/4] Loading {EMBED_MODEL} on {device.upper()}...")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    # 3. Load Evaluation Data (Ground Truth)
    data_path = TRAINING_MASTER_PATH
    
    print(f"[3/4] Sampling {EVAL_SAMPLE_SIZE} records for ground truth...")
    eval_set = []
    with open(data_path, "r", encoding="utf-8") as f:
        # We sample distributed through the file for variety
        lines = f.readlines()
        step = max(1, len(lines) // EVAL_SAMPLE_SIZE)
        for i in range(0, len(lines), step):
            if len(eval_set) >= EVAL_SAMPLE_SIZE: break
            try:
                doc = json.loads(lines[i])
                if doc.get("instruction") and doc.get("task"):
                    eval_set.append({
                        "query": doc["instruction"],
                        "expected_task": doc["task"]
                    })
            except:
                continue

    # 4. Run Benchmarks
    print(f"\n[4/4] Executing benchmarks on {len(eval_set)} queries...")
    results = []
    latencies = []

    for i, item in enumerate(eval_set):
        query = item["query"]
        expected_task = item["expected_task"]

        t_start = time.time()
        # A. Embed
        q_vec = embedder.encode([query], show_progress_bar=False)[0].tolist()
        t_embed = time.time() - t_start

        # B. Retrieve
        t_ret_start = time.time()
        retrieved = collection.query(
            query_embeddings=[q_vec],
            n_results=max(TOP_K_VALUES),
            include=["metadatas"]
        )
        t_retrieve = time.time() - t_ret_start
        t_total = time.time() - t_start
        
        latencies.append({
            "embed": t_embed * 1000,
            "retrieve": t_retrieve * 1000,
            "total": t_total * 1000
        })

        # C. Evaluate
        retrieved_tasks = [m["task"] for m in retrieved["metadatas"][0]]
        
        hit_at = {k: (expected_task in retrieved_tasks[:k]) for k in TOP_K_VALUES}
        
        # Reciprocal Rank calculation
        mrr_score = 0
        if expected_task in retrieved_tasks:
            rank = retrieved_tasks.index(expected_task) + 1
            mrr_score = 1.0 / rank

        results.append({
            "hits": hit_at,
            "mrr": mrr_score
        })

        if (i + 1) % 10 == 0:
            print(f"      Processed {i+1}/{len(eval_set)} queries...")

    # --- CALCULATE METRICS ---
    avg_lat_total = statistics.mean([l["total"] for l in latencies])
    avg_lat_embed = statistics.mean([l["embed"] for l in latencies])
    avg_lat_ret   = statistics.mean([l["retrieve"] for l in latencies])
    
    mrr = statistics.mean([r["mrr"] for r in results])
    recall = {k: statistics.mean([1 if r["hits"][k] else 0 for r in results]) for k in TOP_K_VALUES}

    # --- OUTPUT REPORT ---
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS (N={len(eval_set)})")
    print(f"{'=' * 60}")
    
    print(f"\n  LATENCY (Averages)")
    print(f"  - Query Embedding:   {avg_lat_embed:.2f} ms")
    print(f"  - Vector Retrieval:  {avg_lat_ret:.2f} ms")
    print(f"  - End-to-End:        {avg_lat_total:.2f} ms")
    print(f"  - Throughput:        {1000/avg_lat_total:.1f} queries/sec")

    print(f"\n  RETRIEVAL QUALITY")
    for k in TOP_K_VALUES:
        print(f"  - Recall@{k:<2}:         {recall[k]*100:6.1f}%")
    print(f"  - MRR:               {mrr:.4f}")

    print(f"\n  SYSTEM STATS")
    print(f"  - DB Records:        {total_chunks:,}")
    print(f"  - Model:             {EMBED_MODEL}")
    print(f"  - Device:            {device.upper()}")
    print(f"{'=' * 60}")

    # Save report
    report_path = os.path.join(CHROMA_DB_PATH, "eval_report.json")
    report = {
        "metrics": {
            "recall": recall,
            "mrr": mrr,
            "latency_ms": {"avg_total": avg_lat_total, "avg_embed": avg_lat_embed, "avg_retrieve": avg_lat_ret}
        },
        "config": {"sample_size": EVAL_SAMPLE_SIZE, "model": EMBED_MODEL}
    }
    ensure_parent_dir(report_path)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n  Full report saved to: {report_path}")

if __name__ == "__main__":
    main()
