"""
Background Pinecone Ingestion Script (1GB Scale)
Reads ~65,000 chunks (approx. 1GB) from the raw dataset, embeds them
using the correct all-mpnet-base-v2 model, and uploads them to Pinecone. 
Designed to run in the background.
"""

import json
import os
import time
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch

# ---- CONFIG ----
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-search-agent" # Target index
NEW_MODEL = "sentence-transformers/all-mpnet-base-v2"
LIMIT = 65000 # ~1GB of data
BATCH_SIZE = 100
PARALLEL = 10

def main():
    print(f"Starting 1GB Background Ingestion Pipeline to Pinecone ({LIMIT} records)...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "vectorized_chunks_master.jsonl")

    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {NEW_MODEL} on {device.upper()}...")
    model = SentenceTransformer(NEW_MODEL, device=device)

    # 2. Connect to Pinecone
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME, pool_threads=PARALLEL)

    print("\nStarting Streaming Process (Read -> Embed -> Upload)...")
    t_start = time.time()
    
    total_processed = 0
    records_batch = []
    async_res = []

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if total_processed >= LIMIT:
                break
                
            rec = json.loads(line)
            
            records_batch.append({
                "id": f"chunk_{i}",
                "text": rec.get("text", ""),
                "task": rec.get("task_group", ""),
                "source": rec.get("source", ""),
            })

            # Process in chunks of BATCH_SIZE
            if len(records_batch) >= BATCH_SIZE:
                # 1. Embed on GPU
                texts = [r["text"] for r in records_batch]
                embeddings = model.encode(texts, show_progress_bar=False)
                
                # 2. Format for Pinecone
                upload_batch = []
                for j, r in enumerate(records_batch):
                    meta = {
                        "text": r["text"][:500], # Keep metadata lightweight for networking
                        "task": r["task"],
                        "source": r["source"],
                    }
                    upload_batch.append((r["id"], embeddings[j].tolist(), meta))
                
                # 3. Async Upsert
                res = index.upsert(vectors=upload_batch, async_req=True)
                async_res.append(res)
                
                total_processed += len(records_batch)
                records_batch = []

                # Ensure we don't overwhelm the thread pool
                if len(async_res) >= PARALLEL:
                    for r in async_res:
                        r.get()
                    async_res = []
                    
                    elapsed = time.time() - t_start
                    speed = total_processed / elapsed
                    print(f"  -> Uploaded {total_processed:,} / {LIMIT:,} chunks... (Speed: {speed:.1f} v/s)")

    # Process remaining stragglers
    if records_batch:
        texts = [r["text"] for r in records_batch]
        embeddings = model.encode(texts, show_progress_bar=False)
        upload_batch = [(r["id"], embeddings[j].tolist(), {"text": r["text"][:500], "task": r["task"], "source": r["source"]}) for j, r in enumerate(records_batch)]
        index.upsert(vectors=upload_batch)
        total_processed += len(records_batch)

    # Wait for all async tasks to finish
    for r in async_res:
        r.get()

    t_total = time.time() - t_start
    print(f"\n==================================================")
    print(f"1GB UPLOAD COMPLETE")
    print(f"  Vectors:    {total_processed:,}")
    print(f"  Time:       {t_total/60:.2f} minutes")
    print(f"  Avg Speed:  {total_processed/t_total:.1f} vectors/second")
    print(f"==================================================")

if __name__ == "__main__":
    main()
