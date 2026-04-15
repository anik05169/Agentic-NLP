"""
Pinecone Indexing Script
Pulls the properly embedded vectors from local ChromaDB (all-mpnet-base-v2)
and uploads them to Pinecone Cloud.
"""

import os
import time

import chromadb
from pinecone import Pinecone, ServerlessSpec

# ---- CONFIG ----
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-search-agent"
DIMENSION = 768
BATCH_SIZE = 100
PARALLEL = 10

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "chroma_db_fixed")

    # 1. Pull vectors from local ChromaDB
    print("[1/3] Loading vectors from local ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="legal-search-agent")
    count = collection.count()
    print(f"       Found {count:,} vectors.")

    print("       Extracting all vectors + metadata...")
    data = collection.get(include=["embeddings", "metadatas", "documents"])
    ids = data["ids"]
    embeddings = data["embeddings"]
    metadatas = data["metadatas"]
    documents = data["documents"]
    print(f"       Extracted {len(ids):,} records.")

    # 2. Setup Pinecone
    print("[2/3] Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME in pc.list_indexes().names():
        print(f"       Deleting old '{INDEX_NAME}' index (had broken LegalBERT vectors)...")
        pc.delete_index(INDEX_NAME)
        time.sleep(5)

    print(f"       Creating fresh index '{INDEX_NAME}' (768-dim, cosine)...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10)

    index = pc.Index(INDEX_NAME, pool_threads=PARALLEL)

    # 3. Upload
    print(f"[3/3] Uploading {len(ids):,} vectors to Pinecone...")
    start_time = time.time()
    batch = []
    async_res = []
    total = 0

    for i in range(len(ids)):
        meta = {
            "text": (documents[i] or "")[:500],
            "task": metadatas[i].get("task", ""),
            "source": metadatas[i].get("source", ""),
        }
        batch.append((ids[i], embeddings[i], meta))

        if len(batch) >= BATCH_SIZE:
            res = index.upsert(vectors=batch, async_req=True)
            async_res.append(res)
            total += len(batch)
            batch = []

            if len(async_res) >= PARALLEL:
                for r in async_res:
                    r.get()
                async_res = []
                speed = total / (time.time() - start_time)
                print(f"       Uploaded {total:,} vectors ({speed:.1f} v/s)")

    if batch:
        index.upsert(vectors=batch)
        total += len(batch)

    for r in async_res:
        r.get()

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"UPLOAD COMPLETE")
    print(f"  Vectors:    {total:,}")
    print(f"  Time:       {elapsed:.1f}s")
    print(f"  Avg Speed:  {total/elapsed:.1f} v/s")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
