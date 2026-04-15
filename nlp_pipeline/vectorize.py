"""
Legal Text NLP Pipeline: Semantic Chunking & Vectorization
Uses LegalBERT or ModernBERT to generate dense embeddings for Legal Document Search.
"""

import json
import os
import re

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None

# Model defined in your roadmap
DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"

def semantic_chunking(text: str, max_words: int = 400) -> list:
    """
    Splits text by sentence boundaries (periods/question marks).
    Groups them into chunks staying under max_words to safely fit into the 512-token model limit.
    """
    sentences = re.split(r'(?<=[.!?])\s+', str(text).replace("\n", " "))
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        words = sent.split()
        if not words: continue
        
        if current_len + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = words
            current_len = len(words)
        else:
            current_chunk.extend(words)
            current_len += len(words)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "training_master.jsonl")
    out_path = os.path.join(base_dir, "data", "vectorized_chunks_master.jsonl")
    
    # 1. Load Data
    print(f"Loading data from {data_path}...")
    documents = []
    with open(data_path, "r", encoding="utf-8") as f:
        # Load the full unified 235,533 training master for embedding
        for i, line in enumerate(f):
            try:
                documents.append(json.loads(line))
            except:
                pass
                
    print(f"Loaded {len(documents)} records for vectorization pipeline test.")
    
    # 2. Semantic Chunking
    print("Applying Semantic Chunking (max ~400 words per chunk to fit 512 constraint)...")
    chunked_data = []
    for doc in documents:
        # We chunk the core textual content
        chunks = semantic_chunking(doc.get("instruction", ""))
        for idx, chunk_text in enumerate(chunks):
            chunked_data.append({
                "task_group": doc.get("task"),
                "chunk_id": idx,
                "text": chunk_text,
                "labels": doc.get("response", ""),
                "source": doc.get("source", "")
            })
            
    print(f"Generated {len(chunked_data)} total semantic chunks from {len(documents)} documents.")
    
    # 3. Vectorization (Embeddings)
    if SentenceTransformer is None:
        print("\n[!] WARNING: `sentence-transformers` or `torch` not installed.")
        print("To run the deep learning embedder locally, please run:")
        print("    pip install sentence-transformers torch")
        print("\nSkipping massive vector math; saving the chunked text data only.")
        
        with open(out_path, "w", encoding="utf-8") as f:
            for item in chunked_data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved unvectorized semantic chunks to {out_path}")
        return

    print(f"\nLoading Vectorization Model: {DEFAULT_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device.upper()}")
    
    model = SentenceTransformer(DEFAULT_MODEL, device=device)
    
    print("Computing LegalBERT embeddings in batches...")
    texts_to_embed = [item["text"] for item in chunked_data]
    
    # This generates a dense vector array (768 dimensions) for every chunk mathematically mapping its legal semantics
    embeddings = model.encode(texts_to_embed, batch_size=32, show_progress_bar=True)
    
    print("Vectorization complete. Saving embeddings alongside text chunks...")
    
    # Attach embeddings (convert numpy arrays to lists for JSON saving)
    for i, embedding in enumerate(embeddings):
        chunked_data[i]["vector"] = embedding.tolist()
        
    with open(out_path, "w", encoding="utf-8") as f:
        for item in chunked_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully saved {len(chunked_data)} vectorized records to {out_path}")
    print(f"Each record now contains its raw text AND a perfectly mapped {len(chunked_data[0]['vector'])}-dimensional tensor embedding.")
    print("These vectors are ready to be uploaded to a Vector Database (like Pinecone/Chroma) for the Legal Document Search Agent!")

if __name__ == "__main__":
    main()
