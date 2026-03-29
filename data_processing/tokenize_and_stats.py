"""
Data tokenization statistics analyzer.
Computes token-based lengths to help finalize max sequence length.
"""

import json
import os
import statistics
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Please `pip install transformers` to use AutoTokenizer.")
    AutoTokenizer = None

from config import PATHS

# The default model tokenizer for reference
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def main():
    print(f"Tokenization Analysis using {MODEL_NAME} tokenizer")
    
    if AutoTokenizer is None:
        print("Skipping detailed token stats because `transformers` is not installed.")
        return
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Failed to load tokenizer {MODEL_NAME}: {e}")
        return

    out_file = PATHS["training_master"]
    
    if not os.path.exists(out_file):
        print(f"Could not find {out_file}. Run merge_master.py first.")
        return
        
    token_lengths = []
    
    print("Tokenizing sample of 5,000 records...")
    with open(out_file, 'r', encoding='utf-8') as f:
        records = []
        for i, line in enumerate(f):
            if i > 5000:
                break
            try:
                records.append(json.loads(line))
            except:
                continue
                
    for r in records:
        text_str = f"Instruction: {r.get('instruction', '')}\nAnswer: {r.get('response', '')}"
        tokens = tokenizer.encode(text_str, add_special_tokens=False)
        token_lengths.append(len(tokens))
        
    if not token_lengths:
        return
        
    print("\nToken Length Statistics (Sample):")
    print(f"  Count: {len(token_lengths):,}")
    print(f"    Min: {min(token_lengths):,}")
    print(f"    Max: {max(token_lengths):,}")
    print(f"   Mean: {round(statistics.mean(token_lengths)):,}")
    print(f" Median: {round(statistics.median(token_lengths)):,}")
    print(f"    P95: {round(sorted(token_lengths)[int(len(token_lengths) * 0.95)]):,}")
    
    print("\nRecommendation:")
    p95 = sorted(token_lengths)[int(len(token_lengths) * 0.95)]
    if p95 <= 512:
        print("  A max_seq_length of 512 is sufficient to preserve 95% of data.")
    elif p95 <= 2048:
        print("  A max_seq_length of 2048 is recommended.")
    else:
        print("  Dataset contains very long texts. Consider a max_seq_length of 4096 or chunking.")

if __name__ == "__main__":
    main()
