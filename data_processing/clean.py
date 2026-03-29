"""
Core cleaning functions applicable across all datasets.
Strips whitespace, removes empty pairs, deduplicates, and limits length based on config.
"""

import json
import os
from config import PROCESSING_CONFIG, PATHS

def word_count(text) -> int:
    return len(str(text).split())

def clean_dataset(input_file: str, output_file: str, max_words: int):
    print(f"Cleaning {os.path.basename(input_file)}...")
    
    if not os.path.exists(input_file):
        print(f"  File not found: {input_file}")
        return
        
    seen = set()
    cleaned = 0
    removed_empty = 0
    removed_dup = 0
    removed_length = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except:
                continue
                
            instruction = str(r.get("instruction", "")).strip()
            response = str(r.get("response", "")).strip()
            
            # 1. Empty responses constraint
            if len(response) < PROCESSING_CONFIG["min_response_length_chars"]:
                removed_empty += 1
                continue
                
            # 2. Deduplication
            if PROCESSING_CONFIG["dedup_strategy"] == "exact_match":
                key = (instruction, response)
                if key in seen:
                    removed_dup += 1
                    continue
                seen.add(key)
                
            # 3. Length Filtering
            if word_count(instruction) > max_words:
                removed_length += 1
                continue
                
            r["instruction"] = instruction
            r["response"] = response
            
            fout.write(json.dumps(r) + "\n")
            cleaned += 1
            
    total = cleaned + removed_empty + removed_dup + removed_length
    print(f"  Total: {total:,} | Cleaned: {cleaned:,}")
    print(f"  Removed -> Empty: {removed_empty:,} | Dups: {removed_dup:,} | Too Long: {removed_length:,}\n")

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    lb_out = os.path.join(base, "legalbench_clean.jsonl")
    lg_out = os.path.join(base, "lex_glue_clean.jsonl")
    aux_out = os.path.join(base, "auxiliary_clean.jsonl")
    
    # 1. Clean LegalBench (Short-form)
    clean_dataset(
        PATHS["lb_master"], 
        lb_out, 
        PROCESSING_CONFIG["max_instruction_words"]
    )
    
    # 2. Clean LexGLUE (Long-form)
    clean_dataset(
        PATHS["lg_master"], 
        lg_out, 
        PROCESSING_CONFIG["max_long_form_words"]
    )
    
    # 3. Clean Auxiliary (Short-form)
    clean_dataset(
        PATHS["aux_master"], 
        aux_out, 
        PROCESSING_CONFIG["max_instruction_words"]
    )
    
if __name__ == "__main__":
    main()
