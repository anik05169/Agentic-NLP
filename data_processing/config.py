"""
Configuration parameters for NLP data processing.
Maps data processing variables to features like max length, split ratios, etc.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSING_CONFIG = {
    "max_instruction_words": 512,       # ~700 tokens
    "max_long_form_words": 2048,        # ~2700 tokens, for LexGLUE
    "min_response_length_chars": 1,     # Filter empty responses
    "auxiliary_ratio": 0.15,            # Dolly mix ratio to prevent catastrophic forgetting
    "dedup_strategy": "exact_match",    # Exact string matching
    "cleaning_steps": ["whitespace", "unicode", "dedup", "length_filter"]
}

PATHS = {
    "lb_master": os.path.join(BASE_DIR, "legalbench", "data", "legalbench_master.jsonl"),
    "lg_dir": os.path.join(BASE_DIR, "lex_glue", "lex_glue_data"),
    "aux_master": os.path.join(BASE_DIR, "auxiliary", "data", "auxiliary_master.jsonl"),
    
    "lg_master": os.path.join(BASE_DIR, "data_processing", "lex_glue_master.jsonl"),
    "training_master": os.path.join(BASE_DIR, "data", "training_master.jsonl")
}
