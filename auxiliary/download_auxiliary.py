import os
import json
import datasets

def main():
    print("Downloading Databricks Dolly 15k for auxiliary language training...")
    try:
        # Databricks Dolly is a gold-standard instruction tuning dataset for general NLP
        dataset = datasets.load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "auxiliary_master.jsonl")
    
    print(f"Loaded {len(dataset)} high-quality conversational records. Unifying schema to match LegalBench...")
    
    success_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in dataset:
            # Combine instruction and context if context exists
            instruction = row["instruction"]
            if row.get("context", ""):
                instruction = f"{instruction}\n\nContext: {row['context']}"
                
            # We map this into the exact same unified schema we used for LegalBench and LexGLUE
            master_record = {
                "task": f"dolly_{row['category']}", # Keeps track of whether it's summarization, QA, etc.
                "split": "train",
                "instruction": instruction,
                "response": row["response"]
            }
            
            f.write(json.dumps(master_record) + "\n")
            success_count += 1
            
    print(f"Successfully saved {success_count} auxiliary records to:")
    print(output_file)

if __name__ == "__main__":
    main()
