"""
Unifies LexGLUE formatting into the same Instruction-Tuning schema as LegalBench/Dolly.
Converts {text, labels[]} -> {task, split, instruction, response}.
"""

import os
import json
from glob import glob
from config import PATHS

# Simple human-readable prompt templates for each sub-dataset
LEXGLUE_PROMPTS = {
    "eurlex": "Classify the following EU legal document into the relevant EUROVOC concepts:\n\n{text}\n\nRelevant concepts:",
    "ledgar": "Identify the main provision type (e.g., Governing Law, Severability) in the following contract clause:\n\n{text}\n\nProvision type:",
    "case_hold": "Identify the correct legal holding that is cited in the following judicial decision excerpt:\n\n{text}\n\nHolding citing:",
    "ecthr_a": "Based on the following facts from an ECtHR case, does it violate any articles of the ECHR? Identify the violated articles:\n\n{text}\n\nViolated articles:",
    "ecthr_b": "Based on the following facts, which ECHR articles were allegedly violated (regardless of court decision)?\n\n{text}\n\nAllegedly violated articles:",
    "scotus": "Classify the main issue area of the following US Supreme Court opinion:\n\n{text}\n\nIssue area:",
    "unfair_tos": "Read the following terms of service clause and determine if it contains any unfair terms. If yes, classify the unfairness types:\n\n{text}\n\nUnfair terms:"
}

def list_to_text(text_field) -> str:
    """Handles cases where text is a list of paragraphs (e.g., ECtHR)."""
    if isinstance(text_field, list):
        return "\n".join(str(s) for s in text_field)
    return str(text_field)

def main():
    print("Unifying LexGLUE into generic instruction-tuning format...")
    
    files = sorted(glob(os.path.join(PATHS["lg_dir"], "*.jsonl")))
    
    success_count = 0
    with open(PATHS["lg_master"], 'w', encoding='utf-8') as outfile:
        for fp in files:
            name = os.path.basename(fp).replace(".jsonl", "")
            
            # Extract dataset and split
            known_prefixes = ["ecthr_a", "ecthr_b", "unfair_tos", "case_hold"]
            dataset = None
            for prefix in known_prefixes:
                if name.startswith(prefix + "_"):
                    dataset = prefix
                    split = name[len(prefix) + 1:]
                    break
            if dataset is None:
                parts = name.rsplit("_", 1)
                dataset = parts[0]
                split = parts[1] if len(parts) > 1 else "unknown"
                
            prompt_template = LEXGLUE_PROMPTS.get(dataset, "Process the following document:\n\n{text}\n\nOutput:")
            
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        text = list_to_text(row.get("text", ""))
                        
                        # Get labels
                        labels = row.get("labels", row.get("label", []))
                        if not isinstance(labels, list):
                            labels = [labels]
                            
                        # Response: comma separated string of label IDs
                        if not labels:
                            response = "None"
                        else:
                            response = ", ".join(map(str, labels))
                            
                        # Some LexGLUE docs are exceptionally long, but we unify them first.
                        instruction = prompt_template.replace("{text}", text)
                        
                        master_record = {
                            "task": f"lexglue_{dataset}",
                            "split": split,
                            "instruction": instruction,
                            "response": response
                        }
                        
                        outfile.write(json.dumps(master_record) + "\n")
                        success_count += 1
                    except Exception as e:
                        print(f"Error processing record in {name}: {e}")
                        continue
                        
    print(f"\nSuccessfully unified {success_count} LexGLUE records into {PATHS['lg_master']}")

if __name__ == "__main__":
    main()
