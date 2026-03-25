import os
import json
import pandas as pd
from glob import glob

def generate_prompt(prompt_template: str, row_dict: dict) -> str:
    """Uses LegalBench's template substitution logic."""
    prompt = str(prompt_template)
    for k, v in row_dict.items():
        placeholder = "{{" + k + "}}"
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, str(v))
    return prompt.strip()

def get_target_value(row_dict: dict, prompt_template: str) -> str:
    """Finds the ground truth answer/label that was not part of the prompt template."""
    # Common target column names in LegalBench
    candidates = ["answer", "label", "class", "target", "verdict"]
    
    for candidate in candidates:
        if candidate in row_dict:
            # Verify the candidate isn't part of the input prompt
            if "{{" + candidate + "}}" not in prompt_template:
                return str(row_dict[candidate])
                
    # Fallback to any column not used in the template (ignoring 'index' or 'id')
    for k, v in row_dict.items():
        if k.lower() not in ["index", "id"] and "{{" + k + "}}" not in prompt_template:
            return str(v)
            
    return ""

def main():
    raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    output_file = os.path.join(os.path.dirname(__file__), "data", "legalbench_master.jsonl")
    tasks_dir = os.path.join(os.path.dirname(__file__), "tasks")
    
    raw_files = glob(os.path.join(raw_dir, "*.jsonl"))
    
    print(f"Found {len(raw_files)} raw split files. Starting unification process...")
    
    success_count = 0
    missing_template = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in raw_files:
            file_name = os.path.basename(file_path)
            # Extact task name from file name (e.g. abercrombie_test.jsonl -> abercrombie)
            # Note: tasks can contain underscores, so we split from the right
            task_name = file_name.rsplit("_", 1)[0]
            split_name = file_name.rsplit("_", 1)[1].split(".")[0]
            
            prompt_file = os.path.join(tasks_dir, task_name, "base_prompt.txt")
            
            if not os.path.exists(prompt_file):
                missing_template += 1
                continue
                
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
                
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        row_dict = json.loads(line)
                        instruction = generate_prompt(prompt_template, row_dict)
                        response = get_target_value(row_dict, prompt_template)
                        
                        master_record = {
                            "task": task_name,
                            "split": split_name,
                            "instruction": instruction,
                            "response": response
                        }
                        
                        outfile.write(json.dumps(master_record) + "\n")
                        success_count += 1
                    except Exception as e:
                        pass
                        
    print(f"\n--- Unification Summary ---")
    print(f"Total prompt-response pairs generated: {success_count:,}")
    print(f"Tasks skipped due to missing prompt template: {missing_template // 2}") 
    print(f"Master dataset saved successfully at: {output_file}")

if __name__ == "__main__":
    main()
