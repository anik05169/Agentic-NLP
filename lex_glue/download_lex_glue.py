import os
import datasets

def main():
    dataset_name = "lex_glue"
    print(f"Fetching configuration names for Hugging Face benchmark: {dataset_name}...")
    
    try:
        # Get all subsets in LexGLUE (eurlex, scotus, ecthr_a, ecthr_b, unfair_tos, case_hold, ledgar)
        tasks = datasets.get_dataset_config_names(dataset_name)
    except Exception as e:
        print(f"Error fetching tasks: {e}")
        return

    print(f"Successfully found {len(tasks)} LexGLUE tasks: {tasks}")
    
    # Create an output directory for LexGLUE data
    output_dir = os.path.join(os.path.dirname(__file__), "lex_glue_data")
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failure_count = 0
    
    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Downloading LexGLUE subset: {task}...")
        try:
            dataset = datasets.load_dataset(dataset_name, task)
            
            # Save Train, Validation, and Test splits if they exist
            for split_name in dataset.keys():
                split_df = dataset[split_name].to_pandas()
                
                output_file = os.path.join(output_dir, f"{task}_{split_name}.jsonl")
                split_df.to_json(output_file, orient="records", lines=True)
                
            success_count += 1
            
        except Exception as e:
            print(f"Failed to download task '{task}'. Error: {e}")
            failure_count += 1
            
    print("\n--- LexGLUE Download Summary ---")
    print(f"Successfully saved subsets: {success_count}")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main()
