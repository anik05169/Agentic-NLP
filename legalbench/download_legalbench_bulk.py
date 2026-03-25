import os
import datasets
import json

def main():
    dataset_name = "nguha/legalbench"
    print(f"Fetching configuration names (tasks) for {dataset_name}...")
    
    try:
        # Get all 162 tasks dynamically from HuggingFace
        tasks = datasets.get_dataset_config_names(dataset_name)
    except Exception as e:
        print(f"Error fetching tasks: {e}")
        return

    print(f"Successfully found {len(tasks)} tasks. Starting bulk download...")
    
    # Create an output directory for the raw files
    output_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failure_count = 0
    
    # Iterate through all tasks
    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Downloading task: {task}...")
        try:
            # We don't download the full dataset at once to save memory; we download task by task
            dataset = datasets.load_dataset(dataset_name, task)
            
            # Save Train and Test splits if they exist
            for split_name in dataset.keys():
                split_df = dataset[split_name].to_pandas()
                
                # We use JSONL format, which handles multiline strings and complex schemas better than CSV
                output_file = os.path.join(output_dir, f"{task}_{split_name}.jsonl")
                split_df.to_json(output_file, orient="records", lines=True)
                
            success_count += 1
            
        except Exception as e:
            print(f"Failed to download task '{task}'. Error: {e}")
            failure_count += 1
            
    print("\n--- Download Summary ---")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successfully saved: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"All files saved to: {output_dir}")

if __name__ == "__main__":
    main()
