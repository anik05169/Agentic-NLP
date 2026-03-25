import datasets
import pandas as pd
import os

def main():
    print("Loading LegalBench 'abercrombie' task dataset...")
    # Load a specific task from LegalBench
    dataset = datasets.load_dataset("nguha/legalbench", "abercrombie", trust_remote_code=True)
    
    # We will use the 'test' split to get a larger sample
    df = dataset["test"].to_pandas()
    
    print(f"Loaded {len(df)} rows from the dataset.")
    
    # Save the dataframe to a CSV file
    output_path = "abercrombie_sample.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Sample CSV successfully saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
