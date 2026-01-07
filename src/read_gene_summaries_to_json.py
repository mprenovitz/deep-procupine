"""
This script reads gene summary files into one json file.
It can also be used with csv files, but the first line must be a a header, with the 
summaries column name passed to the '--colname' cli argument when running script.
"""
from multiprocessing import Pool
import os
from pathlib import Path
import argparse
import pandas as pd

def read_gene_summary_files(fp: str | Path):
    try:
        if not os.path.exists(fp):
            print(f"File {fp} does not exist.")
            return [] # Translates to NaN, removed using panda.
        elif os.path.splitext(fp)[-1] == ".DS_Store":
            return []
        with open(fp, 'r') as fh:
            summary = fh.read().strip()
        return [Path(fp).stem, summary]  # Use Path to ensure stem works
    except Exception as e:
        print(f"Error reading file {fp}: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read text files of gene summaries and into a consolidated json file",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing all gene summary text files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Directory to store json file. If not provided will be stored in the same level as parent of this script.",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        required=False,
        help="Number of cpu cores to use. If not provided will use all but one of the available cpus cores."
    )

    args = parser.parse_args()
    n_cpus = args.ncpus
    output_dir = args.output_dir
    input_dir = args.input_dir

    if not os.path.exists(input_dir):
        print(f"Provided input directory doesn't exist. Exiting program.")
        exit(0)
    
    if output_dir is None:
        output_dir = os.path.join(os.path.split(os.getcwd())[0], "data")
        os.makedirs(output_dir, exist_ok=True)  # Create if doesn't exist
        print(f"Output directory not provided. Using {output_dir}")

    if n_cpus is None:
        n_cpus = max(1, os.cpu_count() - 1)
    
    # Process the Gene summaries
    summary_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    print(f"{len(summary_files)} files read from {input_dir}.\nNow processing gene summaries...")

    # Run Process pool to process gene summary
    chunksize = max(1, len(summary_files) // (n_cpus * 4))

    with Pool(n_cpus) as p:
        results = list(p.imap_unordered(read_gene_summary_files, summary_files, chunksize=chunksize))
    
    columns = ["gene", "summary"]
    df = pd.DataFrame(data=results, columns=columns,)
    df.dropna()

    print(f"Successfully processed {df.shape[0]} files")
    print(f"First 5 results: {df.head()}")
    
    try: 
        fullpath = os.path.join(output_dir, "gene_summaries.json")
        # sample_csv_path = os.path.join(output_dir, "sample.csv")
        # sample_json_path = os.path.join(output_dir, "sample.json")

        df.to_json(fullpath, orient='records', indent=2)
        # df.head().to_json(sample_json_path, orient='records', indent=2)
        # df.head().to_csv(sample_csv_path, index=False)
        
        print(f"Saved full data to {fullpath}")
        #print(f"Saved sample data to {sample_csv_path} and {sample_json_path}")
    except Exception as e:
        print(f"Error saving summaries to file: {e}")