import os
import glob
import pandas as pd
import argparse
from evaluation import run_evaluation

# Default parameters for batch runs, can be overridden by command-line args
DEFAULT_NUM_DISTINCT_QUERIES_BATCH = 10 # Default for batch runs
DEFAULT_BBF_T_BATCH = 200

def run_batch_evaluation(data_dir, num_distinct_queries, bbf_t, output_summary_csv):
    """
    Runs evaluation for all .txt datasets in the specified directory.
    Saves a summary CSV of all results.
    """
    dataset_files = glob.glob(os.path.join(data_dir, "*.txt"))
    # Exclude known dummy files from specific test scripts if necessary, or ensure they are not in the primary data_dir for batch.
    dataset_files = [f for f in dataset_files if "dummy_" not in os.path.basename(f)]

    if not dataset_files:
        print(f"No .txt dataset files found in {data_dir}. Ensure datasets are present.")
        print("If you have dummy files like 'dummy_dataset.txt' or 'dummy_eval_dataset.txt', they are ignored by default.")
        print("If your actual data files contain 'dummy_' in their names, please adjust the filter in run_batch.py.")
        return

    print(f"Found dataset files for batch processing: {dataset_files}")
    all_results = []

    for dataset_path in dataset_files:
        print(f"\nProcessing dataset: {os.path.basename(dataset_path)} with num_distinct_queries={num_distinct_queries}, BBF t={bbf_t}")
        results = run_evaluation(
            dataset_filepath=dataset_path,
            num_distinct_queries=num_distinct_queries,
            bbf_t_param=bbf_t
        )
        if results and 'error' not in results: # Also check for info-only results if they should be skipped for main summary
            all_results.append(results)
        elif results and 'error' in results:
            print(f"Skipping {dataset_path} from summary due to error: {results['error']}")
        elif results and 'info' in results: # e.g. empty dataset, no queries run
             print(f"Dataset {dataset_path} processed with info: {results['info']}. It might not appear in the main summary if it lacks timing/accuracy data.")
             # Optionally, decide if these info-only results should also go into a simplified summary or be logged differently.
             # For now, only results with actual metrics will make it to the CSV.
             if 'kdtree_avg_time_s' in results: # A proxy to see if it has full data
                all_results.append(results)
        else:
            print(f"Skipping {dataset_path} as evaluation returned None or an unexpected result.")

    if not all_results:
        print("No valid results were generated from any dataset to form a summary.")
        return

    summary_df = pd.DataFrame(all_results)
    
    # Define a specific order for columns for better readability
    cols_order = [
        'dataset', 'dimensions', 'num_data_points', 'num_distinct_queries_tested', 
        'repetitions_per_query_for_timing', 'bbf_t',
        'kdtree_build_time_s', 'kdtree_nodes', 'memory_data_mb',
        'bruteforce_avg_time_s', 'bruteforce_accuracy',
        'kdtree_avg_time_s', 'kdtree_accuracy',
        'bbf_avg_time_s', 'bbf_accuracy'
    ]
    # Ensure only existing columns are selected, in the desired order
    summary_df = summary_df[[col for col in cols_order if col in summary_df.columns]]

    # Ensure results directory exists for the summary CSV
    results_dir = os.path.dirname(output_summary_csv)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    try:
        summary_df.to_csv(output_summary_csv, index=False)
        print(f"\nBatch evaluation complete. Summary saved to: {output_summary_csv}")
        print("\nSummary Table (from batch run):")
        print(summary_df.to_string())
    except Exception as e:
        print(f"Error saving batch summary CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run batch k-NN algorithm evaluations.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset files (default: data/).")
    parser.add_argument("--num_distinct_queries", type=int, default=DEFAULT_NUM_DISTINCT_QUERIES_BATCH, 
                        help=f"Number of distinct query points per dataset (default: {DEFAULT_NUM_DISTINCT_QUERIES_BATCH}). Each timed over repetitions.")
    parser.add_argument("--bbf_t", type=int, default=DEFAULT_BBF_T_BATCH, 
                        help=f"BBF parameter t (max leaves, default: {DEFAULT_BBF_T_BATCH}).")
    parser.add_argument("--output_csv", type=str, default="results/batch_summary.csv", 
                        help="Path to save the batch summary CSV (default: results/batch_summary.csv).")
    
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
        # Attempt to create a dummy file to guide the user
        if args.data_dir == "data":
            if not os.path.exists("data"):
                os.makedirs("data")
            dummy_file_path = "data/example_dataset.txt"
            if not os.path.exists(dummy_file_path):
                 with open(dummy_file_path, 'w') as f:
                    f.write("10 2\n") # 10 points, 2 dimensions
                    for i in range(10):
                        f.write(f"{i*0.5} {i*0.2}\n")
                 print(f"Created a sample dataset: {dummy_file_path}. Please populate {args.data_dir} with your actual datasets.")
        return

    run_batch_evaluation(args.data_dir, args.num_distinct_queries, args.bbf_t, args.output_csv)

if __name__ == "__main__":
    main()

    # To run this script:
    # 1. Make sure you have dataset files (e.g., dataset1.txt, dataset2.txt) in the 'data/' directory.
    #    The format should be: 
    #    <num_points> <dimensions>
    #    <data_point_1_dim1> <data_point_1_dim2> ...
    #    ...
    # 2. Run from the terminal: python run_batch.py
    #    Optional arguments: python run_batch.py --data_dir path/to/your/data --num_distinct_queries 50 --bbf_t 100 --output_csv custom_summary.csv 