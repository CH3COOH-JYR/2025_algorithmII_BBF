import numpy as np
import time
import psutil
import os
import sys # For sys.getsizeof if used for more detailed memory

from data_loader import load_data
from bruteforce import bruteforce_search, euclidean_distance
from kdtree import build_kdtree, kdtree_search, KDNode
from bbf import bbf_search

# Number of times to repeat each search for a single query point to get stable average time
NUM_REPETITIONS_PER_QUERY_FOR_TIMING = 1

def get_node_count(node):
    """Recursively counts the number of nodes in the k-d tree."""
    if node is None:
        return 0
    return 1 + get_node_count(node.left) + get_node_count(node.right)

def get_memory_usage_process():
    """Returns current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # Convert bytes to MB

def run_evaluation(dataset_filepath, num_distinct_queries=1, bbf_t_param=200):
    """
    Runs the evaluation for all algorithms on a given dataset.
    For each of 'num_distinct_queries' query points, each search algorithm is run ONCE.
    The "average query time" reported for the dataset will be the average over these num_distinct_queries runs.
    Records overall average query time, accuracy, and memory metrics.
    """
    print(f"\nStarting evaluation for dataset: {dataset_filepath}")
    print(f"Parameters: num_distinct_queries_from_this_file={num_distinct_queries}, BBF t={bbf_t_param}")

    try:
        points, dimensions = load_data(dataset_filepath)
    except Exception as e:
        print(f"Error loading dataset {dataset_filepath}: {e}")
        return None

    if len(points) == 0:
        print(f"Dataset {dataset_filepath} is empty or could not be loaded processed further.")
        # Return minimal info for empty datasets if they were successfully parsed as having 0 points
        if dimensions is not None: # data_loader might return (np.empty((0,D)), D)
             return {
                'dataset': os.path.basename(dataset_filepath),
                'dimensions': dimensions,
                'num_data_points': 0,
                'num_distinct_queries_tested': 0,
                'bbf_t': bbf_t_param,
                'error': 'Empty dataset'
            }
        return None
    
    if num_distinct_queries <= 0 :
        print(f"Warning: num_distinct_queries ({num_distinct_queries}) must be positive. Setting to 1.")
        num_distinct_queries = 1
        
    actual_num_distinct_queries = min(num_distinct_queries, len(points))
    if actual_num_distinct_queries == 0 and len(points) > 0: # Should not happen if len(points) > 0
        print("Warning: No query points selected despite data presence. Setting to 1 if possible.")
        actual_num_distinct_queries = 1 if len(points) > 0 else 0
    
    if actual_num_distinct_queries == 0:
        print("No query points to run (dataset likely too small or num_distinct_queries set too low). Skipping detailed evaluation.")
        return {
            'dataset': os.path.basename(dataset_filepath),
            'dimensions': dimensions,
            'num_data_points': len(points),
            'num_distinct_queries_tested': 0,
            'bbf_t': bbf_t_param,
            'info': 'No query points tested'
        }

    query_indices = np.random.choice(len(points), size=actual_num_distinct_queries, replace=False)
    actual_distinct_query_points = points[query_indices]

    results = {
        'dataset': os.path.basename(dataset_filepath),
        'dimensions': dimensions,
        'num_data_points': len(points),
        'num_distinct_queries_tested': len(actual_distinct_query_points),
        'bbf_t': bbf_t_param
    }

    results['memory_data_mb'] = points.nbytes / (1024 * 1024)
    print("Building k-d tree...")
    build_start_time = time.time()
    kdtree_root = build_kdtree(points)
    build_end_time = time.time()
    results['kdtree_build_time_s'] = build_end_time - build_start_time
    results['kdtree_nodes'] = get_node_count(kdtree_root)

    algo_performance = {
        'bruteforce': {'times_for_each_distinct_query': [], 'distances': [], 'found_nns': [], 'successful_queries': 0},
        'kdtree': {'times_for_each_distinct_query': [], 'distances': [], 'found_nns': [], 'successful_queries': 0},
        'bbf': {'times_for_each_distinct_query': [], 'distances': [], 'found_nns': [], 'successful_queries': 0}
    }

    print(f"Running {len(actual_distinct_query_points)} distinct query points (each timed once)...")
    
    for i, query_point in enumerate(actual_distinct_query_points):
        start_time_bf = time.perf_counter()
        true_nn, true_dist = bruteforce_search(points, query_point)
        end_time_bf = time.perf_counter()
        algo_performance['bruteforce']['times_for_each_distinct_query'].append(end_time_bf - start_time_bf)
        algo_performance['bruteforce']['distances'].append(true_dist)
        algo_performance['bruteforce']['found_nns'].append(true_nn)
        if true_dist is not None and true_dist >= 0: algo_performance['bruteforce']['successful_queries'] += 1

        start_time_kdt = time.perf_counter()
        kdt_nn, kdt_dist = kdtree_search(kdtree_root, query_point)
        end_time_kdt = time.perf_counter()
        algo_performance['kdtree']['times_for_each_distinct_query'].append(end_time_kdt - start_time_kdt)
        algo_performance['kdtree']['distances'].append(kdt_dist)
        algo_performance['kdtree']['found_nns'].append(kdt_nn)
        if true_dist is not None and kdt_dist is not None:
            if true_dist > 1e-9: 
                if (kdt_dist / true_dist) <= 1.050000001: algo_performance['kdtree']['successful_queries'] += 1
            elif abs(kdt_dist - true_dist) < 1e-9: algo_performance['kdtree']['successful_queries'] += 1

        start_time_bbf = time.perf_counter()
        bbf_nn, bbf_dist = bbf_search(kdtree_root, query_point, bbf_t_param)
        end_time_bbf = time.perf_counter()
        algo_performance['bbf']['times_for_each_distinct_query'].append(end_time_bbf - start_time_bbf)
        algo_performance['bbf']['distances'].append(bbf_dist)
        algo_performance['bbf']['found_nns'].append(bbf_nn)
        if true_dist is not None and bbf_dist is not None:
            if true_dist > 1e-9: 
                if (bbf_dist / true_dist) <= 1.050000001: algo_performance['bbf']['successful_queries'] += 1
            elif abs(bbf_dist - true_dist) < 1e-9: algo_performance['bbf']['successful_queries'] += 1
    
    for algo_name, perf_data in algo_performance.items():
        if perf_data['times_for_each_distinct_query']:
            results[f'{algo_name}_avg_time_s'] = np.mean(perf_data['times_for_each_distinct_query'])
            results[f'{algo_name}_accuracy'] = perf_data['successful_queries'] / len(perf_data['times_for_each_distinct_query'])
        else: # Should not happen if actual_distinct_query_points > 0
            results[f'{algo_name}_avg_time_s'] = 0
            results[f'{algo_name}_accuracy'] = 0
            
    print(f"Evaluation finished for: {dataset_filepath} (ran {len(actual_distinct_query_points)} distinct queries)")
    return results

if __name__ == '__main__':
    dummy_eval_path = "data/dummy_eval_dataset.txt"
    if not os.path.exists("data"): os.makedirs("data")
    
    np.random.seed(42)
    num_points_eval = 500 
    dimensions_eval = 5
    with open(dummy_eval_path, 'w') as f:
        f.write(f"{num_points_eval} {dimensions_eval}\n")
        for _ in range(num_points_eval):
            f.write(" ".join(map(str, np.random.rand(dimensions_eval) * 100)) + "\n")
    print(f"Created dummy dataset for evaluation: {dummy_eval_path}")
    
    # Test with 3 distinct queries, each run once for timing.
    eval_results = run_evaluation(dummy_eval_path, num_distinct_queries=3, bbf_t_param=10)
    
    if eval_results and 'error' not in eval_results and 'info' not in eval_results:
        print("\n--- Evaluation Results (run_evaluation test) ---")
        for key, value in eval_results.items():
            if isinstance(value, float): print(f"{key}: {value:.6f}")
            else: print(f"{key}: {value}")
    elif eval_results:
        print("\nEvaluation for dummy dataset produced info/error message:", eval_results)
    else:
        print("Evaluation (run_evaluation test) failed or produced no results.")

    # Test empty dataset handling
    dummy_empty_path = "data/dummy_empty_dataset.txt"
    with open(dummy_empty_path, 'w') as f:
        f.write("0 8\n") # 0 points, 8 dimensions
    print(f"\nCreated dummy empty dataset: {dummy_empty_path}")
    eval_empty_results = run_evaluation(dummy_empty_path, num_distinct_queries=1)
    if eval_empty_results:
        print("Results for empty dataset:", eval_empty_results) 