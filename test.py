import numpy as np
import pandas as pd
import time
import os
import tracemalloc
import matplotlib.pyplot as plt
from bruteforce import bruteforce_search
from kdtree import build_kdtree, kdtree_search
from bbf import bbf_search

def read_data_from_file(filename):
    """Read data and query points from file."""
    with open(filename, 'r') as file:
        n, m, d = map(int, file.readline().split())
        
        # Read data points
        data_points = np.zeros((n, d))
        for i in range(n):
            data_points[i] = np.array(list(map(float, file.readline().split())))
        
        # Read query points
        query_points = np.zeros((m, d))
        for i in range(m):
            query_points[i] = np.array(list(map(float, file.readline().split())))
            
    return data_points, query_points

def evaluate_algorithms(data_points, query_points, max_queries=10, bbf_t_value=200):
    """
    Evaluate all three algorithms (brute force, KD-tree, BBF) on the given data.
    Returns metrics for each algorithm.
    """
    # Build KD-tree
    build_start = time.time()
    kdtree_root = build_kdtree(data_points)
    build_time = time.time() - build_start
    
    results = []
    num_queries = min(len(query_points), max_queries)
    
    for i in range(num_queries):
        query = query_points[i]
        result = {'query_id': i}
        
        # Brute force search
        tracemalloc.start()
        bf_start = time.time()
        bf_nn, bf_dist = bruteforce_search(data_points, query)
        bf_time = (time.time() - bf_start) * 1e6  # microseconds
        bf_memory = tracemalloc.get_traced_memory()[1]  # peak memory
        tracemalloc.stop()
        
        # KD-tree search
        tracemalloc.start()
        kd_start = time.time()
        kd_nn, kd_dist = kdtree_search(kdtree_root, query)
        kd_time = (time.time() - kd_start) * 1e6  # microseconds
        kd_memory = tracemalloc.get_traced_memory()[1]  # peak memory
        tracemalloc.stop()
        
        # BBF search
        tracemalloc.start()
        bbf_start = time.time()
        bbf_nn, bbf_dist = bbf_search(kdtree_root, query, bbf_t_value)
        bbf_time = (time.time() - bbf_start) * 1e6  # microseconds
        bbf_memory = tracemalloc.get_traced_memory()[1]  # peak memory
        tracemalloc.stop()
        
        # Calculate accuracy (ratio of distances)
        kd_accuracy = bf_dist / kd_dist if bf_dist > 0 and kd_dist > 0 else 1.0
        bbf_accuracy = bf_dist / bbf_dist if bf_dist > 0 and bbf_dist > 0 else 1.0
        
        # Store results
        result.update({
            'bf_time': bf_time,
            'bf_memory': bf_memory,
            'kd_time': kd_time,
            'kd_memory': kd_memory,
            'kd_accuracy': kd_accuracy,
            'bbf_time': bbf_time,
            'bbf_memory': bbf_memory,
            'bbf_accuracy': bbf_accuracy
        })
        
        results.append(result)
    
    # Calculate average metrics
    avg_results = {
        'build_time': build_time,
        'avg_bf_time': np.mean([r['bf_time'] for r in results]),
        'avg_bf_memory': np.mean([r['bf_memory'] for r in results]),
        'avg_kd_time': np.mean([r['kd_time'] for r in results]),
        'avg_kd_memory': np.mean([r['kd_memory'] for r in results]),
        'avg_kd_accuracy': np.mean([r['kd_accuracy'] for r in results]),
        'avg_bbf_time': np.mean([r['bbf_time'] for r in results]),
        'avg_bbf_memory': np.mean([r['bbf_memory'] for r in results]),
        'avg_bbf_accuracy': np.mean([r['bbf_accuracy'] for r in results])
    }
    
    return results, avg_results

def process_data_files(data_dir='./data', max_files=3, max_queries=10, bbf_t_value=200):
    """Process a small number of data files for testing."""
    all_results = []
    summary_results = []
    
    # Create results directory if it doesn't exist
    results_dir = './test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    for i in range(1, max_files + 1):
        filename = os.path.join(data_dir, f"{i}.txt")
        
        if not os.path.exists(filename):
            print(f"File {filename} not found, skipping.")
            continue
        
        print(f"Processing file {filename}...")
        
        try:
            # Read data from file
            data_points, query_points = read_data_from_file(filename)
            
            # Print some basic info about the data
            print(f"  Data points: {data_points.shape[0]}, Query points: {query_points.shape[0]}, Dimensions: {data_points.shape[1]}")
            
            # Evaluate algorithms
            results, avg_results = evaluate_algorithms(
                data_points, query_points, max_queries, bbf_t_value
            )
            
            # Add file info to results
            for r in results:
                r['file_id'] = i
                r['dimensions'] = data_points.shape[1]
                r['num_points'] = data_points.shape[0]
                all_results.append(r)
            
            # Add file info to summary
            avg_results['file_id'] = i
            avg_results['dimensions'] = data_points.shape[1]
            avg_results['num_points'] = data_points.shape[0]
            summary_results.append(avg_results)
            
            # Print summary for this file
            print(f"  Brute Force: {avg_results['avg_bf_time']:.2f} µs, KD-Tree: {avg_results['avg_kd_time']:.2f} µs (acc: {avg_results['avg_kd_accuracy']:.4f}), BBF: {avg_results['avg_bbf_time']:.2f} µs (acc: {avg_results['avg_bbf_accuracy']:.4f})")
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    # Save results to CSV
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(os.path.join(results_dir, 'test_results.csv'), index=False)
        print(f"Saved detailed results to {os.path.join(results_dir, 'test_results.csv')}")
    
    return all_results, summary_results

if __name__ == '__main__':
    # Process just a few data files for quick testing
    all_results, summary_results = process_data_files(
        data_dir='./data',
        max_files=3,      # Only process first 3 files
        max_queries=10,   # Only test 10 queries per file
        bbf_t_value=200
    )
    
    print("Test complete.") 