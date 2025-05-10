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

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2)**2))

def evaluate_algorithms(data_points, query_points, max_queries=100, bbf_t_value=200):
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
        # If brute force distance is 0, we set accuracy to 1.0 for exact matches
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

def process_data_files(data_dir='./data', max_files=100, max_queries=100, bbf_t_value=200):
    """Process all data files in the specified directory."""
    all_results = []
    summary_results = []
    
    # Create results directory if it doesn't exist
    results_dir = './results'
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
            
            print(f"  Completed processing file {i}.txt")
        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    # Save all results to CSV
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)
    
    # Save summary results to CSV
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(results_dir, 'summary_results.csv'), index=False)
    
    return all_results_df, summary_df

def plot_results(summary_df, results_dir='./results'):
    """Create plots to visualize the results."""
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Average Query Time Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['file_id'], summary_df['avg_bf_time'], 'b-', label='Brute Force')
    plt.plot(summary_df['file_id'], summary_df['avg_kd_time'], 'g-', label='KD-Tree')
    plt.plot(summary_df['file_id'], summary_df['avg_bbf_time'], 'r-', label='BBF')
    plt.xlabel('File ID')
    plt.ylabel('Average Query Time (microseconds)')
    plt.title('Average Query Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'query_time_comparison.png'))
    
    # Plot 2: Memory Usage Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['file_id'], summary_df['avg_bf_memory'], 'b-', label='Brute Force')
    plt.plot(summary_df['file_id'], summary_df['avg_kd_memory'], 'g-', label='KD-Tree')
    plt.plot(summary_df['file_id'], summary_df['avg_bbf_memory'], 'r-', label='BBF')
    plt.xlabel('File ID')
    plt.ylabel('Average Memory Usage (bytes)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'memory_usage_comparison.png'))
    
    # Plot 3: Accuracy Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['file_id'], summary_df['avg_kd_accuracy'], 'g-', label='KD-Tree')
    plt.plot(summary_df['file_id'], summary_df['avg_bbf_accuracy'], 'r-', label='BBF')
    plt.axhline(y=1.0, color='b', linestyle='--', label='Optimal (Brute Force)')
    plt.xlabel('File ID')
    plt.ylabel('Average Accuracy Ratio (distance ratio)')
    plt.title('Accuracy Comparison (lower is better)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'))
    
    # Plot 4: Dimensions vs. Time
    plt.figure(figsize=(12, 6))
    
    # Group by dimensions
    dim_groups = summary_df.groupby('dimensions')
    dims = []
    bf_times = []
    kd_times = []
    bbf_times = []
    
    for dim, group in dim_groups:
        dims.append(dim)
        bf_times.append(group['avg_bf_time'].mean())
        kd_times.append(group['avg_kd_time'].mean())
        bbf_times.append(group['avg_bbf_time'].mean())
    
    plt.plot(dims, bf_times, 'bo-', label='Brute Force')
    plt.plot(dims, kd_times, 'go-', label='KD-Tree')
    plt.plot(dims, bbf_times, 'ro-', label='BBF')
    plt.xlabel('Dimensions')
    plt.ylabel('Average Query Time (microseconds)')
    plt.title('Effect of Dimensions on Query Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'dimensions_vs_time.png'))
    
    # Plot 5: Number of Points vs. Time
    plt.figure(figsize=(12, 6))
    
    # Bin the number of points
    summary_df['points_bin'] = pd.cut(summary_df['num_points'], bins=10)
    bin_groups = summary_df.groupby('points_bin')
    bins = []
    bf_times_by_bin = []
    kd_times_by_bin = []
    bbf_times_by_bin = []
    
    for bin_name, group in bin_groups:
        bins.append(bin_name.mid)
        bf_times_by_bin.append(group['avg_bf_time'].mean())
        kd_times_by_bin.append(group['avg_kd_time'].mean())
        bbf_times_by_bin.append(group['avg_bbf_time'].mean())
    
    plt.plot(bins, bf_times_by_bin, 'bo-', label='Brute Force')
    plt.plot(bins, kd_times_by_bin, 'go-', label='KD-Tree')
    plt.plot(bins, bbf_times_by_bin, 'ro-', label='BBF')
    plt.xlabel('Number of Points')
    plt.ylabel('Average Query Time (microseconds)')
    plt.title('Effect of Number of Points on Query Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'num_points_vs_time.png'))

if __name__ == '__main__':
    # Process all data files and get results
    all_results, summary_results = process_data_files(
        data_dir='./data',
        max_files=100,
        max_queries=100,
        bbf_t_value=200
    )
    
    # Plot the results
    plot_results(summary_results, results_dir='./results')
    
    print("Processing complete. Results are stored in the 'results' directory.") 