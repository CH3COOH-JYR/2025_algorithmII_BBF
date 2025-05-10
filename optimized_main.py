import numpy as np
import pandas as pd
import time
import os
import psutil
import matplotlib.pyplot as plt
from bruteforce import bruteforce_search
from kdtree import build_kdtree, kdtree_search
from bbf import bbf_search
from multiprocessing import Pool, cpu_count
from functools import partial

def read_data_from_file(filename):
    """Read data and query points from file."""
    with open(filename, 'r') as file:
        n, m, d = map(int, file.readline().split())
        
        # 一次性预分配数组
        data_points = np.zeros((n, d), dtype=np.float32)  # 使用float32而不是float64减少内存
        for i in range(n):
            data_points[i] = np.fromstring(file.readline().strip(), sep=' ', dtype=np.float32)
        
        query_points = np.zeros((m, d), dtype=np.float32)
        for i in range(m):
            query_points[i] = np.fromstring(file.readline().strip(), sep=' ', dtype=np.float32)
            
    return data_points, query_points

def get_memory_usage():
    """获取当前进程的内存使用量（以字节为单位）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def evaluate_single_query(query_idx, query, data_points, kdtree_root, bbf_t_value):
    """评估单个查询点的性能"""
    result = {'query_id': query_idx}
    
    # 暴力搜索
    start_mem = get_memory_usage()
    bf_start = time.time()
    bf_nn, bf_dist = bruteforce_search(data_points, query)
    bf_time = (time.time() - bf_start) * 1e6  # 微秒
    bf_memory = get_memory_usage() - start_mem
    
    # KD树搜索
    start_mem = get_memory_usage()
    kd_start = time.time()
    kd_nn, kd_dist = kdtree_search(kdtree_root, query)
    kd_time = (time.time() - kd_start) * 1e6  # 微秒
    kd_memory = get_memory_usage() - start_mem
    
    # BBF搜索
    start_mem = get_memory_usage()
    bbf_start = time.time()
    bbf_nn, bbf_dist = bbf_search(kdtree_root, query, bbf_t_value)
    bbf_time = (time.time() - bbf_start) * 1e6  # 微秒
    bbf_memory = get_memory_usage() - start_mem
    
    # 计算准确率（距离比率）
    kd_accuracy = bf_dist / kd_dist if bf_dist > 0 and kd_dist > 0 else 1.0
    bbf_accuracy = bf_dist / bbf_dist if bf_dist > 0 and bbf_dist > 0 else 1.0
    
    # 存储结果
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
    
    return result

def evaluate_algorithms(data_points, query_points, max_queries=100, bbf_t_value=200, use_multiprocessing=True):
    """评估所有算法（暴力搜索、KD树、BBF）的性能"""
    # 构建KD树
    build_start = time.time()
    kdtree_root = build_kdtree(data_points)
    build_time = time.time() - build_start
    
    # 限制查询点数量
    num_queries = min(len(query_points), max_queries)
    queries = query_points[:num_queries]
    
    results = []
    if use_multiprocessing and num_queries >= 4:  # 只有当查询点数量足够多时才使用多进程
        # 使用多进程处理查询点
        num_processes = min(cpu_count(), 4)  # 限制进程数量，避免过多进程开销
        with Pool(processes=num_processes) as pool:
            eval_func = partial(
                evaluate_single_query, 
                data_points=data_points, 
                kdtree_root=kdtree_root, 
                bbf_t_value=bbf_t_value
            )
            # 将查询点与索引打包
            query_items = [(i, query) for i, query in enumerate(queries)]
            # 展开参数传递给eval_func
            results = pool.starmap(eval_func, [(idx, q) for idx, q in query_items])
    else:
        # 单进程处理
        for i, query in enumerate(queries):
            result = evaluate_single_query(i, query, data_points, kdtree_root, bbf_t_value)
            results.append(result)
    
    # 计算平均指标
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
    """处理指定目录中的所有数据文件"""
    all_results = []
    summary_results = []
    
    # 创建结果目录（如果不存在）
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 处理指定数量的文件
    for i in range(1, max_files + 1):
        filename = os.path.join(data_dir, f"{i}.txt")
        
        if not os.path.exists(filename):
            print(f"文件 {filename} 不存在，跳过。")
            continue
        
        # 测量处理每个文件的时间
        file_start_time = time.time()
        print(f"处理文件 {filename}...")
        
        try:
            # 读取数据
            data_points, query_points = read_data_from_file(filename)
            
            # 评估算法
            results, avg_results = evaluate_algorithms(
                data_points, query_points, max_queries, bbf_t_value
            )
            
            # 添加文件信息到结果
            for r in results:
                r['file_id'] = i
                r['dimensions'] = data_points.shape[1]
                r['num_points'] = data_points.shape[0]
                all_results.append(r)
            
            # 添加文件信息到摘要
            avg_results['file_id'] = i
            avg_results['dimensions'] = data_points.shape[1]
            avg_results['num_points'] = data_points.shape[0]
            summary_results.append(avg_results)
            
            # 计算并显示每个文件处理时间
            file_process_time = time.time() - file_start_time
            print(f"  完成处理文件 {i}.txt (耗时: {file_process_time:.2f}秒)")
            
            # 每10个文件保存一次中间结果
            if i % 10 == 0:
                temp_df = pd.DataFrame(summary_results)
                temp_df.to_csv(os.path.join(results_dir, f'summary_results_temp_{i}.csv'), index=False)
                print(f"  已保存中间结果到 {os.path.join(results_dir, f'summary_results_temp_{i}.csv')}")
        
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
    
    # 保存所有结果到CSV
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)
    
    # 保存摘要结果到CSV
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(os.path.join(results_dir, 'summary_results.csv'), index=False)
    
    return all_results, summary_results

def plot_results(summary_df, results_dir='./results'):
    """创建可视化结果的图表"""
    os.makedirs(results_dir, exist_ok=True)
    
    # 图1：平均查询时间比较
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['file_id'], summary_df['avg_bf_time'], 'b-', label='暴力搜索')
    plt.plot(summary_df['file_id'], summary_df['avg_kd_time'], 'g-', label='KD树')
    plt.plot(summary_df['file_id'], summary_df['avg_bbf_time'], 'r-', label='BBF')
    plt.xlabel('文件ID')
    plt.ylabel('平均查询时间 (微秒)')
    plt.title('平均查询时间比较')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'query_time_comparison.png'))
    plt.close()
    
    # 图2：内存使用比较
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['file_id'], summary_df['avg_bf_memory'], 'b-', label='暴力搜索')
    plt.plot(summary_df['file_id'], summary_df['avg_kd_memory'], 'g-', label='KD树')
    plt.plot(summary_df['file_id'], summary_df['avg_bbf_memory'], 'r-', label='BBF')
    plt.xlabel('文件ID')
    plt.ylabel('平均内存使用 (字节)')
    plt.title('内存使用比较')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'memory_usage_comparison.png'))
    plt.close()
    
    # 图3：准确率比较
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['file_id'], summary_df['avg_kd_accuracy'], 'g-', label='KD树')
    plt.plot(summary_df['file_id'], summary_df['avg_bbf_accuracy'], 'r-', label='BBF')
    plt.axhline(y=1.0, color='b', linestyle='--', label='最优 (暴力搜索)')
    plt.xlabel('文件ID')
    plt.ylabel('平均准确率 (距离比)')
    plt.title('准确率比较 (越低越好)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 图4：维度 vs 时间
    if len(set(summary_df['dimensions'])) > 1:  # 只在维度有多个值时绘制
        plt.figure(figsize=(12, 6))
        
        # 按维度分组
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
        
        plt.plot(dims, bf_times, 'bo-', label='暴力搜索')
        plt.plot(dims, kd_times, 'go-', label='KD树')
        plt.plot(dims, bbf_times, 'ro-', label='BBF')
        plt.xlabel('维度')
        plt.ylabel('平均查询时间 (微秒)')
        plt.title('维度对查询时间的影响')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'dimensions_vs_time.png'))
        plt.close()
    
    # 图5：数据点数量 vs 时间
    if len(set(summary_df['num_points'])) > 1:  # 只在数据点数量有多个值时绘制
        plt.figure(figsize=(12, 6))
        
        # 将数据点数量分箱
        summary_df['points_bin'] = pd.cut(summary_df['num_points'], bins=min(10, len(set(summary_df['num_points']))))
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
        
        plt.plot(bins, bf_times_by_bin, 'bo-', label='暴力搜索')
        plt.plot(bins, kd_times_by_bin, 'go-', label='KD树')
        plt.plot(bins, bbf_times_by_bin, 'ro-', label='BBF')
        plt.xlabel('数据点数量')
        plt.ylabel('平均查询时间 (微秒)')
        plt.title('数据点数量对查询时间的影响')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'num_points_vs_time.png'))
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='评估近邻搜索算法性能')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据文件目录')
    parser.add_argument('--max_files', type=int, default=100, help='处理的最大文件数量')
    parser.add_argument('--max_queries', type=int, default=100, help='每个文件处理的最大查询点数量')
    parser.add_argument('--bbf_t', type=int, default=200, help='BBF搜索的最大叶节点访问数量')
    parser.add_argument('--no_multiprocessing', action='store_true', help='禁用多进程')
    
    args = parser.parse_args()
    
    print("=== 启动近邻搜索算法评估 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"最大文件数: {args.max_files}")
    print(f"每个文件最大查询点数: {args.max_queries}")
    print(f"BBF最大叶节点访问数: {args.bbf_t}")
    print(f"多进程: {'禁用' if args.no_multiprocessing else '启用'}")
    print("="*30)
    
    # 处理所有数据文件并获取结果
    all_results, summary_results = process_data_files(
        data_dir=args.data_dir,
        max_files=args.max_files,
        max_queries=args.max_queries,
        bbf_t_value=args.bbf_t
    )
    
    # 绘制结果
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        plot_results(summary_df, results_dir='./results')
    
    print("处理完成。结果已保存在 'results' 目录中。")

if __name__ == '__main__':
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"总运行时间: {total_time:.2f}秒") 