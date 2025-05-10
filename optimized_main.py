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
import matplotlib as mpl
import gc # 添加垃圾回收模块

# 配置matplotlib以正确显示中文
def configure_matplotlib_chinese():
    # 尝试不同的方法来支持中文字体
    try:
        # 方法1: 使用系统中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'STHeiti', 'SimSun', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 测试中文显示
        test_fig = plt.figure(figsize=(2, 1))
        plt.title("测试中文")
        plt.close(test_fig)
        print("成功配置中文字体")
    except Exception as e:
        print(f"中文字体配置失败，将使用英文标题: {e}")
        # 如果中文配置失败，使用英文标题
        global USE_ENGLISH_LABELS
        USE_ENGLISH_LABELS = True

# 全局变量，控制是否使用英文标题
USE_ENGLISH_LABELS = False

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

def _measure_algo_performance(algo_func, *args):
    """Helper to measure time and memory for a given algorithm function, using gc.collect()."""
    gc.collect()
    mem_before = get_memory_usage()
    time_start = time.time()
    
    algo_result = algo_func(*args) # Expected to return (nn, dist)
    
    time_taken = (time.time() - time_start) * 1e6 # microseconds
    gc.collect()
    mem_after = get_memory_usage()
    mem_delta = mem_after - mem_before # Raw delta, can be negative
    
    return algo_result, time_taken, mem_delta

def evaluate_single_query(query_idx, query, data_points, kdtree_root, bbf_t_value):
    """评估单个查询点的性能"""
    result = {'query_id': query_idx}
    
    # 暴力搜索
    (bf_nn, bf_dist), bf_time, bf_memory = _measure_algo_performance(
        bruteforce_search, data_points, query
    )
    
    # KD树搜索
    (kd_nn, kd_dist), kd_time, kd_memory = _measure_algo_performance(
        kdtree_search, kdtree_root, query
    )
    
    # BBF搜索
    (bbf_nn, bbf_dist), bbf_time, bbf_memory = _measure_algo_performance(
        bbf_search, kdtree_root, query, bbf_t_value
    )
    
    # 计算准确率（距离比率）
    kd_accuracy = bf_dist / kd_dist if bf_dist > 0 and kd_dist > 0 else 1.0
    bbf_accuracy = bf_dist / bbf_dist if bf_dist > 0 and bbf_dist > 0 else 1.0
    
    # 存储结果
    result.update({
        'bf_time': bf_time,
        'bf_memory': bf_memory, # Raw value, can be negative
        'kd_time': kd_time,
        'kd_memory': kd_memory, # Raw value, can be negative
        'kd_accuracy': kd_accuracy,
        'bbf_time': bbf_time,
        'bbf_memory': bbf_memory, # Raw value, can be negative
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
            results = pool.starmap(eval_func, query_items)
    else:
        # 单进程处理
        for i, query in enumerate(queries):
            result = evaluate_single_query(i, query, data_points, kdtree_root, bbf_t_value)
            results.append(result)
    
    # 计算平均指标
    avg_results = {
        'build_time': build_time,
        'avg_bf_time': np.mean([r['bf_time'] for r in results if r is not None]),
        'avg_bf_memory': np.mean([r['bf_memory'] for r in results if r is not None]),
        'avg_kd_time': np.mean([r['kd_time'] for r in results if r is not None]),
        'avg_kd_memory': np.mean([r['kd_memory'] for r in results if r is not None]),
        'avg_kd_accuracy': np.mean([r['kd_accuracy'] for r in results if r is not None]),
        'avg_bbf_time': np.mean([r['bbf_time'] for r in results if r is not None]),
        'avg_bbf_memory': np.mean([r['bbf_memory'] for r in results if r is not None]),
        'avg_bbf_accuracy': np.mean([r['bbf_accuracy'] for r in results if r is not None])
    }
    
    return results, avg_results

def process_data_files(data_dir='./data', max_files=100, max_queries=100, bbf_t_value=200, use_multiprocessing=True):
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
            results_for_file, avg_results_for_file = evaluate_algorithms(
                data_points, query_points, max_queries, bbf_t_value, use_multiprocessing
            )
            
            # 添加文件信息到结果
            for r in results_for_file:
                if r is None: continue
                r['file_id'] = i
                r['dimensions'] = data_points.shape[1]
                r['num_points'] = data_points.shape[0]
                all_results.append(r)
            
            # 添加文件信息到摘要
            avg_results_for_file['file_id'] = i
            avg_results_for_file['dimensions'] = data_points.shape[1]
            avg_results_for_file['num_points'] = data_points.shape[0]
            summary_results.append(avg_results_for_file)
            
            # 计算并显示每个文件处理时间
            file_process_time = time.time() - file_start_time
            print(f"  完成处理文件 {i}.txt (耗时: {file_process_time:.2f}秒)")
            
            # 每10个文件保存一次中间结果
            if i % 10 == 0 and summary_results:
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
    
    # 根据全局变量决定使用中文还是英文标签
    labels = {
        'title1': '平均查询时间比较' if not USE_ENGLISH_LABELS else 'Average Query Time Comparison',
        'title2': '内存使用比较' if not USE_ENGLISH_LABELS else 'Memory Usage Comparison',
        'title3': '准确率比较 (越低越好)' if not USE_ENGLISH_LABELS else 'Accuracy Comparison (lower is better)',
        'title4': '维度对查询时间的影响' if not USE_ENGLISH_LABELS else 'Effect of Dimensions on Query Time',
        'title5': '数据点数量对查询时间的影响' if not USE_ENGLISH_LABELS else 'Effect of Number of Points on Query Time',
        'xlabel1': '文件ID' if not USE_ENGLISH_LABELS else 'File ID',
        'xlabel4': '维度' if not USE_ENGLISH_LABELS else 'Dimensions',
        'xlabel5': '数据点数量' if not USE_ENGLISH_LABELS else 'Number of Points',
        'ylabel1': '平均查询时间 (微秒)' if not USE_ENGLISH_LABELS else 'Average Query Time (microseconds)',
        'ylabel2': '平均内存使用 (字节)' if not USE_ENGLISH_LABELS else 'Average Memory Usage (bytes)',
        'ylabel3': '平均准确率 (距离比)' if not USE_ENGLISH_LABELS else 'Average Accuracy Ratio (distance ratio)',
        'label1': '暴力搜索' if not USE_ENGLISH_LABELS else 'Brute Force',
        'label2': 'KD树' if not USE_ENGLISH_LABELS else 'KD-Tree',
        'label3': 'BBF' if not USE_ENGLISH_LABELS else 'BBF',
        'label4': '最优 (暴力搜索)' if not USE_ENGLISH_LABELS else 'Optimal (Brute Force)'
    }
    
    # 图1：平均查询时间比较
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['file_id'], summary_df['avg_bf_time'], 'b-', label=labels['label1'])
    plt.plot(summary_df['file_id'], summary_df['avg_kd_time'], 'g-', label=labels['label2'])
    plt.plot(summary_df['file_id'], summary_df['avg_bbf_time'], 'r-', label=labels['label3'])
    plt.xlabel(labels['xlabel1'])
    plt.ylabel(labels['ylabel1'])
    plt.title(labels['title1'])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'query_time_comparison.png'))
    plt.close()
    
    # 图2：内存使用比较
    plt.figure(figsize=(12, 6))
    # For plotting, clamp negative average memory to 0 for visual clarity. Raw data in CSV remains.
    avg_bf_memory_plot = np.maximum(0, summary_df['avg_bf_memory'])
    avg_kd_memory_plot = np.maximum(0, summary_df['avg_kd_memory'])
    avg_bbf_memory_plot = np.maximum(0, summary_df['avg_bbf_memory'])
    plt.plot(summary_df['file_id'], avg_bf_memory_plot, 'b-', label=labels['label1'])
    plt.plot(summary_df['file_id'], avg_kd_memory_plot, 'g-', label=labels['label2'])
    plt.plot(summary_df['file_id'], avg_bbf_memory_plot, 'r-', label=labels['label3'])
    plt.xlabel(labels['xlabel1'])
    plt.ylabel(labels['ylabel2'])
    plt.title(labels['title2'])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'memory_usage_comparison.png'))
    plt.close()
    
    # 图3：准确率比较
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['file_id'], summary_df['avg_kd_accuracy'], 'g-', label=labels['label2'])
    plt.plot(summary_df['file_id'], summary_df['avg_bbf_accuracy'], 'r-', label=labels['label3'])
    plt.axhline(y=1.0, color='b', linestyle='--', label=labels['label4'])
    plt.xlabel(labels['xlabel1'])
    plt.ylabel(labels['ylabel3'])
    plt.title(labels['title3'])
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
        
        plt.plot(dims, bf_times, 'bo-', label=labels['label1'])
        plt.plot(dims, kd_times, 'go-', label=labels['label2'])
        plt.plot(dims, bbf_times, 'ro-', label=labels['label3'])
        plt.xlabel(labels['xlabel4'])
        plt.ylabel(labels['ylabel1'])
        plt.title(labels['title4'])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'dimensions_vs_time.png'))
        plt.close()
    
    # 图5：数据点数量 vs 时间
    if len(set(summary_df['num_points'])) > 1:  # 只在数据点数量有多个值时绘制
        plt.figure(figsize=(12, 6))
        
        # 将数据点数量分箱
        summary_df['points_bin'] = pd.cut(summary_df['num_points'], bins=min(10, len(set(summary_df['num_points']))))
        bin_groups = summary_df.groupby('points_bin', observed=False) # Use observed=False for newer pandas
        bins = []
        bf_times_by_bin = []
        kd_times_by_bin = []
        bbf_times_by_bin = []
        
        for bin_name, group in bin_groups:
            if group.empty: continue
            bins.append(bin_name.mid)
            bf_times_by_bin.append(group['avg_bf_time'].mean())
            kd_times_by_bin.append(group['avg_kd_time'].mean())
            bbf_times_by_bin.append(group['avg_bbf_time'].mean())
        
        if bins: # Ensure there's data to plot
            plt.plot(bins, bf_times_by_bin, 'bo-', label=labels['label1'])
            plt.plot(bins, kd_times_by_bin, 'go-', label=labels['label2'])
            plt.plot(bins, bbf_times_by_bin, 'ro-', label=labels['label3'])
            plt.xlabel(labels['xlabel5'])
            plt.ylabel(labels['ylabel1'])
            plt.title(labels['title5'])
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, 'num_points_vs_time.png'))
        plt.close()

def main():
    import argparse
    
    # 配置matplotlib以支持中文
    configure_matplotlib_chinese()
    
    parser = argparse.ArgumentParser(description='评估近邻搜索算法性能')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据文件目录')
    parser.add_argument('--max_files', type=int, default=100, help='处理的最大文件数量')
    parser.add_argument('--max_queries', type=int, default=100, help='每个文件处理的最大查询点数量')
    parser.add_argument('--bbf_t', type=int, default=200, help='BBF搜索的最大叶节点访问数量')
    parser.add_argument('--no_multiprocessing', action='store_true', help='禁用多进程')
    parser.add_argument('--english', action='store_true', help='使用英文标签')
    
    args = parser.parse_args()
    
    # 如果指定使用英文，设置全局变量
    if args.english:
        global USE_ENGLISH_LABELS
        USE_ENGLISH_LABELS = True
    
    print("=== 启动近邻搜索算法评估 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"最大文件数: {args.max_files}")
    print(f"每个文件最大查询点数: {args.max_queries}")
    print(f"BBF最大叶节点访问数: {args.bbf_t}")
    print(f"多进程: {'禁用' if args.no_multiprocessing else '启用'}")
    print(f"标签语言: {'英文' if USE_ENGLISH_LABELS else '中文'}")
    print("="*30)
    
    # 处理所有数据文件并获取结果
    all_results, summary_results = process_data_files(
        data_dir=args.data_dir,
        max_files=args.max_files,
        max_queries=args.max_queries,
        bbf_t_value=args.bbf_t,
        use_multiprocessing=not args.no_multiprocessing
    )
    
    # 绘制结果
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        if not summary_df.empty:
            plot_results(summary_df, results_dir='./results')
    
    print("处理完成。结果已保存在 'results' 目录中。")

if __name__ == '__main__':
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"总运行时间: {total_time:.2f}秒") 