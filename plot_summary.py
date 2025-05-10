import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib as mpl

# 全局变量，控制是否使用英文标题
USE_ENGLISH_LABELS = False

# 配置matplotlib以正确显示中文
def configure_matplotlib_chinese():
    global USE_ENGLISH_LABELS
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'STHeiti', 'SimSun', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 测试中文显示，如果失败则USE_ENGLISH_LABELS会自动在plot_results中被处理或在main中设置
        test_fig = plt.figure(figsize=(2, 1))
        plt.title("测试中文") # Plotting a test title
        plt.close(test_fig)
        print("成功配置中文字体")
    except Exception as e:
        print(f"中文字体配置失败，将自动尝试使用英文标题: {e}")
        USE_ENGLISH_LABELS = True # Fallback to English if Chinese font setup fails

def plot_summary_results(summary_df, output_dir):
    """根据提供的DataFrame绘制结果图表。"""
    global USE_ENGLISH_LABELS
    os.makedirs(output_dir, exist_ok=True)
    
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
    plt.savefig(os.path.join(output_dir, 'summary_query_time_comparison.png'))
    plt.close()
    print(f"已保存图表: {os.path.join(output_dir, 'summary_query_time_comparison.png')}")

    # 图2：内存使用比较
    plt.figure(figsize=(12, 6))
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
    plt.savefig(os.path.join(output_dir, 'summary_memory_usage_comparison.png'))
    plt.close()
    print(f"已保存图表: {os.path.join(output_dir, 'summary_memory_usage_comparison.png')}")

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
    plt.savefig(os.path.join(output_dir, 'summary_accuracy_comparison.png'))
    plt.close()
    print(f"已保存图表: {os.path.join(output_dir, 'summary_accuracy_comparison.png')}")

    # 图4：维度 vs 时间
    if 'dimensions' in summary_df.columns and len(set(summary_df['dimensions'])) > 1:
        plt.figure(figsize=(12, 6))
        dim_groups = summary_df.groupby('dimensions')
        dims, bf_times, kd_times, bbf_times = [], [], [], []
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
        plt.savefig(os.path.join(output_dir, 'summary_dimensions_vs_time.png'))
        plt.close()
        print(f"已保存图表: {os.path.join(output_dir, 'summary_dimensions_vs_time.png')}")
    else:
        print("跳过 '维度 vs 时间' 图表：维度数据不足或不存在。")

    # 图5：数据点数量 vs 时间
    if 'num_points' in summary_df.columns and len(set(summary_df['num_points'])) > 1:
        plt.figure(figsize=(12, 6))
        # Ensure bins are at most the number of unique points, and at least 1.
        n_unique_points = len(set(summary_df['num_points']))
        n_bins = min(10, n_unique_points) if n_unique_points > 0 else 1
        summary_df['points_bin'] = pd.cut(summary_df['num_points'], bins=n_bins)
        
        bin_groups = summary_df.groupby('points_bin', observed=False)
        bins, bf_times_by_bin, kd_times_by_bin, bbf_times_by_bin = [], [], [], []
        
        for bin_name, group in bin_groups:
            if group.empty:
                continue
            bins.append(bin_name.mid)
            bf_times_by_bin.append(group['avg_bf_time'].mean())
            kd_times_by_bin.append(group['avg_kd_time'].mean())
            bbf_times_by_bin.append(group['avg_bbf_time'].mean())
        
        if bins: # Proceed only if there's data to plot
            plt.plot(bins, bf_times_by_bin, 'bo-', label=labels['label1'])
            plt.plot(bins, kd_times_by_bin, 'go-', label=labels['label2'])
            plt.plot(bins, bbf_times_by_bin, 'ro-', label=labels['label3'])
            plt.xlabel(labels['xlabel5'])
            plt.ylabel(labels['ylabel1'])
            plt.title(labels['title5'])
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'summary_num_points_vs_time.png'))
            plt.close()
            print(f"已保存图表: {os.path.join(output_dir, 'summary_num_points_vs_time.png')}")
        else:
            print("跳过 '数据点数量 vs 时间' 图表：分组后无有效数据点。")
    else:
        print("跳过 '数据点数量 vs 时间' 图表：数据点数量数据不足或不存在。")


def main():
    global USE_ENGLISH_LABELS
    parser = argparse.ArgumentParser(description="从summary_results.csv生成图表")
    parser.add_argument(
        "--csv_file", 
        type=str, 
        default="results/summary_results.csv", 
        help="包含汇总结果的CSV文件路径 (默认: results/summary_results.csv)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/plots_from_summary/", 
        help="保存生成图表的目录 (默认: results/plots_from_summary/)"
    )
    parser.add_argument(
        "--english", 
        action="store_true", 
        help="使用英文标签和标题"
    )
    
    args = parser.parse_args()

    if args.english:
        USE_ENGLISH_LABELS = True
    
    # 尝试配置中文，如果失败，USE_ENGLISH_LABELS 会被设为 True
    if not USE_ENGLISH_LABELS: # Only configure if not already forced to English by command line
        configure_matplotlib_chinese()

    print(f"将从以下文件读取数据: {args.csv_file}")
    print(f"图表将保存到: {args.output_dir}")
    print(f"标签语言: {'英文' if USE_ENGLISH_LABELS else '中文'}")

    try:
        summary_df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"错误: CSV文件未找到于 '{args.csv_file}'")
        return
    except Exception as e:
        print(f"读取CSV文件时发生错误: {e}")
        return

    if summary_df.empty:
        print("CSV文件为空，无法生成图表。")
        return

    plot_summary_results(summary_df, args.output_dir)
    print("所有图表已生成。")

if __name__ == "__main__":
    main() 