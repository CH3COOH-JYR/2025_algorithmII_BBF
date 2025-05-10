import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def generate_plots(summary_df_orig, results_dir):
    """Generates and saves various comparison plots, aggregated by dimension."""
    if summary_df_orig.empty:
        print("Summary DataFrame is empty. No plots will be generated.")
        return

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory for plots: {results_dir}")

    agg_funcs = {
        'bruteforce_avg_time_s': 'mean',
        'kdtree_avg_time_s': 'mean',
        'bbf_avg_time_s': 'mean',
        'bruteforce_accuracy': 'mean',
        'kdtree_accuracy': 'mean',
        'bbf_accuracy': 'mean',
        'kdtree_build_time_s': 'mean',
        'kdtree_nodes': 'mean',
        'memory_data_mb': 'mean',
        'num_data_points': 'mean',
        'bbf_t': 'first',
        'dataset': 'count'
    }
    
    existing_cols_agg_funcs = {k: v for k, v in agg_funcs.items() if k in summary_df_orig.columns}
    
    if not existing_cols_agg_funcs or 'dimensions' not in summary_df_orig.columns:
        print("Relevant columns ('dimensions' or metrics) not found for aggregation. Skipping plot generation.")
        return

    summary_df = summary_df_orig.groupby('dimensions').agg(existing_cols_agg_funcs).reset_index()
    summary_df = summary_df.rename(columns={'dataset': 'num_datasets_in_dim'})
    summary_df = summary_df.sort_values(by='dimensions').reset_index(drop=True)

    if summary_df.empty:
        print("DataFrame is empty after grouping by dimension. No plots will be generated.")
        return

    dimensions_labels = summary_df['dimensions'].astype(str)
    x_indices = np.arange(len(dimensions_labels))
    bbf_t_val = summary_df["bbf_t"].iloc[0] if "bbf_t" in summary_df.columns and not summary_df["bbf_t"].empty else "N/A"

    plt.figure(figsize=(12, 7))
    bar_width = 0.25
    if 'bruteforce_avg_time_s' in summary_df.columns: plt.bar(x_indices - bar_width, summary_df['bruteforce_avg_time_s'], width=bar_width, label='Brute-force', color='#1f77b4')
    if 'kdtree_avg_time_s' in summary_df.columns: plt.bar(x_indices, summary_df['kdtree_avg_time_s'], width=bar_width, label='k-d Tree', color='#ff7f0e')
    if 'bbf_avg_time_s' in summary_df.columns: plt.bar(x_indices + bar_width, summary_df['bbf_avg_time_s'], width=bar_width, label=f'BBF (t={bbf_t_val})', color='#2ca02c')
    plt.xlabel('Data Dimension'); plt.ylabel('Average Query Time (seconds, log scale)'); plt.title('Average Query Time vs Dimension')
    plt.xticks(x_indices, dimensions_labels); plt.yscale('log'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dim_query_time_comparison.png"))
    print(f"Saved dimension-based query time plot to {results_dir}/dim_query_time_comparison.png"); plt.close()

    plt.figure(figsize=(12, 7))
    if 'kdtree_accuracy' in summary_df.columns: plt.bar(x_indices - bar_width/2, summary_df['kdtree_accuracy'], width=bar_width, label='k-d Tree Accuracy', color='#ff7f0e')
    if 'bbf_accuracy' in summary_df.columns: plt.bar(x_indices + bar_width/2, summary_df['bbf_accuracy'], width=bar_width, label=f'BBF Accuracy (t={bbf_t_val})', color='#2ca02c')
    plt.xlabel('Data Dimension'); plt.ylabel('Average Accuracy (Fraction successful)'); plt.title('Average Accuracy vs Dimension')
    plt.xticks(x_indices, dimensions_labels); plt.ylim(0, 1.05); plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dim_accuracy_comparison.png"))
    print(f"Saved dimension-based accuracy plot to {results_dir}/dim_accuracy_comparison.png"); plt.close()

    if 'kdtree_build_time_s' in summary_df.columns and 'kdtree_nodes' in summary_df.columns:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        color = 'tab:red'; ax1.set_xlabel('Data Dimension'); ax1.set_ylabel('Avg k-d Tree Build Time (s)', color=color)
        ax1.bar(x_indices, summary_df['kdtree_build_time_s'], color=color, alpha=0.6, width=0.4, label='Build Time'); ax1.tick_params(axis='y', labelcolor=color); plt.xticks(x_indices, dimensions_labels)
        ax2 = ax1.twinx(); color = 'tab:blue'; ax2.set_ylabel('Avg k-d Tree Nodes', color=color)
        ax2.plot(x_indices, summary_df['kdtree_nodes'], color=color, marker='o', linestyle='--', label='Nodes'); ax2.tick_params(axis='y', labelcolor=color)
        if not summary_df['kdtree_nodes'].empty and summary_df['kdtree_nodes'].min() > 0: ax2.set_yscale('log')
        fig.tight_layout(); plt.title('k-d Tree Build Time and Node Count vs Dimension')
        lines, labels_ax1 = ax1.get_legend_handles_labels(); lines2, labels_ax2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels_ax1 + labels_ax2, loc='upper center'); plt.savefig(os.path.join(results_dir, "dim_build_time_nodes_comparison.png"))
        print(f"Saved dimension-based k-d tree build/nodes plot to {results_dir}/dim_build_time_nodes_comparison.png"); plt.close()
    
    if 'bruteforce_avg_time_s' in summary_df.columns:
        if 'kdtree_avg_time_s' in summary_df.columns: summary_df['speedup_kdtree'] = summary_df['bruteforce_avg_time_s'] / summary_df['kdtree_avg_time_s'].replace(0, np.nan)
        if 'bbf_avg_time_s' in summary_df.columns: summary_df['speedup_bbf'] = summary_df['bruteforce_avg_time_s'] / summary_df['bbf_avg_time_s'].replace(0, np.nan)
        plt.figure(figsize=(12, 7))
        if 'speedup_kdtree' in summary_df.columns: plt.bar(x_indices - bar_width/2, summary_df['speedup_kdtree'].fillna(0), width=bar_width, label='k-d Tree Speedup', color='#ff7f0e')
        if 'speedup_bbf' in summary_df.columns: plt.bar(x_indices + bar_width/2, summary_df['speedup_bbf'].fillna(0), width=bar_width, label=f'BBF Speedup (t={bbf_t_val})', color='#2ca02c')
        plt.xlabel('Data Dimension'); plt.ylabel('Average Speedup Factor (vs Brute-Force)'); plt.title('Query Speedup vs Dimension')
        plt.xticks(x_indices, dimensions_labels); plt.axhline(y=1, color='gray', linestyle='--'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "dim_speedup_comparison.png"))
        print(f"Saved dimension-based speedup plot to {results_dir}/dim_speedup_comparison.png"); plt.close()

def generate_summary_table_markdown(summary_df_orig, output_path):
    if summary_df_orig.empty: return ""
    agg_funcs = {
        'num_data_points': 'mean', 'kdtree_build_time_s': 'mean', 'kdtree_nodes': 'mean',
        'memory_data_mb': 'mean', 'bruteforce_avg_time_s': 'mean', 'kdtree_avg_time_s': 'mean',
        'bbf_avg_time_s': 'mean', 'kdtree_accuracy': 'mean', 'bbf_accuracy': 'mean',
        'bbf_t': 'first', 'dataset': 'count'}
    existing_cols_agg_funcs = {k: v for k, v in agg_funcs.items() if k in summary_df_orig.columns}
    if not existing_cols_agg_funcs or 'dimensions' not in summary_df_orig.columns: return ""
    table_df_agg = summary_df_orig.groupby('dimensions').agg(existing_cols_agg_funcs).reset_index()
    table_df_agg = table_df_agg.rename(columns={'dataset': 'Num Datasets'}).sort_values(by='dimensions')
    column_map = {
        'dimensions': 'Dim', 'num_data_points': 'Avg. N', 'Num Datasets': 'Files',
        'kdtree_build_time_s': 'Build Time (s)', 'kdtree_nodes': 'k-d Nodes',
        'bruteforce_avg_time_s': 'T_bf (s)', 'kdtree_avg_time_s': 'T_kdt (s)',
        'bbf_avg_time_s': 'T_bbf (s)', 'kdtree_accuracy': 'Acc_kdt',
        'bbf_accuracy': 'Acc_bbf', 'bbf_t': 'BBF t'}
    final_columns = [k for k, v in column_map.items() if k in table_df_agg.columns]
    table_df_display = table_df_agg[final_columns].copy().rename(columns=column_map)
    for col_name in table_df_display.columns:
        original_col_key = next((k for k, v in column_map.items() if v == col_name), None)
        if original_col_key and original_col_key in table_df_agg and pd.api.types.is_numeric_dtype(table_df_agg[original_col_key]):
            fmt = '.0f' if any(s in col_name for s in ['Nodes', 'Avg. N', 'Files']) else (
                  '.4f' if any(s in col_name for s in ['Time', 'Acc', '(s)']) else '.2f')
            table_df_display[col_name] = table_df_agg[original_col_key].apply(lambda x: f"{x:{fmt}}" if pd.notnull(x) else 'N/A')
        elif col_name == 'BBF t' and 'bbf_t' in table_df_agg: # Ensure BBF t is integer like
             table_df_display[col_name] = table_df_agg['bbf_t'].apply(lambda x: f'{int(x)}' if pd.notnull(x) and not np.isnan(x) else ('N/A' if not pd.notnull(x) or np.isnan(x) else str(x)))
    markdown_table = "## Experiment Results Summary (Aggregated by Dimension)\n\n" + table_df_display.to_markdown(index=False) + "\n\n"
    try:
        with open(output_path, 'w') as f: f.write(markdown_table)
        print(f"Dimension-aggregated summary table saved to {output_path}")
    except Exception as e: print(f"Error saving markdown table: {e}")
    return markdown_table

def main():
    parser = argparse.ArgumentParser(description="Analyze batch evaluation results, aggregate by dimension, and generate plots/tables.")
    parser.add_argument("summary_csv_path", type=str, help="Path to batch summary CSV (e.g., results/batch_summary.csv).")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save plots/tables (default: results/).")
    args = parser.parse_args()
    if not os.path.exists(args.summary_csv_path): print(f"Error: Summary CSV not found: {args.summary_csv_path}"); return
    if not os.path.exists(args.results_dir): os.makedirs(args.results_dir); print(f"Created results directory: {args.results_dir}")
    try: summary_df_loaded = pd.read_csv(args.summary_csv_path)
    except Exception as e: print(f"Error reading summary CSV: {e}"); return
    if 'error' in summary_df_loaded.columns: summary_df_loaded = summary_df_loaded[summary_df_loaded['error'].isna()].copy()
    if 'info' in summary_df_loaded.columns: 
        summary_df_loaded = summary_df_loaded[~summary_df_loaded['info'].str.contains("No query points tested", na=False)].copy()
    if summary_df_loaded.empty: print("Loaded summary data is empty after filtering. Cannot analyze."); return
    print("--- Analyzing Results (Aggregated by Dimension) ---")
    print("Summary Data Loaded (pre-filtered):")
    print(summary_df_loaded.to_string())
    generate_plots(summary_df_loaded, args.results_dir)
    md_table_path = os.path.join(args.results_dir, "dim_experiment_summary_table.md")
    generate_summary_table_markdown(summary_df_loaded, md_table_path)
    print(f"\nDimension-based analysis complete. Output in: {args.results_dir}")

if __name__ == "__main__":
    main() 