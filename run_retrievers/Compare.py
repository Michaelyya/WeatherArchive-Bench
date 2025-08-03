import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

def load_strategy_results(folder_path):
    strategy_files = glob(os.path.join(folder_path, "*.csv"))
    strategies = {}
    
    for file in strategy_files:
        strategy_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)
        
        # Calc mean value 
        metrics = {
            'recall@1': df['recall@1'].mean(),
            'recall@5': df['recall@5'].mean(),
            'recall@10': df['recall@10'].mean(),
            'mrr@1': df['mrr@1'].mean(),
            'mrr@5': df['mrr@5'].mean(),
            'mrr@10': df['mrr@10'].mean(),
            'success_rate': (df['hit_rank'] != -1).mean(),
            'avg_hit_rank': df[df['hit_rank'] != -1]['hit_rank'].mean()
        }
        strategies[strategy_name] = metrics
    
    return pd.DataFrame.from_dict(strategies, orient='index')

def plot_comparison(comparison_df, output_path):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 1. Recall graph
    plt.figure(figsize=(10, 6))
    comparison_df[['recall@1', 'recall@5', 'recall@10']].plot(kind='bar')
    plt.title('Recall@K Comparison Across Strategies')
    plt.ylabel('Recall Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'recall_comparison.png'))
    plt.close()
    
    # 2. MRR graph
    plt.figure(figsize=(10, 6))
    comparison_df[['mrr@1', 'mrr@5', 'mrr@10']].plot(kind='bar')
    plt.title('MRR@K Comparison Across Strategies')
    plt.ylabel('MRR Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'mrr_comparison.png'))
    plt.close()
    
    # 3. Ranking of success_rate & avg hit rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    comparison_df['success_rate'].plot(kind='bar', ax=ax1)
    ax1.set_title('Success Rate Comparison')
    ax1.set_ylabel('Success Rate')
    ax1.tick_params(axis='x', rotation=45)
    
    comparison_df['avg_hit_rank'].plot(kind='bar', ax=ax2)
    ax2.set_title('Average Hit Rank (Successful Queries Only)')
    ax2.set_ylabel('Average Rank')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'success_metrics.png'))
    plt.close()
    
    #Set up metrics
    metrics = ['recall@1', 'recall@5', 'recall@10', 'mrr@1', 'mrr@5', 'mrr@10', 'success_rate']
    num_vars = len(metrics)
    
    angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for idx, strategy in enumerate(comparison_df.index):
        values = comparison_df.loc[strategy, metrics].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=strategy)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Strategy Performance Radar Chart', size=14, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'radar_chart.png'))
    plt.close()

def plot_summary_table(df, output_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, ax = plt.subplots(figsize=(12, len(df)*0.5 + 2))
    ax.axis('off')

    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: round(x, 4) if pd.notnull(x) else x)

    table = ax.table(cellText=display_df.values,
                     rowLabels=display_df.index,
                     colLabels=display_df.columns,
                     loc='center',
                     cellLoc='center')


    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor('white')            
        cell.set_edgecolor('black')            
        
       
        if row == 0:
            cell.set_text_props(weight='bold') 
            cell.set_linewidth(1.5)            
       
        else:
            cell.set_linewidth(0.8)

        if col == -1 and row != 0:
            cell.set_text_props(weight='bold') 

    plt.title("Retrieval Strategies Comparison", fontsize=14, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'general_cmparison.png'), dpi=400)
    plt.close()



def compare_strategies(input_folder, output_folder):
    
    comparison_df = load_strategy_results(input_folder)
    
    #Store result
    plot_comparison(comparison_df, output_folder)
    
    plot_summary_table(comparison_df, output_folder)
    
    print(f"Comparison complete! Results saved to {output_folder} folder.")
    return comparison_df


if __name__ == "__main__":
    results = compare_strategies(input_folder="C:/Users/14821/PyCharmMiscProject/run_retrievers/retriever_eval", output_folder="C:/Users/14821/PyCharmMiscProject/run_retrievers/retriever_eval/Result_IMG")
    
    print("\nStrategy Comparison Results:")
    print(results)