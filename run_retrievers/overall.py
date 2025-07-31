import os
import pandas as pd

# 设置目录路径
BASE_ADDRESS = "run_retrievers/retriever_eval"
folder_path = os.path.join(os.getcwd(), BASE_ADDRESS)

# 查找所有 CSV 文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# 所需的列（不包含 id 和 query）
metrics_columns = [
    "hit_rank",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr@1",
    "mrr@5",
    "mrr@10",
]

overall_data = []

for idx, filename in enumerate(sorted(csv_files), 1):
    file_path = os.path.join(folder_path, filename)
    retriever_type = os.path.splitext(filename)[0]

    df = pd.read_csv(file_path)

    # 检查是否包含全部需要的列
    required_columns = ["id", "query"] + metrics_columns
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"{filename} is missing required columns.")

    # 计算平均值
    averages = df[metrics_columns].mean().to_dict()

    # 构造结果行
    result_row = {"index": idx, "retriever_type": retriever_type, **averages}
    overall_data.append(result_row)

# 转为 DataFrame 并保存为 CSV
overall_df = pd.DataFrame(overall_data)
overall_df.to_csv(f"{BASE_ADDRESS}/overall.csv", index=False)

print("overall.csv 已成功生成。")
