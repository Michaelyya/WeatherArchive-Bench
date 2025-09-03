import os
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 设置路径
BASE_ADDRESS = "run_retrievers/retriever_eval"
raw_folder_path = os.path.join(os.getcwd(), BASE_ADDRESS)

# 加载 correct_passages.csv 和 concatenated_chunks.csv
correct_passages = pd.read_csv(f"run_retrievers/qa_pair/correct_passages.csv")
gold_texts = correct_passages["correct_passage"].tolist()

chunk_df = pd.read_csv("run_retrievers/qa_pair/concatenated_chunks.csv")
id_to_text = dict(zip(chunk_df["id"].astype(str), chunk_df["Text"]))

# 指标参数
ks = [1, 3, 5, 10, 50, 100]
smooth_fn = SmoothingFunction().method1

# 查找所有 raw_xxx.csv 文件
csv_files = sorted(
    [
        f
        for f in os.listdir(raw_folder_path)
        if f.startswith("raw_") and f.endswith(".csv")
    ]
)

results = []

for filename in csv_files:
    retriever_type = os.path.splitext(filename)[0]
    df = pd.read_csv(os.path.join(raw_folder_path, filename))

    # 提取 top_k 的 id
    topk_cols = [col for col in df.columns if col.startswith("top_")]
    assert len(topk_cols) == 100, f"{filename} should have 100 top_k columns"

    recalls = {f"recall@{k}": [] for k in ks}
    ndcgs = {f"ndcg@{k}": [] for k in ks}
    mrrs = {f"mrr@{k}": [] for k in ks}
    bleus = []

    for i, row in df.iterrows():
        query = row["query"]
        retrieved_ids = [str(row[col]) for col in topk_cols if pd.notna(row[col])]
        retrieved_texts = [id_to_text.get(rid, "") for rid in retrieved_ids]

        # 找到第一个包含任何 gold_text 的 retrieved chunk（作为 hit）
        hit_rank = -1
        for j, chunk_text in enumerate(retrieved_texts):
            if any(gold_text in chunk_text for gold_text in gold_texts):
                hit_rank = j + 1  # rank starts from 1
                break

        for k in ks:
            # Recall@k
            recalls[f"recall@{k}"].append(1.0 if 0 < hit_rank <= k else 0.0)
            # nDCG@k
            ndcgs[f"ndcg@{k}"].append(
                1.0 / np.log2(hit_rank + 1) if 0 < hit_rank <= k else 0.0
            )
            # MRR@k
            mrrs[f"mrr@{k}"].append(1.0 / hit_rank if 0 < hit_rank <= k else 0.0)

        # BLEU@1
        bleu_score = max(
            [
                sentence_bleu(
                    [gold.split()],
                    retrieved_texts[0].split(),
                    smoothing_function=smooth_fn,
                )
                for gold in gold_texts
            ]
        )
        bleus.append(bleu_score)

    row_result = {"retriever_type": retriever_type}
    row_result.update({k: np.mean(v) for k, v in recalls.items()})
    row_result.update({k: np.mean(v) for k, v in ndcgs.items()})
    row_result.update({k: np.mean(v) for k, v in mrrs.items()})
    row_result["BLEU@1"] = np.mean(bleus)

    results.append(row_result)

# 输出结果为 overall.csv
overall_df = pd.DataFrame(results)
overall_df.to_csv(os.path.join(BASE_ADDRESS, "overall.csv"), index=False)
print("✅ overall.csv 已成功生成")
