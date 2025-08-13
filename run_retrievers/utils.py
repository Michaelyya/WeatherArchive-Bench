BASE_ADDRESS = "run_retrievers/retriever_eval"

import pandas as pd
from typing import Dict, List

# 新：现在的 gold 来自这两个文件
from constant.constants import FILE_QUERY_ADDRESS, FILE_CORRECT_PASSAGES_ADDRESS


def evaluate_retriever_performance(
    retrieval_results: Dict[str, List[str]],
    output_path: str,
    ks: List[int] = (1, 5, 10, 50, 100),
):
    """
    评估检索效果：
    - retrieval_results: {qid(str): [passage_text, ...]}
    - gold: 现在来自 correct_passages.csv 的 'correct_passage'，与 query.csv 一一对应
    - 计算 recall@k 与 mrr@k，k 默认包含 1/5/10/50/100
    """
    # 读取 query 与 gold，并对齐 id
    qdf = pd.read_csv(FILE_QUERY_ADDRESS)
    if "id" not in qdf.columns:
        qdf = qdf.reset_index().rename(columns={"index": "id"})
    qdf["id"] = qdf["id"].astype(str)

    gdf = pd.read_csv(FILE_CORRECT_PASSAGES_ADDRESS)
    if "correct_passage" not in gdf.columns or len(gdf) != len(qdf):
        raise ValueError(
            "correct_passages.csv 必须包含列 'correct_passage'，且行数需与 query.csv 对齐。"
        )

    # 将 gold 附在 qdf，保证一一对应（按行对齐）
    qdf = qdf.copy()
    qdf["correct_passage"] = gdf["correct_passage"].astype(str)

    results = []
    ks = list(sorted(set(int(k) for k in ks)))  # 去重并排序，健壮性

    for _, row in qdf.iterrows():
        qid = row["id"]
        query = str(row["query"])
        golden_answer = str(row["correct_passage"])

        retrieved_passages = retrieval_results.get(qid, [])

        # 初始化各 k 的指标
        hit_k = {f"recall@{k}": 0 for k in ks}
        mrr_k = {f"mrr@{k}": 0.0 for k in ks}
        hit_rank = -1  # 首次命中的名次（1-based），未命中则 -1

        # 遍历检索列表，找到首次命中位置
        max_k = max(ks) if ks else len(retrieved_passages)
        for rank, p in enumerate(retrieved_passages[:max_k], start=1):
            if str(p).strip() == golden_answer.strip():
                hit_rank = rank
                for k in ks:
                    if rank <= k:
                        hit_k[f"recall@{k}"] = 1
                        mrr_k[f"mrr@{k}"] = 1.0 / rank
                break  # 只计首次命中

        results.append(
            {
                "id": qid,
                "query": query,
                "hit_rank": hit_rank,
                **hit_k,
                **mrr_k,
            }
        )

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
