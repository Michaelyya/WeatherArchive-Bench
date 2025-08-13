import pandas as pd
from tqdm import tqdm
from run_retrievers.hybrid_retriver import (
    hybrid_retrieve,
    bm25,
    collection,
)
from run_retrievers.utils import evaluate_retriever_performance
from run_retrievers.utils import BASE_ADDRESS

# 新：使用 query/gold 文件（若你未在 constants 中定义，可直接用字符串路径）
from constant.constants import (
    FILE_QUERY_ADDRESS,  # e.g., "data/query.csv"（列：query，可无 id）
    FILE_CORRECT_PASSAGES_ADDRESS,  # e.g., "data/correct_passages.csv"（列：correct_passage）
)


def run_and_eval_retrievers(
    top_k: int = 100, disaster_threshold: float = 0.0, use_rerank: bool = False
):
    # 读取 query 和 gold（用于评估）
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)
    if "id" not in df_queries.columns:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries.index.astype(str)
    else:
        df_queries["id"] = df_queries["id"].astype(str)

    # （可选）对齐 gold，便于 evaluate 在内部使用
    try:
        df_golds = pd.read_csv(FILE_CORRECT_PASSAGES_ADDRESS)
        if "correct_passage" in df_golds.columns and len(df_golds) == len(df_queries):
            df_queries = df_queries.copy()
            df_queries["correct_passage"] = df_golds["correct_passage"]
        else:
            print(
                "WARNING: correct_passages.csv 行数与 query.csv 不一致，或缺少 'correct_passage' 列，跳过对齐。"
            )
    except Exception as e:
        print(f"WARNING: 读取 correct_passages.csv 失败：{e}")

    bm25_results = {}
    semantic_results = {}

    for _, row in tqdm(
        df_queries.iterrows(), total=len(df_queries), desc="Hybrid retrieving"
    ):
        qid = str(row["id"])
        query = str(row["query"])

        # 纯 BM25（bm25_weight=1.0）
        bm25_top = hybrid_retrieve(
            query,
            bm25,
            collection,
            top_k=top_k,
            bm25_weight=1.0,
            disaster_threshold=disaster_threshold,
            rerank=use_rerank,
        )

        # 纯语义（bm25_weight=0.0）
        semantic_top = hybrid_retrieve(
            query,
            bm25,
            collection,
            top_k=top_k,
            bm25_weight=0.0,
            disaster_threshold=disaster_threshold,
            rerank=use_rerank,
        )

        bm25_results[qid] = [doc for _, _, doc, _, _ in bm25_top]
        semantic_results[qid] = [doc for _, _, doc, _, _ in semantic_top]

    evaluate_retriever_performance(bm25_results, f"{BASE_ADDRESS}/bm25.csv")
    evaluate_retriever_performance(semantic_results, f"{BASE_ADDRESS}/semantic.csv")


if __name__ == "__main__":
    run_and_eval_retrievers(top_k=100, disaster_threshold=0.0, use_rerank=False)
