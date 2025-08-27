import pandas as pd
import numpy as np
import tqdm
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from sentence_transformers import CrossEncoder
from constant.constants import FILE_QUERY_ADDRESS, FILE_CONCATENATED_CHUNKS_ADDRESS
from run_retrievers.utils import BASE_ADDRESS


# -----------------------------
# BM25 检索（返回 id 而不是文本）
# -----------------------------
def bm25_retrieve_local_ids(df_queries, df_chunks, bm25_builder, top_k=100):
    passages = df_chunks["Text"].astype(str).tolist()
    ids = df_chunks["id"].astype(str).tolist()

    tokenized_corpus = [p.split() for p in passages]  # 简单分词
    bm25 = bm25_builder(tokenized_corpus)

    results = {}
    for _, row in tqdm.tqdm(df_queries.iterrows(), total=len(df_queries), desc=f"BM25-{bm25_builder.__name__}"):
        qid = str(row["id"])
        query = str(row["query"])
        tokenized_query = query.split()

        # rank_bm25 的 get_top_n 只能返回文本，这里我们手动算分数
        scores = bm25.get_scores(tokenized_query)  # ndarray(len(corpus),)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results[qid] = [ids[i] for i in top_idx]

    return results


def save_raw_results(results, df_queries, output_path, top_k=100):
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["id"])
        query = row["query"]
        retrieved_ids = results.get(qid, [])
        row_data = [query] + retrieved_ids + [""] * (top_k - len(retrieved_ids))
        rows.append(row_data)

    columns = ["query"] + [f"top_{i}" for i in range(1, top_k + 1)]
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


# -----------------------------
# CE Cross-Encoder 重排序（对 BM25 的候选进行 rerank）
# -----------------------------
def ce_rerank_ids(df_queries, df_chunks, bm25_results, ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=100, batch_size=128):
    # 预备 id -> text 映射
    id2text = dict(zip(df_chunks["id"].astype(str), df_chunks["Text"].astype(str)))

    model = CrossEncoder(ce_model_name)  # 自动用 GPU（若可用）
    results = {}

    for _, row in tqdm.tqdm(df_queries.iterrows(), total=len(df_queries), desc=f"CE rerank ({ce_model_name})"):
        qid = str(row["id"])
        query = str(row["query"])

        candidate_ids = bm25_results.get(qid, [])
        if not candidate_ids:
            results[qid] = []
            continue

        pairs = [(query, id2text[cid]) for cid in candidate_ids if cid in id2text]

        # 批量预测分数
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_scores = model.predict(batch_pairs)  # 越大越相关
            scores.extend(batch_scores.tolist() if hasattr(batch_scores, "tolist") else list(batch_scores))

        # 按分数排序并取前 top_k
        order = np.argsort(scores)[::-1]
        reranked_ids = [candidate_ids[i] for i in order[:top_k]]
        results[qid] = reranked_ids

    return results


def run_and_eval_retrievers():
    TOP_K = 100
    CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # 读 queries
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)
    if "id" not in df_queries.columns:
        df_queries = df_queries.reset_index().rename(columns={"index": "id"})
    df_queries["id"] = df_queries["id"].astype(str)

    # 读 chunks，加 id
    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)
    if "id" not in df_chunks.columns:
        df_chunks = df_chunks.reset_index().rename(columns={"index": "id"})
    df_chunks["id"] = df_chunks["id"].astype(str)

    # -------- BM25 三种原始结果（3 个 CSV）--------
    bm25_variants = [
        ("BM25Plus", BM25Plus),
        ("BM25L", BM25L),
        ("BM25Okapi", BM25Okapi),
    ]

    bm25_results_map = {}

    for name, builder in bm25_variants:
        res = bm25_retrieve_local_ids(df_queries, df_chunks, builder, top_k=TOP_K)
        bm25_results_map[name] = res
        out_raw = f"{BASE_ADDRESS}/raw_{name}_result.csv"
        save_raw_results(res, df_queries, out_raw, top_k=TOP_K)

    # -------- CE 对三种 BM25 的候选做重排（3 个 CSV）--------
    for name, _ in bm25_variants:
        ce_res = ce_rerank_ids(df_queries, df_chunks, bm25_results_map[name], ce_model_name=CE_MODEL, top_k=TOP_K)
        out_ce = f"{BASE_ADDRESS}/ce_reranked_{name}_result.csv"
        save_raw_results(ce_res, df_queries, out_ce, top_k=TOP_K)


if __name__ == "__main__":
    run_and_eval_retrievers()
