import pandas as pd
import tqdm
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
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
    for _, row in tqdm.tqdm(df_queries.iterrows(), total=len(df_queries)):
        qid = str(row["id"])
        query = str(row["query"])
        tokenized_query = query.split()

        # rank_bm25 的 get_top_n 只能返回文本，这里我们手动算分数
        scores = bm25.get_scores(tokenized_query)
        top_idx = scores.argsort()[::-1][:top_k]
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
    print(f"Raw results saved to {output_path}")


def run_and_eval_retrievers():
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

    # 只跑一个 BM25（你可以改成多种）
    results = bm25_retrieve_local_ids(df_queries, df_chunks, BM25Plus, top_k=100)
    output_path = f"{BASE_ADDRESS}/raw_BM25Plus_result.csv"
    save_raw_results(results, df_queries, output_path, top_k=100)

    results = bm25_retrieve_local_ids(df_queries, df_chunks, BM25L, top_k=100)
    output_path = f"{BASE_ADDRESS}/raw_BM25L_result.csv"
    save_raw_results(results, df_queries, output_path, top_k=100)

    results = bm25_retrieve_local_ids(df_queries, df_chunks, BM25Okapi, top_k=100)
    output_path = f"{BASE_ADDRESS}/raw_BM25Okapi_result.csv"
    save_raw_results(results, df_queries, output_path, top_k=100)


if __name__ == "__main__":
    run_and_eval_retrievers()
