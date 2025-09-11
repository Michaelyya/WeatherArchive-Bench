import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tqdm

# 常量：与你工程一致
from constant.constants import (
    FILE_QUERY_ADDRESS,  # e.g., "data/queries.csv"，需要 'query' 列
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # e.g., "data/concatenated_chunks.csv"，需要 'Text' 列
)
from run_retrievers.utils import BASE_ADDRESS  # e.g., "run_retrievers/retriever_eval"


def ensure_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)
    return df


def dense_retrieve_local_ids(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    model_name: str,
    top_k: int = 100,
):
    """
    使用 SentenceTransformer + FAISS 做密集检索，返回每个 query 的 top_k 文档 **id 列表**。
    - df_queries: 需要列 ['id', 'query']
    - df_chunks: 需要列 ['id', 'Text']
    返回：{qid(str): [doc_id(str), ...]}
    """
    passages = df_chunks["Text"].astype(str).tolist()
    ids = df_chunks["id"].astype(str).tolist()
    if len(passages) == 0:
        raise ValueError("语料为空：未在 concatenated_chunks.csv 中找到可用的 'Text'。")

    print(f"[{model_name}] Loading model...")
    model = SentenceTransformer(model_name)

    print(f"[{model_name}] Encoding passages...")
    passage_embeddings = model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dim = passage_embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # 向量已归一化 -> 内积等价余弦
    index.add(passage_embeddings)

    results = {}
    print(f"[{model_name}] Retrieving ...")
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc=f"Retrieving with {model_name}",
    ):
        qid = str(row["id"])
        query = str(row["query"])
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        k = min(top_k, len(passages))
        scores, indices = index.search(q_emb, k)
        top_idx = indices[0]
        results[qid] = [ids[i] for i in top_idx]

    return results


def save_raw_results_as_csv(
    results: dict,
    df_queries: pd.DataFrame,
    output_path: str,
    top_k: int = 100,
):
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["id"])
        query = str(row["query"])
        retrieved_ids = results.get(qid, [])
        retrieved_ids = (retrieved_ids + [""] * top_k)[:top_k]
        rows.append([query] + retrieved_ids)

    columns = ["query"] + [f"top_{i}" for i in range(1, top_k + 1)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")


def run_retrievers_and_dump_raw():
    # 读取 queries 与 chunks，并确保有 id 列
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)  # 需要 'query'
    if "query" not in df_queries.columns:
        raise ValueError("queries.csv 需要包含列 'query'。")
    df_queries = ensure_id_column(df_queries, "id")

    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)  # 需要 'Text'
    if "Text" not in df_chunks.columns:
        raise ValueError("concatenated_chunks.csv 需要包含列 'Text'。")
    df_chunks = ensure_id_column(df_chunks, "id")

    # === 新的可选模型清单（排除 Qwen；优先易于本地直载） ===
    retrievers = [
        # 多语检索向，Arctic Embed 2.0
        ("arctic", "Snowflake/snowflake-arctic-embed-l-v2.0"),
        # Granite Embedding R2（企业检索向）
        ("granite", "ibm-granite/granite-embedding-english-r2"),
        # （可选）NV-Embed v2：如果显存不足会自动被 try/except 跳过
        ("nvidia_nv", "nvidia/NV-Embed-v2"),
    ]

    for short_name, model_name in retrievers:
        print(f"\n=== Running retriever: {short_name} ({model_name}) ===")
        try:
            results = dense_retrieve_local_ids(
                df_queries=df_queries,
                df_chunks=df_chunks,
                model_name=model_name,
                top_k=100,
            )
            out_path = os.path.join(BASE_ADDRESS, f"raw_model_result_{short_name}.csv")
            save_raw_results_as_csv(results, df_queries, out_path, top_k=100)
        except Exception as e:
            print(f"[Skipped] retriever '{short_name}' failed: {e}")


if __name__ == "__main__":
    run_retrievers_and_dump_raw()
