import os
import numpy as np
import pandas as pd
import faiss
import tqdm
from openai import OpenAI
from run_retrievers.utils import BASE_ADDRESS
from constant.constants import (
    FILE_QUERY_ADDRESS,  # 需要包含 'query' 列
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # 需要包含 'Text' 列；若无 'id' 会自动生成
)

import dotenv

dotenv.load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ---------- Utils ----------
def ensure_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """确保存在字符串类型的唯一 id 列；如果没有则用行号生成。"""
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)
    return df


def encode_with_openai(texts, model="text-embedding-3-large", batch_size=64):
    """
    使用 OpenAI Embeddings API 编码文本（批量）。
    返回：List[List[float]]
    """
    embeddings = []
    for i in tqdm.trange(0, len(texts), batch_size, desc=f"Encoding with {model}"):
        batch_texts = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch_texts)
        batch_embeddings = [d.embedding for d in resp.data]
        embeddings.extend(batch_embeddings)
    return embeddings


def dense_retrieve_openai_ids(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    model_name: str = "text-embedding-3-large",
    top_k: int = 100,
):
    """
    基于 OpenAI Embeddings + FAISS 的本地稠密检索。
    返回：{qid(str): [doc_id(str), ...]} 仅包含前 top_k 的文档 id。
    """
    # 语料与 id（顺序将决定向量与索引的对齐）
    passages = df_chunks["Text"].astype(str).tolist()
    doc_ids = df_chunks["id"].astype(str).tolist()
    if not passages:
        raise ValueError("concatenated_chunks.csv 的 'Text' 为空。")

    # 1) 编码语料并建索引（内积 + 归一化 = 余弦相似度）
    passage_embs = encode_with_openai(passages, model=model_name)
    passage_embs = np.asarray(passage_embs, dtype="float32")
    dim = passage_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embs)

    # 2) 逐条（或可批量）编码 query 并检索
    results = {}
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc=f"Retrieving with {model_name}",
    ):
        qid = str(row["id"])
        query = str(row["query"])
        q_emb = encode_with_openai([query], model=model_name)[0]
        q_emb = np.asarray([q_emb], dtype="float32")
        k = min(top_k, len(passages))
        scores, indices = index.search(q_emb, k)
        top_idx = indices[0]
        results[qid] = [doc_ids[i] for i in top_idx]
    return results


def save_raw_results_as_csv(
    results: dict, df_queries: pd.DataFrame, output_path: str, top_k: int = 100
):
    """将 {qid: [doc_id,...]} 存为 CSV：query, top_1..top_k（按 df_queries 原顺序对齐）。"""
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["id"])
        query = str(row["query"])
        ids = results.get(qid, [])
        ids = (ids + [""] * top_k)[:top_k]  # 不足补空，超出截断
        rows.append([query] + ids)

    cols = ["query"] + [f"top_{i}" for i in range(1, top_k + 1)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")


# ---------- Main ----------
def run_retrievers_and_dump_raw():
    # 读取数据
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)
    if "query" not in df_queries.columns:
        raise ValueError("queries.csv 需要包含列 'query'。")
    df_queries = ensure_id_column(df_queries, "id")

    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)
    if "Text" not in df_chunks.columns:
        raise ValueError("concatenated_chunks.csv 需要包含列 'Text'。")
    df_chunks = ensure_id_column(df_chunks, "id")

    # 要跑的 OpenAI embedding 模型（可按需增删）
    retrievers = [
        ("openai-3-large", "text-embedding-3-large"),
        ("openai-3-small", "text-embedding-3-small"),
        ("openai-ada-002", "text-embedding-ada-002"),
    ]

    for short_name, model_name in retrievers:
        print(f"\n=== Running retriever: {short_name} ({model_name}) ===")
        try:
            results = dense_retrieve_openai_ids(
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
