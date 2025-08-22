import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import tqdm
from run_retrievers.utils import evaluate_retriever_performance
from run_retrievers.utils import BASE_ADDRESS

# 新路径常量（若未在 constants 中定义，可直接用字符串路径替换）
from constant.constants import (
    FILE_QUERY_ADDRESS,  # e.g., "data/query.csv"
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # e.g., "data/concatenated_chunks.csv"
    FILE_CORRECT_PASSAGES_ADDRESS,  # e.g., "data/correct_passages.csv"
)


def dense_retrieve_local(
    df_queries: pd.DataFrame, passages_corpus, model_name: str, top_k: int = 100
):
    """
    在全量 passages_corpus 上为每个 query 检索 top_k 条，返回 {qid: [passage,...]}。
    与原 evaluate_retriever_performance 接口保持一致。
    """
    # 若无 id 列，则用行号作为 id
    if "id" not in df_queries.columns:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries.index.astype(str)
    else:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries["id"].astype(str)

    model = SentenceTransformer(model_name)

    # 预编码整库（按模型一次）
    passages = [str(p) for p in passages_corpus if pd.notna(p)]
    if not passages:
        raise ValueError(
            "passages_corpus 为空，请检查 concatenated_chunks.csv 的 Text 列。"
        )

    passage_embeddings = model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)

    results = {}
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc=f"Retrieving with {model_name}",
    ):
        qid = str(row["id"])
        query = str(row["query"])

        query_embedding = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        k = min(top_k, len(passages))
        scores, indices = index.search(query_embedding, k)
        results[qid] = [passages[i] for i in indices[0]]
    return results


def retrieve_with_ance(df_queries, passages_corpus, top_k: int = 100):
    return dense_retrieve_local(
        df_queries, passages_corpus, "castorini/ance-msmarco-passage", top_k=top_k
    )


def retrieve_with_colbert(df_queries, passages_corpus, top_k: int = 100):
    return dense_retrieve_local(
        df_queries, passages_corpus, "colbert-ir/colbertv1.9", top_k=top_k
    )


def retrieve_with_unicoil(df_queries, passages_corpus, top_k: int = 100):
    return dense_retrieve_local(
        df_queries, passages_corpus, "castorini/unicoil-msmarco-passage", top_k=top_k
    )


def run_and_eval_retrievers():
    # 读取三份文件
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)  # 列: query（可无 id）
    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)  # 列: Text
    df_golds = pd.read_csv(
        FILE_CORRECT_PASSAGES_ADDRESS
    )  # 列: correct_passage（与 query 对齐）

    # 统一 id，并把 gold 对齐到 queries（方便后续评估函数内部使用）
    if "id" not in df_queries.columns:
        df_queries["id"] = df_queries.index.astype(str)
    else:
        df_queries["id"] = df_queries["id"].astype(str)

    if "correct_passage" in df_golds.columns and len(df_golds) == len(df_queries):
        df_queries = df_queries.copy()
        df_queries["correct_passage"] = df_golds["correct_passage"]
    else:
        print(
            "WARNING: correct_passages.csv 行数与 query.csv 不一致，或缺少 'correct_passage' 列，跳过对齐。"
        )

    passages_corpus = df_chunks["Text"].astype(str).tolist()

    retrievers = [
        ("ance", retrieve_with_ance, f"{BASE_ADDRESS}/ance.csv"),
        ("colbert", retrieve_with_colbert, f"{BASE_ADDRESS}/colbert.csv"),
        ("unicoil", retrieve_with_unicoil, f"{BASE_ADDRESS}/unicoil.csv"),
    ]

    for name, fn, path in retrievers:
        print(f"\nRunning retriever: {name}")
        try:
            results = fn(df_queries, passages_corpus)
            evaluate_retriever_performance(results, path)
        except Exception as e:
            print(f"Retriever '{name}' failed with error:\n{e}")
            continue


if __name__ == "__main__":
    run_and_eval_retrievers()
