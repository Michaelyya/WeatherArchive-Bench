import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import tqdm
from run_retrievers.utils import evaluate_retriever_performance
from run_retrievers.utils import BASE_ADDRESS

# 新的常量，指向三份文件；若你尚未在 constants 中添加，可直接用字符串路径替代
from constant.constants import (
    FILE_QUERY_ADDRESS,  # e.g., "data/query.csv"
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # e.g., "data/concatenated_chunks.csv"
    FILE_CORRECT_PASSAGES_ADDRESS,  # e.g., "data/correct_passages.csv"
)


def dense_retrieve_local(
    df_queries: pd.DataFrame, passages_corpus, model_name: str, top_k: int = 100
):
    """
    基于给定模型，在全量 passages_corpus 上为每个 query 检索 top_k 条。
    返回: {qid(str): [passage_str, ...]}，与原 evaluate_retriever_performance 兼容。
    """
    # 构造 qid：若 df_queries 中没有 'id' 列，则用行号作为 id
    if "id" not in df_queries.columns:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries.index.astype(str)
    else:
        df_queries["id"] = df_queries["id"].astype(str)

    model = SentenceTransformer(model_name)

    # 1) 预编码整个语料库（只做一次/模型）
    passages = [str(p) for p in passages_corpus if pd.notna(p)]
    if len(passages) == 0:
        raise ValueError(
            "passages_corpus 为空，无法检索。请检查 concatenated_chunks.csv 的 Text 列。"
        )

    passage_embeddings = model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # 2) 建索引（内积 + 归一化 = 余弦相似度）
    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)

    # 3) 遍历所有 query，逐一检索
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


def retrieve_with_deepct(df_queries, passages_corpus):
    return dense_retrieve_local(
        df_queries,
        passages_corpus,
        "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    )


def retrieve_with_sbert(df_queries, passages_corpus):
    return dense_retrieve_local(
        df_queries,
        passages_corpus,
        "sentence-transformers/msmarco-distilbert-base-tas-b",
    )


def retrieve_with_colbert_v2(df_queries, passages_corpus):
    return dense_retrieve_local(df_queries, passages_corpus, "colbert-ir/colbertv2.0")


def retrieve_with_splade(df_queries, passages_corpus):
    return dense_retrieve_local(
        df_queries, passages_corpus, "naver/splade-cocondenser-ensembledistil"
    )


def run_and_eval_retrievers():
    # 读取三份文件
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)  # 只有 'query' 一列；可无 'id'
    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)  # 只有 'Text' 一列
    df_golds = pd.read_csv(
        FILE_CORRECT_PASSAGES_ADDRESS
    )  # 只有 'correct_passage' 一列（与 query 对齐）

    # 为了与后续评估/保存兼容，给 queries 加上 id，并把 gold 附在 df_queries 上（按行对齐）
    if "id" not in df_queries.columns:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries.index.astype(str)

    if "correct_passage" in df_golds.columns and len(df_golds) == len(df_queries):
        df_queries = df_queries.copy()
        df_queries["correct_passage"] = df_golds["correct_passage"]
    else:
        print(
            "WARNING: correct_passages.csv 与 query.csv 行数不一致，或缺少 'correct_passage' 列，跳过对齐。"
        )

    # 全量语料库
    passages_corpus = df_chunks["Text"].astype(str).tolist()

    retrievers = [
        ("deepct", retrieve_with_deepct, f"{BASE_ADDRESS}/deepct.csv"),
        ("sbert", retrieve_with_sbert, f"{BASE_ADDRESS}/sbert.csv"),
        ("colbert_v2", retrieve_with_colbert_v2, f"{BASE_ADDRESS}/colbert_v2.csv"),
        ("splade", retrieve_with_splade, f"{BASE_ADDRESS}/splade.csv"),
    ]

    for name, fn, path in retrievers:
        print(f"\nRunning retriever: {name}")
        try:
            results = fn(df_queries, passages_corpus)
            # 仍沿用原有评估与导出逻辑；如 evaluate_retriever_performance 需要 gold，可在其内部读取或在此函数中已对齐在 df_queries 中
            evaluate_retriever_performance(results, path)
        except Exception as e:
            print(f"Retriever '{name}' failed with error:\n{e}")
            continue


if __name__ == "__main__":
    run_and_eval_retrievers()
