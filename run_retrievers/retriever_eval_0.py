import re
import pandas as pd
import tqdm
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from run_retrievers.utils import evaluate_retriever_performance
from run_retrievers.utils import BASE_ADDRESS

# 仍沿用你的常量
from constant.constants import (
    FILE_QUERY_ADDRESS,  # e.g., "data/query.csv"
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # e.g., "data/concatenated_chunks.csv"
    FILE_CORRECT_PASSAGES_ADDRESS,  # e.g., "data/correct_passages.csv"
)

# -----------------------------
# 简单英文分词器（小写 + 去掉非字母数字）
# -----------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# -----------------------------
# 通用 BM25 检索流程（适配 evaluate 接口）
# -----------------------------
def bm25_retrieve_local(
    df_queries: pd.DataFrame,
    passages_corpus,
    bm25_builder,  # 传入 BM25Okapi / BM25L / BM25Plus 类
    top_k: int = 100,
):
    """
    基于 BM25（Okapi/L/Plus）在全量 passages_corpus 上为每个 query 检索 top_k 条。
    返回: {qid(str): [passage_str, ...]}，与 evaluate_retriever_performance 兼容。
    """
    # 1) 规范化 qid
    if "id" not in df_queries.columns:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries.index.astype(str)
    else:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries["id"].astype(str)

    # 2) 清洗/准备语料
    passages = [str(p) for p in passages_corpus if pd.notna(p)]
    if len(passages) == 0:
        raise ValueError(
            "passages_corpus 为空，无法检索。请检查 concatenated_chunks.csv 的 Text 列。"
        )

    tokenized_corpus = [tokenize(p) for p in passages]

    # 3) 构建 BM25 索引
    bm25 = bm25_builder(tokenized_corpus)

    # 4) 遍历 queries 检索
    results = {}
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc=f"Retrieving with {bm25_builder.__name__}",
    ):
        qid = str(row["id"])
        query = str(row["query"])
        tokenized_query = tokenize(query)

        # rank_bm25 提供 get_top_n：直接返回对应的原始文档
        k = min(top_k, len(passages))
        top_docs = bm25.get_top_n(tokenized_query, passages, n=k)
        results[qid] = top_docs

    return results


# -----------------------------
# 三种 BM25 变体（名称与导出路径）
# -----------------------------
def retrieve_with_bm25_okapi(df_queries, passages_corpus):
    return bm25_retrieve_local(df_queries, passages_corpus, BM25Okapi)


def retrieve_with_bm25l(df_queries, passages_corpus):
    return bm25_retrieve_local(df_queries, passages_corpus, BM25L)


def retrieve_with_bm25plus(df_queries, passages_corpus):
    return bm25_retrieve_local(df_queries, passages_corpus, BM25Plus)


# -----------------------------
# 主流程：读取文件、跑三种 BM25、评估并导出
# -----------------------------
def run_and_eval_retrievers():
    # 读取三份文件
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)  # 需要 'query' 列；可无 'id'
    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)  # 需要 'Text' 列
    df_golds = pd.read_csv(
        FILE_CORRECT_PASSAGES_ADDRESS
    )  # 需要 'correct_passage' 列（与 query 对齐）

    # 给 queries 加 id，并尝试把 gold 附上（按行对齐）
    if "id" not in df_queries.columns:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries.index.astype(str)
    else:
        df_queries = df_queries.copy()
        df_queries["id"] = df_queries["id"].astype(str)

    if "correct_passage" in df_golds.columns and len(df_golds) == len(df_queries):
        df_queries["correct_passage"] = df_golds["correct_passage"]
    else:
        print(
            "WARNING: correct_passages.csv 与 query.csv 行数不一致，或缺少 'correct_passage' 列，跳过对齐。"
        )

    # 全量语料库（英文）
    if "Text" not in df_chunks.columns:
        raise ValueError("在 FILE_CONCATENATED_CHUNKS_ADDRESS 中未找到 'Text' 列。")
    passages_corpus = df_chunks["Text"].astype(str).tolist()

    retrievers = [
        # ("bm25_okapi", retrieve_with_bm25_okapi, f"{BASE_ADDRESS}/bm25_okapi.csv"),
        # ("bm25l", retrieve_with_bm25l, f"{BASE_ADDRESS}/bm25l.csv"),
        ("bm25plus", retrieve_with_bm25plus, f"{BASE_ADDRESS}/bm25plus.csv"),
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
