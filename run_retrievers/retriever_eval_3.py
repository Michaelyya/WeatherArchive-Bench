import os
import math
import pandas as pd
import numpy as np
import faiss
import tqdm
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from run_retrievers.utils import evaluate_retriever_performance

# ========== 优先尝试使用 hybrid_retriver 中现成的 bm25 / 语料 ==========
_USE_HYBRID_BM25 = False
_HYBRID_BM25 = None
_HYBRID_CORPUS = None  # 若可从 hybrid 模块拿到 documents，则作为 passages_list 使用

try:
    # 你之前那份 hybrid_retriver.py 里暴露了 bm25 和 documents（模块级变量）
    from run_retrievers.hybrid_retriver import (
        bm25 as _HYBRID_BM25,
        documents as _HYBRID_CORPUS,
    )

    if (
        _HYBRID_BM25 is not None
        and isinstance(_HYBRID_CORPUS, list)
        and len(_HYBRID_CORPUS) > 0
    ):
        _USE_HYBRID_BM25 = True
        _HYBRID_BM25 = _HYBRID_BM25
        _HYBRID_CORPUS = _HYBRID_CORPUS
except Exception:
    # 没有 hybrid 亦或其不符合预期，则走本地 BM25
    _USE_HYBRID_BM25 = False
    _HYBRID_BM25 = None
    _HYBRID_CORPUS = None


# ====================== 本地 BM25（作为回退）======================
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        corpus: list of list of tokens (e.g. [["climate","change"],["global","warming"]])
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        self.df = defaultdict(int)
        self.doc_tf = []

        for doc in corpus:
            tf = Counter(doc)
            self.doc_tf.append(tf)
            for term in tf:
                self.df[term] += 1

        # Precompute IDF
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query, index):
        if index >= self.N:
            return 0.0
        score = 0.0
        doc_tf = self.doc_tf[index]
        dl = self.doc_len[index]
        for term in query:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            idf = self.idf.get(term, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (
                1.0 - self.b + self.b * dl / max(self.avgdl, 1e-8)
            )
            score += idf * (numerator / max(denominator, 1e-8))
        return score

    def get_scores(self, query):
        return [self.score(query, idx) for idx in range(self.N)]


# ====================== 工具函数 =======================
def _ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    df["id"] = df["id"].astype(str)
    return df


def _tokenize(text: str):
    # 轻量 tokenizer：对英文/数字而言够用；若需要可替换为 nltk.word_tokenize
    return [w for w in str(text).lower().split() if w.isalnum()]


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x_min, x_max = float(x.min()), float(x.max())
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min + 1e-8)


# ====================== 稠密检索（全库一次性编码）======================
def dense_retrieve_local(
    df: pd.DataFrame, model_name: str, top_k: int = 100, passages_list=None
):
    """
    与原接口保持兼容，但当提供 passages_list 时：
    - 预编码语料只做一次
    - 为每个 query 做检索
    返回 {qid: [passage, ...]}
    """
    df = _ensure_ids(df)
    # 语料
    if passages_list is None:
        raise ValueError(
            "请提供 passages_list（一般来自 concatenated_chunks.csv 的 Text 列）"
        )
    passages = [str(p) for p in passages_list if pd.notna(p)]
    if not passages:
        raise ValueError("passages_list 为空。")

    # 编码与建索引（一次）
    model = SentenceTransformer(model_name)
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
        df.iterrows(), total=len(df), desc=f"Retrieving with {model_name}"
    ):
        qid = row["id"]
        query = str(row["query"])
        query_emb = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        k = min(top_k, len(passages))
        _, idxs = index.search(query_emb, k)
        results[qid] = [passages[i] for i in idxs[0]]
    return results


# ====================== Hybrid（BGE + BM25）======================
def hybrid_retrieve_bge_bm25(
    df: pd.DataFrame,
    alpha: float = 0.3,
    top_k: int = 100,
    passages_list=None,
    bge_model: str = "BAAI/bge-small-en-v1.5",
):
    """
    Dense(BGE) + BM25 融合：
    final = alpha * bm25_norm + (1 - alpha) * dense_norm
    - 预编码语料 & 建索引只做一次
    - BM25 也只建一次（优先使用 hybrid 模块中的 bm25，否则使用本地 BM25）
    返回 {qid: [passage, ...]}
    """
    df = _ensure_ids(df)

    # 语料：优先用 hybrid 的 documents，否则用传入的 passages_list
    if (
        _USE_HYBRID_BM25
        and isinstance(_HYBRID_CORPUS, list)
        and len(_HYBRID_CORPUS) > 0
    ):
        passages = [str(p) for p in _HYBRID_CORPUS]
    else:
        if passages_list is None:
            raise ValueError(
                "请提供 passages_list（或确保 hybrid_retriver 可用以提供 documents）"
            )
        passages = [str(p) for p in passages_list if pd.notna(p)]

    if not passages:
        raise ValueError("语料为空。")

    # 1) 稠密侧：一次性编码全库并建索引
    dense_model = SentenceTransformer(bge_model)
    passage_emb = dense_model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dim = passage_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_emb)

    # 2) BM25：优先使用 hybrid 模块中的 bm25，否则构建本地 BM25
    if _USE_HYBRID_BM25 and _HYBRID_BM25 is not None:
        bm25_model = _HYBRID_BM25
        # 注意：hybrid 内 bm25 的文档顺序需与 passages 一致（你之前的 hybrid 即如此）
        bm25_on_local = False
    else:
        tokenized_passages = [_tokenize(p) for p in passages]
        bm25_model = BM25(tokenized_passages)
        bm25_on_local = True

    results = {}
    N = len(passages)

    for _, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc="Hybrid retrieve (bge + bm25)"
    ):
        qid = row["id"]
        query = str(row["query"])

        # 稠密得分：取全量（k=N），再还原为 length=N 的分数数组
        q_emb = dense_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        dense_scores_sorted, dense_indices_sorted = index.search(q_emb, N)
        dense_scores_sorted = dense_scores_sorted[0]
        dense_indices_sorted = dense_indices_sorted[0]

        dense_scores = np.zeros(N, dtype=np.float32)
        dense_scores[dense_indices_sorted] = (
            dense_scores_sorted  # 还原到对应 passage 位置
        )

        # BM25 得分
        tokenized_query = _tokenize(query)
        bm25_scores = np.array(bm25_model.get_scores(tokenized_query), dtype=np.float32)
        # 防御：有些第三方 BM25 可能返回长度与 N 不一致
        if bm25_scores.shape[0] != N:
            # 强制对齐长度（极少发生；若发生，截断或填零）
            bm25_scores = np.pad(bm25_scores[:N], (0, max(0, N - bm25_scores.shape[0])))

        # 归一化并线性融合
        bm25_norm = _minmax(bm25_scores)
        dense_norm = _minmax(dense_scores)
        final_score = alpha * bm25_norm + (1.0 - alpha) * dense_norm

        topk = min(top_k, N)
        ranked_idx = np.argsort(final_score)[::-1][:topk]
        results[qid] = [passages[i] for i in ranked_idx]

    return results


# ====================== 入口（读新 CSV，跑并评估）======================
def run_and_eval_retrievers(top_k: int = 100, alpha: float = 0.3):
    # 读取 query/gold & 语料（当不使用 hybrid 的 documents 时）
    qdf = pd.read_csv("query.csv")
    if "query" not in qdf.columns:
        raise ValueError("query.csv 必须包含列 'query'")
    qdf = _ensure_ids(qdf)

    passages_list = None
    if not _USE_HYBRID_BM25:
        pdf = pd.read_csv("concatenated_chunks.csv")
        if "Text" not in pdf.columns:
            raise ValueError("concatenated_chunks.csv 必须包含列 'Text'")
        passages_list = pdf["Text"].dropna().astype(str).tolist()

    retrievers = [
        # 仅示例一个（你也可继续保留/新增其他检索器）
        (
            "hybrid_bge_bm25",
            lambda d=qdf: hybrid_retrieve_bge_bm25(
                d,
                alpha=alpha,
                top_k=top_k,
                passages_list=passages_list,
                bge_model="BAAI/bge-small-en-v1.5",
            ),
            f"./retriever_eval/hybrid_bge_bm25_top{top_k}.csv",
        ),
        (
            "dense_bge_only",
            lambda d=qdf: dense_retrieve_local(
                d,
                model_name="BAAI/bge-small-en-v1.5",
                top_k=top_k,
                passages_list=(
                    passages_list if passages_list is not None else _HYBRID_CORPUS
                ),
            ),
            f"./retriever_eval/dense_bge_only_top{top_k}.csv",
        ),
    ]

    for name, fn, path in retrievers:
        print(f"\nRunning retriever: {name}")
        try:
            results = fn()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            evaluate_retriever_performance(results, path)
        except Exception as e:
            print(f"Retriever '{name}' failed with error:\n{e}")
            continue


if __name__ == "__main__":
    run_and_eval_retrievers(top_k=100, alpha=0.3)
